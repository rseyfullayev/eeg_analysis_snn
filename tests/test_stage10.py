"""Stage 10 — the LIVE LOOP gate. Run the real training loop, not a copy of it.

WHY THIS EXISTS (pod crash, 2026-11-26): `train.training_loop()` referenced a bare
`config` at train.py:689/707 (`num_workers=int(config.data.get(...))`,
`config.training.get('probe_max_windows', ...)`), but `config` is not a parameter of
`training_loop` and not a module global -> NameError. It sat inside the Phase-1A
offline-probe branch, so it only fired at `epoch % probe_interval == 0` on a real
run: at billed GPU prices, after the first epoch of the smoke job.

Stage 9 already "preflights the live pipeline" — but it RE-IMPLEMENTS the loop inline
instead of calling `train.training_loop()`, so a NameError inside the real function is
invisible to it. That is the gap this file closes, in two layers:

  1. STATIC — every function/method in train.py, test.py, main.py,
     precompute_features.py, generate_dataset.py is disassembled and every name it
     loads with LOAD_GLOBAL is checked against that module's globals + builtins.
     This finds the exact bug class in branches that no test can reach cheaply
     (probe cadence, resume, early-stop, the burst PNG diagnostic) without running them.
  2. DYNAMIC — `train.run_training()` is executed end-to-end on a synthetic pseudo
     SEED-V slice with the REAL SpikingMobileNetProjector, REAL SWEEPDataset,
     REAL SMCLossEngine and `probe_interval=1`, so the loop body, `validate()`,
     the offline-probe branch (the crash site), the ZeroStep preview, the SMC metric
     logging and the checkpoint writes ALL actually run. The test asserts the probe
     keys reached the writer — a green run that skipped the branch is a failure here.

Run from repo root:  venv\\Scripts\\python.exe tests\\test_stage10.py
                   (pod: python tests/test_stage10.py)
                   Optional args filter by test-name substring, e.g.
                   python tests/test_stage10.py undefined_globals
Exit code 0 = all pass. CPU runs are mandatory; CUDA runs execute when
torch.cuda.is_available() and are announced, never silently skipped.
Measured on this box (RTX A2000): 8/8 in ~75 s.

BUGS THIS GATE FOUND ON ITS FIRST RUN (2026-11-26, all fixed):
  1. train.training_loop read a bare `config` (the pod crash above) — no parameter,
     no module global. NameError at the first probe epoch.
  2. SupMoCoState.enqueue got `timestamps` as (B,1)/(B,K) while the queue slots are (B,)
     -> "expand(torch.LongTensor{[10,1]}, size=[10])" on the legacy (loss.sm off) path.
     train.py now passes the same (B,) `wins` vector the SMC branch already used.
  3. validate(..., only_classification=True) was called with the TRAINING criterion:
     SupMoCoLoss.forward needs 4+ args (TypeError on the first val pass) and
     ContrastiveLoss silently graded logits contrastively. Val now uses CrossEntropyLoss.
  4. test._linear_probe_loto saved evidence/cm_loto_*.png with no os.makedirs("evidence")
     -> FileNotFoundError inside the probe branch on any checkout where --masks was
     never run (the pod's own run dir).
  5. test.py's cuSVC / cuLogisticRegression / cuUMAP are only bound when the module-level
     cuml import succeeds, yet are referenced by name -> undefined globals on a
     cuML-less box (harmless behind _USE_CUML, but exactly this bug class). Now bound.
"""

import dis
import importlib
import inspect
import io
import os
import random
import sys
import tempfile
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NC = 5            # SEED-V emotion count (kept real: PKSampler defaults to 5)
SUBJECTS = [1, 2, 3, 4]   # loso=[1,2] -> VAL subjects {1,2}, TRAIN subjects {3,4}
BAGS_PER_SUBJECT = NC     # one bag per class per subject
WINDOWS_PER_BAG = 6
T, C, GRID = 2, 5, 32     # window shape (T, freq_bands, H, Wg) — matches config/data/seed_v_ea


# =========================================================== layer 1: static scan

def _iter_code_objects(code):
    yield code
    for const in code.co_consts:
        if hasattr(const, "co_code"):
            yield from _iter_code_objects(const)


def undefined_globals(fn, extra_names=()):
    """Names this function loads as a global that exist neither in its module globals,
    nor in builtins, nor in `extra_names`. Closure variables are LOAD_DEREF, so names
    legitimately captured from an enclosing scope are never reported."""
    module_globals = getattr(fn, "__globals__", {})
    missing = []
    for code in _iter_code_objects(fn.__code__):
        line = code.co_firstlineno
        for ins in dis.get_instructions(code):
            # 3.14 reports starts_line=False (not None) between statements; positions
            # is the reliable line source.
            pos = getattr(ins, "positions", None)
            lineno = getattr(pos, "lineno", None)
            if isinstance(lineno, int):
                line = lineno
            if ins.opname != "LOAD_GLOBAL":
                continue
            name = ins.argval
            if name is None or name in module_globals or hasattr(__builtins__, name) \
                    or name in extra_names or name in code.co_varnames:
                continue
            missing.append((name, code.co_name, line))
    return missing


def _module_functions(module):
    out = {}
    for value in vars(module).values():
        targets = [value]
        if inspect.isclass(value):
            targets += [v for _, v in inspect.getmembers(value, inspect.isfunction)]
        for t in targets:
            if inspect.isfunction(t) and t.__module__ == module.__name__:
                out[f"{module.__name__}.{t.__qualname__}"] = t
    return out


LIVE_MODULES = ("train", "test", "main", "precompute_features", "generate_dataset")


def test_undefined_globals_scanner_is_not_vacuous():
    """Harness sanity (both directions): the scanner must catch a planted NameError in
    a branch that never executes, and must stay quiet on a function that closes over a
    local / a parameter (which is what `config` was wrongly assumed to be)."""
    probe_src = """
def broken(inner, flag):
    def helper(v):
        return v + 1
    if not flag:                   # cold at call time, live in the bytecode —
        return helper(inner) + MISSING_NAME   # exactly the probe-branch situation
    return helper(inner) + MISSING_NAME

def shadowed(value):
    MISSING_NAME = 1               # a local: LOAD_FAST, must NOT be flagged
    return MISSING_NAME + value

def closure_ok(value):
    captured = value * 2           # LOAD_DEREF from the enclosing frame, not a global
    def inner(v):
        return v + captured
    return inner(1)
"""
    ns = {}
    exec(compile(probe_src, "<scanner-sanity>", "exec"), ns)
    hits = {name for name, _, _ in undefined_globals(ns["broken"])}
    assert hits == {"MISSING_NAME"}, f"scanner missed the planted global: {sorted(hits)}"
    assert undefined_globals(ns["shadowed"]) == [], "scanner flagged a local as global"
    assert undefined_globals(ns["closure_ok"]) == [], "scanner flagged a closure var"


def test_live_entry_modules_have_no_undefined_globals():
    os.environ.setdefault("WANDB_MODE", "disabled")
    offenders = []
    per_module = {}
    seen = set()
    for mod_name in LIVE_MODULES:
        module = importlib.import_module(mod_name)
        functions = _module_functions(module)
        per_module[mod_name] = len(functions)
        for qual, fn in sorted(functions.items()):
            seen.add(qual)
            for name, code_name, line in undefined_globals(fn):
                offenders.append(f"{qual} (inner {code_name!r}, line ~{line}): "
                                 f"LOAD_GLOBAL {name!r} is not defined in module "
                                 f"{module.__name__} nor in builtins")
    thin = {m: n for m, n in per_module.items() if n == 0}
    assert not thin, f"scanner saw no functions in {thin} — the import did not work"
    for required in ("train.training_loop", "train.validate", "train.run_training",
                     "train.phase_one_a", "test.test"):
        assert required in seen, f"{required} was not scanned (renamed? scanner scope?)"
    assert not offenders, "undefined global(s) on the live path (pod NameError):\n  " + \
        "\n  ".join(offenders)


def test_training_loop_declares_every_name_it_reads():
    """Pin the specific regression: training_loop must receive `config` explicitly —
    the probe branch reads config.data.num_workers and config.training.probe_max_windows."""
    import train
    params = inspect.signature(train.training_loop).parameters
    for needed in ("config", "train_loader", "val_loader", "smc_engine", "probe_interval"):
        assert needed in params, f"train.training_loop has no parameter {needed!r}"
    src = inspect.getsource(train.training_loop)
    assert "config.data" in src and "config.training" in src, \
        "probe branch no longer reads config — update this pin or the probe block"
    # One caller must exist and must pass config (a required kw-only arg fails loudly
    # at the call site otherwise; no default = no silent None fallback).
    caller = inspect.getsource(train.phase_one_a)
    assert "config=config" in caller, "phase_one_a does not pass config to training_loop"


# ========================================================= layer 2: dynamic smoke

def _write_windows(root):
    """Class-STRUCTURED pseudo windows (see test_stage9's note): each emotion carries
    its own quadrant offset so the untrained encoder's embeddings cluster like real
    Phase-1A data instead of going near-orthogonal (which stalls Sinkhorn)."""
    os.makedirs(root, exist_ok=True)
    rows = []
    for s in SUBJECTS:
        for y in range(NC):
            bag_id = 100 * s + y
            for w in range(WINDOWS_PER_BAG):
                fname = f"{s}_{bag_id}_s{w}.pt"
                x = 0.25 * torch.rand(T, C, GRID, GRID)
                quad = (y % 2) * (GRID // 2)
                band = (y // 2)
                x[:, band, quad:quad + GRID // 2, :] += 1.2
                if y == 1 and w % 3 == 0:
                    x[:, 0, :GRID // 4, :GRID // 4] += 5.0   # burst windows
                torch.save(x, os.path.join(root, fname))
                rows.append((fname, bag_id, y))
    with open(os.path.join(root, "index.csv"), "w") as f:
        f.write("filename,bag_id,emotion_id\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]}\n")


def _sm_cfg(n_subjects):
    """loss.sm for the toy corpus. Everything except the sizes that cannot fit a
    4-subject fixture (n_q, seed_min_windows, max_rank, warmup) and the toy-geometry eps
    (see below) matches the pod smoke config
    config/ea_supcon_fda_adamw_lotomasking.yaml.

    eps 0.15 -> 0.6: measured here on 256-D real-encoder embeddings, the pod's pinned
    smoke eps leaves a 2.8e-3..3.6e-3 marginal residual after 200 Sinkhorn iters on THIS
    toy (4 subjects x 5 classes x 6 windows), while the real 36k-window corpus converges
    at ~2e-7. That is a fixture-geometry artifact, NOT a production signal — so the fixture
    raises eps (the §11.5 ladder's first rung) instead of relaxing residual_cap, which
    stays at the shipped 1e-3 so the acceptance check itself is still exercised."""
    return dict(
        enabled=True, feature_dim=256, num_subjects=n_subjects, n_q=64,
        tau_norm=0.25, rank_energy_band=0.85, max_rank=1, warmup_epochs=0,
        space_momentum=0.9, rho_min=0.05, seed=dict(s=2.0), sigma_sq=0.05,
        tau_dcl=0.1, anchor_on="projected",
        anneal=dict(t_start=2.0, t_end=1.0, frac=0.3),
        c=0.5, g_floor=0.1, lambda_anc=1.0, lambda_sub=0.1, lambda_rho=0.1,
        sinkhorn=dict(tol=1e-3, T_sc=30, n_iters=400, solver_tol=1e-10, residual_cap=1e-3),
        eps=0.6, eps_max=1.0, queue_momentum=0.9, gate_ema_momentum=0.9,
        tau_0=0.35, kappa_cost=0.5, lambda_max=1.0, allow_smoke_sizing=True,
        seed_max_batches=16, seed_min_windows=8, calib=dict(bisection_steps=25, ema_momentum=0.95),
        ot_loto_mask=True, force_balanced=False, supmoco_momentum_end=0.99,
        sizing_C_const=1.0, sizing_d_eff=20.0, sizing_tol=0.5, sizing_eps_tol=20000.0,
        sizing_L=1.0,
    )


def _cfg(root, sm=True):
    from omegaconf import OmegaConf
    loss = dict(temperature=0.1, use_scda=False, decoupled=True, exclude_same_trial=True,
                cosface_margin=0.0, smooth=0.0, lambda_seg=0.0, lambda_con=0.0)
    if sm:
        loss["sm"] = _sm_cfg(len(SUBJECTS))
    return OmegaConf.create({
        "experiment_name": "stage10-smoke",
        "data": {"dataset_path": root, "save_path": root, "n_emotions": NC,
                 "grid_size": GRID, "freq_bands": C, "num_timesteps": T,
                 "num_workers": 0, "preload_ram": False, "train_size": 0.8,
                 "mil_bag_size": WINDOWS_PER_BAG, "prefetch_factor": 2},
        "mask": {"type": "Statistical Mask", "sigma": 0.25, "radius": 0.7},
        "model": {"num_classes": NC},
        "training": {"phase": "1a", "phase_1_epochs": 2, "batch_size": 10,
                    "bagging": False, "probe_interval": 1, "probe_max_windows": 48,
                    "learning_rate": 1e-3, "lr_min": 1e-5, "weight_decay": 0.01,
                    "warmup_epochs": 0, "accumulation_steps": 1, "use_muon": False,
                    "use_pk_sampler": True, "subject_diverse_k": False,
                    "use_supmoco": True, "supmoco_momentum": 0.9, "supmoco_queue_size": 32,
                    "bf16": False, "use_fourier_mixup": True, "key_view": "light",
                    "temporal_decay_enabled": False, "temporal_decay_factor": 0.99,
                    "fourier_mixup": {"retention_ratio": 0.95, "p": 0.5, "beta_alpha": 1.0}},
        "logger": {"enabled": False, "offline": True, "project_name": "stage10-smoke",
                   "run_name": "stage10-smoke", "tags": ["smoke"], "log_interval": 1,
                   "telegram": {"token": None, "chat_id": None}},
        "loss": loss,
    })


def _model(device):
    from src.snn_modeling.models.encoders import SpikingMobileNetEncoder
    from src.snn_modeling.models.unet import SpikingMobileNetProjector
    enc = SpikingMobileNetEncoder(in_channels=C, p_drop=0.3, use_odconv=True,
                                  use_batchnorm=False, spike_model="SiLU")
    return SpikingMobileNetProjector(encoder_backbone=enc, num_classes=NC).to(device)


class _RecordingWriter:
    """Delegates to the real SummaryWriter and records every scalar key, so a test can
    prove the probe branch actually executed instead of being silently skipped."""

    instances = []

    def __init__(self, log_dir):
        from torch.utils.tensorboard import SummaryWriter
        self._inner = SummaryWriter(log_dir=log_dir)
        self.scalars = {}
        _RecordingWriter.instances.append(self)

    def add_scalar(self, tag, value, step=None):
        self.scalars.setdefault(tag, []).append(value)
        return self._inner.add_scalar(tag, value, step)

    def close(self):
        return self._inner.close()


def _wandb_offline():
    """train.training_loop / test.test() log through the module-level wandb without
    initializing it (main.py owns the init). Start a disabled run so those logging lines
    execute instead of raising — this is what the pod does, minus the network."""
    import wandb
    if wandb.run is None:
        wandb.init(project="stage10-smoke", mode="disabled",
                   settings=wandb.Settings(_disable_stats=True, _disable_meta=True))


class _cwd:
    """run_training writes results/<run>/ and the probe block writes evidence/*.png
    relative to CWD, so each live run executes inside its own temp dir."""

    def __init__(self, path):
        self.path, self.old = path, os.getcwd()

    def __enter__(self):
        os.chdir(self.path)
        return self

    def __exit__(self, *exc):
        os.chdir(self.old)
        return False


def _run_training(dev):
    """Real entry point -> phase_one_a -> training_loop -> validate -> probes -> ckpt."""
    import train

    os.environ["WANDB_MODE"] = "disabled"
    os.environ.setdefault("MPLBACKEND", "Agg")
    random.seed(0)
    torch.manual_seed(0)
    _RecordingWriter.instances.clear()
    real = train.SummaryWriter
    train.SummaryWriter = _RecordingWriter
    try:
        with tempfile.TemporaryDirectory() as root:
            _write_windows(root)
            with _cwd(root):
                cfg = _cfg(root, sm=True)
                device = torch.device(dev)
                train.run_training(cfg, _model(device), device, phase="1a", resume=False,
                                   loso=[1, 2])
            writer = _RecordingWriter.instances[-1]
            ckpt_dir = os.path.join(root, "saved_models", "phase1a", "stage10-smoke")
            return set(writer.scalars), sorted(os.listdir(ckpt_dir)), writer.scalars
    finally:
        train.SummaryWriter = real


# Keys that ONLY exist if the Phase-1A probe branch ran (the pod crash site).
PROBE_KEYS = [
    "Phase1a/Train/Emotion_LOTO_Mean_Acc",
    "Phase1a/Val/Emotion_LOTO_Mean_Acc",
    "Phase1a/Val/Subject_Probe_Val_Acc",
    "Phase1a/Val/Subject_Probe_Chance",
]
SMC_KEYS = ["Phase1a/SMC/Gate", "Phase1a/SMC/OTResidualMax", "Phase1a/SMC/StdIntraTrial"]


def test_training_loop_smc_cpu_runs_every_branch():
    keys, files, scalars = _run_training("cpu")
    missing = [k for k in PROBE_KEYS + SMC_KEYS + ["Phase1a/Val/Loss",
               "Phase1a/Train/Loss", "LR"] if k not in keys]
    assert not missing, f"loop branches never ran / keys missing: {missing}"
    for f in ("checkpoint_last.pt", "checkpoint_best.pt", "lockpass_1a.pt"):
        assert f in files, f"missing artifact {f} in checkpoint dir: {files}"
    # ZeroStep preview (inside the probe branch, behind smc_engine) must have logged too.
    assert "Phase1a/Val/ZeroStep_Bag_Acc" in keys, f"ZeroStep preview skipped: {sorted(keys)}"
    for tag in ("Phase1a/Val/Loss", "Phase1a/SMC/OTResidualMax"):
        vals = [float(v) for v in scalars[tag]]
        assert all(v == v and abs(v) != float("inf") for v in vals), f"{tag} non-finite: {vals}"


def test_training_loop_smc_cuda_runs_every_branch():
    if not torch.cuda.is_available():
        print("[SKIP] cuda unavailable — the pod runs this gate before launching")
        return
    keys, files, _ = _run_training("cuda")
    missing = [k for k in PROBE_KEYS + SMC_KEYS if k not in keys]
    assert not missing, f"cuda loop branches missing: {missing}"
    assert "checkpoint_last.pt" in files


def test_legacy_supmoco_path_runs_end_to_end():
    """loss.sm disabled -> the SupMoCoLoss + SupMoCoState queue branch (get_queue /
    enqueue / momentum_update), the other half of the loop body."""
    import train
    import torch.nn as nn
    from src.snn_modeling.dataloader.dataset import SWEEPDataset
    from src.snn_modeling.utils.augmentations import (GaussianNoise, FrequencyDropout,
                                                      VideoRandomErasing,
                                                      VideoTemporalMasking, SignalJitter)
    from torch.utils.data import DataLoader

    os.environ["WANDB_MODE"] = "disabled"
    random.seed(0)
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as root:
        _write_windows(root)
        cfg = _cfg(root, sm=False)
        device = torch.device("cpu")
        model = _model(device)
        train_set = SWEEPDataset(cfg, loso=[1, 2], split="train", precomputed=False)
        chain = [GaussianNoise(std=0.05), FrequencyDropout(p=0.1),
                 VideoRandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
                 VideoTemporalMasking(p=0.1, max_mask_len=8),
                 SignalJitter(lower=0.5, upper=2.0)]
        train_set.augmentations = nn.Sequential(*chain)
        train_set.key_augmentations = nn.Sequential(*chain)   # no FourierMixup: clean keys
        tr = DataLoader(train_set, batch_size=10, shuffle=True)
        val_set = SWEEPDataset(cfg, loso=[1, 2], split="val", precomputed=False)
        va = DataLoader(val_set, batch_size=10, shuffle=False)
        _RecordingWriter.instances.clear()
        _wandb_offline()
        with _cwd(root):
            train.phase_one_a(cfg, model, device, tr, va,
                              _RecordingWriter(root), root, False)
        keys = set(_RecordingWriter.instances[-1].scalars)
        for required in PROBE_KEYS + ["Phase1a/Train/QueueSize", "Phase1a/Val/Loss"]:
            assert required in keys, f"legacy path missing {required!r}: {sorted(keys)}"
        assert os.path.exists(os.path.join(root, "checkpoint_last.pt"))


def test_validate_returns_finite_classification_metrics():
    """train.validate() on its own — the --val entry (main.py) uses exactly this call."""
    import train
    import torch.nn as nn
    from src.snn_modeling.dataloader.dataset import SWEEPDataset
    from torch.utils.data import DataLoader

    with tempfile.TemporaryDirectory() as root:
        _write_windows(root)
        cfg = _cfg(root, sm=False)
        device = torch.device("cpu")
        model = _model(device)
        val_set = SWEEPDataset(cfg, loso=[1, 2], split="val", precomputed=False)
        loader = DataLoader(val_set, batch_size=8, shuffle=False)
        out = train.validate(model, loader, nn.CrossEntropyLoss(), device,
                             only_classification=True)
        assert len(out) == 7, f"validate() arity changed: {len(out)} values"
        avg_loss, acc, bal = out[0], out[1], out[2]
        assert all(isinstance(v, (int, float)) and v == v and abs(v) != float("inf")
                   for v in (avg_loss, acc, bal)), f"non-finite val metric: {out}"
        assert avg_loss > 0.0, f"validate() averaged an empty loop: {out}"
        assert 0.0 <= acc <= 1.0 and 0.0 <= bal <= 1.0, out
        # only_classification=True returns the four segmentation slots as hard zeros.
        assert out[3:] == (0, 0, 0, 0), f"segmentation slots leaked into the cls-only path: {out}"


def test_offline_evaluation_entry_runs_end_to_end():
    """main.py --test -> test.test(): the UMAP/kNN/class-cosine/evidence-PNG loops the
    pod runs after training. Cheap enough to gate here, never covered before."""
    import test as eval_module
    os.environ["WANDB_MODE"] = "disabled"
    os.environ.setdefault("MPLBACKEND", "Agg")
    random.seed(0)
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as root:
        _write_windows(root)
        cfg = _cfg(root, sm=False)
        device = torch.device("cpu")
        model = _model(device)
        _wandb_offline()
        with _cwd(root):
            eval_module.test(cfg, [1, 2], None, device, model)
        figures = [os.path.join(d, f) for d, _, fs in os.walk(root)
                   for f in fs if f.endswith((".png", ".pdf"))]
        assert figures, f"evaluation produced no figures under {root}"


# ==================================================================== runner

def main():
    os.chdir(REPO)
    filters = [a for a in sys.argv[1:] if not a.startswith("-")]
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)
             and (not filters or any(f in k for f in filters))]
    if not tests:
        print(f"no test matched {filters}")
        return 1
    failed = 0
    for t in tests:
        t0 = time.time()
        buf = io.StringIO()
        try:
            real = sys.stdout
            sys.stdout = buf
            try:
                t()
            finally:
                sys.stdout = real
            print(f"PASS  {t.__name__}  ({time.time() - t0:.1f}s)")
        except Exception as e:
            failed += 1
            real = sys.stdout
            sys.stdout = real
            print(buf.getvalue()[-4000:])
            import traceback
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
