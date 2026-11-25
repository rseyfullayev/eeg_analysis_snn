"""Stage 9 preflight — the LIVE pipeline, fast, on BOTH devices.

Runs (in order, mirroring train.py exactly): real SWEEPDataset batches through a
real DataLoader with the train-loop unpack semantics (8-tuple incl. the
`timestamps.to(device)` / `[:, 0]` lines that crashed on the pod 2026-11-26), the
Option-B pooled pre-pass + seed_priors, seeded SMCLossEngine forward/backward
steps incl. the NullspaceState rank-recompute (searchsorted — the pod crash
site) and all M-step side effects, a classification-only validation pass, then
lock_fullpass -> Phase1BReadout -> LBFGS finetune -> evaluate -> head_state
roundtrip.

WHY THIS EXISTS: every earlier stage test runs the same code on CPU tensors
only, so device-placement bugs (a CPU scalar tensor meeting a CUDA buffer)
could only be discovered by the pod — at billed GPU prices. This file runs the
whole live path once per available device in well under a minute, so:
  (a) locally (this box HAS an RTX A2000) it gates every commit, and
  (b) on the pod it is the LAUNCH GATE: run it FIRST, before the training line.

Run from repo root:  venv\\Scripts\\python.exe tests\\test_stage9_preflight.py
                   (pod: python tests/test_stage9_preflight.py)
Exit code 0 = all pass. CPU run is mandatory; CUDA run executes when
torch.cuda.is_available() and is announced (never silently skipped).
"""

import os
import sys
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.snn_modeling.adapters.mil_readout import Phase1BReadout
from src.snn_modeling.dataloader.dataset import SWEEPDataset, PKSampler
from src.snn_modeling.utils.sm_loss import SMCLossEngine, validate_lockpass

D = 16          # engine feature dim (toy encoder output)
NC = 2          # fixture emotion classes
NS = 4          # index-wide subjects (loso [1,2] -> TRAIN subjects {3,4}: n_present=2
                # forces the SVD + searchsorted rank path from the very first update)
BATCH = 4
EPOCHS = 2


def _sm_cfg():
    return dict(
        tau_norm=0.25, rank_energy_band=0.85, max_rank=1, warmup_epochs=0,
        space_momentum=0.9, rho_min=0.05, seed=dict(s=2.0), sigma_sq=0.05,
        tau_dcl=0.1, anchor_on="projected",
        anneal=dict(t_start=2.0, t_end=1.0, frac=0.3),
        c=0.5, g_floor=0.1, lambda_anc=1.0, lambda_sub=0.1, lambda_rho=0.1,
        sinkhorn=dict(tol=1e-3, T_sc=30, n_iters=200, solver_tol=1e-10, residual_cap=1e-3),
        # ^ ONLY n_iters deviates from the pod smoke config (25 -> 200): the pod's OWN
        # step-1 OT passed the 1e-3 cap at 25 iters (its crash was in the M-step rank
        # update AFTER the solve), so eps=0.15/25 is validated on real geometry. Tiny
        # n=2-rows x small-m toy blocks converge slower than the cap allows at 25;
        # bumping iters keeps every solver/eps/cap/device code path identical without
        # asserting toy geometry == SEED-V geometry.
        eps=0.15, eps_max=1.0, n_q=96, queue_momentum=0.9, gate_ema_momentum=0.9,
        tau_0=0.35, kappa_cost=0.5, lambda_max=1.0, allow_smoke_sizing=True,
    )


def _preflight_dataset_dir(tmp, n_subjects=4, n_bags_per=4, n_windows=5):
    """Class-STRUCTURED synthetic windows (T=2,C=1,H=4,W=4): each class carries its
    own additive quadrant pattern (so the untrained encoder's normalized embeddings
    CLUSTER like real 1A data) and class-1 bags additionally get a 20% burst-window
    boost. Random-uniform windows would make embeddings near-orthogonal ->
    uniform cost ~2 everywhere, which stalls Sinkhorn at eps=0.15/25 iters and
    makes the pooled Option-B delta spread too narrow for the Banach fixed point.
    Both are FIXTURE artifacts, not real SEED-V geometry (measured here 2026-11-26)."""
    rows = []
    for s in range(1, n_subjects + 1):
        for b in range(n_bags_per):
            y = (s + b) % NC
            bag_id = 100 * s + b
            for w in range(n_windows):
                fname = f"{s}_{bag_id}_s{w}.pt"
                x = 0.25 * torch.rand(2, 1, 4, 4)   # low noise floor: the class pattern
                # dominates the untrained embedding direction, like trained-adjacent real
                # data (the pod itself passed OT at eps=0.15/25 iters on step 1 — its
                # crash was in the M-step rank update AFTER the solve). A noise-dominated
                # toy makes the residual a coin-flip at the cap — harness artifact.
                if y == 1:
                    x[:, :, 2:, :] += 1.2
                    if w % 5 == 0:
                        x[:, :, :2, :2] += 5.0  # burst windows (~20% of class 1)
                else:
                    x[:, :, :2, :2] += 1.2
                rows.append((fname, bag_id, y))
                torch.save(x, os.path.join(tmp, fname))
    with open(os.path.join(tmp, "index.csv"), "w") as f:
        f.write("filename,bag_id,emotion_id\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]}\n")


def _build(tmp):
    from tests.test_stage0 import _cfg
    _preflight_dataset_dir(tmp)
    cfg = _cfg(tmp, bagging=False)
    ds_tr = SWEEPDataset(cfg, loso=[1, 2], split='train', precomputed=False)
    ds_tr.augmentations = nn.Sequential()          # identity dual view (8-tuple)
    ds_tr.key_augmentations = None
    ds_va = SWEEPDataset(cfg, loso=[1, 2], split='val', precomputed=False)
    ld_tr = DataLoader(ds_tr, batch_sampler=PKSampler(ds_tr, batch_size=BATCH, n_classes=NC,
                                                      subject_diverse_k=False))
    ld_va = DataLoader(ds_va, batch_size=BATCH, shuffle=False)
    return cfg, ld_tr, ld_va


def run_pipeline(dev: str):
    import random
    random.seed(0)                              # PKSampler/dataset use Python random too —
    torch.manual_seed(0)                        # IDENTICAL data+order per device: a pass/
    device = torch.device(dev)                  # fail difference here IS a device bug
    with tempfile.TemporaryDirectory() as tmp:
        cfg, ld_tr, ld_va = _build(tmp)
        enc = nn.Flatten()
        model = nn.Sequential(enc, nn.Linear(2 * 1 * 4 * 4, D)).to(device)
        head = nn.Linear(D, NC).to(device)
        opt = torch.optim.SGD(list(model.parameters()) + list(head.parameters()), lr=0.05)

        eng = SMCLossEngine(_sm_cfg(), feature_dim=D, num_classes=NC,
                            num_subjects=NS, total_steps=EPOCHS * max(1, len(ld_tr))) \
            .to(device)
        assert eng.space.U.device.type == dev

        # ---- Option-B pre-pass STATS lines (mirrors train.py exactly): exercised on
        # the loader so the pooling/standardize/log_coeff path runs per device. The
        # PRIORS themselves are then seeded from a synthetic norm population at
        # realistic burst rates (0.06-0.26/class): a 40-row toy CDF is too coarse
        # for the 1e-10 Banach tolerance (residual stalls as a staircase) and the
        # untrained encoder cannot cluster away the class-amplitude conflation —
        # FIXTURE artifacts of this fast harness, not production bugs (the pod
        # pre-pass sees >= 36k pooled rows).
        pooled = {}
        with torch.no_grad():
            for b in ld_tr:
                view, lab = b[0], b[3]
                r = view.to(torch.float64).mean(dim=(1, 3, 4)).cpu()
                for y in lab.unique().tolist():
                    pooled.setdefault(int(y), []).append(r[lab == y])
        all_r = torch.cat([torch.cat(v) for v in pooled.values()])
        med = all_r.median(dim=0).values
        iqr = (torch.quantile(all_r, 0.75, dim=0) - torch.quantile(all_r, 0.25, dim=0)) / 1.349
        assert torch.isfinite(med).all() and (iqr > 0).all() and all_r.shape[1] == 1
        b_dim = float(all_r.shape[1])
        rng = torch.Generator().manual_seed(11)
        norms = {}
        for y in range(NC):                      # d^2 ~ chi^2(b_dim) bulk; burst rows ~ s^2 x
            base = torch.linalg.vector_norm(
                torch.randn(240, 32, generator=rng, dtype=torch.float64), dim=1) ** 2 / 32.0
            burst = base[:int((0.06 + 0.10 * y) * 240)] * 4.0
            norms[y] = torch.cat([base, burst])
        info = eng.seed_priors(norms, log_coeff=b_dim)
        rho = eng.rho.detach()
        assert rho.shape == (NC,) and torch.isfinite(rho).all() \
            and bool(((rho > 0) & (rho < 1)).all()), f"seed rho broken: {rho}"

        eng.run_sizing_checks(batch_size=BATCH, epochs=EPOCHS, steps_per_epoch=len(ld_tr),
                              L_const=1.0, d_eff=float(D), tol_eff=0.1, C_const=4.0,
                              eps_tol=1e9, allow_smoke_sizing=True)

        # ---- training loop: train.py's exact unpack + engine call
        bag_embs, bag_ys = {}, {}
        losses = []
        for ep in range(EPOCHS):
            eng.space.set_epoch(ep)
            for batch in ld_tr:
                # ---- the pod crash lines, verbatim contract (train.py:389-391, 461-464)
                inp1, inp2, targets, targets_c, subject_labels, _, video_ids, timestamps = batch
                video_ids = video_ids.to(device)
                timestamps = timestamps.to(device)
                inp1, inp2 = inp1.to(device), inp2.to(device)
                subject_labels = subject_labels.to(device)
                targets, targets_c = targets.to(device), targets_c.to(device)
                assert isinstance(timestamps, torch.Tensor), "slot 7 must arrive as a tensor"
                wins = timestamps.reshape(-1) if timestamps.dim() == 1 else timestamps[:, 0]

                z_q = F.normalize(model(inp1.to(device)), dim=1, eps=1e-6)
                with torch.no_grad():
                    key_features = F.normalize(model(inp2.to(device)), dim=1, eps=1e-6)
                out = eng(z_q, key_features, targets_c, subject_labels, wins, video_ids)
                loss = out["total"]
                assert torch.isfinite(loss), f"non-finite engine loss epoch {ep}"
                opt.zero_grad()
                loss.backward()
                for p in model.parameters():
                    assert p.grad is not None and torch.isfinite(p.grad).all(), "bad grad"
                opt.step()
                losses.append(float(loss.detach()))
                d = out["diagnostics"]
                assert d["ot_residual"] == {} or max(d["ot_residual"].values()) < 1e-3
                zc = z_q.detach().cpu()
                for bid, zi, yi in zip(video_ids.tolist(), zc.tolist(), targets_c.tolist()):
                    bag_embs.setdefault(int(bid), []).append(zi)
                    bag_ys[int(bid)] = int(yi)
                if len(losses) >= EPOCHS * len(ld_tr):
                    break   # safety cap: exactly EPOCHS * steps_per_epoch engine steps

        # rank machinery must have actually run (the crash path, not the early return)
        assert not eng.space.warmup_active and eng.space.U.device.type == dev
        assert int(eng.space.rank.item()) <= 1 and float(eng.sigma_hat_sq) >= 0.05 - 1e-12
        assert eng.queue.keys.device.type == dev and int(eng.queue.filled.item()) > 0

        # ---- classification-only validation pass (validate() device moves)
        ce = nn.CrossEntropyLoss()
        with torch.no_grad():
            tot, correct, n = 0.0, 0, 0
            for batch in ld_va:
                inputs, targets, targets_c, _, video_ids, timestamps = batch[:6]
                inputs, targets_c = inputs.to(device), targets_c.to(device)
                video_ids = video_ids.to(device)
                timestamps = timestamps.to(device)
                logits = head(model(inputs))
                tot += float(ce(logits, targets_c))
                correct += int((logits.argmax(1) == targets_c).sum())
                n += targets_c.numel()
            assert n > 0 and torch.isfinite(torch.tensor(tot / n))

        # ---- Phase 1B (CPU by contract: payloads/lockpasses are CPU artifacts)
        lock = eng.lock_fullpass()
        validate_lockpass(lock)
        assert lock["schema"] == "smc-lockpass-v2" and torch.isfinite(lock["c"]).all()
        rd = Phase1BReadout(lock)
        ztr = [torch.stack([torch.tensor(v) for v in bag_embs[b]]) for b in sorted(bag_embs)]
        ytr = torch.tensor([bag_ys[b] for b in sorted(bag_embs)], dtype=torch.long)
        R, stats = rd.encode_bags(ztr, ytr)
        assert torch.isfinite(R).all() and float(R.norm(dim=1).max()) <= 2.0 + 1e-5
        assert all(s["kappa"] < 1.0 for s in stats)
        rd.fit_nearest_centroid(R, ytr)
        rd.init_bayes_head()
        assert torch.equal(rd.head_predict(R), rd.centroid_predict(R)), \
            "Eq. 18 head must match centroid at warm start"
        hist = rd.finetune(R, ytr, epochs=30, lambda_wd=0.05)
        assert hist and min(h["loss"] for h in hist) <= hist[0]["loss"] + 1e-6
        acc = rd.evaluate(R, ytr, use="head")
        assert 0.0 < acc <= 1.0
        st = rd.head_state()
        rd2 = Phase1BReadout(lock)
        rd2.fit_nearest_centroid(R, ytr)
        rd2.load_head_state(st)
        assert torch.equal(rd2.head_predict(R), rd.head_predict(R))
        return dict(steps=len(losses), first_loss=losses[0], last_loss=losses[-1],
                    rank=int(eng.space.rank.item()), rho=[round(float(v), 4) for v in rho],
                    val_n=n, bags=len(ztr), ft_acc=round(acc, 4))


def main():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    else:
        print("[SKIP] cuda unavailable on this box — RUN THIS FILE ON THE POD before "
              "launching the real job; device bugs only exist there.")
    failed = 0
    for dev in devices:
        try:
            res = run_pipeline(dev)
            print(f"PASS [{dev}] steps={res['steps']} rank<={res['rank']} "
                  f"rho={res['rho']} bags={res['bags']} ft_acc={res['ft_acc']} "
                  f"loss {res['first_loss']:.3f} -> {res['last_loss']:.3f}")
        except Exception as e:
            failed += 1
            import traceback
            print(f"FAIL [{dev}]: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"{len(devices) - failed}/{len(devices)} device passes")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
