"""Stage 7 regression tests — Phase 1B MIL readout adapter (adapters/mil_readout.py).

Run from repo root:  venv\\Scripts\\python.exe tests\\test_stage7.py
Exit code 0 = all pass. (Plain asserts — the venv has no pytest.)

Covers plan §5 Stage-7 checklist: strict lockpass load; certified per-bag
R_b (Eq. 17) via two_step_baseline; zero-train nearest centroid == Eq. 18
Bayes warm-start (decision-identical by Prop 10.4's algebra); CE+L2 finetune
within the approved lambda_wd band; the TWO new Failure-Catalogue rows:
dropping the 1/T_b normalization must add duration bias; dropping b_y must
kill burst-rate discrimination. The lockpass is produced by a REAL
SMCLossEngine run and its class-mean/subject-offset geometry is reused by the
bag generator (no hand-tuned surrogate).
"""

import math
import os
import sys

import torch
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.snn_modeling.adapters.mil_readout import (Phase1BReadout, HEAD_SCHEMA, run_phase_1b)
from src.snn_modeling.utils.sm_loss import SMCLossEngine, LOCKPASS_SCHEMA

D, C, S = 16, 3, 4


def _expect(exc, fn, msg):
    try:
        fn()
    except exc:
        return
    raise AssertionError(msg)


def _cfg():
    return dict(feature_dim=D, num_subjects=S, n_q=64, warmup_epochs=0, tau_dcl=0.25,
                sigma_sq=1.0, rho_min=0.02, tau_norm=0.25, rank_energy_band=0.85,
                space_momentum=0.9, c=0.5, g_floor=0.1, lambda_sub=1.0, lambda_rho=0.1,
                sinkhorn=dict(tol=1e-3, T_sc=30, n_iters=600, solver_tol=1e-8),
                eps_max=1.0, gate_ema_momentum=0.9, seed_batches=96, seed_min_windows=64,
                anneal=dict(t_start=20.0, t_end=1.0, frac=0.3),
                seed=dict(s=2.0, log_coeff=float(D)))


def make_lockpass(seed=0):
    """Real engine run -> lockpass; also returns the (mu, off) it was built on."""
    torch.manual_seed(seed)
    eng = SMCLossEngine(_cfg(), feature_dim=D, num_classes=C, num_subjects=S, total_steps=50)
    norms = {y: (D + 2.0 * torch.rand(200, dtype=torch.float64)) for y in range(C)}
    eng.seed_priors(norms)
    mu = Fn.normalize(torch.randn(C, D), dim=1)
    off = Fn.normalize(torch.randn(S, D), dim=1)
    B = 16
    for step in range(4):
        labels = torch.arange(B) % C
        subj = torch.arange(B) % S
        z = Fn.normalize(mu[labels] + 0.8 * off[subj] + 0.05 * torch.randn(B, D), dim=1)
        k = Fn.normalize(mu[labels] + 0.8 * off[subj] + 0.05 * torch.randn(B, D), dim=1)
        z = z.detach().requires_grad_(True)   # the engine's grad_norm differentiates through z~
        wins = torch.arange(B) + step * B
        out = eng(z, k, labels, subj, wins, torch.arange(B) // 4 + step * 4)
        assert torch.isfinite(out["total"]), "engine loss diverged while filling the lock"
    return eng.lock_fullpass(), mu, off


def gen_bags(subjects, mu, off, T_list, seed=0, rho_true=(0.10, 0.25, 0.45),
             per_class=4, noise=0.05, gain=1.0):
    """Bags over the SAME geometry the engine learned: subject offset (with per-bag
    jitter — a fresh 'session'), class bursts at rate rho_true, burst amplitude
    `gain`·mu_y. The readout never receives subject ids, so unseen-vs-seen is only
    about the jitter seed."""
    g = torch.Generator().manual_seed(seed)
    zs, labels = [], []
    for s in subjects:
        base = Fn.normalize(off[s % off.shape[0]] + 0.15 * torch.randn(D, generator=g), dim=0)
        for y in range(C):
            for b in range(per_class):
                T = T_list[(y + b) % len(T_list)]
                burst = (torch.rand(T, generator=g) < rho_true[y]).float()
                rows = burst.view(-1, 1) * (gain * mu[y]).unsqueeze(0) \
                    + 0.8 * base.unsqueeze(0) + noise * torch.randn(T, D, generator=g)
                rows = rows + 1e-6                     # keep baseline rows off strict zero
                zs.append(Fn.normalize(rows, dim=1))
                labels.append(y)
    return zs, torch.tensor(labels, dtype=torch.long)


# ------------------------------------------------------------------ strict loading

def test_lockpass_strict():
    payload, mu, off = make_lockpass()
    assert payload["schema"] == LOCKPASS_SCHEMA
    Phase1BReadout(payload)                              # happy path
    bad = dict(payload); bad["schema"] = "v0"
    _expect(ValueError, lambda: Phase1BReadout(bad), "wrong schema accepted")
    missing = dict(payload)
    missing["clouds"] = {y: c for y, c in payload["clouds"].items() if y != 1}
    _expect(ValueError, lambda: Phase1BReadout(missing), "clouds missing class 1 accepted")
    badrho = dict(payload); badrho["rho"] = torch.tensor([0.0, 0.5, 0.5])
    _expect(ValueError, lambda: Phase1BReadout(badrho), "rho=0 accepted")
    _expect(ValueError, lambda: Phase1BReadout(payload, sigma_sq=0.0), "sigma_sq=0 accepted")
    _expect(ValueError, lambda: Phase1BReadout(dict(payload, eps=0.0)), "eps=0 accepted")


# ------------------------------------------------------------------ Eq. 17 + certification + duration row

def test_encode_bag_certified_and_duration_row():
    payload, mu, off = make_lockpass()
    rd = Phase1BReadout(payload)
    zs, labels = gen_bags([0, 1, 2, 3], mu, off, [6, 8, 12], seed=1)
    R, stats = rd.encode_bags(zs, labels)
    assert R.shape == (len(zs), D) and torch.isfinite(R).all()
    for s in stats:
        assert 0.0 <= s["kappa"] < 1.0, f"Prop 10.2 certificate breached: {s['kappa']}"
        assert math.isfinite(s["bound"]) and s["bound"] >= 0.0        # Eq. 16 logged
        assert float(s["R"].norm()) <= 2.0 + 1e-5, "Prop 10.3: ||R_b|| <= 2 on B^d"
        assert 0.0 <= s["alpha_mean"] <= 1.0 and s["T_b"] >= 1
    # FAILURE ROW (omit 1/T_b): duplicating a bag's windows leaves R_b invariant,
    # while the unnormalized sum inflates magnitude with duration.
    z, s0 = zs[0], stats[0]
    R4, _ = rd.encode_bags([torch.cat([z] * 4, dim=0)])
    assert torch.allclose(R4[0], s0["R"], atol=1e-5), "R_b not invariant to window duplication"
    from src.snn_modeling.utils.estep import two_step_baseline
    h4, lp4 = rd._bag_geometry(torch.cat([z] * 4, dim=0))
    tb4 = two_step_baseline(h4, lp4, rd.rho, rd.sigma_sq)
    R_sum = (tb4["alpha"].unsqueeze(1) * h4).sum(dim=0)               # the bug being guarded
    assert float(R_sum.norm()) > 3.0 * float(s0["R"].norm()), "duration-bias guard toothless"


# ------------------------------------------------------------------ Eq. 18 head

def test_head_matches_centroid_and_finetunes():
    payload, _, _ = make_lockpass()
    rd = Phase1BReadout(payload)
    torch.manual_seed(5)
    R = torch.randn(48, D).abs() + 0.1
    labels = torch.arange(C).repeat_interleave(16)
    fit = rd.fit_nearest_centroid(R, labels)
    assert fit["sigma_R_sq"] > 0 and fit["class_counts"] == [16, 16, 16]
    rd.init_bayes_head()
    # Eq. 18 init: w_y = rho_y mu_y / sigma^2 = R_bar_y / sigma^2 (E[R|y] = rho_y mu_y)
    assert float((rd.head.weight.detach() - rd.centroids / float(rd.sigma_R_sq)).abs().max()) < 1e-4
    Rte = torch.randn(40, D).abs() + 0.1
    assert torch.equal(rd.head_predict(Rte), rd.centroid_predict(Rte)), \
        "Eq. 18 warm-start must be decision-identical to the centroid rule"
    hist = rd.finetune(R, labels, epochs=3, lr=1e-3, lambda_wd=0.05)
    assert len(hist) == 3 and hist[-1]["loss"] <= hist[0]["loss"] + 1e-6
    _expect(ValueError, lambda: rd.finetune(R, labels, epochs=1, lambda_wd=1e-3),
            "lambda_wd below approved band accepted")
    _expect(ValueError, lambda: rd.finetune(R, labels, epochs=1, lambda_wd=0.5),
            "lambda_wd above approved band accepted")
    _expect(ValueError, lambda: rd.fit_nearest_centroid(R[:32], labels[:32]),
            "missing-class fit accepted silently")            # only classes 0,1 present
    _expect(ValueError, lambda: rd.fit_nearest_centroid(R, labels + C), "OOR labels accepted")
    _expect(ValueError, lambda: rd.encode_bags([]), "empty bag list accepted")
    st = rd.head_state()
    assert st["schema"] == HEAD_SCHEMA
    _expect(ValueError, lambda: rd.load_head_state(dict(st, schema="v0")), "bad head schema accepted")
    _expect(ValueError, lambda: rd.load_head_state(dict(st, centroids=st["centroids"] + 1.0)),
            "centroid mismatch accepted")


def test_burst_rate_bias_row():
    """FAILURE ROW (omit b_y): with a SHARED direction and rho-scaled magnitudes
    (E[R|y] = rho_y mu_y), the Eq.18 bias is what keeps burst-rate discrimination;
    a bias-free (cosine-style) head collapses to the largest-rate class."""
    payload, _, _ = make_lockpass()
    rd = Phase1BReadout(payload)
    rho = torch.tensor([0.1, 0.3, 0.9])
    e0 = torch.zeros(C, D); e0[:, 0] = rho          # mu shared (e0), rho_y scales magnitude
    labels = torch.arange(C).repeat_interleave(4)
    torch.manual_seed(3)
    R = e0.repeat_interleave(4, dim=0) + 0.01 * torch.randn(C * 4, D)
    rd.fit_nearest_centroid(R, labels)
    rd.init_bayes_head()
    probe = torch.zeros(1, D); probe[0, 0] = 0.3    # a class-1 (rho=0.3) bag
    assert int(rd.centroid_predict(probe)[0]) == 1, "centroid rule lost the middle rate"
    no_bias = (probe @ rd.head.weight.detach().T).argmax(dim=1)
    assert int(no_bias[0]) == 2, "bias-free head must (mis)collapse to the largest-rate class"
    assert int(rd.head_predict(probe)[0]) == 1, "Eq.18 head with b_y must recover the true class"
    st = rd.head_state()
    rd2 = Phase1BReadout(payload)
    rd2.fit_nearest_centroid(R, labels)
    rd2.load_head_state(st)
    assert torch.equal(rd2.head_predict(probe), rd.head_predict(probe)), "state roundtrip changed behaviour"


# ------------------------------------------------------------------ rolling buffer + end-to-end

def test_rolling_buffer_variant():
    payload, mu, off = make_lockpass()
    rd = Phase1BReadout(payload)
    zs, _ = gen_bags([0, 1], mu, off, [8, 8], seed=2)
    _expect(ValueError, lambda: rd.encode_bag(zs[0], mode="rolling_buffer"),
            "rolling mode accepted without a buffer")
    _expect(ValueError, lambda: rd.encode_bag(zs[0], mode="per_bag", buffer_z=[zs[1]]),
            "buffer silently ignored in per_bag mode")
    R_pb, _ = rd.encode_bags(zs)
    buffers = [[zs[j] for j in range(i)] if i > 0 else [zs[-1]] for i in range(len(zs))]
    R_rl, stats = rd.encode_bags(zs, mode="rolling_buffer", buffers=buffers)
    assert R_pb.shape == R_rl.shape and torch.isfinite(R_rl).all()
    for s in stats:
        assert 0.0 <= s["kappa"] < 1.0 and math.isfinite(s["bound"])


def test_run_phase_1b_end_to_end():
    payload, mu, off = make_lockpass()
    # The synthetic lockpass seeds rho at the rho_min floor (probe-verified): all
    # classes share the floor rate, so R_b carries no burst-rate signal BY DESIGN.
    # Simulate a 1A run that resolved well-separated rates instead.
    payload["rho"] = torch.tensor([0.15, 0.45, 0.85])
    rho_true = [0.15, 0.45, 0.85]
    # gain=2 + sigma_sq=0.25 raise the harness SNR: with eps~0.96 the cloud kernel
    # is deliberately flat, so unit-amplitude bursts under 0.8·(subject base)
    # barely move alpha (probe: 0.71 acc at gain=1, sigma_sq=1). sigma_sq is a
    # config knob; the pipeline itself is unchanged.
    ztr, ltr = gen_bags([0, 1, 2], mu, off, [6, 8, 10, 12], seed=11, rho_true=rho_true, gain=2.0)
    zte, lte = gen_bags([0, 1, 2, 3], mu, off, [8, 8], seed=12, rho_true=rho_true, gain=2.0)
    rep = run_phase_1b(payload, ztr, ltr, zte, lte, finetune_epochs=0, sigma_sq=0.25)
    assert rep["n_train"] == len(ztr) and rep["n_eval"] == len(zte)
    assert rep["zero_shot_acc"] >= 0.8, f"zero-train nearest-centroid acc too low: {rep}"
    assert rep["kappa_max"] < 1.0 and rep["sigma_R_sq"] > 0.0
    rep2 = run_phase_1b(payload, ztr, ltr, zte, lte, finetune_epochs=15, lambda_wd=0.05,
                        sigma_sq=0.25)
    assert rep2["finetuned_acc"] >= 0.8
    _expect(ValueError, lambda: run_phase_1b(payload, ztr, ltr, zte, lte,
                                             finetune_epochs=0, mode="rolling_buffer"),
            "rolling mode accepted without explicit buffers")


# ------------------------------------------------------------------ runner

def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
