"""Stage 6 regression tests — SMCLossEngine (sm_loss.py): Eq. 23 assembly,
bootstrap order (§11.4ii), uid composition (§9.4), lock artifact, state roundtrip.

Run from repo root:  venv\\Scripts\\python.exe tests\\test_stage6.py
Exit code 0 = all pass. (Plain asserts — the venv has no pytest.)
"""

import math
import os
import sys

import torch
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.snn_modeling.utils.sm_loss import (LOCKPASS_SCHEMA, SMCLossEngine,
                                            validate_lockpass)

C, S, D = 4, 6, 16


def _expect(exc, fn, msg):
    try:
        fn()
    except exc:
        return
    raise AssertionError(msg)


def _cfg(**over):
    cfg = dict(
        tau_norm=0.25, rank_energy_band=0.85, warmup_epochs=0, space_momentum=0.9,
        rho_min=0.02, seed=dict(s=2.0), sigma_sq=1.0, tau_dcl=0.25, anchor_on="projected",
        anneal=dict(t_start=20.0, t_end=1.0, frac=0.3),
        c=0.5, g_floor=0.1, lambda_sub=1.0, lambda_rho=0.1,
        sinkhorn=dict(tol=1e-3, T_sc=30, n_iters=800, solver_tol=1e-9),
        eps_max=1.0, n_q=256, queue_momentum=0.9, gate_ema_momentum=0.9,
    )
    cfg.update(over)
    return cfg


def _engine(**over):
    return SMCLossEngine(_cfg(**over), feature_dim=D, num_classes=C,
                         num_subjects=S, total_steps=200)


def _seeded_engine():
    torch.manual_seed(0)
    eng = _engine()
    norms = {}
    for y in range(C):
        base = (D + math.sqrt(2.0 * D) * torch.randn(240, dtype=torch.float64)).clamp(min=1e-3)
        k = int(round((0.08 + 0.06 * y) * 240))          # distinct burst rates per class
        burst = (3.0 * D + D * torch.rand(k, dtype=torch.float64)).clamp(min=1e-3)
        norms[y] = torch.cat([base, burst])
    info = eng.seed_priors(norms)
    return eng, info


def _batch(eng, step, B=24):
    """Class centres + subject offsets in a random 'subject subspace' + noise."""
    torch.manual_seed(100 + step)
    g = torch.Generator().manual_seed(7)
    centers = Fn.normalize(torch.randn(C, D, generator=g), dim=1)
    offs = Fn.normalize(torch.randn(S, D, generator=g), dim=1) * 0.9
    labels = torch.arange(B, dtype=torch.long) % C
    subjects = torch.arange(B, dtype=torch.long) % S
    noise = 0.05 * torch.randn(B, D)
    z = Fn.normalize(centers[labels] + offs[subjects] + noise, dim=1)
    key_z = Fn.normalize(centers[labels] + offs[subjects] + 0.05 * torch.randn(B, D), dim=1)
    trials = torch.arange(B, dtype=torch.long) // 4 + step * (B // 4)
    wins = torch.arange(B, dtype=torch.long) + step * B
    return z.requires_grad_(True), key_z, labels, subjects, wins, trials


# ------------------------------------------------------------------ seeding + eps gate

def test_seed_priors_interior_and_eps_admissible():
    eng, info = _seeded_engine()
    rho = eng.rho
    assert rho.shape == (C,) and torch.isfinite(rho).all()
    assert ((rho >= 0.02 - 1e-9) & (rho <= 0.98 + 1e-9)).all(), f"rho outside interiority: {rho}"
    # burst rates 0.08..0.26 -> seeded priors must sit above floor and below 0.6
    assert (rho > 0.02).all() and (rho < 0.6).all(), f"uninformative seed: {rho}"
    # B2a: resolved eps must satisfy the per-class admissibility check (seed ran it loudly)
    from src.snn_modeling.utils.sinkhorn import check_eps_admissible, dustbin_price
    for y in range(C):
        check_eps_admissible(eng.eps, eng.sn_tol, eng.sn_T_sc,
                             float(dustbin_price(eng.eps, float(rho[y]))))
    assert 0.9 < eng.eps <= 1.0, f"midpoint eps {eng.eps} off Thm 12.2 interval"


def test_forward_requires_seed():
    eng = _engine()
    z, kz, lb, sb, w, tr = _batch(eng, 0)
    _expect(RuntimeError, lambda: eng.forward(z, kz, lb, sb, w, tr), "forward ran on unseeded rho")


def test_eps_too_small_rejected_at_seed():
    eng = _engine()
    eng.eps = 0.01  # below the ~0.92 B2a bound: seed_priors must raise, not train
    norms = {y: (D + torch.rand(100, dtype=torch.float64)) for y in range(C)}
    _expect(ValueError, lambda: eng.seed_priors(norms), "inadmissible eps accepted at seed")


# ------------------------------------------------------------------ bootstrap order §11.4ii

def test_step_zero_bootstrap_order():
    eng, _ = _seeded_engine()
    rho0 = eng.rho.detach().clone()          # rho_mstep updates the buffer INSIDE forward
    z, kz, lb, sb, w, tr = _batch(eng, 0)
    out = eng.forward(z, kz, lb, sb, w, tr)
    d = out["diagnostics"]
    assert d["cold_anchor"] and d["cold_em"], "step 0 must have no negatives/clouds"
    assert d["gate"] == eng.g_floor and d["lambda_ot"] == 0.0, "gate must sit at floor when cold"
    assert torch.allclose(out["alpha"].detach(), rho0[lb]), \
        "cold E-step must fall back to the seeded rho"
    assert math.isfinite(float(out["total"])) and out["total"].requires_grad
    out["total"].backward()
    assert z.grad is not None and torch.isfinite(z.grad).all() and float(z.grad.norm()) > 0.0, \
        "L_sub must move the encoder even while anchor/OT are cold"
    # side effects: queue now holds step 0, rho moved off seed via Eq. 12, step advanced
    assert int(eng.queue.filled.item()) == 24 and int(eng.step.item()) == 1


def test_anchor_and_emotion_warm_in_after_first_step():
    eng, _ = _seeded_engine()
    for t in range(3):
        z, kz, lb, sb, w, tr = _batch(eng, t)
        out = eng.forward(z, kz, lb, sb, w, tr)
        d = out["diagnostics"]
        if t == 0:
            continue
        assert not d["cold_anchor"], "queue keys from step 0 must serve step 1+ negatives"
        assert d["n_negatives"] > 0 and 0.0 <= d["gate"] <= 1.0
        assert d["lambda_ot"] >= 0.0 and math.isfinite(float(out["total"]))
    # clouds filled with class-balanced queue -> E-step mostly hot:
    assert d["cold_alpha_rows"] == 0, f"all classes in queue yet {d['cold_alpha_rows']} cold rows"
    assert 0.0 < d["alpha_std"], "alpha collapsed to the prior after clouds exist (Table-1 #4)"


def test_diagnostics_carry_table1_fields():
    eng, _ = _seeded_engine()
    for t in range(4):
        z, kz, lb, sb, w, tr = _batch(eng, t)
        out = eng.forward(z, kz, lb, sb, w, tr)
    d = out["diagnostics"]
    for key in ("T", "gate", "lambda_ot", "C_K", "D_norm", "ess", "queue", "space", "eps"):
        assert key in d, f"missing Table-1 diagnostic {key}"
    assert d["T"] <= eng.t_anneal and d["T"] >= eng.t_end
    assert all(v >= 0.0 for v in d["ess"].values())


# ------------------------------------------------------------------ uids + fail-loud

def test_uid_composition_guards():
    eng, _ = _seeded_engine()
    z, kz, lb, sb, w, tr = _batch(eng, 0)
    _expect(ValueError, lambda: eng.forward(z, kz, lb, sb, w - 1, tr),
            "padded window id (-1) reached the queue")
    _expect(ValueError, lambda: eng.forward(z, kz, lb, sb, torch.zeros_like(w), tr),
            "duplicate window ids within batch accepted")
    _expect(ValueError, lambda: eng.forward(z, kz, lb, sb, w, torch.full_like(tr, -1)),
            "negative trial id accepted")
    _expect(ValueError, lambda: eng.forward(z, kz, lb, sb, w + (2 ** 20), tr),
            "window id exceeding UID_SHIFT silently folded")
    _expect(ValueError, lambda: eng.forward(z[:0], kz[:0], lb[:0], sb[:0], w[:0], tr[:0]),
            "empty batch accepted (B14)")
    _expect(ValueError, lambda: eng.forward(z, kz, lb + C, sb, w, tr),
            "out-of-range label accepted")


def test_loto_ghosting_is_trial_scoped():
    # negatives from the query's own trial must be ghosted: same-window masking alone
    # would still let OTHER windows of the current bag act as negatives.
    eng, _ = _seeded_engine()
    z, kz, lb, sb, w, tr = _batch(eng, 0)          # trials 0..5, 4 windows each
    eng.forward(z, kz, lb, sb, w, tr)              # fills queue with step-0 keys
    z2, kz2, lb2, sb2, w2, tr2 = _batch(eng, 1)   # fresh trials
    out = eng.forward(z2, kz2, lb2, sb2, w2, tr2)
    assert out["diagnostics"]["n_negatives"] == 24  # all step-0 keys survive (disjoint trials)
    # re-run step-1 batch again: its own windows excluded, but its own trials ghosted too
    z3, kz3, lb3, sb3, w3, tr3 = _batch(eng, 1)
    out3 = eng.forward(z3, kz3, lb3, sb3, w3, tr3)
    # queue now holds steps 0,1,1': only the 24 step-0 keys remain usable (steps 1/1' share trials with the query)
    assert out3["diagnostics"]["n_negatives"] <= 24 + 24, "LOTO ghosting must bound same-trial negatives"


# ------------------------------------------------------------------ state + lock

def test_state_roundtrip_forward_identical():
    eng, _ = _seeded_engine()
    for t in range(2):
        z, kz, lb, sb, w, tr = _batch(eng, t)
        eng.forward(z, kz, lb, sb, w, tr)
    sd = {k: v.clone() if torch.is_tensor(v) else v for k, v in eng.state_dict().items()}
    sm = eng.sm_state()
    fresh = _engine()
    fresh.load_state_dict(sd)
    fresh.load_sm_state(sm)
    assert torch.equal(fresh.rho, eng.rho) and int(fresh.step) == int(eng.step)
    z, kz, lb, sb, w, tr = _batch(eng, 2)
    a = eng.forward(z.clone().requires_grad_(True), kz, lb, sb, w, tr)["total"].detach()
    b = fresh.forward(z.clone().requires_grad_(True), kz, lb, sb, w, tr)["total"].detach()
    assert torch.allclose(a, b, atol=1e-7), "restored engine must reproduce the loss exactly"


def test_lock_fullpass_schema():
    eng, _ = _seeded_engine()
    for t in range(5):
        z, kz, lb, sb, w, tr = _batch(eng, t)
        eng.forward(z, kz, lb, sb, w, tr)
    lock = eng.lock_fullpass()
    assert lock["schema"] == LOCKPASS_SCHEMA and validate_lockpass(lock) is lock
    assert lock["rho"].shape == (C,) and (lock["rho"] > 0).all() and (lock["rho"] < 1).all()
    assert lock["clouds"], "1A ran 5 steps and still has no clouds"
    for y, c in lock["clouds"].items():
        assert abs(float(c["weights"].sum()) - 1.0) < 1e-4 and c["keys"].shape[1] == D
    # unseeded engine refuses to emit a NaN-rho artifact
    _expect(RuntimeError, lambda: _engine().lock_fullpass(), "unseeded lockpass allowed")
    bad = dict(lock)
    bad["rho"] = torch.full((C,), float("nan"))
    _expect(ValueError, lambda: validate_lockpass(bad), "NaN rho lock accepted")


# ------------------------------------------------------------------ §9.6 / §9.4 init asserts

def test_sizing_and_staleness_init_checks():
    eng, _ = _seeded_engine()          # n_q = 256
    floor, ceiling = eng.run_sizing_checks(batch_size=24, epochs=10, steps_per_epoch=50,
                                           L_const=1.0, d_eff=8.0, tol_eff=2.5,
                                           C_const=4.0, eps_tol=20.0)
    assert floor <= 256 <= ceiling, f"engine's own queue {256} outside [{floor}, {ceiling}]"
    _expect(ValueError, lambda: eng.run_sizing_checks(
        batch_size=24, epochs=10, steps_per_epoch=50, L_const=1.0, d_eff=64.0,
        tol_eff=0.1, C_const=4.0, eps_tol=10.0), "infeasible sizing accepted (§9.6 gate)")


def test_bad_config_inputs_rejected():
    _expect(ValueError, lambda: _engine(anchor_on="cosface"), "unknown anchor_on accepted")
    _expect(ValueError, lambda: _engine(c=1.0), "c>=1 accepted (B5iii)")
    _expect(ValueError, lambda: _engine(g_floor=1.0), "g_floor>=1 accepted")
    _expect(ValueError, lambda: _engine(eps=0.0), "eps<=0 accepted")
    _expect(ValueError, lambda: _engine(eps=2.0), "eps above eps_max accepted")
    _expect(ValueError, lambda: _engine(anneal=dict(t_start=1.0, t_end=20.0, frac=0.3)),
            "inverted anneal accepted")
    _expect(ValueError, lambda: SMCLossEngine(_cfg(), feature_dim=D, num_classes=C,
                                             num_subjects=S, total_steps=0), "total_steps=0 accepted")


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
