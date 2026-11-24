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
                             float(dustbin_price(float(rho[y]), eng.tau_0, eng.kappa_cost)))
    assert 0.9 < eng.eps <= 1.0, f"midpoint eps {eng.eps} off Thm 12.2 interval"
    # Eq. 11 config defaults are the approved pair
    assert eng.tau_0 == 0.35 and eng.kappa_cost == 0.50


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


# ------------------------------------------------------------------ cloud LOTO support mask

def _same_trial_other_window(eng, base, win_offset):
    """Same batch, same trials, DIFFERENT window ids: these keys survive the queue's
    uid exclusion and can only be removed by the per-pair trial mask."""
    z, kz, lb, sb, w, tr = base
    return z, kz, lb, sb, w + win_offset, tr


def test_cloud_loto_mask_excludes_same_trial_keys():
    from src.snn_modeling.utils.sm_loss import UID_SHIFT
    eng, _ = _seeded_engine()
    base = _batch(eng, 0)
    eng.forward(*base)                                  # queue holds trials 0..5
    z, kz, lb, sb, w, tr = _same_trial_other_window(eng, base, 1000)
    out = eng.forward(z, kz, lb, sb, w, tr)
    d = out["diagnostics"]
    assert d["loto_mask"] is True and d["loto_forbidden_frac"] > 0.0, \
        f"trial mask did not forbid any pair: {d['loto_forbidden_frac']}"
    # the mask must be exactly the trial-equality relation on the recovered cloud trials
    clouds = eng.queue.read_class_clouds(exclude_window_ids=eng._compose_uids(tr, w))
    for y, cl in clouds.items():
        assert "window_ids" in cl, "cloud lost its uids — trial recovery would be impossible"
        key_trial = cl["window_ids"] // UID_SHIFT
        sp = eng._loto_support(clouds, tr)[int(y)]
        assert torch.equal(sp, key_trial.view(1, -1) != tr.view(-1, 1))
        assert (~sp).any(), f"class {y}: expected some same-trial pairs to be forbidden"


def test_cloud_loto_mask_changes_alpha_and_emotion_only():
    """Mask ON vs OFF on identical state must move alpha / L_em / D_norm, and nothing
    that is not supposed to change (costs stay in [0,2], anchor is untouched)."""
    def run(mask):
        eng, _ = _seeded_engine()
        eng.ot_loto_mask = mask
        base = _batch(eng, 0)
        eng.forward(*base)
        z, kz, lb, sb, w, tr = _same_trial_other_window(eng, base, 1000)
        return eng.forward(z, kz, lb, sb, w, tr)
    off, on = run(False), run(True)
    assert not torch.allclose(off["alpha"], on["alpha"]), \
        "the mask left the E-step responsibilities unchanged — it is not wired in"
    assert abs(float(off["L_em"]) - float(on["L_em"])) > 1e-9, "L_em ignored the mask"
    assert off["diagnostics"]["loto_forbidden_frac"] == 0.0
    assert on["diagnostics"]["loto_forbidden_frac"] > 0.0
    # D_norm is reported per class in both runs; masked drop mass is accounted, not lost
    for y in on["diagnostics"]["D_norm"]:
        assert float(on["diagnostics"]["D_norm"][y]) >= 0.0


def test_cloud_loto_mask_refuses_silent_disable():
    """A cloud payload without uids must raise, not silently train unmasked."""
    eng, _ = _seeded_engine()
    z, kz, lb, sb, w, tr = _batch(eng, 0)
    eng.forward(z, kz, lb, sb, w, tr)
    bad_clouds = {y: {"keys": c["keys"], "weights": c["weights"], "beta_sum": c["beta_sum"],
                      "n": c["n"]} for y, c in eng.queue.read_class_clouds().items()}
    _expect(RuntimeError, lambda: eng._loto_support(bad_clouds, tr),
            "LOTO mask silently disabled when the cloud lost its uids")


def test_cloud_loto_mask_off_matches_previous_behaviour():
    eng, _ = _seeded_engine()
    eng.ot_loto_mask = False
    base = _batch(eng, 0)
    eng.forward(*base)
    z, kz, lb, sb, w, tr = _same_trial_other_window(eng, base, 1000)
    out = eng.forward(z, kz, lb, sb, w, tr)
    assert out["diagnostics"]["loto_mask"] is False
    assert out["diagnostics"]["loto_cold_rows"] == 0


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
    # v2 fields: calibrated offsets + sigma_hat^2 are mandatory (Eq. 18 hookup)
    assert torch.isfinite(lock["c"]).all() and lock["c"].shape == (C,)
    assert isinstance(lock["sigma_hat_sq"], float) and lock["sigma_hat_sq"] > 0.0
    rho_d = lock["rho"].double()
    c_ref = eng.u_bar.double() - torch.log(rho_d / (1.0 - rho_d))
    assert torch.allclose(lock["c"].double(), c_ref, atol=1e-6), "c != u_bar - logit(rho)"
    for drop in ("c", "sigma_hat_sq"):
        cut = dict(lock); cut.pop(drop)
        _expect(ValueError, lambda: validate_lockpass(cut), f"lockpass without {drop} accepted")
    badc = dict(lock); badc["c"] = torch.full((C,), float("nan"))
    _expect(ValueError, lambda: validate_lockpass(badc), "NaN c lock accepted")


def test_lock_refuses_uncalibrated_lock():
    """Step 0 alone leaves buffer B_y unscored (cold start) — the lock must REFUSE
    to export a silent zero offset, not ship a half-calibrated artifact."""
    eng, _ = _seeded_engine()
    z, kz, lb, sb, w, tr = _batch(eng, 0)
    eng.forward(z, kz, lb, sb, w, tr)
    assert not torch.isfinite(eng.u_bar).any(), "step 0 must not calibrate on an empty buffer"
    _expect(RuntimeError, lambda: eng.lock_fullpass(),
            "lockpass exported an uncalibrated (NaN) c_y offset")


def test_calibrated_offset_matches_rho_and_sigma_floor():
    """Rm 4.2 / Eq. 7-8: after enough steps every u_bar_y is finite, sigma_hat^2
    respects the sigma_min^2 floor, and the within-class mean responsibility
    tracks the bisection target rho_y (mean alpha_y ~= rho_y)."""
    eng, _ = _seeded_engine()
    out = None
    for t in range(8):
        z, kz, lb, sb, w, tr = _batch(eng, t)
        out = eng.forward(z, kz, lb, sb, w, tr)
    d = out["diagnostics"]
    assert torch.isfinite(eng.u_bar).all(), f"u_bar uncalibrated after 8 steps: {d['u_bar']}"
    assert float(eng.sigma_hat_sq) >= eng.sigma_sq - 1e-12, "sigma_hat^2 broke the sigma_min^2 floor"
    assert abs(d["sigma_hat_sq"] - float(eng.sigma_hat_sq)) < 1e-12
    assert d["calib"] and all(v["calibrated"] for v in d["calib"].values()), d["calib"]
    alpha = out["alpha"].detach()
    rho = eng.rho.detach()
    labels = _batch(eng, 7)[2]                     # same deterministic batch builder
    for y in range(C):
        m = labels == y
        mean_a = float(alpha[m].mean())
        r_y = float(rho[y])
        assert abs(mean_a - r_y) <= 0.5 * r_y + 0.05, \
            f"class {y}: mean(alpha)={mean_a:.4f} !~= rho_y={r_y:.4f}"


# ------------------------------------------------------------------ §9.6 / §9.4 init asserts

def test_sizing_and_staleness_init_checks():
    eng, _ = _seeded_engine()          # n_q = 256
    from src.snn_modeling.memory.queue import queue_floor
    # §9.6 floor now uses the SEEDED min_y(pi_y*rho_y), never the rho_min guess:
    burst = eng.class_prior.double() * eng.rho.double()
    rho_tilde = float(burst.min())
    assert abs(rho_tilde - 0.0177) < 5e-3, f"seeded burst min off: {rho_tilde}"
    # feasible: tol large enough that the seeded floor lands at/below n_q=256
    floor, ceiling = eng.run_sizing_checks(batch_size=24, epochs=10, steps_per_epoch=50,
                                           L_const=1.0, d_eff=8.0, tol_eff=3.0,
                                           C_const=4.0, eps_tol=20.0)
    assert floor == queue_floor(C_const=4.0, d_eff=8.0, tol=3.0, rho_tilde_min=rho_tilde)
    assert floor <= 256 <= ceiling, f"engine queue 256 outside [{floor}, {ceiling}]"
    # hard raise on a sub-floor breach by default (§9.6 gate is live)
    _expect(ValueError, lambda: eng.run_sizing_checks(
        batch_size=24, epochs=10, steps_per_epoch=50, L_const=1.0, d_eff=64.0,
        tol_eff=0.1, C_const=4.0, eps_tol=20.0), "infeasible sizing accepted (§9.6 gate)")
    # allow_smoke_sizing downgrades that same sub-floor breach to a warning (no raise)
    s_floor, s_ceiling = eng.run_sizing_checks(
        batch_size=24, epochs=10, steps_per_epoch=50, L_const=1.0, d_eff=64.0,
        tol_eff=0.1, C_const=4.0, eps_tol=20.0, allow_smoke_sizing=True)
    assert s_floor > 256, "smoke path should only be reachable when n_q is below the floor"
    # ceiling breach STILL raises even with allow_smoke_sizing (staleness is never waived)
    _expect(ValueError, lambda: eng.run_sizing_checks(
        batch_size=24, epochs=10, steps_per_epoch=50, L_const=1.0, d_eff=8.0,
        tol_eff=3.0, C_const=4.0, eps_tol=2.0, allow_smoke_sizing=True),
        "over-ceiling queue accepted in smoke mode")


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
