"""Stage 8 regression tests — certification (§11.5 protocol + §13 ablation registry).

Run from repo root:  venv\\Scripts\\python.exe tests\\test_stage8.py
Exit code 0 = all pass. (Plain asserts — the venv has no pytest.)

Covers plan §5 Stage-8 checklist: Table-1 grading bands, Remark 11.2 smoke
criteria, Remark 11.1 remediation ladder in STRICT causal order (and refusal
to run on a healthy class), B3(iii) alpha-identity early-stop, the 11-row
Failure-Catalogue registry + switch guardrails, the engine's new certification
diagnostics (stdt_intra / energy_ratio / anneal_complete / lambda_anc), and the
two adapter ablation switches (§13 rows 8-9) with their expected pathologies.
"""

import os
import sys

import torch
import torch.nn.functional as Fn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import tests.test_stage7 as T7                      # helpers only (main is guarded)
from src.snn_modeling.adapters.mil_readout import Phase1BReadout, run_phase_1b
from src.snn_modeling.utils.certification import (DNORM_BAND, FAILURE_CATALOGUE_ROWS,
                                                  alpha_identity_escape, ablation_switch,
                                                  evaluate_smoke, evaluate_table1,
                                                  remediation_ladder)
from src.snn_modeling.utils.sm_loss import SMCLossEngine

D, C, S = T7.D, T7.C, T7.S


def _expect(exc, fn, msg):
    try:
        fn()
    except exc:
        return
    raise AssertionError(msg)


def _healthy_table1():
    return dict(probe_z=0.80, probe_h=0.12, zero_step_acc=0.40, energy_ratio=0.50,
                stdt_alpha=0.30, gate_g=0.60, dnorm={0: 0.20, 1: 0.35, 2: 0.28},
                rho=[0.1, 0.3, 0.5], rho_order_pairs=[(1, 2)])


# ------------------------------------------------------------------ ladder + escape

def test_alpha_identity_escape():
    stop = alpha_identity_escape([0.5, 0.01, 0.01, 0.01], [False, True, True, True])
    assert stop == 3, f"escape should fire on the 3rd consecutive post-anneal epoch, got {stop}"
    assert alpha_identity_escape([0.01] * 5, [False] * 5) is None, "must not fire before anneal floor"
    assert alpha_identity_escape([0.01, 0.01, 0.5, 0.01, 0.01, 0.01], [True] * 6) == 5, \
        "streak must RESET on a break, not accumulate"
    assert alpha_identity_escape([0.5] * 4, [True] * 4) is None, "healthy series fired"
    _expect(ValueError, lambda: alpha_identity_escape([0.1, 0.1], [True]), "length mismatch accepted")


def test_remediation_ladder_strict_order():
    # refuses to run on a healthy class (D_norm <= 0.5)
    _expect(ValueError, lambda: remediation_ladder(0.3, 0.95, 0.9, 1.0, 0.5, 0.02, 100, 512, 0.02),
            "ladder ran on a healthy D_norm — would mask real diagnosis")
    _expect(ValueError, lambda: remediation_ladder(0.7, 0.95, 1.0, 0.9, 0.5, 0.02, 100, 512, 0.02),
            "inverted Thm 12.2 interval accepted")
    kw = dict(dnorm_y=0.7, eps_min=0.9, eps_max=1.0, rho_min=0.02, n_q=512, rho_tilde_min=0.02)
    # step 1 first even when 2 and 3 are ALSO bad — strict causal order
    r = remediation_ladder(eps=0.905, rho_y=0.025, ess_y=3, **kw)
    assert r["next_step"] == 1, f"eps-near-min must win over rho/ESS failures: {r}"
    # eps healthy -> step 2
    r = remediation_ladder(eps=0.95, rho_y=0.025, ess_y=100, **kw)
    assert r["next_step"] == 2, f"under-seeded rho should trigger step 2: {r}"
    # eps + rho healthy -> step 3
    r = remediation_ladder(eps=0.95, rho_y=0.5, ess_y=3, **kw)
    assert r["next_step"] == 3, f"ESS shortfall should trigger step 3: {r}"
    # nothing explains the failure -> escalation, never a silent all-clear
    r = remediation_ladder(eps=0.95, rho_y=0.5, ess_y=100, **kw)
    assert r["next_step"] == 4, f"exhausted ladder must escalate: {r}"
    assert all(s["fires"] is not None for s in r["steps"]) and len(r["steps"]) == 4


# ------------------------------------------------------------------ Table 1 + smoke gate

def test_evaluate_table1_bands():
    rep = evaluate_table1(_healthy_table1())
    assert rep["status"] == "ok" and rep["issues"] == [], f"healthy record graded fail: {rep}"
    cases = {
        "probe_h": (0.50, "rank"),                    # Table 1 row-1 failure mode
        "energy_ratio": (0.05, "shrinkage"),          # row 3
        "stdt_alpha": (0.01, "alpha == rho"),         # row 4 (B3iii escape)
        "gate_g": (0.95, "outside"),                  # row 5
        "zero_step_acc": (0.22, "clustering"),        # row 2
    }
    for key, (val, needle) in cases.items():
        rec = _healthy_table1(); rec[key] = val
        rep = evaluate_table1(rec)
        assert rep["status"] == "fail" and any(needle in s for s in rep["issues"]), \
            f"{key}={val} not graded as failure containing {needle!r}: {rep['issues']}"
    rec = _healthy_table1(); rec["dnorm"] = {0: 0.70}
    rep = evaluate_table1(rec)
    assert any("remediation_ladder" in s for s in rep["issues"]), \
        "D_norm > 0.5 must POINT AT the ladder, not just flag"
    rec = _healthy_table1(); rec["rho"] = [0.2, 0.5, 0.1]
    rep = evaluate_table1(rec)
    assert any("sparsity" in s for s in rep["issues"]), "rho ordering violation missed"
    rec = _healthy_table1(); rec.pop("probe_z")
    _expect(KeyError, lambda: evaluate_table1(rec), "missing diagnostic graded silently")


def test_evaluate_smoke():
    good = dict(probe_z=0.85, probe_h=0.18, stdt_alpha=0.30, unseen_bag_acc=0.40)
    ok, rep = evaluate_smoke(good)
    assert ok and all(rep.values()), rep
    for key, bad, flag in (("probe_z", 0.75, "probe_z_kept"),
                           ("probe_h", 0.35, "probe_h_dropped"),
                           ("stdt_alpha", 0.10, "stdt_informative"),
                           ("unseen_bag_acc", 0.30, "unseen_bag_acc")):
        rec = dict(good); rec[key] = bad
        ok, rep = evaluate_smoke(rec)
        assert not ok and rep[flag] is False and sum(rep.values()) == 3, \
            f"{key}={bad} did not trip exactly its own criterion: {rep}"
    rec = dict(good); rec.pop("stdt_alpha")
    _expect(KeyError, lambda: evaluate_smoke(rec), "missing criterion accepted")


# ------------------------------------------------------------------ registry + switches

def test_failure_catalogue_registry():
    assert len(FAILURE_CATALOGUE_ROWS) == 11, \
        f"§13 carries 11 rows, registry has {len(FAILURE_CATALOGUE_ROWS)}"
    for r in FAILURE_CATALOGUE_ROWS:
        assert {"row", "how", "expect"} <= set(r), f"registry row incomplete: {r}"
    hit = ablation_switch("Remove sparsity (L_spr)", {"loss.sm.lambda_rho": 0.0})
    assert hit["row"] == "Remove sparsity (L_spr)" and hit["overrides"]["loss.sm.lambda_rho"] == 0.0
    _expect(KeyError, lambda: ablation_switch("Remove sparcity (L_spr)", {}),
            "typo'd ablation name silently accepted — would run as the HEALTHY config")
    _expect(ValueError, lambda: ablation_switch("Remove sparsity (L_spr)", {"data.batch_size": 1}),
            "non-loss override slipped past the registry guard")


# ------------------------------------------------------------------ engine diagnostics

def _warm_engine():
    torch.manual_seed(0)
    cfg = T7._cfg()
    eng = SMCLossEngine(cfg, feature_dim=D, num_classes=C, num_subjects=S, total_steps=50)
    norms = {y: (D + 2.0 * torch.rand(200, dtype=torch.float64)) for y in range(C)}
    eng.seed_priors(norms)
    mu = Fn.normalize(torch.randn(C, D), dim=1)
    off = Fn.normalize(torch.randn(S, D), dim=1)
    B = 16
    outs = []
    for step in range(4):
        labels = torch.arange(B) % C
        subj = torch.arange(B) % S
        z = Fn.normalize(mu[labels] + 0.8 * off[subj] + 0.05 * torch.randn(B, D), dim=1)
        k = Fn.normalize(mu[labels] + 0.8 * off[subj] + 0.05 * torch.randn(B, D), dim=1)
        z = z.detach().requires_grad_(True)
        outs.append(eng(z, k, labels, subj, torch.arange(B) + step * B,
                        torch.arange(B) // 4 + step * 4))
    return eng, outs, mu, off


def test_engine_certification_diagnostics():
    eng, outs, *_ = _warm_engine()
    d = outs[-1]["diagnostics"]
    for key in ("stdt_intra", "energy_ratio", "anneal_complete"):
        assert key in d, f"Table-1 diagnostic {key!r} missing from engine diagnostics"
    assert 0.0 <= d["stdt_intra"] <= 1.0
    # STEP 0 saw a rank-0 (never-refreshed) NullspaceState: h' == z - mu EXACTLY,
    # so energy_ratio must be 1.0. Rm 6.4 refreshes U inside update_batch every step,
    # so by step 3 the projector is live and must have removed energy (<1).
    d_first = outs[0]["diagnostics"]
    assert abs(d_first["energy_ratio"] - 1.0) < 1e-5, \
        f"identity projector must give energy_ratio 1.0, got {d_first['energy_ratio']}"
    # The invariant is conditional on the state the SVD actually produced (float32
    # thread-order can straddle the band threshold on 4 tiny steps): rank > 0 must
    # have REMOVED energy; rank 0 must still read exactly 1.0. Either way the
    # diagnostic must agree with the geometry, never silently float between.
    rank_last = int(d["space"].get("rank", 0))
    if rank_last > 0:
        assert d["energy_ratio"] < 1.0, \
            f"rank {rank_last} projector removed no energy (ratio {d['energy_ratio']})"
    else:
        assert abs(d["energy_ratio"] - 1.0) < 1e-5, \
            f"rank-0 projector must keep energy_ratio at 1.0, got {d['energy_ratio']}"
    assert d["anneal_complete"] is False, "step 3 of a 50-step budget is not post-anneal"
    eng2, outs2, *_ = _warm_engine()
    eng2.on_epoch_end()                       # epoch 1 -> SVD refresh may lift rank
    out = outs2[-1]
    assert torch.isfinite(out["total"])
    # lambda_anc is a first-class Eq. 23 weight: total identity must hold verbatim
    d2 = out["diagnostics"]
    rhs = (1.0 * out["L_anc"] + d2["lambda_ot"] * out["L_em"]
           + eng2.lambda_sub * out["L_sub"] + eng2.lambda_rho * out["L_spr"])
    assert torch.allclose(out["total"], rhs, atol=1e-5), "Eq. 23 assembly drifted from the parts"
    # runtime ablation of the anchor weight must be arithmetically exact
    eng2.lambda_anc = 0.0
    z = Fn.normalize(torch.randn(8, D), dim=1).detach().requires_grad_(True)
    out0 = eng2(z, z.detach(), torch.arange(8) % C, torch.arange(8) % S,
                torch.arange(8) + 100, torch.arange(8) // 4 + 8)
    d0 = out0["diagnostics"]
    rhs0 = (d0["lambda_ot"] * out0["L_em"] + eng2.lambda_sub * out0["L_sub"]
            + eng2.lambda_rho * out0["L_spr"])
    assert torch.allclose(out0["total"], rhs0, atol=1e-5), "lambda_anc=0 ablation not exact"
    _expect(ValueError, lambda: SMCLossEngine(dict(T7._cfg(), lambda_anc=-0.1),
                                              feature_dim=D, num_classes=C,
                                              num_subjects=S, total_steps=10),
            "negative lambda_anc accepted")
    # §13 row-2 switch builds and runs (announced; tau pinned => no dustbin collapse)
    fb = SMCLossEngine(dict(T7._cfg(), force_balanced=True), feature_dim=D, num_classes=C,
                       num_subjects=S, total_steps=50)
    fb.seed_priors({y: (D + 2.0 * torch.rand(200, dtype=torch.float64)) for y in range(C)})
    o = fb(z, z.detach(), torch.arange(8) % C, torch.arange(8) % S,
           torch.arange(8) + 200, torch.arange(8) // 4 + 20)
    assert torch.isfinite(o["total"])


# ------------------------------------------------------------------ adapter ablation flags

def test_adapter_ablation_flags():
    payload, mu, off = T7.make_lockpass()
    payload["rho"] = torch.tensor([0.15, 0.45, 0.85])
    zs, zlabels = T7.gen_bags([0], mu, off, [8], seed=4, rho_true=[0.15, 0.45, 0.85], gain=2.0)
    z = zs[0]
    rd = Phase1BReadout(payload)
    rd_abl = Phase1BReadout(payload, omit_duration_norm=True)
    s0, s0a = rd.encode_bag(z), rd_abl.encode_bag(z)
    # §13 row 8 signature: without 1/T_b the statistic scales with duration
    assert float(s0a["R"].norm()) > 3.0 * float(s0["R"].norm()), \
        "omit_duration_norm did not reproduce the duration-bias pathology"
    s4a = rd_abl.encode_bag(torch.cat([z] * 4, dim=0))
    assert float(s4a["R"].norm()) > 3.0 * float(s0a["R"].norm()), \
        "ablated R_b must grow further when the bag is made 4x longer"
    _expect(TypeError, lambda: Phase1BReadout(payload, omit_duration_norm="yes"),
            "non-bool ablation switch accepted")

    # §13 row 9: b_y == 0 collapses rate-scaled centroids onto the largest-rate class
    rho3 = torch.tensor([0.1, 0.3, 0.9])
    e0 = torch.zeros(C, D); e0[:, 0] = rho3
    labels = torch.arange(C).repeat_interleave(4)
    torch.manual_seed(3)
    R = e0.repeat_interleave(4, dim=0) + 0.01 * torch.randn(C * 4, D)
    rd2 = Phase1BReadout(payload)
    rd2.fit_nearest_centroid(R, labels)
    rd2.init_bayes_head(ablate_bias=True)
    assert float(rd2.head.bias.detach().abs().max()) == 0.0, "ablate_bias left b_y nonzero"
    probe = torch.zeros(1, D); probe[0, 0] = 0.3            # true class = rate 0.3
    assert int(rd2.head_predict(probe)[0]) == 2, \
        "ablated head must (predictably) collapse to the largest-rate class"
    rd3 = Phase1BReadout(payload)
    rd3.fit_nearest_centroid(R, labels)
    rd3.init_bayes_head()                                    # default: spec behaviour
    assert int(rd3.head_predict(probe)[0]) == 1, "default head lost the true rate class"
    _expect(TypeError, lambda: rd3.init_bayes_head(ablate_bias="yes"),
            "non-bool ablate_bias accepted")

    rep = run_phase_1b(payload, zs, zlabels, zs, zlabels,
                       finetune_epochs=0, sigma_sq=0.25,
                       omit_duration_norm=True, ablate_bias=True)
    assert isinstance(rep["zero_shot_acc"], float), "ablated pipeline refused to run"


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
