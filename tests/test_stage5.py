"""Stage 5 regression tests — dcl.py + gates.py (fixed.pdf §7.4-7.5, §8.3, §9.7 Eq.13, B5(iii), 12.4iv/12.5i).

Run from repo root:  venv\\Scripts\\python.exe tests\\test_stage5.py
Exit code 0 = all pass. (Plain asserts — the venv has no pytest.)

Plan §5 Stage-5 checklist: gate opens iff the encoder becomes informative;
constant encoder ⇒ C_K = 0 ⇒ g at the floor (g_floor=0 ⇒ OT inert, Thm 12.5(i));
clamp monotone / Lipschitz 1/log(K-1) (Prop 12.4iv); decoupled partition (pos
excluded from denominator); LOTO ghosting with correct per-query counts;
EMA + lambda arithmetic; c <= c_max < 1 guard; fail-loud everywhere.
"""

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.snn_modeling.utils.dcl import ProjectedAnchor, dcl_loss_certificate
from src.snn_modeling.utils.gates import OTCoefficient, check_c_gate, grad_norm, readiness_gate


def _expect(exc, fn, msg):
    try:
        fn()
    except exc:
        return
    raise AssertionError(msg)


def _f64(*shape, seed=0):
    torch.manual_seed(seed)
    return torch.rand(*shape, dtype=torch.float64)


# ------------------------------------------------------------------ DCL math

def test_dcl_hand_computed():
    q = torch.tensor([[1.0, 0.0], [0.8, 0.6]], dtype=torch.float64)
    k = q.clone()
    neg = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    tau = 1.0
    L0 = torch.logsumexp(torch.tensor([1.0, 0.0], dtype=torch.float64), 0) - 1.0
    L1 = torch.logsumexp(torch.tensor([0.8, 0.6], dtype=torch.float64), 0) - 1.0
    C0 = math.log(2) - float(L0)
    C1 = math.log(2) - float(L1)
    res = dcl_loss_certificate(q, k, neg, tau)
    assert abs(float(res["L_DCL"]) - (float(L0) + float(L1)) / 2) < 1e-10
    assert abs(float(res["C_K"]) - (C0 + C1) / 2) < 1e-10
    assert res["M"] == 2 and res["K"] == 3 and res["n_active"] == 2
    assert torch.allclose(res["per_query_L"], torch.stack([L0, L1]), atol=1e-12)


def test_dcl_decoupled_partition():
    # positive is NOT in the denominator: L_decoupled < L_coupled for the same data
    q = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    k = q.clone()
    neg = torch.tensor([[0.5, 0.5 ** 0.5], [-0.5, 0.5 ** 0.5]], dtype=torch.float64)
    res = dcl_loss_certificate(q, k, neg, tau=1.0)
    lse_dec = torch.logsumexp(torch.tensor([0.5, -0.5], dtype=torch.float64), 0)
    lse_coup = torch.logsumexp(torch.tensor([1.0, 0.5, -0.5], dtype=torch.float64), 0)
    L_dec = float(lse_dec - 1.0)
    L_coup = float(lse_coup - 1.0)
    assert abs(float(res["L_DCL"]) - L_dec) < 1e-12, "DCL must use the decoupled (§7.4) partition"
    assert res["L_DCL"].item() < L_coup - 1e-6


def test_dcl_loto_ghosting():
    q = _f64(2, 3, seed=1)
    q = F.normalize(q, dim=1)
    k = F.normalize(_f64(2, 3, seed=2), dim=1)
    neg = F.normalize(_f64(3, 3, seed=3), dim=1)
    qg = torch.tensor([5, 7], dtype=torch.long)
    ng = torch.tensor([5, 9, 5], dtype=torch.long)   # query0 keeps 1 neg, query1 keeps 3
    res = dcl_loss_certificate(q, k, neg, tau=1.0, query_group=qg, neg_group=ng)
    assert res["M_i"].tolist() == [1, 3] and res["n_active"] == 2
    s0 = float((q[0] * k[0]).sum())
    L0 = float(torch.logsumexp((q[0] @ neg[1].unsqueeze(1)) * 1.0, 0)) - s0  # only col 1 survives
    C0 = math.log(1) - L0
    assert abs(float(res["per_query_L"][0]) - L0) < 1e-10
    assert abs(float(res["per_query_C"][0]) - C0) < 1e-10
    # ghosting is a hard mask, not a downweight: without groups all three negatives count
    res2 = dcl_loss_certificate(q, k, neg, tau=1.0)
    assert res2["M_i"].tolist() == [3, 3]


def test_dcl_fail_loud():
    q = F.normalize(_f64(2, 3, seed=4), dim=1)
    k = q.clone()
    neg = F.normalize(_f64(4, 3, seed=5), dim=1)
    _expect(ValueError, lambda: dcl_loss_certificate(q, k, neg.new_empty(0, 3), 1.0),
            "empty negative pool accepted")
    _expect(ValueError, lambda: dcl_loss_certificate(q, k, neg, 0.0), "tau=0 accepted")
    _expect(ValueError, lambda: dcl_loss_certificate(q, k[:1], neg, 1.0), "k/q shape mismatch accepted")
    _expect(ValueError, lambda: dcl_loss_certificate(q, k, neg, 1.0, query_group=torch.tensor([1, 2])),
            "half a ghosting pair accepted")
    _expect(ValueError, lambda: dcl_loss_certificate(q, k, neg, 1.0,
                                                    query_group=q.new_zeros(2), neg_group=neg.new_zeros(4)),
            "float group ids accepted")
    outside = q.clone(); outside[0, 0] = 2.0
    _expect(ValueError, lambda: dcl_loss_certificate(outside, k, neg, 1.0), "q outside B^d accepted")
    outside2 = neg.clone(); outside2[0, 0] = 1.5
    _expect(ValueError, lambda: dcl_loss_certificate(q, k, outside2, 1.0), "neg outside B^d accepted")
    # stop-grad reference material: grad-tracking k/neg REJECTED, not silently detached
    _expect(ValueError, lambda: dcl_loss_certificate(q, k.clone().requires_grad_(True), neg, 1.0),
            "grad-tracking key accepted")
    _expect(ValueError, lambda: dcl_loss_certificate(q, k, neg.clone().requires_grad_(True), 1.0),
            "grad-tracking negatives accepted")
    # every query fully ghosted -> masking bug, not a training step
    _expect(RuntimeError, lambda: dcl_loss_certificate(q, k, neg, 1.0,
                                                      query_group=torch.tensor([1, 1], dtype=torch.long),
                                                      neg_group=torch.tensor([1, 1, 1, 1], dtype=torch.long)),
            "fully-ghosted batch accepted")


def test_dcl_gradient_flows_through_query_only():
    q = F.normalize(_f64(3, 3, seed=6), dim=1).requires_grad_(True)
    k = q.detach().clone()
    neg = F.normalize(_f64(5, 3, seed=7), dim=1)
    res = dcl_loss_certificate(q, k, neg, tau=1.0)
    assert res["L_DCL"].requires_grad and res["C_K"].requires_grad
    res["L_DCL"].backward()
    assert q.grad is not None and torch.isfinite(q.grad).all() and float(q.grad.norm()) > 0.0


# ------------------------------------------------------------------ Readiness gate (Eq. 13, Prop 12.4iv)

def test_gate_constant_encoder_inert():
    # q = k = neg = same unit vector: s_pos == every neg score -> L = log M -> C_K = 0 exactly
    v = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    neg = v.expand(4, 2).contiguous()
    res = dcl_loss_certificate(v, v.clone(), neg, tau=1.0)
    assert abs(float(res["C_K"])) < 1e-12, f"constant encoder must certify 0, got {res['C_K']}"
    g = readiness_gate(float(res["C_K"]), res["K"], g_floor=0.1)
    assert g == 0.1, "floor-clamped, never below, never inflated"
    g0 = readiness_gate(float(res["C_K"]), res["K"], g_floor=0.0)
    assert g0 == 0.0
    coeff = OTCoefficient().observe(1.0, 1.0)
    assert coeff.value(0.5, g0) == 0.0, "Thm 12.5(i): gate closed -> OT fully inert"


def test_gate_opens_with_information():
    q = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    k = torch.tensor([[0.9, math.sqrt(1 - 0.81)]], dtype=torch.float64)
    neg = torch.tensor([[0.4, math.sqrt(1 - 0.16)], [0.4, -math.sqrt(1 - 0.16)]], dtype=torch.float64)
    res = dcl_loss_certificate(q, k, neg, tau=1.0)
    ratio = float(res["C_K"]) / math.log(2)
    assert 0.3 < ratio < 0.95, f"designed mid-margin case should be open, got {ratio}"
    g = readiness_gate(float(res["C_K"]), res["K"], g_floor=0.1)
    assert abs(g - ratio) < 1e-12, "gate must equal the raw ratio mid-range"
    # perfect separation -> clamped at 1
    neg2 = torch.tensor([[0.0, 1.0], [0.0, -1.0]], dtype=torch.float64)
    res2 = dcl_loss_certificate(q, torch.tensor([[1.0, 0.0]], dtype=torch.float64), neg2, tau=1.0)
    assert readiness_gate(float(res2["C_K"]), res2["K"], 0.1) == 1.0


def test_gate_monotone_lipschitz_prop124():
    K, denom, floor = 5, math.log(4), 0.1
    xs = torch.linspace(-3.0, 3.0, 241, dtype=torch.float64)
    gs = [readiness_gate(float(x), K, floor) for x in xs]
    assert all(b >= a - 1e-15 for a, b in zip(gs, gs[1:])), "gate not monotone"
    assert all(floor <= g <= 1.0 for g in gs)
    for i in range(0, 241 - 9, 7):
        x, y = float(xs[i]), float(xs[i + 9])
        assert abs(gs[i] - gs[i + 9]) <= abs(x - y) / denom + 1e-12, "Lipschitz 1/log(K-1) violated"
    _expect(ValueError, lambda: readiness_gate(0.5, 1), "K<2 accepted")
    _expect(ValueError, lambda: readiness_gate(float("nan"), 5), "nan C_K accepted")
    _expect(ValueError, lambda: readiness_gate(0.5, 5, g_floor=1.0), "g_floor=1 accepted")
    _expect(ValueError, lambda: readiness_gate(0.5, 5, g_floor=-0.1), "negative g_floor accepted")
    assert isinstance(readiness_gate(torch.tensor(0.3, dtype=torch.float64), 5, floor), float)


# ------------------------------------------------------------------ lambda_OT (Eq. 13, §8.3)

def test_lambda_ema_arithmetic():
    coeff = OTCoefficient(momentum=0.9)
    _expect(RuntimeError, lambda: coeff.value(0.5, 1.0), "value defined before any observation")
    coeff.observe(2.0, 4.0)
    assert abs(coeff.value(0.5, 1.0) - 0.25) < 1e-12
    coeff.observe(4.0, 4.0)  # anc: 0.9*2 + 0.1*4 = 2.2 ; em stays 4.0
    assert abs(coeff.anc - 2.2) < 1e-12 and abs(coeff.em - 4.0) < 1e-12
    assert abs(coeff.value(0.5, 1.0) - 0.5 * (2.2 / 4.0)) < 1e-12
    assert coeff.value(0.5, 0.0) == 0.0
    raw = OTCoefficient(momentum=0.0)  # latest-wins sanity
    raw.observe(1.0, 2.0); raw.observe(5.0, 2.0)
    assert abs(raw.value(1.0, 1.0) - 2.5) < 1e-12
    _expect(ValueError, lambda: OTCoefficient(momentum=1.0), "momentum=1 accepted")
    _expect(ValueError, lambda: raw.observe(-1.0, 2.0), "negative norm accepted")
    _expect(ValueError, lambda: raw.observe(float("nan"), 2.0), "nan norm accepted")
    _expect(ValueError, lambda: raw.value(0.5, 1.5), "g>1 accepted")


def test_lambda_inert_zero_em_gradient():
    coeff = OTCoefficient().observe(1.0, 1e-12)
    assert coeff.inert() and coeff.value(0.5, 1.0) == 0.0, "OT-inert regime must give lambda=0"


def test_lambda_state_dict_roundtrip():
    src = OTCoefficient(momentum=0.8)
    src.observe(3.0, 6.0); src.observe(2.0, 5.0)
    st = src.state_dict()
    dst = OTCoefficient().load_state_dict(st)
    assert dst.momentum == 0.8 and abs(dst.anc - src.anc) < 1e-15
    assert abs(dst.value(0.5, 0.7) - src.value(0.5, 0.7)) < 1e-15
    _expect(ValueError, lambda: OTCoefficient().load_state_dict({"anc": 1.0}), "partial state accepted")


# ------------------------------------------------------------------ c-guard + grad_norm util

def test_check_c_gate_B5iii():
    assert check_c_gate(0.5) == 0.5
    _expect(ValueError, lambda: check_c_gate(1.0), "c=1 accepted")
    _expect(ValueError, lambda: check_c_gate(0.0), "c=0 accepted")
    _expect(ValueError, lambda: check_c_gate(0.95), "c above c_max accepted")
    _expect(ValueError, lambda: check_c_gate(0.01), "c below c_min accepted")
    _expect(ValueError, lambda: check_c_gate(0.5, c_max=1.0), "config c_max=1 accepted (B5iii: <1)")
    _expect(ValueError, lambda: check_c_gate(0.5, c_min=0.0), "config c_min=0 accepted")
    assert check_c_gate(0.8, c_min=0.1, c_max=0.8) == 0.8


def test_grad_norm_utility():
    z = _f64(3, 4, seed=8).requires_grad_(True)
    loss = (z ** 2).sum()
    expect = 2.0 * float(z.detach().norm())
    v1 = grad_norm(loss, z)
    v2 = grad_norm(loss, z)  # retain_graph default: second call must work
    assert abs(v1 - expect) < 1e-10 and v1 == v2
    _expect(ValueError, lambda: grad_norm((z ** 2), z), "non-scalar loss accepted")
    _expect(ValueError, lambda: grad_norm(loss, z.detach()), "z without graph accepted")
    fresh = _f64(2, 2, seed=9)
    _expect(ValueError, lambda: grad_norm((fresh ** 2).sum(), fresh), "leaf without requires_grad accepted")


# ------------------------------------------------------------------ assembled anchor module

def test_projected_anchor_module():
    anchor = ProjectedAnchor(tau_dcl=0.1, anchor_on="projected")
    q = F.normalize(_f64(6, 4, seed=10), dim=1).requires_grad_(True)
    k = F.normalize(q.detach() + 0.1 * _f64(6, 4, seed=11), dim=1)
    neg = F.normalize(_f64(20, 4, seed=12), dim=1)
    qg = torch.randint(0, 2, (6,), dtype=torch.long)
    ng = torch.randint(0, 2, (20,), dtype=torch.long)
    res = anchor(q, k, neg, query_group=qg, neg_group=ng)
    assert set(res) >= {"L_DCL", "C_K", "M", "K"} and res["K"] == res["M"] + 1
    assert math.isfinite(float(res["C_K"].detach())) and int(res["n_ghosted"]) + int(res["n_active"]) <= 6
    g = readiness_gate(float(res["C_K"].detach()), res["K"])
    assert 0.1 <= g <= 1.0
    lam = OTCoefficient().observe(1.0, 0.5).value(check_c_gate(0.5), g)
    assert lam > 0
    # raw ablation mode (§11.4i): same tensor outside B^d passes on raw, raises on projected
    big = q.detach() * 3.0
    _expect(ValueError, lambda: anchor(big, big, neg), "projected mode tolerated non-ball rows")
    ProjectedAnchor(tau_dcl=0.1, anchor_on="raw")(big, big, neg)  # must not raise
    _expect(ValueError, lambda: ProjectedAnchor(anchor_on="cosface"), "unknown anchor_on accepted")
    _expect(ValueError, lambda: ProjectedAnchor(tau_dcl=0.0), "tau_dcl=0 accepted")


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
