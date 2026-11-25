"""Stage 4 regression tests — Dustbin Sinkhorn OT (sinkhorn.py) per fixed.pdf §5.3-5.6, §8.

Run from repo root:  venv\\Scripts\\python.exe tests\\test_stage4.py
Exit code 0 = all pass. (Plain asserts — the venv has no pytest.)

Plan §5 Stage-4 checklist: symmetry; S_eps(mu,mu) = 0; two-point saturation
min(c, 2*tau_y) flat for c >= 2*tau_y (B3); Prop 8.2 zero-gradient regimes on
B^d; queue-vs-queue gradient exactly 0 (Prop 9.1); no underflow at eps_min;
D_norm_y / ESS bands (Table 1 #5). Plus the B2a eps-interval gates and every
fail-loud validation.
"""

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.snn_modeling.utils.sinkhorn import (augment_cost, blur_bound, burst_marginal,
                                             check_eps_admissible, class_ot_term,
                                             dustbin_price, eps_lower_bound, min_iters_for_tol,
                                             sinkhorn_plan, solve_w, W_with_grad)

D = 4


def _expect(exc, fn, msg):
    try:
        fn()
    except exc:
        return
    raise AssertionError(msg)


def _rand_marginal(k, seed):
    torch.manual_seed(seed)
    v = torch.rand(k, dtype=torch.float64) + 0.1
    return v / v.sum()


def _ball(n, seed, dtype=torch.float64):
    torch.manual_seed(seed)
    return F.normalize(torch.randn(n, D, dtype=dtype), dim=1)


# ------------------------------------------------------------------ §5.4 drop price

def test_dustbin_price():
    # Eq. 11 decoupled geometric drop price: tau_y = tau_0 - kappa_cost*log1p(-rho_y)
    # (defaults tau_0=0.35, kappa_cost=0.50) — NO LONGER a function of eps.
    t = dustbin_price(0.5)
    assert abs(t - (0.35 - 0.50 * math.log(0.5))) < 1e-12
    tau_0, kappa = 0.35, 0.50
    t = dustbin_price(0.3, tau_0, kappa)
    assert abs(t - (tau_0 - kappa * math.log1p(-0.3))) < 1e-12
    assert dustbin_price(0.9, tau_0, kappa) > dustbin_price(0.1, tau_0, kappa), \
        "tau must be strictly increasing in rho_y"
    # decoupling: the price signature carries NO eps — the old Eq. 8 lock
    # tau = -eps*log1p(-rho) must NOT be reconstructable at any eps.
    assert all(abs(dustbin_price(0.4, tau_0, kappa) + e * math.log1p(-0.4)) > 1e-6
               for e in (0.1, 0.4, 1.0)), "price drifted back to the eps-coupled form"
    tv = dustbin_price(torch.tensor([0.1, 0.5], dtype=torch.float64), tau_0, kappa)
    assert torch.allclose(tv, torch.tensor([tau_0 - kappa * math.log1p(-0.1),
                                            tau_0 - kappa * math.log1p(-0.5)],
                                           dtype=torch.float64))
    # tau_y >= tau_0 > 0 for every interior rho (drop is never free)
    assert min(dustbin_price(r, tau_0, kappa) for r in (0.01, 0.5, 0.99)) >= tau_0 - 1e-12
    for bad in [lambda: dustbin_price(0.0), lambda: dustbin_price(1.0),
                lambda: dustbin_price(float("nan")),
                lambda: dustbin_price(0.5, tau_0=-0.1), lambda: dustbin_price(0.5, tau_0=0.35,
                                                                             kappa_cost=0.0),
                lambda: dustbin_price(0.5, tau_0=float("nan"))]:
        _expect(ValueError, bad, "bad price input accepted")


def test_eps015_25iter_smoke_residual():
    """2026-11-25 REVISED smoke contract: eps=0.15, 25 iterations, LOG domain. The B2a
    worst-case floor (~0.9233 for tol 1e-3 / T_sc 30) is a START WARNING now; the live
    guarantee is the EMPIRICAL marginal residual max(|P1-a|,|P'b|) < 1e-3 — measured
    ~2e-7 here (worst-case theory is astronomically loose on realistic clouds). The
    high-tau corner (rho 0.9 -> tau ~ 1.5, exp(-tau/eps) ~ 4e-5 must not underflow)
    is included."""
    torch.manual_seed(1)
    n, m = 60, 256
    z = _ball(n, seed=40)
    keys = _ball(m, seed=41)
    a = burst_marginal(0.02 + 0.9 * torch.rand(n, dtype=torch.float64))
    b = burst_marginal(0.02 + 0.9 * torch.rand(m, dtype=torch.float64))
    # seeded regime of the smoke run (rho 0.05..0.27 -> tau <= 0.5): 25 iters meet
    # the 1e-3 budget in EXACT arithmetic, not just under fp32 noise.
    for tau in (float(dustbin_price(0.1)), float(dustbin_price(0.25))):
        res = class_ot_term(z.clone().requires_grad_(True), a, keys, b,
                            eps=0.15, tau=tau, n_iters=25, tol=1e-10, residual_cap=1e-3)
        d = res["diagnostics"]
        assert d["cross_residual"] < 1e-3 and d["self_mu_residual"] < 1e-3, \
            f"eps=0.15/25-iter residual breach at tau={tau:.3f}: {d}"
        assert torch.isfinite(res["S"].detach())
    # exact-arith corners: rho 0.5/0.9 -> tau 0.70/1.50, dustbin kernel as small as
    # exp(-tau/eps) = 4.5e-5; fp64 shows 25 iters genuinely CANNOT carry the
    # contraction (2.5e-3 / 8.7e-3). The cap RAISES instead of shipping — and the
    # same solve meets < 1e-3 once given iterations (log-domain never underflows;
    # the B2a worst-case 2.1M-iter estimate is astronomically loose: 1000 suffice).
    for tau in (float(dustbin_price(0.5)), float(dustbin_price(0.9))):
        _expect(RuntimeError, lambda tau=tau: class_ot_term(
            z.clone().requires_grad_(True), a, keys, b, eps=0.15, tau=tau,
            n_iters=25, tol=1e-10, residual_cap=1e-3),
            f"breached solve shipped at tau={tau:.3f}")
    res_hi = class_ot_term(z.clone().requires_grad_(True), a, keys, b, eps=0.15,
                           tau=float(dustbin_price(0.9)), n_iters=1000, tol=1e-12,
                           residual_cap=1e-3)
    assert res_hi["diagnostics"]["cross_residual"] < 1e-3, res_hi["diagnostics"]
    # the solver's own raise floor still guards a genuinely under-solved batch
    _expect(RuntimeError, lambda: sinkhorn_plan(1.0 - z[:8] @ keys[:8].t(),
                                                burst_marginal(torch.full((8,), 0.5)),
                                                burst_marginal(torch.rand(8, dtype=torch.float64)),
                                                eps=0.02, tau=0.5, n_iters=1, tol=1e-10),
            "under-solved coupling shipped silently")


def test_half_sq_ground_cost_interior_and_range():
    """Eq. 9 / Prop 8.1: C_ij = 0.5*||z_i - r_j||^2 on B^d — C(v, v) = 0 EVEN IN THE
    INTERIOR (||v|| < 1, where the old cosine cost gave 1 - ||v||^2 > 0), values stay
    in [0, 2], and class_ot_term now builds that cost (self-mu term vanishes on the
    diagonal rows -> S(mu, mu) = 0 exactly also for interior clouds)."""
    from src.snn_modeling.utils.estep import half_sq_cost
    torch.manual_seed(4)
    v = torch.randn(5, D, dtype=torch.float64)
    v = v / v.norm(dim=1, keepdim=True)                       # direction…
    v = v * torch.linspace(0.3, 0.9, 5, dtype=torch.float64).unsqueeze(1)  # …interior radii
    assert (v.norm(dim=1) < 1.0).all()
    c = half_sq_cost(v, v)
    assert torch.allclose(c.diagonal(), torch.zeros(5, dtype=torch.float64), atol=1e-15), \
        "C(v, v) != 0 in the interior of B^d"
    assert float(c.min()) >= 0.0 and float(c.max()) <= 2.0 + 1e-12
    # agrees with cosine exactly on the sphere (norm-1 rows)
    z, r = _ball(4, seed=30), _ball(4, seed=31)
    assert torch.allclose(half_sq_cost(z, r), 1.0 - z @ r.t(), atol=1e-12)
    # class_ot_term path: interior mu == nu cloud -> S = 0 exactly (was > 0 under cosine)
    vi = v.clone().requires_grad_(True)
    a = burst_marginal(torch.full((5,), 0.3, dtype=torch.float64))
    res = class_ot_term(vi, a, v.detach(), a, eps=0.4, tau=0.3)
    assert abs(float(res["S"].detach())) < 1e-9, f"interior self-term not debiased to 0: {res['S']}"
    _expect(ValueError, lambda: half_sq_cost(torch.zeros(3), torch.zeros(3, 3)),
            "wrong-shape cost accepted")


# ------------------------------------------------------------------ B2a eps interval + B2b blur

def test_eps_interval_B2a():
    tol, T = 1e-3, 30
    x = tol ** (1.0 / T)
    lb_ref = 2.0 / (2.0 * 0.5 * math.log((1 + x) / (1 - x)))
    assert abs(eps_lower_bound(tol, T, 1.0) - lb_ref) < 1e-9      # M~ = max(2, tau) = 2
    assert abs(eps_lower_bound(tol, T, 5.0) - 5.0 / (2.0 * 0.5 * math.log((1 + x) / (1 - x)))) < 1e-9
    got = check_eps_admissible(lb_ref + 0.1, tol, T, 1.0)
    assert abs(got - lb_ref) < 1e-9
    _expect(ValueError, lambda: check_eps_admissible(0.5, tol, T, 1.0), "inadmissible eps accepted")
    mi = min_iters_for_tol(lb_ref, tol, 1.0)
    assert 29 <= mi <= 31, f"at the lower bound min_iters must be ~T_sc={T}, got {mi}"
    assert min_iters_for_tol(1.0, tol, 1.0) == 26  # ceil(log(1e-3)/log(tanh(1)))
    _expect(ValueError, lambda: min_iters_for_tol(0.01, tol, 1.0), "eps too small accepted")
    assert abs(blur_bound(0.5, 2, 3) - 0.5 * math.log(12)) < 1e-12
    for bad in [lambda: eps_lower_bound(1.0, 30, 1.0), lambda: eps_lower_bound(0.5, 0, 1.0),
                lambda: eps_lower_bound(0.5, 30, -1.0), lambda: blur_bound(0.1, 0, 3),
                lambda: min_iters_for_tol(0.0, 0.5, 1.0)]:
        _expect(ValueError, bad, "bad interval input accepted")


# ------------------------------------------------------------------ augmented structure + solver

def test_augment_cost_structure_and_grad():
    cost = torch.full((2, 3), 1.0, dtype=torch.float64, requires_grad=True)
    C = augment_cost(cost, 0.75)
    assert C.shape == (3, 4)
    assert float(C[0, 0].detach()) == 0.0
    assert torch.allclose(C[0, 1:], torch.full((3,), 0.75, dtype=torch.float64))
    assert torch.allclose(C[1:, 0], torch.full((2,), 0.75, dtype=torch.float64))
    assert torch.equal(C[1:, 1:], cost)
    C.sum().backward()
    assert torch.equal(cost.grad, torch.ones(2, 3, dtype=torch.float64)), "grad must flow via C only"
    for bad in [lambda: augment_cost(torch.zeros(2, dtype=torch.float64), 1.0),
                lambda: augment_cost(torch.zeros(2, 2, dtype=torch.float64), float("nan")),
                lambda: augment_cost(torch.zeros(2, 2, dtype=torch.float64), -0.1)]:
        _expect(ValueError, bad, "bad augment input accepted")


def test_sinkhorn_plan_marginals():
    torch.manual_seed(0)
    cost = (torch.rand(5, 3, dtype=torch.float64) * 2.0).clamp(0, 2)
    a, b = _rand_marginal(5, 1), _rand_marginal(3, 2)
    eps, tau = 0.3, 0.5
    res = sinkhorn_plan(cost, a, b, eps, tau, n_iters=5000, tol=1e-12)
    P = res["plan"]
    assert P.shape == (6, 4) and (P >= 0).all() and torch.isfinite(P).all()
    r = torch.cat([torch.ones(1, dtype=torch.float64), a])
    cvec = torch.cat([torch.ones(1, dtype=torch.float64), b])
    assert torch.allclose(P.sum(dim=1), r, atol=1e-9), "row marginal (1, a) not met"
    assert torch.allclose(P.sum(dim=0), cvec, atol=1e-9), "col marginal (1, b) not met"
    assert torch.allclose(res["di"], P[1:, 0], atol=0) and (res["uj"] == P[0, 1:]).all()
    assert float(res["di"].sum()) <= 1.0 + 1e-9
    assert math.isfinite(res["W"])
    # Prop 5.3 symmetry: swapping sides transposes the plan
    W_t, P_t = solve_w(cost.t(), b, a, eps, tau)
    assert abs(W_t - res["W"]) < 1e-9, "W_eps not symmetric"
    assert torch.allclose(P, P_t.t(), atol=1e-9)
    # determinism
    r2 = sinkhorn_plan(cost, a, b, eps, tau)
    assert torch.allclose(r2["plan"], P, atol=1e-12)
    for bad in [lambda: sinkhorn_plan(cost * 1.5, a, b, eps, tau),          # cost > 2
                lambda: sinkhorn_plan(cost, a * 0.9, b, eps, tau),          # not a measure
                lambda: sinkhorn_plan(cost, a, -b, eps, tau),
                lambda: sinkhorn_plan(cost, a, b, 0.0, tau),
                lambda: sinkhorn_plan(cost, torch.zeros_like(a), b, eps, tau),  # zero mass row
                lambda: sinkhorn_plan(cost, a, b, eps, tau, n_iters=0),
                lambda: solve_w(cost, a, b, eps, tau, n_iters=1, tol=1e-30)]:   # under-solved -> raise
        _expect((ValueError, RuntimeError), bad, "bad solver input accepted")


def test_burst_marginal():
    alpha = torch.tensor([0.1, 0.4, 0.2], dtype=torch.float64)
    a = burst_marginal(alpha)
    assert torch.allclose(a, alpha / 0.7) and abs(float(a.sum()) - 1.0) < 1e-12
    for bad in [lambda: burst_marginal(torch.tensor([0.0, 0.5], dtype=torch.float64)),
                lambda: burst_marginal(torch.tensor([0.5, 1.0], dtype=torch.float64)),
                lambda: burst_marginal(torch.rand(2, 2, dtype=torch.float64))]:
        _expect(ValueError, bad, "bad alpha accepted")


# ------------------------------------------------------------------ Def. 5.4 debias + B3

def test_self_zero_and_nonnegativity():
    z = _ball(4, seed=3).requires_grad_(True)
    a = burst_marginal(torch.rand(4, dtype=torch.float64) * 0.8 + 0.1)
    keys = z.detach().clone()
    res = class_ot_term(z, a, keys, a, eps=0.4, tau=0.3)
    assert float(res["S"].detach().abs()) < 1e-12, "S_eps(mu, mu) must vanish exactly"
    assert res["S"].requires_grad
    res["S"].backward()
    assert z.grad is not None and float(z.grad.norm()) < 1e-6, "grad must vanish at mu = nu"
    # near cloud: S >= 0 (B3 definiteness), D_norm small (close match -> little dropping)
    z2 = _ball(4, seed=3)
    keys2 = z2 + 0.05 * torch.randn(4, D, dtype=torch.float64)
    res2 = class_ot_term(z2, a, F.normalize(keys2, dim=1), a, eps=0.1, tau=0.5)
    assert float(res2["S"].detach()) > -1e-6, "S_eps went negative"
    assert res2["diagnostics"]["D_norm_y"] < 0.1, f"close cloud should barely drop: {res2['diagnostics']}"


def test_two_point_saturation_B3():
    """S_eps(d_x, d_y) -> min(c, 2*tau_y); flat above 2*tau_y (B3 step 2).
    eps=0.1: at eps=0.05 the worst-case contraction tanh(M~/2eps)=1-8e-9 makes
    any tight tol unreachable, which is precisely what §12.2's eps floor exists
    to prevent; the value bands carry the B2b blur bound eps*log((n+1)(m+1))."""
    eps, tau = 0.1, 0.7
    e0 = torch.zeros(1, D, dtype=torch.float64); e0[0, 0] = 1.0
    e1 = torch.zeros(1, D, dtype=torch.float64); e1[0, 1] = 1.0
    one = torch.ones(1, dtype=torch.float64)

    def S_of_c(c):
        k = (1.0 - c) * e0 + math.sqrt(max(0.0, 1.0 - (1.0 - c) ** 2)) * e1
        z = e0.clone().requires_grad_(True)
        out = class_ot_term(z, one, k.detach(), one, eps=eps, tau=tau, n_iters=20000, tol=1e-6)
        return float(out["S"].detach())

    s_active = S_of_c(0.5)
    s_active2 = S_of_c(1.0)
    s_near = S_of_c(1.9)     # 2*tau = 1.4 < c <= 2: saturated
    s_max = S_of_c(2.0)      # cost cap on B^d, also saturated
    assert abs(s_active - 0.5) < 0.3, f"min(c,2tau) wrong at c=0.5: {s_active}"
    assert abs(s_active2 - 1.0) < 0.3, f"min(c,2tau) wrong at c=1.0: {s_active2}"
    assert s_active < s_active2, "W_eps must be strictly increasing in the active regime"
    assert abs(s_near - 2.0 * tau) < 0.3, f"saturation value wrong: {s_near}"
    # B3 content at finite eps = SLOPE COLLAPSE (dW/dc = Pi*11 ~ e^{-(c-2tau)/eps}),
    # not exact flatness: strictly increasing but far below the active slope.
    assert 0.0 < s_max - s_near < 0.02, f"slope must collapse above 2*tau_y: {s_near} vs {s_max}"
    assert s_max - s_near < 0.2 * (s_active2 - s_active), "saturated slope rivals active slope"


# ------------------------------------------------------------------ §8 gradient structure

def test_grad_bounds_prop81():
    z = _ball(6, seed=5).requires_grad_(True)
    keys = _ball(9, seed=6)
    a = burst_marginal(torch.rand(6, dtype=torch.float64) * 0.8 + 0.1)
    b = _rand_marginal(9, 7)
    eps, tau = 0.4, 0.3
    res = class_ot_term(z, a, keys, b, eps=eps, tau=tau)
    res["S"].backward()
    gnorms = z.grad.norm(dim=1)
    assert torch.all(gnorms <= 2.0 * a + 1e-6), f"Prop 8.1(b) bound 2*a_i violated: {gnorms}"
    # cross-only functional obeys the tighter bound ||grad|| <= a_i (Prop 8.1(a))
    z2 = z.detach().clone().requires_grad_(True)
    cost_cross = 1.0 - z2 @ keys.t()
    plan = sinkhorn_plan(cost_cross, a, b, eps, tau)["plan"]
    Wc = W_with_grad(cost_cross, plan, tau, eps) - res["W_self_nu"]
    Wc.backward()
    assert torch.all(z2.grad.norm(dim=1) <= a + 1e-6), f"Prop 8.1(a) bound a_i violated"
    # (a) fully-baseline row: a_i -> 0 forces its gradient to zero across all of B^d (Prop 8.2a)
    a_thin = a.clone()
    a_thin[0] = 1e-9
    a_thin = a_thin / a_thin.sum()
    z3 = _ball(6, seed=5).requires_grad_(True)
    res3 = class_ot_term(z3, a_thin, keys, b, eps=eps, tau=tau)
    res3["S"].backward()
    assert float(z3.grad[0].norm()) < 1e-6, "baseline row leaked OT gradient"


def test_confident_match_radial_only_prop82b():
    """One-hot confident match on the sphere: the CROSS functional's gradient is
    -a_i*r_j*, purely radial in z~_i, so the upstream normalisation Jacobian
    (I - z~z~^T)/||h'|| of NullspaceState.project annihilates it (Prop 8.2b).
    The full debiased S has zero gradient at mu = nu by symmetry - tested in
    test_self_zero - so this asserts the one-sided queue-debiased functional."""
    z = torch.zeros(1, D, dtype=torch.float64)
    z[0, 0] = 1.0
    z.requires_grad_(True)
    keys = z.detach().clone()  # confident match: cost 0, one-hot coupling
    one = torch.ones(1, dtype=torch.float64)
    cost_cross = 1.0 - z @ keys.t()
    plan = sinkhorn_plan(cost_cross, one, one, 0.1, 0.5)["plan"]
    W = W_with_grad(cost_cross, plan, 0.5, 0.1) - solve_w(
        (1.0 - keys @ keys.t()).detach(), one, one, 0.1, 0.5)[0]
    W.backward()
    g = z.grad[0].detach()
    z0 = z.detach()[0]
    assert float(g.norm()) > 1e-6, "confident match must still carry a (radial) gradient"
    cos = float(torch.dot(g, z0) / (g.norm() * z0.norm()))
    assert abs(cos) > 1.0 - 1e-9, "confident-match gradient must be purely radial"
    tangent = g - float(torch.dot(g, z0)) * z0
    assert float(tangent.norm()) < 1e-12, "tangential residue on a settled burst (ghost drift)"


def test_queue_vs_queue_is_dead_prop91():
    z = _ball(5, seed=8)              # requires_grad=False on BOTH sides
    keys = _ball(5, seed=8)
    a = _rand_marginal(5, 1)
    res = class_ot_term(z, a, keys, a, eps=0.4, tau=0.3)
    assert res["S"].grad_fn is None, "queue-vs-queue must produce a constant (dead) loss"

    def _inside_no_grad():
        with torch.no_grad():
            class_ot_term(z, a, keys, a, eps=0.4, tau=0.3)
    _expect(ValueError, _inside_no_grad, "class_ot_term allowed inside no_grad")


# ------------------------------------------------------------------ eps_min, cache, Table-1 diagnostics

def test_no_underflow_at_eps_min():
    tau = 0.8
    eps_min = eps_lower_bound(1e-3, 30, tau)
    check_eps_admissible(eps_min, 1e-3, 30, tau)
    cost = (torch.rand(5, 7, dtype=torch.float64) * 2.0).clamp(0, 2)
    a, b = _rand_marginal(5, 11), _rand_marginal(7, 12)
    res = sinkhorn_plan(cost, a, b, eps_min, tau, tol=1e-8)
    assert torch.isfinite(res["plan"]).all() and res["plan"].min() >= 0.0
    assert res["residual"] < 1e-8 and math.isfinite(res["W"])
    res32 = sinkhorn_plan(cost.float(), a.float(), b.float(), eps_min, tau, tol=1e-5)
    assert torch.isfinite(res32["plan"]).all() and res32["residual"] < 1e-5


def test_self_nu_cache_contract():
    z = _ball(4, seed=13)
    keys = _ball(6, seed=14)
    a = burst_marginal(torch.rand(4, dtype=torch.float64) * 0.7 + 0.1)
    b = _rand_marginal(6, 15)
    r1 = class_ot_term(z, a, keys, b, eps=0.4, tau=0.3)
    cache = r1["self_nu_cache"]
    assert cache["schema"] == "smc-sinkhorn-selfnu-v1" and cache["m"] == 6
    r2 = class_ot_term(z, a, keys, b, eps=0.4, tau=0.3, self_nu_cache=cache)
    assert r2["W_self_nu"] == r1["W_self_nu"] and abs(float(r2["S"]) - float(r1["S"])) < 1e-12
    stale = dict(cache); stale["schema"] = "v0"
    bad_m = dict(cache); bad_m["m"] = 5
    bad_eps = dict(cache); bad_eps["eps"] = 0.5
    _expect(ValueError, lambda: class_ot_term(z, a, keys, b, eps=0.4, tau=0.3, self_nu_cache=stale),
            "invalid cache schema accepted")
    _expect(ValueError, lambda: class_ot_term(z, a, keys, b, eps=0.4, tau=0.3, self_nu_cache=bad_m),
            "stale cloud size accepted")
    _expect(ValueError, lambda: class_ot_term(z, a, keys, b, eps=0.4, tau=0.3, self_nu_cache=bad_eps),
            "cache reused across (eps, tau) change")
    # cloud genuinely far -> D_norm_y large (dropping dominates when c >> 2*tau_y),
    # Table-1 #5 direction. EVERY pair must cost 2: identical z~ rows at +e0 with all
    # prototypes at the exact antipode -e0 (an earlier draft used elementwise per-index
    # antipodes, which left off-diagonal costs ~1 and the solver legitimately matched
    # most mass - the scenario was wrong, not the optimizer).
    zc = torch.zeros(3, D, dtype=torch.float64)
    zc[:, 0] = 1.0
    zc.requires_grad_(True)
    kf = torch.zeros(3, D, dtype=torch.float64)
    kf[:, 0] = -1.0
    far = class_ot_term(zc, burst_marginal(torch.full((3,), 0.5, dtype=torch.float64)),
                        kf, _rand_marginal(3, 18), eps=0.4, tau=0.02)
    assert far["diagnostics"]["D_norm_y"] > 0.9, "far cloud must mostly drop mass"


def test_class_ot_term_fail_loud():
    z = _ball(4, seed=19).requires_grad_(True)
    keys = _ball(5, seed=20)
    a = _rand_marginal(4, 21)
    b = _rand_marginal(5, 22)
    outside = z.clone(); outside[0, 0] = 2.0
    _expect(ValueError, lambda: class_ot_term(outside, a, keys, b, 0.4, 0.3), "z~ left B^d")
    for bad in [lambda: class_ot_term(z, a[:2], keys, b, 0.4, 0.3),          # len mismatch
                lambda: class_ot_term(z, a, keys, b, 0.0, 0.3),              # eps
                lambda: class_ot_term(z, a, keys, b, 0.4, -0.3),             # tau
                lambda: class_ot_term(z, a, keys, b, 0.4, float("nan")),
                lambda: class_ot_term(torch.zeros(4, 4, dtype=torch.float64), a,
                                      keys, b, 0.4, 0.3, n_iters=1, tol=1e-30)]:  # under-solved
        _expect((ValueError, RuntimeError), bad, "bad class_ot_term input accepted")


# ------------------------------------------------------------------ LOTO support mask

def test_support_mask_is_structural_zero():
    """Forbidden pairs must carry EXACTLY zero plan mass and zero cost-gradient, while
    the cost tensor itself stays in [0, 2] so M~ = max(2, tau) — and therefore every
    Thm 12.2 / B2a / B2b bound — is unchanged by masking."""
    z, keys = _ball(4, seed=41), _ball(5, seed=42)
    a, b = _rand_marginal(4, 43), _rand_marginal(5, 44)
    eps, tau = 0.4, 0.3
    cost = 1.0 - z @ keys.t()

    full = sinkhorn_plan(cost, a, b, eps, tau)
    sup = torch.ones(4, 5, dtype=torch.bool)
    sup[0, 1] = sup[2, 3] = False
    res = sinkhorn_plan(cost, a, b, eps, tau, support=sup)
    plan = res["plan"]
    assert plan[1, 2].item() == 0.0 and plan[3, 4].item() == 0.0, "forbidden cell carries mass"
    assert res["forbidden_mass"] == 0.0
    assert torch.isfinite(plan).all() and res["residual"] < 1e-8
    assert float(cost.max()) <= 2.0 + 1e-9, "mask leaked into the cost tensor"
    # M~ = max(2, tau) is untouched by masking, so the B2a lower bound stays ~0.92.
    # (A 1e4 cost sentinel would push the diameter to 1e4 -> lb ~4617, outside eps_max.)
    assert abs(eps_lower_bound(1e-3, 30, max(2.0, tau)) - 0.9233) < 1e-3
    assert eps_lower_bound(1e-3, 30, 1e4) > 4000.0
    # Restricting the feasible set can only raise the optimal matching value.
    assert res["W"] >= full["W"] - 1e-12, f"masked W {res['W']} < unmasked {full['W']}"
    # Fully-open mask must be identical to no mask at all (the short-circuit path).
    same = sinkhorn_plan(cost, a, b, eps, tau, support=torch.ones(4, 5, dtype=torch.bool))
    assert abs(same["W"] - full["W"]) < 1e-12

    # Danskin derivative restricted to the support: dW/dC == 0 on forbidden cells.
    cleaf = cost.detach().clone().requires_grad_(True)
    plan_m = sinkhorn_plan(cleaf, a, b, eps, tau, support=sup)["plan"]
    g = torch.autograd.grad(W_with_grad(cleaf, plan_m, tau, eps), cleaf)[0]
    assert g[0, 1].item() == 0.0 and g[2, 3].item() == 0.0, "gradient leaked through a forbidden pair"
    assert (g[sup] != 0).any(), "mask killed the whole gradient"


def test_support_mask_over_masked_row_drops_to_dustbin():
    """A row with no transportable key must dump into the (always-open) dustbin column,
    i.e. d_i == a_i — not make the balanced problem infeasible and raise."""
    z, keys = _ball(3, seed=51), _ball(4, seed=52)
    a, b = _rand_marginal(3, 53), _rand_marginal(4, 54)
    cost = 1.0 - z @ keys.t()
    sup = torch.ones(3, 4, dtype=torch.bool)
    sup[1, :] = False                       # row 1 fully forbidden
    res = sinkhorn_plan(cost, a, b, 0.4, 0.3, support=sup)
    assert res["rows_fully_masked"] == 1
    assert abs(float(res["di"][1]) - float(a[1])) < 1e-9, "fully-masked row must drop in full"
    assert res["residual"] < 1e-8, "closed row broke marginal feasibility"


def test_support_mask_rejects_sentinel_and_bad_input():
    """The approved mechanism is a structural zero. A large-cost sentinel — the obvious
    wrong way — must be refused, and a silent clamp must not be allowed to flatten it."""
    z, keys = _ball(3, seed=61), _ball(4, seed=62)
    a, b = _rand_marginal(3, 63), _rand_marginal(4, 64)
    good = 1.0 - z @ keys.t()
    sentinel = good.clone()
    sentinel[0, 0] = 1e4                    # what a "hard guard" on the cost looks like
    _expect(ValueError, lambda: sinkhorn_plan(sentinel, a, b, 0.4, 0.3),
            "1e4 cost sentinel accepted instead of raising")
    assert float(sentinel[0, 0].clamp(0.0, 2.0)) == 2.0, \
        "clamp-to-2 no longer flattens the sentinel — re-check the silent-degradation guard"
    for bad in [torch.ones(3, 4, dtype=torch.float32),          # not bool
                torch.ones(4, 4, dtype=torch.bool),             # wrong shape
                torch.ones(3, 5, dtype=torch.bool)]:
        _expect(ValueError, lambda: sinkhorn_plan(good, a, b, 0.4, 0.3, support=bad),
                f"bad support {tuple(bad.shape)}/{bad.dtype} accepted")


def test_class_ot_term_support_contract():
    """The OT cross term is masked; the self terms are NOT (masking W(mu,mu) would
    penalise a trial for containing a burst) and the nu-self cache stays mask-free."""
    z = _ball(4, seed=71).requires_grad_(True)
    keys = _ball(5, seed=72)
    a, b = _rand_marginal(4, 73), _rand_marginal(5, 74)
    sup = torch.ones(4, 5, dtype=torch.bool)
    sup[0, 0] = sup[1, 1] = sup[2, 2] = False
    out = class_ot_term(z, a, keys, b, 0.4, 0.3, support=sup, self_nu_cache=None)
    d = out["diagnostics"]
    assert d["mask_forbidden_mass"] == 0.0
    assert d["mask_forbidden_frac"] > 0.0 and d["mask_cols_fully_forbidden"] == 0
    assert out["plan"][1, 1].item() == 0.0
    assert torch.isfinite(out["S"]), "masked OT produced a non-finite value"
    out["S"].backward()
    assert torch.isfinite(z.grad).all() and float(z.grad.abs().sum()) > 0.0
    # self-nu cache is support-independent: reusing it under a different mask is legal
    cache = out["self_nu_cache"]
    z2 = _ball(4, seed=71).requires_grad_(True)
    sup2 = torch.ones(4, 5, dtype=torch.bool)
    sup2[3, 4] = False
    again = class_ot_term(z2, a, keys, b, 0.4, 0.3, support=sup2, self_nu_cache=cache)
    assert abs(again["W_self_nu"] - cache["W"]) < 1e-12, "cache wrongly masked"
    _expect(ValueError, lambda: class_ot_term(z, a, keys, b, 0.4, 0.3,
                                             support=torch.ones(9, 9, dtype=torch.bool)),
            "class_ot_term accepted a mis-shaped support")


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
