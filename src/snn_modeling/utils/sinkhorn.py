"""Dustbin Sinkhorn OT for the SMC emotion term (fixed.pdf §5.3-5.6, §8, B2/B3).

Balanced log-domain matrix scaling on the AUGMENTED (n+1)x(m+1) kernel:
  C~ = [[0, tau...],[tau..., C]],  row marginal (1, a), col marginal (1, b)
with tau = tau_y = -eps*log(1-rho_y) (§5.4 Eq. 8): the Gibbs drop weight
exp(-tau/eps) = 1 - rho_y is EXACTLY the prior baseline probability, so
no free drop-rate hyperparameter exists (and none may be added).

Def. 5.2/5.4:  W_eps(mu,nu) = min <P~,C~> + R_eps(P~)  (slack constraints (7)
make this an EXACT balanced problem — unbalanced Sinkhorn is the dead list);
S_eps = W(mu,nu) - 1/2 W(mu,mu) - 1/2 W(nu,nu).

Prop 8.3 (Danskin): the optimal coupling is stop-gradient — the gradient of
the value flows ONLY through the cost, evaluated at the fixed optimal plan.
The nu-self solve is fully detached (cacheable per queue refresh); the mu-self
solve carries gradients (Prop 8.1(b): ||grad_z~ S|| <= 2 a_i <= 2 on B^d).

Cost matrices come from estep.cloud_log_p_matrix's SHARED kernel (Rm 4.2) —
this module never recomputes 1 - z~^T r for the cross term.
"""

import math

import torch

__all__ = ["dustbin_price", "eps_lower_bound", "check_eps_admissible", "min_iters_for_tol",
           "blur_bound", "augment_cost", "burst_marginal", "sinkhorn_plan", "solve_w",
           "class_ot_term", "SELF_NU_SCHEMA"]

SELF_NU_SCHEMA = "smc-sinkhorn-selfnu-v1"
_SUM_TOL = 1e-4        # probability-marginal normalisation tolerance
_LOG_TINY = 1e-300


def _finfo_tiny(dtype):
    return torch.finfo(dtype).tiny


# ------------------------------------------------------------------ §5.4 drop price + §12.2/B2a eps

def dustbin_price(eps, rho):
    """tau_y = -eps * log(1 - rho_y)  (Eq. 8). exp(-tau/eps) = 1 - rho_y exactly."""
    if not (eps > 0.0) or not math.isfinite(eps):
        raise ValueError(f"dustbin_price: eps must be positive/finite, got {eps}")
    if torch.is_tensor(rho):
        if not torch.isfinite(rho).all() or (rho <= 0).any() or (rho >= 1).any():
            raise ValueError("dustbin_price: rho_y must lie strictly in (0,1) — "
                             "the [rho_min, 1-rho_min] clamp happens upstream")
        return -eps * torch.log1p(-rho)
    rho = float(rho)
    if not (0.0 < rho < 1.0):
        raise ValueError(f"dustbin_price: rho_y must lie strictly in (0,1), got {rho}")
    return -eps * math.log1p(-rho)


def _artanh(x):
    if not (0.0 < x < 1.0):
        raise ValueError(f"artanh argument must lie in (0,1), got {x}")
    return 0.5 * math.log((1.0 + x) / (1.0 - x))


def eps_lower_bound(tol, T_sc, tau):
    """B2a / §12.2: eps >= M~ / (2 * artanh(tol^(1/T_sc))), M~ = max(2, tau_y)."""
    if not (0.0 < tol < 1.0):
        raise ValueError(f"eps_lower_bound: tol must lie in (0,1), got {tol}")
    if int(T_sc) < 1:
        raise ValueError(f"eps_lower_bound: T_sc >= 1 required, got {T_sc}")
    if not math.isfinite(tau) or tau < 0.0:
        raise ValueError(f"eps_lower_bound: tau must be finite >= 0, got {tau}")
    return max(2.0, float(tau)) / (2.0 * _artanh(tol ** (1.0 / float(T_sc))))


def check_eps_admissible(eps, tol, T_sc, tau):
    """Hard config gate (Failure-Catalogue row: never unroll/never underprice):
    raises unless eps >= eps_lower_bound(tol, T_sc, tau)."""
    lb = eps_lower_bound(tol, T_sc, tau)
    if eps < lb - 1e-12:
        raise ValueError(
            f"§12.2/B2a eps-admissibility violated: eps={eps:.6f} < lower bound {lb:.6f} "
            f"(M~=max(2,tau)={max(2.0, float(tau)):.3f}, tol={tol}, T_sc={T_sc}). "
            "Raise eps or T_sc — do NOT silently ship a non-contracting schedule.")
    return lb


def min_iters_for_tol(eps, tol, tau):
    """Smallest T with [tanh(M~/(2 eps))]^T <= tol (B2a). Raises when no finite T
    works (tanh underflows the float64 representation of 1 — eps too small)."""
    if not (eps > 0.0) or not math.isfinite(eps):
        raise ValueError(f"min_iters_for_tol: eps must be positive/finite, got {eps}")
    t = math.tanh(max(2.0, float(tau)) / (2.0 * eps))
    if t >= 1.0:
        raise ValueError(
            f"min_iters_for_tol: tanh(M~/2eps)={t} is indistinguishable from 1 in float64 — "
            "tol is unreachable in finitely many iterations at this eps; raise eps")
    if t <= 0.0:
        return 1
    return int(math.ceil(math.log(tol) / math.log(t)))


def blur_bound(eps, n, m):
    """B2b: |W_eps - W_0| <= eps * log((n+1)(m+1)) (negative-entropy bound)."""
    if n < 1 or m < 1:
        raise ValueError(f"blur_bound: n,m >= 1 required, got {n}/{m}")
    return eps * math.log((n + 1) * (m + 1))


# ------------------------------------------------------------------ augmented problem

def augment_cost(cost, tau):
    """C~ = [[0, tau...],[tau..., C]] — gradient flows through the real block only."""
    if cost.dim() != 2 or cost.shape[0] < 1 or cost.shape[1] < 1:
        raise ValueError(f"augment_cost: cost must be (n, m) with n,m >= 1, got {tuple(cost.shape)}")
    if not torch.isfinite(cost.detach()).all():
        raise ValueError("augment_cost: non-finite cost")
    if not math.isfinite(tau) or tau < 0.0:
        raise ValueError(f"augment_cost: tau must be finite >= 0, got {tau}")
    n, m = cost.shape
    top = torch.cat([cost.new_zeros(1, 1), cost.new_full((1, m), float(tau))], dim=1)
    bot = torch.cat([cost.new_full((n, 1), float(tau)), cost], dim=1)
    return torch.cat([top, bot], dim=0)


def _validate_marginal(v, name):
    if v.dim() != 1 or v.numel() < 1:
        raise ValueError(f"{name}: probability marginal must be (k,), k >= 1, got {tuple(v.shape)}")
    if not torch.isfinite(v).all():
        raise ValueError(f"{name}: non-finite marginal")
    if (v <= 0).any():
        raise ValueError(
            f"{name}: every entry must be strictly positive — zero mass rows contradict the "
            "Rm 4.2 interiority of alpha (use the annealed alpha, never a hard mask)")
    if abs(float(v.sum().item()) - 1.0) > _SUM_TOL:
        raise ValueError(f"{name}: marginal must sum to 1 (Def. 5.2), got {float(v.sum()):.6f}")


def _validate_cost(cost, name):
    if cost.dim() != 2:
        raise ValueError(f"{name}: cost must be 2-D")
    cd = cost.detach()
    if not torch.isfinite(cd).all():
        raise ValueError(f"{name}: non-finite cost")
    lo, hi = float(cd.min().item()), float(cd.max().item())
    if lo < -1e-5 or hi > 2.0 + 1e-5:
        raise ValueError(
            f"{name}: costs must lie in [0, 2] on B^d (got [{lo:.6f}, {hi:.6f}]) — "
            "build them via estep.cloud_log_p_matrix / NullspaceState.project")


def burst_marginal(alpha):
    """a_i = alpha_i / A_y — the class-y posterior-burst measure of Def. 5.2 (mu side)."""
    if alpha.dim() != 1 or alpha.numel() < 1:
        raise ValueError("burst_marginal: alpha must be (n,), n >= 1")
    if not torch.isfinite(alpha).all():
        raise ValueError("burst_marginal: non-finite alpha")
    if (alpha <= 0).any() or (alpha >= 1).any():
        raise ValueError(f"burst_marginal: alpha must lie strictly in (0,1) (Rm 4.2), got "
                         f"[{float(alpha.min()):.6f}, {float(alpha.max()):.6f}]")
    return alpha / alpha.sum()


# ------------------------------------------------------------------ solver (stop-grad, Prop 8.3)

def sinkhorn_plan(cost, a, b, eps, tau, n_iters=5000, tol=1e-10):
    """Balanced log-domain scaling on the augmented marginals (1,a) x (1,b).
    Fully no_grad: the plan is an M-step object; gradients reach the loss only
    through W_with_grad below (Prop 8.3/Danskin). Non-convergence RAISES
    (never a silently under-solved coupling)."""
    _validate_cost(cost, "sinkhorn_plan cost")
    _validate_marginal(a, "sinkhorn_plan a")
    _validate_marginal(b, "sinkhorn_plan b")
    if not (eps > 0.0) or not math.isfinite(eps):
        raise ValueError(f"sinkhorn_plan: eps must be positive/finite, got {eps}")
    if int(n_iters) < 1 or not (tol > 0.0):
        raise ValueError(f"sinkhorn_plan: n_iters >= 1 and tol > 0 required, got {n_iters}/{tol}")
    with torch.no_grad():
        cd = cost.detach()
        C_aug = augment_cost(cd, tau)
        logK = -C_aug / eps
        # §5.3: the dummy baseline node carries marginal capacity a0 = b0 = 1 on
        # BOTH sides (NOT 0 — that would zero the dustbin column and silently
        # degrade the problem to exact balanced OT, ignoring tau entirely).
        r = torch.cat([cd.new_ones(1), a.detach()], dim=0)
        cvec = torch.cat([cd.new_ones(1), b.detach()], dim=0)
        log_r, log_c = r.log(), cvec.log()
        f = torch.zeros_like(log_r)
        g = torch.zeros_like(log_c)
        it = 0
        for it in range(1, int(n_iters) + 1):
            f_prev = f.clone()
            f = log_r - torch.logsumexp(logK + g.unsqueeze(0), dim=1)
            g = log_c - torch.logsumexp(logK + f.unsqueeze(1), dim=0)
            if float((f - f_prev).abs().max().item()) < tol * min(1.0, eps):
                break
        plan = torch.exp(f.unsqueeze(1) + logK + g.unsqueeze(0))
        row_res = float((plan.sum(dim=1) - r).abs().max().item())
        col_res = float((plan.sum(dim=0) - cvec).abs().max().item())
        residual = max(row_res, col_res)
        if residual > max(tol, 1000.0 * torch.finfo(plan.dtype).eps):
            raise RuntimeError(
                f"sinkhorn_plan: marginals not met to tol after {it} iterations "
                f"(residual {residual:.3e} > {tol:.3e}; eps={eps}, M~=max(2,tau)={max(2.0, float(tau)):.3f}) — "
                "apply the §11.5 ladder (raise eps first, then re-check the rho seed); "
                "never ship an under-solved coupling")
        e_const = eps * float((plan * (plan.clamp(min=_finfo_tiny(plan.dtype)).log() - 1.0)).sum())
        W = float((plan * C_aug).sum()) + e_const
    return {"plan": plan, "W": W, "di": plan[1:, 0], "uj": plan[0, 1:],
            "residual": residual, "iters": it}


def W_with_grad(cost, plan, tau, eps):
    """Prop 8.3: dW/dC_ij = Pi*_ij — value re-evaluated at the FIXED plan so the
    gradient flows only through the cost (exact to first order; no unrolling)."""
    _validate_cost(cost, "W_with_grad cost")
    if plan.shape != (cost.shape[0] + 1, cost.shape[1] + 1):
        raise ValueError(f"W_with_grad: plan must be the augmented ({cost.shape[0]+1},"
                         f"{cost.shape[1]+1}) coupling, got {tuple(plan.shape)}")
    e_const = eps * float((plan * (plan.clamp(min=_finfo_tiny(plan.dtype)).log() - 1.0)).sum())
    return (plan * augment_cost(cost, tau)).sum() + e_const


def solve_w(cost, a, b, eps, tau, **kw):
    """Fully-detached matching value (symmetry tests, self-terms)."""
    res = sinkhorn_plan(cost, a, b, eps, tau, **kw)
    return res["W"], res["plan"]


# ------------------------------------------------------------------ per-class emotion term

def class_ot_term(z_tilde, a, keys, b, eps, tau, self_nu_cache=None, n_iters=5000, tol=1e-10):
    """S^y_eps(mu_y, nu_y) — Def. 5.4 debiased matching value for one class (Prop 5.5).

    z_tilde: (n, D) current-window ball features for class y, grad-carrying.
    a: (n,) burst marginal a_i = alpha_i / A_y (burst_marginal).
    keys: (m, D) queue cloud r_j (stop-grad material), b: (m,) its measure.
    self_nu_cache: SELF_NU_SCHEMA dict from a previous call — the nu-self solve is
      detached, so it is cacheable per queue refresh (Stage 4 §5 line 3).

    Returns S (grad tensor through the cross AND mu-self sides, Prop 8.1(b)),
    plus plans, per-window drops d_i, unused mass u_j, D_norm_y = sum_i d_i
    (Rm 5.1 / Table 1 #5), and a fresh self_nu_cache to store.
    """
    if z_tilde.dim() != 2 or keys.dim() != 2 or keys.shape[1] != z_tilde.shape[1]:
        raise ValueError(f"class_ot_term: z_tilde (n,D) and keys (m,D) required, got "
                         f"{tuple(z_tilde.shape)}/{tuple(keys.shape)}")
    for name, t in (("z_tilde", z_tilde), ("keys", keys)):
        if not torch.isfinite(t).all():
            raise ValueError(f"class_ot_term: non-finite {name}")
        nrm = t.norm(dim=1)
        if (nrm > 1.0 + 1e-5).any():
            raise ValueError(f"class_ot_term: {name} must lie in B^d — use NullspaceState.project")
    if not torch.is_grad_enabled():
        raise ValueError("class_ot_term: mu side must be differentiable (queue-vs-queue is DEAD, "
                         "Prop 9.1) — call the detached path via solve_w instead")
    n = z_tilde.shape[0]
    _validate_marginal(a, "class_ot_term a")
    _validate_marginal(b, "class_ot_term b")
    if a.shape[0] != n or b.shape[0] != keys.shape[0]:
        raise ValueError("class_ot_term: marginal lengths must match z_tilde / keys rows")
    if not (eps > 0.0) or not math.isfinite(eps):
        raise ValueError(f"class_ot_term: eps must be positive/finite, got {eps}")
    if not math.isfinite(tau) or tau < 0.0:
        raise ValueError(f"class_ot_term: tau must be finite >= 0, got {tau}")

    cost_cross = 1.0 - z_tilde @ keys.t()          # grad-carrying; Rm 4.2 form
    cost_self_mu = 1.0 - z_tilde @ z_tilde.t()
    for name, C in (("cross", cost_cross), ("self-mu", cost_self_mu)):
        _validate_cost(C, f"class_ot_term {name} cost")

    cross = sinkhorn_plan(cost_cross, a, b, eps, tau, n_iters=n_iters, tol=tol)
    W_cross = W_with_grad(cost_cross, cross["plan"], tau, eps)

    self_mu = sinkhorn_plan(cost_self_mu.detach(), a, a, eps, tau, n_iters=n_iters, tol=tol)
    W_self_mu = W_with_grad(cost_self_mu, self_mu["plan"], tau, eps)

    if self_nu_cache is None:
        cost_self_nu = (1.0 - keys @ keys.t()).detach()
        _validate_cost(cost_self_nu, "class_ot_term self-nu cost")
        snu = sinkhorn_plan(cost_self_nu, b, b, eps, tau, n_iters=n_iters, tol=tol)
        W_self_nu = snu["W"]
        fresh_cache = {"schema": SELF_NU_SCHEMA, "W": W_self_nu, "plan": snu["plan"],
                       "m": int(keys.shape[0]), "eps": float(eps), "tau": float(tau)}
    else:
        if not isinstance(self_nu_cache, dict) or self_nu_cache.get("schema") != SELF_NU_SCHEMA:
            raise ValueError(f"class_ot_term: self_nu_cache must be a {SELF_NU_SCHEMA} dict — "
                             "refusing a silently invalid OT cache")
        if self_nu_cache["m"] != keys.shape[0]:
            raise ValueError("class_ot_term: cached nu-self cloud size != keys rows (stale cache)")
        if abs(self_nu_cache["eps"] - eps) > 1e-12 or abs(self_nu_cache["tau"] - tau) > 1e-12:
            raise ValueError("class_ot_term: cached nu-self solved at different (eps, tau) — "
                             "re-solve, do not reuse")
        W_self_nu = float(self_nu_cache["W"])
        fresh_cache = dict(self_nu_cache)

    S = W_cross - 0.5 * W_self_mu - 0.5 * W_self_nu
    D_norm = float(cross["di"].sum().item())
    return {"S": S, "W_cross": W_cross, "W_self_mu": W_self_mu, "W_self_nu": W_self_nu,
            "plan": cross["plan"], "d": cross["di"], "u": cross["uj"],
            "self_nu_cache": fresh_cache,
            "diagnostics": {"D_norm_y": D_norm, "matched_mass": float(a.sum()) - D_norm,
                            "cross_residual": cross["residual"], "self_mu_residual": self_mu["residual"]}}
