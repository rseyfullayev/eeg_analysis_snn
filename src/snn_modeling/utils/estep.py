"""E-step responsibilities for the SMC objective (SWEEP-Net Rev. Certified
Eq. 6-9/14-17, Rm 4.2, Prop 8.1/10.1).

Densities (Rm 4.2 — the FAILURE-CATALOGUE row-5 guard):
* log p_y(z~) = logsumexp_j [log b_j - 0.5*||z~ - r_j||^2 / eps]   over the queue's
  beta-weighted class cloud (Def. 5.2 nu_y). The ground cost is the HALF-SQUARED
  EUCLIDEAN metric on B^d (Eq. 9 / Prop 8.1): C(v, v) = 0 holds in the INTERIOR
  (||v|| < 1), not just on the sphere as with the old cosine cost 1 - v^T v.
  The cost matrix is RETURNED alongside: Stage 4's Sinkhorn consumes the exact
  same object (Rm 4.2 kernel-matrix sharing — one computation per class per step).
* log p_s(h') = -||h' - mu_hat'||^2 / (2 sigma^2) — evaluated on nullspace
  coordinates ONLY (h' from NullspaceState.project/apply_projector). Never
  call it with full-space z; the split-space failure row stays unrepresentable.
  sigma^2 is the CALIBRATED EMA nullspace baseline variance (Rm 4.2 / Eq. 7-8),
  not a free constant — see sm_loss.SMCLossEngine.sigma_hat_sq.

Responsibilities:
* alpha^(1A) (Eq. 6, Prop 4.1): sigma((log p_y - log p_s)/T + logit(rho_y)),
  T annealed high -> low (deterministic annealing, Rose [3]). The calibrated
  form (Rm 4.2 / Eq. 7-8) replaces logit(rho_y) with the buffer-solved offset
  u_bar_y: alpha = sigma((s_t + u_bar_y)/T) with s_t = log p_y - log p_s — see
  solve_offset_bisection / alpha_calibrated below; the c_y = u_bar_y - logit(rho_y)
  correction is what Phase 1B inherits through the lockpass (Eq. 18).
* alpha^(1B) (Eq. 14, Prop 10.1): label-free marginal — invariant to any
  permutation of the class axis by construction.
* Bag baseline mu_hat_b = G^2(h_bar'_b) (Eq. 15, Prop 10.2) with the CERTIFIED
  contraction constant kappa_mu and the explicit error bound (16). kappa >= 1
  raises; the certificate is checked from the orbit, never assumed.

Nothing inside detaches alpha: the E-step output is a deterministic function
of the current model (Rm 4.1), and Stage 6 owns the gradient routing
(queue betas detached per Prop 8.4; L_spr/L_sub grads flow per Eq. 23).
"""

import math

import torch

from .sparsity import _logit_tensor

__all__ = ["half_sq_cost", "cloud_log_p_matrix", "log_baseline_gaussian",
           "alpha_1a", "alpha_1b", "anneal_temperature", "two_step_baseline",
           "batch_two_step_baselines", "bag_statistic",
           "solve_offset_bisection", "alpha_calibrated"]

_NORM_TOL = 1e-5


def half_sq_cost(z, r):
    """C_ij = 0.5 * ||z_i - r_j||^2 — half-squared Euclidean ground cost on B^d
    (Eq. 9 / Prop 8.1). Gram-expansion form (no sqrt): smooth at z_i == r_j, so
    C(v, v) = 0 exactly in the INTERIOR ||v|| < 1, and on B^d the cost stays in
    [0, 2] (diameter bound 0.5 * (||z|| + ||r||)^2 <= 2).
    """
    if z.dim() != 2 or r.dim() != 2 or z.shape[1] != r.shape[1]:
        raise ValueError(f"half_sq_cost: (n, D) vs (m, D) required, got "
                         f"{tuple(z.shape)}/{tuple(r.shape)}")
    d2 = (z.pow(2).sum(dim=1, keepdim=True) + r.pow(2).sum(dim=1).unsqueeze(0)
          - 2.0 * (z @ r.t()))
    return 0.5 * d2.clamp(min=0.0)


def _check_ball(t, name):
    if t.dim() != 2 or t.shape[1] < 1:
        raise ValueError(f"{name} must be (n, D), got {tuple(t.shape)}")
    if not torch.isfinite(t).all():
        raise ValueError(f"{name} contains non-finite values")
    nrm = t.norm(dim=1)
    if (nrm > 1.0 + _NORM_TOL).any():
        raise ValueError(
            f"{name} must live in B^d (norm <= 1) — max row norm {nrm.max():.6f}; "
            "produce it via NullspaceState.project")
    return nrm


# ------------------------------------------------------------------ densities

def cloud_log_p_matrix(z_tilde, clouds, eps, support=None):
    """log p_y(z~) for every class in `clouds` — Eq. 6/Rm 4.2 cloud score.

    z_tilde: (n, D) current-window ball features (grad allowed — this is the
    differentiable side that Stage 4's coupling gradient later mirrors).
    clouds:  {y: {'keys': (m_y, D), 'weights': (m_y,)}} from
             SupMoCoQueue.read_class_clouds() (stop-grad material).
    support: optional {y: (n, m_y) bool} — False = pair forbidden. Applied as a
             STRUCTURAL ZERO inside the logsumexp (never a cost sentinel), so the
             E-step density and Stage 4's coupling see the SAME transportable graph
             (Rm 4.2). A window whose whole class-y cloud is forbidden gets
             log p_y = -inf; the caller must treat those rows as cold (alpha = rho),
             never as a density.
    Returns (lp, costs): lp (n, C_max) with column y = log p_y, costs = {y:
    C matrix (n, m_y)} — THE shared kernel matrices (Rm 4.2; Stage 4 must not
    recompute them). Costs stay in [0, 2] under a mask: the mask restricts the
    support, so the B2a eps bound (which scales with the cost DIAMETER) is unchanged.
    """
    _check_ball(z_tilde, "z_tilde")
    if not (eps > 0) or not math.isfinite(eps):
        raise ValueError(f"eps must be positive/finite, got {eps}")
    if not clouds:
        raise ValueError("cloud_log_p_matrix: empty cloud dict — cold queue is handled by the "
                         "annealing schedule (Stage 6), never by a silent uniform lp")
    n = z_tilde.shape[0]
    classes = sorted(int(y) for y in clouds)
    if len(classes) != len(clouds):
        raise ValueError("cloud_log_p_matrix: duplicate class keys")
    lp = torch.full((n, len(classes)), float("nan"), device=z_tilde.device, dtype=z_tilde.dtype)
    costs = {}
    for col, y in enumerate(classes):
        c = clouds[y]
        keys, w = c["keys"], c["weights"]
        _check_ball(keys, f"class {y} cloud keys")
        if w.shape != (keys.shape[0],) or w.numel() == 0:
            raise ValueError(f"class {y}: weights must be (m_y,) matching keys, got {tuple(w.shape)}")
        if not torch.isfinite(w).all() or (w < 0).any():
            raise ValueError(f"class {y}: cloud weights must be finite and non-negative")
        wsum = float(w.sum().item())
        if abs(wsum - 1.0) > 1e-3:
            raise ValueError(f"class {y}: cloud weights must sum to 1 (Def. 5.2), got {wsum:.6f}")
        cost = half_sq_cost(z_tilde, keys)
        lo, hi = float(cost.min().item()), float(cost.max().item())
        if lo < -1e-4 or hi > 2.0 + 1e-4:
            raise RuntimeError(
                f"cost 0.5*||z~ - r||^2 left [0, 2] (min {lo:.6f}, max {hi:.6f}) — "
                "inputs violate B^d")
        cost = cost.clamp(0.0, 2.0)
        lw = torch.full_like(w, float("-inf"))
        pos = w > 0
        if not bool(pos.any()):
            raise ValueError(f"class {y}: all-zero cloud mass — Rm 4.2 interiority violated upstream")
        lw[pos] = torch.log(w[pos])
        logits = lw - cost / eps
        if support is not None:
            sp = support.get(int(y)) if hasattr(support, "get") else support[y]
            if sp.dtype != torch.bool:
                raise ValueError(f"class {y}: support mask must be bool, got {sp.dtype}")
            if sp.shape != cost.shape:
                raise ValueError(f"class {y}: support mask {tuple(sp.shape)} != cost "
                                 f"{tuple(cost.shape)}")
            if bool(sp.all()):
                pass                      # fully open column: nothing to zero out
            else:
                logits = logits.masked_fill(~sp, float("-inf"))
        lp[:, col] = torch.logsumexp(logits, dim=1)
        costs[y] = cost
    # -inf is legal ONLY as a support-mask verdict ("no transportable key of this class
    # for this window"); NaN/+inf remain impossible on B^d. Written as isinf & (lp > 0)
    # rather than torch.isposinf — the RunPod image ships torch 2.8.0, so the guard must
    # not depend on an API whose availability differs between 2.8 and the 2.9 dev box.
    if bool((torch.isnan(lp) | (torch.isinf(lp) & (lp > 0))).any()):
        raise RuntimeError("non-finite class log-density — impossible on B^d; check eps")
    return lp, costs


def log_baseline_gaussian(h_prime, baseline, sigma_sq):
    """log p_s(h') = -||h' - baseline||^2 / (2 sigma^2) (Rm 4.2 / Eq. 14).

    h_prime and `baseline` must BOTH be nullspace coordinates (h' or G(h'/mu'),
    per Rm 6.5). `baseline` broadcasts: (D,) shared or (n, D) per-window.
    """
    if h_prime.dim() != 2:
        raise ValueError(f"h_prime must be (n, D), got {tuple(h_prime.shape)}")
    if not torch.isfinite(h_prime).all():
        raise ValueError("h_prime contains non-finite values")
    if not (sigma_sq > 0) or not math.isfinite(sigma_sq):
        raise ValueError(f"sigma_sq must be positive/finite, got {sigma_sq}")
    b = torch.as_tensor(baseline, device=h_prime.device, dtype=h_prime.dtype)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    if b.shape not in ((1, h_prime.shape[1]), h_prime.shape):
        raise ValueError(f"baseline must be (D,) or (n, D)={tuple(h_prime.shape)}, got {tuple(b.shape)}")
    if not torch.isfinite(b).all():
        raise ValueError("baseline contains non-finite values")
    return -((h_prime - b) ** 2).sum(dim=1) / (2.0 * sigma_sq)


# ------------------------------------------------------------------ responsibilities

def alpha_1a(log_p_y, log_p_s, rho_y, T):
    """alpha^(1A) = sigma((log p_y - log p_s)/T + logit(rho_y)) — Eq. 6 (Prop 4.1)
    with the deterministic-annealing temperature T on the log-ratio.

    rho_y: per-window class prior (or per-class vector gathered by the caller).
    Returns (alpha, info). Rm 4.2: with rho_y in [rho_min, 1-rho_min] and both
    densities bounded on B^d, the logit is bounded, so alpha lies STRICTLY in
    (0, 1) — verified here, and [alpha_min, alpha_max] := sigma(±max|logit|)
    of the batch are reported for Table-1 logging and the Stage-6 clamp knob.
    """
    if not (T > 0) or not math.isfinite(T):
        raise ValueError(f"annealing temperature T must be positive/finite, got {T}")
    if log_p_y.shape != log_p_s.shape or rho_y.shape != log_p_s.shape:
        raise ValueError(f"alpha_1a: log_p_y/log_p_s/rho_y must share shape, got "
                         f"{tuple(log_p_y.shape)}/{tuple(log_p_s.shape)}/{tuple(rho_y.shape)}")
    if log_p_y.numel() == 0:
        raise ValueError("alpha_1a on an empty batch (B14 discipline)")
    for name, t in (("log_p_y", log_p_y), ("log_p_s", log_p_s)):
        if not torch.isfinite(t).all():
            raise ValueError(f"alpha_1a: {name} non-finite")
    if (rho_y <= 0).any() or (rho_y >= 1).any():
        raise ValueError(f"alpha_1a: rho_y must lie strictly in (0,1), got "
                         f"[{rho_y.min():.6f}, {rho_y.max():.6f}]")
    logit = (log_p_y - log_p_s) / T + _logit_tensor(rho_y)
    alpha = torch.sigmoid(logit)
    if (alpha <= 0).any() or (alpha >= 1).any():
        raise RuntimeError("interiority violated: saturated alpha — logit magnitudes exceed "
                           "float32 range; check eps/sigma_sq/rho_min")
    m = logit.abs().max()  # bounds via the SAME torch.sigmoid path as alpha itself,
    info = {"logit_abs_max": float(m.item()), "alpha_min": float(torch.sigmoid(-m).item()),
            "alpha_max": float(torch.sigmoid(m).item()),  # so Rm 4.2 holds bit-exactly
            "alpha_std": float(alpha.std().item()) if alpha.numel() > 1 else 0.0}
    return alpha, info


# ------------------------------------------------------------------ Rm 4.2 / Eq. 7-8 calibrated offset

def solve_offset_bisection(scores, rho_y, T, steps=25, seed_offset=None):
    """Solve  (1/|B_y|) sum_{t in B_y} sigma((s_t + u)/T) = rho_y  for u_y* by
    BISECTION on the rolling class-buffer scores (Rm 4.2, Eq. 7-8).

    scores: (N,) stored raw scores s = log p_y - log p_s of buffer B_y
    (float vector, no grad — a scalar-root problem). g(u) is strictly increasing
    (sigma is), so a bracket exists for any rho_y in (0,1): it is found by
    DOUBLING around seed_offset (default T*logit(rho_y), the Eq. 6 anchor) and
    then EXACTLY `steps` (spec: 25) bisection steps are taken. Raises when the
    scores are all-identical and no finite bracket is reached — never returns an
    unbracketed midpoint. Returns (u_star, diag).
    """
    s = torch.as_tensor(scores, dtype=torch.float64).reshape(-1)
    if s.numel() == 0:
        raise ValueError("solve_offset_bisection: empty buffer B_y — calibrate only after "
                         "the queue holds scored entries (no silent zero offset)")
    if not torch.isfinite(s).all():
        raise ValueError("solve_offset_bisection: NaN/inf stored score — the queue score "
                         "buffer was filled without calibration scores (wiring bug)")
    if not (T > 0) or not math.isfinite(T):
        raise ValueError(f"solve_offset_bisection: T must be positive/finite, got {T}")
    if int(steps) < 1:
        raise ValueError(f"solve_offset_bisection: steps >= 1 required, got {steps}")
    rho_y = float(rho_y)
    if not (0.0 < rho_y < 1.0) or not math.isfinite(rho_y):
        raise ValueError(f"solve_offset_bisection: rho_y must lie strictly in (0,1), got {rho_y}")

    def g(u):
        return float(torch.sigmoid((s + u) / T).mean().item()) - rho_y

    u0 = T * math.log(rho_y / (1.0 - rho_y)) if seed_offset is None else float(seed_offset)
    if not math.isfinite(u0):
        raise ValueError(f"solve_offset_bisection: non-finite seed_offset {seed_offset}")
    span = max(T, 1.0)
    lo, hi = u0 - span, u0 + span
    for _ in range(60):
        if g(lo) <= 0.0 <= g(hi):
            break
        span *= 2.0
        lo, hi = u0 - span, u0 + span
    else:
        raise RuntimeError(
            f"solve_offset_bisection: no bracket within +-u0 {span:.3e} — degenerate buffer "
            "(all scores identical cannot root every rho); check the stored score pipeline")
    for _ in range(int(steps)):
        mid = 0.5 * (lo + hi)
        if g(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    u_star = 0.5 * (lo + hi)
    return u_star, {"residual": abs(g(u_star)), "bracket": (lo, hi), "steps": int(steps)}


def alpha_calibrated(scores, offsets, T):
    """alpha^(1A) = sigma((s_t + u_bar_{y_t})/T) — the calibrated-offset form of
    Eq. 6/7-8 (Rm 4.2): offsets carry the buffer-solved u_bar_y (EMA-smoothed),
    which reduces to T*logit(rho_y) before the first calibration solve, so the
    uncalibrated Eq. 6 path and this one are the SAME formula at seed time.

    Returns (alpha, info) with the same interiority guarantee as alpha_1a.
    """
    if not (T > 0) or not math.isfinite(T):
        raise ValueError(f"alpha_calibrated: T must be positive/finite, got {T}")
    if scores.shape != offsets.shape:
        raise ValueError(f"alpha_calibrated: scores/offsets must share shape, got "
                         f"{tuple(scores.shape)}/{tuple(offsets.shape)}")
    if scores.numel() == 0:
        raise ValueError("alpha_calibrated on an empty batch (B14 discipline)")
    if not torch.isfinite(scores).all() or not torch.isfinite(offsets).all():
        raise ValueError("alpha_calibrated: non-finite score/offset (NaN u_bar means the "
                         "class was never calibrated — route cold rows to alpha=rho instead)")
    alpha = torch.sigmoid((scores + offsets) / T)
    if (alpha <= 0).any() or (alpha >= 1).any():
        raise RuntimeError("interiority violated: saturated alpha — check eps/sigma_hat_sq/T")
    logit = (scores + offsets) / T
    info = {"logit_abs_max": float(logit.abs().max().item()),
            "alpha_std": float(alpha.std().item()) if alpha.numel() > 1 else 0.0}
    return alpha, info


def alpha_1b(log_p_by_class, log_p_s, rho, calibration=None):
    """alpha^(1B) — Eq. 14 (Prop 10.1), the label-free nullspace marginal:

        sigma( log[(1/C) sum_y rho_y exp(l_y + c_y)] - log p_s(h') - log(1 - rho_bar) )

    log_p_by_class: (n, C); rho: (C,) strictly in (0,1). Invariant under any
    permutation of the class axis (no label is ever read) — Phase 1B safety.
    calibration: optional (C,) per-class offsets c_y = u_bar_y - logit(rho_y)
    exported by the Phase 1A lockpass (Eq. 18 / Prop 10.1); None = no calibrated
    correction (the raw Eq. 14 rule). It shifts only the class-score axis, so
    permutation invariance is preserved (c travels with its column).
    """
    if log_p_by_class.dim() != 2:
        raise ValueError(f"alpha_1b: log_p_by_class must be (n, C), got {tuple(log_p_by_class.shape)}")
    C = log_p_by_class.shape[1]
    rho = torch.as_tensor(rho, device=log_p_by_class.device, dtype=log_p_by_class.dtype).view(-1)
    if rho.shape[0] != C:
        raise ValueError(f"alpha_1b: rho must have C={C} entries, got {rho.shape[0]}")
    if log_p_by_class.numel() == 0:
        raise ValueError("alpha_1b on an empty batch (B14 discipline)")
    if not torch.isfinite(log_p_by_class).all() or not torch.isfinite(log_p_s).all():
        raise ValueError("alpha_1b: non-finite inputs")
    if (rho <= 0).any() or (rho >= 1).any():
        raise ValueError(f"alpha_1b: rho must lie strictly in (0,1), got "
                         f"[{rho.min():.6f}, {rho.max():.6f}]")
    rho_bar = float(rho.mean().item())
    if rho_bar >= 1.0:
        raise ValueError("alpha_1b: marginal burst prior rho_bar >= 1 — no baseline mass left")
    lp = log_p_by_class
    if calibration is not None:
        c = torch.as_tensor(calibration, device=log_p_by_class.device,
                            dtype=log_p_by_class.dtype).view(-1)
        if c.shape[0] != C:
            raise ValueError(f"alpha_1b: calibration must have C={C} entries, got {c.shape[0]}")
        if not torch.isfinite(c).all():
            raise ValueError("alpha_1b: non-finite calibration offset c_y — the lockpass was "
                             "written before calibration completed (wiring bug)")
        lp = lp + c.unsqueeze(0)
    log_num = torch.logsumexp(torch.log(rho).unsqueeze(0) + lp, dim=1) - math.log(C)
    logit = log_num - log_p_s - math.log(1.0 - rho_bar)
    return torch.sigmoid(logit)


def anneal_temperature(step, total_steps, t_start=20.0, t_end=1.0, anneal_frac=0.3):
    """Deterministic-annealing schedule (fixed.pdf §4 "high to low"; knob table:
    20 -> 1 across the first 30 % of training, linear in step, then t_end)."""
    if total_steps < 1 or step < 0:
        raise ValueError(f"step/total_steps must satisfy total_steps >= 1, 0 <= step, got "
                         f"{step}/{total_steps}")
    if not (t_start > t_end > 0):
        raise ValueError(f"need t_start > t_end > 0 (anneal high->low), got {t_start}->{t_end}")
    if not 0.0 < anneal_frac <= 1.0:
        raise ValueError(f"anneal_frac must lie in (0, 1], got {anneal_frac}")
    frac = min(1.0, float(step) / (anneal_frac * total_steps))
    return t_start + (t_end - t_start) * frac


# ------------------------------------------------------------------ Eq. 15/16 bag baseline

def two_step_baseline(h_prime_bag, log_p_by_class, rho, sigma_sq, max_iter=64, tol=1e-12,
                      calibration=None):
    """Certified two-step bag baseline mu_hat_b = G^2(h_bar'_b) (Eq. 15) with the
    Prop 10.2 contraction certificate kappa_mu and the explicit bound (16).

    G(mu') = sum_t (1 - alpha_t(1B; mu')) h'_t / sum_t (1 - alpha_t(1B; mu')),
    where ONLY log p_s depends on mu' (the burst numerator log[(1/C)sum rho_y p_y]
    is mu'-free by Prop 10.1). `calibration` (C,) = the locked c_y offsets
    (Eq. 18) enters the numerator exactly as in alpha_1b; sigma_sq is the locked
    CALIBRATED sigma_hat^2 (Rm 4.2 / Eq. 7-8), not a free constant. Returns a dict:
      mu_hat  : G^2(h_bar') — the estimator actually used downstream (Eq. 14/17)
      alpha   : alpha^(1B) evaluated AT mu_hat (the responsibilities for R_b)
      kappa   : max contraction estimate over the visited orbit (incl. fixed point)
      bound   : kappa^2/(1-kappa) * ||G(h_bar') - h_bar'||   (Eq. 16, last term)
      mu_star : converged fixed point of G (certification only; not used downstream)
    Raises if kappa >= 1 (Prop 10.2 premise violated) — never a silent extra step.
    """
    if h_prime_bag.dim() != 2 or h_prime_bag.shape[0] < 1:
        raise ValueError(f"two_step_baseline: h_prime_bag must be (T_b, D), T_b >= 1, got "
                         f"{tuple(h_prime_bag.shape)}")
    Tb = h_prime_bag.shape[0]
    if log_p_by_class.dim() != 2 or log_p_by_class.shape[0] != Tb:
        raise ValueError("two_step_baseline: log_p_by_class must be (T_b, C)")
    if not torch.isfinite(h_prime_bag).all() or not torch.isfinite(log_p_by_class).all():
        raise ValueError("two_step_baseline: non-finite inputs")
    rho = rho.view(-1).to(h_prime_bag)
    C = log_p_by_class.shape[1]
    if rho.shape[0] != C:
        raise ValueError(f"two_step_baseline: rho must have {C} entries")
    if (rho <= 0).any() or (rho >= 1).any():
        raise ValueError("two_step_baseline: rho must lie strictly in (0,1)")
    if not (sigma_sq > 0) or not math.isfinite(sigma_sq):
        raise ValueError(f"sigma_sq must be positive/finite, got {sigma_sq}")
    if int(max_iter) < 1 or tol <= 0:
        raise ValueError(f"two_step_baseline: max_iter >= 1 and tol > 0 required, got "
                         f"{max_iter}/{tol}")
    rho_bar = float(rho.mean().item())
    lp = log_p_by_class
    if calibration is not None:
        c = torch.as_tensor(calibration, device=log_p_by_class.device,
                            dtype=log_p_by_class.dtype).view(-1)
        if c.shape[0] != C:
            raise ValueError(f"two_step_baseline: calibration must have {C} entries, "
                             f"got {c.shape[0]}")
        if not torch.isfinite(c).all():
            raise ValueError("two_step_baseline: non-finite calibration offset c_y")
        lp = lp + c.unsqueeze(0)
    log_num = (torch.logsumexp(torch.log(rho).unsqueeze(0) + lp, dim=1)
               - math.log(C))
    log_base_term = math.log(1.0 - rho_bar)

    def g(mu):
        alpha = torch.sigmoid(log_num - log_baseline_gaussian(h_prime_bag, mu, sigma_sq)
                              - log_base_term)
        w = 1.0 - alpha
        den = float(w.sum().item())
        if den <= 1e-9:
            raise ValueError("G undefined: bag fully burst-flagged (sum(1-alpha) ~ 0) — "
                             "no baseline mass to average (Rm 4.2 interiority upstream)")
        return (w.unsqueeze(1) * h_prime_bag).sum(dim=0) / den, alpha, w / den

    mu0 = h_prime_bag.mean(dim=0)                    # h_bar'_b, mu'^(0) = mean (Eq. 15)
    # tol is a FLOOR: a float32 orbit plateaus near machine eps, so compare against
    # max(tol, eps-scale) instead of demanding an unreachable absolute residual.
    tol_eff = max(float(tol), 1000.0 * torch.finfo(mu0.dtype).eps
                  * max(1.0, float(mu0.norm().item())))
    mu1, a1, w1 = g(mu0)
    mu2, a2, w2 = g(mu1)                             # mu_hat_b = G^2(h_bar') (Eq. 15)
    mu3, a_hat, w_hat = g(mu2)                       # alpha^(1B) evaluated AT mu_hat (Eq. 14/17)
    mu_star, a_s, w_s = mu3, a_hat, w_hat
    resid = float((mu_star - mu2).norm().item())
    for _ in range(int(max_iter)):
        nxt, a_n, w_n = g(mu_star)
        resid = float((nxt - mu_star).norm().item())
        mu_star, a_s, w_s = nxt, a_n, w_n
        if resid < tol_eff:
            break
    else:
        raise RuntimeError(f"two_step_baseline: G did not reach its fixed point in {max_iter} "
                           f"steps (residual {resid:.3e}) — contraction premise kappa < 1 is false")
    kappa = 0.0
    for mu, a, w, Gmu in ((mu0, a1, w1, mu1), (mu1, a2, w2, mu2),
                          (mu2, a_hat, w_hat, mu3), (mu_star, a_s, w_s, mu_star)):
        kn = float((w * a * (h_prime_bag - Gmu).norm(dim=1) * (h_prime_bag - mu).norm(dim=1))
                   .sum().item() / (sigma_sq * Tb))
        kappa = max(kappa, kn)
    # The sup over mu' in Prop 10.2 is uncomputable, so the certificate also takes the
    # MEASURED two-step orbit ratios against the converged fixed point: gap = d2 =
    # r01*r12*d0 <= kappa^2*d0 holds by construction once kappa >= max(r01, r12). Deeper
    # orbit ratios are NOT added — at the float noise floor d3/d2 can exceed 1 spuriously.
    dists = [float((p - mu_star).norm().item()) for p in (mu0, mu1, mu2)]
    for i in range(2):
        if dists[i] > tol_eff:
            kappa = max(kappa, dists[i + 1] / dists[i])
    if kappa >= 1.0:
        raise RuntimeError(
            f"two_step_baseline: kappa_mu = {kappa:.4f} >= 1 — Prop 10.2 certification FAILED "
            "(burst fraction >= 1/2 or baseline spread > sigma^2). Do NOT ship mu_hat_b: "
            "re-seed rho / raise sigma_sq per the §11.5 ladder")
    step_norm = float((mu1 - mu0).norm().item())
    bound = kappa ** 2 / (1.0 - kappa) * step_norm
    h_bar_minus_star = float((mu0 - mu_star).norm().item())
    return {"mu_hat": mu2, "alpha": a_hat, "kappa": kappa, "bound": bound, "mu_star": mu_star,
            "gap": float((mu2 - mu_star).norm().item()),
            "gap_theory": kappa ** 2 * h_bar_minus_star}


def batch_two_step_baselines(bags, log_p_by_class_batch, rho, sigma_sq, **kw):
    """Per-bag loop over list-of-tensors (bags: list of (T_b, D); T_b may differ
    per bag — the (1/T_b) normalization of Prop 10.2 is applied bag by bag)."""
    if len(bags) != len(log_p_by_class_batch):
        raise ValueError("batch_two_step_baselines: bags and log_p matrices must match in count")
    if len(bags) == 0:
        raise ValueError("batch_two_step_baselines: empty batch (B14 discipline)")
    return [two_step_baseline(h, lp, rho, sigma_sq, **kw)
            for h, lp in zip(bags, log_p_by_class_batch)]


def bag_statistic(alpha_1b_bag, h_prime_bag):
    """R_b = (1/T_b) sum_t alpha_t^(1B) h'_t (Eq. 17 — the sufficient statistic)."""
    if alpha_1b_bag.dim() != 1 or h_prime_bag.dim() != 2 or \
            alpha_1b_bag.shape[0] != h_prime_bag.shape[0]:
        raise ValueError(f"bag_statistic: alpha (T_b,) and h' (T_b, D) required, got "
                         f"{tuple(alpha_1b_bag.shape)}/{tuple(h_prime_bag.shape)}")
    if alpha_1b_bag.numel() == 0:
        raise ValueError("bag_statistic: T_b = 0 bag — dataset must not emit empty bags")
    if (alpha_1b_bag < 0).any() or (alpha_1b_bag > 1).any():
        raise ValueError("bag_statistic: responsibilities must lie in [0,1]")
    return (alpha_1b_bag.unsqueeze(1) * h_prime_bag).sum(dim=0) / float(alpha_1b_bag.shape[0])
