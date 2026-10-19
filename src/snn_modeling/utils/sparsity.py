"""Burst prior rho_y and the sparsity term (fixed.pdf §9.5, Eq. 12/21, Prop 12.1/B6).

Everything here consumes ONLY (alpha, rho) — the geometry lives in space.py,
the E-step densities in estep.py. Pieces:

* bkl + L_spr (Eq. 21): Prop 12.1 — non-negative, zero iff a = rho, strictly
  convex in a with f'' = 1/(a(1-a)) >= 4, continuous extension to a in {0, 1}.
* rho_mstep (Eq. 12): self-consistency M-step rho_y = mean_{t in y} alpha_t,
  detached, clamped to [rho_min, 1 - rho_min] to preserve strict interiority.
* option_b_seed (§9.5, B6(v)): threshold-free fixed-point seed for rho_y —
  Banach iteration of F(u) = logit(mean_t sigma(Delta_t + u)) on the
  interiority box; the contraction identity F'(u) = E[P(1-P)] / (m(1-m)) < 1
  is asserted from the iterates, never assumed.
"""

import math

import torch

__all__ = ["bkl", "sparsity_loss", "rho_mstep", "burst_delta_scores", "option_b_seed",
           "logit_scalar"]

_LOG_EPS = 1e-12


def logit_scalar(p, eps=_LOG_EPS):
    p = min(max(float(p), eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def _logit_tensor(p, eps=_LOG_EPS):
    p = p.clamp(eps, 1.0 - eps)
    return torch.log(p) - torch.log1p(-p)


# ------------------------------------------------------------------ bkl / L_spr

def bkl(a, rho):
    """Elementwise binary divergence a*log(a/rho) + (1-a)*log((1-a)/(1-rho)).

    Prop 12.1(ii): the 0*log-0 = 0 extension is applied where a hits {0, 1}.
    rho is the DETACHED M-step parameter (Eq. 12) — gradients flow to `a`
    only; passing a grad-tracking rho raises instead of silently detaching
    (owner policy, B6(i)).
    """
    if not (torch.is_tensor(a) and torch.is_tensor(rho)):
        raise TypeError("bkl expects torch tensors for a and rho")
    if rho.requires_grad:
        raise ValueError("bkl: rho must be detached (Eq. 12 is an M-step quantity)")
    if not torch.isfinite(a).all() or not torch.isfinite(rho).all():
        raise ValueError("bkl: non-finite input")
    if (a < 0).any() or (a > 1).any():
        raise ValueError(f"bkl: a must lie in [0,1], got [{a.min():.4f}, {a.max():.4f}]")
    if (rho <= 0).any() or (rho >= 1).any():
        raise ValueError(f"bkl: rho must lie strictly in (0,1) — interiority clamp violated, "
                         f"got [{rho.min():.6f}, {rho.max():.6f}]")
    tiny = torch.finfo(a.dtype).tiny  # dtype-safe 0*log-0 extension (1e-300 is not a float32)
    pos = a.clamp(min=0.0)
    t1 = torch.where(pos > 0, pos * (pos.clamp(min=tiny).log() - torch.log(rho)),
                     torch.zeros_like(pos))
    neg = (1.0 - a).clamp(min=0.0)
    t2 = torch.where(neg > 0, neg * (neg.clamp(min=tiny).log() - torch.log1p(-rho)),
                     torch.zeros_like(neg))
    return t1 + t2


def sparsity_loss(alpha, labels, rho):
    """L_spr (Eq. 21) = sum_t bkl(alpha_t, rho_{y_t}) — summed, NOT averaged:
    §12.1 bounds its per-term gradient by log(alpha_max(1-rho_min)/(alpha_min(1-rho_max)))
    which is what keeps the whole objective Lipschitz on the compact box."""
    if alpha.dim() != 1 or labels.shape != alpha.shape:
        raise ValueError(f"sparsity_loss expects alpha (B,) and labels (B,), "
                         f"got {tuple(alpha.shape)}/{tuple(labels.shape)}")
    if labels.dtype != torch.long:
        raise ValueError(f"labels must be long, got {labels.dtype}")
    if rho.dim() != 1 or rho.shape[0] <= int(labels.max().item() if labels.numel() else -1):
        raise ValueError("sparsity_loss: rho must be a 1-D vector covering every label present")
    if alpha.numel() == 0:
        raise ValueError("sparsity_loss called on an empty batch (B14 discipline)")
    return bkl(alpha, rho[labels]).sum()


# ------------------------------------------------------------------ Eq. 12 M-step

def rho_mstep(alpha, labels, num_classes, rho_min, prev_rho=None):
    """Self-consistency rho_y = (1/N_y) sum_{t in y} alpha_t (Eq. 12), detached
    and clamped to [rho_min, 1 - rho_min].

    Classes absent from this update keep prev_rho (an explicit caller-supplied
    table — pass None to demand full class coverage instead).
    """
    if alpha.dim() != 1 or labels.shape != alpha.shape:
        raise ValueError("rho_mstep: alpha/labels must both be (B,)")
    if alpha.numel() == 0:
        raise ValueError("rho_mstep on an empty batch — nothing to refine (B14)")
    if labels.dtype != torch.long:
        raise ValueError(f"rho_mstep: labels must be long, got {labels.dtype}")
    if (labels < 0).any() or labels.max() >= num_classes:
        raise ValueError(f"rho_mstep: labels must lie in [0, {num_classes})")
    if not torch.isfinite(alpha).all() or (alpha < 0).any() or (alpha > 1).any():
        raise ValueError(f"rho_mstep: alpha must be finite in [0,1], got "
                         f"[{alpha.min():.4f}, {alpha.max():.4f}]")
    if not 0.0 < rho_min < 0.5:
        raise ValueError(f"rho_min must lie in (0, 0.5), got {rho_min}")
    a = alpha.detach().float()
    rho = torch.empty(int(num_classes), device=a.device, dtype=a.dtype)
    missing = []
    for c in range(int(num_classes)):
        m = labels == c
        if not bool(m.any()):
            if prev_rho is None:
                missing.append(c)
                rho[c] = rho_min  # overwritten below unless we raise
            else:
                rho[c] = float(prev_rho[c])
        else:
            rho[c] = float(a[m].mean().item())
    if missing:
        raise ValueError(
            f"rho_mstep: classes {missing} absent from this batch and no prev_rho supplied — "
            "refusing a silent rho fallback (owner policy)")
    return rho.clamp(rho_min, 1.0 - rho_min)


# ------------------------------------------------------------------ §9.5 Option-B seed

def burst_delta_scores(norm_sq, s, log_coeff):
    """Delta_t = ((1 - s^-2) / 2) * ||r_t||^2 - log_coeff * log(s) (§9.5).

    Derivation check (§9.5): the log-likelihood ratio of the raw window norm
    under the burst N(0, s^2 I) versus the standardised baseline bulk N(0, I)
    in d dimensions is exactly this with log_coeff = d — the PDF's "B log s"
    is that Gaussian normalisation term, so the caller default is the feature
    dimension. Exposed as a knob (config `loss.sm.seed.log_coeff`).
    """
    if not (s > 1.0) or not math.isfinite(s):
        raise ValueError(f"burst spread constant s must be > 1 (finite), got {s}")
    if not (log_coeff > 0) or not math.isfinite(log_coeff):
        raise ValueError(f"log_coeff must be positive/finite (baseline is d*log s), got {log_coeff}")
    d2 = torch.as_tensor(norm_sq, dtype=torch.float32)
    if not torch.isfinite(d2).all() or (d2 < 0).any():
        raise ValueError("burst_delta_scores: norm_sq must be finite and >= 0")
    return 0.5 * (1.0 - s ** -2) * d2 - log_coeff * math.log(s)


def option_b_seed(delta, rho_min, max_iter=64, tol=1e-10):
    """Threshold-free fixed-point seed for rho_y (§9.5, B6(v)).

    Solves u = F(u) with F(u) = logit(mean_t sigma(Delta_t + u)) by plain
    Banach iteration projected onto the interiority box
    I = [logit(rho_min), -logit(rho_min)], so the returned
    rho = sigma(u) always lies in [rho_min, 1 - rho_min] (the Eq. 12 clamp:
    when no interior fixed point exists the seed legitimately saturates at
    the boundary, which is exactly the clamped self-consistency solution).

    Returns (rho, diag) with diag = {iterations, residual, kappa, u_star,
    saturated}. F'(u) = E[P(1-P)] / (m(1-m)) is evaluated along the orbit;
    an interior fixed point with kappa >= 1 contradicts B6(v) and raises —
    the contraction is CERTIFIED, not assumed (Track-B assertion discipline).
    """
    # float64 orbit: float32 jitter around the fixed point (~ulp of the logit) would make
    # the absolute convergence test below unreachable and falsely trip the no-convergence raise.
    d = torch.as_tensor(delta).reshape(-1).to(torch.float64)
    if d.numel() == 0:
        raise ValueError("option_b_seed: empty score vector")
    if not torch.isfinite(d).all():
        raise ValueError("option_b_seed: non-finite Delta scores")
    if not 0.0 < rho_min < 0.5:
        raise ValueError(f"rho_min must lie in (0, 0.5), got {rho_min}")
    if max_iter < 1 or tol <= 0:
        raise ValueError("option_b_seed: max_iter >= 1 and tol > 0 required")
    if float(d.std()) == 0.0:
        raise ValueError(
            "option_b_seed: degenerate scores (all Delta equal) — Var(P) = 0 makes F'(u) = 1, "
            "the B6(v) contraction hypothesis is violated by construction")

    lo, hi = logit_scalar(rho_min), -logit_scalar(rho_min)
    u = 0.0
    kappa = 0.0
    residual = float("inf")
    it = 0
    for it in range(1, int(max_iter) + 1):
        p = torch.sigmoid(d + u)
        m = float(p.mean().item())
        mc = min(max(m, _LOG_EPS), 1.0 - _LOG_EPS)
        f_u = math.log(mc / (1.0 - mc))
        num = float((p * (1.0 - p)).mean().item())
        den = m * (1.0 - m)
        if den > 0:
            kappa = max(kappa, num / den)
        u_new = min(max(f_u, lo), hi)
        residual = abs(u_new - u)
        u = u_new
        if residual < tol:
            break
    else:
        raise RuntimeError(
            f"option_b_seed: Banach iteration did not converge in {max_iter} steps "
            f"(residual {residual:.3e}) — check tol/rho_min box; do NOT ship a non-fixed seed")
    interior = (lo + 1e-6) < u < (hi - 1e-6)
    if interior and kappa >= 1.0 - 1e-9:
        raise RuntimeError(
            f"option_b_seed: contraction certificate failed at an interior fixed point "
            f"(kappa = {kappa:.6f} >= 1) — B6(v) violated, scores are numerically degenerate")
    rho = 1.0 / (1.0 + math.exp(-u))
    diag = {"iterations": it, "residual": residual, "kappa": kappa, "u_star": u,
            "saturated": not interior}
    return rho, diag
