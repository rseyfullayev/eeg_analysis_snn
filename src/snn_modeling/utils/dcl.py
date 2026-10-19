"""Projected DCL anchor + finite-negative information certificate (fixed.pdf §7.4-7.5, Eq. 22).

L_DCL is the decoupled contrastive loss on the shared ball geometry — online
queries z~ (grad-carrying) against their momentum keys and the uniform queue
negatives; score s(x, z~) = (z~_q · z~_k)/tau_DCL with the positive REMOVED
from the negative partition [7] (decoupled form). It exposes the finite
negative certificate

    C_K^DCL = log(K - 1) - L_DCL,      IDV(s) <= C_inf <= C_K <= I(x; z~)   (Thm 7.3)

which drives the readiness gate g (Eq. 13, gates.py) — Rm 7.4: because the
anchor is evaluated directly on z~ in B^d ∩ span(U)^perp, the retained mutual
information is certified INSIDE the exact nullspace OT and R_b read on.

LOTO ghosting retained: negatives sampled from a query's own trial are
ghosted (neither positive nor negative) — they are not independent draws of
p(z~) and would turn the certificate into an identity shortcut (cross-trial
agreement only).

Gradient routing is never silent: `key`/`negatives` are stop-grad material
(grad-tracking tensors are REJECTED, not detached here); gradients may flow
through `query` only.
"""

import math

import torch

__all__ = ["dcl_loss_certificate", "ProjectedAnchor"]

_BALL_TOL = 1e-5


def _check_rows(t, name, ball=True):
    if t.dim() != 2:
        raise ValueError(f"dcl: {name} must be 2-D (n, D), got {tuple(t.shape)}")
    if not torch.isfinite(t.detach()).all():
        raise ValueError(f"dcl: {name} contains non-finite values")
    if ball:
        nrm = t.detach().norm(dim=1)
        if (nrm > 1.0 + _BALL_TOL).any():
            raise ValueError(
                f"dcl: {name} rows must lie in B^d (max norm {float(nrm.max()):.6f}) — "
                "produce them via NullspaceState.project")


def _check_stop_grad(t, name):
    if t.requires_grad:
        raise ValueError(
            f"dcl: {name} is stop-grad M-step/reference material — detach at the source "
            "(queue readout / momentum path); this module never silently detaches (owner policy)")


def dcl_loss_certificate(query, key, negatives, tau, *, query_group=None, neg_group=None,
                         ball_check=True):
    """Decoupled contrastive loss + C_K^DCL certificate (Thm 7.3/7.4).

    query (B,D)   online z~ — the ONLY grad-carrying input.
    key (B,D)     momentum-encoder z~ for the same windows (stop-grad).
    negatives (M,D) uniform whole-queue z~ keys (stop-grad) — K = M + 1.
    query_group (B,), neg_group (M,)  long trial ids for LOTO ghosting
                     (None disables ghosting; both must be given together).
    ball_check    enforce the B^d domain (§11.4i 'projected' mode); the 'raw'
                     ablation feeds unit-norm encoder outputs z and relaxes it.

    Per query i with ghost-filtered counts M_i >= 1:
        L_i   = logsumexp_j (q_i·n_j)/tau - (q_i·k_i)/tau
        C_i   = log(M_i) - L_i                      (finite-negative certificate)
    Batch:  L_DCL = mean L_i,  C_K = mean C_i  (= log(K-1) - L_DCL for M_i = M).
    Fully-ghosted pools raise — a batch without negatives is not a DCL step.
    """
    _check_rows(query, "query", ball=ball_check)
    _check_rows(key, "key", ball=ball_check)
    _check_rows(negatives, "negatives", ball=ball_check)
    _check_stop_grad(key, "key")
    _check_stop_grad(negatives, "negatives")
    if query.shape != key.shape:
        raise ValueError(f"dcl: key {tuple(key.shape)} must match query {tuple(query.shape)}")
    if negatives.shape[0] == 0:
        raise ValueError(
            "dcl: empty negative pool — bootstrap order (§11.4ii) keeps the anchor off until "
            "the queue has turned over; do NOT run a step with no negatives")
    if query.shape[1] != negatives.shape[1]:
        raise ValueError(f"dcl: dim mismatch query D={query.shape[1]} vs negatives D={negatives.shape[1]}")
    if not (tau > 0.0) or not math.isfinite(tau):
        raise ValueError(f"dcl: temperature tau must be positive/finite, got {tau}")
    if (query_group is None) != (neg_group is None):
        raise ValueError("dcl: query_group and neg_group must be provided together (LOTO ghosting)")
    if query_group is not None:
        if query_group.dtype != torch.long or query_group.shape != (query.shape[0],):
            raise ValueError(f"dcl: query_group must be long (B,), got {query_group.dtype}/{tuple(query_group.shape)}")
        if neg_group.dtype != torch.long or neg_group.shape != (negatives.shape[0],):
            raise ValueError(f"dcl: neg_group must be long (M,), got {neg_group.dtype}/{tuple(neg_group.shape)}")

    s_pos = (query * key).sum(dim=1) / tau
    logits = query @ negatives.t() / tau
    if query_group is not None:
        keep = query_group.view(-1, 1) != neg_group.view(1, -1)
        logits = torch.where(keep, logits, logits.detach().new_full((), float("-inf")))
        m_i = keep.sum(dim=1)
    else:
        m_i = torch.full((query.shape[0],), negatives.shape[0], dtype=torch.long,
                         device=query.device)

    active = m_i > 0
    if not bool(active.any()):
        raise RuntimeError(
            "dcl: every query lost ALL negatives to LOTO ghosting — trial-id derivation is "
            "broken (refusing a zero-loss step, audit §3.5 row-8 discipline)")
    lse = torch.logsumexp(logits[active], dim=1)
    L_i = lse - s_pos[active]
    C_i = m_i[active].to(L_i.dtype).log() - L_i  # stay in the data dtype (.float() would
    #   silently downgrade a float64 certificate by ~5e-8 and break the exact C_K = 0
    #   constant-encoder guarantee of Thm 12.5(i))
    return {"L_DCL": L_i.mean(), "C_K": C_i.mean(), "per_query_L": L_i, "per_query_C": C_i,
            "M": int(negatives.shape[0]), "K": int(negatives.shape[0] + 1), "M_i": m_i[active],
            "n_active": int(active.sum()), "n_ghosted": int(query.shape[0]) - int(active.sum())}


class ProjectedAnchor(torch.nn.Module):
    """Eq. 22 anchor term L_anc = L_DCL(z~) with certificate; §11.4(i) ablation knob.

    anchor_on = "projected" (spec): every side must lie in B^d — enforced, because
                  the shared geometry is what makes the certificate certify the
                  nullspace the OT and R_b live in (Rm 7.4).
    anchor_on = "raw"       (ablation only): ball checks relaxed to the unit
                  encoder outputs z; the §11.3 anchor_on comparison runs the
                  identical math on the un-projected representation.
    """

    def __init__(self, tau_dcl=0.1, anchor_on="projected"):
        super().__init__()
        if not (tau_dcl > 0.0) or not math.isfinite(tau_dcl):
            raise ValueError(f"ProjectedAnchor: tau_dcl must be positive/finite, got {tau_dcl}")
        if anchor_on not in ("projected", "raw"):
            raise ValueError("ProjectedAnchor: anchor_on must be 'projected' or 'raw' (§11.4i), "
                             f"got {anchor_on!r}")
        self.tau_dcl = float(tau_dcl)
        self.anchor_on = anchor_on

    def forward(self, query, key, negatives, query_group=None, neg_group=None):
        return dcl_loss_certificate(query, key, negatives, self.tau_dcl,
                                    query_group=query_group, neg_group=neg_group,
                                    ball_check=self.anchor_on == "projected")
