"""Readiness gate + OT gradient-ratio coefficient (fixed.pdf §8.3, §9.7 Eq. 13, §11.1,
Prop 12.4(iv), Thm 12.5(i), B5(iii)).

Eq. 13:   g = clamp(C_K^DCL / log(K-1), g_floor, 1),
          lambda_OT = c * ||grad_z~ L_anc|| / ||grad_z~ L_em|| * g

Both norms are gradients w.r.t. the SAME variable z~ (Remark 8.2) — the ratio
is measured on the projected-embedding tensor via torch.autograd.grad
(grad_norm below); the batch norms are EMA-carried between steps so single
spikes cannot throttle or unleash the OT term. The product
lambda_OT * ||grad L_em|| = c * ||grad L_anc|| * g stays bounded as
||grad L_em|| -> 0: that regime returns EXACTLY 0 (Thm 12.5(i) OT-inert),
never a capped surrogate.

Prop 12.4(iv): the gate is continuous, non-decreasing in C_K and
1/log(K-1)-Lipschitz away from the clamp boundaries.
B5(iii): c <= c_max < 1 keeps phi'(0) <= -(1 - c*g)A < 0 along the collapse
escape — enforced at config load by check_c_gate, never silently clipped.

Gradient routing is never invented: value() raises before the first
observation (Stage 6 must skip the OT term during bootstrap).
"""

import math

import torch

__all__ = ["readiness_gate", "OTCoefficient", "grad_norm", "check_c_gate"]

_EM_FLOOR = 1e-8  # OT-inert numerical floor (float32-safe)


def readiness_gate(C_K, K, g_floor=0.1):
    """g = clamp(C_K / log(K-1), g_floor, 1) — Eq. 13.

    A non-positive finite-sample certificate (possible: worse than random
    negatives) floors at g_floor; K < 2 or non-finite inputs raise. C_K may
    be a scalar tensor — it is only READ here (a diagnostic, never part of a
    backward graph)."""
    K = int(K)
    if K < 2:
        raise ValueError(f"readiness_gate: K must be >= 2 (log(K-1) > 0), got {K}")
    if not (0.0 <= g_floor < 1.0):
        raise ValueError(f"readiness_gate: g_floor must lie in [0,1), got {g_floor}")
    c_val = float(C_K.detach().item()) if torch.is_tensor(C_K) else float(C_K)
    if not math.isfinite(c_val):
        raise ValueError(f"readiness_gate: non-finite certificate C_K={c_val}")
    ratio = c_val / math.log(K - 1)
    return min(1.0, max(g_floor, ratio))


def check_c_gate(c, c_min=0.05, c_max=0.9):
    """B5(iii) config guard: 0 < c_min and c_max < 1 and c_min <= c <= c_max.
    Raises at config load; the coefficient is NEVER clipped into range."""
    if not (0.0 < c_min):
        raise ValueError(f"check_c_gate: c_min must be > 0, got {c_min}")
    if not (c_max < 1.0):
        raise ValueError(f"check_c_gate: c_max must be strictly < 1 (B5(iii) anchor "
                         f"dominance), got {c_max}")
    c = float(c)
    if not math.isfinite(c) or not (c_min <= c <= c_max):
        raise ValueError(f"check_c_gate: c={c} outside admissible [{c_min}, {c_max}]")
    return c


class OTCoefficient:
    """EMA-carried lambda_OT = c * (anc_ema / em_ema) * g (§8.3, §11.1).

    Pure float bookkeeping (no tensors) — checkpointable via state_dict.
    ``momentum`` is the carry weight on the PREVIOUS value, the same
    convention as NullspaceState/queue EMAs; momentum=0 => latest-wins.
    First observation initialises (no zero-bias); value() before that raises.
    """

    def __init__(self, momentum=0.9, em_floor=_EM_FLOOR):
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"OTCoefficient: momentum must lie in [0,1), got {momentum}")
        if not (em_floor > 0.0) or not math.isfinite(em_floor):
            raise ValueError(f"OTCoefficient: em_floor must be positive/finite, got {em_floor}")
        self.momentum = float(momentum)
        self.em_floor = float(em_floor)
        self.anc = None
        self.em = None

    @property
    def ready(self):
        return self.anc is not None and self.em is not None

    @staticmethod
    def _check_norm(v, name):
        v = float(v)
        if not math.isfinite(v) or v < 0.0:
            raise ValueError(f"OTCoefficient: {name} must be finite >= 0, got {v}")
        return v

    def observe(self, anchor_grad_norm, em_grad_norm):
        a = self._check_norm(anchor_grad_norm, "anchor_grad_norm")
        e = self._check_norm(em_grad_norm, "em_grad_norm")
        m = self.momentum
        self.anc = a if self.anc is None else m * self.anc + (1.0 - m) * a
        self.em = e if self.em is None else m * self.em + (1.0 - m) * e
        return self

    def value(self, c, g):
        c = float(c)
        g = float(g)
        if not self.ready:
            raise RuntimeError(
                "OTCoefficient.value() before any observe(): Stage 6 bootstraps with the OT "
                "term OFF (g=0 / gate closed) — inventing a coefficient here is a wiring bug")
        if not (0.0 <= g <= 1.0):
            raise ValueError(f"OTCoefficient: gate g must lie in [0,1], got {g}")
        if not math.isfinite(c) or c < 0.0:
            raise ValueError(f"OTCoefficient: c must be finite >= 0, got {c}")
        if g == 0.0 or self.anc == 0.0:
            return 0.0
        if self.em <= self.em_floor:
            return 0.0  # Thm 12.5(i): OT gradient vanishing => lambda is exactly 0
        return c * (self.anc / self.em) * g

    def inert(self):
        return self.ready and self.em <= self.em_floor

    def state_dict(self):
        return {"anc": self.anc, "em": self.em, "momentum": self.momentum}

    def load_state_dict(self, payload):
        """Restores state AND hyperparameters — a checkpoint trained at momentum
        0.8 must resume at momentum 0.8, silently switching smoothing would
        change the dynamics the gate was tuned under."""
        if not isinstance(payload, dict) or not {"anc", "em", "momentum"} <= set(payload):
            raise ValueError("OTCoefficient.load_state_dict: payload needs keys "
                             "anc/em/momentum (strict, no partial restore)")
        self.momentum = self._check_norm(payload["momentum"], "momentum")
        if not (0.0 <= self.momentum < 1.0):
            raise ValueError(f"OTCoefficient.load_state_dict: momentum out of range "
                             f"{payload['momentum']}")
        self.anc = None if payload["anc"] is None else self._check_norm(payload["anc"], "anc")
        self.em = None if payload["em"] is None else self._check_norm(payload["em"], "em")
        return self


def grad_norm(loss, z_tilde, retain_graph=True):
    """||grad_{z~} loss|| over the whole batch (Frobenius norm of the stacked
    per-window gradients), measured with torch.autograd.grad so parameter
    .grad buffers are untouched (§8.3). z~ may be a non-leaf (it is
    NullspaceState.project output); a loss disconnected from it raises.
    retain_graph defaults to True — the same graph is backwarded later."""
    if not (torch.is_tensor(loss) and loss.dim() == 0):
        shape = tuple(loss.shape) if torch.is_tensor(loss) else type(loss).__name__
        raise ValueError(f"grad_norm: loss must be a scalar tensor, got {shape}")
    if not (torch.is_tensor(z_tilde) and z_tilde.requires_grad):
        raise ValueError("grad_norm: z_tilde must require grad (the projected embedding "
                         "tensor or any upstream activation the term differentiates through)")
    if not loss.requires_grad:
        raise ValueError("grad_norm: loss carries no autograd graph w.r.t. z~")
    (gz,) = torch.autograd.grad(loss, z_tilde, retain_graph=retain_graph,
                                create_graph=False, allow_unused=False)
    return float(gz.norm().item())
