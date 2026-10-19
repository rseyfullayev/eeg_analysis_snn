"""Queue v2 for the SMC objective (fixed.pdf §9).

One ring buffer, TWO readouts (Rm 9.2):
  * beta-weighted per-class clouds -> unbalanced OT reference nu_y (Stage 4)
  * uniform whole-queue negatives  -> projected DCL anchor (Stage 5)

Each entry stores: the norm-floored nullspace key r_j in B^d, its class y_j,
its burst weight beta_j = alpha^(1A)_j (detached), its window_id (§9.4 leakage
guard: same-window keys are masked at READ time on BOTH readouts), and the
enqueue step (staleness accounting, §9.4). Queue-vs-queue use is forbidden
(Prop 9.1): every readout pairs with current-batch gradients upstream.

§9.6 sizing helpers (floor/ceiling) and the §9.4 staleness bound are exact
formula ports; training setup (Stage 6) must call check_sizing()/check_staleness().
"""

import math

import torch
import torch.nn as nn


# ------------------------------------------------------------------ §9.6 / §9.4

def queue_floor(C_const, d_eff, tol, rho_tilde_min):
    """N_q >= C * d_eff / (tol^2 * rho_tilde_min)  (§9.6, prototype richness)."""
    for name, v in (("C_const", C_const), ("d_eff", d_eff), ("tol", tol),
                    ("rho_tilde_min", rho_tilde_min)):
        if not (v > 0) or not math.isfinite(v):
            raise ValueError(f"queue_floor: {name} must be positive/finite, got {v}")
    return int(math.ceil(C_const * d_eff / (tol * tol * rho_tilde_min)))


def queue_ceiling(eps_tol, B, T_total, L, D):
    """N_q <= eps_tol * B * T_total / (10 * L * D)  (§9.6, staleness ceiling)."""
    for name, v in (("eps_tol", eps_tol), ("B", B), ("T_total", T_total),
                    ("L", L), ("D", D)):
        if not (v > 0) or not math.isfinite(v):
            raise ValueError(f"queue_ceiling: {name} must be positive/finite, got {v}")
    return int((eps_tol * B * T_total) // (10 * L * D))


def check_sizing(n_q, floor, ceiling):
    """§9.6: N*_q is the floor rounded up, accepted only if it lies under the ceiling."""
    if not floor <= n_q <= ceiling:
        raise ValueError(
            f"§9.6 sizing violated: n_q={n_q} outside [floor={floor}, ceiling={ceiling}]. "
            "Raise tolerance or change batch/epoch budget — do NOT shrink the queue silently.")
    return n_q


def max_staleness(n_q, B, momentum, L, D):
    """staleness <= L * (N_q/B) * (1 - m) * D  (§9.4)."""
    if not 0.0 <= momentum < 1.0:
        raise ValueError(f"momentum must be in [0,1), got {momentum}")
    return L * (n_q / B) * (1.0 - momentum) * D


def check_staleness(n_q, B, momentum, L, D, eps_tol):
    bound = max_staleness(n_q, B, momentum, L, D)
    if bound > eps_tol:
        raise ValueError(
            f"§9.4 staleness bound violated: {bound:.4f} > eps_tol={eps_tol} "
            "(n_q, B, momentum, L, D as given) — shrink queue or raise momentum.")
    return bound


# ------------------------------------------------------------------ queue

class SupMoCoQueue(nn.Module):
    """Ring buffer of (key, class, beta, window_id, step) entries with the two
    §9.2 readouts. Buffers are registered so checkpoint save/load stays strict."""

    def __init__(self, queue_size, feature_dim, num_classes=5):
        super().__init__()
        if queue_size < 1 or feature_dim < 1:
            raise ValueError(f"queue_size/feature_dim must be >=1, got {queue_size}/{feature_dim}")
        self.queue_size = int(queue_size)
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.register_buffer("keys", torch.zeros(self.queue_size, self.feature_dim))
        self.register_buffer("labels", torch.full((self.queue_size,), -1, dtype=torch.long))
        self.register_buffer("betas", torch.zeros(self.queue_size))
        self.register_buffer("window_ids", torch.full((self.queue_size,), -1, dtype=torch.long))
        self.register_buffer("steps", torch.full((self.queue_size,), -1, dtype=torch.long))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("filled", torch.zeros(1, dtype=torch.long))

    # ---------------------------------------------------------------- write

    @torch.no_grad()
    def enqueue(self, keys, labels, betas, window_ids, step):
        """Append current-window entries. keys must ALREADY be projected
        z-tilde (NullspaceState.project) and detached (they are stop-grad
        references; Prop 8.4/§9.1)."""
        n = keys.shape[0]
        if keys.requires_grad or betas.requires_grad:
            raise RuntimeError("enqueue expects detached keys/betas (stop-grad queue, Prop 8.4)")
        if keys.shape != (n, self.feature_dim):
            raise ValueError(f"keys must be (n, {self.feature_dim}), got {tuple(keys.shape)}")
        for name, t in (("labels", labels), ("betas", betas), ("window_ids", window_ids)):
            if t.shape != (n,):
                raise ValueError(f"{name} must be (n,)={n}, got {tuple(t.shape)}")
        if not torch.isfinite(keys).all() or not torch.isfinite(betas).all():
            raise ValueError("non-finite keys/betas in enqueue")
        kn = keys.norm(dim=1)
        if (kn > 1.0 + 1e-5).any():
            raise ValueError(
                f"queue keys must live in B^d (norm<=1), max row norm {kn.max():.6f} — "
                "project through NullspaceState first")
        if labels.dtype != torch.long or (labels < 0).any() or (labels >= self.num_classes).any():
            raise ValueError(f"labels must be long in [0,{self.num_classes})")
        if (betas < 0).any() or (betas > 1).any():
            raise ValueError(f"betas (alpha^(1A)) must lie in [0,1], got [{betas.min():.3f}, {betas.max():.3f}]")
        if window_ids.dtype != torch.long or (window_ids < 0).any():
            raise ValueError("window_ids must be long >= 0 (the -1 pad sentinel is not queue material)")
        if torch.unique(window_ids).numel() != n:
            raise ValueError("duplicate window_ids within one enqueue batch — window id derivation is broken")

        step = int(step)
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")

        if n >= self.queue_size:
            keep = slice(-self.queue_size, None)
            self.keys.copy_(keys[keep])
            self.labels.copy_(labels[keep])
            self.betas.copy_(betas[keep])
            self.window_ids.copy_(window_ids[keep])
            self.steps.copy_(torch.full((self.queue_size,), step, dtype=torch.long))
            self.ptr.zero_()
            self.filled.fill_(self.queue_size)
            return

        ptr = int(self.ptr.item())
        end = ptr + n
        if end <= self.queue_size:
            self.keys[ptr:end] = keys
            self.labels[ptr:end] = labels
            self.betas[ptr:end] = betas
            self.window_ids[ptr:end] = window_ids
            self.steps[ptr:end] = step
        else:
            first = self.queue_size - ptr
            second = n - first
            for buf, src in ((self.keys, keys), (self.labels, labels), (self.betas, betas),
                             (self.window_ids, window_ids)):
                buf[ptr:] = src[:first]
                buf[:second] = src[first:]
            self.steps[ptr:] = step
            self.steps[:second] = step
        self.ptr[0] = (ptr + n) % self.queue_size
        self.filled[0] = min(self.queue_size, int(self.filled.item()) + n)

    # ---------------------------------------------------------------- reads

    def _active(self):
        filled = int(self.filled.item())
        return slice(0, filled), filled

    def _not_excluded(self, sl, exclude_window_ids):
        if exclude_window_ids is None:
            return torch.ones(sl.stop - sl.start, dtype=torch.bool, device=self.keys.device)
        ex = torch.as_tensor(exclude_window_ids, dtype=torch.long, device=self.keys.device).view(-1)
        if ex.numel() == 0:
            return torch.ones(sl.stop - sl.start, dtype=torch.bool, device=self.keys.device)
        return ~torch.isin(self.window_ids[sl], ex)

    def read_negatives(self, exclude_window_ids=None):
        """Uniform whole-queue readout (anchor negatives, §9.2). Converges to the
        nullspace marginal p(z-tilde) (Prop 9.2). Returns (keys, labels, betas,
        window_ids) of surviving entries; possibly empty during cold start."""
        sl, filled = self._active()
        if filled == 0:
            empty = self.keys[:0]
            return (empty, empty.new_empty(0, dtype=torch.long), empty.new_empty(0),
                    empty.new_empty(0, dtype=torch.long))
        keep = self._not_excluded(sl, exclude_window_ids)
        return (self.keys[sl][keep], self.labels[sl][keep],
                self.betas[sl][keep], self.window_ids[sl][keep])

    def read_class_clouds(self, exclude_window_ids=None):
        """Per-class beta-weighted clouds (OT reference nu_y, Rm 9.2/9.3).

        Returns {y: {'keys','weights','beta_sum','n'}} for classes with entries;
        weights = beta/sum(beta) so the class slice is a probability cloud
        (Def. 5.2's nu_y). Classes with positive mass but sum(beta)=0 raise —
        that contradicts Rm 4.2 interiority and means upstream alpha is wrong.
        """
        sl, filled = self._active()
        out = {}
        if filled == 0:
            return out
        keep = self._not_excluded(sl, exclude_window_ids)
        keys = self.keys[sl][keep]
        labels = self.labels[sl][keep]
        betas = self.betas[sl][keep]
        for y in labels.unique().tolist():
            m = labels == y
            bsum = float(betas[m].sum().item())
            if bsum <= 0.0:
                raise ValueError(
                    f"class {y} cloud has {int(m.sum())} entries but sum(beta)=0 — "
                    "alpha^(1A) interiority (Rm 4.2) violated upstream")
            out[int(y)] = {
                "keys": keys[m],
                "weights": betas[m] / bsum,
                "beta_sum": bsum,
                "n": int(m.sum()),
            }
        return out

    # ---------------------------------------------------------------- diagnostics

    def ess_by_class(self, exclude_window_ids=None):
        """ESS_y = (sum beta)^2 / sum beta^2 (fixed.pdf §11.5, ladder step 3)."""
        clouds = {}
        sl, filled = self._active()
        if filled == 0:
            return clouds
        keep = self._not_excluded(sl, exclude_window_ids)
        betas = self.betas[sl][keep]
        labels = self.labels[sl][keep]
        for y in labels.unique().tolist():
            b = betas[labels == y]
            denom = float((b ** 2).sum().item())
            clouds[int(y)] = (float(b.sum().item()) ** 2 / denom) if denom > 0 else 0.0
        return clouds

    def summary(self):
        filled = int(self.filled.item())
        return {
            "fill_ratio": filled / float(max(1, self.queue_size)),
            "filled": filled,
            "queue_size": self.queue_size,
            "oldest_step": int(self.steps[:filled].min().item()) if filled else -1,
            "newest_step": int(self.steps[:filled].max().item()) if filled else -1,
        }
