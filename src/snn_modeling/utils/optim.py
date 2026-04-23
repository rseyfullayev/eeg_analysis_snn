"""
Muon (MomentUm Orthogonalized by Newton-schulz) optimizer and hybrid wrappers.

Reference: https://kellerjordan.github.io/posts/muon/
Based on: https://github.com/KellerJordan/Muon

Muon internally runs standard SGD-momentum, and then performs an orthogonalization
post-processing step, in which each 2D parameter's update is replaced with the
nearest orthogonal matrix via Newton-Schulz iteration.

Muon should only be used for hidden weight layers (ndim >= 2).
Biases, normalization parameters, SNN neuron dynamics, and other 1D/scalar
parameters should be optimized using a standard method such as AdamW.
"""

import torch
import torch.optim as optim


# ──────────────────────── Newton-Schulz core ──────────────────────── #

def zeropower_via_newtonschulz5(G, steps: int = 5):
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.

    Uses a quintic iteration whose coefficients are selected to maximize the
    slope at zero for rapid convergence in bfloat16 on GPU.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum_buf, beta=0.95, ns_steps=5, nesterov=True):
    """Compute a single Muon update: momentum + Newton-Schulz orthogonalization."""
    original_shape = grad.shape
    momentum_buf.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum_buf, beta) if nesterov else momentum_buf
    if update.ndim > 2:  # conv / higher-dim filters: collapse to 2D for NS
        update = update.view(update.size(0), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update.view(original_shape)


# ──────────────────────── Single-device Muon ──────────────────────── #

class Muon(torch.optim.Optimizer):
    """Single-device Muon optimizer for >= 2D hidden-layer parameters.

    Args:
        params: List of parameters (must all be ndim >= 2).
        lr: Learning rate in units of spectral norm per update.
        weight_decay: AdamW-style weight decay.
        momentum: Momentum coefficient (0.95 is typically fine).
        ns_steps: Number of Newton-Schulz iterations.
    """

    def __init__(self, params, lr=0.02, weight_decay=0.0, momentum=0.95, ns_steps=5):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(
                    p.grad, state["momentum_buffer"],
                    beta=group["momentum"],
                    ns_steps=group["ns_steps"],
                )
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


# ──────────────────────── Hybrid Wrappers ──────────────────────── #

class HybridOptimizer:
    """Wraps a Muon optimizer and an AdamW optimizer behind a unified interface.

    Transparently forwards .step(), .zero_grad(), .state_dict(), and
    .load_state_dict() so the rest of the training loop does not need
    to know about the split.
    """

    def __init__(self, opt_muon, opt_adamw):
        self.opt_muon = opt_muon
        self.opt_adamw = opt_adamw

    # --- Core training interface ---
    def step(self):
        if self.opt_muon is not None:
            self.opt_muon.step()
        if self.opt_adamw is not None:
            self.opt_adamw.step()

    def zero_grad(self, set_to_none=True):
        if self.opt_muon is not None:
            self.opt_muon.zero_grad(set_to_none=set_to_none)
        if self.opt_adamw is not None:
            self.opt_adamw.zero_grad(set_to_none=set_to_none)

    # --- Checkpointing ---
    def state_dict(self):
        return {
            'muon': self.opt_muon.state_dict() if self.opt_muon else None,
            'adamw': self.opt_adamw.state_dict() if self.opt_adamw else None,
        }

    def load_state_dict(self, state_dict):
        # Support loading from both hybrid and legacy (pure AdamW) checkpoints
        if 'muon' in state_dict and 'adamw' in state_dict:
            if self.opt_muon is not None and state_dict.get('muon') is not None:
                self.opt_muon.load_state_dict(state_dict['muon'])
            if self.opt_adamw is not None and state_dict.get('adamw') is not None:
                self.opt_adamw.load_state_dict(state_dict['adamw'])
        else:
            # Legacy checkpoint: entire state is for a single AdamW
            if self.opt_adamw is not None:
                self.opt_adamw.load_state_dict(state_dict)

    # --- Expose param_groups so clip_grad_norm_ can find all parameters ---
    @property
    def param_groups(self):
        groups = []
        if self.opt_muon is not None:
            groups.extend(self.opt_muon.param_groups)
        if self.opt_adamw is not None:
            groups.extend(self.opt_adamw.param_groups)
        return groups


class HybridScheduler:
    """Wraps two LR schedulers (one for Muon, one for AdamW) behind a unified interface.

    Forwards .step() to both and returns the AdamW LR from .get_last_lr()
    (since that's the one logged to W&B / TensorBoard).
    """

    def __init__(self, sched_muon, sched_adamw):
        self.sched_muon = sched_muon
        self.sched_adamw = sched_adamw

    def step(self):
        if self.sched_muon is not None:
            self.sched_muon.step()
        if self.sched_adamw is not None:
            self.sched_adamw.step()

    def get_last_lr(self):
        """Return AdamW LR (primary) for logging."""
        if self.sched_adamw is not None:
            return self.sched_adamw.get_last_lr()
        if self.sched_muon is not None:
            return self.sched_muon.get_last_lr()
        return [0.0]

    # --- Checkpointing ---
    def state_dict(self):
        return {
            'muon': self.sched_muon.state_dict() if self.sched_muon else None,
            'adamw': self.sched_adamw.state_dict() if self.sched_adamw else None,
        }

    def load_state_dict(self, state_dict):
        # Support loading from both hybrid and legacy (pure scheduler) checkpoints
        if 'muon' in state_dict and 'adamw' in state_dict:
            if self.sched_muon is not None and state_dict.get('muon') is not None:
                self.sched_muon.load_state_dict(state_dict['muon'])
            if self.sched_adamw is not None and state_dict.get('adamw') is not None:
                self.sched_adamw.load_state_dict(state_dict['adamw'])
        else:
            # Legacy checkpoint: entire state is for a single scheduler
            if self.sched_adamw is not None:
                self.sched_adamw.load_state_dict(state_dict)
