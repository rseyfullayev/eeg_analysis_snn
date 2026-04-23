"""
Muon (MomentUm Orthogonalized by Newton-schulz) optimizer and hybrid wrappers.

Uses the official torch.optim.Muon (PyTorch >= 2.9) for the core optimizer.
Muon should only be used for hidden weight layers (ndim >= 2).
Biases, normalization parameters, SNN neuron dynamics, and other 1D/scalar
parameters should be optimized using a standard method such as AdamW.
"""

import torch
import torch.optim as optim


# ──────────────────────── Muon Wrapper ──────────────────────── #

class Muon:
    """Wrapper around torch.optim.Muon that handles >2D tensor reshaping.

    torch.optim.Muon expects 2D matrices for Newton-Schulz orthogonalization.
    Conv filters (e.g. [out, in, kH, kW]) must be reshaped to 2D
    [out, in*kH*kW] before the step, and restored to their original shape
    after. This wrapper handles that transparently.

    The wrapper exposes the same interface as a standard optimizer so it
    works seamlessly with HybridOptimizer/HybridScheduler.
    """

    def __init__(self, param_groups, lr=0.02, weight_decay=0.0, momentum=0.95):
        # Record original shapes for every >2D parameter
        self._nd_shapes = {}  # id(param) -> original shape
        for group in param_groups:
            for p in group['params']:
                if p.ndim > 2:
                    self._nd_shapes[id(p)] = p.shape
                    # Flatten to 2D NOW so torch.optim.Muon.__init__ passes
                    # its ndim==2 validation check
                    p.data = p.data.view(p.size(0), -1)

        self._inner = torch.optim.Muon(
            param_groups, lr=lr, weight_decay=weight_decay, momentum=momentum
        )

        # Immediately restore original shapes so model.forward() works
        for group in param_groups:
            for p in group['params']:
                if id(p) in self._nd_shapes:
                    p.data = p.data.view(self._nd_shapes[id(p)])

    # --- Reshape helpers ---
    def _flatten_nd(self):
        """Reshape >2D param.data and param.grad to 2D [fan_out, fan_in]."""
        for group in self._inner.param_groups:
            for p in group['params']:
                if id(p) in self._nd_shapes:
                    p.data = p.data.view(p.size(0), -1)
                    if p.grad is not None:
                        p.grad = p.grad.view(p.size(0), -1)

    def _restore_nd(self):
        """Restore >2D params back to their original shapes."""
        for group in self._inner.param_groups:
            for p in group['params']:
                if id(p) in self._nd_shapes:
                    orig = self._nd_shapes[id(p)]
                    p.data = p.data.view(orig)
                    if p.grad is not None:
                        p.grad = p.grad.view(orig)

    # --- Core optimizer interface ---
    @torch.no_grad()
    def step(self, closure=None):
        self._flatten_nd()
        loss = self._inner.step(closure)
        self._restore_nd()
        return loss

    def zero_grad(self, set_to_none=True):
        self._inner.zero_grad(set_to_none=set_to_none)

    @property
    def param_groups(self):
        return self._inner.param_groups

    def state_dict(self):
        return self._inner.state_dict()

    def load_state_dict(self, state_dict):
        self._inner.load_state_dict(state_dict)


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
