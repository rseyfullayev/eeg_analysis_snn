"""
NaN / Inf Monitor for SNN Training Pipeline.

Diagnoses the root cause of numerical instabilities by inspecting
activations, gradients, loss components, and optimizer state at each
checkpoint in the forward-backward-step cycle.

Usage:
    monitor = NaNMonitor(model, verbose=True)
    ...
    # After forward pass:
    monitor.check_activations(backbone_feats=backbone_feats, outputs=outputs, ...)
    # After loss computation:
    monitor.check_loss(loss=loss, loss_dann=loss_dann, loss_mmd=loss_mmd, ...)
    # After backward:
    monitor.check_gradients()
    # After optimizer step:
    monitor.check_weights()
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict


class NaNMonitor:
    """Monitors model for NaN/Inf with diagnostic reasoning.

    Tracks where numerical instability first appears in the
    forward → loss → backward → step pipeline and provides
    likely root-cause explanations.
    """

    # Common failure modes and their signatures
    _DIAGNOSES = {
        'representational_collapse': (
            "REPRESENTATIONAL COLLAPSE: All features have near-zero variance.\n"
            "  → Muon's Newton-Schulz can amplify collapsed representations.\n"
            "  → cuML SVM will crash with ZeroDivisionError on gamma = 1/(n*var).\n"
            "  → Consider: lower Muon LR, warmup epochs, or check if encoder outputs are too similar."
        ),
        'gradient_explosion': (
            "GRADIENT EXPLOSION: Gradient norms exceed safe range (>1e4).\n"
            "  → Often caused by: high LR, missing grad clipping, or GRL alpha too large.\n"
            "  → Consider: reduce LR, add/tighten grad clipping, reduce DANN alpha."
        ),
        'loss_nan': (
            "LOSS NaN: The loss value itself is NaN before backward.\n"
            "  → Likely cause: log(0), division by zero in loss computation, or\n"
            "    degenerate inputs (e.g. all-same features → zero denominator in contrastive loss).\n"
            "  → Check: temperature scaling, partition function clamping, feature normalization."
        ),
        'weight_nan': (
            "WEIGHT NaN: Model weights contain NaN after optimizer step.\n"
            "  → Likely cause: NaN gradients propagated through optimizer update.\n"
            "  → This is usually a downstream effect — check gradient/loss diagnostics above."
        ),
        'grl_instability': (
            "GRL INSTABILITY: DANN head outputs contain extreme values.\n"
            "  → Gradient Reversal Layer amplifies gradients by alpha, which can\n"
            "    create feedback loops with high learning rates.\n"
            "  → Consider: reduce dann_alpha, use slower alpha annealing schedule."
        ),
        'mmd_kernel_overflow': (
            "MMD KERNEL OVERFLOW: Kernel matrix contains Inf/NaN.\n"
            "  → exp(-gamma * dist²) overflows when gamma * dist² is very large.\n"
            "  → Usually caused by very small median distance (collapsed features)\n"
            "    producing huge gamma = 1/(2σ²).\n"
            "  → Consider: clamp bandwidths, add epsilon to sigma."
        ),
        'contrastive_partition_collapse': (
            "CONTRASTIVE PARTITION COLLAPSE: Denominator in log-softmax is zero or NaN.\n"
            "  → All logits are -Inf after max-shift, causing exp(logits) ≈ 0 everywhere.\n"
            "  → Usually caused by: identical features (collapse), extreme temperature.\n"
            "  → Consider: increase temperature, check feature normalization."
        ),
    }

    def __init__(self, model, verbose=True, max_reports_per_epoch=3, live_tracker=None):
        self.model = model
        self.verbose = verbose
        self.max_reports = max_reports_per_epoch
        self._report_count = 0
        self._first_nan_location = None
        self._history = defaultdict(list)  # tracks stats over time
        self._live_tracker = live_tracker  # LiveTracker instance for Telegram/Discord alerts

    def reset_epoch(self):
        """Call at the start of each epoch to reset report count."""
        self._report_count = 0
        self._first_nan_location = None

    def _should_report(self):
        return self._report_count < self.max_reports

    def _report(self, tag, diagnosis_key, details=""):
        if not self._should_report():
            return
        self._report_count += 1
        if self._first_nan_location is None:
            self._first_nan_location = tag

        msg = f"\n{'='*70}\n"
        msg += f"  ⚠️  NaN/Inf DETECTED — {tag}\n"
        msg += f"{'='*70}\n"
        if details:
            msg += f"  Details: {details}\n"
        diagnosis_text = self._DIAGNOSES.get(diagnosis_key, '')
        if diagnosis_text:
            msg += f"\n  Diagnosis:\n  {diagnosis_text}\n"
        msg += f"{'='*70}\n"
        print(msg)

        # Forward to Telegram/Discord if live tracker is attached
        if self._live_tracker is not None:
            self._live_tracker.nan_alert(tag, diagnosis_text, details)

    @staticmethod
    def _tensor_stats(t, name="tensor"):
        """Return a dict of diagnostic stats for a tensor."""
        if t is None:
            return f"{name}: None"
        if not isinstance(t, torch.Tensor):
            return f"{name}: {t}"

        with torch.no_grad():
            has_nan = torch.isnan(t).any().item()
            has_inf = torch.isinf(t).any().item()
            if has_nan or has_inf:
                nan_count = torch.isnan(t).sum().item()
                inf_count = torch.isinf(t).sum().item()
                return (f"{name}: shape={list(t.shape)} | "
                        f"NaN={nan_count}/{t.numel()} | Inf={inf_count}/{t.numel()}")

            t_float = t.float()
            mn = t_float.min().item()
            mx = t_float.max().item()
            mean = t_float.mean().item()
            std = t_float.std().item() if t.numel() > 1 else 0.0
            return (f"{name}: shape={list(t.shape)} | "
                    f"min={mn:.4g} max={mx:.4g} mean={mean:.4g} std={std:.4g}")

    @staticmethod
    def _has_bad_values(t):
        """Check if tensor has NaN or Inf."""
        if t is None or not isinstance(t, torch.Tensor):
            return False
        return (torch.isnan(t).any().item() or torch.isinf(t).any().item())

    @staticmethod
    def _is_collapsed(t, var_threshold=1e-8):
        """Check if features show representational collapse (near-zero variance)."""
        if t is None or not isinstance(t, torch.Tensor) or t.numel() < 2:
            return False
        with torch.no_grad():
            var = t.float().var(dim=0).mean().item()
            return var < var_threshold

    # ──────────────────────── Check Points ──────────────────────── #

    def check_activations(self, epoch=None, batch_idx=None, **named_tensors):
        """Check intermediate activations for NaN/Inf/collapse.

        Pass named tensors: backbone_feats=..., outputs=..., key_features=..., etc.
        """
        problems = []
        collapsed = []

        for name, t in named_tensors.items():
            if t is None:
                continue
            if self._has_bad_values(t):
                problems.append(self._tensor_stats(t, name))
            if self._is_collapsed(t):
                collapsed.append(name)

        if problems:
            loc = f"Epoch {epoch} Batch {batch_idx}" if epoch is not None else "unknown"
            details = f"Location: {loc}\n" + "\n".join(f"    {p}" for p in problems)
            self._report("Activations", 'representational_collapse' if collapsed else 'loss_nan', details)
            return True

        if collapsed:
            # Silently count — summary printed once at epoch end
            if not hasattr(self, '_collapse_count'):
                self._collapse_count = 0
                self._collapse_names = set()
            self._collapse_count += 1
            self._collapse_names.update(collapsed)
            return False  # Not NaN, but worth noting

        return False

    def check_loss(self, epoch=None, batch_idx=None, **named_losses):
        """Check loss values for NaN/Inf.

        Pass named losses: loss=..., loss_dann=..., loss_mmd=..., etc.
        """
        problems = []
        for name, val in named_losses.items():
            if val is None:
                continue
            if isinstance(val, torch.Tensor):
                if self._has_bad_values(val):
                    problems.append(f"{name}={val.item() if val.numel() == 1 else 'tensor'} (NaN/Inf)")
            elif isinstance(val, float):
                if np.isnan(val) or np.isinf(val):
                    problems.append(f"{name}={val} (NaN/Inf)")

        if problems:
            loc = f"Epoch {epoch} Batch {batch_idx}" if epoch is not None else "unknown"
            details = f"Location: {loc}\n    " + "\n    ".join(problems)
            self._report("Loss Computation", 'loss_nan', details)
            return True
        return False

    def check_gradients(self, epoch=None, batch_idx=None):
        """Check all model gradients for NaN/Inf/explosion after backward().

        Returns True if problems found.
        """
        bad_grads = []
        max_grad_norm = 0.0
        total_params = 0
        nan_params = 0

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            total_params += 1
            grad = param.grad

            if self._has_bad_values(grad):
                nan_params += 1
                bad_grads.append(f"{name}: {self._tensor_stats(grad, 'grad')}")
            else:
                norm = grad.float().norm().item()
                max_grad_norm = max(max_grad_norm, norm)

        if bad_grads:
            loc = f"Epoch {epoch} Batch {batch_idx}" if epoch is not None else "unknown"
            details = (f"Location: {loc}\n"
                       f"    {nan_params}/{total_params} parameters have NaN/Inf gradients\n"
                       f"    First offenders:\n" +
                       "\n".join(f"      {g}" for g in bad_grads[:5]))
            self._report("Gradients", 'gradient_explosion', details)
            return True

        if max_grad_norm > 1e4 and self.verbose and self._should_report():
            loc = f"Epoch {epoch} Batch {batch_idx}" if epoch is not None else ""
            print(f"  [NaNMonitor] ⚡ Large gradient norm ({loc}): {max_grad_norm:.2e}")

        return False

    def check_weights(self, epoch=None, batch_idx=None):
        """Check model weights after optimizer step for NaN/Inf.

        Returns True if problems found.
        """
        bad_weights = []
        for name, param in self.model.named_parameters():
            if self._has_bad_values(param.data):
                bad_weights.append(f"{name}: {self._tensor_stats(param.data, 'weight')}")

        if bad_weights:
            loc = f"Epoch {epoch} Batch {batch_idx}" if epoch is not None else "unknown"
            details = (f"Location: {loc}\n"
                       f"    {len(bad_weights)} parameters have NaN/Inf weights\n"
                       f"    First offenders:\n" +
                       "\n".join(f"      {w}" for w in bad_weights[:5]))
            self._report("Weights (post-step)", 'weight_nan', details)
            return True
        return False

    def check_dann(self, backbone_feats, subj_preds=None, dann_alpha=None,
                   epoch=None, batch_idx=None):
        """Specialized check for DANN head instability.

        Call after model.dann_head(backbone_feats).
        """
        if self._has_bad_values(backbone_feats):
            self._report("DANN Input", 'grl_instability',
                         f"backbone_feats contain NaN/Inf BEFORE DANN head.\n"
                         f"    {self._tensor_stats(backbone_feats, 'backbone_feats')}")
            return True

        if subj_preds is not None and self._has_bad_values(subj_preds):
            details = (f"DANN head output has NaN/Inf.\n"
                       f"    {self._tensor_stats(subj_preds, 'subj_preds')}\n"
                       f"    {self._tensor_stats(backbone_feats, 'backbone_feats')}")
            if dann_alpha is not None:
                details += f"\n    GRL alpha = {dann_alpha:.4f}"
            self._report("DANN Output", 'grl_instability', details)
            return True

        # Check for extreme logit values that might overflow softmax
        if subj_preds is not None:
            with torch.no_grad():
                max_logit = subj_preds.float().abs().max().item()
                if max_logit > 50.0 and self.verbose and self._should_report():
                    print(f"  [NaNMonitor] ⚡ DANN logits extreme: max|logit|={max_logit:.1f} "
                          f"(alpha={dann_alpha:.4f})")
        return False

    def check_mmd(self, features, kernel_matrix=None, epoch=None, batch_idx=None):
        """Specialized check for MMD kernel overflow."""
        if self._has_bad_values(features):
            self._report("MMD Input", 'mmd_kernel_overflow',
                         f"Features contain NaN/Inf.\n"
                         f"    {self._tensor_stats(features, 'features')}")
            return True

        if kernel_matrix is not None and self._has_bad_values(kernel_matrix):
            details = (f"Kernel matrix has NaN/Inf.\n"
                       f"    {self._tensor_stats(kernel_matrix, 'kernel_matrix')}\n"
                       f"    {self._tensor_stats(features, 'features')}")
            self._report("MMD Kernel", 'mmd_kernel_overflow', details)
            return True
        return False

    def full_check(self, epoch=None, batch_idx=None, **named_tensors):
        """Convenience: run all activation + loss checks at once.

        Returns the first diagnosis key if problems found, else None.
        """
        if self.check_activations(epoch=epoch, batch_idx=batch_idx, **named_tensors):
            return 'activations'
        return None

    def summary(self):
        """Print end-of-epoch summary."""
        if self._report_count > 0:
            print(f"\n  [NaNMonitor] Epoch summary: {self._report_count} NaN/Inf events detected. "
                  f"First occurrence at: {self._first_nan_location}")
        elif self.verbose:
            print(f"  [NaNMonitor] Epoch clean — no NaN/Inf detected.")
        
        # Collapse summary (throttled to 1 line per epoch instead of per-batch spam)
        collapse_count = getattr(self, '_collapse_count', 0)
        if collapse_count > 0:
            names = ', '.join(sorted(getattr(self, '_collapse_names', set())))
            msg = f"Collapse: {collapse_count} batches had near-zero variance ({names})"
            print(f"  [NaNMonitor] ⚡ {msg}")
            if self._live_tracker is not None:
                self._live_tracker.add_warning(msg)
            self._collapse_count = 0
            self._collapse_names = set()
