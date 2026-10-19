"""Phase 1A model: Atrous-MobileNet encoder + contrastive projection head.

Stripped in Stage 0 tranche 2 of the SMC rebuild. Removed (fixed.pdf §13 dead
list + audit B3/B8/B21): GRL/DANN heads, VIB head, the SwiGLU-MIL
WindowReRanker, the inverse-weighted h_dmn path, the RTFM Top-K sieve, and the
broken SpikingUNet/UNet placeholders (B8). Phase 1B returns as the R_b
sufficient-statistic readout adapter (plan §5 Stage 7), not an MLP attention
tower, so nothing here should be re-added.
"""

import torch
import torch.nn as nn

from ..layers.stem import ProjectionHead


class SpikingMobileNetProjector(nn.Module):
    """Backbone encoder + ProjectionHead for the SupMoCo/DCL anchor.

    Legacy Phase-1B flags (use_swiglu/use_dann/use_vib/use_subj) are still
    accepted ONLY so stale configs fail loudly instead of silently building a
    different model than requested; any True value raises.
    """

    _RETIRED_FLAGS = {
        'use_swiglu': 'SwiGLU-MIL attention reranker',
        'use_dann': 'GRL/DANN adversarial subject head',
        'use_vib': 'variational information bottleneck head',
        'use_subj': 'explicit subject-routing head',
    }

    def __init__(self, encoder_backbone,
                 num_classes=5, feature_dim=256,
                 use_batchnorm=False,
                 use_swiglu=False, use_dann=False, use_vib=False, use_subj=False,
                 num_subjects=15, max_windows=1000):
        super().__init__()
        flag_values = {'use_swiglu': use_swiglu, 'use_dann': use_dann,
                       'use_vib': use_vib, 'use_subj': use_subj}
        requested = [name for name, flag in flag_values.items() if flag]
        if requested:
            raise NotImplementedError(
                "Retired Phase-1B adapter flags: "
                + "; ".join(f"{r} = {self._RETIRED_FLAGS[r]}" for r in requested)
                + ". The MIL/VIB/DANN adapter is replaced by the frozen-state "
                  "R_b = (1/T_b) * sum(alpha_t * h'_t) readout (fixed.pdf §10-§11, "
                  "plan §5 Stage 7). Remove these flags from the config.")

        self.encoder = encoder_backbone
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # Auto-detect if encoder uses batchnorm to sync the ProjectionHead
        has_bn = any(isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
                     for m in self.encoder.modules())
        self.classifier = ProjectionHead(feature_dim, 128,
                                         use_batchnorm=use_batchnorm or has_bn)

    def extract_features(self, x, K=None):
        """Backbone embedding BEFORE the projection head.

        Accepts raw time-first video (T, B, C, H, W) or precomputed 3D bag
        features (B, W, C) (v2 artifact 'z'; Stage 7 replaces the plain bag GAP
        with the alpha-weighted R_b). Returns (batch, feature_dim).
        """
        if x.dim() == 3:
            B, W, C_dim = x.shape
            if C_dim != self.feature_dim:
                raise ValueError(
                    f"precomputed bag features have dim {C_dim}, expected {self.feature_dim}")
            return x.mean(dim=1)  # bag GAP — placeholder until Stage 7's R_b

        features, _ = self.encoder(x)
        if features.dim() == 5:
            out = features.mean(dim=[0, 3, 4])  # mean over T, H, W
        else:
            out = features.mean(dim=[-2, -1])

        if K is not None and K > 1:
            B_total, C = out.shape
            if B_total % K != 0:
                raise ValueError(f"batch rows {B_total} not divisible by bag count K={K}")
            out = out.view(B_total // K, K, C).mean(dim=1)
        return out

    def forward(self, x, K=None, mask=None):
        if mask is not None:
            raise NotImplementedError(
                "Bag-mask MIL pooling was retired with the legacy 1B adapter; "
                "Stage 7 computes R_b from precomputed v2 window features directly.")
        out = self.extract_features(x, K=K)
        return self.classifier(out)
