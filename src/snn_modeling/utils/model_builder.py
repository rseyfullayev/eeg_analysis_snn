"""Model construction: Hydra instantiation only.

The legacy manual-build branch was broken by construction (called
SpikingUNet(config=...) against an incompatible signature, referenced an
undefined `encoder`; audit B22-family) and was deleted in Stage 0 tranche 2.
"""

from hydra.utils import instantiate


def build_model(config):
    if hasattr(config.model, '_target_'):
        return instantiate(config.model, _recursive_=True)
    raise ValueError(
        "config.model must declare a Hydra `_target_`; the legacy manual builder "
        "branch was removed in Stage 0 tranche 2 (REVIEW_SMC_AUDIT.md §8).")
