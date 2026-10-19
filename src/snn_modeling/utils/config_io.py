"""Single authoritative config loader for every entry point.

Fixes audit B2: scripts that skip Hydra composition (raw OmegaConf.load) or the
legacy architecture->model alias crash at startup. Every script must load config
through load_config(); none may re-implement the compose + alias dance.
"""

import os
import random

from omegaconf import open_dict
from hydra import initialize_config_dir, compose


def seed_everything(seed=42):
    import numpy as np
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path, overrides=(), seed=42):
    """Compose a Hydra config (resolving its defaults tree), apply the
    architecture->model alias, and seed RNGs. Raises loudly on missing files.
    """
    config_path = os.path.abspath(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config_dir = os.path.dirname(config_path)
    config_name = os.path.splitext(os.path.basename(config_path))[0]

    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        config = compose(config_name=config_name, overrides=list(overrides))

    with open_dict(config):
        if hasattr(config, 'architecture'):
            config.model = config.architecture

    if seed is not None:
        seed_everything(seed)
    return config
