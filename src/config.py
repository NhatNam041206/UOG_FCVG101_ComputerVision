"""[CFG] Load the project configuration and resolve paths (rule R6).

Flow: called first by every entry point (notebook, app.py, tests) -> passed down to run_task*().
"""
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def _to_namespace(value):
    """Turn nested dicts into attribute access (cfg.task1a.square_size_mm)."""
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _to_namespace(item) for key, item in value.items()})
    return value


def load_config(path=DEFAULT_CONFIG):
    """[CFG] Read config.yaml and turn every entry under `paths` into an absolute Path.

    Purpose:  One source for all settings, so no value is hard-coded in src/.
    Flow:     called by notebook / app.py / tests  ->  result passed to run_task1a/1b/2()

    Inputs:
        path: path to a YAML file; defaults to configs/config.yaml.

    Returns:
        SimpleNamespace, e.g. cfg.seed, cfg.paths.calibration_images (Path), cfg.task1a.fix_k3.

    Raises:
        FileNotFoundError: the config file does not exist.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["paths"] = {name: PROJECT_ROOT / rel for name, rel in raw["paths"].items()}
    cfg = _to_namespace(raw)
    cfg.project_root = PROJECT_ROOT
    return cfg


def set_seed(seed):
    """[CFG] Seed Python, NumPy and (if installed) PyTorch for reproducible runs (rule R7).

    Flow:     called at the start of every run_task*()

    Inputs:
        seed: integer seed, normally cfg.seed.

    Returns:
        None.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
