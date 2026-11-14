"""Skeleton checks: config loads, every module imports, the assistant routes modes."""
import importlib

import pytest

from src.config import load_config

MODULES = ["config", "io_utils", "calibration", "filtering", "frequency", "edges", "features",
           "task1b", "data_pipeline", "model", "train", "evaluate", "inference", "assistant", "visualize"]


def test_config_loads():
    cfg = load_config()
    assert cfg.seed == 42
    assert cfg.task1a.pattern_cols == 6 and cfg.task1a.pattern_rows == 9
    assert cfg.paths.calibration_images.is_absolute()


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(f"src.{name}")


def test_unknown_mode_is_rejected():
    from src.assistant import run_mode
    with pytest.raises(ValueError):
        run_mode("dance", None, load_config())
