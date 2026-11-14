"""[3] Task 3 - the assistant's routing: one user request -> one module -> one result (rubric 3.1, 3.2).

Flow:  app.py / notebook -> [3.1] run_mode(mode, ...) -> MODES[mode] -> Task 1 / Task 2 function -> ModeResult
"""
from dataclasses import dataclass, field


@dataclass
class ModeResult:
    """What every mode returns to the UI."""
    images: dict = field(default_factory=dict)    # caption -> image (BGR or grayscale uint8)
    numbers: dict = field(default_factory=dict)   # name -> value, shown as a table
    message: str = ""                             # short explanation or warning for the user


def undistort_mode(image, cfg, image_2=None, params=None):
    """[3.2a] Undistort with artifacts/calibration.npz.  ->  calibration.load_calibration, undistort_image [1A.5]"""
    raise NotImplementedError("[3.2a] undistort_mode")


def filter_mode(image, cfg, image_2=None, params=None):
    """[3.2b] Box/Gaussian/median.  ->  filtering.apply_spatial_filters [1B.2]"""
    raise NotImplementedError("[3.2b] filter_mode")


def frequency_mode(image, cfg, image_2=None, params=None):
    """[3.2c] Spectrum and low/high-pass.  ->  frequency.filter_frequency [1B.3]"""
    raise NotImplementedError("[3.2c] frequency_mode")


def edges_mode(image, cfg, image_2=None, params=None):
    """[3.2d] Sobel and Canny.  ->  edges.compute_edges [1B.4]"""
    raise NotImplementedError("[3.2d] edges_mode")


def features_mode(image, cfg, image_2=None, params=None):
    """[3.2e] Keypoint matching between two images.  ->  features.match_features [1B.5]"""
    raise NotImplementedError("[3.2e] features_mode")


def classify_mode(image, cfg, image_2=None, params=None):
    """[3.2f] Class label and score.  ->  inference.load_model [3.0], predict [3.3]"""
    raise NotImplementedError("[3.2f] classify_mode")


# [3.2] The one place that lists what the assistant can do.
MODES = {
    "undistort": undistort_mode,   # -> [1A.5]
    "filter": filter_mode,         # -> [1B.2]
    "frequency": frequency_mode,   # -> [1B.3]
    "edges": edges_mode,           # -> [1B.4]
    "features": features_mode,     # -> [1B.5], needs two images
    "classify": classify_mode,     # -> [3.3]
}


def run_mode(mode, image, cfg, image_2=None, params=None):
    """[3.1] Route a user request to the right module and return its result.

    Purpose:  Application logic: identify the requested task and run it (rubric 3.1).
    Flow:     called by app.py and the notebook  ->  MODES[mode](...)

    Inputs:
        mode:    a key of MODES.
        image:   np.ndarray uint8 BGR (H, W, 3).
        cfg:     config from load_config().
        image_2: second image, only for "features".
        params:  optional dict of UI overrides (kernel size, thresholds, ...).

    Returns:
        ModeResult.

    Raises:
        ValueError: unknown mode, or a required second image is missing.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown mode '{mode}'. Choose one of: {', '.join(MODES)}")
    if mode == "features" and image_2 is None:
        raise ValueError("Feature matching needs a second image.")
    return MODES[mode](image, cfg, image_2=image_2, params=params)
