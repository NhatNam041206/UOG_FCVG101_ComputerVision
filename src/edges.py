"""[1B] Task 1B - image gradients and edges (rubric 1.2).

Flow:  task1b.run_task1b(cfg) -> [1B.4] compute_edges

Baseline: cv2.Sobel + cv2.Canny with fixed thresholds. Advanced: auto thresholds from the median.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class EdgeResult:
    """Output of [1B.4]."""
    smoothed: np.ndarray          # uint8 (H, W), Gaussian pre-smoothed input
    grad_x: np.ndarray            # float32 (H, W), signed; strong on vertical edges
    grad_y: np.ndarray            # float32 (H, W), signed; strong on horizontal edges
    magnitude: np.ndarray         # float32 (H, W), sqrt(gx^2 + gy^2)
    orientation: np.ndarray       # float32 (H, W), atan2(gy, gx) in radians
    canny: np.ndarray             # uint8 (H, W), 0 or 255
    thresholds: tuple             # (low, high) actually used
    edge_density: float           # fraction of pixels marked as Canny edges


def auto_canny_thresholds(gray, spread=0.33):
    """[1B.4a] Canny thresholds from the image median (advanced variant).

    Flow:     called by compute_edges(auto_threshold=True)
    Method:   low = (1 - spread) * median, high = (1 + spread) * median, clipped to 0..255.

    Inputs:
        gray:   np.ndarray uint8 (H, W), normally the smoothed image.
        spread: fraction around the median.

    Returns:
        (low: int, high: int)
    """
    median = float(np.median(gray))
    low = int(max(0, (1.0 - spread) * median))
    high = int(min(255, (1.0 + spread) * median))
    return low, max(high, low + 1)


def compute_edges(gray, low=80, high=160, blur_ksize=5, blur_sigma=1.0, auto_threshold=False):
    """[1B.4] Sobel gradients and Canny edges.

    Purpose:  Find intensity edges (rubric 1.2).
    Flow:     called by run_task1b(), assistant.edges_mode()
              -> calls cv2.GaussianBlur, cv2.Sobel (3x3), cv2.Canny
              previous step: [1B.3]   next step: [1B.5] match_features
    Method:   baseline = fixed (low, high); auto_threshold=True = median-based thresholds.

    Inputs:
        gray:           np.ndarray uint8 (H, W).
        low, high:      Canny hysteresis thresholds (ignored if auto_threshold).
        blur_ksize:     odd pre-smoothing kernel size.
        blur_sigma:     pre-smoothing sigma.
        auto_threshold: advanced variant, thresholds from the image median.

    Returns:
        EdgeResult.

    Raises:
        ValueError: image is not uint8 (H, W) or low >= high.
    """
    if not isinstance(gray, np.ndarray) or gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("[1B.4] compute_edges expects a uint8 grayscale (H, W) image")
    smoothed = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), blur_sigma)
    if auto_threshold:
        low, high = auto_canny_thresholds(smoothed)
    elif low >= high:
        raise ValueError(f"[1B.4] Canny needs low < high, got ({low}, {high})")
    grad_x = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    canny = cv2.Canny(smoothed, low, high)
    return EdgeResult(smoothed=smoothed,
                      grad_x=grad_x,
                      grad_y=grad_y,
                      magnitude=np.hypot(grad_x, grad_y).astype(np.float32),
                      orientation=np.arctan2(grad_y, grad_x).astype(np.float32),
                      canny=canny,
                      thresholds=(int(low), int(high)),
                      edge_density=float((canny > 0).mean()))
