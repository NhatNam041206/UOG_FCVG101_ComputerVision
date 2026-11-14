"""[1B] Task 1B - frequency-domain analysis and filtering (rubric 1.2).

Flow:  task1b.run_task1b(cfg) -> [1B.3] filter_frequency (uses make_frequency_mask)

Baseline: ideal circular low/high-pass mask. Advanced: Gaussian and Butterworth masks, which avoid the
ringing that the ideal mask's sharp edge causes.
"""
from dataclasses import dataclass

import numpy as np

KINDS = ("lowpass", "highpass")
PROFILES = ("ideal", "gaussian", "butterworth")


@dataclass
class FrequencyResult:
    """Output of [1B.3]."""
    log_spectrum: np.ndarray      # float32 (H, W), log(1 + |F|), centred (DC in the middle)
    mask: np.ndarray              # float32 (H, W), values 0..1
    filtered: np.ndarray          # float32 (H, W), real part of the inverse FFT. A high-pass result is
                                  # signed (centred on 0): show it with a symmetric colour range.
    filtered_uint8: np.ndarray    # uint8 (H, W), display copy: low-pass clipped to 0..255,
                                  # high-pass shifted to 128 + value


def make_frequency_mask(shape, kind="lowpass", radius=30, profile="ideal", order=2):
    """[1B.3a] Build a centred frequency mask.

    Flow:     called by filter_frequency()
    Method:   D = distance of each frequency from the centre (after fftshift).
              ideal:       1 if D <= radius else 0
              gaussian:    exp(-D^2 / (2 radius^2))          (radius acts as sigma)
              butterworth: 1 / (1 + (D / radius)^(2 order))
              high-pass = 1 - low-pass for every profile.

    Inputs:
        shape:   (H, W).
        kind:    "lowpass" or "highpass".
        radius:  cut-off distance from the centre, pixels (> 0).
        profile: "ideal" (baseline), "gaussian" or "butterworth" (advanced).
        order:   Butterworth order.

    Returns:
        np.ndarray float32 (H, W), 0..1.

    Raises:
        ValueError: unknown kind/profile or radius <= 0.
    """
    if kind not in KINDS:
        raise ValueError(f"[1B.3a] kind must be one of {KINDS}, got '{kind}'")
    if profile not in PROFILES:
        raise ValueError(f"[1B.3a] profile must be one of {PROFILES}, got '{profile}'")
    if radius <= 0:
        raise ValueError(f"[1B.3a] radius must be > 0, got {radius}")
    height, width = shape[:2]
    rows, cols = np.ogrid[:height, :width]
    dist_sq = (rows - height // 2) ** 2 + (cols - width // 2) ** 2
    if profile == "ideal":
        lowpass = (dist_sq <= radius ** 2).astype(np.float64)
    elif profile == "gaussian":
        lowpass = np.exp(-dist_sq / (2.0 * radius ** 2))
    else:
        lowpass = 1.0 / (1.0 + (np.sqrt(dist_sq) / radius) ** (2 * order))
    mask = lowpass if kind == "lowpass" else 1.0 - lowpass
    return mask.astype(np.float32)


def filter_frequency(gray, kind="lowpass", radius=30, profile="ideal", order=2):
    """[1B.3] FFT -> fftshift -> mask -> inverse FFT.

    Purpose:  Show the image spectrum and blur/sharpen in the frequency domain (rubric 1.2).
    Flow:     called by run_task1b(), assistant.frequency_mode()  ->  calls make_frequency_mask()
              previous step: [1B.2]   next step: [1B.4] compute_edges

    Inputs:
        gray:    np.ndarray uint8 (H, W).
        kind, radius, profile, order: see make_frequency_mask().

    Returns:
        FrequencyResult.

    Raises:
        ValueError: image is not 2D.
    """
    if not isinstance(gray, np.ndarray) or gray.ndim != 2:
        raise ValueError(f"[1B.3] filter_frequency expects a grayscale (H, W) image, "
                         f"got {getattr(gray, 'shape', type(gray).__name__)}")
    spectrum = np.fft.fftshift(np.fft.fft2(gray.astype(np.float64)))
    mask = make_frequency_mask(gray.shape, kind, radius, profile, order)
    filtered = np.fft.ifft2(np.fft.ifftshift(spectrum * mask)).real
    if kind == "lowpass":
        display = np.clip(filtered, 0, 255)
    else:
        display = np.clip(filtered + 128.0, 0, 255)
    return FrequencyResult(log_spectrum=np.log1p(np.abs(spectrum)).astype(np.float32),
                           mask=mask,
                           filtered=filtered.astype(np.float32),
                           filtered_uint8=np.rint(display).astype(np.uint8))
