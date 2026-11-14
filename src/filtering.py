"""[1B] Task 1B - noise and spatial filtering (rubric 1.2).

Flow:  task1b.run_task1b(cfg) -> [1B.1] add_noise -> [1B.2] apply_spatial_filters -> [1B.2m] compare_to_reference

Baseline: cv2.blur / cv2.GaussianBlur / cv2.medianBlur. Advanced: method="scratch" box and Gaussian
through our own convolve2d(), checked against OpenCV.
"""
import cv2
import numpy as np


def _check_uint8_image(image, step):
    """Shared input check: uint8, (H, W) or (H, W, 3)."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError(f"[{step}] expected a uint8 NumPy image, got {type(image).__name__} "
                         f"{getattr(image, 'dtype', '')}")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
        raise ValueError(f"[{step}] expected shape (H, W) or (H, W, 3), got {image.shape}")


def add_noise(bgr, kind="gaussian", level=20.0, seed=42):
    """[1B.1] Add controlled noise to a clean image.

    Purpose:  A test image whose clean version is known, so filters can be scored.
    Flow:     called by run_task1b()   next step: [1B.2] apply_spatial_filters
    Method:   "gaussian" = additive N(0, level^2), clipped to 0..255;
              "salt_pepper" = a fraction `level` of pixels set to 0 or 255 (half each).

    Inputs:
        bgr:   np.ndarray uint8 (H, W, 3) or (H, W).
        kind:  "gaussian" (std = level, 0-255 scale) or "salt_pepper" (fraction = level).
        level: noise strength.
        seed:  random seed.

    Returns:
        np.ndarray uint8, same shape as input.

    Raises:
        ValueError: unknown kind, or level out of range.
    """
    _check_uint8_image(bgr, "1B.1")
    rng = np.random.default_rng(seed)
    if kind == "gaussian":
        if level < 0:
            raise ValueError("[1B.1] Gaussian noise std must be >= 0")
        noise = rng.normal(0.0, level, bgr.shape)
        return np.clip(bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if kind == "salt_pepper":
        if not 0 <= level <= 1:
            raise ValueError("[1B.1] salt-and-pepper level is a fraction in 0..1")
        noisy = bgr.copy()
        draw = rng.random(bgr.shape[:2])          # same pixel hit in every channel
        noisy[draw < level / 2] = 0
        noisy[(draw >= level / 2) & (draw < level)] = 255
        return noisy
    raise ValueError(f"[1B.1] unknown noise kind '{kind}', use 'gaussian' or 'salt_pepper'")


def make_gaussian_kernel(ksize, sigma):
    """[1B.2k] Normalised 2D Gaussian kernel, the same weights cv2.GaussianBlur uses.

    Flow:     called by apply_spatial_filters(method="scratch")

    Inputs:
        ksize: odd kernel size.
        sigma: standard deviation in pixels.

    Returns:
        np.ndarray float32 (ksize, ksize), sums to 1.
    """
    half = ksize // 2
    axis = np.arange(-half, half + 1, dtype=np.float64)
    weights_1d = np.exp(-(axis ** 2) / (2.0 * sigma ** 2))
    weights_1d /= weights_1d.sum()
    return np.outer(weights_1d, weights_1d).astype(np.float32)


def convolve2d(gray, kernel):
    """[1B.2s] Correlate an image with a kernel, written from scratch (advanced variant).

    Flow:     called by apply_spatial_filters(method="scratch"); checked against cv2.filter2D
    Method:   reflect-pad by k//2, then sum kernel[i, j] * shifted image over all (i, j).
              Looping over the k*k kernel taps (not the pixels) keeps it fast in NumPy.

    Inputs:
        gray:   np.ndarray (H, W), uint8 or float32.
        kernel: np.ndarray float32 (k, k), k odd.

    Returns:
        np.ndarray float32 (H, W), reflect-padded borders (same as cv2.BORDER_REFLECT_101).

    Raises:
        ValueError: image is not 2D or the kernel is not square with odd size.
    """
    if gray.ndim != 2:
        raise ValueError(f"[1B.2s] convolve2d expects a 2D image, got {gray.shape}")
    k_rows, k_cols = kernel.shape
    if k_rows != k_cols or k_rows % 2 == 0:
        raise ValueError(f"[1B.2s] kernel must be square with odd size, got {kernel.shape}")
    half = k_rows // 2
    padded = np.pad(gray.astype(np.float32), half, mode="reflect")
    height, width = gray.shape
    out = np.zeros((height, width), np.float32)
    for row in range(k_rows):
        for col in range(k_cols):
            out += kernel[row, col] * padded[row:row + height, col:col + width]
    return out


def _filter_scratch(image, kernel):
    """Apply convolve2d per channel and round back to uint8."""
    if image.ndim == 2:
        result = convolve2d(image, kernel)
    else:
        result = np.dstack([convolve2d(image[:, :, ch], kernel) for ch in range(3)])
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def apply_spatial_filters(image, ksize=5, sigma=1.0, method="opencv"):
    """[1B.2] Denoise with box, Gaussian and median filters of the same kernel size.

    Purpose:  Compare the three standard linear/non-linear filters (rubric 1.2).
    Flow:     called by run_task1b(), assistant.filter_mode()
              previous step: [1B.1]   next step: [1B.2m] compare_to_reference
    Method:   "opencv" = baseline cv2.blur / cv2.GaussianBlur / cv2.medianBlur;
              "scratch" = box and Gaussian via convolve2d() (median always uses OpenCV,
              it is not a convolution).

    Inputs:
        image:  np.ndarray uint8 (H, W, 3) or (H, W).
        ksize:  odd kernel size >= 3.
        sigma:  Gaussian sigma in pixels.
        method: "opencv" or "scratch".

    Returns:
        dict {"box": img, "gaussian": img, "median": img}, each uint8 same shape as input.

    Raises:
        ValueError: ksize even or < 3, sigma <= 0, unknown method.
    """
    _check_uint8_image(image, "1B.2")
    if ksize < 3 or ksize % 2 == 0:
        raise ValueError(f"[1B.2] kernel size must be odd and >= 3, got {ksize}")
    if sigma <= 0:
        raise ValueError(f"[1B.2] Gaussian sigma must be > 0, got {sigma}")
    median = cv2.medianBlur(image, ksize)
    if method == "opencv":
        return {"box": cv2.blur(image, (ksize, ksize)),
                "gaussian": cv2.GaussianBlur(image, (ksize, ksize), sigma),
                "median": median}
    if method == "scratch":
        box_kernel = np.full((ksize, ksize), 1.0 / (ksize * ksize), np.float32)
        return {"box": _filter_scratch(image, box_kernel),
                "gaussian": _filter_scratch(image, make_gaussian_kernel(ksize, sigma)),
                "median": median}
    raise ValueError(f"[1B.2] unknown method '{method}', use 'opencv' or 'scratch'")


def mse(reference, test):
    """[1B.2m] Mean squared error between two images of the same shape (0-255 scale)."""
    if reference.shape != test.shape:
        raise ValueError(f"[1B.2m] shapes differ: {reference.shape} vs {test.shape}")
    diff = reference.astype(np.float64) - test.astype(np.float64)
    return float(np.mean(diff * diff))


def psnr(reference, test, max_value=255.0):
    """[1B.2m] Peak signal-to-noise ratio in dB; inf when the images are identical."""
    error = mse(reference, test)
    return float("inf") if error == 0 else float(10.0 * np.log10(max_value ** 2 / error))


def compare_to_reference(filtered, clean):
    """[1B.2m] Score filtered images against the clean original.

    Purpose:  Fair, numeric filter comparison: lower MSE / higher PSNR = closer to the clean image.
    Flow:     called by run_task1b()   previous step: [1B.2]

    Inputs:
        filtered: dict name -> uint8 image (from [1B.2]; add "noisy" to get the starting point).
        clean:    uint8 image, same shape.

    Returns:
        dict name -> {"mse": float, "psnr_db": float}
    """
    return {name: {"mse": mse(clean, image), "psnr_db": psnr(clean, image)}
            for name, image in filtered.items()}
