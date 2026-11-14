"""Task 1B checks: each core function returns the right shape/type, and the advanced variants agree
with OpenCV. Uses small generated images only, so no photos are needed."""
import cv2
import numpy as np
import pytest

from src.edges import compute_edges
from src.features import detect_harris, match_features
from src.filtering import add_noise, apply_spatial_filters, compare_to_reference, convolve2d
from src.frequency import filter_frequency, make_frequency_mask


@pytest.fixture
def scene():
    """Textured 160x200 BGR test image with clear corners and edges."""
    rng = np.random.default_rng(0)
    image = np.full((160, 200, 3), 200, np.uint8)
    for _ in range(40):
        x, y = rng.integers(10, 190), rng.integers(10, 150)
        colour = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.rectangle(image, (int(x), int(y)), (int(x) + 12, int(y) + 9), colour, -1)
    return image


def test_noise_shapes_and_seed(scene):
    gauss = add_noise(scene, "gaussian", 20, seed=1)
    assert gauss.shape == scene.shape and gauss.dtype == np.uint8
    assert np.array_equal(gauss, add_noise(scene, "gaussian", 20, seed=1))
    salt = add_noise(scene[:, :, 0], "salt_pepper", 0.1, seed=1)
    assert 0.07 < np.mean((salt == 0) | (salt == 255)) < 0.13
    with pytest.raises(ValueError):
        add_noise(scene, "speckle", 1)


def test_scratch_convolution_matches_opencv(scene):
    gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    kernel = np.random.default_rng(1).random((5, 5)).astype(np.float32)
    ours = convolve2d(gray, kernel)
    theirs = cv2.filter2D(gray.astype(np.float32), -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    assert np.allclose(ours, theirs, atol=1e-3)
    opencv = apply_spatial_filters(gray, 5, 1.2, "opencv")
    scratch = apply_spatial_filters(gray, 5, 1.2, "scratch")
    for name in ("box", "gaussian"):
        assert np.abs(opencv[name].astype(int) - scratch[name].astype(int)).max() <= 1


def test_filters_improve_salt_pepper(scene):
    gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    noisy = add_noise(gray, "salt_pepper", 0.05)
    scores = compare_to_reference({"noisy": noisy, **apply_spatial_filters(noisy, 5, 1.0)}, gray)
    assert scores["median"]["psnr_db"] > scores["noisy"]["psnr_db"]
    with pytest.raises(ValueError):
        apply_spatial_filters(gray, ksize=4)


def test_frequency_masks_and_identity(scene):
    gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    for profile in ("ideal", "gaussian", "butterworth"):
        low = make_frequency_mask(gray.shape, "lowpass", 20, profile)
        high = make_frequency_mask(gray.shape, "highpass", 20, profile)
        assert np.allclose(low + high, 1.0)
        assert low[80, 100] == pytest.approx(1.0)        # centre (DC) kept by the low-pass
    huge = filter_frequency(gray, "lowpass", radius=10_000)
    assert np.allclose(huge.filtered, gray, atol=1e-3)    # all-pass mask gives the image back
    high = filter_frequency(gray, "highpass", 20)
    assert abs(float(high.filtered.mean())) < 1.0         # DC removed -> signed, mean ~ 0
    assert high.filtered_uint8.dtype == np.uint8


def test_edges(scene):
    gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    result = compute_edges(gray, 80, 160)
    assert result.grad_x.shape == gray.shape and result.canny.dtype == np.uint8
    assert set(np.unique(result.canny)) <= {0, 255} and 0 < result.edge_density < 1
    auto = compute_edges(gray, auto_threshold=True)
    assert auto.thresholds[0] < auto.thresholds[1]
    with pytest.raises(ValueError):
        compute_edges(gray, 160, 80)


def test_harris_opencv_and_scratch(scene):
    gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    for method in ("opencv", "scratch"):
        response, corners = detect_harris(gray, method, threshold=0.05, max_corners=50)
        assert response.shape == gray.shape and corners.shape[1] == 2 and 0 < len(corners) <= 50


@pytest.mark.parametrize("detector", ["orb", "sift", "harris"])
def test_matching_recovers_known_shift(scene, detector):
    shift = np.float32([[1, 0, 15], [0, 1, 10]])
    moved = cv2.warpAffine(scene, shift, (scene.shape[1], scene.shape[0]), borderValue=(200, 200, 200))
    result = match_features(scene, moved, detector, 0.75, use_ransac=True)
    assert result.stats["good_matches"] >= 4 and result.stats["inliers"] >= 4
    inliers = [m for m, keep in zip(result.good_matches, result.inlier_mask) if keep]
    shifts = np.array([np.subtract(result.keypoints_2[m.trainIdx].pt, result.keypoints_1[m.queryIdx].pt)
                       for m in inliers])
    assert np.median(shifts, axis=0) == pytest.approx([15, 10], abs=1.5)


def test_matching_rejects_blank_image(scene):
    with pytest.raises(ValueError):
        match_features(scene, np.zeros_like(scene), "orb")
