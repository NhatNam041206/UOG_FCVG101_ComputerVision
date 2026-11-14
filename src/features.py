"""[1B] Task 1B - interest points and matching (rubric 1.3).

Flow:  task1b.run_task1b(cfg) -> [1B.5] match_features (uses detect_keypoints, ratio_test, filter_ransac)

Baseline: ORB/SIFT + BFMatcher + 0.75 ratio test. Advanced: Harris from scratch, RANSAC homography inliers.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

DETECTORS = ("orb", "sift", "harris")
HARRIS_KEYPOINT_SIZE = 31.0       # patch size ORB uses to describe a Harris corner


@dataclass
class MatchResult:
    """Output of [1B.5]."""
    keypoints_1: list             # cv2.KeyPoint list, image 1
    keypoints_2: list             # cv2.KeyPoint list, image 2
    good_matches: list            # cv2.DMatch list after the ratio test
    inlier_mask: np.ndarray = None    # bool (len(good_matches),) after RANSAC, else None
    homography: np.ndarray = None     # (3, 3) from RANSAC, else None
    stats: dict = field(default_factory=dict)   # counts and inlier ratio


def _to_gray(image):
    """BGR or grayscale uint8 -> grayscale uint8."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise ValueError("[1B.5] expected a uint8 image, (H, W) or (H, W, 3)")
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def harris_response_scratch(gray, k=0.04, window_sigma=1.0):
    """[1B.5s] Harris response written from the formula (advanced variant).

    Flow:     called by detect_harris(method="scratch")
    Method:   Ix, Iy = Sobel 3x3 -> structure tensor entries Ixx, Iyy, Ixy smoothed by a Gaussian
              window -> R = det(M) - k * trace(M)^2.

    Inputs:
        gray:         np.ndarray uint8 (H, W).
        k:            Harris constant (0.04-0.06).
        window_sigma: Gaussian window sigma in pixels.

    Returns:
        np.ndarray float32 (H, W), Harris response R.
    """
    image = gray.astype(np.float32) / 255.0
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    ixx = cv2.GaussianBlur(grad_x * grad_x, (0, 0), window_sigma)
    iyy = cv2.GaussianBlur(grad_y * grad_y, (0, 0), window_sigma)
    ixy = cv2.GaussianBlur(grad_x * grad_y, (0, 0), window_sigma)
    return (ixx * iyy - ixy * ixy - k * (ixx + iyy) ** 2).astype(np.float32)


def detect_harris(gray, method="opencv", k=0.04, threshold=0.01, max_corners=500):
    """[1B.5h] Harris corner response and corner locations.

    Purpose:  Show where both gradient directions are strong, i.e. corners (rubric 1.3).
    Flow:     called by run_task1b() (Harris figure) and detect_keypoints(detector="harris")
    Method:   "opencv" = cv2.cornerHarris(blockSize=3, ksize=3); "scratch" = harris_response_scratch().
              Then non-maximum suppression (3x3 local maxima), keep R > threshold * max(R),
              strongest first.

    Inputs:
        gray:        np.ndarray uint8 (H, W).
        method:      "opencv" or "scratch".
        k:           Harris constant.
        threshold:   fraction of max response kept.
        max_corners: keep at most this many, strongest first.

    Returns:
        (response: float32 (H, W), corners: np.ndarray int (M, 2) as (x, y))

    Raises:
        ValueError: unknown method.
    """
    gray = _to_gray(gray)
    if method == "opencv":
        response = cv2.cornerHarris(gray.astype(np.float32) / 255.0, blockSize=3, ksize=3, k=k)
    elif method == "scratch":
        response = harris_response_scratch(gray, k=k)
    else:
        raise ValueError(f"[1B.5h] unknown method '{method}', use 'opencv' or 'scratch'")
    peak = float(response.max())
    is_local_max = response == cv2.dilate(response, None)
    rows, cols = np.where(is_local_max & (response > threshold * max(peak, 0.0)))
    order = np.argsort(response[rows, cols])[::-1][:max_corners]
    corners = np.stack([cols[order], rows[order]], axis=1).astype(int) if len(order) else np.zeros((0, 2), int)
    return response.astype(np.float32), corners


def detect_keypoints(gray, detector="orb", max_features=1000):
    """[1B.5a] Detect keypoints and compute descriptors.

    Flow:     called by match_features()
              -> cv2.ORB_create / cv2.SIFT_create, or [1B.5h] detect_harris + ORB descriptors

    Inputs:
        gray:         np.ndarray uint8 (H, W) (BGR is converted).
        detector:     "orb", "sift" or "harris" (Harris corners described with ORB).
        max_features: cap on keypoints.

    Returns:
        (keypoints: list of cv2.KeyPoint, descriptors: np.ndarray or None)

    Raises:
        ValueError: unknown detector.
    """
    gray = _to_gray(gray)
    if detector == "orb":
        keypoints, descriptors = cv2.ORB_create(nfeatures=max_features).detectAndCompute(gray, None)
    elif detector == "sift":
        keypoints, descriptors = cv2.SIFT_create(nfeatures=max_features).detectAndCompute(gray, None)
    elif detector == "harris":
        _, corners = detect_harris(gray, threshold=0.01, max_corners=max_features)
        candidates = [cv2.KeyPoint(float(x), float(y), HARRIS_KEYPOINT_SIZE) for x, y in corners]
        keypoints, descriptors = cv2.ORB_create().compute(gray, candidates)
    else:
        raise ValueError(f"[1B.5a] unknown detector '{detector}', use one of {DETECTORS}")
    return list(keypoints), descriptors


def ratio_test(desc_1, desc_2, detector="orb", ratio=0.75):
    """[1B.5b] 2-nearest-neighbour matching with Lowe's ratio test.

    Purpose:  Drop ambiguous matches whose best and second-best candidates are nearly as close.
    Flow:     called by match_features()  ->  cv2.BFMatcher (NORM_HAMMING for ORB/Harris, NORM_L2 for SIFT)

    Inputs:
        desc_1, desc_2: descriptor arrays.
        detector:       decides the distance norm.
        ratio:          keep a match if best < ratio * second best.

    Returns:
        list of cv2.DMatch, sorted by distance.
    """
    if desc_1 is None or desc_2 is None or len(desc_1) == 0 or len(desc_2) < 2:
        return []
    norm = cv2.NORM_L2 if detector == "sift" else cv2.NORM_HAMMING
    pairs = cv2.BFMatcher(norm).knnMatch(desc_1, desc_2, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]
    return sorted(good, key=lambda match: match.distance)


def filter_ransac(keypoints_1, keypoints_2, matches, reproj_threshold=5.0):
    """[1B.5c] Keep geometrically consistent matches with a RANSAC homography (advanced).

    Flow:     called by match_features(use_ransac=True)  ->  cv2.findHomography(RANSAC)
    Method:   only valid when the two views are related by a homography (planar scene or pure
              camera rotation); otherwise read the inlier count as a rough consistency check.

    Inputs:
        keypoints_1, keypoints_2: keypoint lists.
        matches:                  list of cv2.DMatch from [1B.5b].
        reproj_threshold:         RANSAC inlier distance, pixels.

    Returns:
        (homography: (3, 3) or None, inlier_mask: bool array (len(matches),))
    """
    if len(matches) < 4:
        return None, np.zeros(len(matches), bool)
    points_1 = np.float32([keypoints_1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    points_2 = np.float32([keypoints_2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(points_1, points_2, cv2.RANSAC, reproj_threshold)
    if mask is None:
        return None, np.zeros(len(matches), bool)
    return homography, mask.ravel().astype(bool)


def match_features(image_1, image_2, detector="orb", ratio=0.75, use_ransac=False,
                   max_features=1000, reproj_threshold=5.0):
    """[1B.5] Detect, describe and match interest points between two views.

    Purpose:  Find corresponding points across views (rubric 1.3).
    Flow:     called by run_task1b(), assistant.features_mode()
              -> [1B.5a] detect_keypoints (x2) -> [1B.5b] ratio_test -> [1B.5c] filter_ransac (optional)
    Method:   baseline = ORB + ratio test; advanced = SIFT, Harris, RANSAC inliers.

    Inputs:
        image_1, image_2: np.ndarray uint8, BGR or grayscale.
        detector:         "orb", "sift" or "harris".
        ratio:            ratio-test threshold.
        use_ransac:       also estimate a homography and count inliers.
        max_features:     keypoint cap per image.
        reproj_threshold: RANSAC inlier distance, pixels.

    Returns:
        MatchResult. stats = {detector, keypoints_1, keypoints_2, good_matches,
                              and with RANSAC: inliers, inlier_ratio}.

    Raises:
        ValueError: an image has too few keypoints to match.
    """
    keypoints_1, desc_1 = detect_keypoints(image_1, detector, max_features)
    keypoints_2, desc_2 = detect_keypoints(image_2, detector, max_features)
    for label, keypoints in (("first", keypoints_1), ("second", keypoints_2)):
        if len(keypoints) < 2:
            raise ValueError(f"[1B.5] the {label} image has only {len(keypoints)} {detector} keypoints; "
                             f"use an image with more texture")
    good = ratio_test(desc_1, desc_2, detector, ratio)
    result = MatchResult(keypoints_1, keypoints_2, good,
                         stats={"detector": detector, "ratio": ratio,
                                "keypoints_1": len(keypoints_1), "keypoints_2": len(keypoints_2),
                                "good_matches": len(good)})
    if use_ransac:
        result.homography, result.inlier_mask = filter_ransac(keypoints_1, keypoints_2, good, reproj_threshold)
        inliers = int(result.inlier_mask.sum())
        result.stats.update(inliers=inliers, inlier_ratio=inliers / len(good) if good else 0.0)
    return result
