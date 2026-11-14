"""[1A] Task 1A - camera calibration and undistortion (rubric 1.1).

Flow:  run_task1a(cfg)
         -> [1A.1] load_calibration_images
         -> [1A.2a] make_board_points
         -> [1A.2] detect_corners            (per image)
         -> [1A.3] calibrate_camera          (baseline, then fix_k3 variant)
         -> [1A.4] compute_reprojection_errors
         -> [1A.5] undistort_image
         -> [1A.6] save_calibration          -> artifacts/calibration.npz (read by Task 3)

Baseline: OpenCV 5-coefficient model. Advanced: fix_k3=True (k3 held at 0 to stop k2/k3 overfitting).
The course originals in Cousework/ stay unchanged; this module mirrors calibration_task1a.py as functions
(same detector order, same error definitions, same npz keys).
"""
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.config import set_seed
from src.io_utils import list_images, read_image, save_csv, save_image, save_json
from src.visualize import plot_view_errors

MIN_VIEWS = 3


@dataclass
class CalibrationResult:
    """Output of [1A.3]."""
    K: np.ndarray                 # (3, 3) intrinsic matrix, pixels
    dist: np.ndarray              # (1, 5) [k1, k2, p1, p2, k3]
    rms_px: float                 # RMS reprojection error from cv2.calibrateCamera
    image_size: tuple             # (width, height) in pixels
    rvecs: list = field(default_factory=list)   # per-view board rotation
    tvecs: list = field(default_factory=list)   # per-view board translation, metres


@dataclass
class Task1AReport:
    """Output of run_task1a(): everything the notebook and report need."""
    calibration: CalibrationResult            # the variant chosen in config (saved to artifacts/)
    accepted: list                # file names used for calibration
    rejected: list                # (file name, reason)
    view_errors: list             # (file name, mean_px, rms_px)
    mean_error_px: float
    variants: dict = field(default_factory=dict)    # "baseline" / "fix_k3" -> summary numbers
    figure_paths: list = field(default_factory=list)
    metrics_path: object = None


def load_calibration_images(folder):
    """[1A.1] Read every checkerboard photo and check they share one resolution.

    Purpose:  Ingest the calibration set from one camera (requirement a).
    Flow:     called by run_task1a()  ->  calls io_utils.list_images(), io_utils.read_image()
              next step: [1A.2] detect_corners

    Inputs:
        folder: Path to data/calibration.

    Returns:
        list of (file_name: str, bgr: np.ndarray uint8 (H, W, 3)).

    Raises:
        FileNotFoundError: folder missing or empty.
        ValueError: images have different resolutions (never resize calibration photos silently).
    """
    paths = list_images(folder)
    if not paths:
        raise FileNotFoundError(f"[1A.1] No images in {folder}. Put 10-20 checkerboard photos from ONE camera "
                                f"there (capture_checker_task1a.py helps you take them).")
    images, first_size = [], None
    for path in paths:
        bgr = read_image(path)
        size = (bgr.shape[1], bgr.shape[0])
        first_size = first_size or size
        if size != first_size:
            raise ValueError(f"[1A.1] Mixed image sizes: {path.name} is {size}, expected {first_size}. "
                             f"Use one camera at one resolution.")
        images.append((path.name, bgr))
    return images


def make_board_points(pattern_size, square_size_m):
    """[1A.2a] Build the 3D coordinates of the inner corners on the board plane (Z = 0).

    Purpose:  The known 3D side of the 3D-2D correspondences.
    Flow:     called by run_task1a()  ->  used with every accepted view in [1A.3]
    Method:   corner (i, j) -> (i * square, j * square, 0), in the same order OpenCV returns corners.

    Inputs:
        pattern_size:  (cols, rows) of INNER corners, e.g. (6, 9).
        square_size_m: side of one square in metres.

    Returns:
        np.ndarray float32 (cols*rows, 3), metres.

    Raises:
        ValueError: fewer than 3 corners per side or square size <= 0.
    """
    cols, rows = pattern_size
    if cols < 3 or rows < 3 or square_size_m <= 0:
        raise ValueError(f"[1A.2a] need >= 3x3 inner corners and a positive square size, "
                         f"got {pattern_size}, {square_size_m} m")
    points = np.zeros((cols * rows, 3), np.float32)
    points[:, :2] = np.mgrid[:cols, :rows].T.reshape(-1, 2) * square_size_m
    return points


def detect_corners(gray, pattern_size):
    """[1A.2] Find the checkerboard's inner corners with sub-pixel accuracy.

    Purpose:  The measured 2D side of the 3D-2D correspondences.
    Flow:     called by run_task1a() for each image  ->  calls cv2.findChessboardCornersSB
              (fallback cv2.findChessboardCorners + cv2.cornerSubPix)
              previous step: [1A.1]   next step: [1A.3] calibrate_camera

    Inputs:
        gray:         np.ndarray uint8 (H, W).
        pattern_size: (cols, rows) of INNER corners.

    Returns:
        (found: bool, corners: np.ndarray float32 (N, 1, 2) or None, detector: str)
    """
    if gray.ndim != 2:
        raise ValueError(f"[1A.2] detect_corners expects a grayscale (H, W) image, got {gray.shape}")
    found, corners = cv2.findChessboardCornersSB(gray, pattern_size)
    if found:
        return True, corners.reshape(-1, 1, 2).astype(np.float32), "SB"
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found:
        return False, None, "-"
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners.reshape(-1, 1, 2).astype(np.float32), "classic + subpixel"


def calibrate_camera(object_points, image_points, image_size, fix_k3=False):
    """[1A.3] Estimate camera intrinsics K and lens distortion from checkerboard views.

    Purpose:  Fit the pinhole + distortion model so pixels can be related to 3D rays (rubric 1.1).
    Flow:     called by run_task1a()  ->  calls cv2.calibrateCamera()
              previous step: [1A.2] detect_corners   next step: [1A.4] compute_reprojection_errors
    Method:   baseline = OpenCV 5-coefficient model; fix_k3=True = advanced variant that stops
              k2/k3 overfitting (same RMS, sensible coefficients).

    Inputs:
        object_points: list of (N, 3) float32 arrays, board corners in metres, Z = 0.
        image_points:  list of (N, 1, 2) float32 arrays, detected corners in pixels.
        image_size:    (width, height) in pixels, identical for all views.
        fix_k3:        if True, k3 is held at 0.

    Returns:
        CalibrationResult with K (3x3), dist (1x5), rms_px (float), rvecs, tvecs.

    Raises:
        ValueError: fewer than 3 views, or the solver returns non-finite values.
    """
    if len(object_points) < MIN_VIEWS or len(object_points) != len(image_points):
        raise ValueError(f"[1A.3] need at least {MIN_VIEWS} matching views, got {len(object_points)} "
                         f"(10-20 varied views are recommended)")
    flags = cv2.CALIB_FIX_K3 if fix_k3 else 0
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, tuple(image_size),
                                                     None, None, flags=flags)
    if not (np.isfinite(K).all() and np.isfinite(dist).all()):
        raise ValueError("[1A.3] calibration produced invalid numbers; check the corner overlays")
    return CalibrationResult(K=K, dist=dist.reshape(1, -1)[:, :5], rms_px=float(rms),
                             image_size=tuple(int(v) for v in image_size),
                             rvecs=list(rvecs), tvecs=list(tvecs))


def compute_reprojection_errors(result, object_points, image_points):
    """[1A.4] Re-project the board points and measure the pixel error per view.

    Purpose:  Quantify calibration quality (rubric 1.1 evidence).
    Flow:     called by run_task1a()  ->  calls cv2.projectPoints()
              previous step: [1A.3]   next step: [1A.5] undistort_image
    Method:   per corner, Euclidean distance between detected and re-projected pixel;
              per view: mean and RMS of those distances (same definitions as calibration_task1a.py).

    Inputs:
        result:        CalibrationResult from [1A.3].
        object_points: same list passed to [1A.3].
        image_points:  same list passed to [1A.3].

    Returns:
        (per_view: list of (mean_px: float, rms_px: float), overall_mean_px: float)
    """
    per_view, all_distances = [], []
    for board, detected, rvec, tvec in zip(object_points, image_points, result.rvecs, result.tvecs):
        projected, _ = cv2.projectPoints(board, rvec, tvec, result.K, result.dist)
        distances = np.linalg.norm(detected.reshape(-1, 2) - projected.reshape(-1, 2), axis=1)
        per_view.append((float(distances.mean()), float(np.sqrt(np.mean(distances ** 2)))))
        all_distances.append(distances)
    return per_view, float(np.concatenate(all_distances).mean())


def undistort_image(bgr, K, dist, alpha=0.0):
    """[1A.5] Remove lens distortion from an image taken by the calibrated camera.

    Purpose:  Produce the corrected image (rubric 1.1; Task 3 "undistort" mode).
    Flow:     called by run_task1a() and assistant.undistort_mode()
              -> calls cv2.getOptimalNewCameraMatrix(), cv2.undistort()

    Inputs:
        bgr:   np.ndarray uint8 (H, W, 3), same camera and resolution as calibration.
        K:     (3, 3) intrinsic matrix.
        dist:  (1, 5) distortion coefficients.
        alpha: 0 crops to valid pixels, 1 keeps all source pixels (black borders).

    Returns:
        np.ndarray uint8 (H, W, 3), undistorted.

    Raises:
        ValueError: image size differs from the calibration size (K principal point is used as a check).
    """
    height, width = bgr.shape[:2]
    if not (0 < K[0, 2] < width and 0 < K[1, 2] < height):
        raise ValueError(f"[1A.5] image is {width}x{height} but K was estimated for another resolution; "
                         f"use a photo from the same camera at the same resolution")
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (width, height), alpha, (width, height))
    return cv2.undistort(bgr, K, dist, None, new_K)


def save_calibration(result, path):
    """[1A.6] Save K, dist and image size so Task 3 and undistort_view_task1a.py can load them.

    Flow:     called by run_task1a()  ->  writes artifacts/calibration.npz

    Inputs:
        result: CalibrationResult.
        path:   output Path (.npz).

    Returns:
        Path written. Keys: K, dist, width, height, rms_px.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, K=result.K, dist=result.dist, width=result.image_size[0],
             height=result.image_size[1], rms_px=result.rms_px)
    return path


def load_calibration(path):
    """[1A.6b] Load a saved calibration.

    Flow:     called by assistant.undistort_mode()

    Inputs:
        path: Path to calibration.npz.

    Returns:
        (K: (3, 3), dist: (1, 5), image_size: (width, height))

    Raises:
        FileNotFoundError: the file does not exist (run Task 1A first).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"[1A.6b] {path} not found: run Task 1A (run_task1a) first")
    data = np.load(path)
    return data["K"], data["dist"], (int(data["width"]), int(data["height"]))


def _summarise(result, per_view, mean_px):
    """Numbers that go in the baseline vs fix_k3 table."""
    k1, k2, p1, p2, k3 = result.dist.ravel()[:5]
    return {"rms_px": round(result.rms_px, 4), "mean_error_px": round(mean_px, 4),
            "worst_view_mean_px": round(max(m for m, _ in per_view), 4),
            "fx": round(float(result.K[0, 0]), 2), "fy": round(float(result.K[1, 1]), 2),
            "cx": round(float(result.K[0, 2]), 2), "cy": round(float(result.K[1, 2]), 2),
            "k1": float(k1), "k2": float(k2), "p1": float(p1), "p2": float(p2), "k3": float(k3)}


def run_task1a(cfg):
    """[1A.R] Run the whole Task 1A flow and save its results.

    Purpose:  One call that produces every Task 1A number and figure for the report.
    Flow:     called by notebook / README command
              -> [1A.1] -> [1A.2a] -> [1A.2] -> [1A.3] (baseline + fix_k3) -> [1A.4] -> [1A.5] -> [1A.6]
              writes results/task1a/ (corners/ overlays, detection_log.csv, view_errors.csv,
              calibration_variants.csv, view_errors.png, before_after.jpg, metrics.json)
              and artifacts/calibration.npz (the variant chosen by cfg.task1a.fix_k3)

    Inputs:
        cfg: config from load_config() (uses cfg.paths, cfg.task1a, cfg.seed).

    Returns:
        Task1AReport.
    """
    set_seed(cfg.seed)
    settings = cfg.task1a
    out_dir = cfg.paths.results / "task1a"
    pattern_size = (settings.pattern_cols, settings.pattern_rows)

    images = load_calibration_images(cfg.paths.calibration_images)                       # [1A.1]
    board = make_board_points(pattern_size, settings.square_size_mm / 1000.0)            # [1A.2a]
    object_points, image_points, accepted, rejected, log_rows = [], [], [], [], []
    for name, bgr in images:                                                             # [1A.2]
        found, corners, detector = detect_corners(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), pattern_size)
        if not found:
            reason = f"no complete {pattern_size[0]}x{pattern_size[1]} inner-corner grid"
            rejected.append((name, reason))
            log_rows.append({"file_name": name, "status": "REJECTED", "detector": "-", "notes": reason})
            continue
        object_points.append(board)
        image_points.append(corners)
        accepted.append(name)
        log_rows.append({"file_name": name, "status": "ACCEPTED", "detector": detector, "notes": ""})
        overlay = bgr.copy()
        cv2.drawChessboardCorners(overlay, pattern_size, corners, True)
        save_image(overlay, out_dir / "corners" / f"{Path(name).stem}_corners.jpg")
    if len(accepted) < settings.min_views:
        raise ValueError(f"[1A.R] only {len(accepted)} usable views (need {settings.min_views}). Check that "
                         f"pattern_cols/rows count INNER corners ({pattern_size}) and look at rejected photos.")
    image_size = (images[0][1].shape[1], images[0][1].shape[0])

    variants, results = {}, {}
    for label, fix_k3 in (("baseline", False), ("fix_k3", True)):                        # [1A.3] + [1A.4]
        result = calibrate_camera(object_points, image_points, image_size, fix_k3=fix_k3)
        per_view, mean_px = compute_reprojection_errors(result, object_points, image_points)
        results[label] = (result, per_view, mean_px)
        variants[label] = _summarise(result, per_view, mean_px)
    chosen, per_view, mean_px = results["fix_k3" if settings.fix_k3 else "baseline"]
    view_errors = [(name, m, r) for name, (m, r) in zip(accepted, per_view)]

    sample_name = getattr(settings, "undistort_sample", None)                           # [1A.5]
    sample = read_image(cfg.project_root / sample_name) if sample_name else dict(images)[accepted[0]]
    undistorted = undistort_image(sample, chosen.K, chosen.dist, settings.undistort_alpha)
    both = np.concatenate([sample, undistorted], axis=1)
    for text, x, colour in (("BEFORE", 16, (0, 0, 255)), ("AFTER", sample.shape[1] + 16, (0, 160, 0))):
        cv2.putText(both, text, (x, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 3)
    figure_paths = [save_image(both, out_dir / "before_after.jpg")]

    fig = plot_view_errors(view_errors)
    figure_paths.append(out_dir / "view_errors.png")
    fig.savefig(figure_paths[-1], dpi=150)
    plt.close(fig)

    save_calibration(chosen, cfg.paths.artifacts / "calibration.npz")                   # [1A.6]
    save_csv(log_rows, out_dir / "detection_log.csv")
    save_csv([{"file_name": n, "mean_reprojection_error_px": round(m, 5), "rms_reprojection_error_px": round(r, 5)}
              for n, m, r in view_errors], out_dir / "view_errors.csv")
    save_csv([{"variant": label, **numbers} for label, numbers in variants.items()],
             out_dir / "calibration_variants.csv")
    metrics_path = save_json({
        "input_folder": str(cfg.paths.calibration_images), "accepted": len(accepted), "total": len(images),
        "pattern_inner_corners": list(pattern_size), "square_size_mm": settings.square_size_mm,
        "image_size_wh": list(image_size), "chosen_variant": "fix_k3" if settings.fix_k3 else "baseline",
        "K": chosen.K, "dist": chosen.dist.ravel(), "rms_px": chosen.rms_px, "mean_error_px": mean_px,
        "variants": variants, "rejected": rejected,
    }, out_dir / "metrics.json")
    return Task1AReport(calibration=chosen, accepted=accepted, rejected=rejected, view_errors=view_errors,
                        mean_error_px=mean_px, variants=variants, figure_paths=figure_paths,
                        metrics_path=metrics_path)
