"""[1A] Task 1A - camera calibration and undistortion (rubric 1.1).

Flow:  run_task1a(cfg, source)
         -> [1A.0] collect_calibration_images  (source "upload" = photo files, "camera" = live capture tool)
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
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.config import set_seed
from src.io_utils import list_images, read_image, save_csv, save_image, save_json
from src.visualize import plot_view_errors

MIN_VIEWS = 3
SOURCES = ("upload", "camera")   # [1A.0] the two ways to get calibration photos


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


def _tool_path(cfg, key):
    """Absolute path of one of nam's Task 1A tools in tools/task1a (run unchanged, never re-implemented)."""
    path = cfg.project_root / getattr(cfg.task1a, key)
    if not path.is_file():
        raise FileNotFoundError(f"[1A.0] {path} not found; keep capture_checker_task1a.py, undistort_view_task1a.py "
                                f"and calibration_task1a.py together in tools/task1a/")
    return path


def _new_session_folder(cfg, key):
    """data/calibration_<live|uploads>/session_YYYYmmdd_HHMMSS, so sessions are never mixed."""
    folder = cfg.project_root / getattr(cfg.task1a, key) / time.strftime("session_%Y%m%d_%H%M%S")
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def collect_calibration_images(cfg, source=None, uploads=None):
    """[1A.0] Get the checkerboard photos from one of two sources and return the folder that holds them.

    Purpose:  Let the user either upload photos or take them live from the camera (one camera, one resolution).
    Flow:     called by run_task1a() (and the app)  ->  next step: [1A.1] load_calibration_images
    Method:   "upload": uploads given -> written to a new data/calibration_uploads/session_* folder;
                        no uploads -> the photos already in paths.calibration_images are used.
              "camera": runs tools/task1a/capture_checker_task1a.py --camera (nam's guided capture tool,
                        unchanged) into a new data/calibration_live/session_* folder; its window guides the
                        13-shot plan and closes when done (Q quits early).

    Inputs:
        cfg:     config from load_config() (uses cfg.task1a.source, camera_index, screen_board, tool paths).
        source:  "upload" | "camera"; None = cfg.task1a.source.
        uploads: optional list of (file_name: str, data: bytes), e.g. from st.file_uploader(accept_multiple_files).

    Returns:
        Path of the folder with the photos.

    Raises:
        ValueError: unknown source.
        FileNotFoundError: capture tool missing, or no photos were saved.
        subprocess.CalledProcessError: the capture tool failed (camera not found, ...).
    """
    settings = cfg.task1a
    source = source or settings.source
    if source not in SOURCES:
        raise ValueError(f"[1A.0] source must be one of {SOURCES}, got {source!r}")
    if source == "upload":
        if not uploads:
            return cfg.paths.calibration_images
        folder = _new_session_folder(cfg, "upload_save_root")
        for name, data in uploads:
            (folder / Path(name).name).write_bytes(data)
        return folder
    folder = _new_session_folder(cfg, "live_save_root")
    command = [sys.executable, str(_tool_path(cfg, "capture_tool")), "--camera", str(settings.camera_index),
               "--save-dir", str(folder), "--output", str(cfg.paths.results / "task1a" / "capture"),
               "--cols", str(settings.pattern_cols), "--rows", str(settings.pattern_rows),
               "--square-mm", str(settings.square_size_mm), "--force"]
    if settings.screen_board:
        command.append("--screen")
    subprocess.run(command, check=True, cwd=cfg.project_root)
    if not list_images(folder):
        raise FileNotFoundError(f"[1A.0] the capture tool saved no photos in {folder}")
    return folder


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


def run_task1a(cfg, source=None, uploads=None):
    """[1A.R] Run the whole Task 1A flow and save its results.

    Purpose:  One call that produces every Task 1A number and figure for the report.
    Flow:     called by notebook / README command
              -> [1A.0] -> [1A.1] -> [1A.2a] -> [1A.2] -> [1A.3] (baseline + fix_k3) -> [1A.4] -> [1A.5] -> [1A.6]
              writes results/task1a/ (corners/ overlays, detection_log.csv, view_errors.csv,
              calibration_variants.csv, view_errors.png, before_after.jpg, metrics.json)
              and artifacts/calibration.npz (the variant chosen by cfg.task1a.fix_k3)

    Inputs:
        cfg:     config from load_config() (uses cfg.paths, cfg.task1a, cfg.seed).
        source:  "upload" | "camera"; None = cfg.task1a.source (see [1A.0]).
        uploads: optional list of (file_name, bytes) for source "upload".

    Returns:
        Task1AReport.
    """
    set_seed(cfg.seed)
    settings = cfg.task1a
    out_dir = cfg.paths.results / "task1a"
    pattern_size = (settings.pattern_cols, settings.pattern_rows)

    folder = collect_calibration_images(cfg, source, uploads)                             # [1A.0]
    images = load_calibration_images(folder)                                             # [1A.1]
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
        "input_folder": folder.relative_to(cfg.project_root).as_posix() if folder.is_relative_to(cfg.project_root) else str(folder), "accepted": len(accepted), "total": len(images),
        "pattern_inner_corners": list(pattern_size), "square_size_mm": settings.square_size_mm,
        "image_size_wh": list(image_size), "chosen_variant": "fix_k3" if settings.fix_k3 else "baseline",
        "K": chosen.K, "dist": chosen.dist.ravel(), "rms_px": chosen.rms_px, "mean_error_px": mean_px,
        "variants": variants, "rejected": rejected,
    }, out_dir / "metrics.json")
    return Task1AReport(calibration=chosen, accepted=accepted, rejected=rejected, view_errors=view_errors,
                        mean_error_px=mean_px, variants=variants, figure_paths=figure_paths,
                        metrics_path=metrics_path)


def view_undistorted(cfg, source="camera", image_path=None):
    """[1A.7] Live BEFORE | AFTER window with nam's undistort_view_task1a.py (run unchanged).

    Purpose:  Show the saved calibration working on the camera feed or on one photo.
    Flow:     called by the CLI below (or the app)  ->  runs tools/task1a/undistort_view_task1a.py
              with artifacts/calibration.npz; keys in the window: G grid, A crop, S snapshot, Q quit.

    Inputs:
        cfg:        config from load_config().
        source:     "camera" (live, cfg.task1a.camera_index) or "upload" (one photo file).
        image_path: photo for source "upload".

    Returns:
        None (snapshots go to results/task1a/undistort_live/).

    Raises:
        ValueError: unknown source, or "upload" without image_path.
        FileNotFoundError: calibration.npz or the tool is missing.
    """
    calib = cfg.paths.artifacts / "calibration.npz"
    if not calib.is_file():
        raise FileNotFoundError(f"[1A.7] {calib} not found: run Task 1A (run_task1a) first")
    if source not in SOURCES or (source == "upload" and not image_path):
        raise ValueError(f"[1A.7] source must be 'camera', or 'upload' with image_path; got {source!r}")
    command = [sys.executable, str(_tool_path(cfg, "undistort_tool")), "--calib", str(calib),
               "--output", str(cfg.paths.results / "task1a" / "undistort_live")]
    command += ["--camera", str(cfg.task1a.camera_index)] if source == "camera" else ["--image", str(image_path)]
    subprocess.run(command, check=True, cwd=cfg.project_root)


if __name__ == "__main__":
    # python -m src.calibration --source upload|camera [--view]
    import argparse

    from src.config import load_config

    parser = argparse.ArgumentParser(description="[1A.R] Task 1A: calibrate from uploaded photos or the live camera")
    parser.add_argument("--source", choices=SOURCES, help="default: task1a.source in configs/config.yaml")
    parser.add_argument("--view", action="store_true", help="open the live BEFORE | AFTER undistort window afterwards")
    args = parser.parse_args()
    config = load_config()
    report = run_task1a(config, source=args.source)
    print(f"{len(report.accepted)} views used, {len(report.rejected)} rejected, RMS {report.calibration.rms_px:.3f} px, "
          f"mean error {report.mean_error_px:.3f} px -> artifacts/calibration.npz, results/task1a/")
    if args.view:
        view_undistorted(config, "camera")
