"""Task 1A checks on rendered checkerboards with a known camera, so no photos are needed."""
import cv2
import numpy as np
import pytest

from src.calibration import (calibrate_camera, compute_reprojection_errors, detect_corners, load_calibration,
                             make_board_points, save_calibration, undistort_image)

TRUE_K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1]])
PATTERN = (6, 5)           # inner corners
SQUARE_M = 0.03


def render_view(rvec, tvec, size=(640, 480), square_px=40):
    """Draw a (cols+1) x (rows+1) board and project it with TRUE_K (no distortion)."""
    cols, rows = PATTERN[0] + 1, PATTERN[1] + 1
    texture = np.full((rows * square_px, cols * square_px), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2:
                texture[r * square_px:(r + 1) * square_px, c * square_px:(c + 1) * square_px] = 0
    # Board origin at the first inner corner, so the outer border starts one square before it.
    outer = np.float32([[-1, -1, 0], [cols - 1, -1, 0], [cols - 1, rows - 1, 0], [-1, rows - 1, 0]]) * SQUARE_M
    projected, _ = cv2.projectPoints(outer, rvec, tvec, TRUE_K, None)
    src = np.float32([[0, 0], [cols * square_px, 0], [cols * square_px, rows * square_px], [0, rows * square_px]])
    warp = cv2.getPerspectiveTransform(src, projected.reshape(4, 2).astype(np.float32))
    canvas = np.full((size[1] + 40, size[0] + 40), 255, np.uint8)
    return cv2.warpPerspective(texture, warp, size, dst=canvas[:size[1], :size[0]].copy(),
                               borderMode=cv2.BORDER_CONSTANT, borderValue=180)


@pytest.fixture(scope="module")
def views():
    rng = np.random.default_rng(3)
    grays = []
    for _ in range(8):
        rvec = rng.uniform(-0.35, 0.35, 3)
        tvec = np.array([-0.09 + rng.uniform(-0.03, 0.03), -0.07 + rng.uniform(-0.03, 0.03), rng.uniform(0.45, 0.7)])
        grays.append(render_view(rvec, tvec))
    return grays


def test_board_points():
    points = make_board_points((6, 9), 0.0091)
    assert points.shape == (54, 3) and points.dtype == np.float32
    assert np.allclose(points[1], [0.0091, 0, 0]) and np.all(points[:, 2] == 0)
    with pytest.raises(ValueError):
        make_board_points((2, 9), 0.01)


def test_detect_and_calibrate_recovers_known_camera(views, tmp_path):
    board = make_board_points(PATTERN, SQUARE_M)
    image_points = []
    for gray in views:
        found, corners, _ = detect_corners(gray, PATTERN)
        assert found and corners.shape == (PATTERN[0] * PATTERN[1], 1, 2)
        image_points.append(corners)
    object_points = [board] * len(image_points)
    result = calibrate_camera(object_points, image_points, (640, 480), fix_k3=True)
    assert result.rms_px < 0.5
    assert result.K[0, 0] == pytest.approx(600, rel=0.03) and result.K[0, 2] == pytest.approx(320, abs=12)
    per_view, mean_px = compute_reprojection_errors(result, object_points, image_points)
    assert len(per_view) == len(views) and mean_px < 0.5
    assert result.dist.ravel()[4] == 0.0                 # fix_k3 holds k3 at 0

    path = save_calibration(result, tmp_path / "calibration.npz")
    K, dist, size = load_calibration(path)
    assert np.allclose(K, result.K) and size == (640, 480)
    assert set(np.load(path).files) >= {"K", "dist", "width", "height"}   # keys undistort_view_task1a.py reads


def test_calibrate_needs_three_views():
    with pytest.raises(ValueError):
        calibrate_camera([make_board_points(PATTERN, SQUARE_M)] * 2, [np.zeros((30, 1, 2), np.float32)] * 2,
                         (640, 480))


def test_undistort_shape_and_size_check():
    image = np.zeros((480, 640, 3), np.uint8)
    dist = np.array([[-0.2, 0.05, 0, 0, 0]])
    assert undistort_image(image, TRUE_K, dist).shape == image.shape
    with pytest.raises(ValueError):
        undistort_image(np.zeros((200, 300, 3), np.uint8), TRUE_K, dist)


def test_missing_calibration_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_calibration(tmp_path / "nope.npz")


def test_collect_calibration_images_sources(tmp_path):
    """[1A.0] upload with files -> new session folder; upload without files -> config folder; bad source -> error."""
    from src.calibration import collect_calibration_images
    from src.config import load_config
    cfg = load_config()
    cfg.project_root = tmp_path
    cfg.task1a.upload_save_root = "uploads"
    folder = collect_calibration_images(cfg, "upload", [("a.png", b"x"), ("b.png", b"y")])
    assert folder.parent == tmp_path / "uploads" and sorted(p.name for p in folder.iterdir()) == ["a.png", "b.png"]
    assert collect_calibration_images(cfg, "upload") == cfg.paths.calibration_images
    with pytest.raises(ValueError):
        collect_calibration_images(cfg, "webcam")
