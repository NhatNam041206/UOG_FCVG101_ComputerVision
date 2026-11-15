---
name: fcvg101-task3-assistant
description: How to build Task 3 of FCVG101, the assistant modes in src/assistant.py and the Streamlit app.py, by calling the finished Task 1A/1B/2 functions. Use when implementing or changing any *_mode function, run_mode, MODES, or app.py.
---

# Task 3: assistant + app

`src/assistant.py` already has `ModeResult(images: dict, numbers: dict, message: str)`, the `MODES` table and
`run_mode(mode, image, cfg, image_2=None, params=None)`. Each `*_mode(image, cfg, image_2=None, params=None)`
currently raises `NotImplementedError`. Fill them; do not change `run_mode`'s signature (the app and notebook call it).

`params` is an optional dict of UI overrides; fall back to `cfg` values: `value = (params or {}).get("ksize", cfg.task1b.kernel_size)`.
`images` values are BGR or grayscale uint8, keyed by a caption string. `numbers` holds plain floats/ints/str.

## Mode -> existing function (read the target's docstring first; never modify Task 1/2 functions)

| Mode | Call | Notes |
| --- | --- | --- |
| `undistort` [3.2a] | `calibration.load_calibration(cfg.paths.artifacts / "calibration.npz")` -> `(K, dist, (w, h))`; `calibration.undistort_image(bgr, K, dist, alpha=cfg.task1a.undistort_alpha)` | Image must be exactly w x h (640x480): if not, raise ValueError telling the user. Show BEFORE and AFTER; numbers: fx, fy, cx, cy, k1, k2, RMS (npz key `rms_px`). |
| `filter` [3.2b] | `filtering.add_noise(bgr, kind, level)` (optional) then `filtering.apply_spatial_filters(image, ksize, sigma, method)` -> `{"box", "gaussian", "median"}`; `filtering.compare_to_reference(filtered, clean)` -> PSNR/MSE | Use `cfg.task1b.kernel_size`, `gaussian_sigma`, `filter_method`, `noise_tests[0]`. |
| `frequency` [3.2c] | `frequency.filter_frequency(gray, kind, radius, profile, order)` -> `FrequencyResult(log_spectrum, mask, filtered, filtered_uint8)` | Takes GRAYSCALE. Normalise `log_spectrum` to uint8 for display. Config: `fft_radius`, `butterworth_order`. |
| `edges` [3.2d] | `edges.compute_edges(gray, low, high, blur_ksize, blur_sigma, auto_threshold)` -> `EdgeResult(smoothed, grad_x, grad_y, magnitude, orientation, canny, thresholds)` | Grayscale in. Show magnitude (scaled to uint8) and canny. Config: `canny_low/high`, `edge_blur_*`. |
| `features` [3.2e] | `features.match_features(img1, img2, detector, ratio, use_ransac, max_features, reproj_threshold)` -> `MatchResult`; draw with `visualize.draw_matches(img1, img2, result, max_matches)` | Needs `image_2` (run_mode already checks). numbers = `result.stats`. Config: `detector`, `ratio_threshold`, `use_ransac`, `ransac_threshold_px`, `draw_max_matches`. |
| `classify` [3.2f] | `inference.load_model(cfg.paths.artifacts / "model.pth", cfg.paths.artifacts / "classes.json")` -> `(model, meta)`; `inference.predict(model, meta, bgr)` -> `Prediction(label, score, top3)` | Load the model ONCE (module-level cache or `functools.lru_cache`; in app.py wrap with `st.cache_resource`). Classes: daisy, dandelion, roses, sunflowers, tulips. Say the score is a softmax score, not a calibrated probability. |

Grayscale: `cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)`.

## app.py (layer 3: UI only)
- Mode select box -> input choice radio: **Upload image** (`st.file_uploader`) or **Camera** (`st.camera_input`).
  Both give bytes -> `io_utils.decode_image_bytes(data)` -> BGR. `features` needs two images (two uploaders / two snapshots).
- Note for `undistort`: browser camera snapshots may not be 640x480; show the ValueError message clearly.
- "Calibrate camera" section with the same two choices:
  - Upload: `st.file_uploader(..., accept_multiple_files=True)` -> `calibration.run_task1a(cfg, "upload", [(f.name, f.getvalue()) for f in files])`.
  - Live: button -> `calibration.run_task1a(cfg, "camera")` (opens the owner's capture tool window on this PC; blocks until it closes).
  - Then show `report.calibration.rms_px`, `report.mean_error_px`, `report.view_errors` table, `results/task1a/before_after.jpg`.
  - Optional button: `calibration.view_undistorted(cfg, "camera")` for the live BEFORE | AFTER window.
- Catch `ValueError`, `FileNotFoundError`, `NotImplementedError` and show `st.error` / `st.warning`, as the current app.py does.
- Display with `st.image(io_utils.to_rgb(img), caption=...)`; numbers with `st.table`.

## Tests and notebook
- Add `tests/test_task3.py`: for every mode, call `run_mode` on a small synthetic image (e.g. a drawn checkerboard,
  640x480 for undistort) and assert a `ModeResult` with at least one image. Skip `classify` with
  `pytest.importorskip("torch")` if torch is missing, and skip `undistort` if `artifacts/calibration.npz` is missing.
- Notebook section 3: the same `run_mode` calls the app makes, one cell per mode, with a short explanation after each.
- Update the README status line.
