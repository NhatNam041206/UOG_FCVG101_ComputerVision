# FCVG101 Coursework – Project Guide

This file sets the project's structure and coding rules. Every piece of code follows it, so you can open any
function and see straight away where it sits in the flow, what goes in, what comes out, and why it exists.

---

## 1. The flow in one picture

```
                         configs/config.yaml
                                 |
                          load_config()            [CFG]
                                 |
     +---------------------------+---------------------------+
     |                           |                           |
 TASK 1A  calibration.py     TASK 1B  filtering.py ...   TASK 2  data_pipeline.py ...
 run_task1a(cfg)             run_task1b(cfg)             run_task2(cfg)
   1A.1 load_calibration_images   1B.1 add_noise           2.1 build_manifest
   1A.2 detect_corners            1B.2 apply_spatial_filters 2.2 clean_dataset
   1A.3 calibrate_camera          1B.3 filter_frequency    2.3 split_dataset
   1A.4 compute_reprojection_errors 1B.4 compute_edges     2.4 make_dataloaders
   1A.5 undistort_image           1B.5 match_features      2.5 build_model
   1A.6 save_calibration                                   2.6 train_model
     |                           |                         2.7 evaluate_model
     v                           v                         2.8 save_model
 artifacts/calibration.npz   results/task1b/*              artifacts/model.pth, classes.json
     |                           |                           |
     +---------------------------+---------------------------+
                                 |
                  TASK 3  assistant.py  run_mode(mode, image, ...)   [3.1 route, 3.2 execute]
                                 |
            +--------------------+--------------------+
            |                                         |
   notebooks/FCVG101_Coursework.ipynb          app.py (Streamlit)
   (report figures + numbers)                  (user uploads image, picks mode)
```

**Three layers, always in this order:**

| Layer | Where | Does | Never does |
| --- | --- | --- | --- |
| 1. Core functions | `src/*.py` | One algorithm step: arrays in, arrays/results out | Read config, touch files, plot, print |
| 2. Task runners | `run_task1a()`, `run_task1b()`, `run_task2()` | Chain core steps in order, save results | Contain algorithm maths |
| 3. Entry points | notebook, `app.py` | Call runners or `assistant.run_mode()`, show results | Contain algorithm code |

So when you read the notebook or the app, every call goes **down** one layer, and every algorithm lives in exactly
one place in `src/`.

---

## 2. Folder structure

```
FCVG101-Coursework/
  PROJECT_GUIDE.md            this file
  README.md                   install + run commands (for the marker)
  requirements.txt
  configs/
    config.yaml               every path, size, seed, threshold lives here
  src/
    __init__.py
    config.py                 [CFG]  load_config()
    io_utils.py               [IO]   read_image(), save_image(), save_json(), to_rgb()
    calibration.py            [1A]   Task 1A steps + run_task1a()
    filtering.py              [1B]   noise + box/Gaussian/median + metrics
    frequency.py              [1B]   FFT spectrum, low/high-pass masks
    edges.py                  [1B]   Sobel, Canny
    features.py               [1B]   Harris/SIFT/ORB, ratio test, RANSAC
    task1b.py                 [1B]   run_task1b() only: chains the four files above
    data_pipeline.py          [2]    manifest, cleaning, split, augmentation, loaders
    model.py                  [2]    CNN definitions
    train.py                  [2]    train_model(), run_task2()
    evaluate.py               [2]    metrics, confusion matrix
    inference.py              [2→3]  load_model(), predict()
    assistant.py              [3]    MODES table, run_mode()
    visualize.py              [VIS]  all plotting (figures for notebook and app)
  notebooks/
    FCVG101_Coursework.ipynb  one section per task, calls runners, shows figures
  app.py                      Streamlit UI, calls assistant.run_mode() only
  data/
    calibration/              your real checkerboard photos (same camera, same resolution)
    samples/                  test images for Task 1B (noisy, image pairs)
    raw/                      Task 2 dataset as downloaded, never edited
    splits/                   generated train/ val/ test/
  artifacts/                  outputs other tasks load: calibration.npz, model.pth, classes.json
  results/
    task1a/  task1b/  task2/  figures, CSVs and metrics.json used in the report
  tests/                      small checks that each core function returns the right shape/type
```

Your existing Task 1A scripts (`calibration_task1a.py`, `capture_checker_task1a.py`, `undistort_view_task1a.py`)
stay in `Cousework/` unchanged. `capture_checker_task1a.py` remains the tool for taking photos. `src/calibration.py`
holds the same calibration steps as importable functions, and its output (`artifacts/calibration.npz`) uses the
same keys (`K`, `dist`, `width`, `height`), so `undistort_view_task1a.py` can still read it.

---

## 3. Coding rules

### R1. Every function has a step ID and a fixed docstring

The step ID (for example `[1A.3]`) is the same in the code, the flow picture above, the step catalogue (section 4),
the notebook headings and the report. Search for the ID and you find every place it is used.

```python
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
        ValueError: fewer than 3 views.
    """
```

### R2. One function, one step

A function does one thing from the catalogue. If a docstring needs "and then" twice, split it.

### R3. Inputs and outputs are explicit

- Core functions take arrays and plain values, never the whole config and never a file path (except the `io_utils`
  readers).
- Anything with more than two outputs returns a small `@dataclass` (for example `CalibrationResult`,
  `MatchResult`, `EvalResult`), never a bare tuple, so the names show what came back.
- No `print()` in `src/`. Runners return results; the notebook or app displays them.

### R4. Image conventions (one rule everywhere)

| What | Format |
| --- | --- |
| Colour image inside `src/` | BGR, `uint8`, shape (H, W, 3), as OpenCV reads it |
| Grayscale | `uint8`, shape (H, W) |
| Float processing (FFT, gradients) | `float32`, stated in the docstring |
| Display (notebook, Streamlit) | convert with `to_rgb()` at the very last moment |
| Model input | RGB tensor, resized and normalised by the same `make_transforms()` used in training |

### R5. Baseline and advanced share one function

Each method has a baseline (the OpenCV call the brief names) and optional advanced variants behind the **same
signature**, chosen by a parameter, not a separate file:

```python
match_features(img1, img2, detector="orb", ratio=0.75, use_ransac=False)
detect_corners_harris(gray, method="opencv")      # or method="scratch"
calibrate_camera(..., fix_k3=False)
build_model(name="small_cnn")                     # or name="resnet18"
```

The notebook runs the baseline first, then the advanced variants, and puts them in one comparison table. This is
how the report shows "baseline → improvement" with real numbers.

### R6. No hard-coded values

Paths, image sizes, seeds, kernel sizes, thresholds and board size all come from `configs/config.yaml`, read once by
`load_config()` and passed down by the runners. Paths use `pathlib.Path` relative to the project root, so the
project runs on any machine.

### R7. Reproducible

One seed (`cfg.seed`, default 42) is set for Python, NumPy and PyTorch at the start of every runner. Each runner
writes its numbers to `results/<task>/metrics.json` and its figures to `results/<task>/`, so every number in the
report can be traced to a file.

### R8. Names

Functions use `verb_noun` (`detect_corners`, `apply_spatial_filters`). Variables say what they hold and their unit
where it matters (`square_size_m`, `rms_px`, `ratio_threshold`). No one-letter names except `K` and loop indices.

### R9. Fail loudly with a helpful message

Check inputs at the start of each core function (missing image, wrong shape, too few views, missing model file) and
raise a clear error. The app catches these and shows the message to the user (rubric 3.1).

---

## 4. Step catalogue

This is the map of the whole flow. Each row is one function.

### Task 1A – Camera calibration (`src/calibration.py`, rubric 1.1)

| ID | Function | Input | Output | Purpose |
| --- | --- | --- | --- | --- |
| 1A.1 | `load_calibration_images(folder)` | folder path | list of (name, BGR image) | Read photos, reject unreadable or mixed-size ones |
| 1A.2 | `detect_corners(gray, pattern_size)` | grayscale, (cols, rows) inner corners | `found`, corners (N,1,2) | Find the board grid, refine to sub-pixel |
| 1A.3 | `calibrate_camera(obj_pts, img_pts, size, fix_k3)` | 3D/2D point lists, image size | `CalibrationResult` | Estimate K and distortion |
| 1A.4 | `compute_reprojection_errors(result, obj_pts, img_pts)` | calibration, point lists | per-view mean/RMS error (px) | Measure fit quality |
| 1A.5 | `undistort_image(bgr, K, dist, alpha)` | image, K, dist | undistorted image | Remove lens distortion |
| 1A.6 | `save_calibration(result, path)` | result, path | `calibration.npz` | Hand K and dist to Task 3 |
| 1A.R | `run_task1a(cfg)` | config | `Task1AReport` + files in results/task1a | Run 1A.1–1A.6 in order |

### Task 1B – Classical processing (rubric 1.2, 1.3)

| ID | Function | Input | Output | Purpose |
| --- | --- | --- | --- | --- |
| 1B.1 | `add_noise(bgr, kind, level, seed)` | clean image | noisy image | Make a controlled test image |
| 1B.2 | `apply_spatial_filters(img, ksize, sigma)` | image | dict box/gaussian/median | Denoise with three filters |
| 1B.2m | `compare_to_reference(filtered, clean)` | images | MSE, PSNR (dB) | Score each filter fairly |
| 1B.3 | `filter_frequency(gray, kind, radius)` | grayscale | spectrum, mask, filtered | Low/high-pass in the FFT domain |
| 1B.4 | `compute_edges(gray, low, high)` | grayscale | Sobel x/y/magnitude, Canny | Find intensity edges |
| 1B.5 | `match_features(img1, img2, detector, ratio, use_ransac)` | two images | `MatchResult` (keypoints, good matches, inliers) | Find and match interest points |
| 1B.R | `run_task1b(cfg)` | config | `Task1BReport` + figures | Run 1B.1–1B.5 in order |

### Task 2 – Data pipeline and model (rubric 2.1–2.3)

| ID | Function | Input | Output | Purpose |
| --- | --- | --- | --- | --- |
| 2.0 | `download_and_extract(url, dest)` (io_utils) | dataset URL | `data/raw/<class>/` | Fetch the dataset once |
| 2.1 | `build_manifest(raw_dir)` | dataset folder | DataFrame (file, label, size) | Ingest and list every image |
| 2.2 | `clean_dataset(manifest)` | manifest | cleaned manifest + removal log | Drop corrupt, tiny and duplicate images |
| 2.3 | `split_dataset(manifest, ratios, seed)` | manifest | train/val/test manifests | Stratified split before augmentation |
| 2.4 | `make_dataloaders(splits, img_size, batch)` | splits | train/val/test loaders | Resize, normalise, augment train only |
| 2.5 | `build_model(name, n_classes)` | model name | `nn.Module` | Baseline small CNN or transfer model |
| 2.5d | `resolve_device(name)` | "auto" / "cpu" / "cuda" | device string | GPU when available (Colab), else CPU |
| 2.6 | `train_model(model, loaders, cfg)` | model, loaders | trained model + history | Train, keep best on validation |
| 2.6t | `tune_hyperparameters(splits, classes, exp, grid, ...)` | grid of values | chosen values + trials | Short grid search on validation |
| 2.7 | `evaluate_model(model, test_loader)` | model, test data | `EvalResult` (acc, macro F1, confusion matrix) | Final honest test score |
| 2.7b | `benchmark_model(model, image_size)` | trained model | params, MB, latency, throughput | Size/speed criteria for choosing the model |
| 2.8 | `save_model(model, classes, path)` | model, class names | `model.pth`, `classes.json` | Hand the model to Task 3 |
| 2.R | `run_task2(cfg)` | config | `Task2Report` + files | Run 2.1–2.8 in order |

### Task 3 – Assistant (`src/assistant.py`, `src/inference.py`, rubric 3.1, 3.2)

| ID | Function | Input | Output | Purpose |
| --- | --- | --- | --- | --- |
| 3.0 | `load_model(path)` | artifact path | model, classes | Load once, cached by the app |
| 3.1 | `run_mode(mode, image, image2=None, params=None)` | mode name, image(s) | `ModeResult` (images, numbers, message) | Pick the right function for the user's request |
| 3.2 | `MODES` table | – | mode → function mapping | One place that lists what the app can do |
| 3.3 | `predict(model, classes, bgr)` | model, image | label, score, top-3 | Classify an uploaded image |

The `MODES` table makes the app's routing readable at a glance:

```python
MODES = {
    "undistort": undistort_mode,     # -> [1A.5] using artifacts/calibration.npz
    "filter":    filter_mode,        # -> [1B.2]
    "frequency": frequency_mode,     # -> [1B.3]
    "edges":     edges_mode,         # -> [1B.4]
    "features":  features_mode,      # -> [1B.5] needs two images
    "classify":  classify_mode,      # -> [3.3]  using artifacts/model.pth
}
```

---

## 5. How the notebook is laid out

The notebook only calls `src/` functions and displays results. Each heading carries the step ID:

```
0  Setup            load_config(), set seed, check data folders
1A Calibration      1A.1 ... 1A.6, table: baseline vs fix_k3
1B Processing       1B.1 ... 1B.5, tables: filter PSNR, SIFT vs ORB vs RANSAC
2  Deep learning    2.1 ... 2.8, curves, confusion matrix, small CNN vs transfer model
3  Assistant        run_mode() for each mode, the same calls app.py makes
```

---

## 6. Build order

1. Skeleton: folders, `config.yaml`, `config.py`, `io_utils.py`, empty modules with docstrings only.
2. Task 1A in `src/calibration.py`, checked against `calibration_task1a.py` on the same photos.
3. Task 1B baselines, then advanced variants.
4. Task 2 baseline CNN end to end, then the transfer model.
5. `assistant.py` and `app.py`.
6. Notebook sections filled as each task lands; report written from `results/`.

Task 2 dataset: TensorFlow flowers (5 classes, CC-BY 2.0); any folder-per-class dataset can replace it.
