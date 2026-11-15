# Starting prompt for Antigravity (paste everything below the line)

---

You are helping me finish my FCVG101 Computer Vision coursework. The project folder is open in this workspace
(`FCVG101-Coursework/`). Read this whole message, then read `PROJECT_GUIDE.md` and `README.md` before writing code.
This workspace has three skills in `.agents/skills/`; use them: `fcvg101-coding-rules` (before any code change),
`fcvg101-task3-assistant` (the mode -> function map for Task 3), `fcvg101-run-and-verify` (tests and run commands).
The `.venv` is already created and `requirements.txt` installed; do not install anything else without asking me.

## The coursework

One MVP "Vision Inspection Assistant" (60% of the module):

- Task 1 – classical CV engine (20 pts): 1A camera calibration + undistortion, 1B filtering / FFT / edges / features.
- Task 2 – deep-learning model + data pipeline (35 pts).
- Task 3 – a Streamlit app that calls Task 1 and Task 2 through one routing layer (30 pts).
- Documentation / report (15 pts), submitted as one PDF of at most 5 MB.

Tasks 1A, 1B and 2 are DONE. **Your job is Task 3**: `src/assistant.py` + `app.py`.

## Project rules (from PROJECT_GUIDE.md – follow them strictly)

1. Three layers: core functions in `src/` (one job each) → task runners `run_task1a/1b/2(cfg)` that chain them →
   notebook + `app.py`, which only call `src/` and display results. No algorithm code in `app.py`.
2. Every function has a step ID in its docstring (e.g. `[1A.5]`, `[3.2a]`) and the fixed docstring template:
   Purpose / Flow / Method / Inputs / Returns / Raises. Keep the same IDs as the step catalogue in PROJECT_GUIDE.md.
3. Every path, size, threshold and seed lives in `configs/config.yaml` (loaded with `src.config.load_config()`);
   nothing hard-coded in `src/`.
4. Baseline and advanced variants share one signature and are switched by a parameter.
5. Small, scoped changes. Do not rewrite or reformat working Task 1/2 code; do not add extra scripts unless I ask.
6. Images inside the code are BGR `uint8` (OpenCV); convert with `io_utils.to_rgb` only for display.

## What exists and its results

| Part | Code | Result |
| --- | --- | --- |
| 1A calibration | `src/calibration.py` (`[1A.0]`–`[1A.7]`, `run_task1a`) | 13 webcam views (640×480, board 6×9 inner corners on a phone screen, 9.1 mm squares), 13/13 accepted, RMS 0.097 px, mean 0.081 px. `fix_k3` variant: same RMS, saner coefficients (k1 −0.31, k2 0.11). Saved to `artifacts/calibration.npz` (keys K, dist, width, height, rms_px). |
| 1A inputs | `run_task1a(cfg, source="upload" \| "camera", uploads=None)` | Two input options: `upload` = photo files (folder or `(name, bytes)` list from the app), `camera` = my guided capture tool `tools/task1a/capture_checker_task1a.py`, run unchanged. `view_undistorted(cfg, "camera")` opens `tools/task1a/undistort_view_task1a.py` for a live before/after window. |
| 1B processing | `src/filtering.py`, `frequency.py`, `edges.py`, `features.py`, `task1b.py` | Noise + box/Gaussian/median (PSNR), FFT low/high-pass, Sobel/Canny, Harris/ORB/SIFT + ratio test + RANSAC. Results in `results/task1b/`. |
| 2 deep learning | `src/data_pipeline.py`, `model.py`, `train.py`, `evaluate.py`, `inference.py`, `baselines.py`, `explain.py` | TensorFlow flowers (5 classes). Small CNN F1 0.776 (no aug) / 0.766 (aug); MobileNetV3 transfer **F1 0.945**, 16 MB, 16.6 ms/img on CPU → selected, saved as `artifacts/model.pth` + `classes.json`. Baselines, Grad-CAM/XAI in `results/task2/`. |
| 3 assistant | `src/assistant.py` | **Not built**: `run_mode` and `MODES` exist; every `*_mode` function raises `NotImplementedError`. |
| App | `app.py` | Thin Streamlit shell: mode select box, upload, calls `run_mode`, shows `ModeResult`. |

## Task 3: what to build

1. Fill the mode functions in `src/assistant.py`, each one calling the existing Task 1/2 function named in its docstring
   and returning a `ModeResult(images, numbers, message)`:
   - `undistort_mode` [3.2a] → `calibration.load_calibration` + `undistort_image` (show before/after, RMS from the npz).
   - `filter_mode` [3.2b] → `filtering.apply_spatial_filters`
   - `frequency_mode` [3.2c] → `frequency.filter_frequency`
   - `edges_mode` [3.2d] → `edges.compute_edges`
   - `features_mode` [3.2e] → `features.match_features` (needs a second image)
   - `classify_mode` [3.2f] → `inference.load_model` (cache it) + `predict` → label, score, top-3.
   Read each target function's signature first; do not change those functions. `params` carries UI overrides.
2. In `app.py` give every mode two input options: **Upload image** (`st.file_uploader`) or **Camera**
   (`st.camera_input` for one snapshot). Add a "Calibrate camera" section with the same two options:
   upload several photos (`accept_multiple_files=True` → `run_task1a(cfg, "upload", [(f.name, f.getvalue()), ...])`)
   or live capture (`run_task1a(cfg, "camera")`, which opens my capture tool's window), then show RMS, per-view errors
   and `results/task1a/before_after.jpg`. Optional button: `view_undistorted(cfg, "camera")`.
3. Handle errors in the UI (wrong resolution for undistort, missing second image, missing model) with clear messages.
4. Add `tests/test_task3.py` (each mode returns a `ModeResult` on a small synthetic image) and fill notebook section 3
   with the same `run_mode` calls the app makes.
5. Update the README status line when done.

## Local setup (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest tests                        # all tests should pass
python -m src.calibration --source upload     # re-run Task 1A on data/calibration
python -m src.calibration --source camera --view   # live capture + live before/after
streamlit run app.py
```

## Notes

- The calibration photos look like they came through the capture tool's simulated-lens option (black curved border,
  recovered k1/k2 close to −0.3/0.08). Undistort only frames from that same pipeline and resolution (640×480).
- Square size only scales the extrinsics (board distance); K and distortion do not depend on it.
- Before you change anything, tell me your plan in a short list and wait for my OK.
