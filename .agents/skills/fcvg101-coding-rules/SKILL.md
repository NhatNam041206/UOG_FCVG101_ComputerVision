---
name: fcvg101-coding-rules
description: House rules for writing or editing any Python in the FCVG101 coursework (src/, app.py, tests, notebook). Use before creating or changing a function, docstring, config value or file in this project.
---

# FCVG101 coding rules

The full guide is `PROJECT_GUIDE.md` in the project root; read sections 2-4 once per session. The rules below are the
ones that are checked most often.

## Layers (never mix them)
1. **Core functions** in `src/<module>.py`: one job each, no file I/O unless the function's job is saving.
2. **Task runners** `run_task1a/1b/2(cfg)` and `assistant.run_mode()`: chain core functions in order, save results.
3. **Notebook and `app.py`**: only call `src/` functions and display. No OpenCV/torch maths in `app.py`.

## Every function
- First docstring line starts with its step ID, e.g. `"""[3.2a] Undistort with artifacts/calibration.npz."""`.
  IDs must match the step catalogue in PROJECT_GUIDE.md section 4 (Task 3 = `[3.0]`-`[3.3]`, modes `[3.2a]`-`[3.2f]`).
- Template, in this order: Purpose / Flow (called by -> calls, previous / next step) / Method (if not obvious) /
  Inputs / Returns / Raises. Copy the layout of `src/calibration.py:calibrate_camera`.
- More than two outputs -> return a small `@dataclass` (like `ModeResult`, `EdgeResult`).
- Baseline and advanced variant share one signature, switched by a parameter (e.g. `method="opencv" | "scratch"`).

## Data and config
- Images in code are **BGR uint8** (OpenCV order). Convert with `src.io_utils.to_rgb` only when displaying.
- Every path, size, threshold, seed lives in `configs/config.yaml`; read it with `src.config.load_config()` and pass
  `cfg` down. Never hard-code a number in `src/`. Add new settings under the task's section (e.g. a new `task3:` block).
- Paths under `paths:` are already absolute `Path`s after `load_config()`; other relative paths join `cfg.project_root`.
- Errors: raise `ValueError` / `FileNotFoundError` with the step ID and a fix hint, e.g.
  `raise ValueError("[3.2e] Feature matching needs a second image.")`.

## Scope
- Make the smallest change that does the job. Do not reformat, rename or "improve" working Task 1/2 code.
- Do not add new scripts or files beyond what the task needs; `tools/task1a/` holds the owner's own tools: never edit them.
- After changing code, follow the `fcvg101-run-and-verify` skill.
