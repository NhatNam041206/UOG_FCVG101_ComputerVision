---
name: fcvg101-run-and-verify
description: Commands to run, test and check the FCVG101 project on Windows (venv, pytest, Task 1A/1B/2 runners, Streamlit app). Use after any code change, before saying a task is done, or when asked how to run something.
---

# Run and verify (Windows PowerShell, project root)

Always use the project venv: `.venv\Scripts\python.exe` (or activate with `.venv\Scripts\Activate.ps1`).
Do not `pip install` anything new without asking the owner; everything needed is in `requirements.txt`.

## After a change
1. Run only the tests for what changed, e.g. `python -m pytest tests/test_task3.py -q`.
2. When those pass, run the full suite once: `python -m pytest tests -q`.
3. For app changes: `streamlit run app.py` and try each mode once with an upload (sample photos: `data/samples/`,
   calibration photos: `data/calibration/`, flowers: `data/raw/flower_photos/<class>/`).
4. Report what you ran and the result. Never claim a check you did not run.

## Runners (do NOT re-run Task 2 unless asked: it retrains for ~55 min on CPU and overwrites artifacts/)
| What | Command |
| --- | --- |
| Task 1A from photo files | `python -m src.calibration --source upload` |
| Task 1A live webcam + before/after window | `python -m src.calibration --source camera --view` |
| Task 1B | `python -c "from src.config import load_config; from src.task1b import run_task1b; run_task1b(load_config())"` |
| App | `streamlit run app.py` |

Outputs: `artifacts/` (calibration.npz, model.pth, classes.json) are read by Task 3; `results/task*/` hold report
figures. Do not delete or regenerate them unless asked.

Known results to sanity-check against: Task 1A RMS 0.097 px on 13 views (640x480); Task 2 MobileNetV3 test macro F1 0.945.
