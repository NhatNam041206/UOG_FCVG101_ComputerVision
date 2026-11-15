# FCVG101 Coursework – Vision Inspection Assistant

Structure and coding rules: see `PROJECT_GUIDE.md`.

## Install (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run on Google Colab (GPU)

1. Upload `FCVG101-Coursework.zip` to *My Drive* (top level, keep the name).
2. Colab: **File → Upload notebook** → `notebooks/FCVG101_Coursework.ipynb`; **Runtime → Change runtime type → T4 GPU**.
3. **Runtime → Run all.** The first code cell mounts Drive, unzips to `/content/FCVG101-Coursework`, installs `timm`;
   Task 2 downloads the dataset (≈230 MB) and retrains on the GPU (`config.yaml: task2.device: auto`).
4. The last cell saves `results/`, `artifacts/`, `data/splits/` and `exports/` to *My Drive/FCVG101-outputs.zip*.
   Unzip it over your project folder to keep the new models and figures.

Training time is measured on the device used (GPU on Colab); model latency/throughput is always measured on the CPU,
because that is where the Task 3 app runs.

## Run

| What | Command |
| --- | --- |
| Check the skeleton imports | `python -m pytest tests` |
| Task 1A calibration, photo files in `data/calibration/` | `python -m src.calibration --source upload` |
| Task 1A calibration, live webcam (guided capture tool) | `python -m src.calibration --source camera --view` (`--view` opens the live before/after window) |
| Task 1B (filters, FFT, edges, features) | put `clean.jpg`, `pair_1.jpg`, `pair_2.jpg` (+ optional `unrelated.jpg`) in `data/samples/`, then `python -c "from src.config import load_config; from src.task1b import run_task1b; run_task1b(load_config())"` |
| Task 2 (data pipeline + models) | `python -c "from src.config import load_config; from src.train import run_task2; run_task2(load_config())"` (downloads the flowers dataset to `data/raw/` on first run; about 55 min on a 4-core CPU, 10–15 min on a Colab GPU) |
| Notebook (report figures) | open `notebooks/FCVG101_Coursework.ipynb` (stored without outputs; the rendered run is `exports/FCVG101_Coursework.html`) |
| App | `streamlit run app.py` |
| Visual demo of every step (calibration by webcam, 1B controls, Task 2 results, live prediction) | `streamlit run demo_app.py` |
| Report (.docx) | `python report/build_report.py` (writes `report/FCVG101_Coursework_Report.docx`; `--with-code` adds the source appendix; needs pandoc) |

Status: Task 1A implemented (`src/calibration.py`, `tests/test_task1a.py`; calibrated on 13 webcam views in `data/calibration/`, RMS 0.097 px; source option upload | camera, tools in `tools/task1a/`). Task 1B implemented (`src/filtering.py`, `frequency.py`, `edges.py`, `features.py`, `task1b.py`; tests in `tests/test_task1b.py`). Task 2 implemented (`src/data_pipeline.py`, `model.py`, `train.py`, `evaluate.py`, `inference.py`, `baselines.py`, `explain.py`; tests in `tests/test_task2.py`; selected model in `artifacts/model.pth`, every experiment in `artifacts/models/<name>/`). Task 3 (`assistant.py`) still raises `NotImplementedError`.
