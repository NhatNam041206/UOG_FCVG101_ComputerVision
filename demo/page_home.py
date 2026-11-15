"""[DEMO] Overview page: what each page shows and the headline numbers from results/."""
import streamlit as st

from demo.common import get_config, load_json

cfg = get_config()
results = cfg.paths.results

st.title("FCVG101 Vision Lab")
st.write("An interactive tour of the coursework. Pick a task in the bar at the top; inside each page the "
         "buttons under the title switch between its steps. Every control re-runs the real `src/` code.")

task1a = load_json(results / "task1a" / "metrics.json")
task2 = load_json(results / "task2" / "metrics.json")
task1b = load_json(results / "task1b" / "metrics.json")

col_1, col_2, col_3, col_4 = st.columns(4)
with col_1, st.container(border=True):
    st.metric("1A reprojection RMS", f"{task1a['rms_px']:.3f} px" if task1a else "not run",
              help="results/task1a/metrics.json")
with col_2, st.container(border=True):
    st.metric("1A views used", f"{task1a['accepted']} / {task1a['total']}" if task1a else "not run")
with col_3, st.container(border=True):
    if task2:
        best = next(e for e in task2["experiments"] if e["experiment"] == task2["selected"])
        st.metric("2 selected model, test macro F1", f"{best['test_macro_f1']:.3f}", help=task2["selected"])
    else:
        st.metric("2 selected model", "not run")
with col_4, st.container(border=True):
    st.metric("2 dataset (clean images)", f"{task2['clean_images']:,}" if task2 else "not run",
              help=task2["dataset"] if task2 else None)

st.divider()
cards = [
    (":material/photo_camera: 1A Calibration",
     "Take checkerboard photos with your webcam (or upload / use the project photos), watch the corners being "
     "found, calibrate, inspect per-view errors and residuals, then undistort and save the result."),
    (":material/tune: 1B Image processing",
     "Add noise and compare box / Gaussian / median filters with PSNR, filter in the FFT domain, tune Sobel and "
     "Canny, find Harris corners and match ORB / SIFT features with RANSAC."),
    (":material/monitoring: 2 Training results",
     "Dataset cleaning and split, interactive learning curves, hyper-parameter tuning, model comparison, "
     "confusion matrices, classical baselines and explainability results."),
    (":material/local_florist: 2 Try the model",
     "Classify a flower from the test split, an upload or your camera with any trained model and see the "
     "Grad-CAM heatmap of where it looked."),
]
columns = st.columns(2)
for index, (title, text) in enumerate(cards):
    with columns[index % 2], st.container(border=True):
        st.markdown(f"**{title}**")
        st.write(text)

if task1b is None:
    st.info("Task 1B results are missing; the 1B page still works on the sample images.")
