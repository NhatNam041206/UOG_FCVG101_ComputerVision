"""[DEMO] Interactive visual demo of every coursework step (Task 1A, 1B, 2), one page per task.

Run:  streamlit run demo_app.py
Flow: top navigation bar -> one page per task (demo/page_*.py); inside each page a segmented control
      switches between the steps. Pages call the src/ functions directly, like the notebook does
      (layer 3); no algorithm code lives here. app.py (Task 3 assistant) is separate and unchanged.
"""
import streamlit as st

st.set_page_config(page_title="FCVG101 Vision Lab", page_icon=":material/visibility:", layout="wide")

pages = [
    st.Page("demo/page_home.py", title="Overview", icon=":material/home:", default=True),
    st.Page("demo/page_calibration.py", title="1A Calibration", icon=":material/photo_camera:"),
    st.Page("demo/page_processing.py", title="1B Image processing", icon=":material/tune:"),
    st.Page("demo/page_training.py", title="2 Training results", icon=":material/monitoring:"),
    st.Page("demo/page_classify.py", title="2 Try the model", icon=":material/local_florist:"),
]
st.navigation(pages, position="top").run()
