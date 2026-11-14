# SYNTHETIC checkerboards – not real camera evidence

16 views made by the course pack's `generate_synthetic_checkerboards.py` (FCVG101_Task1A_Workshop.zip, seed 42).
Board: 10 x 7 squares = 9 x 6 inner corners, 18 mm squares. Rendered with a known camera:
K = [[900, 0, 480], [0, 890, 360], [0, 0, 1]], dist = [-0.14, 0.04, 0.001, -0.0005, -0.005], 960 x 720 px.

The notebook uses this folder only while `data/calibration/` is empty, to show that the pipeline recovers a known
camera. Never report these numbers as a real-camera calibration.
