# Task 1B test images – sources

All four are real photographs from OpenCV's official test-data repository **opencv_extra**
(https://github.com/opencv/opencv_extra/tree/4.x/testdata/stitching), which OpenCV uses for its image-stitching
tests. Downloaded 2026-11-04. The repository does not state a separate licence for these files; cite them as
OpenCV test data and do not redistribute them outside the coursework.

| File here | Original file | Change | Used for |
| --- | --- | --- | --- |
| `clean.jpg` | `testdata/stitching/boat1.jpg` (3888 x 2592) | resized to 1600 px wide, JPEG q95 | [1B.1]–[1B.4], [1B.5h]: noise, filters, FFT, edges, Harris |
| `pair_1.jpg` | `testdata/stitching/s1.jpg` (1246 x 700) | none | [1B.5] same scene, view 1 (Pont du Gard) |
| `pair_2.jpg` | `testdata/stitching/s2.jpg` (1385 x 700) | none | [1B.5] same scene, view 2 (camera turned) |
| `unrelated.jpg` | `testdata/stitching/a1.png` (600 x 768) | saved as JPEG q95 | [1B.5] unrelated scene (cathedral interior), negative control |

Why these:
- `clean.jpg` is a sharp, low-noise outdoor photo with both smooth regions (sky, water) and fine detail (ship
  rigging, buildings, trees), so filter differences and edge maps are easy to see.
- `pair_1`/`pair_2` were taken from one position while the camera rotated (they are a panorama pair). Two views
  related by pure rotation are linked by a homography, so the RANSAC homography check [1B.5c] is valid.
- `unrelated.jpg` shares no content with the pair, so any matches it gets show the false-match rate.

Suggested report citation: OpenCV contributors, "opencv_extra test data", https://github.com/opencv/opencv_extra.
