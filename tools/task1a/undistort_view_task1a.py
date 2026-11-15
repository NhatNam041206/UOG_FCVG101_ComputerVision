"""FCVG101 Task 1A: show BEFORE | AFTER undistortion with a saved calibration.

Usage:
  # live webcam (same camera + resolution that was calibrated)
  python undistort_view_task1a.py --calib results/live/calibration.npz --camera 0
  # one photo from the same camera
  python undistort_view_task1a.py --calib results/live/calibration.npz --image my_photo.jpg

Keys: G toggle straight-line grid, A toggle crop (alpha 0/1), S save snapshot, Q quit.
Lines that are straight in the scene should follow the grid better in AFTER.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np


def load_calibration(path):
    data = np.load(path)
    return data['K'], data['dist'], (int(data['width']), int(data['height']))


def make_maps(K, dist, size, alpha):
    # alpha=1 keeps every source pixel (black curved border), alpha=0 crops to valid pixels only.
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, size, alpha, size)
    return cv2.initUndistortRectifyMap(K, dist, None, new_K, size, cv2.CV_16SC2)


def side_by_side(before, after, grid):
    both = np.concatenate([before, after], axis=1)
    h, w = before.shape[:2]
    if grid:
        for x in range(0, 2 * w, w // 8):
            cv2.line(both, (x, 0), (x, h), (0, 255, 255), 1)
        for y in range(0, h, h // 6):
            cv2.line(both, (0, y), (2 * w, y), (0, 255, 255), 1)
    cv2.line(both, (w, 0), (w, h), (255, 255, 255), 3)
    for text, x, colour in (('BEFORE', 16, (0, 0, 255)), ('AFTER', w + 16, (0, 160, 0))):
        cv2.putText(both, text, (x, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(both, text, (x, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, colour, 2, cv2.LINE_AA)
    return both


def main():
    p = argparse.ArgumentParser(description='FCVG101 - before/after undistortion viewer')
    p.add_argument('--calib', default='results/live/calibration.npz', help='calibration.npz from calibration_task1a')
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--camera', type=int, help='webcam index (same camera that was calibrated)')
    src.add_argument('--image', help='photo from the same camera and resolution')
    p.add_argument('--output', default='results/live', help='where S saves snapshots')
    args = p.parse_args()

    K, dist, size = load_calibration(args.calib)
    alpha, grid, shot = 1.0, True, 0
    maps = make_maps(K, dist, size, alpha)
    cap = None
    if args.camera is not None:
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    still = cv2.imread(args.image) if args.image else None
    if args.image and still is None:
        raise FileNotFoundError(f'Image unreadable: {args.image}')
    Path(args.output).mkdir(parents=True, exist_ok=True)

    while True:
        if cap is not None:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'Cannot read camera {args.camera}')
        else:
            frame = still
        if (frame.shape[1], frame.shape[0]) != size:
            raise ValueError(f'Frame is {frame.shape[1]}x{frame.shape[0]} but calibration is {size[0]}x{size[1]}; '
                             'use the same camera mode/resolution.')
        after = cv2.remap(frame, *maps, cv2.INTER_LINEAR)
        view = side_by_side(frame, after, grid)
        cv2.imshow('FCVG101 Task 1A - before | after', view)
        key = cv2.waitKey(1 if cap is not None else 30) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('g'):
            grid = not grid
        if key == ord('a'):
            alpha = 0.0 if alpha else 1.0
            maps = make_maps(K, dist, size, alpha)
        if key == ord('s'):
            shot += 1
            out = Path(args.output) / f'before_after_live_{shot:02d}.jpg'
            cv2.imwrite(str(out), side_by_side(frame, after, False), [cv2.IMWRITE_JPEG_QUALITY, 92])
            print('Saved', out)
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
