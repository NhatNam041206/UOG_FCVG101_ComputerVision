"""FCVG101 Task 1A: enforce the "Chup anh checkerboard dung cach" capture rules.

Implements section 4 of the Task 1A practice guide as automatic checks:
  * every view shows all 9x6 = 54 inner corners, same camera, same resolution;
  * evenly lit, sharp, whole board in frame, no occlusion or glare;
  * varied poses: a few near-frontal, tilted left/right and up/down, board near
    the four frame corners, large and small (near/far);
  * 10-20 accepted views, no near-duplicate poses, no synthetic/real mixing.

Usage:
  # 1) Audit a folder of photos you already took (phone/DSLR), then calibrate
  python capture_checker_task1a.py --check data/calibration --calibrate

  # 2) Live webcam, fully automatic: the window tells you where to move/tilt the board,
  #    saves a FULL frame when it passes every rule, then calibrates when the plan is done
  python capture_checker_task1a.py --camera 0 --save-dir data/live --output results/live

  # 3) Board shown on a tablet/phone screen instead of paper
  python capture_checker_task1a.py --make-board board_screen.png
  python capture_checker_task1a.py --camera 0 --screen --square-mm 9.6

  # 4) Simulated lens with KNOWN distortion (report true vs recovered k1, k2, ...)
  python capture_checker_task1a.py --camera 0 --save-dir data/sim --output results/sim \
      --simulate-distortion=-0.3,0.08 --sim-real-calib results/live/calibration.npz

Pose angles are estimated with an approximate K (f ~ image width), so they are only
good enough to sort views into "frontal / tilted" bins, not as measurements.
"""
import argparse
import csv
import shutil
import sys
import time
from pathlib import Path
import cv2
import numpy as np

from calibration_task1a import EXTENSIONS, find_inner_corners, load_paths, run as run_calibration

# ---- pose-diversity thresholds (edit here; quality limits are command-line flags) ----
GRID = 4                  # coverage grid GRID x GRID over the image
FRONTAL_DEG = 12          # |tilt| below this counts as near-frontal
TILT_DEG = 15             # yaw/pitch beyond this counts as tilted left/right/up/down
SMALL_AREA, LARGE_AREA = .12, .25   # board area / image area for far / near views
DUP_POS, DUP_SCALE, DUP_DEG = .06, .04, 6   # closer than all three = same pose
NEED = {'frontal': 2, 'left': 1, 'right': 1, 'up': 1, 'down': 1,
        'near_TL': 1, 'near_TR': 1, 'near_BL': 1, 'near_BR': 1, 'small': 2, 'large': 2}


def board_mask(shape, corners, pattern, grow_one_square=True):
    """Polygon of the board's inner-corner area, optionally grown by one square on each side."""
    grid = corners.reshape(pattern[1], pattern[0], 2)
    quad = np.float32([grid[0, 0], grid[0, -1], grid[-1, -1], grid[-1, 0]])
    centre = quad.mean(axis=0)
    grow = np.float32([(pattern[0] + 1) / (pattern[0] - 1), (pattern[1] + 1) / (pattern[1] - 1)]).max()
    outer = centre + (quad - centre) * grow if grow_one_square else quad
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, outer.astype(np.int32), 255)
    return mask, outer


def square_px(corners, pattern):
    grid = corners.reshape(pattern[1], pattern[0], 2)
    dx = np.linalg.norm(np.diff(grid, axis=1), axis=2).mean()
    dy = np.linalg.norm(np.diff(grid, axis=0), axis=2).mean()
    return float((dx + dy) / 2)


def estimate_pose(corners, world, size):
    """Board yaw/pitch/tilt in degrees from solvePnP with a rough pinhole K."""
    w, h = size
    K = np.float64([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]])
    ok, rvec, _ = cv2.solvePnP(world, corners, K, None)
    if not ok:
        return 0.0, 0.0, 0.0
    n = cv2.Rodrigues(rvec)[0][:, 2]         # board normal in camera frame
    if n[2] < 0:
        n = -n
    yaw = np.degrees(np.arctan2(n[0], n[2]))     # + = board turned right
    pitch = np.degrees(np.arctan2(n[1], n[2]))   # + = board tilted down
    tilt = np.degrees(np.arccos(np.clip(n[2], -1, 1)))
    return float(yaw), float(pitch), float(tilt)


def assess(bgr, pattern, world, args):
    """Run every per-image rule. Returns (ok, reasons, info dict)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    ok, corners, detector = find_inner_corners(gray, pattern)
    if not ok:
        return False, ['board not found / not all inner corners visible (occluded or wrong --cols/--rows)'], {}
    reasons, info = check_board(gray, corners, pattern, world, args)
    info['detector'] = detector
    return not reasons, reasons, info


def check_board(gray, corners, pattern, world, args):
    """Quality rules + pose descriptors for an already detected corner grid."""
    h, w = gray.shape
    info = {'corners': corners}
    reasons = []

    # Whole board in frame: outer squares must not touch the image border.
    sq = square_px(corners, pattern)
    pts = corners.reshape(-1, 2)
    margin = min(pts[:, 0].min(), pts[:, 1].min(), w - 1 - pts[:, 0].max(), h - 1 - pts[:, 1].max())
    if margin < 1.2 * sq:
        reasons.append(f'board too close to frame edge (margin {margin:.0f}px < {1.2 * sq:.0f}px)')

    # Sharpness (edge width in px, lower is sharper) and contrast.
    if hasattr(cv2, 'estimateChessboardSharpness'):
        stats, per_corner = cv2.estimateChessboardSharpness(gray, pattern, corners)
        sharp, dark, bright = stats[0], stats[1], stats[2]
        bright_each = np.asarray(per_corner)[:, 4]
    else:
        mask, _ = board_mask(gray.shape, corners, pattern)
        lap = cv2.Laplacian(gray, cv2.CV_64F)[mask > 0].var()
        sharp = 2.0 if lap > 100 else 5.0
        vals = gray[mask > 0]
        dark, bright = np.percentile(vals, 10), np.percentile(vals, 90)
        bright_each = np.array([bright])
    info.update(sharpness_px=float(sharp), contrast=float(bright - dark))
    if sharp > args.max_blur:
        reasons.append(f'blurry: edge width {sharp:.2f}px > {args.max_blur}px (hold still / refocus)')
    if bright - dark < args.min_contrast:
        reasons.append(f'low contrast {bright - dark:.0f} < {args.min_contrast} (add more light)')

    # Even lighting: white-square brightness should be similar across the board.
    unevenness = float(np.std(bright_each) / max(np.mean(bright_each), 1))
    info['unevenness'] = unevenness
    if unevenness > args.max_uneven:
        reasons.append(f'uneven lighting / shadow {unevenness:.2f} > {args.max_uneven}')

    # Glare: blown-out pixels inside the grid (reflections on glossy paper/screen, overexposure).
    inner, _ = board_mask(gray.shape, corners, pattern, grow_one_square=False)
    glare = float((gray[inner > 0] >= 254).mean())
    info['glare'] = glare
    if glare > args.max_glare:
        reasons.append(f'glare/overexposed: {glare * 100:.1f}% clipped pixels on board > {args.max_glare * 100:.1f}%')

    # Pose descriptors for the diversity rules.
    _, outer = board_mask(gray.shape, corners, pattern)
    area = cv2.contourArea(outer.astype(np.float32)) / (w * h)
    yaw, pitch, tilt = estimate_pose(corners, world, (w, h))
    cx, cy = pts.mean(axis=0) / (w, h)
    cells = {(min(int(x / w * GRID), GRID - 1), min(int(y / h * GRID), GRID - 1)) for x, y in pts}
    info.update(area=float(area), yaw=yaw, pitch=pitch, tilt=tilt, cx=float(cx), cy=float(cy), cells=cells)
    return reasons, info


def tags_of(info):
    t = set()
    if info['tilt'] < FRONTAL_DEG: t.add('frontal')
    if info['yaw'] < -TILT_DEG: t.add('left')
    if info['yaw'] > TILT_DEG: t.add('right')
    if info['pitch'] < -TILT_DEG: t.add('up')
    if info['pitch'] > TILT_DEG: t.add('down')
    if info['area'] < SMALL_AREA: t.add('small')
    if info['area'] > LARGE_AREA: t.add('large')
    corner_cells = {(0, 0): 'near_TL', (GRID - 1, 0): 'near_TR', (0, GRID - 1): 'near_BL', (GRID - 1, GRID - 1): 'near_BR'}
    t.update(name for cell, name in corner_cells.items() if cell in info['cells'])
    return t


def duplicate_of(info, accepted):
    for name, other in accepted:
        if (np.hypot(info['cx'] - other['cx'], info['cy'] - other['cy']) < DUP_POS
                and abs(info['area'] - other['area']) < DUP_SCALE
                and abs(info['yaw'] - other['yaw']) < DUP_DEG and abs(info['pitch'] - other['pitch']) < DUP_DEG):
            return name
    return None


class Session:
    """Accumulates accepted views and tracks the diversity checklist."""

    def __init__(self, args):
        self.args = args
        self.pattern = (args.cols, args.rows)
        self.world = np.zeros((args.cols * args.rows, 3), np.float32)
        self.world[:, :2] = np.mgrid[:args.cols, :args.rows].T.reshape(-1, 2) * args.square_mm / 1000
        self.accepted, self.records = [], []
        self.size, self.camera_id = None, None

    def consider(self, name, bgr, exif_id=None):
        size = (bgr.shape[1], bgr.shape[0])
        if self.size is None:
            self.size, self.camera_id = size, exif_id
        if size != self.size:
            return self.log(name, False, [f'resolution {size} differs from {self.size}: same camera/mode only'], {})
        if exif_id and self.camera_id and exif_id != self.camera_id:
            return self.log(name, False, [f'camera/lens/zoom changed: {exif_id} vs {self.camera_id}'], {})
        if len(self.accepted) >= self.args.max_views:
            return self.log(name, False, [f'already have {self.args.max_views} views'], {})
        ok, reasons, info = assess(bgr, self.pattern, self.world, self.args)
        if ok:
            dup = duplicate_of(info, self.accepted)
            if dup:
                ok, reasons = False, [f'same pose as {dup}: move/tilt the board, do not repeat one angle']
        if ok:
            self.accepted.append((name, info))
        return self.log(name, ok, reasons, info)

    def log(self, name, ok, reasons, info):
        tags = ' '.join(sorted(tags_of(info))) if 'tilt' in info else ''
        pose = f"yaw={info['yaw']:+.0f} pitch={info['pitch']:+.0f} area={info['area']:.2f}" if 'tilt' in info else ''
        self.records.append((name, 'ACCEPTED' if ok else 'REJECTED', pose, tags, '; '.join(reasons)))
        print(f"[{'OK' if ok else 'REJECT'}] {name} {pose} {('[' + tags + ']') if tags else ''}")
        for r in reasons:
            print('       -', r)
        return ok, reasons, info

    def missing(self):
        have = {k: 0 for k in NEED}
        cells = set()
        for _, info in self.accepted:
            for t in tags_of(info):
                have[t] += 1
            cells |= info['cells']
        todo = [f'{k} x{NEED[k] - have[k]}' for k in NEED if have[k] < NEED[k]]
        coverage = len(cells) / GRID ** 2
        if coverage < self.args.min_coverage:
            todo.append(f'coverage {coverage * 100:.0f}% < {self.args.min_coverage * 100:.0f}% (fill empty areas)')
        if len(self.accepted) < self.args.min_views:
            todo.append(f'views {len(self.accepted)}/{self.args.min_views}')
        return todo, coverage, cells

    def summary(self):
        todo, coverage, _ = self.missing()
        print(f'\nAccepted views: {len(self.accepted)} (target {self.args.min_views}-{self.args.max_views}), '
              f'frame coverage {coverage * 100:.0f}%')
        print('Capture plan complete.' if not todo else 'Still needed: ' + ', '.join(todo))
        print('Manual check: keep at least one view with straight lines near the image edge '
              '(to judge undistortion visually).')
        return not todo


def exif_camera(path):
    """Make/model/focal length from EXIF, if Pillow is available; None otherwise."""
    try:
        from PIL import Image
        exif = Image.open(path).getexif()
        sub = exif.get_ifd(0x8769)
        fields = [exif.get(0x010F), exif.get(0x0110), sub.get(0x920A), sub.get(0xA404)]  # make, model, focal, zoom
        return ' / '.join(str(f) for f in fields if f is not None) or None
    except Exception:
        return None


def check_folder(session, folder):
    paths = load_paths(folder)
    if not paths:
        raise RuntimeError(f'No images in {folder}')
    synthetic = (folder / 'README_SYNTHETIC.txt').exists() or any('synthetic' in p.name.lower() for p in paths)
    if synthetic and not session.args.allow_synthetic:
        raise RuntimeError('Folder contains synthetic views. Never mix synthetic and real photos; '
                           'use a clean folder of real photos (or --allow-synthetic for a code test).')
    for p in paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            session.log(p.name, False, ['unreadable file'], {})
            continue
        session.consider(p.name, bgr, exif_camera(p))
    return {p.name: p for p in paths}


# ---- guided live capture ------------------------------------------------------
# Shot plan in SCREEN coordinates (what you see in the window). pos = target board
# centre as a fraction of the frame; corner = frame cell the board must reach;
# area / tilt / yaw / pitch = allowed ranges. Together they cover the guide's table.
T = TILT_DEG + 3
SHOTS = [
    dict(name='Centre, facing the camera', pos=(.5, .5), area=(SMALL_AREA, LARGE_AREA), tilt=(0, FRONTAL_DEG)),
    dict(name='Close-up, facing the camera', pos=(.5, .5), area=(LARGE_AREA, .8), tilt=(0, FRONTAL_DEG)),
    dict(name='Far away, facing the camera', pos=(.5, .5), area=(0, SMALL_AREA), tilt=(0, FRONTAL_DEG)),
    dict(name='Turned: LEFT edge toward camera', pos=(.5, .5), area=(.05, .8), yaw=(-70, -T)),
    dict(name='Turned: RIGHT edge toward camera', pos=(.5, .5), area=(.05, .8), yaw=(T, 70)),
    dict(name='Tilted: TOP edge toward camera', pos=(.5, .5), area=(.05, .8), pitch=(-70, -T)),
    dict(name='Tilted: BOTTOM edge toward camera', pos=(.5, .5), area=(.05, .8), pitch=(T, 70)),
    dict(name='Top-left corner of the frame', pos=(.25, .25), corner=(0, 0), area=(0, .3)),
    dict(name='Top-right corner of the frame', pos=(.75, .25), corner=(GRID - 1, 0), area=(0, .3)),
    dict(name='Bottom-left corner of the frame', pos=(.25, .75), corner=(0, GRID - 1), area=(0, .3)),
    dict(name='Bottom-right corner of the frame', pos=(.75, .75), corner=(GRID - 1, GRID - 1), area=(0, .3)),
    dict(name='Close-up AND tilted (any way)', pos=(.5, .5), area=(LARGE_AREA * .8, .8), tilt=(T, 70)),
    dict(name='Far away AND tilted (any way)', pos=(.5, .5), area=(0, SMALL_AREA), tilt=(T, 70)),
]
POS_TOL = .12     # board centre within this distance of the target (fraction of frame)
HOLD_S = .8       # board must be steady this long before the automatic capture
PANEL_W, PANEL_MIN_H = 460, 640
FONT = cv2.FONT_HERSHEY_SIMPLEX
QUALITY_HINTS = [('blurry', 'Blurry: hold still / let it focus'), ('glare', 'Glare: tilt board away from the light'),
                 ('contrast', 'Low contrast: add more light'), ('uneven', 'Shadow on board: light it evenly'),
                 ('edge', 'Too close to frame edge: move inward')]


def quick_detect(gray, pattern, width=640):
    """Fast per-frame detection on a downscaled frame; corners returned in full-size pixels."""
    scale = min(1.0, width / gray.shape[1])
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    ok, corners = cv2.findChessboardCorners(small, pattern, flags)
    if not ok:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 20, .01)
    return cv2.cornerSubPix(gray, (corners / scale).astype(np.float32), (5, 5), (-1, -1), criteria)


def guidance(shot, info, reasons):
    """Checks for the current shot as (ok, text); a failing text says what to do."""
    if 'area' not in info:
        return [(False, 'Show the WHOLE board to the camera')]
    hint = next((h for key, h in QUALITY_HINTS for r in reasons if key in r), reasons[0] if reasons else '')
    checks = [(True, 'Whole board visible'), (not reasons, hint or 'Sharp, no glare, even light')]

    dx, dy = shot['pos'][0] - info['cx'], shot['pos'][1] - info['cy']
    if 'corner' in shot:
        ok = shot['corner'] in info['cells'] and abs(dx) < .25 and abs(dy) < .25
        words = ['LEFT' if shot['corner'][0] == 0 else 'RIGHT', 'UP' if shot['corner'][1] == 0 else 'DOWN']
    else:
        ok = abs(dx) < POS_TOL and abs(dy) < POS_TOL
        words = ([('RIGHT' if dx > 0 else 'LEFT')] if abs(dx) >= POS_TOL / 2 else []) + \
                ([('DOWN' if dy > 0 else 'UP')] if abs(dy) >= POS_TOL / 2 else [])
    checks.append((ok, 'Position OK' if ok else 'Move board ' + ' + '.join(words) + ' (follow the arrow)'))

    lo, hi = shot['area']
    a = info['area']
    checks.append((lo <= a <= hi, 'Distance OK' if lo <= a <= hi else
                   'Move board CLOSER to camera' if a < lo else 'Move board FARTHER from camera'))

    angle = (True, 'Angle OK')
    if 'tilt' in shot and not shot['tilt'][0] <= info['tilt'] <= shot['tilt'][1]:
        angle = (False, 'Face the board straight at the camera' if info['tilt'] > shot['tilt'][1]
                 else 'Tilt the board more (about 30 deg, any way)')
    if 'yaw' in shot and not shot['yaw'][0] <= info['yaw'] <= shot['yaw'][1]:
        more_right = info['yaw'] < shot['yaw'][0]   # yaw > 0: right edge closer to the camera
        angle = (False, f"Turn: bring {'RIGHT' if more_right else 'LEFT'} edge toward camera")
    if 'pitch' in shot and not shot['pitch'][0] <= info['pitch'] <= shot['pitch'][1]:
        more_bottom = info['pitch'] < shot['pitch'][0]   # pitch > 0: bottom edge closer
        angle = (False, f"Tilt: bring {'BOTTOM' if more_bottom else 'TOP'} edge toward camera")
    checks.append(angle)
    return checks


def put(img, text, y, scale=.55, colour=(225, 225, 225), thick=1, x=16):
    cv2.putText(img, text, (x, y), FONT, scale, colour, thick, cv2.LINE_AA)


def wrap(text, width):
    import textwrap
    return textwrap.wrap(text, width) or ['']


def render(view, pattern, idx, status, checks, progress, info, message):
    """Camera view with target + board outline only; all text goes on a side panel."""
    vis = view.copy()
    h, w = vis.shape[:2]
    shot = SHOTS[idx]
    yellow, green, red, grey = (0, 220, 255), (80, 220, 80), (80, 80, 255), (140, 140, 140)
    tx, ty = int(shot['pos'][0] * w), int(shot['pos'][1] * h)
    if 'corner' in shot:
        gx, gy = shot['corner']
        cv2.rectangle(vis, (gx * w // GRID + 3, gy * h // GRID + 3), ((gx + 1) * w // GRID - 3, (gy + 1) * h // GRID - 3),
                      yellow, 3)
    else:
        cv2.circle(vis, (tx, ty), int(POS_TOL * w), yellow, 2)
    cv2.drawMarker(vis, (tx, ty), yellow, cv2.MARKER_CROSS, 24, 2)
    if 'corners' in info:
        _, outer = board_mask(vis.shape, info['corners'], pattern)
        all_ok = all(ok for ok, _ in checks)
        cv2.polylines(vis, [outer.astype(np.int32)], True, green if all_ok else (0, 140, 255), 3)
        if not checks[2][0]:
            bx, by = int(info['cx'] * w), int(info['cy'] * h)
            cv2.arrowedLine(vis, (bx, by), (tx, ty), yellow, 3, tipLength=.08)
    if message.startswith('Captured'):
        cv2.rectangle(vis, (0, 0), (w - 1, h - 1), (255, 255, 255), 12)

    ph = max(h, PANEL_MIN_H)
    panel = np.full((ph, PANEL_W, 3), 35, np.uint8)
    put(panel, 'FCVG101 - calibration capture', 30, .62, (255, 255, 255), 2)
    put(panel, f'Shot {idx + 1}/{len(SHOTS)}:', 62, .55, grey)
    y = 62
    for line in wrap(SHOTS[idx]['name'], 30):
        put(panel, line, y, .6, yellow, 2, x=110); y += 26
    y += 10
    todo = next((text for ok, text in checks if not ok), None)
    put(panel, 'DO NOW:', y, .5, grey); y += 28
    for line in wrap(todo or 'HOLD STILL...', 26):
        put(panel, line, y, .75, red if todo else green, 2); y += 30
    y += 4
    cv2.rectangle(panel, (16, y), (PANEL_W - 16, y + 14), (80, 80, 80), 1)
    cv2.rectangle(panel, (16, y), (16 + int((PANEL_W - 32) * min(progress, 1)), y + 14), green, -1)
    y += 38
    for ok, text in checks:
        put(panel, ('[OK] ' if ok else '[  ] ') + text[:42], y, .48, green if ok else red); y += 22
    y += 6
    for line in wrap(message, 46):
        put(panel, line, y, .48, (255, 255, 255)); y += 20
    y += 10
    put(panel, 'Plan:', y, .5, grey); y += 22
    marks = {'done': ('+', green), 'skip': ('-', grey), 'todo': (' ', (170, 170, 170))}
    for i, s in enumerate(SHOTS):
        mark, colour = ('>', yellow) if i == idx else marks[status[i]]
        put(panel, f'{mark} {i + 1:2d}. {s["name"]}', y, .45, colour); y += 19
    put(panel, 'Q quit    U undo last    S skip shot', ph - 14, .5, grey)
    if ph > h:
        vis = np.vstack([vis, np.zeros((ph - h, w, 3), np.uint8)])
    return np.hstack([vis, panel])


class LensSimulator:
    """Fake lens with KNOWN distortion, applied to every frame with cv2.remap.

    For each pixel of the simulated (distorted) frame we find where it comes from in a clean
    pinhole image: undistort it with (K_sim, dist_sim), then project into the source camera.
    With real_calib the webcam's own lens is removed in the same remap, so the simulated
    camera's true K and dist are exactly K_sim and dist_sim (ground truth for the report).
    """

    def __init__(self, spec, real_calib=None, focal=None):
        vals = [float(v) for v in spec.split(',')]
        if not 1 <= len(vals) <= 5:
            raise ValueError('--simulate-distortion takes 1-5 values: k1[,k2[,p1[,p2[,k3]]]]')
        self.dist = np.zeros(5)
        self.dist[:len(vals)] = vals
        self.real_calib, self.focal, self.maps, self.K = real_calib, focal, None, None

    def _build(self, size):
        w, h = size
        K_src, d_src = None, None
        if self.real_calib:
            data = np.load(self.real_calib)
            if (int(data['width']), int(data['height'])) != size:
                raise ValueError(f'{self.real_calib} was calibrated at another resolution than {w}x{h}')
            K_src, d_src = data['K'], data['dist']
        f = self.focal or (K_src[0, 0] if K_src is not None else float(w))
        self.K = np.float64([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
        if K_src is None:
            K_src, d_src = self.K, np.zeros(5)
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        pts = np.stack([xs.ravel(), ys.ravel()], 1).reshape(-1, 1, 2)
        crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-9)
        if hasattr(cv2, 'undistortPointsIter'):          # OpenCV 4.x
            norm = cv2.undistortPointsIter(pts, self.K, self.dist, None, None, crit)
        else:                                            # OpenCV 5.x
            norm = cv2.undistortPoints(pts, self.K, self.dist, criteria=crit)
        rays = np.concatenate([norm.reshape(-1, 2), np.ones((len(pts), 1))], 1)
        src, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), K_src, d_src)
        src = src.reshape(h, w, 2).astype(np.float32)
        self.maps = (src[..., 0], src[..., 1])

    def __call__(self, frame):
        if self.maps is None:
            self._build((frame.shape[1], frame.shape[0]))
        return cv2.remap(frame, *self.maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def report(self, K, dist, output):
        """True vs recovered parameters of the simulated lens."""
        d = np.zeros(5)
        d[:min(5, dist.size)] = dist.ravel()[:5]
        rows = [('fx', self.K[0, 0], K[0, 0]), ('fy', self.K[1, 1], K[1, 1]),
                ('cx', self.K[0, 2], K[0, 2]), ('cy', self.K[1, 2], K[1, 2])]
        rows += [(n, t, e) for n, t, e in zip(('k1', 'k2', 'p1', 'p2', 'k3'), self.dist, d)]
        lines = ['Simulated lens: true vs recovered',
                 'real lens removed: '
                 + ('yes' if self.real_calib else 'NO (webcam lens adds its own distortion)'),
                 f"{'param':<6}{'true':>12}{'recovered':>12}{'error':>12}"]
        lines += [f'{n:<6}{t:>12.5f}{e:>12.5f}{e - t:>12.5f}' for n, t, e in rows]
        text = '\n'.join(lines)
        print('\n' + text)
        (Path(output) / 'simulated_lens_truth_vs_estimate.txt').write_text(text + '\n', encoding='utf-8')


def add_simulation_args(p):
    p.add_argument('--simulate-distortion', metavar='K1,K2,P1,P2,K3',
                   help='apply a fake lens with these known coefficients to every frame; write it with = '
                   'because of the minus sign, e.g. --simulate-distortion=-0.3,0.08')
    p.add_argument('--sim-real-calib', help='calibration.npz of the real camera: remove its lens first')
    p.add_argument('--sim-focal', type=float, help='focal length in px of the simulated lens '
                   '(default: real fx, or image width)')


def simulator_from(args):
    if not getattr(args, 'simulate_distortion', None):
        return None
    return LensSimulator(args.simulate_distortion, args.sim_real_calib, args.sim_focal)


def live_capture(session, camera, save_dir, mirror=True, simulate=None):
    """Guided capture: tells you where to move the board, captures automatically when the
    frame passes the quick blur/glare/pose checks AND the full check. Returns (files, finished)."""
    save_dir.mkdir(parents=True, exist_ok=True)
    if any(p.suffix.lower() in EXTENSIONS for p in save_dir.iterdir()):
        raise RuntimeError(f'{save_dir} already has images; use an empty folder so sessions are not mixed.')
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open camera {camera}')
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)   # rule 1: fix the camera mode (ignored by some drivers)
    print('Fix zoom/lens/resolution now; do not change them during the session.')
    status = ['todo'] * len(SHOTS)
    taken, saved = [], {}                # taken: (shot index, file name), newest last
    still_since, last, message, msg_until = None, None, '', 0
    finished = False
    while True:
        grabbed, raw = cap.read()
        if not grabbed:
            break
        if simulate is not None:
            raw = simulate(raw)              # the simulated lens IS the camera from here on
        idx = next((i for i, s in enumerate(status) if s == 'todo'), None)
        if idx is None:
            finished = True
            break
        view = cv2.flip(raw, 1) if mirror else raw   # mirrored preview feels natural; raw frame is saved
        gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
        corners = quick_detect(gray, session.pattern)
        reasons, info = check_board(gray, corners, session.pattern, session.world, session.args) \
            if corners is not None else ([], {})
        checks = guidance(SHOTS[idx], info, reasons)
        ready = all(ok for ok, _ in checks)

        # Prefilter passed -> wait until the board is steady, then run the full check on the raw frame.
        now = time.time()
        steady = (last is not None and 'cx' in info
                  and np.hypot(info['cx'] - last[0], info['cy'] - last[1]) < .004
                  and abs(info['area'] - last[2]) < .004)
        last = (info['cx'], info['cy'], info['area']) if 'cx' in info else None
        still_since = (still_since or now) if (ready and steady) else None
        progress = (now - still_since) / HOLD_S if still_since else 0
        if progress >= 1:
            name = f'view_{len(saved) + 1:02d}.png'
            accepted, why, _ = session.consider(name, raw)
            if accepted:
                cv2.imwrite(str(save_dir / name), raw)
                saved[name] = save_dir / name
                taken.append((idx, name))
                status[idx] = 'done'
                message = f'Captured {name}! Next shot...'
            else:
                message = 'Full check rejected it: ' + why[0]
            msg_until, still_since = now + 1.5, None
        elif now > msg_until:
            message = ''

        cv2.imshow('FCVG101 Task 1A capture', render(view, session.pattern, idx, status, checks,
                                                     progress, info, message))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('s'):
            status[idx] = 'skip'
            print('[SKIP]', SHOTS[idx]['name'])
        if key == ord('u') and taken:
            shot_idx, name = taken.pop()
            status[shot_idx] = 'todo'
            session.accepted = [a for a in session.accepted if a[0] != name]
            saved.pop(name).unlink(missing_ok=True)
            message, msg_until = f'Removed {name}', now + 1.5
            print('[UNDO] removed', name)
    cap.release()
    cv2.destroyAllWindows()
    skipped = [s['name'] for s, st in zip(SHOTS, status) if st == 'skip']
    if skipped:
        print('Skipped shots (calibration may be less reliable):', ', '.join(skipped))
    return saved, finished


def sanity_after_calibration(K, dist, size, output):
    """Plausibility checks from the guide's troubleshooting table."""
    w, h = size
    warn = []
    if abs(K[0, 0] / K[1, 1] - 1) > .05:
        warn.append(f'fx/fy = {K[0, 0] / K[1, 1]:.3f} far from 1')
    if abs(K[0, 2] - w / 2) > .15 * w or abs(K[1, 2] - h / 2) > .15 * h:
        warn.append(f'principal point ({K[0, 2]:.0f}, {K[1, 2]:.0f}) far from image centre')
    if np.abs(dist.ravel()[:2]).max() > 1 or (dist.size > 4 and abs(dist.ravel()[4]) > 1):
        warn.append(f'large distortion coefficients {np.round(dist.ravel(), 3)} (likely overfit: add tilted/corner views)')
    errors = np.genfromtxt(output / 'view_errors.csv', delimiter=',', skip_header=1, usecols=1, ndmin=1)
    names = np.genfromtxt(output / 'view_errors.csv', delimiter=',', skip_header=1, usecols=0, dtype=str, ndmin=1)
    bad = [n for n, e in zip(names, errors) if e > 2 * np.median(errors)]
    if bad:
        warn.append('views with error > 2x median (re-check/replace): ' + ', '.join(bad))
    print('\nSanity checks:', 'all passed' if not warn else '')
    for w_ in warn:
        print('  [WARN]', w_)


def make_board(path, cols, rows, px=120):
    """Board image to show full-screen on a tablet/phone: (cols+1) x (rows+1) squares + white border."""
    board = np.full(((rows + 3) * px, (cols + 3) * px), 255, np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                board[(r + 1) * px:(r + 2) * px, (c + 1) * px:(c + 2) * px] = 0
    cv2.imwrite(str(path), board)
    print('Saved', path, '- open it full-screen (no zoom/fit), then MEASURE one square with a ruler '
          'and pass that value as --square-mm.')


def main():
    p = argparse.ArgumentParser(description='FCVG101 - enforce correct checkerboard capture, then calibrate')
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--check', help='audit an existing folder of photos from ONE camera')
    src.add_argument('--camera', type=int, help='webcam index for live guided capture')
    src.add_argument('--make-board', help='write a board PNG for display on a tablet/phone screen')
    p.add_argument('--save-dir', default='data/calibration', help='where live captures are saved (must be empty)')
    p.add_argument('--output', default='results/real', help='calibration output folder')
    p.add_argument('--cols', type=int, default=9, help='number of INNER corners horizontally')
    p.add_argument('--rows', type=int, default=6, help='number of INNER corners vertically')
    p.add_argument('--square-mm', type=float, default=None, help='measured square size in mm (default 18, printed A4)')
    p.add_argument('--screen', action='store_true', help='board is shown on a tablet/phone screen')
    p.add_argument('--min-views', type=int, default=10)
    p.add_argument('--max-views', type=int, default=20)
    p.add_argument('--min-coverage', type=float, default=.75, help='fraction of the 4x4 frame grid touched by corners')
    p.add_argument('--max-blur', type=float, default=3.0, help='max chessboard edge width in px')
    p.add_argument('--min-contrast', type=float, default=80)
    p.add_argument('--max-uneven', type=float, default=.12)
    p.add_argument('--max-glare', type=float, default=.02)
    p.add_argument('--no-mirror', dest='mirror', action='store_false', help='live mode: do not mirror the preview')
    p.add_argument('--calibrate', action='store_true', help='folder mode: run calibration_task1a on the accepted views '
                   '(live mode always calibrates when the shot plan is finished)')
    add_simulation_args(p)
    p.add_argument('--force', action='store_true', help='calibrate even if the capture plan is incomplete')
    p.add_argument('--allow-synthetic', action='store_true', help='code test only: accept synthetic views')
    args = p.parse_args()
    if args.cols < 3 or args.rows < 3:
        p.error('cols/rows must be >= 3')
    if args.make_board:
        make_board(Path(args.make_board), args.cols, args.rows)
        return
    if args.screen:
        if args.square_mm is None:
            p.error('--screen needs --square-mm measured with a ruler on the screen itself')
        args.max_glare = min(args.max_glare, .005)   # screens reflect lamps/windows easily
        print('Screen mode: brightness max and fixed (auto-brightness/Night Shift/True Tone off), '
              'screen lock off, no zoom, tilt the screen away from lamps and windows.')
    args.square_mm = args.square_mm or 18.0

    session = Session(args)
    simulate = None
    if args.check:
        files = check_folder(session, Path(args.check))
        complete = session.summary()
    else:
        # Live mode follows its own shot plan and calibrates as soon as the plan is finished.
        simulate = simulator_from(args)
        files, finished = live_capture(session, args.camera, Path(args.save_dir), mirror=args.mirror,
                                       simulate=simulate)
        print(f"\nLive session {'finished' if finished else 'stopped'}: {len(session.accepted)} views saved "
              f"to {args.save_dir} (full, uncropped frames).")
        complete = finished and len(session.accepted) >= args.min_views
        args.calibrate = args.calibrate or finished

    Path(args.output).mkdir(parents=True, exist_ok=True)
    with Path(args.output, 'capture_log.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['file_name', 'status', 'pose', 'tags', 'reasons'])
        w.writerows(session.records)

    if not args.calibrate:
        return
    if not complete and not args.force:
        sys.exit('Capture plan incomplete - take the missing views (or pass --force).')
    accepted_dir = Path(args.output) / 'accepted_views'
    shutil.rmtree(accepted_dir, ignore_errors=True)
    accepted_dir.mkdir(parents=True)
    for name, _ in session.accepted:
        shutil.copy2(files[name], accepted_dir / name)
    calib_args = argparse.Namespace(input=str(accepted_dir), output=args.output, sample=None,
                                    cols=args.cols, rows=args.rows, square_mm=args.square_mm)
    K, dist, _ = run_calibration(calib_args)
    sanity_after_calibration(K, dist, session.size, Path(args.output))
    if simulate is not None:
        simulate.report(K, dist, args.output)


if __name__ == '__main__':
    main()
