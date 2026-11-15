"""[DEMO 1A] Camera calibration page: capture -> calibrate -> inspect -> undistort -> save.

Flow:  webcam / upload / project photos -> [1A.2] detect_corners (each photo, live)
       -> [1A.3] calibrate_camera (baseline and fix_k3) -> [1A.4] compute_reprojection_errors
       -> [1A.5] undistort_image -> [1A.6] save_calibration
"""
import io
import time

import altair as alt
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from demo.common import IMAGE_TYPES, colour_map, get_config, show, step_header
from src.calibration import (calibrate_camera, collect_calibration_images, compute_reprojection_errors,
                             detect_corners, load_calibration, load_calibration_images, make_board_points,
                             save_calibration, undistort_image, view_undistorted)
from src.io_utils import decode_image_bytes, list_images, read_image, save_image

cfg = get_config()
settings = cfg.task1a
state = st.session_state
state.setdefault("calib_views", [])        # list of dicts: name, bgr, corners, detector
state.setdefault("calib_last_shot", None)
state.setdefault("calib_message", None)

st.title("Task 1A · Camera calibration")

with st.expander("Board settings", icon=":material/grid_on:"):
    col_1, col_2, col_3 = st.columns(3)
    pattern_cols = col_1.number_input("Inner corners across", 3, 20, settings.pattern_cols)
    pattern_rows = col_2.number_input("Inner corners down", 3, 20, settings.pattern_rows)
    square_mm = col_3.number_input("Square size (mm)", 1.0, 100.0, float(settings.square_size_mm), step=0.1)
    st.caption(f"A board of {pattern_cols + 1} x {pattern_rows + 1} squares has {pattern_cols} x {pattern_rows} "
               "inner corners. The square size only scales the board poses, not K or the distortion.")
pattern_size = (int(pattern_cols), int(pattern_rows))

step = st.segmented_control("Step", ["1 Capture", "2 Calibrate", "3 Inspect", "4 Undistort", "5 Save"],
                            default="1 Capture", key="calib_step", label_visibility="collapsed") or "1 Capture"


# ------------------------------------------------------------------------------------------------ helpers
def draw_board(bgr, corners, found=True):
    """Copy of the photo with the detected grid drawn on it."""
    canvas = bgr.copy()
    if corners is not None:
        cv2.drawChessboardCorners(canvas, pattern_size, corners, found)
    return canvas


def add_view(name, bgr):
    """[1A.2] Detect the board in one photo and keep it when the full grid is found.

    Returns (accepted: bool, message: str, overlay: BGR image).
    """
    views = state.calib_views
    if views and bgr.shape[:2] != views[0]["bgr"].shape[:2]:
        h, w = views[0]["bgr"].shape[:2]
        return False, f"{name}: {bgr.shape[1]}x{bgr.shape[0]} differs from {w}x{h}; use one camera and resolution", bgr
    found, corners, detector = detect_corners(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), pattern_size)
    if not found:
        return False, f"{name}: no complete {pattern_size[0]}x{pattern_size[1]} grid found", bgr
    views.append({"name": name, "bgr": bgr, "corners": corners, "detector": detector})
    return True, f"{name}: grid found ({detector}), view {len(views)} kept", draw_board(bgr, corners)


def coverage_map(views):
    """Heat map of where the board has been in the frame (good calibration covers the whole frame)."""
    h, w = views[0]["bgr"].shape[:2]
    heat = np.zeros((h, w), np.float32)
    for view in views:
        hull = cv2.convexHull(view["corners"].reshape(-1, 2).astype(np.int32))
        layer = np.zeros_like(heat)
        cv2.fillConvexPoly(layer, hull, 1.0)
        heat += layer
    heat = (heat / max(heat.max(), 1) * 255).astype(np.uint8)
    blended = cv2.addWeighted(views[-1]["bgr"], 0.35, colour_map(heat, cv2.COLORMAP_VIRIDIS), 0.65, 0)
    return blended, float((heat > 0).mean())


def tilt_degrees(corners):
    """Rough board tilt: how far the grid's left/right edge lengths differ (0 = facing the camera)."""
    grid = corners.reshape(pattern_size[1], pattern_size[0], 2)
    left = np.linalg.norm(grid[-1, 0] - grid[0, 0])
    right = np.linalg.norm(grid[-1, -1] - grid[0, -1])
    top = np.linalg.norm(grid[0, -1] - grid[0, 0])
    bottom = np.linalg.norm(grid[-1, -1] - grid[-1, 0])
    ratio = max(left / right, right / left, top / bottom, bottom / top)
    return float(np.degrees(np.arccos(min(1.0, 1.0 / ratio))))


@st.cache_data(show_spinner="Calibrating ...")
def run_calibration(corners_list, image_size, pattern, square_size_mm, fix_k3):
    """[1A.2a] -> [1A.3] -> [1A.4] for one variant; cached on the exact views and settings."""
    board = make_board_points(pattern, square_size_mm / 1000.0)
    object_points = [board] * len(corners_list)
    result = calibrate_camera(object_points, list(corners_list), image_size, fix_k3=fix_k3)
    per_view, mean_px = compute_reprojection_errors(result, object_points, list(corners_list))
    return result, per_view, mean_px, board


def current_calibration(fix_k3):
    """Calibration of the views in this session, or None when there are too few."""
    views = state.calib_views
    if len(views) < settings.min_views:
        return None
    h, w = views[0]["bgr"].shape[:2]
    return run_calibration(tuple(v["corners"] for v in views), (w, h), pattern_size, square_mm, fix_k3)


def need_views():
    st.info(f"Add at least {settings.min_views} views in **1 Capture** first "
            f"(10-20 varied views give a stable result).", icon=":material/info:")


# ------------------------------------------------------------------------------------------------ 1 Capture
if step == "1 Capture":
    step_header("1A.0-1A.2", "Collect checkerboard views",
                "Every photo is checked as soon as it arrives: only views where the whole inner-corner grid is "
                "found are kept. Move the board to the corners and edges of the frame and tilt it.")
    left, right = st.columns([3, 2], gap="large")
    with left:
        source = st.segmented_control("Source", ["Camera", "Upload", "Project photos"], default="Camera",
                                      key="calib_source") or "Camera"
        if source == "Camera":
            shot = st.camera_input("Point the camera at the board and take a photo", key="calib_camera")
            if shot is not None and shot.file_id != state.calib_last_shot:
                state.calib_last_shot = shot.file_id
                ok, message, overlay = add_view(f"cam_{len(state.calib_views) + 1:02d}",
                                                decode_image_bytes(shot.getvalue()))
                state.calib_message = (ok, message, overlay)
        elif source == "Upload":
            uploads = st.file_uploader("Checkerboard photos from one camera", type=IMAGE_TYPES,
                                       accept_multiple_files=True, key="calib_uploads")
            if uploads and st.button(f"Check and add {len(uploads)} photos", type="primary"):
                results = [add_view(u.name, decode_image_bytes(u.getvalue())) for u in uploads]
                kept = sum(ok for ok, _, _ in results)
                state.calib_message = (kept > 0, f"{kept} of {len(uploads)} photos kept. "
                                       + " ".join(m for ok, m, _ in results if not ok), results[-1][2])
        else:
            folder = cfg.paths.calibration_images
            count = len(list_images(folder)) if folder.is_dir() else 0
            st.write(f"`{folder.relative_to(cfg.project_root).as_posix()}` holds **{count}** photos "
                     "(the set used for the report).")
            if count and st.button("Load project photos", type="primary"):
                results = [add_view(name, bgr) for name, bgr in load_calibration_images(folder)]
                kept = sum(ok for ok, _, _ in results)
                state.calib_message = (kept > 0, f"{kept} of {count} photos kept.", results[-1][2])
            with st.expander("Desktop tools (run on this computer, open their own window)"):
                st.caption("`capture_checker_task1a.py --camera`: the guided 13-shot capture plan with "
                           "auto-capture. The photos it saves are loaded here when its window closes.")
                if st.button("Open guided capture window"):
                    try:
                        with st.spinner("Capture window is open; finish or press Q there ..."):
                            live_folder = collect_calibration_images(cfg, "camera")
                        results = [add_view(n, b) for n, b in load_calibration_images(live_folder)]
                        state.calib_message = (True, f"{sum(r[0] for r in results)} views loaded from "
                                               f"{live_folder.name}", results[-1][2])
                    except Exception as err:           # camera missing, tool failed, no photos saved
                        st.error(f"Capture tool failed: {err}")

        if state.calib_message:
            ok, message, overlay = state.calib_message
            (st.success if ok else st.warning)(message)
            show(overlay, caption="Last photo with the detected grid")

    with right:
        views = state.calib_views
        target = 15
        st.metric("Views kept", len(views))
        st.progress(min(len(views) / target, 1.0), text=f"{len(views)} / {target} suggested")
        if views:
            heat, covered = coverage_map(views)
            show(heat, caption=f"Frame coverage: {covered:.0%} of the frame has seen the board")
            tilts = [tilt_degrees(v["corners"]) for v in views]
            st.caption(f"Board tilt per view (approx.): {min(tilts):.0f}° to {max(tilts):.0f}°. "
                       "A spread of tilts helps separate focal length from distance.")
            with st.popover("Manage views", icon=":material/photo_library:"):
                remove = st.multiselect("Remove", [v["name"] for v in views])
                col_a, col_b = st.columns(2)
                if col_a.button("Remove selected", disabled=not remove):
                    state.calib_views = [v for v in views if v["name"] not in remove]
                    st.rerun()
                if col_b.button("Clear all", type="secondary"):
                    state.calib_views, state.calib_message = [], None
                    st.rerun()

    if state.calib_views:
        st.markdown("##### Kept views")
        grid = st.columns(6)
        for index, view in enumerate(state.calib_views):
            with grid[index % 6]:
                show(draw_board(view["bgr"], view["corners"]), caption=view["name"])

# ------------------------------------------------------------------------------------------------ 2 Calibrate
elif step == "2 Calibrate":
    step_header("1A.3-1A.4", "Estimate K and lens distortion",
                "cv2.calibrateCamera fits the pinhole model plus 5 distortion coefficients. The advanced variant "
                "holds k3 at 0, which stops k2/k3 from trading off against each other on small view sets.")
    calibrated = current_calibration(False)
    if calibrated is None:
        need_views()
        st.stop()
    fixed = current_calibration(True)
    use_fix = st.toggle("Use the fix_k3 variant (advanced)", value=state.get("fix_k3_choice", bool(settings.fix_k3)))
    state.fix_k3_choice = use_fix
    result, per_view, mean_px, _ = fixed if use_fix else calibrated

    cols = st.columns(5)
    cols[0].metric("Views", len(per_view))
    cols[1].metric("RMS error", f"{result.rms_px:.3f} px")
    cols[2].metric("Mean error", f"{mean_px:.3f} px")
    cols[3].metric("fx / fy", f"{result.K[0, 0]:.0f} / {result.K[1, 1]:.0f}")
    cols[4].metric("Principal point", f"{result.K[0, 2]:.0f}, {result.K[1, 2]:.0f}")

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Intrinsic matrix K (pixels)**")
        st.latex(r"K = \begin{bmatrix} %.2f & %.2f & %.2f \\ %.2f & %.2f & %.2f \\ %.0f & %.0f & %.0f \end{bmatrix}"
                 % tuple(result.K.ravel()))
        st.markdown("**Baseline vs fix_k3**")
        rows = []
        for label, (res, _, mean, _) in (("baseline", calibrated), ("fix_k3", fixed)):
            k1, k2, p1, p2, k3 = res.dist.ravel()[:5]
            rows.append({"variant": label, "RMS px": res.rms_px, "mean px": mean, "fx": res.K[0, 0],
                         "k1": k1, "k2": k2, "p1": p1, "p2": p2, "k3": k3})
        st.dataframe(pd.DataFrame(rows).round(4), hide_index=True)
        st.caption("If the RMS is almost the same but k2/k3 are large with opposite signs, the 5-coefficient "
                   "model is over-fitting; fix_k3 gives the same fit with sensible coefficients.")
    with right:
        st.markdown("**Reprojection error per view**")
        errors = pd.DataFrame({"view": [v["name"] for v in state.calib_views],
                               "mean_px": [m for m, _ in per_view], "rms_px": [r for _, r in per_view]})
        bars = alt.Chart(errors).mark_bar().encode(
            x=alt.X("view:N", sort=None, title=None), y=alt.Y("mean_px:Q", title="mean error (px)"),
            color=alt.condition(alt.datum.mean_px > 2 * mean_px, alt.value("#e45756"), alt.value("#4c78a8")),
            tooltip=["view", alt.Tooltip("mean_px:Q", format=".3f"), alt.Tooltip("rms_px:Q", format=".3f")])
        rule = alt.Chart(pd.DataFrame({"y": [mean_px]})).mark_rule(strokeDash=[4, 4]).encode(y="y:Q")
        st.altair_chart(bars + rule, width="stretch")
        st.caption("Red = more than twice the overall mean; such a view is a candidate to retake.")

# ------------------------------------------------------------------------------------------------ 3 Inspect
elif step == "3 Inspect":
    step_header("1A.4", "Inspect one view", "Green = detected corner, red = the same corner re-projected through the "
                "fitted model. The scatter shows the residual of every corner (should be a tight blob around 0).")
    calibrated = current_calibration(state.get("fix_k3_choice", bool(settings.fix_k3)))
    if calibrated is None:
        need_views()
        st.stop()
    result, per_view, mean_px, board = calibrated
    names = [v["name"] for v in state.calib_views]
    name = st.select_slider("View", names, value=names[int(np.argmax([m for m, _ in per_view]))])
    index = names.index(name)
    view = state.calib_views[index]
    projected, _ = cv2.projectPoints(board, result.rvecs[index], result.tvecs[index], result.K, result.dist)
    detected, projected = view["corners"].reshape(-1, 2), projected.reshape(-1, 2)
    canvas = view["bgr"].copy()
    for (dx, dy), (px, py) in zip(detected, projected):
        cv2.circle(canvas, (int(round(dx)), int(round(dy))), 4, (60, 200, 60), 1, cv2.LINE_AA)
        cv2.drawMarker(canvas, (int(round(px)), int(round(py))), (40, 40, 230), cv2.MARKER_CROSS, 6, 1)
    left, right = st.columns([3, 2], gap="large")
    with left:
        show(canvas, caption=f"{name}: mean {per_view[index][0]:.3f} px, RMS {per_view[index][1]:.3f} px")
    with right:
        residual = pd.DataFrame(detected - projected, columns=["dx", "dy"])
        residual["corner"] = range(len(residual))
        limit = float(np.abs(residual[["dx", "dy"]]).to_numpy().max()) * 1.2 or 1.0
        scale = alt.Scale(domain=[-limit, limit])
        chart = alt.Chart(residual).mark_circle(size=40, opacity=0.7).encode(
            x=alt.X("dx:Q", scale=scale, title="dx (px)"), y=alt.Y("dy:Q", scale=scale, title="dy (px)"),
            tooltip=["corner", alt.Tooltip("dx:Q", format=".3f"), alt.Tooltip("dy:Q", format=".3f")])
        st.altair_chart(chart.properties(height=360), width="stretch")
        rotation, _ = cv2.Rodrigues(result.rvecs[index])
        distance_mm = float(np.linalg.norm(result.tvecs[index])) * 1000
        angle = float(np.degrees(np.arccos(np.clip(abs(rotation[2, 2]), -1, 1))))
        st.metric("Board distance", f"{distance_mm:.0f} mm")
        st.metric("Board tilt from the image plane", f"{angle:.1f}°")

# ------------------------------------------------------------------------------------------------ 4 Undistort
elif step == "4 Undistort":
    step_header("1A.5", "Remove the lens distortion",
                "alpha = 0 crops to pixels that are valid everywhere; alpha = 1 keeps every source pixel "
                "(black curved borders show how much the lens bends the image).")
    saved_path = cfg.paths.artifacts / "calibration.npz"
    options = (["This session"] if len(state.calib_views) >= settings.min_views else []) + \
              (["Saved calibration.npz"] if saved_path.is_file() else [])
    if not options:
        need_views()
        st.stop()
    which = st.segmented_control("Calibration", options, default=options[0], key="calib_which") or options[0]
    if which == "This session":
        result = current_calibration(state.get("fix_k3_choice", bool(settings.fix_k3)))[0]
        K, dist, size = result.K, result.dist, result.image_size
    else:
        K, dist, size = load_calibration(saved_path)

    col_1, col_2, col_3 = st.columns([2, 2, 1])
    image_source = col_1.segmented_control("Image", ["Kept view", "Camera", "Upload"], default="Kept view"
                                           if state.calib_views else "Upload", key="und_source") or "Upload"
    alpha = col_2.slider("alpha", 0.0, 1.0, float(settings.undistort_alpha), 0.05)
    draw_grid = col_3.toggle("Grid", value=True)

    image = None
    if image_source == "Kept view" and state.calib_views:
        names = [v["name"] for v in state.calib_views]
        image = state.calib_views[names.index(st.selectbox("View", names))]["bgr"]
    elif image_source == "Kept view" and saved_path.is_file():
        files = list_images(cfg.paths.calibration_images)
        image = read_image(files[0]) if files else None
    elif image_source == "Camera":
        shot = st.camera_input("Photo to undistort", key="und_camera")
        image = decode_image_bytes(shot.getvalue()) if shot else None
    else:
        upload = st.file_uploader("Photo from the same camera", type=IMAGE_TYPES, key="und_upload")
        image = decode_image_bytes(upload.getvalue()) if upload else None
    if image is None:
        st.info("Choose an image taken by the calibrated camera.")
        st.stop()
    if (image.shape[1], image.shape[0]) != tuple(size):
        st.error(f"This image is {image.shape[1]}x{image.shape[0]} but the calibration is for {size[0]}x{size[1]}.")
        st.stop()

    undistorted = undistort_image(image, K, dist, alpha)

    def with_grid(img):
        out = img.copy()
        if draw_grid:
            for x in range(0, out.shape[1], 40):
                cv2.line(out, (x, 0), (x, out.shape[0]), (255, 200, 0), 1)
            for y in range(0, out.shape[0], 40):
                cv2.line(out, (0, y), (out.shape[1], y), (255, 200, 0), 1)
        return out

    left, right = st.columns(2)
    with left:
        show(with_grid(image), caption="Before: straight lines on the board bend near the edges")
    with right:
        show(with_grid(undistorted), caption="After: [1A.5] undistort_image")

    st.markdown("##### Distortion field")
    width, height = size
    xs, ys = np.meshgrid(np.linspace(0, width - 1, 17), np.linspace(0, height - 1, 13))
    points = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32).reshape(-1, 1, 2)
    corrected = cv2.undistortPoints(points, K, dist, P=K).reshape(-1, 2)
    shift = corrected - points.reshape(-1, 2)
    magnitude = np.linalg.norm(shift, axis=1)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    quiver = ax.quiver(points[:, 0, 0], points[:, 0, 1], shift[:, 0], shift[:, 1], magnitude, cmap="viridis",
                       angles="xy", scale_units="xy", scale=1)
    ax.set_xlim(-40, width + 40)
    ax.set_ylim(height + 40, -40)
    ax.set_aspect("equal")
    ax.set_title("Where each pixel moves when the distortion is removed (arrow length = shift in px)")
    fig.colorbar(quiver, ax=ax, label="shift (px)")
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.caption(f"Largest shift {magnitude.max():.1f} px (frame corners), smallest {magnitude.min():.2f} px near "
               "the principal point. Negative k1 = barrel distortion, which pulls the edges inward.")

# ------------------------------------------------------------------------------------------------ 5 Save
else:
    step_header("1A.6", "Save the calibration",
                "Saving writes artifacts/calibration.npz (K, dist, width, height, rms_px). Task 3 and "
                "undistort_view_task1a.py read that file, so this replaces the calibration they use.")
    calibrated = current_calibration(state.get("fix_k3_choice", bool(settings.fix_k3)))
    saved_path = cfg.paths.artifacts / "calibration.npz"
    if calibrated is None:
        need_views()
    else:
        result = calibrated[0]
        st.write(f"Session calibration: **{len(state.calib_views)} views**, RMS **{result.rms_px:.3f} px**, "
                 f"{result.image_size[0]}x{result.image_size[1]}.")
        buffer = io.BytesIO()
        np.savez(buffer, K=result.K, dist=result.dist, width=result.image_size[0],
                 height=result.image_size[1], rms_px=result.rms_px)
        col_1, col_2, col_3 = st.columns(3)
        col_1.download_button("Download calibration.npz", buffer.getvalue(), "calibration.npz",
                              icon=":material/download:")
        confirm = col_2.checkbox("Replace artifacts/calibration.npz")
        if col_2.button("Save to artifacts", type="primary", disabled=not confirm):
            save_calibration(result, saved_path)
            st.success(f"Saved {saved_path.relative_to(cfg.project_root).as_posix()}")
        if col_3.button("Save the photos"):
            folder = cfg.project_root / settings.live_save_root / time.strftime("session_%Y%m%d_%H%M%S")
            for view in state.calib_views:
                save_image(view["bgr"], folder / f"{view['name'].rsplit('.', 1)[0]}.png")
            st.success(f"{len(state.calib_views)} photos in {folder.relative_to(cfg.project_root).as_posix()}")

    st.divider()
    st.markdown("**Live check on this computer**")
    st.caption("Opens undistort_view_task1a.py with the saved calibration: a live BEFORE | AFTER window "
               "(G grid, A crop, S snapshot, Q quit).")
    if st.button("Open live undistort window", disabled=not saved_path.is_file()):
        try:
            with st.spinner("Live window is open; press Q there to close it ..."):
                view_undistorted(cfg, "camera")
        except Exception as err:
            st.error(f"Live viewer failed: {err}")
