"""[DEMO 1B] Classical image processing page: one segment per step, every control re-runs the src/ function.

Flow:  [1B.1] add_noise -> [1B.2] apply_spatial_filters -> [1B.2m] compare_to_reference
       [1B.3] filter_frequency   [1B.4] compute_edges   [1B.5h] detect_harris   [1B.5] match_features
"""
import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from demo.common import colour_map, get_config, image_input, show, step_header, to_display
from src.edges import compute_edges
from src.features import detect_harris, match_features
from src.filtering import add_noise, apply_spatial_filters, compare_to_reference
from src.frequency import filter_frequency
from src.visualize import draw_corners, draw_matches

cfg = get_config()
settings = cfg.task1b
root = cfg.project_root
max_side = settings.max_side_px

st.title("Task 1B · Image processing")
step = st.segmented_control("Step", ["Noise & filters", "Frequency domain", "Edges", "Harris corners",
                                     "Feature matching"], default="Noise & filters", key="proc_step",
                            label_visibility="collapsed") or "Noise & filters"


def odd(label, low, high, value, key):
    return st.slider(label, low, high, value, step=2, key=key)


# ------------------------------------------------------------------------------------------- filters
if step == "Noise & filters":
    step_header("1B.1-1B.2m", "Denoise and score three filters",
                "Noise is added to a clean photo so each filter can be scored against the original "
                "(higher PSNR = closer to clean). Median wins on salt & pepper, Gaussian on Gaussian noise.")
    controls, view = st.columns([1, 3], gap="large")
    with controls:
        image = image_input("filt", root / settings.sample_image, max_side=max_side)
        kind = st.segmented_control("Noise", ["gaussian", "salt_pepper"], default="gaussian", key="noise_kind")
        kind = kind or "gaussian"
        level = (st.slider("Std (0-255)", 0.0, 60.0, 20.0, 1.0) if kind == "gaussian"
                 else st.slider("Fraction of pixels", 0.0, 0.3, 0.05, 0.01))
        ksize = odd("Kernel size", 3, 15, settings.kernel_size, "filt_k")
        sigma = st.slider("Gaussian sigma", 0.3, 5.0, float(settings.gaussian_sigma), 0.1)
        method = st.segmented_control("Convolution", ["opencv", "scratch"], default="opencv", key="filt_method",
                                      help="scratch = own convolve2d() (slower, same result)") or "opencv"
    if image is None:
        view.info("Choose an image.")
        st.stop()
    noisy = add_noise(image, kind, level, seed=cfg.seed) if level > 0 else image.copy()
    filtered = apply_spatial_filters(noisy, ksize, sigma, method)
    scores = compare_to_reference({"noisy": noisy, **filtered}, image)
    with view:
        best = max(filtered, key=lambda name: scores[name]["psnr_db"])
        cols = st.columns(4)
        for col, name in zip(cols, ["noisy", "box", "gaussian", "median"]):
            col.metric(name, f"{scores[name]['psnr_db']:.2f} dB",
                       None if name == "noisy" else f"{scores[name]['psnr_db'] - scores['noisy']['psnr_db']:+.2f} dB")
        row_1 = st.columns(2)
        with row_1[0]:
            show(image, caption="clean original")
        with row_1[1]:
            show(noisy, caption=f"noisy ({kind}, {level:g})")
        row_2 = st.columns(3)
        for col, name in zip(row_2, ["box", "gaussian", "median"]):
            with col:
                show(filtered[name], caption=f"{name} {ksize}x{ksize}" + ("  ★ best" if name == best else ""))
        with st.expander("Zoom: centre crop", expanded=False):
            h, w = image.shape[:2]
            crop = slice(h // 2 - h // 8, h // 2 + h // 8), slice(w // 2 - w // 8, w // 2 + w // 8)
            cols = st.columns(4)
            for col, (name, img) in zip(cols, [("noisy", noisy), *filtered.items()]):
                with col:
                    show(cv2.resize(img[crop], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST), caption=name)
        table = pd.DataFrame([{"image": n, "PSNR dB": s["psnr_db"], "MSE": s["mse"]} for n, s in scores.items()])
        st.altair_chart(alt.Chart(table).mark_bar().encode(
            x=alt.X("PSNR dB:Q"), y=alt.Y("image:N", sort=None, title=None),
            color=alt.condition(alt.datum.image == best, alt.value("#54a24b"), alt.value("#4c78a8")),
            tooltip=["image", alt.Tooltip("PSNR dB:Q", format=".2f"), alt.Tooltip("MSE:Q", format=".1f")]
        ).properties(height=160), width="stretch")

# ------------------------------------------------------------------------------------------- frequency
elif step == "Frequency domain":
    step_header("1B.3", "Filter in the Fourier domain",
                "FFT -> shift the zero frequency to the centre -> multiply by a mask -> inverse FFT. Low-pass "
                "keeps the centre (smooth shapes); high-pass keeps the outside (edges and fine texture).")
    controls, view = st.columns([1, 3], gap="large")
    with controls:
        image = image_input("freq", root / settings.sample_image, max_side=max_side)
        kind = st.segmented_control("Filter", ["lowpass", "highpass"], default="lowpass", key="freq_kind")
        profile = st.segmented_control("Profile", ["ideal", "gaussian", "butterworth"], default="ideal",
                                       key="freq_profile", help="ideal rings (Gibbs); gaussian/butterworth do not")
        radius = st.slider("Cut-off radius (px)", 2, 150, settings.fft_radius)
        order = st.slider("Butterworth order", 1, 10, settings.butterworth_order) if profile == "butterworth" else 2
    if image is None:
        view.info("Choose an image.")
        st.stop()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = filter_frequency(gray, kind or "lowpass", radius, profile or "ideal", order)
    with view:
        cols = st.columns(2)
        with cols[0]:
            show(gray, caption="input (grayscale)")
        with cols[1]:
            show(result.filtered_uint8, caption=f"{kind} {profile}, radius {radius}"
                 + (" (shown as 128 + value)" if kind == "highpass" else ""))
        cols = st.columns(3)
        spectrum = to_display(result.log_spectrum)
        with cols[0]:
            show(colour_map(spectrum), caption="log magnitude spectrum")
        with cols[1]:
            show((result.mask * 255).astype(np.uint8), caption="mask (white = kept)")
        with cols[2]:
            show(colour_map(to_display(result.log_spectrum * result.mask)), caption="spectrum x mask")
        energy = (np.expm1(result.log_spectrum.astype(np.float64)) ** 2)
        st.metric("Spectral energy kept by the mask", f"{float((energy * result.mask ** 2).sum() / energy.sum()):.2%}")

# ------------------------------------------------------------------------------------------- edges
elif step == "Edges":
    step_header("1B.4", "Sobel gradients and Canny",
                "Sobel gives the signed gradient in x and y; magnitude combines them. Canny thins those to "
                "one-pixel edges and keeps weak edges only when they touch strong ones (hysteresis).")
    controls, view = st.columns([1, 3], gap="large")
    with controls:
        image = image_input("edge", root / settings.sample_image, max_side=max_side)
        blur = odd("Pre-blur kernel", 3, 15, settings.edge_blur_ksize, "edge_k")
        blur_sigma = st.slider("Pre-blur sigma", 0.3, 5.0, float(settings.edge_blur_sigma), 0.1)
        auto = st.toggle("Automatic thresholds (median rule)", value=False)
        low, high = st.slider("Canny low / high", 0, 400, (settings.canny_low, settings.canny_high),
                              disabled=auto)
    if image is None:
        view.info("Choose an image.")
        st.stop()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = compute_edges(gray, low, max(high, low + 1), blur, blur_sigma, auto_threshold=auto)
    with view:
        cols = st.columns(3)
        cols[0].metric("Canny thresholds", f"{result.thresholds[0]:.0f} / {result.thresholds[1]:.0f}")
        cols[1].metric("Edge density", f"{result.edge_density:.2%}")
        cols[2].metric("Max gradient", f"{float(result.magnitude.max()):.0f}")
        cols = st.columns(2)
        with cols[0]:
            show(result.smoothed, caption="pre-smoothed")
        with cols[1]:
            show(result.canny, caption="Canny edges")
        cols = st.columns(3)
        with cols[0]:
            show(to_display(result.grad_x, signed=True), caption="Sobel x (vertical edges)")
        with cols[1]:
            show(to_display(result.grad_y, signed=True), caption="Sobel y (horizontal edges)")
        with cols[2]:
            hsv = np.zeros((*gray.shape, 3), np.uint8)
            hsv[..., 0] = ((result.orientation + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
            hsv[..., 1] = 255
            hsv[..., 2] = to_display(result.magnitude)
            show(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), caption="magnitude, colour = direction")

# ------------------------------------------------------------------------------------------- harris
elif step == "Harris corners":
    step_header("1B.5h", "Harris corner response",
                "R = det(M) - k trace(M)^2 is large where the gradient is strong in both directions. Local "
                "maxima above a fraction of the strongest response are kept.")
    controls, view = st.columns([1, 3], gap="large")
    with controls:
        image = image_input("harris", root / settings.pair_image_1, max_side=max_side)
        method = st.segmented_control("Implementation", ["opencv", "scratch"], default="opencv",
                                      key="harris_method") or "opencv"
        k = st.slider("k", 0.02, 0.15, float(settings.harris_k), 0.005)
        threshold = st.slider("Threshold (fraction of max R)", 0.001, 0.2, float(settings.harris_threshold), 0.001,
                              format="%.3f")
        max_corners = st.slider("Max corners", 10, 500, settings.harris_max_corners, 10)
    if image is None:
        view.info("Choose an image.")
        st.stop()
    response, corners = detect_harris(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), method, k, threshold, max_corners)
    with view:
        st.metric("Corners kept", len(corners))
        cols = st.columns(2)
        with cols[0]:
            show(draw_corners(image, corners), caption="corners (strongest first)")
        with cols[1]:
            show(colour_map(to_display(np.sign(response) * np.sqrt(np.abs(response)))),
                 caption="response R (sqrt-scaled; bright = corner, dark = edge)")

# ------------------------------------------------------------------------------------------- matching
else:
    step_header("1B.5", "Detect, describe and match",
                "Keypoints in both views are matched by descriptor distance; Lowe's ratio test drops ambiguous "
                "matches and RANSAC keeps the ones consistent with one homography (green = inlier, red = outlier).")
    controls, view = st.columns([1, 3], gap="large")
    with controls:
        image_1 = image_input("m1", root / settings.pair_image_1, label="View 1", max_side=max_side)
        default_2 = root / settings.pair_image_2
        if st.toggle("Use the unrelated image as view 2", value=False, help="negative control"):
            default_2 = root / settings.unrelated_image
        image_2 = image_input("m2", default_2, label="View 2", max_side=max_side)
        detector = st.segmented_control("Detector", ["orb", "sift", "harris"], default=settings.detector,
                                        key="m_detector") or "orb"
        ratio = st.slider("Ratio test", 0.4, 1.0, float(settings.ratio_threshold), 0.01)
        use_ransac = st.toggle("RANSAC homography", value=bool(settings.use_ransac))
        threshold = st.slider("RANSAC threshold (px)", 1.0, 20.0, float(settings.ransac_threshold_px), 0.5,
                              disabled=not use_ransac)
        max_features = st.slider("Max keypoints", 100, 3000, settings.max_features, 100)
        draw_n = st.slider("Matches drawn", 5, 200, settings.draw_max_matches, 5)
    if image_1 is None or image_2 is None:
        view.info("Choose two images.")
        st.stop()
    with view:
        try:
            result = match_features(image_1, image_2, detector, ratio, use_ransac, max_features, threshold)
        except (ValueError, cv2.error) as err:
            st.error(str(err))
            st.stop()
        stats = result.stats
        cols = st.columns(4)
        cols[0].metric("Keypoints", f"{stats['keypoints_1']} / {stats['keypoints_2']}")
        cols[1].metric("Good matches", stats["good_matches"])
        cols[2].metric("RANSAC inliers", stats.get("inliers", "-"))
        cols[3].metric("Inlier ratio", f"{stats['inlier_ratio']:.0%}" if "inlier_ratio" in stats else "-")
        show(draw_matches(image_1, image_2, result, max_matches=draw_n),
             caption=f"{detector.upper()}: strongest {min(draw_n, stats['good_matches'])} matches")
        if use_ransac and result.homography is not None:
            h, w = image_1.shape[:2]
            outline = cv2.perspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2),
                                               result.homography)
            warped = image_2.copy()
            cv2.polylines(warped, [np.int32(outline)], True, (60, 200, 60), 3, cv2.LINE_AA)
            cols = st.columns(2)
            with cols[0]:
                show(warped, caption="view 1's outline mapped into view 2 by the homography")
            with cols[1]:
                aligned = cv2.warpPerspective(image_1, result.homography, (image_2.shape[1], image_2.shape[0]))
                show(cv2.addWeighted(image_2, 0.5, aligned, 0.5, 0), caption="view 1 warped onto view 2 (50% blend)")
        st.caption("Few inliers or a low inlier ratio means the views probably do not show the same planar scene; "
                   "try the unrelated image to see what a failed match looks like.")
