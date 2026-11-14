"""[1B.R] Task 1B runner: chains filtering, frequency, edges and features. No algorithm code here.

Flow:  run_task1b(cfg) -> [1B.1] -> [1B.2] -> [1B.2m] -> [1B.3] -> [1B.4] -> [1B.5]
       writes results/task1b/: 01_* filtering, 02_* frequency, 03_* edges, 04_* Harris, 05_* matching,
       metrics_filtering.csv, metrics_matching.csv, metrics.json
"""
import time
from dataclasses import dataclass, field

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.config import set_seed
from src.edges import compute_edges
from src.features import detect_harris, match_features
from src.filtering import add_noise, apply_spatial_filters, compare_to_reference
from src.frequency import filter_frequency
from src.io_utils import read_image, save_csv, save_image, save_json
from src.visualize import draw_corners, draw_matches, plot_filter_scores, show_grid


@dataclass
class Task1BReport:
    """Output of run_task1b()."""
    filter_scores: dict           # from [1B.2m]: noise label -> method -> {mse, psnr_db}
    match_stats: list             # from [1B.5]: one row per pair x detector
    figure_paths: list            # figures written to results/task1b/
    edge_stats: list = field(default_factory=list)    # from [1B.4]: thresholds -> edge density
    harris_stats: dict = field(default_factory=dict)  # from [1B.5h]: corner counts, opencv vs scratch
    metrics_path: object = None   # results/task1b/metrics.json


def _noise_label(test):
    """'gaussian' 20 -> 'gaussian (std 20)', 'salt_pepper' 0.05 -> 'salt_pepper (5%)'."""
    if test["kind"] == "salt_pepper":
        return f"salt_pepper ({test['level'] * 100:g}%)"
    return f"{test['kind']} (std {test['level']:g})"


def _save_figure(fig, path, figure_paths):
    """Save a matplotlib figure at report resolution, close it, record the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figure_paths.append(path)


def _load_inputs(cfg):
    """Read the Task 1B photos named in config. The unrelated image is optional."""
    settings = cfg.task1b
    root = cfg.project_root
    required = {"sample_image": settings.sample_image,
                "pair_image_1": settings.pair_image_1,
                "pair_image_2": settings.pair_image_2}
    missing = [f"{key}: {root / rel}" for key, rel in required.items() if not (root / rel).is_file()]
    if missing:
        raise FileNotFoundError("[1B.R] Task 1B needs your own photos in data/samples/. Missing:\n  "
                                + "\n  ".join(missing))
    images = {key: read_image(root / rel, settings.max_side_px) for key, rel in required.items()}
    unrelated_path = root / settings.unrelated_image if settings.unrelated_image else None
    images["unrelated_image"] = (read_image(unrelated_path, settings.max_side_px)
                                 if unrelated_path is not None and unrelated_path.is_file() else None)
    return images


def _run_filtering(clean_gray, settings, seed, out_dir, figure_paths):
    """[1B.1] -> [1B.2] -> [1B.2m] for every noise test in config. Returns (scores, csv rows, scratch check)."""
    scores, rows, grid_images, grid_titles = {}, [], [], []
    for index, test in enumerate(settings.noise_tests):
        label = _noise_label(test)
        noisy = add_noise(clean_gray, test["kind"], test["level"], seed=seed + index)          # [1B.1]
        filtered = apply_spatial_filters(noisy, settings.kernel_size, settings.gaussian_sigma,  # [1B.2]
                                         method=settings.filter_method)
        scores[label] = compare_to_reference({"noisy": noisy, **filtered}, clean_gray)          # [1B.2m]
        rows += [{"noise": label, "method": method, "mse": round(score["mse"], 3),
                  "psnr_db": round(score["psnr_db"], 3)} for method, score in scores[label].items()]
        grid_images += [clean_gray, noisy, filtered["box"], filtered["gaussian"], filtered["median"]]
        grid_titles += ["Clean", f"Noisy: {label}"] + [
            f"{name} k={settings.kernel_size}  PSNR {scores[label][name]['psnr_db']:.1f} dB"
            for name in ("box", "gaussian", "median")]
    _save_figure(show_grid(grid_images, grid_titles, cols=5,
                           suptitle="[1B.2] Spatial filtering on controlled noise"),
                 out_dir / "01_filter_comparison.png", figure_paths)
    _save_figure(plot_filter_scores(scores), out_dir / "01b_filter_psnr.png", figure_paths)

    # Advanced check: own convolution vs OpenCV on the first noisy image (should differ by <= 1 grey level).
    first = settings.noise_tests[0]
    noisy = add_noise(clean_gray, first["kind"], first["level"], seed=seed)
    opencv = apply_spatial_filters(noisy, settings.kernel_size, settings.gaussian_sigma, method="opencv")
    scratch = apply_spatial_filters(noisy, settings.kernel_size, settings.gaussian_sigma, method="scratch")
    scratch_check = {name: int(np.abs(opencv[name].astype(int) - scratch[name].astype(int)).max())
                     for name in ("box", "gaussian")}
    return scores, rows, scratch_check


def _run_frequency(clean_gray, settings, out_dir, figure_paths):
    """[1B.3] Spectrum, ideal low/high-pass, Gaussian and Butterworth low-pass, radius sweep."""
    radius = settings.fft_radius
    ideal_low = filter_frequency(clean_gray, "lowpass", radius, "ideal")
    ideal_high = filter_frequency(clean_gray, "highpass", radius, "ideal")
    gauss_low = filter_frequency(clean_gray, "lowpass", radius, "gaussian")
    butter_low = filter_frequency(clean_gray, "lowpass", radius, "butterworth", settings.butterworth_order)
    signed_limit = float(np.percentile(np.abs(ideal_high.filtered), 99)) or 1.0
    _save_figure(show_grid(
        [clean_gray, ideal_low.log_spectrum, ideal_low.mask, ideal_high.mask,
         ideal_low.filtered_uint8, ideal_high.filtered, gauss_low.filtered_uint8, butter_low.filtered_uint8],
        ["Original", "Log |FFT| (fftshift, DC at centre)", f"Ideal low-pass mask r={radius}",
         "Ideal high-pass mask", "Ideal low-pass (ringing)", "Ideal high-pass (signed)",
         "Gaussian low-pass", f"Butterworth low-pass n={settings.butterworth_order}"],
        cols=4, cmaps=[None, "magma", None, None, None, "coolwarm", None, None],
        limits=[None, None, (0, 1), (0, 1), None, (-signed_limit, signed_limit), None, None],
        suptitle="[1B.3] Frequency-domain filtering"), out_dir / "02_frequency_domain.png", figure_paths)

    sweep_images, sweep_titles = [], []
    for sweep_radius in settings.fft_radius_sweep:
        for profile in ("ideal", "gaussian"):
            result = filter_frequency(clean_gray, "lowpass", sweep_radius, profile)
            sweep_images.append(result.filtered_uint8)
            sweep_titles.append(f"{profile} low-pass r={sweep_radius}")
    _save_figure(show_grid(sweep_images, sweep_titles, cols=2,
                           suptitle="[1B.3] Cut-off radius: blur and ringing"),
                 out_dir / "02b_fft_radius_sweep.png", figure_paths)


def _run_edges(clean_gray, settings, out_dir, figure_paths):
    """[1B.4] Sobel + Canny at the config thresholds, a threshold sweep and the auto variant."""
    baseline = compute_edges(clean_gray, settings.canny_low, settings.canny_high,
                             settings.edge_blur_ksize, settings.edge_blur_sigma)
    gradient_limit = float(np.percentile(np.abs(baseline.grad_x), 99)) or 1.0
    _save_figure(show_grid(
        [clean_gray, baseline.smoothed, baseline.grad_x, baseline.grad_y, baseline.magnitude, baseline.canny],
        ["Original", f"Gaussian smoothed s={settings.edge_blur_sigma}", "Sobel Gx (signed, vertical edges)",
         "Sobel Gy (signed, horizontal edges)", "Gradient magnitude",
         f"Canny {baseline.thresholds}  density {baseline.edge_density:.3f}"],
        cols=3, cmaps=[None, None, "coolwarm", "coolwarm", "gray", None],
        limits=[None, None, (-gradient_limit, gradient_limit), (-gradient_limit, gradient_limit), None, None],
        suptitle="[1B.4] Gradients and edges"), out_dir / "03_edges.png", figure_paths)

    variants = [compute_edges(clean_gray, low, high, settings.edge_blur_ksize, settings.edge_blur_sigma)
                for low, high in settings.canny_sweep]
    variants.append(compute_edges(clean_gray, blur_ksize=settings.edge_blur_ksize,
                                  blur_sigma=settings.edge_blur_sigma, auto_threshold=True))
    labels = [f"fixed {tuple(pair)}" for pair in settings.canny_sweep] + ["auto (median)"]
    _save_figure(show_grid([v.canny for v in variants],
                           [f"{label} -> {v.thresholds}, density {v.edge_density:.3f}"
                            for label, v in zip(labels, variants)],
                           cols=len(variants), suptitle="[1B.4] Canny thresholds"),
                 out_dir / "03b_canny_sweep.png", figure_paths)
    return [{"variant": label, "low": v.thresholds[0], "high": v.thresholds[1],
             "edge_density": round(v.edge_density, 4)} for label, v in zip(labels, variants)]


def _run_harris(clean_bgr, clean_gray, settings, out_dir, figure_paths):
    """[1B.5h] Harris corners with OpenCV and with the from-scratch response."""
    harris = {}
    for method in ("opencv", "scratch"):
        harris[method] = detect_harris(clean_gray, method, settings.harris_k,
                                       settings.harris_threshold, settings.harris_max_corners)
    _save_figure(show_grid(
        [draw_corners(clean_bgr, harris["opencv"][1]), np.maximum(harris["opencv"][0], 0),
         draw_corners(clean_bgr, harris["scratch"][1])],
        [f"Harris (OpenCV) corners n={len(harris['opencv'][1])}", "Harris response R > 0",
         f"Harris (scratch) corners n={len(harris['scratch'][1])}"],
        cols=3, cmaps=[None, "hot", None], suptitle="[1B.5h] Harris corners"),
        out_dir / "04_harris.png", figure_paths)
    return {method: len(corners) for method, (_, corners) in harris.items()}


def _run_matching(images, settings, out_dir, figure_paths):
    """[1B.5] Every detector on the same-scene pair (and the unrelated pair, if given)."""
    pairs = [("same_scene", images["pair_image_1"], images["pair_image_2"])]
    if images["unrelated_image"] is not None:
        pairs.append(("unrelated", images["pair_image_1"], images["unrelated_image"]))
    rows = []
    for pair_name, image_1, image_2 in pairs:
        for detector in settings.detectors_compare:
            started = time.perf_counter()
            try:
                result = match_features(image_1, image_2, detector, settings.ratio_threshold,
                                        settings.use_ransac, settings.max_features,
                                        settings.ransac_threshold_px)
            except ValueError as error:          # too few keypoints: record it, keep going
                rows.append({"pair": pair_name, "detector": detector, "error": str(error)})
                continue
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            stats = result.stats
            rows.append({"pair": pair_name, "detector": detector,
                         "keypoints_1": stats["keypoints_1"], "keypoints_2": stats["keypoints_2"],
                         "good_matches": stats["good_matches"],
                         "ransac_inliers": stats.get("inliers", ""),
                         "inlier_ratio": round(stats["inlier_ratio"], 3) if "inlier_ratio" in stats else "",
                         "time_ms": round(elapsed_ms, 1)})
            canvas = draw_matches(image_1, image_2, result, settings.draw_max_matches)
            path = out_dir / f"05_matches_{pair_name}_{detector}.png"
            save_image(canvas, path)
            figure_paths.append(path)
    return rows


def run_task1b(cfg):
    """[1B.R] Run every Task 1B step, baseline first then advanced variants, and save results.

    Purpose:  One call that produces every Task 1B number and figure for the report.
    Flow:     called by notebook  ->  filtering, frequency, edges, features, visualize
              [1B.1]+[1B.2]+[1B.2m] -> [1B.3] -> [1B.4] -> [1B.5h] -> [1B.5]
              writes results/task1b/ (figures, metrics_filtering.csv, metrics_matching.csv, metrics.json)

    Inputs:
        cfg: config from load_config() (uses cfg.paths, cfg.task1b, cfg.seed).

    Returns:
        Task1BReport.

    Raises:
        FileNotFoundError: one of sample_image / pair_image_1 / pair_image_2 is missing.
    """
    set_seed(cfg.seed)
    settings = cfg.task1b
    out_dir = cfg.paths.results / "task1b"
    out_dir.mkdir(parents=True, exist_ok=True)
    images = _load_inputs(cfg)
    clean_bgr = images["sample_image"]
    clean_gray = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2GRAY)
    figure_paths = []

    filter_scores, filter_rows, scratch_check = _run_filtering(clean_gray, settings, cfg.seed, out_dir,
                                                               figure_paths)
    _run_frequency(clean_gray, settings, out_dir, figure_paths)
    edge_stats = _run_edges(clean_gray, settings, out_dir, figure_paths)
    harris_stats = _run_harris(clean_bgr, clean_gray, settings, out_dir, figure_paths)
    match_rows = _run_matching(images, settings, out_dir, figure_paths)

    save_csv(filter_rows, out_dir / "metrics_filtering.csv")
    save_csv(match_rows, out_dir / "metrics_matching.csv")
    metrics_path = save_json({
        "inputs": {key: list(image.shape) if image is not None else None for key, image in images.items()},
        "settings": {"kernel_size": settings.kernel_size, "gaussian_sigma": settings.gaussian_sigma,
                     "filter_method": settings.filter_method, "fft_radius": settings.fft_radius,
                     "canny": [settings.canny_low, settings.canny_high],
                     "ratio_threshold": settings.ratio_threshold, "use_ransac": settings.use_ransac,
                     "ransac_threshold_px": settings.ransac_threshold_px, "seed": cfg.seed},
        "filter_scores": filter_scores,
        "filter_scratch_vs_opencv_max_abs_diff": scratch_check,
        "edge_stats": edge_stats,
        "harris_corners": harris_stats,
        "matching": match_rows,
        "figures": [path.name for path in figure_paths],
    }, out_dir / "metrics.json")
    return Task1BReport(filter_scores=filter_scores, match_stats=match_rows, figure_paths=figure_paths,
                        edge_stats=edge_stats, harris_stats=harris_stats, metrics_path=metrics_path)
