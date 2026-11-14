"""[VIS] Every figure in the project. Functions take results and return a matplotlib Figure.

Flow:  run_task*() save these figures; the notebook shows them. Images are converted with io_utils.to_rgb().
"""
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.io_utils import to_rgb


def show_grid(images, titles, cols=3, figsize=None, cmaps=None, limits=None, suptitle=None):
    """[VIS.1] Side-by-side images (filters, edges, before/after).

    Inputs:
        images:   list of images. uint8 BGR is shown in colour; uint8 grayscale on a fixed 0..255 scale;
                  float arrays (spectrum, gradients, masks) are auto-scaled.
        titles:   list of str, same length.
        cols:     images per row.
        figsize:  (w, h) inches; default 4.2 x 3.4 per cell.
        cmaps:    optional list of colormap names per image (None = gray).
        limits:   optional list of (vmin, vmax) or None per image (use (-a, a) for signed images).
        suptitle: optional figure title.

    Returns:
        matplotlib.figure.Figure.

    Raises:
        ValueError: images and titles have different lengths.
    """
    if len(images) != len(titles):
        raise ValueError(f"[VIS.1] {len(images)} images but {len(titles)} titles")
    rows = math.ceil(len(images) / cols)
    figsize = figsize or (4.2 * cols, 3.4 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True, squeeze=False)
    cmaps = cmaps or [None] * len(images)
    limits = limits or [None] * len(images)
    for ax in axes.flat:
        ax.axis("off")
    for ax, image, title, cmap, lims in zip(axes.flat, images, titles, cmaps, limits):
        if image.ndim == 3:
            ax.imshow(to_rgb(image))
        else:
            options = {"cmap": cmap or "gray"}
            if lims is not None:
                options.update(vmin=lims[0], vmax=lims[1])
            elif image.dtype == np.uint8:
                options.update(vmin=0, vmax=255)
            ax.imshow(image, **options)
        ax.set_title(title, fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    return fig


def plot_view_errors(view_errors):
    """[VIS.2] Bar chart of per-view reprojection error (px) for Task 1A.

    Inputs:
        view_errors: list of (file name, mean_px, rms_px).

    Returns:
        matplotlib.figure.Figure. The dashed line is the overall mean; bars far above it are views to
        re-take or drop.
    """
    names = [name for name, _, _ in view_errors]
    means = np.array([mean for _, mean, _ in view_errors])
    fig, ax = plt.subplots(figsize=(max(6.0, 0.45 * len(names) + 2), 3.6), constrained_layout=True)
    ax.bar(np.arange(len(names)), means, color="#4C72B0")
    ax.axhline(means.mean(), color="#C44E52", linestyle="--", linewidth=1.2,
               label=f"overall mean {means.mean():.3f} px")
    ax.set_xticks(np.arange(len(names)), names, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("mean reprojection error (px)")
    ax.set_title("[1A.4] Reprojection error per view", fontsize=10)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def draw_matches(image_1, image_2, match_result, max_matches=50):
    """[VIS.3] Draw good (and RANSAC inlier) matches between two images.

    Method:   without RANSAC every drawn match is green; with RANSAC inliers are green and
              outliers red. The strongest (lowest distance) matches are drawn first.

    Inputs:
        image_1, image_2: uint8 images.
        match_result:     features.MatchResult.
        max_matches:      how many to draw.

    Returns:
        np.ndarray uint8 BGR, both images side by side with lines.
    """
    def as_bgr(image):
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image

    matches = match_result.good_matches[:max_matches]
    mask = match_result.inlier_mask
    canvas = cv2.drawMatches(as_bgr(image_1), match_result.keypoints_1, as_bgr(image_2),
                             match_result.keypoints_2, [], None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    for index, match in enumerate(matches):
        inlier = mask is None or bool(mask[index])
        colour = (60, 200, 60) if inlier else (40, 40, 230)
        canvas = cv2.drawMatches(as_bgr(image_1), match_result.keypoints_1, as_bgr(image_2),
                                 match_result.keypoints_2, [match], canvas, matchColor=colour,
                                 flags=cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG
                                 | cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return canvas


def draw_corners(image, corners, radius=4):
    """[VIS.6] Mark Harris corners on a copy of the image.

    Inputs:
        image:   uint8 BGR or grayscale.
        corners: np.ndarray int (M, 2) as (x, y), from features.detect_harris().
        radius:  circle radius in pixels.

    Returns:
        np.ndarray uint8 BGR with red circles.
    """
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    for x, y in corners:
        cv2.circle(canvas, (int(x), int(y)), radius, (0, 0, 255), 1, cv2.LINE_AA)
    return canvas


def plot_filter_scores(scores):
    """[VIS.7] Grouped bar chart of PSNR (dB) per filter, one group per noise type.

    Inputs:
        scores: dict noise label -> {method -> {"mse", "psnr_db"}} (from [1B.2m]).

    Returns:
        matplotlib.figure.Figure.
    """
    noise_labels = list(scores)
    methods = list(next(iter(scores.values())))
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(7.5, 3.6), constrained_layout=True)
    for offset, method in enumerate(methods):
        values = [scores[label][method]["psnr_db"] for label in noise_labels]
        positions = np.arange(len(noise_labels)) + (offset - (len(methods) - 1) / 2) * width
        bars = ax.bar(positions, values, width, label=method)
        ax.bar_label(bars, fmt="%.1f", fontsize=8)
    ax.set_xticks(np.arange(len(noise_labels)), noise_labels)
    ax.set_ylabel("PSNR vs clean (dB, higher is better)")
    ax.legend(fontsize=8, ncols=len(methods), loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def plot_training_history(history, title=""):
    """[VIS.4] Train/val loss and val accuracy / macro F1 per epoch.

    Inputs:
        history: dict from train.train_model() (train_loss, val_loss, train_acc, val_acc, val_macro_f1, best_epoch).
        title:   experiment name.

    Returns:
        matplotlib.figure.Figure. The dotted line marks the epoch whose weights were kept.
    """
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
    ax_loss.plot(epochs, history["train_loss"], label="train loss")
    ax_loss.plot(epochs, history["val_loss"], label="val loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("cross-entropy")
    ax_acc.plot(epochs, history["train_acc"], label="train accuracy")
    ax_acc.plot(epochs, history["val_acc"], label="val accuracy")
    ax_acc.plot(epochs, history["val_macro_f1"], label="val macro F1", linestyle="--")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylim(0, 1)
    for ax in (ax_loss, ax_acc):
        ax.axvline(history.get("best_epoch", epochs[-1]), color="grey", linestyle=":", linewidth=1)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"[2.6] Training history {title}", fontsize=11)
    return fig


def plot_confusion_matrix(confusion, classes, title=""):
    """[VIS.5] Confusion matrix heatmap with counts.

    Inputs:
        confusion: (n, n) array, rows = true.
        classes:   class names.
        title:     experiment name.

    Returns:
        matplotlib.figure.Figure. Diagonal = correct; each off-diagonal cell = how often the row class was
        predicted as the column class.
    """
    confusion = np.asarray(confusion)
    fig, ax = plt.subplots(figsize=(4.8, 4.2), constrained_layout=True)
    ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=40, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    threshold = confusion.max() / 2
    for row in range(len(classes)):
        for col in range(len(classes)):
            ax.text(col, row, int(confusion[row, col]), ha="center", va="center", fontsize=9,
                    color="white" if confusion[row, col] > threshold else "black")
    ax.set_title(f"[2.7] Test confusion matrix {title}", fontsize=10)
    return fig


def plot_class_counts(class_counts, raw_counts=None):
    """[VIS.8] Images per class in each split (and in the raw data), to show balance and stratification.

    Inputs:
        class_counts: dict split -> {class -> count}.
        raw_counts:   optional {class -> count} before cleaning.

    Returns:
        matplotlib.figure.Figure.
    """
    classes = list(next(iter(class_counts.values())))
    series = ({"raw": raw_counts} if raw_counts else {}) | class_counts
    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(8, 3.4), constrained_layout=True)
    for offset, (name, counts) in enumerate(series.items()):
        positions = np.arange(len(classes)) + (offset - (len(series) - 1) / 2) * width
        bars = ax.bar(positions, [counts[c] for c in classes], width, label=name)
        ax.bar_label(bars, fontsize=7)
    ax.set_xticks(np.arange(len(classes)), classes)
    ax.set_ylabel("images")
    ax.legend(fontsize=8)
    ax.set_title("[2.1]-[2.3] Images per class", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def show_class_examples(table, classes, per_class=4, seed=0):
    """[VIS.9] A few random training images per class (one row per class).

    Inputs:
        table:     manifest DataFrame with path and label.
        classes:   class names, row order.
        per_class: images per row.

    Returns:
        matplotlib.figure.Figure.
    """
    images, titles = [], []
    for name in classes:
        rows = table[table.label == name].sample(per_class, random_state=seed)
        for path, width, height in zip(rows.path, rows.width, rows.height):
            images.append(cv2.imread(str(path)))
            titles.append(f"{name}  {width}x{height}")
    return show_grid(images, titles, cols=per_class, figsize=(2.6 * per_class, 2.2 * len(classes)))


def show_augmentations(path, image_size, count=7, seed=0):
    """[VIS.10] One training image, its eval version and several random augmentations of it.

    Inputs:
        path:       image file.
        image_size: model input size.
        count:      number of augmented samples.

    Returns:
        matplotlib.figure.Figure.
    """
    import torch

    from src.data_pipeline import IMAGENET_MEAN, IMAGENET_STD, load_rgb, make_transforms

    torch.manual_seed(seed)
    image = load_rgb(path)
    mean, std = torch.tensor(IMAGENET_MEAN).view(3, 1, 1), torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def to_bgr(tensor):
        rgb = ((tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    eval_view = to_bgr(make_transforms(image_size, train=False)(image))
    train_tf = make_transforms(image_size, train=True)
    views = [eval_view] + [to_bgr(train_tf(image)) for _ in range(count)]
    titles = ["eval transform (resize + centre crop)"] + [f"augmented #{i + 1}" for i in range(count)]
    return show_grid(views, titles, cols=4, figsize=(11, 6), suptitle="[2.4] Train-time augmentation")


def show_misclassified(misclassified, max_images=8, title=""):
    """[VIS.11] The most confident mistakes: true label, predicted label and score.

    Inputs:
        misclassified: list of (path, true, predicted, score) from evaluate.EvalResult.
        max_images:    how many to show.

    Returns:
        matplotlib.figure.Figure.
    """
    picks = misclassified[:max_images]
    if not picks:
        fig, ax = plt.subplots(figsize=(4, 1))
        ax.axis("off")
        ax.text(0.5, 0.5, "no misclassified images", ha="center")
        return fig
    images = [cv2.imread(str(path)) for path, _, _, _ in picks]
    titles = [f"true {true} -> {pred} ({score:.2f})" for _, true, pred, score in picks]
    return show_grid(images, titles, cols=4, figsize=(12, 3.0 * ((len(picks) + 3) // 4)),
                     suptitle=f"[2.7] Most confident test mistakes {title}")


def plot_tuning(trials, keys, title=""):
    """[VIS.12] Tuning results: validation macro F1 per setting (bars) and per epoch (curves).

    Inputs:
        trials: list of dicts from train.tune_hyperparameters() (values, val_macro_f1, curve).
        keys:   tuned parameter names.
        title:  experiment name.

    Returns:
        matplotlib.figure.Figure. The best setting is the tallest bar (highlighted).
    """
    labels = [", ".join(f"{k}={t[k]:g}" if isinstance(t[k], float) else f"{k}={t[k]}" for k in keys) for t in trials]
    scores = [t["val_macro_f1"] for t in trials]
    best = int(np.argmax(scores))
    fig, (ax_bar, ax_curve) = plt.subplots(1, 2, figsize=(11, 3.4), constrained_layout=True)
    bars = ax_bar.bar(range(len(trials)), scores, color=["#DD8452" if i == best else "#4C72B0" for i in range(len(trials))])
    ax_bar.bar_label(bars, fmt="%.3f", fontsize=8)
    ax_bar.set_xticks(range(len(trials)), labels, rotation=20, ha="right", fontsize=8)
    ax_bar.set_ylabel("best val macro F1")
    ax_bar.set_ylim(0, 1)
    for label, trial in zip(labels, trials):
        ax_curve.plot(range(1, len(trial["curve"]) + 1), trial["curve"], marker="o", label=label)
    ax_curve.set_xlabel("epoch (short tuning budget)")
    ax_curve.set_ylabel("val macro F1")
    ax_curve.legend(fontsize=8)
    for ax in (ax_bar, ax_curve):
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"[2.6t] Hyperparameter tuning {title}", fontsize=11)
    return fig


def plot_tradeoff(rows):
    """[VIS.13] Accuracy vs CPU latency for every model; bubble area = size on disk.

    Inputs:
        rows: experiment rows from train.run_task2() (test_macro_f1, latency_ms, size_MB, experiment).

    Returns:
        matplotlib.figure.Figure. Top-left = accurate and fast (best trade-off).
    """
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for row in rows:
        ax.scatter(row["latency_ms"], row["test_macro_f1"], s=60 + 40 * row["size_MB"], alpha=0.6)
        ax.annotate(f"{row['experiment']}\n{row['size_MB']} MB, {row['train_minutes']} min train",
                    (row["latency_ms"], row["test_macro_f1"]), textcoords="offset points", xytext=(8, -4), fontsize=8)
    ax.set_xlabel("CPU latency per image (ms, lower is better)")
    ax.set_ylabel("test macro F1 (higher is better)")
    ax.set_title("[2.7b] Model trade-off: quality vs speed vs size", fontsize=10)
    ax.margins(x=0.35, y=0.15)
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def plot_method_ladder(table):
    """[VIS.14] Test macro F1 of every method, grouped and coloured by family (chance / classical ML / deep learning).

    Inputs:
        table: DataFrame with columns label, family, test_macro_f1 (one row per method).

    Returns:
        matplotlib.figure.Figure, methods sorted from worst to best.
    """
    palette = {"chance baseline": "#8C8C8C", "classical CV + ML": "#4C72B0",
               "pretrained features + ML": "#55A868", "deep learning": "#DD8452"}
    table = table.sort_values("test_macro_f1")
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(table) + 1.2), constrained_layout=True)
    bars = ax.barh(table["label"], table["test_macro_f1"], color=[palette.get(f, "#999999") for f in table["family"]])
    ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("test macro F1")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in palette.values()]
    ax.legend(handles, palette.keys(), fontsize=8, loc="lower right")
    ax.set_title("[2.9] From chance to deep learning: every method on the same test split", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    return fig
