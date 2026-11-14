"""[2.9] Task 2 - non-deep-learning baselines, to show what deep learning adds (rubric 2.2, 2.3).

Flow:  run_baselines(cfg)
         -> [2.9a] extract_handcrafted_features   (colour histogram + HOG + LBP, classical CV from Task 1)
         -> [2.9b] extract_deep_features           (frozen ImageNet MobileNetV3, no fine-tuning)
         -> [2.9c] fit_baseline                    (dummy / logistic regression / SVM / random forest / k-NN,
                                                    small grid on validation)
         -> [2.9d] group_permutation_importance    (which feature family the best ML model relies on)
       writes results/task2/baselines.csv, baselines.json

Same train/val/test split as the CNNs (data/splits/*.csv), same metrics, so every row is comparable.
"""
import io
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEATURE_SIZE = 128                  # images are centre-cropped to this square before hand-crafted features
HOG_CELL, HOG_BINS = 16, 9          # HOG: 16 px cells, 2 x 2-cell blocks, 9 unsigned orientation bins


# ----------------------------------------------------------------------------------------------------------- [2.9a]
def _centre_square(bgr, size=FEATURE_SIZE):
    """Resize the short side to size and centre-crop a square (same framing as the CNN eval transform)."""
    height, width = bgr.shape[:2]
    scale = size / min(height, width)
    resized = cv2.resize(bgr, (max(size, round(width * scale)), max(size, round(height * scale))),
                         interpolation=cv2.INTER_AREA)
    top, left = (resized.shape[0] - size) // 2, (resized.shape[1] - size) // 2
    return resized[top:top + size, left:left + size]


def local_binary_pattern(gray):
    """[2.9a-lbp] 8-neighbour LBP code per pixel: bit k = neighbour k >= centre (texture descriptor, from scratch)."""
    centre = gray[1:-1, 1:-1].astype(np.int16)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    codes = np.zeros_like(centre, dtype=np.uint8)
    height, width = gray.shape
    for bit, (dy, dx) in enumerate(offsets):
        neighbour = gray[1 + dy:height - 1 + dy, 1 + dx:width - 1 + dx].astype(np.int16)
        codes |= ((neighbour >= centre).astype(np.uint8) << bit)
    return codes


def histogram_of_gradients(gray, cell=HOG_CELL, bins=HOG_BINS):
    """[2.9a-hog] HOG descriptor from scratch (Dalal & Triggs): local edge-direction histograms.

    Method:   Sobel gradients -> magnitude and unsigned orientation (0-180 deg) -> per cell, a 9-bin histogram
              of orientations weighted by magnitude -> blocks of 2 x 2 cells (stride 1 cell) normalised with L2-Hys
              (L2, clip at 0.2, L2 again) so the descriptor ignores local contrast.

    Inputs:
        gray: np.ndarray uint8 (S, S), S divisible by cell.

    Returns:
        np.ndarray float32 ((S/cell - 1)^2 * 4 * bins,), 1764 values for 128 px.
    """
    image = gray.astype(np.float32)
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=1)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=1)
    magnitude = np.hypot(grad_x, grad_y)
    angle = np.rad2deg(np.arctan2(grad_y, grad_x)) % 180.0
    bin_index = np.minimum((angle / (180.0 / bins)).astype(int), bins - 1)
    cells_y, cells_x = gray.shape[0] // cell, gray.shape[1] // cell
    histograms = np.zeros((cells_y, cells_x, bins), np.float32)
    for b in range(bins):
        weight = np.where(bin_index == b, magnitude, 0.0)
        histograms[:, :, b] = weight[:cells_y * cell, :cells_x * cell].reshape(cells_y, cell, cells_x, cell).sum(axis=(1, 3))
    blocks = []
    for y in range(cells_y - 1):
        for x in range(cells_x - 1):
            block = histograms[y:y + 2, x:x + 2].ravel()
            block = block / (np.linalg.norm(block) + 1e-6)
            block = np.minimum(block, 0.2)
            blocks.append(block / (np.linalg.norm(block) + 1e-6))
    return np.concatenate(blocks).astype(np.float32)


def handcrafted_feature_groups():
    """Names and lengths of the three feature families, in vector order."""
    return {"colour (HSV histogram)": 16 * 4 * 4, "shape (HOG)": (FEATURE_SIZE // HOG_CELL - 1) ** 2 * 4 * HOG_BINS, "texture (LBP)": 256}


def extract_handcrafted_features(bgr):
    """[2.9a] Classical image descriptor: colour + shape + texture, concatenated.

    Purpose:  The pre-deep-learning way to classify images: describe them with fixed, hand-designed features
              (built from Task 1 ideas: gradients -> HOG, local intensity patterns -> LBP) and train an ML model.
    Flow:     called by run_baselines() for every image
    Method:   centre-crop to 128 x 128, then
              colour:  HSV joint histogram 16 (hue) x 4 (saturation) x 4 (value) = 256 bins, sums to 1;
              shape:   HOG (9 orientation bins, 16 px cells, 32 px blocks) = 1764 values;
              texture: histogram of 8-neighbour LBP codes = 256 bins, sums to 1.

    Inputs:
        bgr: np.ndarray uint8 (H, W, 3).

    Returns:
        np.ndarray float32 (2276,)
    """
    square = _centre_square(bgr)
    hsv = cv2.cvtColor(square, cv2.COLOR_BGR2HSV)
    colour = cv2.calcHist([hsv], [0, 1, 2], None, [16, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    colour /= max(colour.sum(), 1)
    gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
    shape = histogram_of_gradients(gray)
    texture = np.bincount(local_binary_pattern(gray).ravel(), minlength=256).astype(np.float32)
    texture /= max(texture.sum(), 1)
    return np.concatenate([colour, shape, texture]).astype(np.float32)


# ----------------------------------------------------------------------------------------------------------- [2.9b]
def extract_deep_features(paths, image_size=160, batch_size=64):
    """[2.9b] 1280-d embeddings from MobileNetV3 pretrained on ImageNet, NOT fine-tuned (a "linear probe" input).

    Purpose:  Separates two effects: how much comes from pretrained features alone, and how much from fine-tuning.
    Flow:     called by run_baselines()  ->  model.build_model("mobilenet_v3") + data_pipeline eval transform

    Inputs:
        paths:      list of image paths.
        image_size: input size (same as the fine-tuned model).
        batch_size: batch size.

    Returns:
        np.ndarray float32 (N, 1280)
    """
    import torch

    from src.data_pipeline import load_rgb, make_transforms
    from src.model import build_model

    model = build_model("mobilenet_v3", 1000).eval()
    model.reset_classifier(0)                       # outputs the pooled 1280-d features
    transform = make_transforms(image_size, train=False)
    features = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch = torch.stack([transform(load_rgb(p)) for p in paths[start:start + batch_size]])
            features.append(model(batch).numpy())
    return np.concatenate(features).astype(np.float32)


# ----------------------------------------------------------------------------------------------------------- [2.9c]
BASELINE_GRIDS = {
    "majority class (dummy)": (lambda: DummyClassifier(strategy="most_frequent"), [{}]),
    "random by class share (dummy)": (lambda: DummyClassifier(strategy="stratified", random_state=0), [{}]),
    "k-NN": (lambda **kw: make_pipeline(StandardScaler(), KNeighborsClassifier(**kw)),
             [{"n_neighbors": k} for k in (5, 15, 31)]),
    "logistic regression": (lambda **kw: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, **kw)),
                            [{"C": c} for c in (0.01, 0.1, 1.0)]),
    "SVM (RBF kernel)": (lambda **kw: make_pipeline(StandardScaler(), SVC(kernel="rbf", **kw)),
                         [{"C": c} for c in (1.0, 10.0)]),
    "random forest": (lambda **kw: RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0, **kw),
                      [{"max_features": m} for m in ("sqrt", 0.05)]),
}


@dataclass
class BaselineResult:
    """Output of [2.9c] for one model on one feature set."""
    name: str
    features: str
    params: dict
    val_macro_f1: float
    test_accuracy: float
    test_macro_f1: float
    fit_seconds: float
    predict_ms_per_image: float           # classifier only; feature extraction time is added by the runner
    size_MB: float
    model: object = field(default=None, repr=False)


def fit_baseline(name, features_name, x_train, y_train, x_val, y_val, x_test, y_test):
    """[2.9c] Fit one baseline: try each grid setting on train, keep the best on validation, score on test.

    Flow:     called by run_baselines()
    Method:   same protocol as the CNN tuning ([2.6t]): validation chooses, test is scored once.

    Inputs:
        name:          key of BASELINE_GRIDS.
        features_name: label of the feature set (for the table).
        x_*, y_*:      feature matrices and integer labels of the three splits.

    Returns:
        BaselineResult.
    """
    factory, grid = BASELINE_GRIDS[name]
    best = None
    for params in grid:
        started = time.perf_counter()
        model = factory(**params).fit(x_train, y_train)
        fit_seconds = time.perf_counter() - started
        val_f1 = f1_score(y_val, model.predict(x_val), average="macro")
        if best is None or val_f1 > best[0]:
            best = (val_f1, params, model, fit_seconds)
    val_f1, params, model, fit_seconds = best
    started = time.perf_counter()
    predicted = model.predict(x_test)
    predict_ms = 1000 * (time.perf_counter() - started) / len(x_test)
    buffer = io.BytesIO()
    pickle.dump(model, buffer)
    return BaselineResult(name=name, features=features_name, params=params, val_macro_f1=float(val_f1),
                          test_accuracy=float(accuracy_score(y_test, predicted)),
                          test_macro_f1=float(f1_score(y_test, predicted, average="macro")),
                          fit_seconds=round(fit_seconds, 2), predict_ms_per_image=round(predict_ms, 3),
                          size_MB=round(buffer.getbuffer().nbytes / 2 ** 20, 2), model=model)


# ----------------------------------------------------------------------------------------------------------- [2.9d]
def group_permutation_importance(model, x_val, y_val, groups, repeats=5, seed=0):
    """[2.9d] Drop in validation macro F1 when one whole feature family is shuffled across images (XAI for ML).

    Purpose:  Explains WHAT the classical model uses: if shuffling the colour columns ruins the score, the model
              decides mostly by colour. Shuffling breaks the link between that family and the label while keeping
              its distribution.
    Flow:     called by run_baselines() on the best hand-crafted-feature model

    Inputs:
        model:   fitted classifier.
        x_val:   validation features (N, D).
        y_val:   validation labels.
        groups:  dict name -> length, in column order (from handcrafted_feature_groups()).
        repeats: shuffles per group (the mean is reported).

    Returns:
        dict group -> {"f1_drop": mean drop, "std": spread}
    """
    rng = np.random.default_rng(seed)
    baseline = f1_score(y_val, model.predict(x_val), average="macro")
    result, start = {}, 0
    for group, length in groups.items():
        drops = []
        for _ in range(repeats):
            shuffled = x_val.copy()
            shuffled[:, start:start + length] = shuffled[rng.permutation(len(x_val)), start:start + length]
            drops.append(baseline - f1_score(y_val, model.predict(shuffled), average="macro"))
        result[group] = {"f1_drop": float(np.mean(drops)), "std": float(np.std(drops))}
        start += length
    return result


# ----------------------------------------------------------------------------------------------------------- [2.9.R]
def run_baselines(cfg):
    """[2.9.R] Train every baseline on the saved Task 2 split and write the comparison table.

    Purpose:  Answer "is deep learning worth it here?" with numbers: chance level (dummy), classical CV features
              + ML, pretrained deep features + a linear model, against the CNNs from run_task2().
    Flow:     called by notebook after run_task2()  ->  [2.9a] -> [2.9b] -> [2.9c] -> [2.9d]
              writes results/task2/baselines.csv, baselines.json (features are cached in features_cache.npz)

    Inputs:
        cfg: config from load_config().

    Returns:
        dict with "rows" (table), "importance" (from [2.9d]), "feature_seconds_per_image".

    Raises:
        FileNotFoundError: data/splits/*.csv missing (run run_task2 first).
    """
    from src.io_utils import read_image, save_csv, save_json

    out_dir = cfg.paths.results / "task2"
    splits = {}
    for name in ("train", "val", "test"):
        path = cfg.paths.splits / f"{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"[2.9.R] {path} missing: run run_task2(cfg) first")
        splits[name] = pd.read_csv(path)
    classes = sorted(splits["train"].label.unique())
    resolve = lambda p: Path(p) if Path(p).is_absolute() else cfg.project_root / p

    cache = out_dir / "features_cache.npz"
    all_paths = [str(p) for name in ("train", "val", "test") for p in splits[name].path]
    if cache.is_file() and list(np.load(cache, allow_pickle=True)["paths"]) == all_paths:
        data = np.load(cache, allow_pickle=True)
        handcrafted, deep = data["handcrafted"], data["deep"]
        hand_seconds, deep_seconds = float(data["hand_seconds"]), float(data["deep_seconds"])
    else:
        started = time.perf_counter()
        handcrafted = np.stack([extract_handcrafted_features(read_image(resolve(p))) for p in all_paths])
        hand_seconds = (time.perf_counter() - started) / len(all_paths)
        started = time.perf_counter()
        deep = extract_deep_features([resolve(p) for p in all_paths])
        deep_seconds = (time.perf_counter() - started) / len(all_paths)
        np.savez_compressed(cache, handcrafted=handcrafted, deep=deep, paths=np.array(all_paths, dtype=object),
                            hand_seconds=hand_seconds, deep_seconds=deep_seconds)

    sizes = [len(splits[name]) for name in ("train", "val", "test")]
    cut = np.cumsum(sizes)[:-1]
    labels = np.array([classes.index(l) for name in ("train", "val", "test") for l in splits[name].label])
    y_train, y_val, y_test = np.split(labels, cut)
    feature_sets = {"hand-crafted (colour+HOG+LBP)": (handcrafted, hand_seconds),
                    "ImageNet MobileNetV3 features (frozen)": (deep, deep_seconds)}
    rows, results = [], {}
    for name in BASELINE_GRIDS:
        sets = list(feature_sets.items())[:1] if "dummy" in name else feature_sets.items()
        for features_name, (matrix, seconds) in sets:
            if "ImageNet" in features_name and name not in ("logistic regression", "SVM (RBF kernel)"):
                continue                                  # a linear/kernel probe is the standard test for embeddings
            x_train, x_val, x_test = np.split(matrix, cut)
            result = fit_baseline(name, features_name, x_train, y_train, x_val, y_val, x_test, y_test)
            results[(name, features_name)] = result
            rows.append({"model": name, "features": "none (ignores the image)" if "dummy" in name else features_name,
                         "chosen_params": str(result.params), "val_macro_f1": round(result.val_macro_f1, 4),
                         "test_accuracy": round(result.test_accuracy, 4), "test_macro_f1": round(result.test_macro_f1, 4),
                         "fit_seconds": result.fit_seconds,
                         "latency_ms": round(result.predict_ms_per_image + (0 if "dummy" in name else 1000 * seconds), 2),
                         "size_MB": result.size_MB})
    hand_rows = {key: r for key, r in results.items() if key[1].startswith("hand") and "dummy" not in key[0]}
    best_key = max(hand_rows, key=lambda key: hand_rows[key].val_macro_f1)
    x_val = np.split(handcrafted, cut)[1]
    importance = group_permutation_importance(results[best_key].model, x_val, y_val, handcrafted_feature_groups())
    save_csv(rows, out_dir / "baselines.csv")
    summary = {"rows": rows, "importance_model": best_key[0], "importance": importance,
               "feature_seconds_per_image": {"handcrafted": hand_seconds, "deep_frozen": deep_seconds}}
    save_json(summary, out_dir / "baselines.json")
    return summary
