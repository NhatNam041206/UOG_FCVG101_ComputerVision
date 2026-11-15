"""[2.10] Task 2 - explainable AI (XAI): why the models decide what they decide (rubric 2.3).

Flow:  run_explanations(cfg)
         -> [2.10a] grad_cam                 where in the image the CNN looks for its decision
         -> [2.10b] deletion_test            is that explanation faithful? (masking the hot pixels must hurt)
         -> [2.10c] calibration_report       are the confidence scores trustworthy? (reliability, ECE)
         -> [2.10d] embedding_separation     how well each representation separates the classes (5-NN accuracy, silhouette, t-SNE)
       writes results/task2/xai_*.png and xai.json
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from src.data_pipeline import load_rgb, make_transforms

_trapz = getattr(np, "trapezoid", None) or np.trapz       # numpy 2 renamed trapz; Colab and older installs differ


# ---------------------------------------------------------------------------------------------------------- [2.10a]
def target_layer(model):
    """The last convolutional feature map of each architecture (what Grad-CAM explains)."""
    if hasattr(model, "features"):                      # SmallCNN: last block's ReLU (before the final pool)
        return model.features[-2]
    return model.blocks[-1]                              # timm MobileNetV3: last inverted-residual stage


def grad_cam(model, tensor, class_index=None):
    """[2.10a] Grad-CAM heatmap (Selvaraju et al., 2017).

    Purpose:  Show which image regions raised the score of the predicted class.
    Method:   forward pass keeps the last conv feature maps A_k; backward pass gives d(score_c)/dA_k;
              weight of map k = mean of its gradient; heatmap = ReLU(sum_k weight_k * A_k), upsampled to the
              input size and scaled to 0..1. Red = evidence FOR the class.

    Inputs:
        model:       nn.Module in eval mode.
        tensor:      (1, 3, S, S) normalised input.
        class_index: class to explain (default: the predicted one).

    Returns:
        (heatmap: np.ndarray float32 (S, S) in 0..1, class_index: int, probabilities: np.ndarray (C,))
    """
    store = {}
    layer = target_layer(model)
    def keep(module, inputs, output):
        store["act"] = output
        output.register_hook(lambda grad: store.__setitem__("grad", grad))     # gradient w.r.t. this feature map

    forward_hook = layer.register_forward_hook(keep)
    try:
        model.eval()
        tensor = tensor.clone().requires_grad_(True)
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].detach().numpy()
        class_index = int(logits.argmax(1)) if class_index is None else class_index
        model.zero_grad()
        logits[0, class_index].backward()
        weights = store["grad"].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * store["act"]).sum(dim=1))[0].detach().numpy()
    finally:
        forward_hook.remove()
    cam = cv2.resize(cam, (tensor.shape[3], tensor.shape[2]), interpolation=cv2.INTER_LINEAR)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.astype(np.float32), class_index, probabilities


def overlay_heatmap(rgb, heatmap, alpha=0.45):
    """[2.10a-vis] Blend a 0..1 heatmap (JET colours) over an RGB uint8 image; returns BGR uint8."""
    coloured = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 1 - alpha, coloured, alpha, 0)


def model_input(path, image_size):
    """Eval-transformed tensor (1, 3, S, S) and the matching un-normalised RGB crop (S, S, 3) for display."""
    image = load_rgb(path)
    tensor = make_transforms(image_size, train=False)(image).unsqueeze(0)
    from torchvision import transforms
    crop = transforms.CenterCrop(image_size)(transforms.Resize(int(round(image_size * 1.14)))(image))
    return tensor, np.asarray(crop)


# ---------------------------------------------------------------------------------------------------------- [2.10b]
@torch.no_grad()
def _class_probability(model, batch, class_index):
    return torch.softmax(model(batch), dim=1)[:, class_index].numpy()


def deletion_test(model, tensor, heatmap, class_index, fractions=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5), seed=0):
    """[2.10b] Faithfulness check: grey out the most important pixels first and watch the score fall.

    Purpose:  A heatmap can look plausible and still be wrong. If Grad-CAM marks the pixels the model really uses,
              removing them should drop the class probability much faster than removing random pixels.
    Method:   for each fraction f, replace the top-f pixels by Grad-CAM (or a random f of pixels) with 0, i.e. the
              dataset mean colour after normalisation; record the probability of the originally predicted class.

    Inputs:
        model, tensor, heatmap, class_index: from grad_cam().
        fractions: shares of pixels removed.

    Returns:
        dict "fractions", "gradcam" (probabilities), "random" (probabilities)
    """
    order = np.argsort(heatmap.ravel())[::-1]
    random_order = np.random.default_rng(seed).permutation(order.size)
    batches = {"gradcam": [], "random": []}
    for fraction in fractions:
        for key, ranking in (("gradcam", order), ("random", random_order)):
            mask = np.ones(order.size, np.float32)
            mask[ranking[:int(fraction * order.size)]] = 0.0
            batches[key].append(tensor[0] * torch.from_numpy(mask.reshape(heatmap.shape)))
    return {"fractions": list(fractions),
            "gradcam": _class_probability(model, torch.stack(batches["gradcam"]), class_index).tolist(),
            "random": _class_probability(model, torch.stack(batches["random"]), class_index).tolist()}


# ---------------------------------------------------------------------------------------------------------- [2.10c]
def calibration_report(confidences, correct, bins=10):
    """[2.10c] Reliability of the confidence scores: expected calibration error (ECE).

    Method:   group predictions by confidence (10 bins); in each bin compare mean confidence with accuracy.
              ECE = sum over bins of (bin share) x |accuracy - confidence|. 0 = perfectly calibrated.

    Inputs:
        confidences: np.ndarray (N,) top softmax score.
        correct:     np.ndarray bool (N,).

    Returns:
        dict ece, bins (list of {low, high, count, confidence, accuracy})
    """
    edges = np.linspace(0, 1, bins + 1)
    rows, ece = [], 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        in_bin = (confidences > low) & (confidences <= high)
        if in_bin.any():
            conf, acc = float(confidences[in_bin].mean()), float(correct[in_bin].mean())
            ece += in_bin.mean() * abs(acc - conf)
            rows.append({"low": low, "high": high, "count": int(in_bin.sum()), "confidence": conf, "accuracy": acc})
    return {"ece": float(ece), "bins": rows}


# ---------------------------------------------------------------------------------------------------------- [2.10d]
@torch.no_grad()
def penultimate_features(model, paths, image_size, batch_size=64):
    """[2.10d-a] Features just before the classifier layer (what the last linear layer separates)."""
    transform = make_transforms(image_size, train=False)
    outputs = []
    for start in range(0, len(paths), batch_size):
        batch = torch.stack([transform(load_rgb(p)) for p in paths[start:start + batch_size]])
        if hasattr(model, "forward_head"):
            outputs.append(model.forward_head(model.forward_features(batch), pre_logits=True).numpy())
        else:
            outputs.append(model.classifier[:-1](model.features(batch)).numpy())
    return np.concatenate(outputs)


def embedding_separation(features, labels, seed=0):
    """[2.10d] How well a representation separates the classes: 5-NN accuracy, silhouette score, 2-D t-SNE map.

    Method:   5-NN accuracy (cosine distance, 5-fold cross-validation): share of images whose nearest neighbours
              carry the same class -> 1 = classes form clean neighbourhoods, 1/n_classes = mixed. Robust in high
              dimensions, so it is the main number.
              Silhouette (cosine) of the true labels: +1 = tight, separate clusters, 0 = overlapping. Strict in high
              dimensions (values around 0.2 are already good there), so it is reported as a second view.
              t-SNE (perplexity 30) of the L2-normalised features, only for the picture (t-SNE distances are not meaningful).

    Inputs:
        features: (N, D) array.
        labels:   (N,) integer labels.

    Returns:
        (knn_accuracy: float, silhouette: float, points_2d: np.ndarray (N, 2))
    """
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import normalize

    knn = float(cross_val_score(KNeighborsClassifier(5, metric="cosine"), features, labels, cv=5).mean())
    score = float(silhouette_score(features, labels, metric="cosine"))
    points = TSNE(n_components=2, perplexity=30, init="pca", random_state=seed).fit_transform(normalize(features))
    return knn, score, points


# ---------------------------------------------------------------------------------------------------------- [2.10.R]
def run_explanations(cfg, n_deletion=60, seed=0):
    """[2.10.R] Run every XAI analysis on the saved models and the test split; save figures and numbers.

    Purpose:  Turn "the model scores 0.94" into evidence of WHY: where it looks, whether that explanation is
              faithful, whether its confidence can be trusted, and how its learned features compare.
    Flow:     called by notebook after run_task2() and run_baselines()
              -> inference.load_model for each artifacts/models/<experiment> -> [2.10a]-[2.10d]

    Inputs:
        cfg:        config from load_config().
        n_deletion: test images used for the deletion test (per model).
        seed:       sampling seed.

    Returns:
        dict of all numbers (also saved to results/task2/xai.json).

    Raises:
        FileNotFoundError: trained models or splits missing (run run_task2 first).
    """
    import matplotlib.pyplot as plt

    from src.inference import load_model
    from src.io_utils import save_json
    from src.visualize import show_grid

    out_dir = cfg.paths.results / "task2"
    resolve = lambda p: Path(p) if Path(p).is_absolute() else cfg.project_root / p
    test = pd.read_csv(cfg.paths.splits / "test.csv")
    test["path"] = [str(resolve(p)) for p in test.path]
    models = {}
    for folder in sorted((cfg.paths.artifacts / "models").iterdir()):
        if (folder / "model.pth").is_file():
            models[folder.name] = load_model(folder / "model.pth", folder / "classes.json")
    if not models:
        raise FileNotFoundError("[2.10.R] no models in artifacts/models/: run run_task2(cfg) first")
    classes = next(iter(models.values()))[1]["classes"]
    labels = np.array([classes.index(l) for l in test.label])
    rng = np.random.default_rng(seed)
    results, figure_paths = {"models": {}}, []

    def save(fig, name):
        fig.savefig(out_dir / name, dpi=130)
        plt.close(fig)
        figure_paths.append(out_dir / name)

    # [2.10a] the same test images explained by every model: 2 per class, chosen at random
    picks = [rng.choice(np.nonzero(labels == c)[0]) for c in range(len(classes)) for _ in range(2)]
    images, titles = [], []
    for index in picks:
        row = test.iloc[index]
        largest = max(meta["image_size"] for _, meta in models.values())
        images.append(cv2.cvtColor(model_input(row.path, largest)[1], cv2.COLOR_RGB2BGR))
        titles.append(f"true: {row.label}")
        for name, (model, meta) in models.items():
            tensor, rgb = model_input(row.path, meta["image_size"])
            cam, predicted, probabilities = grad_cam(model, tensor)
            images.append(overlay_heatmap(rgb, cam))
            mark = "" if classes[predicted] == row.label else "  WRONG"
            titles.append(f"{name}: {classes[predicted]} {probabilities[predicted]:.2f}{mark}")
    save(show_grid(images, titles, cols=len(models) + 1, figsize=(3.2 * (len(models) + 1), 3.1 * len(picks)),
                   suptitle="[2.10a] Grad-CAM: where each model looks (red = evidence for the predicted class)"),
         "xai_gradcam_compare.png")

    for name, (model, meta) in models.items():
        info = {}
        # [2.10a] mistakes of this model, explained
        paths = list(test.path)
        tensors = [model_input(p, meta["image_size"])[0] for p in paths]
        with torch.no_grad():
            probabilities = torch.cat([torch.softmax(model(torch.cat(tensors[i:i + 64])), 1)
                                       for i in range(0, len(tensors), 64)]).numpy()
        predicted = probabilities.argmax(1)
        confidence = probabilities.max(1)
        wrong = np.nonzero(predicted != labels)[0]
        wrong = wrong[np.argsort(-confidence[wrong])][:6]
        images, titles = [], []
        for index in wrong:
            tensor, rgb = model_input(paths[index], meta["image_size"])
            cam, cls, prob = grad_cam(model, tensor)
            images += [cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), overlay_heatmap(rgb, cam)]
            titles += [f"true {classes[labels[index]]}", f"pred {classes[cls]} ({prob[cls]:.2f})"]
        if images:
            save(show_grid(images, titles, cols=4, figsize=(12, 3.0 * ((len(images) + 3) // 4)),
                           suptitle=f"[2.10a] Grad-CAM on the most confident mistakes: {name}"),
                 f"xai_mistakes_{name}.png")

        # [2.10b] deletion test on random test images
        curves = {"gradcam": [], "random": []}
        for index in rng.choice(len(paths), size=min(n_deletion, len(paths)), replace=False):
            tensor = tensors[index]
            cam, cls, _ = grad_cam(model, tensor)
            curve = deletion_test(model, tensor, cam, cls)
            curves["gradcam"].append(curve["gradcam"])
            curves["random"].append(curve["random"])
        fractions = curve["fractions"]
        mean_cam, mean_rand = np.mean(curves["gradcam"], 0), np.mean(curves["random"], 0)
        info["deletion"] = {"fractions": fractions, "gradcam": mean_cam.tolist(), "random": mean_rand.tolist(),
                            "auc_gradcam": float(_trapz(mean_cam, fractions)),
                            "auc_random": float(_trapz(mean_rand, fractions))}
        # [2.10c] calibration
        info["calibration"] = calibration_report(confidence, predicted == labels)
        info["accuracy"] = float((predicted == labels).mean())
        # [2.10d] learned representation
        features = penultimate_features(model, paths, meta["image_size"])
        info["knn_accuracy"], info["silhouette"], points = embedding_separation(features, labels, seed)
        info["tsne"] = points.tolist()
        results["models"][name] = info

    # [2.10d] the hand-crafted descriptor of the baselines, for comparison
    cache = out_dir / "features_cache.npz"
    if cache.is_file():
        data = np.load(cache, allow_pickle=True)
        n_test = len(test)
        hand = data["handcrafted"][-n_test:]
        knn, score, points = embedding_separation(hand, labels, seed)
        results["handcrafted"] = {"knn_accuracy": knn, "silhouette": score, "tsne": points.tolist()}
        deep = data["deep"][-n_test:]
        knn, score, points = embedding_separation(deep, labels, seed)
        results["imagenet_frozen"] = {"knn_accuracy": knn, "silhouette": score, "tsne": points.tolist()}

    # figures: deletion curves, reliability diagrams, t-SNE maps
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 3.4), constrained_layout=True, squeeze=False)
    for ax, (name, info) in zip(axes[0], results["models"].items()):
        d = info["deletion"]
        ax.plot(d["fractions"], d["gradcam"], marker="o", label=f"Grad-CAM order (AUC {d['auc_gradcam']:.3f})")
        ax.plot(d["fractions"], d["random"], marker="o", label=f"random order (AUC {d['auc_random']:.3f})")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("share of pixels removed")
        ax.set_ylabel("probability of predicted class")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    fig.suptitle("[2.10b] Deletion test: removing the Grad-CAM pixels first should hurt more", fontsize=11)
    save(fig, "xai_deletion.png")

    fig, axes = plt.subplots(1, len(models), figsize=(4.0 * len(models), 3.6), constrained_layout=True, squeeze=False)
    for ax, (name, info) in zip(axes[0], results["models"].items()):
        bins = info["calibration"]["bins"]
        ax.plot([0, 1], [0, 1], color="grey", linestyle=":", label="perfect calibration")
        ax.bar([b["low"] + 0.05 for b in bins], [b["accuracy"] for b in bins], width=0.09, alpha=0.7, label="accuracy")
        ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], marker="o", color="#C44E52")
        ax.set_title(f"{name}  ECE {info['calibration']['ece']:.3f}", fontsize=10)
        ax.set_xlabel("confidence")
        ax.set_ylabel("accuracy")
        ax.legend(fontsize=8)
    fig.suptitle("[2.10c] Reliability diagrams (bars on the diagonal = trustworthy scores)", fontsize=11)
    save(fig, "xai_calibration.png")

    maps = [(f"hand-crafted features\n5-NN acc {results['handcrafted']['knn_accuracy']:.2f}, silhouette {results['handcrafted']['silhouette']:.2f}", results["handcrafted"]["tsne"])] \
        if "handcrafted" in results else []
    maps += [(f"ImageNet MobileNetV3 (frozen)\n5-NN acc {results['imagenet_frozen']['knn_accuracy']:.2f}, silhouette {results['imagenet_frozen']['silhouette']:.2f}",
              results["imagenet_frozen"]["tsne"])] if "imagenet_frozen" in results else []
    maps += [(f"{name} (trained)\n5-NN acc {info['knn_accuracy']:.2f}, silhouette {info['silhouette']:.2f}", info["tsne"]) for name, info in results["models"].items()]
    fig, axes = plt.subplots(1, len(maps), figsize=(3.6 * len(maps), 3.8), constrained_layout=True, squeeze=False)
    for ax, (title, points) in zip(axes[0], maps):
        points = np.asarray(points)
        for c, name in enumerate(classes):
            ax.scatter(points[labels == c, 0], points[labels == c, 1], s=6, label=name)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0][0].legend(fontsize=7, markerscale=2)
    fig.suptitle("[2.10d] t-SNE of the test images in each representation (colour = true class)", fontsize=11)
    save(fig, "xai_tsne.png")

    for info in results["models"].values():
        info.pop("tsne")
    for key in ("handcrafted", "imagenet_frozen"):
        results.get(key, {}).pop("tsne", None)
    results["figures"] = [p.name for p in figure_paths]
    save_json(results, out_dir / "xai.json")
    return results
