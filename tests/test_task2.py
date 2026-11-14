"""Task 2 checks on a tiny generated dataset (no download, a few seconds on CPU)."""
import copy

import numpy as np
import pytest
import torch
from PIL import Image

from src.config import load_config
from src.data_pipeline import build_manifest, clean_dataset, make_dataloaders, make_transforms, split_dataset
from src.evaluate import benchmark_model, evaluate_model
from src.inference import load_model, predict
from src.model import build_model, set_backbone_trainable
from src.train import run_task2, save_model, train_model

CLASSES = ("blue", "red", "stripes")


def make_image(kind, rng, size=(96, 80)):
    image = rng.integers(0, 40, (size[1], size[0], 3)).astype(np.uint8)
    if kind == "red":
        image[..., 0] += 180
    elif kind == "blue":
        image[..., 2] += 180
    else:
        image[:, ::8] = 230
    return Image.fromarray(image)


@pytest.fixture
def dataset(tmp_path):
    """3 classes x 14 images, plus one corrupt file, one tiny image and one exact duplicate."""
    rng = np.random.default_rng(0)
    root = tmp_path / "raw"
    for kind in CLASSES:
        (root / kind).mkdir(parents=True)
        for index in range(14):
            make_image(kind, rng).save(root / kind / f"{index}.png")
    (root / "red" / "broken.jpg").write_bytes(b"not an image")
    make_image("red", rng, size=(20, 20)).save(root / "red" / "tiny.png")
    Image.open(root / "blue" / "0.png").save(root / "blue" / "copy_of_0.png")
    return root


def test_manifest_and_cleaning(dataset):
    manifest = build_manifest(dataset)
    assert len(manifest) == 3 * 14 + 3 and (~manifest.readable).sum() == 1
    clean, log = clean_dataset(manifest, min_side=32, dedup_distance=0)
    reasons = " ".join(log.reason)
    assert "unreadable" in reasons and "smaller than" in reasons and "duplicate of" in reasons
    assert len(clean) == 3 * 14


def test_split_is_stratified_and_disjoint(dataset):
    clean, _ = clean_dataset(build_manifest(dataset), min_side=32)
    splits = split_dataset(clean, (0.5, 0.25, 0.25), seed=1)
    paths = [set(part.path) for part in splits.values()]
    assert not (paths[0] & paths[1] or paths[0] & paths[2] or paths[1] & paths[2])
    assert sum(len(part) for part in splits.values()) == len(clean)
    for part in splits.values():
        assert set(part.label) == set(CLASSES)
    with pytest.raises(ValueError):
        split_dataset(clean, (0.5, 0.5, 0.5))


def test_transforms_shapes():
    image = Image.new("RGB", (120, 90), (200, 30, 30))
    for train in (True, False):
        tensor = make_transforms(64, train=train)(image)
        assert tensor.shape == (3, 64, 64) and tensor.dtype == torch.float32


def test_train_save_load_predict(dataset, tmp_path):
    clean, _ = clean_dataset(build_manifest(dataset), min_side=32)
    splits = split_dataset(clean, (0.5, 0.25, 0.25), seed=1)
    classes = sorted(CLASSES)
    loaders = make_dataloaders(splits, classes, image_size=48, batch_size=8, augment=False)
    model, history = train_model(build_model("small_cnn", 3), loaders, epochs=4, learning_rate=0.01, weight_decay=0)
    assert len(history["val_acc"]) == 4 and 1 <= history["best_epoch"] <= 4
    result = evaluate_model(model, loaders["test"], classes)
    assert result.confusion.shape == (3, 3) and result.accuracy > 0.5     # colours are easy to learn
    cost = benchmark_model(model, 48, repeats=3, batch_size=4)
    assert cost["size_MB"] > 0 and cost["latency_ms"] > 0

    path = save_model(model, classes, "small_cnn", 48, tmp_path / "model.pth")
    loaded, meta = load_model(path, tmp_path / "classes.json")
    bgr = np.zeros((60, 60, 3), np.uint8)
    bgr[..., 2] = 220                                                       # red in BGR
    prediction = predict(loaded, meta, bgr)
    assert prediction.label in classes and len(prediction.top3) == 3
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "missing.pth", tmp_path / "classes.json")


def test_freeze_backbone_and_unknown_model():
    model = build_model("small_cnn", 4)
    total = sum(p.numel() for p in model.parameters())
    assert set_backbone_trainable(model, False) < total
    assert set_backbone_trainable(model, True) == total
    with pytest.raises(ValueError):
        build_model("vit_giant", 4)


def test_run_task2_end_to_end(dataset, tmp_path):
    cfg = load_config()
    cfg.paths.raw_dataset = dataset
    cfg.paths.results, cfg.paths.artifacts, cfg.paths.splits = tmp_path / "res", tmp_path / "art", tmp_path / "splits"
    cfg.task2 = copy.deepcopy(cfg.task2)
    cfg.task2.min_side, cfg.task2.batch_size, cfg.task2.tune_epochs = 32, 8, 1
    cfg.task2.experiments = [
        {"name": "a", "model_name": "small_cnn", "image_size": 48, "augment": False, "epochs": 2, "learning_rate": 0.01},
        {"name": "b", "model_name": "small_cnn", "image_size": 48, "augment": True, "epochs": 2, "learning_rate": 0.01,
         "tune": {"learning_rate": [0.01, 0.001]}},
    ]
    report = run_task2(cfg)
    assert report.selected in ("a", "b") and (tmp_path / "art" / "model.pth").is_file()
    assert len(report.comparison) == 2 and len(report.tuning["b"]["trials"]) == 2
    assert {"latency_ms", "size_MB", "train_minutes", "test_macro_f1"} <= set(report.comparison[0])
    assert (tmp_path / "res" / "task2" / "metrics.json").is_file()


def test_handcrafted_features_and_baseline():
    from src.baselines import extract_handcrafted_features, fit_baseline, handcrafted_feature_groups, histogram_of_gradients
    rng = np.random.default_rng(0)
    bgr = rng.integers(0, 255, (90, 140, 3)).astype(np.uint8)
    features = extract_handcrafted_features(bgr)
    assert features.shape == (sum(handcrafted_feature_groups().values()),) and np.isfinite(features).all()
    gray = np.zeros((128, 128), np.uint8)
    gray[:, 64:] = 255                                       # one vertical edge -> HOG energy in the 0-degree bin
    hog = histogram_of_gradients(gray).reshape(-1, 9)
    assert hog[:, 0].sum() > hog[:, 4].sum()
    x = rng.normal(size=(60, 4)); y = (x[:, 0] > 0).astype(int)
    result = fit_baseline("logistic regression", "toy", x[:30], y[:30], x[30:45], y[30:45], x[45:], y[45:])
    assert result.test_accuracy > 0.7 and result.size_MB >= 0


def test_grad_cam_and_deletion():
    from src.explain import calibration_report, deletion_test, grad_cam
    model = build_model("small_cnn", 3).eval()
    tensor = torch.randn(1, 3, 64, 64)
    cam, index, probabilities = grad_cam(model, tensor)
    assert cam.shape == (64, 64) and 0 <= cam.min() and cam.max() <= 1 and abs(probabilities.sum() - 1) < 1e-5
    curve = deletion_test(model, tensor, cam, index, fractions=(0.0, 0.5))
    assert len(curve["gradcam"]) == 2 and curve["gradcam"][0] == pytest.approx(curve["random"][0])
    report = calibration_report(np.array([0.9, 0.8, 0.3]), np.array([True, True, False]))
    assert 0 <= report["ece"] <= 1
