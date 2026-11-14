"""[2] Task 2 - evaluation on a held-out split (rubric 2.3).

Flow:  train.run_task2(cfg) -> [2.7] evaluate_model
"""
import time
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


@dataclass
class EvalResult:
    """Output of [2.7]."""
    accuracy: float
    macro_f1: float
    per_class: dict               # class -> precision, recall, f1, support
    confusion: np.ndarray         # (n_classes, n_classes), rows = true, cols = predicted
    misclassified: list           # (path, true label, predicted label, score)
    mean_ms_per_image: float = 0.0    # CPU inference time, batch included


@torch.no_grad()
def predict_loader(model, loader, device="cpu"):
    """[2.7a] Run the model over a DataLoader once.

    Inputs:
        model:  nn.Module.
        loader: DataLoader returning (images, label ids).
        device: "cpu" or "cuda".

    Returns:
        (true ids: np.ndarray (N,), predicted ids: np.ndarray (N,), softmax scores: np.ndarray (N, C), seconds)
    """
    model.eval()
    targets, scores = [], []
    started = time.perf_counter()
    for images, labels in loader:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            logits = model(images.to(device, memory_format=torch.channels_last, non_blocking=device == "cuda"))
        scores.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        targets.append(labels.numpy())
    elapsed = time.perf_counter() - started
    scores = np.concatenate(scores)
    return np.concatenate(targets), scores.argmax(axis=1), scores, elapsed


def evaluate_model(model, test_loader, classes, device="cpu"):
    """[2.7] Predict a split once and compute the report metrics.

    Purpose:  The honest final score: accuracy, macro F1 (every class counts equally, so a large class cannot
              hide a weak one), per-class precision/recall/F1 and the confusion matrix.
    Flow:     called by run_task2()   previous: [2.6]   next: [2.8] save_model
              -> sklearn.metrics (accuracy, f1, classification_report, confusion_matrix)

    Inputs:
        model:       trained nn.Module in eval mode.
        test_loader: DataLoader for the split (its dataset must expose .paths, see ImageTableDataset).
        classes:     ordered class names.
        device:      "cpu" or "cuda".

    Returns:
        EvalResult.
    """
    true_ids, predicted_ids, scores, seconds = predict_loader(model, test_loader, device)
    report = classification_report(true_ids, predicted_ids, labels=list(range(len(classes))),
                                   target_names=classes, output_dict=True, zero_division=0)
    per_class = {name: {"precision": report[name]["precision"], "recall": report[name]["recall"],
                        "f1": report[name]["f1-score"], "support": int(report[name]["support"])} for name in classes}
    paths = getattr(test_loader.dataset, "paths", [""] * len(true_ids))
    wrong = [(paths[i], classes[true_ids[i]], classes[predicted_ids[i]], float(scores[i].max()))
             for i in np.nonzero(true_ids != predicted_ids)[0]]
    wrong.sort(key=lambda item: -item[3])          # most confident mistakes first
    return EvalResult(accuracy=float(accuracy_score(true_ids, predicted_ids)),
                      macro_f1=float(f1_score(true_ids, predicted_ids, average="macro")),
                      per_class=per_class,
                      confusion=confusion_matrix(true_ids, predicted_ids, labels=list(range(len(classes)))),
                      misclassified=wrong,
                      mean_ms_per_image=1000.0 * seconds / max(len(true_ids), 1))


@torch.no_grad()
def benchmark_model(model, image_size, repeats=30, batch_size=32, device="cpu"):
    """[2.7b] Deployment cost of a model: size on disk, parameters, latency and throughput.

    Purpose:  Accuracy is not the only criterion: the Task 3 app runs on a laptop CPU, so a model must also be
              small and fast. These numbers let the report pick the model that suits the deployment.
    Flow:     called by run_task2() for every experiment

    Inputs:
        model:      trained nn.Module.
        image_size: input side length used by the model.
        repeats:    timed single-image runs (the median is reported).
        batch_size: batch for the throughput measurement.
        device:     "cpu" or "cuda".

    Returns:
        dict parameters_M, size_MB (saved state_dict), latency_ms (1 image, median), throughput_img_s (batched).
    """
    import io

    model.eval().to(device)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    single = torch.randn(1, 3, image_size, image_size, device=device).contiguous(memory_format=torch.channels_last)
    batch = torch.randn(batch_size, 3, image_size, image_size, device=device).contiguous(memory_format=torch.channels_last)
    for _ in range(3):                                 # warm-up
        model(single)
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        model(single)
        times.append(time.perf_counter() - started)
    started = time.perf_counter()
    for _ in range(3):
        model(batch)
    throughput = 3 * batch_size / (time.perf_counter() - started)
    return {"parameters_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 3),
            "size_MB": round(buffer.getbuffer().nbytes / 2 ** 20, 2),
            "latency_ms": round(1000 * float(np.median(times)), 2),
            "throughput_img_s": round(throughput, 1)}
