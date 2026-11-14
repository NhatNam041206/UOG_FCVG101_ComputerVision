"""[2] Task 2 - training loop and the Task 2 runner (rubric 2.2).

Flow:  run_task2(cfg)
         -> [2.0] io_utils.download_and_extract (first run only)
         -> data_pipeline [2.1]-[2.4] -> model [2.5] -> [2.6] train_model   (once per experiment in config)
         -> evaluate [2.7] evaluate_model -> pick the best experiment on VALIDATION macro F1
         -> [2.8] save_model -> artifacts/model.pth, classes.json
"""
import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from src.config import set_seed
from src.data_pipeline import build_manifest, clean_dataset, make_dataloaders, split_dataset
from src.evaluate import benchmark_model, evaluate_model, predict_loader
from src.io_utils import download_and_extract, save_csv, save_json
from src.model import build_model, resolve_device, set_backbone_trainable
from src.visualize import (plot_class_counts, plot_confusion_matrix, plot_training_history, plot_tradeoff,
                           plot_tuning, show_augmentations, show_class_examples, show_misclassified)


@dataclass
class Task2Report:
    """Output of run_task2()."""
    class_counts: dict            # per split, per class
    history: dict                 # experiment -> train_loss, val_loss, val_acc ... per epoch
    test_metrics: dict            # experiment -> metrics from [2.7] on the test split
    model_path: str               # artifacts/model.pth (the selected experiment)
    selected: str = ""            # experiment saved for Task 3
    val_metrics: dict = field(default_factory=dict)      # experiment -> metrics on the validation split
    cleaning: dict = field(default_factory=dict)         # counts per removal reason
    comparison: list = field(default_factory=list)       # one row per experiment: quality + size/speed/time
    tuning: dict = field(default_factory=dict)           # experiment -> grid, chosen values, trials
    figure_paths: list = field(default_factory=list)
    metrics_path: object = None


def _run_epoch(model, loader, loss_fn, optimizer=None, device="cpu", scaler=None):
    """One pass over a loader; trains when an optimizer is given. Returns (mean loss, accuracy).

    On a GPU the forward pass runs in mixed precision (float16 with loss scaling): about 2x faster on
    tensor-core GPUs (T4, L4, A100) with the same accuracy. On a CPU everything stays float32.
    """
    training = optimizer is not None
    model.train(training)
    cuda = device == "cuda"
    scaler = scaler or torch.amp.GradScaler("cuda", enabled=cuda)
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, memory_format=torch.channels_last, non_blocking=cuda)
            labels = labels.to(device, non_blocking=cuda)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=cuda):
                logits = model(images)
                loss = loss_fn(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += len(labels)
    return total_loss / seen, correct / seen


def train_model(model, loaders, epochs, learning_rate, weight_decay, device="cpu", freeze_epochs=0,
                fine_tune_lr=None, label_smoothing=0.0):
    """[2.6] Train with AdamW + cross-entropy and a cosine learning-rate schedule; keep the best validation weights.

    Purpose:  Fit the model on train, use val to decide which epoch's weights to keep (early-stopping style),
              never look at test.
    Flow:     called by run_task2()   previous: [2.4], [2.5]   next: [2.7] evaluate_model
    Method:   freeze_epochs > 0 (transfer learning): stage 1 trains only the new head with the backbone frozen,
              stage 2 unfreezes everything at fine_tune_lr (smaller, so pretrained filters are not destroyed).
              After every epoch the validation macro-F1 is measured; the best epoch's weights are returned.

    Inputs:
        model:           nn.Module from [2.5].
        loaders:         dict with "train" and "val" DataLoaders.
        epochs:          total epochs (stage 1 + stage 2).
        learning_rate:   AdamW learning rate (head / whole model when not freezing).
        weight_decay:    AdamW weight decay (L2-style regularisation).
        device:          "cpu" or "cuda".
        freeze_epochs:   epochs with a frozen backbone (0 = train everything from the start).
        fine_tune_lr:    learning rate after unfreezing (default learning_rate / 10).
        label_smoothing: cross-entropy label smoothing.

    Returns:
        (best_model: nn.Module, history: dict of per-epoch lists:
         train_loss, train_acc, val_loss, val_acc, val_macro_f1, lr, seconds, best_epoch)
    """
    from sklearn.metrics import f1_score

    model.to(device, memory_format=torch.channels_last)     # faster convolutions on CPU and GPU
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    history = {key: [] for key in ("train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1", "lr", "seconds")}
    best_f1, best_state, best_epoch = -1.0, None, 0
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")    # mixed-precision loss scaling (GPU only)

    def make_optimizer(lr, n_epochs):
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        return optimizer, torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(n_epochs, 1))

    stages = [(freeze_epochs, False, learning_rate), (epochs - freeze_epochs, True, fine_tune_lr or learning_rate / 10)] \
        if freeze_epochs else [(epochs, True, learning_rate)]
    for stage_epochs, backbone_trainable, stage_lr in stages:
        if freeze_epochs:
            set_backbone_trainable(model, backbone_trainable)
        optimizer, scheduler = make_optimizer(stage_lr, stage_epochs)
        for _ in range(stage_epochs):
            started = time.perf_counter()
            train_loss, train_acc = _run_epoch(model, loaders["train"], loss_fn, optimizer, device, scaler)
            val_loss, _ = _run_epoch(model, loaders["val"], loss_fn, None, device)
            true_ids, predicted_ids, _, _ = predict_loader(model, loaders["val"], device)
            val_f1 = f1_score(true_ids, predicted_ids, average="macro")
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(float((true_ids == predicted_ids).mean()))
            history["val_macro_f1"].append(float(val_f1))
            history["lr"].append(optimizer.param_groups[0]["lr"])
            history["seconds"].append(time.perf_counter() - started)
            scheduler.step()
            if val_f1 > best_f1:
                best_f1, best_state, best_epoch = val_f1, copy.deepcopy(model.state_dict()), len(history["val_acc"])
    if freeze_epochs:
        set_backbone_trainable(model, True)
    model.load_state_dict(best_state)
    history["best_epoch"] = best_epoch
    return model.eval(), history


def tune_hyperparameters(splits, classes, experiment, grid, tune_epochs, settings, seed=42, device="cpu"):
    """[2.6t] Small grid search: train briefly with each setting, keep the one with the best validation macro F1.

    Purpose:  Show that hyperparameters were chosen by evidence, not guessed (tuning process for rubric 2.2).
    Flow:     called by run_task2() before the full training of an experiment that has a `tune` grid
              -> [2.4] make_dataloaders -> [2.5] build_model -> [2.6] train_model (short budget)
    Method:   every combination of the grid values (cartesian product) is trained for tune_epochs with the same
              seed and data; only the VALIDATION split is used to compare, test stays untouched.
              Short runs rank settings well enough while costing a fraction of a full run.

    Inputs:
        splits:      dict from [2.3].
        classes:     class names.
        experiment:  dict from cfg.task2.experiments (base settings).
        grid:        dict parameter -> list of values, e.g. {"learning_rate": [0.003, 0.001, 0.0003]}.
        tune_epochs: epochs per trial.
        settings:    cfg.task2 (batch size, weight decay, workers, label smoothing).
        seed:        random seed (same for every trial, so only the hyperparameter changes).
        device:      "cpu" or "cuda".

    Returns:
        (best: dict of chosen values, trials: list of dicts with the values, val metrics, curve and seconds)
    """
    import itertools

    keys = list(grid)
    trials = []
    for values in itertools.product(*(grid[key] for key in keys)):
        trial = {**experiment, **dict(zip(keys, values))}
        set_seed(seed)
        torch.manual_seed(seed)
        loaders = make_dataloaders({k: splits[k] for k in ("train", "val")}, classes, trial["image_size"],
                                   settings.batch_size, seed, augment=trial["augment"], num_workers=settings.num_workers)
        model = build_model(trial["model_name"], len(classes))
        started = time.perf_counter()
        freeze = min(trial.get("freeze_epochs", 0), max(tune_epochs - 1, 0))
        _, history = train_model(model, loaders, tune_epochs, trial["learning_rate"], settings.weight_decay,
                                 device=device, freeze_epochs=freeze, fine_tune_lr=trial.get("fine_tune_lr"),
                                 label_smoothing=settings.label_smoothing)
        trials.append({**dict(zip(keys, values)), "val_macro_f1": round(max(history["val_macro_f1"]), 4),
                       "val_acc": round(max(history["val_acc"]), 4), "seconds": round(time.perf_counter() - started, 1),
                       "curve": [round(v, 4) for v in history["val_macro_f1"]]})
    best = max(trials, key=lambda row: row["val_macro_f1"])
    return {key: best[key] for key in keys}, trials


def save_model(model, classes, model_name, image_size, path):
    """[2.8] Save weights plus the metadata inference needs to rebuild the same model.

    Flow:     called by run_task2()  ->  writes model.pth and classes.json in artifacts/
              read back by inference.load_model() [3.0]

    Inputs:
        model:      trained nn.Module.
        classes:    ordered class names.
        model_name: name passed to build_model().
        image_size: input size used in training.
        path:       output Path for model.pth.

    Returns:
        Path written (classes.json is written next to it).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"classes": list(classes), "model_name": model_name, "image_size": int(image_size)}
    torch.save({"state_dict": model.state_dict(), **meta}, path)
    (path.parent / "classes.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def _metrics_dict(result):
    """EvalResult -> plain numbers for JSON/CSV."""
    return {"accuracy": round(result.accuracy, 4), "macro_f1": round(result.macro_f1, 4),
            "ms_per_image": round(result.mean_ms_per_image, 2), "errors": len(result.misclassified)}


def _save_figure(fig, path, figure_paths):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    figure_paths.append(path)


def run_task2(cfg, experiments=None):
    """[2.R] Run the whole Task 2 flow and save the model and results.

    Purpose:  One call from raw images to a saved, evaluated model.
    Flow:     called by notebook  ->  [2.0] -> [2.1] -> [2.2] -> [2.3] -> [2.4] -> ([2.5] -> [2.6] -> [2.7]) per
              experiment -> select on validation -> [2.8]
              writes data/splits/*.csv, results/task2/ (figures, CSVs, metrics.json), artifacts/model.pth (selected)
              and artifacts/models/<experiment>/model.pth (every experiment)

    Inputs:
        cfg:         config from load_config() (uses cfg.paths, cfg.task2, cfg.seed).
        experiments: optional list of experiment names from cfg.task2.experiments to run (default: all).

    Returns:
        Task2Report.
    """
    set_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    settings = cfg.task2
    device = resolve_device(getattr(settings, "device", "auto"))                       # CPU locally, GPU on Colab
    torch.backends.cudnn.benchmark = device == "cuda"                                   # pick the fastest conv kernels
    print(f"[2.R] training on {device}")
    out_dir = cfg.paths.results / "task2"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = cfg.paths.raw_dataset
    if not raw_dir.is_dir() or not any(raw_dir.iterdir()):                               # [2.0]
        download_and_extract(settings.dataset_url, raw_dir.parent)
    figure_paths = []

    manifest = build_manifest(raw_dir)                                                  # [2.1]
    save_csv(manifest.to_dict("records"), out_dir / "manifest_raw.csv")
    clean, removal_log = clean_dataset(manifest, settings.min_side, settings.max_aspect,  # [2.2]
                                       settings.dedup_distance)
    save_csv(removal_log.to_dict("records") or [{"path": "", "label": "", "reason": "nothing removed"}],
             out_dir / "removal_log.csv")
    classes = sorted(clean.label.unique())
    splits = split_dataset(clean, tuple(settings.split_ratios), cfg.seed)               # [2.3]
    cfg.paths.splits.mkdir(parents=True, exist_ok=True)
    for name, table in splits.items():
        save_csv(table[["path", "label", "width", "height"]].to_dict("records"), cfg.paths.splits / f"{name}.csv")
    class_counts = {name: table.label.value_counts().reindex(classes, fill_value=0).to_dict()
                    for name, table in splits.items()}

    _save_figure(plot_class_counts(class_counts, manifest.label.value_counts().reindex(classes).to_dict()),
                 out_dir / "01_class_counts.png", figure_paths)
    _save_figure(show_class_examples(splits["train"], classes), out_dir / "02_class_examples.png", figure_paths)
    _save_figure(show_augmentations(splits["train"].path.iloc[0], settings.experiments[0]["image_size"], seed=cfg.seed),
                 out_dir / "03_augmentation_examples.png", figure_paths)

    histories, val_metrics, test_metrics, results, rows, tuning = {}, {}, {}, {}, [], {}
    wanted = experiments or [exp["name"] for exp in settings.experiments]
    for base in [e for e in settings.experiments if e["name"] in wanted]:
        exp = dict(base)
        if exp.get("tune"):                                                                          # [2.6t]
            best, trials = tune_hyperparameters(splits, classes, exp, exp["tune"], settings.tune_epochs, settings,
                                                cfg.seed, device)
            exp.update(best)
            tuning[exp["name"]] = {"grid": exp["tune"], "chosen": best, "trials": trials}
            save_csv([{k: v for k, v in t.items() if k != "curve"} for t in trials],
                     out_dir / f"tuning_{exp['name']}.csv")
            _save_figure(plot_tuning(trials, list(exp["tune"]), exp["name"]), out_dir / f"04a_tuning_{exp['name']}.png",
                         figure_paths)
        set_seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        loaders = make_dataloaders(splits, classes, exp["image_size"], settings.batch_size, cfg.seed,  # [2.4]
                                   augment=exp["augment"], num_workers=settings.num_workers)
        model = build_model(exp["model_name"], len(classes))                                         # [2.5]
        started = time.perf_counter()
        model, history = train_model(model, loaders, exp["epochs"], exp["learning_rate"],            # [2.6]
                                     settings.weight_decay, device=device, freeze_epochs=exp.get("freeze_epochs", 0),
                                     fine_tune_lr=exp.get("fine_tune_lr"),
                                     label_smoothing=settings.label_smoothing)
        train_minutes = (time.perf_counter() - started) / 60
        val_result = evaluate_model(model, loaders["val"], classes, device)                          # [2.7]
        test_result = evaluate_model(model, loaders["test"], classes, device)
        cost = benchmark_model(model, exp["image_size"], device="cpu")    # [2.7b] always CPU: the Task 3 app runs there
        histories[exp["name"]] = history
        val_metrics[exp["name"]] = _metrics_dict(val_result)
        test_metrics[exp["name"]] = {**_metrics_dict(test_result), "per_class": test_result.per_class,
                                     "confusion": test_result.confusion}
        results[exp["name"]] = (model, exp, test_result)
        save_model(model, classes, exp["model_name"], exp["image_size"],                              # [2.8]
                   cfg.paths.artifacts / "models" / exp["name"] / "model.pth")   # every experiment, for XAI/comparison
        rows.append({"experiment": exp["name"], "model": exp["model_name"], "image_size": exp["image_size"],
                     "augment": exp["augment"], "learning_rate": exp["learning_rate"],
                     "fine_tune_lr": exp.get("fine_tune_lr", ""), "epochs": exp["epochs"],
                     "best_epoch": history["best_epoch"],
                     "val_accuracy": val_metrics[exp["name"]]["accuracy"], "val_macro_f1": val_metrics[exp["name"]]["macro_f1"],
                     "test_accuracy": test_metrics[exp["name"]]["accuracy"],
                     "test_macro_f1": test_metrics[exp["name"]]["macro_f1"],
                     **cost, "train_minutes": round(train_minutes, 1), "trained_on": device,
                     "sec_per_epoch": round(sum(history["seconds"]) / len(history["seconds"]), 1)})
        _save_figure(plot_training_history(history, exp["name"]), out_dir / f"04_history_{exp['name']}.png", figure_paths)
        _save_figure(plot_confusion_matrix(test_result.confusion, classes, exp["name"]),
                     out_dir / f"05_confusion_{exp['name']}.png", figure_paths)
    _save_figure(plot_tradeoff(rows), out_dir / "07_model_tradeoff.png", figure_paths)

    selected = max(val_metrics, key=lambda name: (val_metrics[name]["macro_f1"], val_metrics[name]["accuracy"]))
    best_model, best_exp, best_test = results[selected]
    _save_figure(show_misclassified(best_test.misclassified, max_images=8, title=selected),
                 out_dir / f"06_misclassified_{selected}.png", figure_paths)
    model_path = save_model(best_model, classes, best_exp["model_name"], best_exp["image_size"],      # [2.8]
                            cfg.paths.artifacts / "model.pth")
    save_csv(rows, out_dir / "experiments.csv")
    removal_counts = removal_log.reason.str.replace(r" of .*| \[.*", "", regex=True).value_counts().to_dict()
    metrics_path = save_json({
        "dataset": settings.dataset_name, "classes": classes, "raw_images": len(manifest),
        "clean_images": len(clean), "removed": removal_counts, "class_counts": class_counts,
        "experiments": rows, "tuning": tuning, "selected": selected, "selection_rule": "highest validation macro F1",
        "test_metrics": test_metrics, "histories": histories, "model_path": str(model_path),
        "misclassified_examples": best_test.misclassified[:20],
    }, out_dir / "metrics.json")
    return Task2Report(class_counts=class_counts, history=histories, test_metrics=test_metrics,
                       model_path=str(model_path), selected=selected, val_metrics=val_metrics,
                       comparison=rows, tuning=tuning,
                       cleaning=removal_counts, figure_paths=figure_paths, metrics_path=metrics_path)
