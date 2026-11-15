"""[DEMO 2] Training results page: reads results/task2 (written by run_task2 and run_explanations) and lets the
user explore it. Nothing is retrained here.

Flow:  results/task2/metrics.json, baselines.csv, xai.json  ->  interactive charts per segment
"""
import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from demo.common import get_config, load_csv, load_json, show, step_header
from src.io_utils import read_image

cfg = get_config()
out_dir = cfg.paths.results / "task2"
metrics = load_json(out_dir / "metrics.json")

st.title("Task 2 · Training results")
if metrics is None:
    st.warning("results/task2/metrics.json is missing: run Task 2 (run_task2) first.")
    st.stop()

classes = metrics["classes"]
experiments = pd.DataFrame(metrics["experiments"])
experiments["fine_tune_lr"] = pd.to_numeric(experiments["fine_tune_lr"], errors="coerce")   # "" = not used
names = list(experiments["experiment"])
selected = metrics["selected"]

step = st.segmented_control("Step", ["Dataset", "Learning curves", "Tuning", "Model comparison", "Confusion matrix",
                                     "Baselines", "Explainability"], default="Learning curves", key="train_step",
                            label_visibility="collapsed") or "Learning curves"


def figure(name, caption=None):
    path = out_dir / name
    if path.is_file():
        show(read_image(path), caption=caption)
    else:
        st.caption(f"{name} not found")


# ------------------------------------------------------------------------------------------- dataset
if step == "Dataset":
    step_header("2.1-2.4", "Ingest, clean, split, augment",
                f"{metrics['dataset']}. Images are listed, corrupt / tiny / panorama / near-duplicate files are "
                "removed, then a stratified 70/15/15 split is made before any augmentation.")
    cols = st.columns(4)
    cols[0].metric("Raw images", f"{metrics['raw_images']:,}")
    cols[1].metric("After cleaning", f"{metrics['clean_images']:,}",
                   f"-{metrics['raw_images'] - metrics['clean_images']}", delta_color="off")
    cols[2].metric("Classes", len(classes))
    cols[3].metric("Train / val / test", " / ".join(str(sum(metrics["class_counts"][s].values()))
                                                    for s in ("train", "val", "test")))
    left, right = st.columns([3, 2], gap="large")
    with left:
        counts = pd.DataFrame([{"split": split, "class": c, "images": n}
                               for split, per_class in metrics["class_counts"].items() for c, n in per_class.items()])
        st.altair_chart(alt.Chart(counts).mark_bar().encode(
            x=alt.X("class:N", title=None), y=alt.Y("images:Q"), color=alt.Color("split:N", sort=["train", "val", "test"]),
            xOffset=alt.XOffset("split:N", sort=["train", "val", "test"]), tooltip=["split", "class", "images"]
        ).properties(height=320, title="Images per class and split"), width="stretch")
    with right:
        st.markdown("**Removed while cleaning [2.2]**")
        st.dataframe(pd.DataFrame(list(metrics["removed"].items()), columns=["reason", "images"]), hide_index=True)
        log = load_csv(out_dir / "removal_log.csv")
        if log is not None:
            with st.expander(f"Removal log ({len(log)} rows)"):
                st.dataframe(log, hide_index=True)

    st.markdown("##### Augmentation preview [2.4a]")
    st.caption("The training transform is random: each click draws new crops, flips, rotations, colour jitter "
               "and noise. Validation and test images only get resize + centre crop.")
    splits = cfg.paths.splits / "train.csv"
    if splits.is_file():
        import torch
        from src.data_pipeline import IMAGENET_MEAN, IMAGENET_STD, load_rgb, make_transforms
        table = load_csv(splits)
        col_a, col_b = st.columns([1, 4])
        flower = col_a.selectbox("Class", classes)
        if col_a.button("New samples", icon=":material/shuffle:"):
            st.session_state.aug_seed = int(time.time())
        seed = st.session_state.get("aug_seed", 0)
        rows = table[table["label"] == flower]
        path = cfg.project_root / rows.iloc[seed % len(rows)]["path"]
        if path.is_file():
            torch.manual_seed(seed)
            image = load_rgb(path)
            transform = make_transforms(160, train=True, augment=True)
            mean, std = np.array(IMAGENET_MEAN)[:, None, None], np.array(IMAGENET_STD)[:, None, None]
            tiles = [np.asarray(make_transforms(160, train=False)(image).numpy() * std + mean)]
            tiles += [np.asarray(transform(image).numpy() * std + mean) for _ in range(5)]
            with col_b:
                cols = st.columns(6)
                for index, (col, tile) in enumerate(zip(cols, tiles)):
                    col.image(np.clip(tile.transpose(1, 2, 0), 0, 1), caption="eval" if index == 0 else f"train #{index}",
                              width="stretch")
        else:
            st.caption("Raw dataset not on disk; showing the saved figure.")
            figure("03_augmentation_examples.png")
    else:
        figure("03_augmentation_examples.png")
    with st.expander("Examples per class"):
        figure("02_class_examples.png")

# ------------------------------------------------------------------------------------------- curves
elif step == "Learning curves":
    step_header("2.6", "Watch the models learn",
                "Per-epoch loss, accuracy and validation macro F1 recorded during training. Drag the epoch slider "
                "or press Play to replay training; the dot marks the epoch kept (best validation macro F1).")
    col_a, col_b, col_c, col_d = st.columns([3, 3, 2, 1], vertical_alignment="bottom")
    chosen = col_a.multiselect("Experiments", names, default=names)
    metric = col_b.segmented_control("Metric", ["loss", "accuracy", "macro F1", "learning rate"], default="loss",
                                     key="curve_metric") or "loss"
    max_epoch = max(len(metrics["histories"][n]["train_loss"]) for n in names)
    upto = col_c.slider("Up to epoch", 1, max_epoch, max_epoch, key="curve_upto")
    play = col_d.button("Play", icon=":material/play_arrow:")
    view = st.container()
    rows = []
    for name in chosen:
        history = metrics["histories"][name]
        series = {"loss": [("train", "train_loss"), ("val", "val_loss")],
                  "accuracy": [("train", "train_acc"), ("val", "val_acc")],
                  "macro F1": [("val", "val_macro_f1")], "learning rate": [("train", "lr")]}[metric]
        for split, key in series:
            for epoch, value in enumerate(history[key], start=1):
                rows.append({"experiment": name, "split": split, "epoch": epoch, "value": value,
                             "best": epoch == history["best_epoch"]})
    data = pd.DataFrame(rows)

    def chart(limit):
        shown = data[data["epoch"] <= limit]
        base = alt.Chart(shown).encode(
            x=alt.X("epoch:Q", scale=alt.Scale(domain=[1, max_epoch]), axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("value:Q", title=metric, scale=alt.Scale(zero=False, type="log" if metric == "learning rate" else "linear")),
            color=alt.Color("experiment:N"), strokeDash=alt.StrokeDash("split:N", sort=["train", "val"], legend=alt.Legend(symbolType="stroke")),
            tooltip=["experiment", "split", "epoch", alt.Tooltip("value:Q", format=".4f")])
        lines = base.mark_line(point=True)
        best = base.transform_filter(alt.datum.best).mark_point(size=180, filled=False, strokeWidth=2)
        return (lines + best).properties(height=420)

    with view:
        if data.empty:
            st.info("Pick at least one experiment.")
            st.stop()
        holder = st.empty()
        if play:
            for epoch in range(1, max_epoch + 1):
                holder.altair_chart(chart(epoch), width="stretch")
                time.sleep(0.35)
        else:
            holder.altair_chart(chart(upto), width="stretch")
        summary = experiments.set_index("experiment").loc[chosen, ["best_epoch", "epochs", "val_macro_f1",
                                                                     "train_minutes", "sec_per_epoch"]]
        st.dataframe(summary, width="stretch")
        st.caption("Train loss above validation loss early on is expected with augmentation and label smoothing; "
                   "the transfer model's jump after epoch 2 is where its backbone is unfrozen for fine-tuning.")

# ------------------------------------------------------------------------------------------- tuning
elif step == "Tuning":
    step_header("2.6t", "Hyper-parameter search",
                "A short grid search on the validation set; each trial trains for a few epochs and the value with "
                "the best validation macro F1 is used for the full run.")
    tuning = metrics.get("tuning", {})
    if not tuning:
        st.info("No tuning results.")
        st.stop()
    name = st.segmented_control("Experiment", list(tuning), default=list(tuning)[0], key="tune_exp") or list(tuning)[0]
    info = tuning[name]
    parameter = list(info["grid"])[0]
    rows = [{parameter: str(t[parameter]), "epoch": e, "val macro F1": v}
            for t in info["trials"] for e, v in enumerate(t["curve"], start=1)]
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.altair_chart(alt.Chart(pd.DataFrame(rows)).mark_line(point=True).encode(
            x=alt.X("epoch:Q", axis=alt.Axis(tickMinStep=1)), y=alt.Y("val macro F1:Q", scale=alt.Scale(zero=False)),
            color=alt.Color(f"{parameter}:N"), tooltip=[parameter, "epoch", alt.Tooltip("val macro F1:Q", format=".4f")]
        ).properties(height=380, title=f"{name}: {parameter} trials"), width="stretch")
    with right:
        st.metric(f"Chosen {parameter}", info["chosen"][parameter])
        trials = pd.DataFrame(info["trials"]).drop(columns=["curve"])
        st.dataframe(trials, hide_index=True)

# ------------------------------------------------------------------------------------------- compare
elif step == "Model comparison":
    step_header("2.7-2.7b", "Accuracy vs size vs speed",
                f"The model saved for Task 3 is chosen by {metrics['selection_rule']}; size and CPU latency are "
                "the other criteria. Pick the axes to see the trade-off.")
    best = experiments.set_index("experiment").loc[selected]
    cols = st.columns(4)
    cols[0].metric("Selected", selected)
    cols[1].metric("Test macro F1", f"{best['test_macro_f1']:.3f}")
    cols[2].metric("Size", f"{best['size_MB']:.1f} MB")
    cols[3].metric("CPU latency", f"{best['latency_ms']:.1f} ms / image")
    numeric = ["test_macro_f1", "test_accuracy", "val_macro_f1", "size_MB", "parameters_M", "latency_ms",
               "throughput_img_s", "train_minutes"]
    col_a, col_b = st.columns(2)
    x_axis = col_a.selectbox("x axis", numeric, index=numeric.index("size_MB"))
    y_axis = col_b.selectbox("y axis", numeric, index=0)
    st.altair_chart(alt.Chart(experiments).mark_circle(opacity=0.85).encode(
        x=alt.X(f"{x_axis}:Q", scale=alt.Scale(zero=False, padding=30)),
        y=alt.Y(f"{y_axis}:Q", scale=alt.Scale(zero=False, padding=30)),
        size=alt.Size("latency_ms:Q", scale=alt.Scale(range=[150, 1200])), color="experiment:N",
        tooltip=list(experiments.columns)).properties(height=380)
        + alt.Chart(experiments).mark_text(dy=-22).encode(x=f"{x_axis}:Q", y=f"{y_axis}:Q", text="experiment:N"),
        width="stretch")
    st.dataframe(experiments, hide_index=True)

# ------------------------------------------------------------------------------------------- confusion
elif step == "Confusion matrix":
    step_header("2.7", "Where the models go wrong (test split)",
                "Rows = true class, columns = predicted class. Normalised rows show recall per class.")
    col_a, col_b = st.columns([2, 1])
    name = col_a.segmented_control("Model", names, default=selected, key="cm_model") or selected
    normalise = col_b.toggle("Normalise rows", value=True)
    test = metrics["test_metrics"][name]
    matrix = np.array(test["confusion"], float)
    shown = matrix / matrix.sum(1, keepdims=True) if normalise else matrix
    cells = pd.DataFrame([{"true": classes[i], "predicted": classes[j], "value": shown[i, j], "count": int(matrix[i, j])}
                          for i in range(len(classes)) for j in range(len(classes))])
    left, right = st.columns([3, 2], gap="large")
    with left:
        heat = alt.Chart(cells).encode(x=alt.X("predicted:N", sort=classes), y=alt.Y("true:N", sort=classes))
        st.altair_chart((heat.mark_rect().encode(color=alt.Color("value:Q", scale=alt.Scale(scheme="blues")),
                                                 tooltip=["true", "predicted", "count",
                                                          alt.Tooltip("value:Q", format=".2f")])
                         + heat.mark_text().encode(text=alt.Text("value:Q", format=".2f" if normalise else ".0f"),
                                                   color=alt.condition(alt.datum.value > shown.max() / 2,
                                                                       alt.value("white"), alt.value("black"))))
                        .properties(height=380), width="stretch")
    with right:
        cols = st.columns(2)
        cols[0].metric("Accuracy", f"{test['accuracy']:.3f}")
        cols[1].metric("Macro F1", f"{test['macro_f1']:.3f}")
        per_class = pd.DataFrame(test["per_class"]).T.round(3)
        st.dataframe(per_class, width="stretch")
        off = matrix.copy()
        np.fill_diagonal(off, 0)
        i, j = np.unravel_index(off.argmax(), off.shape)
        st.caption(f"Most common mistake: **{classes[i]}** predicted as **{classes[j]}** ({int(off[i, j])} images).")
    if name == selected and metrics.get("misclassified_examples"):
        st.markdown("##### Misclassified test images")
        cols = st.columns(5)
        for index, (path, true, predicted, score) in enumerate(metrics["misclassified_examples"][:10]):
            full = cfg.project_root / path
            if full.is_file():
                with cols[index % 5]:
                    show(read_image(full, max_side=320), caption=f"{true} → {predicted} ({score:.0%})")

# ------------------------------------------------------------------------------------------- baselines
elif step == "Baselines":
    step_header("2.9", "Is deep learning worth it?",
                "Dummy classifiers set the floor; classical models on hand-crafted colour + HOG + LBP features and "
                "a linear probe on frozen ImageNet features sit between them and the trained CNNs.")
    baselines = load_csv(out_dir / "baselines.csv")
    if baselines is None:
        st.info("results/task2/baselines.csv is missing.")
        st.stop()
    ladder = pd.concat([
        baselines[["model", "features", "test_macro_f1", "latency_ms", "size_MB"]],
        experiments.assign(model=experiments["experiment"], features="learned end to end (CNN)")[
            ["model", "features", "test_macro_f1", "latency_ms", "size_MB"]]], ignore_index=True)
    st.altair_chart(alt.Chart(ladder).mark_bar().encode(
        x=alt.X("test_macro_f1:Q", title="test macro F1", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("model:N", sort="x", title=None), color=alt.Color("features:N", legend=alt.Legend(orient="bottom")),
        tooltip=list(ladder.columns)).properties(height=40 + 28 * len(ladder)), width="stretch")
    st.dataframe(baselines, hide_index=True)

# ------------------------------------------------------------------------------------------- XAI
else:
    step_header("2.10", "Explainability",
                "Grad-CAM shows where each CNN looks; the deletion test checks the heatmap is faithful (masking "
                "the hottest pixels should hurt more than masking random ones); the reliability diagram checks "
                "whether confidence scores can be trusted.")
    xai = load_json(out_dir / "xai.json")
    if xai is None:
        st.info("results/task2/xai.json is missing: run explain.run_explanations first.")
        st.stop()
    name = st.segmented_control("Model", list(xai["models"]), default=selected, key="xai_model") or selected
    info = xai["models"][name]
    deletion, calibration = info["deletion"], info["calibration"]
    cols = st.columns(4)
    cols[0].metric("Deletion gap (AUC random - Grad-CAM)", f"{deletion['auc_random'] - deletion['auc_gradcam']:+.3f}",
                   help="positive = the heatmap points at pixels the model really uses")
    cols[1].metric("ECE", f"{calibration['ece']:.3f}", help="expected calibration error, 0 = perfectly calibrated")
    cols[2].metric("5-NN accuracy in embedding", f"{info['knn_accuracy']:.3f}")
    cols[3].metric("Silhouette", f"{info['silhouette']:.3f}")
    left, right = st.columns(2, gap="large")
    with left:
        curve = pd.DataFrame({"fraction masked": deletion["fractions"] * 2,
                              "class probability": deletion["gradcam"] + deletion["random"],
                              "masking": ["Grad-CAM hottest first"] * len(deletion["fractions"])
                              + ["random"] * len(deletion["fractions"])})
        st.altair_chart(alt.Chart(curve).mark_line(point=True).encode(
            x=alt.X("fraction masked:Q", axis=alt.Axis(format="%")), y="class probability:Q", color="masking:N",
            tooltip=["masking", "fraction masked", alt.Tooltip("class probability:Q", format=".3f")]
        ).properties(height=320, title="Deletion test"), width="stretch")
    with right:
        bins = pd.DataFrame(calibration["bins"])
        bars = alt.Chart(bins).mark_bar(opacity=0.8).encode(
            x=alt.X("confidence:Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("accuracy:Q", scale=alt.Scale(domain=[0, 1])),
            size=alt.value(18), tooltip=["low", "high", "count", alt.Tooltip("confidence:Q", format=".3f"),
                                         alt.Tooltip("accuracy:Q", format=".3f")])
        diagonal = alt.Chart(pd.DataFrame({"confidence": [0, 1], "accuracy": [0, 1]})).mark_line(
            strokeDash=[4, 4], color="gray").encode(x="confidence:Q", y="accuracy:Q")
        st.altair_chart((bars + diagonal).properties(height=320, title="Reliability diagram"), width="stretch")
        st.caption("Bars above the diagonal = the model is more often right than its score says (under-confident).")
    view = st.segmented_control("Figure", ["Grad-CAM comparison", "Grad-CAM on mistakes", "t-SNE embedding"],
                                default="Grad-CAM comparison", key="xai_fig") or "Grad-CAM comparison"
    figure({"Grad-CAM comparison": "xai_gradcam_compare.png", "Grad-CAM on mistakes": f"xai_mistakes_{name}.png",
            "t-SNE embedding": "xai_tsne.png"}[view])
    st.caption("Live Grad-CAM on your own image: open **2 Try the model**.")
