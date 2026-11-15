"""[DEMO 2] Try the trained model: classify one image and explain the decision with Grad-CAM.

Flow:  image (test split / upload / camera) -> [3.0] load_model (cached) -> [3.3] predict
       -> [2.10a] grad_cam -> overlay_heatmap
"""
import time

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from torchvision import transforms

from demo.common import IMAGE_TYPES, get_config, load_classifier, load_csv, show, step_header
from src.data_pipeline import make_transforms
from src.explain import grad_cam, overlay_heatmap
from src.inference import predict
from src.io_utils import decode_image_bytes, read_image

cfg = get_config()
artifacts = cfg.paths.artifacts
state = st.session_state

st.title("Task 2 · Try the model")

model_dirs = {p.name: p for p in sorted((artifacts / "models").glob("*")) if (p / "model.pth").is_file()}
if (artifacts / "model.pth").is_file():
    model_dirs = {"selected (artifacts/model.pth)": artifacts, **model_dirs}
if not model_dirs:
    st.warning("No trained model in artifacts/: run Task 2 first.")
    st.stop()

controls, view = st.columns([1, 2], gap="large")
with controls:
    model_name = st.selectbox("Model", list(model_dirs))
    model, meta = load_classifier(str(model_dirs[model_name]))
    st.caption(f"{meta['model_name']}, input {meta['image_size']} px, classes: {', '.join(meta['classes'])}")
    source = st.segmented_control("Image", ["Test split", "Upload", "Camera"], default="Test split",
                                  key="cls_source") or "Test split"
    image, true_label = None, None
    if source == "Test split":
        test = load_csv(cfg.paths.splits / "test.csv")
        if test is None:
            st.info("data/splits/test.csv not found; use Upload or Camera.")
        else:
            flower = st.selectbox("Class", ["any"] + meta["classes"])
            pool = test if flower == "any" else test[test["label"] == flower]
            if st.button("Random test image", icon=":material/shuffle:") or "cls_index" not in state:
                state.cls_index = int(np.random.default_rng().integers(1 << 30))
            row = pool.iloc[state.cls_index % len(pool)]
            path = cfg.project_root / row["path"]
            if path.is_file():
                image, true_label = read_image(path), row["label"]
            else:
                st.info("The raw dataset is not on disk; use Upload or Camera.")
    elif source == "Upload":
        upload = st.file_uploader("Flower photo", type=IMAGE_TYPES, key="cls_upload")
        image = decode_image_bytes(upload.getvalue()) if upload else None
    else:
        shot = st.camera_input("Photo of a flower", key="cls_camera")
        image = decode_image_bytes(shot.getvalue()) if shot else None
    explain = st.toggle("Grad-CAM heatmap", value=True)
    alpha = st.slider("Heatmap opacity", 0.0, 1.0, 0.45, 0.05, disabled=not explain)

with view:
    step_header("3.3 + 2.10a", "Prediction and explanation",
                "The image gets the same resize + centre crop + normalisation as the validation set. Grad-CAM "
                "highlights the regions that raised the score of the explained class.")
    if image is None:
        st.info("Choose an image.")
        st.stop()
    start = time.perf_counter()
    prediction = predict(model, meta, image)
    latency_ms = (time.perf_counter() - start) * 1000
    cols = st.columns(3)
    verdict = None if true_label is None else ("correct" if true_label == prediction.label else f"true: {true_label}")
    cols[0].metric("Prediction", prediction.label, verdict, delta_color="normal" if verdict == "correct" else "inverse")
    cols[1].metric("Score", f"{prediction.score:.1%}", help="softmax score, not a calibrated probability")
    cols[2].metric("Inference", f"{latency_ms:.0f} ms", help="CPU, includes preprocessing")

    size = meta["image_size"]
    rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    tensor = make_transforms(size, train=False)(rgb).unsqueeze(0)
    crop = np.asarray(transforms.CenterCrop(size)(transforms.Resize(int(round(size * 1.14)))(rgb)))
    explained = st.segmented_control("Explain class", meta["classes"], default=prediction.label,
                                     key="cls_explain") if explain else None
    heatmap, _, probabilities = grad_cam(model, tensor, meta["classes"].index(explained or prediction.label))

    left, right = st.columns(2)
    with left:
        show(image, caption="input")
    with right:
        if explain:
            show(overlay_heatmap(crop, heatmap, alpha), caption=f"Grad-CAM for '{explained}' (model's {size} px view)")
        else:
            show(cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), caption=f"what the model sees ({size} px)")
    scores = pd.DataFrame({"class": meta["classes"], "score": probabilities})
    st.altair_chart(alt.Chart(scores).mark_bar().encode(
        x=alt.X("score:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        y=alt.Y("class:N", sort="-x", title=None),
        color=alt.condition(alt.datum["class"] == prediction.label, alt.value("#54a24b"), alt.value("#9ecae9")),
        tooltip=["class", alt.Tooltip("score:Q", format=".2%")]).properties(height=200), width="stretch")
    st.caption("Images outside the five training classes still get one of the five labels: the model has no "
               "'unknown' class, so a low top score is the only warning sign.")
