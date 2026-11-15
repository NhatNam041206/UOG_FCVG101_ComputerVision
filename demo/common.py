"""[DEMO] Shared helpers for the demo pages: cached config/model loading, image input, display.

Display only (rule R4: convert to RGB at the last moment); every algorithm lives in src/.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.config import load_config
from src.io_utils import decode_image_bytes, read_image, to_rgb

IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp"]


@st.cache_resource
def get_config():
    """Load configs/config.yaml once per app process."""
    return load_config()


@st.cache_data
def load_json(path):
    """Read a results/*.json file (cached; returns None when the task has not been run yet)."""
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


@st.cache_data
def load_csv(path):
    """Read a results/*.csv file (cached; None when missing)."""
    path = Path(path)
    return pd.read_csv(path, encoding="utf-8-sig") if path.is_file() else None


@st.cache_resource
def load_classifier(model_dir):
    """[3.0] Load a trained model once per folder (model.pth + classes.json)."""
    from src.inference import load_model
    model_dir = Path(model_dir)
    return load_model(model_dir / "model.pth", model_dir / "classes.json")


def show(image, caption=None, **kwargs):
    """Show a BGR / grayscale uint8 image (converted to RGB here)."""
    st.image(to_rgb(image), caption=caption, width="stretch", **kwargs)


def to_display(array, signed=False):
    """Scale a float map (gradient, spectrum, response) to uint8 for display.

    signed=True keeps 0 at mid-grey so negative and positive values stay distinguishable.
    """
    array = np.asarray(array, np.float32)
    if signed:
        peak = float(np.abs(array).max()) or 1.0
        return np.clip(128 + 127 * array / peak, 0, 255).astype(np.uint8)
    low, high = float(array.min()), float(array.max())
    return ((array - low) / ((high - low) or 1.0) * 255).astype(np.uint8)


def colour_map(gray_uint8, cmap=cv2.COLORMAP_INFERNO):
    """Apply an OpenCV colour map (returns BGR)."""
    return cv2.applyColorMap(gray_uint8, cmap)


def image_input(key, default_path=None, label="Image", max_side=None, sources=("Sample", "Upload", "Camera")):
    """Let the user pick an image from a sample file, an upload or the webcam.

    Returns:
        BGR uint8 image, or None while nothing is chosen.
    """
    if default_path is None or not Path(default_path).is_file():
        sources = tuple(s for s in sources if s != "Sample")
    source = st.segmented_control(label, sources, default=sources[0], key=f"{key}_source")
    image = None
    if source == "Sample":
        image = read_image(default_path)
    elif source == "Upload":
        upload = st.file_uploader(f"{label} file", type=IMAGE_TYPES, key=f"{key}_upload",
                                  label_visibility="collapsed")
        if upload is not None:
            image = decode_image_bytes(upload.getvalue())
    elif source == "Camera":
        shot = st.camera_input(f"{label} from camera", key=f"{key}_camera", label_visibility="collapsed")
        if shot is not None:
            image = decode_image_bytes(shot.getvalue())
    if image is not None and max_side:
        scale = max_side / max(image.shape[:2])
        if scale < 1:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image


def step_header(step_id, title, text):
    """Title line with the step ID used across code, notebook and report (rule R1)."""
    st.markdown(f"#### `{step_id}` {title}")
    st.caption(text)
