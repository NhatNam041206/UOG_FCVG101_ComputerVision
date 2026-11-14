"""[2->3] Load the saved model and classify one image (the bridge from Task 2 to Task 3).

Flow:  assistant.classify_mode() -> [3.0] load_model (once) -> [3.3] predict
"""
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
from PIL import Image

from src.data_pipeline import make_transforms
from src.model import build_model


@dataclass
class Prediction:
    """Output of [3.3]."""
    label: str
    score: float                  # softmax score, not a calibrated probability
    top3: list                    # [(label, score), ...]


def load_model(model_path, classes_path, device="cpu"):
    """[3.0] Rebuild the network from metadata and load the trained weights.

    Flow:     called by assistant.classify_mode() (cached by app.py)  ->  model.build_model(pretrained=False)

    Inputs:
        model_path:   Path to artifacts/model.pth.
        classes_path: Path to artifacts/classes.json.
        device:       "cpu" or "cuda".

    Returns:
        (model: nn.Module in eval mode, meta: dict with classes, model_name, image_size)

    Raises:
        FileNotFoundError: an artifact is missing (run Task 2 first).
    """
    model_path, classes_path = Path(model_path), Path(classes_path)
    for path in (model_path, classes_path):
        if not path.is_file():
            raise FileNotFoundError(f"[3.0] {path} not found: run Task 2 (run_task2) first")
    meta = json.loads(classes_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(model_path, map_location=device)
    model = build_model(meta["model_name"], len(meta["classes"]), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), meta


@torch.no_grad()
def predict(model, meta, bgr, device="cpu"):
    """[3.3] Classify one BGR image with the same preprocessing as validation.

    Flow:     called by assistant.classify_mode()  ->  data_pipeline.make_transforms(train=False)

    Inputs:
        model: from [3.0].
        meta:  from [3.0].
        bgr:   np.ndarray uint8 (H, W, 3).

    Returns:
        Prediction.

    Raises:
        ValueError: not a 3-channel uint8 image.
    """
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("[3.3] predict expects a colour image (H, W, 3)")
    image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    tensor = make_transforms(meta["image_size"], train=False)(image).unsqueeze(0).to(device)
    scores = torch.softmax(model(tensor), dim=1)[0].cpu()
    top = torch.topk(scores, k=min(3, len(meta["classes"])))
    top3 = [(meta["classes"][i], float(s)) for s, i in zip(top.values, top.indices)]
    return Prediction(label=top3[0][0], score=top3[0][1], top3=top3)
