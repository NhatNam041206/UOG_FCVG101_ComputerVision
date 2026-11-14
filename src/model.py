"""[2] Task 2 - model definitions (rubric 2.2).

Flow:  train.run_task2(cfg) -> [2.5] build_model ; inference.load_model() -> [2.5] build_model

Baseline: SmallCNN, a VGG-style Conv-BN-ReLU-Pool stack trained from scratch.
Advanced: MobileNetV3-Large pretrained on ImageNet (transfer learning), new classifier head, fine-tuned.
"""
import torch
from torch import nn

MODEL_NAMES = ("small_cnn", "mobilenet_v3")
# ImageNet weights for timm's mobilenetv3_large_100, hosted on GitHub (reachable where Hugging Face is not).
MOBILENET_WEIGHTS_URL = ("https://github.com/rwightman/pytorch-image-models/releases/download/"
                         "v0.1-weights/mobilenetv3_large_100_ra-f55367f5.pth")


class SmallCNN(nn.Module):
    """[2.5a] Baseline CNN from scratch: 4 x (Conv3x3-BN-ReLU-MaxPool2) -> global average pool -> Dropout -> FC.

    Channels 32 -> 64 -> 128 -> 256; each block halves the resolution and doubles the channels, so early
    blocks see edges/colours and later blocks see petal and flower-head patterns. Global average pooling
    (not a big flatten layer) keeps the parameter count small (~0.39 M) and works at any input size.
    One conv per block keeps CPU training practical (about 2.5x faster than two convs per block).
    """

    def __init__(self, n_classes, widths=(32, 64, 128, 256), dropout=0.3):
        super().__init__()
        layers, in_channels = [], 3
        for width in widths:
            layers += [nn.Conv2d(in_channels, width, 3, padding=1, bias=False), nn.BatchNorm2d(width),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            in_channels = width
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                        nn.Linear(in_channels, n_classes))

    def forward(self, images):
        return self.classifier(self.features(images))


def build_model(name, n_classes, pretrained=True):
    """[2.5] Create the network by name.

    Purpose:  One switch between the baseline CNN and the transfer-learning model (rule R5).
    Flow:     called by run_task2() and inference.load_model()
    Method:   "small_cnn":    SmallCNN, random initialisation.
              "mobilenet_v3": timm mobilenetv3_large_100 with ImageNet weights; its 1000-class head is
                              replaced by a new n_classes head. Pretrained filters already detect edges,
                              textures and object parts, so far fewer labelled images are needed.

    Inputs:
        name:       "small_cnn" (baseline) or "mobilenet_v3" (advanced).
        n_classes:  number of output classes.
        pretrained: load ImageNet weights (mobilenet_v3 only); False when weights come from model.pth.

    Returns:
        torch.nn.Module that outputs logits (batch, n_classes).

    Raises:
        ValueError: unknown model name.
    """
    if name == "small_cnn":
        return SmallCNN(n_classes)
    if name == "mobilenet_v3":
        import timm
        model = timm.create_model("mobilenetv3_large_100", pretrained=False, num_classes=1000)
        if pretrained:
            state = torch.hub.load_state_dict_from_url(MOBILENET_WEIGHTS_URL, map_location="cpu", progress=False)
            model.load_state_dict(state)
        model.reset_classifier(n_classes)
        return model
    raise ValueError(f"[2.5] unknown model '{name}', use one of {MODEL_NAMES}")


def resolve_device(name="auto"):
    """[2.5d] Pick the device to train on.

    Purpose:  The same code runs on a laptop CPU and on a Colab GPU; only cfg.task2.device changes.
    Flow:     called by train.run_task2() and baselines.extract_deep_features()

    Inputs:
        name: "auto" (GPU if available, else CPU), "cpu" or "cuda".

    Returns:
        str "cpu" or "cuda".

    Raises:
        ValueError: "cuda" requested but no GPU is available.
    """
    if name in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("[2.5d] device 'cuda' requested but no GPU is available (Colab: Runtime > Change runtime type)")
    return name


def set_backbone_trainable(model, trainable):
    """[2.5b] Freeze or unfreeze everything except the classifier head (transfer-learning stage 1 / 2).

    Inputs:
        model:     nn.Module from [2.5].
        trainable: False = only the head learns (feature extraction); True = fine-tune everything.

    Returns:
        int, number of trainable parameters after the change.
    """
    head = model.get_classifier() if hasattr(model, "get_classifier") else model.classifier
    head_ids = {id(p) for p in head.parameters()}
    for parameter in model.parameters():
        parameter.requires_grad = trainable or id(parameter) in head_ids
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters(model):
    """[2.5c] Total number of parameters (for the model comparison table)."""
    return sum(p.numel() for p in model.parameters())
