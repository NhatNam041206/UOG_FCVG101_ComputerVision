"""[2] Task 2 - data pipeline: ingestion -> cleaning -> split -> transforms -> loaders (rubric 2.1).

Flow:  train.run_task2(cfg)
         -> [2.0] io_utils.download_and_extract (only if data/raw is empty)
         -> [2.1] build_manifest -> [2.2] clean_dataset -> [2.3] split_dataset -> [2.4] make_dataloaders

Rule: split BEFORE augmentation; augmentation on train only; the eval transform is reused by inference.
Works for any dataset laid out as one sub-folder per class.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.io_utils import IMAGE_EXTENSIONS

IMAGENET_MEAN = (0.485, 0.456, 0.406)   # normalisation shared by every model (needed by the pretrained one)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------------------------------------------- [2.1]
def build_manifest(raw_dir):
    """[2.1] List every image with its label (folder name), size, colour mode and readability.

    Purpose:  Ingestion step: one table that records the raw data before anything is changed.
    Flow:     called by run_task2()   next step: [2.2] clean_dataset
    Method:   walk raw_dir/<class>/*; open each file with PIL (header + full decode) so broken files are
              found here, not in the middle of training.

    Inputs:
        raw_dir: Path, one sub-folder per class.

    Returns:
        pandas.DataFrame columns: path, label, width, height, mode, file_kb, readable, error.

    Raises:
        FileNotFoundError: raw_dir missing or has no class folders with images.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"[2.1] dataset folder not found: {raw_dir}")
    rows = []
    for class_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            row = {"path": str(path), "label": class_dir.name, "width": 0, "height": 0, "mode": "",
                   "file_kb": round(path.stat().st_size / 1024, 1), "readable": True, "error": ""}
            try:
                with Image.open(path) as image:
                    image.load()                       # full decode: catches truncated files
                    row.update(width=image.width, height=image.height, mode=image.mode)
            except Exception as error:                 # PIL raises many types for broken files
                row.update(readable=False, error=f"{type(error).__name__}: {error}")
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"[2.1] no class folders with images in {raw_dir}")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------- [2.2]
def difference_hash(path, hash_size=8):
    """[2.2a] 64-bit difference hash (dHash): a fingerprint that survives resizing and re-compression.

    Flow:     called by clean_dataset()
    Method:   grey, resize to (hash_size+1) x hash_size, bit = "pixel brighter than its right neighbour".
              Two images whose hashes differ in only a few bits look almost identical.

    Inputs:
        path:      image file.
        hash_size: 8 -> 64 bits.

    Returns:
        np.ndarray bool (hash_size * hash_size,)
    """
    with Image.open(path) as image:
        small = np.asarray(ImageOps.exif_transpose(image).convert("L").resize((hash_size + 1, hash_size),
                                                                              Image.Resampling.LANCZOS))
    return (small[:, 1:] > small[:, :-1]).ravel()


def find_duplicates(hashes, max_distance=4):
    """[2.2b] Group images whose hashes are within max_distance bits.

    Flow:     called by clean_dataset()
    Method:   all-pairs Hamming distance with NumPy (fine for a few thousand images), then union-find so
              chains of near-copies form one group.

    Inputs:
        hashes:       np.ndarray bool (N, 64).
        max_distance: largest Hamming distance counted as a duplicate (0 = exact hash match).

    Returns:
        list of groups, each a sorted list of row indices with 2+ members.
    """
    count = len(hashes)
    parent = list(range(count))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    packed = np.packbits(hashes, axis=1)
    for index in range(count - 1):
        distances = np.unpackbits(packed[index] ^ packed[index + 1:], axis=1).sum(axis=1)
        for other in np.nonzero(distances <= max_distance)[0] + index + 1:
            parent[root(int(other))] = root(index)
    groups = {}
    for index in range(count):
        groups.setdefault(root(index), []).append(index)
    return [sorted(members) for members in groups.values() if len(members) > 1]


def clean_dataset(manifest, min_side=64, max_aspect=3.0, dedup_distance=4):
    """[2.2] Remove unreadable, too-small, oddly shaped and duplicate images.

    Purpose:  Bad images waste training and duplicates leak between train and test (inflated scores).
    Flow:     called by run_task2()   previous: [2.1]   next step: [2.3] split_dataset
    Method:   rules applied in order, each removal logged with its reason:
              1. unreadable file            2. min(width, height) < min_side
              3. aspect ratio > max_aspect  4. duplicate group (dHash within dedup_distance bits):
                 same label -> keep the largest copy; mixed labels -> drop the whole group (label conflict).
              Images that are greyscale / CMYK / palette are kept and converted to RGB at load time.

    Inputs:
        manifest:       DataFrame from [2.1].
        min_side:       smallest allowed width/height, pixels.
        max_aspect:     largest allowed long side / short side.
        dedup_distance: Hamming distance for near-duplicates (0 = exact only).

    Returns:
        (clean_manifest: DataFrame (+ column 'hash'), removal_log: DataFrame path, label, reason)
    """
    removed = []

    def drop(frame, mask, reason):
        removed.extend({"path": p, "label": l, "reason": reason} for p, l in zip(frame.path[mask], frame.label[mask]))
        return frame[~mask].reset_index(drop=True)

    frame = manifest.copy()
    frame = drop(frame, ~frame.readable, "unreadable / corrupt file")
    frame = drop(frame, frame[["width", "height"]].min(axis=1) < min_side, f"smaller than {min_side} px")
    aspect = frame[["width", "height"]].max(axis=1) / frame[["width", "height"]].min(axis=1)
    frame = drop(frame, aspect > max_aspect, f"aspect ratio above {max_aspect}")

    hashes = np.stack([difference_hash(p) for p in frame.path]) if len(frame) else np.zeros((0, 64), bool)
    duplicate_mask = np.zeros(len(frame), bool)
    reasons = {}
    for group in find_duplicates(hashes, dedup_distance):
        labels = set(frame.label.iloc[group])
        if len(labels) > 1:
            for index in group:
                reasons[index] = f"duplicate with conflicting labels {sorted(labels)}"
            duplicate_mask[group] = True
        else:
            areas = (frame.width * frame.height).iloc[group]
            keep = int(areas.idxmax())
            for index in group:
                if index != keep:
                    reasons[index] = f"duplicate of {Path(frame.path.iloc[keep]).name}"
                    duplicate_mask[index] = True
    for index in np.nonzero(duplicate_mask)[0]:
        removed.append({"path": frame.path.iloc[index], "label": frame.label.iloc[index], "reason": reasons[index]})
    frame["hash"] = [np.packbits(h).tobytes().hex() for h in hashes]
    frame = frame[~duplicate_mask].reset_index(drop=True)
    return frame, pd.DataFrame(removed, columns=["path", "label", "reason"])


# ----------------------------------------------------------------------------------------------------------- [2.3]
def split_dataset(manifest, ratios=(0.70, 0.15, 0.15), seed=42):
    """[2.3] Stratified train/val/test split.

    Purpose:  Three disjoint sets with the same class proportions: train fits weights, val picks epochs and
              models, test is touched once at the end for the honest score.
    Flow:     called by run_task2()   previous: [2.2]   next step: [2.4] make_dataloaders
    Method:   two calls to sklearn train_test_split(stratify=label). Done BEFORE augmentation and after
              de-duplication, so no image (or near copy) appears in two splits.

    Inputs:
        manifest: cleaned DataFrame.
        ratios:   (train, val, test), sums to 1.
        seed:     random seed.

    Returns:
        dict {"train": DataFrame, "val": DataFrame, "test": DataFrame}

    Raises:
        ValueError: ratios do not sum to 1.
    """
    train_ratio, val_ratio, test_ratio = ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"[2.3] split ratios must sum to 1, got {ratios}")
    train, rest = train_test_split(manifest, train_size=train_ratio, stratify=manifest.label, random_state=seed)
    val, test = train_test_split(rest, train_size=val_ratio / (val_ratio + test_ratio), stratify=rest.label,
                                 random_state=seed)
    return {name: part.reset_index(drop=True) for name, part in (("train", train), ("val", val), ("test", test))}


# ----------------------------------------------------------------------------------------------------------- [2.4]
class AddGaussianNoise:
    """[2.4b] Noise injection augmentation on a normalised-range tensor (0..1): x + N(0, std^2), clipped."""

    def __init__(self, std=0.03, probability=0.5):
        self.std, self.probability = std, probability

    def __call__(self, tensor):
        if torch.rand(1).item() < self.probability:
            tensor = (tensor + torch.randn_like(tensor) * self.std).clamp(0.0, 1.0)
        return tensor

    def __repr__(self):
        return f"AddGaussianNoise(std={self.std}, p={self.probability})"


def make_transforms(image_size, train, augment=True):
    """[2.4a] Preprocessing (resize, tensor, normalise) plus train-only augmentation.

    Flow:     called by make_dataloaders() and inference.predict() (same eval transform)
    Method:   eval:  resize short side to 1.14 x size -> centre crop -> tensor -> ImageNet normalisation.
              train: random resized crop (70-100 % area) -> horizontal flip -> rotation +-15 deg ->
                     colour jitter -> tensor -> Gaussian noise -> normalisation.

    Inputs:
        image_size: side length in pixels.
        train:      True for the training split.
        augment:    False gives the eval transform even for train (the "no augmentation" ablation).

    Returns:
        torchvision transform (PIL RGB -> normalised float tensor (3, size, size)).
    """
    normalise = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train and augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.03),
            transforms.ToTensor(),
            AddGaussianNoise(std=0.03, probability=0.5),
            normalise,
        ])
    return transforms.Compose([
        transforms.Resize(int(round(image_size * 1.14))),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalise,
    ])


def load_rgb(path):
    """[2.4c] Open an image as RGB, applying the EXIF rotation (greyscale/CMYK/palette are converted here)."""
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


class ImageTableDataset(Dataset):
    """[2.4d] Dataset over a manifest table: returns (normalised tensor, label id)."""

    def __init__(self, table, classes, transform):
        self.paths = list(table.path)
        self.targets = [classes.index(label) for label in table.label]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return self.transform(load_rgb(self.paths[index])), self.targets[index]


def make_dataloaders(splits, classes, image_size=128, batch_size=32, seed=42, augment=True, num_workers=0):
    """[2.4] Wrap each split in a Dataset and DataLoader.

    Flow:     called by run_task2()   previous: [2.3]   next step: [2.5] build_model / [2.6] train_model

    Inputs:
        splits:      dict from [2.3].
        classes:     ordered list of class names (index = label id).
        image_size:  model input size.
        batch_size:  batch size.
        seed:        shuffling seed.
        augment:     use the augmented train transform.
        num_workers: DataLoader worker processes (keep 0 in Windows notebooks).

    Returns:
        dict {"train": DataLoader, "val": DataLoader, "test": DataLoader}
    """
    generator = torch.Generator().manual_seed(seed)
    loaders = {}
    for name, table in splits.items():
        is_train = name == "train"
        dataset = ImageTableDataset(table, classes, make_transforms(image_size, is_train, augment))
        loaders[name] = DataLoader(dataset, batch_size=batch_size, shuffle=is_train, generator=generator,
                                   num_workers=num_workers, persistent_workers=num_workers > 0,
                                   pin_memory=torch.cuda.is_available())
    return loaders
