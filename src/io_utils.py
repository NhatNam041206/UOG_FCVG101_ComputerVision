"""[IO] The only module in src/ that reads or writes files (rule R3, R4).

Flow: used by run_task*() runners and app.py; core functions never call it.
"""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(folder):
    """[IO] List image files in a folder, sorted by name.

    Inputs:
        folder: Path to a directory.

    Returns:
        list of Path, only files with an image extension.

    Raises:
        FileNotFoundError: the folder does not exist.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def read_image(path, max_side=None):
    """[IO] Read an image as BGR uint8, shape (H, W, 3).

    Inputs:
        path:     Path to an image file.
        max_side: if set, shrink (never enlarge) so the longer side is at most this many pixels,
                  keeping the aspect ratio (cv2.INTER_AREA).

    Returns:
        np.ndarray, BGR uint8.

    Raises:
        FileNotFoundError: the file is missing or cannot be decoded.
    """
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if max_side:
        scale = max_side / max(image.shape[:2])
        if scale < 1:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image


def decode_image_bytes(data):
    """[IO] Decode uploaded file bytes (Streamlit) to BGR uint8.

    Inputs:
        data: bytes of a JPG/PNG file.

    Returns:
        np.ndarray, BGR uint8.

    Raises:
        ValueError: the bytes are not a readable image.
    """
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded file is not a readable image.")
    return image


def save_image(image, path):
    """[IO] Write a BGR or grayscale uint8 image, creating parent folders.

    Inputs:
        image: np.ndarray, uint8.
        path:  output Path (.jpg or .png).

    Returns:
        Path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    return path


def save_json(data, path):
    """[IO] Write a dict as indented JSON (metrics.json files, rule R7).

    Inputs:
        data: JSON-serialisable dict (NumPy values are converted).
        path: output Path.

    Returns:
        Path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_to_builtin), encoding="utf-8")
    return path


def save_csv(rows, path):
    """[IO] Write a list of flat dicts as a CSV table (UTF-8 with BOM so Excel opens it cleanly).

    Inputs:
        rows: list of dicts; columns are every key seen, in first-seen order (missing cells are blank).
        path: output Path.

    Returns:
        Path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def download_and_extract(url, dest_dir):
    """[2.0] Download a dataset archive (.tgz / .tar.gz / .zip) once and unpack it into dest_dir.

    Flow:     called by train.run_task2() only when the dataset folder is missing or empty.

    Inputs:
        url:      archive URL.
        dest_dir: folder to unpack into (normally data/raw).

    Returns:
        Path of dest_dir.

    Raises:
        urllib.error.URLError: the download failed (no network): download the archive by hand and unpack it
        into dest_dir instead.
    """
    import shutil
    import tarfile
    import urllib.request
    import zipfile

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive = dest_dir / Path(url).name
    if not archive.is_file():
        with urllib.request.urlopen(url) as response, archive.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(dest_dir)
    else:
        with tarfile.open(archive) as bundle:
            bundle.extractall(dest_dir, filter="data")
    archive.unlink()
    return dest_dir


def _to_builtin(value):
    """Convert NumPy scalars/arrays so json can write them."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def to_rgb(image):
    """[IO] Convert BGR to RGB for display only (rule R4). Grayscale is returned unchanged.

    Inputs:
        image: np.ndarray, BGR (H, W, 3) or grayscale (H, W).

    Returns:
        np.ndarray in RGB order (or the same grayscale array).
    """
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
