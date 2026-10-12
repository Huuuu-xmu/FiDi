"""Image loading, native-resolution center cropping and boundary targets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _crop(image: Image.Image, *, input_size: int = 448) -> Image.Image:
    if input_size <= 0:
        raise ValueError("input_size must be positive")
    width, height = image.size
    if min(width, height) < input_size:
        raise ValueError("Source image is smaller than the requested crop; resizing and padding are not applied")
    left = (width - input_size) // 2
    top = (height - input_size) // 2
    return image.crop((left, top, left + input_size, top + input_size))


def load_image(path: str | Path, *, input_size: int = 448) -> np.ndarray:
    with Image.open(path) as image:
        image = _crop(image.convert("RGB"), input_size=input_size)
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return (array - 0.5) / 0.5


def load_mask(path: str | Path, *, input_size: int = 448) -> np.ndarray:
    with Image.open(path) as mask:
        array = np.asarray(_crop(mask, input_size=input_size), dtype=np.uint8)
    if array.ndim != 2 or not np.isin(array, range(7)).all():
        raise ValueError(f"Expected a 2D label mask with values 0..6: {path}")
    return array


class ManifestDataset:
    def __init__(self, rows, *, input_size: int = 448, boundary_classes=(1, 4)):
        self.rows = rows
        self.input_size = input_size
        self.boundary_classes = boundary_classes

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import cv2
        import torch

        row = self.rows[index]
        image = load_image(row["image_path"], input_size=self.input_size)
        mask = load_mask(row["mask_path"], input_size=self.input_size)
        # LSLE uses closed/open glottis labels 1 and 4. A different task must
        # supply its own glottis label IDs explicitly.
        glottis = np.isin(mask, self.boundary_classes).astype(np.uint8) * 255
        boundary = (cv2.Canny(glottis, 1, 3) > 0).astype(np.float32)
        return (
            torch.from_numpy(image.copy()),
            torch.from_numpy(mask.astype(np.int64)),
            torch.from_numpy(boundary),
        )
