"""Image preprocessing recovered from the development training script."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _crop(image: Image.Image, *, is_mask: bool, input_size: int = 448, resize_width: int = 600) -> Image.Image:
    if input_size <= 0 or resize_width < input_size:
        raise ValueError("input_size must be positive and resize_width >= input_size")
    resampling = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
    image = image.resize((resize_width, input_size), resampling)
    left = (resize_width - input_size) // 2
    return image.crop((left, 0, left + input_size, input_size))


def load_image(path: str | Path, *, input_size: int = 448, resize_width: int = 600) -> np.ndarray:
    with Image.open(path) as image:
        image = _crop(image.convert("RGB"), is_mask=False, input_size=input_size, resize_width=resize_width)
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return (array - 0.5) / 0.5


def load_mask(path: str | Path, *, input_size: int = 448, resize_width: int = 600) -> np.ndarray:
    with Image.open(path) as mask:
        array = np.asarray(_crop(mask, is_mask=True, input_size=input_size, resize_width=resize_width), dtype=np.uint8)
    if array.ndim != 2 or not np.isin(array, range(7)).all():
        raise ValueError(f"Expected a 2D label mask with values 0..6: {path}")
    return array


class ManifestDataset:
    def __init__(self, rows, *, input_size: int = 448, resize_width: int = 600, boundary_classes=(1, 4)):
        self.rows = rows
        self.input_size = input_size
        self.resize_width = resize_width
        self.boundary_classes = boundary_classes

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import cv2
        import torch

        row = self.rows[index]
        image = load_image(row["image_path"], input_size=self.input_size, resize_width=self.resize_width)
        mask = load_mask(row["mask_path"], input_size=self.input_size, resize_width=self.resize_width)
        # LSLE uses closed/open glottis labels 1 and 4. A different task must
        # supply its own glottis label IDs explicitly.
        glottis = np.isin(mask, self.boundary_classes).astype(np.uint8) * 255
        boundary = (cv2.Canny(glottis, 1, 3) > 0).astype(np.float32)
        return (
            torch.from_numpy(image.copy()),
            torch.from_numpy(mask.astype(np.int64)),
            torch.from_numpy(boundary),
        )
