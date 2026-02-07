"""Motion extraction helpers."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def read_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img.astype(np.float32)


def compute_motion(
    prev: np.ndarray,
    curr: np.ndarray,
    threshold: float = 15.0,
    blur_ksize: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    if prev.shape != curr.shape:
        raise ValueError("Frame sizes do not match")

    if blur_ksize and blur_ksize > 1:
        prev = cv2.GaussianBlur(prev, (blur_ksize, blur_ksize), 0)
        curr = cv2.GaussianBlur(curr, (blur_ksize, blur_ksize), 0)

    diff = cv2.absdiff(prev, curr)
    mask = diff > threshold
    coords = np.column_stack(np.nonzero(mask))
    weights = diff[mask]
    return coords, weights
