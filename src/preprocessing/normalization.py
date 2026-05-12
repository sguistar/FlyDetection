from __future__ import annotations

import cv2
import numpy as np


def resize_with_aspect(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("resize_with_aspect requires a non-empty image.")
    return cv2.resize(image, size)


def normalize_bgr_to_tensor_input(
    image: np.ndarray,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
) -> np.ndarray:
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    img = np.transpose(img, (2, 0, 1))
    return img.astype(np.float32)


def prepare_handcrafted_crop(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("prepare_handcrafted_crop requires a non-empty image.")
    crop = resize_with_aspect(image, size)
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    return crop.astype(np.uint8)
