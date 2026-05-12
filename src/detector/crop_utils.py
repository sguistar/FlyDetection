from __future__ import annotations

import cv2
import numpy as np


def safe_clip_bbox(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(round(x1)), max(width - 1, 0)))
    y1 = max(0, min(int(round(y1)), max(height - 1, 0)))
    x2 = max(0, min(int(round(x2)), width))
    y2 = max(0, min(int(round(y2)), height))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def expand_bbox(
    bbox: tuple[float, float, float, float],
    pad: int = 0,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return x1 - pad, y1 - pad, x2 + pad, y2 + pad


def is_border_bbox(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    margin: int = 2,
) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 <= margin or y1 <= margin or x2 >= width - margin or y2 >= height - margin


def crop_from_bbox(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    out_size: tuple[int, int] | None = None,
    pad: int = 0,
    border_value: int | tuple[int, int, int] = 0,
) -> np.ndarray | None:
    if frame is None or frame.size == 0:
        return None

    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        return None

    padded_bbox = expand_bbox(bbox, pad=pad)
    x1, y1, x2, y2 = safe_clip_bbox(padded_bbox, w, h)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    crop = crop.copy()
    if out_size is not None:
        crop = cv2.resize(crop, out_size)
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    if crop.size == 0:
        return None
    if isinstance(border_value, tuple):
        _ = border_value
    return crop
