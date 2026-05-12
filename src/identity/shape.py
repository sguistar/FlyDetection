from __future__ import annotations

import cv2
import numpy as np

from core.structures import Detection


SHAPE_FEATURE_DIM = 10


def _normalize_feature(feature: np.ndarray) -> np.ndarray:
    feature = np.nan_to_num(feature.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(feature, 0.0, 1.0)


def compute_crop_shape_descriptor(crop: np.ndarray | None) -> np.ndarray:
    if crop is None or crop.size == 0:
        return np.zeros((5,), dtype=np.float32)

    if crop.ndim == 2:
        gray = crop.astype(np.uint8)
    else:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape[:2]
    if width <= 0 or height <= 0:
        return np.zeros((5,), dtype=np.float32)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if float(mask.mean()) > 200.0:
        mask = 255 - mask

    occupancy = float(np.mean(mask > 0))
    ys, xs = np.nonzero(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return np.zeros((5,), dtype=np.float32)

    centroid_x = float(xs.mean() / max(width - 1, 1))
    centroid_y = float(ys.mean() / max(height - 1, 1))
    x_spread = float(np.std(xs) / max(width, 1))
    y_spread = float(np.std(ys) / max(height, 1))

    grad_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    edge_density = float(np.mean(cv2.magnitude(grad_x, grad_y) > 12.0))

    return _normalize_feature(
        np.array(
            [
                occupancy,
                centroid_x,
                centroid_y,
                np.clip(x_spread, 0.0, 1.0),
                np.clip(y_spread + 0.5 * edge_density, 0.0, 1.0),
            ],
            dtype=np.float32,
        )
    )


def compute_shape_feature(det: Detection) -> np.ndarray:
    x1, y1, x2, y2 = det.bbox
    width = max(x2 - x1, 1e-6)
    height = max(y2 - y1, 1e-6)
    frame_w, frame_h = det.frame_size
    frame_w = max(frame_w, 1)
    frame_h = max(frame_h, 1)

    area_norm = det.area / float(frame_w * frame_h)
    crop_descriptor = compute_crop_shape_descriptor(det.crop)
    feature = np.array(
        [
            float(np.clip(area_norm, 0.0, 1.0)),
            float(np.clip(width / frame_w, 0.0, 1.0)),
            float(np.clip(height / frame_h, 0.0, 1.0)),
            float(np.clip(width / max(height, 1e-6) / 4.0, 0.0, 1.0)),
            float(np.clip(np.log1p(det.area) / 10.0, 0.0, 1.0)),
            *crop_descriptor.tolist(),
        ],
        dtype=np.float32,
    )
    feature = _normalize_feature(feature)
    if feature.shape[0] < SHAPE_FEATURE_DIM:
        padding = np.zeros((SHAPE_FEATURE_DIM - feature.shape[0],), dtype=np.float32)
        feature = np.concatenate([feature, padding], axis=0)
    return feature[:SHAPE_FEATURE_DIM]
