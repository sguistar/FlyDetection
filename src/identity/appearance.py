from __future__ import annotations

import cv2
import numpy as np


def compute_simple_appearance_feature(crop: np.ndarray | None, bins: int = 8) -> np.ndarray | None:
    if crop is None or crop.size == 0:
        return None

    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    hs_hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    hs_hist = cv2.normalize(hs_hist, hs_hist).flatten().astype(np.float32)

    gray_hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
    gray_hist = cv2.normalize(gray_hist, gray_hist).flatten().astype(np.float32)

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    edge_summary = np.array(
        [
            float(np.mean(mag)),
            float(np.std(mag)),
            float(np.percentile(mag, 75)),
            float(np.mean(mag > 10.0)),
        ],
        dtype=np.float32,
    )

    feat = np.concatenate([hs_hist, gray_hist, edge_summary], axis=0).astype(np.float32)
    norm = np.linalg.norm(feat) + 1e-8
    return feat / norm
