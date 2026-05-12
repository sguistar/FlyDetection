from __future__ import annotations

import numpy as np
try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from preprocessing.normalization import normalize_bgr_to_tensor_input, prepare_handcrafted_crop


def build_reid_input(
    crop: np.ndarray,
    *,
    backend: str = "cnn",
    size: tuple[int, int] = (96, 96),
) -> np.ndarray | "torch.Tensor":
    backend = backend.lower()
    prepared = prepare_handcrafted_crop(crop, size=size)
    if backend == "handcrafted":
        return prepared
    if backend == "cnn":
        array = normalize_bgr_to_tensor_input(prepared)
        if torch is None:
            return array
        return torch.from_numpy(array)
    raise NotImplementedError(f"Encoder backend '{backend}' is not implemented in this scope.")
