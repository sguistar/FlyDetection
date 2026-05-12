from __future__ import annotations

from pathlib import Path

import numpy as np

from core.structures import Detection
from identity.shape import SHAPE_FEATURE_DIM, compute_crop_shape_descriptor

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None


SPACIAL_CONTEXT_DIM = 8
SPACIAL_INPUT_DIM = SHAPE_FEATURE_DIM + SPACIAL_CONTEXT_DIM


def _normalize(vector: np.ndarray | None) -> np.ndarray | None:
    if vector is None:
        return None
    vector = vector.astype(np.float32)
    return vector / (np.linalg.norm(vector) + 1e-8)


def _resize_vector(feature: np.ndarray, output_dim: int) -> np.ndarray:
    if feature.shape[0] == output_dim:
        return feature.astype(np.float32)
    x_old = np.linspace(0.0, 1.0, num=feature.shape[0], dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=output_dim, dtype=np.float32)
    return np.interp(x_new, x_old, feature.astype(np.float32)).astype(np.float32)


def _load_compatible_state_dict(model, state_dict: dict) -> bool:
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape)
    }
    if not compatible:
        return False
    model.load_state_dict(compatible, strict=False)
    return True


def pseudo_context_from_crop(crop: np.ndarray | None) -> np.ndarray:
    """Estimate coarse spacial context from a crop when richer scene metadata is unavailable.

    当缺少更丰富的场景元数据时，从裁剪图估计粗略空间上下文。
    """
    descriptor = compute_crop_shape_descriptor(crop)
    if descriptor.shape[0] < 5:
        descriptor = np.pad(descriptor, (0, 5 - descriptor.shape[0]), constant_values=0.0)
    occupancy, centroid_x, centroid_y, spread_x, spread_y = descriptor[:5].tolist()
    context = np.array(
        [
            centroid_x,
            centroid_y,
            1.0,
            1.0,
            0.0,
            float(np.clip(occupancy, 0.0, 1.0)),
            float(np.clip(occupancy * 0.5 + 0.25 * (spread_x + spread_y), 0.0, 1.0)),
            0.0,
        ],
        dtype=np.float32,
    )
    return context


def build_detection_spatial_input(det: Detection) -> np.ndarray:
    """Assemble the SC input vector for one detection from shape and context cues.

    根据形状和上下文线索组装单个检测的 SC 输入向量。
    """
    shape_feature = (
        det.shape_feature.astype(np.float32)
        if det.shape_feature is not None
        else np.zeros((SHAPE_FEATURE_DIM,), dtype=np.float32)
    )
    context_feature = (
        det.context_feature.astype(np.float32)
        if det.context_feature is not None
        else pseudo_context_from_crop(det.crop)
    )
    if context_feature.shape[0] < SPACIAL_CONTEXT_DIM:
        context_feature = np.pad(
            context_feature,
            (0, SPACIAL_CONTEXT_DIM - context_feature.shape[0]),
            constant_values=0.0,
        )
    context_feature = context_feature[:SPACIAL_CONTEXT_DIM]
    return np.concatenate([shape_feature[:SHAPE_FEATURE_DIM], context_feature], axis=0).astype(np.float32)


def build_crop_spatial_input(crop: np.ndarray | None) -> np.ndarray:
    """Create a train-time SC input directly from an image crop.

    直接从图像裁剪生成训练阶段使用的 SC 输入。
    """
    shape_feature = compute_crop_shape_descriptor(crop)
    if shape_feature.shape[0] < SHAPE_FEATURE_DIM:
        shape_feature = np.pad(shape_feature, (0, SHAPE_FEATURE_DIM - shape_feature.shape[0]), constant_values=0.0)
    shape_feature = shape_feature[:SHAPE_FEATURE_DIM]
    context_feature = pseudo_context_from_crop(crop)
    return np.concatenate([shape_feature, context_feature], axis=0).astype(np.float32)


if nn is not None:
    class SpacialContextNet(nn.Module):
        """Small MLP that maps geometric and context features into a spatial embedding.

        小型 MLP，将几何与上下文特征映射为空间嵌入。
        """
        def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int = 96, dropout: float = 0.10) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embedding_dim),
            )

        def forward(self, spatial_inputs: torch.Tensor) -> torch.Tensor:
            embeddings = self.net(spatial_inputs)
            return F.normalize(embeddings, dim=-1)
else:  # pragma: no cover
    class SpacialContextNet:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("torch is required to build the spacial context.")


def build_spacial_context(
    *,
    input_dim: int = SPACIAL_INPUT_DIM,
    embedding_dim: int = 128,
    hidden_dim: int = 96,
    dropout: float = 0.10,
) -> SpacialContextNet:
    """Factory wrapper so runtime and training share the same SC architecture.

    工厂封装，确保运行时和训练阶段共享同一套 SC 架构。
    """
    return SpacialContextNet(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )


class SpacialContext:
    def __init__(
        self,
        *,
        embedding_dim: int = 128,
        hidden_dim: int = 96,
        dropout: float = 0.10,
        checkpoint_path: str | None = None,
        device: str = "cpu",
    ) -> None:
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.device = device
        self.model = None
        self.loaded_checkpoint = False
        self.status_message = "spacial_context_handcrafted_fallback"
        self._build_backend(checkpoint_path)

    def _build_backend(self, checkpoint_path: str | None) -> None:
        if torch is None:
            return
        self.model = build_spacial_context(
            input_dim=SPACIAL_INPUT_DIM,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        ).to(self.device)
        self.model.eval()
        if not checkpoint_path:
            self.status_message = "spacial_context_missing_checkpoint_handcrafted_fallback"
            return
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            self.status_message = "spacial_context_missing_checkpoint_handcrafted_fallback"
            return
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        state_dict = checkpoint.get("spacial_context_state_dict")
        if not state_dict:
            self.status_message = "spacial_context_missing_in_bundle_handcrafted_fallback"
            return
        if _load_compatible_state_dict(self.model, state_dict):
            self.loaded_checkpoint = True
            self.status_message = f"spacial_context_loaded:{checkpoint_file.name}"
        else:
            self.status_message = "spacial_context_incompatible_checkpoint_handcrafted_fallback"

    def _encode_handcrafted(self, spatial_input: np.ndarray) -> np.ndarray:
        """Fallback spatial token based on normalized handcrafted spatial features.

        基于归一化手工空间特征生成的回退空间 token。
        """
        normalized = _normalize(_resize_vector(spatial_input, self.embedding_dim))
        if normalized is None:
            return np.zeros((self.embedding_dim,), dtype=np.float32)
        return normalized

    def encode_detection(self, det: Detection) -> np.ndarray:
        """Encode one detection into a spatial token, falling back to handcrafted features when needed.

        将单个检测编码为空间 token，必要时回退到手工特征。
        """
        spatial_input = build_detection_spatial_input(det)
        if self.model is None or torch is None or not self.loaded_checkpoint:
            return self._encode_handcrafted(spatial_input)
        with torch.no_grad():
            tensor = torch.from_numpy(spatial_input).unsqueeze(0).to(self.device)
            embedding = self.model(tensor)[0].detach().cpu().numpy().astype(np.float32)
        normalized = _normalize(embedding)
        if normalized is None:
            return np.zeros((self.embedding_dim,), dtype=np.float32)
        return normalized
