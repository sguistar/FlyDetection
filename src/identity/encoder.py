from __future__ import annotations

from pathlib import Path

import numpy as np

from core.structures import Detection
from identity.appearance import compute_simple_appearance_feature
from identity.identity_memory import IdentityMemory
from identity.shape import compute_shape_feature
from identity.spacial_context import SpacialContext
from identity.transforms import build_reid_input

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None


def _resize_feature_vector(feature: np.ndarray, output_dim: int) -> np.ndarray:
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


if nn is not None:
    class AppearanceEncoderNet(nn.Module):
        """Lightweight CNN that converts one crop into the appearance embedding used by the runtime.

        轻量级 CNN，将单个裁剪图转换为运行时使用的外观嵌入。
        """
        def __init__(
            self,
            embedding_dim: int = 128,
            *,
            width: int = 32,
            dropout: float = 0.10,
            num_classes: int = 0,
        ) -> None:
            super().__init__()
            self.embedding_dim = embedding_dim
            self.width = width
            self.dropout = dropout
            self.num_classes = num_classes
            self.backbone = nn.Sequential(
                nn.Conv2d(3, width, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
                nn.Conv2d(width, width * 2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(width * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(width * 2, width * 4, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(width * 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(width * 4, width * 4, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(width * 4),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.projection = nn.Sequential(
                nn.Flatten(),
                nn.Linear(width * 4, width * 4),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(width * 4, embedding_dim),
            )
            self.classifier = nn.Linear(embedding_dim, num_classes) if num_classes > 0 else None

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
            features = self.backbone(x)
            embedding = self.projection(features)
            embedding = F.normalize(embedding, dim=1)
            logits = self.classifier(embedding) if self.classifier is not None else None
            return embedding, logits
else:  # pragma: no cover
    class AppearanceEncoderNet:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("torch is required to build the CNN appearance encoder.")


def build_appearance_encoder(
    embedding_dim: int = 128,
    *,
    width: int = 32,
    dropout: float = 0.10,
    num_classes: int = 0,
) -> AppearanceEncoderNet:
    """Factory wrapper so training and runtime instantiate the same appearance backbone.

    工厂封装，确保训练和运行时实例化同一套外观骨干网络。
    """
    return AppearanceEncoderNet(embedding_dim=embedding_dim, width=width, dropout=dropout, num_classes=num_classes)


class IdentityEncoder:
    """Runtime encoder that assembles appearance, SC, and IM outputs into the final detection embedding.

    运行时编码器，将外观、SC 和 IM 输出组合成最终检测嵌入。
    """
    def __init__(
        self,
        embedding_dim: int = 128,
        *,
        backend: str = "cnn",
        crop_size: tuple[int, int] = (96, 96),
        checkpoint_path: str | None = None,
        cnn_width: int = 32,
        cnn_dropout: float = 0.10,
        identity_hidden_dim: int = 128,
        identity_dropout: float = 0.10,
        spatial_hidden_dim: int = 96,
        spatial_dropout: float = 0.10,
        use_identity_memory: bool = True,
        use_spacial_context: bool = True,
        device: str = "cpu",
        fallback_to_handcrafted_when_untrained: bool = True,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.backend = backend.lower()
        self.crop_size = crop_size
        self.checkpoint_path = checkpoint_path
        self.cnn_width = cnn_width
        self.cnn_dropout = cnn_dropout
        self.identity_hidden_dim = identity_hidden_dim
        self.identity_dropout = identity_dropout
        self.spatial_hidden_dim = spatial_hidden_dim
        self.spatial_dropout = spatial_dropout
        self.use_identity_memory = use_identity_memory
        self.use_spacial_context = use_spacial_context
        self.device = device
        self.fallback_to_handcrafted_when_untrained = fallback_to_handcrafted_when_untrained
        self.model = None
        self.loaded_checkpoint = False
        self.status_message = "handcrafted_fallback"
        self.identity_memory_status = "identity_memory_disabled" if not self.use_identity_memory else "identity_memory_handcrafted_fallback"
        self.spacial_context_status = "spacial_context_disabled" if not self.use_spacial_context else "spacial_context_handcrafted_fallback"
        self.spacial_context = SpacialContext(
            embedding_dim=self.embedding_dim,
            hidden_dim=self.spatial_hidden_dim,
            dropout=self.spatial_dropout,
            checkpoint_path=self.checkpoint_path if self.use_spacial_context else None,
            device=self.device,
        )
        self.identity_memory = IdentityMemory(
            embedding_dim=self.embedding_dim,
            hidden_dim=self.identity_hidden_dim,
            dropout=self.identity_dropout,
            checkpoint_path=self.checkpoint_path if self.use_identity_memory else None,
            device=self.device,
        )
        self.spacial_context_status = self.spacial_context.status_message if self.use_spacial_context else "spacial_context_disabled"
        self.identity_memory_status = self.identity_memory.status_message if self.use_identity_memory else "identity_memory_disabled"
        self._build_backend()

    def _build_backend(self) -> None:
        """Load the CNN appearance branch from checkpoint when configured, otherwise keep fallback mode.

        在配置检查点时加载 CNN 外观分支，否则保持回退模式。
        """
        if self.backend != "cnn" or torch is None:
            return
        self.model = build_appearance_encoder(
            embedding_dim=self.embedding_dim,
            width=self.cnn_width,
            dropout=self.cnn_dropout,
        )
        self.model.to(self.device)
        self.model.eval()
        self.status_message = "cnn_untrained"
        if not self.checkpoint_path:
            if self.fallback_to_handcrafted_when_untrained:
                self.status_message = "cnn_missing_checkpoint_handcrafted_fallback"
            return
        checkpoint_file = Path(self.checkpoint_path)
        if not checkpoint_file.exists():
            if self.fallback_to_handcrafted_when_untrained:
                self.status_message = "cnn_missing_checkpoint_handcrafted_fallback"
            return
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        state_dict = checkpoint.get("appearance_state_dict", checkpoint.get("state_dict", checkpoint))
        if _load_compatible_state_dict(self.model, state_dict):
            self.loaded_checkpoint = True
            self.status_message = f"cnn_loaded:{checkpoint_file.name}"
        else:
            self.status_message = "cnn_incompatible_checkpoint_handcrafted_fallback"

    def _encode_handcrafted(self, det: Detection) -> np.ndarray | None:
        """Fallback appearance encoding path used when CNN weights are absent or disabled.

        当 CNN 权重缺失或被禁用时使用的外观编码回退路径。
        """
        if det.crop is None or det.crop.size == 0:
            return None
        prepared = build_reid_input(det.crop, backend="handcrafted", size=self.crop_size)
        det.embedding_source = "handcrafted"
        det.appearance_feature = compute_simple_appearance_feature(prepared)
        return det.appearance_feature

    def _encode_cnn(self, det: Detection) -> np.ndarray | None:
        """Run the trained CNN appearance encoder on one detection crop.

        对单个检测裁剪图运行已训练的 CNN 外观编码器。
        """
        if self.model is None or torch is None or det.crop is None or det.crop.size == 0:
            return None
        det.embedding_source = "cnn"
        tensor = build_reid_input(det.crop, backend="cnn", size=self.crop_size)
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.from_numpy(np.asarray(tensor, dtype=np.float32))
        tensor = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding, _ = self.model(tensor)
        return embedding[0].detach().cpu().numpy().astype(np.float32)

    def _finalize_embedding(self, vector: np.ndarray | None) -> np.ndarray | None:
        """Resize and normalize any feature vector so downstream modules see a consistent embedding width.

        调整并归一化任意特征向量，使下游模块看到一致的嵌入宽度。
        """
        if vector is None:
            return None
        vector = _resize_feature_vector(vector, self.embedding_dim)
        norm = np.linalg.norm(vector) + 1e-8
        return (vector / norm).astype(np.float32)

    def encode_detection(self, det: Detection) -> np.ndarray | None:
        """Populate shape, appearance, spatial, identity, and final embedding fields for one detection.

        为单个检测填充形状、外观、空间、身份以及最终嵌入字段。
        """
        det.shape_feature = compute_shape_feature(det)
        det.appearance_feature = None
        det.identity_feature = None
        det.spatial_feature = None
        detector_prefix = "rescue_" if det.is_rescued or det.detector_source == "rescue" else ""
        
        if det.embedding is None:
            det.embedding_source = "none"

        if self.backend == "cnn":
            if self.loaded_checkpoint or not self.fallback_to_handcrafted_when_untrained:
                det.appearance_feature = self._encode_cnn(det)
            if det.appearance_feature is None and self.fallback_to_handcrafted_when_untrained:
                det.appearance_feature = self._encode_handcrafted(det)
                if not self.loaded_checkpoint:
                    self.status_message = "cnn_missing_checkpoint_handcrafted_fallback"
        else:
            det.appearance_feature = self._encode_handcrafted(det)
            self.status_message = "handcrafted"

        base_embedding = self._finalize_embedding(det.appearance_feature)
        if self.use_spacial_context:
            det.spatial_feature = self.spacial_context.encode_detection(det)
        elif det.shape_feature is not None:
            det.spatial_feature = self._finalize_embedding(det.shape_feature)
        else:
            det.spatial_feature = None

        if self.use_identity_memory:
            det.identity_feature = self.identity_memory.encode_features(base_embedding, det.spatial_feature)
        else:
            det.identity_feature = base_embedding

        det.embedding = det.identity_feature if det.identity_feature is not None else base_embedding
        if det.embedding is not None and detector_prefix and not det.embedding_source.startswith(detector_prefix):
            det.embedding_source = f"{detector_prefix}{det.embedding_source}"
        if det.embedding is not None:
            suffixes = []
            if self.use_identity_memory:
                suffixes.append("ia")
            if self.use_spacial_context:
                suffixes.append("sa")
            if suffixes:
                det.embedding_source = f"{det.embedding_source}_{'+'.join(suffixes)}"
        return det.embedding
