from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None


def _normalize(vector: np.ndarray | None) -> np.ndarray | None:
    if vector is None:
        return None
    vector = vector.astype(np.float32)
    return vector / (np.linalg.norm(vector) + 1e-8)


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
    class IdentityMemoryNet(nn.Module):
        """Fuse appearance and spatial embeddings into the identity representation used downstream.

        将外观和空间嵌入融合为下游使用的身份表示。
        """
        def __init__(self, embedding_dim: int, hidden_dim: int = 128, dropout: float = 0.10) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(embedding_dim * 2, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embedding_dim),
            )

        def forward(self, appearance_embedding: torch.Tensor, spatial_embedding: torch.Tensor) -> torch.Tensor:
            fused = torch.cat([appearance_embedding, spatial_embedding], dim=-1)
            return F.normalize(self.net(fused), dim=-1)
else:  # pragma: no cover
    class IdentityMemoryNet:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("torch is required to build the identity memory.")


def build_identity_memory(
    *,
    embedding_dim: int = 128,
    hidden_dim: int = 128,
    dropout: float = 0.10,
) -> IdentityMemoryNet:
    """Factory wrapper so runtime and training instantiate the same IM module.

    工厂封装，确保运行时和训练阶段实例化同一套 IM 模块。
    """
    return IdentityMemoryNet(
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )


class IdentityMemory:
    def __init__(
        self,
        *,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
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
        self.status_message = "identity_memory_handcrafted_fallback"
        self._build_backend(checkpoint_path)

    def _build_backend(self, checkpoint_path: str | None) -> None:
        if torch is None:
            return
        self.model = build_identity_memory(
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        ).to(self.device)
        self.model.eval()
        if not checkpoint_path:
            self.status_message = "identity_memory_missing_checkpoint_handcrafted_fallback"
            return
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            self.status_message = "identity_memory_missing_checkpoint_handcrafted_fallback"
            return
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        state_dict = checkpoint.get("identity_memory_state_dict")
        if not state_dict:
            self.status_message = "identity_memory_missing_in_bundle_handcrafted_fallback"
            return
        if _load_compatible_state_dict(self.model, state_dict):
            self.loaded_checkpoint = True
            self.status_message = f"identity_memory_loaded:{checkpoint_file.name}"
        else:
            self.status_message = "identity_memory_incompatible_checkpoint_handcrafted_fallback"

    def _encode_handcrafted(
        self,
        appearance_embedding: np.ndarray | None,
        spatial_embedding: np.ndarray | None,
    ) -> np.ndarray | None:
        """Fallback fusion rule that keeps IM usable even without a trained checkpoint.

        在没有已训练检查点时仍能使用回退融合规则。
        """
        if appearance_embedding is None and spatial_embedding is None:
            return None
        if appearance_embedding is None:
            return _normalize(spatial_embedding)
        if spatial_embedding is None:
            return _normalize(appearance_embedding)
        fused = 0.75 * appearance_embedding.astype(np.float32) + 0.25 * spatial_embedding.astype(np.float32)
        return _normalize(fused)

    def encode_features(
        self,
        appearance_embedding: np.ndarray | None,
        spatial_embedding: np.ndarray | None,
    ) -> np.ndarray | None:
        """Fuse appearance and spatial inputs into the identity embedding consumed by association.

        将外观和空间输入融合为关联模块消费的身份嵌入。
        """
        if self.model is None or torch is None or not self.loaded_checkpoint:
            return self._encode_handcrafted(appearance_embedding, spatial_embedding)
        if appearance_embedding is None and spatial_embedding is None:
            return None
        if appearance_embedding is None:
            appearance_embedding = spatial_embedding
        if spatial_embedding is None:
            spatial_embedding = appearance_embedding
        if appearance_embedding is None or spatial_embedding is None:
            return None
        with torch.no_grad():
            appearance_tensor = torch.from_numpy(appearance_embedding.astype(np.float32)).unsqueeze(0).to(self.device)
            spatial_tensor = torch.from_numpy(spatial_embedding.astype(np.float32)).unsqueeze(0).to(self.device)
            embedding = self.model(appearance_tensor, spatial_tensor)[0].detach().cpu().numpy().astype(np.float32)
        return _normalize(embedding)
