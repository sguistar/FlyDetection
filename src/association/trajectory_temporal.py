from __future__ import annotations

from pathlib import Path

import numpy as np

from core.structures import Detection, Track
from identity.spacial_context import SPACIAL_CONTEXT_DIM
from motion.kinematics import acceleration_from_history, velocity_from_history

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


TEMPORAL_SCALAR_DIM = 14


def _normalize(vector: np.ndarray | None) -> np.ndarray | None:
    if vector is None:
        return None
    vector = vector.astype(np.float32)
    return vector / (np.linalg.norm(vector) + 1e-8)


def _resize_vector(vector: np.ndarray, output_dim: int) -> np.ndarray:
    if vector.shape[0] == output_dim:
        return vector.astype(np.float32)
    x_old = np.linspace(0.0, 1.0, num=vector.shape[0], dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=output_dim, dtype=np.float32)
    return np.interp(x_new, x_old, vector.astype(np.float32)).astype(np.float32)


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


def build_detection_identity_vector(det: Detection, embedding_dim: int) -> np.ndarray:
    """Collapse one detection's identity cues into a single fixed-width vector for TT fallback logic.

    将单个检测的身份线索压缩为固定宽度向量，供 TT 回退逻辑使用。
    """
    pieces: list[np.ndarray] = []
    if det.embedding is not None:
        pieces.append(_resize_vector(det.embedding.astype(np.float32), embedding_dim))
    if det.identity_feature is not None:
        pieces.append(_resize_vector(det.identity_feature.astype(np.float32), embedding_dim))
    if det.spatial_feature is not None:
        pieces.append(_resize_vector(det.spatial_feature.astype(np.float32), embedding_dim))
    if det.shape_feature is not None:
        pieces.append(_resize_vector(det.shape_feature.astype(np.float32), embedding_dim))
    if det.context_feature is not None:
        pieces.append(_resize_vector(det.context_feature.astype(np.float32), embedding_dim))
    if not pieces:
        return np.zeros((embedding_dim,), dtype=np.float32)
    vector = np.mean(np.stack(pieces, axis=0), axis=0)
    normalized = _normalize(vector)
    if normalized is None:
        return np.zeros((embedding_dim,), dtype=np.float32)
    return normalized


def build_temporal_scalar_features(
    *,
    vx: float,
    vy: float,
    ax: float,
    ay: float,
    area: float,
    aspect: float,
    reid_quality: float,
    is_crowded: bool,
    is_merged_risk: bool,
    interpolated: bool,
    frame_gap: float,
    memory_reliability: float,
    x_norm: float,
    y_norm: float,
) -> np.ndarray:
    """Pack motion, quality, and coarse context scalars in the exact order used by TT.

    按 TT 使用的精确顺序打包运动、质量和粗略上下文标量。
    """
    return np.array(
        [
            float(vx),
            float(vy),
            float(ax),
            float(ay),
            float(area),
            float(aspect),
            float(np.clip(reid_quality, 0.0, 1.0)),
            float(is_crowded),
            float(is_merged_risk),
            float(interpolated),
            float(max(frame_gap, 0.0)),
            float(np.clip(memory_reliability, 0.0, 1.0)),
            float(np.clip(x_norm, 0.0, 1.0)),
            float(np.clip(y_norm, 0.0, 1.0)),
        ],
        dtype=np.float32,
    )


def build_track_sequence_features(
    track: Track,
    *,
    history_len: int = 16,
    embedding_dim: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the sequence tensor and padding mask consumed by the trajectory temporal.

    构建时间适配器消费的序列张量和 padding mask。
    """
    observations = sorted(track.trajectory, key=lambda obs: obs.frame_idx)
    all_features_by_frame = {
        int(item["frame_idx"]): item
        for item in track.feature_history
        if item.get("frame_idx") is not None
    }
    features_by_frame = {
        int(item["frame_idx"]): item
        for item in track.recent_feature_items(trusted_only=True, trust_for="temporal")
        if item.get("frame_idx") is not None
        and not bool(item.get("is_merged_risk", False))
        and float(item.get("reid_quality", 0.0)) >= 0.15
    }
    embeddings_by_frame = {
        int(rec.frame_idx): rec.vector.astype(np.float32)
        for rec in track.embedding_records
        if rec.frame_idx is not None
    }

    if not observations:
        empty = np.zeros((history_len, embedding_dim * 2 + TEMPORAL_SCALAR_DIM), dtype=np.float32)
        mask = np.ones((history_len,), dtype=np.float32)
        return empty, mask

    trusted_observations = [obs for obs in observations if obs.frame_idx in features_by_frame]
    weak_evidence = len(trusted_observations) < 2
    active_observations = trusted_observations if not weak_evidence else observations[-min(len(observations), history_len):]
    active_features_by_frame = features_by_frame if not weak_evidence else all_features_by_frame

    active_observations = active_observations[-history_len:]
    seq = np.zeros((history_len, embedding_dim * 2 + TEMPORAL_SCALAR_DIM), dtype=np.float32)
    mask = np.ones((history_len,), dtype=np.float32)

    for out_idx, obs in enumerate(active_observations, start=history_len - len(active_observations)):
        mask[out_idx] = 0.0
        feature_item = active_features_by_frame.get(obs.frame_idx, all_features_by_frame.get(obs.frame_idx, {}))
        embedding = embeddings_by_frame.get(obs.frame_idx)
        if embedding is None:
            embedding = track.short_term_embedding(recent=1)
        if embedding is None:
            embedding = track.prototype_embedding
        if embedding is None:
            embedding = np.zeros((embedding_dim,), dtype=np.float32)
        embedding = _resize_vector(embedding, embedding_dim)
        spatial = feature_item.get("spatial")
        if spatial is None:
            spatial = track.reid_state.spatial_token
        if spatial is None:
            spatial = track.mean_feature("spatial")
        if spatial is None:
            spatial = np.zeros((embedding_dim,), dtype=np.float32)
        spatial = _resize_vector(spatial.astype(np.float32), embedding_dim)

        point_history = [
            (item.frame_idx, item.center[0], item.center[1])
            for item in active_observations[: out_idx - (history_len - len(active_observations)) + 1]
        ]
        vx, vy = velocity_from_history(point_history)
        ax, ay = acceleration_from_history(point_history)
        context = feature_item.get("context")
        if context is None:
            context = np.zeros((SPACIAL_CONTEXT_DIM,), dtype=np.float32)
        else:
            context = _resize_vector(context.astype(np.float32), SPACIAL_CONTEXT_DIM)
        frame_gap = 0.0
        if out_idx > history_len - len(active_observations):
            prev_obs = active_observations[out_idx - (history_len - len(active_observations)) - 1]
            frame_gap = float(max(obs.frame_idx - prev_obs.frame_idx, 0)) / max(float(history_len), 1.0)
        reid_quality = float(feature_item.get("reid_quality", 0.0))
        memory_reliability = float(track.reid_state.memory_reliability)
        if weak_evidence:
            reid_quality = min(reid_quality, 0.35)
            memory_reliability *= 0.5
        scalars = build_temporal_scalar_features(
            vx=vx,
            vy=vy,
            ax=ax,
            ay=ay,
            area=float(feature_item.get("area", 0.0)),
            aspect=float(feature_item.get("aspect", 0.0)),
            reid_quality=reid_quality,
            is_crowded=bool(feature_item.get("is_crowded", False)),
            is_merged_risk=bool(feature_item.get("is_merged_risk", False)),
            interpolated=bool(obs.interpolated),
            frame_gap=frame_gap,
            memory_reliability=memory_reliability,
            x_norm=float(context[0]) if context.shape[0] > 0 else 0.0,
            y_norm=float(context[1]) if context.shape[0] > 1 else 0.0,
        )
        seq[out_idx] = np.concatenate([embedding, spatial, scalars], axis=0)
    return seq, mask


if nn is not None:
    class TemporalPositionEncoding(nn.Module):
        def __init__(self, dim: int, max_len: int = 64) -> None:
            super().__init__()
            position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-np.log(10000.0) / max(dim, 2)))
            pe = torch.zeros(max_len, dim, dtype=torch.float32)
            pe[:, 0::2] = torch.sin(position * div_term)
            if dim > 1:
                pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
            self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.pe[:, : x.shape[1]].to(x.device)

    class TrajectoryTemporalNet(nn.Module):
        """Transformer-GRU hybrid that turns recent track history into one temporal identity token.

        Transformer-GRU 混合网络，将近期轨迹历史编码成一个时间身份 token。
        """
        def __init__(
            self,
            input_dim: int,
            token_dim: int,
            *,
            hidden_dim: int = 128,
            num_layers: int = 1,
            num_heads: int = 4,
            dropout: float = 0.10,
        ) -> None:
            super().__init__()
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.position = TemporalPositionEncoding(hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=max(1, num_heads),
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, num_layers))
            self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
            self.output_proj = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, token_dim),
            )

        def forward(self, sequence: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
            x = self.input_proj(sequence)
            x = self.position(x)
            padding_mask = mask >= 0.5 if mask is not None else None
            x = self.encoder(x, src_key_padding_mask=padding_mask)
            outputs, hidden = self.rnn(x)
            token = hidden[-1]
            if mask is not None:
                valid = (1.0 - mask).sum(dim=1, keepdim=True).clamp(min=1.0)
                pooled = (outputs * (1.0 - mask).unsqueeze(-1)).sum(dim=1) / valid
                token = 0.35 * token + 0.65 * pooled
            token = self.output_proj(token)
            return token / torch.linalg.norm(token, dim=1, keepdim=True).clamp(min=1e-8)
else:  # pragma: no cover
    class TrajectoryTemporalNet:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("torch is required to build the trajectory temporal.")


def build_trajectory_temporal(
    *,
    input_dim: int,
    token_dim: int,
    hidden_dim: int = 128,
    num_layers: int = 1,
    num_heads: int = 4,
    dropout: float = 0.10,
) -> TrajectoryTemporalNet:
    """Factory wrapper so training and runtime instantiate the same TT architecture.

    工厂封装，确保训练和运行时实例化同一套 TT 架构。
    """
    return TrajectoryTemporalNet(
        input_dim=input_dim,
        token_dim=token_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
    )


class TrajectoryTemporal:
    def __init__(
        self,
        *,
        embedding_dim: int = 128,
        history_len: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.10,
        checkpoint_path: str | None = None,
        device: str = "cpu",
    ) -> None:
        self.embedding_dim = embedding_dim
        self.history_len = history_len
        self.input_dim = embedding_dim * 2 + TEMPORAL_SCALAR_DIM
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.device = device
        self.model = None
        self.loaded_checkpoint = False
        self.status_message = "trajectory_temporal_handcrafted_fallback"
        self._build_backend(checkpoint_path)

    def _build_backend(self, checkpoint_path: str | None) -> None:
        if torch is None:
            return
        self.model = build_trajectory_temporal(
            input_dim=self.input_dim,
            token_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
        ).to(self.device)
        self.model.eval()
        if not checkpoint_path:
            self.status_message = "trajectory_temporal_missing_checkpoint_handcrafted_fallback"
            return
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            self.status_message = "trajectory_temporal_missing_checkpoint_handcrafted_fallback"
            return
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        state_dict = checkpoint.get("trajectory_temporal_state_dict")
        if not state_dict:
            self.status_message = "trajectory_temporal_missing_in_bundle_handcrafted_fallback"
            return
        if _load_compatible_state_dict(self.model, state_dict):
            self.loaded_checkpoint = True
            self.status_message = f"trajectory_temporal_loaded:{checkpoint_file.name}"
        else:
            self.status_message = "trajectory_temporal_incompatible_checkpoint_handcrafted_fallback"

    def _encode_handcrafted(self, track: Track) -> np.ndarray | None:
        """Fallback token built by pooling trusted sequence features when no trained TT is available.

        当没有可用的已训练 TT 时，通过池化可信序列特征构建回退 token。
        """
        sequence, mask = build_track_sequence_features(
            track,
            history_len=self.history_len,
            embedding_dim=self.embedding_dim,
        )
        valid = sequence[mask < 0.5]
        if valid.size == 0:
            return track.prototype_embedding
        pooled = valid.mean(axis=0)
        token = _resize_vector(pooled, self.embedding_dim)
        return _normalize(token)

    def encode_track(self, track: Track) -> np.ndarray | None:
        """Encode one track into a temporal token, with safe fallback for short or noisy histories.

        将单条轨迹编码为时间 token，并为过短或噪声较多的历史提供安全回退。
        """
        if self.model is None or torch is None or not self.loaded_checkpoint:
            return self._encode_handcrafted(track)
        sequence, mask = build_track_sequence_features(
            track,
            history_len=self.history_len,
            embedding_dim=self.embedding_dim,
        )
        if int(np.sum(mask < 0.5)) < 2:
            return self._encode_handcrafted(track)
        with torch.no_grad():
            seq_tensor = torch.from_numpy(sequence).unsqueeze(0).to(self.device)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(self.device)
            token = self.model(seq_tensor, mask_tensor)[0].detach().cpu().numpy().astype(np.float32)
        return _normalize(token)
