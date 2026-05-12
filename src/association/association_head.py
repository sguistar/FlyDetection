from __future__ import annotations

from pathlib import Path

import numpy as np

from core.structures import Detection, Track
from .trajectory_temporal import build_detection_identity_vector

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


ASSOCIATION_FEATURE_NAMES = [
    "appearance_long",
    "appearance_short",
    "temporal_distance",
    "identity_distance",
    "spatial_distance",
    "shape_cost",
    "motion_cost",
    "kf_cost",
    "direction_cost",
    "det_reid_quality",
    "memory_reliability",
    "can_update_reid",
    "is_crowded",
    "is_merged_risk",
    "normalized_hits",
    "normalized_missed",
    "has_identity_slot",
    "switch_risk_hint",
    "local_density",
    "border_risk",
]
ASSOCIATION_FEATURE_DIM = len(ASSOCIATION_FEATURE_NAMES)


def _cosine_distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.5
    na = np.linalg.norm(a) + 1e-8
    nb = np.linalg.norm(b) + 1e-8
    return float(1.0 - float(a @ b) / float(na * nb))


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


def build_association_feature_vector(
    *,
    appearance_long: float,
    appearance_short: float,
    temporal_distance: float,
    identity_distance: float,
    spatial_distance: float,
    shape_cost: float,
    motion_cost: float,
    kf_cost: float,
    direction_cost: float,
    det_reid_quality: float,
    memory_reliability: float,
    can_update_reid: bool,
    is_crowded: bool,
    is_merged_risk: bool,
    normalized_hits: float,
    normalized_missed: float,
    has_identity_slot: bool,
    switch_risk_hint: float,
    local_density: float,
    border_risk: float,
) -> np.ndarray:
    """Build the canonical pair-feature vector shared by runtime scoring and training.

    构建运行时评分和训练共用的标准成对特征向量。
    """
    return np.array(
        [
            float(appearance_long),
            float(appearance_short),
            float(temporal_distance),
            float(identity_distance),
            float(spatial_distance),
            float(shape_cost),
            float(motion_cost),
            float(kf_cost),
            float(direction_cost),
            float(det_reid_quality),
            float(memory_reliability),
            float(can_update_reid),
            float(is_crowded),
            float(is_merged_risk),
            float(normalized_hits),
            float(normalized_missed),
            float(has_identity_slot),
            float(np.clip(switch_risk_hint, 0.0, 1.0)),
            float(np.clip(local_density, 0.0, 1.0)),
            float(np.clip(border_risk, 0.0, 1.0)),
        ],
        dtype=np.float32,
    )


def build_pair_feature_vector(
    track: Track,
    det: Detection,
    *,
    appearance_long: float,
    appearance_short: float,
    temporal_distance: float,
    identity_distance: float,
    spatial_distance: float,
    shape_cost: float,
    motion_cost: float,
    kf_cost: float,
    direction_cost: float,
) -> np.ndarray:
    """Bind one track-detection pair to the canonical association feature schema.

    将一组轨迹和检测绑定到标准关联特征结构中。
    """
    return build_association_feature_vector(
        appearance_long=appearance_long,
        appearance_short=appearance_short,
        temporal_distance=temporal_distance,
        identity_distance=identity_distance,
        spatial_distance=spatial_distance,
        shape_cost=shape_cost,
        motion_cost=motion_cost,
        kf_cost=kf_cost,
        direction_cost=direction_cost,
        det_reid_quality=det.reid_quality,
        memory_reliability=track.reid_state.memory_reliability,
        can_update_reid=track.can_update_reid(det.frame_idx),
        is_crowded=det.is_crowded,
        is_merged_risk=det.is_merged_risk,
        normalized_hits=float(np.clip(track.hits / 10.0, 0.0, 1.0)),
        normalized_missed=float(np.clip(track.missed / 10.0, 0.0, 1.0)),
        has_identity_slot=track.identity_slot is not None,
        switch_risk_hint=det.switch_risk_hint,
        local_density=float(det.context_feature[4]) if det.context_feature is not None and det.context_feature.shape[0] > 4 else 0.0,
        border_risk=float(det.context_feature[7]) if det.context_feature is not None and det.context_feature.shape[0] > 7 else 0.0,
    )


if nn is not None:
    class AssociationHeadNet(nn.Module):
        """Small learned head that predicts match confidence and switch risk for a pair.

        小型学习头，用于预测一对轨迹和检测的匹配置信度与切换风险。
        """
        def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2),
            )

        def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            logits = self.net(features)
            return logits[..., :1], logits[..., 1:]
else:  # pragma: no cover
    class AssociationHeadNet:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("torch is required to build the association head.")


def build_association_head(input_dim: int, hidden_dim: int = 64) -> AssociationHeadNet:
    return AssociationHeadNet(input_dim=input_dim, hidden_dim=hidden_dim)


class AssociationHead:
    def __init__(
        self,
        *,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        checkpoint_path: str | None = None,
        device: str = "cpu",
    ) -> None:
        self.embedding_dim = embedding_dim
        self.input_dim = ASSOCIATION_FEATURE_DIM
        self.hidden_dim = hidden_dim
        self.device = device
        self.model = None
        self.loaded_checkpoint = False
        self.status_message = "association_handcrafted_fallback"
        self._build_backend(checkpoint_path)

    def _build_backend(self, checkpoint_path: str | None) -> None:
        if torch is None:
            return
        self.model = build_association_head(self.input_dim, hidden_dim=self.hidden_dim).to(self.device)
        self.model.eval()
        if not checkpoint_path:
            self.status_message = "association_missing_checkpoint_handcrafted_fallback"
            return
        checkpoint_file = Path(checkpoint_path)
        if not checkpoint_file.exists():
            self.status_message = "association_missing_checkpoint_handcrafted_fallback"
            return
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        state_dict = checkpoint.get("association_head_state_dict")
        if not state_dict:
            self.status_message = "association_missing_in_bundle_handcrafted_fallback"
            return
        if _load_compatible_state_dict(self.model, state_dict):
            self.loaded_checkpoint = True
            self.status_message = f"association_loaded:{checkpoint_file.name}"
        else:
            self.status_message = "association_incompatible_checkpoint_handcrafted_fallback"

    def heuristic_scores(self, pair_features: np.ndarray) -> tuple[float, float]:
        """Fallback scoring path when the learned head is disabled or unavailable.

        当学习头被禁用或不可用时使用的回退评分路径。
        """
        appearance_long, appearance_short, temporal_distance, identity_distance, spatial_distance = pair_features[:5]
        shape_cost, motion_cost, kf_cost, direction_cost = pair_features[5:9]
        det_quality, memory_rel = pair_features[9], pair_features[10]
        crowded, merged = pair_features[12], pair_features[13]
        switch_hint = pair_features[17]
        local_density = pair_features[18]
        border_risk = pair_features[19]

        appearance_cost = min(appearance_long, appearance_short, temporal_distance, identity_distance)
        match_cost = (
            0.26 * motion_cost
            + 0.20 * kf_cost
            + 0.10 * direction_cost
            + 0.12 * shape_cost
            + 0.12 * spatial_distance
            + 0.20 * appearance_cost
        )
        match_score = float(np.clip(1.0 - match_cost + 0.15 * det_quality + 0.10 * memory_rel, 0.0, 1.0))
        switch_risk = float(
            np.clip(
                0.24 * appearance_long
                + 0.16 * temporal_distance
                + 0.14 * identity_distance
                + 0.10 * spatial_distance
                + 0.12 * crowded
                + 0.08 * merged
                + 0.06 * local_density
                + 0.05 * border_risk
                + 0.10 * switch_hint
                + 0.10 * (1.0 - memory_rel),
                0.0,
                1.0,
            )
        )
        return match_score, switch_risk

    def score_pair(
        self,
        track: Track,
        det: Detection,
        *,
        appearance_long: float,
        appearance_short: float,
        temporal_distance: float,
        identity_distance: float,
        spatial_distance: float,
        shape_cost: float,
        motion_cost: float,
        kf_cost: float,
        direction_cost: float,
    ) -> tuple[float, float, np.ndarray]:
        """Return match score, switch risk, and the exact feature vector used to produce them.

        返回匹配分数、切换风险，以及生成它们时使用的精确特征向量。
        """
        pair_features = build_pair_feature_vector(
            track,
            det,
            appearance_long=appearance_long,
            appearance_short=appearance_short,
            temporal_distance=temporal_distance,
            identity_distance=identity_distance,
            spatial_distance=spatial_distance,
            shape_cost=shape_cost,
            motion_cost=motion_cost,
            kf_cost=kf_cost,
            direction_cost=direction_cost,
        )
        if self.model is None or torch is None or not self.loaded_checkpoint:
            match_score, switch_risk = self.heuristic_scores(pair_features)
            return match_score, switch_risk, pair_features

        with torch.no_grad():
            tensor = torch.from_numpy(pair_features).unsqueeze(0).to(self.device)
            match_logit, switch_logit = self.model(tensor)
            match_score = float(torch.sigmoid(match_logit)[0, 0].item())
            switch_risk = float(torch.sigmoid(switch_logit)[0, 0].item())
        return match_score, switch_risk, pair_features


def temporal_detection_distance(track: Track, det: Detection, embedding_dim: int) -> float:
    """Compare a track's temporal token against the detection-side identity summary.

    比较轨迹的时间 token 与检测侧身份摘要之间的差异。
    """
    track_token = track.reid_state.temporal_token
    if det.identity_feature is not None:
        det_vector = det.identity_feature.astype(np.float32)
    elif det.embedding is not None:
        det_vector = det.embedding.astype(np.float32)
    else:
        det_vector = build_detection_identity_vector(det, embedding_dim=embedding_dim)
    return _cosine_distance(track_token, det_vector)


def spatial_detection_distance(track: Track, det: Detection, embedding_dim: int) -> float:
    """Compare a track's spatial token against detection spatial evidence.

    比较轨迹的空间 token 与检测空间证据之间的差异。
    """
    track_token = track.reid_state.spatial_token
    if det.spatial_feature is None:
        det_vector = build_detection_identity_vector(det, embedding_dim=embedding_dim)
    else:
        det_vector = det.spatial_feature.astype(np.float32)
    return _cosine_distance(track_token, det_vector)
