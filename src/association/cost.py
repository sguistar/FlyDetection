from __future__ import annotations

import math

import numpy as np

from core.structures import Detection, Track
from motion.kalman_filter import SimpleKalmanFilter
from motion.kinematics import velocity_from_history

from .association_head import AssociationHead, spatial_detection_distance, temporal_detection_distance

_KALMAN = SimpleKalmanFilter()


def _cosine_distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.5
    na = np.linalg.norm(a) + 1e-8
    nb = np.linalg.norm(b) + 1e-8
    return float(1.0 - float(a @ b) / float(na * nb))


def _shape_distance(track: Track, det: Detection) -> float:
    """Compare the track's average shape signature against the current detection.

    比较轨迹的平均形状特征与当前检测目标的形状特征。
    """
    track_shape = track.mean_feature("shape")
    if track_shape is None or det.shape_feature is None:
        return 0.25
    return float(np.linalg.norm(track_shape - det.shape_feature) / np.sqrt(track_shape.shape[0]))


def _spatial_distance(track: Track, det: Detection) -> float:
    """Measure distance between cached track spatial token and detection spatial feature.

    度量缓存的轨迹空间 token 与检测空间特征之间的距离。
    """
    spatial_token = track.reid_state.spatial_token
    spatial_feature = det.spatial_feature
    if spatial_token is None or spatial_feature is None:
        return 0.25
    return _cosine_distance(spatial_token, spatial_feature)


def _motion_distance(track: Track, det: Detection, motion_gate: float) -> float:
    """Normalized center-distance gate based on the predicted track position.

    基于轨迹预测位置计算归一化中心距离门控。
    """
    cx, cy = track.predicted_center if track.predicted_center != (0.0, 0.0) else track.center
    dx, dy = det.center
    return float(np.hypot(cx - dx, cy - dy) / max(motion_gate, 1e-6))


def _kalman_distance(track: Track, det: Detection, kf_gate: float) -> float:
    """Mahalanobis-style Kalman gating distance used to reject implausible matches early.

    计算类似马氏距离的卡尔曼门控距离，用于提前拒绝不合理匹配。
    """
    if track.kf_mean is None or track.kf_cov is None:
        return 0.0
    return float(_KALMAN.gating_distance(track.kf_mean, track.kf_cov, det.center) / max(kf_gate, 1e-6))


def _direction_distance(track: Track, det: Detection) -> float:
    """Penalize detections that move against the recent track direction of travel.

    惩罚与轨迹近期运动方向相反的检测候选。
    """
    vx, vy = velocity_from_history(track.xy_history(include_interpolated=False))
    speed = math.hypot(vx, vy)
    if speed < 1.0:
        return 0.0
    move = (det.center[0] - track.center[0], det.center[1] - track.center[1])
    move_norm = math.hypot(move[0], move[1])
    if move_norm < 1e-6:
        return 0.0
    cos_sim = (vx * move[0] + vy * move[1]) / max(speed * move_norm, 1e-6)
    cos_sim = max(-1.0, min(1.0, cos_sim))
    return float(0.5 * (1.0 - cos_sim))


def _appearance_distance(
    track: Track,
    det: Detection,
    *,
    recent_embedding_window: int | None = None,
) -> tuple[float, float]:
    """Return long-term and short-term appearance distances for the current pair.

    返回当前轨迹-检测对的长期和短期外观距离。
    """
    if det.embedding is None:
        return 1.0, 1.0

    long_term = _cosine_distance(track.prototype_embedding, det.embedding)
    short_term = _cosine_distance(track.short_term_embedding(recent=recent_embedding_window), det.embedding)
    return long_term, short_term


def _identity_distance(
    track: Track,
    det: Detection,
    *,
    recent_embedding_window: int | None = None,
) -> tuple[float, bool]:
    """Compare detection identity features against trusted track identity memory when available.

    在可用时将检测身份特征与可信轨迹身份记忆进行比较。
    """
    if det.identity_feature is not None:
        det_vector = det.identity_feature
    elif det.embedding is not None:
        det_vector = det.embedding
    else:
        return 0.5, False

    track_vector = track.mean_feature("identity", recent=recent_embedding_window, trusted_only=True)
    if track_vector is None:
        track_vector = track.mean_feature("identity", recent=recent_embedding_window)
    if track_vector is None:
        track_vector = track.prototype_embedding
    if track_vector is None:
        return 0.5, False
    return _cosine_distance(track_vector, det_vector), True


def _context_value(det: Detection, index: int, default: float = 0.0) -> float:
    if det.context_feature is None or det.context_feature.shape[0] <= index:
        return float(default)
    return float(det.context_feature[index])


def _identity_conflict_intensity(
    det: Detection,
    *,
    min_risk: float,
    min_density: float,
) -> float:
    """Return how strongly IM should influence this pair based on crowding and switch risk cues.

    根据拥挤程度和切换风险线索，返回 IM 应对该候选对产生多强影响。
    """
    local_density = _context_value(det, 5, 0.0)
    conflict_risk = max(
        float(np.clip(det.switch_risk_hint, 0.0, 1.0)),
        local_density if local_density >= min_density else 0.0,
        0.60 if det.is_crowded else 0.0,
        0.85 if det.is_merged_risk else 0.0,
    )
    if conflict_risk <= min_risk:
        return 0.0
    return float(np.clip((conflict_risk - min_risk) / max(1.0 - min_risk, 1e-6), 0.0, 1.0))


def compute_association_matrices(
    tracks: list[Track],
    detections: list[Detection],
    *,
    motion_weight: float = 1.0,
    appearance_weight: float = 0.0,
    temporal_weight: float = 0.0,
    shape_weight: float = 0.2,
    spatial_weight: float = 0.0,
    direction_weight: float = 0.0,
    motion_gate: float = 80.0,
    kf_gate: float = 9.0,
    appearance_gate: float = 1.0,
    temporal_gate: float = 1.0,
    shape_gate: float = 1.0,
    spatial_gate: float = 1.0,
    large_cost: float = 1e6,
    recent_embedding_window: int | None = None,
    hard_conflict_identity_blend: float = 0.18,
    hard_conflict_min_risk: float = 0.45,
    hard_conflict_min_density: float = 0.18,
    association_head: AssociationHead | None = None,
    use_learned_head: bool = True,
    handcrafted_blend_weight: float = 0.55,
    learned_blend_weight: float = 0.45,
    low_match_penalty: float = 0.35,
    high_switch_penalty: float = 0.25,
    risk_adaptive_weights: bool = True,
    switch_risk_weight: float = 0.50,
    match_score_gate: float = 0.12,
    embedding_dim: int = 128,
    use_temporal_signal: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build cost, match-score, and switch-risk matrices for one frame of association.

    为单帧关联构建代价矩阵、匹配分数矩阵和切换风险矩阵。
    """
    if len(tracks) == 0 or len(detections) == 0:
        shape = (len(tracks), len(detections))
        return (
            np.zeros(shape, dtype=np.float32),
            np.zeros(shape, dtype=np.float32),
            np.zeros(shape, dtype=np.float32),
        )

    cost = np.full((len(tracks), len(detections)), large_cost, dtype=np.float32)
    score_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    switch_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)

    for i, track in enumerate(tracks):
        for j, det in enumerate(detections):
            motion_cost = _motion_distance(track, det, motion_gate=motion_gate)
            kf_cost = _kalman_distance(track, det, kf_gate=kf_gate)
            if motion_cost > 1.0 or kf_cost > 1.0:
                continue

            appearance_long, appearance_short = _appearance_distance(
                track,
                det,
                recent_embedding_window=recent_embedding_window,
            )
            identity_distance, has_identity_signal = _identity_distance(
                track,
                det,
                recent_embedding_window=recent_embedding_window,
            )
            shape_cost = _shape_distance(track, det)
            spatial_cost = _spatial_distance(track, det)
            temporal_distance = temporal_detection_distance(track, det, embedding_dim=embedding_dim)
            temporal_ready = (
                use_temporal_signal
                and track.reid_state.temporal_token is not None
                and det.embedding is not None
                and track.hits >= 8
                and track.reid_state.memory_reliability >= 0.35
                and not det.is_rescued
            )
            temporal_consistent = (
                temporal_distance <= min(appearance_long, appearance_short) + 0.12
                or temporal_distance <= 0.22
            )
            temporal_active = temporal_ready and temporal_consistent
            effective_appearance_gate = appearance_gate + (0.20 if det.is_rescued else 0.0)
            effective_temporal_gate = temporal_gate + (0.10 if det.is_rescued else 0.0)
            effective_shape_gate = shape_gate + (0.10 if det.is_rescued else 0.0)
            effective_spatial_gate = spatial_gate + (0.10 if det.is_rescued else 0.0)
            if (
                min(appearance_long, appearance_short) > effective_appearance_gate
                or (
                    temporal_active
                    and temporal_distance > effective_temporal_gate
                )
                or shape_cost > effective_shape_gate
                or spatial_cost > effective_spatial_gate
            ):
                continue

            direction_cost = _direction_distance(track, det)
            spatial_distance = spatial_detection_distance(track, det, embedding_dim=embedding_dim)
            base_appearance_cost = min(appearance_long, appearance_short)
            identity_blend = 0.0
            if has_identity_signal and hard_conflict_identity_blend > 0.0:
                identity_blend = hard_conflict_identity_blend * _identity_conflict_intensity(
                    det,
                    min_risk=hard_conflict_min_risk,
                    min_density=hard_conflict_min_density,
                )
            appearance_identity_cost = (
                (1.0 - identity_blend) * base_appearance_cost
                + identity_blend * identity_distance
            )

            effective_motion_weight = motion_weight
            effective_appearance_weight = appearance_weight
            effective_temporal_weight = temporal_weight if temporal_active else 0.0
            effective_shape_weight = shape_weight
            effective_spatial_weight = spatial_weight
            effective_direction_weight = direction_weight
            if risk_adaptive_weights:
                # In crowded or merged situations, trust appearance/spatial evidence slightly more
                # and pure motion slightly less so we do not over-commit to a bad geometric guess.
                risk = max(
                    float(np.clip(det.switch_risk_hint, 0.0, 1.0)),
                    0.7 if det.is_crowded else 0.0,
                    0.9 if det.is_merged_risk else 0.0,
                )
                effective_motion_weight = motion_weight * (1.0 - 0.40 * risk)
                effective_appearance_weight = appearance_weight * (1.0 + 1.00 * risk)
                effective_temporal_weight = effective_temporal_weight * (1.0 + 0.55 * risk)
                effective_spatial_weight = spatial_weight * (1.0 + 0.80 * risk)
                effective_shape_weight = shape_weight * (1.0 + 0.25 * risk)
                effective_direction_weight = direction_weight * (1.0 - 0.20 * risk)

            match_score = float(np.clip(1.0 - (0.65 * motion_cost + 0.35 * kf_cost), 0.0, 1.0))
            switch_risk = float(np.clip(0.5 * min(appearance_long, appearance_short) + 0.3 * temporal_distance, 0.0, 1.0))
            if use_learned_head and association_head is not None:
                match_score, switch_risk, _ = association_head.score_pair(
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

            if det.reid_quality < 0.50:
                effective_appearance_weight *= 0.30
                effective_temporal_weight *= 0.70
            if not track.can_update_reid(det.frame_idx):
                effective_appearance_weight *= 0.50
                effective_temporal_weight *= 0.80

            handcrafted_total = (
                effective_motion_weight * (0.6 * motion_cost + 0.4 * kf_cost)
                + effective_appearance_weight * appearance_identity_cost
                + effective_temporal_weight * temporal_distance
                + effective_shape_weight * shape_cost
                + effective_spatial_weight * min(spatial_cost, spatial_distance)
                + effective_direction_weight * direction_cost
            )
            # Blend the explainable handcrafted score with the learned match/switch estimate when enabled.
            learned_total = (1.0 - match_score) + switch_risk_weight * switch_risk
            blend_total = handcrafted_blend_weight * handcrafted_total + learned_blend_weight * learned_total
            total = blend_total if use_learned_head else handcrafted_total

            if match_score < match_score_gate:
                total += low_match_penalty
            if switch_risk > 0.75:
                total += high_switch_penalty

            cost[i, j] = float(total)
            score_matrix[i, j] = float(match_score)
            switch_matrix[i, j] = float(switch_risk)

    return cost, score_matrix, switch_matrix


def compute_cost_matrix(
    tracks: list[Track],
    detections: list[Detection],
    *,
    motion_weight: float = 1.0,
    appearance_weight: float = 0.0,
    temporal_weight: float = 0.0,
    shape_weight: float = 0.2,
    spatial_weight: float = 0.0,
    direction_weight: float = 0.0,
    motion_gate: float = 80.0,
    kf_gate: float = 9.0,
    appearance_gate: float = 1.0,
    temporal_gate: float = 1.0,
    shape_gate: float = 1.0,
    spatial_gate: float = 1.0,
    large_cost: float = 1e6,
    recent_embedding_window: int | None = None,
    hard_conflict_identity_blend: float = 0.18,
    hard_conflict_min_risk: float = 0.45,
    hard_conflict_min_density: float = 0.18,
    handcrafted_blend_weight: float = 0.55,
    learned_blend_weight: float = 0.45,
    low_match_penalty: float = 0.35,
    high_switch_penalty: float = 0.25,
    risk_adaptive_weights: bool = True,
    use_temporal_signal: bool = False,
) -> np.ndarray:
    """Compatibility wrapper for callers that only need the final cost matrix.

    为只需要最终代价矩阵的调用方提供兼容封装。
    """
    cost, _, _ = compute_association_matrices(
        tracks,
        detections,
        motion_weight=motion_weight,
        appearance_weight=appearance_weight,
        temporal_weight=temporal_weight,
        shape_weight=shape_weight,
        spatial_weight=spatial_weight,
        direction_weight=direction_weight,
        motion_gate=motion_gate,
        kf_gate=kf_gate,
        appearance_gate=appearance_gate,
        temporal_gate=temporal_gate,
        shape_gate=shape_gate,
        spatial_gate=spatial_gate,
        large_cost=large_cost,
        recent_embedding_window=recent_embedding_window,
        hard_conflict_identity_blend=hard_conflict_identity_blend,
        hard_conflict_min_risk=hard_conflict_min_risk,
        hard_conflict_min_density=hard_conflict_min_density,
        association_head=None,
        use_learned_head=False,
        handcrafted_blend_weight=handcrafted_blend_weight,
        learned_blend_weight=learned_blend_weight,
        low_match_penalty=low_match_penalty,
        high_switch_penalty=high_switch_penalty,
        risk_adaptive_weights=risk_adaptive_weights,
        use_temporal_signal=use_temporal_signal,
    )
    return cost
