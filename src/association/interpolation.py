from __future__ import annotations

import copy
import math

from core.structures import Track, TrackObservation
from motion.kinematics import velocity_from_history


def _interp(value_a: float, value_b: float, alpha: float) -> float:
    return float(value_a + alpha * (value_b - value_a))


def _interp_bbox(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
    alpha: float,
) -> tuple[float, float, float, float]:
    return (
        _interp(box_a[0], box_b[0], alpha),
        _interp(box_a[1], box_b[1], alpha),
        _interp(box_a[2], box_b[2], alpha),
        _interp(box_a[3], box_b[3], alpha),
    )


def _bbox_size(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return max(float(box[2] - box[0]), 1.0), max(float(box[3] - box[1]), 1.0)


def _bbox_from_center(
    center: tuple[float, float],
    *,
    width: float,
    height: float,
) -> tuple[float, float, float, float]:
    half_w = 0.5 * max(float(width), 1.0)
    half_h = 0.5 * max(float(height), 1.0)
    return (
        float(center[0] - half_w),
        float(center[1] - half_h),
        float(center[0] + half_w),
        float(center[1] + half_h),
    )


def _local_velocity(
    observations: list[TrackObservation],
    *,
    anchor_idx: int,
    window: int,
    reverse: bool = False,
) -> tuple[float, float]:
    if reverse:
        segment = observations[anchor_idx:min(len(observations), anchor_idx + max(int(window), 2))]
        points = [(obs.frame_idx, obs.center[0], obs.center[1]) for obs in segment]
        vx, vy = velocity_from_history(points)
        return float(vx), float(vy)
    segment = observations[max(0, anchor_idx - max(int(window), 2) + 1):anchor_idx + 1]
    points = [(obs.frame_idx, obs.center[0], obs.center[1]) for obs in segment]
    vx, vy = velocity_from_history(points)
    return float(vx), float(vy)


def _hermite_center(
    left: TrackObservation,
    right: TrackObservation,
    *,
    left_velocity: tuple[float, float],
    right_velocity: tuple[float, float],
    frame_idx: int,
) -> tuple[float, float]:
    gap = max(int(right.frame_idx - left.frame_idx), 1)
    t = float(frame_idx - left.frame_idx) / float(gap)
    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2
    scale = float(gap)
    x = (
        h00 * float(left.center[0])
        + h10 * float(left_velocity[0]) * scale
        + h01 * float(right.center[0])
        + h11 * float(right_velocity[0]) * scale
    )
    y = (
        h00 * float(left.center[1])
        + h10 * float(left_velocity[1]) * scale
        + h01 * float(right.center[1])
        + h11 * float(right_velocity[1]) * scale
    )
    return float(x), float(y)


def _relative_delta(value_a: float, value_b: float) -> float:
    denom = max(abs(float(value_a)), abs(float(value_b)), 1.0)
    return float(abs(float(value_a) - float(value_b)) / denom)


def _should_bridge_long_gap(
    left: TrackObservation,
    right: TrackObservation,
    *,
    left_velocity: tuple[float, float],
    right_velocity: tuple[float, float],
    max_gap: int,
    min_gap: int,
    endpoint_tol_per_frame: float,
    max_step_per_frame: float,
    shape_ratio_tol: float,
) -> bool:
    gap = int(right.frame_idx - left.frame_idx)
    if gap <= max(int(min_gap), 1) or gap > max(int(max_gap), 1):
        return False

    direct_distance = math.hypot(right.center[0] - left.center[0], right.center[1] - left.center[1])
    direct_step = float(direct_distance / max(gap, 1))
    if direct_step > float(max_step_per_frame):
        return False

    forward_pred = (
        float(left.center[0] + left_velocity[0] * gap),
        float(left.center[1] + left_velocity[1] * gap),
    )
    backward_pred = (
        float(right.center[0] - right_velocity[0] * gap),
        float(right.center[1] - right_velocity[1] * gap),
    )
    forward_error = math.hypot(forward_pred[0] - right.center[0], forward_pred[1] - right.center[1])
    backward_error = math.hypot(backward_pred[0] - left.center[0], backward_pred[1] - left.center[1])
    mean_endpoint_error = float(0.5 * (forward_error + backward_error) / max(gap, 1))
    if mean_endpoint_error > float(endpoint_tol_per_frame):
        return False

    left_w, left_h = _bbox_size(left.bbox)
    right_w, right_h = _bbox_size(right.bbox)
    if _relative_delta(left_w, right_w) > float(shape_ratio_tol):
        return False
    if _relative_delta(left_h, right_h) > float(shape_ratio_tol):
        return False
    return True


def bridge_long_gaps_spatiotemporal(
    tracks: list[Track],
    *,
    min_gap: int = 5,
    max_gap: int = 2400,
    velocity_window: int = 6,
    endpoint_tol_per_frame: float = 2.5,
    max_step_per_frame: float = 14.0,
    shape_ratio_tol: float = 0.75,
    min_conf_scale: float = 0.12,
) -> list[Track]:
    """Bridge long detector-failure gaps with a smooth path between the last and next trusted observations.

    在前后可信观测之间生成平滑路径，桥接检测器长时间失效造成的轨迹缺口。
    """
    result: list[Track] = []
    for track in tracks:
        updated = copy.deepcopy(track)
        observations = sorted(updated.trajectory, key=lambda obs: obs.frame_idx)
        if len(observations) < 2:
            result.append(updated)
            continue

        new_observations: list[TrackObservation] = []
        interpolated_frames = set(updated.interpolated_frames)
        for idx, (left, right) in enumerate(zip(observations[:-1], observations[1:])):
            new_observations.append(left)
            gap = int(right.frame_idx - left.frame_idx)
            if gap <= 1:
                continue

            left_velocity = _local_velocity(
                observations,
                anchor_idx=idx,
                window=velocity_window,
                reverse=False,
            )
            right_velocity = _local_velocity(
                observations,
                anchor_idx=idx + 1,
                window=velocity_window,
                reverse=True,
            )
            if not _should_bridge_long_gap(
                left,
                right,
                left_velocity=left_velocity,
                right_velocity=right_velocity,
                max_gap=max_gap,
                min_gap=min_gap,
                endpoint_tol_per_frame=endpoint_tol_per_frame,
                max_step_per_frame=max_step_per_frame,
                shape_ratio_tol=shape_ratio_tol,
            ):
                continue

            left_w, left_h = _bbox_size(left.bbox)
            right_w, right_h = _bbox_size(right.bbox)
            conf_scale = max(
                float(min_conf_scale),
                min(1.0, 0.70 - 0.0015 * float(gap)),
            )
            bridged_conf = float(min(left.conf, right.conf) * conf_scale)
            for step in range(1, gap):
                frame_idx = int(left.frame_idx + step)
                center = _hermite_center(
                    left,
                    right,
                    left_velocity=left_velocity,
                    right_velocity=right_velocity,
                    frame_idx=frame_idx,
                )
                alpha = float(step / gap)
                width = _interp(left_w, right_w, alpha)
                height = _interp(left_h, right_h, alpha)
                new_observations.append(
                    TrackObservation(
                        frame_idx=frame_idx,
                        bbox=_bbox_from_center(center, width=width, height=height),
                        center=center,
                        conf=bridged_conf,
                        state=left.state,
                        interpolated=True,
                    )
                )
                interpolated_frames.add(frame_idx)

        new_observations.append(observations[-1])
        updated.trajectory = sorted(new_observations, key=lambda obs: obs.frame_idx)
        updated.interpolated_frames = interpolated_frames
        latest = updated.latest_observation()
        if latest is not None:
            updated.bbox = latest.bbox
            updated.center = latest.center
            updated.last_frame = latest.frame_idx
            updated.conf = latest.conf
        updated.stats_cache.clear()
        result.append(updated)
    return result


def interpolate_short_gaps(tracks: list[Track], max_gap: int = 5) -> list[Track]:
    result: list[Track] = []
    for track in tracks:
        updated = copy.deepcopy(track)
        observations = sorted(updated.trajectory, key=lambda obs: obs.frame_idx)
        if len(observations) < 2:
            result.append(updated)
            continue

        new_observations: list[TrackObservation] = []
        updated.interpolated_frames = {
            int(obs.frame_idx)
            for obs in observations
            if bool(getattr(obs, "interpolated", False))
        }
        for left, right in zip(observations[:-1], observations[1:]):
            new_observations.append(left)
            gap = right.frame_idx - left.frame_idx
            if 1 < gap <= max_gap:
                for step in range(1, gap):
                    alpha = step / gap
                    bbox = _interp_bbox(left.bbox, right.bbox, alpha)
                    center = (
                        _interp(left.center[0], right.center[0], alpha),
                        _interp(left.center[1], right.center[1], alpha),
                    )
                    frame_idx = left.frame_idx + step
                    new_observations.append(
                        TrackObservation(
                            frame_idx=frame_idx,
                            bbox=bbox,
                            center=center,
                            conf=min(left.conf, right.conf),
                            state=left.state,
                            interpolated=True,
                        )
                    )
                    updated.interpolated_frames.add(frame_idx)
        new_observations.append(observations[-1])
        updated.trajectory = sorted(new_observations, key=lambda obs: obs.frame_idx)
        latest = updated.latest_observation()
        if latest is not None:
            updated.bbox = latest.bbox
            updated.center = latest.center
            updated.last_frame = latest.frame_idx
        updated.stats_cache.clear()
        result.append(updated)
    return result
