from __future__ import annotations

import math
from typing import Iterable, Sequence

from core.structures import TrackObservation


def _to_points(history: Sequence[TrackObservation] | Sequence[tuple[int, float, float]]) -> list[tuple[int, float, float]]:
    points: list[tuple[int, float, float]] = []
    for item in history:
        if isinstance(item, TrackObservation):
            points.append((item.frame_idx, item.center[0], item.center[1]))
        else:
            points.append((int(item[0]), float(item[1]), float(item[2])))
    return points


def velocity_from_history(history: Sequence[TrackObservation] | Sequence[tuple[int, float, float]]) -> tuple[float, float]:
    points = _to_points(history)
    if len(points) < 2:
        return 0.0, 0.0
    f0, x0, y0 = points[-2]
    f1, x1, y1 = points[-1]
    dt = max(f1 - f0, 1)
    return (x1 - x0) / dt, (y1 - y0) / dt


def acceleration_from_history(history: Sequence[TrackObservation] | Sequence[tuple[int, float, float]]) -> tuple[float, float]:
    points = _to_points(history)
    if len(points) < 3:
        return 0.0, 0.0
    v0 = velocity_from_history(points[:-1])
    v1 = velocity_from_history(points)
    return v1[0] - v0[0], v1[1] - v0[1]


def speed_from_history(history: Sequence[TrackObservation] | Sequence[tuple[int, float, float]]) -> float:
    vx, vy = velocity_from_history(history)
    return float(math.hypot(vx, vy))


def direction_from_history(history: Sequence[TrackObservation] | Sequence[tuple[int, float, float]]) -> float:
    vx, vy = velocity_from_history(history)
    return float(math.atan2(vy, vx)) if (vx != 0 or vy != 0) else 0.0


def direction_change_rate(history: Sequence[TrackObservation] | Sequence[tuple[int, float, float]]) -> float:
    points = _to_points(history)
    if len(points) < 3:
        return 0.0
    d0 = direction_from_history(points[:-1])
    d1 = direction_from_history(points)
    diff = math.atan2(math.sin(d1 - d0), math.cos(d1 - d0))
    return float(abs(diff))


def mean_step_distance(history: Sequence[TrackObservation] | Sequence[tuple[int, float, float]]) -> float:
    points = _to_points(history)
    if len(points) < 2:
        return 0.0
    distances = []
    for (_, x0, y0), (_, x1, y1) in zip(points[:-1], points[1:]):
        distances.append(math.hypot(x1 - x0, y1 - y0))
    return float(sum(distances) / max(len(distances), 1))
