from __future__ import annotations

from collections import Counter

import numpy as np

from core.structures import Track
from motion.kinematics import acceleration_from_history, direction_change_rate, mean_step_distance, velocity_from_history


def extract_track_statistics(track: Track) -> dict:
    if track.stats_cache:
        return dict(track.stats_cache)

    points = track.xy_history(include_interpolated=True)
    vx, vy = velocity_from_history(points)
    ax, ay = acceleration_from_history(points)
    xs = [point[1] for point in points] if points else [track.center[0]]
    ys = [point[2] for point in points] if points else [track.center[1]]
    areas = [entry["area"] for entry in track.feature_history if "area" in entry]
    aspects = [entry["aspect"] for entry in track.feature_history if "aspect" in entry]
    detector_sources = Counter(str(entry.get("detector_source", "main")) for entry in track.feature_history)
    rescue_count = int(sum(bool(entry.get("is_rescued", False)) for entry in track.feature_history))
    main_count = int(detector_sources.get("main", 0))
    total_feature_points = max(len(track.feature_history), 1)
    mean_reid_quality = float(
        np.mean([entry["reid_quality"] for entry in track.feature_history if "reid_quality" in entry])
    ) if track.feature_history else 0.0
    track_supported_count = int(
        sum(
            1
            for entry in track.feature_history
            if bool(entry.get("is_track_supported", False))
        )
    )
    duration = max(track.end_frame - track.start_frame + 1, 1)

    display_id = track.identity_slot if track.identity_slot is not None else track.track_id
    stats = {
        "display_id": display_id,
        "track_id": track.track_id,
        "identity_slot": -1 if track.identity_slot is None else track.identity_slot,
        "length": len(track.trajectory),
        "duration": duration,
        "missing_rate": float((duration - len(track.trajectory)) / duration),
        "mean_speed": float(np.hypot(vx, vy)),
        "mean_step": float(mean_step_distance(points)),
        "mean_acc": float(np.hypot(ax, ay)),
        "direction_change_rate": float(direction_change_rate(points)),
        "x_span": float(max(xs) - min(xs)),
        "y_span": float(max(ys) - min(ys)),
        "mean_area": float(np.mean(areas)) if areas else 0.0,
        "var_area": float(np.var(areas)) if areas else 0.0,
        "mean_aspect": float(np.mean(aspects)) if aspects else 0.0,
        "var_aspect": float(np.var(aspects)) if aspects else 0.0,
        "mean_reid_quality": mean_reid_quality,
        "main_count": main_count,
        "rescue_count": int(rescue_count),
        "rescue_ratio": float(rescue_count / total_feature_points),
        "track_supported_count": track_supported_count,
        "start_frame": track.start_frame,
        "end_frame": track.end_frame,
        "missed": track.missed,
        "interpolated_points": len(track.interpolated_frames),
    }
    track.stats_cache = dict(stats)
    return stats
