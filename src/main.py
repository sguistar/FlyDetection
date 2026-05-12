from __future__ import annotations

import copy
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from association import (
    AssociationHead,
    TrajectoryTemporal,
    apply_track_support_bias,
    apply_slot_stickiness,
    bridge_long_gaps_spatiotemporal,
    cascade_match,
    compute_association_matrices,
    global_reassign_ids,
    interpolate_short_gaps,
    recover_track_supported_matches,
    suppress_slot_swaps,
)
from config import Config, get_config
from core.states import TrackState
from detector import YOLODetector
from events import InteractionDetector, LineCrossingDetector
from evaluation.metrics import (
    compute_basic_tracking_metrics,
    evaluate_tracks,
    evaluate_tracks_with_audit,
    summarize_detector_miss_by_id,
    summarize_detector_miss_segments,
    summarize_fn_breakdown,
    summarize_hard_case_buckets,
    summarize_detection_recall,
    summarize_temporal_window_metrics,
    tracks_to_frame_points,
)
from identity import IdentityEncoder
from io_utils import (
    create_video_writer,
    get_video_meta,
    load_detection_cache,
    log_kv,
    open_video,
    save_detection_cache,
    setup_logger,
    write_detections_csv,
    write_events_csv,
    write_metrics_csv,
    write_recall_audit_csv,
    write_table_csv,
    write_tracks_csv,
    write_track_stats_csv,
)
from preprocessing import QualityFilter
from render import Renderer
from tracklet import SplitMergeResolver, TrackBuilder
from tracklet.features import extract_track_statistics


def _build_output_paths(cfg: Config) -> dict[str, Path]:
    """Resolve the standard output files used by one pipeline run.

    解析单次流水线运行要使用的标准输出文件路径。
    """
    return {
        "video": cfg.paths.videos / "result_20s.mp4",
        "detections_csv": cfg.paths.csv / "detections.csv",
        "tracks_csv": cfg.paths.csv / "tracks.csv",
        "track_stats_csv": cfg.paths.csv / "track_stats.csv",
        "events_csv": cfg.paths.csv / "events.csv",
        "metrics_csv": cfg.paths.csv / "metrics.csv",
        "recall_audit_csv": cfg.paths.csv / "recall_audit.csv",
        "fn_breakdown_csv": cfg.paths.csv / "fn_breakdown.csv",
        "detector_miss_segments_csv": cfg.paths.csv / "detector_miss_segments.csv",
        "detector_miss_by_id_csv": cfg.paths.csv / "detector_miss_by_id.csv",
        "stage_metrics_csv": cfg.paths.csv / "stage_metrics.csv",
        "temporal_window_metrics_csv": cfg.paths.csv / "temporal_window_metrics.csv",
        "hard_case_summary_csv": cfg.paths.csv / "hard_case_summary.csv",
        "log": cfg.paths.logs / "run.log",
    }


def _detach_detection_crops(detections: list) -> list:
    for det in detections:
        det.crop = None
    return detections


def _annotate_detection_context(detections: list, frame_shape: tuple[int, int, int] | None = None) -> None:
    """Attach crowding, border, and local-density cues that downstream SC/IM/association can reuse.

    为检测结果附加拥挤、边界和局部密度信息，供后续 SC、IM 和关联模块复用。
    """
    if not detections:
        return
    width = float(frame_shape[1]) if frame_shape is not None and len(frame_shape) > 1 else 1.0
    height = float(frame_shape[0]) if frame_shape is not None and len(frame_shape) > 0 else 1.0

    centers = np.array([det.center for det in detections], dtype=np.float32)
    if len(detections) > 1:
        dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
        dists += np.eye(len(detections), dtype=np.float32) * 1e6
    else:
        dists = np.full((1, 1), 1e6, dtype=np.float32)

    for idx, det in enumerate(detections):
        nearest = float(np.min(dists[idx])) if len(detections) > 1 else float(max(width, height))
        sorted_dists = np.sort(dists[idx]) if len(detections) > 1 else np.array([float(max(width, height))], dtype=np.float32)
        second_nearest = float(sorted_dists[1]) if sorted_dists.shape[0] > 1 else nearest
        crowd_count = int(np.sum(dists[idx] < 40.0)) if len(detections) > 1 else 0
        x_norm = float(det.center[0] / max(width, 1.0))
        y_norm = float(det.center[1] / max(height, 1.0))
        area_norm = float(det.area / max(width * height, 1.0))
        x1, y1, x2, y2 = det.bbox
        border_distance = min(x1, y1, max(width - x2, 0.0), max(height - y2, 0.0))
        border_risk = float(np.clip(1.0 - border_distance / 24.0, 0.0, 1.0))
        local_density = float(np.clip(np.sum(np.maximum(0.0, 48.0 - dists[idx])) / (48.0 * 4.0), 0.0, 1.0)) if len(detections) > 1 else 0.0
        det.is_crowded = crowd_count >= 1
        det.is_merged_risk = crowd_count >= 2 or nearest < 18.0
        det.switch_risk_hint = float(np.clip((40.0 - nearest) / 40.0, 0.0, 1.0))
        det.context_feature = np.array(
            [
                x_norm,
                y_norm,
                float(np.clip(nearest / max(width, height, 1.0), 0.0, 1.0)),
                float(np.clip(second_nearest / max(width, height, 1.0), 0.0, 1.0)),
                float(np.clip(crowd_count / 5.0, 0.0, 1.0)),
                local_density,
                area_norm,
                border_risk,
            ],
            dtype=np.float32,
        )


def _detections_to_points(detections: list) -> list[dict]:
    return [
        {
            "x": float(det.center[0]),
            "y": float(det.center[1]),
            "conf": float(det.conf),
            "detector_source": str(getattr(det, "detector_source", "main")),
            "is_rescued": bool(getattr(det, "is_rescued", False)),
        }
        for det in detections
    ]


def _track_recent_size(track) -> tuple[float, float]:
    """Estimate a track's recent footprint so rescue ROIs stay tied to its observed scale.

    估计轨迹近期目标尺寸，使救援 ROI 与已观测到的尺度保持一致。
    """
    observations = sorted(track.trajectory, key=lambda obs: obs.frame_idx)[-5:]
    if not observations:
        x1, y1, x2, y2 = track.bbox
        return max(float(x2 - x1), 1.0), max(float(y2 - y1), 1.0)
    widths = [max(float(obs.bbox[2] - obs.bbox[0]), 1.0) for obs in observations]
    heights = [max(float(obs.bbox[3] - obs.bbox[1]), 1.0) for obs in observations]
    return float(np.mean(widths)), float(np.mean(heights))


def _mark_track_supported_detections(cfg: Config, builder: TrackBuilder, detections: list) -> None:
    """Mark raw detections that sit close to a confirmed/lost track prediction so they are not hard-dropped too early.

    标记靠近已确认或丢失轨迹预测位置的原始检测，避免它们过早被硬过滤。
    """
    if not cfg.preprocess.keep_track_supported_low_quality or not detections:
        return
    candidate_tracks = [
        track
        for track in builder.tracks
        if track.state in {TrackState.CONFIRMED, TrackState.LOST}
        and track.identity_slot is not None
    ]
    if not candidate_tracks:
        return

    for det in detections:
        best_track = None
        best_distance = None
        for track in candidate_tracks:
            predicted = track.predicted_center if track.predicted_center != (0.0, 0.0) else track.center
            width, height = _track_recent_size(track)
            support_radius = max(12.0, 0.75 * cfg.preprocess.track_support_radius_scale * max(width, height))
            support_radius = min(float(cfg.preprocess.track_support_max_distance), float(support_radius + 4.0 * min(track.missed, 4)))
            distance = float(np.hypot(det.center[0] - predicted[0], det.center[1] - predicted[1]))
            if distance > support_radius:
                continue
            if best_distance is None or distance < best_distance:
                best_track = track
                best_distance = distance
        if best_track is None:
            continue
        det.is_track_supported = True
        det.support_track_id = int(best_track.track_id)
        if det.conf < cfg.detection.conf_thres or det.is_border:
            if det.reid_quality_cap is None:
                det.reid_quality_cap = 0.45
            else:
                det.reid_quality_cap = min(float(det.reid_quality_cap), 0.45)


def _available_slot_ids(cfg: Config, builder: TrackBuilder) -> list[int]:
    """List currently free identity slots so rescue logic knows what it is trying to recover.

    列出当前空闲的身份槽位，让救援逻辑明确需要恢复哪些目标。
    """
    if not cfg.track.use_identity_slots:
        return []
    if getattr(builder, "available_track_ids", None) is not None:
        return sorted(int(slot_id) for slot_id in builder.available_track_ids)
    used = {
        int(track.identity_slot)
        for track in builder.tracks
        if track.identity_slot is not None and track.state != TrackState.REMOVED
    }
    return [slot_id for slot_id in range(cfg.track.identity_slots) if slot_id not in used]


def _build_rescue_rois(cfg: Config, builder: TrackBuilder, detections: list, frame_shape: tuple[int, int, int]) -> list[tuple[int, tuple[float, float, float, float]]]:
    """Create slot-conditioned rescue ROIs around tracks that should exist but are currently unmatched.

    围绕应存在但当前未匹配的轨迹，按身份槽位生成救援检测 ROI。
    """
    slot_cap = cfg.track.identity_slots if cfg.track.use_identity_slots else cfg.track.num_flies
    if not cfg.detection.rescue_enabled:
        return []
    full_frame = len(detections) >= slot_cap
    if full_frame and not cfg.detection.rescue_when_full:
        return []

    candidate_entries: list[tuple[int, tuple[float, float], float, float, tuple]] = []
    occupied_slots: set[int] = set()
    for track in builder.tracks:
        if track.identity_slot is None or track.state == TrackState.REMOVED:
            continue
        occupied_slots.add(int(track.identity_slot))
        if track.predicted_center == (0.0, 0.0):
            continue
        if full_frame and (
            track.state not in {TrackState.CONFIRMED, TrackState.LOST}
            or track.hits < cfg.detection.rescue_min_track_hits
        ):
            continue
        width, height = _track_recent_size(track)
        coverage_radius = max(
            float(cfg.detection.rescue_coverage_radius_min),
            float(cfg.detection.rescue_coverage_radius_scale * max(width, height)),
        )
        if any(np.hypot(det.center[0] - track.predicted_center[0], det.center[1] - track.predicted_center[1]) <= coverage_radius for det in detections):
            continue
        cx, cy = track.predicted_center
        half_w = 0.5 * width * cfg.detection.rescue_roi_scale
        half_h = 0.5 * height * cfg.detection.rescue_roi_scale
        candidate_entries.append(
            (
                int(track.identity_slot),
                (float(cx), float(cy)),
                float(width),
                float(height),
                (
                    track.state == TrackState.LOST,
                    track.hits,
                    track.missed,
                    -track.track_id,
                ),
            )
        )

    if getattr(builder, "enable_latent_slot_reconnect", False):
        for latent in builder.latent_slot_candidates():
            if int(latent.slot_id) in occupied_slots:
                continue
            cx, cy = latent.predicted_center
            if (float(cx), float(cy)) == (0.0, 0.0):
                continue
            width, height = latent.bbox_size
            coverage_radius = max(
                float(cfg.detection.rescue_coverage_radius_min),
                float(cfg.detection.rescue_coverage_radius_scale * max(width, height)),
            )
            if any(
                np.hypot(det.center[0] - cx, det.center[1] - cy) <= coverage_radius
                for det in detections
            ):
                continue
            candidate_entries.append(
                (
                    int(latent.slot_id),
                    (float(cx), float(cy)),
                    float(width),
                    float(height),
                    (
                        True,
                        max(0.0, float(latent.memory_reliability)),
                        -float(latent.frames_since_seen),
                        1,
                    ),
                )
            )

    candidate_entries.sort(key=lambda item: item[4], reverse=True)
    shortage = (
        min(len(candidate_entries), max(int(cfg.detection.rescue_max_rois_when_full), 0))
        if full_frame
        else max(slot_cap - len(detections), 0)
    )
    return [
        (
            int(slot_id),
            (
                float(center[0] - 0.5 * width * cfg.detection.rescue_roi_scale),
                float(center[1] - 0.5 * height * cfg.detection.rescue_roi_scale),
                float(center[0] + 0.5 * width * cfg.detection.rescue_roi_scale),
                float(center[1] + 0.5 * height * cfg.detection.rescue_roi_scale),
            ),
        )
        for slot_id, center, width, height, _ in candidate_entries[:shortage]
    ]


def _run_rescue_pass(
    cfg: Config,
    frame_idx: int,
    frame,
    *,
    builder: TrackBuilder,
    detector,
    detections: list,
) -> list:
    """Run ROI-level rescue detection for missing slots using track motion priors.

    利用轨迹运动先验，对缺失槽位执行 ROI 级别的救援检测。
    """
    if detector is None or not hasattr(detector, "detect_roi"):
        return []
    rescue_rois = _build_rescue_rois(cfg, builder, detections, frame.shape)
    rescued: list = []
    for slot_id, roi_bbox in rescue_rois:
        roi_cx = 0.5 * (roi_bbox[0] + roi_bbox[2])
        roi_cy = 0.5 * (roi_bbox[1] + roi_bbox[3])
        roi_max_dim = max(float(roi_bbox[2] - roi_bbox[0]), float(roi_bbox[3] - roi_bbox[1]))
        center_gate = max(
            float(cfg.detection.rescue_center_gate_min),
            float(cfg.detection.rescue_center_gate_scale * roi_max_dim),
        )
        rescue_candidates = detector.detect_roi(
            frame_idx,
            frame,
            roi_bbox,
            conf_thres=cfg.detection.rescue_conf_thres,
            max_det=3,
        )
        ranked_candidates = sorted(
            rescue_candidates,
            key=lambda det: (
                float(np.hypot(det.center[0] - roi_cx, det.center[1] - roi_cy)),
                -float(det.conf),
            ),
        )
        kept_for_slot = 0
        for det in ranked_candidates:
            center_distance = float(np.hypot(det.center[0] - roi_cx, det.center[1] - roi_cy))
            if center_distance > center_gate:
                continue
            if kept_for_slot >= max(int(cfg.detection.rescue_max_per_slot), 1):
                break
            det.is_rescued = True
            det.detector_source = "rescue"
            det.embedding_source = "rescue"
            det.rescue_slot_id = slot_id
            det.reid_quality_cap = 0.55
            rescued.append(det)
            kept_for_slot += 1
    return rescued


def _run_low_conf_full_frame_pass(
    cfg: Config,
    frame_idx: int,
    frame,
    *,
    detector,
) -> list:
    """Run a low-threshold full-frame rescue pass when the primary detections underfill the slot count.

    当主检测数量不足槽位数量时，执行低阈值全帧救援检测。
    """
    if detector is None:
        return []
    try:
        detections = detector.detect_frame(
            frame_idx,
            frame,
            conf_thres=cfg.detection.rescue_conf_thres,
            max_det=max(cfg.detection.max_det, (cfg.track.identity_slots if cfg.track.use_identity_slots else cfg.track.num_flies) * 8),
        )
    except TypeError:
        return []

    for det in detections:
        det.is_rescued = True
        det.detector_source = "rescue_full"
        det.embedding_source = "rescue"
        det.reid_quality_cap = 0.45
    return detections


def _bootstrap_tile_bboxes(frame_shape: tuple[int, int, int], *, grid_size: int, overlap: float) -> list[tuple[float, float, float, float]]:
    """Split the frame into overlapping tiles for optional cold-start bootstrap detection.

    将画面切成带重叠的小块，用于可选的冷启动引导检测。
    """
    height, width = frame_shape[:2]
    grid_size = max(int(grid_size), 1)
    overlap = float(np.clip(overlap, 0.0, 0.45))
    tile_width = width / grid_size
    tile_height = height / grid_size
    expand_x = tile_width * overlap
    expand_y = tile_height * overlap

    bboxes: list[tuple[float, float, float, float]] = []
    for gy in range(grid_size):
        for gx in range(grid_size):
            x1 = max(gx * tile_width - expand_x, 0.0)
            y1 = max(gy * tile_height - expand_y, 0.0)
            x2 = min((gx + 1) * tile_width + expand_x, float(width))
            y2 = min((gy + 1) * tile_height + expand_y, float(height))
            bboxes.append((float(x1), float(y1), float(x2), float(y2)))
    return bboxes


def _run_bootstrap_tile_pass(
    cfg: Config,
    frame_idx: int,
    frame,
    *,
    builder: TrackBuilder,
    detector,
    detections: list,
) -> list:
    """Probe tiled ROIs during early frames to cold-start slots the main pass missed entirely.

    在早期帧探测分块 ROI，为主检测完全漏掉的槽位进行冷启动。
    """
    slot_cap = cfg.track.identity_slots if cfg.track.use_identity_slots else cfg.track.num_flies
    available_slots = _available_slot_ids(cfg, builder)
    if (
        detector is None
        or not hasattr(detector, "detect_roi")
        or not cfg.detection.bootstrap_tile_enabled
        or frame_idx >= cfg.detection.bootstrap_frames
        or len(detections) >= slot_cap
        or not available_slots
    ):
        return []

    rescued: list = []
    tile_bboxes = _bootstrap_tile_bboxes(
        frame.shape,
        grid_size=cfg.detection.bootstrap_grid_size,
        overlap=cfg.detection.bootstrap_tile_overlap,
    )
    for tile_bbox in tile_bboxes:
        for det in detector.detect_roi(
            frame_idx,
            frame,
            tile_bbox,
            conf_thres=cfg.detection.rescue_conf_thres,
            max_det=slot_cap,
        ):
            det.is_rescued = True
            det.detector_source = "bootstrap_tile"
            det.embedding_source = "rescue"
            det.reid_quality_cap = 0.50
            rescued.append(det)
    return rescued


def _enhance_rescue_frame(frame):
    """Apply a lightweight contrast enhancement used only by the enhanced rescue detection pass.

    应用轻量级对比度增强，仅供增强型救援检测流程使用。
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.merge((l_channel, a_channel, b_channel))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    return cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0.0)


def _run_enhanced_full_frame_pass(
    cfg: Config,
    frame_idx: int,
    frame,
    *,
    detector,
) -> list:
    """Run a final enhanced-image rescue pass for stubborn low-visibility misses.

    对难以发现的低可见度漏检执行最后一轮增强图像救援检测。
    """
    if detector is None or not getattr(cfg.detection, "enhanced_rescue_enabled", True):
        return []
    try:
        detections = detector.detect_frame(
            frame_idx,
            _enhance_rescue_frame(frame),
            conf_thres=cfg.detection.rescue_conf_thres,
            max_det=max(
                cfg.detection.max_det,
                (cfg.track.identity_slots if cfg.track.use_identity_slots else cfg.track.num_flies) * 8,
            ),
        )
    except TypeError:
        return []

    for det in detections:
        det.is_rescued = True
        det.detector_source = "rescue_enhanced"
        det.embedding_source = "rescue"
        det.reid_quality_cap = 0.40
    return detections


def _snapshot_builder_points(builder: TrackBuilder, frame_idx: int) -> tuple[list[dict], list[dict]]:
    """Capture online and lost-track point views for later recall auditing.

    记录在线轨迹和丢失轨迹的点位视图，供后续召回审计使用。
    """
    online_rows: list[dict] = []
    lost_rows: list[dict] = []
    for track in builder.tracks:
        latest = track.latest_observation()
        if latest is not None and latest.frame_idx == frame_idx:
            online_rows.append(
                {
                    "id": int(track.track_id),
                    "identity_slot": -1 if track.identity_slot is None else int(track.identity_slot),
                    "x": float(latest.center[0]),
                    "y": float(latest.center[1]),
                    "state": track.state.value,
                }
            )
        elif track.state == TrackState.LOST:
            lost_rows.append(
                {
                    "id": int(track.track_id),
                    "identity_slot": -1 if track.identity_slot is None else int(track.identity_slot),
                    "x": float(track.predicted_center[0]),
                    "y": float(track.predicted_center[1]),
                    "state": track.state.value,
                }
            )
    return online_rows, lost_rows


def _frame_motion_summary(builder: TrackBuilder, frame_idx: int) -> tuple[float, bool]:
    """Estimate whether the current frame contains unusually fast motion among active tracks.

    估计当前帧的活跃轨迹中是否存在异常快速运动。
    """
    max_speed = 0.0
    for track in builder.tracks:
        observations = sorted(track.trajectory, key=lambda item: item.frame_idx)
        if len(observations) < 2:
            continue
        latest = observations[-1]
        previous = observations[-2]
        if latest.frame_idx != frame_idx:
            continue
        gap = max(int(latest.frame_idx - previous.frame_idx), 1)
        speed = float(np.hypot(latest.center[0] - previous.center[0], latest.center[1] - previous.center[1]) / gap)
        max_speed = max(max_speed, speed)
    return max_speed, bool(max_speed >= 18.0)


def _build_frame_debug_summary(frame_idx: int, raw_detections: list, filtered_detections: list, dropped_detections: list, builder: TrackBuilder) -> dict:
    """Summarize per-frame risk factors so FN hard-case buckets can be aggregated after evaluation.

    汇总每帧风险因素，便于评估后聚合假阴性困难案例类别。
    """
    del filtered_detections
    max_speed, high_speed_motion = _frame_motion_summary(builder, frame_idx)
    return {
        "frame": int(frame_idx),
        "num_raw": int(len(raw_detections)),
        "num_dropped": int(len(dropped_detections)),
        "has_border": bool(any(getattr(det, "is_border", False) for det in raw_detections)),
        "has_blur": bool(any("blur_low" in getattr(det, "quality_flags", []) for det in raw_detections + dropped_detections)),
        "has_rescue": bool(any(bool(getattr(det, "is_rescued", False)) for det in raw_detections)),
        "has_track_supported": bool(any(bool(getattr(det, "is_track_supported", False)) for det in raw_detections)),
        "has_close_interaction": bool(any(bool(getattr(det, "is_crowded", False)) for det in raw_detections)),
        "has_merge_risk": bool(any(bool(getattr(det, "is_merged_risk", False)) for det in raw_detections)),
        "high_speed_motion": bool(high_speed_motion),
        "max_speed": float(max_speed),
    }


def _load_cache_payload(cfg: Config, logger) -> dict | None:
    """Load cached detections when cache mode is enabled, otherwise return None.

    在启用缓存模式时加载检测缓存，否则返回 None。
    """
    if not (cfg.cache.enabled and cfg.cache.use_detection_cache):
        return None
    cache_payload = load_detection_cache(cfg.paths.cache, cfg.runtime.video_path, cfg.config_hash)
    if cache_payload is not None:
        log_kv(logger, 20, "Loaded detection cache", num_frames=len(cache_payload))
    return cache_payload


def build_runtime_components(cfg: Config, logger, cache_payload=None) -> dict:
    """Construct the detector, encoders, tracker, and postprocess helpers used by the runtime.

    构建运行时所需的检测器、编码器、跟踪器和后处理辅助组件。
    """
    detector = None if cache_payload is not None else YOLODetector(cfg)
    # Temporary clean runtime baseline from patch checklist:
    # - cfg.detection.use_tracking_api = False
    # - cfg.reid.use_trajectory_temporal = False
    # - cfg.association.use_learned_head = False
    quality_filter = QualityFilter(
        min_conf=cfg.detection.conf_thres,
        min_area=cfg.preprocess.min_area,
        max_area=cfg.preprocess.max_area,
        min_aspect=cfg.preprocess.min_aspect,
        max_aspect=cfg.preprocess.max_aspect,
        min_blur_score=cfg.preprocess.min_blur_score,
        border_margin=cfg.preprocess.border_margin,
        duplicate_iou_thres=cfg.preprocess.duplicate_iou_thres,
        duplicate_center_thres=cfg.preprocess.duplicate_center_thres,
        keep_low_quality_border=cfg.preprocess.keep_low_quality_border,
        keep_track_supported_low_quality=cfg.preprocess.keep_track_supported_low_quality,
    )
    encoder = IdentityEncoder(
        embedding_dim=cfg.feature.embedding_dim,
        backend=cfg.feature.encoder_backend,
        crop_size=cfg.feature.crop_size,
        checkpoint_path=cfg.feature.encoder_checkpoint,
        cnn_width=cfg.feature.cnn_width,
        cnn_dropout=cfg.feature.cnn_dropout,
        identity_hidden_dim=cfg.feature.identity_hidden_dim,
        identity_dropout=cfg.feature.identity_dropout,
        spatial_hidden_dim=cfg.feature.spatial_hidden_dim,
        spatial_dropout=cfg.feature.spatial_dropout,
        use_identity_memory=cfg.feature.use_identity_memory,
        use_spacial_context=cfg.feature.use_spacial_context,
        device=cfg.device,
        fallback_to_handcrafted_when_untrained=cfg.feature.fallback_to_handcrafted_when_untrained,
    )
    trajectory_temporal = TrajectoryTemporal(
        embedding_dim=cfg.feature.embedding_dim,
        history_len=cfg.feature.history_len,
        hidden_dim=cfg.feature.temporal_hidden_dim,
        num_layers=cfg.feature.temporal_num_layers,
        num_heads=cfg.feature.temporal_num_heads,
        dropout=cfg.feature.temporal_dropout,
        checkpoint_path=cfg.feature.encoder_checkpoint,
        device=cfg.device,
    )
    association_head = AssociationHead(
        embedding_dim=cfg.feature.embedding_dim,
        hidden_dim=cfg.association.association_hidden_dim,
        checkpoint_path=cfg.feature.encoder_checkpoint,
        device=cfg.device,
    )
    log_kv(logger, 20, "Identity encoder ready", backend=cfg.feature.encoder_backend, status=encoder.status_message)
    log_kv(logger, 20, "Identity memory ready", status=encoder.identity_memory_status)
    log_kv(logger, 20, "Spacial context ready", status=encoder.spacial_context_status)
    log_kv(logger, 20, "Trajectory temporal ready", status=trajectory_temporal.status_message)
    log_kv(logger, 20, "Association head ready", status=association_head.status_message)
    builder = TrackBuilder(
        confirm_hits=cfg.track.confirm_hits,
        max_missed=cfg.track.max_missed,
        remove_tentative_after=cfg.track.remove_tentative_after,
        max_tracks=cfg.track.identity_slots if cfg.track.use_identity_slots else cfg.track.num_flies,
        reid_update_quality_thres=cfg.track.reid_update_quality_thres,
        suspicious_appearance_thres=cfg.track.suspicious_appearance_thres,
        suspicious_hits=cfg.track.suspicious_hits,
        reid_freeze_frames=cfg.track.reid_freeze_frames,
        use_identity_slots=cfg.track.use_identity_slots,
        recall_mode=cfg.track.recall_mode,
        recovery_confirm_hits=cfg.track.recovery_confirm_hits,
        short_term_window=cfg.feature.short_term_window,
        long_term_momentum=cfg.feature.long_term_momentum,
        quarantine_min_quality=cfg.feature.quarantine_min_quality,
        trajectory_temporal=trajectory_temporal if cfg.reid.use_trajectory_temporal else None,
        enable_latent_slot_reconnect=cfg.track.enable_latent_slot_reconnect,
        latent_slot_max_age=cfg.track.latent_slot_max_age,
        latent_motion_gate=cfg.track.latent_motion_gate,
        latent_shape_ratio_tol=cfg.track.latent_shape_ratio_tol,
        latent_reconnect_min_reliability=cfg.track.latent_reconnect_min_reliability,
        enable_weak_match_motion_blend=cfg.track.enable_weak_match_motion_blend,
        weak_match_min_hits=cfg.track.weak_match_min_hits,
        weak_match_score_thres=cfg.track.weak_match_score_thres,
        weak_match_quality_thres=cfg.track.weak_match_quality_thres,
        weak_match_switch_risk_thres=cfg.track.weak_match_switch_risk_thres,
        weak_match_position_alpha=cfg.track.weak_match_position_alpha,
    )
    builder.association_head = association_head
    resolver = SplitMergeResolver(
        duplicate_iou_threshold=cfg.events.duplicate_iou_threshold,
        merged_iou_threshold=cfg.events.merged_iou_threshold,
        center_threshold=cfg.events.split_merge_center_threshold,
    )
    return {
        "detector": detector,
        "quality_filter": quality_filter,
        "encoder": encoder,
        "trajectory_temporal": trajectory_temporal,
        "association_head": association_head,
        "builder": builder,
        "resolver": resolver,
    }


def _resolve_detections(frame_idx: int, frame, *, cfg: Config, builder: TrackBuilder, cache_payload, detector, quality_filter, encoder) -> dict:
    """Resolve one frame's detections, including filtering, rescue passes, encoding, and debug views.

    处理单帧检测结果，包括过滤、救援检测、特征编码和调试视图生成。
    """
    if cache_payload is not None:
        detections = cache_payload.get(frame_idx, [])
        _annotate_detection_context(detections, frame.shape)
        return {
            "raw_detections": detections,
            "filtered_detections": detections,
            "dropped_detections": [],
            "accepted_detections": detections,
        }
    raw_detections = detector.detect_frame(frame_idx, frame)
    _mark_track_supported_detections(cfg, builder, raw_detections)
    detections, dropped_detections = quality_filter.filter_with_debug(raw_detections)

    rescued_detections = _run_rescue_pass(
        cfg,
        frame_idx,
        frame,
        builder=builder,
        detector=detector,
        detections=detections,
    )
    if rescued_detections:
        rescue_kept, rescue_dropped = quality_filter.filter_with_debug(
            rescued_detections,
            rescue_mode=True,
            anchors=detections,
        )
        raw_detections.extend(rescued_detections)
        dropped_detections.extend(rescue_dropped)
        detections.extend(rescue_kept)

    slot_cap = cfg.track.identity_slots if cfg.track.use_identity_slots else cfg.track.num_flies
    if len(detections) < slot_cap:
        fallback_rescue = _run_low_conf_full_frame_pass(
            cfg,
            frame_idx,
            frame,
            detector=detector,
        )
        if fallback_rescue:
            fallback_kept, fallback_dropped = quality_filter.filter_with_debug(
                fallback_rescue,
                rescue_mode=True,
                anchors=detections,
            )
            raw_detections.extend(fallback_rescue)
            dropped_detections.extend(fallback_dropped)
            detections.extend(fallback_kept)

    if len(detections) < slot_cap:
        bootstrap_rescue = _run_bootstrap_tile_pass(
            cfg,
            frame_idx,
            frame,
            builder=builder,
            detector=detector,
            detections=detections,
        )
        if bootstrap_rescue:
            bootstrap_kept, bootstrap_dropped = quality_filter.filter_with_debug(
                bootstrap_rescue,
                rescue_mode=True,
                anchors=detections,
            )
            for slot_id, det in zip(_available_slot_ids(cfg, builder), sorted(bootstrap_kept, key=lambda item: item.conf, reverse=True)):
                det.rescue_slot_id = slot_id
            raw_detections.extend(bootstrap_rescue)
            dropped_detections.extend(bootstrap_dropped)
            detections.extend(bootstrap_kept)

    if len(detections) < slot_cap:
        enhanced_rescue = _run_enhanced_full_frame_pass(
            cfg,
            frame_idx,
            frame,
            detector=detector,
        )
        if enhanced_rescue:
            enhanced_kept, enhanced_dropped = quality_filter.filter_with_debug(
                enhanced_rescue,
                rescue_mode=True,
                anchors=detections,
            )
            raw_detections.extend(enhanced_rescue)
            dropped_detections.extend(enhanced_dropped)
            detections.extend(enhanced_kept)

    for det in detections:
        encoder.encode_detection(det)
    _annotate_detection_context(detections, frame.shape)
    accepted = _detach_detection_crops(detections)
    raw_detections = _detach_detection_crops(raw_detections)
    dropped_detections = _detach_detection_crops(dropped_detections)
    return {
        "raw_detections": raw_detections,
        "filtered_detections": accepted,
        "dropped_detections": dropped_detections,
        "accepted_detections": accepted,
    }


def _associate_frame(cfg: Config, builder: TrackBuilder, detections: list):
    """Build pairwise association costs for one frame and run the cascade matcher.

    为单帧构建成对关联代价，并运行级联匹配器。
    """
    active_tracks = list(builder.tracks)
    slot_cap = cfg.track.identity_slots if cfg.track.use_identity_slots else cfg.track.num_flies
    recall_shortage = max(slot_cap - len(detections), 0) if cfg.track.recall_mode else 0
    recall_relax = 0.0
    reconnect_bonus = 0.0
    if recall_shortage > 0:
        recall_relax = 0.06 + 0.02 * min(recall_shortage, 2)
        reconnect_bonus = 0.10 + 0.02 * min(recall_shortage, 2)
    effective_match_score_gate = cfg.association.match_score_gate - recall_relax
    cost_matrix, score_matrix, switch_risk_matrix = compute_association_matrices(
        active_tracks,
        detections,
        motion_weight=cfg.feature.motion_weight,
        appearance_weight=cfg.feature.appearance_weight,
        temporal_weight=cfg.feature.temporal_weight,
        shape_weight=cfg.feature.shape_weight,
        spatial_weight=cfg.feature.spatial_weight,
        direction_weight=cfg.feature.direction_weight,
        motion_gate=cfg.association.motion_gate,
        kf_gate=cfg.association.kf_gate,
        appearance_gate=cfg.feature.appearance_gate,
        temporal_gate=cfg.feature.temporal_gate,
        shape_gate=cfg.feature.shape_gate,
        spatial_gate=cfg.feature.spatial_gate,
        large_cost=cfg.association.large_cost,
        recent_embedding_window=cfg.feature.recent_embedding_window,
        hard_conflict_identity_blend=cfg.association.hard_conflict_identity_blend,
        hard_conflict_min_risk=cfg.association.hard_conflict_min_risk,
        hard_conflict_min_density=cfg.association.hard_conflict_min_density,
        association_head=getattr(builder, "association_head", None),
        use_learned_head=cfg.association.use_learned_head,
        handcrafted_blend_weight=cfg.association.handcrafted_blend_weight,
        learned_blend_weight=cfg.association.learned_blend_weight,
        low_match_penalty=cfg.association.low_match_penalty,
        high_switch_penalty=cfg.association.high_switch_penalty,
        risk_adaptive_weights=cfg.association.risk_adaptive_weights,
        switch_risk_weight=cfg.association.switch_risk_weight,
        match_score_gate=max(effective_match_score_gate, 0.0),
        embedding_dim=cfg.feature.embedding_dim,
        use_temporal_signal=cfg.reid.use_trajectory_temporal,
    )
    if recall_shortage > 0 and cost_matrix.size > 0:
        for row_idx, track in enumerate(active_tracks):
            if track.identity_slot is not None and track.state in {TrackState.CONFIRMED, TrackState.LOST}:
                valid = cost_matrix[row_idx] < cfg.association.large_cost
                cost_matrix[row_idx, valid] = np.maximum(cost_matrix[row_idx, valid] - reconnect_bonus, 0.0)
    cost_matrix, score_matrix, switch_risk_matrix = apply_track_support_bias(
        active_tracks,
        detections,
        cost_matrix,
        score_matrix=score_matrix,
        switch_risk_matrix=switch_risk_matrix,
        motion_gate=cfg.association.motion_gate,
        large_cost=cfg.association.large_cost,
        support_distance_thres=cfg.preprocess.track_support_max_distance,
        support_reconnect_bonus=cfg.association.support_reconnect_bonus,
        lost_track_bonus=cfg.association.support_lost_track_bonus,
        fallback_cost_thres=cfg.association.support_fallback_cost_thres,
        score_floor=cfg.association.support_score_floor,
        switch_risk_cap=cfg.association.support_switch_risk_cap,
    )
    result = cascade_match(active_tracks, detections, cost_matrix, gate=None, large_cost=cfg.association.large_cost)
    result.score_matrix = score_matrix
    result.switch_risk_matrix = switch_risk_matrix
    result = recover_track_supported_matches(
        active_tracks,
        detections,
        result,
        large_cost=cfg.association.large_cost,
        cost_margin=cfg.association.support_reconnect_bonus,
    )
    if cfg.association.slot_swap_suppress_enabled:
        result = suppress_slot_swaps(
            active_tracks,
            detections,
            result,
            large_cost=cfg.association.large_cost,
            cost_margin=cfg.association.slot_swap_cost_margin,
            distance_thres=cfg.association.slot_swap_distance_thres,
            stable_hits=cfg.association.slot_swap_stable_hits,
            max_missed=cfg.association.slot_swap_max_missed,
        )
    result = builder.recover_latent_slot_matches(
        detections,
        result,
        large_cost=cfg.association.large_cost,
    )
    return result


def process_video_frames(cfg: Config, components: dict, logger, cache_payload=None) -> dict:
    """Run the online frame loop and collect all per-frame artifacts needed for export and evaluation.

    执行在线视频帧循环，并收集导出与评估所需的逐帧产物。
    """
    cap = open_video(cfg.runtime.video_path)
    meta = get_video_meta(cap)
    detections_by_frame: dict[int, list] = {}
    raw_detections_by_frame: dict[int, list[dict]] = {}
    filtered_detections_by_frame: dict[int, list[dict]] = {}
    dropped_detections_by_frame: dict[int, list[dict]] = {}
    online_tracks_by_frame: dict[int, list[dict]] = {}
    lost_tracks_by_frame: dict[int, list[dict]] = {}
    frame_debug_by_frame: dict[int, dict] = {}

    frame_idx = 0
    try:
        while True:
            if cfg.runtime.max_frames is not None and frame_idx >= cfg.runtime.max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break

            components["builder"].predict()
            detection_info = _resolve_detections(
                frame_idx,
                frame,
                cfg=cfg,
                builder=components["builder"],
                cache_payload=cache_payload,
                detector=components["detector"],
                quality_filter=components["quality_filter"],
                encoder=components["encoder"],
            )
            detections = detection_info["accepted_detections"]
            detections_by_frame[frame_idx] = detections
            raw_detections_by_frame[frame_idx] = _detections_to_points(detection_info["raw_detections"])
            filtered_detections_by_frame[frame_idx] = _detections_to_points(detection_info["filtered_detections"])
            dropped_detections_by_frame[frame_idx] = _detections_to_points(detection_info["dropped_detections"])
            assoc = _associate_frame(cfg, components["builder"], detections)
            components["builder"].update(frame_idx, detections, assoc)
            online_rows, lost_rows = _snapshot_builder_points(components["builder"], frame_idx)
            online_tracks_by_frame[frame_idx] = online_rows
            lost_tracks_by_frame[frame_idx] = lost_rows
            frame_debug_by_frame[frame_idx] = _build_frame_debug_summary(
                frame_idx,
                detection_info["raw_detections"],
                detection_info["filtered_detections"],
                detection_info["dropped_detections"],
                components["builder"],
            )

            if frame_idx % max(cfg.runtime.log_every_n_frames, 1) == 0:
                log_kv(
                    logger,
                    20,
                    "Processed frame",
                    frame_idx=frame_idx,
                    num_detections=len(detections),
                    active_tracks=len(components["builder"].tracks),
                )
            frame_idx += 1
    finally:
        cap.release()

    return {
        "meta": meta,
        "detections_by_frame": detections_by_frame,
        "raw_detections_by_frame": raw_detections_by_frame,
        "filtered_detections_by_frame": filtered_detections_by_frame,
        "dropped_detections_by_frame": dropped_detections_by_frame,
        "online_tracks_by_frame": online_tracks_by_frame,
        "lost_tracks_by_frame": lost_tracks_by_frame,
        "frame_debug_by_frame": frame_debug_by_frame,
        "detection_summary": summarize_detection_recall(
            filtered_detections_by_frame,
            expected_num_flies=cfg.track.identity_slots if cfg.track.use_identity_slots else cfg.track.num_flies,
        ),
    }


def _detect_events(cfg: Config, tracks: list, resolver: SplitMergeResolver) -> list[dict]:
    """Generate crossing, interaction, and split/merge events from the finalized tracks.

    根据最终轨迹生成越线、交互以及分裂/合并事件。
    """
    events = list(resolver.last_events)
    if cfg.events.enable_crossing:
        crossing_detector = LineCrossingDetector(x_line=cfg.events.x_line, y_line=cfg.events.y_line)
        events.extend(crossing_detector.detect_many(tracks))
    if cfg.events.enable_interaction:
        interaction_detector = InteractionDetector(
            distance_thres=cfg.events.interaction_distance,
            overlap_iou_thres=cfg.events.merged_iou_threshold,
        )
        events.extend(interaction_detector.detect_many(tracks))
    events.sort(key=lambda item: (int(item.get("frame_idx", -1)), str(item.get("type", ""))))
    return events


def _compute_metrics(cfg: Config, tracks: list, frame_results: dict) -> tuple[dict, list[dict]]:
    """Compute tracking metrics and the recall audit using the configured GT protocol.

    按照配置的 GT 协议计算跟踪指标和召回审计结果。
    """
    metrics = compute_basic_tracking_metrics(tracks)
    metrics.update(frame_results.get("detection_summary", {}))
    recall_audit: list[dict] = []
    gt_path = Path(cfg.evaluation.gt_csv_path)
    if cfg.evaluation.enabled and gt_path.exists():
        eval_metrics, recall_audit = evaluate_tracks_with_audit(
            tracks,
            str(gt_path),
            raw_detections_by_frame=frame_results.get("raw_detections_by_frame"),
            filtered_detections_by_frame=frame_results.get("filtered_detections_by_frame"),
            online_tracks_by_frame=frame_results.get("online_tracks_by_frame"),
            lost_tracks_by_frame=frame_results.get("lost_tracks_by_frame"),
            match_dist=cfg.evaluation.point_match_distance,
            compute_hota=cfg.evaluation.compute_hota,
            ignore_unlabeled_frames=cfg.evaluation.ignore_unlabeled_frames,
            gt_frame_stride=cfg.evaluation.gt_frame_stride,
            gt_frame_offset=cfg.evaluation.gt_frame_offset,
            prediction_id_source=cfg.evaluation.prediction_id_source,
        )
        metrics.update(eval_metrics)
    else:
        metrics["evaluation_skipped"] = 1
    return metrics, recall_audit


def _prune_low_conf_short_tracks(cfg: Config, tracks: list) -> list:
    """Drop very short, weak-confidence ghost tracks before final export and evaluation.

    在最终导出和评估前移除很短且置信度较低的幽灵轨迹。
    """
    if not cfg.track.prune_short_low_conf_tracks:
        return tracks
    pruned: list = []
    for track in tracks:
        trajectory_len = len(track.trajectory)
        if trajectory_len == 0:
            continue
        mean_conf = float(np.mean([obs.conf for obs in track.trajectory]))
        if (
            trajectory_len <= cfg.track.low_conf_track_max_length
            and mean_conf <= cfg.track.low_conf_track_mean_conf
        ):
            continue
        pruned.append(track)
    return pruned


def _track_source_summary(track) -> dict[str, float | int]:
    """Summarize how strongly one track depends on rescue detections instead of the main detector.

    汇总单条轨迹对救援检测而非主检测器的依赖程度。
    """
    stats = extract_track_statistics(track)
    feature_points = max(int(len(track.feature_history)), 1)
    main_count = int(stats.get("main_count", 0))
    rescue_count = int(stats.get("rescue_count", 0))
    rescue_ratio = float(stats.get("rescue_ratio", 0.0))
    mean_conf = float(np.mean([obs.conf for obs in track.trajectory])) if track.trajectory else 0.0
    return {
        "feature_points": feature_points,
        "main_count": main_count,
        "main_ratio": float(main_count / feature_points),
        "rescue_count": rescue_count,
        "rescue_ratio": rescue_ratio,
        "mean_conf": mean_conf,
    }


def _is_rescue_heavy_ghost_track(cfg: Config, track) -> bool:
    """Flag tracks that are almost entirely sustained by ultra-low-confidence rescue detections.

    标记几乎完全依赖超低置信度救援检测维持的幽灵轨迹。
    """
    if not cfg.track.prune_rescue_heavy_tracks:
        return False
    if not track.trajectory or not track.feature_history:
        return False

    summary = _track_source_summary(track)
    if int(summary["feature_points"]) < cfg.track.rescue_ghost_min_feature_points:
        return False
    if float(summary["rescue_ratio"]) < cfg.track.rescue_ghost_min_ratio:
        return False

    mean_conf = float(summary["mean_conf"])
    main_count = int(summary["main_count"])
    main_ratio = float(summary["main_ratio"])
    obvious_rescue_ghost = (
        mean_conf <= cfg.track.rescue_ghost_mean_conf
        and main_count <= cfg.track.rescue_ghost_max_main_count
    )
    extended_rescue_ghost = (
        mean_conf <= cfg.track.rescue_ghost_extreme_mean_conf
        and main_ratio <= cfg.track.rescue_ghost_max_main_ratio
    )
    return bool(obvious_rescue_ghost or extended_rescue_ghost)


def _prune_rescue_heavy_tracks(cfg: Config, tracks: list) -> list:
    """Drop rescue-heavy ghost tracks before final export so recall gains do not turn into easy false positives.

    在最终导出前移除救援占比过高的幽灵轨迹，避免召回收益变成明显假阳性。
    """
    if not cfg.track.prune_rescue_heavy_tracks:
        return tracks
    return [track for track in tracks if not _is_rescue_heavy_ghost_track(cfg, track)]


def _stage_avg_points_per_frame(tracks: list) -> float:
    frame_points = tracks_to_frame_points(tracks, prediction_id_source="track_id")
    if not frame_points:
        return 0.0
    return float(sum(len(items) for items in frame_points.values()) / max(len(frame_points), 1))


def _evaluate_stage_metrics(cfg: Config, frame_results: dict, stage_tracks: list[tuple[str, list]]) -> list[dict]:
    """Evaluate every postprocess stage separately so FN jumps can be localized to one step.

    分别评估每个后处理阶段，以便定位假阴性突增发生在哪一步。
    """
    rows: list[dict] = []
    previous_fn = None
    previous_fp = None
    gt_path = Path(cfg.evaluation.gt_csv_path)
    for order, (stage_name, tracks) in enumerate(stage_tracks, start=1):
        row = {
            "stage_order": int(order),
            "stage": stage_name,
            "num_tracks": int(len(tracks)),
            "avg_points_per_frame": _stage_avg_points_per_frame(tracks),
        }
        row.update(compute_basic_tracking_metrics(tracks))
        if cfg.evaluation.enabled and gt_path.exists():
            row.update(
                evaluate_tracks(
                    tracks,
                    str(gt_path),
                    match_dist=cfg.evaluation.point_match_distance,
                    compute_hota=cfg.evaluation.compute_hota,
                    ignore_unlabeled_frames=cfg.evaluation.ignore_unlabeled_frames,
                    gt_frame_stride=cfg.evaluation.gt_frame_stride,
                    gt_frame_offset=cfg.evaluation.gt_frame_offset,
                    prediction_id_source=cfg.evaluation.prediction_id_source,
                )
            )
        else:
            row["evaluation_skipped"] = 1
        if previous_fn is None:
            row["fn_delta_from_prev"] = 0
            row["fp_delta_from_prev"] = 0
            row["stability_killer"] = 0
        else:
            fn_delta = int(row.get("fn", 0)) - int(previous_fn)
            fp_delta = int(row.get("fp", 0)) - int(previous_fp)
            row["fn_delta_from_prev"] = fn_delta
            row["fp_delta_from_prev"] = fp_delta
            row["stability_killer"] = int(fn_delta > 0 and fp_delta >= 0)
        previous_fn = int(row.get("fn", 0))
        previous_fp = int(row.get("fp", 0))
        rows.append(row)
    return rows


def postprocess_tracks(cfg: Config, components: dict, frame_results: dict | None = None) -> dict:
    """Apply offline ID reassignment, split/merge cleanup, interpolation, events, and metrics.

    执行离线 ID 重分配、分裂/合并清理、插值、事件检测和指标计算。
    """
    builder: TrackBuilder = components["builder"]
    resolver: SplitMergeResolver = components["resolver"]
    enable_offline_id_merge = cfg.track.enable_global_reid and cfg.reid.enabled and cfg.reid.use_slot_reassign
    frame_results = frame_results or {}
    stage_tracks: list[tuple[str, list]] = []

    merge_same_track_id = enable_offline_id_merge and not cfg.track.recall_mode
    tracks = builder.export_tracks(
        min_length=1 if cfg.track.recall_mode else cfg.track.min_track_length,
        merge_same_track_id=merge_same_track_id,
        reassign_track_ids=not enable_offline_id_merge,
    )
    stage_tracks.append(("online_export", copy.deepcopy(tracks)))
    if enable_offline_id_merge:
        tracks = global_reassign_ids(
            tracks,
            max_link_gap=cfg.association.max_link_gap,
            merge_threshold=cfg.reid.merge_threshold,
            appearance_threshold=cfg.reid.appearance_threshold,
            shape_threshold=cfg.reid.shape_threshold,
            spatial_threshold=cfg.reid.spatial_threshold,
            motion_threshold=cfg.reid.motion_threshold,
            fragment_min_len=cfg.reid.fragment_min_len,
            fragment_max_internal_gap=cfg.reid.fragment_max_internal_gap,
            offline_window=cfg.reid.offline_window,
            max_identities=cfg.track.identity_slots if cfg.track.use_identity_slots else cfg.track.num_flies,
            merge_fragments=not cfg.track.recall_mode,
            force_assign_when_full=cfg.reid.offline_force_assign_when_full if not cfg.track.recall_mode else False,
        )
        stage_tracks.append(("global_reassign", copy.deepcopy(tracks)))
    tracks = resolver(tracks)
    stage_tracks.append(("split_merge_resolve", copy.deepcopy(tracks)))
    if cfg.reid.slot_stickiness_enabled:
        tracks = apply_slot_stickiness(
            tracks,
            max_fragment_len=cfg.reid.slot_stickiness_max_fragment_len,
            max_gap=cfg.reid.slot_stickiness_max_gap,
            max_speed=cfg.reid.slot_stickiness_max_speed,
            min_anchor_len=cfg.reid.slot_stickiness_min_anchor_len,
        )
        stage_tracks.append(("slot_stickiness", copy.deepcopy(tracks)))
    if cfg.track.enable_long_gap_bridge:
        tracks = bridge_long_gaps_spatiotemporal(
            tracks,
            min_gap=cfg.track.long_gap_bridge_min_gap,
            max_gap=cfg.track.long_gap_bridge_max_gap,
            velocity_window=cfg.track.long_gap_bridge_velocity_window,
            endpoint_tol_per_frame=cfg.track.long_gap_bridge_endpoint_tol_per_frame,
            max_step_per_frame=cfg.track.long_gap_bridge_max_step_per_frame,
            shape_ratio_tol=cfg.track.long_gap_bridge_shape_ratio_tol,
            min_conf_scale=cfg.track.long_gap_bridge_min_conf_scale,
        )
        stage_tracks.append(("long_gap_bridge", copy.deepcopy(tracks)))
    if cfg.track.enable_interpolation:
        tracks = interpolate_short_gaps(tracks, max_gap=cfg.association.max_interpolation_gap)
        stage_tracks.append(("interpolation", copy.deepcopy(tracks)))
    if cfg.track.prune_rescue_heavy_tracks:
        tracks = _prune_rescue_heavy_tracks(cfg, tracks)
        stage_tracks.append(("prune_rescue_ghosts", copy.deepcopy(tracks)))
    tracks = _prune_low_conf_short_tracks(cfg, tracks)
    stage_tracks.append(("prune_low_conf", copy.deepcopy(tracks)))

    events = _detect_events(cfg, tracks, resolver)
    metrics, recall_audit = _compute_metrics(cfg, tracks, frame_results)
    stage_tracks.append(("final_export", copy.deepcopy(tracks)))
    stage_metrics = _evaluate_stage_metrics(cfg, frame_results, stage_tracks)
    fn_breakdown = summarize_fn_breakdown(recall_audit)
    detector_miss_segments = summarize_detector_miss_segments(
        recall_audit,
        frame_gap=cfg.evaluation.gt_frame_stride or 1,
    )
    gt_path = Path(cfg.evaluation.gt_csv_path)
    if cfg.evaluation.enabled and gt_path.exists():
        detector_miss_by_id = summarize_detector_miss_by_id(
            recall_audit,
            frame_gap=cfg.evaluation.gt_frame_stride or 1,
            min_segment_points=cfg.evaluation.detector_miss_long_segment_points,
        )
        temporal_window_metrics = summarize_temporal_window_metrics(
            tracks,
            str(gt_path),
            fps=float(frame_results.get("meta", {}).get("fps", 0.0)),
            window_seconds=cfg.evaluation.temporal_window_sec,
            audit_rows=recall_audit,
            match_dist=cfg.evaluation.point_match_distance,
            compute_hota=cfg.evaluation.compute_hota,
            ignore_unlabeled_frames=cfg.evaluation.ignore_unlabeled_frames,
            gt_frame_stride=cfg.evaluation.gt_frame_stride,
            gt_frame_offset=cfg.evaluation.gt_frame_offset,
            prediction_id_source=cfg.evaluation.prediction_id_source,
        )
    else:
        detector_miss_by_id = []
        temporal_window_metrics = []
    hard_case_summary = summarize_hard_case_buckets(
        recall_audit,
        frame_results.get("frame_debug_by_frame"),
    )
    return {
        "tracks": tracks,
        "events": events,
        "metrics": metrics,
        "recall_audit": recall_audit,
        "stage_metrics": stage_metrics,
        "fn_breakdown": fn_breakdown,
        "detector_miss_segments": detector_miss_segments,
        "detector_miss_by_id": detector_miss_by_id,
        "temporal_window_metrics": temporal_window_metrics,
        "hard_case_summary": hard_case_summary,
    }


def _render_video(cfg: Config, output_video: Path, tracks, events, meta: dict) -> None:
    """Render the final annotated result video from exported tracks and events.

    根据导出的轨迹和事件渲染最终带标注的视频结果。
    """
    renderer = Renderer(
        trail_len=cfg.render.trail_len,
        draw_labels=cfg.render.draw_labels,
        draw_hud=cfg.render.draw_hud,
        bbox_thickness=cfg.render.bbox_thickness,
        max_event_lines=cfg.render.max_event_lines,
    )
    frame_index = renderer.build_frame_index(tracks)
    events_by_frame: dict[int, list[dict]] = defaultdict(list)
    for event in events:
        events_by_frame[int(event.get("frame_idx", -1))].append(event)

    cap = open_video(cfg.runtime.video_path)
    writer = create_video_writer(output_video, meta["fps"], meta["width"], meta["height"])
    frame_idx = 0
    try:
        while True:
            if cfg.runtime.max_frames is not None and frame_idx >= cfg.runtime.max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break
            canvas = renderer.draw(
                frame,
                frame_idx,
                frame_index.get(frame_idx, []),
                events=events_by_frame.get(frame_idx, []),
                meta={"total_tracks": len(tracks)},
            )
            writer.write(canvas)
            frame_idx += 1
    finally:
        writer.release()
        cap.release()


def export_outputs(
    cfg: Config,
    output_paths: dict[str, Path],
    *,
    detections_by_frame: dict[int, list],
    tracks: list,
    events: list[dict],
    metrics: dict,
    recall_audit: list[dict],
    fn_breakdown: list[dict],
    detector_miss_segments: list[dict],
    detector_miss_by_id: list[dict],
    stage_metrics: list[dict],
    temporal_window_metrics: list[dict],
    hard_case_summary: list[dict],
    meta: dict,
) -> None:
    """Write all requested CSV/video outputs for the completed run.

    为已完成的运行写出所有请求的 CSV 和视频结果。
    """
    if cfg.runtime.save_detection_csv:
        write_detections_csv(output_paths["detections_csv"], detections_by_frame)
    if cfg.runtime.save_track_csv:
        write_tracks_csv(output_paths["tracks_csv"], tracks)
        write_track_stats_csv(output_paths["track_stats_csv"], tracks)
    if cfg.runtime.save_event_csv:
        write_events_csv(output_paths["events_csv"], events)
    if cfg.runtime.save_metrics_csv:
        write_metrics_csv(output_paths["metrics_csv"], metrics)
        write_recall_audit_csv(output_paths["recall_audit_csv"], recall_audit)
        write_table_csv(output_paths["fn_breakdown_csv"], fn_breakdown)
        write_table_csv(output_paths["detector_miss_segments_csv"], detector_miss_segments)
        write_table_csv(output_paths["detector_miss_by_id_csv"], detector_miss_by_id)
        write_table_csv(output_paths["stage_metrics_csv"], stage_metrics)
        write_table_csv(output_paths["temporal_window_metrics_csv"], temporal_window_metrics)
        write_table_csv(output_paths["hard_case_summary_csv"], hard_case_summary)
    if cfg.runtime.save_video:
        _render_video(cfg, output_paths["video"], tracks, events, meta)


def run_pipeline(cfg: Config) -> dict:
    """Top-level orchestration for one MOT run from input video to exported results.

    顶层调度一次 MOT 运行，从输入视频处理到结果导出。
    """
    cfg.paths.mkdirs()
    output_paths = _build_output_paths(cfg)
    logger = setup_logger(output_paths["log"])
    try:
        log_kv(
            logger,
            20,
            "Start MOT pipeline",
            config_hash=cfg.config_hash,
            video_path=cfg.runtime.video_path,
            device=cfg.device,
        )

        cache_payload = _load_cache_payload(cfg, logger)
        components = build_runtime_components(cfg, logger, cache_payload=cache_payload)
        frame_results = process_video_frames(cfg, components, logger, cache_payload=cache_payload)

        if cache_payload is None and cfg.cache.enabled and cfg.cache.write_detection_cache:
            cache_path = save_detection_cache(
                cfg.paths.cache,
                cfg.runtime.video_path,
                cfg.config_hash,
                frame_results["detections_by_frame"],
            )
            log_kv(logger, 20, "Wrote detection cache", cache_path=str(cache_path))

        postprocess_results = postprocess_tracks(cfg, components, frame_results=frame_results)
        export_outputs(
            cfg,
            output_paths,
            detections_by_frame=frame_results["detections_by_frame"],
            tracks=postprocess_results["tracks"],
            events=postprocess_results["events"],
            metrics=postprocess_results["metrics"],
            recall_audit=postprocess_results["recall_audit"],
            fn_breakdown=postprocess_results["fn_breakdown"],
            detector_miss_segments=postprocess_results["detector_miss_segments"],
            detector_miss_by_id=postprocess_results["detector_miss_by_id"],
            stage_metrics=postprocess_results["stage_metrics"],
            temporal_window_metrics=postprocess_results["temporal_window_metrics"],
            hard_case_summary=postprocess_results["hard_case_summary"],
            meta=frame_results["meta"],
        )

        log_kv(
            logger,
            20,
            "Finished MOT pipeline",
            num_tracks=len(postprocess_results["tracks"]),
            num_events=len(postprocess_results["events"]),
            outputs={key: str(value) for key, value in output_paths.items()},
        )
        return {
            "tracks": postprocess_results["tracks"],
            "events": postprocess_results["events"],
            "metrics": postprocess_results["metrics"],
            "recall_audit": postprocess_results["recall_audit"],
            "fn_breakdown": postprocess_results["fn_breakdown"],
            "detector_miss_segments": postprocess_results["detector_miss_segments"],
            "detector_miss_by_id": postprocess_results["detector_miss_by_id"],
            "stage_metrics": postprocess_results["stage_metrics"],
            "temporal_window_metrics": postprocess_results["temporal_window_metrics"],
            "hard_case_summary": postprocess_results["hard_case_summary"],
            "output_paths": output_paths,
        }
    finally:
        for handler in list(logger.handlers):
            try:
                handler.flush()
                handler.close()
            finally:
                logger.removeHandler(handler)


def main() -> None:
    cfg = get_config()
    run_pipeline(cfg)

if __name__ == "__main__":
    main()
