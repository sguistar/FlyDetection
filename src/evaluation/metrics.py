from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from core.structures import Track
from io_utils.csv_io import read_points_csv
from scipy.optimize import linear_sum_assignment



def _resolve_prediction_id(item, *, prediction_id_source: str = "track_id") -> int | None:
    if hasattr(item, "track_id"):
        track_id = getattr(item, "track_id", None)
        identity_slot = getattr(item, "identity_slot", None)
    else:
        track_id = item.get("track_id", item.get("id"))
        identity_slot = item.get("identity_slot")

    if prediction_id_source == "identity_slot" and identity_slot is not None:
        return int(identity_slot)
    if prediction_id_source == "auto" and identity_slot is not None:
        return int(identity_slot)
    if track_id is None and identity_slot is not None:
        return int(identity_slot)
    return None if track_id is None else int(track_id)


def tracks_to_frame_points(
    tracks: list[Track],
    *,
    prediction_id_source: str = "track_id",
) -> dict[int, list[dict]]:
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for track in tracks:
        pred_id = _resolve_prediction_id(track, prediction_id_source=prediction_id_source)
        for obs in track.trajectory:
            by_frame[obs.frame_idx].append(
                {
                    "id": pred_id,
                    "track_id": int(track.track_id),
                    "identity_slot": None if track.identity_slot is None else int(track.identity_slot),
                    "x": float(obs.center[0]),
                    "y": float(obs.center[1]),
                }
            )
    return by_frame


def _nearest_distance(
    x: float,
    y: float,
    items: list[dict],
    *,
    x_key: str = "x",
    y_key: str = "y",
) -> tuple[float | None, dict | None]:
    if not items:
        return None, None
    best_item = None
    best_dist = None
    for item in items:
        dist = math.hypot(float(item[x_key]) - x, float(item[y_key]) - y)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_item = item
    return best_dist, best_item


def _frame_assignment(
    gts: list[dict],
    preds: list[dict],
    *,
    match_dist: float,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    large_cost = 1e9
    if not gts or not preds:
        return [], set(), set()
    cost = np.full((len(gts), len(preds)), large_cost, dtype=np.float32)
    for i, gt in enumerate(gts):
        for j, pred in enumerate(preds):
            distance = math.hypot(float(pred["x"]) - float(gt["x"]), float(pred["y"]) - float(gt["y"]))
            if distance <= match_dist:
                cost[i, j] = float(distance)

    row_ind, col_ind = linear_sum_assignment(cost)
    matches: list[tuple[int, int, float]] = []
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    for row, col in zip(row_ind.tolist(), col_ind.tolist()):
        distance = float(cost[row, col])
        if distance >= large_cost:
            continue
        matches.append((row, col, distance))
        matched_gt.add(row)
        matched_pred.add(col)
    return matches, matched_gt, matched_pred


def summarize_detection_recall(
    detections_by_frame: dict[int, list[dict] | list],
    *,
    expected_num_flies: int = 6,
) -> dict:
    frame_count = len(detections_by_frame)
    det_counts = [len(items) for items in detections_by_frame.values()]
    if not det_counts:
        return {
            "avg_detections_per_frame": 0.0,
            "frames_below_num_flies": 0,
            "frames_with_lt_4_detections": 0,
        }
    return {
        "avg_detections_per_frame": float(sum(det_counts) / max(frame_count, 1)),
        "frames_below_num_flies": int(sum(count < expected_num_flies for count in det_counts)),
        "frames_with_lt_4_detections": int(sum(count < 4 for count in det_counts)),
    }


def build_recall_audit(
    gt_by_frame: dict[int, list[dict]],
    pred_by_frame: dict[int, list[dict]],
    *,
    raw_detections_by_frame: dict[int, list[dict]] | None = None,
    filtered_detections_by_frame: dict[int, list[dict]] | None = None,
    online_tracks_by_frame: dict[int, list[dict]] | None = None,
    lost_tracks_by_frame: dict[int, list[dict]] | None = None,
    match_dist: float = 30.0,
    gt_frame_stride: int | None = None,
    gt_frame_offset: int = 0,
    prediction_id_source: str = "track_id",
) -> list[dict]:
    gt_by_frame = _filter_gt_frames(gt_by_frame, gt_frame_stride, gt_frame_offset)
    all_frames = sorted(gt_by_frame.keys())
    audit_rows: list[dict] = []

    for frame_idx in all_frames:
        gts = gt_by_frame.get(frame_idx, [])
        preds = pred_by_frame.get(frame_idx, [])
        matches, _, _ = _frame_assignment(gts, preds, match_dist=match_dist)
        matched_pred_by_gt = {row: col for row, col, _ in matches}

        raw_rows = raw_detections_by_frame.get(frame_idx, []) if raw_detections_by_frame is not None else []
        filtered_rows = filtered_detections_by_frame.get(frame_idx, []) if filtered_detections_by_frame is not None else []
        online_rows = online_tracks_by_frame.get(frame_idx, []) if online_tracks_by_frame is not None else []
        lost_rows = lost_tracks_by_frame.get(frame_idx, []) if lost_tracks_by_frame is not None else []

        for gt_idx, gt in enumerate(gts):
            gt_x = float(gt["x"])
            gt_y = float(gt["y"])
            pred_col = matched_pred_by_gt.get(gt_idx)

            nearest_raw_dist, _ = _nearest_distance(gt_x, gt_y, raw_rows)
            nearest_filtered_dist, _ = _nearest_distance(gt_x, gt_y, filtered_rows)
            nearest_online_dist, nearest_online = _nearest_distance(gt_x, gt_y, online_rows)
            nearest_lost_dist, _ = _nearest_distance(gt_x, gt_y, lost_rows)
            track_candidates = [dist for dist in (nearest_online_dist, nearest_lost_dist) if dist is not None]
            nearest_track_dist = min(track_candidates) if track_candidates else None

            matched = pred_col is not None
            pred_track_id = int(preds[pred_col]["id"]) if pred_col is not None and preds[pred_col].get("id") is not None else None

            if matched:
                miss_stage = ""
            elif nearest_raw_dist is None or nearest_raw_dist > match_dist:
                miss_stage = "detector_miss"
            elif nearest_filtered_dist is None or nearest_filtered_dist > match_dist:
                miss_stage = "lost_before_match" if nearest_lost_dist is not None and nearest_lost_dist <= match_dist else "quality_filter_drop"
            elif nearest_online_dist is not None and nearest_online_dist <= match_dist:
                online_track_id = (
                    _resolve_prediction_id(nearest_online, prediction_id_source=prediction_id_source)
                    if nearest_online is not None
                    else None
                )
                miss_stage = "association_miss" if online_track_id == pred_track_id else "postprocess_loss"
            elif nearest_lost_dist is not None and nearest_lost_dist <= match_dist:
                miss_stage = "lost_before_match"
            else:
                miss_stage = "association_miss"

            audit_rows.append(
                {
                    "frame": int(frame_idx),
                    "id": int(gt["id"]),
                    "matched": int(matched),
                    "pred_track_id": pred_track_id,
                    "nearest_det_dist": None if nearest_filtered_dist is None else float(nearest_filtered_dist),
                    "nearest_track_dist": None if nearest_track_dist is None else float(nearest_track_dist),
                    "miss_stage": miss_stage,
                }
            )

    return audit_rows


def compute_point_tracking_metrics(
    gt_by_frame: dict[int, list[dict]],
    pred_by_frame: dict[int, list[dict]],
    *,
    match_dist: float = 30.0,
    compute_hota: bool = True,
    ignore_unlabeled_frames: bool = True,
    gt_frame_stride: int | None = None,
    gt_frame_offset: int = 0,
) -> dict:
    if linear_sum_assignment is None:
        raise ImportError("scipy is required for metric computation.")

    fp = 0
    fn = 0
    idsw = 0
    matched_points = 0
    total_pred_points = 0
    localization_errors: list[float] = []
    gt_totals: Counter[int] = Counter()
    gt_matched: Counter[int] = Counter()
    identity_pairs: Counter[tuple[int, int]] = Counter()
    last_pred_for_gt: dict[int, int] = {}

    gt_by_frame = _filter_gt_frames(gt_by_frame, gt_frame_stride, gt_frame_offset)

    if ignore_unlabeled_frames:
        all_frames = sorted(gt_by_frame.keys())
    else:
        all_frames = sorted(set(gt_by_frame.keys()) | set(pred_by_frame.keys()))
    for frame_idx in all_frames:
        gts = gt_by_frame.get(frame_idx, [])
        preds = pred_by_frame.get(frame_idx, [])
        total_pred_points += len(preds)
        for gt in gts:
            gt_totals[int(gt["id"])] += 1

        if not gts and not preds:
            continue
        if not gts:
            fp += len(preds)
            continue
        if not preds:
            fn += len(gts)
            continue

        matches, matched_gt, matched_pred = _frame_assignment(gts, preds, match_dist=match_dist)
        for row, col, distance in matches:
            matched_gt.add(row)
            matched_pred.add(col)
            matched_points += 1
            localization_errors.append(distance)

            gt_id = int(gts[row]["id"])
            pred_id = int(preds[col]["id"])
            gt_matched[gt_id] += 1
            identity_pairs[(gt_id, pred_id)] += 1

            previous_pred = last_pred_for_gt.get(gt_id)
            if previous_pred is not None and previous_pred != pred_id:
                idsw += 1
            last_pred_for_gt[gt_id] = pred_id

        frame_fn = len(gts) - len(matched_gt)
        frame_fp = len(preds) - len(matched_pred)
        fn += frame_fn
        fp += frame_fp

    total_gt_points = sum(gt_totals.values())
    idtp = _compute_optimal_identity_matches(identity_pairs)
    idfp = max(total_pred_points - idtp, 0)
    idfn = max(total_gt_points - idtp, 0)
    idp = float(idtp / max(idtp + idfp, 1))
    idr = float(idtp / max(idtp + idfn, 1))
    idf1 = float(2.0 * idp * idr / max(idp + idr, 1e-8))
    mota_like = float(1.0 - (fn + fp + idsw) / max(total_gt_points, 1))
    mean_loc_error = float(sum(localization_errors) / max(len(localization_errors), 1))

    coverages = [float(gt_matched[gt_id] / max(count, 1)) for gt_id, count in gt_totals.items()]
    mostly_tracked = sum(coverage >= 0.8 for coverage in coverages)
    partially_tracked = sum(0.2 <= coverage < 0.8 for coverage in coverages)
    mostly_lost = sum(coverage < 0.2 for coverage in coverages)
    track_coverage = float(sum(coverages) / max(len(coverages), 1))

    det_a = float(matched_points / max(matched_points + fp + fn, 1))
    assoc_a = float(idtp / max(idtp + idfp + idfn, 1))
    point_hota = float(math.sqrt(max(det_a, 0.0) * max(assoc_a, 0.0))) if compute_hota else 0.0

    return {
        "gt_total_points": int(total_gt_points),
        "evaluated_frames": int(len(all_frames)),
        "ignored_unlabeled_frames": int(bool(ignore_unlabeled_frames)),
        "gt_frame_stride": int(gt_frame_stride or 0),
        "gt_frame_offset": int(gt_frame_offset),
        "matched_points": int(matched_points),
        "fp": int(fp),
        "fn": int(fn),
        "idtp": int(idtp),
        "idfp": int(idfp),
        "idfn": int(idfn),
        "idsw": int(idsw),
        "idp": idp,
        "idr": idr,
        "idf1": idf1,
        "mota_like": mota_like,
        "mostly_tracked": int(mostly_tracked),
        "partially_tracked": int(partially_tracked),
        "mostly_lost": int(mostly_lost),
        "track_coverage": track_coverage,
        "mean_localization_error": mean_loc_error,
        "det_a": det_a,
        "assoc_a": assoc_a,
        "point_hota": point_hota,
    }


def _filter_gt_frames(
    gt_by_frame: dict[int, list[dict]],
    frame_stride: int | None,
    frame_offset: int = 0,
) -> dict[int, list[dict]]:
    if frame_stride is None or frame_stride <= 1:
        return gt_by_frame
    return {
        frame_idx: points
        for frame_idx, points in gt_by_frame.items()
        if (int(frame_idx) - int(frame_offset)) % int(frame_stride) == 0
    }


def _compute_optimal_identity_matches(identity_pairs: Counter[tuple[int, int]]) -> int:
    if not identity_pairs:
        return 0
    gt_ids = sorted({pair[0] for pair in identity_pairs})
    pred_ids = sorted({pair[1] for pair in identity_pairs})
    cost = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float32)
    for row, gt_id in enumerate(gt_ids):
        for col, pred_id in enumerate(pred_ids):
            cost[row, col] = -float(identity_pairs.get((gt_id, pred_id), 0))
    rows, cols = linear_sum_assignment(cost)
    return int(sum(identity_pairs.get((gt_ids[row], pred_ids[col]), 0) for row, col in zip(rows, cols)))


def evaluate_tracks(
    tracks: list[Track],
    gt_csv_path: str,
    *,
    match_dist: float = 30.0,
    compute_hota: bool = True,
    ignore_unlabeled_frames: bool = True,
    gt_frame_stride: int | None = None,
    gt_frame_offset: int = 0,
    prediction_id_source: str = "track_id",
) -> dict:
    gt_by_frame = read_points_csv(gt_csv_path)
    pred_by_frame = tracks_to_frame_points(tracks, prediction_id_source=prediction_id_source)
    return compute_point_tracking_metrics(
        gt_by_frame,
        pred_by_frame,
        match_dist=match_dist,
        compute_hota=compute_hota,
        ignore_unlabeled_frames=ignore_unlabeled_frames,
        gt_frame_stride=gt_frame_stride,
        gt_frame_offset=gt_frame_offset,
    )


def evaluate_tracks_with_audit(
    tracks: list[Track],
    gt_csv_path: str,
    *,
    raw_detections_by_frame: dict[int, list[dict]] | None = None,
    filtered_detections_by_frame: dict[int, list[dict]] | None = None,
    online_tracks_by_frame: dict[int, list[dict]] | None = None,
    lost_tracks_by_frame: dict[int, list[dict]] | None = None,
    match_dist: float = 30.0,
    compute_hota: bool = True,
    ignore_unlabeled_frames: bool = True,
    gt_frame_stride: int | None = None,
    gt_frame_offset: int = 0,
    prediction_id_source: str = "track_id",
) -> tuple[dict, list[dict]]:
    gt_by_frame = read_points_csv(gt_csv_path)
    pred_by_frame = tracks_to_frame_points(tracks, prediction_id_source=prediction_id_source)
    metrics = compute_point_tracking_metrics(
        gt_by_frame,
        pred_by_frame,
        match_dist=match_dist,
        compute_hota=compute_hota,
        ignore_unlabeled_frames=ignore_unlabeled_frames,
        gt_frame_stride=gt_frame_stride,
        gt_frame_offset=gt_frame_offset,
    )
    audit_rows = build_recall_audit(
        gt_by_frame,
        pred_by_frame,
        raw_detections_by_frame=raw_detections_by_frame,
        filtered_detections_by_frame=filtered_detections_by_frame,
        online_tracks_by_frame=online_tracks_by_frame,
        lost_tracks_by_frame=lost_tracks_by_frame,
        match_dist=match_dist,
        gt_frame_stride=gt_frame_stride,
        gt_frame_offset=gt_frame_offset,
        prediction_id_source=prediction_id_source,
    )
    return metrics, audit_rows


def compute_basic_tracking_metrics(tracks: list[Track]) -> dict:
    lengths = [len(track.trajectory) for track in tracks]
    return {
        "num_tracks": len(tracks),
        "mean_track_length": float(sum(lengths) / max(len(lengths), 1)),
        "max_track_length": max(lengths) if lengths else 0,
    }


def summarize_fn_breakdown(audit_rows: list[dict]) -> list[dict]:
    stage_order = [
        "detector_miss",
        "quality_filter_drop",
        "association_miss",
        "lost_before_match",
        "postprocess_loss",
    ]
    descriptions = {
        "detector_miss": "detector 根本没看到",
        "quality_filter_drop": "检测到了但被过滤",
        "association_miss": "检测存在但未正确关联",
        "lost_before_match": "轨迹提前死亡",
        "postprocess_loss": "后处理或导出阶段被抹掉",
    }
    misses = [row for row in audit_rows if not bool(int(row.get("matched", 0)))]
    total = max(len(misses), 1)
    counts = Counter(str(row.get("miss_stage", "")) for row in misses)
    return [
        {
            "miss_stage": stage,
            "count": int(counts.get(stage, 0)),
            "ratio": float(counts.get(stage, 0) / total),
            "description": descriptions[stage],
        }
        for stage in stage_order
    ]


def summarize_detector_miss_segments(
    audit_rows: list[dict],
    *,
    frame_gap: int = 1,
) -> list[dict]:
    """Group contiguous detector misses into segments so long blind spots can be inspected by identity.

    将连续检测漏检分组成片段，便于按身份检查长时间盲区。
    """
    detector_misses = sorted(
        (
            row
            for row in audit_rows
            if str(row.get("miss_stage", "")) == "detector_miss"
        ),
        key=lambda row: (int(row.get("id", -1)), int(row.get("frame", -1))),
    )
    if not detector_misses:
        return []

    frame_gap = max(int(frame_gap), 1)
    rows: list[dict] = []
    current_segment: list[dict] = []

    def flush_segment() -> None:
        if not current_segment:
            return
        det_dists = [
            float(row["nearest_det_dist"])
            for row in current_segment
            if row.get("nearest_det_dist") not in (None, "")
        ]
        track_dists = [
            float(row["nearest_track_dist"])
            for row in current_segment
            if row.get("nearest_track_dist") not in (None, "")
        ]
        coupled = [
            row
            for row in current_segment
            if row.get("nearest_det_dist") not in (None, "")
            and row.get("nearest_track_dist") not in (None, "")
        ]
        rows.append(
            {
                "id": int(current_segment[0]["id"]),
                "start_frame": int(current_segment[0]["frame"]),
                "end_frame": int(current_segment[-1]["frame"]),
                "num_points": int(len(current_segment)),
                "frame_span": int(current_segment[-1]["frame"] - current_segment[0]["frame"]),
                "mean_nearest_det_dist": float(sum(det_dists) / max(len(det_dists), 1)) if det_dists else None,
                "max_nearest_det_dist": float(max(det_dists)) if det_dists else None,
                "mean_nearest_track_dist": float(sum(track_dists) / max(len(track_dists), 1)) if track_dists else None,
                "mean_track_minus_det_dist": float(
                    sum(float(row["nearest_track_dist"]) - float(row["nearest_det_dist"]) for row in coupled) / max(len(coupled), 1)
                ) if coupled else None,
                "track_follows_det_ratio": float(
                    sum(
                        abs(float(row["nearest_track_dist"]) - float(row["nearest_det_dist"])) <= 1e-3
                        for row in coupled
                    ) / max(len(coupled), 1)
                ) if coupled else 0.0,
            }
        )

    previous_id = None
    previous_frame = None
    for row in detector_misses:
        row_id = int(row["id"])
        frame = int(row["frame"])
        contiguous = (
            previous_id is not None
            and row_id == previous_id
            and previous_frame is not None
            and frame - previous_frame == frame_gap
        )
        if not contiguous:
            flush_segment()
            current_segment = []
        current_segment.append(row)
        previous_id = row_id
        previous_frame = frame
    flush_segment()
    return rows


def summarize_detector_miss_by_id(
    audit_rows: list[dict],
    *,
    frame_gap: int = 1,
    min_segment_points: int = 1,
) -> list[dict]:
    """Aggregate detector-miss segments per GT id so long blind spots stand out immediately.

    按 GT ID 聚合检测漏检片段，让长时间盲区更容易被发现。
    """
    segments = summarize_detector_miss_segments(audit_rows, frame_gap=frame_gap)
    grouped: dict[int, list[dict]] = defaultdict(list)
    for segment in segments:
        grouped[int(segment["id"])].append(segment)

    rows: list[dict] = []
    for gt_id, id_segments in grouped.items():
        long_segments = [
            segment
            for segment in id_segments
            if int(segment.get("num_points", 0)) >= int(min_segment_points)
        ]
        longest_segment = max(
            id_segments,
            key=lambda segment: (int(segment.get("num_points", 0)), int(segment.get("frame_span", 0))),
        )
        longest_long_segment = max(
            long_segments,
            key=lambda segment: (int(segment.get("num_points", 0)), int(segment.get("frame_span", 0))),
            default=None,
        )
        rows.append(
            {
                "id": int(gt_id),
                "num_segments": int(len(id_segments)),
                "num_long_segments": int(len(long_segments)),
                "total_detector_miss_points": int(sum(int(segment.get("num_points", 0)) for segment in id_segments)),
                "total_long_segment_points": int(sum(int(segment.get("num_points", 0)) for segment in long_segments)),
                "longest_segment_points": int(longest_segment.get("num_points", 0)),
                "longest_segment_start_frame": int(longest_segment.get("start_frame", -1)),
                "longest_segment_end_frame": int(longest_segment.get("end_frame", -1)),
                "longest_segment_span": int(longest_segment.get("frame_span", 0)),
                "mean_track_follows_det_ratio": float(
                    sum(float(segment.get("track_follows_det_ratio", 0.0)) for segment in id_segments) / max(len(id_segments), 1)
                ),
                "longest_long_segment_points": int(longest_long_segment.get("num_points", 0)) if longest_long_segment is not None else 0,
                "longest_long_segment_start_frame": int(longest_long_segment.get("start_frame", -1)) if longest_long_segment is not None else -1,
                "longest_long_segment_end_frame": int(longest_long_segment.get("end_frame", -1)) if longest_long_segment is not None else -1,
            }
        )
    rows.sort(
        key=lambda row: (
            -int(row["total_detector_miss_points"]),
            -int(row["longest_segment_points"]),
            int(row["id"]),
        )
    )
    return rows


def summarize_temporal_window_metrics(
    tracks: list[Track],
    gt_csv_path: str,
    *,
    fps: float,
    window_seconds: float = 20.0,
    audit_rows: list[dict] | None = None,
    match_dist: float = 30.0,
    compute_hota: bool = True,
    ignore_unlabeled_frames: bool = True,
    gt_frame_stride: int | None = None,
    gt_frame_offset: int = 0,
    prediction_id_source: str = "track_id",
) -> list[dict]:
    """Compute fixed-length time-window metrics so long videos can be inspected for temporal drift.

    计算固定长度时间窗口指标，用于检查长视频中的时间漂移。
    """
    gt_by_frame = _filter_gt_frames(read_points_csv(gt_csv_path), gt_frame_stride, gt_frame_offset)
    if not gt_by_frame:
        return []

    pred_by_frame = tracks_to_frame_points(tracks, prediction_id_source=prediction_id_source)
    valid_fps = float(fps) if fps and fps > 0.0 else 0.0
    if valid_fps > 0.0:
        window_frames = max(int(round(valid_fps * max(float(window_seconds), 1e-3))), 1)
    else:
        window_frames = max(int(gt_frame_stride or 1), 1)

    track_start_frames = [int(track.start_frame) for track in tracks if track.trajectory]
    all_gt_frames = sorted(gt_by_frame.keys())
    rows: list[dict] = []
    start_frame = int(all_gt_frames[0])
    final_frame = int(all_gt_frames[-1])
    window_index = 0

    while start_frame <= final_frame:
        end_frame = start_frame + window_frames - 1
        gt_window = {
            frame_idx: points
            for frame_idx, points in gt_by_frame.items()
            if start_frame <= int(frame_idx) <= end_frame
        }
        if gt_window:
            pred_window = {
                frame_idx: points
                for frame_idx, points in pred_by_frame.items()
                if start_frame <= int(frame_idx) <= end_frame
            }
            metrics = compute_point_tracking_metrics(
                gt_window,
                pred_window,
                match_dist=match_dist,
                compute_hota=compute_hota,
                ignore_unlabeled_frames=ignore_unlabeled_frames,
                gt_frame_stride=None,
                gt_frame_offset=0,
            )
            window_audit = [
                row
                for row in (audit_rows or [])
                if start_frame <= int(row.get("frame", -1)) <= end_frame
            ]
            window_misses = [row for row in window_audit if not bool(int(row.get("matched", 0)))]
            miss_counts = Counter(str(row.get("miss_stage", "")) for row in window_misses)
            rows.append(
                {
                    "window_index": int(window_index),
                    "start_frame": int(start_frame),
                    "end_frame": int(min(end_frame, final_frame)),
                    "start_sec": float(start_frame / valid_fps) if valid_fps > 0.0 else None,
                    "end_sec": float(min(end_frame, final_frame) / valid_fps) if valid_fps > 0.0 else None,
                    "window_gt_points": int(sum(len(points) for points in gt_window.values())),
                    "new_track_count": int(sum(start_frame <= frame_idx <= end_frame for frame_idx in track_start_frames)),
                    "matched_points": int(metrics["matched_points"]),
                    "fp": int(metrics["fp"]),
                    "fn": int(metrics["fn"]),
                    "idsw": int(metrics["idsw"]),
                    "idp": float(metrics["idp"]),
                    "idr": float(metrics["idr"]),
                    "idf1": float(metrics["idf1"]),
                    "track_coverage": float(metrics["track_coverage"]),
                    "point_hota": float(metrics["point_hota"]),
                    "detector_miss": int(miss_counts.get("detector_miss", 0)),
                    "quality_filter_drop": int(miss_counts.get("quality_filter_drop", 0)),
                    "association_miss": int(miss_counts.get("association_miss", 0)),
                    "lost_before_match": int(miss_counts.get("lost_before_match", 0)),
                    "postprocess_loss": int(miss_counts.get("postprocess_loss", 0)),
                }
            )
        start_frame = end_frame + 1
        window_index += 1
    return rows


def summarize_hard_case_buckets(
    audit_rows: list[dict],
    frame_debug_by_frame: dict[int, dict] | None = None,
) -> list[dict]:
    misses = [row for row in audit_rows if not bool(int(row.get("matched", 0)))]
    total = max(len(misses), 1)
    frame_debug_by_frame = frame_debug_by_frame or {}
    bucket_specs = [
        ("close_interaction_frames", "crossing / close-interaction frames", "has_close_interaction"),
        ("border_frames", "border frames", "has_border"),
        ("blur_frames", "blur frames", "has_blur"),
        ("rescue_generated_detections", "rescue-generated detections", "has_rescue"),
        ("track_supported_low_quality", "track-supported low-quality detections", "has_track_supported"),
        ("high_speed_motion_frames", "high-speed motion frames", "high_speed_motion"),
        ("merge_risk_frames", "merge-risk / crowded frames", "has_merge_risk"),
    ]
    rows: list[dict] = []
    for bucket, description, key in bucket_specs:
        count = sum(
            1
            for miss in misses
            if bool(frame_debug_by_frame.get(int(miss["frame"]), {}).get(key, False))
        )
        rows.append(
            {
                "bucket": bucket,
                "count": int(count),
                "ratio": float(count / total),
                "description": description,
            }
        )
    return rows
