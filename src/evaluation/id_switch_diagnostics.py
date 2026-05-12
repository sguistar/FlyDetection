from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import csv

import numpy as np

from association.association_head import spatial_detection_distance, temporal_detection_distance
from association.cost import (
    _appearance_distance,
    _direction_distance,
    _identity_conflict_intensity,
    _identity_distance,
    _kalman_distance,
    _motion_distance,
    _shape_distance,
    _spatial_distance,
)
from config import clone_config, get_config
from evaluation.metrics import _filter_gt_frames, _frame_assignment, tracks_to_frame_points
from io_utils import open_video, read_points_csv, setup_logger
from main import _associate_frame, _resolve_detections, build_runtime_components, postprocess_tracks, process_video_frames


def _write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _find_identity_switch_events(
    gt_by_frame: dict[int, list[dict]],
    pred_by_frame: dict[int, list[dict]],
    *,
    match_dist: float,
    gt_frame_stride: int | None = None,
    gt_frame_offset: int = 0,
) -> list[dict]:
    """Locate labeled GT frames where the matched prediction id changes across time.

    定位带标注 GT 中匹配预测 ID 随时间发生变化的帧。
    """
    gt_by_frame = _filter_gt_frames(gt_by_frame, gt_frame_stride, gt_frame_offset)
    last_pred_for_gt: dict[int, int] = {}
    switch_events: list[dict] = []
    event_id = 0

    for frame_idx in sorted(gt_by_frame.keys()):
        gts = gt_by_frame.get(frame_idx, [])
        preds = pred_by_frame.get(frame_idx, [])
        matches, _, _ = _frame_assignment(gts, preds, match_dist=match_dist)
        for row_idx, col_idx, distance in matches:
            gt = gts[row_idx]
            pred = preds[col_idx]
            gt_id = int(gt["id"])
            pred_id = int(pred["id"])
            previous_pred = last_pred_for_gt.get(gt_id)
            if previous_pred is not None and previous_pred != pred_id:
                switch_events.append(
                    {
                        "event_id": int(event_id),
                        "frame": int(frame_idx),
                        "gt_id": gt_id,
                        "previous_pred_id": int(previous_pred),
                        "pred_id": int(pred_id),
                        "gt_x": float(gt["x"]),
                        "gt_y": float(gt["y"]),
                        "pred_x": float(pred["x"]),
                        "pred_y": float(pred["y"]),
                        "match_distance": float(distance),
                    }
                )
                event_id += 1
            last_pred_for_gt[gt_id] = pred_id
    return switch_events


def _track_state_name(track) -> str:
    state = getattr(track, "state", None)
    return getattr(state, "value", str(state))


def _candidate_detection_indices(
    detections: list,
    gt_x: float,
    gt_y: float,
    *,
    max_candidate_detections: int,
    match_dist: float,
) -> list[int]:
    if not detections:
        return []
    ranked = sorted(
        range(len(detections)),
        key=lambda idx: float(np.hypot(detections[idx].center[0] - gt_x, detections[idx].center[1] - gt_y)),
    )
    close = [
        idx
        for idx in ranked
        if float(np.hypot(detections[idx].center[0] - gt_x, detections[idx].center[1] - gt_y)) <= 1.5 * match_dist
    ]
    chosen = close[:max_candidate_detections] if close else ranked[:1]
    return chosen


def _capture_event_candidates(
    event: dict,
    *,
    cfg,
    active_tracks: list,
    detections: list,
    assoc,
    topk: int,
    max_candidate_detections: int,
) -> list[dict]:
    gt_x = float(event["gt_x"])
    gt_y = float(event["gt_y"])
    det_indices = _candidate_detection_indices(
        detections,
        gt_x,
        gt_y,
        max_candidate_detections=max_candidate_detections,
        match_dist=cfg.evaluation.point_match_distance,
    )
    if not det_indices:
        return [
            {
                **event,
                "det_idx": -1,
                "track_id": -1,
                "identity_slot": -1,
                "pair_rank": -1,
                "pair_cost": None,
                "match_score": None,
                "switch_risk": None,
                "conflict_intensity": 0.0,
                "identity_blend": 0.0,
                "note": "no_detections",
            }
        ]

    match_lookup = set(assoc.matches)
    rows: list[dict] = []
    for det_idx in det_indices:
        det = detections[det_idx]
        det_gt_dist = float(np.hypot(det.center[0] - gt_x, det.center[1] - gt_y))
        ranked_tracks = sorted(
            [
                idx
                for idx in range(len(active_tracks))
                if assoc.cost_matrix is not None and float(assoc.cost_matrix[idx, det_idx]) < cfg.association.large_cost
            ],
            key=lambda idx: float(assoc.cost_matrix[idx, det_idx]),
        )[:topk]
        if not ranked_tracks:
            rows.append(
                {
                    **event,
                    "det_idx": int(det_idx),
                    "det_x": float(det.center[0]),
                    "det_y": float(det.center[1]),
                    "det_conf": float(det.conf),
                    "det_gt_dist": det_gt_dist,
                    "detector_source": str(det.detector_source),
                    "is_crowded": int(bool(det.is_crowded)),
                    "is_merged_risk": int(bool(det.is_merged_risk)),
                    "switch_risk_hint": float(det.switch_risk_hint),
                    "local_density": float(det.context_feature[5]) if det.context_feature is not None and det.context_feature.shape[0] > 5 else 0.0,
                    "pair_rank": -1,
                    "track_id": -1,
                    "identity_slot": -1,
                    "pair_cost": None,
                    "match_score": None,
                    "switch_risk": None,
                    "conflict_intensity": float(
                        _identity_conflict_intensity(
                            det,
                            min_risk=cfg.association.hard_conflict_min_risk,
                            min_density=cfg.association.hard_conflict_min_density,
                        )
                    ),
                    "identity_blend": 0.0,
                    "note": "no_valid_tracks",
                }
            )
            continue

        for rank, track_idx in enumerate(ranked_tracks, start=1):
            track = active_tracks[track_idx]
            appearance_long, appearance_short = _appearance_distance(
                track,
                det,
                recent_embedding_window=cfg.feature.recent_embedding_window,
            )
            identity_distance, _ = _identity_distance(
                track,
                det,
                recent_embedding_window=cfg.feature.recent_embedding_window,
            )
            shape_cost = _shape_distance(track, det)
            spatial_cost = _spatial_distance(track, det)
            spatial_distance = spatial_detection_distance(track, det, embedding_dim=cfg.feature.embedding_dim)
            temporal_distance = temporal_detection_distance(track, det, embedding_dim=cfg.feature.embedding_dim)
            motion_cost = _motion_distance(track, det, motion_gate=cfg.association.motion_gate)
            kf_cost = _kalman_distance(track, det, kf_gate=cfg.association.kf_gate)
            direction_cost = _direction_distance(track, det)
            conflict_intensity = float(
                _identity_conflict_intensity(
                    det,
                    min_risk=cfg.association.hard_conflict_min_risk,
                    min_density=cfg.association.hard_conflict_min_density,
                )
            )
            identity_blend = float(cfg.association.hard_conflict_identity_blend * conflict_intensity)
            rows.append(
                {
                    **event,
                    "det_idx": int(det_idx),
                    "det_x": float(det.center[0]),
                    "det_y": float(det.center[1]),
                    "det_conf": float(det.conf),
                    "det_gt_dist": det_gt_dist,
                    "detector_source": str(det.detector_source),
                    "is_crowded": int(bool(det.is_crowded)),
                    "is_merged_risk": int(bool(det.is_merged_risk)),
                    "switch_risk_hint": float(det.switch_risk_hint),
                    "local_density": float(det.context_feature[5]) if det.context_feature is not None and det.context_feature.shape[0] > 5 else 0.0,
                    "pair_rank": int(rank),
                    "track_id": int(track.track_id),
                    "identity_slot": -1 if track.identity_slot is None else int(track.identity_slot),
                    "track_state": _track_state_name(track),
                    "track_hits": int(track.hits),
                    "track_missed": int(track.missed),
                    "is_assoc_match": int((track_idx, det_idx) in match_lookup),
                    "pair_cost": float(assoc.cost_matrix[track_idx, det_idx]),
                    "match_score": float(assoc.score_matrix[track_idx, det_idx]) if assoc.score_matrix is not None else None,
                    "switch_risk": float(assoc.switch_risk_matrix[track_idx, det_idx]) if assoc.switch_risk_matrix is not None else None,
                    "appearance_long": float(appearance_long),
                    "appearance_short": float(appearance_short),
                    "identity_distance": float(identity_distance),
                    "temporal_distance": float(temporal_distance),
                    "shape_cost": float(shape_cost),
                    "spatial_cost": float(spatial_cost),
                    "spatial_distance": float(spatial_distance),
                    "motion_cost": float(motion_cost),
                    "kf_cost": float(kf_cost),
                    "direction_cost": float(direction_cost),
                    "conflict_intensity": conflict_intensity,
                    "identity_blend": identity_blend,
                    "hard_conflict_active": int(identity_blend > 0.0),
                }
            )
    return rows


def _summarize_switch_rows(events: list[dict], candidate_rows: list[dict]) -> dict:
    """Aggregate a compact view of switch frequency and whether conflict-aware IM was actually active.

    汇总 ID 切换频率，并简要呈现冲突感知 IM 是否实际生效。
    """
    top_rows = [row for row in candidate_rows if int(row.get("pair_rank", -1)) == 1]
    conflict_rows = [row for row in candidate_rows if float(row.get("identity_blend", 0.0) or 0.0) > 0.0]
    summary = {
        "num_switch_events": int(len(events)),
        "num_unique_gt_ids": int(len({int(row["gt_id"]) for row in events})) if events else 0,
        "num_candidate_rows": int(len(candidate_rows)),
        "num_top_rows": int(len(top_rows)),
        "frames_with_conflict_active": int(len({int(row["frame"]) for row in conflict_rows})),
        "mean_conflict_intensity": float(np.mean([float(row.get("conflict_intensity", 0.0) or 0.0) for row in candidate_rows])) if candidate_rows else 0.0,
        "mean_top_identity_blend": float(np.mean([float(row.get("identity_blend", 0.0) or 0.0) for row in top_rows])) if top_rows else 0.0,
        "mean_top_match_score": float(np.mean([float(row.get("match_score", 0.0) or 0.0) for row in top_rows])) if top_rows else 0.0,
        "mean_top_switch_risk": float(np.mean([float(row.get("switch_risk", 0.0) or 0.0) for row in top_rows])) if top_rows else 0.0,
    }
    type_counts = defaultdict(int)
    for row in events:
        type_counts[str(row.get("event_type", "unclassified"))] += 1
    for event_type, count in sorted(type_counts.items()):
        summary[f"type_{event_type}"] = int(count)
    return summary


def _classify_switch_events(
    events: list[dict],
    candidate_rows: list[dict],
    *,
    swap_window: int = 20,
    reclaim_cost_margin: float = 0.15,
    reclaim_min_hits: int = 24,
    reclaim_min_hit_advantage: int = 48,
) -> list[dict]:
    """Tag each switch with the most plausible failure mode so targeted fixes can stay narrow.

    为每次切换标记最可能的失败模式，便于有针对性地缩小修复范围。
    """
    rows_by_event: dict[int, list[dict]] = defaultdict(list)
    events_by_gt: dict[int, list[dict]] = defaultdict(list)
    for row in candidate_rows:
        rows_by_event[int(row["event_id"])].append(row)
    for event in events:
        events_by_gt[int(event["gt_id"])].append(event)
    for gt_events in events_by_gt.values():
        gt_events.sort(key=lambda item: int(item["frame"]))

    classified: list[dict] = []
    for event in events:
        event_id = int(event["event_id"])
        gt_id = int(event["gt_id"])
        event_rows = rows_by_event.get(event_id, [])
        top_rows = [row for row in event_rows if int(row.get("pair_rank", -1)) == 1]
        top_row = min(
            top_rows,
            key=lambda row: (
                float(row.get("det_gt_dist", 1e9) or 1e9),
                float(row.get("pair_cost", 1e9) or 1e9),
            ),
        ) if top_rows else None
        sibling_rows = [
            row
            for row in event_rows
            if top_row is not None
            and int(row.get("det_idx", -1)) == int(top_row.get("det_idx", -1))
            and int(row.get("pair_rank", -1)) > 1
        ]
        next_event = None
        for candidate in events_by_gt.get(gt_id, []):
            if int(candidate["frame"]) <= int(event["frame"]):
                continue
            if int(candidate["frame"]) - int(event["frame"]) > swap_window:
                break
            if (
                int(candidate["previous_pred_id"]) == int(event["pred_id"])
                and int(candidate["pred_id"]) == int(event["previous_pred_id"])
            ):
                next_event = candidate
                break

        event_type = "single_candidate_rebind"
        repair_target = "inspect_detector_or_memory"
        if next_event is not None:
            event_type = "swap_back_pair"
            repair_target = "slot_stickiness_or_lifecycle"
        elif top_row is not None and float(top_row.get("conflict_intensity", 0.0) or 0.0) > 0.0:
            event_type = "crowded_conflict"
            repair_target = "conflict_association"
        elif top_row is not None:
            reclaim_candidate = next(
                (
                    row
                    for row in sibling_rows
                    if str(row.get("track_state")) == "Lost"
                    and int(row.get("track_hits", 0) or 0) >= reclaim_min_hits
                    and int(row.get("track_hits", 0) or 0) >= int(top_row.get("track_hits", 0) or 0) + reclaim_min_hit_advantage
                    and float(row.get("pair_cost", 1e9) or 1e9) <= float(top_row.get("pair_cost", 1e9) or 1e9) + reclaim_cost_margin
                ),
                None,
            )
            if reclaim_candidate is not None:
                event_type = "lost_slot_reclaim_candidate"
                repair_target = "lifecycle_reactivation"
            elif sibling_rows:
                event_type = "multi_candidate_rebind"
                repair_target = "association_ranking"

        enriched = dict(event)
        enriched["event_type"] = event_type
        enriched["repair_target"] = repair_target
        enriched["candidate_count"] = int(len(event_rows))
        enriched["top_conflict_intensity"] = 0.0 if top_row is None else float(top_row.get("conflict_intensity", 0.0) or 0.0)
        enriched["top_identity_blend"] = 0.0 if top_row is None else float(top_row.get("identity_blend", 0.0) or 0.0)
        classified.append(enriched)
    return classified


def run_id_switch_diagnostics(
    cfg=None,
    *,
    output_root: str | Path | None = None,
    topk: int = 4,
    max_candidate_detections: int = 2,
) -> dict[str, object]:
    cfg = clone_config(cfg or get_config())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    diag_root = Path(output_root) if output_root is not None else cfg.paths.root / "diagnostics" / f"id_switch_{timestamp}"
    diag_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(diag_root / "id_switch.log")

    cfg.runtime.save_video = False
    cfg.runtime.save_cache = False
    cfg.cache.enabled = False
    cfg.cache.use_detection_cache = False
    cfg.cache.write_detection_cache = False

    components = build_runtime_components(cfg, logger, cache_payload=None)
    frame_results = process_video_frames(cfg, components, logger, cache_payload=None)
    postprocess = postprocess_tracks(cfg, components, frame_results)

    gt_by_frame = read_points_csv(cfg.evaluation.gt_csv_path)
    pred_by_frame = tracks_to_frame_points(
        postprocess["tracks"],
        prediction_id_source=cfg.evaluation.prediction_id_source,
    )
    switch_events = _find_identity_switch_events(
        gt_by_frame,
        pred_by_frame,
        match_dist=cfg.evaluation.point_match_distance,
        gt_frame_stride=cfg.evaluation.gt_frame_stride,
        gt_frame_offset=cfg.evaluation.gt_frame_offset,
    )
    events_by_frame: dict[int, list[dict]] = defaultdict(list)
    for event in switch_events:
        events_by_frame[int(event["frame"])].append(event)

    replay_components = build_runtime_components(cfg, logger, cache_payload=None)
    cap = open_video(cfg.runtime.video_path)
    candidate_rows: list[dict] = []
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            replay_components["builder"].predict()
            detection_info = _resolve_detections(
                frame_idx,
                frame,
                cfg=cfg,
                builder=replay_components["builder"],
                cache_payload=None,
                detector=replay_components["detector"],
                quality_filter=replay_components["quality_filter"],
                encoder=replay_components["encoder"],
            )
            detections = detection_info["accepted_detections"]
            active_tracks = list(replay_components["builder"].tracks)
            assoc = _associate_frame(cfg, replay_components["builder"], detections)

            for event in events_by_frame.get(frame_idx, []):
                candidate_rows.extend(
                    _capture_event_candidates(
                        event,
                        cfg=cfg,
                        active_tracks=active_tracks,
                        detections=detections,
                        assoc=assoc,
                        topk=topk,
                        max_candidate_detections=max_candidate_detections,
                    )
                )

            replay_components["builder"].update(frame_idx, detections, assoc)
            frame_idx += 1
    finally:
        cap.release()

    switch_events = _classify_switch_events(
        switch_events,
        candidate_rows,
        reclaim_cost_margin=0.15,
        reclaim_min_hits=24,
        reclaim_min_hit_advantage=48,
    )
    summary = _summarize_switch_rows(switch_events, candidate_rows)
    summary["metrics_idsw"] = int(postprocess["metrics"].get("idsw", 0))
    summary["metric_idf1"] = float(postprocess["metrics"].get("idf1", 0.0))

    events_path = diag_root / "id_switch_events.csv"
    candidates_path = diag_root / "id_switch_pair_candidates.csv"
    summary_path = diag_root / "id_switch_summary.csv"
    _write_rows(events_path, switch_events)
    _write_rows(candidates_path, candidate_rows)
    _write_rows(summary_path, [summary])

    logger.info("finished id-switch diagnostics | %s", {"summary": summary, "summary_path": str(summary_path)})
    return {
        "diagnostic_root": str(diag_root),
        "events_path": str(events_path),
        "candidates_path": str(candidates_path),
        "summary_path": str(summary_path),
        "summary": summary,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose frame-level association context around ID switches.")
    parser.add_argument("--output-root", default=None, help="Optional diagnostic output directory.")
    parser.add_argument("--topk", type=int, default=4, help="How many candidate tracks to keep per target detection.")
    parser.add_argument("--max-candidate-detections", type=int, default=2, help="How many detections near the GT point to inspect.")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    result = run_id_switch_diagnostics(
        output_root=args.output_root,
        topk=args.topk,
        max_candidate_detections=args.max_candidate_detections,
    )
    print(result)


if __name__ == "__main__":
    main()
