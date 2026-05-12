from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from core.structures import Detection, Track
from tracklet.features import extract_track_statistics


def _display_track_id(track: Track) -> int:
    return int(track.identity_slot) if track.identity_slot is not None else int(track.track_id)


def write_detections_csv(path: str | Path, detections_by_frame: dict[int, list[Detection]]) -> None:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
            [
                "frame",
                "x1",
                "y1",
                "x2",
                "y2",
                "cx",
                "cy",
                "conf",
                "cls",
                "raw_tid",
                "area",
                "aspect",
                "blur_score",
                "quality_flags",
            ]
        )
        for frame_idx in sorted(detections_by_frame):
            for det in detections_by_frame[frame_idx]:
                x1, y1, x2, y2 = det.bbox
                writer.writerow(
                    [
                        frame_idx,
                        x1,
                        y1,
                        x2,
                        y2,
                        det.center[0],
                        det.center[1],
                        det.conf,
                        det.cls_id,
                        det.raw_tid,
                        det.area,
                        det.aspect,
                        det.blur_score,
                        "|".join(det.quality_flags),
                    ]
                )


def write_tracks_csv(path: str | Path, tracks: list[Track]) -> None:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
            [
                "track_id",
                "identity_slot",
                "display_id",
                "frame",
                "x",
                "y",
                "x1",
                "y1",
                "x2",
                "y2",
                "state",
                "conf",
                "interpolated",
            ]
        )
        for track in sorted(tracks, key=lambda item: item.track_id):
            for obs in sorted(track.trajectory, key=lambda item: item.frame_idx):
                display_id = _display_track_id(track)
                x1, y1, x2, y2 = obs.bbox
                writer.writerow(
                    [
                        track.track_id,
                        "" if track.identity_slot is None else track.identity_slot,
                        display_id,
                        obs.frame_idx,
                        obs.center[0],
                        obs.center[1],
                        x1,
                        y1,
                        x2,
                        y2,
                        obs.state,
                        obs.conf,
                        int(obs.interpolated),
                    ]
                )


def write_track_stats_csv(path: str | Path, tracks: list[Track]) -> None:
    path = Path(path)
    stats = [extract_track_statistics(track) for track in tracks]
    if not stats:
        return
    fieldnames = list(stats[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats)


def write_events_csv(path: str | Path, events: list[dict]) -> None:
    path = Path(path)
    if not events:
        with path.open("w", newline="", encoding="utf-8") as file_obj:
            writer = csv.writer(file_obj)
            writer.writerow(["frame_idx", "type"])
        return
    preferred_order = [
        "frame_idx",
        "type",
        "track_id",
        "display_id",
        "identity_slot",
        "fragment_track_id",
        "track_a",
        "track_b",
        "display_track_a",
        "display_track_b",
        "identity_slot_a",
        "identity_slot_b",
        "fragment_track_a",
        "fragment_track_b",
        "keep_track_id",
        "drop_track_id",
        "display_keep_track_id",
        "display_drop_track_id",
        "keep_identity_slot",
        "drop_identity_slot",
        "fragment_keep_track_id",
        "fragment_drop_track_id",
    ]
    seen: set[str] = set()
    fieldnames: list[str] = []
    for name in preferred_order:
        if any(name in event for event in events):
            fieldnames.append(name)
            seen.add(name)
    for event in events:
        for key in event.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(events)


def write_metrics_csv(path: str | Path, metrics: dict) -> None:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def write_table_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    if not rows:
        path.touch()
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_recall_audit_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    fieldnames = [
        "frame",
        "id",
        "matched",
        "pred_track_id",
        "nearest_det_dist",
        "nearest_track_dist",
        "miss_stage",
    ]
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def read_points_csv(path: str | Path) -> dict[int, list[dict]]:
    path = Path(path)
    by_frame: dict[int, list[dict]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            frame_idx = int(row["frame"])
            by_frame[frame_idx].append({"id": int(row["id"]), "x": float(row["x"]), "y": float(row["y"])})
    return by_frame


def read_tracks_csv(path: str | Path) -> dict[int, list[dict]]:
    path = Path(path)
    by_frame: dict[int, list[dict]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            frame_idx = int(row["frame"])
            track_id = int(row["track_id"])
            identity_slot = None if row.get("identity_slot", "") == "" else int(row["identity_slot"])
            display_id_raw = row.get("display_id", "")
            if identity_slot is not None:
                display_id = identity_slot
            elif display_id_raw not in (None, ""):
                display_id = int(display_id_raw)
            else:
                display_id = track_id
            by_frame[frame_idx].append(
                {
                    "id": display_id,
                    "track_id": track_id,
                    "identity_slot": identity_slot,
                    "display_id": display_id,
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "state": row["state"],
                    "interpolated": bool(int(row["interpolated"])),
                }
            )
    return by_frame
