from __future__ import annotations

import csv
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any


def _coerce_value(value: str | None) -> Any:
    if value is None:
        return None
    text = value.strip()
    if text == "":
        return None
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if all(char not in text for char in ".eE"):
            return int(text)
        return float(text)
    except ValueError:
        return text


def read_table_csv(path: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    csv_path = Path(path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return []
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            rows.append({key: _coerce_value(value) for key, value in row.items()})
            if limit is not None and len(rows) >= limit:
                break
    return rows


def read_metrics_csv(path: str | Path) -> dict[str, Any]:
    rows = read_table_csv(path, limit=1)
    return rows[0] if rows else {}


def parse_log_progress(path: str | Path) -> dict[str, Any]:
    log_path = Path(path)
    progress: dict[str, Any] = {
        "processed_frame": None,
        "active_tracks": None,
        "num_detections": None,
        "finished": False,
    }
    if not log_path.exists():
        return progress

    with log_path.open("r", encoding="utf-8", errors="replace") as file_obj:
        for line in file_obj:
            if "Processed frame" in line:
                payload = _json_payload_from_log_line(line)
                if payload:
                    progress["processed_frame"] = payload.get("frame_idx", progress["processed_frame"])
                    progress["active_tracks"] = payload.get("active_tracks", progress["active_tracks"])
                    progress["num_detections"] = payload.get("num_detections", progress["num_detections"])
            elif "Finished MOT pipeline" in line:
                progress["finished"] = True
                payload = _json_payload_from_log_line(line)
                if payload:
                    progress["num_tracks"] = payload.get("num_tracks")
                    progress["num_events"] = payload.get("num_events")
    return progress


def _json_payload_from_log_line(line: str) -> dict[str, Any] | None:
    marker = " | {"
    idx = line.find(marker)
    if idx < 0:
        return None
    payload = line[idx + 3 :].strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def tail_text(path: str | Path, *, lines: int = 200) -> str:
    log_path = Path(path)
    if not log_path.exists():
        return ""
    buffer: deque[str] = deque(maxlen=max(int(lines), 1))
    with log_path.open("r", encoding="utf-8", errors="replace") as file_obj:
        for line in file_obj:
            buffer.append(line.rstrip("\n"))
    return "\n".join(buffer)


def parse_tracks_csv(
    path: str | Path,
    *,
    stride: int = 1,
    from_frame: int | None = None,
    to_frame: int | None = None,
) -> dict[str, Any]:
    csv_path = Path(path)
    stride = max(int(stride), 1)
    tracks: dict[int, list[dict[str, Any]]] = defaultdict(list)
    bounds = {
        "minX": None,
        "maxX": None,
        "minY": None,
        "maxY": None,
        "minFrame": None,
        "maxFrame": None,
    }

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return {"tracks": [], "bounds": bounds, "frameCount": 0}

    with csv_path.open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            frame = int(float(row["frame"]))
            if from_frame is not None and frame < from_frame:
                continue
            if to_frame is not None and frame > to_frame:
                continue
            if frame % stride != 0:
                continue

            track_id = int(float(row["track_id"]))
            display_raw = row.get("display_id") or row.get("identity_slot") or row["track_id"]
            display_id = int(float(display_raw))
            x = float(row["x"])
            y = float(row["y"])
            point = {
                "f": frame,
                "x": x,
                "y": y,
                "trackId": track_id,
                "displayId": display_id,
                "state": row.get("state") or "",
                "conf": float(row.get("conf") or 0.0),
                "interpolated": bool(int(float(row.get("interpolated") or 0))),
            }
            tracks[display_id].append(point)
            _update_bounds(bounds, x, y, frame)

    track_payload = [
        {"id": display_id, "points": sorted(points, key=lambda item: item["f"])}
        for display_id, points in sorted(tracks.items(), key=lambda item: item[0])
    ]
    frame_count = 0
    if bounds["minFrame"] is not None and bounds["maxFrame"] is not None:
        frame_count = int(bounds["maxFrame"]) - int(bounds["minFrame"]) + 1
    return {"tracks": track_payload, "bounds": bounds, "frameCount": frame_count}


def _update_bounds(bounds: dict[str, Any], x: float, y: float, frame: int) -> None:
    bounds["minX"] = x if bounds["minX"] is None else min(bounds["minX"], x)
    bounds["maxX"] = x if bounds["maxX"] is None else max(bounds["maxX"], x)
    bounds["minY"] = y if bounds["minY"] is None else min(bounds["minY"], y)
    bounds["maxY"] = y if bounds["maxY"] is None else max(bounds["maxY"], y)
    bounds["minFrame"] = frame if bounds["minFrame"] is None else min(bounds["minFrame"], frame)
    bounds["maxFrame"] = frame if bounds["maxFrame"] is None else max(bounds["maxFrame"], frame)

