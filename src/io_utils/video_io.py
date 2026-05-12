from __future__ import annotations

from pathlib import Path

import cv2


def open_video(video_path: str | Path):
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")
    return cap


def get_video_meta(cap) -> dict:
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return {"fps": fps, "width": width, "height": height, "total_frames": total_frames}


def create_video_writer(path: str | Path, fps: float, width: int, height: int):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise OSError(f"Failed to create video writer: {path}")
    return writer
