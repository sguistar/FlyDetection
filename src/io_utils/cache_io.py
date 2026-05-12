from __future__ import annotations

import pickle
from pathlib import Path


def save_pickle(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file_obj:
        pickle.dump(obj, file_obj)


def load_pickle(path: str | Path):
    path = Path(path)
    with path.open("rb") as file_obj:
        return pickle.load(file_obj)


def build_cache_path(cache_root: str | Path, video_path: str | Path, config_hash: str, kind: str) -> Path:
    cache_root = Path(cache_root)
    video_stem = Path(video_path).stem
    return cache_root / f"{video_stem}.{kind}.{config_hash}.pkl"


def save_detection_cache(cache_root: str | Path, video_path: str | Path, config_hash: str, payload) -> Path:
    path = build_cache_path(cache_root, video_path, config_hash, kind="detections")
    save_pickle(path, {"config_hash": config_hash, "payload": payload})
    return path


def load_detection_cache(cache_root: str | Path, video_path: str | Path, config_hash: str):
    path = build_cache_path(cache_root, video_path, config_hash, kind="detections")
    if not path.exists():
        return None
    envelope = load_pickle(path)
    if envelope.get("config_hash") != config_hash:
        return None
    return envelope.get("payload")
