from __future__ import annotations

import csv
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import get_config
from detector.crop_utils import crop_from_bbox
from io_utils.csv_io import read_points_csv
from io_utils.video_io import open_video
from training.dataset import has_reid_layout

_CROP_NAME_PATTERN = re.compile(r"frame_(\d+)_id_(\d+)", re.IGNORECASE)


def _build_identity_splits(
    gt_by_frame: dict[int, list[dict]],
    *,
    val_split: float,
    min_images_per_identity: int,
    seed: int,
) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    frames_by_identity: dict[int, list[int]] = defaultdict(list)
    for frame_idx, rows in gt_by_frame.items():
        for row in rows:
            frames_by_identity[int(row["id"])].append(int(frame_idx))

    rng = random.Random(seed)
    train_frames: dict[int, set[int]] = {}
    val_frames: dict[int, set[int]] = {}
    for identity, frame_indices in sorted(frames_by_identity.items()):
        unique_frames = sorted(set(frame_indices))
        if len(unique_frames) < min_images_per_identity:
            continue
        rng.shuffle(unique_frames)
        split_idx = max(1, int(round(len(unique_frames) * (1.0 - val_split))))
        split_idx = min(split_idx, len(unique_frames) - 1) if len(unique_frames) > 1 else len(unique_frames)
        train_frames[identity] = set(unique_frames[:split_idx])
        val_frames[identity] = set(unique_frames[split_idx:] if split_idx < len(unique_frames) else unique_frames[-1:])
    return train_frames, val_frames


def _build_frame_local_stats(gt_by_frame: dict[int, list[dict]]) -> dict[tuple[int, int], dict[str, float]]:
    stats: dict[tuple[int, int], dict[str, float]] = {}
    for frame_idx, rows in gt_by_frame.items():
        centers = [(float(row["x"]), float(row["y"]), int(row["id"])) for row in rows]
        for cx, cy, identity in centers:
            distances = [
                ((other_x - cx) ** 2 + (other_y - cy) ** 2) ** 0.5
                for other_x, other_y, other_id in centers
                if other_id != identity
            ]
            if distances:
                nearest = min(distances)
                neighbor_count = sum(dist < 40.0 for dist in distances)
                local_density = float(
                    min(
                        sum(max(0.0, 48.0 - dist) for dist in distances) / (48.0 * 4.0),
                        1.0,
                    )
                )
            else:
                nearest = 9999.0
                neighbor_count = 0
                local_density = 0.0
            stats[(int(frame_idx), int(identity))] = {
                "nearest_neighbor_dist": float(nearest),
                "neighbor_count": float(neighbor_count),
                "local_density": float(local_density),
            }
    return stats


def _write_metadata_csv(data_root: Path, metadata_rows: list[dict[str, object]]) -> Path:
    metadata_path = data_root / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as file_obj:
        fieldnames = [
            "split",
            "identity",
            "frame_idx",
            "image_path",
            "center_x",
            "center_y",
            "crop_w",
            "crop_h",
            "area_proxy",
            "aspect_proxy",
            "frame_w",
            "frame_h",
            "border_flag",
            "local_density",
            "neighbor_count",
            "nearest_neighbor_dist",
        ]
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)
    return metadata_path


def _video_frame_size(video_path: str | Path) -> tuple[int, int]:
    cap = open_video(video_path)
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width > 0 and height > 0:
            return width, height
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Unable to read frame size from video: {video_path}")
        return int(frame.shape[1]), int(frame.shape[0])
    finally:
        cap.release()


def _parse_existing_crop(path: Path) -> tuple[int, int] | None:
    match = _CROP_NAME_PATTERN.search(path.stem)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def backfill_reid_metadata(
    *,
    data_root: str | Path,
    video_path: str | Path,
    gt_csv_path: str | Path,
    train_subdir: str = "train",
    val_subdir: str = "val",
) -> dict:
    data_root = Path(data_root)
    gt_csv_path = Path(gt_csv_path)
    if not gt_csv_path.exists():
        raise FileNotFoundError(f"GT csv not found for metadata backfill: {gt_csv_path}")

    gt_by_frame = read_points_csv(gt_csv_path)
    if not gt_by_frame:
        raise RuntimeError(f"No point annotations found in {gt_csv_path}")

    gt_lookup = {
        (int(frame_idx), int(row["id"])): row
        for frame_idx, frame_rows in gt_by_frame.items()
        for row in frame_rows
    }
    frame_local_stats = _build_frame_local_stats(gt_by_frame)
    frame_w, frame_h = _video_frame_size(video_path)
    metadata_rows: list[dict[str, object]] = []

    for split_name, split_root in (
        ("train", data_root / train_subdir),
        ("val", data_root / val_subdir),
    ):
        if not split_root.exists():
            continue
        for identity_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
            for crop_path in sorted(identity_dir.glob("*.jpg")):
                parsed = _parse_existing_crop(crop_path)
                if parsed is None:
                    continue
                frame_idx, identity = parsed
                gt_row = gt_lookup.get((frame_idx, identity))
                if gt_row is None:
                    continue
                crop = cv2.imread(str(crop_path))
                if crop is None:
                    continue
                crop_w = int(crop.shape[1])
                crop_h = int(crop.shape[0])
                cx = float(gt_row["x"])
                cy = float(gt_row["y"])
                border_flag = float(
                    cx - crop_w * 0.5 < 0
                    or cy - crop_h * 0.5 < 0
                    or cx + crop_w * 0.5 > frame_w
                    or cy + crop_h * 0.5 > frame_h
                )
                local_stats = frame_local_stats.get((frame_idx, identity), {})
                metadata_rows.append(
                    {
                        "split": split_name,
                        "identity": int(identity),
                        "frame_idx": int(frame_idx),
                        "image_path": str(crop_path.relative_to(data_root)),
                        "center_x": cx,
                        "center_y": cy,
                        "crop_w": crop_w,
                        "crop_h": crop_h,
                        "area_proxy": float(crop_w * crop_h),
                        "aspect_proxy": float(crop_w / max(crop_h, 1)),
                        "frame_w": frame_w,
                        "frame_h": frame_h,
                        "border_flag": border_flag,
                        "local_density": float(local_stats.get("local_density", 0.0)),
                        "neighbor_count": float(local_stats.get("neighbor_count", 0.0)),
                        "nearest_neighbor_dist": float(local_stats.get("nearest_neighbor_dist", 9999.0)),
                    }
                )

    if not metadata_rows:
        raise RuntimeError(f"No existing crop images found for metadata backfill under {data_root}")

    metadata_path = _write_metadata_csv(data_root, metadata_rows)
    return {
        "data_root": str(data_root),
        "prepared": False,
        "reason": "metadata_backfilled",
        "metadata_path": str(metadata_path),
        "num_metadata_rows": len(metadata_rows),
    }


def prepare_reid_dataset(
    *,
    data_root: str | Path,
    video_path: str | Path,
    gt_csv_path: str | Path,
    crop_size: tuple[int, int] = (96, 96),
    train_subdir: str = "train",
    val_subdir: str = "val",
    val_split: float = 0.2,
    min_images_per_identity: int = 2,
    seed: int = 42,
    write_clip_metadata: bool = True,
) -> dict:
    data_root = Path(data_root)
    gt_csv_path = Path(gt_csv_path)
    if has_reid_layout(data_root, train_subdir=train_subdir, val_subdir=val_subdir):
        return {"data_root": str(data_root), "prepared": False, "reason": "already_exists"}
    if not gt_csv_path.exists():
        raise FileNotFoundError(f"GT csv not found for ReID dataset preparation: {gt_csv_path}")

    gt_by_frame = read_points_csv(gt_csv_path)
    if not gt_by_frame:
        raise RuntimeError(f"No point annotations found in {gt_csv_path}")

    train_frames, val_frames = _build_identity_splits(
        gt_by_frame,
        val_split=val_split,
        min_images_per_identity=min_images_per_identity,
        seed=seed,
    )
    frame_local_stats = _build_frame_local_stats(gt_by_frame)
    if not train_frames:
        raise RuntimeError("Not enough labeled identities to prepare a ReID dataset.")

    train_root = data_root / train_subdir
    val_root = data_root / val_subdir
    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)

    cap = open_video(video_path)
    frame_idx = 0
    target_frames = set(gt_by_frame.keys())
    saved_train = 0
    saved_val = 0
    metadata_rows: list[dict[str, object]] = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx not in target_frames:
                frame_idx += 1
                continue

            frame_rows = gt_by_frame.get(frame_idx, [])
            for row in frame_rows:
                identity = int(row["id"])
                if identity not in train_frames:
                    continue
                if frame_idx in train_frames[identity]:
                    split_root = train_root
                elif frame_idx in val_frames.get(identity, set()):
                    split_root = val_root
                else:
                    continue

                cx = float(row["x"])
                cy = float(row["y"])
                half_w = crop_size[0] / 2.0
                half_h = crop_size[1] / 2.0
                crop = crop_from_bbox(
                    frame,
                    (cx - half_w, cy - half_h, cx + half_w, cy + half_h),
                    out_size=crop_size,
                    pad=0,
                )
                if crop is None:
                    continue

                identity_dir = split_root / str(identity)
                identity_dir.mkdir(parents=True, exist_ok=True)
                crop_path = identity_dir / f"frame_{frame_idx:06d}_id_{identity:02d}.jpg"
                if cv2.imwrite(str(crop_path), crop):
                    split_name = "train" if split_root == train_root else "val"
                    if split_root == train_root:
                        saved_train += 1
                    else:
                        saved_val += 1
                    if write_clip_metadata:
                        crop_w = int(crop.shape[1]) if crop is not None else int(crop_size[0])
                        crop_h = int(crop.shape[0]) if crop is not None else int(crop_size[1])
                        border_flag = float(
                            cx - half_w < 0
                            or cy - half_h < 0
                            or cx + half_w > frame.shape[1]
                            or cy + half_h > frame.shape[0]
                        )
                        local_stats = frame_local_stats.get((frame_idx, identity), {})
                        metadata_rows.append(
                            {
                                "split": split_name,
                                "identity": int(identity),
                                "frame_idx": int(frame_idx),
                                "image_path": str(crop_path.relative_to(data_root)),
                                "center_x": float(cx),
                                "center_y": float(cy),
                                "crop_w": crop_w,
                                "crop_h": crop_h,
                                "area_proxy": float(crop_w * crop_h),
                                "aspect_proxy": float(crop_w / max(crop_h, 1)),
                                "frame_w": int(frame.shape[1]),
                                "frame_h": int(frame.shape[0]),
                                "border_flag": border_flag,
                                "local_density": float(local_stats.get("local_density", 0.0)),
                                "neighbor_count": float(local_stats.get("neighbor_count", 0.0)),
                                "nearest_neighbor_dist": float(local_stats.get("nearest_neighbor_dist", 9999.0)),
                            }
                        )
            frame_idx += 1
    finally:
        cap.release()

    manifest_path = data_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["split", "identity", "num_images"])
        for split_name, split_root in (("train", train_root), ("val", val_root)):
            for identity_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
                num_images = len(list(identity_dir.glob("*.jpg")))
                writer.writerow([split_name, identity_dir.name, num_images])

    if write_clip_metadata:
        metadata_path = _write_metadata_csv(data_root, metadata_rows)
    else:
        metadata_path = data_root / "metadata.csv"

    return {
        "data_root": str(data_root),
        "prepared": True,
        "saved_train": saved_train,
        "saved_val": saved_val,
        "num_identities": len(train_frames),
        "manifest_path": str(manifest_path),
        "metadata_path": str(metadata_path) if write_clip_metadata else None,
    }


def ensure_reid_dataset(cfg) -> dict:
    data_root = Path(cfg.training.data_root)
    if has_reid_layout(
        data_root,
        train_subdir=cfg.training.train_subdir,
        val_subdir=cfg.training.val_subdir,
    ):
        metadata_path = data_root / "metadata.csv"
        if cfg.training.write_clip_metadata and not metadata_path.exists():
            return backfill_reid_metadata(
                data_root=data_root,
                video_path=cfg.runtime.video_path,
                gt_csv_path=cfg.evaluation.gt_csv_path,
                train_subdir=cfg.training.train_subdir,
                val_subdir=cfg.training.val_subdir,
            )
        return {"data_root": str(data_root), "prepared": False, "reason": "already_exists"}
    if not cfg.training.auto_prepare_from_gt:
        return {"data_root": str(data_root), "prepared": False, "reason": "auto_prepare_disabled"}
    return prepare_reid_dataset(
        data_root=data_root,
        video_path=cfg.runtime.video_path,
        gt_csv_path=cfg.evaluation.gt_csv_path,
        crop_size=cfg.feature.crop_size,
        train_subdir=cfg.training.train_subdir,
        val_subdir=cfg.training.val_subdir,
        val_split=cfg.training.val_split,
        min_images_per_identity=cfg.training.min_images_per_identity,
        seed=cfg.training.random_seed,
        write_clip_metadata=cfg.training.write_clip_metadata,
    )


def prepare_reid_dataset_main() -> None:
    cfg = get_config()
    summary = ensure_reid_dataset(cfg)
    print(summary)


if __name__ == "__main__":
    prepare_reid_dataset_main()
