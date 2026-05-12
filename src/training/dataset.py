from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from torch.utils.data import Dataset

from identity.spacial_context import SPACIAL_INPUT_DIM, build_crop_spatial_input
from identity.transforms import build_reid_input

_FRAME_PATTERN = re.compile(r"frame_(\d+)", re.IGNORECASE)
TEMPORAL_META_FIELDS = (
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
)
TEMPORAL_META_DIM = len(TEMPORAL_META_FIELDS)


@dataclass(frozen=True)
class ClipSample:
    paths: tuple[str, ...]
    label: int
    frame_indices: tuple[int, ...]
    anchor_frame: int = -1
    difficulty: float = 0.0
    hard_frame_count: int = 0
    source_kind: str = "base"


def _list_identity_dirs(root: Path) -> list[Path]:
    return sorted([path for path in root.iterdir() if path.is_dir()])


def _gather_images(identity_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([path for path in identity_dir.rglob("*") if path.suffix.lower() in exts])


def _parse_frame_index(path: str | Path) -> int:
    match = _FRAME_PATTERN.search(Path(path).stem)
    if match is not None:
        return int(match.group(1))
    return -1


def _normalize_metadata_key(path: str | Path) -> str:
    return str(Path(path)).replace("\\", "/")


def _candidate_metadata_keys(path: str | Path, metadata_root: str | Path | None) -> list[str]:
    path_obj = Path(path)
    candidates = [_normalize_metadata_key(path_obj.name)]
    if metadata_root is not None:
        root = Path(metadata_root)
        try:
            candidates.insert(0, _normalize_metadata_key(path_obj.relative_to(root)))
        except Exception:
            pass
    return candidates


def _metadata_row_for_path(
    path: str | Path,
    *,
    metadata_by_path: dict[str, dict[str, float]],
    metadata_root: str | Path | None,
) -> dict[str, float] | None:
    for key in _candidate_metadata_keys(path, metadata_root):
        row = metadata_by_path.get(key)
        if row is not None:
            return row
    return None


def _load_temporal_metadata(metadata_root: str | Path | None) -> dict[str, dict[str, float]]:
    if metadata_root is None:
        return {}
    metadata_path = Path(metadata_root) / "metadata.csv"
    if not metadata_path.exists():
        return {}
    import csv

    mapping: dict[str, dict[str, float]] = {}
    with metadata_path.open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            image_path = row.get("image_path")
            if not image_path:
                continue
            mapping[_normalize_metadata_key(image_path)] = {
                field: float(row.get(field, 0.0) or 0.0)
                for field in TEMPORAL_META_FIELDS
            }
    return mapping


def _temporal_meta_from_path(
    path: str | Path,
    *,
    image: np.ndarray,
    metadata_by_path: dict[str, dict[str, float]],
    metadata_root: str | Path | None,
) -> np.ndarray:
    row = _metadata_row_for_path(
        path,
        metadata_by_path=metadata_by_path,
        metadata_root=metadata_root,
    )

    if row is None:
        height, width = image.shape[:2]
        row = {
            "center_x": float(width * 0.5),
            "center_y": float(height * 0.5),
            "crop_w": float(width),
            "crop_h": float(height),
            "area_proxy": float(width * height),
            "aspect_proxy": float(width / max(height, 1)),
            "frame_w": float(width),
            "frame_h": float(height),
            "border_flag": 0.0,
            "local_density": 0.0,
            "neighbor_count": 0.0,
            "nearest_neighbor_dist": 9999.0,
        }

    return np.asarray([float(row.get(field, 0.0)) for field in TEMPORAL_META_FIELDS], dtype=np.float32)


def _score_clip_difficulty(
    window_paths: list[str],
    *,
    metadata_by_path: dict[str, dict[str, float]],
    metadata_root: str | Path | None,
) -> tuple[float, int, int]:
    if not metadata_by_path:
        return 0.0, 0, -1

    rows: list[dict[str, float]] = []
    frame_indices: list[int] = []
    for path in window_paths:
        row = _metadata_row_for_path(
            path,
            metadata_by_path=metadata_by_path,
            metadata_root=metadata_root,
        )
        if row is None:
            continue
        rows.append(row)
        frame_indices.append(_parse_frame_index(path))

    if not rows:
        return 0.0, 0, -1

    densities = [float(row.get("local_density", 0.0)) for row in rows]
    border_flags = [float(row.get("border_flag", 0.0)) for row in rows]
    crowded_flags = [
        float(row.get("neighbor_count", 0.0) >= 1.0 or row.get("local_density", 0.0) >= 0.20)
        for row in rows
    ]
    nearest_proxy = [
        float(np.clip((36.0 - float(row.get("nearest_neighbor_dist", 9999.0))) / 36.0, 0.0, 1.0))
        for row in rows
    ]
    difficulty = float(
        np.clip(
            0.45 * float(np.mean(densities))
            + 0.25 * float(np.max(densities))
            + 0.15 * float(np.mean(border_flags))
            + 0.10 * float(np.mean(crowded_flags))
            + 0.05 * float(np.mean(nearest_proxy)),
            0.0,
            1.0,
        )
    )
    hard_frame_count = sum(
        1
        for row in rows
        if float(row.get("local_density", 0.0)) >= 0.20
        or float(row.get("border_flag", 0.0)) >= 1.0
        or float(row.get("nearest_neighbor_dist", 9999.0)) <= 32.0
    )
    anchor_idx = max(
        range(len(rows)),
        key=lambda idx: (
            float(rows[idx].get("local_density", 0.0)),
            float(rows[idx].get("neighbor_count", 0.0)),
            float(rows[idx].get("border_flag", 0.0)),
        ),
    )
    anchor_frame = frame_indices[anchor_idx] if anchor_idx < len(frame_indices) else -1
    return difficulty, int(hard_frame_count), int(anchor_frame)


def _build_clip_samples(
    samples: list[tuple[str, int]],
    *,
    clip_len: int,
    min_clip_len: int = 2,
    stride: int | None = None,
    metadata_by_path: dict[str, dict[str, float]] | None = None,
    metadata_root: str | Path | None = None,
    hard_mining: bool = False,
    hard_threshold: float = 0.20,
    hard_oversample: int = 2,
) -> list[ClipSample]:
    grouped: dict[int, list[tuple[int, str]]] = {}
    for path, label in samples:
        grouped.setdefault(label, []).append((_parse_frame_index(path), str(path)))

    clip_samples: list[ClipSample] = []
    metadata_by_path = metadata_by_path or {}
    step = max(1, stride or max(1, clip_len // 2))
    for label, items in grouped.items():
        items = sorted(items, key=lambda item: (item[0], item[1]))
        if len(items) < min_clip_len:
            continue
        if len(items) <= clip_len:
            difficulty, hard_frame_count, anchor_frame = _score_clip_difficulty(
                [path for _, path in items],
                metadata_by_path=metadata_by_path,
                metadata_root=metadata_root,
            )
            clip_sample = ClipSample(
                paths=tuple(path for _, path in items),
                label=label,
                frame_indices=tuple(frame_idx for frame_idx, _ in items),
                anchor_frame=anchor_frame if anchor_frame >= 0 else items[len(items) // 2][0],
                difficulty=difficulty,
                hard_frame_count=hard_frame_count,
                source_kind="hard_window" if hard_frame_count > 0 else "base",
            )
            clip_samples.append(clip_sample)
            if hard_mining and hard_frame_count > 0 and difficulty >= hard_threshold:
                for _ in range(max(int(hard_oversample), 1) - 1):
                    clip_samples.append(
                        ClipSample(
                            paths=clip_sample.paths,
                            label=clip_sample.label,
                            frame_indices=clip_sample.frame_indices,
                            anchor_frame=clip_sample.anchor_frame,
                            difficulty=clip_sample.difficulty,
                            hard_frame_count=clip_sample.hard_frame_count,
                            source_kind="hard_oversample",
                        )
                    )
            continue

        starts = list(range(0, len(items) - clip_len + 1, step))
        if starts[-1] != len(items) - clip_len:
            starts.append(len(items) - clip_len)
        if hard_mining and metadata_by_path:
            hard_starts: set[int] = set()
            for item_idx, (_, path) in enumerate(items):
                row = _metadata_row_for_path(
                    path,
                    metadata_by_path=metadata_by_path,
                    metadata_root=metadata_root,
                )
                if row is None:
                    continue
                if (
                    float(row.get("local_density", 0.0)) >= hard_threshold
                    or float(row.get("border_flag", 0.0)) >= 1.0
                    or float(row.get("nearest_neighbor_dist", 9999.0)) <= 32.0
                ):
                    hard_starts.add(int(np.clip(item_idx - clip_len // 2, 0, len(items) - clip_len)))
            starts = sorted(set(starts) | hard_starts)
        for start in starts:
            window = items[start:start + clip_len]
            window_paths = [path for _, path in window]
            difficulty, hard_frame_count, anchor_frame = _score_clip_difficulty(
                window_paths,
                metadata_by_path=metadata_by_path,
                metadata_root=metadata_root,
            )
            clip_sample = ClipSample(
                paths=tuple(window_paths),
                label=label,
                frame_indices=tuple(frame_idx for frame_idx, _ in window),
                anchor_frame=anchor_frame if anchor_frame >= 0 else window[len(window) // 2][0],
                difficulty=difficulty,
                hard_frame_count=hard_frame_count,
                source_kind="hard_window" if hard_frame_count > 0 else "base",
            )
            clip_samples.append(clip_sample)
            if hard_mining and hard_frame_count > 0 and difficulty >= hard_threshold:
                for _ in range(max(int(hard_oversample), 1) - 1):
                    clip_samples.append(
                        ClipSample(
                            paths=clip_sample.paths,
                            label=clip_sample.label,
                            frame_indices=clip_sample.frame_indices,
                            anchor_frame=clip_sample.anchor_frame,
                            difficulty=clip_sample.difficulty,
                            hard_frame_count=clip_sample.hard_frame_count,
                            source_kind="hard_oversample",
                        )
                    )
    return clip_samples


def describe_reid_layout(
    data_root: str | Path,
    *,
    train_subdir: str = "train",
    val_subdir: str = "val",
) -> str:
    root = Path(data_root)
    return (
        f"Expected ReID dataset at '{root}' with one of these layouts:\n"
        f"  1. {root / train_subdir / '<identity>' / '*.jpg'} and {root / val_subdir / '<identity>' / '*.jpg'}\n"
        f"  2. {root / '<identity>' / '*.jpg'}\n"
        "You can generate this structure from point GT by running src/training/prepare_reid_dataset.py."
    )


def has_reid_layout(
    data_root: str | Path,
    *,
    train_subdir: str = "train",
    val_subdir: str = "val",
) -> bool:
    root = Path(data_root)
    if not root.exists():
        return False

    train_root = root / train_subdir
    val_root = root / val_subdir
    if train_root.exists() or val_root.exists():
        train_ok = any(path.is_dir() for path in train_root.iterdir()) if train_root.exists() else False
        val_ok = any(path.is_dir() for path in val_root.iterdir()) if val_root.exists() else False
        if train_ok or val_ok:
            return True

    return any(path.is_dir() for path in root.iterdir())


def build_reid_splits(
    data_root: str | Path,
    *,
    train_subdir: str = "train",
    val_subdir: str = "val",
    val_split: float = 0.2,
    min_images_per_identity: int = 2,
    seed: int = 42,
) -> tuple[list[tuple[str, int]], list[tuple[str, int]], dict[str, int]]:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Training data root not found: {root}\n{describe_reid_layout(root, train_subdir=train_subdir, val_subdir=val_subdir)}")

    train_root = root / train_subdir
    val_root = root / val_subdir
    rng = random.Random(seed)
    train_samples: list[tuple[str, int]] = []
    val_samples: list[tuple[str, int]] = []

    if train_root.exists() and val_root.exists():
        identity_names = sorted({path.name for path in _list_identity_dirs(train_root)} | {path.name for path in _list_identity_dirs(val_root)})
        class_to_idx = {name: idx for idx, name in enumerate(identity_names)}
        for identity_name, label in class_to_idx.items():
            train_images = _gather_images(train_root / identity_name) if (train_root / identity_name).exists() else []
            val_images = _gather_images(val_root / identity_name) if (val_root / identity_name).exists() else []
            if len(train_images) + len(val_images) < min_images_per_identity:
                continue
            train_samples.extend((str(path), label) for path in train_images)
            val_samples.extend((str(path), label) for path in val_images)
        return train_samples, val_samples, class_to_idx

    identity_dirs = _list_identity_dirs(root)
    class_to_idx = {path.name: idx for idx, path in enumerate(identity_dirs)}
    for identity_dir in identity_dirs:
        images = _gather_images(identity_dir)
        if len(images) < min_images_per_identity:
            continue
        rng.shuffle(images)
        split_idx = max(1, int(round(len(images) * (1.0 - val_split))))
        split_idx = min(split_idx, len(images) - 1) if len(images) > 1 else len(images)
        train_images = images[:split_idx]
        val_images = images[split_idx:] if split_idx < len(images) else images[-1:]
        label = class_to_idx[identity_dir.name]
        train_samples.extend((str(path), label) for path in train_images)
        val_samples.extend((str(path), label) for path in val_images)
    if not train_samples and not val_samples:
        raise RuntimeError(
            f"No identity-organized training images found under: {root}\n"
            f"{describe_reid_layout(root, train_subdir=train_subdir, val_subdir=val_subdir)}"
        )
    return train_samples, val_samples, class_to_idx


def build_reid_clip_splits(
    data_root: str | Path,
    *,
    train_subdir: str = "train",
    val_subdir: str = "val",
    val_split: float = 0.2,
    min_images_per_identity: int = 2,
    seed: int = 42,
    clip_len: int = 6,
    min_clip_len: int = 2,
    clip_stride: int | None = None,
    metadata_root: str | Path | None = None,
    temporal_hard_mining: bool = True,
    temporal_hard_threshold: float = 0.20,
    temporal_hard_oversample: int = 2,
) -> tuple[list[ClipSample], list[ClipSample], dict[str, int]]:
    root = Path(data_root)
    train_samples, val_samples, class_to_idx = build_reid_splits(
        data_root,
        train_subdir=train_subdir,
        val_subdir=val_subdir,
        val_split=val_split,
        min_images_per_identity=min_images_per_identity,
        seed=seed,
    )
    effective_metadata_root = metadata_root if metadata_root is not None else root
    metadata_by_path = _load_temporal_metadata(effective_metadata_root)
    train_clips = _build_clip_samples(
        train_samples,
        clip_len=clip_len,
        min_clip_len=min_clip_len,
        stride=clip_stride,
        metadata_by_path=metadata_by_path,
        metadata_root=effective_metadata_root,
        hard_mining=temporal_hard_mining,
        hard_threshold=temporal_hard_threshold,
        hard_oversample=temporal_hard_oversample,
    )
    val_clips = _build_clip_samples(
        val_samples,
        clip_len=clip_len,
        min_clip_len=min_clip_len,
        stride=clip_stride,
        metadata_by_path=metadata_by_path,
        metadata_root=effective_metadata_root,
        hard_mining=False,
        hard_threshold=temporal_hard_threshold,
        hard_oversample=1,
    )
    return train_clips, val_clips, class_to_idx


class ReIDDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[str, int]],
        transform: Callable | None = None,
        *,
        backend: str = "cnn",
        crop_size: tuple[int, int] = (96, 96),
    ) -> None:
        if Dataset is object:
            raise ImportError("torch is required to build the ReID training dataset.")
        self.samples = samples
        self.transform = transform
        self.backend = backend
        self.crop_size = crop_size
        self.labels = [label for _, label in samples]
        self.label_to_indices: dict[int, list[int]] = {}
        for index, label in enumerate(self.labels):
            self.label_to_indices.setdefault(label, []).append(index)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(path)
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = build_reid_input(img, backend=self.backend, size=self.crop_size)
        if isinstance(img, np.ndarray):
            img = img.astype(np.float32)
        return img, label


class ReIDClipDataset(Dataset):
    def __init__(
        self,
        samples: list[ClipSample],
        transform: Callable | None = None,
        *,
        backend: str = "cnn",
        crop_size: tuple[int, int] = (96, 96),
        clip_len: int = 6,
        metadata_root: str | Path | None = None,
    ) -> None:
        if Dataset is object:
            raise ImportError("torch is required to build the ReID clip dataset.")
        self.samples = samples
        self.transform = transform
        self.backend = backend
        self.crop_size = crop_size
        self.clip_len = clip_len
        self.metadata_root = None if metadata_root is None else Path(metadata_root)
        self.metadata_by_path = _load_temporal_metadata(self.metadata_root)
        self.labels = [sample.label for sample in samples]
        self.sample_anchor_frames = [sample.anchor_frame for sample in samples]
        self.sample_difficulties = [sample.difficulty for sample in samples]
        self.label_to_indices: dict[int, list[int]] = {}
        self.label_to_hard_indices: dict[int, list[int]] = {}
        for index, label in enumerate(self.labels):
            self.label_to_indices.setdefault(label, []).append(index)
            if samples[index].hard_frame_count > 0 or samples[index].difficulty >= 0.20:
                self.label_to_hard_indices.setdefault(label, []).append(index)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        images: list[np.ndarray] = []
        spatial_inputs: list[np.ndarray] = []
        temporal_meta_values: list[np.ndarray] = []
        for path in sample.paths:
            img = cv2.imread(str(path))
            if img is None:
                raise FileNotFoundError(path)
            temporal_meta_values.append(
                _temporal_meta_from_path(
                    path,
                    image=img,
                    metadata_by_path=self.metadata_by_path,
                    metadata_root=self.metadata_root,
                )
            )
            spatial_inputs.append(build_crop_spatial_input(img))
            if self.transform is not None:
                img = self.transform(img)
            else:
                img = build_reid_input(img, backend=self.backend, size=self.crop_size)
            if hasattr(img, "detach"):
                img_array = img.detach().cpu().numpy().astype(np.float32)
            else:
                img_array = np.asarray(img, dtype=np.float32)
            images.append(img_array)

        if not images:
            raise RuntimeError(f"Clip sample has no images for label {sample.label}.")

        image_shape = images[0].shape
        clip = np.zeros((self.clip_len,) + image_shape, dtype=np.float32)
        mask = np.ones((self.clip_len,), dtype=np.float32)
        frame_indices = np.full((self.clip_len,), fill_value=-1, dtype=np.int64)
        spatial_clip = np.zeros((self.clip_len, SPACIAL_INPUT_DIM), dtype=np.float32)
        temporal_meta = np.zeros((self.clip_len, TEMPORAL_META_DIM), dtype=np.float32)

        valid_len = min(len(images), self.clip_len)
        for item_idx in range(valid_len):
            clip[item_idx] = images[item_idx]
            mask[item_idx] = 0.0
            frame_indices[item_idx] = sample.frame_indices[item_idx] if item_idx < len(sample.frame_indices) else -1
            spatial_clip[item_idx] = spatial_inputs[item_idx]
            temporal_meta[item_idx] = temporal_meta_values[item_idx]

        return {
            "images": clip,
            "mask": mask,
            "label": int(sample.label),
            "frame_indices": frame_indices,
            "spatial_inputs": spatial_clip,
            "temporal_meta": temporal_meta,
            "clip_difficulty": np.float32(sample.difficulty),
            "hard_frame_count": np.int64(sample.hard_frame_count),
            "anchor_frame": np.int64(sample.anchor_frame),
        }
