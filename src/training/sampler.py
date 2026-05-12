from __future__ import annotations

import math
import random

try:
    from torch.utils.data import Sampler
except Exception:  # pragma: no cover
    Sampler = object


class IdentityBalancedSampler(Sampler):
    def __init__(
        self,
        data_source,
        *,
        identities_per_batch: int = 4,
        samples_per_identity: int = 4,
        hard_sample_ratio: float = 0.5,
        seed: int = 42,
    ) -> None:
        if Sampler is object:
            raise ImportError("torch is required to build the identity-balanced sampler.")
        self.data_source = data_source
        self.identities_per_batch = identities_per_batch
        self.samples_per_identity = samples_per_identity
        self.hard_sample_ratio = float(max(0.0, min(1.0, hard_sample_ratio)))
        self.seed = seed
        self.epoch = 0
        self.labels = sorted(getattr(data_source, "label_to_indices", {}).keys())
        self.label_to_hard_indices = getattr(data_source, "label_to_hard_indices", {})
        self.anchor_frames = list(getattr(data_source, "sample_anchor_frames", []))
        self.batch_size = max(1, identities_per_batch * samples_per_identity)
        self.num_batches = max(1, math.ceil(len(data_source) / self.batch_size))

    def _anchor_frame(self, idx: int) -> int:
        if 0 <= idx < len(self.anchor_frames):
            return int(self.anchor_frames[idx])
        return -1

    def _sample_diverse_indices(self, indices: list[int], count: int, rng: random.Random) -> list[int]:
        if count <= 0 or not indices:
            return []
        if len(indices) == 1:
            return [indices[0] for _ in range(count)]
        chosen: list[int] = [rng.choice(indices)]
        while len(chosen) < count:
            best_idx = None
            best_score = float("-inf")
            for candidate in indices:
                if candidate in chosen and len(indices) >= count:
                    continue
                anchor = self._anchor_frame(candidate)
                if anchor < 0:
                    score = rng.random()
                else:
                    score = min(abs(anchor - self._anchor_frame(existing)) for existing in chosen)
                score += 0.01 * rng.random()
                if score > best_score:
                    best_score = score
                    best_idx = candidate
            chosen.append(best_idx if best_idx is not None else rng.choice(indices))
        return chosen

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        batch: list[int] = []
        for _ in range(self.num_batches):
            if not self.labels:
                break
            if len(self.labels) >= self.identities_per_batch:
                chosen_labels = rng.sample(self.labels, self.identities_per_batch)
            else:
                chosen_labels = [rng.choice(self.labels) for _ in range(self.identities_per_batch)]
            for label in chosen_labels:
                indices = self.data_source.label_to_indices[label]
                chosen_indices: list[int] = []
                hard_indices = self.label_to_hard_indices.get(label, [])
                wants_hard = hard_indices and rng.random() < self.hard_sample_ratio
                if wants_hard:
                    chosen_indices.extend(self._sample_diverse_indices(hard_indices, 1, rng))
                remaining = max(self.samples_per_identity - len(chosen_indices), 0)
                pool = indices if remaining > 0 else []
                if remaining > 0:
                    diverse = self._sample_diverse_indices(pool, remaining, rng)
                    chosen_indices.extend(diverse)
                if len(chosen_indices) < self.samples_per_identity:
                    chosen_indices.extend(rng.choices(indices, k=self.samples_per_identity - len(chosen_indices)))
                batch.extend(chosen_indices[: self.samples_per_identity])
        return iter(batch)

    def __len__(self):
        return self.num_batches * self.batch_size
