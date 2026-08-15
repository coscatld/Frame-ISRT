from __future__ import annotations

from collections import defaultdict
from math import ceil
from typing import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler


class IdentityBalancedBatchSampler(Sampler[list[int]]):
    """PK sampler: sample P identities and K samples for each identity per batch."""

    def __init__(
        self,
        labels: Sequence[str],
        identities_per_batch: int,
        samples_per_identity: int,
        seed: int = 42,
        drop_last: bool = False,
        require_distinct_samples: bool = False,
    ) -> None:
        if identities_per_batch <= 0 or samples_per_identity <= 0:
            raise ValueError("identities_per_batch and samples_per_identity must be positive.")
        self.labels = [str(label) for label in labels]
        self.identities_per_batch = int(identities_per_batch)
        self.samples_per_identity = int(samples_per_identity)
        self.seed = int(seed)
        self.drop_last = drop_last
        self.require_distinct_samples = bool(require_distinct_samples)
        self.indices_by_identity: dict[str, list[int]] = defaultdict(list)
        for index, label in enumerate(self.labels):
            self.indices_by_identity[label].append(index)
        if not self.indices_by_identity:
            raise ValueError("IdentityBalancedBatchSampler received no labels.")
        if self.require_distinct_samples:
            self.identities = sorted(
                identity
                for identity, indices in self.indices_by_identity.items()
                if len(indices) >= self.samples_per_identity
            )
            self.excluded_identities = sorted(set(self.indices_by_identity) - set(self.identities))
            if not self.identities:
                raise ValueError("No identities have enough distinct samples for the requested PK sampler.")
        else:
            self.identities = sorted(self.indices_by_identity)
            self.excluded_identities = []
        self.eligible_sample_count = sum(len(self.indices_by_identity[identity]) for identity in self.identities)
        self.batch_size = self.identities_per_batch * self.samples_per_identity
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        # Preserve reproducibility without replaying the exact same PK batches
        # on every epoch.
        epoch = self.epoch
        self.epoch += 1
        rng = np.random.default_rng(self.seed + epoch)
        n_batches = len(self)
        for _ in range(n_batches):
            replace_ids = len(self.identities) < self.identities_per_batch
            chosen_ids = rng.choice(self.identities, size=self.identities_per_batch, replace=replace_ids)
            batch: list[int] = []
            for identity in chosen_ids:
                pool = self.indices_by_identity[str(identity)]
                replace_samples = len(pool) < self.samples_per_identity
                if self.require_distinct_samples and replace_samples:
                    raise RuntimeError("Distinct-sample PK invariant was violated.")
                chosen = rng.choice(pool, size=self.samples_per_identity, replace=replace_samples)
                batch.extend(int(index) for index in chosen)
            rng.shuffle(batch)
            yield batch

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative.")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        if self.drop_last:
            return max(1, self.eligible_sample_count // self.batch_size)
        return max(1, ceil(self.eligible_sample_count / self.batch_size))


def labels_for_sampler(dataset: object) -> list[str]:
    """Extract identity labels without changing Dataset public output."""

    if hasattr(dataset, "_record_subject_ids"):
        return [str(item) for item in getattr(dataset, "_record_subject_ids")]
    if hasattr(dataset, "num_identities") and hasattr(dataset, "samples_per_identity"):
        num_identities = int(getattr(dataset, "num_identities"))
        samples_per_identity = int(getattr(dataset, "samples_per_identity"))
        return [f"id_{identity:03d}" for identity in range(num_identities) for _ in range(samples_per_identity)]
    labels: list[str] = []
    for index in range(len(dataset)):  # type: ignore[arg-type]
        item = dataset[index]  # type: ignore[index]
        labels.append(str(item.get("subject_id", item.get("identity", item.get("label", index)))))
    return labels
