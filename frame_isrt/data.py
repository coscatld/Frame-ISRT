from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .model import KGBD_BONES


class SixFrameWindowDataset(Dataset[dict[str, torch.Tensor]]):
    """Read a frozen six-frame, 20-joint protocol directory."""

    def __init__(self, root: str | Path, role: str) -> None:
        self.root = Path(root)
        self.role = str(role)
        if self.role not in {"train", "gallery", "probe"}:
            raise ValueError("role must be train, gallery, or probe")
        self.skeletons = np.load(self.root / f"{role}_skeleton.npy", mmap_mode="r")
        self.identities = np.load(self.root / f"{role}_identity.npy", mmap_mode="r")
        self.recordings = np.load(self.root / f"{role}_recording.npy", mmap_mode="r")
        if self.skeletons.ndim != 4 or self.skeletons.shape[1:] != (6, 20, 3):
            raise ValueError(f"Expected [N,6,20,3], got {self.skeletons.shape}")
        if len(self.identities) != len(self.skeletons) or len(self.recordings) != len(self.skeletons):
            raise ValueError("KGBD metadata length mismatch")
        self.labels = [str(int(value)) for value in self.identities]

    def __len__(self) -> int:
        return int(len(self.skeletons))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "skeleton": torch.from_numpy(np.array(self.skeletons[index], copy=True)),
            "identity": torch.tensor(int(self.identities[index]), dtype=torch.long),
            "recording": torch.tensor(int(self.recordings[index]), dtype=torch.long),
        }


KGBDWindowDataset = SixFrameWindowDataset


class CorruptedKGBDWindowDataset(Dataset[dict[str, torch.Tensor]]):
    """Deterministic query-only corruption wrapper for robustness evaluation."""

    def __init__(
        self,
        base: KGBDWindowDataset,
        corruption: str,
        severity: float,
        *,
        seed: int,
    ) -> None:
        self.base = base
        self.corruption = str(corruption)
        self.severity = float(severity)
        self.seed = int(seed)
        if self.corruption not in {"clean", "gaussian", "missing"}:
            raise ValueError(f"Unknown corruption: {self.corruption}")
        if self.severity < 0.0 or self.severity > 1.0:
            raise ValueError("severity must be in [0, 1]")

    def __len__(self) -> int:
        return len(self.base)

    def _generator(self, index: int) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(self.seed * 1_000_003 + int(index))
        return generator

    @staticmethod
    def _bone_scale(skeleton: torch.Tensor) -> torch.Tensor:
        parent = torch.tensor([edge[0] for edge in KGBD_BONES], dtype=torch.long)
        child = torch.tensor([edge[1] for edge in KGBD_BONES], dtype=torch.long)
        bones = skeleton.index_select(1, child) - skeleton.index_select(1, parent)
        return torch.linalg.vector_norm(bones, dim=-1).median().clamp_min(1e-6)

    @staticmethod
    def _nearest_temporal_fill(skeleton: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        original = skeleton.clone()
        for joint in range(skeleton.shape[1]):
            available = torch.where(~mask[:, joint])[0]
            if len(available) == 0:
                skeleton[:, joint] = original[:, 11]
                continue
            for frame in torch.where(mask[:, joint])[0].tolist():
                nearest = available[(available - frame).abs().argmin()]
                skeleton[frame, joint] = original[nearest, joint]
        return skeleton

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.base[index]
        skeleton = item["skeleton"].clone().float()
        if self.corruption == "gaussian" and self.severity > 0:
            noise = torch.randn(
                skeleton.shape,
                generator=self._generator(index),
                dtype=skeleton.dtype,
            )
            skeleton = skeleton + noise * (self.severity * self._bone_scale(skeleton))
        elif self.corruption == "missing" and self.severity > 0:
            mask = torch.rand(
                skeleton.shape[:2], generator=self._generator(index)
            ) < self.severity
            mask[:, 11] = False
            skeleton = self._nearest_temporal_fill(skeleton, mask)
        return {**item, "skeleton": skeleton}
