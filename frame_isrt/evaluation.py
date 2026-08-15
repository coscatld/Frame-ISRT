from __future__ import annotations

from collections import defaultdict

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    embeddings: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    recordings: list[torch.Tensor] = []
    for batch in loader:
        embeddings.append(F.normalize(model(batch["skeleton"].to(device, non_blocking=True)), dim=-1).cpu())
        labels.append(batch["identity"])
        recordings.append(batch["recording"])
    return torch.cat(embeddings), torch.cat(labels), torch.cat(recordings)


@torch.no_grad()
def retrieval_metrics(
    gallery: torch.Tensor,
    gallery_labels: torch.Tensor,
    query: torch.Tensor,
    query_labels: torch.Tensor,
    *,
    device: torch.device,
    chunk_size: int,
) -> dict[str, float | int]:
    gallery = F.normalize(gallery.float(), dim=-1).to(device)
    gallery_labels_device = gallery_labels.long().to(device)
    query = F.normalize(query.float(), dim=-1)
    ap_parts: list[torch.Tensor] = []
    rank_hits = {1: 0, 5: 0, 10: 0}
    total = 0
    for start in range(0, len(query), chunk_size):
        current = query[start : start + chunk_size].to(device)
        labels = query_labels[start : start + chunk_size].long().to(device)
        order = torch.argsort(current @ gallery.T, dim=1, descending=True)
        matches = gallery_labels_device[order].eq(labels[:, None])
        positives = matches.sum(dim=1)
        if torch.any(positives == 0):
            raise RuntimeError("Query without gallery positive")
        cumulative = matches.cumsum(dim=1)
        positions = torch.arange(1, matches.shape[1] + 1, device=device, dtype=torch.float32)
        ap_parts.append((((cumulative.float() / positions[None]) * matches).sum(dim=1) / positives).cpu())
        for rank in rank_hits:
            rank_hits[rank] += int(matches[:, :rank].any(dim=1).sum())
        total += len(labels)
    return {
        "mAP": float(torch.cat(ap_parts).mean()),
        "rank1": rank_hits[1] / total,
        "rank5": rank_hits[5] / total,
        "rank10": rank_hits[10] / total,
        "queries": total,
        "gallery": int(len(gallery_labels)),
    }


@torch.no_grad()
def retrieval_per_query(
    gallery: torch.Tensor,
    gallery_labels: torch.Tensor,
    query: torch.Tensor,
    query_labels: torch.Tensor,
    *,
    device: torch.device,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return AP and Rank-1 hit for each query without changing retrieval rules."""
    gallery = F.normalize(gallery.float(), dim=-1).to(device)
    gallery_labels_device = gallery_labels.long().to(device)
    query = F.normalize(query.float(), dim=-1)
    average_precision: list[torch.Tensor] = []
    rank1: list[torch.Tensor] = []
    for start in range(0, len(query), chunk_size):
        current = query[start : start + chunk_size].to(device)
        labels = query_labels[start : start + chunk_size].long().to(device)
        order = torch.argsort(current @ gallery.T, dim=1, descending=True)
        matches = gallery_labels_device[order].eq(labels[:, None])
        positives = matches.sum(dim=1)
        if torch.any(positives == 0):
            raise RuntimeError("Query without gallery positive")
        cumulative = matches.cumsum(dim=1)
        positions = torch.arange(1, matches.shape[1] + 1, device=device, dtype=torch.float32)
        average_precision.append(
            (((cumulative.float() / positions[None]) * matches).sum(dim=1) / positives).cpu()
        )
        rank1.append(matches[:, 0].float().cpu())
    return torch.cat(average_precision), torch.cat(rank1)


def aggregate_recordings(
    embeddings: torch.Tensor, labels: torch.Tensor, recordings: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    groups: dict[int, list[int]] = defaultdict(list)
    for index, recording in enumerate(recordings.tolist()):
        groups[int(recording)].append(index)
    output_embeddings: list[torch.Tensor] = []
    output_labels: list[int] = []
    for recording in sorted(groups):
        indices = torch.tensor(groups[recording], dtype=torch.long)
        group_labels = labels.index_select(0, indices)
        if not torch.all(group_labels == group_labels[0]):
            raise ValueError("Recording spans identities")
        output_embeddings.append(F.normalize(embeddings.index_select(0, indices).mean(dim=0), dim=0))
        output_labels.append(int(group_labels[0]))
    return torch.stack(output_embeddings), torch.tensor(output_labels)
