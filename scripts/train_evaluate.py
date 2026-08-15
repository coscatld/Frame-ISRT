from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from frame_isrt.samplers import IdentityBalancedBatchSampler
from frame_isrt.arcface import ArcFaceLoss
from frame_isrt.metric import TripletLoss
from frame_isrt.data import SixFrameWindowDataset
from frame_isrt.evaluation import (
    aggregate_recordings,
    extract_embeddings,
    retrieval_metrics,
)
from frame_isrt.model import build_mva_model


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    """Write a file atomically (temp + rename) so a crash never leaves a partial
    file that downstream tooling mistakes for a complete artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def git_head(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError:
        pointer = root / ".git"
        if not pointer.is_file():
            raise
        value = pointer.read_text(encoding="utf-8").strip()
        if not value.startswith("gitdir: "):
            raise RuntimeError(f"Malformed linked-worktree pointer: {pointer}")
        git_dir_text = value.removeprefix("gitdir: ").replace("\\", "/")
        if len(git_dir_text) >= 3 and git_dir_text[1:3] == ":/":
            git_dir = Path("/mnt") / git_dir_text[0].lower() / git_dir_text[3:]
        else:
            git_dir = Path(git_dir_text)
        head_value = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head_value.startswith("ref: "):
            return head_value
        reference = head_value.removeprefix("ref: ")
        common_dir = (git_dir / (git_dir / "commondir").read_text(encoding="utf-8").strip()).resolve()
        loose_reference = common_dir / reference
        if loose_reference.exists():
            return loose_reference.read_text(encoding="utf-8").strip()
        for line in (common_dir / "packed-refs").read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#") and line.split(" ", 1)[-1] == reference:
                return line.split(" ", 1)[0]
        raise RuntimeError(f"Unable to resolve Git reference {reference}")


def compute_norm_match_factor(
    train_set, model, device: torch.device, num_workers: int
) -> float:
    """Protocol-level RMS ratio mean_RMS(C)/mean_RMS(N) over the training set.

    Used only by the norm_matched residual arm: the normalized residual is
    re-scaled by this factor so its global RMS/variance matches the absolute
    (root-centered) input, isolating per-sample scale content from magnitude.
    """
    loader = DataLoader(
        train_set,
        batch_size=128,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    builder = model.builder
    sum_abs = 0.0
    sum_norm = 0.0
    count = 0
    with torch.no_grad():
        for batch in loader:
            skeleton = batch["skeleton"].to(device, non_blocking=True)
            normalized, _, _, _ = builder(skeleton)
            centered = builder.root_center(skeleton)
            sum_abs += float(centered.flatten(1).norm(dim=1).sum())
            sum_norm += float(normalized.flatten(1).norm(dim=1).sum())
            count += len(skeleton)
    if count == 0:
        raise RuntimeError("Empty training set for norm_match factor")
    return (sum_abs / count) / (sum_norm / count)


def validate_protocol_overlaps(
    protocol: dict[str, object], data_config: dict[str, object]
) -> tuple[dict[str, int], bool]:
    overlaps = protocol["raw_payload_role_overlaps"]
    counts = {str(key): len(value) for key, value in overlaps.items()}
    allowed = bool(data_config.get("allow_protocol_payload_overlap", False))
    if any(counts.values()) and not allowed:
        raise RuntimeError("Protocol payload leakage")
    if allowed:
        expected = {
            str(key): int(value)
            for key, value in data_config.get(
                "expected_payload_overlap_counts", {}
            ).items()
        }
        if counts != expected:
            raise RuntimeError(
                f"Disclosed protocol overlap counts changed: {counts} != {expected}"
            )
    return counts, allowed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    protocol_path = Path(config["data"]["protocol"])
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    overlap_counts, overlap_allowed = validate_protocol_overlaps(
        protocol, config["data"]
    )
    if int(protocol["data_split_seed"]) != int(config["data"]["split_seed"]):
        raise RuntimeError("Protocol/config split mismatch")
    if args.seed not in [int(value) for value in config["seeds"]]:
        raise ValueError("Seed is outside frozen confirmation set")
    if not torch.cuda.is_available():
        raise RuntimeError("Confirmation training requires CUDA")
    seed_everything(args.seed)
    device = torch.device("cuda")
    root = Path(config["data"]["root"])
    train_set = SixFrameWindowDataset(root, "train")
    gallery_set = SixFrameWindowDataset(root, "gallery")
    probe_set = SixFrameWindowDataset(root, "probe")
    training = config["training"]
    sampler = IdentityBalancedBatchSampler(
        train_set.labels,
        identities_per_batch=int(training["identities_per_batch"]),
        samples_per_identity=int(training["samples_per_identity"]),
        seed=args.seed,
        require_distinct_samples=True,
    )
    train_loader = DataLoader(
        train_set,
        batch_sampler=sampler,
        num_workers=int(training["num_workers"]),
        pin_memory=True,
        persistent_workers=int(training["num_workers"]) > 0,
    )
    model_layout = str(config["model"].get("layout", "kgbd_reindexed20"))
    model = build_mva_model(
        args.model,
        embedding_dim=int(config["model"]["embedding_dim"]),
        dropout=float(config["model"]["dropout"]),
        layout=model_layout,
        mamba_d_state=int(config["model"].get("mamba_d_state", 16)),
    ).to(device)
    if getattr(model, "residual_input_type", None) == "norm_matched":
        factor = compute_norm_match_factor(
            train_set, model, device, int(training["num_workers"])
        )
        model.residual_energy_factor.fill_(factor)
        print(
            f"{args.model} norm_match factor={factor:.6f} "
            f"(mean_RMS(C)/mean_RMS(N), {len(train_set)} train samples)",
            flush=True,
        )
    atomic_write_text(
        args.output_dir / "config.json",
        json.dumps(
            {
                "model": args.model,
                "model_layout": model_layout,
                "seed": args.seed,
                "config_path": str(args.config),
                "config_sha256": sha256_file(args.config),
                "protocol_sha256": sha256_file(protocol_path),
                "residual_input_type": getattr(model, "residual_input_type", None),
                "residual_energy_factor": (
                    float(model.residual_energy_factor.item())
                    if hasattr(model, "residual_energy_factor")
                    else None
                ),
                "selection_policy": "fixed_final_epoch",
                "probe_gallery_used_for_selection": False,
                "epochs": int(training["epochs"]),
            },
            indent=2,
            ensure_ascii=False,
        ),
    )
    arcface = ArcFaceLoss(
        int(config["data"].get("num_classes", len(set(train_set.labels)))),
        int(config["model"]["embedding_dim"]),
        scale=float(training["arcface_scale"]),
        margin=float(training["arcface_margin"]),
    ).to(device)
    triplet = TripletLoss(margin=float(training["triplet_margin"]))
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(arcface.parameters()),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    epochs = int(training["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    started = time.perf_counter()
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        samples = 0
        for batch in train_loader:
            skeleton = batch["skeleton"].to(device, non_blocking=True)
            labels = batch["identity"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            embeddings = model(skeleton)
            if hasattr(model, "training_objective"):
                loss, _ = model.training_objective(labels, arcface.weight)
            else:
                arc_loss, _ = arcface(embeddings, labels)
                tri_loss = triplet(embeddings, labels)
                loss = arc_loss + float(training["triplet_weight"]) * tri_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
            samples += len(labels)
        scheduler.step()
        print(f"{args.model} seed={args.seed} epoch={epoch + 1}/{epochs} loss={total_loss / samples:.6f}", flush=True)

    evaluation = config["evaluation"]
    loaders = [
        DataLoader(
            dataset,
            batch_size=int(evaluation["batch_size"]),
            shuffle=False,
            num_workers=int(training["num_workers"]),
            pin_memory=True,
            persistent_workers=int(training["num_workers"]) > 0,
        )
        for dataset in (gallery_set, probe_set)
    ]
    gallery_embeddings, gallery_labels, gallery_recordings = extract_embeddings(model, loaders[0], device)
    probe_embeddings, probe_labels, probe_recordings = extract_embeddings(model, loaders[1], device)
    window = retrieval_metrics(
        gallery_embeddings,
        gallery_labels,
        probe_embeddings,
        probe_labels,
        device=device,
        chunk_size=int(evaluation["chunk_size"]),
    )
    recording = None
    if bool(evaluation.get("recording_metrics", True)):
        gallery_record_embeddings, gallery_record_labels = aggregate_recordings(
            gallery_embeddings, gallery_labels, gallery_recordings
        )
        probe_record_embeddings, probe_record_labels = aggregate_recordings(
            probe_embeddings, probe_labels, probe_recordings
        )
        recording = retrieval_metrics(
            gallery_record_embeddings,
            gallery_record_labels,
            probe_record_embeddings,
            probe_record_labels,
            device=device,
            chunk_size=int(evaluation["chunk_size"]),
        )
    project_root = Path(__file__).resolve().parents[1]
    checkpoint = args.output_dir / "final_epoch.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_name": args.model,
            "model_layout": model_layout,
            "seed": args.seed,
            "epoch": epochs,
            "config_sha256": sha256_file(args.config),
            "protocol_sha256": sha256_file(protocol_path),
        },
        checkpoint,
    )
    result = {
        "model": args.model,
        "model_layout": model_layout,
        "seed": args.seed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "epochs": epochs,
        "selection_policy": "fixed_final_epoch",
        "training_objective": (
            "transg_gpc_stpr_reproduction"
            if hasattr(model, "training_objective")
            else "arcface_plus_triplet"
        ),
        "probe_gallery_used_for_selection": False,
        "protocol_payload_overlap_counts": overlap_counts,
        "protocol_payload_overlap_allowed": overlap_allowed,
        "training_seconds": time.perf_counter() - started,
        "window": window,
        "recording": recording,
        "residual_gate": (
            float(torch.sigmoid(model.residual_gate.detach()).cpu())
            if hasattr(model, "residual_gate")
            else None
        ),
        "residual_input_type": getattr(model, "residual_input_type", None),
        "residual_energy_factor": (
            float(model.residual_energy_factor.item())
            if hasattr(model, "residual_energy_factor")
            else None
        ),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config_sha256": sha256_file(args.config),
        "protocol_sha256": sha256_file(protocol_path),
        "git_head": git_head(project_root),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    atomic_write_text(
        args.output_dir / "result.json",
        json.dumps(result, indent=2, ensure_ascii=False),
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
