"""Faithful TranSG training loop (port of TranSG-main/TranSG.py Train mode).

Fidelity points vs the official TF1.14 loop:
- class_samp_gen class-balanced resampling (batch_num = total//batch*2 blocks,
  every block contains all classes, batch_per_class = 256//num_classes, random
  re-sample with replacement when a class is exhausted), then ONE global shuffle.
- Per epoch: (1) eval-mode feature pass over the fixed-order train set
  (drop last partial batch) which also draws and stores the per-batch STPR
  masks; (2) eval-mode gallery pass (recomputed every epoch); (3) probe pass +
  Euclidean mAP/Rank-1/5/10; (4) best-model selection by probe Rank-1 with
  epoch 0 excluded, patience counter; (5) GT-mean class prototypes from the
  train features; (6) training steps in fixed order reusing the stored masks,
  optional per-batch Laplacian eigenvector sign flip (rand_flip).
- Adam lr=3.5e-4, batch 256, patience 120 (KGBD/CASIA 60).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from frame_isrt.transg_faithful import TranSGFaithful


def average_precision_score_binary(y_true: np.ndarray) -> float:
    """AP for binary relevance, ordered as given (equivalent to sklearn's
    average_precision_score with scores sorted descending)."""
    hits = y_true.astype(np.float64)
    n_rel = hits.sum()
    if n_rel == 0:
        return float("nan")
    prec_at_hit = hits.cumsum()[hits == 1] / (np.arange(len(hits))[hits == 1] + 1)
    return float(prec_at_hit.sum() / n_rel)


def load_role(data_dir: Path, role: str) -> tuple[np.ndarray, np.ndarray]:
    skel = np.load(data_dir / f"{role}_skeleton.npy").astype(np.float32)
    ids = np.load(data_dir / f"{role}_identity.npy")
    return skel, ids


def class_samp_gen(y: np.ndarray, num_classes: int, batch_size: int) -> np.ndarray:
    """Return resampled train indices, replicating class_samp_gen + shuffle."""
    ids_ = [np.where(y == c)[0].tolist() for c in range(num_classes)]
    total = y.shape[0]
    batch_num = total // batch_size * 2
    batch_per_class = batch_size // num_classes
    all_idx: list[int] = []
    for i in range(batch_num):
        for c in range(num_classes):
            v = ids_[c]
            chunk = v[batch_per_class * i : batch_per_class * (i + 1)]
            if len(chunk) < batch_per_class:
                chunk = np.random.choice(len(v), batch_per_class).tolist()
                chunk = [v[j] for j in chunk]
            all_idx.extend(chunk)
    all_idx = np.array(all_idx)
    return all_idx[np.random.permutation(all_idx.shape[0])]


@torch.no_grad()
def extract_seq_features(model: TranSGFaithful, skel: torch.Tensor, batch_size: int) -> torch.Tensor:
    model.eval()
    feats = []
    n_full = skel.shape[0] // batch_size  # drop last partial batch (official)
    for i in range(n_full):
        batch = skel[i * batch_size : (i + 1) * batch_size]
        feats.append(model.encode(batch)["seq_ftr"])
    return torch.cat(feats, dim=0)


@torch.no_grad()
def extract_train_features_fast(model: TranSGFaithful, skel: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Eval-mode pass in larger chunks (BN uses moving stats in eval, so chunk
    size does not change outputs); returns the same drop-last truncation."""
    model.eval()
    n = (skel.shape[0] // batch_size) * batch_size
    feats = []
    chunk = batch_size * 4
    for i in range(0, n, chunk):
        feats.append(model.encode(skel[i : min(i + chunk, n)])["seq_ftr"])
    return torch.cat(feats, dim=0)


def evaluate(gal_f: np.ndarray, gal_y: np.ndarray, pro_f: np.ndarray, pro_y: np.ndarray):
    a = torch.from_numpy(pro_f)
    b = torch.from_numpy(gal_f)
    m, n = a.size(0), b.size(0)
    dist = torch.pow(a, 2).sum(1, keepdim=True).expand(m, n) + torch.pow(b, 2).sum(1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, a, b.t())
    dist = dist.clamp(min=1e-12).sqrt().numpy()
    indices = np.argsort(dist, axis=1)
    matches = gal_y[indices] == pro_y[:, None]
    aps = []
    for i in range(m):
        y_true = matches[i]
        if not np.any(y_true):
            continue
        aps.append(average_precision_score_binary(y_true))
    map_ = float(np.mean(aps))
    sort = np.argsort(dist, axis=1)
    r1 = float(np.mean([pro_y[i] in gal_y[sort[i, :1]] for i in range(m)]))
    r5 = float(np.mean([pro_y[i] in gal_y[sort[i, :5]] for i in range(m)]))
    r10 = float(np.mean([pro_y[i] in gal_y[sort[i, :10]] for i in range(m)]))
    return map_, r1, r5, r10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=120)
    ap.add_argument("--rand-flip", type=int, default=1)
    ap.add_argument("--isrt", type=int, default=0)
    ap.add_argument("--max-epochs", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.00035)
    ap.add_argument("--resume", action="store_true", help="resume from checkpoint.pt if present")
    ap.add_argument("--ckpt-every", type=int, default=5)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tr_skel_np, tr_ids = load_role(data_dir, "train")
    pro_skel_np, pro_ids = load_role(data_dir, "probe")
    gal_skel_np, gal_ids = load_role(data_dir, "gallery")

    classes = np.unique(tr_ids)
    num_classes = len(classes)
    id2idx = {int(v): i for i, v in enumerate(classes)}
    tr_y = np.array([id2idx[int(v)] for v in tr_ids])
    pro_y = np.array([id2idx[int(v)] for v in pro_ids])
    gal_y = np.array([id2idx[int(v)] for v in gal_ids])

    samp_idx = class_samp_gen(tr_y, num_classes, args.batch_size)
    tr_skel_np = tr_skel_np[samp_idx]
    tr_y = tr_y[samp_idx]

    tr_skel = torch.from_numpy(tr_skel_np).to(device)
    pro_skel = torch.from_numpy(pro_skel_np).to(device)
    gal_skel = torch.from_numpy(gal_skel_np).to(device)

    model = TranSGFaithful(isrt=bool(args.isrt), rand_flip=bool(args.rand_flip)).to(device)
    base_pos_enc = model._laplacian_pos_enc(device)
    model.pos_enc = base_pos_enc.clone()
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    B = args.batch_size
    T, J = 6, 20
    n_batch = tr_skel.shape[0] // B

    best = {"rank1": -1.0, "mAP": 0.0, "rank5": 0.0, "rank10": 0.0, "epoch": -1}
    cur_patience = 0
    history: list[dict] = []
    wall_so_far = 0.0
    start_epoch = 0
    ckpt_path = out_dir / "checkpoint.pt"
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        cur_patience = ckpt["patience"]
        best = ckpt["best"]
        history = ckpt["history"]
        wall_so_far = ckpt["wall_time_sec"]
        np.random.set_state(ckpt["np_rng"])
        torch.set_rng_state(ckpt["torch_rng"])
        if ckpt.get("torch_cuda_rng"):
            torch.cuda.set_rng_state_all(ckpt["torch_cuda_rng"])
        print(
            f"RESUMED from epoch {ckpt['epoch']} (patience {cur_patience}, "
            f"best R1 {best['rank1']:.4f} @ {best['epoch']})",
            flush=True,
        )
    t0 = time.time() - wall_so_far

    for epoch in range(start_epoch, args.max_epochs):
        # (1) train feature pass (eval mode) + per-batch mask draw/store.
        model.eval()
        node_masks: list[torch.Tensor] = []
        seq_masks: list[torch.Tensor] = []
        train_feats = extract_train_features_fast(model, tr_skel, B)
        for i in range(n_batch):
            node_keep = torch.from_numpy(
                np.setdiff1d(np.arange(J), np.random.choice(J, 10, replace=False))
            )
            seq_keep = torch.from_numpy(
                np.sort(np.random.choice(T, T - 2, replace=False))
            )
            node_masks.append(node_keep)
            seq_masks.append(seq_keep)
        train_labels = torch.from_numpy(tr_y[: n_batch * B]).to(device)

        # (2) gallery pass (recomputed every epoch, official).
        gal_feats = extract_train_features_fast(model, gal_skel, B)
        gal_y_used = gal_y[: gal_feats.shape[0]]

        # (3) probe pass + metrics.
        pro_feats = extract_train_features_fast(model, pro_skel, B)
        pro_y_used = pro_y[: pro_feats.shape[0]]
        map_, r1, r5, r10 = evaluate(
            gal_feats.cpu().numpy(), gal_y_used, pro_feats.cpu().numpy(), pro_y_used
        )

        cur_patience += 1
        if epoch > 0 and r1 > best["rank1"]:
            best = {"rank1": r1, "mAP": map_, "rank5": r5, "rank10": r10, "epoch": epoch}
            torch.save(model.state_dict(), out_dir / "best.pt")
            cur_patience = 0

        # (5) GT-mean prototypes from train features.
        class_ftr = torch.stack(
            [train_feats[train_labels == c].mean(dim=0) for c in range(num_classes)]
        )

        # (6) training steps in fixed order with stored masks.
        model.train()
        ep_gpc = ep_recon = 0.0
        for i in range(n_batch):
            batch = tr_skel[i * B : (i + 1) * B]
            labels = train_labels[i * B : (i + 1) * B]
            if args.rand_flip:
                sign = torch.where(
                    torch.rand(base_pos_enc.shape[1], device=device) >= 0.5, 1.0, -1.0
                )
                model.pos_enc.data = base_pos_enc * sign
            total, parts = model.total_loss(
                batch, labels, class_ftr, node_masks[i].to(device), seq_masks[i].to(device)
            )
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            ep_gpc += float(parts["gpc"])
            ep_recon += float(parts["recon"])
        if args.rand_flip:
            model.pos_enc.data = base_pos_enc.clone()

        history.append(
            {
                "epoch": epoch,
                "mAP": map_,
                "rank1": r1,
                "rank5": r5,
                "rank10": r10,
                "gpc": ep_gpc / n_batch,
                "recon": ep_recon / n_batch,
            }
        )
        if epoch % 5 == 0 or cur_patience == 0:
            print(
                f"epoch {epoch} | mAP {map_:.4f} R1 {r1:.4f} R5 {r5:.4f} R10 {r10:.4f} "
                f"| gpc {ep_gpc / n_batch:.4f} recon {ep_recon / n_batch:.4f} "
                f"| best R1 {best['rank1']:.4f} @ {best['epoch']} | pat {cur_patience}",
                flush=True,
            )
        if cur_patience == args.patience:
            break

        if epoch % args.ckpt_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "patience": cur_patience,
                    "best": best,
                    "history": history,
                    "wall_time_sec": round(time.time() - t0, 1),
                    "np_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state(),
                    "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
                ckpt_path,
            )

    # Final: reload best and re-evaluate (official Eval mode reports best.ckpt).
    if best["epoch"] >= 0:
        model.load_state_dict(torch.load(out_dir / "best.pt", weights_only=True))
        gal_feats = extract_train_features_fast(model, gal_skel, B)
        pro_feats = extract_train_features_fast(model, pro_skel, B)
        map_, r1, r5, r10 = evaluate(
            gal_feats.cpu().numpy(),
            gal_y[: gal_feats.shape[0]],
            pro_feats.cpu().numpy(),
            pro_y[: pro_feats.shape[0]],
        )
        final = {"mAP": map_, "rank1": r1, "rank5": r5, "rank10": r10, "epoch": best["epoch"]}
    else:
        final = best

    result = {
        "data_dir": str(data_dir),
        "seed": args.seed,
        "isrt": bool(args.isrt),
        "rand_flip": bool(args.rand_flip),
        "patience": args.patience,
        "parameters": n_params,
        "num_classes": num_classes,
        "train_windows_resampled": int(tr_skel.shape[0]),
        "batches_per_epoch": n_batch,
        "epochs_run": len(history),
        "wall_time_sec": round(time.time() - t0, 1),
        "best_by_probe_rank1": best,
        "final_best_reload": final,
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)
    np.save(out_dir / "history.npy", np.array([list(h.values()) for h in history]))
    print("FINAL", json.dumps(final))


if __name__ == "__main__":
    main()
