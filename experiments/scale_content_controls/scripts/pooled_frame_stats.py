"""Pooled paired statistics for the Frame track across protocol x backbone x seed cells.

Pools every (protocol, backbone, seed) cell into one paired observation per arm
(baseline mAP vs ISR arm mAP), then reports:

  * mean Delta_pp + SD over cells
  * 95% bootstrap CI over cells (sampling cells with replacement)
  * paired permutation p-value (random sign-flip of the paired diffs, preserving
    cell structure; two-sided)
  * one-sample Wilcoxon signed-rank p-value over the paired diffs

Seed handling (P1-2): with seeds 42/123/2026 the arms read from the same
sources as the aggregation scripts:
  baseline/absolute -> frozen_matrix picker (frozen Table 1)
  normalized        -> experiments/scale_normalization_ablation/results
  norm_matched/random_matched -> this experiment's results/
With the two extra seeds (2027/2028) all arms -- including the baseline and the
absolute/normalized arms -- are read from the P1-2 seed-extension directory
(experiments/seed_extension/results), which re-trains every model
under a config copy whose only difference is the extended seed list. Cells
whose extension result is missing are skipped, so the script degrades to the
3-seed pool until the extension run completes.

Usage:
  python3 \
    experiments/scale_content_controls/scripts/pooled_frame_stats.py \
    [--arms abs,norm,normmatch,random] [--out <dir>]
"""
from __future__ import annotations

import argparse
import json
import os
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from frozen_matrix import (  # noqa: E402
    ALLOWED_SHA,
    PREFERRED_SOURCE,
    ROOT as FROZEN_ROOT,
    SCAN_ROOTS,
)

EXP = Path(os.environ.get("FRAME_ISRT_SCALE_CONTROLS_ROOT", str(Path(__file__).resolve().parents[3] / "experiments" / "scale_content_controls")))
NEW_ROOT = EXP / "results"
NORM_ROOT = Path(
    os.environ.get("FRAME_ISRT_NORM_ROOT", str(Path(__file__).resolve().parents[3] / "experiments" / "scale_normalization_ablation" / "results"))
)
SEED_EXTEND_ROOT = Path(
    os.environ.get("FRAME_ISRT_SEED_EXTEND_ROOT", str(Path(__file__).resolve().parents[3] / "experiments" / "seed_extension" / "results"))
)
PROTOS = ["biwi_s", "biwi_w", "ias_a", "ias_b", "kgbd", "kgbd_dedup"]
BACKBONES = [
    ("gru", "isrgru", "isrgru_norm", "isrgru_normmatch", "isrgru_randommatch"),
    ("transformer", "isr_transformer", "isr_transformer_norm",
     "isr_transformer_normmatch", "isr_transformer_randommatch"),
]
OLD_SEEDS = [42, 123, 2026]
NEW_SEEDS = [2027, 2028]
SEEDS = OLD_SEEDS + NEW_SEEDS
ARMS = {
    "abs": (1,),
    "norm": (2,),
    "normmatch": (3,),
    "random": (4,),
}
SHA_TO_LABEL = {sha: label for label, shas in ALLOWED_SHA.items() for sha in shas}


def pick_frozen_cell(label, model, seed):
    cands = []
    for scan in SCAN_ROOTS:
        scan_root = FROZEN_ROOT / scan
        if not scan_root.exists():
            continue
        for rp in scan_root.rglob("result.json"):
            if "_d64_" in str(rp):
                continue
            try:
                payload = json.loads(rp.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            if payload.get("model") != model or payload.get("seed") != seed:
                continue
            if SHA_TO_LABEL.get((payload.get("protocol_sha256") or "").lower()[:12]) != label:
                continue
            mAP = (payload.get("window") or {}).get("mAP")
            if mAP is None:
                continue
            cands.append({"mAP": mAP, "path": str(rp.relative_to(FROZEN_ROOT))})
    preferred = PREFERRED_SOURCE.get((label, model))
    if preferred:
        match = [c for c in cands if preferred in c["path"]]
        if match:
            return match[0]
    return sorted(cands, key=lambda c: c["path"])[-1]


def load_new_cell(root, proto, model, seed):
    rp = root / proto / model / f"seed{seed}" / "result.json"
    if not rp.exists():
        return None
    payload = json.loads(rp.read_text(encoding="utf-8"))
    return (payload.get("window") or {}).get("mAP")


def permutation_pvalue(diffs, n_perms=100_000, seed=20260813):
    """Two-sided p-value of mean(diffs) under random sign-flip permutation."""
    rng = __import__("random").Random(seed)
    diffs = [float(d) for d in diffs]
    n = len(diffs)
    obs = statistics.mean(diffs)
    if n == 0:
        return float("nan")
    if all(d == 0 for d in diffs):
        return 1.0
    # sign-flip: each pair flips with p=0.5 -> equivalent to permutation on the
    # sign of the paired difference (valid for paired data).
    count = 0
    for _ in range(n_perms):
        total = 0.0
        for d in diffs:
            total += d if rng.getrandbits(1) else -d
        count += 1 if total / n >= abs(obs) else 0
    return 2.0 * (count / n_perms)


def wilcoxon_pvalue(diffs, seed=20260813):
    """One-sample signed-rank test on nonzero diffs (normal approx, tie-corrected)."""
    rng = __import__("random").Random(seed + 1)
    diffs = [float(d) for d in diffs if d != 0]
    n = len(diffs)
    if n == 0:
        return float("nan")
    ranks = {}
    order = sorted(enumerate(diffs), key=lambda iv: abs(iv[1]))
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(order[j + 1][1]) == abs(order[i][1]):
            j += 1
        avg = sum(k + 1 for k in range(i, j + 1)) / (j - i + 1)
        for k in range(i, j + 1):
            ranks[order[k][0]] = avg
        i = j + 1
    w_plus = sum(ranks[k] for k, d in enumerate(diffs) if d > 0)
    mu = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24
    # tie correction
    diffs_abs = sorted(abs(d) for d in diffs)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and diffs_abs[j + 1] == diffs_abs[i]:
            j += 1
        t = j - i + 1
        if t > 1:
            var -= (t ** 3 - t) / 48
        i = j + 1
    z = (w_plus - mu) / math.sqrt(var)
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2))))


def bootstrap_ci(diffs, samples=100_000, seed=20260813, alpha=0.05):
    import random
    rng = random.Random(seed)
    n = len(diffs)
    if n == 0:
        return float("nan"), float("nan")
    means = []
    for _ in range(samples):
        total = 0.0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    lo = means[int(samples * alpha / 2)]
    hi = means[int(samples * (1 - alpha / 2))]
    return lo, hi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default="abs,norm,normmatch,random")
    parser.add_argument("--out", type=Path, default=EXP / "aggregate")
    args = parser.parse_args()
    want = [a.strip() for a in args.arms.split(",") if a.strip()]

    cells = {arm: [] for arm in want}  # arm -> list of (proto, backbone, seed, delta_pp)
    for proto in PROTOS:
        for row in BACKBONES:
            base_name = row[0]
            backbone = base_name
            for seed in SEEDS:
                if seed in OLD_SEEDS:
                    base = pick_frozen_cell(proto, base_name, seed)
                    if not base:
                        continue
                    b = base["mAP"] * 100
                else:
                    # new seeds: baseline re-run lives in the seed-extension dir
                    cell = load_new_cell(SEED_EXTEND_ROOT, proto, base_name, seed)
                    if cell is None:
                        continue
                    b = cell * 100
                for arm in want:
                    idx = ARMS[arm][0]
                    model = row[idx]
                    if seed in OLD_SEEDS:
                        if idx == 1:
                            cell = pick_frozen_cell(proto, model, seed)
                            m = cell["mAP"] * 100 if cell else float("nan")
                        elif idx == 2:
                            cell = load_new_cell(NORM_ROOT, proto, model, seed)
                            m = cell * 100 if cell is not None else float("nan")
                        else:
                            cell = load_new_cell(NEW_ROOT, proto, model, seed)
                            m = cell * 100 if cell is not None else float("nan")
                    else:
                        cell = load_new_cell(SEED_EXTEND_ROOT, proto, model, seed)
                        m = cell * 100 if cell is not None else float("nan")
                    if math.isfinite(m):
                        cells[arm].append((proto, backbone, seed, m - b))

    out = []
    header = ("arm | n_cells | mean_D_pp | sd_pp | 95%_CI_pp | permutation_p | "
              "wilcoxon_p | positive_cells")
    out.append(header)
    out.append("|---|---|---|---|---|---|---|---|")
    for arm in want:
        rows = cells[arm]
        diffs = [r[3] for r in rows]
        n = len(diffs)
        if n == 0:
            out.append(f"| {arm} | 0 | -- | -- | -- | -- | -- | -- |")
            continue
        mean_d = statistics.mean(diffs)
        sd = statistics.stdev(diffs) if n > 1 else 0.0
        lo, hi = bootstrap_ci(diffs)
        perm_p = permutation_pvalue(diffs)
        w_p = wilcoxon_pvalue(diffs)
        pos = sum(1 for d in diffs if d > 0)
        out.append(
            f"| {arm} | {n} | {mean_d:+.2f} | {sd:.2f} | [{lo:+.2f}, {hi:+.2f}] | "
            f"{perm_p:.4f} | {w_p:.4f} | {pos}/{n} |"
        )
        print(f"arm={arm:10s} n={n} mean={mean_d:+.3f}pp sd={sd:.3f} "
              f"CI95=[{lo:+.3f},{hi:+.3f}] perm_p={perm_p:.4f} wilcoxon_p={w_p:.4f} "
              f"pos={pos}/{n}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "pooled_frame_stats.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
