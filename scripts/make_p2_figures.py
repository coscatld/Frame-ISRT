"""P2 figures from the 5-seed Frame results (read from the per-seed CSV).

Produces four publication-style figures into ../figures/:
  fig1_forest_pooled.png     pooled arm effect (mean delta + bootstrap 95% CI)
  fig2_cross_arm_paired.png  abs vs norm_matched / random_matched paired scatter
  fig3_gate.png              mean learned residual gate per arm (error bars)
  fig4_protocol_arm_heatmap.png  5-seed mean delta per protocol x arm
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV = str(REPO_ROOT / "results" / "per_seed_frame.csv")
OUT = REPO_ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

PROTOS = ["biwi_s", "biwi_w", "ias_a", "ias_b", "kgbd", "kgbd_dedup"]
PROTO_LABELS = ["BIWI-S", "BIWI-W", "IAS-A", "IAS-B", "KGBD", "KGBD dup-safe"]
MODELS = {
    "gru": {"baseline": "gru", "abs": "isrgru", "norm": "isrgru_norm",
            "normmatch": "isrgru_normmatch", "random": "isrgru_randommatch"},
    "transformer": {"baseline": "transformer", "abs": "isr_transformer",
                    "norm": "isr_transformer_norm",
                    "normmatch": "isr_transformer_normmatch",
                    "random": "isr_transformer_randommatch"},
}
ARMS = ["abs", "norm", "normmatch", "random"]
ARM_LABELS = ["absolute", "normalized", "norm_matched", "random_matched"]
GATES = {"abs": 0.2035, "norm": 0.1957, "normmatch": 0.1981, "random": 0.1266}
GATE_SD = {"abs": 0.0101, "norm": 0.0077, "normmatch": 0.0080, "random": 0.0150}


def load_deltas():
    lut = {}
    with open(CSV, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            lut[(row["protocol"], row["seed"], row["model"])] = float(row["mAP"])
    cells = []  # list of dicts {protocol, backbone, seed, abs, norm, normmatch, random}
    for proto in PROTOS:
        for seed in ("42", "123", "2026", "2027", "2028"):
            for backbone, m in MODELS.items():
                base = lut.get((proto, seed, m["baseline"]))
                if base is None:
                    continue
                d = {"protocol": proto, "backbone": backbone, "seed": int(seed)}
                ok = True
                for arm in ARMS:
                    v = lut.get((proto, seed, m[arm]))
                    if v is None:
                        ok = False
                        break
                    d[arm] = v - base  # CSV already stores mAP in percent
                if ok:
                    cells.append(d)
    return cells


def bootstrap_ci(values, samples=100_000, seed=20260813):
    import random
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(samples):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    return means[int(samples * 0.025)], means[int(samples * 0.975)]


FAMILY = {
    "biwi_s": "BIWI", "biwi_w": "BIWI",
    "ias_a": "IAS", "ias_b": "IAS",
    "kgbd": "KGBD", "kgbd_dedup": "KGBD-dedup",
}


def cluster_ci(cells, arm, samples=100_000, seed=20260813):
    """Cluster block bootstrap 95% CI (resampling unit = dataset family x backbone x seed)."""
    import random
    cl = {}
    for c in cells:
        cl.setdefault((FAMILY[c["protocol"]], c["backbone"], c["seed"]), []).append(c[arm])
    rng = random.Random(seed)
    keys = list(cl.keys())
    n = len(keys)
    means = []
    for _ in range(samples):
        total, cnt = 0.0, 0
        for _ in range(n):
            for v in cl[keys[rng.randrange(n)]]:
                total += v
                cnt += 1
        means.append(total / cnt)
    means.sort()
    return means[int(samples * 0.025)], means[int(samples * 0.975)]


def fig1_forest(cells):
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    rows = []
    for arm, label in zip(ARMS, ARM_LABELS):
        d = [c[arm] for c in cells]
        rows.append((label, statistics.mean(d), *cluster_ci(cells, arm),
                     sum(1 for x in d if x > 0), len(d)))
    ypos = range(len(rows))
    for y, (label, mean, lo, hi, pos, n) in zip(ypos, rows):
        ax.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]], fmt="o", ms=5,
                    color="#1f77b4", ecolor="#1f77b4", elinewidth=1.5,
                    capsize=3)
        sig = " *" if lo > 0 or hi < 0 else ""
        ax.text(hi + 0.03, y, f"{mean:+.2f} [{lo:+.2f}, {hi:+.2f}]{sig}",
                va="center", fontsize=8)
    ax.axvline(0, color="0.3", lw=0.8, ls="--")
    ax.set_yticks(list(ypos))
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("pooled mean ΔmAP (pp), 5 seeds, 40 clusters / 60 cells")
    ax.set_xlim(-1.0, 2.0)
    ax.set_title("Scale-content control arms (cluster bootstrap 95% CI)")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_forest_pooled.png", dpi=300)
    plt.close(fig)


def fig2_cross_arm(cells):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharex=True, sharey=True)
    for ax, (ka, kb, title) in zip(
            axes,
            [("normmatch", "norm_matched", "abs vs norm_matched"),
             ("random", "random_matched", "abs vs random_matched")]):
        xs = [c[ka] for c in cells]
        ys = [c["abs"] for c in cells]
        ax.scatter(xs, ys, s=12, alpha=0.55, color="#1f77b4",
                   edgecolors="none")
        lim = [min(xs + ys) - 1, max(xs + ys) + 1]
        ax.plot(lim, lim, "k--", lw=0.8)
        above = sum(1 for x, y in zip(xs, ys) if y > x)
        ax.text(0.97, 0.05, f"abs above control: {above}/{len(cells)}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f"{kb} ΔmAP (pp)")
        ax.set_ylabel("absolute ΔmAP (pp)")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_cross_arm_paired.png", dpi=300)
    plt.close(fig)


def fig3_gate():
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    labels = ARM_LABELS
    means = [GATES[a] for a in ARMS]
    sds = [GATE_SD[a] for a in ARMS]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    bars = ax.bar(range(len(ARMS)), means, yerr=sds, capsize=4,
                  color=colors, width=0.6, edgecolor="black", linewidth=0.6)
    for i, (m, sd) in enumerate(zip(means, sds)):
        ax.text(i, m + sd + 0.004, f"{m:.3f}", ha="center", fontsize=8)
    ax.axhline(0.2, color="0.3", lw=0.8, ls="--")
    ax.text(3.4, 0.201, "0.20", fontsize=7, va="bottom")
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 0.24)
    ax.set_ylabel("mean learned residual gate σ(g)")
    ax.set_title("Residual gate by arm (5 seeds, 60 cells/arm)")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_gate.png", dpi=300)
    plt.close(fig)


def fig4_heatmap(cells):
    import numpy as np
    means = np.zeros((len(PROTOS), len(ARMS)))
    for i, proto in enumerate(PROTOS):
        for j, arm in enumerate(ARMS):
            d = [c[arm] for c in cells if c["protocol"] == proto]
            means[i, j] = statistics.mean(d) if d else float("nan")
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    im = ax.imshow(means, cmap="RdBu_r", vmin=-2.5, vmax=2.5,
                   aspect="auto")
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels(ARM_LABELS, fontsize=8)
    ax.set_yticks(range(len(PROTOS)))
    ax.set_yticklabels(PROTO_LABELS, fontsize=8)
    for i in range(len(PROTOS)):
        for j in range(len(ARMS)):
            v = means[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8,
                    color="black" if abs(v) < 2.2 else "white")
    ax.set_title("5-seed mean ΔmAP (pp) by protocol × arm")
    fig.colorbar(im, ax=ax, shrink=0.85, label="ΔmAP (pp)")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_protocol_arm_heatmap.png", dpi=300)
    plt.close(fig)


def main():
    cells = load_deltas()
    print(f"cells={len(cells)}")
    for arm, label in zip(ARMS, ARM_LABELS):
        d = [c[arm] for c in cells]
        lo, hi = cluster_ci(cells, arm)
        print(f"{label:14s} mean={statistics.mean(d):+.3f} clusterCI=[{lo:+.3f},{hi:+.3f}] "
              f"pos={sum(1 for x in d if x > 0)}/{len(d)}")
    fig1_forest(cells)
    fig2_cross_arm(cells)
    fig3_gate()
    fig4_heatmap(cells)
    for f in sorted(OUT.glob("*.png")):
        print(f.name, f.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
