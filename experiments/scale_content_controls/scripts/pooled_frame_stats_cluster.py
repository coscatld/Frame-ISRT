"""Cluster-aware pooled statistics for the Frame scale-content controls.

Reviewer point: the 60 cells in §4.5.1 are NOT independent observations.
BIWI-S/BIWI-W and IAS-A/IAS-B share byte-identical training windows AND
training checkpoints (gallery/probe roles swapped); KGBD official and KGBD
duplicate-safe are separate train sets. The valid resampling unit is therefore
the independent training run = (dataset family, backbone, seed), giving
4 families x 2 backbones x 5 seeds = 40 clusters, with 60 cells nested inside.

This script keeps the cell-level point estimate (mean Delta_pp over the 60
cells, identical to pooled_frame_stats.py) but reports cluster-aware inference:

  * cluster block bootstrap 95% CI  (resample the 40 clusters with replacement)
  * cluster sign-flip permutation p (flip every delta inside a cluster together)
  * cluster-mean one-sample t-test p with explicit df = n_clusters - 1 = 39
    (each cluster summarized by its mean Delta_pp)

Same structure for the cross-arm paired deltas (abs - norm_matched,
abs - random_matched, abs - normalized).

Data source: docs/per_seed_results_frame.csv (all arms, all 5 seeds), so the
point estimates match the reported pooled numbers exactly.

Usage:
  python3 \
    experiments/scale_content_controls/scripts/pooled_frame_stats_cluster.py
"""
from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CSV_PATH = REPO_ROOT / "results" / "per_seed_frame.csv"
OUT_PATH = Path(
    REPO_ROOT / "experiments" / "scale_content_controls" / "aggregate"
) / "pooled_frame_stats_cluster.md"

FAMILY = {
    "biwi_s": "BIWI", "biwi_w": "BIWI",
    "ias_a": "IAS", "ias_b": "IAS",
    "kgbd": "KGBD",
    "kgbd_dedup": "KGBD-dedup",
}
ARMS = ["abs", "norm", "normmatch", "random"]
MODEL_BACKBONE = {
    "gru": "gru", "transformer": "transformer",
    "isrgru": "gru", "isr_transformer": "transformer",
    "isrgru_norm": "gru", "isr_transformer_norm": "transformer",
    "isrgru_normmatch": "gru", "isr_transformer_normmatch": "transformer",
    "isrgru_randommatch": "gru", "isr_transformer_randommatch": "transformer",
}
BASELINE_MODEL = {"gru": "gru", "transformer": "transformer"}
N_CLUST = 40
N_PERMS = 100_000
N_BOOT = 100_000
SEED = 20260815


def load_cells():
    """Return list of dicts: protocol, family, backbone, seed, delta per arm (pp)."""
    cells = {}
    with open(CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["protocol"], row["model"], int(row["seed"]))
            cells[key] = {"mAP": float(row["mAP"])}
    out = []
    # group by (protocol, backbone, seed); find baseline + each arm
    by_cell = {}
    for (protocol, model, seed), rec in cells.items():
        bb = MODEL_BACKBONE.get(model)
        if bb is None:
            continue
        k = (protocol, bb, seed)
        by_cell.setdefault(k, {})[model] = rec["mAP"]
    for (protocol, bb, seed), m in by_cell.items():
        base = m.get(BASELINE_MODEL[bb])
        if base is None:
            continue
        d = {arm: m.get(arm_model(bb, arm)) - base for arm in ARMS}
        d = {a: v for a, v in d.items() if v is not None}
        if len(d) == len(ARMS):
            out.append({
                "protocol": protocol,
                "family": FAMILY[protocol],
                "backbone": bb,
                "seed": seed,
                "deltas": d,
            })
    return out


def arm_model(backbone, arm):
    return {
        ("gru", "abs"): "isrgru",
        ("gru", "norm"): "isrgru_norm",
        ("gru", "normmatch"): "isrgru_normmatch",
        ("gru", "random"): "isrgru_randommatch",
        ("transformer", "abs"): "isr_transformer",
        ("transformer", "norm"): "isr_transformer_norm",
        ("transformer", "normmatch"): "isr_transformer_normmatch",
        ("transformer", "random"): "isr_transformer_randommatch",
    }[(backbone, arm)]


def clusters_for(cells, arm):
    """dict cluster_key -> list of cell deltas (pp)."""
    cl = {}
    for c in cells:
        cl.setdefault((c["family"], c["backbone"], c["seed"]), []).append(c["deltas"][arm])
    return cl


def cluster_block_bootstrap_ci(clusters, samples=N_BOOT, seed=SEED, alpha=0.05):
    import random
    rng = random.Random(seed)
    keys = list(clusters.keys())
    n = len(keys)
    if n == 0:
        return float("nan"), float("nan")
    means = []
    for _ in range(samples):
        total, count = 0.0, 0
        for _ in range(n):
            for d in clusters[keys[rng.randrange(n)]]:
                total += d
                count += 1
        means.append(total / count)
    means.sort()
    return means[int(samples * alpha / 2)], means[int(samples * (1 - alpha / 2))]


def cluster_permutation_p(clusters, n_perms=N_PERMS, seed=SEED):
    """Cluster sign-flip permutation: flip all deltas inside a cluster together."""
    import random
    rng = random.Random(seed)
    keys = list(clusters.keys())
    obs_vals = [d for k in keys for d in clusters[k]]
    obs = statistics.mean(obs_vals)
    if all(v == 0 for v in obs_vals):
        return 1.0
    count = 0
    for _ in range(n_perms):
        total, cnt = 0.0, 0
        for k in keys:
            flip = rng.getrandbits(1)
            for d in clusters[k]:
                total += -d if flip else d
                cnt += 1
        if abs(total / cnt) >= abs(obs):
            count += 1
    return (1 + count) / (1 + n_perms)


def cluster_mean_t_p(clusters):
    """One-sample t-test on cluster means vs 0; df = n_clusters - 1."""
    means = [statistics.mean(v) for v in clusters.values()]
    n = len(means)
    if n < 2:
        return float("nan"), n - 1
    m = statistics.mean(means)
    sd = statistics.stdev(means)
    t = m / (sd / math.sqrt(n))
    # two-sided p for Student's t with n-1 df
    from math import gamma as _gam
    df = n - 1
    # regularized incomplete beta / student t survival via scipy not guaranteed
    # -> use the scipy.stats.t that ships in this env.
    from scipy.stats import t as _t  # noqa: PLC0415
    return float(_t.sf(abs(t), df) * 2.0), df


def report_arm(name, cells, clusters, out):
    vals = [d for c in clusters.values() for d in c]
    n_cells = len(vals)
    n_clust = len(clusters)
    mean = statistics.mean(vals)
    sd = statistics.stdev(vals)
    lo, hi = cluster_block_bootstrap_ci(clusters)
    p_perm = cluster_permutation_p(clusters)
    p_t, df = cluster_mean_t_p(clusters)
    pos = sum(1 for v in vals if v > 0)
    row = (
        f"| {name} | {n_cells} | {n_clust} | {mean:+.2f} | {sd:.2f} | "
        f"[{lo:+.2f}, {hi:+.2f}] | {p_perm:.4f} | {p_t:.4f} (df={df}) | {pos}/{n_cells} |"
    )
    out.append(row)
    print(f"arm={name:10s} cells={n_cells} clusters={n_clust} "
          f"mean={mean:+.3f}pp sd={sd:.3f} clusterCI95=[{lo:+.3f},{hi:+.3f}] "
          f"clusterPerm_p={p_perm:.4f} clustert_p={p_t:.4f}(df={df}) pos={pos}/{n_cells}")
    return mean


def report_paired(control_arm, cells, out):
    # per-cell paired diff (abs − control), clustered by the same cluster key
    paired = {}
    for c in cells:
        key = (c["family"], c["backbone"], c["seed"])
        da = c["deltas"].get("abs")
        db = c["deltas"].get(control_arm)
        if da is None or db is None:
            continue
        paired.setdefault(key, []).append(da - db)
    vals = [d for v in paired.values() for d in v]
    n_cells, n_clust = len(vals), len(paired)
    mean = statistics.mean(vals)
    lo, hi = cluster_block_bootstrap_ci(paired)
    p_perm = cluster_permutation_p(paired)
    p_t, df = cluster_mean_t_p(paired)
    pos = sum(1 for v in vals if v > 0)
    row = (
        f"| abs−{control_arm} | {n_cells} | {n_clust} | {mean:+.2f} | -- | "
        f"[{lo:+.2f}, {hi:+.2f}] | {p_perm:.4f} | {p_t:.4f} (df={df}) | {pos}/{n_cells} |"
    )
    out.append(row)
    print(f"paired abs-{control_arm:10s} cells={n_cells} clusters={n_clust} "
          f"mean={mean:+.3f}pp clusterCI95=[{lo:+.3f},{hi:+.3f}] "
          f"clusterPerm_p={p_perm:.4f} clustert_p={p_t:.4f}(df={df}) pos={pos}/{n_cells}")


def main():
    cells = load_cells()
    assert len(cells) == 60, f"expected 60 cells, got {len(cells)}"
    n_clust = len({(c["family"], c["backbone"], c["seed"]) for c in cells})
    print(f"cells={len(cells)} clusters={n_clust} (expect 40)")

    out = []
    out.append("# Cluster-aware pooled statistics (Frame scale-content controls)")
    out.append("")
    out.append("Resampling unit = independent training run `(dataset family, backbone, seed)`;")
    out.append("`n = 40` clusters with 60 cells nested inside (BIWI: biwi_s+biwi_w, IAS: ias_a+ias_b,"
               " KGBD, KGBD-dedup; 2 backbones x 5 seeds).")
    out.append("")
    out.append(f"- Cluster block bootstrap 95% CI: resample the {N_CLUST} clusters with replacement"
               f", `{N_BOOT:,}` draws, percentile interval.")
    out.append("- Cluster sign-flip permutation p: flip every delta inside a cluster together"
               f", `{N_PERMS:,}` draws, two-sided (`(1+count)/(1+n)`).")
    out.append("- Cluster-mean one-sample t-test p with `df = n_clusters - 1 = 39`"
               " (each cluster summarized by its mean Delta_pp).")
    out.append("")
    header = ("| arm | n_cells | n_clusters | mean Δpp | sd_pp | cluster-boot 95% CI | "
              "cluster-perm p | cluster-t p (df) | positive cells |")
    out.append(header)
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm in ARMS:
        cl = clusters_for(cells, arm)
        report_arm(arm, cells, cl, out)
    out.append("")
    out.append("## Cross-arm paired deltas (abs minus each control), cluster-aware")
    out.append("")
    out.append("Same resampling unit; the paired cell diff (abs − control) is clustered by"
               " `(family, backbone, seed)`.")
    out.append("")
    out.append(header)
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm in ("norm", "normmatch", "random"):
        report_paired(arm, cells, out)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
