"""P2-3: arm x backbone interaction on the 5-seed Frame pool (60 cells).

For every (protocol, seed) subject with both backbones, we hold the four arm
deltas (abs, norm, normmatch, random) for GRU and Transformer. The script
reports:

  1. A two-way ANOVA table (arm x backbone), treated as descriptive because the
     arms within a subject share one baseline.
  2. A permutation interaction test: within each subject the GRU/Transformer
     arm-vectors are swapped with p=0.5 (100k draws) and the interaction sum of
     squares is recomputed.
  3. The focused interaction contrast (abs - mean(controls)) x backbone as a
     paired one-sample test across subjects (bootstrap CI, sign-flip perm p,
     Wilcoxon).
  4. Arm and backbone main effects.

Reads the same per-seed CSV as the other P2 scripts.
"""
from __future__ import annotations

import csv
import random
import statistics
from scipy import stats as st

PROTOS = ["biwi_s", "biwi_w", "ias_a", "ias_b", "kgbd", "kgbd_dedup"]
BACKBONES = ["gru", "transformer"]
ARMS = ["abs", "norm", "normmatch", "random"]
CONTROLS = ["norm", "normmatch", "random"]
MODELS = {
    "gru": {"baseline": "gru", "abs": "isrgru", "norm": "isrgru_norm",
            "normmatch": "isrgru_normmatch", "random": "isrgru_randommatch"},
    "transformer": {"baseline": "transformer", "abs": "isr_transformer",
                    "norm": "isr_transformer_norm",
                    "normmatch": "isr_transformer_normmatch",
                    "random": "isr_transformer_randommatch"},
}
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
CSV = str(REPO_ROOT / "results" / "per_seed_frame.csv")
OUT = str(REPO_ROOT / "experiments" / "scale_content_controls" / "aggregate" / "interaction_summary.md")
NPERM = 100_000
RNG_SEED = 20260814


def load_subjects():
    lut = {}
    with open(CSV, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            lut[(row["protocol"], row["seed"], row["model"])] = float(row["mAP"])
    subjects = {}
    for proto in PROTOS:
        for seed in ("42", "123", "2026", "2027", "2028"):
            sd = {}
            for bb in BACKBONES:
                base = lut.get((proto, seed, MODELS[bb]["baseline"]))
                if base is None:
                    break
                for arm in ARMS:
                    v = lut.get((proto, seed, MODELS[bb][arm]))
                    if v is None:
                        break
                    sd[(bb, arm)] = v - base
            if len(sd) == len(BACKBONES) * len(ARMS):
                subjects[(proto, seed)] = sd
    return subjects


def anova_cells(subjects):
    """Return SS_arm, SS_backbone, SS_ab, SS_e, df_arm, df_backbone, df_ab, df_e."""
    cell = {}
    for sd in subjects.values():
        for (bb, arm), val in sd.items():
            cell.setdefault((arm, bb), []).append(val)
    a, b, n = len(ARMS), len(BACKBONES), len(subjects)
    grand = statistics.mean(v for lst in cell.values() for v in lst)
    arm_mean = {arm: statistics.mean(v for (ar, _), lst in cell.items()
                                     if ar == arm for v in lst)
                for arm in ARMS}
    bb_mean = {bb: statistics.mean(v for (_, br), lst in cell.items()
                                   if br == bb for v in lst)
               for bb in BACKBONES}
    cell_mean = {k: statistics.mean(v) for k, v in cell.items()}
    ss_a = b * n * sum((arm_mean[arm] - grand) ** 2 for arm in ARMS)
    ss_b = a * n * sum((bb_mean[bb] - grand) ** 2 for bb in BACKBONES)
    ss_ab = n * sum((cell_mean[(arm, bb)] - arm_mean[arm] - bb_mean[bb] + grand) ** 2
                    for arm in ARMS for bb in BACKBONES)
    ss_e = sum((val - cell_mean[(arm, bb)]) ** 2 for (arm, bb), lst in cell.items() for val in lst)
    df_a, df_b, df_ab, df_e = a - 1, b - 1, (a - 1) * (b - 1), a * b * (n - 1)
    return ss_a, ss_b, ss_ab, ss_e, df_a, df_b, df_ab, df_e


def f_p(ss, df, ms_e):
    return 1 - st.f.cdf((ss / df) / ms_e, df, 4 * 2 * (30 - 1))


def ss_ab_perm(subjects, rng):
    perm = {}
    for key, sd in subjects.items():
        if rng.getrandbits(1):
            perm[key] = {
                (bb, arm): sd[(BACKBONES[1 - BACKBONES.index(bb)], arm)]
                for bb in BACKBONES for arm in ARMS
            }
        else:
            perm[key] = dict(sd)
    return anova_cells(perm)[2]


def bootstrap_ci(values, samples=NPERM, seed=RNG_SEED):
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


def main():
    subjects = load_subjects()
    n = len(subjects)
    ss_a, ss_b, ss_ab, ss_e, df_a, df_b, df_ab, df_e = anova_cells(subjects)
    ms_e = ss_e / df_e
    f_ab = (ss_ab / df_ab) / ms_e
    p_f = 1 - st.f.cdf(f_ab, df_ab, df_e)

    rng = random.Random(RNG_SEED)
    obs = ss_ab
    cnt = sum(1 for _ in range(NPERM) if ss_ab_perm(subjects, rng) >= obs)
    perm_p = cnt / NPERM

    cs = []
    for sd in subjects.values():
        a_g = sd[("gru", "abs")] - statistics.mean(sd[("gru", c)] for c in CONTROLS)
        a_t = sd[("transformer", "abs")] - statistics.mean(sd[("transformer", c)] for c in CONTROLS)
        cs.append(a_g - a_t)
    mean_c, sd_c = statistics.mean(cs), statistics.stdev(cs)
    lo, hi = bootstrap_ci(cs)
    w_p = st.wilcoxon(cs).pvalue if any(c != 0 for c in cs) else 1.0
    rng2 = random.Random(RNG_SEED + 2)
    cnt2 = 0
    for _ in range(NPERM):
        tot = 0.0
        for c in cs:
            tot += c if rng2.getrandbits(1) else -c
        if tot / len(cs) >= abs(mean_c):
            cnt2 += 1
    perm_c_p = 2.0 * (cnt2 / NPERM)

    cell = {}
    for sd in subjects.values():
        for (bb, arm), val in sd.items():
            cell.setdefault((arm, bb), []).append(val)

    lines = []
    lines.append("# P2-3: arm x backbone interaction (5-seed Frame pool)")
    lines.append("")
    lines.append(f"n = {n} subjects, each (protocol, seed) with 4 arms x 2 backbones; "
                 "240 deltas in pp.")
    lines.append("")
    lines.append("## Two-way ANOVA (descriptive; arms within a subject share one baseline)")
    lines.append("")
    lines.append("| source | SS | df | MS | F | p |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(f"| arm | {ss_a:.2f} | {df_a} | {ss_a/df_a:.2f} | "
                 f"{(ss_a/df_a)/ms_e:.3f} | {f_p(ss_a, df_a, ms_e):.4f} |")
    lines.append(f"| backbone | {ss_b:.2f} | {df_b} | {ss_b/df_b:.2f} | "
                 f"{(ss_b/df_b)/ms_e:.3f} | {f_p(ss_b, df_b, ms_e):.4f} |")
    lines.append(f"| arm x backbone | {ss_ab:.2f} | {df_ab} | {ss_ab/df_ab:.2f} | "
                 f"{f_ab:.3f} | {p_f:.4f} |")
    lines.append(f"| error | {ss_e:.2f} | {df_e} | {ms_e:.2f} | | |")
    lines.append("")
    lines.append(f"Interaction permutation p (within-subject backbone swap, "
                 f"{NPERM} draws): **{perm_p:.4f}**")
    lines.append("")
    lines.append("## Focused interaction contrast (abs - mean(controls)) x backbone")
    lines.append("")
    lines.append("GRU-minus-Transformer difference of (abs minus the three-control mean) "
                 "per subject:")
    lines.append("")
    lines.append(f"mean = {mean_c:+.3f} pp, sd = {sd_c:.3f}, 95% CI "
                 f"[{lo:+.3f}, {hi:+.3f}], sign-flip p = {perm_c_p:.4f}, "
                 f"Wilcoxon p = {w_p:.4f}.")
    lines.append("")
    lines.append("## Main effects")
    lines.append("")
    lines.append("| arm | mean Δpp | GRU | Transformer |")
    lines.append("|---|---:|---:|---:|")
    for arm in ARMS:
        g, t = cell[(arm, "gru")], cell[(arm, "transformer")]
        lines.append(f"| {arm} | {statistics.mean(g + t):+.2f} | "
                     f"{statistics.mean(g):+.2f} | {statistics.mean(t):+.2f} |")
    lines.append("")
    lines.append("Interpretation: a null interaction means the arm-effect pattern is not "
                 "reliably backbone-dependent; the pooled 5-seed conclusion in the "
                 "manuscript §4.5.1 holds within each backbone.")
    body = "\n".join(lines)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(body)
    print(body)


if __name__ == "__main__":
    main()
