# P2-3: arm x backbone interaction (5-seed Frame pool)

n = 30 subjects, each (protocol, seed) with 4 arms x 2 backbones; 240 deltas in pp.

## Two-way ANOVA (descriptive; arms within a subject share one baseline)

| source | SS | df | MS | F | p |
|---|---:|---:|---:|---:|---:|
| arm | 26.56 | 3 | 8.85 | 11.584 | 0.0000 |
| backbone | 0.97 | 1 | 0.97 | 1.270 | 0.2609 |
| arm x backbone | 0.71 | 3 | 0.24 | 0.309 | 0.8192 |
| error | 177.30 | 232 | 0.76 | | |

Interaction permutation p (within-subject backbone swap, 100000 draws): **0.2228**

## Focused interaction contrast (abs - mean(controls)) x backbone

GRU-minus-Transformer difference of (abs minus the three-control mean) per subject:

mean = +0.229 pp, sd = 0.666, 95% CI [+0.015, +0.482], sign-flip p = 0.0704, Wilcoxon p = 0.1840.

## Main effects

| arm | mean Δpp | GRU | Transformer |
|---|---:|---:|---:|
| abs | +0.83 | +0.86 | +0.81 |
| norm | +0.16 | +0.07 | +0.24 |
| normmatch | +0.01 | -0.05 | +0.07 |
| random | +0.06 | -0.07 | +0.19 |

Interpretation: a null interaction means the arm-effect pattern is not reliably backbone-dependent; the pooled 5-seed conclusion in the manuscript §4.5.1 holds within each backbone.

> 2026-08-15 addendum (external review Item 1/2): the ANOVA and permutation above
> are descriptive — the 30 subjects still share training runs (BIWI pair, IAS
> pair), the same non-independence the manuscript now addresses with cluster-aware
> pooled statistics in `aggregate/pooled_frame_stats_cluster.md`. Manuscript
> §4.5.1 labels the interaction "descriptive" and no longer derives backbone
> independence from the null interaction.