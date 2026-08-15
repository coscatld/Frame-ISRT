# Cluster-aware pooled statistics (Frame scale-content controls)

Resampling unit = independent training run `(dataset family, backbone, seed)`;
`n = 40` clusters with 60 cells nested inside (BIWI: biwi_s+biwi_w, IAS: ias_a+ias_b, KGBD, KGBD-dedup; 2 backbones x 5 seeds).

- Cluster block bootstrap 95% CI: resample the 40 clusters with replacement, `100,000` draws, percentile interval.
- Cluster sign-flip permutation p: flip every delta inside a cluster together, `100,000` draws, two-sided (`(1+count)/(1+n)`).
- Cluster-mean one-sample t-test p with `df = n_clusters - 1 = 39` (each cluster summarized by its mean Delta_pp).

| arm | n_cells | n_clusters | mean Δpp | sd_pp | cluster-boot 95% CI | cluster-perm p | cluster-t p (df) | positive cells |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| abs | 60 | 40 | +0.83 | 0.93 | [+0.54, +1.16] | 0.0000 | 0.0000 (df=39) | 52/60 |
| norm | 60 | 40 | +0.16 | 0.81 | [-0.09, +0.42] | 0.2663 | 0.4315 (df=39) | 35/60 |
| normmatch | 60 | 40 | +0.01 | 0.90 | [-0.28, +0.29] | 0.9457 | 0.6498 (df=39) | 28/60 |
| random | 60 | 40 | +0.06 | 0.83 | [-0.20, +0.34] | 0.6781 | 0.8927 (df=39) | 29/60 |

## Cross-arm paired deltas (abs minus each control), cluster-aware

Same resampling unit; the paired cell diff (abs − control) is clustered by `(family, backbone, seed)`.

| arm | n_cells | n_clusters | mean Δpp | sd_pp | cluster-boot 95% CI | cluster-perm p | cluster-t p (df) | positive cells |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| abs−norm | 60 | 40 | +0.68 | -- | [+0.53, +0.83] | 0.0000 | 0.0000 (df=39) | 56/60 |
| abs−normmatch | 60 | 40 | +0.82 | -- | [+0.54, +1.11] | 0.0000 | 0.0000 (df=39) | 50/60 |
| abs−random | 60 | 40 | +0.77 | -- | [+0.63, +0.93] | 0.0000 | 0.0000 (df=39) | 60/60 |
