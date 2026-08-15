## Frame-track scale-control: absolute vs normalized vs norm-matched vs random

36 cells (protocol × backbone × seed), mAP in pp, Δ vs same-seed baseline.

| protocol | backbone | Δ_abs | Δ_norm | Δ_normmatch | Δ_random | Δ_abs−Δ_normmatch | Δ_abs−Δ_random |
|---|---|---|---|---|---|---|---|
| biwi_s | gru | +1.12 | +0.16 | -0.48±0.47 | +0.12±0.55 | +1.60±0.10 | +1.00±0.05 |
| biwi_s | transformer | +2.42 | +2.04 | +0.54±1.85 | +1.93±1.14 | +1.88±0.75 | +0.50±0.15 |
| biwi_w | gru | +1.30 | -0.23 | -0.76±1.34 | -0.30±0.86 | +2.06±1.38 | +1.61±0.10 |
| biwi_w | transformer | +1.26 | +0.62 | -0.32±1.64 | +0.75±1.88 | +1.58±0.95 | +0.51±0.38 |
| ias_a | gru | +0.22 | +0.01 | +0.32±0.38 | -0.14±0.76 | -0.10±0.24 | +0.36±0.17 |
| ias_a | transformer | +0.43 | -0.05 | +0.77±1.61 | -0.13±1.21 | -0.34±0.69 | +0.55±0.25 |
| ias_b | gru | +0.17 | -0.14 | -0.00±0.28 | -0.23±0.62 | +0.17±0.29 | +0.39±0.09 |
| ias_b | transformer | +0.16 | -0.27 | +0.20±0.82 | -0.24±0.79 | -0.03±0.24 | +0.41±0.42 |
| kgbd | gru | +0.66 | -0.16 | -0.10±0.43 | -0.36±0.32 | +0.77±0.20 | +1.03±0.08 |
| kgbd | transformer | +1.04 | +0.12 | -0.30±0.33 | +0.08±0.11 | +1.33±0.11 | +0.96±0.30 |
| kgbd_dedup | gru | +0.63 | -0.17 | -0.11±0.40 | -0.37±0.32 | +0.74±0.21 | +1.00±0.09 |
| kgbd_dedup | transformer | +1.09 | +0.16 | -0.30±0.39 | +0.12±0.16 | +1.39±0.11 | +0.97±0.29 |

**Grand mean over 36 cells:**
- Δ_abs +0.88pp (31/36 positive)
- Δ_norm +0.17pp (21/36 positive)
- Δ_normmatch -0.04pp (16/36 positive)
- Δ_random +0.10pp (17/36 positive)
- Δ_abs−Δ_normmatch +0.92pp (30/36 abs>normmatch)
- Δ_abs−Δ_random +0.77pp (36/36 abs>random)

Gate sanity: Learned σ(g) means — absolute 0.203, normalized 0.195, normmatch 0.197, random 0.126.
norm_matched factor γ = mean_RMS(C)/mean_RMS(N): mean 0.2260 over protocols (per-cell in scale_results.csv).
