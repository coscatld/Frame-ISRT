# CLEAN_PROTOCOL_AUDIT — Phase 1 Clean Checkpoint Comparison（Q1）

日期：2026-08-10
脚本：`scripts/train_transg_clean.py`、`scripts/run_clean_matrix.sh`、`analysis/aggregate_clean.py`
结果目录：`results/clean/{proto}/{model}/seed{seed}/`（24 run，全部 rc=0）
数据：`transg_official_{biwi_s,biwi_w,ias_a,ias_b}_f6`（官方 f6 预处组数组，未改）

---

## 1. 目的

Q1 的诚实回答：移除 probe 选模与 probe 早停后，Node-ISRT 相对 base TranSG 是否有干净可复现的收益。
固定 final epoch N=200（基于 train-loss 收敛的诚实预算），probe/gallery 指标**仅记录、绝不用于选模**。

## 2. 修改代码

| 文件 | 改动 |
|---|---|
| `scripts/train_transg_clean.py`（新建） | 去 probe 选模/patience/best.pt；固定 `--max-epochs N`；保留 class_samp_gen、每 epoch train 特征 pass（GT-mean 原型）、STPR 掩码、rand_flip、Adam lr=3.5e-4、batch 256、dropout 0.5；每 epoch 记录 train loss + 可选 probe 指标（仅分析）。`checkpoint_rule=fixed_final_epoch`、`probe_gallery_used_for_selection=false`。 |
| `scripts/run_clean_matrix.sh`（新建） | 24-run 驱动，幂等（已有 metrics.json 跳过）+ `--resume`。 |
| `analysis/aggregate_clean.py`（新建） | 聚合 → all_results.csv + paired_delta.csv + 统计摘要。 |
| `skeleton_auth/mva_research/transg_faithful.py` | 新增 `num_frames=6` 默认参数（traj_dec2 输出 T*3），T=6 时逐位不变；仅 Phase 7 需要。 |
| `scripts/train_transg_clean.py` | 从数据推断 T、STPR 掩码 `max(T-2,1)`；f6 行为与改动前逐位一致。 |

> 冻结的 `train_transg_faithful.py` 与 `E:\paper_yolo_mva_results_20260803\*` 结果未触碰。

## 3. 配置（每个 run 的 config.json 均有完整记录）

- 固定末轮：N=200（`selection_epoch=199`），无 probe 选模、无早停。
- 训练：GPC(0.5·seq + 0.5·ske) + STPR(0.5·struct + 0.5·traj)；total=0.5·GPC+0.5·recon。
- Adam lr=3.5e-4，batch 256，dropout 0.5，rand_flip，`class_samp_gen` 重采样。
- Node-ISRT：`scale_residual_projection=Linear(3,128,bias=False)` 零初始化 + `residual_gate=Param(-1.5)`；W_s=0 初始化下 isrt≡base（逐位一致，已单测）。
- 参数：transg 266,126；node_isrt 266,511（+385，+0.145%）。
- 种子：42 / 123 / 2026（成对）。协议：BIWI-S / BIWI-W / IAS-A / IAS-B。

## 4. 完整结果表（fixed-epoch mAP / rank1 / rank5 / rank10）

| protocol | seed | transg mAP | transg R1 | node_isrt mAP | node_isrt R1 | ΔmAP | ΔR1 |
|---|---|---|---|---|---|---|---|
| biwi_s | 42 | 0.2371 | 0.5410 | 0.2256 | 0.5742 | −0.0115 | +0.0332 |
| biwi_s | 123 | 0.2162 | 0.5293 | 0.1985 | 0.4238 | −0.0177 | −0.1055 |
| biwi_s | 2026 | 0.2349 | 0.5371 | 0.2543 | 0.5957 | +0.0194 | +0.0586 |
| biwi_w | 42 | 0.2329 | 0.2917 | 0.2476 | 0.3216 | +0.0146 | +0.0299 |
| biwi_w | 123 | 0.2355 | 0.2982 | 0.2470 | 0.3164 | +0.0115 | +0.0182 |
| biwi_w | 2026 | 0.2388 | 0.3047 | 0.2334 | 0.3086 | −0.0054 | +0.0039 |
| ias_a | 42 | 0.3351 | 0.4189 | 0.3231 | 0.4082 | −0.0119 | −0.0107 |
| ias_a | 123 | 0.2987 | 0.3447 | 0.3107 | 0.3750 | +0.0120 | +0.0303 |
| ias_a | 2026 | 0.3400 | 0.4063 | 0.3299 | 0.4170 | −0.0102 | +0.0107 |
| ias_b | 42 | 0.4482 | 0.5320 | 0.4158 | 0.4945 | −0.0324 | −0.0375 |
| ias_b | 123 | 0.4115 | 0.5164 | 0.4224 | 0.5227 | +0.0109 | +0.0063 |
| ias_b | 2026 | 0.4050 | 0.4820 | 0.4214 | 0.5047 | +0.0164 | +0.0227 |

（完整 rank5/rank10、train_loss、wall_time、git_head 见 `analysis/all_results.csv`。）

## 5. 配对差值统计

| protocol | base mAP | isrt mAP | paired ΔmAP | 正种子 | n |
|---|---|---|---|---|---|
| biwi_s | 0.2294 | 0.2261 | **−0.0033** | 1/3 | 3 |
| biwi_w | 0.2357 | 0.2427 | **+0.0069** | 2/3 | 3 |
| ias_a | 0.3246 | 0.3212 | **−0.0034** | 1/3 | 3 |
| ias_b | 0.4215 | 0.4199 | **−0.0017** | 2/3 | 3 |
| **总体** | | | **−0.0004**（std 0.0159） | **6/12** | 12 |

leave-one-protocol-out ΔmAP：去 biwi_s +0.0006；去 biwi_w −0.0028；去 ias_a +0.0007；去 ias_b +0.0001。
整体 ΔR1 ≈ **+0.005 pp**（12/12 中 7/12 正）。

## 6. 结论（verdict）

**在固定末轮（N=200）诚实评估下，Node-ISRT 相对 base 无干净可复现收益：总体配对 ΔmAP −0.0004 pp（6/12 正，std 0.0159），ΔR1 ≈ +0.005 pp。**

对照冻结 probe-选模轨（ΔmAP +1.60 pp，9/12 正），**clean 收益归零**。这是 Phase 0 审计第 4 条的机制确认：train loss 收敛后（BIWI ~200 ep）probe mAP 仍爬升，isrt 的残差分支提供额外优化容量，使 probe-选模在 >200 ep 挑到更高尖峰。该增益是 **checkpoint 选择效应**，不是检索能力本身。

论文 Node-track 现有"ΔmAP +1.60"是 probe-选模轨数字，**不得**当 clean 结论引用（审计 §9.2 双轨呈现要求）。

## 7. 论文结论是否改变

- **是（重大）**：Node-track 的"Node-ISRT 提升检索精度"在 clean 口径下不成立。可保留的正确表述：
  1. 固定预算下 Node-ISRT 与 base **期望无差异**（ΔmAP≈0，12 种子 6 正）；
  2. 其 probe-选模 +1.60 pp 全部由选模效应解释：probe 选模抬 base +0.96 pp、抬 isrt +2.60 pp，差值 +1.64 pp ≈ +1.60（§10 双轨分解，恒等检验通过）；**例外：biwi_w 有干净收益 +0.69 pp**；
  3. Node-ISRT 是**受控重参数化**（+0.145% 参数，W_s 零初始化），机制定位而非精度提升。
- **否（Frame 轨道不受影响）**：五方向转移矩阵（GRU/Transformer/Mamba +2.42/+1.26/+0.43/+0.16/+1.04，12/15 正）是 Frame-track、15-epoch 固定末轮、ArcFace+triplet，与 Node-track 的 GPC/STPR、N=200 不同，不受本结论影响。
- KGBD Node-track 冻结 −1.61（probe-选模）负边界保留；clean 轨 KGBD 由 Phase 3 补 3 种子。

## 8. 新问题

1. **双轨并存的呈现问题**：probe-选模（官方复现）与 clean（固定末轮）数字必须分开，不得混用（审计 §9.2）。
2. **机制问题**：若 clean 下 Δ≈0，residual_gate σ(g)≈0.19-0.21 近常数观察 + Phase 2 参数匹配控制（归一化残差）能区分"residual 输入是什么"是否根本无关 → Phase 2/5 回答。
3. **尺度敏感性是独立的强信号**：Phase 6 已验证 base 对查询尺度极敏感（±10% → mAP 0.237→0.105）。这是比"ΔmAP"更值得写的机制发现，且与 clean 归零不矛盾（敏感性与平均收益是两个量）。
4. **绝对 mAP 普遍低于 probe-选模**（如 biwi_s 0.2294 vs 0.2659）：任何与官方 30.1 的比较必须注明选模口径。

## 9. 下一阶段

Phase 2 参数匹配控制（node_isrt_norm，C/s 归一化残差，12 run）→ Phase 3 KGBD duplicate-safe clean 3 种子（6 run）→ Phase 5 组件消融（24 run）→ Phase 6 尺度干预（eval-only）→ Phase 7 窗口长度（24 run）。已由 `run_all_post_phase1.sh` 自动串联。

---

## 10. 双轨量化：probe 选模效应分解（selection-effect quantification）

脚本 `analysis/aggregate_dualtrack.py` → `analysis/dualtrack.csv`。
冻结 probe-选模轨（`E:\paper_yolo_mva_results_20260803\transg_faithful_20260807\{proto}_{base|isrt}\seed{seed}\result.json` 的 `best_by_probe_rank1`）vs clean 固定末轮轨（`results/clean/{proto}/{model}/seed{seed}/metrics.json`）。两端数据协议相同（num_classes=50、66500 windows、seeds 42/123/2026），唯一变量是 checkpoint 选择规则 → 选模效应可逐 cell 直接量化。

定义 `inflate = probe_selected_mAP − clean_mAP`（probe 选模为该模型抬升的 mAP）。

| protocol | d_probe | d_clean | inflate_base | inflate_isrt |
|---|---|---|---|---|
| biwi_s | +0.0033 | −0.0033 | +0.0305 | +0.0372 |
| biwi_w | +0.0052 | +0.0069 | +0.0270 | +0.0253 |
| ias_a | +0.0137 | −0.0034 | −0.0024 | +0.0147 |
| ias_b | +0.0420 | −0.0017 | −0.0166 | +0.0271 |
| **总体** | **+0.0160**（9/12 正） | **−0.0004**（6/12 正） | **+0.0096** | **+0.0260** |

恒等检验（分解自洽）：d_probe = d_clean + (inflate_isrt − inflate_base) = −0.0004 + (0.0260 − 0.0096) = +0.0160 ✓

**分解结论**：
1. probe 选模把**两个**模型都抬高了：base +0.96 pp、isrt +2.60 pp。isrt 被抬得比 base 多 **+1.64 pp** —— 论文现轨 "+1.60 pp isrt 优势" 的几乎全部来源（0.0164 ≈ 0.0160）。
2. 机制与 §6 假设一致：residual 分支提供额外优化容量，train loss 平台期后 probe mAP 仍爬升，probe 选模在 >200 ep 为 isrt 挑到更高尖峰。
3. **例外协议 biwi_w**：clean d = +0.0069（2/3 正），是 4 协议中唯一有**干净（非选模）**isrt 收益的。诚实表述应为 "3/4 协议 Δ≈0、biwi_w 有小的干净收益"，不得一刀切写成 "永远无收益"。
4. 选模效应集中在 IAS 数据：ias_b（d_probe +4.20 → clean −0.17）与 ias_a（+1.37 → −0.34）。IAS 上 isrt 的 probe 尖峰最高，clean 下反而略负 —— 全部为选模伪差。
