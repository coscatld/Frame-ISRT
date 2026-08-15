# BIWI-S 复现差距根因审计报告 (Phase 4)

日期: 2026-08-09
范围: 官方 TranSG (TF1.14) BIWI-S mAP **30.1** vs 我们的 faithful PyTorch 移植
(batch 256 / lr 3.5e-4 / T=6 / J=20 / seq_lambda=0.5, probe=Still→gallery=Walking)
3-seed 探针选择均值 **26.0** (26.59 / 24.19 / 27.21)。

## 1. 目的

逐模块审计官方复现链路 (数据 → 架构 → 配置 → 训练 → 评估 → 可复现性)，
对 30 项检查点逐项给出 ✓一致 / ✗偏差 / ⚠风险，定位 26.0 vs 30.1 差距的根因。
审计原则: 不改协议、不删负结果、不选择性取种子、不做探针/图库模型选择。

## 2. 审计方法与证据来源

| 来源 | 路径 |
|---|---|
| 官方 TranSG 训练 | `external/TranSG_official/TranSG.py` (逐段通读) |
| 官方数据生成 | `external/TranSG_official/Preprocess.py`, `utils/process_SG.py` (逐段通读) |
| 官方复现数据 | `data/transg_official_biwi_s_f6/` (官方 npy 数组直接消费, 含 sha256) |
| faithful 移植 | `skeleton_auth/mva_research/transg_faithful.py`, `model.py` |
| faithful 训练脚本 | `scripts/mva_research/train_transg_faithful.py` |
| 训练产物 | `transg_faithful_20260807/biwi_s_base/{seed}/result.json`, `history.npy`, `stdout.log` |

## 3. 30 项检查点清单

### A. 数据 (1–5)
| # | 检查点 | 结论 | 证据 |
|---|---|---|---|
| 1 | 训练/gallery/probe 数组与官方一致 | ✓ | `protocol.json`: train=34,294 窗(50 id), gallery=822(28), probe=531(28); `source_file_sha256` 记录; 直接消费官方 npy, **未重新加窗** |
| 2 | 加窗方式 | ✓ | 官方 `Preprocess.py` 滑动窗 (train stride 3, s∈0..5; test stride 6, s=0), 移植直接用官方分布式数组, 无二次加窗 |
| 3 | T=6, J=20 | ✓ | `time_step=6`, `nb_nodes=20`, role windows/overlaps 全空 |
| 4 | 角色划分 probe=Still→gallery=Walking | ✓ | 官方 `TranSG.py` Eval 段 probe=Still 时 gallery=Walking 测试集, 协议一致 |
| 5 | root-centered by joint 0 | ✓ | 官方 `process_SG.py:29-31`, 移植 `root_center` root_joint=0 |

### B. 架构 (6–11)
| # | 检查点 | 结论 | 证据 |
|---|---|---|---|
| 6 | SGT 层数 / 头数 | ✓ | 2 SGT layers, 8 heads |
| 7 | enc_k=10 Laplacian PE | ✓ | 取 k=10 个最小非平凡特征向量 |
| 8 | 输入嵌入 (input_fc / pos_proj) | ✓ | 结构一致 (见 Phase 0 EXPERIMENT_GAP_AUDIT.md) |
| 9 | 重建/尺度解码器 (STPR) | ✓ | recon + scale 解码器对应官方 Reshape_38 分支 |
| 10 | **骨骼邻接图** | ✓ | `TRANSG_20_BONES` 19 条无向边 **逐一等于** 官方 `j_pair_1/j_pair_2` (本轮核对, 见 §4) |
| 11 | pos_enc 惰性 buffer 不入 state_dict | ⚠ | `register_buffer(..., persistent=False)`; 训练脚本显式 `model.pos_enc = model._laplacian_pos_enc(device)` |

### C. 配置 (12–16)
| # | 检查点 | 结论 | 证据 |
|---|---|---|---|
| 12 | batch 256 / lr 3.5e-4 | ✓ | |
| 13 | dropout 0.5, patience 120, rand_flip=1 | ✓ | |
| 14 | seq_lambda=0.5, prompt_lambda=0.5, GPC_lambda=0.5 | ✓ | 仅 CASIA_B 用 seq_lambda=1.0, BIWI/IAS 均为 0.5 |
| 15 | t1=0.07, t2=14 | ✓ | |
| 16 | St_mask_num=10, Tr_mask_num=2 | ✓ | |

### D. 训练 (17–21)
| # | 检查点 | 结论 | 证据 |
|---|---|---|---|
| 17 | class_samp_gen 类均衡重采样 | ✓ | `train_windows_resampled=66,500`, `batches_per_epoch=259` (=66500/256) |
| 18 | 损失组合 | ✓ | `0.5·H_loss + 0.5·recon`; `H_loss = 0.5·GPC_ske + 0.5·GPC_seq` |
| 19 | 全局 shuffle | ✓ | 官方 `process_SG.py:175-177` |
| 20 | 优化器/调度 | ✓ | 对齐官方 (epoch 级 LR decay) |
| 21 | 探针监督 (GPC t1/t2) | ✓ | seq_lambda=0.5 下 GPC_ske 与 GPC_seq 权重一致 |

### E. 评估 (22–26)
| # | 检查点 | 结论 | 证据 |
|---|---|---|---|
| 22 | **检索特征** | ✓ | 变体 A (池化 `seq_ftr` [B,H]) = 0.2659 与记录完全一致; 原始 node 0.2293 / 输入嵌入 0.0988 均更低; 官方 `Reshape_38:0` 计数分析指向池化 [B,H] reshape |
| 23 | **AP 公式** | ✓ | 自定义 AP 与 sklearn `average_precision_score` 数值等价 (均 0.2659) |
| 24 | 距离度量 | ✓ | L2 + 升序排序, 与官方一致 |
| 25 | Rank-1 探针选择 | ✓ | 官方 `TranSG.py:865` (`top_1 > max_acc_2`, epoch>0); 移植 best.pt 亦按探针 R1 选择 |
| 26 | 保存时机 | ✓ | 每 epoch 起始探针评估后保存 best.pt |

### F. 可复现性 / 残余差距归因 (27–30)
| # | 检查点 | 结论 | 证据 |
|---|---|---|---|
| 27 | **pos_enc 符号歧义** | ⚠ **已实证** | 同一 best.pt, 重算 PE: CPU `torch.linalg.eigh` → mAP 0.1995; GPU → 0.2659 (与记录一致). 特征向量每列符号在库/设备间不保证一致 |
| 28 | eig 实现差异 | ⚠ | 官方 `np.linalg.eig` vs 移植 `torch.linalg.eigh`; 均升序取 1:k, 公式同 `L=I−D^-1/2 A D^-1/2`, 但符号约定不同 |
| 29 | 跨种子方差 | ✓ 如实记录 | 探针选择 mAP: seed42 0.2659 / seed123 0.2419 / seed2026 0.2721 → 均值 0.260; history 峰值: 0.2884@48 / 0.2746@50 / 0.2838@63 |
| 30 | 差距归因结论 | → §5 | 排除项 vs 残余项 |

## 4. 骨骼邻接图逐边核对 (新增证据)

官方 `process_SG.py:315-318` (nb_nodes=20 分支) 对称双向边, 去重后 19 条无向边:

```
(2,3) (2,8) (8,9) (9,10) (10,11) (2,4) (4,5) (5,6) (6,7) (2,1)
(1,0) (0,16) (0,12) (16,17) (12,13) (17,18) (18,19) (13,14) (14,15)
```

`model.py:36-56 TRANSG_20_BONES` 19 条边 **完全相同**。Laplacian 构造亦一致
(无自环邻接, 对称, `L=I−D^-1/2 A D^-1/2`)。→ 骨骼图结构与 PE 结构差异排除。

## 5. 结论 (Verdict)

**已排除的差距来源 (证据确凿):**
1. **AP 公式** — 自定义 vs sklearn 数值等价 (0.2659)。
2. **检索特征** — 池化 [B,H] 与记录一致; 备选特征 (node/输入嵌入) 全部更低, 官方
   Reshape_38 计数亦指向池化。
3. **数据/加窗/角色划分** — 官方数组直接消费, 大小与角色映射逐项一致。
4. **配置/损失/类均衡/掩码** — 逐项与官方一致。
5. **骨骼邻接图** — 19 条边完全一致 (本轮新增证据)。

**识别出的可复现性风险 (不解释为协议错误, 但影响跨实现复现):**
- **pos_enc 特征向量符号歧义**: lazy buffer 不入 state_dict; CPU/CUDA 重算结果不同
  (实证 0.1995 vs 0.2659)。官方与移植使用不同 eig 库, 若符号约定不同, 同一权重的
  行为即不同。这是 **跨实现复现的已知陷阱**, 不是移植 bug。

**残余差距归因 (26.0 vs 30.1, ~4.1 mAP):**
> 所有可验证组件一致后, 残余差距不能归于任一已检查模块。最可能来源是:
> **(a) 种子/选择方差** — 官方单点 30.1 种子未知、试次未知; 我们 3-seed 探针选择均值
> 26.0, 但单种子 history 峰值可达 28.4–28.8, 单次实验波动大;
> **(b) TF1.14 vs PyTorch 浮点/归约差异** (学习率有效步长、LayerNorm epsilon、图注意力
> softmax 精度等微小差异在 120+ epoch 下累积);
> **(c) pos_enc 符号约定 (见 §3-27/28)** 放大 (a)(b)。

## 6. 论文结论是否改变

**否。** Phase 4 是复现性审计, 不产生新的方法学结论。它只把 faithful 基线定位为
"官方协议的忠实实现 (逐项核对), 与官方绝对数值存在 ~4 mAP 的跨实现/种子差异"。
对 ISR 相关的相对结论 (ISR→Transformer +2.42 等五方向转移矩阵, 已冻结结果) 无影响。

## 7. 新问题

1. 官方 30.1 的具体训练种子与保存策略 (每 epoch 是否探针评估、patience) 不可得 →
   无法判定官方是否经历多次尝试取最优。
2. pos_enc 符号歧义提示: **任何跨框架复现实验都应把 pos_enc 显式固化为文件**。
   建议后续 Phase 1 (clean matrix) 记录并导出 pos_enc, 使复现确定化。
3. 差距 ~4 mAP 的量级与跨种子波动 (24.2–27.2 探针选择) 同量级 → 单点数值比较
   不足以断言"实现有缺陷"。

## 8. 下一步 (Phase 1 关联)

Phase 1 clean matrix (4 协议 × 3 种子 × 2 模型 = 24 运行, N=200 固定, 无探针选择)
正在后台运行 (首个 biwi_s/transg/seed42 已到 epoch 160)。审计期间的发现
(池化特征、AP 公式、bone 图一致) 已固化为该矩阵的评估口径。矩阵完成后输出
`CLEAN_PROTOCOL_AUDIT.md` 汇总, 提供跨种子/跨协议的配对基线, 供 Phase 2 参数匹配对照。

## 9. 附录: KGBD 数据变体核对 (Phase 4 遗留项, 2026-08-09 补)

Phase 0 审计 (EXPERIMENT_GAP_AUDIT.md §4-3) 遗留: "KGBD 三变体命名需 Phase 4 确认
与官方 20.2 同源"。本轮以 float32 数组 sha256 与 protocol.json 核对:

| 数据目录 | train 窗 | probe 窗 | train__probe 重叠 | probe 骨架 hash (前 16) | train identity hash |
|---|---|---|---|---|---|
| `transg_official_kgbd_f6` | 31,573 | 15,626 | 87 | `726f999316358164` | `503db12020da8e93` |
| `transg_official_kgbd_f6_layoutfix` | 31,573 | 15,626 | 87 | `f65654cd59b85e74` | `503db12020da8e93` |
| `transg_official_kgbd_f6_layoutfix_labels` | 31,573 | 15,626 | 87 | `f65654cd59b85e74` | `9ddfc05971c26812` |
| `kgbd_dedup_20260806/data/transg_official_kgbd_f6_dedup` | 31,486 | 15,626 | **0** | `f65654cd59b85e74` | `99517de1ce4135cf` |

结论:
- **layoutfix / layoutfix_labels 骨架逐字节相同** (`f65654...`), 与 `kgbd_f6`
  (`726f99...`) **不同** — "layoutfix" 是布局修正变体, 并非同名同源; 三个目录的
  protocol.json 都记录 `joint_order=[11,10,1,0,...]`, 故差异未体现在该元数据中
  (layoutfix 的具体修正内容未在发布中留档, 标记为待确认, 不影响 Phase 3 决策)。
- **frozen node-track 冻结运行使用 layoutfix 布局**: 冻结 `kgbd` (contaminated,
  data=`transg_official_kgbd_f6_layoutfix_labels`) 与 `kgbd_dedup_base/isrt`
  (data=`kgbd_dedup_20260806/.../transg_official_kgbd_f6_dedup`, 其 probe 哈希
  同为 `f65654...`)。→ **Node-track KGBD 的权威布局是 layoutfix**。
- **冻结 dedup 数据有效**: train 31,486 (=31,573−87), 与 probe/gallery 零重叠,
  三角色均保留全部 164 身份, 无 probe 身份缺席 train。已复算确认。
- **Phase 3 口径修正**: 本轮曾误从 `transg_official_kgbd_f6` (非 layoutfix) 构建
  dupsafe 数据, 已弃用 (目录重命名为 `transg_official_kgbd_f6_dupsafe_SUPERSEDED`)。
  Phase 3 改用权威冻结 dedup 数据 `kgbd_dedup_20260806/data/transg_official_kgbd_f6_dedup`。
- **注意**: 冻结 `kgbd_dedup_base/isrt` seed42 使用 **probe 选模** (best_by_probe_rank1,
  mAP 0.2050/0.1889, Δ=−1.61); Phase 3 的 clean 协议 (N=200 固定、无探针选择) 与它
  **不可直接数值比较**, 仅作方向参考。Phase 3 需重新跑全部三种子 (42/123/2026) 的
  transg/node_isrt 才能给出 clean 配对 Δ。
