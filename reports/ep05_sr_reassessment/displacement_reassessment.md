# EP05 — 2x Phase / Alignment Baseline Reassessment

## 1. Baseline Decision

EP01 `is_sr_usable=True` 的 248 帧 clean main input 具备进入 **2x contour-level SR POC** 的采样和对齐基础。当前路线不再以单个小步、单个边缘段或单个静态参考帧判断全局可行性，而是使用 clean main input 的相位覆盖、数据驱动对齐和 held-out 轮廓残差共同约束。

当前使用边界：

- 采用剔除 `R != 0` repeat 后的 248 帧 clean main input 作为默认 SR 输入。
- stage command 只作为位移 prior / 初始化 / 约束。
- 连续相位 prior 优先使用 highpass NCC init 或 filename affine fit。
- 轮廓精修和质量门控使用 data-driven contour refined alignment。
- filename/stage 坐标不再被视为足够准确的实际热像 alignment；data-driven shift 相对 filename affine 的逐帧修正达到 `0.39-0.43 px` 中位量级，需要作为独立证据处理。
- EP04 localization 作为 alignment anchor / quality gate，不作为最终工业交付目标。
- overlay 只作为 visual sanity appendix，不作为 SR metric。
- 3x/4x phase occupancy 只作为风险诊断；本报告不证明 4x 或 5 µm 计量级 SR。

## 2. Displacement Coverage

复测脚本：

```bash
uv run python scripts/run_ep05_displacement_reassessment.py --n-jobs 16
```

核心数值：

| 口径 | highpass NCC 结果 | 用途 |
|---|---:|---|
| clean main X 相邻 2 um 小步 | median `0.0992 px` | 短时线性诊断 |
| clean main X 相邻 4 um 小步 | median `0.1937 px` | 短时线性诊断 |
| R=0 完整 X scanline endpoint, 40 um | median `2.0768 px` | scanline 相位覆盖 |
| R=0 完整 Y column endpoint, 40 um | median `4.4161 px` | column 相位覆盖 |
| clean main 逐帧累计轨迹 span | `2.4942 x 7.1124 px` | raster 二维覆盖 |
| clean main 逐帧路径长度 | `61.1801 px` | 全路径运动诊断 |

这些数值用于约束 2x SR POC 的相位覆盖和对齐设计。不要把任一局部测量单独上升为全局位移结论。

输出：

- `output/ep05_sr_reassessment/registration_pair_table.csv`
- `output/ep05_sr_reassessment/displacement_measurements.csv`
- `output/ep05_sr_reassessment/displacement_summary_by_class.csv`
- `output/ep05_sr_reassessment/main_session_cumulative_trajectory.csv`
- `output/ep05_sr_reassessment/displacement_reassessment_summary.json`
- `output/ep05_sr_reassessment/main_session_cumulative_trajectory.png`
- `output/ep05_sr_reassessment/visible_shift_by_pair_class.png`
- `output/ep05_sr_reassessment/endpoint_displacement_vectors.png`

Notebook 前置页展示 `main_session_cumulative_trajectory.png`、`visible_shift_by_pair_class.png`、`endpoint_displacement_vectors.png`。这些图只说明主 session 具备可见二维微扫描运动；它们不把局部 NCC 位移或 stage command 升格为 alignment truth。

## 3. 2x SR Phase Capacity and Alignment Baseline

统一打分脚本：

```bash
uv run python scripts/run_ep05_alignment_sr_capacity_check.py
```

该脚本在同一参考帧 `6_16_0.txt`、同一中心 ROI、同一 held-out edge points 上比较：

1. `no_alignment`
2. `old_stage_model`
3. `filename_affine_fit`
4. `data_driven_ncc_init`
5. `data_driven_contour_refined`

结果：

| 方法 | held-out Chamfer median / P90 | gradient corr median | 2x occupied / bad bins | 2x min / max | entropy |
|---|---:|---:|---:|---:|---:|
| no alignment | `0.3976 / 0.7080 px` | `0.6985` | `1 / 3` | `0 / 248` | `0.000` |
| old stage model | `0.2462 / 0.4422 px` | `0.8776` | `4 / 0` | `60 / 64` | `1.000` |
| filename affine | `0.1701 / 0.2043 px` | `0.9549` | `4 / 0` | `57 / 65` | `0.999` |
| data-driven NCC init | `0.1558 / 0.1752 px` | `0.9672` | `4 / 0` | `59 / 66` | `0.999` |
| data-driven contour refined | `0.1332 / 0.1610 px` | `0.9492` | `4 / 0` | `59 / 67` | `0.999` |

Multi-scale phase 风险诊断：

| 方法 | 2x occupied/bad | 3x occupied/bad | 4x occupied/bad | 风险解读 |
|---|---:|---:|---:|---|
| no alignment | `1/3` | `1/8` | `1/15` | sanity check，只提供平均/降噪，不提供 phase diversity |
| old stage model | `4/0` | `9/0` | `16/0` | command/prior 几何覆盖，不是热像真实对齐证据 |
| filename affine | `4/0` | `9/0` | `16/0` | 连续 phase prior 较完整，仍不能证明 4x 可行 |
| data-driven NCC init | `4/0` | `9/0` | `16/0` | 数据约束初值覆盖完整，适合作 2x phase prior |
| data-driven contour refined | `4/0` | `4/5` | `4/12` | 高倍率 phase collapse；可作 quality gate，不能作 4x 证据 |

Data-driven correction magnitude：

| 对比 | shift delta median / P90 / max | paired Chamfer delta median | 解读 |
|---|---:|---:|---|
| NCC init - filename affine | `0.2977 / 0.5334 / 0.6173 px` | `-0.0156 px` | NCC init 不是简单复刻 filename affine |
| contour refined - filename affine | `0.3976 / 0.7141 / 1.1703 px` | `-0.0356 px` | contour refined 有独立局部修正 |
| contour refined - NCC init | `0.2500 / 0.3536 / 0.7071 px` | `-0.0201 px` | refinement 是小范围局部锚定 |
| filename affine - stage prior | `0.5797 / 1.4443 / 2.2329 px` | `-0.0738 px` | filename affine 本身也明显不同于旧 stage prior |

结论：

- 2x SR 四个相位格全部被覆盖，且每格约 `57-67` 帧。
- `data_driven_contour_refined` 的 held-out Chamfer 最低，适合作为轮廓锚定和质量门控。
- `data_driven_ncc_init` 和 `filename_affine_fit` 的相位分布更连续，适合作为 SR phase prior。
- 3x/4x occupancy 不能证明高倍率 SR 可行；`data_driven_contour_refined` 在 3x/4x 上出现 phase collapse，尤其不能用于声明 4x。

叠图证据：

| 方法 | sampled frames | density peak | density P99 | near-reference mean | off-reference mean | near/off ratio |
|---|---:|---:|---:|---:|---:|---:|
| no alignment | `80` | `1.0000` | `0.8750` | `0.4990` | `0.0025` | `197.1` |
| data-driven contour refined | `80` | `1.0000` | `0.9962` | `0.5149` | `0.00003` | `16117.4` |

叠图采用 edge-density heatmap，而不是把 248 帧低透明度线条直接叠成不可读图。data-driven contour refined 后，远离参考边缘的背景 edge density 明显下降，和 held-out Chamfer 改善一致。

输出：

- `output/ep05_alignment_sr_capacity/alignment_method_holdout_scores.csv`
- `output/ep05_alignment_sr_capacity/alignment_method_summary.csv`
- `output/ep05_alignment_sr_capacity/phase_bin_summary_2x.csv`
- `output/ep05_alignment_sr_capacity/phase_bin_counts_2x.csv`
- `output/ep05_alignment_sr_capacity/alignment_overlay_density_metrics.csv`
- `output/ep05_alignment_sr_capacity/alignment_sr_capacity_summary.json`
- `output/ep05_alignment_sr_capacity/alignment_method_comparison.png`
- `output/ep05_alignment_sr_capacity/phase_bin_coverage_2x.png`
- `output/ep05_alignment_sr_capacity/alignment_overlay_evidence.png`

## 4. Contour Alignment Baseline

验证脚本：

```bash
uv run python scripts/run_ep05_contour_alignment_validation.py --n-jobs 16
```

方法：

1. 选择 clean main input 中间帧 `6_16_0.txt` 作为对齐坐标系。
2. 每帧使用 highpass NCC 得到自由 2D 平移初值。
3. 用 highpass-gradient 强边缘和 reference distance transform 做 Chamfer refinement。
4. 用未参与细化的 held-out edge points 评价对齐。

结果：

| 指标 | 数值 |
|---|---:|
| 参与帧数 | 248 |
| 成功对齐帧数 | 248 |
| 对齐前 held-out Chamfer median / P90 / max | `0.3976 / 0.7080 / 1.0520 px` |
| NCC 初始化后 held-out Chamfer median / P90 / max | `0.1558 / 0.1752 / 0.1829 px` |
| 轮廓细化后 held-out Chamfer median / P90 / max | `0.1332 / 0.1610 / 0.1817 px` |
| refined Chamfer <= 0.2 px 的帧数 | `248 / 248` |
| data-driven shift span | `4.0166 x 4.9894 px` |
| refined shift norm median / P90 / max | `1.4261 / 2.2816 / 3.1288 px` |

Worst refined Chamfer frames：

| frame | acquisition order | refined Chamfer | NCC peak | refined gradient corr | shift norm |
|---|---:|---:|---:|---:|---:|
| `32_10_0.txt` | 95 | `0.1817 px` | `0.9183` | `0.9157` | `2.0897 px` |
| `10_20_0.txt` | 166 | `0.1793 px` | `0.9249` | `0.9180` | `0.7651 px` |
| `40_40_0.txt` | 256 | `0.1782 px` | `0.9051` | `0.8761` | `1.9959 px` |
| `4_12_0.txt` | 100 | `0.1747 px` | `0.9271` | `0.8951` | `0.0157 px` |

本节优先使用 absolute held-out Chamfer，而不是依赖 improvement pct。improvement pct 仍可作为辅助诊断，但当 before Chamfer 很小时会被放大，不适合作为 EP06 handoff 的主证据。

输出：

- `output/ep05_contour_alignment/contour_alignment_results.csv`
- `output/ep05_contour_alignment/contour_alignment_summary.json`
- `output/ep05_contour_alignment/contour_refined_alignment_shifts.png`
- `output/ep05_contour_alignment/contour_alignment_chamfer_validation.png`
- `output/ep05_contour_alignment/contour_alignment_improvement_timeline.png`
- `output/ep05_contour_alignment/data_driven_coordinate_shift_field.png`

### 4.1 Alignment Tuning Study

调参目标不是把 Chamfer 单指标压到最低，而是在以下约束之间找平衡：

1. held-out Chamfer median/P90 更低；
2. gradient correlation 不明显崩坏；
3. 2x phase bins 保持完整；
4. 3x/4x 不被误解为可行性证据；
5. 后续 SAA split-half 不明显变差。

已跑调参产物：

- `output/ep05_alignment_tuning_study/tuning_summary.csv`
- `output/ep05_alignment_tuning_study/candidate_comparison_summary.csv`
- `output/ep05_alignment_tuning_study/candidate_phase_coverage.csv`
- `output/ep05_alignment_tuning_study/tuning_heatmap_heldout_chamfer.png`
- `output/ep05_alignment_tuning_study/candidate_alignment_comparison.png`

可复现脚本：

```bash
uv run python scripts/run_ep05_alignment_tuning_study.py --mode quick --limit-frames 96 --n-jobs 8
```

关键结果：

| 候选 | self held-out Chamfer median/P90 | fixed 93% eval Chamfer median/P90 | gradient corr median | 相对 NCC init | 相对 filename affine | 2x bins |
|---|---:|---:|---:|---:|---:|---:|
| default refined, step 0.25 | `0.1281 / 0.1579 px` | `0.1281 / 0.1579 px` | `0.9518` | `17.6%` 更低 | `17.8%` 更低 | `4/4` |
| tuned refined, edge 91, step 0.125 | `0.1209 / 0.1561 px` | `0.1209 / 0.1561 px` | `0.9439` | `22.2%` 更低 | `22.4%` 更低 | `4/4` |

调参解释：

- quick study 当前 Chamfer 最强候选为 `edge_percentile=91, refine_radius=0.5, refine_step=0.125`，但它牺牲了一部分 gradient correlation。
- default refined 的 Chamfer 略高，但 gradient correlation 和 SAA split-half 更稳。
- `data_driven_ncc_init` 的 Chamfer 不如 contour refined，但 gradient correlation 最强，且 3x/4x phase coverage 完整。
- contour refined 不论默认还是 tuned，在 3x/4x 上都会出现 phase collapse：3x 只占 `4/9`，4x 只占 `4/16`。这强化了当前边界：**contour refinement 可作为 2x contour gate，不可作为 4x 证据**。

EP06 handoff 因此应保留三路：

1. `NCC init`：连续 phase prior，最适合作 SR shift 主线候选；
2. `default/tuned contour refined`：轮廓锚定和 quality gate；
3. `filename affine`：强 prior/control，不作真实 alignment truth。

## 5. Overlay Baseline

叠图脚本：

```bash
uv run python scripts/run_ep05_overlay_alignment_check.py
uv run python scripts/run_ep05_overlay_4x4_check.py
uv run python scripts/run_ep05_edge_line_overlay.py
```

核心用途：

- 检查不对齐、stage prior、filename affine、data-driven contour 四种方式下的边缘集中程度。
- 同时查看 TXT thermal edge 和 BMP visible/thermal-render edge，避免只依赖单一数值口径。
- 作为人工 sanity check，不作为最终 SR 指标。

代表性结果：

| 组别 | 最优/接近最优叠法 | 关键数值 |
|---|---|---:|
| 全部 R=0, 248 帧 | filename affine / data-driven contour | median `0.0812 / 0.0904 px` |
| `Y=10` X scanline | filename affine | median `0.0419 px` |
| `Y=20` X scanline | data-driven contour | median `0.0993 px` |
| `X=10` Y column | filename affine / data-driven contour | median `0.0862 / 0.0896 px` |
| `X=20` Y column | filename affine / data-driven contour | median `0.0807 / 0.0890 px` |

逐组 median Chamfer 结论：

- `all_r0`、`scanline_y10`、`column_x10`、`column_x20`: filename affine 更低。
- `scanline_y20`: data-driven contour 更低。
- 这说明 overlay 证据必须诚实作为 visual sanity appendix 使用，不能把 overlay 胜负等同于 SR 成功。

输出：

- `output/ep05_overlay_alignment/overlay_alignment_summary.csv`
- `output/ep05_overlay_alignment/all_r0_overlay_grid.png`
- `output/ep05_overlay_alignment/all_main_4x4_txt_bmp_overlay.png`
- `output/ep05_overlay_alignment/all_main_4x4_edge_line_overlay.png`

## 6. EP06 Decision Table

| 项目 | 决策 | 原因 |
|---|---|---|
| EP06 推荐对齐 | `NCC init` 作连续 SR shift 候选，`contour refined` 作局部 gate/tuned 候选 | NCC init 保留连续相位和最高 gradient correlation；contour refined held-out Chamfer 最低 |
| phase prior | 保留 filename affine 和 NCC init | 两者提供连续 2x/3x/4x phase coverage；filename 只作 prior/control |
| 对照组 | no alignment / stage prior / filename affine / NCC init / default refined / tuned refined | 区分显示增益、先验增益、真实数据驱动对齐增益和调参收益 |
| 3x/4x 状态 | 仅作风险诊断 | occupancy 不是 SR 证据，contour refined 高倍率 phase collapse 尤其不能证明 4x |
| overlay 用途 | visual sanity appendix | filename affine 多组更优、scanline_y20 contour 更优；overlay 不是 SR metric |
| 失败风险 | thermal drift / local contour ambiguity / over-refinement / PSF-SNR ceiling | 这些风险可能制造视觉上合理但 split-half 或 held-out 不稳定的结果 |
| 验收指标 | split-half consistency / held-out contour Chamfer / phase-bin coverage / visual contour gain | back-projection residual 或 Tenengrad 不能单独作为 SR 成功证据 |

## 7. Next Implementation Target

下一步直接进入 2x contour-level SR POC：

1. 选取主 session 中内部结构最稳定的 ROI。
2. 用 NCC init / filename affine 生成连续相位 prior，并把 tuned/default refined 作为 alignment ablation。
3. 用 contour refined alignment 做局部锚定和质量门控，避免把 3x/4x phase collapse 写成高倍率证据。
4. 建立带 drift / offset / gain 处理的 forward model。
5. 输出 LR、bicubic 2x、simple stack、2x SR 的并排图和 split-half / held-out 指标。
