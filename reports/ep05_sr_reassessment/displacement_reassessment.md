# EP05 — 2x SR Capacity and Alignment Baseline

## 1. Baseline Decision

主 session 的 255 帧 TXT 温度矩阵具备进入 **2x contour-level SR POC** 的采样和对齐基础。当前路线不再以单个小步、单个边缘段或单个静态参考帧判断全局可行性，而是使用主 session 的全帧相位覆盖、数据驱动对齐和 held-out 轮廓残差共同约束。

当前使用边界：

- 采用 255 帧主 session 作为默认 SR 输入。
- stage command 只作为位移 prior / 初始化 / 约束。
- 连续相位 prior 优先使用 highpass NCC init 或 filename affine fit。
- 轮廓精修和质量门控使用 data-driven contour refined alignment。
- EP04 localization 作为 alignment anchor / quality gate，不作为最终工业交付目标。

## 2. Displacement Coverage

复测脚本：

```bash
uv run python scripts/run_ep05_displacement_reassessment.py --n-jobs 16
```

核心数值：

| 口径 | highpass NCC 结果 | 用途 |
|---|---:|---|
| 主 session X 相邻 2 um 小步 | median `0.0969 px` | 短时线性诊断 |
| 主 session X 相邻 4 um 小步 | median `0.1917 px` | 短时线性诊断 |
| R=0 完整 X scanline endpoint, 40 um | median `2.0836 px` | scanline 相位覆盖 |
| R=0 完整 Y column endpoint, 40 um | median `4.4174 px` | column 相位覆盖 |
| 主 session 逐帧累计轨迹 span | `2.8809 x 9.0435 px` | raster 二维覆盖 |
| 主 session 逐帧路径长度 | `64.6964 px` | 全路径运动诊断 |

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

## 3. 2x SR Phase Capacity and Alignment Baseline

统一打分脚本：

```bash
uv run python scripts/run_ep05_alignment_sr_capacity_check.py
```

该脚本在同一参考帧 `10_16_0.txt`、同一中心 ROI、同一 held-out edge points 上比较：

1. `no_alignment`
2. `old_stage_model`
3. `filename_affine_fit`
4. `data_driven_ncc_init`
5. `data_driven_contour_refined`

结果：

| 方法 | held-out Chamfer median / P90 | gradient corr median | 2x occupied / bad bins | 2x min / max | entropy |
|---|---:|---:|---:|---:|---:|
| no alignment | `0.3813 / 0.7011 px` | `0.7023` | `1 / 3` | `0 / 255` | `0.000` |
| old stage model | `0.2402 / 0.4158 px` | `0.8817` | `4 / 0` | `62 / 65` | `1.000` |
| filename affine | `0.1708 / 0.2087 px` | `0.9551` | `4 / 0` | `60 / 67` | `0.999` |
| data-driven NCC init | `0.1563 / 0.1758 px` | `0.9668` | `4 / 0` | `62 / 65` | `1.000` |
| data-driven contour refined | `0.1341 / 0.1613 px` | `0.9487` | `4 / 0` | `58 / 69` | `0.999` |

结论：

- 2x SR 四个相位格全部被覆盖，且每格约 `58-69` 帧。
- `data_driven_contour_refined` 的 held-out Chamfer 最低，适合作为轮廓锚定和质量门控。
- `data_driven_ncc_init` 和 `filename_affine_fit` 的相位分布更连续，适合作为 SR phase prior。
- `data_driven_contour_refined` 不应用于声明 4x 可行，因为 Chamfer refinement 会把结果吸附到少数局部最优 offset；4x 只保留为后续风险项。

叠图证据：

| 方法 | sampled frames | density peak | density P99 | near-reference mean | off-reference mean | near/off ratio |
|---|---:|---:|---:|---:|---:|---:|
| no alignment | `80` | `1.0000` | `0.8750` | `0.4956` | `0.0029` | `172.5` |
| data-driven contour refined | `80` | `1.0000` | `0.9959` | `0.5136` | `0.00003` | `17483.3` |

叠图采用 edge-density heatmap，而不是把 255 帧低透明度线条直接叠成不可读图。data-driven contour refined 后，远离参考边缘的背景 edge density 明显下降，和 held-out Chamfer 改善一致。

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

1. 选择主 session 中间帧 `10_16_0.txt` 作为对齐坐标系。
2. 每帧使用 highpass NCC 得到自由 2D 平移初值。
3. 用 highpass-gradient 强边缘和 reference distance transform 做 Chamfer refinement。
4. 用未参与细化的 held-out edge points 评价对齐。

结果：

| 指标 | 数值 |
|---|---:|
| 参与帧数 | 255 |
| 成功对齐帧数 | 255 |
| 对齐前 held-out Chamfer median / P90 | `0.3813 / 0.7011 px` |
| NCC 初始化后 held-out Chamfer median / P90 | `0.1563 / 0.1758 px` |
| 轮廓细化后 held-out Chamfer median / P90 | `0.1341 / 0.1613 px` |
| held-out improvement median | `66.5%` |
| refined Chamfer <= 0.2 px 的帧数 | `255 / 255` |
| data-driven shift span | `5.0491 x 6.2536 px` |
| refined shift norm median / P90 / max | `1.3991 / 2.3189 / 4.2197 px` |

输出：

- `output/ep05_contour_alignment/contour_alignment_results.csv`
- `output/ep05_contour_alignment/contour_alignment_summary.json`
- `output/ep05_contour_alignment/contour_refined_alignment_shifts.png`
- `output/ep05_contour_alignment/contour_alignment_chamfer_validation.png`
- `output/ep05_contour_alignment/contour_alignment_improvement_timeline.png`
- `output/ep05_contour_alignment/data_driven_coordinate_shift_field.png`

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
| 全部 R=0, 248 帧 | filename affine / data-driven contour | median `0.0827 / 0.0910 px` |
| `Y=10` X scanline | filename affine | median `0.0391 px` |
| `Y=20` X scanline | data-driven contour | median `0.0970 px` |
| `X=10` Y column | filename affine / data-driven contour | median `0.0833 / 0.0888 px` |
| `X=20` Y column | filename affine / data-driven contour | median `0.0826 / 0.0935 px` |

输出：

- `output/ep05_overlay_alignment/overlay_alignment_summary.csv`
- `output/ep05_overlay_alignment/all_r0_overlay_grid.png`
- `output/ep05_overlay_alignment/all_main_4x4_txt_bmp_overlay.png`
- `output/ep05_overlay_alignment/all_main_4x4_edge_line_overlay.png`

## 6. EP06 Decision Table

| 项目 | 决策 | 原因 |
|---|---|---|
| EP06 推荐对齐 | data-driven NCC init + contour refinement gate | held-out contour Chamfer 最低，同时保留数据约束的局部 anchor |
| phase prior | 保留 filename affine 和 NCC init | 两者提供连续 2x phase coverage，可在局部 refinement 前使用 |
| 对照组 | no alignment / stage prior / filename affine / NCC init | 区分显示增益、先验增益和真正 data-driven 对齐增益 |
| 失败风险 | thermal drift / local contour ambiguity / PSF-SNR ceiling | 这些风险可能制造视觉上合理但 held-out 不稳定的结果 |
| 验收指标 | split-half consistency / held-out contour Chamfer / phase-bin coverage / visual contour gain | back-projection residual 或 Tenengrad 不能单独作为 SR 成功证据 |

## 7. Next Implementation Target

下一步直接进入 2x contour-level SR POC：

1. 选取主 session 中内部结构最稳定的 ROI。
2. 用 NCC init / filename affine 生成连续相位 prior。
3. 用 contour refined alignment 做局部锚定和质量门控。
4. 建立带 drift / offset / gain 处理的 forward model。
5. 输出 LR、bicubic 2x、simple stack、2x SR 的并排图和 split-half / held-out 指标。
