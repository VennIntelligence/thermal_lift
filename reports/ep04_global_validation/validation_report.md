# EP04 Validation Report — Alignment Anchor Benchmark and Quality Gates

## 1. 研究问题

EP04 的任务不是证明超分辨率成败，也不是把高精度边缘定位包装成最终交付。客户目标仍然是后续 EP06 在 LWIR 微扫描数据上提升芯片内部结构、形状和局部轮廓的可见性。

EP04 在这个链路中的定位是：

> 用主 session clean set 中 EP04-local 的完整 `R=0` X-scanline 子集建立 alignment anchor benchmark，识别哪些外轮廓/内轮廓段可作为稳定配准锚点，哪些段适合做 holdout 验证，哪些区域不能当作真值但仍应作为 EP06 SR 的重点形状目标。

因此，本报告中的 pass/fail、split-half、CRB ratio 和通过率只描述“局部定位锚点是否可信”。它们不是 SR 成功率，也不是内部结构是否值得重建的否定证据。stage command 和文件名坐标只作为位移 prior / 初始化 / 约束，不能作为对齐真值。

## 2. 方法摘要

验证单位为 `contour segment × scanline`。输入默认只使用主扫描 clean set 内完整的 `session=2` / `R=0` X scanline。这个 EP04-local 子集是 13 条 X scanline、208 个唯一帧，全部属于当前 `is_sr_usable == True` / `is_main_session == True` 的 248 帧 clean SR input；它不是“255 帧 SR 输入”，也不是对 248 帧全集的 SR 重建验证。外轮廓和内轮廓使用同一套质量门控，保证可比较：

1. 在 contour segment 中心提取局部 ROI。
2. 对相邻帧执行 highpass NCC，获得 data-driven 局部位移。
3. 将位移投影到 segment signed normal，作为 joint ESF 的相位多样性。
4. 对 16 帧 normal profile 做共享 `s, sigma` 的 joint erf 拟合。
5. 用奇偶帧 split-half 拟合评估定位稳定性。
6. 用 fitted `sigma`、`DeltaT` 和 data-driven phase set 计算 CRB。
7. 按 SNR、NCC、phase coverage、fitted sigma、split-half 和 PSF sensitivity 生成质量门控结果。

`fitted_sigma_px` 是表观边缘过渡宽度，包含光学 PSF、热边缘本身、采样和残余配准影响，不等同于纯光学 PSF `sigma≈0.5 px`。EP04 的 sigma gate 是 ESF 模型有效性门控。

## 3. 全局 Anchor 质量分布

| 指标 | 外轮廓 | 内轮廓 |
|---|---:|---:|
| segment 数 | 84 | 390 |
| scanline 数 | 13 | 13 |
| segment × scanline 评估数 | 1092 | 5070 |
| A-class segment 数 | 28 | 139 |
| segment 通过率 | 54.8% | 22.3% |
| A-class segment 通过率 | 46.4% | 12.9% |
| row 通过率 | 52.3% | 24.8% |
| A-class row 通过率 | 40.7% | 17.1% |
| A-class split-half 中位数 | 0.0277 px | 0.0273 px |
| A-class split-half P90 | 0.0622 px | 0.0847 px |
| A-class CRB ratio 中位数 | 1.89× | 2.03× |
| A-class phase coverage 中位数 | 0.4401 px | 0.3723 px |
| A-class NCC peak 中位数 | 0.9825 | 0.9863 |

数据契约边界：

| 项目 | 计数 | 说明 |
|---|---:|---|
| raw main session | 255 frames | `session == 2` 的采集事实，保留作诊断 |
| clean SR input | 248 frames | `is_sr_usable == True` / `is_main_session == True`，当前 SR 默认输入 |
| EP04 complete X scanlines | 13 scanlines | 固定 Y、完整 16 个 X 坐标的 `R=0` scanline |
| EP04 localization frames | 208 unique frames | 13 × 16，全部属于 clean SR input |

外轮廓提供了更连续的 anchor coverage；内轮廓通过率明显更低，但通过段的 split-half 和 NCC 仍可接近外轮廓水平。这说明内轮廓不是“不可重建区域”，而是需要 EP06 使用形状先验、SR forward model 和更严格验证来处理的目标区域。

经典 EP04 诊断图已纳入 Notebook 并重新解释为 alignment-anchor 证据：

- `split_half_distribution.png`
- `crb_ratio_scatter.png`
- `phase_coverage_vs_precision.png`
- `failure_taxonomy.png`
- `cross_scanline_consistency.png`

这些图只回答“哪些外轮廓段可稳定当锚点、哪些段/scanline 应降权或剔除”，不回答 LR/bicubic/SR 对照，也不输出形状重建。

新增 `segment x scanline` heatmap 显示失败既有局部坏段，也有相对弱的 scanline：

| contour | row pass rate | weakest scanline | weakest scanline pass rate | zero-pass scanlines | zero-pass segments |
|---|---:|---:|---:|---:|---:|
| outer | 52.3% | 24 µm | 47.6% | 0 | 20 |
| inner | 24.8% | 12 µm | 22.8% | 0 | 194 |

解释边界：zero-pass segment 表示该段在当前 localization gate 下不能作为强 alignment truth；它不是“没有结构”或“放弃 SR 目标”。weak scanline 表示 EP06 对齐时应考虑 holdout/低权重，而不是把单条线外推为 stage command 真值。

## 4. Anchor Coverage 和 EP06 角色

EP04 将 segment 分为三类 EP06 角色：

| contour | EP06 role | segment 数 | split-half 中位数 | CRB ratio 中位数 | pass rate 中位数 |
|---|---|---:|---:|---:|---:|
| outer | alignment_input | 23 | 0.0193 px | 1.39× | 1.000 |
| outer | holdout_validation | 23 | 0.0250 px | 1.74× | 0.692 |
| outer | sr_target_not_truth | 38 | 0.0607 px | 2.06× | 0.000 |
| inner | alignment_input | 42 | 0.0189 px | 1.25× | 1.000 |
| inner | holdout_validation | 45 | 0.0213 px | 1.52× | 0.692 |
| inner | sr_target_not_truth | 303 | 0.0375 px | 2.55× | 0.000 |

推荐用法：

- `alignment_input`: 可作为 EP06 局部/全局配准的输入段，约束 data-driven alignment。
- `holdout_validation`: 不直接参与配准优化，用于验证 EP06 重建后轮廓一致性和 split-half 稳定性。
- `sr_target_not_truth`: 不可直接当作 alignment truth 或 SR 真值；但这些段通常正是客户关心的内部结构形状区域，应保留为 EP06 视觉和结构一致性评估目标。

EP06 role margin audit 用 alignment-input 数值门槛审计三类 role 距离阈值多近：pass rate ≥70%、split-half ≤0.06 px、CRB ratio ≤5×、phase coverage ≥0.15 px。正 margin 表示离 alignment 输入阈值有余量；负 margin 表示该项不能当强 anchor 条件。

| contour | EP06 role | segment 数 | pass-rate margin | split margin | CRB margin | phase margin | P10 min margin | closest gate |
|---|---|---:|---:|---:|---:|---:|---:|---|
| outer | alignment_input | 23 | +30.0 pp | +0.0407 px | +3.61× | +1.843 px | +0.231 | pass rate |
| outer | holdout_validation | 23 | -0.8 pp | +0.0350 px | +3.26× | +1.788 px | -0.231 | pass rate |
| outer | sr_target_not_truth | 38 | -70.0 pp | -0.0007 px | +2.94× | +0.658 px | -1.795 | pass rate |
| inner | alignment_input | 42 | +30.0 pp | +0.0411 px | +3.75× | +1.824 px | +0.209 | pass rate |
| inner | holdout_validation | 45 | -0.8 pp | +0.0387 px | +3.48× | +1.679 px | -0.231 | pass rate |
| inner | sr_target_not_truth | 303 | -70.0 pp | +0.0225 px | +2.45× | +0.812 px | -1.968 | pass rate |

`sr_target_not_truth` 的负 margin 不表示放弃区域，而是表示不能把这些段当作 alignment truth 或 SR 真值。它们仍然应保留为 EP06 内部结构可见性目标。

## 5. 内轮廓失败模式

段级主要失败原因如下：

| 失败原因 | 外轮廓 segment 数 | 内轮廓 segment 数 |
|---|---:|---:|
| `sigma_out_of_range` | 12 | 145 |
| `fit_error:ValueError` | 6 | 71 |
| `split_half_high` | 6 | 45 |
| `low_phase_coverage` | 7 | 18 |
| `low_delta_t` | 6 | 12 |
| `low_snr` | 1 | 4 |
| `ncc_unreliable` | 0 | 6 |
| `psf_sensitivity_high` | 0 | 2 |

内轮廓的主要瓶颈不是 NCC peak 低，而是表观边缘模型不稳定、局部相位覆盖不足和 split-half 长尾。解释上应写成“这些区域不能直接作为配准真值”，而不是“这些区域不值得 SR”。EP06 应重点检查这些区域是否通过多帧 SR 呈现更稳定的形状轮廓。

row-level 多标签失败诊断进一步支持这一点。内轮廓 3813 个失败 row 中：

| reason | triggered rows | share of failed rows | strongest co-occurrence |
|---|---:|---:|---|
| `sigma_out_of_range` | 2031 | 53.3% | `split_half_high` |
| `split_half_high` | 1305 | 34.2% | `sigma_out_of_range` |
| `fit_error:ValueError` | 1076 | 28.2% | n/a |
| `low_phase_coverage` | 272 | 7.1% | `sigma_out_of_range` |
| `low_snr` | 195 | 5.1% | `sigma_out_of_range` |

这些百分比是多标签比例，不可相加为 100%。同一行可能同时触发 sigma、split-half、phase 或 fit gate。

NCC / ESF 诊断表明，失败行的 NCC peak 并未整体崩溃：

| contour | failed rows | median failed NCC peak | P10 failed NCC peak | failed rows above NCC gate | NCC unreliable share | ESF/model/stability share |
|---|---:|---:|---:|---:|---:|---:|
| outer | 521 | 0.9844 | 0.9779 | 100.0% | 0.0% | 97.9% |
| inner | 3813 | 0.9869 | 0.9827 | 100.0% | 3.2% | 99.1% |

因此当前瓶颈应写成“ESF 表观宽度、拟合、split-half 和 phase coverage 的 localization gate 限制”，而不是“NCC peak 太低”。这保护了 EP06 的正确用法：inner fail 段不能当强锚点，但仍是 SR 目标。

## 6. 定位精度与形状重建的区别

| 项目 | EP04 localization benchmark | EP06 shape / contour SR |
|---|---|---|
| 主要对象 | 局部热边缘 crossing 的亚像素位置 | 芯片内部结构、形状、轮廓和局部细节 |
| 可信证据 | split-half、CRB ratio、NCC quality、phase coverage | forward-model 一致性、holdout 段验证、结构稳定性、视觉轮廓增益 |
| 可用输出 | alignment anchors、quality gates、validation metrics | 2x contour-level SR 图像和内部结构可见性提升 |
| 禁止外推 | 单个 edge localization 不能替代完整形状重建 | SR 不能把 stage command 当作对齐真值 |

EP04 的高精度定位成果是 EP06 的配准支撑，不是最终交付。通过门控的 anchor 可以让 EP06 更稳；未通过的内部轮廓不能当作真值，但仍是 SR 应改善和验证的对象。

## 7. 产物索引

核心 CSV / JSON：

- `output/ep04_global_validation/segment_validation_results.csv`
- `output/ep04_global_validation/segment_summary.csv`
- `output/ep04_global_validation/global_summary.json`
- `output/ep04_global_validation/inner/inner_segment_validation_results.csv`
- `output/ep04_global_validation/inner/inner_segment_summary.csv`
- `output/ep04_global_validation/inner/inner_global_summary.json`
- `output/ep04_global_validation/ep06_gate_recommendations.csv`
- `output/ep04_global_validation/ep06_gate_recommendation_summary.csv`

关键图：

- `output/ep04_global_validation/split_half_distribution.png`
- `output/ep04_global_validation/crb_ratio_scatter.png`
- `output/ep04_global_validation/phase_coverage_vs_precision.png`
- `output/ep04_global_validation/failure_taxonomy.png`
- `output/ep04_global_validation/cross_scanline_consistency.png`
- `output/ep04_global_validation/global_segment_quality_distribution.png`
- `output/ep04_global_validation/anchor_coverage_map.png`
- `output/ep04_global_validation/anchor_scanline_support.png`
- `output/ep04_global_validation/segment_scanline_pass_heatmap.png`
- `output/ep04_global_validation/inner_failure_reasons.png`
- `output/ep04_global_validation/ep06_gate_recommendations.png`
- `output/ep04_global_validation/normal_angle_coverage.png`

## 8. 结论

EP04 给出的是一个正向但有边界的结论：quality-gated, data-driven highpass-NCC / joint-ESF localization 可以在真实 LWIR 主 session clean set 的完整 X-scanline 子集上提供稳定 alignment anchors、holdout validation 段和质量门控指标。

这为 EP06 contour-level SR 提供了配准输入和验证基准。EP04 不承担 SR 成败判定，也不把定位本身当作客户交付。客户目标仍然是通过后续 SR 让芯片内部结构、形状和局部轮廓更清楚、更稳定。
