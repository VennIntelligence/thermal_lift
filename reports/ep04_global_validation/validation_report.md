# EP04 Validation Report — Alignment Anchor Benchmark and Quality Gates

## 1. 研究问题

EP04 的任务不是证明超分辨率成败，也不是把高精度边缘定位包装成最终交付。客户目标仍然是后续 EP06 在 LWIR 微扫描数据上提升芯片内部结构、形状和局部轮廓的可见性。

EP04 在这个链路中的定位是：

> 用真实主 session 数据建立 alignment anchor benchmark，识别哪些外轮廓/内轮廓段可作为稳定配准锚点，哪些段适合做 holdout 验证，哪些区域不能当作真值但仍应作为 EP06 SR 的重点形状目标。

因此，本报告中的 pass/fail、split-half、CRB ratio 和通过率只描述“局部定位锚点是否可信”。它们不是 SR 成功率，也不是内部结构是否值得重建的否定证据。stage command 和文件名坐标只作为位移 prior / 初始化 / 约束，不能作为对齐真值。

## 2. 方法摘要

验证单位为 `contour segment × scanline`。输入默认只使用主扫描 `session=2` 的 `R=0` X scanline。外轮廓和内轮廓使用同一套质量门控，保证可比较：

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
| segment 数 | 84 | 386 |
| scanline 数 | 13 | 13 |
| segment × scanline 评估数 | 1092 | 5018 |
| A-class segment 数 | 28 | 135 |
| segment 通过率 | 53.6% | 21.5% |
| A-class segment 通过率 | 39.3% | 13.3% |
| row 通过率 | 54.3% | 24.2% |
| A-class row 通过率 | 45.1% | 16.8% |
| A-class split-half 中位数 | 0.0265 px | 0.0271 px |
| A-class split-half P90 | 0.0607 px | 0.0800 px |
| A-class CRB ratio 中位数 | 1.93× | 2.08× |
| A-class phase coverage 中位数 | 0.4396 px | 0.3383 px |
| A-class NCC peak 中位数 | 0.9817 | 0.9861 |

外轮廓提供了更连续的 anchor coverage；内轮廓通过率明显更低，但通过段的 split-half 和 NCC 仍可接近外轮廓水平。这说明内轮廓不是“不可重建区域”，而是需要 EP06 使用形状先验、SR forward model 和更严格验证来处理的目标区域。

## 4. Anchor Coverage 和 EP06 角色

EP04 将 segment 分为三类 EP06 角色：

| contour | EP06 role | segment 数 | split-half 中位数 | CRB ratio 中位数 | pass rate 中位数 |
|---|---|---:|---:|---:|---:|
| outer | alignment_input | 26 | 0.0191 px | 1.28× | 1.000 |
| outer | holdout_validation | 19 | 0.0263 px | 1.79× | 0.769 |
| outer | sr_target_not_truth | 39 | 0.0524 px | 1.91× | 0.154 |
| inner | alignment_input | 40 | 0.0192 px | 1.31× | 1.000 |
| inner | holdout_validation | 43 | 0.0254 px | 1.79× | 0.692 |
| inner | sr_target_not_truth | 303 | 0.0340 px | 2.39× | 0.000 |

推荐用法：

- `alignment_input`: 可作为 EP06 局部/全局配准的输入段，约束 data-driven alignment。
- `holdout_validation`: 不直接参与配准优化，用于验证 EP06 重建后轮廓一致性和 split-half 稳定性。
- `sr_target_not_truth`: 不可直接当作 alignment truth 或 SR 真值；但这些段通常正是客户关心的内部结构形状区域，应保留为 EP06 视觉和结构一致性评估目标。

## 5. 内轮廓失败模式

段级主要失败原因如下：

| 失败原因 | 外轮廓 segment 数 | 内轮廓 segment 数 |
|---|---:|---:|
| `sigma_out_of_range` | 11 | 142 |
| `fit_error:ValueError` | 3 | 75 |
| `split_half_high` | 11 | 44 |
| `low_phase_coverage` | 7 | 17 |
| `low_delta_t` | 4 | 15 |
| `low_snr` | 3 | 4 |
| `ncc_unreliable` | 0 | 3 |
| `psf_sensitivity_high` | 0 | 3 |

内轮廓的主要瓶颈不是 NCC peak 低，而是表观边缘模型不稳定、局部相位覆盖不足和 split-half 长尾。解释上应写成“这些区域不能直接作为配准真值”，而不是“这些区域不值得 SR”。EP06 应重点检查这些区域是否通过多帧 SR 呈现更稳定的形状轮廓。

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

- `output/ep04_global_validation/global_segment_quality_distribution.png`
- `output/ep04_global_validation/anchor_coverage_map.png`
- `output/ep04_global_validation/anchor_scanline_support.png`
- `output/ep04_global_validation/inner_failure_reasons.png`
- `output/ep04_global_validation/ep06_gate_recommendations.png`

## 8. 结论

EP04 给出的是一个正向但有边界的结论：quality-gated, data-driven highpass-NCC / joint-ESF localization 可以在真实 LWIR 主 session 上提供稳定 alignment anchors、holdout validation 段和质量门控指标。

这为 EP06 contour-level SR 提供了配准输入和验证基准。EP04 不承担 SR 成败判定，也不把定位本身当作客户交付。客户目标仍然是通过后续 SR 让芯片内部结构、形状和局部轮廓更清楚、更稳定。
