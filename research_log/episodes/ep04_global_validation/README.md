# EP04 — Alignment Anchor Benchmark and Quality Gates

> **目标**: 将 EP04 重建为下游 EP06 contour-level SR 的 alignment anchor benchmark 与质量门控基准。
> **状态**: ✅ 已完成外轮廓/内轮廓联合质量门控、anchor coverage、EP06 gate recommendations。
> **前置**: EP01 数据审计、EP02 位移标定重审、EP03 theoretical limits / segment 34 data-driven POC。

---

## 任务清单

- [x] 工程化 `thermal_core.ep04`，封装 segment/scanline validation pipeline 和 EP04-local segment input 生成逻辑。
- [x] 更新 `scripts/run_ep04a_validation.py`，默认同时运行外轮廓和内轮廓验证。
- [x] 生成全局 segment 质量分布图：precision / CRB / SNR / pass-fail，按外轮廓和内轮廓分组。
- [x] 生成 anchor coverage map，标出可作为 alignment anchor 的 contour/scanline 和缺锚区域。
- [x] 生成内轮廓失败原因图，并明确这些区域是 EP06 SR 的重点形状目标，不是放弃区域。
- [x] 增加 localization precision vs shape reconstruction 的区别表/图。
- [x] 输出 EP06 质量门控建议：alignment input、holdout validation、sr target not truth。
- [x] 补回经典 EP04 诊断图：split-half、CRB ratio、phase coverage、failure taxonomy、cross-scanline consistency，并按 alignment-anchor 语境重新解释。
- [x] 新增 segment × scanline pass heatmap、failure co-occurrence、NCC/ESF failure diagnostic、EP06 role margin 和 normal-angle coverage。
- [x] 更新 notebook fragments、正式报告和研究日志，删除旧的负面成败叙事。

---

## 关键结果

| 指标 | 外轮廓 | 内轮廓 |
|------|------:|------:|
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

segment 34 仍是稳定外轮廓 anchor：13 条 scanline 中 12 条通过，段级 split-half 中位数为 0.0149 px，CRB ratio 中位数约 1.05×。它不再被写成终点，而是作为 EP06 anchor/gate 体系中的强样例。

segment × scanline heatmap 结果显示，失败同时存在局部坏段和较弱 scanline：

| contour | row pass rate | weakest scanline | weakest scanline pass rate | zero-pass scanlines | zero-pass segments |
|---|---:|---:|---:|---:|---:|
| outer | 54.3% | 24 µm | 44.0% | 0 | 16 |
| inner | 24.2% | 14 µm | 22.3% | 0 | 190 |

NCC / ESF 诊断显示内轮廓主要瓶颈不是 NCC peak 低：inner 失败 row 的 median NCC peak 为 0.9868，P10 为 0.9825，100.0% 仍高于 0.85 NCC gate；`ncc_unreliable` 只覆盖 3.1% 失败 row，而 ESF/model/stability 类失败覆盖 99.3%。

---

## EP06 门控建议

| contour | EP06 role | segment 数 | split-half 中位数 | CRB ratio 中位数 | pass rate 中位数 |
|---|---|---:|---:|---:|---:|
| outer | alignment_input | 26 | 0.0191 px | 1.28× | 1.000 |
| outer | holdout_validation | 19 | 0.0263 px | 1.79× | 0.769 |
| outer | sr_target_not_truth | 39 | 0.0524 px | 1.91× | 0.154 |
| inner | alignment_input | 40 | 0.0192 px | 1.31× | 1.000 |
| inner | holdout_validation | 43 | 0.0254 px | 1.79× | 0.692 |
| inner | sr_target_not_truth | 303 | 0.0340 px | 2.39× | 0.000 |

使用边界：

- `alignment_input`: EP06 alignment 的输入锚点，可用于局部/全局配准约束。
- `holdout_validation`: 不直接用于优化，用于 EP06 重建后的轮廓一致性、split-half 和结构稳定性验证。
- `sr_target_not_truth`: 不可直接作为真值；但这些内轮廓和弱轮廓通常是客户关心的内部结构区域，应保留为 SR 可见性提升目标。

Role margin audit 显示 `alignment_input` 对 alignment-input 数值阈值有正余量；`holdout_validation` 接近阈值，适合做泛化检查；`sr_target_not_truth` 主要在 pass-rate margin 上为负，这只表示不能当真值或强 anchor，不表示放弃区域。

---

## 决策记录

- EP04 的 pass/fail 是质量门控，不是 SR 成功/失败判定。
- stage command 和文件名坐标只作为位移 prior、初始化或正则约束，不能作为对齐真值。
- 外轮廓 anchor coverage 明显优于内轮廓，适合优先支撑 EP06 alignment。
- 内轮廓通过率低，但通过段的 A-class split-half 中位数仍接近外轮廓；这说明内部结构中存在可用锚点，同时也暴露了 EP06 需要重点改善的形状区域。
- 内轮廓主要失败原因是 `sigma_out_of_range`、`fit_error:ValueError`、`split_half_high` 和 `low_phase_coverage`，不是简单的 NCC 崩溃。
- row-level 失败原因是多标签，百分比不可相加为 100%；inner 中 `sigma_out_of_range` 覆盖 53.7% 失败 row，`split_half_high` 覆盖 34.2%，`fit_error:ValueError` 覆盖 27.7%。
- normal-angle coverage 是 alignment 几何覆盖诊断，只说明锚点方向分布，不替代 stage-to-pixel 标定，也不证明内部结构真值。
- localization benchmark 是配准支撑；最终交付仍应看 EP06 的 2x contour-level SR、holdout 一致性和内部结构形状可见性。

---

## 统计口径

- `pass_rate` 是 segment 在 13 条 X scanline 上通过质量门控的比例。
- A-class 表示高 SNR 且 stage-prior 法线投影较好的候选 segment；它仍需 data-driven gate 验证。
- `phase_coverage_px` 来自 highpass NCC 位移投影后的法线相位覆盖，用于判断 joint ESF 是否有足够相位多样性。
- `fitted_sigma_px` 是 joint ESF 的表观边缘宽度，不是纯光学 PSF。
- `sr_target_not_truth` 表示不可当作 alignment truth 或 SR 真值，不表示该区域应从 SR 目标中删除。

---

## 产物索引

Notebook:

- `notebooks/ep04_global_validation/fragments/`
- `notebooks/ep04_global_validation/ep04_global_validation.ipynb`

脚本和核心库:

- `core/src/thermal_core/ep04.py`
- `scripts/run_ep04a_validation.py`

报告:

- `reports/ep04_global_validation/validation_report.md`

主要数据产物:

- `output/ep04_global_validation/segment_validation_results.csv`
- `output/ep04_global_validation/segment_summary.csv`
- `output/ep04_global_validation/global_summary.json`
- `output/ep04_global_validation/inner/inner_segment_validation_results.csv`
- `output/ep04_global_validation/inner/inner_segment_summary.csv`
- `output/ep04_global_validation/inner/inner_global_summary.json`
- `output/ep04_global_validation/ep06_gate_recommendations.csv`
- `output/ep04_global_validation/ep06_gate_recommendation_summary.csv`

主要图:

- `output/ep04_global_validation/global_segment_quality_distribution.png`
- `output/ep04_global_validation/anchor_coverage_map.png`
- `output/ep04_global_validation/anchor_scanline_support.png`
- `output/ep04_global_validation/segment_scanline_pass_heatmap.png`
- `output/ep04_global_validation/inner_failure_reasons.png`
- `output/ep04_global_validation/ep06_gate_recommendations.png`
- `output/ep04_global_validation/split_half_distribution.png`
- `output/ep04_global_validation/crb_ratio_scatter.png`
- `output/ep04_global_validation/phase_coverage_vs_precision.png`
- `output/ep04_global_validation/failure_taxonomy.png`
- `output/ep04_global_validation/cross_scanline_consistency.png`
- `output/ep04_global_validation/normal_angle_coverage.png`
