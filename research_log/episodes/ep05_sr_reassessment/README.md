# EP05 — 2x SR Alignment Baseline

> **目标**: 建立 255 帧主 session 的 2x contour-level SR 对齐与相位覆盖基线。
> **状态**: ✅ 已完成 alignment / phase capacity baseline；下一步进入 EP06 2x SR POC。
> **前置**: EP01 数据审计、EP02 raster/stage-prior 诊断、EP03 contour observability、EP04 anchor quality gate。

---

## 任务清单

- [x] 按 acquisition order 复测主 session displacement coverage。
- [x] 完成 data-driven contour alignment 验证。
- [x] 完成 TXT/BMP overlay sanity check。
- [x] 完成 2x SR phase capacity check，比较 no alignment / stage prior / filename affine / NCC init / contour refined。
- [x] 新建并执行 EP05 decision notebook。
- [x] 固化 EP05 baseline 报告、notebook fragments 和输出目录。

---

## 核心结果

| 口径 | 结果 |
|------|---:|
| 主 session X 相邻 2 um 小步 | median 0.0969 px |
| 主 session X 相邻 4 um 小步 | median 0.1917 px |
| R=0 完整 X scanline endpoint | median 2.0836 px |
| R=0 完整 Y column endpoint | median 4.4174 px |
| 主 session 累计轨迹 span | 2.8809 x 9.0435 px |
| data-driven contour held-out Chamfer | median 0.1341 px / P90 0.1613 px |
| contour alignment improvement | median 66.5% |
| refined alignment shift span | 5.0491 x 6.2536 px |
| 全 R=0 overlay 最优 Chamfer | filename affine median 0.0827 px |
| 2x SR phase bins, data-driven refined | 4/4 occupied, min/max 58/69 frames |
| 2x SR phase bins, NCC init | 4/4 occupied, min/max 62/65 frames |
| 2x SR phase bins, no alignment | 1/4 occupied, 3 bad bins |
| contour stack off-reference density | refined 0.00003 vs no alignment 0.00287 |

---

## 决策记录

- 255 帧主 session 具备进入 2x contour-level SR POC 的相位覆盖和对齐基础。
- stage command 继续作为位移 prior / 初始化 / 约束，不作为对齐真值。
- `data_driven_ncc_init` 和 `filename_affine_fit` 更适合作为连续相位 prior。
- `data_driven_contour_refined` 的 held-out Chamfer 最低，适合作为轮廓锚定和质量门控。
- EP04 localization 继续作为 alignment anchor / quality gate，不作为客户最终交付。
- 4x / 5 um 主张留到后续，不能只凭相位覆盖宣称成立。

---

## 产物索引

脚本:

- `scripts/run_ep05_displacement_reassessment.py`
- `scripts/run_ep05_contour_alignment_validation.py`
- `scripts/run_ep05_overlay_alignment_check.py`
- `scripts/run_ep05_overlay_4x4_check.py`
- `scripts/run_ep05_edge_line_overlay.py`
- `scripts/run_ep05_alignment_sr_capacity_check.py`

Notebook:

- `notebooks/ep05_sr_reassessment/fragments/`

报告:

- `reports/ep05_sr_reassessment/displacement_reassessment.md`

主要输出:

- `output/ep05_sr_reassessment/registration_pair_table.csv`
- `output/ep05_sr_reassessment/displacement_measurements.csv`
- `output/ep05_sr_reassessment/displacement_summary_by_class.csv`
- `output/ep05_sr_reassessment/main_session_cumulative_trajectory.csv`
- `output/ep05_sr_reassessment/displacement_reassessment_summary.json`
- `output/ep05_contour_alignment/contour_alignment_results.csv`
- `output/ep05_contour_alignment/contour_alignment_summary.json`
- `output/ep05_overlay_alignment/overlay_alignment_summary.csv`
- `output/ep05_overlay_alignment/all_main_4x4_txt_bmp_overlay.png`
- `output/ep05_overlay_alignment/all_main_4x4_edge_line_overlay.png`
- `output/ep05_alignment_sr_capacity/alignment_method_holdout_scores.csv`
- `output/ep05_alignment_sr_capacity/alignment_method_summary.csv`
- `output/ep05_alignment_sr_capacity/phase_bin_summary_2x.csv`
- `output/ep05_alignment_sr_capacity/phase_bin_counts_2x.csv`
- `output/ep05_alignment_sr_capacity/alignment_overlay_density_metrics.csv`
- `output/ep05_alignment_sr_capacity/alignment_sr_capacity_summary.json`
- `output/ep05_alignment_sr_capacity/alignment_method_comparison.png`
- `output/ep05_alignment_sr_capacity/phase_bin_coverage_2x.png`
- `output/ep05_alignment_sr_capacity/alignment_overlay_evidence.png`
