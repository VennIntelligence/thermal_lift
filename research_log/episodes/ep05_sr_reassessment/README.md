# EP05 — 2x Phase / Alignment Baseline

> **目标**: 建立 EP01 `is_sr_usable=True` 248 帧 clean main input 的 2x contour-level SR 对齐与相位覆盖基线。
> **状态**: ✅ 已补强 displacement / phase / alignment / overlay reassessment；下一步进入 EP06 2x SR POC。
> **前置**: EP01 数据审计、EP02 raster/stage-prior 诊断、EP03 contour observability、EP04 anchor quality gate。

---

## 任务清单

- [x] 按 acquisition order 复测主 session displacement coverage。
- [x] 完成 data-driven contour alignment 验证。
- [x] 完成 TXT/BMP overlay sanity check。
- [x] 完成 2x SR phase capacity check，比较 no alignment / stage prior / filename affine / NCC init / contour refined。
- [x] 补充 multi-scale phase risk diagnostic：3x/4x occupancy 不作为可行性证明。
- [x] 补充 data-driven correction magnitude：说明 NCC/refined 不是简单复刻 filename affine。
- [x] 补充 overlay group summary：filename affine 多组更优，scanline_y20 contour refined 更优。
- [x] 补充 alignment tuning study：比较 default refined、tuned refined、NCC init 和 filename affine 的 Chamfer/phase/split-half tradeoff。
- [x] 新建并执行 EP05 decision notebook。
- [x] 固化 EP05 baseline 报告、notebook fragments 和输出目录。

---

## 核心结果

| 口径 | 结果 |
|------|---:|
| clean main X 相邻 2 um 小步 | median 0.0992 px |
| clean main X 相邻 4 um 小步 | median 0.1937 px |
| R=0 完整 X scanline endpoint | median 2.0768 px |
| R=0 完整 Y column endpoint | median 4.4161 px |
| clean main 累计轨迹 span | 2.4942 x 7.1124 px |
| data-driven contour held-out Chamfer | median 0.1332 px / P90 0.1610 px |
| refined held-out Chamfer tail | max 0.1817 px |
| refined alignment shift span | 4.0166 x 4.9894 px |
| NCC init vs filename affine correction | median 0.2977 px |
| contour refined vs filename affine correction | median 0.3976 px |
| tuned refined held-out Chamfer | median 0.1209 px / P90 0.1561 px |
| tuned refined vs NCC init | Chamfer median 低约 22.2% |
| tuned refined vs filename affine | Chamfer median 低约 22.4% |
| 全 R=0 overlay 最优 Chamfer | filename affine median 0.0812 px |
| scanline_y20 overlay 最优 Chamfer | data-driven contour median 0.0993 px |
| 2x SR phase bins, data-driven refined | 4/4 occupied, min/max 59/67 frames |
| 2x SR phase bins, NCC init | 4/4 occupied, min/max 59/66 frames |
| 4x phase bins, contour refined | 4/16 occupied, 12 bad bins |
| 2x SR phase bins, no alignment | 1/4 occupied, 3 bad bins |
| contour stack off-reference density | refined 0.00003 vs no alignment 0.00287 |

---

## 决策记录

- 248 帧 clean main input 具备进入 2x contour-level SR POC 的相位覆盖和对齐基础。
- stage command 继续作为位移 prior / 初始化 / 约束，不作为对齐真值。
- `data_driven_ncc_init` 和 `filename_affine_fit` 更适合作为连续相位 prior。
- `data_driven_contour_refined` 的 held-out Chamfer 最低，适合作为轮廓锚定和质量门控。
- tuned contour refined quick winner (`edge=91`, `refine_radius=0.5`, `refine_step=0.125`) 能进一步降低 held-out Chamfer，但 gradient correlation 和 EP06 SAA split-half 不自动更优；它是候选 gate，不是无条件替换默认 shift。
- `data_driven_ncc_init` / `data_driven_contour_refined` 相对 filename affine 有可测逐帧修正，不是简单复刻文件名仿射模型。
- `data_driven_contour_refined` 在 3x/4x 上出现 phase collapse；高倍率 occupancy 或局部吸附结果不能证明 4x 可行。
- overlay 只作为 visual sanity appendix：filename affine 在多数组别 median Chamfer 更低，`scanline_y20` 中 data-driven contour 更低。
- EP06 必须把 alignment 作为 ablation 因子：NCC init、filename affine、default refined、tuned refined 至少保留到 SAA/IBP/MAP-TV 对比中。
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

- `paper/reports/ep05_sr_reassessment/displacement_reassessment.md`

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
- `output/ep05_alignment_tuning_study/tuning_summary.csv`
- `output/ep05_alignment_tuning_study/candidate_comparison_summary.csv`
- `output/ep05_alignment_tuning_study/candidate_phase_coverage.csv`
- `output/ep05_alignment_tuning_study/tuning_heatmap_heldout_chamfer.png`
- `output/ep05_alignment_tuning_study/candidate_alignment_comparison.png`
