# EP06 — 2x Contour-Level SR POC (经典物理算法)

> **目标**: 用经典物理 SR 算法在 255 帧主 session 上完成 2x contour-level 重建，证明多帧微扫描对内部结构/轮廓的增益。
> **状态**: ✅ POC 完成（classic 2x SR + 双轨评估）
> **前置**: EP01-EP05 全部完成

---

## 算法表

| 层级 | 方法 | 用途 |
|---|---|---|
| baseline | Bicubic single-frame | 显示倍率对照 |
| baseline | SAA-uniform | 多帧相位覆盖是否有用 |
| baseline+ | SAA-weighted | 质量门控/加权是否足够 |
| classic SR | IBP | forward model baseline |
| regularized physical | MAP-TV | split-half 选择正则强度的物理候选 |

## 双轨输出

1. **主轨 (highpass-input SR)**: 每帧先 highpass → SR 重建 → 输出结构图。评价 gradient magnitude、held-out contour Chamfer、split-half consistency。
2. **控制轨 (raw-temperature SR)**: 原始温度帧 + per-frame offset correction → SR 重建 → 输出端 highpass 可视化。证明主轨不是人为制造结构。

## 评价指标

- Gradient magnitude (轮廓锐度)
- Held-out contour Chamfer distance
- Split-half SR consistency
- Artifact audit (振铃、块状伪影、边缘 ringing)
- 并排可视化对照

## 产物索引

- `algos/ep06_sr_poc/` — 算法实现（独立 UV 环境）
- `scripts/summarize_ep06_alignment_sweep.py` — 汇总 EP06 data-driven alignment sweep 的 CSV/JSON/图表
- `output/ep06_sr_poc/` — 数据产物
- `output/ep06_sr_poc_data_driven_align_sweep/` — default/tuned/NCC-init 三组 full SR sweep 产物
- `notebooks/ep06_sr_poc/` — 可视化 Notebook
- `reports/ep06_sr_poc/` — 正式报告

## 当前结果摘要

- 已在主 session 255 帧上生成 highpass 主轨和 raw-temperature 控制轨的 SAA-uniform、SAA-weighted、IBP、MAP-TV 2x 输出，并完成 default/tuned/NCC-init 三组 data-driven alignment full sweep。
- psf=0.5 sweep 合成验证通过：SAA `30.25 dB`，IBP `30.50 dB`，MAP-TV `30.38 dB`；这只是 smoke test，不是实测轮廓最优证明。
- 真实数据 sanity check 通过：所有输出为 `(960, 1280)`，无 NaN/Inf。
- raw 控制轨在输出端 highpass 后复现主轨主要结构，说明主轨结构不是 highpass 预处理单独制造。
- 当前真实指标应保守解释：default contour refined 仍是 EP06 主 alignment；tuned refined 只作为 sensitivity candidate；NCC init 保留 phase-prior control 角色但 full SR sweep artifact 更高。
- MAP-TV 在三组 alignment 中都选择强正则 `lambda=0.01`，梯度低于 SAA/IBP，不能称为 best 或 sharpest；它是保守正则化候选和 sharpness/regularization 诊断。
- IBP `psf_sigma=0.5` 比旧 `psf_sigma=1.0` 更保守，default 下 std `-0.006953`、P95 gradient `-0.050307`，但 artifact `+0.007619`，不是明确胜者。
- 最新 sweep summary 完整：`sweep_method_metrics.csv` 27 行、`sweep_map_tv_lambda.csv` 24 行、`sweep_validation_summary.csv` 9 行、`sweep_delta_vs_baseline.csv` 27 行，`sweep_summary.json` 的 `missing_files=0`。
- Notebook 源片段已加入 alignment strategy ablation 展示层：自动读取 `output/ep06_alignment_ablation/` 中的 CSV/PNG；产物缺失时给出运行命令并继续执行。
- 当前 SAA alignment ablation 显示 default contour refined 的 split-half NRMSE 最低 (`0.0217`)，tuned contour refined 的 artifact proxy 最低 (`1.4710`) 但 split-half 略差，filename affine 是强 control 而不是最终 alignment truth。

## 复现命令

```bash
uv run python algos/ep06_sr_poc/scripts/run_saa.py --workers 4 --alignment-method data_driven_contour_refined --psf-sigma 0.5 --output-dir output/ep06_sr_poc_data_driven_align_sweep/default_contour_refined_psf05
uv run python algos/ep06_sr_poc/scripts/run_ibp.py --workers 4 --alignment-method data_driven_contour_refined --max-iter 8 --psf-sigma 0.5 --output-dir output/ep06_sr_poc_data_driven_align_sweep/default_contour_refined_psf05
uv run python algos/ep06_sr_poc/scripts/run_map_tv.py --workers 4 --alignment-method data_driven_contour_refined --max-iter 8 --step-size 0.25 --psf-sigma 0.5 --no-fista --lambda-grid 0.0003,0.001,0.003,0.01 --output-dir output/ep06_sr_poc_data_driven_align_sweep/default_contour_refined_psf05
uv run python algos/ep06_sr_poc/scripts/run_evaluation.py --output-dir output/ep06_sr_poc_data_driven_align_sweep/default_contour_refined_psf05 --center-roi-sizes 160,112,80
uv run python scripts/run_ep06_alignment_ablation.py
uv run python scripts/summarize_ep06_alignment_sweep.py --sweep-root output/ep06_sr_poc_data_driven_align_sweep --baseline-dir output/ep06_sr_poc
uv run python scripts/build_notebook.py notebooks/ep06_sr_poc --execute
```

三组 sweep 的差异参数：

- `default_contour_refined_psf05`: 使用默认 alignment CSV 和默认 `contour_refined`。
- `tuned_contour_refined_psf05`: 增加 `--alignment-csv output/ep05_alignment_tuning/full_r360_e93_rad100_s0125/contour_alignment_results.csv --alignment-method data_driven_contour_refined`。
- `ncc_init_psf05`: 增加 `--alignment-method data_driven_ncc_init`。

## Notebook 结论边界

- Alignment ablation 是质量门控，不是 stage command 真值验证；stage command 仍只能作为 prior / 初始化 / 约束。
- 文件名/affine 和 stage command 一样只能作 control 或 prior，不能写成 alignment truth。
- 若 ablation 产物缺失或显示策略敏感，EP06 只能声明“当前 alignment 策略下的 contour-level 候选增益”。
- highpass 是结构图，不是普通温度图；split-half NRMSE、artifact score、Chamfer proxy、gradient 指标都只是辅助证据。最终判断必须同时引用 highpass 结构图、raw-temperature 中心检查和 ROI 视觉一致性。
