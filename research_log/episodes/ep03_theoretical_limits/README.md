# EP03 — SR 物理边界与局部可观测性

> **目标**: 为 2x contour-level SR POC 建立物理边界、必要条件风险图和局部可观测性准则：区分 detector pitch / spatial resolution / output grid Nyquist，量化 PSF/MTF、MTF x SNR recoverability 与噪声底约束，说明 ESF/CRB 只能作为 alignment anchor 与 quality gate。
> **状态**: ✅ Notebook 与报告已按当前主线补强并重新执行
> **前置**: EP01 数据审计；EP02 stage prior 标定；EP04 localization 作为后续 alignment gate

---

## 任务清单

- [x] 保持 EP03 口径为物理边界与局部可观测性，不证明 SR 成败。
- [x] 补充 output grid sample vs Nyquist period 表，明确 `5 um output sample != 5 um spatial resolution`，2x grid Nyquist period 为 `10 um`。
- [x] 补充 MTF x SNR recoverability table/heatmap：`effective_snr = DeltaT * MTF / noise`。
- [x] recoverability 覆盖 reference contrast 与 inner/outer median contour contrast，扫描 `sigma=0.2/0.35/0.5 px` 和 1x/2x/4x Nyquist。
- [x] 补充 CRB sensitivity scan：扫描 `DeltaT`、`sigma`、`n_frames` 和抽象 `phase_coverage`，配合 0.05/0.10 px gate 解读。
- [x] 在 Notebook 每张图/表后保留教程式解读，显式说明能得出什么、不能得出什么。
- [x] 更新正式报告 `reports/ep03_theoretical_limits/theoretical_limits_report.md`。
- [x] 执行 `uv run python scripts/build_notebook.py notebooks/ep03_theoretical_limits --execute` 并记录结果。

---

## 关键结果

| 项目 | 结果 |
|---|---:|
| BMP mm-axis detector pitch | 10.000 um/pixel |
| TXT/BMP contour cross-check pitch | 9.980 um/pixel |
| TXT/BMP outer-mask IoU | 0.9938 |
| Current spatial resolution | 20.000 um |
| 2x SR output sample | 5.000 um |
| 2x grid Nyquist period | 10.000 um |
| 4x exploratory output sample | 2.500 um |
| 4x grid Nyquist period | 5.000 um |
| Noise floor | 0.0724 C |
| 0.3 C / 0.7 C / 1.0 C SNR | 4.14 / 9.67 / 13.81 |
| Outer contour segments | 84 |
| Outer median `|DeltaT|` / SNR | 2.490 C / 34.39 |
| Outer anchor-candidate fraction | 33.3% |
| Inner contour segments | 386 |
| Inner median `|DeltaT|` / SNR | 1.938 C / 26.77 |
| Inner anchor-candidate fraction | 35.0% |
| CRB, `DeltaT=0.7 C`, `sigma=0.5 px`, single frame | 0.1415 px |
| CRB, `DeltaT=0.7 C`, `sigma=0.5 px`, 16-frame known phase | 0.0344 px |
| CRB, `DeltaT=2.0 C`, `sigma=1.0 px`, 16-frame known phase | 0.0170 px |

MTF key values:

| Grid | Sigma 0.20 | Sigma 0.35 | Sigma 0.50 |
|---|---:|---:|---:|
| 1x Nyquist | 0.821 | 0.546 | 0.291 |
| 2x Nyquist | 0.454 | 0.089 | 0.007 |
| 4x Nyquist | 0.042 | 0.000063 | 0.000000003 |

MTF x SNR representative values:

| Contrast scale | Input SNR | 2x sigma=0.20 | 2x sigma=0.35 | 2x sigma=0.50 | 4x sigma=0.20 | 4x sigma=0.35 | 4x sigma=0.50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nominal edge, 0.7 C | 9.67 | 4.390 | 0.861 | 0.070 | 0.411 | 0.001 | ~0 |
| Inner contour median, 1.938 C | 26.77 | 12.157 | 2.386 | 0.193 | 1.138 | 0.002 | ~0 |
| Outer contour median, 2.490 C | 34.39 | 15.616 | 3.064 | 0.247 | 1.462 | 0.002 | ~0 |

CRB sensitivity representative values for `DeltaT=0.7 C`:

| Sigma | Phase coverage | 1 frame | 4 frames | 16 frames | 64 frames | 255 frames |
|---:|---:|---:|---:|---:|---:|---:|
| 0.35 px | 0.0 px | 0.1286 | 0.0643 | 0.0321 | 0.0161 | 0.0081 |
| 0.35 px | 1.0 px | n/a | 0.0564 | 0.0286 | 0.0144 | 0.0072 |
| 0.50 px | 0.0 px | 0.1415 | 0.0707 | 0.0354 | 0.0177 | 0.0089 |
| 0.50 px | 1.0 px | n/a | 0.0684 | 0.0344 | 0.0172 | 0.0086 |

---

## 决策记录

- `10 um/pixel` 是 TXT detector sampling pitch；`20 um` 是当前系统空间分辨率，不得混写。
- `5 um` 目标至少要求 2x output grid，但 `5 um output sample` 不是 `5 um spatial resolution`；2x grid 的 Nyquist period 是 `10 um`。
- 2x 是当前 contour-level SR POC 的合理默认网格；4x 只能作为探索/可视化，必须附带 forward model、MTF/SNR 和 contour consistency 证据。
- `effective_snr = DeltaT * MTF / noise` 是必要条件风险图，不是 SR 成功证明。
- 局部内部/外部 contour 温差显著高于噪声底，支持做局部结构可观测性筛选。
- 局部 ESF/CRB 只能用作 alignment anchor 置信度和 quality gate，不能作为最终芯片内部形状重建目标。
- CRB sensitivity 中的 `phase_coverage` 是抽象理论参数，不是 stage command 真值；0.05/0.10 px gate 是乐观下界风险标签，不是实测 alignment 误差。
- Stage command 和文件名坐标只能作为 prior / 初始化 / 正则约束；实际 alignment 必须由 EP04/EP05 的数据驱动门控修正。
- EP06 SR POC 的清晰度指标必须绑定 shape/contour evidence、alignment quality gate、split/holdout 稳定性和可复现实验记录；residual、Tenengrad、effective SNR 或 CRB 都不能单独作为成功证据。

---

## 产物索引

Notebook:

- `notebooks/ep03_theoretical_limits/fragments/`
- `notebooks/ep03_theoretical_limits/ep03_theoretical_limits.ipynb`

报告:

- `reports/ep03_theoretical_limits/theoretical_limits_report.md`

主要输出:

- `output/ep03_theoretical_limits/sampling_resolution_distinction.png`
- `output/ep03_theoretical_limits/pixel_size_measurement.png`
- `output/ep03_theoretical_limits/mtf_psf_frequency_response.png`
- `output/ep03_theoretical_limits/mtf_snr_recoverability_heatmap.png`
- `output/ep03_theoretical_limits/noise_floor_snr_contrast.png`
- `output/ep03_theoretical_limits/local_contour_candidate_map.png`
- `output/ep03_theoretical_limits/local_anchor_confidence_scatter.png`
- `output/ep03_theoretical_limits/crb_esf_localization_anchor.png`
- `output/ep03_theoretical_limits/crb_sensitivity_surface.png`
- `output/ep03_theoretical_limits/*.csv`

执行命令:

```bash
uv run python scripts/build_notebook.py notebooks/ep03_theoretical_limits --execute
```
