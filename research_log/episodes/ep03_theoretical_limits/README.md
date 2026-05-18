# EP03 — SR 物理边界与局部可观测性

> **目标**: 重建 EP03 为 2x contour-level SR POC 的物理边界说明：区分 detector pitch / spatial resolution / SR grid，量化 PSF/MTF 与噪声底约束，说明 ESF/CRB 只能作为 alignment anchor 与 quality gate。
> **状态**: ✅ Notebook 与报告已按新口径重建并执行
> **前置**: EP01 数据审计；EP02 stage prior 标定；EP04 localization 作为后续 alignment gate

---

## 任务清单

- [x] 重写 `core/src/thermal_core/ep03.py` 为 EP03 物理模型和可观测性工具。
- [x] 重写 `notebooks/ep03_theoretical_limits/fragments/`，移除旧的局部 joint-ESF 失配主线。
- [x] 生成 pixel pitch / spatial resolution / 2x grid 区分图。
- [x] 生成 MTF/PSF 曲线，覆盖 `sigma=0.2/0.35/0.5 px` 和 1x/2x/4x 频率。
- [x] 生成 noise floor / local contrast SNR 图、局部 contour 空间图和 anchor confidence 散点图。
- [x] 生成 ESF/CRB localization anchor 图。
- [x] 在 Notebook 中输出 EP03 中英双语 SR 设计边界表。
- [x] 更新 `scripts/measure_pixel_size.py`，保留 BMP/TXT pitch 测量，去除旧的局部位移尺度叙事。
- [x] 更新正式报告 `reports/ep03_theoretical_limits/theoretical_limits_report.md`。
- [x] 执行 `uv run python scripts/build_notebook.py notebooks/ep03_theoretical_limits --execute`。

---

## 关键结果

| 项目 | 结果 |
|---|---:|
| BMP mm-axis detector pitch | 10.000 um/pixel |
| TXT/BMP contour cross-check pitch | 9.980 um/pixel |
| TXT/BMP outer-mask IoU | 0.9938 |
| Current spatial resolution | 20.000 um |
| 2x SR grid sample | 5.000 um |
| 4x exploratory grid sample | 2.500 um |
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

---

## 决策记录

- `10 um/pixel` 是 TXT detector sampling pitch；`20 um` 是当前系统空间分辨率，不得混写。
- `5 um` 目标至少要求 2x output grid，但 2x grid 不是 5 um 计量级空间分辨率声明。
- 2x 是当前 contour-level SR POC 的合理默认网格；4x 只能作为探索/可视化，必须附带 forward model、MTF/SNR 和 contour consistency 证据。
- 局部内部/外部 contour 温差显著高于噪声底，支持做局部结构可观测性筛选。
- 局部 ESF/CRB 只能用作 alignment anchor 置信度和 quality gate，不能作为最终芯片内部形状重建目标。
- Stage command 和文件名坐标只能作为 prior / 初始化 / 正则约束；实际 alignment 必须由 EP04/EP05 的数据驱动门控修正。
- EP05 清晰度指标必须绑定 shape/contour evidence、alignment quality gate、split/holdout 稳定性和可复现实验记录；residual 或 Tenengrad 不能单独作为成功证据。

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
- `output/ep03_theoretical_limits/noise_floor_snr_contrast.png`
- `output/ep03_theoretical_limits/local_contour_candidate_map.png`
- `output/ep03_theoretical_limits/local_anchor_confidence_scatter.png`
- `output/ep03_theoretical_limits/crb_esf_localization_anchor.png`
- `output/ep03_theoretical_limits/*.csv`

执行命令:

```bash
uv run python scripts/build_notebook.py notebooks/ep03_theoretical_limits --execute
```
