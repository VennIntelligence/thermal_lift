# EP03 — SR 物理边界与局部可观测性

## 1. Executive Summary

EP03 的新定位是：为 **2x contour-level SR POC** 建立物理边界和局部可观测性准则。它回答三件事：

1. `10 um/pixel` 是 detector sampling pitch，`20 um` 是当前空间分辨率，`5 um` 目标至少需要 2x output grid。
2. PSF/MTF 和 `0.0724 C` 噪声底共同决定可恢复高频上限；2x 是合理 POC，4x 需要额外证据。
3. 局部 ESF/CRB 能作为 alignment anchor 和 quality gate 的置信度来源，但不是最终形状重建目标。

EP03 不把 stage command 或文件名坐标当作对齐真值；它们只作为位移 prior、初始化或正则约束。EP03 也不把局部 ESF/NCC 诊断外推成全局 SR 结论。后续 EP05 的验收应围绕 **主 session 内 2x contour/shape evidence、alignment quality gates、结构一致性和可复现实验记录**，不能只看 residual、Tenengrad 或显示倍率。

## 2. Pixel Pitch 与 Spatial Resolution

BMP mm 坐标轴和 TXT/BMP 外轮廓交叉验证确认 detector sampling pitch：

| 证据 | 数值 | 说明 |
|---|---:|---|
| BMP mm-axis ticks | 10.000 um/pixel | 640×480 数据绘图区；1 mm tick spacing = 100 rendered px |
| TXT/BMP contour cross-check | 9.980 um/pixel | 外轮廓 mask IoU = 0.9938 |
| 当前 spatial resolution | 20.000 um | 已校准系统分辨率，不是 detector pitch |

设计含义：

| Quantity | Value | Detector pixels | SR role |
|---|---:|---:|---|
| Detector sampling pitch | 10.0 um | 1.00 | LR 温度矩阵采样单位，stage prior 的像素换算单位 |
| Current spatial resolution | 20.0 um | 2.00 | 光学/热扩散传递函数边界 |
| 2x SR grid sample | 5.0 um | 0.50 | 当前 contour-level POC 的默认输出网格 |
| 4x SR grid sample | 2.5 um | 0.25 | 仅作为探索/可视化，需额外 MTF/SNR 证据 |

结论：`5 um grid` 是采样网格声明，不等于 `5 um spatial resolution` 或 5 um 计量级温度读数。

## 3. PSF/MTF 高频边界

Gaussian PSF MTF 使用 `sigma=0.2/0.35/0.5 px` 扫描。1x/2x/4x grid 的 Nyquist 频率分别为 `0.5/1.0/2.0 cycles per detector pixel`。

| Grid | Grid pitch | Frequency | MTF sigma=0.20 | MTF sigma=0.35 | MTF sigma=0.50 |
|---|---:|---:|---:|---:|---:|
| 1x | 10.0 um | 0.5 cyc/px | 0.821 | 0.546 | 0.291 |
| 2x | 5.0 um | 1.0 cyc/px | 0.454 | 0.089 | 0.007 |
| 4x | 2.5 um | 2.0 cyc/px | 0.042 | 0.000063 | 0.000000003 |

结论：

- 2x grid 在较乐观 PSF 下仍有可用频率余量，在保守 `sigma=0.5 px` 下高频会被强烈衰减，因此 POC 必须聚焦 contour-level improvement 和结构一致性。
- 4x grid 的 Nyquist 频率处衰减更重，默认只能作为可视化/ablation；除非 forward model、SNR 和 shape evidence 同时支持，否则不应作为交付目标。

## 4. Noise Floor 与局部可观测性

噪声底为 `0.0724 C`。参考温差和实测轮廓温差对应的 SNR：

| Contrast scale | Delta T | SNR |
|---|---:|---:|
| Noise floor | 0.0724 C | 1.00 |
| 3x noise gate | 0.2172 C | 3.00 |
| Weak local contrast | 0.3000 C | 4.14 |
| Nominal edge contrast | 0.7000 C | 9.67 |
| Strong local contrast | 1.0000 C | 13.81 |
| Inner contour median | 1.9385 C | 26.77 |
| Outer contour median | 2.4901 C | 34.39 |

本次参考帧的局部 contour observability：

| Source | Segments | Median Delta T | Median SNR | SNR > 5 fraction | Anchor candidate fraction | Median normal projection |
|---|---:|---:|---:|---:|---:|---:|
| Inner | 386 | 1.938 C | 26.77 | 95.3% | 35.0% | 0.823 |
| Outer | 84 | 2.490 C | 34.39 | 94.0% | 33.3% | 0.346 |

结论：

- 噪声底不是对 2x contour-level POC 的直接阻断项；许多内部/外部结构的局部温差显著高于噪声。
- 可观测性是局部属性。高温差不等于高质量 anchor，还必须检查法线投影、曲率、局部稳定性和 EP04 alignment gate。

## 5. ESF/CRB Anchor 置信度

1D ESF CRB 用来定义局部 edge anchor 的理论下界：

| Delta T | Sigma PSF | Single-frame CRB | 16-frame known-phase CRB |
|---:|---:|---:|---:|
| 0.3 C | 0.5 px | 0.3301 px | 0.0802 px |
| 0.7 C | 0.5 px | 0.1415 px | 0.0344 px |
| 1.0 C | 0.5 px | 0.0990 px | 0.0241 px |
| 2.0 C | 0.5 px | 0.0495 px | 0.0120 px |
| 0.7 C | 1.0 px | 0.1947 px | 0.0487 px |
| 2.0 C | 1.0 px | 0.0682 px | 0.0170 px |

正确口径：

- CRB/ESF 能告诉我们某个局部边缘是否适合作为 alignment anchor 或质量门控区域。
- CRB/ESF 不能单独证明芯片内部形状已经被 SR 重建。
- 多帧 CRB 假设相位覆盖、PSF、局部温差和 ESF 模型成立；真实 EP05 必须用数据驱动对齐、split/holdout 检查和 contour consistency 验证。

## 6. EP03 SR 设计边界表

| 状态 | Status | 设计规则/声明 | Design rule / claim | EP05/EP06 约束 |
|---|---|---|---|---|
| 支持 | Supported | 使用 2x 输出网格作为 contour-level POC 默认设置 | Use a 2x output grid for the contour-level POC | 2x 只能作为默认重建网格；不能表述为 5 um 温度计量能力 |
| 支持 | Supported | 用 PSF/MTF 与噪声约束预期 | Constrain expectations with PSF/MTF and noise | 报告 contour/shape evidence 与稳定性；不能只报告显示倍率 |
| 支持 | Supported | 用局部 ESF/CRB 作为 alignment-anchor confidence | Use local ESF/CRB as alignment-anchor confidence | SR fusion 前用 anchor confidence 做帧/patch 质量门控 |
| 支持 | Supported | stage/file-name shift 只作为 prior | Use stage/file-name shifts only as priors | 用数据驱动 alignment 修正 prior，并拒绝不一致区域 |
| 不支持 | Not supported | 从单个局部诊断推出全局 SR no-go | Draw a global SR no-go conclusion from one local diagnostic | 不能把单个 NCC/ESF 失败外推成全局 SR 否定 |
| 不支持 | Not supported | 从插值声称 4x 或 5 um 定量分辨率 | Claim 4x or 5 um quantitative resolution from interpolation | 4x 只能作为 visualization/ablation，除非 forward model 与 contour consistency 验证通过 |
| 不支持 | Not supported | 单独用 residual 或 Tenengrad 证明成功 | Use residual or Tenengrad alone as success evidence | 必须配合 shape/contour evidence、split checks 与 alignment quality gates |

## 7. Generated Artifacts

Notebook:

- `notebooks/ep03_theoretical_limits/fragments/`
- `notebooks/ep03_theoretical_limits/ep03_theoretical_limits.ipynb`

Core outputs:

- `output/ep03_theoretical_limits/sampling_resolution_distinction.png`
- `output/ep03_theoretical_limits/pixel_size_measurement.png`
- `output/ep03_theoretical_limits/mtf_psf_frequency_response.png`
- `output/ep03_theoretical_limits/noise_floor_snr_contrast.png`
- `output/ep03_theoretical_limits/local_contour_candidate_map.png`
- `output/ep03_theoretical_limits/local_anchor_confidence_scatter.png`
- `output/ep03_theoretical_limits/crb_esf_localization_anchor.png`

Tables:

- `sampling_resolution_distinction.csv`
- `pixel_pitch_measurement_summary.csv`
- `mtf_psf_attenuation.csv`
- `local_contour_observability_summary.csv`
- `local_contour_observability_segments.csv`
- `snr_noise_reference.csv`
- `crb_esf_localization_bounds.csv`
