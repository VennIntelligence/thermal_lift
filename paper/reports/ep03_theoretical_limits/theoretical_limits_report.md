# EP03 — SR 物理边界与局部可观测性

## 1. Executive Summary

EP03 的定位是为 **2x contour-level SR POC** 建立物理边界、必要条件风险图和局部可观测性准则。它回答四件事：

1. `10 um/pixel` 是 detector sampling pitch，`20 um` 是当前空间分辨率，`5 um` 是 2x output sample；2x grid 的 Nyquist period 是 `10 um`，不等于 `5 um spatial resolution`。
2. PSF/MTF 和 `0.0724 C` 噪声底共同决定高频成分是否还高于噪声；`effective_snr = DeltaT * MTF / noise` 只能作为必要条件风险图。
3. 局部 contour contrast 显著高于噪声底，说明存在可用于 alignment anchor / quality gate 的候选结构，但这不是 SR 成功证明。
4. ESF/CRB 和 CRB sensitivity scan 只能给出乐观理论下界，用于 0.05/0.10 px 级别的局部门控设计，不能替代真实 alignment 误差或 contour/shape evidence。

EP03 不把 stage command 或文件名坐标当作对齐真值；它们只可作为 prior、初始化或正则约束。EP03 也不把局部 ESF/NCC/CRB 诊断外推成全局 SR 成败结论。后续应由 EP05 建立 alignment/phase baseline，并由 EP06 SR POC 围绕 **主 session 内 2x contour/shape evidence、alignment quality gates、结构一致性和可复现实验记录** 验收；不能只看 residual、Tenengrad、CRB 或显示倍率。

## 2. Pixel Pitch、Spatial Resolution 与 Output Grid Nyquist

BMP mm 坐标轴和 TXT/BMP 外轮廓交叉验证确认 detector sampling pitch：

| 证据 | 数值 | 说明 |
|---|---:|---|
| BMP mm-axis ticks | 10.000 um/pixel | 640×480 数据绘图区；1 mm tick spacing = 100 rendered px |
| TXT/BMP contour cross-check | 9.980 um/pixel | 外轮廓 mask IoU = 0.9938 |
| 当前 spatial resolution | 20.000 um | 已校准系统分辨率，不是 detector pitch |

设计尺度表：

| Quantity | Value | Detector pixels | SR role |
|---|---:|---:|---|
| Detector sampling pitch | 10.0 um | 1.00 | LR 温度矩阵采样单位，stage prior 的像素换算单位 |
| Current spatial resolution | 20.0 um | 2.00 | 光学/热扩散传递函数边界 |
| 2x SR grid sample | 5.0 um | 0.50 | 当前 contour-level POC 的默认输出网格 |
| 4x SR grid sample | 2.5 um | 0.25 | 仅作为探索/可视化，需额外 MTF/SNR 证据 |

Output grid sample 与 Nyquist period 必须分开写：

| Grid | Output sample | Nyquist period | Nyquist frequency | 结论边界 |
|---|---:|---:|---:|---|
| 1x | 10.0 um | 20.0 um | 0.5 cyc/detector px | 与当前 20 um spatial resolution 同量级 |
| 2x | 5.0 um | 10.0 um | 1.0 cyc/detector px | 默认 POC grid；不等于 5 um 分辨率 |
| 4x | 2.5 um | 5.0 um | 2.0 cyc/detector px | 仅探索/可视化；不能默认声明可恢复 |

结论：`5 um output sample` 是采样网格声明，不等于 `5 um spatial resolution` 或 5 um 计量级温度读数；2x grid 的 Nyquist period 也只是 10 um 的采样表示边界，不是实际可分辨周期声明。

## 3. PSF/MTF 高频边界

Gaussian PSF MTF 使用 `sigma=0.2/0.35/0.5 px` 扫描。1x/2x/4x grid 的 Nyquist 频率分别为 `0.5/1.0/2.0 cycles per detector pixel`。

| Grid | Grid pitch | Frequency | MTF sigma=0.20 | MTF sigma=0.35 | MTF sigma=0.50 |
|---|---:|---:|---:|---:|---:|
| 1x | 10.0 um | 0.5 cyc/px | 0.821 | 0.546 | 0.291 |
| 2x | 5.0 um | 1.0 cyc/px | 0.454 | 0.089 | 0.007 |
| 4x | 2.5 um | 2.0 cyc/px | 0.042 | 0.000063 | 0.000000003 |

结论：

- 2x grid 在较乐观 PSF 下仍有可用频率余量，在保守 `sigma=0.5 px` 下高频会被强烈衰减，因此 POC 必须聚焦 contour-level improvement 和结构一致性。
- 4x grid 的 Nyquist 频率处衰减更重，默认只能作为 visualization/ablation；除非 forward model、SNR 和 shape evidence 同时支持，否则不应作为交付目标。

## 4. MTF x SNR Recoverability 风险图

Recoverability 使用必要条件：

`effective_snr = DeltaT * MTF(f, sigma) / noise`

代表性 contrast 的 effective SNR：

| Contrast scale | Input SNR | 2x sigma=0.20 | 2x sigma=0.35 | 2x sigma=0.50 | 4x sigma=0.20 | 4x sigma=0.35 | 4x sigma=0.50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nominal edge, 0.7 C | 9.67 | 4.390 | 0.861 | 0.070 | 0.411 | 0.001 | ~0 |
| Inner contour median, 1.9385 C | 26.77 | 12.157 | 2.386 | 0.193 | 1.138 | 0.002 | ~0 |
| Outer contour median, 2.4901 C | 34.39 | 15.616 | 3.064 | 0.247 | 1.462 | 0.002 | ~0 |

正确口径：

- effective SNR 高于 3x/5x 噪声只说明该局部频率成分有机会高于噪声，是进入 POC 的必要条件之一。
- effective SNR 不能证明 alignment 正确，不能证明 SR 输出形状更真实，也不能单独支持 4x 或 5 um 计量级声明。
- 2x 在乐观 PSF 和高对比 contour 下有可观测窗口；在 `sigma=0.5 px` 下即使 measured median contrast 也会大幅衰减，所以 EP05 必须建立局部质量门控和 alignment/phase baseline，EP06 SR POC 必须继续做 split/holdout 稳定性检查。

## 5. Noise Floor 与局部可观测性

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

## 6. ESF/CRB Anchor 置信度

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
- 多帧 CRB 假设相位覆盖、PSF、局部温差和 ESF 模型成立；真实路线必须先在 EP05 用数据驱动对齐建立 baseline，再在 EP06 SR POC 中用 split/holdout 检查和 contour consistency 验证。

## 7. CRB Sensitivity Surface 与 0.05/0.10 px Gate

CRB sensitivity scan 覆盖：

- `DeltaT = 0.3/0.7/1.0/2.0 C`
- `sigma = 0.2/0.35/0.5/1.0 px`
- `n_frames = 1/4/16/64/255`
- abstract `phase_coverage = 0.0/0.5/1.0 px`

`phase_coverage` 是局部 ESF 理论模型里的抽象已知相位覆盖，不是 stage command 真值，也不是 EP03 的位移标定。

`255 frames` 列保留为 EP03 初始理论敏感性网格中的历史采样量参考；当前真实数据 SR 默认输入仍是剔除 `R != 0` 后的 248 帧 clean set。

代表性 `DeltaT=0.7 C` CRB：

| Sigma | Phase coverage | 1 frame | 4 frames | 16 frames | 64 frames | 255 frames |
|---:|---:|---:|---:|---:|---:|---:|
| 0.35 px | 0.0 px | 0.1286 | 0.0643 | 0.0321 | 0.0161 | 0.0081 |
| 0.35 px | 1.0 px | n/a | 0.0564 | 0.0286 | 0.0144 | 0.0072 |
| 0.50 px | 0.0 px | 0.1415 | 0.0707 | 0.0354 | 0.0177 | 0.0089 |
| 0.50 px | 1.0 px | n/a | 0.0684 | 0.0344 | 0.0172 | 0.0086 |

Gate 解读：

| Condition | 0.10 px gate | 0.05 px gate |
|---|---:|---:|
| Single frame, sigma 0.35/0.50 px | scanned DeltaT ≥ 1.0 C | scanned DeltaT ≥ 2.0 C |
| 16 frames, sigma 0.35/0.50 px | scanned DeltaT ≥ 0.3 C | scanned DeltaT ≥ 0.7 C |
| 64+ frames, sigma 0.35/0.50 px | scanned DeltaT ≥ 0.3 C | scanned DeltaT ≥ 0.3 C |

这些数值是乐观理论下界：真实数据中的热漂移、配准误差、非理想 PSF、非高斯噪声和局部结构变化都会让实际误差变大。CRB gate 可以给 EP05 alignment/phase baseline 和 EP06 SR POC 做局部风险标签和 ROI 筛选，不能替代真实 alignment 误差统计。

## 8. EP03 SR 设计边界表

| 状态 | Status | 设计规则/声明 | Design rule / claim | EP05/EP06 约束 |
|---|---|---|---|---|
| 支持 | Supported | 使用 2x 输出网格作为 contour-level POC 默认设置 | Use a 2x output grid for the contour-level POC | 2x 只能作为默认重建网格；不能表述为 5 um 温度计量能力 |
| 支持 | Supported | 显式报告 output sample 与 Nyquist period | Report output sample and Nyquist period explicitly | `5 um sample` 与 `10 um Nyquist period` 都不是 5 um spatial resolution |
| 支持 | Supported | 用 PSF/MTF 与噪声约束预期 | Constrain expectations with PSF/MTF and noise | 报告 contour/shape evidence 与稳定性；不能只报告显示倍率 |
| 支持 | Supported | 用 MTF x SNR effective SNR 作为必要条件风险图 | Use MTF x SNR effective SNR as a necessary-condition risk map | effective SNR 高也不是 SR 成功证明；低则需要更强门控或放弃该频率声明 |
| 支持 | Supported | 用局部 ESF/CRB 作为 alignment-anchor confidence | Use local ESF/CRB as alignment-anchor confidence | SR fusion 前用 anchor confidence 做帧/patch 质量门控 |
| 支持 | Supported | 用 0.05/0.10 px CRB gate 做局部风险标签 | Use 0.05/0.10 px CRB gates as local risk labels | CRB 是乐观理论下界，不能替代真实 alignment 误差统计 |
| 支持 | Supported | stage/file-name shift 只作为 prior | Use stage/file-name shifts only as priors | 用数据驱动 alignment 修正 prior，并拒绝不一致区域 |
| 不支持 | Not supported | 从单个局部诊断推出全局 SR no-go | Draw a global SR no-go conclusion from one local diagnostic | 不能把单个 NCC/ESF/CRB 失败外推成全局 SR 否定 |
| 不支持 | Not supported | 从插值声称 4x 或 5 um 定量分辨率 | Claim 4x or 5 um quantitative resolution from interpolation | 4x 只能作为 visualization/ablation，除非 forward model 与 contour consistency 验证通过 |
| 不支持 | Not supported | 单独用 residual、Tenengrad、effective SNR 或 CRB 证明成功 | Use residual, Tenengrad, effective SNR, or CRB alone as success evidence | 必须配合 shape/contour evidence、split checks 与 alignment quality gates |

## 9. Generated Artifacts

Notebook:

- `notebooks/ep03_theoretical_limits/fragments/`
- `notebooks/ep03_theoretical_limits/ep03_theoretical_limits.ipynb`

Core figures:

- `output/ep03_theoretical_limits/sampling_resolution_distinction.png`
- `output/ep03_theoretical_limits/pixel_size_measurement.png`
- `output/ep03_theoretical_limits/mtf_psf_frequency_response.png`
- `output/ep03_theoretical_limits/noise_floor_snr_contrast.png`
- `output/ep03_theoretical_limits/local_contour_candidate_map.png`
- `output/ep03_theoretical_limits/local_anchor_confidence_scatter.png`
- `output/ep03_theoretical_limits/mtf_snr_recoverability_heatmap.png`
- `output/ep03_theoretical_limits/crb_esf_localization_anchor.png`
- `output/ep03_theoretical_limits/crb_sensitivity_surface.png`

Tables:

- `sampling_resolution_distinction.csv`
- `output_grid_nyquist_periods.csv`
- `pixel_pitch_measurement_summary.csv`
- `mtf_psf_attenuation.csv`
- `mtf_snr_recoverability.csv`
- `mtf_snr_recoverability_gate_summary.csv`
- `local_contour_observability_summary.csv`
- `local_contour_observability_segments.csv`
- `snr_noise_reference.csv`
- `crb_esf_localization_bounds.csv`
- `crb_sensitivity_scan.csv`
- `crb_sensitivity_gate_summary.csv`
