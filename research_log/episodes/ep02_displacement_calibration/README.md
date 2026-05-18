# EP02 — Raster Path, Stage Prior, and Data-Driven Alignment Evidence

> **目标**: 重建 raster 采集路径，给出 detector-space stage prior 覆盖，并区分 stage prior、小步诊断和 data-driven alignment truth
> **状态**: ✅ 已重写为“raster path / stage prior / data-driven alignment evidence”版本
> **前置**: EP01（数据审计完成）

---

## 🧠 核心问题

旋转角 θ = 47.6° 和 stage 命令坐标仍是后续重建的重要先验，但 EP02 的职责不是给出 SR 成败判决，也不是把命令位移当作对齐真值。
本 Episode 的核心任务是把 TXT/BMP raster 采集顺序、坐标 prior 覆盖、相邻小步诊断和 data-driven 对齐证据分层讲清楚。

当前结论：EP02 提供采集路径、坐标 prior 和局部 smoke test；后续 EP06 需要在主 session 上做 data-driven 对齐与质量门控后，再进入 2x contour-level SR。

---

## 📋 当前交付任务清单

- [x] Raster acquisition path 图：按 `acquisition_order` 连线，显示 X 行内连续、Y 行间跳转。
- [x] Stage prior 位移覆盖图：读取 `configs/stage_calibration.json`，映射 X/Y 到 detector dx/dy，并统计 2x phase bin 覆盖。
- [x] 小步诊断图：保留 X 行内时间相邻方向/线性 smoke test，明确 Y-only 坐标相邻 pair 不能标定。
- [x] Data-driven vs filename/stage 对齐比较：读取已有 EP05 alignment score，证明 data-driven contour/NCC 更适合作 alignment quality gate。
- [x] 结论表：输出 stage prior、data-driven alignment、X 小步、Y 坐标相邻 pair、AVI 连续扫描的用途和禁止用途。

## 📚 历史任务记录

### Phase 1: 互相关位移估计

- [x] 1.1 **亚像素互相关实现** — 对相邻坐标帧做互相关，提取亚像素位移 (dx, dy)
- [x] 1.2 **位移向量场可视化** — 绘制所有帧对的位移向量 (dx, dy)
- [x] 1.3 **位移一致性检查** — 同一坐标差的帧对，位移应一致

### Phase 2: 旋转角 θ 拟合

- [x] 2.1 **θ 拟合** — 从位移向量 (dx, dy) vs 坐标差 (ΔX, ΔY) 反推旋转角
- [x] 2.2 **θ 置信区间** — Bootstrap 估计 θ 的不确定度
- [x] 2.3 **与 47.6° 对比** — 判断旧项目参数是否可信

### Phase 3: 位移响应诊断

- [x] 3.1 **重复定位精度** — 利用 3-repeat 坐标评估同一位置的定位误差
- [x] 3.2 **线性度检查** — 实际位移 vs 标称位移的线性回归
- [x] 3.3 **诊断报告** — 记录哪些指标可用、哪些旧指标不能作为对齐真值

### Phase 4: Session 方向假设检查

- [x] 4.1 **按 session / 扫描轴 / 步长拆分方向** — 检查是否存在 session 级运动方向变化
- [x] 4.2 **方向与幅值诊断图** — 输出 `motion_direction_diagnostic.png`

### Phase 5: 时间相邻与轮廓匹配追加验证

- [x] 5.1 **坐标相邻帧对时间间隔审计** — 区分坐标相邻和真实采集相邻
- [x] 5.2 **真实时间相邻帧对构造** — 按 `acquisition_order` 构造连续帧对
- [x] 5.3 **raw / high-pass / gradient / phase 方法对照** — 比较轮廓/高通预处理是否改善物理一致性
- [x] 5.4 **追加结论写入 Notebook 和报告** — 输出 time-adjacent addendum

### Phase 6: AVI 连续运动配准验证

- [x] 6.1 **AVI 热像区域裁剪** — 从 839×560 AVI 帧中裁剪 640×480 热像区域，排除坐标轴/色条
- [x] 6.2 **重复帧去除与运动段检测** — 按帧差去重，自动识别连续运动段
- [x] 6.3 **连续帧 NCC 配准** — 对 x/y-scan AVI 的唯一连续帧做 high-pass NCC
- [x] 6.4 **轮廓方法对照** — 对同一批 AVI 追加 gradient NCC，区分热纹理与轮廓证据

### Phase 7: AVI-TXT Y 线对应验证

- [x] 7.1 **yN.avi 文件名解释检查** — 同时测试 `yN -> TXT fixed X=N` 和 `yN -> TXT fixed Y=N`
- [x] 7.2 **外轮廓路径验证** — 用中心 ROI Otsu 轮廓质心检查 TXT Y 线是否单调
- [x] 7.3 **TXT-AVI 方向对齐验证** — 比较 high-pass / gradient NCC 轴方向是否与 AVI 一致
- [x] 7.4 **排序/命名/时间 gap 判因** — 区分命名错误、排序错误和非时间相邻热漂移

### Phase 8: AVI-TXT X 线对应验证

- [x] 8.1 **xN.avi 文件名解释检查** — 同时测试 `xN -> TXT fixed Y=N` 和 `xN -> TXT fixed X=N`
- [x] 8.2 **X 侧分类规则验证** — 验证 AVI 前缀是运动轴、数字是固定正交坐标
- [x] 8.3 **X 侧 TXT/AVI 轴差诊断** — 量化 x-scan AVI 与 TXT fixed-Y 行之间的系统角差

### Phase 9: AVI θ 独立验证

- [x] 9.1 **逐 AVI θ 估计** — 从 X/Y-scan AVI 方向反推 θ，16 个独立样本
- [x] 9.2 **θ 统计分析** — 均值、CI、与 47.6° 比较
- [x] 9.3 **森林图** — 逐 AVI θ 估计可视化

---

## 2026-05-17 首轮坐标相邻 NCC 失败诊断

### 关键结果

| 项目 | 结果 |
|------|------|
| 主扫描相邻帧对 | 463 |
| NCC 峰值中位数 | 0.99545 |
| image-row θ 拟合 | 145.690° |
| y-up 诊断 θ 拟合 | 34.053° |
| y-up 95% Bootstrap CI | [33.226°, 34.829°] |
| 47.6° 是否落入 CI | 否 |
| 旧坐标相邻旋转模型 RMS 残差 | 0.1567 px |
| repeat 有效帧对 | 0 / 2 主扫描 repeat pairs |
| repeat 有效 p95 | n/a |
| 线性投影 R² | 0.0001 |
| Y-scan 2 µm 中位幅值 | 0.3515 px |
| Y-scan 4 µm 中位幅值 | 0.2262 px |

### 决策记录

- **不更新** `configs/stage_calibration.json`。
- EP01 已确认旧版 13 sessions 是文件名排序伪影；EP02 已改为只使用主扫描 session=2 的 255 帧数据。
- 当前 NCC 测量仍不能独立验证 θ=47.6°，也不能给出可直接采用的新 θ。
- 主要失败模式仍是 Y-scan：2 µm 与 4 µm 命令步长的实测位移不呈线性倍增，说明问题不是早期低温/补采帧污染。
- 主扫描内 X-scan 方向约 -35° 到 -32°，Y-scan 方向约 52° 到 54°；更大的异常是 Y-scan 4 µm 的实测幅值小于 2 µm。
- 下一步优先排查 Y-scan 帧对构造、扫描回程/反向间隙、轴符号约定，以及 NCC 是否被温度场变化系统性偏置。
- repeatability 只保留同属主扫描 session 且 `fit_ok=True`、非边界峰的 pair；当前 2 个主扫描 repeat pair 都是边界峰，不能给出有效 repeat p95。
- 0.1567 px 是旧坐标相邻 NCC 模型的失败诊断值；它不再被解释为 2×/4× SR 阈值判据，也不作为 SR 成败结论。

## 2026-05-17 追加验证：时间相邻 + 轮廓/高通

### 关键结果

| 项目 | 结果 |
|------|------|
| X-scan 坐标帧对 acquisition gap | 全部 1 帧 |
| Y-scan 2 µm 坐标帧对 acquisition gap | 中位 16 帧 |
| Y-scan 4 µm 坐标帧对 acquisition gap | 中位 16 帧，最大 31 帧 |
| 真实时间相邻 R=0 X 小步 | 232 对 |
| 真实时间相邻 R=0 换行回程 | 15 对 |
| best X-step 方法 | highpass_ncc |
| highpass_ncc X-step visible/nominal projection | 0.475 |
| highpass_ncc X-step RMS vs fixed 10 µm/pixel model | 0.1584 px |
| raw_ncc X-step visible/nominal projection | 0.506 |
| raw_ncc X-step RMS vs fixed 10 µm/pixel model | 0.1501 px |
| raw_ncc row-transition RMS vs fixed 10 µm/pixel model | 1.8748 px |
| phase correlation X-step | 退化为 0 px，不适合本数据的极小位移 |

### 追加决策记录

- 首轮 EP02 的 Y-scan 异常有明确的时间采集混杂解释：Y 坐标相邻帧并不是真实连续采集帧。
- 当前主扫描是 raster 路径：行内 X 连续小步；Y 增量只在换行时出现，并与 X 大回程绑定。
- BMP 标尺确认 TXT 采样 pitch 为 10 µm/pixel 后，2 µm 命令步在名义坐标模型中对应 0.20 px；时间相邻 X 小步在 raw/high-pass NCC 下的可见投影约 0.10 px。这个数值只描述当前 ROI/预处理下的可见图像响应，不能推出全局物理位移“只有一半”。
- X 小步仍形成紧密簇且 4/2 投影比接近 2，说明短时间内的方向、采集顺序和相对线性比首轮全局坐标相邻模型更可信；但它不支持把 stage 幅值直接作为对齐真值。
- 现有数据仍不能独立验证纯 Y 轴 2 µm 微步，也不能给出可替换全局配置的新 θ。
- 若继续追求 Y 方向直接测量，需要新的采集设计：固定 X、连续 Y 小步、同一热状态、记录真实采集顺序。

## 2026-05-17 追加验证：Y-only 轮廓/高通对照

### 关键结果

| 方法 | Y 2 µm 投影比例 | Y 4 µm 投影比例 | 4/2 投影比 | Y 2 µm RMS | Y 4 µm RMS |
|------|----------------:|----------------:|-----------:|-----------:|-----------:|
| raw_ncc | 1.724 | 0.550 | 0.638 | 0.1810 px | 0.1921 px |
| highpass_ncc | 1.601 | 0.511 | 0.638 | 0.1571 px | 0.2027 px |
| gradient_ncc | 1.589 | 0.511 | 0.643 | 0.1586 px | 0.2063 px |

### 追加决策记录

- 三种方法的 Y 方向理论投影均为正，说明文件名 Y 坐标仍保留弱方向意义。
- 但所有方法都不满足单调性：Y 4 µm 的投影只有 Y 2 µm 的约 0.64 倍，而不是约 2 倍。
- high-pass/gradient 能略微降低残差，但无法修复非单调问题；在 10 µm/pixel 固定模型下 Y 2 µm 和 4 µm RMS 约 0.16-0.20 px，只能作为失败模式记录，不能作为对齐真值标定。
- 结论：当前 Y-only 坐标相邻帧对不能作为定量位移输入；它们只能作为命令坐标元数据和失败诊断依据。

## 2026-05-17 追加验证：AVI 连续运动配准

### 关键结果

| 项目 | X-scan AVI | Y-scan AVI |
|------|-----------:|-----------:|
| AVI 数量 | 8 | 8 |
| 重复帧率中位数 | 66.9% | 67.2% |
| 运动帧对数中位数 | 202 | 198 |
| high-pass NCC 方向中位数（row-down） | 51.67° | 131.07° |
| high-pass NCC 方向范围（row-down） | 51.15°–51.84° | 130.56°–131.37° |
| high-pass NCC 帧间幅值中位数 | 0.0884 px | 0.0758 px |
| high-pass NCC 路径直线性最小值 | 0.9997 | 0.9847 |
| high-pass NCC 峰值中位数 | 0.9120 | 0.9099 |
| gradient NCC 方向中位数（row-down） | 48.70° | 135.63° |
| gradient NCC 帧间幅值中位数 | 0.1103 px | 0.0943 px |

### 追加决策记录

- AVI 不是 SR 输入，但可作为连续运动诊断：每个视频都有清晰的静止→连续运动→静止结构，运动段内路径近似直线。
- X/Y 仪器正交性作为硬件事实，不在本次分析中验证；AVI 输出只记录图像中连续运动方向、速度稳定性和异常段。
- high-pass/raw 跟踪热纹理，gradient 更接近“只看轮廓”。两类方法都显示连续运动稳定，但绝对角度有数度差异，因此 AVI 角度暂不直接替换 `configs/stage_calibration.json`。
- Y-scan AVI 明确提供了连续 Y 运动证据：Y 运动段不是杂乱无章，也没有明显反向跳变；这支持“TXT Y-only 坐标相邻失败主要来自非时间相邻帧对和热场演化”，而不是 Y 轴完全不可用。
- `y14um.avi` 的路径直线性略低（high-pass 0.9847，gradient 0.9826），应作为轻微异常视频保留标记，但不改变整体判断。

### 产物

- 脚本: `scripts/avi_y_direction_check.py`
- 报告: `reports/ep02_displacement_calibration/avi_registration_addendum.md`
- high-pass 汇总: `output/ep02_displacement_calibration/avi_direction_summary.csv`
- high-pass 逐帧对: `output/ep02_displacement_calibration/avi_registration_pairs.csv`
- high-pass 图: `output/ep02_displacement_calibration/avi_direction_comparison.png`、`avi_cumulative_motion_paths.png`、`avi_y0um_displacement_timeseries.png`
- gradient 对照: `output/ep02_displacement_calibration/avi_gradient_check/`

## 2026-05-17 追加验证：AVI-TXT Y 线对应

### 关键结果

| 映射假设 | 轮廓轴差 | high-pass NCC 轴差 | gradient NCC 轴差 | 采集间隔 |
|----------|---------:|-------------------:|------------------:|---------:|
| `yN.avi -> TXT fixed X=N` | 中位 1.86° | 中位 5.28° | 中位 6.24° | 中位 16 帧 |
| `yN.avi -> TXT fixed Y=N` | 中位 82.48° | 中位 85.78° | 中位 81.66° | 中位 1 帧 |

### 追加决策记录

- `y0um.avi`、`y2um.avi`、... 明确对应 TXT 的固定 `X=0/2/...` Y 线，而不是固定 `Y=0/2/...` X 行。
- TXT fixed-X 外轮廓随 Y 坐标单调移动；所有测试线的轮廓单调比例为 1.0。
- 因此没有发现全局 Y 命名反了、行列解释反了、或按文件名/采集顺序导致的整体排序错误。
- Y-only TXT NCC 的异常主要来自 raster 采集：固定 X 的相邻 Y 坐标之间 acquisition gap 中位为 16 帧；这段时间内热场演化会偏置强度 NCC 幅值。
- 已知缺失 R=0 点为 `(14,6,0)`、`(16,6,0)`、`(16,16,0)`；`14_6_0.txt` 缺失导致 `X=14` 线最大 gap 达 30 帧。
- 结论：TXT 的 Y 坐标可以保留为命令坐标/顺序元数据；不能把 Y-only 坐标相邻 TXT NCC 当作定量 Y 位移标定。

### 产物

- 脚本: `scripts/avi_txt_yline_match_check.py`
- 报告: `reports/ep02_displacement_calibration/avi_txt_yline_match_addendum.md`
- 汇总: `output/ep02_displacement_calibration/avi_txt_yline_match_summary.csv`
- 逐帧对: `output/ep02_displacement_calibration/avi_txt_yline_pair_measurements.csv`
- 图: `avi_txt_yline_axis_match.png`、`avi_txt_yline_projection_monotonicity.png`、`avi_txt_yline_contour_paths.png`

## 2026-05-17 追加验证：AVI-TXT X 线对应

### 关键结果

| 映射假设 | 轮廓轴差 | high-pass NCC 轴差 | gradient NCC 轴差 | 采集间隔 |
|----------|---------:|-------------------:|------------------:|---------:|
| `xN.avi -> TXT fixed Y=N` | 中位 18.41° | 中位 14.19° | 中位 11.36° | 中位 1 帧 |
| `xN.avi -> TXT fixed X=N` | 中位 77.20° | 中位 74.77° | 中位 80.79° | 中位 16 帧 |

### 追加决策记录

- X 侧分类规则与 Y 侧对称：AVI 前缀表示运动轴，数字表示固定的正交坐标。
- `x.avi` 可视为 fixed `Y=0` 的 X 扫描；`x2um.avi`、`x4um.avi`、... 对应 TXT fixed `Y=2/4/...` 行。
- 与 Y 侧不同，X 侧 TXT fixed-Y 行本身 acquisition gap 为 1，因此 TXT X 小步 NCC 仍是 EP02 最强短时运动诊断证据。
- x-scan AVI 与 TXT fixed-Y 行存在约 11–18° 的系统轴差；因此 x-scan AVI 适合作文件名/分类诊断，不应替代 TXT 时间相邻 X 诊断。

### 产物

- 脚本: `scripts/avi_txt_xline_match_check.py`
- 报告: `reports/ep02_displacement_calibration/avi_txt_xline_match_addendum.md`
- 汇总: `output/ep02_displacement_calibration/avi_txt_xline_match_summary.csv`
- 逐帧对: `output/ep02_displacement_calibration/avi_txt_xline_pair_measurements.csv`
- 图: `avi_txt_xline_axis_match.png`、`avi_txt_xline_projection_monotonicity.png`

## 2026-05-17 追加验证：AVI θ 独立验证

### 关键结果

| 项目 | 值 |
|------|----|
| gradient θ_X 中位数 | 48.70° |
| gradient θ_Y 中位数 | 45.63° |
| gradient combined 均值 | 47.14° |
| gradient combined 95% CI | [46.36°, 47.92°] |
| 47.6° 是否在 CI 内 | 是 |
| high-pass θ_X 中位数 | 51.67° |
| high-pass θ_Y 中位数 | 41.07° |

### 追加决策记录

- AVI 方向重新解读为 θ 估计后，gradient NCC combined 均值为 47.14°（距 47.6° 为 0.46°），是 EP02 首次独立 θ 方向验证。
- X-scan 和 Y-scan 的 θ 估计之间存在约 3.06° 差异，可能来自 AVI 渲染几何、crop 坐标、NCC 预处理或连续视频生成链路的系统偏差。
- 两个方向的估计包夹了 47.6°，且 combined 95% CI 覆盖 47.6°，支持当前 θ 配置的合理性。
- 建议：将 AVI θ 验证结果记录为辅助证据，但仍不替换 `configs/stage_calibration.json`（因为 AVI 是渲染后视频，绝对角度精度有限）。

### 产物

- 脚本: `scripts/avi_theta_estimation.py`
- 估计表: `output/ep02_displacement_calibration/avi_theta_estimates.csv`
- 汇总: `output/ep02_displacement_calibration/avi_theta_summary.csv`
- 结果: `output/ep02_displacement_calibration/avi_theta_result.json`
- 图: `output/ep02_displacement_calibration/avi_theta_forest_plot.png`

## 2026-05-18 Notebook/Report 重写

- EP02 notebook 已重写为“raster path / stage prior / data-driven alignment evidence”版本。
- 新叙事顺序：raster acquisition path → stage prior detector coverage and 2x phase bins → small-step smoke tests → data-driven vs filename/stage alignment comparison → evidence-use decision table。
- 核心图包括：`ep02_raster_acquisition_path.png`、`ep02_stage_prior_coverage.png`、`ep02_small_step_smoke_tests.png`、`ep02_data_driven_alignment_comparison.png`。
- 关键数值：主 session 255 帧、R=0 raster 248 帧、X 行内时间相邻转移 232 对、Y 坐标相邻 gap 中位数 16 帧；stage prior 四个 2x phase bin 均非空。
- alignment 口径：已有 EP05 score 显示 data-driven contour refined 相对 stage-prior-only 的 holdout Chamfer 中位误差下降 44.2%，因此 stage/filename 坐标只作为 prior，alignment truth 由 data-driven contour/NCC 质量指标给出。
- 正式结论：EP02 不把 stage command 当成位移真值，也不从 2 um 小步外推多帧 SR 成败；EP06 应在主 session 上做 data-driven 对齐与质量门控后进入 2x contour-level SR。

### 产物

- Notebook: `notebooks/ep02_displacement_calibration/ep02_displacement_calibration.ipynb`
- 报告: `reports/ep02_displacement_calibration/calibration_report.md`
- 输出目录: `output/ep02_displacement_calibration/`

---

## 📂 相关文件

- 依赖: `output/ep01_data_processing/frame_audit.csv`
- Notebook: `notebooks/ep02_displacement_calibration/`
- 报告: `reports/ep02_displacement_calibration/`
- 输出: `output/ep02_displacement_calibration/`
