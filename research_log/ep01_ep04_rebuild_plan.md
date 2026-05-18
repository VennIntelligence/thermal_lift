# EP01-EP04 Rebuild Plan for the 2x SR Direction

> 目的：按当前 2x contour-level SR 路线重新定义 EP01-EP04 的任务边界、计算产物和可视化。生成的 `.ipynb` 已删除；后续 notebook 应从 fragments 重新构建，不恢复旧输出。

## Narrative

新叙事顺序：

1. **EP01** 证明数据能用。
2. **EP02** 证明 stage command 只能作 prior。
3. **EP03** 证明 contour / internal structure 有可观测热信号，并给出物理风险边界。
4. **EP04** 证明哪些 edge / contour segment 可作为 alignment anchor。
5. **EP06** 首次做真正的 2x contour-level SR POC。

## EP01 — Data Foundation

任务：

- 文件解码、TXT/BMP 配对、坐标覆盖。
- 恢复 `acquisition_order`。
- 标注 3 个温度段和 255 帧主 session。
- 统计主 session drift、noise floor、缺失坐标。

不做：

- 位移标定、theta 拟合、localization、SR。

产物：

- `frame_audit.csv`
- `rename_mapping.csv`
- `bmp_txt_pairing.csv`
- `coordinate_coverage.csv`
- `session_summary.json`
- `noise_floor_summary.json`

图：

- 坐标覆盖 heatmap。
- acquisition-order raster 轨迹。
- 温度时间线 + session 分段。
- missing coordinate map。
- 主 session drift / noise 诊断。

## EP02 — Raster and Stage Prior Diagnostics

任务：

- 说明 raster 采集路径。
- 区分 time-adjacent 与 coordinate-adjacent。
- 生成 per-frame stage-command prior。
- 记录 X 行内短时 NCC/gradient 证据。
- 解释 Y-only coordinate-adjacent 失败原因。
- 保留 AVI 方向 sanity check。

不做：

- 不给对齐真值。
- 不改全局 theta。
- 不裁判 SR 成败。

产物：

- `stage_prior_by_frame.csv`
- `frame_pair_table.csv`
- `time_gap_audit.csv`
- `x_time_adjacent_registration.csv`
- `y_coordinate_failure_diagnostic.csv`
- `avi_theta_summary.csv`
- `evidence_hierarchy.json`

图：

- raster acquisition trajectory。
- coordinate-adjacent vs time-adjacent gap。
- X 小步 visible displacement vs command。
- Y-only projection / gap 诊断。
- AVI theta forest plot。
- evidence hierarchy 总结图。

## EP03 — Physical and Contour Observability

任务：

- 建立 CRB / PSF / SNR 风险边界。
- 构建外轮廓和内轮廓候选库。
- 建立单帧 ESF baseline。
- 给 EP04 输出 candidate manifest。

不做：

- 不做 SR 重建。
- 不用局部 stage-prior 负控裁判全局 SR。

产物：

- `physical_limits.json`
- `crb_grid.csv`
- `outer_contour_segments.csv`
- `inner_contour_segments.csv`
- `single_frame_esf_fits.csv`
- `psf_sensitivity.csv`
- `contour_candidate_manifest.json`

图：

- CRB heatmap / contour plot。
- PSF sensitivity。
- 外轮廓 candidate overlay。
- 内轮廓 candidate map。
- SNR / DeltaT / normal angle 分布。
- 单帧 ESF profile + residual panels。

## EP04 — Alignment Anchor Quality Gate

任务：

- 在主 session 上验证哪些 contour segment 可作为 alignment anchor。
- 计算 data-driven NCC phase、joint ESF split-half、CRB ratio。
- 输出 anchor catalog 和 failure taxonomy。

不做：

- 不输出 SR 图。
- 不做 LR/bicubic/SR 对照。
- 不把 localization 当客户最终交付。

产物：

- `segment_validation_results.csv`
- `segment_summary.csv`
- `anchor_catalog.csv`
- `gate_config.json`
- `global_summary.json`
- `failure_taxonomy.csv`

图：

- split-half distribution。
- phase coverage vs split-half scatter。
- CRB ratio scatter。
- quality-gated contour map。
- failure taxonomy bar chart。
- cross-scanline consistency。
- outer vs inner anchor coverage map。

## EP06 Boundary

EP06 才开始真正 2x contour-level SR POC：

- ROI 选择。
- 对齐策略比较。
- drift / offset / gain 处理。
- forward model 与 regularization。
- LR / bicubic 2x / simple stack / 2x SR 并排图。
- held-out Chamfer、gradient correlation、split-half SR consistency、artifact audit。
