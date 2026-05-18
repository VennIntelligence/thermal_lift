# EP01 — SR 数据基础与主 session 建模

> **目标**: 把原始 LWIR TXT/BMP 数据整理成后续 2x contour-level SR 可直接继承的输入模型。
> **状态**: ✅ 已重写为主 session 数据基础审计
> **起始日期**: 2026-05-16
> **当前口径**: EP01 不判断 SR 是否可行；EP01 只定义哪些帧可一起用于重建、按什么顺序使用、哪些温度段必须隔离。

---

## 核心问题

1. **原始帧是否完整可读？**
   263 个 TXT 温度矩阵全部可读，矩阵尺寸一致为 480×640，无 NaN/Inf。

2. **TXT/BMP 是否一一对应？**
   263 个 TXT 与 263 个 BMP 完整配对。TXT 是数值重建输入，BMP 是同名视觉参考。

3. **坐标覆盖是否足够支撑主 session SR？**
   全数据覆盖 253/256 个实际坐标；缺失坐标为 (14,6)、(16,6)、(16,16)。主 session 同样覆盖这 253 个实际坐标。

4. **真实采集顺序是什么？**
   文件名不是时序。后续必须使用 `acquisition_order`/mtime，而不是按 `X_Y_R` 文件名字母序排序。

5. **哪些帧可用于默认 SR POC？**
   采集顺序下检测到 3 个温度段；主 session 为 session=2，共 255 帧，是默认 2x contour-level SR 输入。

---

## 任务清单

### Phase 1: 文件清单与矩阵审计 ✅

- [x] 扫描 TXT/BMP 文件清单
- [x] 验证 TXT/BMP 配对关系
- [x] 读取全部 TXT 温度矩阵
- [x] 审计矩阵尺寸、NaN/Inf、温度范围
- [x] 补充逐帧中位温、5-95% trimmed mean 等稳健温度统计

### Phase 2: 坐标/R 覆盖建模 ✅

- [x] 统计 `(X,Y,R)` 分布
- [x] 验证缺失坐标和重复坐标
- [x] 绘制全数据坐标覆盖 heatmap
- [x] 绘制主 session vs 其他 session 坐标覆盖 heatmap

### Phase 3: 采集顺序与 session 建模 ✅

- [x] 基于 `acquisition_order`/mtime 建立真实采集顺序
- [x] 绘制文件名顺序 vs 采集顺序温度曲线对比图
- [x] 检测采集顺序下的 3 个温度段
- [x] 标记 `session` 与 `is_main_session`
- [x] 输出主 session 255 帧作为默认 SR 输入规则

### Phase 4: 文档与产物 ✅

- [x] 更新 notebook fragments
- [x] 构建并执行 EP01 notebook
- [x] 更新正式报告 `reports/ep01_data_processing/audit_report.md`
- [x] 输出机器可读 CSV 到 `output/ep01_data_processing/`

---

## 关键数据指标

| 指标 | EP01 结果 | SR 含义 |
|------|----------|---------|
| TXT 数量 | 263 | 完整审计对象 |
| BMP 数量 | 263 | 同名视觉参考完整 |
| 矩阵尺寸 | 480×640 | 全部帧共享同一 detector grid |
| NaN/Inf | 0 / 0 | TXT 可直接用于数值处理 |
| 全数据坐标覆盖 | 253/256 | 近完整二维扫描网格 |
| 主 session | session=2, 255 帧 | 默认 SR POC 输入 |
| 主 session 坐标覆盖 | 253/256 | 主 session 覆盖全部实际存在坐标 |
| 文件名序 session | 13 个表观段 | 排序伪影，不用于后续 |
| 采集顺序 session | 3 个温度段 | 用于帧选择与温度隔离 |
| session 边界跳变 | 1.66°C / 4.16°C | 数十倍噪声底，跨 session 不混合 |
| 主 session 均温跨度 | 0.62°C | 后续对齐和 SR 应在该温度带内完成 |

---

## 决策记录

| 日期 | 决策 | 理由 |
|------|------|------|
| 2026-05-16 | 采用 `X_Y_R.ext` 命名格式 | 消除原始连写文件名歧义，保留坐标/R 信息 |
| 2026-05-16 | TXT 作为数值输入，BMP 作为视觉参考 | TXT 是 raw 温度矩阵，BMP 不是 SR 输入 |
| 2026-05-17 | Session 检测必须使用 `acquisition_order`/mtime | 文件名排序会制造 13 个表观 session |
| 2026-05-17 | 主 session=2 作为默认 SR 输入 | 该段包含 255 帧并覆盖全部实际存在坐标 |
| 2026-05-18 | EP01 范围收敛到 SR 数据基础 | 旋转角、alignment anchor、SR 算法验证分别由后续 Episode 处理 |
| 2026-05-18 | stage/文件名坐标只作为 prior | 对齐真值必须由图像数据与后续 EP04 localization 质量门控约束 |

---

## 输出文件

- Notebook fragments: `notebooks/ep01_data_processing/fragments/`
- Notebook 构建产物: `notebooks/ep01_data_processing/ep01_data_processing.ipynb`
- 正式报告: `reports/ep01_data_processing/audit_report.md`
- 帧审计 CSV: `output/ep01_data_processing/frame_audit.csv`
- 采集顺序审计 CSV: `output/ep01_data_processing/acquisition_order_audit.csv`
- SR 汇总 CSV: `output/ep01_data_processing/sr_data_basis_summary.csv`
- 图表:
  - `coordinate_coverage_map.png`
  - `frame_temperature_statistics.png`
  - `robust_temperature_timeline.png`
  - `order_comparison.png`
  - `session_detection.png`
  - `session_coordinate_coverage.png`

---

## 重建命令

```bash
uv run python scripts/build_notebook.py notebooks/ep01_data_processing --execute
```
