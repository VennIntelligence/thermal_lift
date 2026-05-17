# EP02 — 位移标定与旋转角验证

> **目标**: 独立验证旋转角 θ 和位移台精度 — 这是超分辨率成败的关键参数  
> **状态**: ✅ 已完成首轮验证（NCC 结果不支持更新 θ）  
> **前置**: EP01（数据审计完成）

---

## 🧠 核心问题

旋转角 θ = 47.6° 是旧项目通过数值优化得到的，从未物理测量过。
位移台标称精度 1 µm，但实际重复定位误差未知。
这两个参数直接决定亚像素位移建模的准确性，是 SR 算法的基石。

---

## 📋 任务清单

### Phase 1: 互相关位移估计

- [x] 1.1 **亚像素互相关实现** — 对相邻坐标帧做互相关，提取亚像素位移 (dx, dy)
- [x] 1.2 **位移向量场可视化** — 绘制所有帧对的位移向量 (dx, dy)
- [x] 1.3 **位移一致性检查** — 同一坐标差的帧对，位移应一致

### Phase 2: 旋转角 θ 拟合

- [x] 2.1 **θ 拟合** — 从位移向量 (dx, dy) vs 坐标差 (ΔX, ΔY) 反推旋转角
- [x] 2.2 **θ 置信区间** — Bootstrap 估计 θ 的不确定度
- [x] 2.3 **与 47.6° 对比** — 判断旧项目参数是否可信

### Phase 3: 位移台精度评估

- [x] 3.1 **重复定位精度** — 利用 3-repeat 坐标评估同一位置的定位误差
- [x] 3.2 **线性度检查** — 实际位移 vs 标称位移的线性回归
- [x] 3.3 **精度报告** — 定量评估对 SR 的影响

### Phase 4: Session 方向假设检查

- [x] 4.1 **按 session / 扫描轴 / 步长拆分方向** — 检查是否存在 session 级运动方向变化
- [x] 4.2 **方向与幅值诊断图** — 输出 `motion_direction_diagnostic.png`

---

## 2026-05-17 首轮结论

### 关键结果

| 项目 | 结果 |
|------|------|
| 主扫描相邻帧对 | 463 |
| NCC 峰值中位数 | 0.99545 |
| image-row θ 拟合 | 145.690° |
| y-up 诊断 θ 拟合 | 34.053° |
| y-up 95% Bootstrap CI | [33.226°, 34.829°] |
| 47.6° 是否落入 CI | 否 |
| 旋转模型 RMS 残差 | 0.1567 px |
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
- 对 SR 的影响：0.1567 px 模型残差已超过 2× SR 实用阈值 0.1 px，也超过 4× SR 目标 0.05 px。当前位移证据不足以支撑可靠 SR。

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
