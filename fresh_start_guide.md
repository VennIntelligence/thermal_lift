# Thermal Lift Fresh Start Guide

> **⚠️ 历史文档（2026-07-24 收尾标注）**: 本文档反映项目早期（EP01–EP06 时代）的认知基线，其中的「下一步路线」已全部执行完毕或被取代。当前项目最终结论以 `research_log/algorithm_changelog.md`（ACL-001–080）和 `docs/publication_figures/GALLERY.md` 为准；导航入口见根目录 `README.md` 和 `research_log/README.md`。本文档保留作为早期事实与边界的历史参照。

> 本文档是当前项目的入口基线。它只保留继续推进所需的事实、边界和下一步路线；旧的失败路线、旧 handoff 和生成 notebook 已从项目上下文中移除。

## 1. 项目目标

客户目标是在工业芯片检测场景中，通过 LWIR 微扫描热像数据看清芯片内部结构形状和局部细节。当前工程目标是先完成 **2x contour-level SR POC**，证明内部轮廓或局部结构在热像中比 LR 单帧、bicubic 2x、简单多帧平均更清楚、更稳定。

不要把“边缘定位精度”当作最终交付。EP04 的定位结果只作为后续对齐锚点、质量门控和局部一致性指标。

## 2. 当前可用事实

| 项目 | 当前值 / 结论 |
|---|---|
| TXT 温度矩阵 | 263 帧，全部 480 x 640 |
| 原始主扫描 session | 255 帧，`session=2`，保留为物理温度段诊断 |
| Clean SR 默认输入 | 248 帧，`is_sr_usable=True` / `is_main_session=True`，剔除 `R != 0` 重复/补采帧 |
| TXT 采样 pitch | 20 um/pixel，校准修正（旧 10 um/pixel 为 BMP 标尺 2× 误读） |
| 当前空间分辨率 | 20 um，已校准；不是 TXT 像素 pitch |
| 目标提升 | 先做 2x contour-level POC，再评估更高倍率 |
| 坐标集合 | `{0,2,4,6,8,10,12,14,16,18,20,24,28,32,36,40}` um |
| stage-command 范围 | 40 um = 2.0 px 命令向量幅值，作为 prior / 初始化 / 约束 |
| 相位覆盖 | 248 clean SR 帧在 2x SR 四个相位格中分布均匀，当前采样量足够做 2x POC |
| 温度段 | 3 个温度段；跨 session 不混合 |
| AVI | 只作方向和命名诊断，不作 SR 输入 |

## 3. 数据使用规则

1. 任何位移、session、时间线分析都必须使用 `output/ep01_data_processing/frame_audit.csv` 中的 `acquisition_order`、`session`、`is_sr_usable`、`is_main_session`。
2. SR 和 contour-level POC 默认只使用 248 帧 clean SR set；255 帧主 session 只作为原始物理温度段诊断事实。
3. 文件名 X/Y 坐标是命令坐标和相位 prior，不是最终对齐真值。
4. 实际对齐必须由数据驱动方法约束：NCC init、filename affine prior、EP04/EP05 contour anchor、held-out edge residual、split-half consistency。
5. BMP 是配套可视化和定性 overlay 参考；SR 输入以 TXT 温度矩阵为准。

## 4. 当前推荐路线

### Phase A: 数据基线

- 确认 EP01 frame audit 可重建。
- 确认原始主 session 255 帧、clean SR 默认输入 248 帧、TXT/BMP 配对、缺失坐标和温度漂移。
- 保留 AVI 诊断脚本，但不让 AVI 进入 SR 输入链路。

### Phase B: Alignment Baseline

- 使用 stage command 生成位移 prior。
- 使用 highpass/gradient NCC 得到逐帧连续相位初值。
- 使用 EP04/EP05 contour anchor 做局部修正和质量门控。
- 用 held-out contour Chamfer、gradient correlation、split-half consistency 评估对齐。

### Phase C: 2x Contour SR POC

必须输出同一 ROI 的并排对照：

1. LR 单帧。
2. Bicubic 2x。
3. 简单多帧平均或中值。
4. 2x contour-level SR。

评价重点：

- 内部结构边界是否更清楚。
- split-half / held-out 结果是否一致。
- 对齐残差是否低于当前质量门控阈值。
- 是否出现反卷积振铃、伪边、漂移污染。

## 5. 保留与删除边界

保留：

- EP01 数据审计和 session 结构。
- EP02 raster / acquisition-order 位移诊断。
- EP03 CRB、ESF、局部位移输入边界诊断。
- EP04 quality-gated localization 作为 alignment anchor。
- EP05 2x SR capacity、overlay、data-driven alignment 产物。

删除或不再作为项目上下文：

- 旧 handoff prompt、旧方向 note、生成 `.ipynb`。
- 旧输出图和旧实验产物，除非是 EP05 当前基线。
- 任何把局部小步或局部法线响应外推成“全局位移不足”的叙事。
- 任何把 localization 精度等同于客户所需形状重建的叙事。

## 6. 关键配置

```python
import numpy as np

THETA_DEG = 47.6
PIXEL_SIZE_UM = 20.0

def coordinate_to_shift(x_um, y_um):
    """Stage command to detector-pixel shift prior."""
    theta = np.radians(THETA_DEG)
    dx = (x_um * np.cos(theta) + y_um * np.sin(theta)) / PIXEL_SIZE_UM
    dy = (-x_um * np.sin(theta) + y_um * np.cos(theta)) / PIXEL_SIZE_UM
    return dx, dy
```

这只是命令位移 prior。后续 SR 不应把它当作无需验证的最终对齐。
