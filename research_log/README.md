# Research Log — Episode Index

> 每个 Episode 对应一个独立研究阶段，包含进度追踪、决策记录和分析报告。
> Episode 编号与 `notebooks/` 子目录一一对应。

---

## 当前主线

- 客户目标：工业芯片检测中看清芯片内部结构/形状。
- 数据基础：263 帧 TXT/BMP 中，主 session=2 的 255 帧是 2x contour-level SR POC 的默认输入；跨 session 帧不得混用。
- 位移策略：stage command 只作 prior / 初始化 / 约束，不作对齐真值。
- EP04 定位：localization 是 alignment anchor / quality gate，用来筛选可靠帧对、ROI 和参数，不是最终交付目标。
- POC 验收：以 LR 单帧、bicubic 2×、多帧平均和 2x SR 的轮廓可见性对比为核心。

---

## 📋 Episode 路线图

| Episode | 名称 | 状态 | 目录 |
|---------|------|------|------|
| EP01 | 数据处理与验证 | ✅ 首轮完成 | `ep01_data_processing/` |
| EP02 | 位移标定与旋转角验证 | ✅ 首轮完成 | `ep02_displacement_calibration/` |
| EP03 | 理论边界与最小验证 | ✅ 首轮完成；不作为 SR 否定结论 | `ep03_theoretical_limits/` |
| EP04 | Data-driven localization / alignment quality gate | ✅ EP04-A 完成；作为锚点和质控 | `ep04_global_validation/` |
| EP05 | 2x SR capacity and alignment baseline | ✅ 完成；EP06 输入依据 | `ep05_sr_reassessment/` |
| EP06 | 2x contour-level SR POC | ✅ 完成；classic SR 双轨对比 | `ep06_sr_poc/` |

> 路线图随项目推进更新。新 Episode 创建时在此注册。

---

## 📂 目录对齐规则

每个 Episode `epXX_name` 在四个位置有对应目录：

```
research_log/episodes/epXX_name/README.md  ← 进度 + 任务 + 决策记录
notebooks/epXX_name/                       ← Jupyter 可视化 & 交互分析
reports/epXX_name/                         ← 正式分析报告
output/epXX_name/                          ← 数据产物（CSV、图表等）
```

- `research_log/` = 文字记录（进度、决策、发现）
- `notebooks/` = 代码驱动的可视化和分析
- `reports/` = 正式报告，按 Episode 分子目录
- `output/` = 数据产物（CSV、映射表、图表等）
