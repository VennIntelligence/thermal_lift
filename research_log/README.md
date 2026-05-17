# Research Log — Episode Index

> 每个 Episode 对应一个独立研究阶段，包含进度追踪、决策记录和分析报告。
> Episode 编号与 `notebooks/` 子目录一一对应。

---

## 📋 Episode 路线图

| Episode | 名称 | 状态 | 目录 |
|---------|------|------|------|
| EP01 | 数据处理与验证 | ✅ 首轮完成 | `ep01_data_processing/` |
| EP02 | 位移标定与旋转角验证 | ✅ 首轮完成 | `ep02_displacement_calibration/` |
| EP03 | 理论极限分析 | ⬜ 未开始 | — |
| EP04 | 基线算法实现 | ⬜ 未开始 | — |
| EP05 | 高级 SR 算法 | ⬜ 未开始 | — |

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
