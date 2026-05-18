# EP06 — 2x Contour-Level SR POC (经典物理算法)

> **目标**: 用经典物理 SR 算法在 255 帧主 session 上完成 2x contour-level 重建，证明多帧微扫描对内部结构/轮廓的增益。
> **状态**: ✅ POC 完成（classic 2x SR + 双轨评估）
> **前置**: EP01-EP05 全部完成

---

## 算法表

| 层级 | 方法 | 用途 |
|---|---|---|
| baseline | Bicubic single-frame | 显示倍率对照 |
| baseline | SAA-uniform | 多帧相位覆盖是否有用 |
| baseline+ | SAA-weighted | 质量门控/加权是否足够 |
| classic SR | IBP | forward model baseline |
| best physical | MAP-TV | 当前 contour-level 最有希望的方法 |

## 双轨输出

1. **主轨 (highpass-input SR)**: 每帧先 highpass → SR 重建 → 输出结构图。评价 gradient magnitude、held-out contour Chamfer、split-half consistency。
2. **控制轨 (raw-temperature SR)**: 原始温度帧 + per-frame offset correction → SR 重建 → 输出端 highpass 可视化。证明主轨不是人为制造结构。

## 评价指标

- Gradient magnitude (轮廓锐度)
- Held-out contour Chamfer distance
- Split-half SR consistency
- Artifact audit (振铃、块状伪影、边缘 ringing)
- 并排可视化对照

## 产物索引

- `algos/ep06_sr_poc/` — 算法实现（独立 UV 环境）
- `output/ep06_sr_poc/` — 数据产物
- `notebooks/ep06_sr_poc/` — 可视化 Notebook
- `reports/ep06_sr_poc/` — 正式报告

## 当前结果摘要

- 已在主 session 255 帧上生成 highpass 主轨和 raw-temperature 控制轨的 SAA-uniform、SAA-weighted、IBP、MAP-TV 2x 输出。
- 合成验证通过递进关系：SAA `28.43 dB`，IBP `28.78 dB`，MAP-TV `29.03 dB`。
- 真实数据 sanity check 通过：所有输出为 `(960, 1280)`，无 NaN/Inf。
- raw 控制轨在输出端 highpass 后复现主轨主要结构，说明主轨结构不是 highpass 预处理单独制造。
- MAP-TV 在真实数据中比 SAA/IBP 更锐，但 artifact score 也更高；结论应表述为候选增强方法，而不是计量级分辨率证明。

## 复现命令

```bash
uv run python algos/ep06_sr_poc/scripts/run_saa.py --workers 4
uv run python algos/ep06_sr_poc/scripts/run_ibp.py --workers 4
uv run python algos/ep06_sr_poc/scripts/run_map_tv.py --workers 4 --max-iter 12 --step-size 0.5 --lambda-grid 0.00001,0.0001,0.0003,0.001
uv run python algos/ep06_sr_poc/scripts/run_evaluation.py
uv run python scripts/build_notebook.py notebooks/ep06_sr_poc --execute
```
