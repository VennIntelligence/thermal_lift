# EP09 — PSF Sigma 精确标定

## 状态

已基于 EP06 248 clean-frame MAP-TV/highpass 主 baseline 重跑 CPU-only 三路线标定，结论为 **门控未通过**。

## 关键产物

- 脚本: `algos/ep09_psf_calibration/scripts/`
- Notebook fragments: `notebooks/ep09_psf_calibration/fragments/`
- 输出: `output/ep09_psf_calibration/`
- 配置: `configs/psf_calibration.json`
- 正式报告: `reports/ep09_psf_calibration/psf_calibration_report.md`
- EP06 pseudo-HR: `output/ep06_sr_poc/map_tv_highpass.npy`

## 结果摘要

| Route | 估计 sigma (LR px) | 结论 |
|---|---:|---|
| A Forward residual | 0.2257 | EP06 248 clean-frame 主 baseline 上的小 effective sigma；Route A 内部 CI 较窄 |
| B ESF fitting | 1.1286 | 外轮廓 ESF 宽度明显更大，包含热/几何/边缘宽度风险 |
| C Joint MAP-TV hold-out | 0.1190 | 偏小 sigma，但最小值贴扫描下界 |

## 决策

1. **不启动 4x 作为物理可行主线**: 三路线 spread 约 1.010 px，超过 ±0.05 px 一致性门控。
2. **`configs/psf_calibration.json` 只记录 provisional effective sigma**: 当前值来自 Route A，状态为 `provisional_needs_review`。
3. **后续如果继续 4x，必须先解释 ESF vs forward effective sigma 的系统差异**: 可能来源包括真实热边缘宽度、边缘倾斜、单帧 ESF 采样、MAP-TV pseudo-HR bias 或 Gaussian PSF 模型不适配。

## 复现命令

```bash
uv run python algos/ep09_psf_calibration/scripts/run_forward_residual.py
uv run python algos/ep09_psf_calibration/scripts/run_esf_fitting.py
uv run python algos/ep09_psf_calibration/scripts/run_joint_estimation.py
uv run python algos/ep09_psf_calibration/scripts/summarize_calibration.py
uv run python scripts/build_ep09_cache.py --force
uv run python scripts/build_notebook.py notebooks/ep09_psf_calibration --execute
```
