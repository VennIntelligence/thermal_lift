# EP11 — UNet 2x@4000 vs TGV 2x Benchmark

This episode runs a quick real-data visual benchmark on the 248 clean main-session
thermal frames. It compares the EP07 residual UNet step-4000 checkpoint against
the existing EP10 TGV best 2x artifact in the same highpass domain,
same center ROI, and same colormap range.

## 角色定位

**评测资产（脚本型对照项目，无 `src/`）**。本项目自身不实现算法，只做真实数据上的
同域对照评测与论文证据产出：① UNet 2x vs TGV 2x 视觉对照；② 论文 T1/T2 表与 F5
图的统一真实数据 harness；③ TGV 实测 split-half/FRC 证据硬化。它是 EP07（神经）
与 EP10（经典）两条线在真实 248 帧上交汇的裁判席。

## 目录构成

| 路径 | 职责 |
|------|------|
| `scripts/run_unet_vs_drizzle_2x.py` | 主对照：EP07 UNet checkpoint vs TGV/drizzle 基线（highpass 域中心 ROI 视觉 + split/相关指标） |
| `scripts/run_unified_harness_t1_t2.py` | 论文 T1/T2/F5 统一真实数据 harness（薄编排层，复用 EP06 loader/metrics、EP10 drizzle、EP15 FRC 探针、EP07 推理路径；ACL-021） |
| `scripts/run_tgv_split_frc.py` | CPU-only TGV 实测 split-half + FRC（Task E1 证据硬化，经 EP10 conda 环境子进程重建；ACL-022） |
| `scripts/make_center_zoom4x_png_comparison.py` | 从已有 eval PNG / 缓存 NPY 拼 4x 中心放大对比图（不做 checkpoint 重推理） |
| `pyproject.toml` | UV 项目定义（无 `src/`，`package = false`） |

## 环境安装

独立 UV 项目；`thermal-core`、`ep07-unet-sr`、`ep10-drizzle`、`tcforge` 均通过
`[tool.uv.sources]` 以 editable 路径依赖自动装入，无需手动 `uv pip install -e`：

```bash
cd algos/ep11_dl_benchmark
uv sync
```

主要依赖：torch>=2.2、numpy、scipy、matplotlib、pandas、tqdm、ipython。
pytest 的 `pythonpath` 额外指向 `../ep06_sr_poc/src`。

## Scope

- Input: EP06 clean main session, 248 raw temperature frames.
- Alignment: EP05 `contour_refined` shifts.
- UNet checkpoint: `../ep07_unet_sr/outputs/ep07_unet_sr_v4/checkpoint_step_004000.pt`.
- Baseline artifact: `../../output/ep10_tgv_sr/best_hr_highpass.npy`.
- Baseline metrics: `../../output/ep10_tgv_sr/sweep_results.csv` plus `run_summary.json`.
- Highpass sigma: `5.0`, matching EP10 TGV.
- Default device: `cuda:1`. Bare `--device cuda` is also resolved to `cuda:1` when two or more CUDA devices are visible. `cuda:0` is protected unless `--allow-cuda0` is explicitly passed.

## Run

```bash
cd algos/ep11_dl_benchmark
uv sync
uv run python scripts/run_unet_vs_drizzle_2x.py \
  --checkpoint ../ep07_unet_sr/outputs/ep07_unet_sr_v4/checkpoint_step_004000.pt \
  --baseline-hr ../../output/ep10_tgv_sr/best_hr_highpass.npy \
  --baseline-sweep ../../output/ep10_tgv_sr/sweep_results.csv \
  --baseline-summary ../../output/ep10_tgv_sr/run_summary.json \
  --baseline-name "TGV best 2x" \
  --output-dir ../../output/ep11_dl_benchmark \
  --zoom 3.0 \
  --center-fraction 0.3333333 \
  --device cuda:1
```

Smoke test:

```bash
uv run python scripts/run_unet_vs_drizzle_2x.py \
  --checkpoint ../ep07_unet_sr/outputs/ep07_unet_sr_v4/checkpoint_step_004000.pt \
  --baseline-hr ../../output/ep10_tgv_sr/best_hr_highpass.npy \
  --baseline-sweep ../../output/ep10_tgv_sr/sweep_results.csv \
  --baseline-summary ../../output/ep10_tgv_sr/run_summary.json \
  --baseline-name "TGV best 2x" \
  --output-dir /tmp/ep11_smoke \
  --limit 16 \
  --device cuda:1
```

论文统一 harness（T1/T2 表 + F5 图；参数已核实自 argparse 定义，全部有默认值，
支持 `--only` 选择子任务、`--force` 强制重算、`--skip-f5`）：

```bash
uv run python scripts/run_unified_harness_t1_t2.py --device cuda:0 --workers 4
```

TGV 实测 split-half/FRC（CPU-only，需要 EP10 conda 环境就位；
`--conda-exe` 默认 `/home/ujs/miniforge3/bin/conda`，macOS 上需显式传本机 conda 路径）：

```bash
uv run python scripts/run_tgv_split_frc.py \
  --conda-exe ~/miniforge3/bin/conda \
  --tgv-workers 4
```

## 关键输出

- `run_unet_vs_drizzle_2x.py` → `output/ep11_dl_benchmark/`（可用 `--output-dir` 覆盖）
- `run_unified_harness_t1_t2.py` → `output/ep11_unified_harness/` + `output/paper_figures/`
- `run_tgv_split_frc.py` → `output/ep11_unified_harness/tgv_split_frc/` + `tgv_split_frc.json`

The main comparison script writes:

- `unet_step4000_hr_temp.npy`
- `unet_step4000_hr_highpass.npy`
- `raw_mean_control_2x_hr_temp.npy`
- `raw_mean_control_2x_hr_highpass.npy`
- `unet_vs_tgv_2x_center_zoom3x_highpass.png`
- `unet_step4000_center_zoom3x_temperature.png`
- `comparison_summary.csv`
- `comparison_notes.md`
- `run_manifest.json`

## Interpretation Boundary

This benchmark is a contour-level visual comparison, not a 5 um metrology,
temperature-accuracy conclusion, or 3x SR claim. The reconstruction grid is
still EP07 2x; `--zoom 3.0` is only display magnification for the center ROI.
The fair side-by-side view is the highpass figure. The raw-temperature figure is
a UNet-only sanity view; the raw-mean control is used only for correlation, not
as an algorithm comparison.

The UNet checkpoint is trained on synthetic data and `checkpoint_step_004000.pt`
is a mid-training checkpoint, not the final 25000-step model. Treat any real
inner-contour improvement as domain-gap-sensitive until split-half consistency,
artifact behavior, and raw-control agreement are stable. Tenengrad or sharpness
alone must not be used to declare a winner.

## 相关文档

- Episode 记录：`research_log/episodes/ep11_dl_benchmark/README.md`
- 正式报告：`paper/reports/ep11_dl_benchmark/README.md`、`paper/reports/ep11_dl_benchmark/unet_checkpoint_selection.md`
- 算法变更日志：`research_log/algorithm_changelog.md` — **ACL-004**（checkpoint 推演改为 EP11 真实数据 3x 温度图）、**ACL-021**（论文 T1/T2/F5 统一真实数据 harness）、**ACL-022**（Task E 论文证据硬化：TGV actual split/FRC + F5b ROI2）
- 上游依赖：`algos/ep07_unet_sr/`（checkpoint）、`algos/ep10_tgv_sr/`（TGV 锚点 + conda 环境）、`algos/ep10_drizzle/`、`algos/ep16_budget_robustness/scripts/run_tgv_child.py`（TGV 子进程复用）
