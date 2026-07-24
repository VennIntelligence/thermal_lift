# EP12 4x — UNet@2000 vs Bare Drizzle Benchmark

Quick real-data visual benchmark on the 248 clean main-session frames. Compares the
EP12 drizzle-informed 4x UNet checkpoint against the bare tcforge scatter-add
drizzle mean on the same highpass domain, center ROI, and colormap range.

## 角色定位

**评测资产（脚本型对照项目，无 `src/`）**。为 EP12 4x 主线提供真实数据裁判：
① 4x UNet vs 裸 drizzle 均值；② EP07 2x 输出 x2up 到 4x vs EP12 原生 4x。
它是「4x 是否值得采纳」判断链条中的视觉/指标证据来源之一（另一半是 EP15 的
FRC/M4 锚点门槛）。

## 目录构成

| 路径 | 职责 |
|------|------|
| `scripts/run_ep12_vs_drizzle_4x.py` | EP12 4x UNet checkpoint vs 裸 drizzle 均值（`tcforge.classical_sr.drizzle_features` 通道 0）中心放大对照 |
| `scripts/run_ep07x2up_vs_ep12_4x.py` | EP07 2x 输出上采样至 4x vs EP12 原生 4x（含 zigzag ROI 对照，`--roi-*` 参数） |
| `pyproject.toml` | UV 项目定义（无 `src/`，`package = false`） |

## 环境安装

独立 UV 项目；`ep12-4x-sr`、`thermal-core`、`tcforge` 通过 `[tool.uv.sources]`
以 editable 路径依赖自动装入：

```bash
cd algos/ep12_4x_benchmark
uv sync
```

主要依赖：torch>=2.2、numpy、scipy、matplotlib、pandas、tqdm、ipython。

## Scope

- Input: EP06 clean main session, 248 raw temperature frames.
- Alignment: EP05 `contour_refined` shifts.
- UNet checkpoint: `../ep12_4x_sr/outputs/ep12_large_bucketv2/checkpoint_step_002000.pt`.
- Baseline: bare drizzle mean (`tcforge.classical_sr.drizzle_features` channel 0).
- Highpass sigma: `5.0`.
- Default device: `cuda:1`. `cuda:0` is protected unless `--allow-cuda0` is passed.

## Run

```bash
cd algos/ep12_4x_benchmark
uv sync
uv run python scripts/run_ep12_vs_drizzle_4x.py \
  --checkpoint ../ep12_4x_sr/outputs/ep12_large_bucketv2/checkpoint_step_002000.pt \
  --output-dir ../../output/ep12_4x_benchmark \
  --device cuda:1
```

Smoke test:

```bash
uv run python scripts/run_ep12_vs_drizzle_4x.py \
  --checkpoint ../ep12_4x_sr/outputs/ep12_large_bucketv2/checkpoint_step_002000.pt \
  --output-dir /tmp/ep12_4x_smoke \
  --limit 16 \
  --device cuda:1
```

EP07 2x x2up vs EP12 原生 4x（参数已核实自 argparse 定义；两个 checkpoint 参数分别为
`--ep07-checkpoint` 与 `--ep12-checkpoint`，另有 `--roi-fraction/--roi-x-frac/--roi-y-frac/--roi-zoom`
控制 zigzag ROI）：

```bash
uv run python scripts/run_ep07x2up_vs_ep12_4x.py \
  --ep07-checkpoint ../ep07_unet_sr/outputs/<run>/model_final.pt \
  --ep12-checkpoint ../ep12_4x_sr/outputs/<run>/checkpoint_step_002000.pt \
  --output-dir ../../output/ep12_4x_benchmark \
  --device cuda:1
```

## 关键输出

默认写到 `output/ep12_4x_benchmark/`（Git 忽略，可用 `--output-dir` 覆盖）：

- `run_ep12_vs_drizzle_4x.py`：`ep12_<step>_hr_temp.npy` / `ep12_<step>_hr_highpass.npy`、`drizzle_bare_4x_hr_temp.npy` / `..._highpass.npy`、`raw_mean_control_4x_hr_temp.npy` / `..._highpass.npy`、`ep12_vs_drizzle_4x_center_zoom3x_{highpass,temperature}.png`、`comparison_summary.csv`、`comparison_notes.md`、`config.json`、`run_manifest.json`
- `run_ep07x2up_vs_ep12_4x.py`：`ep07x2up_vs_ep12_center_zoom3x_{highpass,temperature}.png`、`ep07x2up_vs_ep12_zigzag_roi_{highpass,temperature}.png`、`metrics_summary.csv`、`comparison_notes.md`、`config.json`、`run_manifest.json`

## 相关文档

- Episode 记录：`research_log/episodes/ep12_4x_benchmark/README.md`
- 算法变更日志：`research_log/algorithm_changelog.md` — **ACL-013/ACL-014**（EP12 4x 训练侧变更背景）
- 上游依赖：`algos/ep12_4x_sr/`（checkpoint 与 tiled 推理）、`algos/ep07_unet_sr/`（2x 对照臂）、`tcforge/`（drizzle 基线）
- 4x 采纳门槛：见 `algos/ep12_4x_sr/run_training.md` 与 `research_log/episodes/ep15_info_limit/README.md`
