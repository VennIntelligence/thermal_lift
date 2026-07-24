# EP15 Info Limit

Independent UV project for EP15 first-principles information-limit checks.

## 角色定位

**理论/信息上限审计**。EP12 4x 网络失败后，用第一性原理测量真实数据的信息上限，
并建立后续网络方法必须超越的经典下限基准：M1 相位结构 → M2 FRC 信息截止 →
M3 sigma 仲裁 → M4 MAP-TV 去卷积锚点。M4 产出的 `map-tv_*` 锚点被 EP11 论文
harness 和 EP12 4x 采纳门槛直接引用。此外还承载 solver-v2 重设计期间的
Stage 0 系列仪表（split-FRC harness、offset 探针、info-budget 重跑）。

## 目录构成

| 路径 | 职责 |
|------|------|
| `scripts/run_m1_phase_structure.py` | M1：验证 248 帧 clean set 在 contour-refined 对齐下仍具 5x5 微扫描相位格 |
| `scripts/run_m2_frc.py` | M2：split-half FRC 信息截止测量 |
| `scripts/run_m3_sigma_arbitration.py` | M3：ESF / forward / FRC 三路证据仲裁 PSF sigma（判断 EP09 Route B 是否测到热边缘而非光学 PSF） |
| `scripts/run_m4_deconv_anchor.py` | M4：GPU FISTA MAP-TV 去卷积锚点（sigma/lambda 扫描 + 四臂对比） |
| `scripts/run_real_split_frc_v2.py` | Stage 0c：solver-v2 重设计用的真实 split-half FRC harness（多方法、多切分模式） |
| `scripts/probe_pair_offset.py` | Stage 0g/0h：重建对全局亚像素网格偏移探针（相位相关 + FRC 前后对比；ACL-047/048） |
| `scripts/run_stage0b_info_budget2_shift_sweep.py` | Stage 0b：合成 info-budget 重跑 + shift 误差扫描（历史 `info_budget2.py` 的可复现替身） |

无 `src/`，全部逻辑在脚本内；共享工具来自 `thermal_core`。

## Setup

```bash
cd algos/ep15_info_limit
uv sync
```

注：`thermal-core` 已在 `pyproject.toml` 的 `[tool.uv.sources]` 中声明为
editable 路径依赖，`uv sync` 会自动装入；历史文档中的
`uv pip install -e ../../core` 手动步骤已非必需。

M4 uses the CUDA PyTorch wheel in this isolated UV environment
（torch **有意不写入** `pyproject.toml` 依赖，需按下述命令手动装 CUDA wheel）:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv run python -c "import torch; print(torch.cuda.is_available())"
```

## M1 Phase Structure

```bash
uv run python scripts/run_m1_phase_structure.py
```

Outputs are written to `output/ep15_info_limit/m1_phase_structure/`.

## M2 FRC Information Cutoff

```bash
uv run python scripts/run_m2_frc.py
```

Outputs are written to `output/ep15_info_limit/m2_frc/`.

## M3 Sigma Arbitration

```bash
uv run python scripts/run_m3_sigma_arbitration.py
```

Outputs are written to `output/ep15_info_limit/m3_sigma/`.

## M4 MAP-TV Deconvolution Anchor

Run a smoke pass first. Select one idle GPU explicitly; during the 2026-06-10 run GPU 0 was used because GPU 1 was occupied by EP07 training.

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_m4_deconv_anchor.py --smoke --chunk-size 8
```

Run the full single-GPU scan:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_m4_deconv_anchor.py --chunk-size 32
```

Default M4 settings use the M1 grid scale, 248 clean main-session frames, contour-refined shifts, PSF sigma scan `0.2,0.3,0.4,0.5`, lambda scan `3e-4,1e-3,3e-3`, FISTA MAP-TV with 150 iterations, and detector-aperture box integration via `avg_pool2d`. `--no-box` is only for ablation and skips the strict EP06 box-model smoke comparison.

Outputs are written to `output/ep15_info_limit/m4_deconv_anchor/`. `parameter_selection.csv` records every sigma/lambda split-half selection run plus per-sigma full-run timings; `convergence_curves.csv` records the per-sigma full selected-lambda runs. The four-arm comparison uses the EP07 v6 checkpoint at `../ep07_unet_sr/outputs/ep07_v6_physics/model_final.pt` when available.

## Stage 0b Info Budget2 Shift Sweep

Synthetic replacement scaffold for the historical `info_budget2.py` artifact:

```bash
uv run python scripts/run_stage0b_info_budget2_shift_sweep.py
```

Useful MVP smoke:

```bash
uv run python scripts/run_stage0b_info_budget2_shift_sweep.py \
  --lr-size 64 \
  --scene-seeds 11,23 \
  --frame-budgets 16,64 \
  --shift-error-seeds 401
```

Outputs are written to `output/ep15_info_limit/stage0b_info_budget2/`.
The script reads the corrected 20 um detector pitch from
`configs/stage_calibration.json`, accepts `--psf-sigmas-lr-px`, and sweeps
`--shift-error-grid` over LR-pixel Gaussian shift errors for Stage 1a DR
calibration. It reports both a spatial drizzle/Wiener baseline and a Fourier
alias ridge oracle (`alias_multiframe_wiener_*`). For DR calibration, prefer
the full or near-full frame-budget oracle delta columns; tiny budgets can be
conditioning diagnostics rather than monotonic robustness curves.

## Stage 0c Real Split-Half FRC Harness (v2)

solver-v2 重设计期间的真实数据 split-half FRC 仪表，支持多方法
（`--methods`）、多切分模式（`--split-mode`）、cross-pair / artifact-pair
对照与相位分箱：

```bash
uv run python scripts/run_real_split_frc_v2.py --workers 4
```

Outputs are written to `output/stage0c_real_split_frc_v2/`（默认值，可用 `--output-dir` 覆盖）。

## Pair Offset Probe

对两幅重建（`--pair a.npy b.npy`）做全局亚像素网格偏移探测：加窗相位相关估计
(dx, dy) → 频域反移 → 对比校正前后的 FRC 截止（ACL-047/048 的振荡 FRC 判因工具）：

```bash
uv run python scripts/probe_pair_offset.py \
  --pair recon_a.npy recon_b.npy \
  --output-dir ../../output/ep15_info_limit/pair_offset_probe
```

## 相关文档

- Episode 记录：`research_log/episodes/ep15_info_limit/README.md`（M1–M4 结论已回填）
- 算法变更日志：`research_log/algorithm_changelog.md` — **ACL-012**（EP15 M4 GPU MAP-TV 去卷积基准重跑）、**ACL-045**（Stage 1a operator DR + 0a bootstrap CI）、**ACL-046**（Stage 0f 仪表修复）、**ACL-047/ACL-048**（shift 误差瓶颈实锤、偏移探针与精修对齐升级）
- 下游消费者：`algos/ep11_dl_benchmark/scripts/run_unified_harness_t1_t2.py`（默认读取 `output/ep15_info_limit/m4_deconv_anchor/map-tv_*` 锚点）、`algos/ep12_4x_sr/`（4x 采纳门槛）
