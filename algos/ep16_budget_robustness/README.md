# EP16 Budget Robustness

CPU-only classical methods for paper Section 6.4-6.5:

- E1 frame budget: `N={31,62,124,248}` phase-stratified subsets.
- E2 shift robustness: Gaussian noise on contour-refined shifts.
- E3 alignment source ablation: command prior vs contour-refined shifts.

## 角色定位

**评测资产（帧预算 / 鲁棒性消融）**。为论文 6.4–6.5 节提供经典方法
（drizzle + MAP-TGV）在帧数预算、shift 噪声、对齐来源三个轴上的退化曲线；
其 `frame_budget.csv` 也是 EP11 论文 harness 的 TGV proxy 输入。CPU-only，
不涉及神经网络训练。

## 目录构成

| 路径 | 职责 |
|------|------|
| `scripts/run_ep16_classical.py` | 主入口：E1/E2/E3 全部 drizzle 重建 + TGV 子进程编排 + 汇总图表/CSV |
| `scripts/run_tgv_child.py` | 单次 MAP-TGV 重建子进程（`--spec` 读取 JSON 规格；在 EP10 conda 环境中执行；也被 EP11 `run_tgv_split_frc.py` 复用） |
| `pyproject.toml` | UV 项目定义（无 `src/`，`package = false`） |

## 环境安装

独立 UV 项目（`thermal-core` 为 editable 路径依赖，自动装入）；主要依赖：
numpy、scipy、pandas、matplotlib、joblib、tqdm、`drizzle>=2.2`。
TGV 臂额外要求 `algos/ep10_tgv_sr/.venv` conda 环境已按其 README 建好
（`--tgv-env` / `--conda-exe` 可覆盖默认路径）。

```bash
cd algos/ep16_budget_robustness
CUDA_VISIBLE_DEVICES="" uv sync
```

## 运行方法

The runner reconstructs all drizzle runs directly in this UV project. TGV runs
are launched as CPU-only child processes through the existing EP10 conda
environment:

```bash
cd algos/ep16_budget_robustness
CUDA_VISIBLE_DEVICES="" uv run python scripts/run_ep16_classical.py \
  --arms both \
  --run-tgv \
  --tgv-parallel 2 \
  --tgv-workers 6
```

其余常用参数（已核实自 argparse 定义）：`--arms {drizzle,tgv,both}`、
`--skip-tgv`、`--summarize-only`、`--resume/--no-resume`、`--io-workers`、
`--tgv-env`、`--conda-exe`、`--tgv-timeout-sec`，以及 TGV 参数组
`--tgv-lambda-tv/--tgv-psf-sigma/--tgv-alpha-ratio/--tgv-max-iter/--tgv-inner-iter/--tgv-aniso-ratio-y`。
`run_tgv_child.py` 仅接受 `--spec <json>`，由父进程自动生成规格文件，一般不手动调用。

Useful shorter commands:

```bash
# Drizzle only, useful before the overnight TGV queue.
CUDA_VISIBLE_DEVICES="" uv run python scripts/run_ep16_classical.py --arms drizzle --skip-tgv

# Rebuild figures and CSVs from completed run JSON files.
CUDA_VISIBLE_DEVICES="" uv run python scripts/run_ep16_classical.py --summarize-only
```

## 关键输出

Outputs are written under `../../output/ep16_budget_robustness/` and are ignored
by Git. The script writes `run_manifest.json` after each run, so failed TGV
children remain visible and later invocations can resume successful runs.
`frame_budget.csv` 被 `algos/ep11_dl_benchmark/scripts/run_unified_harness_t1_t2.py`
默认引用为 TGV proxy 数据。

Resource contract:

- `CUDA_VISIBLE_DEVICES` is forced to an empty string in parent and child.
- TGV parent parallelism defaults to 2.
- Each TGV child passes `workers<=6` to `reconstruct_map_tgv`.
- BLAS/OpenMP thread environment variables default to 1 unless already set.

## 相关文档

- Episode 记录：`research_log/episodes/ep16_budget_robustness/README.md`（E1/E2/E3 设计与论文 6.4–6.5 节对应关系）
- 上游依赖：`algos/ep10_tgv_sr/`（conda 环境 + `reconstruct_map_tgv`）、`algos/ep06_sr_poc/`（数据加载/指标，经 sys.path 引用）
- 下游消费者：`algos/ep11_dl_benchmark/`（论文 harness 引用 `frame_budget.csv`；其 `run_tgv_split_frc.py` 复用本项目 `run_tgv_child.py`）
