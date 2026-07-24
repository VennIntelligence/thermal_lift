# EP10 MAP-TGV SR

CPU MAP super-resolution experiment using CCPi-Regularisation-Toolkit TGV as
the proximal operator in the EP06 MAP-TV FISTA loop.

## 角色定位

**经典基线（MAP-TGV）**。EP10 方法对比（Drizzle / MAP-TV / MAP-TGV）中的 TGV 臂，
产出的 `best_hr_highpass.npy` 是后续所有神经网络方法（EP07/EP11/EP12）必须对标的
经典 2x contour-level 锚点；EP11 论文 harness 与 EP16 预算/鲁棒性实验的 TGV
子进程也复用本项目的 conda 环境。

## 目录构成

| 路径 | 职责 |
|------|------|
| `environment.yml` | conda 环境定义（本项目**不用 UV**，见下） |
| `src/ep10_tgv_sr/tgv.py` | `reconstruct_map_tgv()` FISTA 主循环 + CCPi TGV proximal 封装、backend provenance（`get_tgv_backend_provenance()`）、本地 Chambolle-Pock 诊断 fallback |
| `scripts/run_tgv_sr.py` | 主入口：参数 sweep + split-half + holdout 完整评估流程 |
| `scripts/run_tgv_quick.py` | 快速单参数运行（lambda=0.003, sigma=0.5，含 ACL-007 aniso/coverage 修复），**无 CLI 参数**，直接跑一次完整重建 + 对比图 |
| `tests/test_tgv.py` | TGV 模块单元测试 |

依赖关系：通过 `sys.path` 直接引用 `../ep06_sr_poc/src`（`common.*` 数据加载/指标）
与 `../../core/src`（`thermal_core.*`），因此运行时需保持仓库目录结构完整。

## Environment

This algorithm intentionally uses a standalone conda environment instead of the
root UV environment. EP10 depends on `ccpi-regulariser`, whose supported binary
distribution is provided through conda channels (`conda-forge` + `ccpi`); using
that package manager keeps the CCPi CPU/CUDA backend reproducible without local
C/C++/CUDA builds. The local Chambolle-Pock implementation is only a diagnostic
fallback and is recorded in the run provenance when CCPi cannot be imported or
executed.

```bash
cd algos/ep10_tgv_sr
conda env create -p .venv -f environment.yml
conda run -p .venv pip install -e ../../core
```

The repository-local run used Miniforge/mamba because `conda` was not already
on PATH:

```bash
~/miniforge3/bin/mamba env create -p .venv -f environment.yml
~/miniforge3/bin/mamba run -p .venv pip install -e ../../core
```

## Run

```bash
~/miniforge3/bin/mamba run -p algos/ep10_tgv_sr/.venv \
  python algos/ep10_tgv_sr/scripts/run_tgv_sr.py
```

常用参数（均已从 `run_tgv_sr.py` argparse 定义核实）：
`--output-dir`、`--workers`、`--lambda-grid`（默认 `0.0003,0.001,0.003`）、
`--psf-grid`（默认 `0.18,0.50`）、`--alpha-ratio`、`--max-iter`、
`--aniso-ratio-y`（默认 1.5）、`--coverage-weighted/--no-coverage-weighted`、
`--tgv-device`（默认 `auto`）、`--resume/--no-resume`、`--force`、
`--max-params`（限制 sweep 点数，可做快速验证）、`--synthetic-only`（只跑合成 gate）。

快速单参数运行（无 CLI 参数，直接使用 lambda=0.003 / sigma=0.5 + ACL-007 修复）：

```bash
~/miniforge3/bin/mamba run -p algos/ep10_tgv_sr/.venv \
  python algos/ep10_tgv_sr/scripts/run_tgv_quick.py
```

测试：

```bash
~/miniforge3/bin/mamba run -p algos/ep10_tgv_sr/.venv python -m pytest algos/ep10_tgv_sr/tests
```

Detached full run used in this workspace (Linux 计算机路径 `/home/ujs/...`，仅供参考):

```bash
setsid bash -c 'cd /home/ujs/mycode/thermal_lift; echo $$ > output/ep10_tgv_sr/full_run.pid; exec algos/ep10_tgv_sr/.venv/bin/python -u algos/ep10_tgv_sr/scripts/run_tgv_sr.py --output-dir output/ep10_tgv_sr --workers 4' \
  > output/ep10_tgv_sr/full_run.log 2>&1 < /dev/null &
```

The runner uses `--tgv-device auto` by default. On a CUDA-capable machine it
uses CCPi-RGL's official GPU backend and assigns one parameter worker per
detected GPU; on CPU-only machines it falls back to a single parameter worker.
Each reconstruction iteration records `tgv_backend`, `tgv_backend_status`,
`tgv_backend_device`, and `tgv_backend_error` in the returned records and output
CSV, so CCPi success and fallback paths are auditable after the run. The latest
proximal call can also be queried in Python with
`ep10_tgv_sr.tgv.get_tgv_backend_provenance()`.
Final comparison images and `best_hr_highpass.npy` are reconstructed from the
248 clean-frame real input set. Holdout reconstructions are used only for the
holdout MSE.

Monitor:

```bash
pid=$(cat output/ep10_tgv_sr/full_run.pid)
ps -p "$pid" -o pid,ppid,etime,stat,%cpu,%mem,cmd
tail -f output/ep10_tgv_sr/full_run.log
```

## 关键输出

Outputs are written to ignored `output/ep10_tgv_sr/`:

- `sweep_results.csv`
- `best_hr_highpass.npy`（下游 EP11/EP16 引用的 TGV 2x 锚点）
- `run_summary.json`
- `tgv_vs_tv_comparison.png`
- `synthetic_validation.png`

The script first runs a 64x64 piecewise-linear denoising gate. It stops before
real data if TGV does not reduce noise while keeping the ramp less staircased
than TV.

## 相关文档

- Episode 记录：`research_log/episodes/ep10_method_comparison/README.md`（Drizzle / MAP-TV / MAP-TGV 三方法对比的目标、范围与质量修订记录）
- 算法变更日志：`research_log/algorithm_changelog.md` — **ACL-007**（TGV 各向异性正则化 + 覆盖率加权数据项，修复横向条纹伪影）
- 下游消费者：`algos/ep11_dl_benchmark/`（以 TGV 2x 为基线）、`algos/ep16_budget_robustness/`（通过本 conda 环境启动 TGV 子进程）
- 跨方法可视化：`notebooks/ep10_method_comparison/`
