# EP10 MAP-TV Sweep

Joint CPU sweep for MAP-TV regularization strength and Gaussian PSF sigma.

## 角色定位

经典对照：在 EP06 MAP-TV 的基础上做 `lambda_tv × psf_sigma` 系统化联合扫描（默认 7×4 = 28 点网格），为 2x highpass-domain SR 提供经过 split-half / holdout / artifact 多指标验证的经典 Pareto 参照，是 EP10 方法对比（Drizzle / MAP-TV / TGV）三条经典对照之一。

## 目录构成

| 路径 | 职责 |
|---|---|
| `src/ep10_map_tv_sweep/map_tv_sweep.py` | 可复用扫描逻辑：输入缓存（含签名防陈旧复用）、逐参数重建、split-half / holdout / artifact 评估、Pareto 汇总与热图 |
| `scripts/run_sweep.py` | 薄 CLI 封装（唯一入口）|
| `tests/test_map_tv_sweep.py` | 单元测试 |

复用 `../ep06_sr_poc/src/common` 的 forward model 与指标（sys.path / pytest `pythonpath` 注入，不修改 EP06 文件）。

## 环境安装

独立 UV 项目，`thermal-core` 为 editable 路径依赖；pytest 已声明：

```bash
cd algos/ep10_map_tv_sweep
uv sync
uv run pytest -q
```

## Method

This experiment evaluates a 2x highpass-domain MAP-TV reconstruction on the
EP10 248 clean-frame real input set. The data term uses the EP06 matrix-free forward model:
shift the HR estimate into each LR frame, apply a Gaussian PSF, and compare
against the highpass-preprocessed LR observation. The prior is isotropic total
variation, applied as a proximal step inside the same FISTA-style outer loop as
EP06.

The optimized proxy objective is:

```text
0.5 * mean_i || A_i x - y_i ||_2^2 + lambda_tv * TV(x)
```

where `A_i` contains the contour-refined shift and Gaussian PSF for frame `i`.
The stage/alignment information is an input prior/anchor; holdout residual and
split-half consistency are quality proxies, not independent optical truth and
not proof of 5 um temperature metrology.

The reusable sweep logic lives in `src/ep10_map_tv_sweep/`; `scripts/run_sweep.py`
is a thin CLI wrapper.

## 运行方法

```bash
cd algos/ep10_map_tv_sweep

# 全量 28 点扫描（默认自动续跑 --resume）
uv run python scripts/run_sweep.py --workers 1 --map-tv-workers 1

# 调试：只跑前 2 个网格点
uv run python scripts/run_sweep.py --max-params 2 --workers 1 --map-tv-workers 1

# 只跑合成验证（不动真实数据）
uv run python scripts/run_sweep.py --synthetic-only
```

其余参数：`--lambda-grid` / `--psf-grid`（逗号分隔浮点网格）、`--max-iter 100`、`--split-half-splits 5`、`--holdout-mod 5`、`--sigma-bg 5.0`、`--alignment-method contour_refined`、`--no-resume` / `--force`（重算）、`--no-reuse-cached-inputs`、`--io-workers`、`--skip-synthetic-validation`。

The default grid is:

- `lambda_tv`: `0.0001,0.0003,0.0005,0.001,0.002,0.005,0.01`
- `psf_sigma`: `0.10,0.18,0.30,0.50`

Outputs are written to `../../output/ep10_map_tv_sweep/`. The script writes
partial `sweep_results.csv` rows as each parameter finishes, so interrupted
runs can be resumed with the default `--resume` behavior.

Key outputs:

- `sweep_results.csv`: one row per lambda/sigma pair, including summary
  split-half, holdout, artifact, raw-control, runtime, and detail-file paths.
- `details/split_half_*.csv`: per split diagnostics for each parameter pair.
- `details/holdout_*.csv`: per holdout frame forward-residual diagnostics.
- `split_half_details.csv` and `holdout_details.csv`: concatenated detail tables.
- `cache/full_hr_*.npy`: ignored full HR highpass caches for every completed
  parameter pair; these are reproducible products, not Git-tracked source.
- `best_params.json`: Pareto frontier and top candidate metadata.
- `sweep_heatmap.png`: CVPR-style parameter heatmap.

The input cache records a signature of `data_dir`, `frame_audit_csv`,
`alignment_csv`, `alignment_method`, and highpass settings. If any of those
change, the cached input arrays are rebuilt instead of being silently reused.

For an auditable detached run（示例中 `/home/ujs/...` 是远程 Linux 计算机路径，本机请替换为实际项目根）:

```bash
cd /home/ujs/mycode/thermal_lift
setsid bash -c 'echo $$ > output/ep10_map_tv_sweep/full_run.pid; trap "rm -f output/ep10_map_tv_sweep/full_run.pid" EXIT; exec algos/ep10_map_tv_sweep/.venv/bin/python -u algos/ep10_map_tv_sweep/scripts/run_sweep.py --output-dir output/ep10_map_tv_sweep --workers 1 --map-tv-workers 1' \
  > output/ep10_map_tv_sweep/full_run.log 2>&1 < /dev/null &
```

## 相关文档

- Episode 记录：`research_log/episodes/ep10_method_comparison/README.md`（扫描范围、计算量说明与质量修正记录）
- 跨方法对比 Notebook：`notebooks/ep10_method_comparison/`
- 兄弟对照项目：`algos/ep10_drizzle/`、`algos/ep10_tgv_sr/`
- `research_log/algorithm_changelog.md`：无专属 ACL 编号（经典对照，进展记录在 ep10 episode README）
