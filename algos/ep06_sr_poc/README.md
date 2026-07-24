# EP06 — 2x Contour-Level SR POC（经典物理算法）

## 角色定位

早期经典物理 SR POC：用 SAA（shift-and-add）、IBP（迭代回投）、MAP-TV 三类经典算法在 248 clean 帧上首次验证 2x contour-level 重建，确立 highpass 主轨 + raw-control 控制轨的双轨评估范式。它同时是全项目的公共基础设施源头——`src/common/`（forward model、数据加载、指标、对齐加载）被 EP09 / EP10 系列子项目通过 sys.path / pythonpath 直接复用（复用方不修改 EP06 文件）。

## 目录构成

| 路径 | 职责 |
|---|---|
| `src/common/alignment.py` | 加载 EP05 对齐 CSV（contour_refined 等）与质量权重 |
| `src/common/data_loader.py` | 248 clean 帧加载、highpass 预处理、bicubic 上采样 |
| `src/common/forward_model.py` | matrix-free 2x forward model（shift + Gaussian PSF + 下采样），被 EP09/EP10 复用 |
| `src/common/metrics.py` | artifact score、split-half consistency 等指标（无 scikit-image 依赖）|
| `src/common/visualization.py` | 直接对比图工具 |
| `src/saa/`、`src/ibp/`、`src/map_tv/` | SAA / IBP / MAP-TV 三种重建算法实现 |
| `scripts/run_saa.py` | SAA baseline（highpass 主轨 + raw-control 控制轨）|
| `scripts/run_ibp.py` | 迭代回投重建 |
| `scripts/run_map_tv.py` | MAP-TV 重建（split-half 自动选 lambda）|
| `scripts/run_evaluation.py` | 汇总评估 + 直接对比图 |
| `tests/` | forward model 与三种算法的单元测试 |

## 环境安装

独立 UV 项目（`[tool.uv] package = false`），`thermal-core` 以 editable 路径依赖（`../../core`）由 `uv sync` 自动安装：

```bash
cd algos/ep06_sr_poc
uv sync
```

注意：`pyproject.toml` 未声明 pytest（`uv.lock` 中也没有），运行 `tests/` 需临时注入：

```bash
uv run --with pytest pytest tests/ -q
```

## 运行方法

四个脚本都会自动向上查找项目根（以 `AGENTS.md` 为标记）并注入 sys.path，可在本目录直接运行。默认输入：`data/data_raw/infrared_avi/`、`output/ep01_data_processing/frame_audit.csv`、EP05 contour alignment CSV（由 `thermal_core.alignment_paths.default_contour_alignment_csv` 解析）。

```bash
cd algos/ep06_sr_poc

# SAA baseline（含合成验证，--skip-synthetic 可跳过）
uv run python scripts/run_saa.py --alignment-method contour_refined --workers 8

# IBP
uv run python scripts/run_ibp.py --max-iter 8 --beta 0.35 --workers 8

# MAP-TV（split-half 自动选 lambda）
uv run python scripts/run_map_tv.py --lambda-grid 0.0003,0.001,0.003 --max-iter 6

# 汇总评估 + 直接对比图（需先跑完上面三个）
uv run python scripts/run_evaluation.py
```

三个重建脚本的公共参数：`--data-dir`、`--frame-audit-csv`、`--alignment-csv`、`--output-dir`、`--alignment-method {contour_refined,ncc_init,data_driven_contour_refined,data_driven_ncc_init}`、`--scale`（默认 2）、`--highpass-sigma`（默认 5.0）、`--psf-sigma`（默认 1.0）、`--splat-sigma`、`--workers`、`--seed`、合成验证相关 `--synthetic-frames` / `--synthetic-noise` / `--skip-synthetic`。

## 关键输出

写到项目根 `output/ep06_sr_poc/`（不入 Git）：各算法 highpass / raw-control 双轨重建 `.npy`、指标 JSON/CSV、对比图。其中 `map_tv_highpass.npy` 被 EP09 用作 pseudo-HR 参考。`run_evaluation.py` 另需读取 `output/ep04_global_validation/segment_summary.csv`。

## 相关文档

- Episode 记录：`research_log/episodes/ep06_sr_poc/README.md`
- 正式报告：`paper/reports/ep06_sr_poc/sr_poc_report.md`
- Notebook：`notebooks/ep06_sr_poc/`
- `research_log/algorithm_changelog.md`：无专属 ACL 编号（ACL 主要覆盖 EP07 起的学习型 SR 算法；EP06 由 episode README 与正式报告记录）
