# EP09 — PSF Sigma Calibration

CPU-only calibration of the Gaussian PSF sigma used by the Thermal Lift forward model.

## 角色定位

标定项目：为 forward model 的 Gaussian PSF sigma 提供三路线独立估计（A forward residual / B ESF / C joint hold-out），结论为一致性门控未通过（spread ≈ 1.0 px > ±0.05 px），据此否决了"4x 物理可行"主线。后续在此项目内追加了两条主线工具：σ 自校准估计器 `sigma_selfcal.py`（ACL-056–059，最终裁决"系统 σ 在本靶上不可自标定"）与 refined alignment 构建 `build_refined_alignment.py`（ACL-048，精修 shift 成为 repo 默认对齐资产）。

## 目录构成

| 路径 | 职责 |
|---|---|
| `src/psf_calibration/forward_sweep.py` | Route A：forward 残差 sigma 扫描 |
| `src/psf_calibration/esf_fitting.py` | Route B：轮廓 ESF 拟合 |
| `src/psf_calibration/joint_sweep.py` | Route C：MAP-TV hold-out 联合估计 |
| `src/psf_calibration/summary.py` | 三路线汇总、门控与报告生成 |
| `src/psf_calibration/sigma_selfcal.py` / `esf_selfcal.py` | σ 自校准估计器（e1e2 / esf 内核，ACL-056/057）|
| `src/psf_calibration/stage0a_mvp.py` | Solver V2 redesign 脚手架（σ/shift 微精修扫描）|
| `src/psf_calibration/refined_alignment.py` | Stage 0a shift 精修 → 对齐 CSV 合成 |
| `src/psf_calibration/data.py` / `utils.py` | 数据加载与路径 / worker 工具 |
| `scripts/` | 7 个 CLI 入口（见下）|
| `tests/` | 单元测试 |

## Environment（环境安装）

**本项目没有 `pyproject.toml`，不建独立 venv**，使用项目根 UV 环境运行。各脚本通过 `bootstrap()` 自动向上找到项目根（`AGENTS.md` 标记）并把本项目 `src/`、`algos/ep06_sr_poc/src`（复用 EP06 forward/数据模块）、`core/src`（`sigma_selfcal.py` 还有 `tcforge/src`）注入 `sys.path`，因此无需 pip 安装本包。

```bash
cd <项目根>   # 例如 /Users/ujs/mycode/thermal_lift
uv sync
uv pip install -e core/
```

测试用根环境运行：`uv run pytest algos/ep09_psf_calibration/tests -q`（需根环境有 pytest；或 `uv run --with pytest ...`）。

## Routes（运行方法）

在项目根运行；默认参数已指向 248 clean 帧与 contour alignment CSV，通常无需额外参数：

```bash
uv run python algos/ep09_psf_calibration/scripts/run_forward_residual.py    # Route A
uv run python algos/ep09_psf_calibration/scripts/run_esf_fitting.py         # Route B
uv run python algos/ep09_psf_calibration/scripts/run_joint_estimation.py    # Route C
uv run python algos/ep09_psf_calibration/scripts/run_stage0a_mvp.py         # Solver V2 脚手架
uv run python algos/ep09_psf_calibration/scripts/summarize_calibration.py   # 三路线汇总 + 门控
```

σ 自校准（ACL-056/057，推荐 `--kernel esf`；bench 模式必须先 PASS 才允许上真实数据）：

```bash
# Step 1: bench 验证（合成池已知真值）
uv run python algos/ep09_psf_calibration/scripts/sigma_selfcal.py \
  --bench-pool-dir data/synthetic/pool_2x_v6_bench48 --workers 48 --kernel esf

# Step 2: 真实 burst 通用模式
uv run python algos/ep09_psf_calibration/scripts/sigma_selfcal.py \
  --burst-npy burst.npy --shifts-csv shifts.csv --scale 2 --kernel esf --crop-lr 160
```

refined alignment 构建（ACL-048，`--output-csv` 为必填）：

```bash
uv run python algos/ep09_psf_calibration/scripts/build_refined_alignment.py \
  --output-csv output/ep09_psf_calibration/refined_alignment.csv
```

## 关键输出

- 三路线 / stage0a：`output/ep09_psf_calibration/`（stage0a 默认子目录 `stage0a_mvp/`）
- σ 自校准：`output/sigma_selfcal/`
- 汇总脚本更新 `configs/psf_calibration.json` 并写报告。**注意（代码与旧文档不一致，2026-07 审计）**：`summarize_calibration.py` 的 `--report-dir` 默认值是项目根 `reports/ep09_psf_calibration/`，而实际正式报告在 `paper/reports/ep09_psf_calibration/`——重跑时需显式传 `--report-dir paper/reports/ep09_psf_calibration`。

## Interpretation

All sigma values are reported in LR detector pixels unless the field name says `hr_px_at_2x`.
Route A is the primary estimate because it directly scores the EP06 forward model against held-out LR
observations. Routes B and C are independent cross-checks and gate diagnostics.

`run_stage0a_mvp.py` is a Solver V2 redesign scaffold, not a replacement for the EP09 report. It
loads the corrected 248 clean real frames, asserts the current `20 um/pixel` detector-pitch contract,
builds a fixed SAA highpass estimate, and compares a `sigma=0.5` no-refine baseline against a small
bounded PSF/shift-refinement sweep using band-limited DC residuals.

## 相关文档

- Episode 记录：`research_log/episodes/ep09_psf_calibration/README.md`（三路线结果摘要与门控决策）
- 正式报告：`paper/reports/ep09_psf_calibration/psf_calibration_report.md`
- Notebook：`notebooks/ep09_psf_calibration/`
- `research_log/algorithm_changelog.md`：ACL-048（refined alignment 升级为默认资产）、ACL-056/057/058/059（σ 自校准线交付与收口）
