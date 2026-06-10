# %% [markdown]
# # EP06 — 2x Contour-Level SR POC
#
# **运行环境**: Notebook 构建与展示使用项目根目录 UV 环境；EP06 算法脚本按当前仓库脚本入口运行，SR 产物写入 `output/ep06_sr_poc/`，alignment ablation 产物写入 `output/ep06_alignment_ablation/`。
#
# ```bash
# cd /path/to/thermal_lift
# uv sync
#
# # 1. 运行 EP06 算法脚本（2x SR 产物）
# uv run python algos/ep06_sr_poc/scripts/run_saa.py --psf-sigma 0.5
# uv run python algos/ep06_sr_poc/scripts/run_ibp.py --max-iter 8 --psf-sigma 0.5
# uv run python algos/ep06_sr_poc/scripts/run_map_tv.py --max-iter 8 --step-size 0.25 --psf-sigma 0.5 --lambda-grid 0.0003,0.001,0.003,0.01 --no-fista
# uv run python algos/ep06_sr_poc/scripts/run_evaluation.py --center-roi-sizes 160,112,80
#
# # 2. 验证产物并构建 4x ROI 缓存
# uv run python scripts/build_ep06_cache.py
#
# # 3. 构建/执行 notebook
# uv run python scripts/build_notebook.py notebooks/ep06_sr_poc --execute
# ```
#
# 如果 `scripts/run_ep06_alignment_ablation.py` 或其产物尚未同步到当前 checkout，本 Notebook 会打印缺失说明并继续执行；不会因为 ablation 产物缺失而中断。
#
# **边界**: 本 EP 只验证 2x contour-level 结构可见性。Highpass 输出是结构图，不是绝对温度 SR；raw track 是控制轨。2x 输出网格不等价于声明 5 um 实际空间分辨率。

# %%
from pathlib import Path

import pandas as pd
from IPython.display import display

from thermal_core.ep06_cache import (
    EP06_REQUIRED_OUTPUTS,
    load_ep06_cache,
)
from thermal_core.notebook_cache import show_fig as _show_cached_fig
from thermal_core.plotting import setup_academic_style

PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "AGENTS.md").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

OUTPUT_DIR = PROJECT_ROOT / "output" / "ep06_sr_poc"
ABLATION_OUTPUT_DIR = PROJECT_ROOT / "output" / "ep06_alignment_ablation"
SWEEP_ROOT = PROJECT_ROOT / "output" / "ep06_sr_poc_data_driven_align_sweep"
SWEEP_SUMMARY_DIR = SWEEP_ROOT / "summary"
REPORT_DIR = PROJECT_ROOT / "reports" / "ep06_sr_poc"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

ABLATION_OUTPUT_PATTERNS = {
    "figures": [
        "strategy_split_half_nrmse.png",
        "strategy_gradient_artifact.png",
        "difference_to_default.png",
        "phase_coverage_2x.png",
        "difference_to_default_panels.png",
    ],
    "tables": [
        "strategy_metrics.csv",
        "split_half_metrics.csv",
        "phase_coverage.csv",
        "phase_bin_counts.csv",
        "alignment_inputs.csv",
    ],
}

setup_academic_style()
cache = load_ep06_cache(project_root_path=PROJECT_ROOT, output_dir=OUTPUT_DIR, require_complete=False)
OUTPUT_DIR_4X = cache.output_dir_4x


def show_fig(name: str, *, subdir: str = "main"):
    """Display a cached EP06 figure (300 dpi PNG from build_ep06_cache.py)."""
    output_dir = {
        "main": cache.output_dir,
        "4x": cache.output_dir_4x,
        "sweep": cache.sweep_summary_dir,
        "ablation": cache.ablation_output_dir,
    }.get(subdir, cache.output_dir)
    hint = cache.manifest.get("rebuild_command", "uv run python scripts/build_ep06_cache.py")
    if subdir == "main" and name in EP06_REQUIRED_OUTPUTS:
        hint = cache.manifest.get("algo_build_hint", hint)
    _show_cached_fig(output_dir, name, rebuild_command=hint)


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def discover_outputs(patterns: list[str], base_dir: Path = OUTPUT_DIR) -> list[Path]:
    found = {}
    for pattern in patterns:
        for path in base_dir.glob(pattern):
            if path.is_file():
                found[path.name] = path
    return [found[name] for name in sorted(found)]


def read_csv_if_exists(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / name
    if not path.exists():
        print(f"Missing table: {relative(path)}")
        return pd.DataFrame()
    return pd.read_csv(path)


def read_csv_path(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"Missing table: {relative(path)}")
        return pd.DataFrame()
    return pd.read_csv(path)


print(f"Project root: {PROJECT_ROOT}")
print(f"EP06 output: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
print(f"EP06 4x output: {OUTPUT_DIR_4X.relative_to(PROJECT_ROOT)}")
print(f"EP06 alignment ablation output: {ABLATION_OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
print(f"EP06 alignment sweep summary: {SWEEP_SUMMARY_DIR.relative_to(PROJECT_ROOT)}")
print(f"Missing required outputs: {len(cache.missing_required)}")
if cache.missing_required:
    print("Run EP06 algo scripts before executing result cells. First missing files:")
    for name in cache.missing_required[:8]:
        print(f"  - {name}")
if cache.manifest:
    print(f"Cache built (UTC): {cache.manifest.get('built_at_utc', 'unknown')}")
    print(f"4x figure built: {cache.manifest.get('four_x_built', False)}")

# %% [markdown]
# 本实验的输入帧序列限定为主扫描会话（session=2），配准参数遵循低分辨率帧至参考帧的对齐约定。为确保算法实现的可维护性与重用性，系统路径中优先载入了算法模块（`algos/ep06_sr_poc/src`），并在模块不可用时自动启用本地后备实现。
# 重建与分析产物均固化于输出目录（`output/ep06_sr_poc/`）中，其中高倍率局部图像由缓存生成脚本预先处理。本分析报告的核心证据链建立在多算法全图对比、局部特征区域（ROI）放大以及高通滤波/原始控制轨的直接目视对比基础之上，定量评估指标表作为辅助支撑工具共同参与论证。
