# %% [markdown]
# # EP10 — Three-Algorithm Method Comparison
#
# **运行环境**: 本 Notebook 使用项目根目录 UV 环境，只读取 `output/ep10_drizzle/`、`output/ep10_map_tv_sweep/`、`output/ep10_tgv_sr/` 中已经生成的 CSV/JSON/NPY/PNG 产物；不会启动 Drizzle、MAP-TV 或 TGV 的长时间实验。
#
# ```bash
# cd /path/to/thermal_lift
# uv sync
# uv pip install -e core/
# uv run python scripts/build_ep10_cache.py
# uv run python scripts/build_notebook.py notebooks/ep10_method_comparison --execute
# ```
#
# **边界**: EP10 比较的是 2x highpass-domain contour-level POC 候选。Drizzle、MAP-TV 和 TGV 的指标来自各自独立脚本产物；本 Notebook 只做展示层汇总、横向图和缺失检查，不声明 5 µm 计量级温度分辨率。

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from thermal_core.ep10_cache import METHOD_ORDER, load_ep10_cache, rank_table
from thermal_core.notebook_cache import show_fig as _show_cached_fig
from thermal_core.plotting import setup_academic_style

PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "AGENTS.md").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

setup_academic_style()
cache = load_ep10_cache(project_root_path=PROJECT_ROOT)

OUTPUT_DIR = cache.output_dir
REPORT_DIR = cache.report_dir
EP10_DIRS = cache.ep10_dirs
artifacts = cache.artifacts
sweeps = cache.sweeps
best_rows = cache.best_rows
summary_table = cache.summary_table
all_candidates = cache.all_candidates
status_table = cache.status_table


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def show_fig(name: str) -> None:
    """Display a cached EP10 comparison figure."""
    path = cache.figure_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cached figure: {path}\nRun: uv run python scripts/build_ep10_cache.py"
    )
    _show_cached_fig(cache.output_dir, name, rebuild_command="uv run python scripts/build_ep10_cache.py")


def show_optional_fig(name: str, pending: str) -> None:
    """Display a cached EP10 figure if present; otherwise print a pending hint."""
    path = cache.figure_path(name)
    if path.exists():
        _show_cached_fig(cache.output_dir, name, rebuild_command="uv run python scripts/build_ep10_cache.py")
    else:
        print(pending)


print(f"Project root: {PROJECT_ROOT}")
print(f"Comparison output: {relative(OUTPUT_DIR)}")
print(f"Cache rebuild: uv run python scripts/build_ep10_cache.py")
display(status_table)

# %% [markdown]
# > **数据说明**: 上表是三套 EP10 产物的读取状态，`sweep_rows` 表示各算法已有的参数/候选行数；`synthetic_json=True` 表示对应算法至少留下了合成 sanity 记录。Notebook 只读取磁盘上的 CSV/JSON/NPY/PNG 文件，缺失时打印路径，不会自动补跑实验。
# >
# > **怎么看**: Drizzle 的参数是 `pixfrac`，MAP-TV 和 TGV 的参数主要是 `lambda` 与 PSF `sigma`。后续横向比较会把不同列名归一到 split-half NRMSE、holdout MSE、artifact score、raw-control correlation 四个共同 proxy。
# >
# > **异常是否正常**: 某个目录缺失或行数为 0 时，后续表格和图会自动跳过该算法；这说明产物尚未生成，不表示算法失败。
# >
# > **核心发现**: EP10 的展示层现在面向三算法统一比较，而不是只展示 MAP-TV 单一路径。
