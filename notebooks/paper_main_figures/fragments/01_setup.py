# %% [markdown]
# # 论文主文图表总览（F1–F7）
#
# **本 notebook 的角色**：主文图的「策展报告」——每张图展示终稿/当前稿、给出
# 权威资产路径与重建命令、并按教程式标准解读。图的**生产**由
# `scripts/paper_figures/` 与各 algo 的 tracked 脚本完成，本 notebook 只做
# 展示与解读，不重复绘图逻辑。
#
# **运行环境**：仓库根 UV 环境。
#
# ```bash
# cd /path/to/thermal_lift
# uv sync && uv pip install -e core/
# uv run python scripts/build_notebook.py notebooks/paper_main_figures --execute
# ```
#
# **先生成图片资产**（CPU，秒级，按需）：
#
# ```bash
# uv run python scripts/paper_figures/fig01_system_calibration.py
# uv run python scripts/paper_figures/fig02_frc.py
# ```
#
# F3/F4/F7 由各自实验管线产出（见对应小节的重建命令）。
# 图表状态与依赖的权威登记：`docs/paper/09_figures_tables_assets.md`。

# %%
from pathlib import Path

from IPython.display import Image as NotebookImage
from IPython.display import Markdown, display

from thermal_core.notebook_cache import project_root

PROJECT_ROOT = project_root(Path.cwd())
PAPER_FIGS = PROJECT_ROOT / "output" / "paper_figures"


def show_figure(path: Path, rebuild: str) -> NotebookImage:
    """Display a saved 300-dpi paper figure with provenance info."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"缺少图片资产: {path}\n重建命令: {rebuild}")
    print(f"资产: {path.relative_to(PROJECT_ROOT)}")
    print(f"重建: {rebuild}")
    return NotebookImage(filename=str(path), retina=True)


print(f"paper_figures 目录: {PAPER_FIGS}")
