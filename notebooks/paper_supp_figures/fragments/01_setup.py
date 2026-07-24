# %% [markdown]
# # 补充材料图表总览（S-F 系列）
#
# **本 notebook 的角色**：supplementary technical appendix 的图表策展报告，
# 原与 `docs/paper/supp/A–E_*.md` 草稿配套（该草稿树 2026-07-24 已删除，
# git 历史可找回）——每张 supp 图展示当前稿、给出权威
# 资产路径、重建命令与教程式解读。生产脚本在 `scripts/paper_figures/` 与各
# algo tracked 脚本中。现行权威图集见 `docs/publication_figures/`。
#
# **运行环境**：仓库根 UV 环境。
#
# ```bash
# cd /path/to/thermal_lift
# uv sync && uv pip install -e core/
# uv run python scripts/build_notebook.py notebooks/paper_supp_figures --execute
# ```
#
# **先生成新图资产**（CPU，秒级）：
#
# ```bash
# uv run python scripts/paper_figures/fig02_frc.py          # 含 figS01
# uv run python scripts/paper_figures/figS02_psf_evidence.py
# uv run python scripts/paper_figures/figS09_fusion_pareto.py
# uv run python scripts/paper_figures/figS10_v9a_strip.py
# ```
#
# 编号对照 `docs/paper/09_figures_tables_assets.md` 的 Supplementary 表。

# %%
from pathlib import Path

from IPython.display import Image as NotebookImage
from IPython.display import Markdown, display

from thermal_core.notebook_cache import project_root

PROJECT_ROOT = project_root(Path.cwd())
PAPER_FIGS = PROJECT_ROOT / "output" / "paper_figures"


def show_figure(path: Path, rebuild: str) -> NotebookImage:
    """Display a saved 300-dpi figure with provenance info."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"缺少图片资产: {path}\n重建命令: {rebuild}")
    print(f"资产: {path.relative_to(PROJECT_ROOT)}")
    print(f"重建: {rebuild}")
    return NotebookImage(filename=str(path), retina=True)


print(f"paper_figures 目录: {PAPER_FIGS}")
