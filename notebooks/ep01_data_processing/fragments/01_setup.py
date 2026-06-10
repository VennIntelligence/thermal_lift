# %% [markdown]
# # EP01 — SR 数据基础与主 session 建模
#
# **目标**: 为后续 LWIR 微扫描 SR 建立可复用的数据底座：确认 raw TXT/BMP 清单、坐标覆盖、
# 真实采集顺序、session 划分和可用于重建的主 session。
#
# **关键口径**: EP01 不判断 SR 是否可行，也不把 stage/文件名位移当作对齐真值。
# EP01 只回答后续 SR 应该使用哪些帧、按什么顺序建模、哪些温度段不能混合。
#
# ---
#
# ### 运行环境
#
# ```bash
# # 本 notebook 只读取 EP01 缓存产物，不重复扫描 263 个 TXT。
# cd /path/to/thermal_lift
# uv sync
#
# # 原始数据或 EP01 逻辑变更时，先重建缓存（约 15–20 s）：
# uv run python scripts/build_ep01_cache.py
#
# # 再构建/执行本 notebook（通常 < 2 s）：
# uv run python scripts/build_notebook.py notebooks/ep01_data_processing --execute
# ```

# %%
from pathlib import Path

import pandas as pd
from IPython.display import Image as NotebookImage, display

from thermal_core.ep01_cache import EP01_FIGURE_ARTIFACTS, load_ep01_cache
from thermal_core.plotting import setup_academic_style

PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "AGENTS.md").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

OUTPUT_DIR = PROJECT_ROOT / "output" / "ep01_data_processing"
REPORT_DIR = PROJECT_ROOT / "reports" / "ep01_data_processing"

setup_academic_style()
cache = load_ep01_cache(project_root=PROJECT_ROOT, output_dir=OUTPUT_DIR)

df = cache.df
pairing = cache.pairing
pairing_detail = cache.pairing_detail
rename_mapping_path = cache.rename_mapping_path
rename_special = cache.rename_special
coord_config = cache.coord_config
VALID_COORDS = cache.valid_coords
noise_floor_c = cache.noise_floor_c
matrix_summary = cache.matrix_audit_summary

def show_fig(name: str):
    """Display a cached EP01 figure (300 dpi PNG from build_ep01_cache.py)."""
    path = cache.figure_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cached figure: {path}\n"
            "Run: uv run python scripts/build_ep01_cache.py"
        )
    display(NotebookImage(filename=str(path), retina=True))

print(f"✅ 项目根目录: {PROJECT_ROOT}")
print(f"✅ EP01 缓存: {OUTPUT_DIR}")
print(f"   构建时间 (UTC): {cache.manifest.get('built_at_utc', 'unknown')}")
print(f"   重建命令: {cache.manifest.get('rebuild_command', 'scripts/build_ep01_cache.py')}")
print(
    f"   审计帧数: {len(df)}  |  原始主 session: {cache.manifest.get('n_raw_main_session_frames', 'unknown')}  |  "
    f"干净 SR 输入: {cache.manifest.get('n_sr_usable_frames', 'unknown')}  |  图表: {len(EP01_FIGURE_ARTIFACTS)} 张"
)

# %% [markdown]
# > [!NOTE]
# > **运行说明**：本 Notebook 仅作为交互式报告层，直接读取缓存的轻量级 CSV 与预生成图像。如果修改了原始数据或审计规则，请在终端运行 `uv run python scripts/build_ep01_cache.py` 以更新缓存。
