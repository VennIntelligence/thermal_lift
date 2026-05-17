# %% [markdown]
# # EP01 — 数据完整性审计与温度统计
#
# **目标**: 从第一性原理验证全部 263 帧 TXT 数据
# **任务**: 矩阵尺寸、NaN/Inf、BMP配对、坐标覆盖、温度统计、Session 检测
#
# > ⚠️ 所有结论以实际数据为准，不预设任何旧项目参数为正确
#
# ---
#
# ### 运行环境
#
# ```bash
# # 本 notebook 使用项目根目录的 UV 环境
# cd /path/to/thermal_lift
# uv sync                  # 安装依赖（含 thermal-core）
# uv run jupyter notebook  # 启动 Jupyter
# # Kernel 选择: 项目 .venv 下的 Python 3
# ```

# %%
%matplotlib inline

import json
from pathlib import Path

import pandas as pd

from thermal_core.plotting import setup_academic_style, savefig_academic
from thermal_core.io import audit_all_frames, check_bmp_txt_pairing, \
    build_coord_repeat_map, compute_coverage_grid, detect_sessions
from thermal_core.viz import plot_coverage_heatmap, plot_temperature_histograms, \
    plot_temperature_timeline, plot_sessions

# 项目路径
PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "AGENTS.md").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data" / "data_raw" / "infrared_avi"
OUTPUT_DIR = PROJECT_ROOT / "output" / "ep01_data_processing"
REPORT_DIR = PROJECT_ROOT / "reports" / "ep01_data_processing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

setup_academic_style()

def save_fig(fig, name):
    savefig_academic(fig, OUTPUT_DIR / name)  # close=True: 从 Gcf 注销，防止 post_execute 二次显示
    print(f"💾 已保存: output/ep01_data_processing/{name}")
    return fig  # Figure 对象仍可用于 Jupyter cell output 的 _repr_png_()

# 加载坐标配置
with open(PROJECT_ROOT / "configs" / "coordinate_set.json") as f:
    coord_config = json.load(f)
VALID_COORDS = set(coord_config["x_coords_um"])

print(f"✅ 项目根目录: {PROJECT_ROOT}")
print(f"✅ 数据目录: {DATA_DIR}")
