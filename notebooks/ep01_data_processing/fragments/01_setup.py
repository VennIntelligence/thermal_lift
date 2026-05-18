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
    build_coord_repeat_map, compute_coverage_grid
from thermal_core.ep01 import (
    add_robust_temperature_stats,
    build_session_model,
    make_ep01_summary_table,
    plot_order_comparison,
    plot_robust_temperature_curve,
    plot_session_coverage_heatmaps,
)
from thermal_core.viz import plot_coverage_heatmap, plot_temperature_histograms, \
    plot_sessions

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

# %% [markdown]
# > **输出说明**: 这里打印 notebook 实际识别到的项目根目录和原始红外数据目录。
# >
# > **怎么读**: 项目根目录应指向 `thermal_lift`，数据目录应指向 `data/data_raw/infrared_avi`。
# > 这两个路径是后续所有读取、缓存和报告输出的基准。
# >
# > **正常/异常理解**: 正常情况下，两行路径都存在，后续 cell 可以直接读取 TXT/BMP 原始数据。
# > 如果路径指向了错误目录，常见原因是从子目录外启动 Jupyter，或新机器上尚未放置 `data/`。
# >
# > **对 EP01 的意义**: EP01 是数据审计而不是算法验证；先确认路径，是为了保证后面的帧数、
# > 坐标覆盖和 session 结论都来自同一份原始数据。
