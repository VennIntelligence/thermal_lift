# %% [markdown]
# # EP02 — 位移标定与旋转角验证
#
# **目标**: 用帧间互相关独立检查旋转角 θ=47.6°、位移台重复定位和线性度。
#
# ---
#
# ### 运行环境
#
# ```bash
# # 本 notebook 使用项目根目录的 UV 环境
# cd /path/to/thermal_lift
# uv sync
# uv run jupyter notebook
# # Kernel 选择: 项目 .venv 下的 Python 3
# ```

# %%
%matplotlib inline

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thermal_core.displacement import (
    bootstrap_theta_ci,
    build_frame_pairs,
    build_repeat_pairs,
    coordinate_to_shift,
    fit_rotation_angle,
    linearity_regression,
    measure_frame_pairs,
)
from thermal_core.plotting import METHOD_COLOR_LIST, make_figure, savefig_academic, setup_academic_style
from thermal_core.viz import plot_displacement_field, plot_linearity, plot_theta_bootstrap

PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "AGENTS.md").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data" / "data_raw" / "infrared_avi"
EP01_OUTPUT_DIR = PROJECT_ROOT / "output" / "ep01_data_processing"
OUTPUT_DIR = PROJECT_ROOT / "output" / "ep02_displacement_calibration"
REPORT_DIR = PROJECT_ROOT / "reports" / "ep02_displacement_calibration"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

setup_academic_style()


def save_fig(fig, name):
    savefig_academic(fig, OUTPUT_DIR / name)  # close=True: 从 Gcf 注销，防止 post_execute 二次显示
    print(f"Saved: output/ep02_displacement_calibration/{name}")
    return fig  # Figure 对象仍可用于 Jupyter cell output 的 _repr_png_()


with open(PROJECT_ROOT / "configs" / "stage_calibration.json") as f:
    stage_config = json.load(f)
with open(PROJECT_ROOT / "configs" / "coordinate_set.json") as f:
    coord_config = json.load(f)

REFERENCE_THETA_DEG = float(stage_config["theta_deg"])
PIXEL_SIZE_UM = float(stage_config["pixel_size_um"])
ROI_SIZE = 320
SEARCH_RADIUS = 5
MAX_PAIR_DELTA_UM = 4

print(f"Project root: {PROJECT_ROOT}")
print(f"Data dir: {DATA_DIR}")
print(f"Reference theta: {REFERENCE_THETA_DEG:.1f} deg")
