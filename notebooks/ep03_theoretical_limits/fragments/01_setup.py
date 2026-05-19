# %% [markdown]
# # EP03 — SR 物理边界与局部可观测性
#
# **运行环境**: 项目根目录 UV 环境。
#
# ```bash
# cd /path/to/thermal_lift
# uv sync
# uv pip install -e core/
# uv run jupyter notebook
# # Kernel 选择: 项目 .venv 下的 Python 3
# ```
#
# **目标**: 用 10 um/pixel detector pitch、20 um 当前空间分辨率、Gaussian PSF/MTF、MTF x SNR recoverability、0.0724 C 噪声底和局部 ESF/CRB，定义 2x contour-level SR POC 的物理边界与质量门控条件。EP03 不把 stage command 当作对齐真值，也不把局部 ESF/NCC 诊断外推成全局 SR 裁决。
#
# **阅读方式**: 本 Notebook 是教程式报告，默认读者具备基础数学/信号处理背景，但不要求熟悉红外热像术语。可以把几个关键词先这样理解：
#
# - **Detector pitch**: 温度矩阵相邻像素中心在样品平面上的间距。本项目为 10 um/pixel。
# - **Spatial resolution**: 光学系统实际能分开的最小结构尺度。本项目当前校准为 20 um，不等于像素间距。
# - **PSF/MTF**: PSF 描述一个点热源被光学系统扩散成多宽；MTF 描述不同空间频率的结构会被保留多少对比度。
# - **SNR**: 局部温差信号相对于噪声底的倍数。SNR 高只说明局部结构更可观测，不自动说明 SR 成功。
# - **ESF/CRB**: ESF 是边缘扩散函数；CRB 是在给定噪声和模型下，边缘位置估计能达到的理论下界。
#
# **边界声明**: EP03 只回答“理论上哪些结构更值得作为 2x contour-level SR 的候选证据，以及哪些声明风险过高”。最终是否真正改善芯片内部结构/形状，必须由 EP05 alignment/phase baseline 与 EP06 主 session 真实 SR POC、对齐质量门控和 contour/shape evidence 共同验证。

# %%
%matplotlib inline

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import Image as NotebookImage, display

from thermal_core.ep03 import (
    build_crb_gate_summary_table,
    build_crb_localization_table,
    build_crb_sensitivity_table,
    build_mtf_attenuation_table,
    build_mtf_snr_recoverability_table,
    build_output_grid_nyquist_table,
    build_sampling_resolution_table,
    build_snr_reference_table,
    load_frame_by_row,
    measure_contour_observability,
    plot_local_anchor_confidence,
    plot_local_contour_candidate_map,
    plot_crb_esf_localization,
    plot_crb_sensitivity_surface,
    plot_mtf_psf_curves,
    plot_mtf_snr_recoverability_heatmap,
    plot_noise_floor_snr,
    plot_sampling_resolution_diagram,
    select_main_scan,
    select_reference_frame_row,
)
from thermal_core.io import load_frame
from thermal_core.plotting import savefig_academic, setup_academic_style

PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "AGENTS.md").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data" / "data_raw" / "infrared_avi"
EP01_OUTPUT_DIR = PROJECT_ROOT / "output" / "ep01_data_processing"
OUTPUT_DIR = PROJECT_ROOT / "output" / "ep03_theoretical_limits"
REPORT_DIR = PROJECT_ROOT / "reports" / "ep03_theoretical_limits"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

setup_academic_style()


def save_fig(fig, name):
    savefig_academic(fig, OUTPUT_DIR / name)
    print(f"Saved: output/ep03_theoretical_limits/{name}")
    return fig


with open(PROJECT_ROOT / "configs" / "stage_calibration.json", encoding="utf-8") as f:
    stage_config = json.load(f)
with open(PROJECT_ROOT / "configs" / "noise_floor.json", encoding="utf-8") as f:
    noise_config = json.load(f)

THETA_DEG = float(stage_config["theta_deg"])
DETECTOR_PITCH_UM = float(stage_config["pixel_size_um"])
SPATIAL_RESOLUTION_UM = float(stage_config["current_spatial_resolution_um"])
NOISE_SIGMA = float(noise_config["noise_floor_celsius"])
TARGET_GRID_UM = 5.0
PSF_SIGMAS = (0.2, 0.35, 0.5)
CRB_SIGMAS = (0.2, 0.35, 0.5, 1.0)
CRB_CONTRASTS = (0.3, 0.7, 1.0, 2.0)
CRB_N_FRAMES = (1, 4, 16, 64, 255)
CRB_PHASE_COVERAGE = (0.0, 0.5, 1.0)
RNG_SEED = 7

audit_df = pd.read_csv(EP01_OUTPUT_DIR / "frame_audit.csv")
main_df = select_main_scan(audit_df)
ref_row = select_reference_frame_row(main_df)
reference_frame = load_frame_by_row(DATA_DIR, ref_row, load_frame)

print(f"Project root: {PROJECT_ROOT}")
print(f"Main session frames: {len(main_df)}")
print(f"Reference frame: {ref_row['file']} (order={ref_row['acquisition_order']}, X={ref_row['X']}, Y={ref_row['Y']})")
print(f"Detector pitch: {DETECTOR_PITCH_UM:.1f} um/pixel")
print(f"Current spatial resolution: {SPATIAL_RESOLUTION_UM:.1f} um")
print(f"Noise floor: {NOISE_SIGMA:.4f} C")

# %% [markdown]
# > **数据说明**: 本 Notebook 读取 EP01 的 `frame_audit.csv`，只使用 `session=2` 主扫描帧，并按 `acquisition_order` 选择中点帧作为局部可观测性参考帧。
# > **读法**: `Main session frames` 是后续理论约束对应的数据范围；`Reference frame` 只是用于局部轮廓诊断的一帧代表样本；`Detector pitch`、`Current spatial resolution` 和 `Noise floor` 是固定物理输入，不是本 Notebook 拟合出来的新结论。
# > **正常/异常理解**: 正常情况下，主 session 应为 255 帧，pitch 应保持 10 um/pixel，spatial resolution 应保持 20 um。若这些值变化，说明输入配置或 EP01 产物已改变，需要先重新审计数据来源，而不能直接比较本次图表。
# > **核心发现**: EP03 的分析边界是“物理可恢复信息与局部 anchor 置信度”。它可以帮助设置 EP05 alignment/phase baseline 和 EP06 SR POC 的实验门槛，但不能替代真实 SR POC，也不能单独裁决 EP06 成败。
