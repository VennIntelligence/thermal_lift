# %% [markdown]
# # EP02 — Raster Path, Stage Prior, and Data-Driven Alignment Evidence
#
# **目标**: 重建主 session 的 raster 采集路径，展示 stage/filename 坐标给出的 detector-space prior 覆盖，并把局部小步 NCC 放回“方向/线性 smoke test”的位置。
#
# EP02 不把 stage command 当作对齐真值，也不从相邻 2 um 小步外推多帧 SR 成败。后续 2x contour-level SR 应先使用 EP04/EP05 的 data-driven alignment anchor 与质量门控，再进入重建。
#
# 读这份 notebook 时，可以把 EP02 理解成“坐标和位移证据的说明书”。我们有两类信息：
# 一类来自电动台命令和文件名坐标，它告诉我们采集时**想要移动到哪里**；另一类来自热像帧本身的图像相似性和轮廓一致性，它告诉我们画面里**实际能可靠对齐到哪里**。
# EP02 的核心任务是把这两类证据分清楚，避免把方便使用的 command prior 误当成真实 alignment。
#
# ### 术语速查
#
# - **displacement / shift**: 两帧在 detector 像素坐标中的相对位移，通常写成 dx/dy，单位是 pixel。它描述的是图像内容在探测器上的移动，不是电动台物理坐标本身。
# - **stage prior**: 由电动台 X/Y command、旋转角 theta 和 10 um/pixel 采样间距换算得到的“预期位移”。它适合做覆盖分析、初始化和正则约束，但不是 ground truth。
# - **acquisition gap**: 两帧在真实采集顺序中的间隔。gap=1 表示连续拍摄；gap 很大时，热场变化会混入位移估计。
# - **NCC**: normalized cross-correlation，归一化互相关。它衡量两块图像在某个位移下是否相似，常用于局部对齐诊断；NCC 高不等于全局 SR 一定成功。
# - **ESF**: edge spread function，边缘扩散函数。它从边缘过渡形状观察系统模糊和边缘响应；在本项目中只能作为局部响应诊断，不能直接外推出全局可恢复分辨率。
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

import subprocess
import sys
from pathlib import Path

import pandas as pd

from thermal_core.ep02 import (
    alignment_improvement_summary,
    ep02_output_dir,
    load_frame_audit,
    load_stage_config,
    plot_alignment_comparison,
    plot_raster_acquisition_path,
    plot_small_step_diagnostics,
    plot_stage_prior_coverage,
    raster_summary,
    small_step_metrics,
    stage_prior_summary,
)
from thermal_core.plotting import setup_academic_style

PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "AGENTS.md").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

OUTPUT_DIR = ep02_output_dir(PROJECT_ROOT)
REPORT_DIR = PROJECT_ROOT / "reports" / "ep02_displacement_calibration"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

setup_academic_style()

stage_config = load_stage_config(PROJECT_ROOT)
REFERENCE_THETA_DEG = float(stage_config["theta_deg"])
PIXEL_SIZE_UM = float(stage_config["pixel_size_um"])

subprocess.run(
    [sys.executable, str(PROJECT_ROOT / "scripts" / "recompute_ep02_displacement_tables.py")],
    cwd=PROJECT_ROOT,
    check=True,
)

frame_audit = load_frame_audit(PROJECT_ROOT)

print(f"Project root: {PROJECT_ROOT}")
print(f"EP02 output: {OUTPUT_DIR}")
print(f"Stage prior: theta={REFERENCE_THETA_DEG:.1f} deg, pitch={PIXEL_SIZE_UM:.1f} um/pixel")

# %% [markdown]
# > **数值说明**: 这里打印的是 notebook 的项目根目录、EP02 输出目录，以及从 `configs/stage_calibration.json` 读取的 theta 和 pixel pitch。
# > **怎么读**: theta 是电动台坐标轴相对 detector 图像坐标轴的旋转角；pitch=10 um/pixel 是 TXT 温度矩阵的采样间距。二者一起把 stage command 换算成 detector-space displacement prior。
# > **正常/异常理解**: 正常情况下 theta 应保持为项目确认的 47.6 deg，pitch 应为 10.0 um/pixel。如果这里变成别的值，后续 prior 覆盖图和 phase 统计都会改变，需要先检查配置而不是解释算法结果。
# > **对本 Episode 的意义**: 这些数值只定义坐标 prior 的换算规则。EP02 后面会反复强调：它们不能替代从图像数据估计到的 alignment truth。
