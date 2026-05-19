# %% [markdown]
# # EP04 — Alignment Anchor Benchmark and Quality Gates
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
# **目标**: 将 EP04 写成服务 EP06 contour-level SR 的 alignment anchor benchmark。定位精度、CRB、SNR 和 split-half 只用于选择可靠配准锚点、质量门控和 held-out 验证；内部结构/形状仍是后续 SR 的目标区域，不因当前 localization gate 失败而被放弃。
#
# **阅读方式**:
#
# - `localization anchor` 是可重复定位的热边缘小段，用来帮助 EP06 估计或约束多帧对齐。
# - `quality gate` 是筛选规则：通过表示这个局部小段适合当配准证据，失败表示它不适合当真值或强约束。
# - `pass/fail` 不是“结构存在/不存在”、不是“SR 成功/失败”，也不是光学显微真值。
# - NCC、CRB、split-half、curvature proxy、Chamfer 等指标都是红外数据内部的一致性或几何代理指标；它们能做质量控制，不能替代未配准的光学显微标注。
# - 本 Notebook 的结论只覆盖 alignment anchor/gate；是否看清芯片内部结构，要在 EP06 的 contour-level SR 对比中继续验证。

# %%
%matplotlib inline

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display

from thermal_core.ep03 import select_main_scan, select_reference_frame_row
from thermal_core.ep04 import (
    build_ep06_gate_recommendations,
    combined_anchor_summary_table,
    create_ep04_anchor_gate_figures,
    ep06_role_margin_table,
    ep06_gate_recommendation_summary,
    failure_cooccurrence_table,
    failure_reason_table,
    ncc_esf_failure_diagnostic_table,
    plot_crb_ratio_scatter,
    plot_anchor_coverage_map,
    plot_anchor_scanline_support,
    plot_ep06_gate_recommendations,
    plot_failure_taxonomy,
    plot_global_segment_quality_distribution,
    plot_inner_failure_reasons,
    plot_cross_scanline_consistency,
    plot_normal_angle_coverage_comparison,
    plot_phase_coverage_vs_precision,
    plot_segment_scanline_pass_heatmap,
    plot_split_half_distribution,
    prepare_ep04_segment_inputs,
    run_all_inner_segments,
    run_all_segments,
    save_ep06_gate_outputs,
    save_validation_outputs,
    scanline_segment_failure_summary_table,
    segment_quality_distribution_table,
)
from thermal_core.io import load_frame
from thermal_core.plotting import savefig_academic, setup_academic_style

PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "AGENTS.md").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data" / "data_raw" / "infrared_avi"
EP01_OUTPUT_DIR = PROJECT_ROOT / "output" / "ep01_data_processing"
EP03_OUTPUT_DIR = PROJECT_ROOT / "output" / "ep03_theoretical_limits"
OUTPUT_DIR = PROJECT_ROOT / "output" / "ep04_global_validation"
REPORT_DIR = PROJECT_ROOT / "reports" / "ep04_global_validation"
INNER_OUTPUT_DIR = OUTPUT_DIR / "inner"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INNER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

setup_academic_style()


def save_fig(fig, name):
    savefig_academic(fig, OUTPUT_DIR / name)
    print(f"Saved: output/ep04_global_validation/{name}")
    return fig


with open(PROJECT_ROOT / "configs" / "stage_calibration.json", encoding="utf-8") as f:
    stage_config = json.load(f)
with open(PROJECT_ROOT / "configs" / "noise_floor.json", encoding="utf-8") as f:
    noise_config = json.load(f)

THETA_DEG = float(stage_config["theta_deg"])
PIXEL_SIZE_UM = float(stage_config["pixel_size_um"])
NOISE_SIGMA = float(noise_config["noise_floor_celsius"])
N_JOBS = max(1, min(os.cpu_count() or 1, 16))
FORCE_RERUN = False
FORCE_SEGMENT_INPUTS = False

audit_df = pd.read_csv(EP01_OUTPUT_DIR / "frame_audit.csv")
main_df = select_main_scan(audit_df)
ref_row = select_reference_frame_row(main_df)
reference_frame = load_frame(DATA_DIR / str(ref_row["file"]))

segment_inputs = prepare_ep04_segment_inputs(
    EP01_OUTPUT_DIR / "frame_audit.csv",
    DATA_DIR,
    OUTPUT_DIR,
    outer_segments_csv=EP03_OUTPUT_DIR / "contour_segments.csv",
    inner_segments_csv=EP03_OUTPUT_DIR / "inner_contour_segments.csv",
    theta_deg=THETA_DEG,
    noise_floor_c=NOISE_SIGMA,
    force=FORCE_SEGMENT_INPUTS,
)
OUTER_SEGMENTS_CSV = segment_inputs["outer_segments_csv"]
INNER_SEGMENTS_CSV = segment_inputs["inner_segments_csv"]

print(f"Project root: {PROJECT_ROOT}")
print(f"Main scan frames: {len(main_df)}")
print(f"Reference frame: {ref_row['file']} (order={ref_row['acquisition_order']})")
print(f"Outer segments: {OUTER_SEGMENTS_CSV}")
print(f"Inner segments: {INNER_SEGMENTS_CSV}")
print(f"theta={THETA_DEG:.1f} deg, pixel={PIXEL_SIZE_UM:.1f} um, noise={NOISE_SIGMA:.4f} C, n_jobs={N_JOBS}")

# %% [markdown]
# > **数据说明**: 本 Notebook 只使用 EP01 `session=2` 主扫描的 TXT 温度矩阵；EP03 segment CSV 若不存在，会在 EP04 输出目录内重建一份输入副本，不写入 EP03 目录。
# > **读法**: 先确认 `Main scan frames` 是否为主 session 的 255 帧，再确认参考帧、outer/inner segment 输入和物理配置是否来自同一套参数。这里的参考帧只给所有图提供共同坐标背景，不是光学真值。
# > **正常/异常理解**: 正常情况是只使用主 session、同一 noise floor、同一 θ 和 10 μm/pixel 配置；如果 frame 数、输入 CSV 或 θ 与项目常量不一致，应先停止解释后续 gate，因为那会改变配准先验和质量阈值语境。
# > **对本 Episode 的意义**: EP04 的所有 pass/fail 都是 anchor quality gate，不是 SR 成败标签；stage/file-name 位移只作为 prior 和诊断背景，实际 anchor 判断来自数据驱动对齐与 split-half/CRB 验证。
