# %% [markdown]
# # EP05: 2x SR Capacity and Alignment Baseline
#
# **运行环境**: 项目根目录 UV 环境。
#
# ```bash
# cd /path/to/thermal_lift
# uv sync
# uv pip install -e core/
# uv run python scripts/run_ep05_alignment_sr_capacity_check.py
# uv run jupyter notebook
# # Kernel 选择: 项目 .venv 下的 Python 3
# ```
#
# **目标**: 汇总 255 帧主 session 的 2x phase-bin 覆盖、对齐方法 holdout 指标和叠图证据，为 EP06 的 2x contour-level SR POC 选择对齐起点。stage/filename 位移只作为 prior 和对照，EP04 localization 只作为 alignment anchor 与 quality gate。
#
# **读者定位**: 本 Notebook 面向有科学/数学基础、但不熟悉微扫描 SR 证据链的读者。它不是直接展示最终 SR 图像，而是在进入 EP06 之前回答四个前置问题：
#
# 1. 主 session 的 255 帧是否真的覆盖了 2x SR 所需的 sub-pixel phase？
# 2. 哪种对齐方式在未参与拟合的轮廓点上更稳定？
# 3. 轮廓叠图是否给出与数值指标一致的视觉证据？
# 4. stage command 为什么只能作为 prior，而不能作为 ground truth？
#
# **证据边界**: 本 Episode 只判断“是否值得启动 2x contour-level SR POC”和“EP06 应从哪种 alignment baseline 起步”。它不声明已经实现 5 µm 计量级温度读数，也不把显示放大、Tenengrad 变大或 back-projection residual 变小单独当作 SR 成功证据。

# %%
%matplotlib inline

from pathlib import Path

import pandas as pd
from IPython.display import Image, display

from thermal_core.ep05 import (
    load_capacity_outputs,
    ordered_method_table,
    overlay_density_table,
    phase_capacity_table,
)
from thermal_core.plotting import setup_academic_style

PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "AGENTS.md").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

OUTPUT_DIR = PROJECT_ROOT / "output" / "ep05_alignment_sr_capacity"
REQUIRED_OUTPUTS = [
    "alignment_sr_capacity_summary.json",
    "alignment_method_summary.csv",
    "alignment_method_holdout_scores.csv",
    "phase_bin_summary_2x.csv",
    "phase_bin_counts_2x.csv",
    "alignment_method_comparison.png",
    "phase_bin_coverage_2x.png",
    "alignment_overlay_evidence.png",
    "alignment_overlay_density_metrics.csv",
]
missing = [name for name in REQUIRED_OUTPUTS if not (OUTPUT_DIR / name).exists()]
if missing:
    raise FileNotFoundError(
        "Missing EP05 capacity outputs. Run: "
        "uv run python scripts/run_ep05_alignment_sr_capacity_check.py. "
        f"Missing: {missing}"
    )

setup_academic_style()
outputs = load_capacity_outputs(OUTPUT_DIR)
summary_json = outputs["summary_json"]

pd.set_option("display.max_colwidth", 120)
print(f"Project root: {PROJECT_ROOT}")
print(f"Capacity output: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
print(f"Main frames scored: {summary_json['n_main_frames_scored']}")
print(f"Reference frame: {summary_json['reference_file']}")
print(f"ROI size: {summary_json['roi_size']} px, edge percentile: {summary_json['edge_percentile']:.1f}")

# %% [markdown]
# > **数据说明**: 本 Notebook 读取 `run_ep05_alignment_sr_capacity_check.py` 生成的 EP05 capacity 输出，输入帧限制在 EP01 标记的主 session。
# > **怎么读**: 上方几行输出确认了项目根目录、结果目录、参与评分的主 session 帧数、参考帧、ROI 大小和边缘阈值。后续所有表格和图片都基于这组固定输入，因此不同方法之间的比较是同一数据、同一 ROI、同一 reference frame 下的横向比较。
# > **正常/异常理解**: 正常情况下，`Main frames scored` 应等于主 session 帧数 255，reference frame 和 ROI 参数应固定。如果帧数明显变少，说明前置筛选或输出文件不完整；如果跨 session 混入，则温度漂移会污染 alignment 和 phase 统计。
# > **核心发现**: 后续判断只讨论 2x contour-level SR POC 是否可以启动；stage 和 filename shift 是 prior/对照，不作为对齐真值，因为它们描述的是命令或文件名坐标下的期望位移，而不是热像数据实际支持的局部对齐结果。
