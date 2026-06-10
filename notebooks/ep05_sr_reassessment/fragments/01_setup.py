# %% [markdown]
# # EP05: 2x SR Capacity and Alignment Baseline
#
# **运行环境**: 项目根目录 UV 环境。
#
# ```bash
# cd /path/to/thermal_lift
# uv sync
# uv pip install -e core/
#
# # EP05 脚本产物变更后，先验证/写入 manifest：
# uv run python scripts/build_ep05_cache.py
#
# # 再构建/执行本 notebook：
# uv run python scripts/build_notebook.py notebooks/ep05_sr_reassessment --execute
# ```
#
# %% [markdown]
# # EP05 — 2x超分辨率物理容量评估与对齐基准标定
#
# **研究目标**：基于亚像素微扫描原理与正向成像模型，定量评估 255 帧主 Session 温度矩阵的亚像素空间相位覆盖率，对比不同数据驱动对齐算法在 Holdout 测试集上的配准指标，为 EP06 轮廓级（Contour-level）超分辨率重建（SR）选定高精度的对齐起点。
#
# ### 1. 超分辨率正向成像数学模型
#
# 设待恢复的高清温度分布图像为 $x$，观测获取的第 $k$ 帧低清温度矩阵为 $y_k$。热成像仪的物理退化过程可表示为如下的正向成像模型（Forward Imaging Model）：
# $$ y_k = D \cdot H \cdot W_k \cdot x + n_k $$
# 其中：
# - $W_k$ 为由平移和旋转运动决定的亚像素几何运动算子（Warping Operator）。
# - $H$ 为光学系统的点扩散函数（PSF）低通模糊算子。
# - $D$ 为物理孔径下采样算子（Downsampling Operator）。
# - $n_k$ 为探测器随机热噪声（加性白噪声，物理底限为 $\sigma_n = 0.0724^\circ\text{C}$）。
#
# ### 2. 2x 轮廓增强与评价指标边界
#
# 本算法框架的 2x 重建物理目标，不是在物理极限之外恢复绝对的微小温度读数（如恢复 5 µm 尺度的计量级温度信号），而是为了**提高芯片内部结构和形状的轮廓可见性（Contour-level Enhancement）**。
# 在超分辨率重建的评价中：
# 1. **回投残差（Back-projection Residual）** $\sum_k \|y_k - D H W_k \hat{x}\|_2^2$ 仅反映前向投影的数学拟合度，容易因图像过度平滑（PSF展宽）或配准过拟合而人为降低。
# 2. **Tenengrad 锐度指标** 容易受到噪声放大和重建伪影的污染而虚高。
#
# 综上，回投残差与锐度指标均不能作为超分辨率成功的唯一性证据。必须强制引入 **空间几何特征一致性（Contour Chamfer Distance）** 和 **梯度相关系数（Gradient Correlation）** 联合校验作为重建成功的验收标准。


# %%
from pathlib import Path

import pandas as pd
from IPython.display import Image as NotebookImage, display

from thermal_core.ep05 import (
    contour_alignment_tail_table,
    data_driven_correction_table,
    fractional_phase_distribution_table,
    multi_scale_phase_coverage_table,
    ordered_method_table,
    overlay_density_table,
    overlay_group_summary_table,
    overlay_group_winner_table,
    phase_capacity_table,
    trajectory_capacity_table,
    visible_shift_key_table,
    worst_contour_frames_table,
)
from thermal_core.ep05_cache import load_ep05_cache
from thermal_core.plotting import setup_academic_style

PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "AGENTS.md").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

setup_academic_style()
cache = load_ep05_cache(project_root_path=PROJECT_ROOT)

DISPLACEMENT_DIR = cache.displacement_dir
CAPACITY_DIR = cache.capacity_dir
CONTOUR_DIR = cache.contour_dir
OVERLAY_DIR = cache.overlay_dir
OUTPUT_DIR = CAPACITY_DIR
TUNING_STUDY_DIR = cache.tuning_study_dir

displacement_outputs = cache.displacement_outputs
outputs = cache.capacity_outputs
contour_outputs = cache.contour_outputs
overlay_outputs = cache.overlay_outputs
tuning_outputs = cache.tuning_outputs
summary_json = cache.summary_json


def show_fig(path: Path, *, width: int | None = None) -> None:
    """Display a cached EP05 PNG via NotebookImage (retina)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cached figure: {path}\n"
            "Run: uv run python scripts/build_ep05_cache.py"
        )
    kwargs = {"filename": str(path), "retina": True}
    if width is not None:
        kwargs["width"] = width
    display(NotebookImage(**kwargs))


pd.set_option("display.max_colwidth", 120)
print(f"Project root: {PROJECT_ROOT}")
print(f"Displacement output: {DISPLACEMENT_DIR.relative_to(PROJECT_ROOT)}")
print(f"Capacity output: {CAPACITY_DIR.relative_to(PROJECT_ROOT)}")
print(f"Contour output: {CONTOUR_DIR.relative_to(PROJECT_ROOT)}")
print(f"Overlay output: {OVERLAY_DIR.relative_to(PROJECT_ROOT)}")
print(f"Main frames scored: {summary_json['n_main_frames_scored']}")
print(f"Reference frame: {summary_json['reference_file']}")
print(f"ROI size: {summary_json['roi_size']} px, edge percentile: {summary_json['edge_percentile']:.1f}")

# %% [markdown]
# %% [markdown]
# ### ⚙️ 空间相位与对齐性能参数审计
#
# 审计确认本 Notebook 的输入严格锁定于 Session 2 的稳定主扫描数据（共 255 帧）。
#
# **💡 算法决策**：全局运动先验仅作为优化重构的初始种子。配准算法的性能优劣必须基于独立特征集（Holdout contours）进行评估。如果 Holdout 测试指标显示名义对齐的 Chamfer 距离大于 0.2 像素，则算法必须激活基于高通 NCC 或轮廓梯度的数据驱动配准精化，防止配准漂移在前向成像模型中引入累积残差。
