# %% [markdown]
# # EP03 — SR 物理边界与局部可观测性
#
# **运行环境**:
#
# ```bash
# cd /path/to/thermal_lift
# uv sync
#
# # 原始数据或 EP03 逻辑变更时，先重建缓存：
# uv run python scripts/build_ep03_cache.py
#
# # 再构建/执行本 notebook：
# uv run python scripts/build_notebook.py notebooks/ep03_theoretical_limits --execute
# ```
#
# %% [markdown]
# # EP03 — 超分辨率重建的物理边界与理论极限评估
#
# **研究目标**：基于长波红外（LWIR 8-14 $\mu\text{m}$）热成像仪的衍射极限（Diffraction Limit）、调制传递函数（MTF）衰减规律及系统噪声底（$0.0724^\circ\text{C}$），定量推导在探测器采样间距 $10.0\,\mu\text{m/pixel}$、当前系统空间分辨率 $20.0\,\mu\text{m}$ 下，实现 2x 轮廓级（Contour-level）超分辨率重建的理论容许范围与物理可行性边界。
#
# ### 1. 长波红外系统衍射极限与埃里斑
#
# 根据物理光学理论，光学系统对点光源的成像受到光瞳衍射的限制。对于工作波段为 $\lambda = 8\text{--}14\,\mu\text{m}$（中位波长 $\bar{\lambda} \approx 10\,\mu\text{m}$）的长波红外成像系统，其衍射极限导致理想点源在像平面上形成埃里斑（Airy Disk），其物理半径 $r_{\text{Airy}}$ 决定了光学系统的最小空间分辨率极限：
# $$ r_{\text{Airy}} = 1.22 \lambda F $$
# 其中 $F$ 为光学系统的光圈数（F-number）。当埃里斑尺寸大于探测器像素的物理尺寸（如本项目中的 $10.0\,\mu\text{m/pixel}$）时，系统处于衍射受限状态。当前的物理空间分辨率经标定为 $20.0\,\mu\text{m}$，处于半衰减的奈奎斯特频率临界点。
#
# ### 2. 光学截止频率与 MTF 衰减模型
#
# 设光学系统截止频率为 $f_c = \frac{1}{\lambda F}$，无像差相干光的调制传递函数 $\text{MTF}(f)$ 随空间频率 $f$ 的衰减公式为：
# $$ \text{MTF}(f) = \frac{2}{\pi} \left[ \arccos\left(\frac{f}{f_c}\right) - \frac{f}{f_c} \sqrt{1 - \left(\frac{f}{f_c}\right)^2} \right], \quad f \le f_c $$
# 在像素级低分辨率采样中，由于探测器像素孔径效应和光学点扩散函数（PSF）的低通平滑作用，频率超过探测器奈奎斯特极限 $f_s/2 = \frac{1}{2 \times 10\,\mu\text{m}} = 0.05\,\mu\text{m}^{-1}$ 的高频信号被严重压制。
#
# 超分辨率重建的本质是通过引入多帧亚像素位移，实现空间截止频率的延拓（由 $0.05\,\mu\text{m}^{-1}$ 延拓至 $0.1\,\mu\text{m}^{-1}$）。在噪声底限制为 $\sigma_n = 0.0724^\circ\text{C}$ 的物理条件下，可恢复的最高空间频率受到局部信号信噪比（SNR）的制约：
# $$ \text{SNR}(f) = \frac{A \cdot \text{MTF}(f)}{\sigma_n} \ge 1.0 $$
# 本阶段的研究重点即在于计算局部边界结构在该 SNR 阈值下的理论定位精度（Cramér-Rao Bound），为后续 EP05 配准基线与 EP06 重建方案提供理论闭环。


# %%
from pathlib import Path

import pandas as pd
from IPython.display import Image as NotebookImage, display

from thermal_core.ep03_cache import EP03_FIGURE_ARTIFACTS, load_ep03_cache
from thermal_core.plotting import setup_academic_style

PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "AGENTS.md").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

OUTPUT_DIR = PROJECT_ROOT / "output" / "ep03_theoretical_limits"
REPORT_DIR = PROJECT_ROOT / "reports" / "ep03_theoretical_limits"

setup_academic_style()
cache = load_ep03_cache(project_root_arg=PROJECT_ROOT, output_dir=OUTPUT_DIR)

THETA_DEG = cache.theta_deg
DETECTOR_PITCH_UM = cache.detector_pitch_um
SPATIAL_RESOLUTION_UM = cache.spatial_resolution_um
NOISE_SIGMA = cache.noise_sigma_c
TARGET_GRID_UM = cache.target_grid_um
PSF_SIGMAS = (0.2, 0.35, 0.5)
CRB_SIGMAS = (0.2, 0.35, 0.5, 1.0)
CRB_CONTRASTS = (0.3, 0.7, 1.0, 2.0)
CRB_N_FRAMES = (1, 4, 16, 64, 255)
CRB_PHASE_COVERAGE = (0.0, 0.5, 1.0)
RNG_SEED = 7


def show_fig(name: str) -> None:
    """Display a cached EP03 figure (300 dpi PNG from build_ep03_cache.py)."""
    path = cache.figure_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cached figure: {path}\n"
            "Run: uv run python scripts/build_ep03_cache.py"
        )
    display(NotebookImage(filename=str(path), retina=True))


print(f"Project root: {PROJECT_ROOT}")
print(f"EP03 cache: {OUTPUT_DIR}")
print(f"   built_at_utc: {cache.manifest.get('built_at_utc', 'unknown')}")
print(f"   rebuild: {cache.manifest.get('rebuild_command', 'scripts/build_ep03_cache.py')}")
print(f"   figures: {len(EP03_FIGURE_ARTIFACTS)}")
print(f"Main session frames: {cache.main_session_frames}")
print(
    f"Reference frame: {cache.reference_file} "
    f"(order={cache.reference_order}, X={cache.reference_xy[0]}, Y={cache.reference_xy[1]})"
)
print(f"Detector pitch: {DETECTOR_PITCH_UM:.1f} um/pixel")
print(f"Current spatial resolution: {SPATIAL_RESOLUTION_UM:.1f} um")
print(f"Noise floor: {NOISE_SIGMA:.4f} C")

# %% [markdown]
# ### ⚙️ 理论物理边界参数审计
#
# 审计确认本 Notebook 加载的各项全局参数具有明确的物理限制作用：
# 1. 物理采样间距（$10.0\,\mu\text{m/pixel}$）和系统标定分辨率（$20.0\,\mu\text{m}$）构成了重构尺度拓展的输入边界。
# 2. 探测器噪声底（$0.0724^\circ\text{C}$）作为随机不确定度的主要来源，决定了高频图像特征能够被算法鲁棒重构的最低对比度底限。
#
# **💡 算法决策**：所有理论推导均基于该组物理常数展开。若后续实验中的物理环境（如镜头、积分时间）发生变更，必须重新审计噪声底与实际空间分辨率，以此校正 MTF 衰减系数和可重建性边界。
