# %% [markdown]
# # EP02 — 步进电机位移先验与数据驱动对齐证据分析
#
# **研究目标**：重建主 Session 的光栅（Raster）采集轨迹，评估电动台指令与文件名坐标提供的探测器空间（Detector-space）位移先验覆盖度，并确定局部小步长互相关（NCC）在方向验证与线性测试中的适用边界。
#
# 在红外超分辨率重建中，步进电机指令坐标仅作为物理先验或初始值，不能作为图像空间对齐的真值。后续的 2x contour-level 超分辨率重建（SR）必须依赖基于图像数据本身的自适应亚像素对齐算法，并引入严格的质量门控机制（Quality Gate）。
#
# 本报告的核心任务是区分并标定两类关键证据：
# 1. **位移先验（Stage Prior）**：基于步进电动台在物理空间中的坐标 $(x_{\text{um}}, y_{\text{um}})$，通过几何投影关系映射到热成像仪探测器网格坐标空间下的相对偏移量 $(\Delta x, \Delta y)$。设电动台相对相机的物理轴线旋转角为 $\theta$，探测器物理像元间距为 $\text{pixel\_size} = 10.0\,\mu\text{m/pixel}$，其投影关系表示为：
#    $$ \Delta x = \frac{x_{\text{um}} \cos\theta + y_{\text{um}} \sin\theta}{\text{pixel\_size}} $$
#    $$ \Delta y = \frac{-x_{\text{um}} \sin\theta + y_{\text{um}} \cos\theta}{\text{pixel\_size}} $$
#    该位移先验仅用于确定亚像素采样点的空间网格分布，并在重构优化中作为正则化约束，不应当作绝对的对齐真值。
# 2. **数据驱动对齐量（Data-Driven Alignment）**：基于物理帧灰度值分布的归一化互相关（NCC）和边缘展开函数（ESF），从热像图像结构中直接提取的几何相对漂移。
#
# ### 📊 核心物理名词定义
#
# - **位移偏移量 (Displacement / Shift, $\Delta x, \Delta y$)**：相邻帧之间在探测器物理网格下的相对平移，单位为像素（pixel）。
# - **采集间距 (Acquisition Gap)**：物理帧在真实采集时序序列中的索引差值。当 Gap 较大时，外界环境扰动与物理热场的演化会混入位移特征估计中。
# - **归一化互相关 (NCC)**：一种衡量图像块在特定平移下重合度的几何测度。其高低主要表征局部灰度相似度，不能单独外推全局重建的成功与否。
# - **边缘扩散函数 (ESF)**：用于物理边界陡峭度及系统调制传递函数（MTF）的局部评估，在无稳定全局对齐门控的情况下，其局部分析不能反映全局分辨率增益。
#
# ---
#
# ### 运行环境
#
# ```bash
# # 本 notebook 只读取 EP02 缓存产物，不重复跑位移 NCC 测量。
# cd /path/to/thermal_lift
# uv sync
#
# # 原始数据或 EP02 逻辑变更时，先重建缓存（耗时取决于 NCC 重算）：
# uv run python scripts/build_ep02_cache.py
#
# # 再构建/执行本 notebook（通常较快）：
# uv run python scripts/build_notebook.py notebooks/ep02_displacement_calibration --execute
# ```

# %%
from pathlib import Path

import pandas as pd
from IPython.display import Image as NotebookImage, display

from thermal_core.ep02_cache import EP02_FIGURE_ARTIFACTS, load_ep02_cache
from thermal_core.plotting import setup_academic_style

PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "AGENTS.md").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

OUTPUT_DIR = PROJECT_ROOT / "output" / "ep02_displacement_calibration"
REPORT_DIR = PROJECT_ROOT / "reports" / "ep02_displacement_calibration"

setup_academic_style()
cache = load_ep02_cache(project_root_arg=PROJECT_ROOT, output_dir=OUTPUT_DIR)

frame_audit = cache.frame_audit
stage_config = cache.stage_config
REFERENCE_THETA_DEG = cache.theta_deg
PIXEL_SIZE_UM = cache.pixel_size_um


def show_fig(name: str) -> None:
    """Display a cached EP02 figure (300 dpi PNG from build_ep02_cache.py)."""
    path = cache.figure_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cached figure: {path}\n"
            "Run: uv run python scripts/build_ep02_cache.py"
        )
    display(NotebookImage(filename=str(path), retina=True))


print(f"Project root: {PROJECT_ROOT}")
print(f"EP02 cache: {OUTPUT_DIR}")
print(f"   built_at_utc: {cache.manifest.get('built_at_utc', 'unknown')}")
print(f"   rebuild: {cache.manifest.get('rebuild_command', 'scripts/build_ep02_cache.py')}")
print(f"   figures: {len(EP02_FIGURE_ARTIFACTS)}")
print(f"Stage prior: theta={REFERENCE_THETA_DEG:.1f} deg, pitch={PIXEL_SIZE_UM:.1f} um/pixel")

# %% [markdown]
# ### ⚙️ 全局配置参数审计
#
# 本阶段加载的标定角 $\theta$ 和探测器像素间距 (Detector Pitch) 具有明确的物理意义。旋转角 $\theta = 47.6^\circ$ 描述了步进电动台物理运动轴线与热像探测器像素格点轴线之间的夹角，而物理采样间距为 $10.0\,\mu\text{m/pixel}$。
#
# **💡 算法决策**：上述参数构成了电动台物理坐标与探测器像素空间位移投影关系的基石。必须保证配置参数的稳定，以防投影映射错误导致全局位移场先验发生偏移。后续的所有图像对齐诊断均基于此基准先验展开。
