# %% [markdown]
# # EP04 — Alignment Anchor Benchmark and Quality Gates
#
# **运行环境**: 项目根目录 UV 环境。
#
# ```bash
# cd /path/to/thermal_lift
# uv sync
# uv pip install -e core/
#
# # 原始数据或 EP04 逻辑变更时，先重建缓存：
# uv run python scripts/build_ep04_cache.py
#
# # 再构建/执行本 notebook：
# uv run python scripts/build_notebook.py notebooks/ep04_global_validation --execute
# ```
#
# %% [markdown]
# # EP04 — 图像对齐锚点基准与全局质量门控分析
#
# **研究目标**：基于归一化互相关（NCC）亚像素配准以及边缘展开函数（ESF）局部特征，建立超分辨率重构前的对齐锚点（Alignment Anchor）选择机制与自适应质量门控（Quality Gate），从而排除局部背景温漂或无结构区域对几何配准的污染。本 Notebook 的验证单位是 `contour segment × complete R=0 X scanline`；它验证 clean SR 输入中的完整 X scanline 子集，不等同于 248 帧 clean SR 默认输入全集。
#
# ### 1. 归一化互相关 (NCC) 亚像素对齐原理
#
# 设观测帧为 $f(x, y)$，参考帧为 $t(x, y)$，归一化互相关系数 $\gamma(u, v)$ 计算公式为：
# $$ \gamma(u, v) = \frac{\sum_{x,y} [f(x,y) - \bar{f}][t(x-u, y-v) - \bar{t}]}{\sqrt{\sum_{x,y} [f(x,y) - \bar{f}]^2 \sum_{x,y} [t(x-u, y-v) - \bar{t}]^2}} $$
# 其中 $\bar{f}$ 与 $\bar{t}$ 分别为对应区域的均值。NCC 能够在一定程度上抵御全局均温起伏的干扰。在求得离散网格的 NCC 矩阵后，通过对峰值邻域进行二次曲面拟合（Quadratic Surface Fitting），可获取亚像素级的位移估计：
# $$ (u^*, v^*) = \arg\max_{u,v} \gamma(u,v) $$
#
# ### 2. 边缘展开函数 (ESF) 局部边缘陡峭度诊断
#
# 引入一维 ESF 分析，主要用于诊断局部结构的有效性与陡峭度。通过沿边缘法线方向采集灰度过渡曲线并拟合三阶样条或高斯误差函数，可以直接估算局部点扩散函数（PSF）的宽度及局部温差对比度。
#
# ### 3. 质量门控 (Quality Gate) 作为对齐锚点的物理准则
#
# 在红外成像场景下，因环境辐射及探测器热平衡演化，图像中存在大范围温漂。同时，无结构（平坦）区域的互相关计算极易受到噪声的随机扰动。
# 质量门控（Quality Gate）的引入作为对齐锚点（Alignment Anchor）的物理准则为：
# 1. **对比度门限**：局部边缘两侧温差 $|\Delta T| \ge 3\sigma_n = 0.217^\circ\text{C}$，防止噪声区被误判为结构。
# 2. **几何单调性与稳定性**：利用折半验证（Split-half Validation）评估多次测量位置偏差，排除在时序推移中几何形状发生漂移的轮廓段。
#
# 该质量门控能剔除不可靠帧对，防止错误的位移先验污染后续 2x 亚像素超分辨率前向模型。


# %%
from pathlib import Path

import pandas as pd
from IPython.display import Image as NotebookImage, display

from thermal_core.ep04_cache import EP04_FIGURE_ARTIFACTS, load_ep04_cache
from thermal_core.plotting import setup_academic_style

PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "AGENTS.md").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

OUTPUT_DIR = PROJECT_ROOT / "output" / "ep04_global_validation"
INNER_OUTPUT_DIR = OUTPUT_DIR / "inner"
REPORT_DIR = PROJECT_ROOT / "reports" / "ep04_global_validation"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

setup_academic_style()
cache = load_ep04_cache(project_root_path=PROJECT_ROOT, output_dir=OUTPUT_DIR)

outer_results = cache.outer_results
outer_segment_summary = cache.outer_segment_summary
outer_global_summary = cache.outer_global_summary
inner_results = cache.inner_results
inner_segment_summary = cache.inner_segment_summary
inner_global_summary = cache.inner_global_summary
ep06_recommendations = cache.ep06_recommendations

OUTER_SEGMENTS_CSV = cache.outer_segments_csv
INNER_SEGMENTS_CSV = cache.inner_segments_csv
THETA_DEG = cache.theta_deg
PIXEL_SIZE_UM = cache.pixel_size_um
NOISE_SIGMA = cache.noise_floor_c


def show_fig(name: str) -> None:
    """Display a cached EP04 figure (300 dpi PNG from build_ep04_cache.py)."""
    path = cache.figure_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing cached figure: {path}\n"
            "Run: uv run python scripts/build_ep04_cache.py"
        )
    display(NotebookImage(filename=str(path), retina=True))


print(f"Project root: {PROJECT_ROOT}")
print(f"EP04 cache: {OUTPUT_DIR}")
print(f"Built at (UTC): {cache.manifest.get('built_at_utc', 'unknown')}")
print(f"Reference frame: {cache.reference_file} (order={cache.reference_order})")
print(f"Outer segments: {OUTER_SEGMENTS_CSV}")
print(f"Inner segments: {INNER_SEGMENTS_CSV}")
print(f"theta={THETA_DEG:.1f} deg, pixel={PIXEL_SIZE_UM:.1f} um, noise={NOISE_SIGMA:.4f} C")
print(f"Figures: {len(EP04_FIGURE_ARTIFACTS)}")

# %%
data_contract_keys = [
    ("Raw session=2 frames", "raw_main_session_frame_count"),
    ("Clean SR input frames", "clean_sr_input_frame_count"),
    ("EP04 complete X scanlines", "ep04_scanline_count"),
    ("EP04 scanline frames", "ep04_scanline_frame_count"),
    ("EP04 unique frames", "ep04_unique_frame_count"),
    ("EP04 clean unique frames", "ep04_clean_unique_frame_count"),
    ("EP04 scanline Y coordinates [um]", "ep04_scanline_y_um"),
    ("EP04 filter", "ep04_scanline_filter"),
]
data_contract_table = pd.DataFrame(
    [
        {"Metric": label, "Value": cache.manifest.get(key, outer_global_summary.get(key, "n/a"))}
        for label, key in data_contract_keys
    ]
)
display(data_contract_table)

# %% [markdown]
# ### 📋 EP04 数据契约边界
#
# > **数据说明**：`session == 2` 的 255 帧是原始主温度段；当前 SR 默认输入是 `is_sr_usable == True` / `is_main_session == True` 的 248 帧 clean set。EP04 只取其中完整的 `R=0` X scanline 做 localization anchor 验证。
# > **数据分布**：本次 EP04 使用 13 条完整 X scanline，每条 16 帧，共 208 个唯一帧；这些帧全部属于 248 帧 clean SR input。未进入 EP04 的 40 个 clean 帧来自不完整 scanline，仍可被后续 SR 算法使用。
# > **核心发现**：EP04 结论只约束 `segment × scanline` localization anchor 与 quality gate；不能写成“全量 248 帧 SR 输入已被 EP04 验证”，更不能回退为“255 帧 SR 输入”口径。

# %% [markdown]
# ### ⚙️ 质量门控与全局标定审计
#
# 审计确认本 Notebook 的输入依赖于由全局标定旋转角 $\theta = 47.6^\circ$、像元大小 $10.0\,\mu\text{m/pixel}$ 以及探测器噪声底限制构成的几何约束体系。
#
# **💡 算法决策**：所有通过质量门控筛选通过（Pass）的局部位移估计段，将作为后续 EP05 配准基线对齐和 EP06 主 Session 重建的几何锚点（Alignment Anchor）。失效（Fail）的帧对或局部区域虽然在配准阶段被屏蔽，但其对应的空域热图像素仍将在超分辨率空域逆求解中被采纳，仅不参与配准决策。
