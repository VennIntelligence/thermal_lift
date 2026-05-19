# %% [markdown]
# # EP07 — ThermalChipPhantom / TCForge 数据集生成架构与 Smoke Test
#
# **运行环境**: TCForge 是独立 UV 项目，依赖与仓库根目录隔离。第一次运行前执行：
#
# ```bash
# cd /path/to/thermal_lift/tcforge
# uv sync --extra dev
# ```
#
# Notebook 从仓库根目录构建和执行：
#
# ```bash
# cd /path/to/thermal_lift
# uv run python scripts/build_notebook.py notebooks/ep07_thermal_chip_phantom --execute
# ```
#
# Notebook 执行时的当前目录是仓库根目录，因此 setup cell 会临时把 `tcforge/src` 加入 `sys.path`。当前 demo 优先调用正式 TCForge API；fallback 代码只用于局部编辑时保持报告可执行，不能替代 CLI smoke 验收。
#
# **本 notebook 要回答的问题**:
#
# 1. ThermalChipPhantom 的数据集是怎样从几何结构、热物理、位移和 forward model 生成的？
# 2. 每个落盘产物的用途是什么，后续 SR 算法应消费哪一类文件？
# 3. 后台 smoke/evaluate 做了哪些检查，哪些指标只能作为数据契约检查，不能当真实 SR 结论？
# 4. Demo 可视化是否遵循项目 CVPR 风格：Times/serif、300 dpi、白底、非 `jet` colormap、图表有解释。
#
# **边界**: EP07 不是新的真实数据 SR 结论。它验证合成数据引擎的架构、forward/highpass 约定、manifest/smoke 验收和小尺寸 demo 可视化。Smoke 通过前，不应报告全幅 benchmark 指标。

# %%
%matplotlib inline

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display
from scipy.ndimage import gaussian_filter, shift as ndi_shift

from thermal_core.plotting import (
    COLORMAPS,
    FIGURE_SIZES,
    METHOD_COLORS,
    format_colorbar,
    savefig_academic,
    setup_academic_style,
)

PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "AGENTS.md").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

TCFORGE_SRC = PROJECT_ROOT / "tcforge" / "src"
if TCFORGE_SRC.exists() and str(TCFORGE_SRC) not in sys.path:
    sys.path.insert(0, str(TCFORGE_SRC))

NOTEBOOK_DIR = PROJECT_ROOT / "notebooks" / "ep07_thermal_chip_phantom"
OUTPUT_DIR = PROJECT_ROOT / "output" / "ep07_thermal_chip_phantom"
DEMO_DIR = OUTPUT_DIR / "demo_dataset"
REGRESSION_DEMO_DIR = PROJECT_ROOT / "output" / "ep07_tcforge_regression_demo"
REGRESSION_EVAL_DIR = PROJECT_ROOT / "output" / "ep07_tcforge_regression_eval"
REPORT_DIR = PROJECT_ROOT / "reports" / "ep07_thermal_chip_phantom"
for path in (OUTPUT_DIR, DEMO_DIR, REPORT_DIR):
    path.mkdir(parents=True, exist_ok=True)

setup_academic_style()

try:
    import tcforge  # type: ignore

    TCFORGE_AVAILABLE = True
    TCFORGE_VERSION = getattr(tcforge, "__version__", "unknown")
except Exception as exc:
    tcforge = None
    TCFORGE_AVAILABLE = False
    TCFORGE_VERSION = f"not importable: {exc.__class__.__name__}: {exc}"


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def save_fig(fig: plt.Figure, name: str) -> plt.Figure:
    savefig_academic(fig, OUTPUT_DIR / name)
    print(f"Saved: {relative(OUTPUT_DIR / name)}")
    return fig


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
    )


def image_limits(image: np.ndarray, *, symmetric: bool = False, q: float = 99.0) -> tuple[float, float]:
    values = np.asarray(image, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (0.0, 1.0)
    if symmetric:
        vmax = float(np.percentile(np.abs(values), q))
        vmax = max(vmax, 1e-8)
        return (-vmax, vmax)
    return (float(np.percentile(values, 100 - q)), float(np.percentile(values, q)))


def show_image(
    ax: plt.Axes,
    image: np.ndarray,
    *,
    title: str,
    cmap: str,
    colorbar_label: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    robust: bool = False,
    symmetric: bool = False,
):
    if robust:
        vmin, vmax = image_limits(image, symmetric=symmetric)
    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    if colorbar_label:
        format_colorbar(cbar, colorbar_label)
    return im


def array_contract_row(name: str, arr: np.ndarray, *, role: str) -> dict[str, object]:
    values = np.asarray(arr)
    finite = np.isfinite(values).all() if np.issubdtype(values.dtype, np.number) else True
    return {
        "array": name,
        "role": role,
        "shape": "x".join(map(str, values.shape)),
        "dtype": str(values.dtype),
        "min": float(np.min(values)) if values.size and np.issubdtype(values.dtype, np.number) else np.nan,
        "max": float(np.max(values)) if values.size and np.issubdtype(values.dtype, np.number) else np.nan,
        "finite": bool(finite),
    }


def compact_numeric_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.loc[:, columns].copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda value: f"{value:.4g}" if pd.notna(value) else "")
    return out


@dataclass(frozen=True)
class DemoConfig:
    scene_id: str = "ep07_demo_easy_000"
    seed: int = 7
    lr_shape: tuple[int, int] = (64, 96)
    scale: int = 2
    n_frames: int = 16
    pixel_size_um: float = 10.0
    base_temp_c: float = 21.0
    delta_temp_c: float = 1.4
    noise_sigma_c: float = 0.03
    psf_sigma_hr_px: float = 1.0
    psf_sigma_lr_px: float = 0.5
    highpass_sigma_lr_px: float = 5.0


DEMO_CONFIG = DemoConfig()
print(f"Project root: {PROJECT_ROOT}")
print(f"TCForge src path exists: {TCFORGE_SRC.exists()} ({relative(TCFORGE_SRC)})")
print(f"TCForge import status: {TCFORGE_VERSION}")
print(f"Demo output: {relative(DEMO_DIR)}")
print(f"Regression demo output exists: {REGRESSION_DEMO_DIR.exists()} ({relative(REGRESSION_DEMO_DIR)})")
print(f"Regression eval output exists: {REGRESSION_EVAL_DIR.exists()} ({relative(REGRESSION_EVAL_DIR)})")
print(f"Demo config: {asdict(DEMO_CONFIG)}")

# %% [markdown]
# > **数据说明**: 本 cell 只建立 Notebook 运行环境，并明确 TCForge 独立 UV 环境与仓库根目录 Notebook 构建环境的分工。`tcforge/src` 被加入 `sys.path`，使从根目录启动的 notebook 能导入独立包源码。
# >
# > **怎么看**: `TCForge import status` 应显示版本号。若显示 `not importable`，说明独立包环境或源码路径异常，后续正式 smoke 不应视为通过。
# >
# > **正常/异常**: Notebook fallback 只用于局部报告编辑。正式验收必须依赖 `tcforge/`、CLI 和 `tcforge/tests/`。
# >
# > **核心发现**: Notebook 已固定环境、输出目录和小尺寸 demo 参数，避免在报告型 notebook 中误生成全幅多场景数据。
