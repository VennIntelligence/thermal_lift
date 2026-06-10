# %% [markdown]
# # EP09 — PSF Sigma 精确标定
#
# **运行环境**: 本 Notebook 使用项目根目录 UV 环境，不需要独立 `algos/` venv。
#
# ```bash
# cd /home/ujs/mycode/thermal_lift
# uv sync
# uv pip install -e core/
# uv run python algos/ep09_psf_calibration/scripts/run_forward_residual.py
# uv run python algos/ep09_psf_calibration/scripts/run_esf_fitting.py
# uv run python algos/ep09_psf_calibration/scripts/run_joint_estimation.py
# uv run python algos/ep09_psf_calibration/scripts/summarize_calibration.py
# uv run python scripts/build_ep09_cache.py
# uv run python scripts/build_notebook.py notebooks/ep09_psf_calibration --execute
# ```
#
# **边界**: 本 EP 的目标是校准 forward model 中的 Gaussian PSF sigma，并用它作为 4x SR 是否启动的前置门控。所有 sigma 默认以 LR detector pixel 为单位；写作 `HR px at 2x` 时才表示 2x 输出网格单位。

# %%
from pathlib import Path

import pandas as pd
from IPython.display import Markdown, display

from thermal_core.ep09_cache import load_ep09_cache
from thermal_core.notebook_cache import show_fig as _show_cached_fig

PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "AGENTS.md").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

OUTPUT_DIR = PROJECT_ROOT / "output" / "ep09_psf_calibration"
REPORT_DIR = PROJECT_ROOT / "reports" / "ep09_psf_calibration"
CONFIG_PATH = PROJECT_ROOT / "configs" / "psf_calibration.json"

cache = load_ep09_cache(project_root_path=PROJECT_ROOT, output_dir=OUTPUT_DIR)

summary = cache.summary
forward = cache.forward
esf = cache.esf
joint = cache.joint
route_table = cache.route_table


def read_json(name: str) -> dict:
    return cache.read_json(name)


def read_csv(name: str) -> pd.DataFrame:
    return cache.read_csv(name)


def show_fig(name: str) -> None:
    """Display a cached EP09 figure (produced by algo scripts, validated by cache builder)."""
    _show_cached_fig(
        cache.output_dir,
        name,
        rebuild_command="uv run python scripts/build_ep09_cache.py",
    )


print(f"Project root: {PROJECT_ROOT}")
print(f"EP09 output: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
print(f"Report: {(REPORT_DIR / 'psf_calibration_report.md').relative_to(PROJECT_ROOT)}")
print(f"Config: {CONFIG_PATH.relative_to(PROJECT_ROOT)}")
print(f"Cache rebuild: uv run python scripts/build_ep09_cache.py")
if summary:
    print(f"Final sigma: {summary['final_sigma_lr_px']:.4f} LR px")
    print(f"4x verdict: {summary['four_x_verdict']}")

# %% [markdown]
# > **数据说明**: Notebook 读取 `output/ep09_psf_calibration/` 中已经由 EP09 脚本生成的 CSV/JSON/PNG，不在展示层重新跑 255 帧 forward sweep 或 MAP-TV。
# >
# > **怎么看**: 下文每条路线分别展示 sigma 曲线或分布，然后在最后做门控汇总。`Route A` 是主 forward-model 估计；`Route B/C` 是独立交叉验证。
# >
# > **正常/异常**: 如果某个路线显示 `minimum_at_grid_edge=True` 或 route spread 很大，表示当前数据不足以给出“精确通过”的 PSF 标定，而不是脚本失败。
# >
# > **核心发现**: 本次 EP09 的关键不是单个 sigma 数字，而是三路线不一致导致 4x 门控未清除。
