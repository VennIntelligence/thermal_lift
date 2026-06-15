# %% [markdown]
# # EP16 Frame Budget and Shift Robustness
#
# **Runtime environment**: This notebook uses the repository root UV
# environment to read finished EP16 outputs. It does not launch the overnight
# TGV queue.
#
# ```bash
# cd /path/to/thermal_lift
# uv sync
# uv pip install -e core/
# uv run python scripts/build_notebook.py notebooks/ep16_budget_robustness --execute
# ```
#
# **Generate EP16 outputs first**:
#
# ```bash
# cd algos/ep16_budget_robustness
# CUDA_VISIBLE_DEVICES="" uv sync
# CUDA_VISIBLE_DEVICES="" uv run python scripts/run_ep16_classical.py --arms drizzle --skip-tgv
# CUDA_VISIBLE_DEVICES="" uv run python scripts/run_ep16_classical.py --arms both --run-tgv --tgv-parallel 2 --tgv-workers 6
# ```
#
# **Scope**: EP16 reports CPU classical methods only: drizzle and TGV. UNet and
# GPU MAP-TV are intentionally outside this notebook.

# %%
from pathlib import Path
import json

import pandas as pd
from IPython.display import Image as NotebookImage
from IPython.display import Markdown, display

from thermal_core.notebook_cache import project_root
from thermal_core.plotting import setup_academic_style

PROJECT_ROOT = project_root(Path.cwd())
OUTPUT_DIR = PROJECT_ROOT / "output" / "ep16_budget_robustness"
PAPER_FIGURE = PROJECT_ROOT / "output" / "paper_figures" / "fig07_budget_robustness.png"
REBUILD_COMMAND = (
    "cd algos/ep16_budget_robustness && "
    "CUDA_VISIBLE_DEVICES=\"\" uv run python scripts/run_ep16_classical.py "
    "--arms both --run-tgv --tgv-parallel 2 --tgv-workers 6"
)

setup_academic_style()


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing EP16 artifact: {path}\nRun: {REBUILD_COMMAND}")
    return path


def show_fig(path: Path) -> None:
    display(NotebookImage(filename=str(require(path)), retina=True))


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(require(OUTPUT_DIR / name))


def read_manifest() -> dict:
    return json.loads(require(OUTPUT_DIR / "run_manifest.json").read_text(encoding="utf-8"))


print(f"Project root: {PROJECT_ROOT}")
print(f"EP16 output: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
print(f"Rebuild command: {REBUILD_COMMAND}")

# %% [markdown]
# > **Data note**: EP16 reads the 248 clean main-session frames and the
# > contour-refined alignment table, then reconstructs highpass-domain HR
# > images for each frame budget, shift perturbation, and alignment source.
# >
# > **How to read it**: The CSVs are the authoritative numeric outputs. Figures
# > aggregate raw-control correlation, a proxy that should be interpreted as
# > stability against the raw-temperature mean control rather than optical
# > ground truth.
# >
# > **Caveat**: TGV split-half and FRC columns use the same drizzle
# > phase-stratified proxy on the identical subset and shifts, because exact
# > five-split TGV would multiply the overnight queue far beyond the intended
# > 17 full TGV runs. TGV-specific columns still come from the full TGV HR image
# > for raw-control correlation, artifact score, and zigzag profiles.
# >
# > **Core finding**: The completed classical CPU matrix contains 37 successful
# > unique HR runs. The frame-budget trend mainly improves stability proxies
# > between 31 and 62 frames, while the shift-perturbation experiment is best
# > read as a pressure test rather than an alignment-error calibration.
