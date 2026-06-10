# %% [markdown]
# # EP12 4x — UNet@2000 vs Bare Drizzle Center-Zoom Benchmark
#
# **运行环境**: 本 Notebook 使用项目根目录 UV 环境读取 `output/ep12_4x_benchmark/` 中已经生成的 EP12 产物；不会启动 EP12 训练，也不会重跑 drizzle feature 构建。
#
# ```bash
# cd /path/to/thermal_lift
# uv sync
# uv pip install -e core/
# uv run python scripts/build_notebook.py notebooks/ep12_4x_benchmark --execute
# ```
#
# **先生成 EP12 产物**:
#
# ```bash
# cd algos/ep12_4x_benchmark
# uv sync
# uv run python scripts/run_ep12_vs_drizzle_4x.py \
#   --checkpoint ../ep12_4x_sr/outputs/ep12_large_bucketv2/checkpoint_step_002000.pt \
#   --output-dir ../../output/ep12_4x_benchmark \
#   --zoom 3.0 \
#   --center-fraction 0.3333333 \
#   --device cuda:1
# ```
#
# **边界**: EP12 4x benchmark 是 contour-level 视觉推演；3x 只是中心 ROI 的显示放大，重建网格仍是真 4x (1920×2560)。UNet@2000 是 synthetic 预训练 checkpoint；裸 drizzle baseline 是 tcforge scatter-add drizzle mean，不是 STScI pixfrac drizzle。

# %%
import json
from pathlib import Path

import pandas as pd
from IPython.display import Image as NotebookImage
from IPython.display import Markdown, display

from thermal_core.notebook_cache import project_root
from thermal_core.plotting import setup_academic_style

PROJECT_ROOT = project_root(Path.cwd())
OUTPUT_DIR = PROJECT_ROOT / "output" / "ep12_4x_benchmark"
REBUILD_COMMAND = (
    "cd algos/ep12_4x_benchmark && "
    "uv run python scripts/run_ep12_vs_drizzle_4x.py "
    "--checkpoint ../ep12_4x_sr/outputs/ep12_large_bucketv2/checkpoint_step_002000.pt "
    "--output-dir ../../output/ep12_4x_benchmark "
    "--zoom 3.0 --center-fraction 0.3333333 --device cuda:1"
)

setup_academic_style()


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def show_fig(name: str) -> None:
    path = OUTPUT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing EP12 figure: {path}\nRun: {REBUILD_COMMAND}")
    display(NotebookImage(filename=str(path), retina=True))


def read_csv(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing EP12 CSV: {path}\nRun: {REBUILD_COMMAND}")
    return pd.read_csv(path)


def read_text(name: str) -> str:
    path = OUTPUT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing EP12 text artifact: {path}\nRun: {REBUILD_COMMAND}")
    return path.read_text(encoding="utf-8")


def read_json(name: str) -> dict:
    return json.loads(read_text(name))


print(f"Project root: {PROJECT_ROOT}")
print(f"EP12 output: {relative(OUTPUT_DIR)}")
print(f"Rebuild command: {REBUILD_COMMAND}")

# %% [markdown]
# > **数据说明**: 本 Notebook 是 EP12 4x 产物展示层，默认读取 `output/ep12_4x_benchmark/` 下的 PNG、CSV、NPY 和 Markdown。长时间 EP12 tiled inference 在 `algos/ep12_4x_benchmark/scripts/run_ep12_vs_drizzle_4x.py` 中完成；裸 drizzle baseline 在同一脚本内由 `tcforge.classical_sr.drizzle_features` 均值通道生成。
# >
# > **怎么看**: 如果这里报缺文件，先运行上面的 EP12 命令；Notebook 本身不自动补跑推理，避免打开报告时意外占用 GPU。
# >
# > **异常是否正常**: 构建 Notebook 时若 EP12 产物不存在，报错是预期行为，表示 benchmark 尚未执行，而不是算法结果失败。
# >
# > **核心发现**: EP12 4x benchmark 把“计算”和“报告展示”分开，保证 EP12@2000 与裸 drizzle 的视觉结论可以复查。
