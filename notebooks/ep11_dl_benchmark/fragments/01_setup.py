# %% [markdown]
# # EP11 — UNet 2x@40000 vs TGV 2x Center-Zoom Benchmark
#
# **运行环境**: 本 Notebook 使用项目根目录 UV 环境读取 `output/ep11_dl_benchmark/` 中已经生成的 EP11 产物；不会启动 UNet 训练，也不会重跑 TGV sweep。
#
# ```bash
# cd /path/to/thermal_lift
# uv sync
# uv pip install -e core/
# uv run python scripts/build_notebook.py notebooks/ep11_dl_benchmark --execute
# ```
#
# **先生成 EP11 产物**:
#
# ```bash
# cd algos/ep11_dl_benchmark
# uv sync
# uv run python scripts/run_unet_vs_drizzle_2x.py \
#   --checkpoint ../ep07_unet_sr/outputs/ep07_run/model_final.pt \
#   --baseline-hr ../../output/ep10_tgv_sr/best_hr_highpass.npy \
#   --baseline-sweep ../../output/ep10_tgv_sr/sweep_results.csv \
#   --baseline-summary ../../output/ep10_tgv_sr/run_summary.json \
#   --baseline-name "TGV best 2x" \
#   --output-dir ../../output/ep11_dl_benchmark \
#   --zoom 3.0 \
#   --center-fraction 0.3333333 \
#   --device cuda:1
# ```
#
# **边界**: EP11 是 2x contour-level 视觉 benchmark；3x 只是中心 ROI 的显示放大，不是 3x SR 重建。UNet@40000 来自 `outputs/ep07_run/model_final.pt`；本 Notebook 不声明 5 um 计量级温度分辨率，也不使用锐度/Tenengrad 单独裁决谁赢。

# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import Image as NotebookImage
from IPython.display import Markdown, display

from thermal_core.notebook_cache import project_root
from thermal_core.plotting import setup_academic_style

PROJECT_ROOT = project_root(Path.cwd())
OUTPUT_DIR = PROJECT_ROOT / "output" / "ep11_dl_benchmark"
REBUILD_COMMAND = (
    "cd algos/ep11_dl_benchmark && "
    "uv run python scripts/run_unet_vs_drizzle_2x.py "
    "--checkpoint ../ep07_unet_sr/outputs/ep07_run/model_final.pt "
    "--baseline-hr ../../output/ep10_tgv_sr/best_hr_highpass.npy "
    "--baseline-sweep ../../output/ep10_tgv_sr/sweep_results.csv "
    "--baseline-summary ../../output/ep10_tgv_sr/run_summary.json "
    "--baseline-name 'TGV best 2x' "
    "--output-dir ../../output/ep11_dl_benchmark "
    "--zoom 3.0 --center-fraction 0.3333333 --overlap 128 --device cuda:1"
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
        raise FileNotFoundError(f"Missing EP11 figure: {path}\nRun: {REBUILD_COMMAND}")
    display(NotebookImage(filename=str(path), retina=True))


def read_csv(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing EP11 CSV: {path}\nRun: {REBUILD_COMMAND}")
    return pd.read_csv(path)


def read_text(name: str) -> str:
    path = OUTPUT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing EP11 text artifact: {path}\nRun: {REBUILD_COMMAND}")
    return path.read_text(encoding="utf-8")


def read_json(name: str) -> dict:
    return json.loads(read_text(name))


print(f"Project root: {PROJECT_ROOT}")
print(f"EP11 output: {relative(OUTPUT_DIR)}")
print(f"Rebuild command: {REBUILD_COMMAND}")

# %% [markdown]
# > **数据说明**: 本 Notebook 是 EP11 产物展示层，默认读取 `output/ep11_dl_benchmark/` 下的 PNG、CSV、NPY 和 Markdown。长时间 UNet tiled inference 在 `algos/ep11_dl_benchmark/scripts/run_unet_vs_drizzle_2x.py` 中完成；TGV baseline 读取 EP10 已生成的 highpass 产物。
# >
# > **怎么看**: 如果这里报缺文件，先运行上面的 EP11 命令；Notebook 本身不自动补跑推理，避免打开报告时意外占用 GPU。
# >
# > **异常是否正常**: 构建 Notebook 时若 EP11 产物不存在，报错是预期行为，表示 benchmark 尚未执行，而不是算法结果失败。
# >
# > **核心发现**: EP11 把“计算”和“报告展示”分开，保证视觉结论可以复查，同时避免重复重训或重跑 TGV sweep。
