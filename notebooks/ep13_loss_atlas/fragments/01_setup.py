# %% [markdown]
# # EP13 — UNet SR Loss Atlas（损失图解 + 训练数据流）
#
# **运行环境**: 项目根目录 UV 环境；依赖 `tcforge/`（构建脚本自动 import）与 `thermal_core`。
#
# ```bash
# cd /path/to/thermal_lift
# uv sync
# uv pip install -e core/
# uv run python scripts/build_ep13_cache.py
# uv run python scripts/build_notebook.py notebooks/ep13_loss_atlas --execute
# ```
#
# **本 Notebook 做什么**:
# 1. 用 **TCForge 真实基础设施**（θ≈47.6° 旋转芯片几何）说明 EP07 训练**输入是什么**、中间经过哪些操作；
# 2. 再用中心 patch 逐步解释 `ContourSRLoss` 每一项怎么从温度场变成 loss 曲线。
#
# **示意边界**: LR burst 只展示 16 帧样品，不是 248 帧全画；离线融合与 `training_pool_2x` 一样用完整 burst。画布为教学用小尺寸（256×320），真实 pool 为 480×640 LR / 960×1280 HR。
#
# **可配置项**（生成器 `scripts/generate_training_pool.py` + `configs/synthetic/training_pool_2x.json`）:
# - `n_frames_per_scene`：每 scene 帧数（当前默认 **248**）
# - `physics_ranges.rotation_deg_center` / `rotation_jitter_deg`：芯片旋转角与抖动
# - `shift_profile` / `shift_jitter_std_px`：位移来源与 jitter
# - `noise_model` / `drift_distribution`：探测器噪声与帧间热漂变

# %%
import json
from pathlib import Path

from IPython.display import Image as NotebookImage
from IPython.display import Markdown, display

from thermal_core.notebook_cache import project_root
from thermal_core.plotting import setup_academic_style

PROJECT_ROOT = project_root(Path.cwd())
OUTPUT_DIR = PROJECT_ROOT / "output" / "ep13_loss_atlas"
REBUILD_COMMAND = "uv run python scripts/build_ep13_cache.py"

setup_academic_style()


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def save_fig(name: str):
    path = OUTPUT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing EP13 figure: {path}\nRun: {REBUILD_COMMAND}")
    print(f"💾 已保存: {relative(path)}")
    display(NotebookImage(filename=str(path), retina=True))
    return path


def read_manifest() -> dict:
    path = OUTPUT_DIR / "loss_breakdown.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing EP13 manifest: {path}\nRun: {REBUILD_COMMAND}")
    return json.loads(path.read_text(encoding="utf-8"))


manifest = read_manifest()
print(f"Project root: {PROJECT_ROOT}")
print(f"EP13 output: {relative(OUTPUT_DIR)}")
print(f"Data source: {manifest.get('data_source', '?')}, rotation={manifest.get('rotation_deg', '?')} deg")
print(f"Figures: {len(manifest['figures'])}")

# %% [markdown]
# > **数据说明**: 所有图由 `scripts/build_ep13_cache.py` 生成：先用 TCForge 造旋转芯片场景，再导出示意图与 loss 分解图。
# >
# > **怎么看**: 缺文件就先跑构建命令；Notebook 不重训 UNet，也不在真实 248 帧上推理。
# >
# > **核心发现**: 训练输入不是「248 张图直接进网络」，而是「离线融合好的 5 通道 obs_features + 在线重建的 HR 温度 GT」。
