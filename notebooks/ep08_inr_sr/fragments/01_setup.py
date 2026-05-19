# %% [markdown]
# # EP08 — INR-based 2x Contour SR
#
# **运行环境**: Notebook 构建与展示使用项目根目录 UV 环境；EP08 算法实现是独立 UV 项目，训练和测试应在 `algos/ep08_inr_sr/` 中运行。
#
# ```bash
# cd /path/to/thermal_lift
# uv sync
# uv pip install -e core/
# uv run python scripts/build_notebook.py notebooks/ep08_inr_sr --execute
# ```
#
# 算法侧命令由 `algos/ep08_inr_sr/` 自己的 `pyproject.toml` 和脚本管理。Notebook 只读取 `output/ep08_inr_sr/`、`output/ep06_sr_poc/` 和配置占位，不在 fragment 中启动训练。
#
# **边界**: EP08 只评估 2x contour-level 结构可见性。Stage command 只能作为 prior / 初始化 / 正则约束，不能作为 alignment 真值；Highpass track 是结构响应图，不是绝对温度 SR。

# %%
%matplotlib inline

import json
from pathlib import Path

import pandas as pd
from IPython.display import Image as NotebookImage
from IPython.display import Markdown, display

from thermal_core.plotting import setup_academic_style

PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "AGENTS.md").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

NOTEBOOK_DIR = PROJECT_ROOT / "notebooks" / "ep08_inr_sr"
OUTPUT_DIR = PROJECT_ROOT / "output" / "ep08_inr_sr"
EP06_OUTPUT_DIR = PROJECT_ROOT / "output" / "ep06_sr_poc"
REPORT_DIR = PROJECT_ROOT / "reports" / "ep08_inr_sr"
EP08_ALGO_DIR = PROJECT_ROOT / "algos" / "ep08_inr_sr"
BASELINE_CONFIG = EP08_ALGO_DIR / "configs" / "ep06_baseline_metrics.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
setup_academic_style()


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"Missing table: {relative(path)}")
        return pd.DataFrame()
    return pd.read_csv(path)


def show_png_if_exists(path: Path):
    if not path.exists():
        print(f"Missing figure: {relative(path)}")
        return None
    return NotebookImage(filename=str(path))


def file_status(paths: dict[str, Path]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "item": name,
                "path": relative(path),
                "exists": path.exists(),
            }
            for name, path in paths.items()
        ]
    )


def method_output_dir(method: str) -> Path:
    aliases = {
        "siren": "siren_stage1",
        "wire": "wire_stage1",
        "deep_decoder": "deep_decoder_stage1",
    }
    return OUTPUT_DIR / aliases.get(method, method)


def method_status(method: str) -> pd.DataFrame:
    method_dir = method_output_dir(method)
    paths = {
        "metrics_json": method_dir / "metrics.json",
        "metrics_csv": method_dir / "metrics.csv",
        "history_csv": method_dir / "training_history.csv",
        "training_curve_png": method_dir / "training_curve.png",
        "highpass_png": method_dir / "hr_highpass.png",
        "raw_control_png": method_dir / "hr_raw_control.png",
        "split_half_png": method_dir / "split_half_difference.png",
        "split_indices_json": method_dir / "split_indices.json",
        "config_used_json": method_dir / "config_used.json",
        "checkpoint_pt": method_dir / "checkpoint.pt",
    }
    return file_status(paths)


baseline_config = read_json_if_exists(BASELINE_CONFIG)
setup_status = file_status(
    {
        "EP08 algorithm dir": EP08_ALGO_DIR,
        "EP08 output dir": OUTPUT_DIR,
        "EP06 output dir": EP06_OUTPUT_DIR,
        "EP06 baseline placeholder": BASELINE_CONFIG,
    }
)
display(setup_status)
print(f"Project root: {PROJECT_ROOT}")
print(f"Baseline placeholder keys: {sorted(baseline_config.keys()) if baseline_config else []}")

# %% [markdown]
# > **数据说明**: 表格只检查 EP08 notebook 需要读取的目录和 baseline placeholder 是否存在，不代表算法结果已经生成。
# >
# > **怎么看**: `exists=True` 表示路径可访问；算法输出目录存在只是 notebook 的落盘目标，不能被解读为训练完成。
# >
# > **正常/异常**: 当前阶段 SIREN / WIRE / Deep Decoder 结果缺失是正常的门控前状态。若 `EP06 baseline placeholder` 缺失，则四方对比无法记录 EP06 指标来源。
# >
# > **核心发现**: EP08 报告层已经建立读取约定，但后续结论必须等待 forward/highpass 等价性和单方法训练门控通过。
