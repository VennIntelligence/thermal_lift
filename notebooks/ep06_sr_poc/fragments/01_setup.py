# %% [markdown]
# # EP06 — 2x Contour-Level SR POC
#
# **运行环境**: Notebook 构建与展示使用项目根目录 UV 环境；EP06 算法脚本按当前仓库脚本入口运行，SR 产物写入 `output/ep06_sr_poc/`，alignment ablation 产物写入 `output/ep06_alignment_ablation/`。
#
# ```bash
# cd /path/to/thermal_lift
# uv run python algos/ep06_sr_poc/scripts/run_saa.py --psf-sigma 0.5
# uv run python algos/ep06_sr_poc/scripts/run_ibp.py --max-iter 8 --psf-sigma 0.5
# uv run python algos/ep06_sr_poc/scripts/run_map_tv.py --max-iter 8 --step-size 0.25 --psf-sigma 0.5 --lambda-grid 0.0003,0.001,0.003,0.01 --no-fista
# uv run python algos/ep06_sr_poc/scripts/run_evaluation.py --center-roi-sizes 160,112,80
# uv run python scripts/run_ep06_alignment_ablation.py
# uv run python scripts/summarize_ep06_alignment_sweep.py
# uv run python scripts/build_notebook.py notebooks/ep06_sr_poc --execute
# ```
#
# 如果 `scripts/run_ep06_alignment_ablation.py` 或其产物尚未同步到当前 checkout，本 Notebook 会打印缺失说明并继续执行；不会因为 ablation 产物缺失而中断。
#
# **边界**: 本 EP 只验证 2x contour-level 结构可见性。Highpass 输出是结构图，不是绝对温度 SR；raw track 是控制轨。2x 输出网格不等价于声明 5 um 实际空间分辨率。

# %%
%matplotlib inline

from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import Image as NotebookImage
from IPython.display import display

from thermal_core.plotting import setup_academic_style

PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "AGENTS.md").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

OUTPUT_DIR = PROJECT_ROOT / "output" / "ep06_sr_poc"
ABLATION_OUTPUT_DIR = PROJECT_ROOT / "output" / "ep06_alignment_ablation"
SWEEP_ROOT = PROJECT_ROOT / "output" / "ep06_sr_poc_data_driven_align_sweep"
SWEEP_SUMMARY_DIR = SWEEP_ROOT / "summary"
REPORT_DIR = PROJECT_ROOT / "reports" / "ep06_sr_poc"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_OUTPUTS = [
    "saa_uniform_highpass.npy",
    "saa_weighted_highpass.npy",
    "saa_uniform_raw.npy",
    "saa_weighted_raw.npy",
    "bicubic_reference.npy",
    "lr_reference.npy",
    "lr_raw_reference.npy",
    "bicubic_raw_reference.npy",
    "saa_synthetic_validation.json",
    "ibp_highpass.npy",
    "ibp_raw.npy",
    "ibp_convergence.csv",
    "ibp_synthetic_validation.json",
    "map_tv_highpass.npy",
    "map_tv_raw.npy",
    "map_tv_lambda_selection.csv",
    "map_tv_convergence.csv",
    "map_tv_synthetic_validation.json",
    "evaluation_summary.csv",
    "comparison_fullview.png",
    "comparison_roi_1.png",
    "comparison_roi_2.png",
    "comparison_roi_3.png",
    "comparison_control_track.png",
    "comparison_center_raw_temperature.png",
    "gradient_magnitude_comparison.png",
    "split_half_consistency.png",
    "artifact_audit.png",
]

ABLATION_OUTPUT_PATTERNS = {
    "figures": [
        "strategy_split_half_nrmse.png",
        "strategy_gradient_artifact.png",
        "difference_to_default.png",
        "phase_coverage_2x.png",
        "difference_to_default_panels.png",
    ],
    "tables": [
        "strategy_metrics.csv",
        "split_half_metrics.csv",
        "phase_coverage.csv",
        "phase_bin_counts.csv",
        "alignment_inputs.csv",
    ],
}

missing = [name for name in REQUIRED_OUTPUTS if not (OUTPUT_DIR / name).exists()]
setup_academic_style()

print(f"Project root: {PROJECT_ROOT}")
print(f"EP06 output: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
print(f"EP06 alignment ablation output: {ABLATION_OUTPUT_DIR.relative_to(PROJECT_ROOT)}")
print(f"EP06 alignment sweep summary: {SWEEP_SUMMARY_DIR.relative_to(PROJECT_ROOT)}")
print(f"Missing outputs: {len(missing)}")
if missing:
    print("Run the EP06 scripts before executing the result cells. First missing files:")
    for name in missing[:8]:
        print(f"  - {name}")


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def discover_outputs(patterns: list[str], base_dir: Path = OUTPUT_DIR) -> list[Path]:
    found = {}
    for pattern in patterns:
        for path in base_dir.glob(pattern):
            if path.is_file():
                found[path.name] = path
    return [found[name] for name in sorted(found)]


def show_png(name: str):
    path = OUTPUT_DIR / name
    if not path.exists():
        print(f"Missing figure: {relative(path)}")
        return None
    return NotebookImage(filename=str(path))


def read_csv_if_exists(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / name
    if not path.exists():
        print(f"Missing table: {relative(path)}")
        return pd.DataFrame()
    return pd.read_csv(path)


def show_png_path(path: Path):
    if not path.exists():
        print(f"Missing figure: {relative(path)}")
        return None
    return NotebookImage(filename=str(path))


def read_csv_path(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"Missing table: {relative(path)}")
        return pd.DataFrame()
    return pd.read_csv(path)


# %% [markdown]
# > **数据说明**: 输入帧限定为 EP01 主 session，位移使用 EP05 的 LR-frame-to-reference 对齐约定。脚本把 `algos/ep06_sr_poc/src` 加入 `sys.path`，优先调用算法模块；模块缺失时使用脚本内 fallback。
# > **数据分布/模式**: 评估结果保存在 `output/ep06_sr_poc/`，Notebook 只读取产物，不在 fragments 中重新跑重建。
# > **核心发现**: 本 Notebook 的主证据是直接视觉对比图，全图、ROI 和 highpass/raw 控制轨优先于单独指标表。
