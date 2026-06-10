# %% [markdown]
# # EP07 — ThermalChipPhantom / TCForge 合成数据引擎验收
#
# **运行环境**: TCForge 为独立 UV 项目；Notebook 使用项目根目录环境读取预生成缓存。
#
# ```bash
# cd /path/to/thermal_lift
# uv sync
# uv pip install -e core/
#
# cd tcforge && uv sync --extra dev && cd ..
#
# uv run python scripts/build_ep07_cache.py --force
# uv run python scripts/build_notebook.py notebooks/ep07_thermal_chip_phantom --execute
# ```
#
# **本 Notebook 要回答的问题**:
#
# 1. 最新 TCForge 引擎如何把几何、热场、PSF、噪声、位移和 forward model 串成可复现数据包？
# 2. 热场分解、EP09 provisional PSF、LR 噪声纹理和 SNR 预算是否与 `phantom_smoke.json` 物理常数一致？
# 3. LR burst / highpass / manifest / smoke 契约是否可供后续 SR 算法直接消费？
#
# **边界**: EP07 不报告真实主 session SR 结论；demo 使用 `lr_shape=(64,96)`、`n_frames=16` 的小尺寸 smoke，不代表 255 帧全幅 benchmark。

# %%
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
from IPython.display import Markdown, display

from thermal_core.ep07_cache import load_ep07_cache
from thermal_core.notebook_cache import show_fig as _show_cached_fig
from thermal_core.plotting import setup_academic_style

PROJECT_ROOT = Path.cwd()
while not (PROJECT_ROOT / "AGENTS.md").exists() and PROJECT_ROOT != PROJECT_ROOT.parent:
    PROJECT_ROOT = PROJECT_ROOT.parent

OUTPUT_DIR = PROJECT_ROOT / "output" / "ep07_thermal_chip_phantom"
DEMO_DIR = OUTPUT_DIR / "demo_dataset"
PHANTOM_CONFIG = PROJECT_ROOT / "configs" / "synthetic" / "phantom_smoke.json"
REBUILD_CMD = "uv run python scripts/build_ep07_cache.py --force"

setup_academic_style()
cache = load_ep07_cache(project_root_path=PROJECT_ROOT, output_dir=OUTPUT_DIR, require_complete=False)


def show_fig(name: str) -> None:
    _show_cached_fig(cache.output_dir, name, rebuild_command=REBUILD_CMD)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def compact_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.loc[:, columns].copy()
    for col in out.columns:
        if pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].map(lambda v: "pass" if v else "fail")
        elif pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda v: f"{v:.4g}" if pd.notna(v) else "")
    return out


print(f"Project root: {PROJECT_ROOT}")
print(f"TCForge available: {cache.tcforge_available}")
print(f"Demo skipped: {cache.demo_skipped}")
print(f"Cache dir: {rel(OUTPUT_DIR)}")
print(f"Phantom config: {rel(PHANTOM_CONFIG)}")
if cache.manifest:
    print(f"Cache version: {cache.manifest.get('version', 'unknown')}")
    print(f"Built (UTC): {cache.manifest.get('built_at_utc', 'unknown')}")
if not cache.demo_skipped:
    print(f"Scene mode: {cache.scene_generation_mode}")
    print(f"Forward mode: {cache.forward_mode}")
    print(f"PSF profile: {cache.demo_config.psf_profile} ({cache.demo_config.psf_sigma_lr_px:.4f} LR px)")
    print(f"Noise model: {cache.demo_config.noise_model}, sigma={cache.demo_config.noise_sigma_c} C")

# %% [markdown]
# > **数据说明**: 所有图表和表格由 `scripts/build_ep07_cache.py` 预生成，Notebook 只读取 `output/ep07_thermal_chip_phantom/`。
# >
# > **怎么看**: `demo_skipped=False` 表示 TCForge 已成功导入并完成 demo 生成；若为 `True`，需先 `cd tcforge && uv sync` 再执行缓存重建命令。
# >
# > **核心发现**: 当前缓存绑定 TCForge v0.1.0 引擎，物理参数与 `phantom_smoke.json` 对齐：噪声 RMS 锚定 0.0724°C，PSF 默认使用 EP09 Route A forward residual 的 provisional σ≈0.226 LR px，easy ΔT=2.5°C。
