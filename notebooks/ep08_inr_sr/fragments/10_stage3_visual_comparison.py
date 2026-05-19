# %% [markdown]
# ## 9. Stage 3 Full-Frame Visual Comparison
#
# 本节从 Stage 3 run directory 读取 `hr_image.npy`、`hr_raw_control.npy`、`split_half_a.npy` 和 `split_half_b.npy`。优先比较 255 帧 run；如果 255 帧尚不存在，则自动选择当前最高帧数的可用 run。

# %%
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thermal_core.plotting import savefig_academic, setup_academic_style

STAGE3_DIR = OUTPUT_DIR / "stage3"


def _stage3_parse_visual_name(run_dir: Path) -> dict:
    parts = run_dir.name.split("_")
    aspect = parts[-1] if parts and parts[-1] in {"preserve", "stretch"} else None
    if aspect is not None:
        parts = parts[:-1]
    patch = parts[-1] if parts else None
    if patch is not None:
        parts = parts[:-1]
    n_frames = None
    if parts and parts[-1].isdigit():
        n_frames = int(parts[-1])
        parts = parts[:-1]
    return {
        "run": run_dir.name,
        "method_from_dir": "_".join(parts) if parts else None,
        "n_frames_from_dir": n_frames,
        "patch_shape_label": patch,
        "coord_aspect_mode_from_dir": aspect,
    }


def _stage3_visual_inventory(stage3_dir: Path = STAGE3_DIR) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(path for path in stage3_dir.glob("*") if path.is_dir()):
        parsed = _stage3_parse_visual_name(run_dir)
        metrics = read_json_if_exists(run_dir / "metrics.json")
        files = {
            "hr_image": run_dir / "hr_image.npy",
            "hr_raw_control": run_dir / "hr_raw_control.npy",
            "split_half_a": run_dir / "split_half_a.npy",
            "split_half_b": run_dir / "split_half_b.npy",
        }
        if not metrics and not any(path.exists() for path in files.values()):
            continue
        rows.append(
            {
                "run": run_dir.name,
                "run_dir": run_dir,
                "method": metrics.get("method") or parsed["method_from_dir"],
                "n_frames": metrics.get("n_frames") or parsed["n_frames_from_dir"],
                "coord_aspect_mode": metrics.get("coord_aspect_mode") or parsed["coord_aspect_mode_from_dir"],
                "patch_shape_label": parsed["patch_shape_label"],
                "hr_shape": metrics.get("hr_shape"),
                "stage_gate": metrics.get("stage_gate") or metrics.get("stage1_gate"),
                "has_hr_image": files["hr_image"].exists(),
                "has_raw_control": files["hr_raw_control"].exists(),
                "has_split_halves": files["split_half_a"].exists() and files["split_half_b"].exists(),
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["n_frames"] = pd.to_numeric(df["n_frames"], errors="coerce")
    return df.sort_values(["n_frames", "method", "coord_aspect_mode", "run"], ascending=[False, True, True, True])


def _load_stage3_image(path: Path, max_side: int = 720) -> np.ndarray:
    image = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
    image = np.squeeze(image)
    if image.ndim != 2:
        raise ValueError(f"expected 2D array, got shape {image.shape}")
    stride = max(1, int(math.ceil(max(image.shape) / max_side)))
    if stride > 1:
        image = image[::stride, ::stride]
    return image


def _symmetric_limit(images: list[np.ndarray], percentile: float = 99.0) -> float:
    finite = []
    for image in images:
        values = np.asarray(image)
        values = values[np.isfinite(values)]
        if values.size:
            finite.append(np.abs(values).ravel())
    if not finite:
        return 1.0
    limit = float(np.percentile(np.concatenate(finite), percentile))
    return limit if np.isfinite(limit) and limit > 0 else 1.0


def _stage3_save_fig(fig, name: str):
    path = OUTPUT_DIR / name
    savefig_academic(fig, path)
    print(f"Saved: {relative(path)}")
    return fig


stage3_visual_inventory = _stage3_visual_inventory()
stage3_visual_fig = None

if stage3_visual_inventory.empty:
    print(f"Pending: no Stage 3 image arrays found under {relative(STAGE3_DIR)}.")
else:
    status_cols = [
        "run",
        "method",
        "n_frames",
        "coord_aspect_mode",
        "patch_shape_label",
        "hr_shape",
        "stage_gate",
        "has_hr_image",
        "has_raw_control",
        "has_split_halves",
    ]
    display(stage3_visual_inventory[status_cols])

    image_ready = stage3_visual_inventory[
        stage3_visual_inventory[["has_hr_image", "has_raw_control", "has_split_halves"]].any(axis=1)
    ].copy()
    image_ready = image_ready.dropna(subset=["n_frames"])
    if image_ready.empty:
        print("Pending: Stage 3 run directories exist, but no plottable full-frame arrays are available yet.")
    else:
        target_frames = 255 if (image_ready["n_frames"] == 255).any() else int(image_ready["n_frames"].max())
        selected = image_ready[image_ready["n_frames"].eq(target_frames)].head(6)
        if selected.empty:
            print("Pending: no selected Stage 3 visual runs after frame-count filtering.")
        else:
            panels = []
            for _, row in selected.iterrows():
                run_dir = Path(row["run_dir"])
                panel = {"row": row, "hr": None, "raw": None, "split": None, "errors": []}
                for key, filename in (("hr", "hr_image.npy"), ("raw", "hr_raw_control.npy")):
                    path = run_dir / filename
                    if path.exists():
                        try:
                            panel[key] = _load_stage3_image(path)
                        except Exception as exc:
                            panel["errors"].append(f"{filename}: {exc}")
                split_a = run_dir / "split_half_a.npy"
                split_b = run_dir / "split_half_b.npy"
                if split_a.exists() and split_b.exists():
                    try:
                        panel["split"] = _load_stage3_image(split_a) - _load_stage3_image(split_b)
                    except Exception as exc:
                        panel["errors"].append(f"split halves: {exc}")
                panels.append(panel)

            hr_limit = _symmetric_limit([p["hr"] for p in panels if p["hr"] is not None])
            raw_limit = _symmetric_limit([p["raw"] for p in panels if p["raw"] is not None])
            split_limit = _symmetric_limit([p["split"] for p in panels if p["split"] is not None])

            setup_academic_style()
            fig, axes = plt.subplots(len(panels), 3, figsize=(7.2, max(2.4, 2.0 * len(panels))), squeeze=False)
            column_specs = [
                ("hr", "HR highpass", hr_limit),
                ("raw", "Raw-control", raw_limit),
                ("split", "Split-half A - B", split_limit),
            ]
            for row_idx, panel in enumerate(panels):
                row = panel["row"]
                row_label = f"{row['method']} | {int(row['n_frames'])}f | {row['coord_aspect_mode']}"
                for col_idx, (key, title, limit) in enumerate(column_specs):
                    ax = axes[row_idx, col_idx]
                    image = panel[key]
                    if image is None:
                        ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                    else:
                        ax.imshow(image, cmap="RdBu_r", vmin=-limit, vmax=limit, origin="upper")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if row_idx == 0:
                        ax.set_title(title)
                    if col_idx == 0:
                        ax.set_ylabel(row_label, rotation=0, ha="right", va="center", labelpad=58)
                if panel["errors"]:
                    print(f"{row['run']}: " + "; ".join(panel["errors"]))
            stage3_visual_fig = _stage3_save_fig(fig, "stage3_visual_comparison.png")

stage3_visual_fig

# %% [markdown]
# > **图表说明**: 每一行是一个 Stage 3 full-frame run；三列分别显示 HR highpass 重建、bicubic raw-control 参照，以及 split-half 两次重建的差异。若 255 帧 run 不存在，图会自动使用当前最高帧数。
# >
# > **怎么看**: HR highpass 用于看芯片内部结构和边缘轮廓是否连贯；raw-control 用于确认增强位置是否有普通视觉参照；split-half difference 越接近白色，表示两个独立帧子集给出的结构越一致。
# >
# > **正常/异常**: Highpass / difference 图里的白色通常表示接近零的局部变化，红色和蓝色分别表示相对局部背景的正/负响应；这不是 raw temperature 图，也不能直接解释为绝对温度升高或降低。规则条纹、棋盘纹或 split-half 中重复出现的强边缘应按 artifact 风险处理。
# >
# > **核心发现**: Full-frame Stage 3 需要同时通过普通参照、highpass 结构和 split-half 稳定性检查；单张边缘更亮的图不足以证明 contour-level SR 成功。

