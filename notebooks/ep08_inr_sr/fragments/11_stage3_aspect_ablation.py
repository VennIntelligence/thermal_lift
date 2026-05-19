# %% [markdown]
# ## 10. Stage 3 Coordinate Aspect Ablation
#
# Stage 3 默认使用 `preserve` 坐标长宽比，因为 full-frame LR 输入是 `480x640` 矩形视场；`stretch` 作为 legacy ablation 保留，用来检查把两个坐标轴都拉到 `[-1, 1]` 是否改变轮廓、artifact 或 split-half 稳定性。

# %%
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thermal_core.plotting import savefig_academic, setup_academic_style

STAGE3_DIR = OUTPUT_DIR / "stage3"


def _stage3_parse_aspect_name(run_dir: Path) -> dict:
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
        "aspect_from_dir": aspect,
    }


def _stage3_collect_aspect_metrics(stage3_dir: Path = STAGE3_DIR) -> pd.DataFrame:
    rows = []
    for metrics_path in sorted(stage3_dir.glob("*/metrics.json")):
        parsed = _stage3_parse_aspect_name(metrics_path.parent)
        if parsed["aspect_from_dir"] not in {"preserve", "stretch"}:
            continue
        metrics = read_json_if_exists(metrics_path)
        if not metrics:
            continue
        row = dict(metrics)
        row.update(parsed)
        row["run_dir"] = metrics_path.parent
        row["metrics_file"] = relative(metrics_path)
        row["method"] = row.get("method") or parsed["method_from_dir"]
        row["n_frames"] = row.get("n_frames") or parsed["n_frames_from_dir"]
        row["coord_aspect_mode"] = parsed["aspect_from_dir"]
        row["stage_gate"] = row.get("stage_gate") or row.get("stage1_gate")
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in (
        "n_frames",
        "holdout_residual",
        "split_half_nrmse",
        "artifact_score",
        "raw_control_agreement",
        "p95_gradient",
        "best_step",
        "final_step",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["method", "n_frames", "patch_shape_label", "coord_aspect_mode", "run"], na_position="last")


def _stage3_save_fig(fig, name: str):
    path = OUTPUT_DIR / name
    savefig_academic(fig, path)
    print(f"Saved: {relative(path)}")
    return fig


aspect_metrics = _stage3_collect_aspect_metrics()
aspect_pairs = pd.DataFrame()
stage3_aspect_fig = None
aspect_metric_cols = [
    "holdout_residual",
    "split_half_nrmse",
    "artifact_score",
    "raw_control_agreement",
    "p95_gradient",
]

if aspect_metrics.empty:
    print(f"Pending: no preserve/stretch Stage 3 metrics found at {relative(STAGE3_DIR)}/*/metrics.json.")
else:
    display_cols = [
        "run",
        "method",
        "n_frames",
        "patch_shape_label",
        "coord_aspect_mode",
        "stage_gate",
        *[col for col in aspect_metric_cols if col in aspect_metrics.columns],
    ]
    display(aspect_metrics[display_cols].round(6))

    pairs = []
    for key, group in aspect_metrics.groupby(["method", "n_frames", "patch_shape_label"], dropna=False, sort=True):
        preserve = group[group["coord_aspect_mode"].eq("preserve")]
        stretch = group[group["coord_aspect_mode"].eq("stretch")]
        if preserve.empty or stretch.empty:
            continue
        preserve_row = preserve.iloc[-1]
        stretch_row = stretch.iloc[-1]
        row = {
            "method": key[0],
            "n_frames": key[1],
            "patch_shape_label": key[2],
            "preserve_run": preserve_row["run"],
            "stretch_run": stretch_row["run"],
        }
        for metric in aspect_metric_cols:
            if metric not in aspect_metrics.columns:
                continue
            row[f"{metric}_preserve"] = preserve_row.get(metric)
            row[f"{metric}_stretch"] = stretch_row.get(metric)
            row[f"{metric}_preserve_minus_stretch"] = preserve_row.get(metric) - stretch_row.get(metric)
        pairs.append(row)

    aspect_pairs = pd.DataFrame(pairs)
    if aspect_pairs.empty:
        print("Pending: preserve and stretch metrics exist, but no matched method/n_frames/patch_shape pair is available yet.")
    else:
        display(aspect_pairs.round(6))
        plot_pairs = aspect_pairs.sort_values(["n_frames", "method"], ascending=[False, True]).head(6).copy()
        plot_pairs["label"] = (
            plot_pairs["method"].astype(str)
            + " "
            + plot_pairs["n_frames"].astype("Int64").astype(str)
            + "f"
        )
        available = [
            metric for metric in aspect_metric_cols
            if f"{metric}_preserve" in plot_pairs.columns
            and pd.to_numeric(plot_pairs[f"{metric}_preserve"], errors="coerce").notna().any()
            and pd.to_numeric(plot_pairs[f"{metric}_stretch"], errors="coerce").notna().any()
        ]
        if not available:
            print("Pending: matched aspect pairs exist, but no numeric ablation metrics are plottable.")
        else:
            ncols = 2
            nrows = int(math.ceil(len(available) / ncols))
            setup_academic_style()
            fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, max(2.4, 2.2 * nrows)), squeeze=False)
            x = np.arange(len(plot_pairs))
            width = 0.36
            for ax in axes.ravel()[len(available):]:
                ax.axis("off")
            for idx, metric in enumerate(available):
                ax = axes.ravel()[idx]
                preserve_values = pd.to_numeric(plot_pairs[f"{metric}_preserve"], errors="coerce").to_numpy(dtype=float)
                stretch_values = pd.to_numeric(plot_pairs[f"{metric}_stretch"], errors="coerce").to_numpy(dtype=float)
                ax.bar(x - width / 2, preserve_values, width, label="preserve", color="#4C72B0")
                ax.bar(x + width / 2, stretch_values, width, label="stretch", color="#C44E52")
                ax.set_title(metric)
                ax.set_xticks(x)
                ax.set_xticklabels(plot_pairs["label"], rotation=25, ha="right")
                ax.grid(True, axis="y", alpha=0.25)
                ax.legend(fontsize=7)
            stage3_aspect_fig = _stage3_save_fig(fig, "stage3_aspect_ablation.png")

stage3_aspect_fig

# %% [markdown]
# > **数据说明**: 表格只读取目录名中带 `preserve` 或 `stretch` 的 Stage 3 `metrics.json`，并只在同一 method、帧数和 patch/full-frame 设置下配对比较。
# >
# > **怎么看**: `preserve` 更符合 `480x640` 矩形视场的物理几何；`stretch` 是 legacy 坐标归一化对照。若 `stretch` 的 hold-out 更低但 split-half 或 artifact 明显更差，应优先按几何偏置或过拟合风险解释，而不是直接判为更好。
# >
# > **正常/异常**: 若当前只有 `preserve` run，pending 是正常状态；这表示 Stage 3 主线已经按默认几何推进，但还没有完成坐标 ablation。空值表示对应指标未生成，不能按 0 参与比较。
# >
# > **核心发现**: 该 ablation 的目的不是重新选择显示比例，而是验证 INR 坐标系是否在 full-frame 长宽比下引入方向性 artifact；最终判断仍需结合视觉比较和 split-half 稳定性。

