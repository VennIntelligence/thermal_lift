# %% [markdown]
# ## 8. Stage 3 Progressive Metrics
#
# 本节只读取 `output/ep08_inr_sr/stage3/*/metrics.json`。如果 64/128/255 帧 full-frame 训练或 EP06 MAP-TV baseline 尚未运行，cell 会输出 pending 信息，不会在 notebook 中启动训练。

# %%
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thermal_core.plotting import METHOD_COLOR_LIST, savefig_academic, setup_academic_style

STAGE3_DIR = OUTPUT_DIR / "stage3"


def _stage3_parse_run_name(run_dir: Path) -> dict:
    name = run_dir.name
    parts = name.split("_")
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
    method = "_".join(parts) if parts else None
    return {
        "run": name,
        "method_from_dir": method,
        "n_frames_from_dir": n_frames,
        "patch_shape_label": patch,
        "coord_aspect_mode_from_dir": aspect,
    }


def _stage3_collect_metrics(stage3_dir: Path = STAGE3_DIR) -> pd.DataFrame:
    rows = []
    for metrics_path in sorted(stage3_dir.glob("*/metrics.json")):
        payload = read_json_if_exists(metrics_path)
        if not payload:
            continue
        parsed = _stage3_parse_run_name(metrics_path.parent)
        row = dict(payload)
        row.update(parsed)
        row["run_dir"] = metrics_path.parent
        row["metrics_file"] = relative(metrics_path)
        row["method"] = row.get("method") or parsed["method_from_dir"]
        row["n_frames"] = row.get("n_frames") or parsed["n_frames_from_dir"]
        row["coord_aspect_mode"] = row.get("coord_aspect_mode") or parsed["coord_aspect_mode_from_dir"]
        row["stage_gate"] = row.get("stage_gate") or row.get("stage1_gate")
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    for col in (
        "n_frames",
        "train_frame_count",
        "val_frame_count",
        "holdout_residual",
        "split_half_nrmse",
        "artifact_score",
        "raw_control_agreement",
        "p95_gradient",
        "best_step",
        "final_step",
        "elapsed_sec",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["method", "coord_aspect_mode", "n_frames", "run"], na_position="last")


def _stage3_save_fig(fig, name: str):
    path = OUTPUT_DIR / name
    savefig_academic(fig, path)
    print(f"Saved: {relative(path)}")
    return fig


stage3_metrics = _stage3_collect_metrics()
stage3_progressive_fig = None

if stage3_metrics.empty:
    print(f"Pending: no Stage 3 metrics found at {relative(STAGE3_DIR)}/*/metrics.json.")
    print("Run the Stage 3 training or EP06 MAP-TV baseline first; this notebook section will then populate automatically.")
else:
    key_cols = [
        "run",
        "method",
        "n_frames",
        "train_frame_count",
        "val_frame_count",
        "lr_shape",
        "hr_shape",
        "coord_aspect_mode",
        "stage_gate",
        "holdout_residual",
        "split_half_nrmse",
        "artifact_score",
        "raw_control_agreement",
        "p95_gradient",
        "best_step",
        "final_step",
        "elapsed_sec",
    ]
    display(stage3_metrics[[col for col in key_cols if col in stage3_metrics.columns]].round(6))

    trend_metrics = [
        ("holdout_residual", "Hold-out residual", "lower"),
        ("split_half_nrmse", "Split-half NRMSE", "lower"),
        ("artifact_score", "Artifact score", "lower"),
        ("raw_control_agreement", "Raw-control agreement", "higher"),
        ("p95_gradient", "P95 gradient", "proxy"),
    ]
    available = [
        item for item in trend_metrics
        if item[0] in stage3_metrics.columns and pd.to_numeric(stage3_metrics[item[0]], errors="coerce").notna().any()
    ]
    plot_df = stage3_metrics.dropna(subset=["n_frames"]).copy()
    if not available or plot_df.empty:
        print("Pending: Stage 3 metrics exist, but no plottable numeric trend columns are available yet.")
    else:
        plot_df["plot_label"] = (
            plot_df["method"].fillna("unknown").astype(str)
            + " / "
            + plot_df["coord_aspect_mode"].fillna("?").astype(str)
        )
        ncols = 2
        nrows = int(math.ceil(len(available) / ncols))
        setup_academic_style()
        fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, max(2.4, 2.2 * nrows)), squeeze=False)
        for ax in axes.ravel()[len(available):]:
            ax.axis("off")
        for idx, (metric, label, direction) in enumerate(available):
            ax = axes.ravel()[idx]
            plotted = False
            for group_idx, (group_label, group) in enumerate(plot_df.groupby("plot_label", sort=True)):
                values = pd.to_numeric(group[metric], errors="coerce")
                keep = values.notna() & group["n_frames"].notna()
                if not keep.any():
                    continue
                group_plot = group.loc[keep].assign(_value=values.loc[keep]).sort_values("n_frames")
                ax.plot(
                    group_plot["n_frames"],
                    group_plot["_value"],
                    marker="o",
                    color=METHOD_COLOR_LIST[group_idx % len(METHOD_COLOR_LIST)],
                    label=group_label,
                )
                plotted = True
            ax.set_title(f"{label} ({direction})")
            ax.set_xlabel("Input frames")
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.25)
            finite_values = pd.to_numeric(plot_df[metric], errors="coerce").dropna()
            if metric in {"holdout_residual", "split_half_nrmse", "artifact_score", "p95_gradient"} and (finite_values > 0).all():
                ax.set_yscale("log")
            if plotted:
                ax.legend(fontsize=7)
        stage3_progressive_fig = _stage3_save_fig(fig, "stage3_progressive_metrics.png")

stage3_progressive_fig

# %% [markdown]
# > **数据说明**: 表格逐个读取 Stage 3 run directory 中的 `metrics.json`，并按方法、帧数、坐标长宽比和质量门控字段汇总；趋势图只绘制已经存在的数值列，不补算缺失指标。
# >
# > **怎么看**: `holdout_residual`、`split_half_nrmse` 和 `artifact_score` 通常越低越好；`raw_control_agreement` 越高表示增强边缘更贴近 bicubic raw-control 参照；`p95_gradient` 只表示强边缘响应，数值变大可能来自真实轮廓，也可能来自噪声、振铃或伪纹理。
# >
# > **正常/异常**: 如果某个 run 缺少 split-half 或 hold-out 字段，表格中的空值表示该协议尚未生成，不应按 0 分解释。64/128/255 的曲线也不是单调性承诺；帧数增加后若 artifact 同时上升，需要回到视觉图和 split-half 差异检查。
# >
# > **核心发现**: 这组趋势用于判断 full-frame progressive expansion 是否健康推进；只有 forward consistency、稳定性、artifact 和视觉结构同时通过，才能把 Stage 3 结果解释为 contour-level 增益。

