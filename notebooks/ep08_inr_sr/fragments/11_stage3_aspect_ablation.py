# %% [markdown]
# ## 10. Stage 3 Coordinate Aspect Ablation
#
# Stage 3 默认使用 `preserve` 坐标长宽比，因为 full-frame LR 输入是 `480x640` 矩形视场；`stretch` 作为 legacy ablation 保留，用来检查把两个坐标轴都拉到 `[-1, 1]` 是否改变轮廓、artifact 或 split-half 稳定性。

# %%
import pandas as pd
from IPython.display import display

from thermal_core.ep08_cache import ASPECT_METRIC_COLS

aspect_metrics = collect_stage3_metrics(STAGE3_DIR, aspect_only=True)
aspect_pairs = pd.DataFrame()

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
        *[col for col in ASPECT_METRIC_COLS if col in aspect_metrics.columns],
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
        for metric in ASPECT_METRIC_COLS:
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
        show_optional_fig(
            "stage3_aspect_ablation.png",
            "Pending: run `uv run python scripts/build_ep08_cache.py` after matched aspect pairs exist.",
        )

# %% [markdown]
# Figure 3: Coordinate aspect ablation. Preserve and stretch coordinate normalizations are compared by stability, artifact, and contour proxies.

# %% [markdown]
# 本项消融实验（Coordinate Aspect Ablation）对隐式神经网络归一化坐标系长宽比的配置方式进行了对比评估。
# 在对比中，`preserve` 模式保留了原始图像视场 $480 \times 640$ 像素的物理几何长宽比例，而 `stretch` 模式则将各维度坐标均拉伸归一化至 $[-1, 1]$ 范围，这二者在同一重建算法、输入帧数及空间裁剪范围下进行了成对对比分析。对于以坐标作为输入的 INR（如 SIREN 和 WIRE）模型，若坐标长宽比被强行拉伸（`stretch`），网络所隐式学习的频率表征在横纵方向上将引入人为的方向性畸变。这在真实数据上常表现为特定方向上出现伪高频条纹或子集一致性恶化。因此，定量参数对比的重点在于分析坐标归一化是否破坏了芯片几何边界的物理对称性与稳定性。
