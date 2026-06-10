# %% [markdown]
# ## 8. Stage 3 Progressive Metrics
#
# 本节只读取 `output/ep08_inr_sr/stage3/*/metrics.json`。如果 64/128/255 帧 full-frame 训练或 EP06 MAP-TV baseline 尚未运行，cell 会输出 pending 信息，不会在 notebook 中启动训练。

# %%
import pandas as pd
from IPython.display import display

stage3_metrics = collect_stage3_metrics(STAGE3_DIR)

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
    show_optional_fig(
        "stage3_progressive_metrics.png",
        "Pending: run `uv run python scripts/build_ep08_cache.py` after Stage 3 metrics exist.",
    )

# %% [markdown]
# Figure 1: Stage 3 progressive metrics. Full-frame INR and MAP-TV runs are compared across frame counts and quality gates.

# %% [markdown]
# 本分析汇总了第三阶段渐进式帧扩展（Stage 3 Progressive Expansion）的物理评估指标。数据从各优化分支目录的 `metrics.json` 文件中动态收集，以跟踪随着输入帧数（64, 128, 255 帧）及坐标长宽比的变化，算法在泛化一致性、重建鲁棒性及计算效率上的响应演化。
# 在参数趋势中，评估依然围绕四项核心约束指标展开：泛化残差、子集一致性误差及伪影得分需处于可控区间，而原始通道一致性则用于约束几何位置。由于多帧热像序列具有更长的采集时间轴（acquisition timeline），更多的帧数输入常引入额外的时变热漂移与环境波动。若随着帧数增加，伪影得分及子集 NRMSE 反而呈上升趋势，则提示位移标定模型或物理退化函数的误差正在网络中累积放大，必须结合高通剖面和一阶统计进一步判定是否存在虚假细节幻觉。
