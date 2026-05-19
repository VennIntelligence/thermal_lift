# %% [markdown]
# ## 5.1 Data-Driven Alignment Sweep for IBP/MAP-TV
#
# 这一节把 EP05/EP06 的 data-driven alignment 定版结果真正传回 forward-model 算法里重跑，而不是只在 SAA ablation 里观察位移策略。三组独立输出目录如下：
#
# | Experiment | Alignment CSV | Method passed to scripts | Role |
# |---|---|---|---|
# | `default_contour_refined_psf05` | `output/ep05_contour_alignment/contour_alignment_results.csv` | `data_driven_contour_refined` | EP06 主线候选 |
# | `tuned_contour_refined_psf05` | `output/ep05_alignment_tuning/full_r360_e93_rad100_s0125/contour_alignment_results.csv` | `data_driven_contour_refined` | EP05 Chamfer-tuned sensitivity check |
# | `ncc_init_psf05` | `output/ep05_contour_alignment/contour_alignment_results.csv` | `data_driven_ncc_init` | NCC init / phase-prior control |
#
# MAP-TV 使用保守设置：`psf_sigma=0.5`、`max_iter=8`、`lambda_grid=0.0003,0.001,0.003,0.01`、`--no-fista`、并用 split-half + artifact/std penalty 选择 lambda。IBP 使用同一 `psf_sigma=0.5` 检查 forward model 对 alignment 的敏感性。

# %%
sweep_summary = read_csv_path(SWEEP_SUMMARY_DIR / "sweep_method_metrics.csv")
sweep_lambda = read_csv_path(SWEEP_SUMMARY_DIR / "sweep_map_tv_lambda.csv")
sweep_delta = read_csv_path(SWEEP_SUMMARY_DIR / "sweep_delta_vs_baseline.csv")
sweep_validation = read_csv_path(SWEEP_SUMMARY_DIR / "sweep_validation_summary.csv")

if not sweep_summary.empty:
    highpass_sweep = sweep_summary[
        sweep_summary["track"].eq("highpass")
        & sweep_summary["method"].isin(["saa_weighted", "ibp", "map_tv"])
    ].copy()
    display(
        highpass_sweep[
            [
                "experiment",
                "method",
                "std",
                "std_ratio_to_lr",
                "mean_gradient",
                "p95_gradient",
                "artifact_score",
                "contour_chamfer_lr_px",
                "corr_to_bicubic",
            ]
        ].round(4)
    )

# %% [markdown]
# > **数据说明**: 这张表来自 `sweep_method_metrics.csv`，只保留 highpass 主轨里的 SAA weighted、IBP、MAP-TV。三个 experiment 使用相同主 session、相同 2x grid、相同 evaluation 口径，只改变 alignment field 或 forward-model 参数。
# >
# > **怎么看**: `artifact_score`、`std_ratio_to_lr` 和 `contour_chamfer_lr_px` 越低通常越稳；`mean_gradient`/`p95_gradient` 越高只表示边缘响应更强，不能单独判胜。这里最重要的是同一算法在不同 alignment 下是否稳定，尤其是 IBP/MAP-TV 有没有因为 alignment residual 变成红蓝过冲或 TV staircasing。
# >
# > **正常/异常**: MAP-TV 的 `mean_gradient` 低于 SAA/IBP 是正常结果，因为它在当前设置下选择强正则；这不代表它失败，但说明它不是“最锐”的候选。NCC init 的 gradient 变高同时 artifact score 也变高，应读成高风险对照，而不是直接读成更清楚。
# >
# > **核心发现**: full sweep 没有推翻 EP06 的保守结论。Default contour refined 仍是主线 alignment；tuned refined 没有让 MAP-TV/IBP 变成明显胜出；NCC init 在 forward-model 算法下更容易抬高 artifact。

# %%
if not sweep_lambda.empty:
    selected_lambda = sweep_lambda[
        sweep_lambda["track"].eq("highpass")
        & sweep_lambda["selected"].astype(str).str.lower().isin(["true", "1"])
    ].copy()
    display(
        selected_lambda[
            [
                "experiment",
                "lambda_tv",
                "split_half_nrmse",
                "artifact_score",
                "mean_gradient",
                "std",
                "std_excess_vs_saa",
                "selection_proxy",
            ]
        ].round(5)
    )

# %% [markdown]
# > **数据说明**: 这张表来自 `sweep_map_tv_lambda.csv`，只显示 highpass track 中被 MAP-TV selection proxy 选中的 lambda。Selection proxy 同时考虑 split-half NRMSE、artifact score 和相对 SAA 的 std 膨胀。
# >
# > **怎么看**: `lambda_tv` 越大，TV 正则越强，通常会降低 split-half 差异和高频伪影，但也会压低真实边缘。`split_half_nrmse` 越小表示两半数据重建更一致；`selection_proxy` 是当前保守规则下的综合分数，越小越好。
# >
# > **正常/异常**: 三个 alignment experiment 都选择 `lambda=0.01`，说明 MAP-TV 需要强正则来压住不稳定高频；这与“alignment residual 被误解释成结构”的风险一致。NCC init 的 split-half 可以更低，但 artifact score 更高，因此不能只按 split-half 判胜。
# >
# > **核心发现**: MAP-TV 在新 data-driven alignment sweep 下仍然是保守正则候选，不是 EP06 的主推荐方法。它的价值是提供一个抑制过冲的上界/诊断，而不是证明 forward-model SR 已经稳健胜过 SAA/IBP。

# %%
if not sweep_delta.empty:
    highpass_delta = sweep_delta[
        sweep_delta["track"].eq("highpass")
        & sweep_delta["method"].isin(["saa_weighted", "ibp", "map_tv"])
    ].copy()
    display(
        highpass_delta[
            [
                "experiment",
                "method",
                "std_delta_vs_baseline",
                "mean_gradient_delta_vs_baseline",
                "p95_gradient_delta_vs_baseline",
                "artifact_score_delta_vs_baseline",
                "contour_chamfer_lr_px_delta_vs_baseline",
            ]
        ].round(5)
    )

# %% [markdown]
# > **数据说明**: 这张 delta 表把 sweep 结果和旧 `output/ep06_sr_poc` baseline 按同一 `track/method` 对齐后相减。它回答“更换 alignment 或 IBP PSF 后，指标相对旧结论改变了多少”。
# >
# > **怎么看**: delta 为正表示当前 sweep 指标比旧 baseline 更高；对 artifact/std 来说通常不是好事，对 gradient 来说只代表更锐或更强响应。IBP 的 delta 还混入了 `psf_sigma=0.5` 与旧输出参数不同的影响，因此要作为 forward-model sensitivity，而不是单纯 alignment 改动。
# >
# > **正常/异常**: Default SAA 和 MAP-TV 与旧 baseline 基本相同，是因为旧 baseline 已经采用了相同 default contour refined 与保守 MAP-TV 设置；IBP 的梯度/std 下降但 artifact 仍上升，说明小 PSF 让它更保守，但没有消除伪影风险。
# >
# > **核心发现**: 新 sweep 支持把 MAP-TV 从“最佳/最锐”降级为 regularized diagnostic；IBP 也不是明确胜出。EP06 主线应优先推荐 default contour refined + SAA/IBP 作为对照组合，MAP-TV 作为保守正则候选展示。

# %%
for figure_name in [
    "sweep_metric_bars.png",
    "sweep_map_tv_lambda_selection.png",
    "sweep_delta_vs_baseline.png",
]:
    figure_path = SWEEP_SUMMARY_DIR / figure_name
    print(f"\nFigure: {relative(figure_path)}")
    display(show_png_path(figure_path))

# %% [markdown]
# > **图表说明**: 这组三张图由 `scripts/summarize_ep06_alignment_sweep.py` 从 summary CSV/JSON 自动生成。`sweep_metric_bars.png` 比较三种 alignment experiment 的 std、P95 gradient 和 artifact；`sweep_map_tv_lambda_selection.png` 展示 MAP-TV lambda sweep 曲线；`sweep_delta_vs_baseline.png` 展示相对旧 baseline 的变化。
# >
# > **怎么看**: 柱状图中 artifact/std 的上升要优先视为风险，P95 gradient 的上升只能作为锐度响应；lambda 曲线中被圈出的点是当前 selection proxy 选中的值。Delta 图以 0 为旧 baseline，偏离越大表示当前 sweep 改变越明显。
# >
# > **正常/异常**: 如果图中某些柱子接近 0，并不表示缺数据，而是表示当前 sweep 与 baseline 基本一致。MAP-TV 曲线单调偏向更大 lambda 是保守调参的表现，说明弱正则会留下更多 split-half 或 artifact 风险。
# >
# > **核心发现**: 图表和数值表一致：tuned refined 没有推翻 default refined，NCC init artifact 更高，MAP-TV 需要 `lambda=0.01` 才稳定。因此 EP06 不应再把 MAP-TV 写成最优锐化算法，而应写成强正则的风险控制候选。
