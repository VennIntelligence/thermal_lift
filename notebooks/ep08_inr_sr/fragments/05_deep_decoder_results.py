# %% [markdown]
# ## 4. Deep Decoder Stage 2 Results
#
# Deep Decoder 是 CNN decoder prior 对照。本阶段使用与 SIREN/WIRE 完全相同的 32 帧、中心 256x256 LR patch、seed=42 train/val split 和 EP08 forward/metric pipeline。

# %%
deep_decoder_dir = method_output_dir("deep_decoder")
deep_decoder_status = method_status("deep_decoder")
display(deep_decoder_status)

deep_decoder_metrics = read_csv_if_exists(deep_decoder_dir / "metrics.csv")
siren_metrics_for_dd = read_csv_if_exists(method_output_dir("siren") / "metrics.csv")
if not deep_decoder_metrics.empty:
    display(deep_decoder_metrics.round(6))

deep_decoder_history = read_csv_if_exists(deep_decoder_dir / "training_history.csv")
if not deep_decoder_history.empty:
    deep_decoder_validation_history = (
        deep_decoder_history.dropna(subset=["holdout_loss"])
        if "holdout_loss" in deep_decoder_history.columns
        else deep_decoder_history.iloc[0:0]
    )
    if deep_decoder_validation_history.empty:
        deep_decoder_validation_history = deep_decoder_history.tail(5)
    display(deep_decoder_validation_history.round(8))

if not deep_decoder_metrics.empty and not siren_metrics_for_dd.empty:
    dd_compare = pd.concat(
        [
            siren_metrics_for_dd.assign(display_method="SIREN"),
            deep_decoder_metrics.assign(display_method="Deep Decoder"),
        ],
        ignore_index=True,
    )[
        [
            "display_method",
            "holdout_residual",
            "split_half_nrmse",
            "artifact_score",
            "raw_control_agreement",
            "p95_gradient",
            "best_step",
            "final_step",
        ]
    ]
    display(dd_compare.round(6))

# %% [markdown]
# > **数据说明**: 状态表检查 Deep Decoder Stage 2 的正式产物；指标表读取 `deep_decoder_stage2/metrics.csv`，并与 SIREN Stage 1 的同 split 结果并排显示。
# >
# > **怎么看**: Hold-out residual、split-half NRMSE 和 artifact score 越小越好；raw-control agreement 越高越好。P95 gradient 较低通常表示边缘响应更弱，但也可能意味着噪声和伪纹理更少。
# >
# > **正常/异常**: Deep Decoder 的 artifact 很低但 hold-out residual 高，通常说明它更保守、更平滑，不能仅凭低 artifact 判为最好。Highpass 图中白色代表接近零局部变化，红/蓝代表相对背景的正/负结构响应。
# >
# > **核心发现**: Deep Decoder 在本轮最稳定、artifact 最低、raw-control agreement 最高，但 hold-out residual 明显高于 SIREN/WIRE，说明它牺牲了 forward consistency 和轮廓强度。

# %%
display(show_png_if_exists(deep_decoder_dir / "training_curve.png"))

# %% [markdown]
# > **图表说明**: 收敛曲线展示 Deep Decoder 的 batch train loss、hold-out loss 和 train-set loss。
# >
# > **数据分布/模式**: 曲线收敛到较平滑的解，最终 best step 接近训练后段，说明没有像 SIREN 那样较早达到最优 hold-out。
# >
# > **核心发现**: 训练稳定不等于重建充分；需要结合 hold-out residual 和视觉图判断是否欠表达。

# %%
display(show_png_if_exists(deep_decoder_dir / "hr_highpass.png"))
display(show_png_if_exists(deep_decoder_dir / "hr_raw_control.png"))
display(show_png_if_exists(deep_decoder_dir / "split_half_difference.png"))

# %% [markdown]
# > **图表说明**: 三张图分别是 Deep Decoder HR highpass、raw-control bicubic 参照和 split-half 差异图。
# >
# > **怎么看**: Highpass 用于看内部轮廓是否连贯；raw-control 用于确认增强位置是否和普通强度/温度结构一致；split-half 差异越小表示不同训练帧子集恢复出的结构越稳定。
# >
# > **正常/异常**: CNN decoder 输出若过平，可能在 split-half 上很好看，但会丢失真实边缘。若出现规则上采样纹理，应按 decoder artifact 风险处理。
# >
# > **核心发现**: Deep Decoder 是一个低 artifact 的保守对照，更适合作为稳定性下界，而不是 Stage 3 的首选增强方法。
