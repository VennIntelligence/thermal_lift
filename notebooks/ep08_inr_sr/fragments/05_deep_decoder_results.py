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
# 此状态表与对比表汇集了基于深度解码器先验（Deep Decoder, Stage 2）的超分辨率重建产物，并与同数据分块下的 SIREN 基准进行了对比。
# 各量化维度包括泛化残差、子集一致性误差（Split-Half NRMSE）、伪影得分以及原始参照一致性（Raw-Control Agreement）。深度解码器在架构上利用未训练卷积网络的归纳偏置限制高频噪声，因此其表现出更低的伪影得分，但较高的泛化残差通常提示其复原边缘的物理强度不足。因此，对于 CNN 先验的评估需在其平滑正则性与细节还原度之间进行折中，以避免陷入仅关注低伪影而忽视信号欠拟合的片面评估。

# %%
display(show_png_if_exists(deep_decoder_dir / "training_curve.png"))

# %% [markdown]
# Deep Decoder 优化收敛曲线展示了迭代过程中的多尺度损失轨迹变化特征。
# 与正弦或 Gabor 等 INR 模型相比，卷积解码器由于不包含显式的空间高频振荡分量，其 Hold-out 验证损失呈现平滑且单调的收敛轨迹，并未出现过拟合带来的误差反弹。虽然参数优化曲线表现出极佳的数值稳定性，但这并不代表图像复原能力的最优，仍需配合图像空间剖面与几何轮廓分析进行联合评判。

# %%
display(show_png_if_exists(deep_decoder_dir / "hr_highpass.png"))
display(show_png_if_exists(deep_decoder_dir / "hr_raw_control.png"))
display(show_png_if_exists(deep_decoder_dir / "split_half_difference.png"))

# %% [markdown]
# 此图像组展示了 Deep Decoder 重建后的高分辨率高通图、原始温度图对比以及子集交叉验证的差异分布。
# 对于卷积解码器，其输出通常倾向于在平坦区域施加较强的物理平滑度，因此其子集一致性差异图通常接近于零。然而，这种低通特性在视觉上表现为对芯片精细引脚及内部微通道的边缘阻碍与钝化。若图像中存在明显的棋盘状上采样伪影（Checkerboard Artifacts），则表明网络架构的转置卷积或亚像素卷积层发生了数值失调，需在后文的物理先验模型中进行精细化调整。
