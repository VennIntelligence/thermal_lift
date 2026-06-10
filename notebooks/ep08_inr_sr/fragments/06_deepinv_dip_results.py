# %% [markdown]
# ## 5. DeepInverse-DIP Stage 2 Results
#
# DeepInverse-DIP 保留 `deepinv==0.4.0` 的 `ConvDecoder` backbone，但使用 EP08 自定义训练循环和 hold-out early stopping：固定 latent `z`，每步经同一个 shift+blur+sample forward operator 回投到训练帧。

# %%
deepinv_dir = method_output_dir("deepinv_dip")
deepinv_status = method_status("deepinv_dip")
display(deepinv_status)

deepinv_metrics = read_csv_if_exists(deepinv_dir / "metrics.csv")
deep_decoder_metrics_for_dip = read_csv_if_exists(method_output_dir("deep_decoder") / "metrics.csv")
if not deepinv_metrics.empty:
    display(deepinv_metrics.round(6))

deepinv_history = read_csv_if_exists(deepinv_dir / "training_history.csv")
if not deepinv_history.empty:
    deepinv_validation_history = (
        deepinv_history.dropna(subset=["holdout_loss"])
        if "holdout_loss" in deepinv_history.columns
        else deepinv_history.iloc[0:0]
    )
    if deepinv_validation_history.empty:
        deepinv_validation_history = deepinv_history.tail(5)
    display(deepinv_validation_history.round(8))

benchmark_metrics = read_csv_if_exists(OUTPUT_DIR / "tcforge_benchmark" / "metrics.csv")
if not benchmark_metrics.empty:
    display(benchmark_metrics.round(6))

if not deepinv_metrics.empty and not deep_decoder_metrics_for_dip.empty:
    dip_compare = pd.concat(
        [
            deep_decoder_metrics_for_dip.assign(display_method="Deep Decoder"),
            deepinv_metrics.assign(display_method="DeepInverse-DIP"),
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
    display(dip_compare.round(6))

# %% [markdown]
# 本评估汇总了基于 DeepInverse 框架内 `ConvDecoder` 先验（Stage 2）的真实数据重建结果，并与 Deep Decoder 在同一数据划分口径下进行了并排对比。
# 评估体系在合成数据集（带高分辨率 Ground Truth 的 TCForge benchmark）与真实芯片观测序列上进行了多层次验证。在真实场景中，较低的泛化残差常指示算法具有更强的前向重投一致性，但若同时伴随着子集一致性误差（Split-Half NRMSE）及伪影评分的上升，则表征卷积网络已经陷入对探测器不均匀噪声与高频微小漂移的过拟合中。对 DeepInverse-DIP 算法的评估需深入审查其泛化性能与伪影控制的物理平衡。

# %%
display(show_png_if_exists(OUTPUT_DIR / "tcforge_benchmark" / "tcforge_benchmark_highpass.png"))

# %% [markdown]
# 仿真基准测试图像（`tcforge_benchmark_highpass.png`）并列展现了高分辨率基准高通真值与两种 CNN 解码器先验的合成重建结构。
# 该对比用于验证模型在大地坐标与前向算子约定下的几何复原精度。虽然在受控仿真场景中可以获得较高的 PSNR 与 SSIM，但合成系统无法模拟真实长波红外（LWIR）成像中的瞬态热演化及非刚性对齐扰动。因此，仿真基准的收敛性仅代表算法架构逻辑的通路完备，真实超分辨率效果的验证依然依赖于后续真实数据物理指标的交叉比对。

# %%
display(show_png_if_exists(deepinv_dir / "training_curve.png"))
display(show_png_if_exists(deepinv_dir / "hr_highpass.png"))
display(show_png_if_exists(deepinv_dir / "hr_raw_control.png"))
display(show_png_if_exists(deepinv_dir / "split_half_difference.png"))

# %% [markdown]
# 此处展示了 DeepInverse-DIP 算法在真实芯片序列上的参数优化轨迹曲线、重建的高分辨率高通结构响应、原始温度控制通道（Raw Control）以及子集分割一致性差异分布。
# 对于强参数容量的 CNN 隐式先验，其具有极低的重投泛化残差，但这极易夹带高频空间纹理幻觉。若子集分割一致性差异图展现出显著且有规律的方向性高频条纹，说明重建出的几何边缘极不稳定，主要由训练子集中的局部噪声及相位不均匀性诱导产生。物理边缘的真实性判定应当以原始温度控制通道的几何契合为依据，从而限制无物理支撑的细节虚构。
