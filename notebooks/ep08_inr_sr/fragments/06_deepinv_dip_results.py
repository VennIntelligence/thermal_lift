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
# > **数据说明**: 第一张表检查 DeepInverse-DIP 正式产物；第二张表读取 TCForge benchmark 的有 GT highpass PSNR/SSIM；第三张表把 DeepInverse-DIP 与本项目 Deep Decoder 放在同一真实数据指标下比较。
# >
# > **怎么看**: TCForge PSNR/SSIM 只验证实现链路在有 HR ground truth 的合成场景上能收敛，不代表真实数据 SR 成功。真实数据上，hold-out residual 越低越好，但 split-half NRMSE 和 artifact score 若同时变差，说明过拟合或 hallucination 风险更高。
# >
# > **正常/异常**: DeepInverse-DIP 的主训练监控 hold-out loss；split-half 使用相同 fixed latent/初始化、较短训练预算，目的是隔离数据分半稳定性，而不是把随机初始化差异混进指标。
# >
# > **核心发现**: TCForge benchmark 中四个方法都通过 HR highpass GT sanity；真实数据上 DeepInverse-DIP 的 hold-out residual 很低，但 artifact 风险仍最高。

# %%
display(show_png_if_exists(OUTPUT_DIR / "tcforge_benchmark" / "tcforge_benchmark_highpass.png"))

# %% [markdown]
# > **图表说明**: TCForge benchmark 图展示 HR highpass GT 以及四个方法的合成场景重建。
# >
# > **怎么看**: 该图只检查两个 CNN decoder 是否能在有真值的小场景上收敛到合理结构。PSNR/SSIM 越高越好，但它不包含真实 LWIR session 的热漂移、alignment uncertainty 和无 GT 风险。
# >
# > **核心发现**: 合成 benchmark 支持继续使用这些实现做真实数据对照，但最终方法选择仍应以真实数据五项指标为准。

# %%
display(show_png_if_exists(deepinv_dir / "training_curve.png"))
display(show_png_if_exists(deepinv_dir / "hr_highpass.png"))
display(show_png_if_exists(deepinv_dir / "hr_raw_control.png"))
display(show_png_if_exists(deepinv_dir / "split_half_difference.png"))

# %% [markdown]
# > **图表说明**: DeepInverse-DIP 的训练摘要、HR highpass、raw-control 参照和 split-half 差异图。
# >
# > **怎么看**: Highpass 轮廓更强或 hold-out residual 更低并不足以通过；需要看 split-half 差异是否集中在噪声/条纹区域，以及 raw-control 是否支持相同结构位置。
# >
# > **正常/异常**: 若 split-half 差异图出现大面积结构差异或方向性纹理，说明该方法对训练帧子集敏感。raw-control agreement 低时，不能把高频结构直接解释为可靠芯片轮廓。
# >
# > **核心发现**: DeepInverse-DIP 更像强拟合上限参照，不适合作为 Stage 3 默认赢家。
