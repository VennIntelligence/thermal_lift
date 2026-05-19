# %% [markdown]
# ## 2b. TCForge HR-GT Benchmark
#
# TCForge benchmark 是一个带 HR ground truth 的合成 sanity/benchmark 层：默认 LR=256×256、HR=512×512、32 帧，比较 SIREN、WIRE、Deep Decoder、DeepInverse-DIP 四种方法。评价只在 highpass 域进行，目标图定义为 `highpass_preprocess(hr_temperature_2x[np.newaxis], sigma_bg=5.0, mode='nearest')[0]`。

# %%
tcforge_dir = OUTPUT_DIR / "tcforge_benchmark"
tcforge_metrics = read_csv_if_exists(tcforge_dir / "metrics.csv")
if not tcforge_metrics.empty:
    display(
        tcforge_metrics[
            ["method", "domain", "psnr_db", "global_ssim", "nrmse", "mae_c", "best_step", "final_step"]
        ].round(6)
    )
else:
    display(
        Markdown(
            "`output/ep08_inr_sr/tcforge_benchmark/metrics.csv` 不存在。可在算法环境中运行 "
            "`python scripts/run_tcforge_benchmark.py --iterations 800`；快速 smoke 可降低 `--lr-shape` 和 `--iterations`。"
        )
    )

# %% [markdown]
# > **数据说明**: 表格读取 TCForge 四方法 benchmark 的 highpass-domain 指标；PSNR/SSIM/NRMSE/MAE 都是相对合成 HR highpass ground truth 的数值，不是实测芯片数据指标。
# >
# > **怎么看**: PSNR 和 global SSIM 越高越好，NRMSE 和 MAE 越低越好；`best_step/final_step` 用来确认每个方法是否真的完成了 bounded 训练循环。DeepInverse-DIP 这里使用自定义早停/有界循环，不使用 deepinv 内置 10000-step DIP 调用。
# >
# > **正常/异常**: 该 benchmark 的 highpass target 来自 HR 温度场先做 highpass，而 LR 输入则由 forward model 生成后再做 highpass；它检查实现、尺寸和 forward wrapper 是否大体自洽，不替代真实数据的 alignment gate。
# >
# > **核心发现**: 当四方法表格和图像都存在时，EP08 可以用同一 TCForge 场景检查 SIREN/WIRE/Deep Decoder/DeepInverse-DIP 的 HR 输出形状、训练闭环和 highpass 结构恢复能力。

# %%
display(show_png_if_exists(tcforge_dir / "tcforge_benchmark_highpass.png"))

# %% [markdown]
# > **图表说明**: 图像从左到右展示 HR highpass ground truth 与各方法重建结果，使用相同的红蓝对称色标。
# >
# > **怎么看**: 白色接近局部背景零响应，红/蓝表示相对局部背景的正/负 highpass 结构；边缘更清楚通常有利于 contour-level 观察，但过强纹理也可能是伪高频。
# >
# > **正常/异常**: 如果某个方法缺图，通常表示 benchmark 只运行了部分 `--methods`；如果 Deep Decoder 尺寸错误，脚本会直接报错，因为 benchmark 要求 native output 匹配 HR shape。
# >
# > **核心发现**: 这张图用于快速发现实现级问题，例如输出错位、尺寸不匹配、DIP 训练失控或 highpass 符号反转；真实芯片结论仍以后续 EP08 patch 指标为准。
