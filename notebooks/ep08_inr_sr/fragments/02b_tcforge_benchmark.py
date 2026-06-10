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
# 此表格汇总了基于 TCForge 仿真环境的四种模型（SIREN, WIRE, Deep Decoder, DeepInverse-DIP）在合成高分辨率 Ground Truth 下的图像复原指标。所包含的 PSNR、结构相似度（SSIM）、NRMSE 及平均绝对误差（MAE）均在高通空间域（Highpass Domain）内计算得出，而非真实热像观测下的实测结果。
# 在基准测试中，较高的 PSNR/SSIM 以及较低的 NRMSE/MAE 指示了算法对于已知退化过程的逆向求解能力。`best_step` 与 `final_step` 字段用于检验网络优化的收敛状态。在此控制分支中，DeepInverse-DIP 重建采用了显式的早停与迭代轮次上限，从而避免了深度网络对背景高频噪声的过度记忆。
# 本仿真基准测试验证了各神经网络的前向传输矩阵及尺寸配置的完备性，提供了深度隐式重建方法在理想退化条件下的前置数学自洽性校验。

# %%
display(show_png_if_exists(tcforge_dir / "tcforge_benchmark_highpass.png"))

# %% [markdown]
# 该高通图像对比图直观展现了仿真高分辨率基准真值（HR Highpass Ground Truth）与各神经网络重建图像的二维特征，图像均采用对称的红蓝色标进行渲染。
# 在图像灰度分布中，白色表征局部零梯度，红蓝指示局部空间温度梯度起伏。边缘结构的高频边缘强度有助于芯片结构的可维性，但也需警惕由神经网络过拟合所夹带的周期性空间振荡。若因训练终止导致某些通道为空，则表明仿真测试未正常跑完所有候选方法。此可视化对比作为快速故障排查工具，用以检测图像错位、网格拉伸及符号反置等实现细节，最终的性能判定仍需依赖真实热像数据的指标评估。
