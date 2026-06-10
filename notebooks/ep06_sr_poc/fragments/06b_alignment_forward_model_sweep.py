# %% [markdown]
# ## 5.1 Data-Driven Alignment Sweep for IBP/MAP-TV
#
# 这一节把 EP05/EP06 的 data-driven alignment 定版结果真正传回 forward-model 算法里重跑，而不是只在 SAA ablation 里观察位移策略。三组独立输出目录如下：
#
# | Experiment | Alignment CSV | Method passed to scripts | Role |
# |---|---|---|---|
# | `default_contour_refined_psf05` | `configs/alignment/contour_alignment_results.csv` | `data_driven_contour_refined` | EP06 主线候选 |
# | `tuned_contour_refined_psf05` | `output/ep05_alignment_tuning/full_r360_e93_rad100_s0125/contour_alignment_results.csv` | `data_driven_contour_refined` | EP05 Chamfer-tuned sensitivity check |
# | `ncc_init_psf05` | `configs/alignment/contour_alignment_results.csv` | `data_driven_ncc_init` | NCC init / phase-prior control |
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
# 此表格汇总了高通成像通道（Highpass Track）下，基于不同对齐方法（配准场先验与前向参数）对 SAA-weighted、IBP 及 MAP-TV 算法进行重新构建的定量评估指标。实验设计控制了相同的输入会话与重建格网大小（2x 尺度），以评估算法在几何与运动约束变化下的物理响应稳定性。
# 评估的重点在于分析同一算法分支对不同配准输入的鲁棒性。`artifact_score`、`std_ratio_to_lr` 以及基于几何定位的 `contour_chamfer_lr_px` 的合理控制是避免病态伪影的关键。对于 IBP 与 MAP-TV，前向位移残差（Alignment Residual）若未在数学模型中正确规避，极易导致边缘过冲或阶梯化伪影（Staircasing Artifacts）。在强正则约束下，MAP-TV 表现出较低的 `mean_gradient`，表明高频细节受到一定程度的平滑，而非算法崩溃。初始互相关对齐（NCC Init）因缺乏几何边界精细化控制，在梯度响应上升的同时伴随着显著的伪影评分增加，表明其相较于轮廓精细化对齐（Contour Refined）具有更高的伪影过拟合风险。
# 完整的配准参数扫描进一步证实了原 2x POC 结论的稳健性。默认轮廓精细化对齐（Default Contour Refined）依然作为首选配准策略，而由 Chamfer 调优的参数及初始互相关在物理重建模型中未能产生更具主导性的优胜优势。

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
# 本表格展示了 MAP-TV 正则化参数 $\lambda_{\text{TV}}$ 扫描中，被综合选择代理（Selection Proxy）判定为最优正则强度的结果。选择代理的目标函数联合最小化了子集均方根误差（Split-Half NRMSE）、伪影得分以及相对 SAA 算法标准差异常膨胀罚项，从而在平滑约束与高频复原之间实现物理平衡。
# 物理上，随着 $\lambda_{\text{TV}}$ 的递增，总变分正则化（TV Regularization）约束增强，从而压制了由于位移误差引发的噪声与伪影，但也对真实轮廓的强梯度形成了衰减。在三组不同的配准实验中，最优正则强度均指向了上限 $\lambda_{\text{TV}} = 0.01$，这表明在当前探测器信噪比限制下，算法需倾向于高强度约束以抑制不可信的细节。虽然初始互相关对齐（NCC Init）在子集 NRMSE 上表现出数值偏低，但其伴随的高伪影评分表明其稳定性源自全局模糊而非真实的细节复原。
# 因此，在当前的运动几何与对齐模型下，MAP-TV 作为强正则控制分支展示，主要提供物理上限诊断，而非作为首选的超分辨率边缘恢复方法。

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
# 本差异表格（Delta Table）反映了本次配准扫描结果相对原 `output/ep06_sr_poc` 预存基线在各项量化指标上的变化差值。该增量分析能够直接指出更换对齐策略或调整点扩散函数（PSF $\sigma = 0.5$）对算法重建状态的具体扰动。
# 差异值为正值表示扫参重建指标高于基线，这在梯度指标上表示边缘强度的增强，而在伪影得分与标准差比例上则通常指示着数值发散的风险。必须注意，IBP 算法由于修改了相机点扩散函数（PSF）参数，其增量中包含了系统模糊度变化和配准改变的复合影响，不可单独解释为对齐改进。
# 数据分析显示，默认的 SAA 与 MAP-TV 算法在两次运行间表现出高度的指标一致性，而 IBP 算法在降低梯度强度的同时仍未能显著消除一阶伪影。这支持了将默认轮廓精细化配准（Default Contour Refined）搭配 SAA/IBP 作为重建基线，并将 MAP-TV 作为强正则校验分支的决策方案。

# %%
figure_path = SWEEP_SUMMARY_DIR / "sweep_metric_bars.png"
print(f"\nFigure: {relative(figure_path)}")
show_fig(figure_path.name, subdir="sweep")

# %% [markdown]
# Figure 11: Alignment forward-model sweep metric bars. Cached summary from the data-driven alignment sensitivity sweep.

# %%
figure_path = SWEEP_SUMMARY_DIR / "sweep_map_tv_lambda_selection.png"
print(f"\nFigure: {relative(figure_path)}")
show_fig(figure_path.name, subdir="sweep")

# %% [markdown]
# Figure 12: Alignment forward-model MAP-TV lambda selection. Cached summary from the data-driven alignment sensitivity sweep.

# %%
figure_path = SWEEP_SUMMARY_DIR / "sweep_delta_vs_baseline.png"
print(f"\nFigure: {relative(figure_path)}")
show_fig(figure_path.name, subdir="sweep")

# %% [markdown]
# Figure 13: Alignment forward-model delta versus baseline. Cached summary from the data-driven alignment sensitivity sweep.

# %% [markdown]
# 此图表组由配准扫参分析脚本自动导出，包含配准方法指标对比柱状图（`sweep_metric_bars.png`）、MAP-TV 正则化寻优曲线（`sweep_map_tv_lambda_selection.png`）以及相对原基线的变动分布图（`sweep_delta_vs_baseline.png`）。
# 在柱状图分析中，标准差与伪影得分的无控制上升应当作为配准模型发散的红色信号。寻优曲线中收敛极值的分布表明了总变分正则化对于高频数值振铃的敏感性，曲线向强约束方向偏移说明弱正则化无法抑制运动矢量微小不一致带来的物理畸变。
# 上述图形与定量数据一致表明，调优参数（Tuned Refined）相较于默认精细化配准未表现出统计学上的显著优势，而初始互相关配准在高通或前向重建模型中表现出较高的不稳定性。这一结果支持了将默认轮廓精细化配准作为 EP06 核心物理对齐基线，并将强正则化的 MAP-TV 用于评估前向物理模型稳定性的结论边界。
