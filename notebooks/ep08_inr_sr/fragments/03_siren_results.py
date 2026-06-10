# %% [markdown]
# ## 2. SIREN Stage 1 Results
#
# SIREN 是正弦 INR baseline。Stage 1 固定 32 帧、中心 256x256 LR patch 和 seed=42 的 train/val split，用于和 WIRE 做 activation ablation。

# %%
siren_dir = method_output_dir("siren")
siren_status = method_status("siren")
display(siren_status)

siren_metrics = read_csv_if_exists(siren_dir / "metrics.csv")
if not siren_metrics.empty:
    display(siren_metrics.round(6))

siren_history = read_csv_if_exists(siren_dir / "training_history.csv")
if not siren_history.empty:
    siren_validation_history = (
        siren_history.dropna(subset=["holdout_loss"])
        if "holdout_loss" in siren_history.columns
        else siren_history.iloc[0:0]
    )
    if siren_validation_history.empty:
        siren_validation_history = siren_history.tail(5)
    display(siren_validation_history.round(8))

# %% [markdown]
# 此状态表与评估指标汇总了 SIREN（Stage 1）正弦激活隐式神经网络超分辨率重建的各项产物与特征。
# 状态校验覆盖了训练指标、收敛历史曲线、高分辨率结构响应、原始控制图像以及子集交叉验证。其中 `train_loss` 为每批次的训练误差，`holdout_loss` 则指示未参与训练相位的泛化残差。在量化评估中，较低的泛化残差与子集 NRMSE 代表重建结果具备较好的泛化稳定性；平均梯度指标则仅指示图像高频细节的丰盈度，需结合伪影评分共同对图像质量进行约束评估。

# %%
display(show_png_if_exists(siren_dir / "training_curve.png"))

# %% [markdown]
# SIREN 收敛曲线图（`training_curve.png`）并列展示了每步迭代的 Batch 训练损失、间隔评估的 Train-Set 损失以及未参与训练帧的 Hold-out 验证损失。
# 在寻优过程中，损失函数的下降轨迹表征了神经网络对多帧亚像素图像退化模型的逼近状况。当 Hold-out 损失曲线在优化后半程发生持续性抬升，则指示网络对于特定的亚像素位移及噪声产生了过拟合。收敛曲线作为网络优化动力学的诊断工具，其在多帧批量更新中的稳步收敛是确立参数健康度的必要前置条件。

# %%
display(show_png_if_exists(siren_dir / "hr_highpass.png"))
display(show_png_if_exists(siren_dir / "hr_raw_control.png"))
display(show_png_if_exists(siren_dir / "split_half_difference.png"))

# %% [markdown]
# 此三幅诊断图像展示了 SIREN 重建后的高分辨率高通结构图、对应的原始对比通道（Raw Control）以及子集分割一致性差异图（Split-Half Difference）。
# 图像评估中，高通响应图用以审查局部高频几何边缘的连续性，并与原始温度参照图进行空间位置对齐，以判断轮廓复原是否产生物理错位。子集一致性图定量揭示了由于训练帧随机组合产生的非结构性偏离。若在高通图像中存在显著条纹，但在原始控制通道中无对应结构，或子集一致性差异图表现出空间相关性，则应归因于神经网络的虚假幻觉，其不可作为超分辨率分辨率提升的物理证据。
