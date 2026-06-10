# %% [markdown]
# ## 3. WIRE Stage 1 Results
#
# WIRE 是 EP08 的主线 INR 候选。本轮实现使用 carrier/envelope 两个独立线性投影的 real-valued Gabor layer，并与 SIREN 共享同一 32 帧数据、split、forward 和指标。

# %%
wire_dir = method_output_dir("wire")
wire_status = method_status("wire")
display(wire_status)

wire_metrics = read_csv_if_exists(wire_dir / "metrics.csv")
if not wire_metrics.empty:
    display(wire_metrics.round(6))

wire_history = read_csv_if_exists(wire_dir / "training_history.csv")
if not wire_history.empty:
    wire_validation_history = (
        wire_history.dropna(subset=["holdout_loss"])
        if "holdout_loss" in wire_history.columns
        else wire_history.iloc[0:0]
    )
    if wire_validation_history.empty:
        wire_validation_history = wire_history.tail(5)
    display(wire_validation_history.round(8))

# %% [markdown]
# 此状态表与评估指标汇总了以 Gabor 函数作为激活的 WIRE（Stage 1）隐式神经网络重建产物。
# 在物理比较中，WIRE 方法在边缘敏感性上的改善不仅应当表征在梯度指标（`p95_gradient`）的提升，更需在泛化残差（`holdout_loss`）和子集 NRMSE 上保持与 SIREN 基准一致或更优的水平，从而证明高频边缘的增强并非由振铃伪影所主导。定量分析直接调取自落盘的 `metrics.csv` 评估文件，以确立数据的统计学可信度。

# %%
display(show_png_if_exists(wire_dir / "training_curve.png"))

# %% [markdown]
# 此收敛曲线图展示了 WIRE 方法训练过程中批次损失及验证损失的动态演化趋势。
# Gabor 小波由于其局部时频局域性（Time-Frequency Localization）优势，理论上能够更稳健地捕获空间局部特征。若 Hold-out 验证损失在迭代过程中出现发散，则提示网络由于网络容量过大，将前向对齐误差或背景热辐射噪声结构化为虚假的细节。该收敛轨迹用于监控参数寻优过程中的数值稳定性，不单独作为超分辨率有效的物理声明。

# %%
display(show_png_if_exists(wire_dir / "hr_highpass.png"))
display(show_png_if_exists(wire_dir / "hr_raw_control.png"))
display(show_png_if_exists(wire_dir / "split_half_difference.png"))

# %% [markdown]
# 该组诊断图呈递了 WIRE 重建后的高分辨率高通结构响应、同一空间位置的原始温度控制图（Raw Control）以及子集交叉验证的差异空间分布。
# 图像几何边缘的完整性与内部轮廓连贯性用于定性评估轮廓增强效果。任何在原始温度通道中缺乏物理对应的条纹或周期性网格伪影，均应定义为伪高频噪声。通过计算分割子集间的差异分布，能够定量评估重建细节的确定性，若两半子集的重建结果出现严重偏离，则证明该方法未能抵抗热像背景噪声的干扰。
