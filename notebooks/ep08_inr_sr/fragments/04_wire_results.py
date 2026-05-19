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
    display(wire_history.tail(10).round(8))

# %% [markdown]
# > **数据说明**: 状态表检查 WIRE Stage 1 的正式产物，包括训练历史和收敛图；指标表只读取已有 `metrics.csv`，不会在 notebook 内补算或伪造缺失数值。
# >
# > **怎么看**: WIRE 若优于 SIREN，应表现为轮廓更稳定、artifact 不升高、split-half 不恶化，而不是仅仅 highpass 边缘更亮。训练历史表用于确认 loss 是否健康下降。
# >
# > **正常/异常**: Gabor 激活可能带来边缘敏感性，也可能引入方向性纹理。若梯度指标升高但 hold-out、split-half 或 artifact 同时恶化，应标为高风险。
# >
# > **核心发现**: WIRE 的判断必须建立在与 SIREN 完全共享的数据 split 和门控链路上，才能把差异归因于激活函数。

# %%
display(show_png_if_exists(wire_dir / "training_curve.png"))

# %% [markdown]
# > **图表说明**: 收敛曲线展示 WIRE 训练过程中的 batch train loss，以及验证间隔上的 hold-out / train-set loss。
# >
# > **数据分布/模式**: 曲线应整体下降；若 hold-out 长期上升或 early stopping 过早触发，说明 Gabor layer 可能过拟合局部高频噪声。
# >
# > **核心发现**: 收敛曲线是 WIRE 是否可训练的直接中间证据，后续仍需结合五项指标和图像检查。

# %%
display(show_png_if_exists(wire_dir / "hr_highpass.png"))
display(show_png_if_exists(wire_dir / "hr_raw_control.png"))
display(show_png_if_exists(wire_dir / "split_half_difference.png"))

# %% [markdown]
# > **图表说明**: 第一张图是 WIRE 的 HR highpass 结构响应；第二张图是 raw-control bicubic 参照；第三张图是 split-half 两次重建的差异。
# >
# > **怎么看**: Highpass 用来看芯片边缘和内部轮廓是否更连贯；raw-control 用来看增强是否与普通温度/强度视图中的结构一致；split-half 差异用于检查结构是否由不同训练帧子集稳定恢复。
# >
# > **正常/异常**: 如果出现规则条纹、棋盘纹或 pin 区域伪线，即使边缘看起来更锐，也不能解释为可靠 SR 增益。
# >
# > **核心发现**: WIRE 只有在视觉结构、收敛轨迹和稳定性指标同时成立时，才可作为 EP08 的候选改进方法。
