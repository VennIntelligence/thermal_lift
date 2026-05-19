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
    display(siren_history.tail(10).round(8))

# %% [markdown]
# > **数据说明**: 状态表检查 SIREN Stage 1 的正式产物，包括指标、训练历史、收敛曲线、HR highpass、raw-control 参照、split-half 差异图、split 记录、配置和 checkpoint。
# >
# > **怎么看**: Hold-out residual、split-half NRMSE 和 artifact score 越小越好；raw-control agreement 越高越好；P95 gradient 只说明边缘响应强度，不能单独证明 SR 成功。训练历史表的 `train_loss` 是 batch loss，`holdout_loss` 和 `train_set_loss` 只在验证间隔写入。
# >
# > **正常/异常**: 若训练曲线下降但 hold-out 持续上升，说明可能过拟合；若指标存在但缺少 split/config/history 产物，则结果不可复现。Highpass 图中红/蓝代表相对局部背景的正/负响应，白色通常接近零变化。
# >
# > **核心发现**: SIREN 只有同时具备收敛轨迹、hold-out 指标、split-half 稳定性和 raw-control 对照，才可作为 WIRE 的公平基线。

# %%
display(show_png_if_exists(siren_dir / "training_curve.png"))

# %% [markdown]
# > **图表说明**: 收敛曲线展示 SIREN 训练过程中的 batch train loss，以及验证间隔上的 hold-out / train-set loss。
# >
# > **数据分布/模式**: 曲线应整体下降；短期震荡是随机抽取帧 batch 的正常现象。hold-out 曲线若长时间反向上升，应优先按泛化风险解释。
# >
# > **核心发现**: 收敛曲线是 Stage 1 的中间产物，用来判断训练是否健康，不能替代最终的五项指标。

# %%
display(show_png_if_exists(siren_dir / "hr_highpass.png"))
display(show_png_if_exists(siren_dir / "hr_raw_control.png"))
display(show_png_if_exists(siren_dir / "split_half_difference.png"))

# %% [markdown]
# > **图表说明**: 第一张图是 SIREN 的 HR highpass 结构响应；第二张图是同一 patch 的 raw-control bicubic 参照；第三张图是 split-half 两次重建的差异。
# >
# > **怎么看**: Highpass 用来看边缘和内部轮廓是否连贯；raw-control 用来看结构位置是否与普通强度/温度参照一致；split-half 差异越接近零，说明强结构越稳定。
# >
# > **正常/异常**: 只在 highpass 中出现、但 raw-control 无对应位置的规则纹理应按 artifact 风险处理。split-half 差异图中的大面积方向性条纹通常不是可靠轮廓证据。
# >
# > **核心发现**: SIREN 的 Stage 1 判断必须把视觉结构、训练轨迹和稳定性指标合并看，不能只看单张锐化图。
