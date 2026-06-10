# %% [markdown]
# ## 4. LR 观测、Forward 与 Highpass
#
# HR 真值经 shift profile、forward model 和 highpass 后，才成为 SR 算法的实际输入。highpass 输出是结构响应图，红/蓝不是绝对温度。

# %%
if cache.demo_skipped:
    display(Markdown("LR 观测图不可用。"))
else:
    display(compact_table(cache.forward_stats, ["array", "role", "shape", "dtype", "min", "max", "finite"]))
    print(f"Shift source: {cache.shift_source}")
    print(f"Highpass source: {cache.highpass_source}")
    print(f"Block-average preview: {cache.block_forward_mode}")
    print(f"Interior crop margin: {cache.edge_vis_margin_lr_px} LR px")
    show_fig("demo_forward_highpass.png")

# %% [markdown]
# > **图表说明**: Figure 5 展示 interior-crop 后的 LR raw、highpass、mean |highpass|、亚像素相位覆盖，以及 point vs block-average 差分。
# >
# > **怎么看**: 相位散点图横轴 dx、纵轴 dy（LR px）；白色 highpass 区表示局部变化接近零。边缘裁剪只为可视化，不改变落盘数组。
# >
# > **核心发现**: `exact_ep06_point` 与 `physical_block_average` 在芯片边缘处存在可分辨差分，后续指标必须分 mode 报告。

# %% [markdown]
# ### 4.1 磁盘数据包总览

# %%
if not cache.demo_skipped:
    show_fig("demo_dataset_overview.png")
    display(compact_table(cache.demo_metrics, ["metric", "value", "interpretation"]))
    print(f"Evaluate source: {cache.eval_source}")

# %% [markdown]
# > **图表说明**: 从磁盘重读 demo 数据包，汇总 HR 真值、LR 均值、highpass std 和相位覆盖。
# >
# > **怎么看**: `lr_highpass_abs_p95_c` 量化结构响应尺度；`shift_norm_max_px` 反映位移覆盖范围。这些是数据健康指标，不是 SR 质量指标。
# >
# > **核心发现**: manifest 指向的文件足以让下游脚本在不依赖 Notebook 内存的情况下复现全部可视化。

# %% [markdown]
# ### 4.2 生成 vs 观测剖面

# %%
if not cache.demo_skipped:
    show_fig("demo_profiles_generation_vs_observation.png")

# %% [markdown]
# > **图表说明**: 左图 HR mask 与温度剖面；右图 LR raw 与 highpass 有符号剖面。
# >
# > **怎么看**: highpass 曲线以零为中心振荡，突出局部边缘；不能当作绝对温度读数。
# >
# > **核心发现**: PSF 使 LR raw 边缘变缓，highpass 则把低频背景扣除后突出结构边界。
