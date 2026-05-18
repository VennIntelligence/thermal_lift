# %% [markdown]
# ## 3. Small-Step Diagnostics as Smoke Tests
#
# 小步 NCC 只回答局部问题：行内 X 时间相邻帧是否有稳定方向和短时线性；Y 坐标相邻帧为什么不能用于定量标定。它不回答“多帧 contour-level SR 是否可行”。
#
# 本节的关键是“诊断”二字。NCC 在局部高通图像上寻找最相似的位移投影，因此它能提示某些边缘/纹理是否随 stage command 方向移动。
# 但红外热像会随时间漂移，局部热结构也可能变化；当 acquisition gap 变大时，NCC 峰值混入的就不只是几何位移。
# 所以这里不会把小步 NCC 或 ESF 局部响应外推为全局 SR 成功/失败结论。

# %%
fig, small_metrics = plot_small_step_diagnostics(
    OUTPUT_DIR,
    OUTPUT_DIR / "ep02_small_step_smoke_tests.png",
)
fig

# %% [markdown]
# > **图表说明**: 左图比较 X/Y 坐标相邻 pair 的采集时间间隔；中图把 X 时间相邻 high-pass NCC 可见投影和 stage-prior 名义幅值放在同一尺度；右图显示 Y 坐标相邻 pair 的 2 um/4 um 投影比例。
# > **怎么读图**: acquisition gap 越小，两帧越接近“只差一个 stage 小步”；visible projection 越接近 nominal prior，说明局部 NCC 响应与命令方向越一致。右图的 4 um/2 um 比例应接近 2 才符合简单线性位移预期。
# > **正常/异常理解**: X pair 的 gap 中位数为 1，是可用的短时 smoke test；4 um/2 um 可见投影保持接近线性，说明方向和数量级有响应。Y pair 的 gap 中位数约 16，且 4 um 组没有达到 2 um 组的两倍，这是时间污染和 raster 路径共同导致的失败诊断。
# > **数据分布**: X 和 Y 的差异主要不是“轴本身好坏”，而是采集路径造成的 pair 质量差异。固定 X 的 Y 坐标相邻点往往隔了一整行，热场演化会污染 NCC。
# > **核心发现**: X 小步是方向和短时线性的 smoke test；Y-only 坐标相邻 NCC 不能作为 Y 位移标定。这个结果也不能反向证明多帧 contour-level SR 不可行。

# %%
display(small_metrics)

# %% [markdown]
# > **数据说明**: 表中数值来自 EP02 重新计算的 pair 表，`visible projection` 是局部 high-pass NCC 投影，`nominal prior` 是 47.6 deg、10 um/pixel stage prior 的期望量级。
# > **怎么读表**: `visible projection` 是图像数据里 NCC 能“看见”的局部投影，不等于完整真实位移；`nominal prior` 是命令位移按配置换算出的预期值；比例项用于检查 2 um 到 4 um 是否呈现合理单调性。
# > **正常/异常理解**: 小步投影低于 nominal prior 并不自动说明 stage 或 theta 错了，因为 PSF 模糊、噪声、高通窗口和局部纹理都会影响 NCC 可见响应。真正危险的是 Y 的 4/2 比例低于 1，它违反了更大 command 应产生更大投影的基本单调性。
# > **数据分布**: X 2 um 的可见投影小于名义 prior，但 X 的 4/2 比例保留短时线性；Y 的 4/2 比例低于 1，说明这组 pair 不适合做定量标定。
# > **核心发现**: 这些结果用于筛选可用诊断 pair，而不是给出 SR 成败判决。后续重建应把 stage prior 作为初始化/正则项，再用 data-driven alignment 作为质量门控。
