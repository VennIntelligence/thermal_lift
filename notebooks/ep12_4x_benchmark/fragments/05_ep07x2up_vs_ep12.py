# %% [markdown]
# ## EP07 2x x2up vs EP12 4x
#
# 这组廉价基线回答一个更直接的问题：EP12 专用 4x 网络是否比“满意的 EP07 2x 结果再 cubic 放大 2 倍”更有价值。四个方法都使用 EP06 clean main session 248 帧和 `contour_refined` shifts，输出统一落到 1920×2560 4x 网格。

# %%
EP07X2UP_DIR = OUTPUT_DIR / "ep07x2up_vs_ep12"
EP07X2UP_COMMAND = (
    "cd algos/ep12_4x_benchmark && "
    "uv run python scripts/run_ep07x2up_vs_ep12_4x.py --device cuda:1"
)


def show_ep07x2up_fig(name: str) -> None:
    path = EP07X2UP_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing EP07x2up-vs-EP12 figure: {path}\nRun: {EP07X2UP_COMMAND}")
    display(NotebookImage(filename=str(path), retina=True))


def read_ep07x2up_csv(name: str) -> pd.DataFrame:
    path = EP07X2UP_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing EP07x2up-vs-EP12 CSV: {path}\nRun: {EP07X2UP_COMMAND}")
    return pd.read_csv(path)


def read_ep07x2up_text(name: str) -> str:
    path = EP07X2UP_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing EP07x2up-vs-EP12 text artifact: {path}\nRun: {EP07X2UP_COMMAND}")
    return path.read_text(encoding="utf-8")


print(f"EP07x2up-vs-EP12 output: {relative(EP07X2UP_DIR)}")
print(f"Rebuild command: {EP07X2UP_COMMAND}")

# %% [markdown]
# > **数据说明**: Arm A 是 `algos/ep07_unet_sr/outputs/ep07_v6_physics/model_final.pt` 原生 2x 推理后用 `scipy.ndimage.zoom(2, order=3)` 上采样到 4x；Arm B 是 `algos/ep12_4x_sr/outputs/ep12_hybrid_v1/checkpoint_step_048000.pt` 原生 4x；Arm C 是 bare drizzle 4x；Arm D 是 raw LR mean bicubic 4x。
# >
# > **怎么看**: 如果 EP12 4x 路线有效，至少应在同口径 highpass 或中心 zigzag ROI 中比 EP07×2up 展示更清楚、更稳定的细轮廓，同时不显著增加伪影。
# >
# > **异常是否正常**: EP07×2up 边缘更锐时也会出现红/蓝双边过冲；EP12 更平滑不必然错误，但如果中心细线可辨识度没有提升，就不能证明专用 4x 网络有投入价值。
# >
# > **核心发现**: 该对照是 4x 路线的低成本 gate，不给出计量级分辨率结论，只判断 contour-level 可视化是否有可见增益。

# %%
show_ep07x2up_fig("ep07x2up_vs_ep12_center_zoom3x_temperature.png")

# %% [markdown]
# > **图表说明**: 温度视图显示中心 1/3 crop 并 3x 展示，四个方法共享全幅 1-99 percentile 色阶。
# >
# > **怎么看**: 普通温度图用于检查内部区域是否自然、中心 zigzag 和大块边界是否真的更清楚，而不是只在 highpass 中被边缘增强。
# >
# > **异常是否正常**: EP07×2up 看起来更硬、更锐；这包含过冲风险。EP12 与 bare drizzle 的边界更软，说明它更像平滑/补洞而不是新增可辨细节。
# >
# > **核心发现**: 在温度域，EP12 没有比 EP07×2up 更清楚地分离中心 zigzag 线，整体更接近 bare drizzle 的模糊轮廓。

# %%
show_ep07x2up_fig("ep07x2up_vs_ep12_center_zoom3x_highpass.png")

# %% [markdown]
# > **图表说明**: Highpass 视图使用 `sigma=5` 背景扣除和共享 symmetric 99th-percentile 色阶，突出局部结构响应。
# >
# > **怎么看**: 红/蓝表示相对局部背景的正/负响应，白色接近零。边缘越强通常越容易看轮廓，但 ringing、噪声和假边缘也会变强，所以不能只看响应幅值。
# >
# > **异常是否正常**: EP07×2up 的红蓝双边明显更强，提示过冲；EP12 的边缘响应弱于 EP07×2up，也没有明显比 bare drizzle 更能分开内部细线。
# >
# > **核心发现**: Highpass 同域比较不支持 EP12 48k 相对 EP07×2up 的可见 contour 增益。

# %%
show_ep07x2up_fig("ep07x2up_vs_ep12_zigzag_roi_temperature.png")

# %% [markdown]
# > **图表说明**: 这是芯片中心 zigzag 线区域的更小 ROI 放大版，温度域展示。
# >
# > **怎么看**: 重点看中心细折线是否能从邻近大块结构中分离，线宽是否稳定，以及拐角处是否被抹平。
# >
# > **异常是否正常**: EP07×2up 的边缘更硬，不能直接等同于真实 4x 信息；但 EP12 如果作为专用 4x 网络，应至少提供相当或更好的细线可辨识度。
# >
# > **核心发现**: 中心 ROI 中 EP12 更钝，zigzag 细线没有超过 EP07×2up 的可辨识度。

# %%
show_ep07x2up_fig("ep07x2up_vs_ep12_zigzag_roi_highpass.png")

# %% [markdown]
# > **图表说明**: 同一中心 zigzag ROI 的 highpass 放大图，用于观察线边缘和局部结构响应。
# >
# > **怎么看**: EP07×2up 的响应更强但过冲也更强；EP12 若有真实优势，应呈现更干净的线分离而不是单纯低振幅平滑。
# >
# > **异常是否正常**: Highpass 的红蓝边是局部响应，不代表温度正负变化；P95 gradient 变大也可能来自振铃。
# >
# > **核心发现**: EP12 没有在最关键的中心 zigzag 线 ROI 中显示比 EP07×2up 更好的轮廓清晰度。

# %%
ep07x2up_metrics = read_ep07x2up_csv("metrics_summary.csv")
display(ep07x2up_metrics.round(6))

# %% [markdown]
# > **数据说明**: 指标 CSV 给出每个方法的 highpass artifact score、与 bicubic raw-control highpass 的 Pearson 相关，以及 highpass P95 gradient。
# >
# > **怎么看**: artifact score 通常越低越少高频伪影；raw-control highpass Pearson 越高表示结构位置更接近 raw 控制轨；P95 gradient 只辅助观察边缘强度，不能单独作为成功证据。
# >
# > **异常是否正常**: Bicubic raw mean 与自己相关为 1 是定义结果；bare drizzle 的 artifact score 低不代表最清楚，因为它也可能更平滑。EP07×2up 的 P95 gradient 低于 EP12 但视觉更锐，说明单个梯度指标不可靠。
# >
# > **核心发现**: EP12 的 raw-control highpass Pearson 低于 EP07×2up (`0.223` vs `0.389`)，artifact score 高于 EP07×2up (`0.535` vs `0.472`)；proxy 不支持 EP12 优于 EP07×2up。

# %%
display(Markdown(read_ep07x2up_text("comparison_notes.md")))

# %% [markdown]
# > **结论**: EP12 48k 4x 相对 EP07 2x×2up 在“轮廓清晰度/中心细 zigzag 线可辨识度”上没有可见增益。当前证据更支持暂停继续投入 EP12 4x 训练，除非后续先解决 target 锯齿、loss 结构和真实数据 gate 的设计问题。
