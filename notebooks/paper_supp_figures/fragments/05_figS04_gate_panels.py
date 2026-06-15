# %% [markdown]
# ## S-F4 — checkpoint 选择的视觉 gate panel（supp D.6 / C.5 配图）
#
# 选点协议第 6 步的实物证据：每个变体 canonical 候选 + 60K 端点的温度域三联图，
# 标注 proxy 数值。诊断面板按训练管线原样存档（非 CVPR 排版），证据价值在
# 内容而非版式。

# %%
EP11_DIR = PROJECT_ROOT / "output" / "ep11_dl_benchmark" / "checkpoint_selection"
REBUILD = "cd algos/ep07_unet_sr && uv run python scripts/plot_checkpoint_selection.py"
for arm in ["v6", "v8.1a", "v8.1b", "v9b"]:
    display(Markdown(f"**panel_{arm}**"))
    display(show_figure(EP11_DIR / f"panel_{arm}.png", REBUILD))

# %% [markdown]
# > **图表说明**: 每行一个变体，三联为「候选 checkpoint（≥5K 间隔的理想点距离
# > top-3）+ 60K 端点对照」的 center-zoom 温度图（inferno，1–99 百分位）。
# > **怎么看**: 与 60K 端点相比，canonical 选点（v6@8K、v8.1a@15K、v8.1b@5K、
# > v9b@11K）的中心结构更接近观测、风格化更弱；v8.1b 行可见 PixelShuffle 的
# > 条纹伪影（负结果档案 supp D.4.1 的视觉证据）。
# > **核心发现**: 机械选点 + 温度域视觉 gate 双重把关是「不默认交付 60K」的
# > 执行层；面板同时构成 60K 端点风格化漂移的可视化证据。
# > **状态**: ✅ 选编即可；⬜ V9A/V9C/V10 panel 等各自选点后由同脚本追加。
