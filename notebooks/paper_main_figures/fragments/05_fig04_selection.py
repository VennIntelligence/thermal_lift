# %% [markdown]
# ## F4 — proxy Pareto 与 checkpoint 选择协议（§6.6，单栏）
#
# 四个 1x 臂全部 checkpoint 在 (artifact, corr) 平面上的散点 + TGV 参考点 +
# 协议选出的 canonical 点。当前稿尚在 EP11 产物目录，V9 系列选点后同脚本重出
# 并迁入 `output/paper_figures/`。

# %%
fig04 = show_figure(
    PROJECT_ROOT / "output" / "ep11_dl_benchmark" / "checkpoint_selection" / "fig_pareto.png",
    "cd algos/ep07_unet_sr && uv run python scripts/plot_checkpoint_selection.py",
)
fig04

# %% [markdown]
# > **图表说明**: TB-scale proxy 平面，每点一个 checkpoint，颜色区分臂；
# > 星标为 TGV 参考点 (0.695, 0.916)；圆圈标注 canonical 选点
# > （v6@8K、v8.1a@15K、v8.1b@5K、v9b@11K）。
# > **怎么看**: 左上方向（低 artifact、高 corr）更好；各臂 60K 端点全部位于
# > 右下——**按端点上报会让每个臂交出最差 checkpoint**（§6.6 金句的数据来源）。
# > **异常是否正常**: 没有任何 UNet checkpoint 接近 TGV 参考点是真实结论
# > （时间轴 Pareto 被经典方法支配），不是绘图问题。
# > **核心发现**: 选点协议是方法的一部分；endpoint reporting 在本问题上会系统性
# > 低报学习臂。
# > **状态**: ✅ 已是学术风格；🔄 V9A/V9C（及 V10）选点落地后重出并迁移。
