# %% [markdown]
# ## S-F3 — V9A（hybrid 输入）漂移轨迹 companion 图
#
# 主文 F3 只放 1x 输入臂；hybrid 输入的 V9A 因 proxy 跨输入模式不可横比
# （supp A.3.4 推论 2）单独成图入 supp。

# %%
figS03 = show_figure(
    PAPER_FIGS / "fig03s_v9a_trajectory.png",
    "cd algos/ep07_unet_sr && uv run python scripts/plot_drift_trajectories_paper.py --refresh",
)
figS03

# %% [markdown]
# > **图表说明**: V9A 的 artifact / corr vs 训练步数（TB-scale），与主文 F3 同
# > 排版；纵轴数值只能在 V9A 自身轨迹内比较，不能与 1x 臂横比。
# > **怎么看**: V9A 是唯一在 30K 后漂移压平的臂（30K→60K artifact −0.014 /
# > corr +0.007），但平台位置仍低于其早期 checkpoint——证据注入改变漂移形态，
# > 不消除先验侵蚀。
# > **核心发现**: 输入端证据注入 + 选点（取 20K 一带）才是可交付组合；
# > fine-window 的保真悬崖证据见 S-F10。
# > **状态**: ✅ 当前稿；🔄 V9C 落地后与 F3 同步 `--refresh`。
