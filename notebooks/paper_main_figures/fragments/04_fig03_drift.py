# %% [markdown]
# ## F3 — 零空间漂移轨迹 + forward-loss inset（§6.2，全文核心机制图）
#
# 1x 输入各变体的 artifact / raw-control corr 随训练步数的轨迹，inset 显示
# `loss/forward_model` 贴底曲线——「地板 + 漂移 = 漂移在零空间」的实测证据。

# %%
fig03 = show_figure(
    PAPER_FIGS / "fig03_nullspace_drift.png",
    "cd algos/ep07_unet_sr && uv run python scripts/plot_drift_trajectories_paper.py --refresh",
)
fig03

# %% [markdown]
# > **图表说明**: 左 panel artifact score（↓ 越小越好）、右 panel raw-control corr
# > （↑ 越大越好）vs 训练步数，TB-scale 口径；○ 为协议选出的 canonical checkpoint，
# > × 为 60K 端点；inset 为带锚变体的 forward loss（log y）。
# > **怎么看**: 所有 1x 变体的两条 proxy 随训练单调恶化（漂移），而 v9b/v9d 的
# > forward loss 自 10K 起平坦在 0.004–0.009——观测域损失对漂移方向**不可见**
# > （supp A.2 Proposition 1 的实测实例）。
# > **异常是否正常**: v6（hot loss）轨迹更陡是预期中的漂移放大参照；各变体曲线
# > 几乎重合正是论点本身（loss 侧旋钮无效）。
# > **核心发现**: 漂移由合成先验驱动、活在观测算子零空间；补救只能来自输入端
# > 证据注入与选点协议，不能来自更强的观测锚定。
# > **状态**: ✅ 当前稿（V9D 数据已闭合）；🔄 V9C 60K 落地后 `--refresh` 出终稿。
# > hybrid 输入变体（V9A）不进本图——proxy 跨输入模式不可横比，见 supp S-F3。
