# %% [markdown]
# ## F7 — 帧数预算与鲁棒性（§6.4–6.5；候选降级 supp）
#
# EP16 经典臂（drizzle/TGV）的三组推理期矩阵：N∈{31,62,124,248} 帧预算、
# shift 扰动 σ∈{0,0.05,0.1,0.2} px、对齐源消融（command vs contour-refined）。

# %%
fig07 = show_figure(
    PAPER_FIGS / "fig07_budget_robustness.png",
    "cd algos/ep16_budget_robustness && CUDA_VISIBLE_DEVICES=\"\" "
    "uv run python scripts/run_ep16_classical.py --summarize-only",
)
fig07

# %% [markdown]
# > **图表说明**: 左 panel 指标 vs 帧数 N（多 seed 子集聚合），右 panel 指标 vs
# > shift 扰动 σ；数据 `output/ep16_budget_robustness/{frame_budget,shift_robustness}.csv`
# > （17+20 行 all success）。
# > **怎么看**: drizzle 的 corr 增益大半在 N=62 前到位（0.747→0.772），FRC/split-half
# > 持续受益于相位覆盖；corr 对 ≤0.2 px 扰动几乎不动而 coverage/FRC 类指标敏感——
# > 鲁棒性结论是 **metric-specific** 的。
# > **异常是否正常**: TGV 的 corr 非单调（0.728→0.754→0.735→0.741）为真实测量，
# > 正则化臂对子集组成更敏感；TGV 的 split-half/FRC 列复用同子集 drizzle proxy
# > （预算考虑），不影响 artifact/corr 列。
# > **核心发现**: E3 对齐源消融是端到端价值证据——refined 对齐让 drizzle corr
# > 0.662→0.771、TGV 0.642→0.741（数据驱动对齐的最终回报）。
# > **状态**: ✅ 经典臂完成；⬜ MAP-TV/UNet 臂等 GPU 窗口；主文保留结论句、
# > 全曲线建议放 supp（S-F8）。
