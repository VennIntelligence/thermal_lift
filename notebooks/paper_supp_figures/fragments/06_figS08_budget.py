# %% [markdown]
# ## S-F8 — EP16 预算/鲁棒性/对齐源全曲线（supp D.5 配图）
#
# 主文 F7 的完整版：三组矩阵各自的单栏图（drizzle + TGV 经典臂）。

# %%
EP16_DIR = PROJECT_ROOT / "output" / "ep16_budget_robustness"
REBUILD = (
    "cd algos/ep16_budget_robustness && CUDA_VISIBLE_DEVICES=\"\" "
    "uv run python scripts/run_ep16_classical.py --summarize-only"
)
for name in ["fig_frame_budget", "fig_shift_robustness", "fig_alignment_source"]:
    display(Markdown(f"**{name}**"))
    display(show_figure(EP16_DIR / f"{name}.png", REBUILD))

# %% [markdown]
# > **图表说明**: 依次为 E1 帧数预算（指标 vs N，多 seed 聚合）、E2 shift 扰动
# > （指标 vs σ）、E3 对齐源消融（command prior vs contour refined）。
# > 全部数字与 supp D.5 的表格同源（`output/ep16_budget_robustness/*.csv`）。
# > **怎么看**: E1 看单调性与收益拐点（corr 拐点在 N=62）；E2 看 metric-specific
# > 敏感性（corr 稳、FRC/coverage 类敏感）；E3 看端到端对齐价值（corr +0.10~0.11）。
# > **异常是否正常**: TGV 的 split-half/FRC 列复用同子集 drizzle proxy（预算考虑，
# > 头注已声明），不要当作 TGV 自身的 split-half。
# > **核心发现**: 数据驱动对齐细化是经典臂收益最大的单一开关；shift 鲁棒性
# > 结论必须按指标分别表述。
# > **状态**: ✅ 经典臂完成；⬜ MAP-TV/UNet 臂等 GPU 窗口补线。
