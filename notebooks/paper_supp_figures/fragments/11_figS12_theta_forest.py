# %% [markdown]
# ## S-F12 — AVI 独立方向验证的 forest plot（supp A.5.2 配图）
#
# 16 个连续扫描 AVI 的 θ 估计（highpass NCC 与 gradient NCC 两种特征域），
# 含 X/Y 子组合并与全体合并的 95% CI。

# %%
figS12 = show_figure(
    PAPER_FIGS / "figS12_theta_forest.png",
    "EP02 notebook 管线重建源图 → uv run python scripts/paper_figures/collect_promoted_supp.py",
)
figS12

# %% [markdown]
# > **图表说明**: 每行一个 AVI 的 θ 估计与不确定度；蓝=X-scan、红=Y-scan；
# > 底部三行为合并估计；竖虚线为标定参考 47.6°，绿色带为合并 95% CI。
# > **怎么看**: gradient NCC（右）合并估计 47.14°、CI [46.36°, 47.92°] 覆盖
# > 47.6°——独立数据源对标定角的方向性背书；同时 X 与 Y 子组存在 ~3° 系统差
# > （highpass 域更明显），这正是 AVI 不能作为高精度标定源的原因。
# > **异常是否正常**: highpass NCC（左）合并 CI 明显更宽且偏离参考——8-bit
# > 渲染视频的特征域敏感性，作为方法学对照保留。
# > **核心发现**: AVI 验证的角色边界被图形本身界定——consistency check 通过，
# > 但 X/Y 系统差禁止用它替换 `stage_calibration.json`（AGENTS 硬教训 13）。
# > **状态**: ✅ 选编收编（06-12）。
