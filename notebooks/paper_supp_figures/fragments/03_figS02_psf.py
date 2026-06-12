# %% [markdown]
# ## S-F2 — PSF 三路标定证据链与 M3 仲裁（supp A.5.3 配图）
#
# EP09 三路标定分歧（spread 1.01 LR px）与 EP15 M3 仲裁的完整视觉证据。

# %%
figS02 = show_figure(
    PAPER_FIGS / "figS02_psf_evidence.png",
    "uv run python scripts/paper_figures/figS02_psf_evidence.py",
)
figS02

# %% [markdown]
# > **图表说明**: (a) Route A（forward 残差，248 帧）与 Route C（joint hold-out）
# > 的相对残差扫描，蓝带为 Route A 95% CI [0.208, 0.240]；(b) Route B 的表观
# > ESF 宽度直方图（外边框 33 段）+ M3 三个边缘族中位线 + 最锐单边缘上界 0.55；
# > (c) M3 FRC 形状拟合 MSE vs σ，蓝带为采纳区间 [0.2, 0.5]，红色标记为三路估计。
# > **怎么看**: (a)(c) 的残差/拟合最优都落在 0.12–0.23 一带；(b) 的表观宽度
# > （0.85–2.0）整体远在采纳区间右侧——它测的是 PSF ⊗ 热/几何边缘宽度
# > （σ²_total = σ²_PSF + w²_edge），不是纯光学 PSF。
# > **异常是否正常**: 三路分歧本身是 finding 而非错误；Route B 的"偏大"被 M3
# > 分解解释，且给反卷积激进度设了物理上限。
# > **核心发现**: 采纳 σ∈[0.2, 0.5] LR px 区间而非单点；所有 PSF 依赖计算
# > 扫区间或显式报告 σ（主文 §3.3 的 supp 背书）。
# > **状态**: ✅ 新落地（本次图优化）。
