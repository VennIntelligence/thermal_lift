# %% [markdown]
# ## S-F9 — 零训练融合 baseline 的 fine-window Pareto（supp D.7 配图）
#
# fused(λ) = (1−λ)·anchor + λ·UNet 的事后线性融合扫描，叠加 V9A 训练时间轴
# 轨迹与 TGV/drizzle 参考点——V10 必须打败的对照前沿。

# %%
figS09 = show_figure(
    PAPER_FIGS / "figS09_fusion_pareto.png",
    "uv run python scripts/paper_figures/figS09_fusion_pareto.py",
)
figS09

# %% [markdown]
# > **图表说明**: 横轴保真（fine-window highpass corr vs drizzle 输入通道，↑），
# > 纵轴锐度 proxy（P95 梯度，↑ 但振铃/假边也会推高）；灰色折线为 V9A
# > checkpoint 轨迹（5K→60K），红/绿曲线为四组融合扫描（λ 从锚点 0 到纯 UNet 1），
# > 右上浅红区为「严格支配 TGV 工作点」象限。
# > **怎么看**: 只有 **TGV + λ·V9A-60K**（红实线）在 λ≈0.1–0.3 进入支配象限
# > （最强候选 λ=0.2：保真 +0.003、锐度 +0.009、格纹 −36%）；drizzle 锚的两条
# > 曲线保真极高但锐度不过线；V9A 自身轨迹整体被融合曲线包络。
# > **异常是否正常**: 轨迹在 15K/25K 处的折返是真实训练动力学（保真-锐度来回
# > 交换）；fine-window 是局部诊断口径，TGV 非 ground truth（supp D.0 限制声明）。
# > **核心发现**: 存在零训练、零 GPU 的事后融合点严格支配 TGV 工作点——
# > Claim 4 的成功判据由此从「越过 TGV」抬高为「越过融合前沿」。
# > **状态**: ✅ 新落地（学术重绘）；⬜ V10 三臂工作点等训练后叠加同图。
