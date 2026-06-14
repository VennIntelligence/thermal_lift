# %% [markdown]
# ## 尚未定稿的主文图表（占位与依赖）
#
# | 图表 | 内容 | 阻塞依赖 | 预期生产路径 |
# |---|---|---|---|
# | F0 teaser | bicubic / TGV / 最优学习臂 中心 zigzag 三列 | F5 终稿 | F5 裁剪 |
# | F5 主视觉对比 | 温度域 + highpass 域 × 6 列 + 消融行 | ① V9A/V9C/V10 选点 ② 统一口径 harness ③ fine-window 坐标冻结 | `run_unet_vs_drizzle_2x.py` 谱系 |
# | T1 主定量表 | 全臂 × 全列单口径数字 | 统一 harness 重跑（GPU 窗口） | 同上 |
# | T2 消融矩阵 | {1x, hybrid} × {none, band, full, legal} | V9C 60K（今晚）+ V10 | supp C.3 表 + harness |
#
# 状态权威登记：`docs/paper/09_figures_tables_assets.md`；
# 写作交接：`docs/paper/10_writing_handover.md`。

# %%
print("F0/F5/T1/T2 等统一 harness 与 V9C/V10 落地，见上表依赖。")
