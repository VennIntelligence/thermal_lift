# %% [markdown]
# ## 尚未定稿的 supp 图表（占位与依赖）
#
# | 编号 | 内容 | 阻塞依赖 | 素材位置 |
# |---|---|---|---|
# | S-F5 | 各臂 step 序列视觉演化选帧（每臂 early/canonical/30K/60K） | V9C/V10 落地后统一选帧 | `algos/ep07_unet_sr/outputs/*/eval_real/` |
# | S-F6 | 负结果档案组图（v8.1b 条纹 crop、EP12 4x 对比、AVI 排除审计） | 选图 + 组版（素材已齐） | EP11/EP12/EP01 输出 |
# | S-F7 | 对齐管线与 gate 图（Chamfer 链 + EP04 角色表） | 选图 + 组版（素材已齐） | `output/ep04_*`、`output/ep05_*` |
# | S-T1 | T1 扩展表 | 统一口径 harness（GPU 窗口） | — |
# | S-T3/S-T4 | 合成参数表 / 融合 λ 全表 | ✅ 已成表于 `docs/paper/supp/` C.1、D.7 | markdown 表，无需图 |
#
# 注：S-T 系列按项目规范保持 **markdown 表格**形态，不渲染为图片。

# %%
print("S-F5/S-F6/S-F7 为下一批组版目标；S-T 系列已在 supp 草稿中成表。")
