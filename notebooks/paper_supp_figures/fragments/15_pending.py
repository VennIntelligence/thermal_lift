# %% [markdown]
# ## 尚未定稿的 supp 图表（占位与依赖）
#
# | 编号 | 内容 | 阻塞依赖 | 素材位置 |
# |---|---|---|---|
# | S-F5 | 各臂 step 序列视觉演化选帧（每臂 early/canonical/30K/60K） | V9C/V10 落地后统一选帧 | `algos/ep07_unet_sr/outputs/*/eval_real/` |
# | S-F6 | 负结果档案组图（v8.1b 条纹 crop、EP12 4x 对比、AVI 排除审计） | 选图 + 组版（素材已齐） | EP11/EP12/EP01 输出 |
# | S-T1 | T1 扩展表 | 统一口径 harness（GPU 窗口） | — |
# | S-T3/S-T4 | 合成参数表 / 融合 λ 全表 | ✅ 已成表于 `docs/paper/supp/` C.1、D.7 | markdown 表，无需图 |
#
# **可选重绘**（价值确认、优先级低）：
#
# | 候选 | 现状 | 重绘动机 |
# |---|---|---|
# | TCForge 平台总览（`output/ep07_thermal_chip_phantom/demo_dataset_overview.png`） | 内容对 supp C.1 有价值，但 colorbar 截断、版面粗糙，且基于旧 demo 池 | 用当前 2x_aa_burst 池重出 4 难度 × (coverage/温度) 画廊 |
# | TGV 条纹修复 before/after | 修复前（各向同性）重建未存档 | 需在 TGV conda 环境重跑 isotropic 臂（~30 min CPU），可与 GPU 臂补线同批 |
# | 位移覆盖三类箱线图（`output/ep05_sr_reassessment/visible_shift_by_pair_class.png`） | 默认 matplotlib 风格 | 数字已入 supp B.3.4 表格，重绘增值有限 |
#
# 注：S-T 系列按项目规范保持 **markdown 表格**形态，不渲染为图片。

# %%
print("S-F5/S-F6 为下一批组版目标；S-F7/S-F11–S-F15 已于 06-12 选编收编。")
