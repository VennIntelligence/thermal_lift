# DC 残差置信分析 — 抹除点在数据一致性残差里**可检出**（AUC 0.68-0.84）

> 2026-07-11，任务 #6（owner 批准的收尾分析之一）。零训练、纯本地。
> 脚本：`algos/ep07_unet_sr/scripts/analyze_dc_residual_confidence.py`（可复跑、参数化）。
> 产物：`output/dc_residual_confidence/`（残差图 npy ×6、per_dot_residual_stats.csv、auc_table.csv、
> example_crops_depb9v9s2.png、run_meta.json）。

## 问题

模型抹除小点时，|y − A(x̂)| 能不能把抹除位置暴露出来？（能 → "自我怀疑仪表"；
不能 → 抹除在前向算子零空间内，DC 残差结构性失明。）

## 方法（复用既有机制，无新算子）

1. x̂ = 渲染包 `remote_inbox/20260710_expab/raw/{arm}_{a,b}.npy`（centered 网格）→
   `from_center_grid`（real_eval.to_center_grid 的精确逆）回到 native 角约定网格。
2. A = `forward_torch.forward_burst`，PSF = **σ=0.5 占位高斯**（= 各 checkpoint 推理期 DC 用的同一算子；
   本分析测的是"求解器自带算子的可部署自检能力"，不是物理真值——σ 未标定的 caveat 与 ACL-032/059 一致）。
3. 帧选择：只用**该半份中 DC 子集（m=12，linspace 选择）之外的 112 帧**（镜像 real_eval 的诚实残差协议，
   排除自拟合）。
4. 逐帧 |y − A(x̂)| 经 `run_m2_frc.bilinear_drizzle`（同帧同 shifts）splat 到 drizzle HR 网格 →
   残差图与 per_dot.csv 坐标同框（render manifest 显示 centered 渲染距 drizzle 网格 ≤0.09 HR px，
   相对 ≥3px 统计窗可忽略）。两半份平均。
5. 逐点统计：窗口（half = clip(⌈2σ_dot⌉+2, 3, 8)）内 max/mean，原图 + 高斯背景扣除（σ_bg=8px）两版；
   null = 3000 个远离任何点 ≥10px 的随机位置。AUC = Mann-Whitney。
6. 对齐验证：12 个最大点中 9 个在 drizzle 图 (y,x) 处比局部中位数暗 → 坐标系 OK。
   sanity：残差图在强结构边缘有系统性痕迹（σ 失配的模型误差，rms_lr≈1.45-1.52）→ 算子装配正确。

## 结果（auc_table.csv 全表在产物目录）

| 臂 | n_erased | AUC erased-vs-kept (win_max) | AUC erased-vs-null | 中位 |resid|：erased / kept / null |
|---|---|---|---|---|
| depb9v6 | 378 | 0.682 | 0.784 | 0.192 / 0.124 / 0.108 |
| depb9v9s2 | 206 | 0.766 | 0.853 | 0.213 / 0.122 / 0.107 |
| depb9v9_3k | 139 | **0.838** | **0.916** | 0.216 / 0.121 / 0.107 |

（bs_max（背景扣除后局部峰）同方向：0.645/0.715/0.798 vs kept。bs_mean 最弱（0.53-0.66）——
信号是**点状局部峰**，不是窗口均值抬升，符合"漏渲染的点在残差里留下点状足迹"的物理预期。）

## 判读

1. **DC 残差对抹除不失明——"自我怀疑仪表"成立**。被抹除的点在 held-out 帧残差里留下可检出的局部峰，
   即使用的是 σ=0.5 未标定占位算子。AUC 0.68-0.84（vs 其余点）/ 0.78-0.92（vs 背景）= 有用但非完美的
   检出器：适合作"此区域低置信、建议人工复核"的标注层，不适合当硬判据。
2. **方向偏置反而加强结论**：erased 点系统性偏小偏浅（probe 三轴数据），小浅点本应留下**更弱**残差——
   观测到的却是更高残差，说明判别不是尺寸混杂的假象。
3. **点保真越好的臂，其残余抹除越"无知觉外露"**（AUC 单调：v6 0.68 < v9s2 0.77 < 3k 0.84）。
   候选解释：好臂只抹它"最想抹"的（先验最强处），这类抹除与数据的冲突最大 → 残差峰最亮；
   差臂大面积抹除里混入大量边缘案例，正类被稀释。对论文是个干净的叙事：**架构上的硬 DC 把
   "先验强行覆盖数据"的行为压进了可观测面**——这正是物理约束求解器相对黑盒网络的可审计性卖点。
4. 结构性含义（C3 章）：抹除**不**完全落在前向算子零空间内（否则 AUC≈0.5）。浅深度小点经 A 降采样
   后仍保留可测能量——与 L1 审计（ACL-068：v9 配方点在输入侧 CNR 可检出）闭环一致。

## 对论文的措辞建议

- C1/C3 加一小节 "Self-auditing via held-out data-consistency residuals"：AUC 表 + example crops 图 +
  威慑性 caveat（σ 占位、AUC 非完美、类别来自 probe 而非人工 GT）。
- Future work 保持 ACL-024 伏笔：要把"标注层"升级为校准的不确定性图，走后验采样（DPS/ΠGDM）。

## 诚实边界

- 类别标签（erased/preserved/blurred）来自点保真探针的深度比阈值，非人工标注；探针与残差用了
  同一重建但信息源不同（类别=重建 vs TGV 深度对比；残差=重建 vs 原始帧物理一致性），非循环。
- σ=0.5 未标定：AUC 是该占位算子下的下界估计；真 σ 标定后可能更高（模型误差背景会降）。
- 单一真实 session（全项目 limitation，此处同样适用）。
