# Figure Atlas Backlog（实验室剧本级全量图集待画清单）

目标：把 relog（ACL-001→076）+ 各 episode 的**所有**有价值对比/结果都画成出版级图，预计上百张。
fig01–fig21 已完成（见 README.md）。本清单从 fig22 起，按系列组织。

## 执行规约（恢复会话后照此办）

1. 风格与流程同 README.md：`pubfig_style.py` 统一风格；每图自包含脚本 + docstring 写数据来源/ACL 引用/运行命令；渲染后**必须回读 PNG 查遮挡**再收工。
2. **分工（owner 指令）**：难图（多面板设计、需要读懂实验语境、图像域处理、数据需推断）→ **spawn Fable 子代理**（Agent tool, model "fable"）；机械图（单 CSV 直画、既有模板复用）→ Sonnet 子代理。主循环只做 QC 和修碰撞。并行 3–4 个代理一批。
3. 数据不在本地时：优先 remote_inbox / output / research_log archive；5090 上的数据（如 output/defect_detectability）用 ssh 拉回（机器信息见 memory thermal-lift-machines；注意 macOS `base64 -D` 陷阱，参考 fig18 docstring）。
4. 每完成一批就把 README.md 表格补上 + 本文件勾掉。任务用 TaskCreate 跟踪。

## A 系列 — solver 内部机制（多为难图 → Fable）

- [x] fig22 K4 方块伪影根因分解（ep07_solver_boundary_artifact/figures 的 01–05 步分解 x0→prox1→dc1→prox2→dc2，重排为出版级多面板；manifest: figure_manifest.tsv）【难】
- [x] fig23 halo 扫描：伪影 vs halo 0/64/96/128 + 墙钟/显存开销双面板（archive README 表格 + figures/06-12）【难】
- [x] fig24 extent-invariance 实验：GroupNorm+SE 远场扰动 1.4σ vs 纯卷积 0（archive DIAGNOSIS.md + diag_extent.py 输出）【难】
- [x] fig25 seam 频谱诊断（figC_seam_spectrum 重绘，outputs/ep07_solver_diag/metrics_arrays*.json）【中】
- [x] fig26 de-waffle x0 + dc_resid 指标演示（ACL-032；de_pb9_probe jpg + 脚本重算）【难】
- [x] fig27 D-E prox 高通残差七臂表→图（ACL-042 结果表格转录）【易→Sonnet】
- [x] fig28 v4/v5-era checkpoint npz 温度图演化条带（remote_inbox/20260627_checkpoint_evolution/solver_v4_acl027/*.npz，配 ACL-027 loss 重设计语境）【中】

## B 系列 — 测量仪器学（stage0 saga 细节）

- [x] fig29 偏移探针跨阶段稳定性：offset_probe_summary_{stage0g,0h,0j,DE,AB,v21}.csv 汇总（remote_inbox/20260713_dotprobe/）——神经臂 ~0.6-0.8px 偏移在每一代复现【中】
- [x] fig30 stage0f 仪表修复前后：t2_frc_method_summary + t3 回归指标可分性（20260704_stage0f/t3_*.csv；ACL-046/047 0d 套件"仅 extent 探针有判别力"）【难】
- [x] fig31 drizzle 多分半控制：odd_even vs phase_stratified vs seed 变体 FRC（20260705_stage0g/drizzle_*_frc_curve.csv + task1_3 summary）【易→Sonnet】
- [x] fig32 same-half 对照（task2_samehalf_method_summary.csv，shared-prior 幻觉论证）【中】
- [x] fig33 stage0j 全臂 FRC 排行榜曲线扩展版（v14 20k/50k、v14×tgv 交叉对；fig06 只画了主线）【易→Sonnet】
- [x] fig34 M1 相位占用（20260704_stage0f/t0e_m1_*.csv：detector bins 5x、stage lattice、command phase bins——4x 不可行论证）【难】

## C 系列 — 训练池与点保真细节

- [x] fig35 L1 可探测性审计角落热图：半径×深度 CNR 病理角（ACL-068；数据在 5090 output/defect_detectability/，需拉回）【难】
- [x] fig36 dot_probe 检出漏斗（output/dot_probe/detection_funnel.json + board_crops 重排）【中】
- [x] fig37 retention vs size 连续曲线 + optical subset 对照（output/dot_probe/retention_vs_size.png 重绘、optical_subset.csv）【中】
- [x] fig38 v22/v24ctrl 臂对照表→图（output/dot_probe_v24ctrl + 20260716_*/probe_out/summary_v22_arms_combined.csv）【易→Sonnet】
- [x] fig39 micro 端点重建视觉条带（20260716_micro_calib npy：micro_v6end/v7end 4k/8k vs depb9v6 vs tgv，同 fig10 变换）【中】
- [x] fig40 v8/v9 代 range_exc 系统性偏高之谜（各代 range_exc 汇总时间线；ACL-071/072/074 未决异常的证据图）【中】
- [x] fig41 tcforge gate 审计（assets/v7_planning/composer_demo_r4/gate_audit.json G1-G8 median/p95/pass）【易→Sonnet】
- [x] fig42 v7 composer 缺陷体系 showcase（scripts/v7_composer_demo.py 三张 sheet 重绘出版级；或 assets/v7_planning sheets 选材）【难】
- [x] fig43 OOD round-2 motif 族预览（organic_blobs/text_serial/rings/voronoi；tmp/motif_previews_20260710/ 若已删则用 tcforge 现场生成，seeds 20260930-33 配置已入库）【难】

## D 系列 — 收尾判决扩展

- [x] fig44 OOD 其余指标：range_excursion / mean_offset / band_rmse 按池×臂（20260712_oodC summary 其余列；v9_9bin mean_offset 系统性负值 -2~-5 是个发现）【中】
- [x] fig45 stage2b 全景：v6bench vs v8bench 双基准所有臂 band_FRC/range_exc 散点（20260717_v8_champion/stage2b_* + 20260711_stage2b/stage2b_summary.csv）【难】
- [x] fig46 DC 残差空间图示例：residmap npy + 点位标注（output/dc_residual_confidence/depb9v9_3k_residmap_a.npy + per_dot 坐标；example_crops 重绘）【难】
- [x] fig47 DC 残差按统计量对比（win_max/win_mean/bs_max/bs_mean 四统计 AUC——"点状局部峰非窗口均值"物理判读）【易→Sonnet】
- [x] fig48 σ 线全景：EP09 三路发散（0.23/1.13/0.12）→ E1/E2 简并 → E3 修复时间线叙事图（ACL-056/057/058/059 + ep09 README 数字）【难】
- [x] fig49（并入 fig48 面板 c） 真实数据 Step2 合法拒绝：0/8 边过质量门（20260712_sigma/ 数据；σ "鲁棒带"策略图）【中】

## E 系列 — 旧 episode 补强（output 缓存缺失的先跑 build 脚本再画）

- [!] fig50 **阻塞**：CSV 三处皆无（5090 无 ep16 目录、数据服务器 100.108.238.8 不可达、git 从未收录）。恢复选项：(a) 帧预算表可从 docs/paper/supp/D_full_results.md L140-149 转录；(c) 4 行可从正文数字转录；(b) σ=0.05/0.1 档只存在于 paper/slides/figures/fig07_budget_robustness.pdf 矢量内。等数据服务器恢复后查 ~/mycode/thermal_lift/output/ep16_budget_robustness/
- [!] fig51 EP15 M4 四臂对比 + zigzag 剖线【阻塞：ep07 v6 checkpoint 缺失】。2026-07-16 复核：M1 已在 5090 重跑成功（m1_phase_structure_summary.json 产出，grid_decision.json 缺失但 M4 的 load_grid_decision 会回退到 summary，非阻塞）；GPU 空闲；configs/stage_calibration.json 在。**真阻塞**：M4 第 4 臂 "EP07 v6 x2.5up" 要一个 scale=2 的 ep07 UNet `model_final.pt`（脚本 line 799 硬断言 scale==2），但默认路径 `ep07_v6_physics/model_final.pt` 不存在、fallback npy `output/ep12_4x_benchmark/ep07x2up_vs_ep12/ep07_2x_x2up_temp.npy` 也不存在，全盘只有两个 scale-2 model_final.pt：`ep07_v10_v3_lam02`(v3)、`v10_v5_sharp`(v5)，**无 v6**（池迁移时删了）。需 owner 决定：指定替代 checkpoint / 重生成 v6 ep07 权重 / 用 v5_sharp 并诚实改标注 / 暂缓。其余输入（248 帧、frame_audit、contour alignment）齐备，一旦 checkpoint 定了即可 `uv run python algos/ep15_info_limit/scripts/run_m4_deconv_anchor.py --ep07-checkpoint <path>`（GPU MAP-TV，约 10-40min）。
- [x] fig52 EP08 INR 五方法表→图（README 表格转录：SIREN/WIRE/DeepDecoder/DIP/MAP-TV × 5 指标）【易→Sonnet】
- [x] fig53 EP02 θ 森林图 + AVI 位移证据重绘（output/ep02 缓存缺失需 build_ep02_cache）【中】
- [x] fig54 EP01 采集会话检测/栅格轨迹重绘（output/ep01_data_processing/frame_audit.csv 在）【中】
- [ ] fig55+ 其余 episode 图按 paper/asset_manifest.json 清单逐个核对补画（每个先查缓存是否可重建）

## F 系列 — 汇总/叙事图（全难 → Fable）

- [x] fig90 全项目时间线图：ACL 编号×日期×主题泳道（数据：changelog 标题行，已可 grep）
- [x] fig91 "神经 vs 经典差距演化"总图：@30µm 差距随时间/阶段收敛曲线（0.053→0.031→…各 ACL 锚点）
- [x] fig92 判据管线示意图：为什么 self-FRC 无效、cross-FRC+offset 校正流程框图
- [x] fig93 三轴验收雷达/平行坐标：champion 候选臂 FRC/点保真/OOD/低频稳定四维

## G 系列 — 多模态视觉可视化 (2026-07-13)

- [x] fig60 光学显微真值配准到 HR 热网格（θ=225.2°，NCC 0.985）+ 走线等高线叠加 4 臂验几何一致性
- [x] fig61 DC 残差自审计实际看到的：抹除点在留出残差图中以局部凸起显形（最清晰正例展示，AUC 0.68-0.84 见 fig04）
- [x] fig62 抹点传奇视觉版：6 个孤立点跨池代追踪，v7 抹掉（43%）→ v8/v9/v9-3k 逐代找回（4.3%/1.6%/0.0%）
- [x] fig63 自分半 FRC 奖励什么：平坦区验收带纹理半幅相关（drizzle r=0.15 vs TGV 0.84 / v6 0.70；v9-3k 0.11 噪声化）
- [x] fig64 halo=96 全帧推理随训练步演化 2×3 蒙太奇（5k/20k/40k 温度 + TB 高通细节，de_pb9 探针存档 JPG）
- [x] fig65 v7 微型训练时长视觉演化：12k/16k/20k/24k vs drizzle/TGV，双行温度+匹配高通
- [x] fig66 FRC 增益住在哪：中心 ROI FFT 环带分解（25-40µm 验收带 + 20-25µm 次验收带诚实标注）
- [x] fig67 合成训练场景解剖：3 场景 × (HR 真值 T / LR 观测 / 覆盖掩膜 cov)，温度列共享色标
- [x] fig68 各臂相对 drizzle 差值图：新增对比度锚定真实走线边缘，TGV 条纹最强，v9-3k 最保守
- [x] fig96 重建对切分 seed 视觉不变：臂 × seed 42/123/456 匹配高通面板（ACL-077 配套，远端渲染于 5090，脚本归档为 fig96_split_visual_consistency_REMOTE.py）
