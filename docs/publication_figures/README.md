# Publication Figures (changelog-era results)

出版级科研图集，覆盖 solver/训练池演化时代（ACL-023 → ACL-080）的全部判决性结果。
旧 episode（ep01–ep16）的论文图由 `scripts/paper_figures/` 管线负责，本目录不重复。
`GALLERY.md` 是全图册，含每图嵌入与中文说明。

## 布局

- `scripts/` — 每图一个自包含脚本 + 共享风格模块 `pubfig_style.py`（实现 `docs/plotting_standards.md`，serif/Times、300 dpi、CVPR 栏宽）。**改图只改对应脚本，不要动别的**；字号/色板一律走 `pubfig_style`，禁止脚本内硬编码。
- `figures/` — 输出（`.png` 预览 + `.pdf` 投稿矢量），由脚本再生，勿手改。
- `data/` — 从 `research_log/algorithm_changelog.md` 判决表格转录的小 CSV（带 ACL 引用），是脚本的稳定数据源；其余数据直接读 `remote_inbox/` 与 `output/`（均 gitignored，路径见各脚本 docstring）。
  - `remote_inbox/20260713_content2ms/ms_curves/` — lb_seed{42,123,456} 逐 seed 的 `frc_curves_long.csv` / `method_summary.csv` / `split_balance.csv` / `run_manifest.json`，2026-07-13 从 5090 `output/stage2p5_multisplit_v2` 取回（ACL-077，fig94 数据源）。

再生任一图：`cd <repo-root> && uv run python docs/publication_figures/scripts/figNN_*.py`

## 图目录（编号 = 脚本前缀）

| 图 | 内容 | 数据源 | ACL |
|---|---|---|---|
| fig00_main_narrative | 主线总结示意图：四幕（物理地基/诚实仪器/算法与先验/权衡定律）→ 冠军判决；GALLERY 开篇图 | 纯示意（数字转录自各判决） | 023-080 |
| fig01_pool_evolution | 训练池 v6→v7→v8→v9→v9-3k 点抹除/保真/FRC 三面板，v7 病理红色高亮，断轴 | `data/pool_evolution.csv` | 066/070/071/072/074 |
| fig02_champion_pareto | FRC@30µm × 孤立点抹除率帕累托前沿 + OOD 胜率标签 | `data/champion_arms.csv` | 064/071/072/074/076 |
| fig03_ood_robustness | 9 个 OOD 极端池 4 臂绝对值 + Δ(neural−oracle) 符号判决；v6 9/9 全胜 | `remote_inbox/20260712_oodC/ood_degradation_summary.csv` | 076 |
| fig04_dc_residual_audit | DC 残差自审计：AUC 随臂保真单调 + 按点类残差箱线 | `output/dc_residual_confidence/` | 075 |
| fig05_training_dynamics | 展开 solver vs UNet 六面板 TB 标量（真实/合成/损失） | `remote_inbox/20260627_checkpoint_evolution/.../{solver_v5_sharp_hybrid,v10_v5_sharp}_scalars.csv` | 029 时代 |
| fig06_frc_leaderboard | 真实域对称 cross-FRC 曲线（TGV/MAP-TV/v14），25–40µm 增益带 + 20µm 孔径零点阴影 | `remote_inbox/20260708_stage0j/frc_curves_long.csv` | 047/048/049 |
| fig07_registration_artifact | +0.5px 网格约定翻案：实测偏移条形 + 校正前后 FRC 哑铃 | `remote_inbox/20260713_dotprobe/offset_probe_summary_stage0h.csv` | 049 |
| fig08_prior_emergence | 抹除先验需要池规模：300 景 24k 步全程 0% vs 5k 池 39.75%；按配方 erased% vs 池规模 | `remote_inbox/20260716_micro_{calib,horizon}/summary_micro.json` + `data/prior_emergence.csv` | 069/074 |
| fig09_dot_probe_stratified | 点探针分层保真（尺寸/深度/孤立度），6 臂 | `output/dot_probe/summary_by_arm_*.csv` | 063 |
| fig10_real_visual_montage | 真实半幅 A 三 ROI × 4 臂蒙太奇；统一 σ=10 AC 耦合；平坦区独立标尺显噪声纹理 | `remote_inbox/20260716_v8_verdict/*.npy`（裁剪窗复用 visboard manifest） | 070 时代 |
| fig11_stage2b_stratified | Stage2b 合成基准按噪声/ΔT 分层曲线（±SEM，n=16/档） | `remote_inbox/20260711_stage2b/stage2b_stratified_*.csv` | 054/062 |
| fig12_checkpoint_evolution_strip | checkpoint 演化条带：solver 5k 步即锐利 vs UNet 需 15k+ | `remote_inbox/20260627_checkpoint_evolution/.../eval_real_png/` | 029 时代 |
| fig13_sigma_selfcal_bench | σ 自校准 E3 内核 bench 判决：σ̂ vs σ 散点 + 相对误差（median 4.1% PASS） | `output/sigma_esf_bench_reverdict/` | 056/057/058 |
| fig14_eta_calibration | η 校准三面板：8 点扫描（η*=0.09）、synth/real 解耦、高原记分板（廉价路线全平） | `data/eta_sweep.csv`、`data/plateau_scoreboard.csv` | 051/052/053 |
| fig15_solver_training_diagnostics | V8/K4 solver 训练诊断六面板（损失分量、DC 残差、伪影、冻结 η/退火） | `research_log/episodes/ep07_solver_v8_k4_fullhalo_eval_archive/scalars/` | 040/041 时代 |
| fig16_forward_selfcheck | 前向算子自检认证：带宽 FRC（HR Nyquist 处 0.715，带内不过 0.5）+ chirp 混叠演示 | `output/forward_selfcheck/` | 023 |
| fig17_v21_convergence | v21 checkpoint 收敛：depb9v6 20k 平台 vs meanDC 15k 后灾难发散 | `output/v21_eval/v21_convergence_table.csv` | 062 时代 |
| fig18_ood_perscene | OOD 逐景配对分布（fig03 的均值不是离群点驱动；胜率 69–98%） | `remote_inbox/20260712_oodC/ood_degradation_long.csv`（曾损坏，已从 5090 重取，见脚本 docstring） | 076 |
| fig19_sampling_resolution | 探测器 pitch / 标定分辨率 / SR 网格区分示意图（调 `thermal_core.ep03`；顺手修了核心函数里 10µm 时代的硬编码标签 bug） | `thermal_core.ep03` | 023 |
| fig20_synthetic_showcase | 合成训练场景 showcase（motif 族画廊 + 单景分解） | `outputs/v7_demo_minipool/` | 065 |
| fig21_shift_refinement | 逐帧 shift 误差瓶颈：248 帧修正散点（RMS 0.31px）+ M2 FRC 34.1→25.5µm | `remote_inbox/20260704_stage0f/` + `20260705_stage0g/` | 047/048 |
| fig22_k4_box_artifact | K4 glow-box 根因：x0→prox1→DC1→prox2→DC2 步分解条带（TB 导出，定性无标尺） | `research_log/episodes/ep07_solver_boundary_artifact/figures/` | 037 |
| fig23_halo_sweep | 外圈 halo 扫描：伪影消退条带 + 墙钟/显存成本（halo128 显存为分配器效应，开标记） | 同上 + `data/halo_sweep.csv` | 038 |
| fig24_extent_invariance | GN+SE 破坏范围不变性：272px 远场扰动 1.4σ vs 纯卷积 0 + K 步放大线（随机初始化架构测量） | `outputs/ep07_solver_diag/metrics_extent.json` | 037/040/041 |
| fig27_de_prox_arms | D-E prox 高通残差七臂点图（E2 σ_hr=4 最均衡；整体阴性） | `data/de_prox_arms.csv` | 042/043 |
| fig29_offset_probe_stability | 偏移探针跨时代：校正前 0.6-0.8px vs 后续各代残余 ≤0.1px | `remote_inbox/20260713_dotprobe/offset_probe_summary_*.csv` | 049+ |
| fig31_drizzle_split_controls | drizzle 分半 FRC 对 split 选择稳健（phase-stratified vs odd/even） | `remote_inbox/20260705_stage0g/drizzle_*_frc_curve.csv` | 048 |
| fig33_frc_band_table | 跨方法 FRC 频带剖面（含方向不对称须、20µm 审计列） | `remote_inbox/20260708_stage0j/method_summary.csv` | 049/050 |
| fig40_range_exc_anomaly | v8/v9 代 range_exc 12-16 之谜（健康带 1.6-4.4 阴影；v9 4-bin 10³ 灾难） | `data/range_exc_by_generation.csv` | 071/072/074 |
| fig41_gate_audit | TCForge G1-G8 质量门审计小倍数（G5 对比度 FAIL 1.45×<1.5× 如实展示） | `research_log/assets/v7_planning/composer_demo_r4/gate_audit.json` | 065 前夜 |
| fig43_ood_motif_previews | 四个 OUT-OF-GRAMMAR motif 族预览蒙太奇（organic blobs / serial text / concentric rings / voronoi cells，4 景/族，逐 tile 标尺+数值范围标注） | `remote_inbox/20260713_content2ms/motif_previews/fig43_previews.npz` | 073/078 |
| fig44_ood_secondary_metrics | OOD 次级指标：v9_9bin 全池 DC 负漂 −2~−5.5°C + range_exc 高一个量级 | `remote_inbox/20260712_oodC/ood_degradation_summary.csv` | 076 |
| fig47_dc_residual_stats | 残差统计量对比：max 类完胜 mean 类（点状局部峰物理判读） | `output/dc_residual_confidence/auc_table.csv` | 075 |
| fig51_ep15_m4_deconv | EP15 M4 反卷积锚四臂对比（bare drizzle/bicubic/MAP-TV/EP07）+ 温度/高通 ROI 蒙太奇 + zigzag 暗迹剖线 + M4 自分半 FRC；EP07 臂=v6 配方在 pool_2x_v9_5k 重训 2026-07-16（非原 v6 池） | `remote_inbox/20260716_ep15_m4/` | EP15 M4 |
| fig52_inr_methods | EP08 学习先验 vs MAP-TV 五方法对比（SIREN 最均衡） | `data/inr_methods.csv` | EP08 |
| fig53_ep02_theta_displacement | EP02 位移标定证据三面板：θ 森林图（AVI 连续扫描 47.1° CI [46.4,47.9] 覆盖配置 47.6°；TXT NCC 各证据层 30-41° 系统偏低，仅作方向 smoke test）+ 可见 vs 命令位移 log-log（时间相邻三档 0.1/0.2/2 px 全落 y=x）+ 逐方法投影比（时间相邻 0.89-1.09；Y 坐标对 2µm 漂移膨胀 ×3.2-3.4） | `remote_inbox/20260715_ep02_recal/` | EP02 |
| fig54_ep01_acquisition | EP01 采集审计三面板：亚像素步进-静止 raster 轨迹（248 可用帧按采集顺序着色 + 排除帧/缺采位）、mtime 会话时间线（冷流产段 +3.6°C 跃变→热稳定主会话→行 0 复检）、263 帧预算分解 | `output/ep01_data_processing/frame_audit.csv` | EP01 |
| fig28_v4era_checkpoint_strip | ACL-027 时代 checkpoint 条带（原始 npz，真 colorbar；2.5k 已基本收敛） | `remote_inbox/20260627_checkpoint_evolution/solver_v4_acl027/` | 027/029 |
| fig30_regression_metric_separability | 0d 回归套件可分性：仅 extent 探针分得开（×2.4-2.9） | `remote_inbox/20260704_stage0f/t3_*.csv` | 046/047 |
| fig32_samehalf_control | 自分半 vs 诚实 cross：~0.4 幻觉膨胀 + ~0.4 配准伪影两段分解 | `remote_inbox/20260703_stage1a` + `20260704_stage0f` + stage0h 校正值 | 046/047/049 |
| fig34_phase_occupancy | 5× 相位占用：实测 11/25 bin、四角堆积 vs 指令 25/25——4x/5x 信息不足 | `remote_inbox/20260704_stage0f/t0e_m1_*.csv` | EP15-M1/048 |
| fig36_dot_probe_funnel | 点探针检出漏斗 5171→3562 + 单点跨臂样例条带 | `output/dot_probe/detection_funnel.json` | 063 |
| fig38_v22_arms_probe | v22 era 臂：配准/增益校准稳定 vs 保真悬殊（0.33-1.13） | `output/dot_probe_v24ctrl` + `20260716_*/probe_out/` | 067/069/070 |
| fig39_micro_endpoint_visuals | 300 景微型臂重建视觉条带（结构合理、更软；配 fig08） | `remote_inbox/20260716_micro_calib/*.npy` | 069 |
| fig46_residmap_spatial | DC 残差空间图 + 139 个被抹点标注 + 4 个最强峰放大（坐标经验证 AUC 0.887） | `output/dc_residual_confidence/` | 075 |
| fig91_gap_evolution | 叙事总图：各臂 FRC@30µm vs TGV 时间线，健康/病理开闭标记（健康纪录 0.6705 天花板） | `data/gap_evolution.csv` | 050-074 |
| fig25_seam_spectrum | seam 自相关 + grid prominence 崩塌 16×（原始渲染数组已失，用 json 汇总，诚实处理） | `outputs/ep07_solver_diag/metrics_arrays*.json` | 037/038 |
| fig37_retention_vs_size | 连续 retention vs 直径 + 42 个光学验证点 + 逐 bin n 密度条 | `output/dot_probe/per_dot.csv` + `optical_subset.csv` | 063 |
| fig45_stage2b_panorama | Stage2b 全景散点（19 个臂×基准点；v20 损坏 checkpoint 10⁵ 标记；η 扫描佐证 range_exc 非 v8 池独有） | `remote_inbox/20260711_stage2b` + `20260717_v8_champion` | 054/071 |
| fig48_sigma_line_narrative | σ 线三幕：EP09 三路发散 FAIL → E3 bench PASS 4.1% → 真实 0/8 边拒绝 → 鲁棒带 [0.1,0.4]px | `data/sigma_line.csv` + `output/sigma_esf_bench_reverdict/` | 056-059 |
| fig90_project_timeline | 全项目编年史：ACL 进度轴 × 六阶段（黑色阶段标题 + 各阶段实验摘要，颜色只在主题彩点；79 条解析、里程碑加星、ACL-073 缺席脚注） | `research_log/algorithm_changelog.md`（live 解析） | 001-080 |
| fig92_criteria_pipeline | 判据管线流程图：强制步骤 + 两条禁用捷径（self-FRC、跳过偏移校正）+ 20µm 警示框 | 纯示意（速览块 #4） | 047/048/049 |
| fig93_champion_axes | 冠军候选四轴平行坐标（v9-3k 的 OOD 盲区如实留白） | `data/champion_axes.csv` | 071/072/074/076 |
| fig26_dewaffle_x0 | de-waffle 暖启动：2px 棋盘伪影 + FFT 谱（合成演示，真实 drizzle 谱佐证；grid score 0.404→0.000） | `remote_inbox/20260716_v8_verdict/drizzle_a.npy` + 合成景 | 032 |
| fig35_l1_detectability_corner | L1 审计角落热图：v7 55%→v8 pilot 22%（经验 CNR<3 比例，数据本次从 5090 拉回） | `remote_inbox/20260713_l1audit/` | 068/070 |
| fig42_composer_defect_showcase | v7 缺陷体系五族画廊（现跑生成器采样，青圈标注目标缺陷） | `scripts/v7_composer_demo.py`（live） | 065 |
| fig94_multisplit_ranking | 多切分判决双面板：(a) tgv>v6>v9 代排序在 seed 42/123/456 全部成立；(b) 三切分完整 cross-FRC 曲线族（fig06 轴约定），~24µm 以上曲线级紧致 | `remote_inbox/20260713_content2ms/ms_verdict.csv` + `ms_curves/lb_seed*/frc_curves_long.csv` | 077 |
| fig95_content2_verdict | 语法外 content 轴 + portable 基线诚实修正：v6 对 oracle 4/4 胜、对 portable 打平 | `remote_inbox/20260713_content2ms/ood2_degradation_summary.csv` | 078 |
| fig99_fidelity_ood_tradeoff | 点保真 ↔ OOD 稳健性权衡（ACL-079 头条）：(a) retention × 13 池平均 ΔFRC(vs oracle) 单调权衡散点，v6 均衡冠军 13/13 / 9bin 两轴皆非 2/13 / 3k 保真冠军但 OOD 全输 0/13；(b) 逐池 ΔFRC 佐证（3k 13/13 垫底，v6>9bin 12/13）；胜场现场重算 fail-loud | `data/champion_arms.csv` + `remote_inbox/20260712_oodC/` + `20260713_content2ms/` 两份 degradation summary | 074/076/078/079 |
| fig96_split_visual_consistency | 重建对切分 seed 视觉不变：臂×seed 42/123/456 匹配高通面板（远端渲染于5090，脚本归档为 `*_REMOTE.py`，DejaVu serif 偏离 pubfig_style） | 5090 `output/stage2p5_multisplit_v2`（+ seed42 臂） | 077 |

### 真实图像 / 多模态可视化 (fig60-68)

| 图 | 内容 | 数据源 | ACL |
|---|---|---|---|
| fig60_optical_registration | 光学显微真值配准到 HR 热网格（θ=225.2°，NCC 0.985）；走线等高线叠加 4 臂验几何一致性 | `remote_inbox/20260713_dotprobe/optical_warp_hr.npy` + `20260710_expab/*_a*.npy` | 071/076 时代 |
| fig61_dc_residual_maps | DC 残差自审计实际看到的：抹除点在留出残差图中以局部凸起显形（最清晰正例，非典型；AUC 0.68-0.84 见 fig04） | `output/dc_residual_confidence/` | 074/075 |
| fig62_erasure_saga_visual | 抹点传奇视觉版：6 个孤立点跨池代追踪，v7 抹掉（43%）→ v8/v9/v9-3k 逐代找回（4.3%/1.6%/0.0%） | `remote_inbox` 三批重建 + `output/dot_probe_v7/intermediate/per_dot_v22_arms.csv` | 066/071/074 |
| fig63_splithalf_flat_band | 自分半 FRC 奖励什么：平坦区验收带纹理半幅相关 r（drizzle 0.15 vs TGV 0.84 / v6 0.70；v9-3k 0.11 噪声化） | `remote_inbox/20260710_expab` | fig31/32 语境 |
| fig64_halo_training_zoom | halo=96 全帧推理随训练步演化 2×3 蒙太奇（5k/20k/40k 温度 + TB 高通细节，存档 JPG 原样） | `output/de_pb9_probe/*.jpg` | de_pb9 探针时代 |
| fig65_v7_horizon_visuals | v7 微型训练时长视觉演化：12k/16k/20k/24k vs drizzle/TGV，双行温度+匹配高通 | `remote_inbox/20260716_micro_horizon/*.npy` | ~060s |
| fig66_band_decomposition | FRC 增益住在哪：中心 ROI FFT 环带分解（25-40µm 验收带 + 20-25µm 次验收带；TGV 次带 rms 0.070 从未验证） | `remote_inbox/20260710_expab` | 049/059/071/072 |
| fig67_synth_scene_anatomy | 合成训练场景解剖：3 场景 × (HR 真值 T / LR 观测 / 覆盖掩膜 cov) 三列，温度列共享色标 | `outputs/v7_demo_minipool/scene_NNN.npz` | v7 池时代 |
| fig68_delta_vs_drizzle | 各臂相对 drizzle 改了哪里：中心 ROI 差值图，新增对比度锚定真实走线边缘；TGV 条纹最强+阶梯块状，v9-3k 最保守 | `remote_inbox/20260710_expab` | 071/076 |

注意（ACL-078 口径修正）：全部 13 个 OOD 池上 tgv_oracle 系统性弱于 tgv_portable（oracle 语义待复核）。fig02/fig03/fig93 的 "OOD 9/9" 均为对 oracle 锚的口径；对 portable 的更严边界见 fig95。

## 维护注意（给后续低阶模型）

1. 动任何标注/图例前先跑一遍再看 PNG，重点检查文字遮挡（本目录所有手调偏移都是为此）。
2. 数值一律来自脚本 docstring 里写明的数据文件；`data/*.csv` 里每行都有 ACL 引用，改数字先对 changelog。
3. 配色/marker 语义固定：TGV 黑、MAP-TV 棕、drizzle 橙、神经臂按池 v6 蓝 / v8 紫 / v9 红 / v9-3k 绿（见 `pubfig_style.METHOD_STYLE`），不要换。
4. 20µm 处的 FRC 值不可采信（探测器孔径零点，速览块 #3）；任何新 FRC 图都要保留该阴影区。
5. fig10 的显示变换（σ=10 HR px AC 耦合）是口径的一部分，见脚本注释与 visboard manifest 的 `real_highpass_note`。
