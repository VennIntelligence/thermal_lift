# Thermal Lift 项目收官总报告

> 日期: 2026-07-24
> 范围: 覆盖 `paper/reports/` 尚无正式报告的 EP10–EP16 各 Episode 与 solver 主线收官（champion 判决），并给出全项目最终结论与材料索引。
> 边界: 本报告是**汇总与溯源**，不含新实验、不改算法。所有定量结论均注明出处（ACL 编号 / GALLERY 图号 / episode README）；找不到确切数字处使用定性表述并说明原因。
> 事实源权威度排序: `research_log/algorithm_changelog.md`（ACL-001–080） > `docs/publication_figures/GALLERY.md` > `research_log/episodes/ep*/README.md` > 本报告。冲突时以 changelog「当前有效结论速览」块为准。

---

## 1. 执行摘要

**任务**：在 20 µm 采样 pitch / 20 µm 空间分辨率的 LWIR 温度矩阵上，用主会话 248 帧微扫描序列验证 2x 轮廓级超分辨率 POC（输出 10 µm/sample 网格），面向工业芯片检测的结构/形状轮廓增强，而非计量级温度读数（GALLERY「项目一页纸」）。

**最终交付**：

1. **均衡神经冠军 depb9v6**——物理约束展开求解器（unrolled solver）+ TCForge v6 合成训练池 + 9-phase-bin drizzle 输入，真实域 corrected cross-FRC@30µm = 0.6611、跨 3 个独立切分排序稳定（ACL-074/077，fig02/fig94）；13 个 OOD 池对 tgv_oracle 锚零符号翻转全胜（均值 Δ+0.171，ACL-076/078/079，fig03/fig99）；低频干净（range_excursion 2–2.7 健康带，ACL-074，fig17）。
2. **权威可恢复频带 25.45 ± 0.73 µm**——修复 ~0.31 px 逐帧对齐误差后从 34.07 µm 改善而来（ACL-047/048，fig21）；真实增益带为 25–40 µm 周期，20 µm 为探测器孔径零点、该处 FRC 一律不采信（changelog 速览块 #3）。
3. **一套诚实的评测方法论**——自分半 FRC 对确定性神经网络无效（奖励可复现幻觉，ACL-047，fig32）；+0.5 HR px 网格角点约定曾系统性压低所有神经×经典对比 ~0.44–0.49 FRC 点，校正后「神经带内破坏」翻案（ACL-049，fig07）；判据管线 = 逐半偏移校正 + 对独立 drizzle 的 cross-FRC（fig92）。
4. **点保真攻坚全链条**——孤立点抹除率 43%（v7 池病理）→ 4.35%（v8）→ 1.55%（v9）→ 0.00%（v9-3k）逐代收复（ACL-066/070/072/074，fig01/fig62），并产出零训练的部署侧「DC 残差自审计」仪表（被抹点在留出残差中可检出，AUC 0.68–0.84，ACL-075，fig04）。

**POC 最终判决：有条件成功**。

- **成立的部分**：2x 轮廓级增益真实存在——FRC 增益集中在 25–40 µm 验收带（fig66），新增对比度锚定在真实走线边缘而非漂浮纹理（fig68），与光学显微真值配准几何一致（NCC 0.985，fig60）；2x 是当前数据的合理倍率（实测相位占用 11/25 bin，>2x 相位饥饿，EP15-M1/ACL-048，fig34）。
- **边界**：精调经典基线 TGV（cross-FRC@30µm 0.7017 / cutoff 23.03 µm，ACL-049，fig06）在纯 FRC 轴上**始终未被神经臂超越**（健康神经臂天花板 ~0.67，历史最佳组合臂 0.6705，ACL-053 回填，fig91）；neural solver 的价值在**点保真可控性、OOD 稳健性、物理可审计性**三条 TGV 不具备的轴上（ACL-077 判读 2、ACL-079 帕累托定稿）。
- **权衡定律**：点保真与 OOD 稳健性沿训练池规模轴单调反相关——点保真冠军 v9-3k（抹除 0.00%）恰是域外最差臂（对 oracle 0/13 全输，Δ−0.275，ACL-079，fig99）。没有全域赢家，champion 选择是显式的多轴帕累托问题（fig02/fig93）。

**冠军决策矩阵**（转录自 GALLERY 第六章，口径：cross-FRC@30µm 对 drizzle、校正后、seed-42 切分；OOD 为 13 池对 tgv_oracle 胜场；数值源 ACL-074/076/077/078/079）：

| 臂 | cross-FRC@30µm | 孤立点抹除率 | OOD 胜场 | range_exc | 判决 |
|---|---|---|---|---|---|
| TGV（经典） | **0.7017** | 无点探针（参照臂） | 参照锚 | 1.6–4.4 健康 | 经典上界，纯分辨率选择 |
| **depb9v6** | 0.6611 | 4.66% | **13/13**（Δ+0.171） | 2–2.7 健康 | **均衡冠军** |
| depb9v9_9bin | 0.6252 | 1.55% | 2/13（Δ−0.136） | 12–16 异常 | 被 3k 弱支配，退出前沿 |
| depb9v9_3k | 0.6245 | **0.00%**（retention 0.798） | 0/13（Δ−0.275） | 12–16 异常 | 点保真冠军，OOD 最差 |


## 2. 主线叙事：solver 时代（ACL-023 → ACL-080，2026-06-25 → 07-13）

solver 主线此前无正式报告，本节按六个阶段给出可溯源的演进叙事（阶段划分同 GALLERY「项目阶段时间线」与 fig90）。UNet 时代（ACL-001–022）由 `paper/reports/ep07_*`、`ep11_dl_benchmark/` 既有报告覆盖，此处不重复。

### 2.1 立项与物理地基（ACL-023/024）

ACL-023 完成两件奠基工作：**探测器 pitch 重标定 10 → 20 µm/pixel**（旧值为 BMP 标尺 2× 误读；系统从「2× 过采样」修正为「临界采样」，直接解释了 4x 路线信息量不足、主线收敛到 2x），以及**前向算子认证**（T1–T5 自检全过，混叠能量占比 2.91%，fig16）。ACL-024 是决策记录：不上 diffusion、不用现成 RGB 底子微调，承诺物理约束 unrolled solver（神经 prox 先验 + 硬 DC 数据一致性交替迭代）——理由是工业计量要 data-consistent 恢复而非生成式幻觉，且 loss 侧 forward 锚定已被 ACL-017/019 证伪，需升级为架构侧硬约束。

### 2.2 架构攻坚（ACL-025–043）

一场教科书式的伪影根因链：K4 共享 solver 的 30 px 方块伪影来自 recurrent prox 残差累积（ACL-037，fig22）；频谱诊断锁定瓦片步长 32 px（ACL-038，fig25）；随机初始化因果实验证明根因是 **GroupNorm+SE 破坏范围不变性**（远场扰动下纯卷积响应 0、GN 0.673、GN+SE 1.412，ACL-040，fig24），促成 noSE+noGN 主线（ACL-041）+ halo=96 外圈推理缓解（峰突出度降约 16 倍，ACL-038，fig23/fig25）。同期修复 drizzle 暖启动的 2 px 棋盘 waffle（grid score 0.404→0.000，ACL-032，fig26）；D-E 高频残差 prox 七臂全阴性、如实归档（ACL-042/043，fig27）。solver 对 UNet 的效率优势：5k 步即锐利 vs UNet 需 15k+（fig12/fig05）。

### 2.3 评测仪器修复「stage0 saga」（ACL-044–050）

项目中期最重要的转折不是新模型，而是发现评测仪表失效（ACL-046 判 inconclusive）。三个连环修复：

1. **逐帧对齐误差 ~0.29–0.31 px 是头号信息瓶颈**，精修对齐后权威频带 34.07 → 25.45 ± 0.73 µm，精修对齐升级为 repo 默认资产（ACL-047/048，fig21）。
2. **自分半 FRC 对神经方法无效**：确定性网络两个半幅复现同样幻觉，自分半 0.92+ 是膨胀值；诚实差距拆为 ~0.4（幻觉）+ ~0.4（配准）两段（ACL-046/047，fig32/fig63）。
3. **+0.5 HR px 网格角点约定翻案**：神经对经典实测偏移 0.6–0.8 px，校正后 v11×TGV cross-FRC@30µm 0.04→0.83，「神经带内净破坏」旧结论作废（ACL-049，fig07）；校正入默认流程后各代残余偏移 ≤0.10 px（fig29）。

产出权威真实域排行榜：TGV 0.7017@30µm / cutoff 23.03 µm，为「要打败的对象」（ACL-049 口径，fig06/fig33）。

### 2.4 η 校准与平台期（ACL-051–053）

Stage 0k 消融显示便宜扩容路线（迭代深度/证据预算/推理期扩帧）全部平盘（ACL-051）；真正移动指针的是 **DC 权重 η 从历史默认 0.5 重校准到 η\*=0.09**，一举缩小约 35% 的 TGV 差距（ACL-052/053，fig14）。神经最佳纪录 = 20k 组合臂 0.6705@30µm（gap 0.031），但 50k/batch8 扩容反而回退（0.647）——单数据集调参就此终止，方向转向 Stage 2b 合成基准与训练池演化（ACL-053 回填，changelog 速览块 #9）。

### 2.5 点保真攻坚与训练池演化（ACL-054–074）

真实域 3562 点探针（ACL-063，fig36）量化出**所有神经臂都衰减小暗缺陷**、失效集中在「小 + 浅 + 孤立」角落（fig09/fig37）。v7 池灾难（孤立点抹除率 ~43%，ACL-066）触发系统归因：对照实验排除训练超参、锁定池缺陷分布（ACL-067）；L1 零训练审计证伪「点物理不可见」、定位病理角落 = 小半径×浅深度（ACL-068，fig35）；300 景微标定发现**抹除先验需要池规模级数据多样性才形成**（300 景全程 0% vs 5000 景 39.75%，ACL-069/072，fig08）。修复严格按归因执行：v8 收紧深度下限（抹除 43%→4.35%，ACL-070）、v9 恢复密度并实锤**浅深度是抹除元凶、密度无罪**（1.55%，ACL-072）、v9-3k 达 0.00%（ACL-074，fig01/fig62）。同期交付 σ 自校准线（点校准不可行 → σ∈[0.1, 0.4] px 鲁棒带，ACL-056–059，fig48/fig13）与 Stage 2b 合成基准 harness（ACL-054/060，fig11/fig45）。

### 2.6 收官判决（ACL-075–080）

- **DC 残差自审计成立**（ACL-075，fig04/fig46/fig61）：被抹除的点在 held-out 帧数据一致性残差中可检出（AUC 0.68–0.84），且检出能力随臂点保真度单调增强——硬 DC 把「先验覆盖数据」压进可观测面，构成零训练的部署侧自我怀疑仪表。
- **13 池 OOD 判决**（ACL-076/078，fig03/fig95）：depb9v6 对 tgv_oracle 锚 13/13 零符号翻转全胜；v9_9bin 大面积倒输；诚实边界——对更强的 tgv_portable 基线，v6 在语法外内容上收窄为打平。
- **多切分验证**（ACL-077，fig94）：排序 tgv > depb9v6 > v9 代在 seed 42/123/456 全部成立，champion 判决非单切分伪影；绝对值有 ~0.02–0.03 切分方差，报值应带 band。
- **权衡定律头条**（ACL-079，fig99）：3k 补测后确认点保真↔OOD 单调权衡；任何含稳健性权重的决策上 3k 被 depb9v6 支配。帕累托前沿定稿为 depb9v6（均衡）× tgv（纯分辨率）× depb9v9_3k（窄角落点保真）。
- **「最锐≠最优」实锤**（ACL-080，fig98）：owner 目视最干净的 v5_hybrid 复评 cross-FRC 仅 0.6256（神经最低档），其锐度大半是 ACL-029 时代的背景 waffle/幻觉——全项目诚实评测方法论最干净的反例演示。


## 3. EP10–EP16 逐节小结

以下 Episode 属 UNet 时代（2026-05 至 06 中旬），此前无正式报告。**重要口径警示**：EP12/EP15 的 µm 数字均产生于 ACL-023（2026-06-25）pitch 重标定**之前**的旧标定坐标系（当时误认 pitch=10 µm/pixel），episode README 从未回改；changelog 速览块 #3 明确「旧值 34 µm（坏对齐）与更早 EP15 数字作废」。引用时需按 2× 换算并叠加精修对齐修正。EP16 引用量为帧数/LR px，不受此换算影响，但其结果数据本身已不可得（见 §3.7）。

### 3.1 EP10 — 经典方法对比（drizzle / MAP-TV / TGV）

- **核心问题**：为 2x 高通域轮廓级 SR 选择经典重建方法，并对 MAP-TV 做 `lambda_tv` × PSF σ 的 28 点系统扫描。
- **方法**：复用 EP06 的 248 帧 clean 高通栈 + EP05 `contour_refined` 对齐；以 split-half NRMSE、留出前向残差、artifact score、raw-control 一致性四指标验证（EP10 README）。
- **结论与意义**：确立了 drizzle / MAP-TV / TGV 三算法对比框架与 proxy 指标体系；TGV/MAP-TV 由此成为贯穿全项目的「要打败的对象」（最终权威数值见 ACL-049 口径：TGV 0.7017@30µm，fig06）。4x drizzle 探索被明确定性为「轮廓过采样/展示支持，不是 2.5 µm 物理分辨率声明」（EP10 README 2026-05-21 条目）。
- **材料**：`research_log/episodes/ep10_method_comparison/README.md`、`notebooks/ep10_method_comparison/`、`output/ep10_map_tv_sweep/`。

### 3.2 EP11 — 学习类 SR 真实数据基准与 checkpoint 选优

- **核心问题**：EP07 UNet 与 EP10 TGV 在真实 248 帧上的同域高通视觉对比；四个 UNet 变体（v6/v8.1a/v8.1b/v9b）的 checkpoint 选优。
- **关键结论**：选优采用 artifact_score × raw_control_corr 理想点距离规则；四变体 canonical checkpoint 分别为 v6@8000、v8.1a@15000、v8.1b@5000、v9b@11000；两 proxy 沿「合成先验风格化」轴反向联动，只能取折中（`paper/reports/ep11_dl_benchmark/unet_checkpoint_selection.md`）。60k 终点一致劣化，印证「合成过训反吃真实细节」。
- **对主线的意义**：proxy 指标不可跨输入模式横比、且非光学 ground truth 的警告，是后来 stage0 仪器修复（§2.3）的伏笔。
- **材料**：`research_log/episodes/ep11_dl_benchmark/README.md`、既有报告 `paper/reports/ep11_dl_benchmark/`。

### 3.3 EP12 — 4x 基准与 4x 路线关闭

- **核心问题**：drizzle-informed 4x UNet 能否超越 bare drizzle 与 EP07 2x + 2 倍上采样（x2up）。
- **关键判决**（EP12 README 四臂 gate，2026-06-10）：EP12 48k 4x 在轮廓清晰度与中心 zigzag 线可分性上**无可见增益**；EP07 x2up 更锐但高通过冲更强；proxy 指标同样不支持 EP12。按预注册规则（三条 gate——FRC 频带、zigzag FWHM/dip、伪影/轮廓质量——任一不达即不采纳），**4x 分支不采纳**，交付回退为 EP07 2x + MAP-TV 后处理上采样。
- **对主线的意义**：「M1–M4 之后 4x 不再被视为可恢复 10–14 µm 信息的证据」（旧标定单位）；ACL-023 重标定后该判决获得第一性原理解释——数据本是临界采样，2x 即 20→10 µm。
- **材料**：`research_log/episodes/ep12_4x_benchmark/README.md`、`output/ep12_4x_benchmark/`。

### 3.4 EP13 — 2x 训练管线与 Loss Atlas（教学）

- **性质**：EP07 训练的可视化教程 notebook，非实验 Episode：TCForge 训练输入管线 + ContourSRLoss 逐项分解；demo 与 `configs/synthetic/training_pool_2x.json` 默认一致（248 帧/景、EP05 refined shifts、detector_realistic 噪声）。
- **意义**：把训练池生成旋钮（`n_frames_per_scene`、rotation/shift/noise/drift 分布）文档化，是后来 v6–v9 训练池代际演化（§2.5）的操作界面。
- **材料**：`research_log/episodes/ep13_loss_atlas/README.md`、`notebooks/ep13_loss_atlas/`。

### 3.5 EP14 — 4x Drizzle-informed Loss Atlas（教学）

- **性质**：EP12 训练的可视化教程 notebook：8 通道输入规格（ch0–2 为 4x drizzle mean/coverage/variance，ch3–7 为 1x 观测特征上采样）、双通道输出（pred + 异方差 log_var）、ThermalSR4xLoss 六项分解（LF/HF/Edge/前向一致性/NLL/逆覆盖 HF detail）。
- **意义**：完整存档了 4x 路线的技术设计；随 EP12 gate 判决与 ACL-023 重标定，该路线关闭，本 Episode 转为历史档案。其中「前向一致性 loss」思想后被 ACL-024 升级为 solver 的硬 DC 约束。
- **材料**：`research_log/episodes/ep14_4x_loss_atlas/README.md`、`notebooks/ep14_4x_loss_atlas/`。

### 3.6 EP15 — 信息上限实测与经典去卷积锚（M1–M4）

**注意：本节 µm 数字为旧标定（10 µm/pixel）单位，README 从未回改；M2 cutoff 17.03 µm 按新标定 2× 换算即 34.07 µm，与 ACL-047/048 时代重测的「坏对齐权威值 34.07 µm」数值一致，后经精修对齐修正为 25.45 µm（ACL-048，fig21）。**

- **M1 相位结构**（PASS WITH CAVEATS）：stage 坐标 5x 相位网格 25/25 覆盖，但 `contour_refined` 在 detector 轴 5x bin 塌缩到 11/25——高倍率相位饥饿的首个证据；该测量在新标定时代由 stage0f 重做并沿用（fig34，EP15-M1/ACL-048 口径）。
- **M2 FRC 信息截止**（负面/RISK）：phase-stratified split-half cutoff 17.03 µm（3 seeds std 0.50 µm，旧单位），不支持把 4x/5x 网格解释为真实物理信息；对照组异常（bicubic 阳性对照反而更低截止、shift-shuffle 阴性对照未崩到 0）被如实记为测量风险信号（EP15 README）。
- **M3 σ 仲裁**：外边框 apparent σ（median 1.015 LR px）明显宽于内部强边缘（0.747），说明 EP09 Route B 的 1.129 是「系统 PSF ⊗ 热边缘宽度」；FRC 形状拟合支持 σ≈0.2 LR px，M4 扫描区间定为 0.2–0.5 LR px（EP15 README）。这一「σ 无法点标定」的结论后被 solver 时代 σ 线（ACL-056–059，fig48）独立复现。
- **M4 去卷积锚**：MAP-TV（σ=0.2、λ=1e-3）建立后续网络必须超越的经典下限；zigzag median FWHM 114→100 µm（旧单位）、增益「有限且混合」，不包装为强阳性（EP15 README）。M4 在 2026-07-16 用新标定重跑并出版（fig51，EP07 臂为 v6 配方在 v9 池重训、原 checkpoint 已丢失，图注如实标注）。
- **对主线的意义**：EP15 确立了「先测信息上限、再谈倍率」的第一性原理纪律，其方法（split-half FRC、σ 仲裁、经典锚）全部被 solver 时代继承升级。
- **材料**：`research_log/episodes/ep15_info_limit/README.md`、`output/ep15_info_limit/`。

### 3.7 EP16 — 帧预算与鲁棒性（经典臂）

- **核心问题**：drizzle/TGV 在 E1 帧预算（N={31,62,124,248} 相位分层子集）、E2 对齐噪声（σ={0,0.05,0.1,0.2} LR px）、E3 对齐来源（command_prior vs contour_refined）三轴上的稳健性。
- **完成状态**：2026-06-11 完成经典 CPU 全量运行，37 个独立 run 全部成功（drizzle 20 + TGV 17；TGV 子进程累计约 353.7 分钟），产物含三张结果 CSV 与论文图 fig07_budget_robustness（EP16 README）。
- **数据现状警示**：结果 CSV（`output/ep16_budget_robustness/*.csv`）**本地、5090、git 三处皆不可得**（output/ 整体 ignore，数据服务器不可达）——出版图 fig50 因此阻塞（BACKLOG.md fig50 条目）。定量结论（各预算档的具体指标值）目前**不可复述**，仅方法与运行清单可溯源；恢复路径见 §5。
- **对主线的意义**：为论文 6.4–6.5 节提供经典臂的预算/鲁棒性包络；UNet 与 GPU MAP-TV 臂被有意留待 GPU 任务、后由 solver 时代评测体系取代。
- **材料**：`research_log/episodes/ep16_budget_robustness/README.md`、`output/ep16_budget_robustness/run_manifest.json`（若可恢复）。


## 4. OOD/鲁棒性与信息上限：方法的适用边界

综合 EP15/EP16 与 solver 时代 OOD 评测（ACL-076–079），本方法的适用边界可以表述为四条：

1. **频率边界**：可信增益仅存在于 25–40 µm 周期验收带（fig66）；权威可恢复截止 25.45 ± 0.73 µm（ACL-048）；20–25 µm 次带高于孔径零点但低于权威截止，其内容从不声称为已恢复；20 µm 探测器孔径零点处任何 FRC 不采信（changelog 速览块 #3）。
2. **倍率边界**：2x 是当前 248 帧数据的合理倍率——detector 轴实测相位占用 11/25 bin 且堆积角落（fig34），>2x 相位饥饿；这与 EP12 四臂 gate 的 4x 否决（§3.3）和 ACL-023 临界采样重标定三方互证。
3. **分布边界**：冠军 depb9v6 的 OOD 稳健性对「合成噪声轴 + 语法内容轴」稳固（13/13 对 oracle 锚，ACL-076/078），但对更强的 tgv_portable 基线在完全语法外的陌生几何上收窄为打平（净 1 胜 1 平 2 负，ACL-078 判读 2，fig95）——即神经增益的可迁移性有诚实上界。极端噪声（noise_amp_x4）下三方法共同塌陷至 0.17–0.31，是「没人赢」的方法失效区（ACL-076 判读 4）。
4. **池规模权衡**：训练池规模同时是 FRC↔点保真（ACL-074 判读 2）与点保真↔OOD（ACL-079 判读 2）两组张力的旋钮，且同向——小池弱先验买到极致点保真（v9-3k 0.00%），代价是最脆的域外行为（0/13，退化幅度约 9bin 的 2 倍）。部署选型必须显式声明轴权重，不存在免费的全能臂（fig99/fig02）。

**部署侧保障**：DC 残差自审计（ACL-075）提供零训练的「低置信区域」标注层——被先验抹除的结构在 held-out 帧残差中留下可检测痕迹（AUC 0.68–0.84，空间核验 0.887，fig46），且臂越诚实检出越强。诚实边界：σ 未标定（AUC 是占位算子下界）、类别来自探针而非人工 GT、单一真实 session。


## 5. 遗留问题与未来方向

### 5.1 未决技术问题（源自 changelog 开放项与 BACKLOG）

| # | 问题 | 现状 | 出处 |
|---|---|---|---|
| 1 | v8/v9 代 range_excursion 系统性偏高（12–16 vs 健康带 1.6–4.4），机制未明 | 已定位到配方层（seed 洗清），疑似深度下限 0.55，需 depth-floor 单变量小消融 | ACL-074 开放项，fig40 |
| 2 | tgv_oracle 系统性弱于 tgv_portable（全 13 OOD 池），oracle 条件的 PSF/参数语义待复核 | 不影响 champion 判决（双基线并记），但列为收尾前需澄清项 | ACL-078/079 开放项 |
| 3 | DC 残差自审计的 σ 未标定，AUC 为占位算子（σ=0.5）下界 | 若 σ 鲁棒带内重估，检出能力可能更高 | ACL-075 诚实边界 |
| 4 | 0d 回归指标套件仅 extent 探针有判别力（2.4–2.9×），阈值重定权未落定 | seam 探针方向反了，应退役或重设计 | ACL-047/速览块 #6，fig30 |
| 5 | v9_9bin OOD 劣化机制（先验强度 vs noise-conditioning）未做消融 | 已知不是 motif 词汇层面（legacymix 诊断点无特异性） | ACL-076 判读 3/开放项 |
| 6 | EP16 结果 CSV 三处（本地/5090/git）皆不可得，fig50 阻塞 | 待数据服务器恢复后查 `~/mycode/thermal_lift/output/ep16_budget_robustness/`；部分数字可从旧论文材料转录 | BACKLOG.md fig50 条目 |
| 7 | 其余旧 episode 出版图（fig55+）未逐个核对补画 | 按 BACKLOG E 系列清单执行 | BACKLOG.md |

### 5.2 未来方向

1. **生成式不确定度先验**：ACL-024 落盘的伏笔——在 unrolled solver 内以 plug-in 后验采样（DPS/ΠGDM 式）提供不确定度，而非替代确定性主干；与 DC 残差自审计（ACL-075）合流为完整的置信度输出层。
2. **新样本/多 session 实验**：现有全部结论建立在单一 248 帧真实会话上（ACL-075/077 反复标注的诚实边界）。新采集应优先：多 session 重复以量化会话间方差、按 fig34 结论设计更均匀的亚像素相位覆盖（解除 >2x 相位饥饿）、以及采集时同步光学显微真值（fig60 配准框架已就绪）。
3. **对齐精度继续挖潜**：0.31 px 逐帧误差修复带来了 34→25.45 µm 的最大单项收益（ACL-048）；残余对齐误差与 20–25 µm 次带的关系值得在新数据上复测。
4. **训练池配方精调**：在 v9 配方基础上做 depth-floor 单变量消融（解 range_exc 之谜），并探索池规模/多样性的中间点——在点保真与 OOD 之间寻找比 v6 更优的帕累托点（fig99 权衡线是否可被推移是开放科学问题）。
5. **交付工程化**：depb9v6 checkpoint + 精修对齐 + halo=96 全帧推理 + DC 残差置信图的端到端打包；新 checkpoint 上线前必过 Stage 2b 合成健康检查（v20 损坏 checkpoint 教训，fig11）。


## 6. 材料索引

### 6.1 本报告引用的 GALLERY 图（`docs/publication_figures/GALLERY.md`）

| 图号 | 内容 | 本报告引用处 |
|---|---|---|
| fig00 / fig90 | 四幕叙事总图 / ACL 编年史 | §2 阶段划分 |
| fig01 / fig62 | 训练池五代演化 / 抹点传奇视觉版 | §1、§2.5 |
| fig02 / fig93 / fig99 | 冠军帕累托前沿 / 四轴平行坐标 / 权衡定律头条 | §1、§2.6、§4 |
| fig03 / fig18 / fig44 / fig95 | OOD round-1 判决 / 逐场景分布 / 次级指标 / 语法外判决 | §2.6、§4 |
| fig04 / fig46 / fig47 / fig61 | DC 残差自审计系列 | §2.6、§4 |
| fig06 / fig33 | 真实域 cross-FRC 排行榜 / 频带表 | §1、§2.3 |
| fig07 / fig29 / fig32 / fig63 / fig92 | 评测方法论修复系列 | §1、§2.3 |
| fig08 / fig35 / fig36 / fig09 / fig37 / fig38 | 点保真归因系列 | §2.5 |
| fig11 / fig45 | Stage 2b 合成基准 | §2.5、§5.2 |
| fig12 / fig05 / fig22–26 / fig27 / fig28 | solver 架构攻坚系列 | §2.2 |
| fig13 / fig48 | σ 校准线 | §2.5、§3.6 |
| fig14 / fig17 | η 校准 / v21 收敛扫描 | §1、§2.4 |
| fig16 | 前向算子自检 | §2.1 |
| fig21 | 逐帧 shift 精修 | §1、§2.3、§5.2 |
| fig34 | 亚像素相位占用 | §1、§3.6、§4 |
| fig40 | range_exc 未决异常 | §5.1 |
| fig51 / fig52 | EP15 M4 重跑 / EP08 INR 对比 | §3.6 |
| fig60 / fig66 / fig68 / fig10 | 最终视觉成果系列 | §1、§4 |
| fig91 / fig94 / fig96 / fig98 / fig97 | 收官判决系列 | §1、§2.6 |
| fig30 | 0d 回归指标可分性 | §5.1 |

### 6.2 本报告引用的关键 ACL 条目（`research_log/algorithm_changelog.md`）

| ACL | 主题 |
|---|---|
| ACL-023/024 | pitch 重标定 + 前向算子认证；unrolled solver 立项决策 |
| ACL-032/037/038/040/041/042/043 | waffle 修复；K4 伪影根因链；noSE+noGN；halo；D-E 阴性 |
| ACL-046/047/048/049/050 | stage0 saga：仪表失效 → 精修对齐 → +0.5 px 翻案 → 对比层永久修正 |
| ACL-051/052/053 | 消融平盘；η 校准；冠军流程与 0.6705 纪录 |
| ACL-054–060 | Stage 2b harness；σ 自校准线收口 |
| ACL-063–074 | 点探针；v7 病理与归因矩阵；v8/v9/v9-3k 修复；champion 候选格局 |
| ACL-075–080 | DC 残差自审计；13 池 OOD；多切分验证；权衡定律；v5 复评 |

注：ACL-073（语法外 motif 族落地）无独立日志标题，内容并入 ACL-074 尾段并由 ACL-078 引用执行（详见 §7 矛盾清单）。

### 6.3 Episode README 与既有报告

| 材料 | 路径 |
|---|---|
| EP10–EP16 README | `research_log/episodes/ep{10_method_comparison,11_dl_benchmark,12_4x_benchmark,13_loss_atlas,14_4x_loss_atlas,15_info_limit,16_budget_robustness}/README.md` |
| 既有正式报告（EP01–EP09、EP11） | `paper/reports/ep0{1..9}_*/`、`paper/reports/ep11_dl_benchmark/`（本报告不重复其内容） |
| 出版图册与待画清单 | `docs/publication_figures/{GALLERY.md,README.md,BACKLOG.md}` |
| DC 残差分析结论文档 | `research_log/dc_residual_confidence_analysis.md` |

## 7. 事实矛盾与口径差异清单（撰写本报告时发现）

1. **EP15 README 数字未随 pitch 重标定回改**：M2 cutoff 17.03 µm、M4 FWHM 114→100 µm 等均为旧标定（10 µm/pixel）单位；changelog 速览块 #3 已明令作废（17.03 µm ≙ 新标定 34.07 µm，后经精修对齐修正为 25.45 µm）。引用 EP15 README 必须换算，本报告 §3.6 已标注。
2. **ACL-073 缺失独立标题**：changelog 中无 `[ACL-073]` 标题行（fig90 图注亦注明「ACL-073 无日志标题故缺席」），其内容（4 个语法外 motif 族落地）以附录形式挂在 ACL-074 条目尾部。编号连续性在形式上断裂，但内容可溯源。
3. **EP16 定量结果不可恢复**：episode README 只保留运行清单（37 runs all success），结果 CSV 在本地/5090/git 三处均不存在（BACKLOG fig50 阻塞条目）；本报告 §3.7 因此仅作定性描述，不转引任何 EP16 具体指标值。
4. **tgv_oracle 与 tgv_portable 的系统性差异未澄清**（ACL-078/079 开放项）：oracle 锚在全部 13 个 OOD 池上弱于 portable，语义待复核。champion 判决对两基线均成立（3k 对两者都 0/13），但「v6 OOD 全胜」的表述严格来说仅对 oracle 锚成立，对 portable 在语法外内容上是打平——GALLERY 决策矩阵口径注与 ACL-078 判读 2 一致，本报告沿用双基线并记。
5. **fig51 的 EP07 臂非原始 checkpoint**：原 ep07_v6_physics 2x checkpoint 在池迁移中丢失，出版图用 v6 配方在 v9 池重训替代（BACKLOG fig51 条目、GALLERY fig51 图注均如实标注）——引用该图时不应称其为「原 v6 checkpoint」。
6. **EP11 README 与正式报告分工**：episode README 仅记进度指针，定量内容全部在 `paper/reports/ep11_dl_benchmark/unet_checkpoint_selection.md`，两者无矛盾但信息密度差异大；本报告以正式报告为准。

