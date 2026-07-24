# Research Log — Episode 路线图与文档索引

> **项目一句话**: 面向工业芯片检测，在 20 µm 采样 pitch / 20 µm 空间分辨率的 LWIR 温度矩阵上，用主 session 248 帧微扫描序列验证 2x contour-level 超分辨率 POC（输出 10 µm/sample 网格）。
> **当前状态**: 收尾 / 可交接（2026-07）。全部 16 个 Episode 闭环，算法主线收敛于物理约束展开求解器（unrolled solver）；最终判决与出版级成果图见 `docs/publication_figures/GALLERY.md`（一页纸 + 72 图）。
> **本文件角色**: AGENTS.md 引用的 Episode 路线图。进度/指标/决策细节下放各 Episode README，本文件只做导览与索引。

---

## 一、项目定位与最终判决速览

- **技术路线**: 物理前向算子（PSF 模糊 + 2x 块降采样）+ unrolled solver（神经 prox 先验与数据一致性 DC 步交替）+ TCForge 程序化合成训练池；经典方法 drizzle / TGV / MAP-TV 作为独立参照与"要打败的对象"。（出处: GALLERY 一页纸）
- **权威可恢复频带**: 25.45 ± 0.73 µm（精修对齐修复 ~0.29 px 逐帧误差后，ACL-048）；20 µm 是探测器孔径零点，该处 FRC 数值不采信。
- **经典基线**: TGV cutoff 23.03 µm / cross-FRC@30µm 0.7017（ACL-049 校正口径）。
- **神经冠军**: depb9v6 —— cross-FRC@30µm 0.6611，13/13 个 OOD 池完胜 TGV(oracle)，低频干净、可审计（ACL-079 champion 帕累托定稿）。
- **权衡定律**: 点保真冠军 depb9v9_3k（孤立点抹除 0.00%）域外 0/13 全输——点保真↔OOD 稳健性沿训练池规模轴单调权衡（ACL-074/079）。
- **倍率口径**: 2x 是当前数据的合理倍率；实测相位占用 11/25 detector bin，>2x 相位饥饿（EP15-M1，GALLERY 头条 #9）。

---

## 二、Episode 路线图（EP01–EP16）

状态图例: ✅ 完成 · ✅⁻ 完成（负结果/门控未过） · 📚 历史文档（结论已被后续主线取代）

| EP | 名称 | 核心问题 | 一句话结论（出处） | 状态 | 链接 |
|---|---|---|---|---|---|
| EP01 | 数据处理 | 263 帧原始数据审计与主 session 建模 | 263 帧全部可读、TXT/BMP 完整配对；按 `acquisition_order` 检出 3 个温度段，主 session=2 共 255 帧，剔除 R≠0 后 **248 帧 clean set** 定为全项目 SR 默认输入（EP01 README） | ✅ | [日志](episodes/ep01_data_processing/) · [报告](../paper/reports/ep01_data_processing/) · [NB](../notebooks/ep01_data_processing/) |
| EP02 | 位移标定 | raster 路径 / stage prior / 数据驱动对齐分层 | θ=47.6° 保留为配置，AVI gradient 独立验证 47.14°（95% CI 覆盖）；stage command 只能作 prior，Y-only 坐标相邻帧不可做定量标定（EP02 README 证据表/决策记录） | ✅ | [日志](episodes/ep02_displacement_calibration/) · [报告](../paper/reports/ep02_displacement_calibration/) · [NB](../notebooks/ep02_displacement_calibration/) |
| EP03 | 理论极限 | MTF / SNR / CRB 物理边界 | 修正探测器 pitch 2× 标尺误读（10→**20 µm/px**）；MTF×SNR 风险图支持 2x 为合理默认网格、4x 仅探索；noise floor 0.0724 °C（EP03 README，含 SUPERSEDED 注记） | ✅ | [日志](episodes/ep03_theoretical_limits/) · [报告](../paper/reports/ep03_theoretical_limits/) · [NB](../notebooks/ep03_theoretical_limits/) |
| EP04 | 全局验证 | alignment anchor benchmark 与质量门控 | 外轮廓 segment 通过率 54.8% vs 内轮廓 22.3%；产出 EP06 三类门控角色：alignment_input / holdout_validation / sr_target_not_truth（EP04 README） | ✅ | [日志](episodes/ep04_global_validation/) · [报告](../paper/reports/ep04_global_validation/) · [NB](../notebooks/ep04_global_validation/) |
| EP05 | SR 再评估 | 248 帧 clean input 的 2x 相位/对齐基线 | 2x 相位 bin 4/4 占满（min 59 帧），contour refined held-out Chamfer median 0.133 px，具备进入 2x POC 的基础；4x 相位塌缩 4/16 是高倍率风险信号（EP05 README） | ✅ | [日志](episodes/ep05_sr_reassessment/) · [报告](../paper/reports/ep05_sr_reassessment/) · [NB](../notebooks/ep05_sr_reassessment/) |
| EP06 | 经典 2x SR POC | SAA / IBP / MAP-TV 双轨重建 | 经典 2x 全管线（highpass 主轨 + raw 控制轨）跑通；default contour refined 的 split-half NRMSE 最低（0.0217）；MAP-TV 选强正则 λ=0.01，是保守候选而非最锐（EP06 README） | ✅ | [日志](episodes/ep06_sr_poc/) · [报告](../paper/reports/ep06_sr_poc/) · [NB](../notebooks/ep06_sr_poc/) |
| EP07 | 多轨（见下节） | TCForge 合成引擎 + UNet→solver 算法主线 | 主线从 UNet（V4→V10）演化为 unrolled solver + v9 代训练池，最终收敛于 champion depb9v6（changelog ACL-024→080；详见「三、EP07 多轨说明」与「四、Solver 时代时间线」） | ✅ 主线 | [引擎](episodes/ep07_thermal_chip_phantom/) · [算法](../algos/ep07_unet_sr/) · [报告](../paper/reports/ep07_v9_attribution/) |
| EP08 | INR SR | SIREN/WIRE/DeepDecoder/DIP 深度先验对照 | 五方对照完成：SIREN 被推荐为 Stage 3 主方法，DIP 是强拟合风险参照；同协议下 EP06 MAP-TV 指标仍最优（hold-out 2.244 / split-half 0.174）；Stage 4 长训未启动，后被 solver 主线取代（EP08 README） | ✅（Stage 4 未启动） | [日志](episodes/ep08_inr_sr/) · [报告](../paper/reports/ep08_inr_sr/) · [NB](../notebooks/ep08_inr_sr/) |
| EP09 | PSF 标定 | σ 三路线精确标定 | 门控未通过：Route A/B/C = 0.226/1.129/0.119 LR px，spread≈1.01 px 超 ±0.05 px 门控；不以物理可行性启动 4x；σ 最终收敛为鲁棒带 0.1–0.4 px（EP09 README；ACL-059、GALLERY fig48） | ✅⁻ | [日志](episodes/ep09_psf_calibration/) · [报告](../paper/reports/ep09_psf_calibration/) · [NB](../notebooks/ep09_psf_calibration/) |
| EP10 | 方法对比 | Drizzle / MAP-TV / TGV 参数扫描与横评 | 三算法 2x 对比与 MAP-TV 28 点 sweep 完成；TGV 由此成为全项目"要打败的"经典基线（cutoff 23.03 µm / FRC@30µm 0.7017，ACL-049 校正口径）（EP10 README；changelog 速览 #7） | ✅ | [日志](episodes/ep10_method_comparison/) · [NB](../notebooks/ep10_method_comparison/) |
| EP11 | DL benchmark | UNet 各变体 vs 经典的真实数据横评 | 四臂 canonical checkpoint 选优完成，v9b@11k 为 proxy 折中最优但需目检确认；UNet-era proxy 口径后被 corrected cross-FRC 取代（EP11 README；`paper/reports/ep11_dl_benchmark/unet_checkpoint_selection.md`） | ✅ | [日志](episodes/ep11_dl_benchmark/) · [报告](../paper/reports/ep11_dl_benchmark/) · [NB](../notebooks/ep11_dl_benchmark/) |
| EP12 | 4x benchmark | 4x UNet vs drizzle 采纳 gate | EP12 48k 4x 相对 EP07 2x x2up 无可见增益，proxy 指标亦不支持；4x 分支未通过采纳门槛，交付 fallback = EP07 2x + MAP-TV 上采样（EP12 README） | ✅⁻ | [日志](episodes/ep12_4x_benchmark/) · [NB](../notebooks/ep12_4x_benchmark/) |
| EP13 | Loss atlas | 2x 训练管线 / ContourSRLoss 教学图集 | 教学图集完成（管线图 00-07 + loss 分解 08-16）；所记录的 ContourSRLoss 已被 ACL-027 重设计取代，保留为 UNet 时代历史教程（EP13 README；ACL-027） | 📚 | [日志](episodes/ep13_loss_atlas/) · [NB](../notebooks/ep13_loss_atlas/) |
| EP14 | 4x loss atlas | 4x drizzle-informed 管线 / ThermalSR4xLoss 教学 | 教学图集完成（8 通道输入 + 六项 loss 分解）；其服务的 4x 分支未获采纳，降级为历史教学材料（EP14 README；EP12 gate、EP15 M2） | 📚 | [日志](episodes/ep14_4x_loss_atlas/) · [NB](../notebooks/ep14_4x_loss_atlas/) |
| EP15 | 信息上限 | FRC 实测信息截止 + 经典去卷积锚 | M2 FRC 不支持 10-14 µm 真实信息（负结果）；M4 建立 MAP-TV 经典锚（σ=0.2/λ=1e-3）；⚠️ README 中 cutoff 17.03 µm 为旧对齐旧口径，权威频带后修正为 25.45±0.73 µm（EP15 README；ACL-048） | ✅⁻ | [日志](episodes/ep15_info_limit/) · 产物 `output/ep15_info_limit/` |
| EP16 | 预算稳健性 | 帧预算 / 位移噪声 / 对齐来源（经典 CPU 部分） | E1/E2/E3 共 37 个 drizzle/TGV run 全部成功，产物进论文 §6.4-6.5（fig07）；UNet / GPU MAP-TV 留待单独 GPU 任务（EP16 README） | ✅（经典部分） | [日志](episodes/ep16_budget_robustness/) · [NB](../notebooks/ep16_budget_robustness/) |

注：EP10/EP12/EP13/EP14/EP15/EP16 无 `paper/reports/` 正式报告；EP15 无 notebook 目录（产物在 `output/ep15_info_limit/`）。solver 时代的正式结论以 `algorithm_changelog.md` 和 GALLERY 为准。

---

## 三、EP07 多轨说明

EP07 是全项目算法主战场，历史上拆成多条轨道，目录名容易混淆：

| 位置 | 角色 | 说明 |
|---|---|---|
| `algos/ep07_unet_sr/` | **算法实现主目录** | UNet 时代（V4→V10）与 solver 时代（ACL-024 之后）的全部训练/推理/评测代码都在这里。目录名保留 "unet_sr" 是历史沿革——solver 主线并未另立目录 |
| `episodes/ep07_thermal_chip_phantom/` | TCForge 合成数据引擎 | 工程 episode：程序化 ThermalChipPhantom 生成器（独立 UV 项目 `tcforge/`）、forward 约定锁定、smoke/manifest 契约；不是真实数据 SR 主张（该 README）。对应 `paper/reports/ep07_thermal_chip_phantom/`、`notebooks/ep07_thermal_chip_phantom/` |
| `episodes/ep07_solver_boundary_artifact/` | solver 边界伪影诊断（2026-06-30→07-02） | 判定方形 "glow box" 伪影源自 patch-local prox 的边界响应：prox 制造、DC 抑制；`halo≥96 HR px` 可压制可见伪影；含 V11 noSE/noGN 中期读数与 ACL-042/043 负结果收官（该 README；对应 ACL-037→043） |
| `episodes/ep07_solver_v8_k4_fullhalo_eval_archive/` | V8/K4 评测证据存档（2026-06-30） | 删除 `outputs/` 前保留的 TensorBoard 证据：GroupNorm+SE 的 extent 分布漂移被第二代理诊断确认（远场扰动改变预测 ~1.4σ，纯卷积为 0），halo 是权衡旋钮不是根治（该 README） |
| `episodes/ep07_unet_sr_task1_audit.md` | 散页审计记录（2026-06-07） | EP07v2 Task 1 的环境准备 / TCForge 代码审计 / 基线测试（29 passed, 1 skipped），无功能改动；保留为过程记录 |
| `paper/reports/ep07_v9_attribution/` | 点保真归因报告 | v7→v9 训练池点保真攻坚的正式报告入口（对应 ACL-063→074） |

一句话关系：**TCForge（thermal_chip_phantom 轨）造数据，`algos/ep07_unet_sr/` 训练与评测，两个 solver 轨是架构攻坚期的诊断/证据存档，audit 散页是过程记录，v9_attribution 是点保真攻坚的报告出口。**

---

## 四、Solver 时代时间线（ACL-023 → ACL-080）

完整编年史见 `algorithm_changelog.md`（读旧条目前先读顶部「当前有效结论速览」）和 GALLERY fig90。六阶段划分（出处: GALLERY 一页纸）：

| 阶段 | 时间 | 主题 | ACL |
|---|---|---|---|
| I | 2026-06 上中旬 | UNet SR 基线与论文 harness（V4→V10 loss 实验、TCForge AA 池、TGV 基线修复、EP12/EP15 基准） | 001–022 |
| II | 06-25 → 07-01 | 像元重标定（20 µm）+ 展开 solver 立项与架构攻坚 | 023–043 |
| III | 07-02 → 07-06 | 评测仪器修复 "stage0 saga"（self-FRC 判无效、精修对齐、+0.5 px 翻案） | 044–050 |
| IV | 07-07 → 07-08 | η 校准、σ 自校准线、Stage 2b 合成基准、v7 生成器落地 | 051–065 |
| V | 07-08 → 07-10 | 点保真传奇与训练池演化（v7 病理 → 归因 → v8/v9 修复） | 063–074 |
| VI | 07-11 → 07-13 | 收官判决（DC 自审计、13 池 OOD、多切分验证、权衡定律、v5 复评） | 075–080 |

主线里程碑（均可在 changelog 对应条目核实）：

- **ACL-023/024（06-25）**: 探测器 pitch 重标定为 20 µm + forward 算子认证；决策不上 diffusion、承诺 unrolled solver——solver 时代开始。
- **ACL-026**: end-on-DC + 冻结 η，把"硬 DC"真正做硬。
- **ACL-037→041**: K4 方块伪影根因（recurrent prox 残差累积）→ 主线改 K2；real-eval 改 full-halo；默认 noSE + noGN + full_halo96（对应 ep07_solver 两条诊断轨）。
- **ACL-046→048（stage0 saga）**: self split-half FRC 对神经方法判无效（奖励可复现幻觉）；0a 半像素约定 bug 修复；精修 shift 升级为 repo 默认资产，权威可恢复频带 34.07 → **25.45 µm**。
- **ACL-049**: 评测方法论翻案——神经输出网格 +0.5 HR px 角点约定曾系统性压低所有神经×经典对比；校正后神经与经典同档（V11 cutoff 平 TGV 23.03 µm），"神经带内破坏"历史结论作废。
- **ACL-053**: 冠军臂 50k 负结果（synth↑/real↓ 第三次反相关）→ 单数据集调参终止，转 Stage 2b 合成基准。
- **ACL-056→059（σ 线）**: E1/E2 自监督估计器对 σ 原理性简并 → E3 多帧投影 ESF 内核 bench PASS（median 4.1%）→ 真实数据合法拒绝（0/8 边过门）→ σ 处理策略转为鲁棒带 0.1–0.4 px。
- **ACL-062→064**: v6 单变量拆解；depb9v6 成为双轴 champion 候选。
- **ACL-063→067（v7 病理）**: 真实小暗点保真探针坐实神经臂抹除缺陷点 → v7 池复训后孤立点 erased% 4.7%→~43%，对照实验锁定为池缺陷分布问题（非超参）。
- **ACL-070/072（修复与归因）**: v8 池修复成立（erased% 43%→4.35%）；v9 归因闭环——浅深度是点抹除元凶（实锤）、密度无罪反有益，depb9v9_9bin erased 1.55% 史上最优；v9 成为生产池。
- **ACL-074**: 3k 池臂点保真史上最优（isolated erased **0.00%** / retention 0.798），成为新帕累托点；池规格降 3K 不获批。
- **ACL-075**: 零训练 DC 残差"自我怀疑仪表"成立——被抹除的点在 held-out 数据一致性残差中可检出（AUC 0.68–0.84）。
- **ACL-076→079（OOD 判决）**: depb9v6 在 13 个 OOD 池对 tgv_oracle 零 sign-flip 完胜；champion 排名跨 3 切分一致；depb9v9_3k OOD 全输 0/13——点保真↔OOD 权衡定律闭环。
- **ACL-080**: "最锐≠最优"实锤——目视最锐的 v5_sharp 复评 cross-FRC 仅 0.6256（神经最低档），锐度大半是背景 waffle/幻觉。

**最终判决（ACL-079 champion 帕累托定稿）**：

- **depb9v6 = 均衡神经冠军**：真实 cross-FRC@30µm 0.6611（跨切分稳）+ 神经臂中唯一 OOD 稳健 + 可审计 + 点保真中庸（4.66%/0.598）。
- **TGV = 纯分辨率选择**：真实 FRC 0.7017 最高，但无点保真/可审计概念。
- **depb9v9_3k**：点保真轴帕累托最优且不可替代（0.00%/0.798），但 OOD 实测最差，仅适用于严格 in-distribution 场景。
- **depb9v9_9bin**：被 3k 弱支配，退出前沿。

> **ACL-073 编号说明**: changelog 中无以 `[ACL-073]` 开头的标题条目（GALLERY fig90 亦注明"无日志标题故缺席"），属历史跳号，**非内容丢失**——ACL-078 引用的"ACL-073 落地的语法外 motif 族"对应工作完整记录在 `ood_content_motif_families_plan.md` 与 `ood_content_motif_families_impl_notes.md`。

---

## 五、顶层设计草稿索引

`research_log/` 顶层散落的设计/协议/心得文档（多为 solver 时代的设计前置或过程记录，权威结论一律以 changelog ACL 条目为准）：

| 文件 | 一句话说明 | 关联 EP / ACL | 状态 |
|---|---|---|---|
| `ep01_ep04_rebuild_plan.md` | 按 2x 主线重定义 EP01–EP04 任务边界与产物的重建计划 | EP01–EP04 | 已执行 |
| `synthetic_data_realism.md` | 合成数据真实感调优心得（真实帧并排对比驱动，2026-06-26） | EP07 / TCForge realism 模块 | 心得记录 |
| `solver_v2_redesign_proposal.md` | solver 真·多帧融合重设计提案；顶部状态更新注明多处已被 Stage 0f/0g 实测推翻 | ACL-046→049 | 部分过时（以 ACL 为准） |
| `network_upgrade_roadmap.md` | post-v3 网络升级路线图；顶部 2026-07-05 状态更新逐条修订 | ACL-046→049 | 已被 ACL 修订 |
| `stage2a_perframe_fusion_design_draft.md` | Stage 2a 逐帧证据融合 + 鲁棒 DC 设计草案 | ACL-051 | DRAFT |
| `stage2b_synth_benchmark_design.md` | Stage 2b 合成基准设计（H1 域差 vs H2 架构判决实验） | ACL-053/054 | 已执行 |
| `sigma_selfcal_prereg_design.md` | σ 自校准预注册方案；E1/E2 估计器已被证伪，E3 换代内核接棒 | ACL-056→059 | 已执行（内核换代） |
| `sigma_knife_edge_calibration_protocol.md` | 刀口 σ 硬件标定采集协议 | ACL-057→060 | DRAFT（待排采集） |
| `lockin_acquisition_protocol.md` | 锁相调制采集协议——攻三堵墙中的 SNR 墙 | 信息预算文献 + ACL-048 | DRAFT（待 owner 评审） |
| `ood_generalization_suite_design.md` | OOD 泛化套件总设计（价值主张转向分布外泛化包络） | ACL-060、076 | 已执行 |
| `ood_generator_support_audit_draft.md` | OOD 套件前置的生成器支持面只读审计 | OOD 套件 §0 | DRAFT（已消化） |
| `ood_content_motif_families_plan.md` | 4 个语法外 motif 族（organic_blobs/text_serial/rings/voronoi）实施计划 | ACL-073（跳号）→078 | 已执行 |
| `ood_content_motif_families_impl_notes.md` | 上述 motif 族的实现偏离记录 | ACL-073（跳号）→078 | 已执行 |
| `v7_composer_defects_integration_plan.md` | v7 构图器 + 缺陷体系进 tcforge 实现方案 | ACL-065 | 已执行 |
| `v7_noise_operator_integration_plan.md` | v7 噪声真实感升级 + 算子随机化实现方案 | ACL-065 | 已执行 |
| `v8_pool_repair_plan.md` | v8 池修复三层验证阶梯（把验证从训练后搬到训练前） | ACL-068→070 | 已执行 |
| `dc_residual_confidence_analysis.md` | DC 残差置信分析——抹除点在残差中可检出（AUC 0.68–0.84） | ACL-075 | 正式分析（已完成） |

另有 `assets/`（日志配图）与 `literature/`（文献笔记，含信息预算分析 `2026_info_budget_and_why_phone_4x.md`）两个子目录。

---

## 六、新接手者阅读指引

按以下顺序读，可在半天内建立全貌：

1. **`AGENTS.md`（根目录）** — 项目规范、物理常数 ground truth、14 条硬教训。所有智能体必读。
2. **`docs/publication_figures/GALLERY.md` 的「项目一页纸」** — 最新的最终叙事：任务、路线、9 条头条成果、六阶段时间线、术语速查。
3. **本文件第二节路线图表** — 定位感兴趣的 Episode，跳转各 README。
4. **`algorithm_changelog.md` 顶部「当前有效结论速览」** — 有效结论的权威口径；**读任何旧 ACL 条目前必先读这里**（旧判断可能已被推翻）。
5. **EP01→EP05 README** — 数据与物理地基（帧审计、位移标定、理论边界、质量门控、对齐基线）。
6. **EP06 / EP10 / EP15 README** — 经典基线与信息上限（SAA/IBP/MAP-TV、TGV 基线、FRC 实测）。
7. **「三、EP07 多轨说明」+ changelog ACL-023 之后按「四、时间线」的里程碑抽读** — solver 主线的立项、攻坚、仪器修复、点保真传奇。
8. **ACL-075→080 + GALLERY 第六/七章** — 收官判决（champion 帕累托、OOD、权衡定律）与最终视觉成果。
9. **`paper/reports/`** — 需要引用正式数字时查对应 Episode 的报告。

两条硬口径（贯穿全部文档）：① 20 µm 空间周期是探测器孔径零点，任何 20 µm 处的 FRC 数值不采信；② 对神经方法只采信"对独立参照（drizzle）的 corrected cross-FRC"，自分半 FRC 无效（ACL-047/049）。
