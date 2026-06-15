# 10. 写作交接文档（主文逐节要点 + Supplementary 材料清单）

> **本文档的角色**: 论文写作的交接总控。任何接手写作的智能体/协作者，按本文档即可知道
> 每一节写什么、数字从哪来、图表用哪张、还缺什么、什么不许写。
> **阅读顺序**: 先 `AGENTS.md`（项目规范）→ `zh/README.md`（叙事骨架索引 + 红队禁写清单 + 图表计划）→ 本文档
> → 对应节中文稿（`zh/01`–`zh/08_*.zh.md`，唯一事实来源）→ `09_figures_tables_assets.md`（图表资产）。
> **语言约定**: 论文正文与 supp 全英文；交接/总控文档中文。

---

## 0. 版式与定档决策（2026-06-11 确认）

- **主文按 8 页 AAAI 版式写**：7 页技术内容 + 第 8 页仅参考文献；模板用
  `paper/aaai/template/anonymous-submission-latex-2026.tex`（AAAI kit，定档后换对应年份 kit 仅是版式微调）。
- **投稿靶子保持弹性**：WACV 2027 R2（8/28）为节奏锚；AAAI 可投 2028 届。
  写作不绑定具体 venue，按「7 页紧凑 + 厚 supp」产出，定档时只调版式与匿名项。
- **Supplementary 形态**: 独立 technical appendix（AAAI/WACV 都接受），
  主文必须自包含——审稿人不读 supp 也能验证全部主张。

## 1. 写作工作流（接手者必读）

1. **草稿在 markdown，不直接写 LaTeX**：`docs/paper/02`–`08` 是单一事实来源；
   改论点/数字先改 markdown，LaTeX 迁移是最后一步机械转换。
2. **`paper/aaai/sections/*.tex` 当前是 EP01–06 旧叙事，迁移时整体重写**，
   不要在旧 tex 上修补；`main.tex` 的标题/摘要按 `zh/01_title_and_abstract.zh.md`（锁定标题）重写。
3. **数字核验铁律**：正文每个数字必须能指到仓库内权威文件
   （`09_figures_tables_assets.md` 与本文档逐节给出路径）。两套指标尺度
   （TB-scale vs EP11-harness-scale）绝不混在同一表/图；终稿 T1/T2 全部数字
   来自统一口径 harness 单次重跑（已完成：`output/ep11_unified_harness/`）。
4. **禁写边界具有否决权**：`zh/README.md` 末节「红队禁写清单（否决权）」是红队清单，
   任何修改后的段落与图注都要过一遍；冲突时删主张，不删边界。
5. **占位规范**：未落地的实验结论一律 `[pending ...]` 或 ⬜ 标注，
   不写预期值；V9A/V9C/V9D/V10/T1/T2/F5 已落地，不应再保留 pending 标签。
   引用用 `[REF: 主题]` 占位，迁 LaTeX 前统一补 `paper/aaai/refs.bib`。

## 2. 主文逐节交接（8 页预算）

> 各节草稿均已存在且数字已初核；下表「待办」即接手者的具体工作项。

### §1 Introduction — `zh/02_introduction.zh.md`（~1.0 页）

- **必须保留的论点结构**: 三问框架 Q1 信息存在 / Q2 信息送达 / Q3 信息忠实；
  C1–C4 贡献；bounded-claim 立场（contour-level 2x，非计量、非 4x）。
- **核心数字**: 20 µm 分辨率 vs 10 µm pitch；17.0 µm FRC cutoff；FWHM 114→100 µm；
  drift artifact 0.37→0.65 同时 forward loss 贴底。
- **待办**: ① 按最终 harness 裁决精炼 Q2/Q3（input evidence 可见，但不构成 learned fidelity win）；
  ② 插入 teaser 图 F0 引用；③ 压缩到 1 页（当前略超）。

### §2 Related Work — `zh/03_related_work.zh.md`（~0.5 页）

- **五桶**: 经典 MFSR / deep burst SR / 热成像 SR / sim-to-real 退化建模 /
  无 GT 评估（FRC、split-half）+ range/null-space 分解与 data consistency。
- **每桶落点**: 一句对比——「无 GT 工业场景下没人量化 hallucination，我们量化并给出机制」。
- **待办**: ① 全部 `[REF:]` 占位补成真实文献（每桶 3–6 篇，共 ~25–35 条）；
  ② 与 §5/§6 的方法名对齐（FRC 引电镜社区原始文献；null-space 引 range-null 分解与
  deep null-space learning 线）。**这是当前唯一可完全并行外包的纯文献任务。**

### §3 Problem & Calibrated Forward Model — `zh/04_problem_forward_model.zh.md`（~1.0 页 + F1）

- **必写**: 仪器与数据（3.1）；raster/session 结构与 248 clean set（3.2）；
  标定链 θ/PSF 仲裁/噪声（3.3，PSF 三路分歧本身是 finding）；观测算子 y_k = D·B·H·S x + n（3.4，
  null space 在此埋伏笔）；claim boundary（3.5）。
- **数字来源**: `configs/stage_calibration.json`；`output/ep03_theoretical_limits/`（pitch、MTF/SNR）；
  `output/ep09_psf_calibration/` + `output/ep15_info_limit/m3_sigma/`（σ 仲裁）；
  `paper/reports/ep01_data_processing/audit_report.md`（session/噪声）。
- **待办**: ① F1 落地后核对图文一致；② EP04 gate 细节表移 supp B.2，正文留 2–3 句；
  ③ 压页（当前材料最厚，目标 1.0 页）。

### §4 Method — `zh/05_method.zh.md`（~1.25 页）

- **必写**: 经典基准三件套（drizzle / MAP-TV 验收基准 / 各向异性 coverage-weighted TGV 及其
  条纹修复 3.870→0.695）；TCForge 物理匹配合成（一段）；学习方法族：固定 UNet 骨干 +
  input 模式 {1x stats, hybrid drizzle} × anchor {none, band, full, legal}——本节结构
  必须让 §6 的 T2 矩阵「直接读出来」；checkpoint 选择规则（方法的一部分，非事后）。
- **待办**: ① 将 V9A/V9C/V10 已落地的 hybrid 输入、legal anchor 1x patch 旁路、
  residual-over-ch5 参数化压缩成正文可读版本；② 加 loss 公式块与选择规则 5 行伪代码（公式细节在 supp C，正文只放
  总式）；③ 损失权重演化史压成一段动机叙述。

### §5 GT-free Evaluation Protocol — `zh/06_evaluation_protocol.zh.md`（~1.0 页 + F2）

- **必写**: 四层协议——FRC（5.1，含控制组失效的诚实披露与 17.0 µm 唯一主张）；
  proxy 对及其**构造性反相关**（5.2，三条推论：温度计/不可跨输入比/看轨迹不看端点）；
  null-space 两曲线诊断（5.3，floor + drift ⇒ 漂移在零空间）；结构指标与双域视觉 gate（5.4）；
  adoption rule（5.5）。
- **数字来源**: `output/ep15_info_limit/m2_frc/`（cutoff/band/控制组）；
  漂移实测数字与 §5.2 共享。
- **待办**: ① δx = δx_range + δx_null 记号与 §3.4 算子统一（正式化推导放 supp A.2，
  正文留陈述 + 引用）；② null-space 投影直接测量（A†A 应用于 checkpoint 差）是 stretch
  goal——若 supp A.2 做出来，正文 5.3 升级一句。

### §6 Experiments — `zh/07_experiments.zh.md`（~1.9 页 + F3/F4/F5 + T1/T2）

- **小节顺序（2026-06-15 重组）**: 5.1 无赢家裁决（T1+F5，含选点一句+第二 ROI 一句）→
  5.2 核心发现：漂移活在零空间（F3，**全文核心**）→
  5.3 消融：输入通路与观测锚定（T2）→ 5.4 帧数预算、鲁棒性与负结果（各一句结论 + 引 supp）。
  旧 §5.6 选点协议实战 → 收入 T1 表注⑦ + 附录 C.5；旧 §5.8 第二 ROI → 收入 §5.1 F5 段尾。
- **当前状态**: V9A/V9C/V9D/V10 选点已过统一 harness；EP16 经典方法数字已回填；
  T1/T2/F5 单口径产物已在 `output/ep11_unified_harness/` 与 `output/paper_figures/`。
  后续是写作压缩、图注与红队边界检查，不再需要重复跑 Task D。
- **叙事预案**: V9A 阳性 → 「证据注入恢复细线」为 Q2 答案；V9A 阴性 → 降级为
  「输入与锚定双阴性 + 协议仍然成立」，C2/C3 不受影响（预案已写入 `00_status_and_plan.md`）。

### §7 Limitations + Conclusion — `zh/08_limitations_conclusion.zh.md`（~0.4 页）

- **已成稿**，待办仅两项：① V9 系列落地后调整结论 (ii)/(iii) 措辞；② 压缩 Limitations
  到 5 条以内（单仪器、无外部 HR 锚、FRC 控制组瑕疵、proxy 天花板、最细结构物理上限——
  PSF 不确定度并入第五条）。

### 摘要与标题

- 标题与摘要以 `zh/01_title_and_abstract.zh.md` 为准（标题已锁定为
  *Detailed but Drifting: Null-Space Hallucination ...*；旧 "Trust but Verify" 候选已废弃）；
  摘要中英双版已对齐，定档 venue 后按版式压缩（≤200 词，AAAI 摘要不设小节）。

## 3. Supplementary 材料清单（technical appendix）

> 草稿落盘位置：`docs/paper/supp/` 下按块建文件（`A_theory.md`、`B_system_data.md`、
> `C_method_details.md`、`D_full_results.md`、`E_reproducibility.md`）。
> **2026-06-12 更新**：五份草稿已全部落盘，按用户决策先以**中文**成稿（迁 LaTeX 时译英），
> 缺口在文内以「⬜ 待回填（等 XX）」占位；新增 D.0（fine-window 诊断口径）与
> D.7（零训练融合 baseline 专节，决策：supp 专节 + 主文 §6 一句话引用）。
> 标 ★ 的条目**不依赖任何未完成实验**。

### A. 理论与推导（A_theory.md）

| 条目 | 内容与验收标准 | 素材来源 | 状态 |
|---|---|---|---|
| ★A.1 MTF/SNR 可行性 | Gaussian PSF MTF(f)=exp(−2π²σ²f²) × 10 µm box sinc 的闭式推导；2x/4x Nyquist 数值表（σ=0.2/0.35/0.5 → 0.454/0.089/0.007；4x ≤0.042）；有效 SNR=ΔT·MTF/noise 判据 → 2x 条件可行、4x 出界；与 EP12 4x 失败互证。验收：表格数值与 `04` §3.3 一致 | `output/ep03_theoretical_limits/`、`paper/reports/ep03_*/theoretical_limits_report.md` | ✍️ 已起草（06-12 中文稿） |
| ★A.2 观测算子与零空间 | A=D·B·H·S 的离散定义；零空间刻画（B·H 的带限衰减 + D 的混叠折叠）；**Proposition 1**: A·δx_null=0 ⇒ 任意观测域 loss 对 δx_null 恒盲（证明 3 行）；两曲线诊断的充分条件陈述（floor + 持续 proxy 漂移 ⇒ 漂移含零空间分量）；可选：A†A 投影直接测量 checkpoint 差的 range/null 分解（GPU-light，可在训练间隙跑） | `04` §3.4、`06` §5.3、`algos/ep06_sr_poc/src/common/forward_model.py` | ✍️ 已起草（06-12 中文稿） |
| ★A.3 proxy 反相关的构造性论证 | artifact score 与 raw-control corr 都是同一 highpass 残差的泛函：写出两者定义式，证明沿「边缘增亮/增宽」扰动方向一阶变化符号相反；推论：联合最大化不可行，只能作温度计与选点准则 | `06` §5.2、`algos/ep06_sr_poc/src/common/metrics.py`、EP11 harness `_pearson_finite` | ✍️ 已起草（06-12 中文稿） |
| ★A.4 FRC 方法学 | phase-stratified split-half 定义；1/7 与 half-bit 判据公式及出处；四个控制组（bicubic 正、shift-shuffle 负、acquisition-drift、zero-coverage 统计）的设计逻辑与各自失效模式；为何只主张 17.0 µm cutoff | `algos/ep15_info_limit/scripts/run_m2_frc.py`、`output/ep15_info_limit/m2_frc/` | ✍️ 已起草（06-12 中文稿） |
| ★A.5 标定不确定度传播 | θ=47.6°±0.1° 经 coordinate_to_shift 的位移误差传播（40 µm command 处 ~0.007 px，给闭式）；PSF σ∈[0.2,0.5] 区间对 MTF/反卷积强度的影响范围；AVI 独立方向验证（θ≈47.14°，CI 覆盖）作为 consistency check 的角色与边界 | `configs/stage_calibration.json`、`core/src/thermal_core/displacement.py`、`paper/reports/ep02_*/calibration_report.md` | ✍️ 已起草（06-12 中文稿） |

### B. 系统与数据细节（B_system_data.md）

| 条目 | 内容 | 素材 | 状态 |
|---|---|---|---|
| ★B.1 数据审计 | 263→255→248 链条；3 温度段与跨 session 边界跳变（audit 权威口径**中位 2.91 / 最大 4.16 °C**；主文/AGENTS 的 3.55 为旧口径待修，见 supp B.1.2 警告框）；命名解码规则与 acquisition_order 教训；AVI/BMP 排除依据（8-bit、67% 重复） | `paper/reports/ep01_data_processing/audit_report.md` | ✍️ 已起草（06-12 中文稿） |
| ★B.2 对齐管线与 gate | highpass NCC 初始化 + contour refinement 细节；Chamfer 0.381→0.240→0.134 px；EP04 84+390 锚点 × 13 扫描线分级表（alignment-input / holdout / target-not-truth 三角色）；相位 bin 保持性 | `paper/reports/ep04_*/validation_report.md`、`paper/reports/ep05_*/displacement_reassessment.md`、`configs/alignment/` | ✍️ 已起草（06-12 中文稿） |
| B.3 微扫描运动学 | raster 各向异性（行内 gap 1 vs 行间 ~16）的热漂移含义；Y-only 标定失效教训（为何 stage 不是 truth 的实证） | `AGENTS.md` 硬教训 12、EP02 报告 | ✍️ 已起草（06-12 中文稿） |

### C. 方法细节（C_method_details.md）

| 条目 | 内容 | 素材 | 状态 |
|---|---|---|---|
| ★C.1 TCForge 合成平台全参数 | 场景几何分布、4×SSAA coverage AA、温度渲染、PSF/box/noise/shift 重放、burst pool K=4 变体表 | `tcforge/src/`、`05` §4.2、训练 pool 构建脚本 | ✍️ 已起草（06-12 中文稿） |
| ★C.2 网络与损失 | UNet 结构表；损失全公式（MSE/highpass/SSIM/grad-vector/edge + thin/gap 权重）与 hot vs conservative 两组权重对照；PixelShuffle 负结果细节 | `algos/ep07_unet_sr/src/.../losses.py`、各 run `config.json` | ✍️ 已起草（06-12 中文稿） |
| C.3 训练 config 对照表 | v6/v8.1a/v8.1b/v9b/v9a/v9c/v9d/V10 全字段差异表（input_mode/in_channels/anchor/band/pool/save_every/residual_mode…） | 各 `outputs/*/config.json` + `output/ep11_unified_harness/run_manifest.json` | ✍️ 已起草，V9C/V10 结果已可回填 |
| ★C.4 经典方法实现细节 | drizzle pixfrac/kernel；MAP-TV FISTA + σ,λ 选择；TGV 各向异性椭圆对偶球投影 + coverage 归一化推导与伪代码 | `algos/ep10_drizzle/`、`algos/ep10_tgv_sr/src/`、`algos/ep15_info_limit/scripts/run_m4_*.py` | ✍️ 已起草（06-12 中文稿） |
| ★C.5 checkpoint 选择伪代码 | 归一化 proxy 对 → 理想点距离 top-3（≥5K 间隔）→ 末端对照 → 视觉 gate；完整伪代码 + 超参 | `algos/ep07_unet_sr/scripts/plot_checkpoint_selection.py` | ✍️ 已起草（06-12 中文稿） |

### D. 完整实验结果（D_full_results.md）

| 条目 | 内容 | 素材 | 状态 |
|---|---|---|---|
| D.0 fine-window 诊断口径（新增） | 中心细线窗口定义 + 四指标（hp_corr_input/tgv、sharp_p95、lattice）+ 限制声明 | `algos/ep07_unet_sr/scripts/v9_review/common.py` | ✍️ 已起草（06-12 中文稿） |
| D.1 T1/T2 扩展版 | 全 selected methods × 全列；TGV/MAP-TV 全参数网格表 | `output/ep11_unified_harness/{all_arm_metrics.csv,t1_metrics.csv,t2_metrics.csv}` + `output/ep10_tgv_sr/sweep_results.csv` | ✅ harness 已完成，文字已回填 |
| D.2 FRC 全档案 | band×seed 全表、控制组完整曲线、MAP-TV 前后 split-half 对照 | `output/ep15_info_limit/m2_frc/`、`m4_deconv_anchor/` | ✍️ 已起草（06-12 中文稿） |
| D.3 漂移视觉演化 | V9 全系数值轨迹表 + 各变体 step 序列 eval_real 选帧 + v9a companion 轨迹 | `output/ep07_v9_review/`、`algos/ep07_unet_sr/outputs/*/eval_real/`、Task B `fig03s` | ✍️ 已起草，V9C 60K 行已可用 |
| D.4 负结果档案 | PixelShuffle 条纹证据、4x 网络失败 + MTF 界互证、loss-side anchoring 无效（v9b+v9d+v9c 已闭合）、AVI 排除 | EP11/EP12 输出、`research_log/algorithm_changelog.md` | ✍️ 已起草，V9C 判定句已可用 |
| D.5 budget/robustness 全曲线 | E1/E2/E3 全部曲线与表（经典方法 + 后补 GPU 方法） | Task C `output/ep16_budget_robustness/` | ✍️ 已起草（经典方法全表已抄录聚合；GPU 方法待跑） |
| D.6 视觉 gate panel 全集 | 四个变体（+V9 系列）panel 图 | `checkpoint_selection/panel_*.png` | ✍️ 已起草（选编原则已定；V9 panel 待选点） |
| D.7 零训练融合 baseline（新增） | fused(λ) 扫描全表 + 支配 TGV 结论 + V10 对照前沿定位 + selection-on-test caveat | `output/ep07_v9_review/fusion_*.{csv,md,png}` + V10 high-λ summary | ✍️ 已起草，V10 λ=1.2@15K 判定句已可回填；第二验证窗仍可选 |

### E. 复现包（E_reproducibility.md）

| 条目 | 内容 | 状态 |
|---|---|---|
| ★E.1 代码与环境 | 仓库结构、各 algo venv（UV/conda）矩阵、core 共享层、`build_all_notebooks.py` 一键重建 | ✍️ 已起草（06-12 中文稿，蓝本 `AGENTS.md`） |
| ★E.2 实验命令清单 | 每个图/表 → 生成脚本 + CLI + venv + 预计耗时 | ✍️ 已起草；T1/T2/F5 harness 命令已回填 |
| E.3 数据声明 | 实验室样品展示说明 + 原始数据不公开 + TCForge/协议可复现 | ✍️ 已起草（实验室样品，全幅 FOV 可用） |

## 4. 任务分派状态（2026-06-11 夜）

| 线 | 任务 | 执行者 | 状态 |
|---|---|---|---|
| 写作基建 | 本文档 + `09_figures_tables_assets.md` 落盘 | 主线 | ✅ 本次 |
| 图表 | F1 / F3 / F7 经典方法 → `todos/paper_prompts.md` Task A/B/C | Codex 并行 | ✅ 已完成 |
| 写作 | Supp ★ 条目起草（A.1–A.5 优先，其次 B/C/E） | 主线（06-12） | ✅ A–E 五份中文草稿落盘 `docs/paper/supp/`，缺口 ⬜ 占位 |
| 写作 | §2 文献补全（`[REF:]` → refs.bib） | 任何写作智能体 | ⬜ 即可领取 |
| 实验 | V9A/V9C/V10 视觉 gate + 选点 | 主线/Codex | ✅ 已入统一 harness |
| 实验 | 统一 harness 全方法重跑 → T1/T2/F5 | Codex（Task D） | ✅ 2026-06-14 完成；`output/ep11_unified_harness/` |
| 写作 | 正文全 7 章中文打磨稿 → `docs/paper/zh/` | 主线（06-14） | ✅ §1–§7 + 标题/摘要中文稿落盘；进修后译英回填 `02–08` |
| 实验 | 证据硬化：TGV 真值 split/FRC + F5 第二 ROI + D.7 第二窗 | Codex（Task E，后台） | 🔄 提示词已落 `todos/paper_prompts.md` Task E（纯 CPU，不挡写作） |

## 5. 接手者快速自检清单

- [ ] 改动是否过了 `zh/README.md` 末节「红队禁写清单（否决权）」？
- [ ] 新增数字是否给出仓库内出处路径？尺度口径是否标注（TB vs harness）？
- [ ] [pending] 占位是否如实保留（而非提前写成结论）？
- [ ] 图表是否登记/更新到 `09_figures_tables_assets.md`？
- [ ] supp 草稿是否落在 `docs/paper/supp/` 并在本文档表格更新状态？
