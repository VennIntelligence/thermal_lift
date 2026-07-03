# Solver V2 重设计提案 — 真·多帧融合 + 求解器中间优化（待讨论）

> ⚠️ **2026-07-05 状态更新（先读这里；本提案多处已被 Stage 0f/0g 实测推翻或修订，权威记录见 ACL-046～048）**：
> 1. **头号杠杆已改判**：真实逐帧 shift 误差实测 ~0.29px（零均值环状分布），0a 精修反馈一轮收敛（残余 0.012/0.071px）；精修对齐已升级为 repo 默认资产（`configs/alignment/stage0f_refined_alignment.csv`），把 EP15 M2 权威可恢复频带从 34.07µm 推进到 **25.45µm**（负对照保持）。§1 的"DR 容忍算子误差"降级为次要手段（DR@0.1px 两臂实测 null）；**shift 校正/联合精化（原 2c）升为主线**。
> 2. **判据修订**：§3 的"real split-half FRC 为主 gate"对神经方法**不可用**——确定性网络在两半复现同一先验高频，self-FRC≈1 是幻觉签名不是分辨率。改用 **cross-method FRC**（`--cross-pair`）+ 同半幅对照 + M2 负对照，且神经×经典比较**必须先做逐半偏移校正**（solver 输出网格带 +0.5 HR px 角点约定，ACL-049；工具 `probe_pair_offset.py` / `real_eval.to_center_grid`）。0d 套件仅 extent 探针有好/坏判别力。
> 2b. **神经臂现状（ACL-049 v4，2026-07-05）**：校正配准后神经与经典**同档**——V11 cutoff 与 TGV 打平（23.03µm），@30µm 落后 0.05-0.06；此前一切"神经比经典差/糊/破坏信息"的结论均需按校正口径重读。当前要打败的基线：TGV×drz 0.702@30µm / cutoff 23.03µm。
> 3. **0a 打分器曾有半像素约定 bug**（forward block 中心 vs SAA 网格差 +0.5 HR px，已修 ACL-046）；本文引用的早期 0a 数字（~5-6%）与后来的 20.07% 均被污染，修正后诚实值 12.6%（iter1）→ 6.9%（iter2 对新对齐）。
> 4. 本文指出的"真实 PSF 占位 σ=0.5"仍是开放项：0a 类仪器测不了 σ（SAA x̂ 自带模糊）。
>
> 写于 2026-07-02。状态：**提案，待 owner 讨论定稿**。
> 背景：ACL-042/043 无人值守队列（D-E σ sweep + 9/16 phase-bin + 96ch 加宽）全部负结果；
> 队列结论写的是"撞到 info-budget 物理天花板，下一杠杆在采集"。本提案基于对全部
> research_log / 代码 / 错误案例 / 文献笔记的重新梳理，论证：**合成域确实到顶了，
> 但真实数据域我们撞的不是物理天花板，而是"模型误差天花板"——它是可修的。**

---

## 0. 核心论点（headline）

ACL-042/043 证明的是：**在 forward 模型精确已知的合成世界里**，输入加 bins、prox 加宽
都换不来 dB——这与 info-budget（+1~1.5 dB @2×，噪声封顶）一致，结论可信。

但真实数据上，我们从来没有让 solver 在"正确的物理算子"下运行过，也从来没有
让它学会容忍算子误差，更没有一个能分辨"真实信息增益"的 real-data 指标。四个证据：

1. **真实 PSF 从未钉死，且 real-eval 用的是明知错的占位值。**
   EP09 三路标定发散：0.119 / 0.2257 / 1.129 LR px（门控 FAIL）；EP15 M3 内部强边
   σ_total≈0.747（含热边宽）。而 `real_eval.py` 的真实 DC 至今用
   `forward_model_psf_sigma=0.5` 各向同性单高斯占位（roadmap §4.3 明说"不是占位 0.5"，
   从未落实）。**σ 的不确定度直接把可恢复频带的能量预算撬动 ~50×**：
   MTF@f=1.0 cyc/LR-px：σ=0.5 → 0.7%；σ=0.2257 → 36.6%。info-budget 的悲观结论
   （+1~1.5 dB）是在 σ≈0.5 这个最悲观端算的。
2. **shift 误差是 roadmap 钦定的头号风险，缓解措施从未实现。**
   contour_refined 拟合不出刚性模型（残差 p95≈0.79 px，命令符合率 4.4%）；
   而训练管线里渲染 y_i 的 shift 和喂给 A_i 的 shift **逐字节相同**
   （`dataset.py _add_burst_to_sample`，"EXACT shifts (no noise)"；
   `_select_burst`/`shift_noise_std_px` 是死代码，零调用点）。solver 从没见过
   "算子不准"这件事，真实数据上 DC 与观测打架时它只能学保守。
   2× SR 需要 shift 精度 ≪0.25 LR px；p95 0.79 px 意味着不少帧的相位信息实际被打散。
3. **"多帧输入"至今没真正做过。** 试过的是 4/9/16 个**预聚合** phase-bin drizzle 通道。
   在 shift 精确的合成世界里，info-budget 证明 4 bins 已携带全部去混叠信号——所以
   E4/E5 在合成指标上必然 null，**这个 null 不能外推到"逐帧输入无用"**：逐帧融合的
   价值恰恰在 shift/PSF 有误差、有 drift、有坏帧的真实域（自适应"信哪帧、信哪块"），
   而我们既没在这个域训练（无 DR），也没在这个域评测（见 4）。
   BSRT 笔记的结构对比正是此点：人家的学习部件看全部帧，我们的 prox 只看一张聚合图。
4. **评测体系测不到我们要的东西。**
   synth PSNR 与真实质量已两次被证反相关（ACL-032：V10 38.8 dB 但真实串珠；
   dewaffle 合成升、真实伪影同步升）；`dc_resid_band` 被 PSF/alignment 错配钉死在
   ~1.21（V9/V11/A/B 四配置同值，无判别力）；`artifact_score` 只是 FM-1 悬崖监视器。
   E4 的真实伪影改善（0.436 vs 0.460）就是被"合成 PSNR 降了"否掉的。
   **缺的是 GT-free 的真实信息量指标**（split-half FRC）。EP15 20µm 重跑至今未做，
   频带权威数字仍是旧标尺下的（roadmap Step 5 欠账）。

另有一笔独立欠账：**V11 的软是"切除术"的代价，不是必然代价。**
DIAGNOSIS.md 已从架构层证明 extent 耦合来源（far-field coupling：纯卷积 0.0 /
GN 0.673 / GN+SE 1.412）。主线的处理是把归一化整个切掉（-2.35 dB synth，视觉变糊），
D-E 和加宽都追不回。但**extent 不变的归一化**（per-pixel channel LayerNorm——统计只
跨通道、不跨空间，far-field coupling 构造性为 0）和 **halo+valid-center 训练**
（训练=推理口径）都还没试过。Restormer 笔记的结论正是"LN+GDFN 可借、MDTA 慎用"。

---

## 1. 重设计方案

### 输入（喂给学习部件什么）
- 保留：5ch fused 统计（mean/median/coverage/variance/highpass）↑2x + x0=aligned_mean。
- 新增（Stage 2）：**逐帧证据的学习融合**。每帧 i 上提到 HR 网格
  （A_iᵀ y_i 伴随 splat 或单帧 drizzle）+ 该帧 coverage mask →
  共享小编码器（2-3 conv）→ **置换不变池化**（mean⊕max⊕std 或轻量跨帧 softmax
  attention）→ 融合特征图 F（通道数与 M 无关；M 训练 24–96、真实 248 天然兼容）。
  prox 输入 = [x_k, F, 5ch]。**取代**固定 phase-bin 通道路线。
- DC 侧输入不变（y_burst + shifts + per-scene PSF），但见"求解器"节的鲁棒化。

### 输出（网络交付什么）
- 不变：单通道 HR 温度场 x_K，end-on-DC 收尾（ACL-026 的架构保证保留）。
- **不做** D-E 硬高通门（已证伪：σ 4/5/8 同跌 ~32.5 dB——把去卷积频段一起夹死了）。
  低/中频诚实交给 warm-start + band-limited DC；若恢复容量后 extent 伪影复发，
  优先用"对 aligned_mean 粗尺度（σ≥16 HR px）的软低频锚 loss"而非结构性硬门。
- 后期（Stage 3，可选）：附加不确定度通道 / 后验采样，维持 ACL-024 的定位不动。

### 求解器中间步（"中间求解器优化"）
- **DR（首要）**：训练时喂给 A_i 的 shift = 存储 shift + N(0, σ_shift)，PSF 参数
  同步加扰（σ ×(1±ε)、角度抖动）——渲染与 DC 出现受控失配，solver 学会带误差工作。
  σ_shift 取值由 Stage 0 标定的真实误差分布定（预计 0.05–0.2 px 档位扫）。
- **鲁棒 DC**：per-frame IRLS/Huber 权重（由残差公式化生成，非黑箱可学），
  可选 per-frame gain/offset 泄压项（对齐真实 drift 模型）。
- **（进阶，靠后）测试时 shift 精修**：真实推理时把 Δs_i 设为有界优化变量，
  以 DC 残差自监督精修——确定性、有界、无幻觉通道，直接打 alignment 瓶颈。

### Prox 容量恢复（extent-safe）
- norm="none" → **per-pixel channel LayerNorm**；SE → 不回，或换 GDFN 式局部门控。
- 训练 patch 带 reflect context halo（≥96 HR px），**loss 只算有效中心**——
  训练口径与 full_halo96 推理口径对齐，prox 不再学到 patch 边界响应。
- 通过 `diag_extent.py` far-field 测试（必须 ≈0）后，才允许加宽/加深。

---

## 2. 分阶段计划（每步单变量、可证伪、预注册判据）

### Stage 0 — 把物理算子和度量衡先弄对（1–2 天，多为 CPU/离线）
| # | 内容 | 产出/判据 |
|---|---|---|
| 0a | 真实 248 帧上**联合标定**：min over（Δs_i 有界、全局 θ/scale、PSF σx/σy/角度）的鲁棒带限残差，交替更新经典 x̂（drizzle/MAP-TV）。EP09 Route A 的联合推广 | σ̂_eff+CI、精修 shifts、**dc_resid 地板 1.21 前后对比**——降幅=模型误差占比的直接证据 |
| 0b | 用 σ̂ 重跑 `info_budget2.py`，并加 shift 误差维度 {0, 0.05, 0.1, 0.2 px} | 修正后的真实天花板；shift 误差每档吃掉多少 dB（给 1a 的 DR 定标） |
| 0c | **真实 split-half FRC harness**：奇/偶帧两路独立重建 → SR 带内 FRC。对 aligned_mean / drizzle / TGV / MAP-TV / V11 / E2 全部跑一遍 | GT-free 真实信息指标 + 现状排行榜；此后作为**主判据** |
| 0d | 把已有错误案例固化为**回归套件**（flat-ROI 伪影、tiled vs full-halo extent 一致性、seam 频谱、串珠探针；素材已在 ep07_solver_diag 与两个归档 episode 里） | 一条命令跑任意 checkpoint；后续所有实验的门槛 |
| 0e | 排队远端重跑 EP15（20µm 权威频带，roadmap Step 5 欠账） | band gate 与论文数字的合法性 |

### Stage 1 — 求解器鲁棒化 + 容量恢复（GPU，v6 池）
- 1a. shift/PSF-jitter DR（幅度按 0b）——其余不动，对照 V11。判据：0c 的真实 FRC↑、
  回归套件通过、synth 不塌。
- 1b. LN(per-pixel) + halo/valid-center 训练 + （可选）GDFN 门控；先 `diag_extent` 过关，
  再谈加宽。判据：synth PSNR 回到 ≥36.5（V9 方向）且回归套件全过。

### Stage 2 — 真·多帧融合（本轮主目标）
- 2a. 逐帧 lift + 置换不变融合（§1 输入设计），在 1a 的 DR 下训练。
  判据：**真实 split-half FRC 与视觉**显著优于 1b；synth 仅作 sanity 下限。
- 2b. 鲁棒 DC 权重（IRLS/Huber + gain/offset 泄压）。
- 2c. （2a/2b 达标后）测试时 shift 精修。

### Stage 3 —（达标后）不确定度图 / 论文级 eval，维持 ACL-024 定位。

---

## 3. 判据与纪律的两个修正

1. **主判据换轨**：合成 PSNR 从"主门"降级为"sanity 下限"；真实 split-half FRC（带内）
   + 回归套件 + 视觉成为主门。理由见 §0.4——继续用合成 PSNR 当门会系统性否掉
   真实域改进（E4 已经被误杀过一次的嫌疑）。
2. **dc_resid 在 0a 完成前不作判据**（被错配钉死）；0a 之后算子对了，它重新变有意义。

其余硬规矩不变：一次一个变量；band gate 一切；截断必 log（AGENTS.md / roadmap §6）。

---

## 4. 已知风险 / 开放问题

- **0a 可能得出 σ̂_eff 落在 v6 池 PSF 范围 [0.15,0.55] 之外**（EP15 M3 的 0.747 含热边宽，
  若拆解后光学 σ 仍 >0.55），则 v6 需要一次 PSF 范围补丁再生成（成本可控，配置一改）。
- 逐帧融合的训练开销：M=12–16 帧 × 小编码器（HR 网格）≈ 2–3× 目前 step 成本；
  5090 空闲 + ForwardBurstPlan 思路可复用，可行但要先 smoke 计时。
- split-half FRC 在真实数据上的方差要先测（奇偶两半各 124 帧，覆盖仍均匀），
  避免又造出一个测不准的指标。
- 采集侧杠杆（更多帧/更长积分/更高 ΔT/一次性 PSF 靶标拍摄）始终是抬天花板的正道
  （info-budget §3），本提案不与之冲突：先把现有 248 帧的信息**吃干净、吃诚实**。

## 5. 本提案吸收的证据清单

- research_log/episodes/ep07_solver_boundary_artifact/（含 ACL-042/043 队列终读）
- research_log/episodes/ep07_solver_v8_k4_fullhalo_eval_archive/diagnosis_20260630/DIAGNOSIS.md
- outputs/ep07_solver_diag/{info_budget2.py, rois.npz, diag_extent.py}
- research_log/literature/{2022_bsrt_burst_sr, 2022_restormer, 2025_2026_restoration_sota, 2026_info_budget_and_why_phone_4x}.md
- research_log/network_upgrade_roadmap.md（§4 风险 1/3/5 = 本提案 Stage 0 的三条欠账）
- algorithm_changelog ACL-023…ACL-043；EP09/EP15 episode README
- 代码事实核查：dataset.py（exact shifts、死代码 _select_burst）、unroll.py（end-on-DC、
  frozen eta、D-E 机制）、forward_torch.py（认证算子、autograd 伴随）、
  real_eval.py（σ=0.5 占位、full_halo96、holdout dc_resid）、fusion.py（5ch 语义）
- 远端状态：v6 池 5000 场景完整；GPU 空闲；本地无 checkpoint（仅诊断 npz/CSV）
