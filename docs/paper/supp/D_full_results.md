# Supplementary D — 完整实验结果档案（中文草稿）

> **角色**: technical appendix 的结果块——主文 §6 每个结论的完整数据背书 + 负结果档案。
> **语言**: 中文草稿（2026-06-12 决策），迁 LaTeX 时翻译为英文。
> **尺度红线**: TB-scale（训练期 eval）与 EP11-harness-scale 的 artifact 定义不同（supp A.3.1），本文件各表逐一标注口径，绝不混表。
> **对应清单**: `10_writing_handover.md` §3.D + 2026-06-12 新增 D.0/D.7（fine-window 口径与融合 baseline，决策：supp D 专节 + 主文 §6 一句话引用）。

---

## D.0 fine-window 诊断口径（方法注，新增）

V9 系列复盘引入的中心细线窗口诊断，是 D.7/D.8 的指标基础（实现 `algos/ep07_unet_sr/scripts/v9_review/common.py`）：

- **窗口**: 2x 网格全幅 960×1280 → center-1/3（rows 320:640, cols 427:853）→ 细线窗 rows 384:518, cols 478:674——中心两条细 zigzag「梯子」及周边粗线，客户最关心的最难结构。
- **四指标**:

| 指标 | 定义 | 方向 | 角色与限制 |
|---|---|---|---|
| hp_corr_input | 窗口 highpass(σ=5) 对 **drizzle 2x mean 输入通道**的 Pearson | ↑ | **保真轴**——网络是否保留观测证据；对「忠实复制模糊」也给高分，度量保真不度量增强 |
| hp_corr_tgv | 同上对 EP10 TGV 的 corr | ↑ | 保真 proxy 交叉验证；TGV 非 ground truth |
| sharp_p95 | 窗口温度图 P95 梯度幅值 | ↑* | **锐度 proxy**——振铃/假边/饱和对比同样推高（`AGENTS.md` 硬教训 8），必须与保真轴联看 |
| lattice_score | highpass 窗口 \|f\|>0.35 cyc/px 频段能量占比 | ↓ | 捕捉 drizzle 格纹/棋盘伪影 |

- **参照点**: drizzle 输入通道 (1.000, 0.503)＝观测域上限（模糊但零幻觉）；TGV (0.960, 0.959)＝经典前沿工作点。
- **与 §5 协议的关系**: fine-window 是**局部诊断窗口**，依赖 TGV 参照，独立性低于 FRC/proxy 对——只用于归因与选型，不替代全局协议（决策 2026-06-12：不升格进主文 §5）。

**D.0 资产依赖**：`algos/ep07_unet_sr/scripts/v9_review/common.py`；`output/ep07_v9_review/`（重建见 E.2）。

---

## D.1 T1/T2 扩展版 ⬜（等统一口径 harness 重跑）

**内容**（落地后填）：全臂 × 全 checkpoint × 全列（split-half NRMSE / artifact / corr / FRC@{16,14,12} / zigzag FWHM & dip / runtime）的 S-T1 扩展表 + TGV/MAP-TV 参数网格全表（S-T2）。

**已有 partial 数据**（不可直接进终表，口径混杂）：

| 来源 | 内容 | 路径 |
|---|---|---|
| EP11 harness | v9b@11K 0.0517/1.732/0.7776 等四臂 canonical | `output/ep11_dl_benchmark/checkpoint_selection/v9b_step11000/comparison_summary.csv` |
| EP10 sweep | TGV 6 行参数网格 | `output/ep10_tgv_sr/sweep_results.csv` |
| EP15 M4 | MAP-TV σ×λ 12 格选择 | `output/ep15_info_limit/m4_deconv_anchor/parameter_selection.csv` |

**⬜ 待回填条件**：① V9A/V9C（及 V10）canonical checkpoint 选定；② `run_unet_vs_drizzle_2x.py` 谱系单次重跑全部臂（GPU 窗口，`00_status_and_plan.md` 状态板最后一行）。

---

## D.2 FRC 全档案（信息存在性证据链）

方法与判据见 supp A.4；本节归档全部数值。

### D.2.1 主曲线与 cutoff

- 1/7 cutoff **17.03 µm**（3-seed 均值曲线），逐 seed 16.17/16.17/17.03，std 0.50 µm；half-bit 判据同值（`output/ep15_info_limit/m2_frc/{frc_summary.json, frc_repeats.csv}`）。
- band 全表（含全部控制组列）：supp A.4.5；源 `frc_band_table.csv`。
- 控制组 cutoff：bicubic 正控 13.58 µm（未通过预期）、shuffle 负控/漂移控 26.20 µm；zero-coverage 均值 27.2%/最大 36.2%（`frc_controls.csv`）。

### D.2.2 MAP-TV 前后 split-half 对照

| 周期 (µm) | bare drizzle | MAP-TV (σ=0.2, λ=1e-3) |
|---:|---:|---:|
| 20 | 0.319 | 0.976 |
| 16 | 0.088 | 0.965 |
| 14 | 0.053 | 0.955 |
| 12 | 0.575 | 0.947 |
| 10 | 0.893 | 0.934 |

（`output/ep15_info_limit/m4_deconv_anchor/frc_verification.csv`）口径：split-half **一致性**提升，非光学分辨率证明；bare cutoff 仍 17.03 µm 与 M2 一致。

### D.2.3 zigzag 轮廓指标（M4 锚的结构收益）

median FWHM **114 → 100 µm**、dip 0.929 → 0.934、3/3 剖面保持分离；逐剖面混合（2 宽 1 显著窄）→ 论文措辞钉死「limited contour enhancement」（`zigzag_profile_metrics.csv`；禁写边界第 6 条）。

**D.2 资产依赖**：`output/ep15_info_limit/{m2_frc/, m4_deconv_anchor/}` 全套 CSV/PNG（S-F1）；重建 `run_m2_frc.py` / `run_m4_deconv_anchor.py`。
**D.2 待回填**：无——可成稿。

---

## D.3 漂移演化档案：数值轨迹 + 视觉序列

### D.3.1 全局漂移指标表（TB-scale，real_eval 248 帧 contour_refined）

artifact（↓）/ raw_control_corr（↑），源 `output/ep07_v9_review/ep07_eval_real_metrics.csv`（`extract_tb_metrics.py` 从 tb_logs 提取）：

| step | v8.1a | V9A | V9B | V9D | V9C |
|---|---|---|---|---|---|
| 10K | 0.390 / 0.756 | 0.446 / 0.719 | 0.369 / 0.758 | 0.379 / 0.758 | 0.516 / 0.714 |
| 20K | 0.476 / 0.729 | 0.514 / 0.702 | 0.486 / 0.735 | 0.575 / 0.642 | 0.689 / 0.655 |
| 30K | 0.602 / 0.703 | 0.660 / 0.663 | 0.611 / 0.709 | 0.615 / 0.694 | ⬜ |
| 40K | 0.627 / 0.698 | 0.656 / 0.665 | 0.640 / 0.697 | 0.672 / 0.681 | ⬜ |
| 60K | 0.643 / 0.689 | 0.646 / 0.669 | 0.655 / 0.688 | 0.677 / 0.677 | ⬜ |

读数：V9A 是唯一 30K→60K 漂移压平的臂（−0.014/+0.007，其余单调恶化），但平台 corr 0.669 < v8.1a 0.689（跨 input-mode 横比无效——A.3.4 推论 2，此处只看各自轨迹形状）；V9D 比 V9B 更差且 1K–28K 震荡。**⬜ V9C 30K+ 行等今晚训练完成后由 `extract_tb_metrics.py` 补齐**（截至 06-12 17:00 已至 ~30K，早期 10K/20K 漂得比 V9A 同期更快，结论以 60K 全曲线为准）。

### D.3.2 fine-window 训练时间轴（V9A，hp_corr_input / sharp_p95）

| 对象 | hp_corr_input | hp_corr_tgv | sharp_p95 | lattice |
|---|---:|---:|---:|---:|
| drizzle 输入通道 | 1.000 | 0.960 | 0.503 | 0.0015 |
| TGV | 0.960 | 1.000 | 0.959 | 0.0169 |
| V9A 5K | 0.968 | 0.945 | 0.677 | 0.0009 |
| V9A 10K | 0.970 | 0.953 | 0.683 | 0.0010 |
| V9A 15K | 0.936 | 0.937 | 1.017 | 0.0028 |
| **V9A 20K** | **0.974** | 0.944 | 0.615 | 0.0009 |
| V9A 25K | 0.935 | 0.931 | 0.831 | 0.0062 |
| V9A 30K | 0.908 | 0.908 | 1.147 | 0.0121 |
| V9A 40–60K | 0.906±0.001 | 0.908 | 1.21–1.25 | ~0.015 |

（`output/ep07_v9_review/v9a_pareto_metrics.csv`）**30K 保真悬崖**：0.974→0.906 后焊死；锐度超过 TGV 的区间（30K+）与去相关重合 = 幻觉过冲。对照 v8.1a 60K 同窗 0.926/0.936——V9A 60K（0.925/0.935）与之无差别，hybrid 早期增益被训练后期完全抹平。**caveat**：35K 中断 bs 128→64 与悬崖重合，混杂未排除（C.3）。

### D.3.3 视觉序列与选帧

- 逐臂训练期序列：`algos/ep07_unet_sr/outputs/{run}/eval_real/unet_step{K}_center_zoom3x_temperature.png`（步距 5K；v6/v9b/v9d 另有 1–2K 密度；V9A 35K 缺失）。
- 已选定的视觉证据：`output/ep07_v9_review/{v9a_checkpoint_strip.png（5K→60K 演化条带）, fine_zigzag_final_panel.png（输入/TGV/v8.1a/V9A 10K/60K 同口径面板）, tight_center_full_comparison.png}`。
- companion 轨迹图：`output/paper_figures/fig03s_v9a_trajectory.*`（S-F3；V9A 不进主图 F3——红线：proxy 跨输入不可横比）。
- ⬜ **S-F5 终选**：等 V9C/V10 落地后统一挑「每臂 4 帧（early/canonical/30K/60K）」组版（选帧脚本已就绪，`render_comparison_panels.py` 参数化 checkpoint 列表）。

**D.3 资产依赖**：`output/ep07_v9_review/`（CSV+PNG+cache）；`algos/ep07_unet_sr/outputs/*/eval_real/`；`output/paper_figures/fig03*.{png,pdf}`。
**D.3 待回填**：⬜ V9C 指标行与视觉帧（等今晚 60K）；⬜ V10 三臂轨迹（等 GPU 实验）；⬜ S-F5 终选组版。

---

## D.4 负结果档案（主文 §6.7 的完整版）

> 四项负结果已全部落地（V9D 完成后解锁），本节可成稿；每项给「现象 → 数字 → 结论边界」。

### D.4.1 PixelShuffle HR 头（v8.1b）

- 现象：中等边框间条纹状亮色伪影；锯齿未改善；中心细线模糊与 bilinear 头（v8.1a）相同。
- 数字（TB-scale）：artifact 0.413→0.709 全程高于 v8.1a 0.390→0.643；corr 0.747→0.667 全程更低（ACL-015；`checkpoint_candidates.csv`）。
- 结论边界：**head 归因失败**——证明中心细线瓶颈不在解码头；保留 bilinear。与 v8.1a 的 loss-cooldown 对照共同构成主文 §6.3 的「两臂归因」：细线模糊对 loss 温度与 head 均不变 → 输入信息瓶颈。

### D.4.2 4x 网络（EP12）

- 现象与数字：四臂 gate（EP07×2up vs EP12-4x）中 EP12 raw-control highpass Pearson **0.223** vs EP07×2up **0.389**；artifact 0.535 vs 0.472——4x 网络两轴皆差（`output/ep12_4x_benchmark/ep07x2up_vs_ep12/metrics_summary.csv`；ACL-012）。
- 理论互证：4x Nyquist MTF ≤ 0.042（σ=0.2）、σ≥0.35 时 ~0（supp A.1.3）；3x/4x 相位 bin collapse（supp B.2.5）。
- 结论边界：4x 无真实增益**且**理论上不应有——负结果与 MTF 界互证是「先第一性原理后实验」工作流的展示案例（`AGENTS.md` 硬教训 6）。

### D.4.3 loss 侧 forward 锚定（V9B band + V9D full，路线关闭）

- 现象：两臂漂移曲线与无锚 v8.1a 几乎重合；`loss/forward_model` 自 10K 起贴底 0.004–0.009 的同时 artifact 持续上爬（ACL-017；`forward_loss_curves.csv`）。
- 数字（TB-scale）：40K→60K 漂移 v9b +0.0145/−0.0082 vs v8.1a +0.016/−0.009（重合）；V9D 60K 0.677/0.677 劣于 V9B 0.655/0.688，且 1K–28K 剧烈震荡（复现 ACL-005 全频低通梯度冲突）。
- 结论边界：**无论 band 选择，1x 观测域锚定对真实漂移不可见**——机制即 supp A.2.4 Proposition 1；「band 太窄」的反对意见被 V9D 关闭。⬜ V9C（hybrid 输入下合法 1x 锚）是该 claim 的最后一臂：若同样无效 → 升级为「即使输入含 2x 证据、合法 1x 锚仍盲」；若有效 → 锚定结论限定于 1x 统计输入。等今晚 60K。

### D.4.4 渲染 AVI 作 SR 输入（数据审计排除）

8-bit 渲染、~67% 重复帧、无温度矩阵（supp B.1.6）→ 仅作方向 consistency check（supp A.5.2）。负结果意义：「能拿到的数据」≠「能进重建的数据」，审计先于算法。

**D.4 资产依赖**：`research_log/algorithm_changelog.md` ACL-005/012/015/017；`output/ep12_4x_benchmark/`（重建：`algos/ep12_4x_benchmark/scripts/run_ep07x2up_vs_ep12_4x.py`）；S-F6（负结果档案图，🔧 选图：v8.1b 条纹 crop、EP12 对比图、AVI 审计图）。
**D.4 待回填**：⬜ V9C 判定句（D.4.3 末，等 60K）；⬜ S-F6 组版。

---

## D.5 frame-budget 与鲁棒性全表（EP16 经典臂）

设计：E1 帧数预算 N∈{31,62,124,248}（phase-stratified 子集，seeds 101/202/303）、E2 shift 扰动 σ∈{0,0.05,0.1,0.2} px（seeds 401–403）、E3 对齐源（command_prior vs contour_refined）；drizzle + TGV 两经典臂，37 个 unique run 全 success（`output/ep16_budget_robustness/run_manifest.json`）。**注意**：这是推理期稳定性研究，不属于 T1 统一 harness；TGV 的 split-half/FRC 列复用同子集 drizzle proxy（预算考虑），TGV 自身列为 artifact/corr/zigzag。

### D.5.1 E1 帧数预算（全表，多 seed 行已聚合为 mean±std；N=248 单 run）

| arm | N | corr | split-half NRMSE | artifact | FRC@16 µm | zigzag FWHM (µm) |
|---|---:|---:|---:|---:|---:|---:|
| drizzle | 31 | 0.747±0.032 | 0.0715±0.0012 | 1.649±0.041 | 0.109±0.065 | 56.7±20.2 |
| drizzle | 62 | 0.772±0.016 | 0.0514±0.0034 | 1.536±0.044 | 0.248±0.070 | 56.7±20.2 |
| drizzle | 124 | 0.770±0.010 | 0.0361±0.0010 | 1.341±0.038 | 0.332±0.018 | 45.0 |
| drizzle | 248 | 0.771 | 0.0306 | 1.145 | 0.479 | 45.0 |
| TGV | 31 | 0.728±0.053 | 0.0717±0.0017* | 0.946±0.003 | 0.075±0.043* | 42.5±3.5 |
| TGV | 62 | 0.754±0.009 | 0.0531±0.0025* | 0.851±0.051 | 0.218±0.066* | 40.0 |
| TGV | 124 | 0.735±0.013 | 0.0366±0.0003* | 0.747±0.011 | 0.328±0.024* | 40.0 |
| TGV | 248 | 0.741 | 0.0306* | 0.708 | 0.479* | 40.0 |

（*TGV 的 split-half/FRC 列为同子集 drizzle proxy，见本节头注；源 `frame_budget.csv` 17 行，本表 2026-06-12 直接聚合抄录。）

读数：drizzle corr 增益大半在 N=62 前到位（0.747→0.772），split-half/artifact/FRC 随 N 单调改善——后三者受益于相位覆盖累积；TGV artifact 全程低于 drizzle（0.946→0.708），corr 非单调（0.728→0.754→0.735→0.741）。**口径注**：主文 §6.4 现写「corr rises to 0.772 at N=248」，CSV 原值 0.7713 应作 0.771——⬜ 主文微修。

### D.5.2 E2 shift 扰动

| arm | σ=0 → 0.2 px | corr | artifact | FRC@16 |
|---|---|---:|---:|---:|
| drizzle | — | 0.771 → 0.770 | 1.145 → 1.434 | 0.479 → 0.340 |
| TGV | — | 0.741 → 0.744 | — | （同 proxy 下降） |

读数（主文 §6.5 措辞已定）：鲁棒性是 metric-specific 的——raw-control corr 对 ≤0.2 px 扰动几乎不动，coverage/FRC 类指标敏感。压力测试口径，非真实对齐误差估计。

### D.5.3 E3 对齐源消融（端到端价值证据）

| arm | command_prior | contour_refined |
|---|---:|---:|
| drizzle corr | 0.662 | **0.771** |
| drizzle FRC@16 | 0.0166 | **0.479** |
| TGV corr | 0.642 | **0.741** |

读数：数据驱动对齐对两个经典臂都是 +0.10~0.11 corr 的端到端增益；FRC@16 从无到有（0.017→0.479）——B.2 对齐链投入的最终回报证据。

**D.5 资产依赖**：`output/ep16_budget_robustness/{frame_budget.csv, shift_robustness.csv, alignment_source.csv, run_manifest.json, fig_*.png}`；`output/paper_figures/fig07_budget_robustness.*`（F7→supp 降级方案 S-F8）；重建 `algos/ep16_budget_robustness/scripts/run_ep16_classical.py`。
**D.5 待回填**：⬜ D.5.1 中间 N 行抄录；⬜ MAP-TV / UNet（GPU）臂三个矩阵（等 GPU 空闲窗口，入口脚本已参数化）。

---

## D.6 视觉 gate panel 选编

- 已有：`output/ep11_dl_benchmark/checkpoint_selection/panel_{v6, v8.1a, v8.1b, v9b}.png`——每臂候选 checkpoint 三联温度域 panel + proxy 标注（C.5 协议的视觉 gate 实物）。
- 选编原则（supp 版面）：每臂 1 行（canonical + 60K 对照即可），v8.1b 行保留条纹证据；配 §6.6 金句。
- ⬜ V9A/V9C/V10 panel（同脚本 `--arms` 扩展后生成，等选点）。

**D.6 资产依赖**：`output/ep11_dl_benchmark/checkpoint_selection/panel_*.png`（S-F4）。

---

## D.7 零训练融合 baseline（专节；V10 必须打败的对照）

> **定位（2026-06-12 决策）**：supp D 专节 + 主文 §6 一句话引用。它把「学习臂的增益是否超出后处理可得」变成可检验命题，是 V10（残差参数化 + λ 惩罚）的对照前沿。

### D.7.1 方法

fused(λ) = (1−λ)·anchor + λ·unet_pred，λ ∈ {0, 0.1, …, 1.0}；anchor ∈ {drizzle 2x mean（248 帧 contour_refined，与 real_eval 同口径）, EP10 TGV}；unet_pred ∈ {V9A 20K（最保真）, V9A 60K（最锐）}。指标 = D.0 fine-window 四指标（实现 `run_fusion_baseline.py`）。

### D.7.2 结果（fine-window 口径）

**TGV 锚 × V9A-60K**（支配区间存在）：

| λ | hp_corr_input | hp_corr_tgv | sharp_p95 | lattice | 支配 TGV 工作点 |
|---:|---:|---:|---:|---:|---|
| 0（=TGV） | 0.960 | 1.000 | 0.959 | 0.0169 | — |
| 0.1 | 0.963 | 0.999 | 0.970 | 0.0134 | **是** |
| **0.2** | **0.963** | 0.995 | **0.968** | **0.0108** | **是（最强候选）** |
| 0.3 | 0.961 | 0.989 | 0.977 | 0.0091 | 是 |

**drizzle 锚 × V9A**（任意 λ 不支配：保真极高但锐度不足，如 drizzle+0.6×V9A60 = (0.954, 0.933)）。

（`output/ep07_v9_review/{fusion_baseline_metrics.csv, fusion_baseline_summary.md, fusion_pareto_overlay.png}`）

### D.7.3 结论与边界

1. **存在事后线性组合严格支配 TGV 工作点**：TGV + 0.2×V9A-60K 同时改善保真 (+0.003)、锐度 (+0.009)、格纹 (−36%)——零训练、零 GPU、推理期一次加权。
2. **对 Claim 4 的含义**：V10 的成功判据从「越过 TGV」抬高为「越过融合前沿」；若 V10 所有 λ 臂都不及融合 baseline → Claim 4 收敛为「学习贡献可被事后融合替代」的诚实结论（同样可发表）。
3. **边界**：fine-window 局部口径 + TGV 参照非 GT（D.0）；λ 在窗口上选出，无独立验证窗 → 终稿如引用「最优 λ」须加 selection-on-test caveat，或 ⬜ 在第二个 held-out 窗口复核（CPU 几分钟，脚本支持改窗口参数）。
4. ⬜ **V10 对照行**（等 GPU 实验）：三臂 λ∈{0.02,0.05,0.15}（G1.5 标定后定）各自工作点叠加进 `fusion_pareto_overlay.png` 同图。

**D.7 资产依赖**：`algos/ep07_unet_sr/scripts/v9_review/run_fusion_baseline.py`；`output/ep07_v9_review/fusion_*.{csv,md,png}`；`output/ep10_tgv_sr/best_hr_temperature.npy`；cache npy（E.2 重建链）。
**D.7 待回填**：⬜ 第二验证窗复核；⬜ V10 三臂叠加与判定句。

---

## D.8 主文图表 → 本档案映射（导航表）

| 主文 | 数据背书 |
|---|---|
| §6.1 T1/F5 | D.1（⬜）+ D.2.3 |
| §6.2 F3 | D.3.1 + D.4.3 |
| §6.3 T2/F6 | D.3.2 + C.3 + D.4.1 |
| §6.4/6.5 F7 | D.5 |
| §6.6 F4 | C.5.3 + D.6 |
| §6.7 | D.4 |
| §6 融合句 | D.7 |
