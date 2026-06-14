# C4 / Claim 3 重写要点（2026-06-13）

> **触发**: 用户目视发现 + 主线量化复核（中心细线 ROI）。
> **角色**: 本文件是 C4「classical-vs-learned 裁决」与 Claim 3「Pareto 前沿」**改写的单一权威底稿**。
> 所有正文/报告改动（`01_outline.md`、`07_experiments.md`、`reports/ep07_v9_attribution/`、`run_v10_highlam.md`）以本文件措辞为准。
> **方向**: A（平衡版核心裁决）+ B（显式 task-level 轮廓可辨指标，诚实标注非保真证据）。

---

## 1. 旧框架错在哪

旧稿把 C4/Claim 3 写成「经典各向异性 TGV（+MAP-TV）是可信交付物，学习方法的时间轴 Pareto 被 TGV 支配」。复核发现这个「支配」是 **proxy 特定的、且部分被伪影误导**：

- **proxy 对连续性失明**：`hp_corr_input`（与模糊 drizzle 的相关）和 `sharp_p95`（P95 梯度）都不度量轮廓**连续性**；`sharp_p95` 甚至**无法区分连续锐线与珠串/颗粒**，珠串和颗粒都会推高它。
- **TGV 有 TV-staircase 小瑕**：在最细对角 trace 上，TGV 把连续线渲染成轻微「珠串/块状」(staircase)，而**观测本身（drizzle）该处是连续的** → 这是 TV 类正则的已知 staircasing 伪影，不是真实结构。TGV 的 `lattice_score=0.0169` 是所有保真参照里最高的。
- **学习输出并非「更干净」**：量化（中心 ROI）显示学习输出（V9A late / V10）携带的**高频内容最多**——`lattice`：drizzle 0.0015 < v9a_20k 0.0009 < TGV 0.0169 < v9a_60k 0.0153 < **v10 0.024**；线尺度 beading 与像素颗粒同序。这些高频是**真细节与不可验证 grain/振铃的混合**，无 GT 不可判定。
- **目视偏好 ≠ 保真**：眼睛把「锐」读成「干净」，但更锐的学习输出恰恰保真更低（`hp_corr_input` 0.96→0.88）。

**一句话**：在无 GT 下，TGV 与学习方法是**互补的失效模式**，没有可认证的单一赢家——这正好印证（而非削弱）「trust but verify」主线与 C2 协议「proxy 只能门控/选点、必须配视觉+结构 gate」。

## 2. 证据底数（中心细线窗口，已复现）

| 对象 | hp_corr_input↑(保真) | sharp_p95↑ | lattice↓(grain/HF) | 目视（中心梳齿） |
|---|---|---|---|---|
| drizzle（观测，软上限） | 1.000 | 0.503 | 0.0015 | 连续但糊，未解析梳齿 |
| v9a_20k（最保真 ckpt） | 0.974 | 0.62 | 0.0009 | 偏软 |
| **TGV（经典锚）** | 0.960 | 0.959 | 0.0169 | 解析梳齿，但轻微 TV-staircase 珠串 |
| v9a_60k | ~0.906 | ~1.2 | 0.0153 | 梳齿被焊粗/合并（过冲） |
| **V10 λ=1.2@15K（高-λ最终工作点）** | 0.922 | 0.987 | 0.0141 | 锐度约等于 TGV、grain 更低，但保真仍低于 TGV |

> 注：MAP-TV 同属 TV 家族，预期亦有 staircase，但本轮未单独量化；写作时标注「likely shares the TV staircase」。

## 3. 新 C4（英文，落 `01_outline.md` / 后续 LaTeX）

> **C4 — Honest classical-vs-learned verdict: complementary failure modes, no GT-certifiable winner.**
> Anisotropic coverage-weighted TGV resolves the raster-anisotropy stripe artifacts (artifact 3.87→0.695, raw-control corr 0.916) and is the most observation-faithful method on the fidelity proxies, but it imprints a mild total-variation **staircase ("beading") on the finest diagonal contours** — an artifact absent from the (continuous) multi-frame observation, and reflected in its being the highest-`lattice` faithful reference. Evidence-injected learned reconstructions (hybrid-drizzle / residual-over-drizzle) are the **sharpest by inspection and resolve the central fine "comb" most crisply**, but they carry the **most high-frequency content** (a mix of plausibly-real detail and unverifiable grain) and drift measurably on observation fidelity. We therefore report a three-part, GT-free verdict — verifiable fidelity proxies (favor classical/drizzle), a structural grain proxy plus dual-domain visual panels, and an **explicitly task-level contour-legibility preference** — and conclude that **no method is a GT-certifiable winner; the task-level visual preference is not fidelity evidence, and nothing in this regime supports metrology-grade claims.**

## 4. 新 Claim 3（英文）

> **Claim 3 — Prior erosion and a proxy-specific (not absolute) Pareto.** Synthetic structure priors erode real observation detail with training step: the learned fidelity–sharpness trajectory does not strictly beat the classical TGV working point **on the observation-fidelity proxies**. We explicitly bound this statement: the proxies (`hp_corr_input`, `sharp_p95`) **do not measure contour continuity**, and `sharp_p95` cannot separate a continuous sharp line from a beaded/grainy one. On a structural-grain reading (`lattice`) and by inspection, classical TGV exhibits TV-staircase on the finest traces while the learned arms exhibit grain — **neither dominates on all axes.** The drift mechanism (null-space, Claim 2) is unchanged; what we retract is any claim of *absolute* classical dominance.

## 5. Claims-to-avoid 新增（`01_outline.md` 红队清单）

- 不得把 TGV 写成「无条件最佳 / the trustworthy deliverable」——必须带 TV-staircase 小瑕 caveat。
- 不得声称学习输出「更干净 / 更保真」——其额外高频不可验证；**目视偏好 ≠ 保真证据**。
- `sharp_p95` 对连续性失明、会被珠串/颗粒推高——**绝不单独用**；报锐度必并报 `lattice` + 视觉。
- 「task-level 轮廓可辨偏好」必须显式标注为主观/任务级、与可验证 proxy 分列，**不得当作分辨率/保真结论**。

## 6. 图表含义

- **F5 主视觉对比**：必须含「中心梳齿高倍裁剪」一行，并排 drizzle(软) / TGV(锐但 staircase 珠串) / V9A-late(过粗) / V10(锐但仍有 grain trade-off)，配温度图+highpass 双域，让 staircase vs grain 双失效**肉眼可见**。终稿已生成：`output/paper_figures/fig05_main_visual.{png,pdf}`。
- **F4 Pareto**：散点轴仍可用 (hp_corr_input, sharp_p95)，但**图注必须声明该平面不含连续性轴**，并叠加 `lattice` 作为第三维（点大小/颜色）。

## 7. 高-λ V10 实验的新成功判据（落 `run_v10_highlam.md` §2）

不再是「在 (hp_corr_input, sharp_p95) 上支配 TGV」。改为：**在保真 `hp_corr_input` 尽量高（趋向 drizzle）的同时，把 grain `lattice` 压到 ≤ TGV(0.0169)，并保持中心梳齿清晰/连续**。即用残差约束换取「锐而不 grain」的工作点。**不声称"打败 TGV"**；结论仍是诚实 trade-off + 无 GT 不可认证。

> **✅ 判据已满足（2026-06-14 高-λ sweep 完成）**：λ∈{0.2,0.5,1.2,3.0}×25K，残差自检通过（缓存温度均值≈23.3°C）。7 个 checkpoint 满足 `hp_corr_input≥0.92 ∧ lattice≤0.0169`；最佳折中 **λ=1.2@15K = (0.922, 0.987, 0.0141)**：锐度 ≈ TGV（+3%）、grain 比 TGV 低 17%、保真刚过门控。残差约束因此把 *fidelity–sharpness–grain* 折中变成**可调 λ 旋钮**（大 λ→高保真低 grain 但软；小 λ→锐但 grain 多）。**但所有点保真仍 < TGV(0.960)**，因此 C4 维持「no GT-certifiable winner / 诚实 trade-off」，**不写成"打败 TGV"**；写作时 C4 可加一句「显式残差约束能换到锐而不 grain 的工作点，但以观测保真为代价，仍不构成可认证支配」。Phase 2 精化已决定跳过。详见 `research_log/algorithm_changelog.md` ACL-020「高-λ sweep 结果」。

## 8. 受影响文件清单

| 文件 | 改动 |
|---|---|
| `docs/paper/01_outline.md` | abstract skeleton 末句、C4、Figures(F5)、Claims-to-avoid |
| `docs/paper/07_experiments.md` | §6.1 verdict line、§6.2/§6.6 措辞、§6.7 可加 staircase/grain 句 |
| `reports/ep07_v9_attribution/README.md` | Claim 3 标题/解读、Claim 4 回填修正状态、Conclusion Status |
| `algos/ep07_unet_sr/scripts/run_v10_highlam.md` | §2 成功判据（已在 HOLD 横幅预告，正式重写） |
| `docs/next_move_plan.md`（工作文档，可后补） | §5 Claim 3、§6 POC 含义 |
