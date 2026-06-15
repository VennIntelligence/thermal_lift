# EP07 V9 Attribution Report Skeleton

> 日期: 2026-06-12  
> 范围: C5 报告骨架，整理 V9A/V9B/V9D 已有归因证据，并为 V9C/V10/EP15 split-half FRC 留出补充位置。  
> 边界: 本报告不启动 GPU、不运行训练或推理、不改算法代码；所有图引用指向 C1 迁移后的 `output/ep07_v9_review/` 路径。

## Executive Summary

V9 系列目前支持三个已证实结论：第一，1x 统计输入会在网络前端丢掉多帧亚像素相位信息，hybrid 2x drizzle 输入能在早期 checkpoint 中透传中心细 zigzag 结构；第二，1x forward consistency 无论 highpass band 还是 full band 都没有压住真实数据漂移，说明漂移主要落在当前 forward 算子不可见或弱约束的方向；第三，UNet 训练时间轴呈现明确的 fidelity-sharpness trade-off，后期锐化超过 TGV 的区域与观测去相关和幻觉过冲重合。

这份报告仍是骨架：V9C 完整 60K 结果、V10 显式残差幅度惩罚实验、以及 EP15 口径 split-half FRC 尚未并入。因此本文只把 Claim 1-3 标为“已证实”，把 Claim 2 的 hybrid+legal forward 第三个变体标为“待 V9C”，把 Claim 4 标为“待 V10”。

## Evidence Conventions

| 口径 | 含义 | 方向 | 限制 |
|---|---|---:|---|
| `hp_corr_input` | 中心细线窗口 highpass 后与 drizzle 2x mean 输入通道的 Pearson correlation | 越高越保真 | 对“忠实复制模糊输入”也会给高分，不度量增强幅度 |
| `hp_corr_tgv` | 中心细线窗口 highpass 后与 EP10 TGV 的 correlation | 越高越接近当前经典参照 | TGV 不是 ground truth，含台阶伪影 |
| `sharp_p95` | 中心窗口温度图梯度幅值 P95 | 越高通常代表边缘更强 | 锐度 proxy；振铃、假边、饱和对比也会推高，不能单独作为 SR 成功证据 |
| `Tenengrad` / gradient 类指标 | 梯度能量或边缘强度 proxy | 越高通常更锐 | 与 `sharp_p95` 同类，不能单独证明 SR 成功 |
| `lattice_score` | highpass 窗口高频格纹能量占比 | 越低越少格纹 | 只捕捉特定频段伪影，不覆盖所有 hallucination |
| `artifact_score` | EP11 同口径真实数据漂移/artifact proxy | 越低越好 | 不是光学 ground truth，不应跨口径混算 |
| `raw_control_corr` | 与控制图的相关性 proxy | 越高越好 | 受结构相似性和风格化共同影响 |

关键图表引用如下，均使用从本 README 出发的相对路径：

| 图 | 预计路径 | 用途 |
|---|---|---|
| 输入/输出/经典对照面板 | `../../output/ep07_v9_review/fine_zigzag_final_panel.png` | 对比 drizzle 输入、TGV、v8.1a 60K、V9A 早晚 checkpoint 的中心细线结构 |
| V9A Pareto 散点 | `../../output/ep07_v9_review/v9a_pareto_scatter.png` | 展示 fidelity (`hp_corr_input`) 与 sharpness (`sharp_p95`) 的时间轴轨迹 |
| V9A checkpoint 演化条带 | `../../output/ep07_v9_review/v9a_checkpoint_strip.png` | 观察 5K-60K 中心窗口从透传、压格纹到后期过锐化的演化 |
| V9 real-eval 漂移曲线 | `../../output/ep07_v9_review/ep07_eval_real_metrics.png` | 对比 v8.1a、V9A、V9B、V9D、V9C 的 artifact/corr 时间序列 |
| V9 fine-window 指标表 | `../../output/ep07_v9_review/v9a_pareto_metrics.csv` | 支撑中心窗口定量读数 |
| V9 real-eval 指标表 | `../../output/ep07_v9_review/ep07_eval_real_metrics.csv` | 支撑全局漂移读数 |

## Claim 1: Phase Information Bottleneck

**Claim.** 多帧亚像素 SR 的学习方法必须在输入端保留相位信息。v8.1 系列的 1x 统计 5ch 输入把 248 帧 clean set 的亚像素相位在进入网络前压扁，使网络上限退化为单帧锐化；V9A 的 hybrid 2x drizzle 输入让中心细 zigzag 结构能够在早期 checkpoint 中被透传。

| Evidence | Observation | Figure / Data | Status |
|---|---|---|---|
| v8.1 A/B 中心区域 | loss/head 改动不能恢复中心细结构，说明瓶颈不在输出 head 或常规 loss 壳 | 待补: `../../output/ep07_v9_review/v8_1_center_comparison.png` | 已证实 |
| V9A hybrid drizzle 输入 | ch5 drizzle mean 在观测域中保留中心“梯子”条纹；V9A 10K-20K highpass corr 约 0.97 | `../../output/ep07_v9_review/fine_zigzag_final_panel.png`; `../../output/ep07_v9_review/v9a_pareto_metrics.csv` | 已证实 |
| V9A 60K 回落 | V9A 60K 与 v8.1a 60K 在同窗口近似无差别，说明早期输入增益会被后期先验侵蚀 | `../../output/ep07_v9_review/v9a_checkpoint_strip.png` | 已证实 |

Interpretation: Claim 1 不声明 2x drizzle 输入本身就是最终 SR 解；它只是证明真实多帧相位信息必须先进入网络。后续是否能转化为可交付增强，需要同时满足保真、锐度、伪影和 split-half 稳定性证据。

## Claim 2: Forward Anchoring Null-Space Failure

**Claim.** 1x forward consistency 无法抑制真实数据上的后期漂移，因为合成先验引导出的幻觉方向主要落在 shift -> PSF -> downsample forward model 的零空间或弱可见空间中。仅在 loss 侧加 highpass/full-band 1x 锚定，不足以约束 2x 输出的真实细节。

| Evidence | Observation | Figure / Data | Status |
|---|---|---|---|
| V9B highpass-band forward anchor | `loss/forward_model` 可以躺到低值，但 artifact 仍随训练上爬，raw corr 下降；曲线与无锚 v8.1a 接近 | `../../output/ep07_v9_review/ep07_eval_real_metrics.png`; `../../output/ep07_v9_review/ep07_eval_real_metrics.csv` | 已证实 |
| V9D full-band forward anchor | full-band 锚定不优于 highpass-band，且早中期震荡更明显，复现全频低通梯度冲突风险 | `../../output/ep07_v9_review/ep07_eval_real_metrics.png` | 已证实 |
| V9C hybrid + legal 1x highpass anchor | 需要完整 60K 曲线确认 hybrid 输入下合法 1x 锚是否仍无法关闭漂移通道 | `../../output/ep07_v9_review/ep07_v9c_metrics.csv` (TODO) | 待 V9C |

Interpretation: 当前证据足以关闭“只靠 1x forward consistency 修 V9”的路线；但 Claim 2 的 2x2 矩阵完整闭合仍需要 V9C 终点。若 V9C 60K 也不能同时改善 artifact/corr 与中心窗口保真，则 Claim 2 可从“1x 输入变体已证实”升级为“hybrid 输入变体也证实”。

## Claim 3: Prior Erosion and a Proxy-Specific (not Absolute) Pareto

**Claim.** 合成结构先验会随训练步数侵蚀真实观测细节。V9A 时间轴不是单调变好，而是在保真和锐度之间移动：20K 附近最保真但偏软，30K 以后 sharp_p95 上升同时 hp_corr_input 塌陷。在**观测保真 proxy** 上，当前 UNet 时间轴前沿不被严格优于 EP10 TGV 工作点——但这一「支配」是 **proxy 特定、非绝对**的：proxy（`hp_corr_input`、`sharp_p95`）不度量轮廓**连续性**，且 `sharp_p95` 无法区分连续锐线与珠串/颗粒。在结构 grain 读数（`lattice`）与目视上，TGV 在最细 trace 上呈 TV-staircase 珠串、学习方法呈 grain，**两者无单轴全胜**（详见 `docs/paper/reframe_c4_claim3.md`）。

| Object / Step | `hp_corr_input` | `hp_corr_tgv` | `sharp_p95` | `lattice_score` | Reading |
|---|---:|---:|---:|---:|---|
| drizzle input | 1.000 | 0.960 | 0.503 | 0.0015 | 观测域保真参考，软但不幻觉 |
| EP10 TGV | 0.960 | 1.000 | 0.959 | 0.0169 | 当前经典参照，锐但含台阶/格纹风险 |
| V9A 10K | 0.970 | 0.953 | 0.683 | 0.0010 | 可透传细结构，仍偏软 |
| V9A 20K | 0.974 | 0.944 | 0.615 | 0.0009 | 最保真，格纹被压下，但锐度不足 |
| V9A 25K | 0.935 | 0.931 | 0.831 | 0.0062 | 开始变锐，但保真已明显下降，proxy 上被 TGV 压过（仅保真/锐度轴） |
| V9A 30K | 0.908 | 0.908 | 1.147 | 0.0121 | 保真悬崖，锐度主要来自过冲/融合 |
| V9A 40K-60K | 0.906 +/- 0.001 | 0.908 | 1.21-1.25 | 0.015 | 后期平台，锐但去相关 |

| Evidence | Observation | Figure / Data | Status |
|---|---|---|---|
| Pareto scatter | TGV 位于高保真/高锐度区域；V9A 高锐度点均伴随 hp_corr_input 明显下降 | `../../output/ep07_v9_review/v9a_pareto_scatter.png` | 已证实 |
| checkpoint strip | 视觉上 30K+ 粗线变极锐，中心粗细线交融成团块；锐度提升不是可信细节恢复 | `../../output/ep07_v9_review/v9a_checkpoint_strip.png` | 已证实 |
| final panel | drizzle/TGV/V9A early/late 同口径对照显示“输入保真、TGV 折中、UNet 后期风格化”的差异 | `../../output/ep07_v9_review/fine_zigzag_final_panel.png` | 已证实 |

Interpretation: `sharp_p95` 和 Tenengrad 只能作为锐度 proxy。它们变大可能来自真实边缘增强，也可能来自振铃、饱和对比或假边缘；因此 Claim 3 必须绑定 `hp_corr_input`、视觉对照、lattice/artifact 指标和后续 split-half FRC，不能用锐度单轴宣称 SR 成功。

补充（2026-06-13 中心细线窗口复核）：`lattice` 排序为 drizzle 0.0015 < v9a_20k 0.0009 < TGV 0.0169 < v9a_60k 0.0153 < **V10 0.024**——学习方法携带的高频内容最多、且无 GT 不可验证；TGV 的珠串是 TV 正则 staircase 伪影（观测 drizzle 本身连续）。据此 **C4 从「经典支配」改写为「互补失效、无 GT 可认证赢家」**，并新增「显式 task-level 轮廓可辨偏好（非保真证据）」轴。`sharp_p95`/`hp_corr_input` 平面不含连续性轴，引用时必须声明。

## Claim 4: Explicit Fidelity-Sharpness Control

**Claim.** 将 fidelity-sharpness 权衡从“训练时长”这个隐式且失控的旋钮，改成显式设计参数，例如 V10 残差幅度惩罚 `lambda`，再检验学习方法能否越过经典 TGV 前沿。

| Planned Evidence | Success Criterion | Figure / Data | Status |
|---|---|---|---|
| V10 lambda sweep（λ=0.02/0.05/0.15, bs64, patch256） | **新判据**（旧「支配 TGV」已弃）：保真 `hp_corr_input` 高 + grain `lattice` ≤ TGV(0.0169) + 梳齿连续 | `../../output/ep07_v9_review/v10_pareto/` | **已跑 + 评估已修正**：曾因评估未把 drizzle base 加回而误判「灾难失败」；修正后落 V9A 折中区（hp_corr 0.88, sharp 1.3, lattice 0.024），**不支配 TGV**；λ 区间过弱 + bs64 混杂 → 高-λ(bs128) 待跑（`run_v10_highlam.md`） |
| zero-training fusion baseline | TGV anchor + V9A 60K 在 lambda=0.1-0.3 上支配 TGV 工作点；drizzle anchor 曲线保真高但不够锐 | `../../output/ep07_v9_review/fusion_pareto_overlay.png`; `../../output/ep07_v9_review/fusion_baseline_metrics.csv` | 已完成 |
| split-half stability | 显式控制后的候选点在 EP15 口径下稳定，而不是只在单次可视化中锐化 | `../../output/ep15_*/` (TODO) | 待 V10 / EP15 |

Interpretation（2026-06-13 更新）: V10（λ=0.02–0.15, bs64）已跑；其 fine-window 评估曾因未把 drizzle base 加回（`common.py` 漏传 `residual_channel`，已修复）而误判「灾难失败 hp_corr≈0.46/lattice≈0.08」。修正后落在 V9A 折中区、不支配 TGV，但 λ 区间过弱（惩罚损失占比仅 ~1–4%）、bs64 混杂未除，故**既非正结果也非干净反证**。同时 C4 已从「经典支配」改写为**「互补失效、无 GT 可认证赢家」**（`docs/paper/reframe_c4_claim3.md`）：TGV 有 TV-staircase、学习方法 grain 最多且不可验证。高-λ(bs128) 扫描的新判据见 `algos/ep07_unet_sr/scripts/run_v10_highlam.md`。zero-training fusion baseline 仍是「学习方法能微微越过 TGV proxy」的最干净后处理证据。

### Zero-Training Fusion Baseline

| Candidate | `hp_corr_input` | `hp_corr_tgv` | `sharp_p95` | `lattice_score` | Reading |
|---|---:|---:|---:|---:|---|
| TGV | 0.9598 | 1.0000 | 0.9593 | 0.0169 | 当前经典工作点 |
| TGV + 0.1 * V9A60 delta | 0.9626 | 0.9988 | 0.9696 | 0.0134 | 严格支配 TGV 工作点，且 lattice 低于 TGV |
| TGV + 0.2 * V9A60 delta | 0.9631 | 0.9951 | 0.9681 | 0.0108 | 当前 fusion baseline 最强候选 |
| TGV + 0.3 * V9A60 delta | 0.9614 | 0.9893 | 0.9765 | 0.0091 | 更锐，但与 TGV 去相关开始增加 |
| drizzle + 0.6 * V9A60 delta | 0.9538 | 0.9429 | 0.9328 | 0.0082 | 接近锐度目标但保真已低于 TGV |

Fusion baseline 的直接含义：V10 不只要打败单个 UNet checkpoint，还要打败“经典 TGV anchor + 少量 UNet late sharpening”的零训练前沿。这个 baseline 仍然只是后处理对照，不证明 V9A late 本身可信；`sharp_p95` 仍需 split-half FRC 和视觉检查约束。

## Conclusion Status

| Category | Conclusion | Evidence State |
|---|---|---|
| 已证实 | 1x 统计输入存在相位信息瓶颈；hybrid 2x drizzle 输入能让真实细结构进入网络并在早期 checkpoint 透传 | Claim 1 已有中心窗口定量与图证据 |
| 已证实 | 1x forward consistency 的 highpass/full-band 锚定不能阻止 v8.1/V9B/V9D 真实数据漂移 | Claim 2 在 1x 输入变体已闭合 |
| 已证实 | V9A 后期锐度提升伴随观测去相关；UNet 时间轴在**观测保真 proxy** 上不被严格优于 TGV（proxy 特定、**非绝对**：TGV 有 TV-staircase、学习方法 grain 最多，无单轴全胜） | Claim 3 已有 Pareto、条带、final panel 证据 + lattice 复核 |
| 待 V9C | hybrid drizzle 输入下的合法 1x forward anchor 是否也失败 | V9C 60K 后补指标与中心窗口诊断 |
| 已完成对照 | zero-training fusion baseline 给出 V10 必须打败的后处理前沿 | TGV + 0.1-0.3 * V9A60 delta 支配 TGV 工作点 |
| 已修正 / 待高-λ | V10（λ0.02–0.15,bs64）评估 bug 已修复：不支配 TGV、落 V9A 区；λ 过弱+bs 混杂 ⇒ 高-λ(bs128) 待跑 | `output/ep07_v9_review/v10_pareto/` + `run_v10_highlam.md` |
| 待 V10 / EP15 | 学习方法候选是否具备 split-half/FRC 稳定性，而不是单次图像锐化 | EP15 口径 FRC 小节待补 |

## Split-Half FRC TODO (EP15 Protocol)

TODO: 按 EP15 口径补充 split-half FRC，而不是继续依赖单图锐度 proxy。

建议记录项：

| Item | Planned Content |
|---|---|
| Split definition | 明确 248 clean SR-usable frames 如何分成两个互斥 half；避免跨 session 或 repeat 混合 |
| Reconstruction set | drizzle、TGV、UNet 20K、V9A late、V10 lambda candidates、fusion baseline |
| FRC metric | 频率轴单位、ring/bin 设定、阈值口径、是否 window/crop |
| Stability reading | 哪些空间频率的结构在两半中一致；哪些只是单次重建的风格化纹理 |
| Decision rule | FRC 通过后才允许把 sharp_p95/Tenengrad 增大解释为可信结构增强 |

## Open TODOs

| TODO | Owner / Trigger | Expected Update |
|---|---|---|
| 更新 C1 迁移后的真实图名 | C1 脚本迁移完成后 | 若文件名不同，修正本文所有 `../../output/ep07_v9_review/...` 链接 |
| 补 V9C 60K | V9C 完成后 | 更新 Claim 2 第三个变体、Conclusion Status |
| 补 V10 lambda sweep | V10 完成后 | 与 TGV 工作点和 zero-training fusion baseline 同时比较，从“待 V10”改为正/负结论 |
| 补 EP15 split-half FRC | EP15 口径确定并产出后 | 填写 split-half FRC TODO 小节 |

## Changed Files

- `paper/reports/ep07_v9_attribution/README.md`
