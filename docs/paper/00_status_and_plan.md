# 论文工作区总控（docs/paper/）

> 本目录是论文的 **markdown 草稿与素材整理区**：先在这里把论点、数字、图表资产钉死，
> 再迁移到 `paper/aaai/`（或目标会议模板）的 LaTeX。
> 旧的 EP01–EP06 时代 LaTeX 骨架在 `paper/`（AAAI kit + SAA/IBP/MAP-TV 故事），
> 叙事骨架现以本目录 `zh/*.zh.md` 中文稿为唯一事实来源（旧 `paper/notes/aaai_outline.md` 与英文 spec `01_outline.md` 均已废弃/删除，升级为 EP07–EP15 全线）。

## 一句话主张

无 ground truth 的多帧热成像增强，什么时候可信？我们用一套全标定 forward model、
一个无 GT 评估协议、以及「先验漂移活在 forward 算子零空间」的实证发现来回答，
并据此构造 2x contour-level 可信增强系统。

> ## 🟢 写作就绪状态（2026-06-14）
>
> **所有过夜消融变体已落地，科学结论已定，可以开始写作。** C1–C4 四条贡献全部 settle：
> V10 高-λ sweep 完成（把 Claim 4 从「凌乱负结果」升级为「可控权衡 + 有价值工作点 λ=1.2@15K」，
> 仍诚实裁决「no GT-certifiable winner」）；V9C/V9D 60K 完成并闭合 input×anchor 矩阵 + null-space drift；
> EP16 经典方法完成；论文图 F1/F2/F3/F5/F7 + method arch 全部生成。
> **统一口径 harness 已完成（Task D，2026-06-14）**：T1/T2/F5 已有单一口径产物
> `output/ep11_unified_harness/{t1_metrics.csv,t2_metrics.csv,run_manifest.json}` 与
> `output/paper_figures/fig05_main_visual.{png,pdf}`。剩余工作转为写作压缩、图注和参考文献。

## 文件结构

| 文件 | 内容 | 语言 | 状态 |
|---|---|---|---|
| `00_status_and_plan.md` | 总控：状态、计划、待补实验 | 中文 | 持续更新 |
| `reframe_c4_claim3.md` | C4/Claim3 改写权威底稿（保留） | 中文 | ✅ 权威 |
| ~~`01_outline.md`~~ | 已删除（2026-06-15）；红队禁写清单 + 图表计划迁入 `zh/README.md`，叙事骨架由 `zh/*.zh.md` 取代 | — | ❌ 已删 |
| ~~`06b_experimental_setup.md`~~ | 已删除（2026-06-15）；英文 setup 散文节内容由 `zh/04`+`zh/07` 覆盖 | — | ❌ 已删 |
| **正文 §0–§7（中文稿 = 唯一事实来源）** | `zh/01_title_and_abstract.zh.md` + `zh/02–08_*.zh.md` | 中文 | ✅ 全文；英文散文稿 02–08 **已删除**（parity 校验后零丢失） |
| `09_figures_tables_assets.md` | figure/table → 仓库资产路径映射 + 生产状态 | 中文 | ✅ 已落盘（06-11） |
| `10_writing_handover.md` | **写作交接总控**：逐节要点 + supp 材料清单 + 分派状态 | 中文 | ✅ 已落盘（06-11） |
| `supp/A–E_*.md` | Supplementary 草稿（理论/数据/方法/结果/复现） | 中文（迁 LaTeX 译英） | ✍️ 五份已落盘（06-12），缺口 ⬜ 占位；新增 D.0 fine-window 口径 + D.7 融合 baseline 专节 |

## 投稿靶子（2026-06-11 决策）

- **主文按 8 页 AAAI 版式写**（7 页技术 + 引用页），supp 走 technical appendix；
  写作不绑定 venue，定档时只调版式与匿名项
- WACV 2027 R2（注册 8/21、截稿 8/28）仍为节奏锚；**AAAI 可投 2028 届**
- ICIP 2027：11/1；ICPR 2027：明年 3/1；PBVS@CVPR2027：明年 3 月（备胎）

## 实验状态板（写作依赖）

| 实验 | 论文角色 | 状态 |
|---|---|---|
| 经典基准：drizzle / MAP-TV / TGV（EP10）、M4 去卷积基准（EP15） | §6 主表 + 经典基准 | ✅ 完成 |
| 信息上限：M1 相位 / M2 FRC / M3 σ 仲裁（EP15） | §5 协议 + §3 边界 | ✅ 完成（带风险标注） |
| UNet 四个变体 + checkpoint 选优（v6/v8.1a/v8.1b/v9b） | §6 漂移轨迹 + Pareto | ✅ 完成（paper/reports/ep11_dl_benchmark） |
| V9A hybrid drizzle 输入 | §6 输入消融主结果 | ✅ 60K 完成；canonical V9A@10K 已入 harness（artifact/corr 1.762/0.719），V9A@60K 用作 F5 late-drift visual control |
| V9D 全频锚 | §6 锚定消融 | ✅ 60K 完成；canonical V9D@7K 已入 harness（1.726/0.771），TB 端点 0.677/0.677 |
| V9C hybrid+合法锚 | §6 锚定消融 | ✅ **60K 完成（2026-06-14）**；canonical V9C@5K 已入 harness（1.669/0.718），TB 端点 0.695/0.669，同样漂移未压平 → 闭合 input×anchor 矩阵 |
| V10 residual-over-observation（高-λ sweep） | §6 Claim 4 | ✅ **高-λ {0.2,0.5,1.2,3.0}×25K 完成（2026-06-14）**；best fine-window λ=1.2@15K (0.922/0.987/0.014)，harness row artifact/corr 2.726/0.711，23°C sanity pass；Phase 2 跳过 |
| 帧数预算 N ∈ {31,62,124,248} | §6 消融（纯推理） | ✅ **经典方法完成（EP16）**；§6.4 数字已回填；learned/GPU matrix 不再是主文硬门槛 |
| shift 扰动鲁棒性 σ 扫描 | §6 消融（纯推理） | ✅ **经典方法完成（EP16 E2）**；§6.5 已回填 |
| 对齐源消融（command prior vs contour_refined） | §6 消融（纯推理） | ✅ **经典方法完成（EP16 E3）**；§6.5 已回填 |
| 论文图 F1（系统+标定链）/ F3（drift+inset） | §3 / §6 主图 | ✅ **完成**（`output/paper_figures/fig01_*`、`fig03_nullspace_drift*` + `fig03s_v9a_trajectory*`） |
| PSF σ 敏感性 | §6 消融 | ✅ M4 网格已有，整理即可 |
| **指标口径统一重跑（同一 harness 出全部终表数字）** | §6 主表 | ✅ **完成（Task D，2026-06-14）**：`algos/ep11_dl_benchmark/scripts/run_unified_harness_t1_t2.py` → `output/ep11_unified_harness/{all_arm_metrics.csv,t1_metrics.csv,t2_metrics.csv,tb_vs_harness_scale_check.csv,run_manifest.json}`；F5 已生成 |

## 写作计划（对应 WACV R2 节奏；分工与领取入口见 `10_writing_handover.md` §4）

1. **W1–2**：等消融变体落地；其间把 §3/§5（材料最厚）从草稿打磨到可投级；图 1（系统+标定链）与图 4（漂移轨迹/Pareto）定稿
2. **W3–4**：§6 回填全部数字（统一口径 harness）；§2/§4 定稿；摘要/标题定稿
3. **W5–6**：LaTeX 迁移（更新 `paper/` 模板为目标会议 kit）、supplementary（复现包说明）、内部红队过一遍 §「禁写边界」
4. **W7–8**：打磨 + 提交

## 待办与风险

- [ ] 主文数字微修（supp 起草时核出，见 `supp/B_system_data.md` B.1.2 / `supp/D_full_results.md` D.5.1）：
  ① §3.2 跨 session 跳变「median 3.55 °C (49×)」→ audit 权威「median 2.91 / max 4.16 °C (40×/57×)」，`AGENTS.md` 同步；
  ② §6.4「corr 0.772 at N=248」→ CSV 原值 0.771
- [ ] D.7 融合 baseline 的 λ 选择需第二验证窗复核（CPU 几分钟，防 selection-on-test）
- [x] 指标口径统一（Task D，`output/ep11_unified_harness/`）
- [ ] §6 叙事保持最终边界：hybrid 输入暴露 sub-pixel evidence，但不认证 learned fidelity；协议与零空间发现不依赖 V9A 阳性
- [ ] 参考文献：`zh/03_related_work.zh.md` 目前是 `\citep` 占位，迁 LaTeX 前补 `paper/aaai/refs.bib`
