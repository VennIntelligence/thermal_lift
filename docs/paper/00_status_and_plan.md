# 论文工作区总控（docs/paper/）

> 本目录是论文的 **markdown 草稿与素材整理区**：先在这里把论点、数字、图表资产钉死，
> 再迁移到 `paper/aaai/`（或目标会议模板）的 LaTeX。
> 旧的 EP01–EP06 时代 LaTeX 骨架在 `paper/`（AAAI kit + SAA/IBP/MAP-TV 故事），
> 其 `paper/notes/aaai_outline.md` 已被本目录 `01_outline.md` 取代（叙事升级为 EP07–EP15 全线）。

## 一句话主张

无 ground truth 的多帧热成像增强，什么时候可信？我们用一套全标定 forward model、
一个无 GT 评估协议、以及「先验漂移活在 forward 算子零空间」的实证发现来回答，
并据此构造 2x contour-level 可信增强系统。

## 文件结构

| 文件 | 内容 | 语言 | 状态 |
|---|---|---|---|
| `00_status_and_plan.md` | 总控：状态、计划、待补实验 | 中文 | 持续更新 |
| `01_outline.md` | 论文大纲 + contributions + 禁写边界 | EN | 初稿 |
| `02_introduction.md` | Introduction 草稿 | EN | 初稿 |
| `03_related_work.md` | Related work 分桶 + 占位引用 | EN | 初稿 |
| `04_problem_forward_model.md` | 系统、数据、标定链（§3） | EN | 初稿，数字已核 |
| `05_method.md` | 合成平台、网络、锚定、输入设计（§4） | EN | 初稿 |
| `06_evaluation_protocol.md` | 无 GT 评估协议（§5） | EN | 初稿 |
| `07_experiments.md` | 实验与消融（§6），含待补占位 | EN | 初稿，待回填 |
| `08_limitations_conclusion.md` | 限制与结论（§7） | EN | 初稿 |
| `09_figures_tables_assets.md` | figure/table → 仓库资产路径映射 + 生产状态 | 中文 | ✅ 已落盘（06-11） |
| `10_writing_handover.md` | **写作交接总控**：逐节要点 + supp 材料清单 + 分派状态 | 中文 | ✅ 已落盘（06-11） |
| `supp/A–E_*.md` | Supplementary 草稿（理论/数据/方法/结果/复现） | EN | ⬜ 按 10 号文档清单起草 |

## 投稿靶子（2026-06-11 决策）

- **主文按 8 页 AAAI 版式写**（7 页技术 + 引用页），supp 走 technical appendix；
  写作不绑定 venue，定档时只调版式与匿名项
- WACV 2027 R2（注册 8/21、截稿 8/28）仍为节奏锚；**AAAI 可投 2028 届**
- ICIP 2027：11/1；ICPR 2027：明年 3/1；PBVS@CVPR2027：明年 3 月（备胎）

## 实验状态板（写作依赖）

| 实验 | 论文角色 | 状态 |
|---|---|---|
| 经典锚：drizzle / MAP-TV / TGV（EP10）、M4 去卷积锚（EP15） | §6 主表 + 经典基准 | ✅ 完成 |
| 信息上限：M1 相位 / M2 FRC / M3 σ 仲裁（EP15） | §5 协议 + §3 边界 | ✅ 完成（带风险标注） |
| UNet 四臂 + checkpoint 选优（v6/v8.1a/v8.1b/v9b） | §6 漂移轨迹 + Pareto | ✅ 完成（reports/ep11_dl_benchmark） |
| V9A hybrid drizzle 输入 | §6 输入消融主结果 | 🔄 GPU 0 训练中，预计 06-12 早 ~6 点 |
| V9D 全频锚 / V9C hybrid+合法锚 | §6 锚定消融 | 🔄 已派 Codex（GPU 1 串行 ~20h） |
| 帧数预算 N ∈ {31,62,124,248} | §6 消融（纯推理） | 🔄 经典臂已派 Codex（EP16，`todos/paper_prompts.md` Task C）；GPU 臂待空闲 |
| shift 扰动鲁棒性 σ 扫描 | §6 消融（纯推理） | 🔄 同上（Task C E2） |
| 对齐源消融（command prior vs contour_refined） | §6 消融（纯推理） | 🔄 同上（Task C E3） |
| 论文图 F1（系统+标定链）/ F3（drift+inset） | §3 / §6 主图 | 🔄 已派 Codex（Task A / Task B） |
| PSF σ 敏感性 | §6 消融 | ✅ M4 网格已有，整理即可 |
| 指标口径统一重跑（同一 harness 出全部终表数字） | §6 主表 | ⬜ 必做：TB eval 与 EP11 横评的 artifact 尺度不同，终稿数字必须单一口径 |

## 写作计划（对应 WACV R2 节奏；分工与领取入口见 `10_writing_handover.md` §4）

1. **W1–2**：等消融臂落地；其间把 §3/§5（材料最厚）从草稿打磨到可投级；图 1（系统+标定链）与图 4（漂移轨迹/Pareto）定稿
2. **W3–4**：§6 回填全部数字（统一口径 harness）；§2/§4 定稿；摘要/标题定稿
3. **W5–6**：LaTeX 迁移（更新 `paper/` 模板为目标会议 kit）、supplementary（复现包说明）、内部红队过一遍 §「禁写边界」
4. **W7–8**：打磨 + 提交

## 待办与风险

- [ ] 客户许可：论文展示芯片热像（可脱敏：匿名型号、仅中心 ROI）——**本周去问**
- [ ] 指标口径统一（见状态板最后一行）
- [ ] V9A 若证伪输入论点，§6 叙事降级为「锚定与输入双阴性 + 协议仍然成立」——协议与零空间发现不依赖 V9A 阳性
- [ ] 参考文献：`03_related_work.md` 目前是 `[REF: ...]` 占位，迁 LaTeX 前补 `paper/aaai/refs.bib`
