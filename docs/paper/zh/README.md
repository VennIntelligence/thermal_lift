# 论文正文 — 中文打磨稿（zh/）

> **角色**：本目录是论文**正文全 6 章的中文稿，即唯一事实来源（single source of truth）**，
> 供用户 + 会话内 AI 人机共写、逐节进修；迁 LaTeX 时由本目录中文稿直接译英。
> **工作流（2026-06-14 翻转）**：改为「**中文优先**」——英文散文权威稿 `docs/paper/02–08_*.md` **已删除**
> （删除前经逐章 parity 校验确认本目录为其完整超集，零信息丢失），不再维护双源。与 supp A–E 的中文优先工作流一致。
> **仍保留的参考件**：`docs/paper/reframe_c4_claim3.md`（C4/Claim3 改写权威底稿，中文）。
> **已删除（2026-06-15）**：`docs/paper/01_outline.md`（英文 spec，标题/摘要/contributions 已随中文稿翻新而过时；
> 其仍生效的「红队禁写清单」已下沉到本文件 §「红队禁写清单（否决权）」，叙事骨架由本目录 `zh/*.zh.md` 取代）、
> `docs/paper/06b_experimental_setup.md`（英文 setup 散文节，无中文镜像、含旧值，内容已被 `zh/04`+`zh/07` 覆盖）。
> **生成日期**：2026-06-14（Task D harness 落地后；§3.2 跨 session 数字已就地修正为 audit 权威值 2.91/4.16 °C）。

## 章节 → 文件映射

| 章 | 标题 | 中文稿（唯一事实来源） | 英文散文稿 | 状态 |
|---|---|---|---|---|
| §0 | 标题与摘要 | `01_title_and_abstract.zh.md` | 已删除 | ✅ 标题锁定 + 中英摘要 |
| §1 | 引言 | `02_introduction.zh.md` | 已删除 | ✅ 全文 |
| §2 | 相关工作 | `03_related_work.zh.md` | 已删除 | ✅ 全文（保留 `\citep` key） |
| §3 | 问题设定与标定化观测模型 | `04_problem_forward_model.zh.md` | 已删除 | ✅ 全文 |
| §4 | 方法 | `05_method.zh.md` | 已删除 | ✅ 全文 |
| ~~§5~~ | ~~无真值评估协议~~ | `06_evaluation_protocol.zh.md` | — | ⛔ 已迁移至附录（A.3/A.4/C.5/C.6） |
| §5 | 实验与消融 | `07_experiments.zh.md` | 已删除 | ✅ 全文（06-15 重组为 4 节：5.1 裁决 / 5.2 漂移 / 5.3 消融 / 5.4 预算+负结果） |
| §6 | 限制与结论 | `08_limitations_conclusion.zh.md` | 已删除 | ✅ 全文 |

## §0 标题与摘要

→ 已迁移到独立文件 **`01_title_and_abstract.zh.md`**（2026-06-14 锁定标题方向）。

- **英文标题（锁定）**：*Detailed but Drifting: Null-Space Hallucination in Ground-Truth-Free Thermal Burst Super-Resolution*
- **中文标题（锁定）**：《细节更多，却在漂移：无真值热成像多帧超分辨中的零空间幻觉》
- 旧候选 "Trust but Verify / Evidence-Bounded ..." **已废弃**（避免暗示"交付了被认证的有界最优解"；卖点改为零空间幻觉诊断 + 诚实裁决）。
- 摘要（中英双版）见该文件；其中 "we prove that observation-side anchoring is blind ..." 一句依赖 `tmp/theorem_proof_handoff/` 的 ε-零空间定理，待回填。

## 进修须知（接手者必读）

1. **本目录中文稿是唯一事实来源**：改论点/数字只改本目录（英文散文稿 02–08 已删除，不再双源）；迁 LaTeX 时由中文稿直接译英。
2. **红队边界具否决权**（见本文件末节「红队禁写清单（否决权）」）：不得写「打败 TGV / 更干净 / 更保真 / 计量级」；
   sharp_p95 必与 lattice + 视觉并报；contour legibility 只作 task-level 主观；TGV 必带 TV-staircase caveat。
   任何修改后的段落与图注都要过一遍这份清单；冲突时删主张，不删边界。
3. **两套尺度不混表**：T1/T2 全用 EP11/harness scale；TB-scale `eval_real/*` 仅 §5.2 轨迹图用、图注声明。
4. **仍待落地的写作侧缺口**（不挡进修）：① §2 `refs.bib` 与 `\citep` key 对齐；② 主文手工图已引用：`02`→`fig_nullspace_drif.pdf`（图 1/F0）、`04`→`observation_pipeline.pdf`（图 4）、`05`→`fig_tcforge_pipeline.pdf`（F-method）；其余 F2–F7 仍见 `09_figures_tables_assets.md`；
   ③ Task E 完成后 T1 的 TGV 真值 split/FRC、F5 第二 ROI、D.7 第二验证窗回填本目录对应处。

## 红队禁写清单（否决权）

> 本清单从已删除的 `01_outline.md` 末节「Claims to avoid」迁入（含 `reframe_c4_claim3.md` §5 的扩充），
> 是**当前唯一权威版本**，对正文/图注/摘要具否决权。任何修改后的段落都要过一遍；冲突时删主张，不删边界。

- 不得声称 5 µm 空间分辨率或温度计量级结论；5 µm 输出采样 ≠ 5 µm 分辨率；2x 网格 Nyquist 周期是 10 µm。
- 不得声称 4x/5x 物理恢复；更细网格只是轮廓过采样（contour oversampling）。
- stage command / 文件名坐标 / EP04 锚点都是**先验与质控门，绝不是对齐真值**。
- FRC 高频回弹（10–12 µm band）只标注为覆盖/lattice + 漂移风险，**不得**当作孔径零点或分辨率证据；cutoff 主张固定在 17.0 µm。
- proxy 指标（split-half、artifact score、raw-control corr、Tenengrad、Chamfer）**绝不单独证明 SR 成功**，只用于门控与选点。
- TGV（及同属 TV 家族的 MAP-TV）在最细对角轮廓上有轻微 staircase/beading 小瑕；**绝不**把经典重建写成无条件「可信交付物」赢家，必带此 caveat；不存在 GT-可认证的单一赢家。
- 学习输出目视更锐，但携带最多高频内容（最高 `lattice`），无 GT 不可验证；**绝不**称其「更干净」或更保真——目视偏好 ≠ 保真证据。
- `sharp_p95` 对轮廓连续性失明、会被 beading 与颗粒同时推高；**绝不单独引用锐度**——必并报 `lattice` + 双域视觉 gate。
- 任何来自目视的「contour legibility」增益是显式 task-level 主观偏好，与可验证 proxy 分列，**绝不**当作分辨率或保真证据。
- MAP-TV zigzag 增益是「有限的轮廓增强」（per-profile 混杂），不是强阳性结果。
- 不得跨输入模式比较 proxy 数值（1x-stat vs hybrid drizzle 输入）。
- 不得混用跨 session 帧；不得把渲染后的 AVI/BMP 当作数值 SR 输入。
- v8.1b（PixelShuffle）与 EP12 4x 作为**负结果如实上报**，不得静默丢弃。

## 图表计划（F1–F7 + T1/T2）

> 同样从已删除的 `01_outline.md` 迁入；资产路径与生产状态以 `docs/paper/09_figures_tables_assets.md` 为准。

- F1 系统 + 标定链 + raster/微扫描示意（含 pitch/分辨率/输出网格区分）
- F2 信息存在性：FRC 曲线（带控制组）+ band 表
- F3 零空间漂移：各变体轨迹（artifact、corr vs step）+ forward-loss 贴底 inset
- F4 proxy Pareto + checkpoint 选点（TGV 参照点）
- F5 主视觉对比：drizzle / TGV / V9A-late / V10 residual（温度 + highpass），含中心梳齿高倍裁剪
- F6 输入模式消融视觉：中心细线、边缘 staircase（v8.1a vs v9a vs v9c）
- F7 frame-budget 与鲁棒性曲线
- T1 主定量表（单口径 harness 数字）；T2 消融矩阵（input × anchor）
