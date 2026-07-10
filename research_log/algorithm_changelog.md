# 算法变更日志 (Algorithm Changelog)

> **用途**: 每次对 SR 算法（网络架构、Loss 设计、训练策略、数据管线）做出修改时，必须在此记录变更条目。  
> **格式**: 按时间倒序排列，每条变更必须包含：问题诊断、修改内容、预期效果、训练结果。  
> **规则**: 见 `AGENTS.md` 中「算法变更日志规则」。

---

## ✅ 当前有效结论速览（2026-07-07 快照 — 读旧条目前先读这里）

旧条目是历史记录，其中的判断随后可能被推翻；与本块冲突时**以本块与其引用的 ACL 为准**：

1. **神经臂与经典同档，"神经带内破坏"已被推翻**（ACL-049 v4）：solver 输出网格带 +0.5 HR px 角点约定（`forward_torch.py`，设计而非 bug），此前所有神经×经典对比被配准伪影压低 ~0.44-0.49 FRC 点。校正后 V11 cutoff 与 TGV 打平（23.03µm），@30µm 落后 0.05-0.06。ACL-046/047/048 中对神经臂 cross-FRC 的定罪段落均已过时。
2. **对齐**：精修对齐是 repo 默认资产（`configs/alignment/stage0f_refined_alignment.csv`，ACL-048）；旧 contour_refined 有 ~0.29px 逐帧误差，勿再用其得出的绝对结论。
3. **权威可恢复频带 = 25.45µm ±0.73**（精修对齐下 EP15 M2，ACL-048）；旧值 34µm（坏对齐）与更早 EP15 数字作废。真实增益带 ~25-40µm 周期；**20µm 是探测器孔径零点，任何 20µm 处的 FRC 值不采信**。
4. **判据管线**：神经×经典比较必须先做逐半偏移校正（`probe_pair_offset.py --save-corrected-dir` 或 `real_eval.to_center_grid`）再算 cross-FRC；self split-half FRC 对神经方法无效（奖励可复现幻觉，ACL-047）；synth PSNR 只是 sanity 下限（ACL-032）。
5. **DR@0.1px 关案**（ACL-049）：C vs D 三重仪器零差。如再讨论 DR，量级为实测残余 ~0.05px。
6. **0d 回归套件**：仅 extent 探针有好/坏判别力（ACL-047），阈值重定权在 owner，尚未落定。
7. **经典基线（要打败的对象）**：TGV×drz cutoff 23.03µm / FRC@30µm 0.702；MAP-TV×drz 24.62µm / 0.690（精修对齐+校正口径，ACL-049）。
8. **基础设施**：`remote_inbox/` 只走 rsync/scp 增量，严禁 git（AGENTS.md 硬规则）。
9. **冠军臂裁决（ACL-053 回填）**：η0.09+band 的 50k/batch8 冠军 0.647@30µm **未达标**，且较 20k/batch4 组合臂倒退 0.023；synth↑/real↓ 反相关再现。神经最佳纪录 = 20k 组合臂 **0.6705**（gap 0.031）。单数据集调参终止，方向转 **Stage 2b 合成基准**（TGV 首次上合成集，判决 H1 域差 vs H2 架构）。
10. ⚠️（已解决，见 #11）**V7 池点保真度崩溃，确认是数据问题非训练问题（ACL-065/066/067，2026-07-08 夜）**：depb9v6 配方在 v7 池复训后 real cross-FRC 小幅改善（0.674/0.676@30µm）但孤立点 erased% 4.7%→~43%；batch/patch 对照实验排除训练超参，病根锁定 v7 缺陷分布设计。**v7 池已删除（2026-07-09，owner 令）。**
11. **V8 池 = 当前生产训练池，点保真崩溃已修复（ACL-068/069/070，2026-07-09 整夜自治）**：`data/synthetic/pool_2x_v8_5k`（5000 景，seed 20260911）= v7 仅改两旋钮——密度 min/max_holes 20-50→**2-8**、深度下限 hole_depth_range 0.3→**0.55**。验证臂 `solver_v25_depb9v8_9bin_30k`：**isolated erased% 4.35%（v6 基线 4.66%，v7 病理 43%）**，ALL retention 0.547。机制发现：(a) "v7 点物理不可见"被 L1 审计证伪，病理角落=半径 1-2px×深度 0.3-0.5（ACL-068）；(b) 抹除先验需要池规模级数据多样性才形成——300 景 24k 步全程 0% 抹除，故 300 景廉价消融对此病理是盲的（ACL-069）；密度-vs-深度归因保持开放。**新池验收制度**：pilot 必过 tcforge gates + `scripts/audit_defect_detectability.py`（L1，已验证有效）+ 生产级验证训练；配置不得含未标定占位参数。三轴选型（ACL-071）无 v8 champion。**归因矩阵已闭环（ACL-072）：v9 池（=v8 恢复 v7 密度 20-50、深度下限 0.55 保留、seed 与 v8 配对）为当前生产池（v8_5k 已删）**——浅深度是点抹除元凶（实锤），密度无罪反有益：depb9v9_9bin erased **1.55%**/retention **0.609** 双双史上最优，FRC 0.625（+0.019 vs v8，未收复 v7 的 0.674，跨代对比含 seed 混杂）；**bin4 配方在密集小点池上低频灾难发散（range_exc ~10³）出局**。champion 权衡待 owner：v9_9bin（点保真绑定）vs 老 depb9v6（FRC 0.661 优先）。未决：v8/v9 代 range_exc 系统性偏高（12-13 vs 历史 1.6-4.4）机制不明。held-out 基准池 `pool_2x_v8_bench48`（seed 20260912）勿训。

---

## 变更记录

### [ACL-074] 2026-07-10（晚，abchain 自动收官）— Track A/B 双判决：**range_exc 低频劣化 + v7 的 FRC 优势都是配方因子，seed 洗清**；3K 池臂点保真史上最优（isolated erased 0.00%/retention 0.798）且真实 FRC 打平 5K——**池规格降 3K 不获批（合成基准不平），但 depb9v9_3k 成为 champion 候选新帕累托点**

**执行**: 固化链 `tmp/ab_chain.sh`（tmux abchain）全自动完成：Track A gen pool_2x_v9s2_5k（v9 配方 × v7 的 seed 20260902）→ 训 solver_v27_depb9v9s2_9bin_30k → 删池；Track B gen pool_2x_v9_3k（v9 配方、同 seed 20260911、3000 景）→ 训 solver_v27_depb9v9_3k_9bin_30k → 删池；渲染 + leaderboard + stage2b 双基准。TGV 控制锚逐位复现（0.7017/23.03µm，第 9 次）；probe sanity 臂 depb9v6 逐位复现（0.5984/4.66%）——两台仪器都可信。probe 阶段在 Mac 跑（渲染 base64-over-ssh 拉回 `remote_inbox/20260710_expab/`）。

**读数**（真实 cross-FRC = corrected vs drizzle；stage2b @n=96 = v8bench/v6bench；点保真 = ALL retention / isolated erased%）:
| 臂 | FRC@30µm | cutoff | band_mean_25_40 | range_exc | retention | erased% |
|---|---|---|---|---|---|---|
| depb9v9s2（A：seed 换 v7 的） | 0.6225 | 25.82µm | 0.385/0.417 | **13.64/15.55** | 0.684 | 1.55 |
| depb9v9_3k（B：3000 景） | 0.6245 | 26.12µm | 0.332/0.385 | 14.91/12.52 | **0.798（史上最优）** | **0.00（史上最优）** |
| 参照 depb9v9_9bin（5k） | 0.6252 | 25.82µm | 0.450/0.455 | 13.02/12.08 | 0.609 | 1.55 |
| 参照 depb9v6 / tgv | 0.661 / 0.7017 | 25.45 / 23.03µm | — / 0.650/0.765 | 2.70 era / 2.10/2.39 | 0.598 | 4.66 |

**判读**:
1. **Track A（seed 归因）双关闭**: (a) range_exc@96 = 13.6/15.6，没有掉回 v7 时代的 ~2 → **v8/v9 代低频劣化是配方（深度下限 0.55 / 缺陷分布）因子，seed/内容洗清**；(b) FRC@30µm = 0.6225 ≉ 0.674 → **v7 的 FRC 优势也不是 seed 混杂，是 v7 配方本身**（浅密小点族的高频纹理贡献）——ACL-072 判读 2 开放项关闭：FRC 与点保真的张力在数据配置层就存在（同一浅点配置既送 FRC 又毁点）。附带：同配方同规模换 seed 在合成基准上差 band ~0.065（0.385 vs 0.450）→ seed 对合成基准有中等内容效应，但解释不了上述两项。
2. **Track B（3K 对照）三轴不平 → 降规格不获批，但结构有趣**: 真实 FRC 打平（Δ0.0007），合成 band 明显差（−0.10~0.12），点保真反而**大幅更好**（0.00%/0.798 vs 1.55%/0.609）。机制与 ACL-069 一致并成为其第二数据点：**抹除先验强度随池规模/多样性增长**——3K 先验更弱 → 抹除更少 + 合成带内匹配更差。池规模本身就是 FRC↔点保真张力轴上的一个旋钮。
3. **champion 备选格局更新（owner 拍板材料）**: depb9v9_3k 在真实域两轴（FRC 打平、点保真占优）**支配** depb9v9_9bin，仅合成基准 band 落后；帕累托前沿现为 depb9v6（FRC 0.661 绑定）vs depb9v9_3k（点保真 0.00%/0.798 绑定）vs tgv（0.7017）。checkpoint solver_v27_depb9v9_3k_9bin_30k 在远端保留。

**开放**: range_exc 劣化已定位到配方层但机制未明（深度下限 0.55 vs 缺陷密度/分布的哪一半，需 depth-floor 单变量小消融）；OOD 退化曲线（oodchain，明早）出来后与 champion 问题一起交 owner。— OOD content 轴生成器缺口补齐：4 个 out-of-grammar motif 族（organic_blobs / text_serial / concentric_rings / voronoi_cells）落地，content 轴第二批池配置就绪

**背景**: 首轮 OOD 套件（seeds 20260920-28，跑于 oodchain）的 content 轴因生成器代码缺口（`research_log/ood_generator_support_audit_draft.md` 任务1 A1：新 motif 族需要新代码分支）退而用了 4 档 config 可达阶梯（texroles/xlmerge/legacymix/tracebus），设计稿 A1 的 4 个新几何语法族被挡在外（handoff 20260710 §4 开放项 3）。本条补上代码缺口。

**实现**（计划 `research_log/ood_content_motif_families_plan.md`，fork agent 于隔离 worktree 实现，主循环独立验收后合并，commit b3363cd + merge）:
- `tcforge/src/tcforge/geometry.py`: `CPU_SCENE_FAMILIES` 注册 4 新族；`_compose_cpu_scene` 入口新增 motif_weights 未知键校验（此前未知键**静默落到 generic**——现 raise ValueError）；4 个 bbox 局部光栅化分支。全部特征守频带诚实下限（宽 ≥ FLOOR 28µm、周期 ≥ pitch_floor 32µm；椭圆环按短轴度量执行 FLOOR/ratio 更严；Voronoi 通道垂直宽度下界 = channel_w ≥ FLOOR；7 段字形净空硬约束 h≥3t+2F、w≥2t+F）。
- **字节级兼容已验证**: family 派发前零新增 rng 抽样、原 6 族分支一字未动、共享 clutter 段的门控是纯键名判断（零 RNG）；golden 测试 `test_scene_mask_legacy_and_v6_paths_match_golden` 通过 = 现有池逐字节不变。
- **偏离计划**（详 `research_log/ood_content_motif_families_impl_notes.md`）: (1) 密度上调（首版 occupancy 0.3-0.5% 会让 48 景带限 FRC 噪声主导 → blobs ~2-3%、text ~1-2%、rings ~8-13%、voronoi ~38-42%）；(2) 共享 CPU clutter（passives/vias/edge-IO）对 4 新族**关闭**——分布内零件内容会稀释 content 轴语法距离。
- 测试 116/116 绿（主循环亲跑；113 现有 + 3 新增：生成+确定性+元数据、12 seeds×4 族 floor 元数据审计、未知族名 raise）。预览目测通过（`tmp/motif_previews_20260710/`，含预览脚本）：环系无莫尔、Voronoi 通道下限成立（宽窄不一=双曲展宽固有几何）、斑块平滑带孔、7 段字形清晰（raised/engraved 两模式）；text 预览中一处内接圆边界裁剪楔形碎片 = pipeline 既有行为（现有族同样被裁），非新代码缺陷。

**池配置**（4 个，未生成——排在 oodchain 之后）: `pool_2x_ood_content2_{organicblobs,textserial,rings,voronoi}.json`，seeds 20260930-33（已 grep 确认与全部历史 seed 不相交），逐字拷贝 legacymix 模板仅改 5 个预期键（程序化断言）。EVAL-ONLY，冻结 checkpoint 纯推理（预注册规则 1），ID 锚沿用 v8_bench48 已有 stage2b 行。

### [ACL-072] 2026-07-10（清晨，v9 链自动收官）— 归因矩阵第三角落定：**浅深度=点抹除元凶（实锤），密度对点保真无害甚至有益**；depb9v9_9bin 点保真史上最优（erased 1.55%/retention 0.609）但 FRC 0.625 未收复 v7 的 0.674；**bin4 配方在 v9 上低频灾难性发散（range_exc ~10³）出局**

**执行**: 固化链 `tmp/v9_chain.sh` 全自动完成（含夜间一次 ~15min Tailscale 掉线，机器未重启、链条自愈无损）：pilot L1 闸通过 → v8_5k 删除（owner 令：不留两个 236G 池）→ v9 5000 景生成（seed 20260911 与 v8 逐景配对，唯一差异 min/max_holes 2-8→20-50）→ solver_v26_depb9v9_{9bin,bin4}_30k 串行训毕 → 三轴评测。仪器锚三重复现（tgv 0.7017/23.03；depb9v6 0.6611；0.5984/4.66%）。

**三轴读数**（本地 `remote_inbox/20260718_v9_verdict/`）:
| 臂 | FRC@30µm | cutoff | isolated erased% | retention | range_exc@v6b |
|---|---|---|---|---|---|
| depb9v9_9bin | 0.6252 | 25.82µm | **1.55%（史上最优）** | **0.609（史上最优）** | 12.08 |
| depb9v9_bin4 | 0.6716 | 22.80µm | 30.43% | 0.381 | **1233（灾难）** |
| 对照 v8_9bin / v8_bin4 | 0.606 / 0.668 | 26.12 / 22.80 | 4.35% / 31.06% | 0.547 / 0.371 | 13.7 / 8.4 |
| 对照 depb9v6 / TGV | 0.661 / 0.702 | 25.45 / 23.03 | 4.66% / — | 0.598 / — | 2.70 era / 2.39 |

**判读（按矩阵）**:
1. **浅深度→点抹除：实锤**。v9（密+深）erased 1.55%——不仅没回到 v7（密+浅）的 43%，反而好于 v8（疏+深）的 4.35% 和 v6 的 4.66%。**密度对点保真无罪，深度下限 0.55 是治点的药**；且"密集但可探测"的点反而教会网络更好地保留（正是 ACL-063 建 v7 的原始直觉，只是当年被浅深度毁了）。ACL-069 的开放归因就此关闭一半。
2. **密度→FRC：只兑现了一小部分**。同 seed 配对下密度恢复给 9-bin +0.019（0.606→0.625），远不到 v7 的 0.674。v7 的 FRC 优势另有来源：候选=浅深度小点族本身的高频纹理贡献，或 v7 池 seed（20260902）与 v8/v9（20260911）的内容差异——**跨代 FRC 对比含 seed 混杂**，此项归因仍开放。
3. **bin4 出局**：在 v9 上 range_excursion 爆到 1048/1233（meandc_eta4 级别的低频发散，ACL-062 同款病），且点抹除 30% 不变。4-bin 配方与密集小点池不相容。
4. **champion 权衡（owner 决策项）**：depb9v9_9bin vs 老 depb9v6 = 点保真大胜（1.55% vs 4.66%，retention 0.609 vs 0.598）/ FRC 小负（0.625 vs 0.661）。若绑定轴是真实芯片的点保真（ACL-063 的 owner 可见缺陷），v9_9bin 是更好的产品臂；若 FRC 优先，老 depb9v6 仍守擂。**v9 池无论如何是迄今最好的训练池**（两臂点保真上限均由它创造）。
5. **未决异常**：v8/v9 两代 9-bin 的 range_exc（12-13）均显著高于 v6/v7 时代（1.6-4.4），疏密无关——共同因子是深度下限或 seed，机制未解，下一个值得查的线索。

**涉及文件**: 远端 `outputs/solver_v26_depb9v9_{9bin,bin4}_30k/`、`output/{stageV26_v9sel_leaderboard,dot_probe_v9sel}/`、stage2b 两基准追加行；本地 inbox `20260718_v9_verdict/`。v8_5k 已删；v9_5k（236G）为当前生产池。

### [ACL-071] 2026-07-09（傍晚，固定化链自动执行）— v8 champion 三轴选型：**无全域赢家，FRC↔点保真权衡在 v8 上重现并放大**；bin4 cutoff 22.80µm 首次超 TGV 但孤立点抹除 31%；9-bin 点保真健康（4.35%）但 cross-FRC 退化到 0.606；**老 depb9v6（v6 池）仍是双轴最优神经臂**

**执行**: owner 指示"固定化代码+后台自跑"。整链固化为 `~/thermal_lift/tmp/champ_chain.sh`（远端）：S0 md5 预检 → S1 tgv_oracle@v8bench48（CPU，与 bin4 训练并行）→ 等训练 → S2 渲染三臂 → S3 cross-FRC 排行榜 → S4-S6 Stage2b@v8bench48 → S7 v8 臂@v6bench48（考古发现 v7 轮 stage2b 实际跑在 v6 bench48 上，故 S7 使 v8 臂与全部历史同池直比）。全链无人值守完成，逐步落盘标志。`solver_v25_depb9v8_bin4_30k`（30k，b8/p384，4-bin ontf，与 v22 bin4 配方逐旗标一致）当日训毕。

**仪器锚（三个全部精确复现，本轮数字可信）**: tgv_x_drz 0.7017@30µm/23.03µm；depb9v6_x_drz 0.6611/25.45（新渲半幅）；v6bench48 tgv__oracle band_FRC 0.76502/range_excursion 2.38760（逐位）。

**三轴读数**（本地 `remote_inbox/20260717_v8_champion/`）:
| 臂 | FRC@30µm | cutoff | isolated erased% | band_FRC@v6b | range_exc@v6b |
|---|---|---|---|---|---|
| depb9v8_9bin | 0.6057 | 26.12µm | 4.35% | 0.437 | 13.75 |
| depb9v8_bin4 | 0.6676 | **22.80µm** | 31.06% | 0.787 | 8.40 |
| depb9v6（历史，v6 池） | 0.6611 | 25.45µm | 4.66% | — | 2.70（ACL-062 era） |
（v8bench48 同向：band_FRC 0.418/0.871/oracle 0.650；range_exc 12.1/43.5/oracle 2.10。）

**判读**:
1. **v6 时代的"bin4 用点保真换 FRC"权衡在 v8 上重现且放大**（v7 上是两臂一起崩没有权衡）：bin4 拿到史上最好神经 cutoff（22.80µm，首次超 TGV 23.03）和 0.668@30µm，代价 31% 孤立点抹除（v6 时代 bin4 是 9.6%）。
2. **depb9v8_9bin 未能继承 v7 的 FRC 增益**（0.674→0.606，-0.068）也低于 v6 时代 9-bin（0.661）；但点保真是全场最优（4.35%）。
3. **双轴纪律下无 v8 champion**：bin4 点保真门不过（31% 接近病理），9-bin FRC 明显退化。**老 depb9v6 checkpoint（0.661/4.66%）仍是双轴综合最优**，champion 维持冻结在它。
4. 两臂 range_excursion（13.7/8.4）均显著高于历史同配方（2.70/4.35）——v8 稀疏缺陷池似乎恶化了低频稳定性；且 9-bin"低频差但点保真好"打破了 ACL-064 的"低频稳↔点活"相关性，机制待解。
5. **归因信号（ACL-069 开放问题的新证据）**: v7（密 20-50，浅 0.3+）→ 9bin FRC 0.674/点崩 43%；v8（疏 2-8，深 0.55+）→ 9bin FRC 0.606/点活 4.35%。与"**密度是 FRC 增益来源，浅深度是点抹除来源**"的分解假说一致 → **v9 候选实验**：恢复 v7 密度（min/max 20-50）+ 保留 v8 深度下限（0.55），若假说成立应同时拿到 FRC≥0.67 与 erased%≤5%。成本 ~6h（生成+训练+三轴链都已固化）。磁盘可同时容纳 v8/v9 两池（650G 空闲），无需先删 v8。

**涉及文件**: 远端 `outputs/solver_v25_depb9v8_bin4_30k/`、`output/{stageV25_v8sel_leaderboard,stage2b_bench_v8,stage2b_v8arms_on_v6bench,dot_probe_v8sel}/`、`tmp/champ_chain.sh`；本地 inbox `20260717_v8_champion/`。空间：v7ab 消融池×4 + v6_micro300 + 微型 step ckpt 已删（owner 令，650G 空闲）。

### [ACL-070] 2026-07-09（清晨，整夜自治收官）— **v8 池修复终裁：成立**。孤立点 erased% 43%→4.35%（略好于 v6 基线 4.66%），ALL retention 0.331→0.547；v8 =v7 仅动密度（min/max_holes 20-50→2-8）+ 深度下限（0.3→0.55）两个旋钮

**执行链**（全部当夜完成，owner 睡眠期间）: v8 pilot 24 景过三闸（tcforge gates 7/8 与 v7 原型持平、L1 可探测性审计 r1|lo 病理角落 55%→~22% 且病理剂量 ~5.3→~0.9 个/景、24 洞 hand-check 目检全清晰）→ **按 owner 硬性顺序先完全删除 v7 池**（rm 同步等待 + ls/du 双重确认）→ 生产 5000 景（seed 20260911，60 workers，~1h05m，236G）→ manifest 完整性检查 → `solver_v25_depb9v8_9bin_30k`（depb9v6 配方，b8/p384/seed42/30k，与 v24 ctrl 逐旗标一致）→ 渲染 halves + 3562 点真实域探针。

**终裁读数**（`remote_inbox/20260716_v8_verdict/`，sanity 臂 depb9v6 同批次精确复现 0.5984/4.66%——仪器可信）:
- **depb9v8: ALL retention 0.5471，isolated erased% 4.35%**。对照：v6 基线 0.598/4.66%，v7 病理 0.331/43.48%（ACL-066）、0.337/39.75%（ACL-067 ctrl）。
- 判读：点保真崩溃**完全逆转**，erased% 回到并略优于 v6 区间；retention 0.547 略低于 v6 的 0.598（-0.05，健康带内）。v8 保留了小点缺陷族（半径 1-4px、每景 2-8 个、深度≥0.55）+ v7 全部 composer/噪声升级。

**含义与未决**: (1) "密度+深度下限"捆绑修复有效，但两者的单变量归因仍开放（ACL-069；可日后 1000 景级消融补做）；(2) v7 曾带来的 cross-FRC 小幅改善（0.661→0.674）与低频稳定性改善是否被 v8 保留——**三轴复评尚未跑**（9-bin vs bin4 的 champion 选型也悬置中），是显式的下一步；(3) pool acceptance gate 制度落地：L1 审计已验证有效并进 gate；L2 微型探针以 300 景尺度对本病理无效（ACL-069），gate 中以"生产级验证训练"替代，未来如需廉价 L2 需先在 ≥1000 景尺度重标定。

**涉及文件**: `configs/synthetic/pool_2x_v8_{pilot,cpu,bench48}.json`；远端 `data/synthetic/pool_2x_v8_5k`（训练池）、`pool_2x_v8_bench48`（held-out，勿训）、`outputs/solver_v25_depb9v8_9bin_30k/`；本地 `remote_inbox/20260716_{micro_calib,micro_horizon,v8_verdict}/`、`output/defect_detectability/`（5090）。v7 池已删（owner 顺序令）；v7ab 消融池×4 + 微型端点 checkpoints 留存于 5090。

### [ACL-069] 2026-07-09（夜间自治）— L2 微型探针标定**阴性且信息量大**：点抹除先验在 300 景尺度上根本不出现（8k-24k 步全程 0% 抹除）——**先验形成依赖数据多样性/池规模**；300 景消融梯取消，v8 改为"捆绑修复+生产级验证"，归因（密度 vs 深度）留为开放问题

**实验序列**: (a) 8k 步端点标定：micro_v7end/micro_v6end（各 300 景、b8/p384/seed42、配方逐旗标复刻 v24 ctrl）+ 各自 4k checkpoint，四臂全部 isolated erased%=0.00、ALL retention 0.89-1.05；sanity 臂（历史 depb9v6 30k）同批精确复现 0.5984/4.66%——仪器无误，是被测物"太年轻"。(b) 出现视界实验：v7 端点 resume 至 24k（12k/16k/20k/24k 四点探针），仍全程 0% 抹除、retention 0.88-0.93 **无任何下降趋势**。对照已知事实：同配方同 b8p384 在 5000 景 v7 池 30k 步 = 39.75% 抹除（ACL-067 ctrl）。

**判读**: 点抹除不是步数问题（24k 无趋势），是**池规模/数据多样性问题**——300 景×10 epoch 走向记忆，5000 景×低重复才形成"密集小暗点=噪声纹理、该抹除"的可泛化先验。这反证密度/统计假说优于"物理不可见"假说（后者在 300 景也该表现）。**方法论结论：L2 廉价消融梯（300 景/短训）对这个病理是盲的，作为 acceptance gate 无效**；未来池验收的 L2 需 ≥1000 景+30k 步量级（约 3h/臂，未标定），或直接依赖 L1（已验证有效）+ 生产级单次验证。

**决策**（时间预算权衡，owner 交付要求"今晚造出 v8 好数据"）: 取消逐参数消融（1000 景×30k×3 臂 ≈ 9h 不可行），v8 捆绑修两个被独立证据指认的参数：min_holes 20→2 / max_holes 50→8（密度，机制+L2 阴性推理）+ hole_depth_range [0.3,1.0]→[0.55,1.0]（L1 实证的 r-lo 病理角落，ACL-068）。其余（radius [1,4]、edge_fraction 0.12、softness、composer、噪声）保持 v7 不动。**归因未隔离，记为开放问题**——若 v8 训练验证点保真恢复，density-vs-depth 的单变量归因可日后用 1000 景级消融补做。配置：`pool_2x_v8_{pilot,cpu,bench48}.json`（seed 20260910/11/12，全部与历史池不相交）。

**涉及文件/产物**: `outputs/micro_{v7end,v6end}_8k/`（含 4k/8k/12k/16k/20k/24k checkpoints，5090）；`output/dot_probe_micro{,2}/`（渲染 halves，5090）+ `remote_inbox/20260716_micro_{calib,horizon}/`（本地探针输入/输出）；消融池 pool_2x_v7ab_{density,radius,depth,edge}300 已生成未使用（留作未来 1000 景级消融的种子配置参考，池本身可删）。**教训**: 微型化验证仪器必须先对"已知阳性"标定——本轮标定纪律（先跑端点复现已知缺口再跑消融）避免了把 3 个消融臂的 0% 误读为"全部解救"。

### [ACL-068] 2026-07-09（夜间自治，owner 授权修数据）— L1 零训练可探测性审计落地并出裁决：**"v7 点物理不可见"强假设被证伪**；不可探测尾部精确定位在"小半径×浅深度"角落（r1|lo：55% 低于 CNR3）；"密集分布教坏先验"升为主嫌

**背景**: ACL-067 把点保真崩溃锁定在 v7 池缺陷分布后，owner 拍板方向：验证必须前移（"不能每次生成 5000 景、训完才发现数据有问题"），授权整夜自治按三层验证阶梯（L1 零训练审计 → L2 微型训练探针 → L3 生产）修出 v8 池。计划文档 `research_log/v8_pool_repair_plan.md`（含预注册判定规则与今后所有新池的 acceptance gate 四条前置）。

**新仪器**: `scripts/audit_defect_detectability.py`（fable 子代理实现，78f9797）——对每个 hole 实例算输入端 CNR 两种：analytic（HR 扰动模板按 realism.py 语义重建，经池生成器自己的 `tcforge.physics.apply_psf_blur`+`forward._block_average_from_blurred` 投影，白噪声匹配滤波多帧合成，另有 AR(1) 惩罚版）+ empirical（模板注入真实含噪 burst 后局部测量，吃得到结构化噪声）。v6 legacy 配方不记录 hole 坐标（realism.py holes_custom 门控）→ `--simulate-from-config` 模式从 defects 配置采样假想洞走同一投影。本地 3 景微型池验证 4 项全过（含独立坐标核对，中位 |offset|≤0.05 LR px，无 0.5px 级系统偏移）。

**正式读数**（v7 池 200 景 7074 洞 instance 模式 vs v6 bench48 48 景 504 洞 simulate 模式，`output/defect_detectability/`，5090 CPU 877s/230s）:
- **强假设死刑**: v7 中位 analytic CNR 93.7（AR1 校正 46.7）、empirical 10.4；analytic 低于 CNR3 仅 1.7%。大多数 v7 小点在输入端明确可恢复——网络"看得见还抹"，不是"看不见只好抹"。
- **真实的不可探测尾部且高度局域化**: 按半径×深度三分位分层，r1|lo 单元中位 empirical CNR 2.8、55%/75% 低于 CNR3/5；r2|lo 23%/42%；而 r3/r4|hi 只有 4-6%/8-11%。尾部质量集中在"半径 1-2px × 深度低三分位（~0.3-0.5）"角落 + edge-fraction 子族（局部对比≈0，但该类在 GT 里同样无对比，属无害 no-op 而非教抹除的病理群体）。
- **v6 参照**: 中位 analytic 216-723，比 v7 高约一个数量级（empirical 尾部偏高是模拟洞落在弱结构区的放置伪影，同属 GT 无对比的 no-op，跨池对比以 analytic 为准）。

**判读**: (1) v7 训练分布里确有一撮"GT 有对比但输入端不可恢复"的浅小点（教网络抹除的病理样本），但占比不足以单独解释 43% 的孤立点抹除率——**ACL-066 假说 1（密度：min_holes=20/景 的密集小点把"小暗点=该抹除的噪声"学成强先验）升为主嫌**，L2 消融梯以 density 臂为最高优先级；(2) 无论 L2 结果如何，v8 的缺陷族都应裁掉病理角落——深度下限 0.3→~0.55，小半径配深深度（radius-gated depth floor）。

**涉及文件**: `scripts/audit_defect_detectability.py`（新）；`algos/ep07_unet_sr/scripts/eval_arms_dot_probe.py`（新，9cbaf4d——两阶段批量点保真评测：5090 渲染 halves/Mac 探针，probe 阶段对历史 inbox 实测复现 depb9v6 0.598/4.66%、depb9v7 0.331/43.2%、v24ctrl 0.337/39.75%，retention 取 by_size/ALL 行的修正一并落地）；微型池/消融配置 5 份 + 计划文档（03b4bfc）。空间清理：v23/v24 中间 step ckpt 已删（各 341M→92M）；v7 池裁到 1000 景（238G→48G，全量 manifest 备份 `manifest_full5000.csv`）。**owner 硬性顺序（2026-07-09）：v8 生产池开跑前先完全删除 v7 剩余池并 ls+du 确认，再启动生成。**

**下一步**: L2 端点标定（micro_v7end/micro_v6end 8k 步，跑着）→ 消融梯（density 优先）→ v8 组装过三闸 → 删 v7 → 生产。

### [ACL-067] 2026-07-08（owner 在线跟进）— 决定性对照实验：batch/patch 对齐历史配置后点保真度依然崩溃 —— **锁定为 v7 池缺陷分布问题，非训练超参问题**

**问题诊断**: ACL-066 记录的 depb9v7 三轴复评存在一个未察觉的混杂变量——本会话从 changelog 文字描述重建 depb9v6 配方时，只核对了 DE-prox 相关旋钮，漏查了 `batch_size`/`patch_size_hr`。事后从历史 checkpoint（`solver_v21_depb9v6_30k`、`solver_v22_depb9v6_bin4_30k`）的 `torch.load(...)["config"]` 直接读出：历史轮次实际用 `batch_size=8, patch_size_hr=384`，而本次 v7 轮用的是 `batch_size=20`（自己探针选的"甜点"）、`patch_size_hr=256`（默认值，未显式设置）。也就是说 ACL-066 的对比同时改变了三件事（池子 + batch + patch size），不是干净的单变量实验——owner 直接追问"到底是数据集问题还是算法训练问题"，倒逼出这个疏漏。

**实验**: 补跑一个对照臂 `solver_v24_depb9v7_9bin_ctrl_b8p384_30k`——v7 池 + depb9v6 9-bin 配方，`batch_size=8, patch_size_hr=384` 与历史逐字段对齐，仅池子（v6→v7）这一个变量在变。30k 步，exit 0。跑完立即用同一套点保真度探针（3562 点检出集，`probe_dot_retention.py --extra-arms`，additive 注册新臂 `depb9v7_ctrl`）+ cross-FRC leaderboard 复评。

**实验记录**（TGV 对照 0.7017/23.03µm 精确复现，仪器可信；v6 历史臂重测漂移 <0.003，探针无漂移）：

| 臂 | 池 | batch/patch | ALL retention | 孤立点 erased% | real cross-FRC@30µm |
|---|---|---|---|---|---|
| depb9v6（历史） | v6 | 8/384 | 0.598 | 4.66% | 0.661 |
| depb9v7（ACL-066，混杂） | v7 | 20/256 | 0.331 | 43.48% | 0.674 |
| **depb9v7_ctrl（本轮，对齐后）** | v7 | **8/384** | **0.337** | **39.75%** | **0.674** |

**判读**: **对齐batch/patch后，点保真度几乎没有恢复**（isolated erased% 43.48%→39.75%，仅降3.7个百分点，相对v6的4.66%这个35个百分点的缺口几乎没有意义变化）；cross-FRC也完全没变（0.674→0.674）。**两条轴一致指向同一个结论：batch/patch的混杂是真实存在的方法学疏漏，但不是点保真度崩溃的成因。真正的病根是v7池本身的缺陷分布设计**（ACL-066 猜想的密度/半径/深度问题）。训练超参可以排除嫌疑；这不是"改算法/训练配置就能解决"的问题，得回到数据生成器。

**回答 owner 的关键问题**："数据集上有问题，得重新生成数据集"——本轮实验就是为了回答这个问题设计的，答案是**数据集**。具体是缺陷族的哪个参数（密度 min_holes=20 vs v6 的 max_holes=6、半径 1-4px vs 4-13px、还是 `hole_depth_range`/`hole_edge_softness_px`/`hole_edge_fraction` 这几个未经 pilot 标定的占位值）导致的，仍需要消融实验才能定位，本轮只证明了"是数据不是训练"，没有进一步拆解数据侧的具体病因。

**下一步（owner 待裁决，未展开）**: 在 v7 defects 配置上做一次小规模（24-48 景）消融——固定其它不变，把 `min_holes`/`hole_radius_px` 调回接近 v6 量级，或先关掉 `hole_edge_fraction`/放宽 `hole_depth_range`，用同一套点保真度探针快速判断哪个参数是主因，再决定是否要重新生成 5000 景主池。

**涉及文件**: `algos/ep07_unet_sr/scripts/probe_dot_retention.py`（additive：`depb9v7_ctrl` 加入 `ARM_FILES`/`EXTRA_ARMS`，默认流程不变）；远端产物 `outputs/solver_v24_depb9v7_9bin_ctrl_b8p384_30k`（checkpoint）、`output/stageDE_v24ctrl_{offset_probe,corrected,leaderboard}/`、`output/dot_probe_v24ctrl/`（未 rsync 回本地）；远端脚本 `render_v24_ctrl_real_halves.py`（v23 同款脚本的改名副本，未提交，留在远端 scratch）。

---

### [ACL-066] 2026-07-08（深夜/凌晨，owner 在线跟进）— v7 池 depb9v7 9-bin/bin4 三轴复评：real cross-FRC 与低频稳定性双双小幅改善，但**点保真度全面崩溃**（ALL retention 0.60/0.48 → ~0.33，孤立点 erased% 4.7%/9.6% → ~43%）——v7 池未能兑现"修复小暗点抹除"的建池初衷，反而更差

**问题诊断**: ACL-065 已把 depb9v6 配方（9-bin/bin4 两臂）搬到 v7 池重训 30k 步（batch20，torch.compile，~390ms/step，均 exit 0）。本条是 owner 要求的下一步——用与 ACL-062/063/064 完全同一套仪器（cross-FRC leaderboard + TGV 对照、Stage2b 合成 band-FRC/range_excursion、3562 点检出集的点保真探针）给这两个 v7 臂做三轴复评，判定 champion。委托子代理执行（远端脚本改造+合成基准+点保真探针三段落地），本轮子代理首次汇报是不完整的中间状态（"等 synth chain 完成"），resume 后补全。

**实验记录**（TGV 对照两处均精确复现：cross-FRC 侧 0.7017/23.03µm，合成侧 tgv__oracle band_FRC 0.765/range_excursion 2.39 与 ACL-062 逐位一致——两套仪器均可信）：

| recipe | pool | real cross-FRC@30µm | cutoff µm | synth range_excursion | dot ALL retention | isolated erased% |
|---|---|---|---|---|---|---|
| 9-bin | v6 (depb9v6) | 0.661 | — | 2.70 | 0.60 | 4.7% |
| 9-bin | **v7 (depb9v7)** | **0.674** | 23.03 | **2.06** | **0.332** | **43.2%** |
| bin4 | v6 (depb9v6_bin4) | 0.668 | — | 4.35 | 0.477 | 9.6% |
| bin4 | **v7 (depb9v7_bin4)** | **0.676** | 22.80 | **1.60** | **0.325** | **43.2%** |
| TGV（经典基线，两套仪器各自对照） | — | 0.7017 | 23.03 | 2.39(oracle) | — | — |

**判读**:
1. **real cross-FRC 小幅提升**：两臂较各自 v6 版本 +0.008~+0.013，双双超过历史神经纪录 v19(0.6705)，与 TGV 差距从 ~0.03-0.04 收窄到 ~0.026-0.028——v7 池的内容/噪声升级对主门确有正贡献，但幅度不大，仍未破 TGV。
2. **低频稳定性同步改善**：bin4 的 range_excursion 从令人担忧的 4.35 降到干净的 1.60；9-bin 从 2.70 降到 2.06。两臂现在都在"低频干净"区间（对照 tgv_oracle 的 2.39 为参照量级），v6 轮里 bin4 相对不稳定的问题在 v7 上消失了。
3. **点保真度是本轮的头号发现，且方向与预期相反**：v7 池专门为修复 ACL-063 发现的"小暗点被神经网络抹除"问题而建（新增半径 1-4px、每景 min_holes=20 的小点缺陷族，v6 的洞半径下限是 4px 且每景至多 6 个，训练分布里事实上不存在这个尺度的孤立小点）。**实测结果是两臂点保真度均较 v6 版本崩溃式下降**：ALL retention 0.60/0.48→~0.33，孤立点 erased% 4.7%/9.6%→~43%（4.5-9 倍恶化）。fwhm_ratio≈1.00（两臂）说明不是"点被模糊变宽"，是被**直接抹平**。9-bin 与 bin4 在这一轴上几乎完全打平（0.332 vs 0.325，43.2% vs 43.2%），v6 轮里"bin4 用点保真换 FRC"的权衡结构在 v7 上不复存在——不是权衡消失变好，是两边一起摔到谷底。
4. **champion 裁决**：**两个 v7 臂都不是好的 champion 候选**。FRC/低频轴的小幅改善不能抵消点保真度的灾难性倒退，尤其鉴于点保真正是建这个池子的**唯一动机**。9-bin vs 4-bin 的选择在三轴上已近乎无差异（FRC 差 0.002，低频小幅偏向 bin4），该问题本身意义降低——核心矛盾变成"v7 池的小点缺陷族设计出了什么问题"。
5. **怀疑方向（未验证，留给下一步）**：v7 的小点缺陷族密度远高于 v6（min_holes 20 vs max_holes 6）且半径更小（1-4px vs 4-13px）——可能让网络把"密集小暗点"学成了比 v6 更强的"这是噪声/该抹除"先验，而不是学会保留它们（v6 训练分布里这个尺度的孤立点几乎不存在，抹除更多是"没见过"；v7 里天天见但仍然抹，更像是"学到的先验就是抹掉"）。也可能是 ACL-065 记录的几个占位参数（`hole_edge_fraction=0.12`、`hole_depth_range=[0.3,1.0]`）设置不当——这两项在两份集成方案文档里都明确标注"待 pilot 目检/标定"，本次为了不多引入一个变量直接用 5000 景生产池验证，尚未做那一步闭环。

**下一步（owner 待裁决，非本轮自主延伸）**: (a) 优先怀疑 v7 defects 配置的密度/半径/深度参数，考虑一次小规模（如 24-48 景）消融：固定其它不变，把 min_holes 调回接近 v6 量级或把 hole_depth_range 变浅，测点保真是否恢复；(b) 也不能排除架构侧交互（DE-prox σ4 + dc_weight0 的强先验路径本身可能对训练分布里"常见的小暗点"更敏感）；(c) 在结论明确前，champion 选型继续冻结，v7 池不建议直接用于任何"对外宣称改进"的产出。

**涉及文件**: `algos/ep07_unet_sr/scripts/probe_dot_retention.py`（additive：`ARM_FILES`/`EXTRA_ARMS` 新增 depb9v7/depb9v7_bin4，默认流程不变，同时复测 depb9v6/depb9v6_bin4 逐位复现 ACL-064 数字，验证探针本身无漂移）；远端产物 `output/stageDE_v23_{offset_probe,corrected,leaderboard}/`、`output/stage2b_bench_v7/`、`output/dot_probe_v7/`（未 rsync 回本地，按 remote_inbox 规则）；远端脚本 `render_v23_depb9v7_real_halves.py`（v22 同款脚本的改名副本，仅改 arm 字典与输出目录，未提交，留在远端 scratch）。

---

### [ACL-065] 2026-07-08（夜间自治，owner 睡眠期间）— V7 生成器落地：composer_v7 + 缺陷体系升级 + 噪声算子升级全部接入 tcforge 生产管线；5000 景 V7 主池已生成；depb9v6 9-bin vs 4-bin 三轴复评（ACL-064 指定的下一步）已在 v7 上起训

**背景**: owner 睡前授权整夜自主执行完整流水线：把 `research_log/v7_composer_defects_integration_plan.md`（内容/缺陷轴）与 `research_log/v7_noise_operator_integration_plan.md`（噪声/PSF/σ-DR 轴）两份"待实现"方案落成生产代码，然后删 v6、生成 v7、起训练。两份方案在会前已由不同规划代理产出（commit 6e81d5a/a9a7c3c），但代码从未写。

**实现**: 两个 Opus 子代理并行在独立 git worktree 里分别实现两份方案（互不干扰同一批文件），完成后本会话亲自复核测试 + merge：
- composer+defects（`v7-composer-defects-integration` 分支 → merge `104793e`）：新模块 `tcforge/src/tcforge/composer_v7.py`（614 行，r4 原型的字节级忠实移植，1200 例 seed 验证与原型逐 bit 一致）；`geometry.build_scene_mask_with_metadata` 新增 `scene_composer="panel_cluster_v7"` 分派（v6/legacy RNG 流零改动，金样本钉死）；`realism.py` 五项升级（统一缺陷实例 schema + `record_instances`、`apply_thermal_defects` 热点/暗斑、`carve_trace_breaks` 断线查表、erosion `hole_margin_px/adaptive/edge_fraction` 旋钮、`render_isothermal_field` zones 分组）全部默认字节不变。tcforge 门禁复跑 7/8（`audit_v7_tcforge_gates.py`，G5 场景内疏密对比度 1.44，与原型自身的 1.446 一致，非回归，划定为内容轴既有局限）。
- 噪声算子（`worktree-agent-a4b1574848700fc6d` 分支 → merge `8395317`）：`field_noise_burst` 四项升级（行条纹 FPN、真 1/f^α 低频场 `physics.powerlaw_field`、静态逐像素 FPN、grain 帧间 AR(1)）；`make_noise`/`sample_psf_parameters` 的 mix_weights / 椭圆比暴露为 config；ep07 侧 σ-DR "绝对谎言" 训练管线（`--solver-dc-psf-sigma-lie-px`）。全部默认路径零额外 RNG draw，金样本 + v6-seed 字节级冒烟测试双重验证。
- 两分支 merge 时 `tcforge/tests/test_realism.py` 有一处结构性冲突（双方都在文件尾追加互不重叠的新测试函数），按纯拼接解决（仅删冲突标记行，两侧内容全保留），合并后 tcforge 113 passed / ep07 113 passed+3 skipped。
- 新增 config：`configs/synthetic/pool_2x_v7_pilot.json`（24 景，seed 20260901）、`pool_2x_v7_cpu.json`（5000 景，seed 20260902）、`pool_2x_v7_bench48.json`（48 景，seed 20260903，暂未生成）——由本会话合并两个子代理各自报告的 config 片段组装，非子代理自动产出（分工上明确排除，避免两边各写一份冲突）。
- commit 6e81d5a→19d1672 全部 push 到 origin/main。

**验证**（本会话亲自复核，未采信子代理报告直接过关）：
1. 本地 tcforge 113 passed、ep07 113 passed+3 skipped（3 skip 是既有 CUDA-only 测试，与本次改动无关）。
2. **远端 5090 环境一致性**：pull 后远端复跑同一测试套件，2 个金样本测试失败（`test_scene_mask_legacy_and_v6_paths_match_golden`、`test_sample_psf_parameters_default_ratio_matches_golden`）——排查后确认是 Mac↔Linux 跨平台浮点数最后一位精度差异（geometry primitive 的 `h_um` 等字段差在 ~1e-13 相对量级，栅格化后被像素网格完全吸收），**不是逻辑 bug**。用更有说服力的验证替代：在远端本机对比 merge 前(6e81d5a)后(19d1672)代码跑同一 v6-seed 3 景，`lr_burst.npy`/`hr_temperature_2x.npy` 逐字节相同、metadata 零差异键——这才是真正要紧的"默认行为不变"证明，在生产机器上坐实。
3. **24 景 v7 pilot**（远端实跑 `generate_training_pool.py`）：`audit_v7_tcforge_gates.py` 7/8、`audit_generated_pool.py` 全绿（24/24 有缺陷标注、0 zero-dot 景、0 结构问题）；肉眼复核 4 张 `hr_mask_4x.png`（含一张验证 360° 旋转）确认面板簇几何、纹理面板、窗口镂空、断线曲线、缺陷点分布均符合设计、无 artifact。
4. **5000 景主池生成**：删除 v6 主池（`data/synthetic/pool_2x_v6_5k`，236GB，bench48 未动）后，61 workers 跑 2h44m 生成完毕（exit 0，manifest 5000 行），磁盘 238GB（655GB 空闲，充裕）。
5. **k=200 抽样 pool 审核**：`POOL AUDIT: INVESTIGATE`（非 PASS）——1/200 景 `scene_1828` 的 `psf_sigma_lr_px=0.6017` 超出审核脚本硬编码的 `[0.10,0.60]` 泛化健康区间。溯源 `physics.py:561`：椭圆 PSF 分支 `psf_sigma_lr_px = sigma * uniform(*elliptical_ratio_range)`，base sigma 上限 0.55 乘椭圆比上限 1.20 可达 0.66，**这是椭圆 PSF 采样的固有统计行为**（v6 用完全相同的硬编码默认比例范围，同样概率会出现，只是从未有人在 v6 上跑过这个深度的 k=200 抽样审核）。非 v7 特有 bug，不阻塞。审核脚本的健康区间硬编码值未来可考虑改为按 `sigma_range × ratio_range` 动态推导，本次未改（不在授权范围内，记录留痕）。

**训练**（owner 授权无需等待的下一步，非本次自主延伸——ACL-064 已明确指定"champion 长训前在 v7 上复评 9-bin vs bin4 三轴"）：
- Batch 探针（bs 8/12/16/20，各 150 步，并发 nvidia-smi 轮询取峰值显存）：bs20 峰值显存 20823MiB（~20GB，命中既定甜点）且吞吐全场最高（14.9 samples/s vs bs8 的 6.7），选定。
- `solver_v23_depb9v7_9bin_30k`（`--phase-bin-channels 9 --solver-phasebin-ontf --solver-warmstart aligned_mean --solver-m-frames 12 --solver-dc-weight 0 --solver-prox-highpass-residual --solver-prox-highpass-sigma-hr 4`，复刻 ACL-062/064 的 depb9v6 配方，仅换 v7 池）与 `solver_v23_depb9v7_bin4_30k`（同配方 `--phase-bin-channels 4`）成对锚定同一 batch(20)/steps(30000)/seed(42)，在远端 tmux 窗口 `v7train` 串行跑（GPU 装不下两个 ~20GB 任务同时跑）。`--save-every 10000 --real-eval-every 1000 --synth-eval-every 1000`（ACL-053 定下的"今后"评测节奏，与 v22 arms 用过的 save5k 不同——本轮不是收敛曲线研究，选更省的节奏）。
- 已验证真正跑起来（非卡死）：torch.compile 冷启动编译耗时约 8-10 分钟（主进程 CPU 占用持续攀升，非空转），编译完成后 step 2300 时稳定在 ~390-420ms/step（较 batch 探针的未编译 ~1.3-1.4s/step 提速 ~3×）。30k 步预计每臂约 3.2-3.5 小时，两臂串行约 6.5-7 小时——9bin 臂预计在 owner 起床前后完成，bin4 臂会跑到之后。
- **依据本轮 v7 池首次系统性纳入小暗点缺陷族（半径 1-4px、每景 min_holes 20、hole_edge_fraction 0.12），预期 dot-retention 三轴指标（尤其 9-bin vs 4-bin 的点保真权衡，ACL-064 判读）应有别于 v6 池上的历史读数——待训练完成后用 `probe_dot_retention.py` 复评，不在本次自主范围内先行判定。**

**已知局限/待 owner 裁决**（记录不隐瞒）：
- `hole_edge_fraction=0.12`、噪声块 `lowfreq_c=[0.04,0.10]`/`vignette_c=[0.10,0.16]` 等参数是两份方案文档标注的"待 pilot 目检/`audit_synth_noise.py` 标定"占位值，本次未做该标定闭环（5000 景已用占位值生成，如后续标定发现需调整，需重新生成——记录在案供 owner 决策是否接受）。
- G5（场景内疏密对比度）仍是唯一未过的门禁，继承自原型本身，非本次引入。
- 审核脚本 `audit_generated_pool.py` 的 psf_sigma 健康区间硬编码，产生良性误报，未修。
- 9-bin vs 4-bin 训练结果本身（含点保真三轴复评）尚未产出，champion 长训决策仍待这轮结果 + owner 裁决。

**涉及文件**: `tcforge/src/tcforge/{composer_v7.py(新),geometry.py,realism.py,physics.py,storage.py,_noise_stats.py(新)}`、`scripts/{generate_training_pool.py,audit_v7_tcforge_gates.py(新),audit_generated_pool.py,audit_real_noise.py}`、`algos/ep07_unet_sr/src/unet_sr/{config.py,dataset.py,solver_train.py}`、`configs/synthetic/pool_2x_v7_{pilot,cpu,bench48}.json(新)`、约 15 个新 tcforge/ep07 测试 + 6 个金样本 fixture。远端产物：`data/synthetic/pool_2x_v7_5k`（5000 景，238GB）、`data/synthetic/pool_2x_v7_pilot`（24 景，保留供后续参考）、`algos/ep07_unet_sr/outputs/solver_v23_depb9v7_{9bin,bin4}_30k`（训练中）。

---

### [ACL-064] 2026-07-08（凌晨自治）— v22 三跟进臂裁决：eta 重标定证实治好 meanDC 灾难性发散(ACL-062 假设成立,range 10^4→~10)但仅平 v14；9-vs-4 bin 隔离**推翻 ACL-062"9-bin 实锤"**(4 bin 双域≥9 bin)；**depb9v6_bin4 = 神经真实域新高 0.668(平 v19 纪录)+低频干净+通道更省 → 双轴 champion 候选**

**问题诊断**: 承 ACL-062 下一步 (a)(b)。三个 30k/save5k 单变量跟进臂(v6 池,可信仪器,TGV×drz 复现 0.7017/0.3558/23.03):
- eta8 `solver_v22_meandc_eta8_30k` = v21_meanDC + `--solver-eta-init 8.0`(mean 归一化下 eta_mean=M×eta_sum,M16→8.0≈v14 的 eta_sum 0.5;confdiff 92 字段仅差 eta);
- eta4 `..._eta4_30k` = 同上 eta 4.0(预注册区间下端);
- bin4 `solver_v22_depb9v6_bin4_30k` = depb9v6 配方仅 9 phase bins→4(派生 in_ch 14→9,confdiff 91 字段仅差此轴)。

**实验记录**(final;合成 N=96 48/48 全格;真实=偏移校正 cross-FRC,六半幅 offset 残余 0.057-0.095 HR px、sign-changes 0→0、网格门 ok):

| 臂 | 合成 band_FRC | range_excursion(final) | 真实 cross-FRC@30 |
|---|---|---|---|
| **depb9v6_bin4** | 0.855 | 4.35(全程干净) | **0.668** |
| meandc_eta8 | 0.858 | 10.1(稳定~10,mean_offset −0.31) | 0.644 |
| meandc_eta4 | 0.871 | 21.3(15k 后漂移,25k 峰 31) | 0.647 |
| depb9v6(9bin) | 0.830 | 2.70 | 0.661 |
| v19_etaB | 0.779 | 1.33 | 0.6705 |
| v14 | 0.870 | 33.6 | 0.649 |
| v21_meanDC(发散) | 0.206 | 10261 | 0.657 |

**判读**:
1. **eta 重标定证实 ACL-062 假设**: mean 归一化 + eta 按 ~M 重标 **治好了 v21_meanDC 灾难性低频发散**(range 10261 → eta8 稳定 ~10、eta4 峰 31,均非 10^4 量级)。收敛轨迹:eta8 全程 range 10-16 平稳;eta4 前 15k 干净(<4)后段漂移(10.9→31→21,fullband_rmse/mean_offset 同步涨)——**eta 越大低频锚越强越稳**(eta8>eta4)。"改归一化必须配 eta 重标"闭环坐实,原则化 DC 可行。**但两臂真实 cross-FRC 仅 0.644/0.647 ≈ 平 v14(0.649),未入 ~0.66 DE-prox 簇;低频也远不及 DE-prox(range 2.7-4.4)**。结论:mean-DC+eta 是合法非病态路径,但**非制胜路径**;DE 高通 prox 仍是低频解耦更优机制。
2. **9-vs-4 bin 隔离 → 推翻 ACL-062"9-phase-bin 实锤"**: 4 bin 合成 0.855>9 bin 的 0.830、真实 0.668>0.661、低频 4.35 仍干净——**4 bin 双域双双 ≥ 9 bin**。故 depb9v6 赢因**不是** 9 这个细粒度(9→4 无损微升),而是"相位分箱输入(粗粒度即可)+DE-prox"。Stage 2a"输入端别抹平子像素相位"方向仍成立,但箱数可降到 4(通道 14→9 更省);ACL-062 该处归因修正。
3. **depb9v6_bin4 = 双轴 champion 候选**: 真实 0.668=神经新高,**平 v19 纪录 0.6705(差 0.002 噪声内)**,低频干净(4.35)、DE-prox 家族、输入通道更少。在 ACL-063 双轴标准(真实 FRC + 点保留门)下**支配 v19**(同 FRC、远好的点保真——v19 抹 68.6% 孤立点;bin4 点保留待补测)。仍低于 TGV 0.702。
4. TGV×drz 复现 0.7017/0.3558/23.03,offset 探针六半幅全 success,结果可信。

**点保真回填(v22 三臂,复用 ACL-063 检出的 3562 点+isolation 标签,同预处理;`output/dot_probe/summary_v22_arms.csv`)**:

| 臂 | ALL retention | isolated erased% | vs_drizzle |
|---|---|---|---|
| depb9v6_bin4 | 0.477 | 9.6% | 0.569 |
| meandc_eta8 | 0.455 | 12.7% | 0.543 |
| meandc_eta4 | 0.197 | **83.5%** | 0.242 |
| (参照)depb9v6 9bin | 0.60 | 4.7% | 0.71 |
| (参照)de_pb9(v5) | 0.68 | 2.5% | 0.85 |
| (参照)v19 | 0.27 | 68.6% | 0.29 |
| (参照)v14 | 0.50 | 6.8% | 0.59 |

**修正判读(加入第三轴=点保真)**:
1. **"4 bin 双域 ≥ 9 bin"不成立于第三轴 → ACL-064 判读 #2/#3 修正**: 点保真上 bin4(0.477,孤立抹除 9.6%)**劣于** 9-bin depb9v6(0.60,4.7%)。9→4 bin **不是免费午餐**:用 −0.12 点保留 + 2× 孤立抹除换 +0.007 真实 FRC。**双轴纪律价值实证**——只看 FRC,bin4 像白捡的赢(FRC↑通道↓);点探针揭出隐藏代价。bin4 不支配 9-bin,两者是 FRC↔点保真真权衡。
2. **champion = 三轴权衡,非橡皮图章**: depb9v6 9-bin(0.661/低频2.70/点0.60·4.7%)=最均衡可复现臂;bin4(0.668/4.35/0.477·9.6%)=FRC 稍高点稍差通道省;v19(0.6705/1.33/0.27·68.6%)=FRC 王点崩;de_pb9(v5 池不可复现,0.667/2.73/0.68·2.5%)=三轴全优但池已删。**owner 护缺陷优先 → 倾向 9-bin depb9v6;FRC 至上 → bin4。属 owner 加权决策。**
3. **meandc_eta4 点保真灾难(83.5% 孤立抹除,劣于 v19)与其后段低频漂移同源**:不稳定同时表现为低频漂移+点摧毁;eta8(稳定)=v14 档(12.7%)。**再证低频稳定性与点保真强相关**(DC 锚强且稳→点活),与 ACL-060 η 反转、ACL-063 弱 DC 抹点三线同指一因。

**下一步(更新)**: (a) champion 建议 **depb9v6 9-bin 为主候选**(护缺陷优先·三轴最均衡可复现),bin4 作 FRC 上限对照;de_pb9 证"三轴全优可存在"(v5),提示 **v7 应能复现其点保真**;(b) champion 长训前在 v7 上复评 9-bin vs bin4 三轴;(c) A3 phase-D 经典跑完回填 ACL-060。

**涉及文件**: `algos/ep07_unet_sr/scripts/probe_dot_retention.py`(Sonnet 加 `--extra-arms/--per-dot-csv` 追加臂能力,默认流程不变);其余无代码变更(实验记录;三臂用现有 harness)。

---

### [ACL-063] 2026-07-08（夜间自治）— 真实域小黑点保真探针（P0）：owner 目视坐实并量化——全部神经臂衰减小暗缺陷,v19(真实纪录臂)抹除 58% 检出点;DE-prox 臂保留最好;cross-FRC 对该失真近盲;神经臂全局增益一致 +15-18%

**问题诊断**: owner 聚合 TB 后目视发现:真实芯片上 TGV 锐利还原的小黑点(局部低温缺陷),神经臂大量模糊甚至抹除。溯源训练分布:v6 池缺陷生成(`tcforge/src/tcforge/realism.py:68` `apply_defects`)最小 hole 半径 4 HR px(=40µm)、每景 ≤6 个、docstring 明写"All defects are > pitch (recoverable)"——**直径 2-4 px 的小暗点在训练分布里不存在,且该尺度的孤立暗斑只以噪声身份出现过**;学好的去噪先验对它们的"正确"动作就是抹除。本条把目视变成可复现测量(探针脚本),并为 A1 内容轴/v7 池提供真实域证据。

**修改内容**: 新分析脚本 `algos/ep07_unet_sr/scripts/probe_dot_retention.py`(仅 numpy/scipy/pandas/matplotlib,种子固定,~6s 一键重跑)。设计预注册:检测只在 TGV 工作图((a+b)/2)上做(多尺度 LoG σ∈{1,1.5,2,3,4});有效性过滤=边缘16px/深度SNR≥4×局部噪声(a−b 差图估计)/a-b 半幅独立一致/r≤6px;逐臂预校验=全幅亚像素相位相关对齐(合成自检 ~0.005px)+高梯度像素 Theil-Sen 增益回归(|slope−1|>0.05 则深度归一);retention=depth_arm/depth_tgv,分类阈值 erased<0.3/blurred 0.3-0.7/preserved≥0.7;20 随机非点位置做对照组。输入=六臂真实重建(v21 世代 center-corrected;drizzle/TGV 无 corrected 版,用原生网格版,实测对齐后全臂偏移 <0.15 px 无需重采样)。

**实验记录**(产物 `output/dot_probe/`,数据 `remote_inbox/20260713_dotprobe/`):
- **跨臂可比性**: 偏移全部 ≤0.034 px;**五个神经臂对 TGV 的增益一致 1.15-1.18**(drizzle 1.035)——ACL-061"轻度过锐化"的定量化;retention 已按增益归一(对神经臂保守)。
- **检出漏斗**: LoG 极大 5171→去重 4559→边缘−393→SNR−410→半幅一致−0→尺寸−194→**3562 点**(深度中位 0.071°C,SNR 中位 8.7)。
- **主表(中位 retention,括号=erased%/preserved%),尺寸=TGV FWHM 直径**:

  | 臂 | ≤3px(≤30µm) | 3-5px | 5-8px | >8px | ALL |
  |---|---|---|---|---|---|
  | drizzle | 0.53 (12/42) | 0.79 (4/76) | 0.85 (2/90) | 0.94 (7/77) | **0.84** (4/80) |
  | v14 | 0.49 (30/32) | 0.48 (16/19) | 0.50 (9/11) | 0.53 (14/22) | **0.50** (14/16) |
  | v19_etaB | 0.36 (39/24) | 0.24 (64/11) | 0.27 (61/6) | 0.33 (40/10) | **0.27** (58/9) |
  | de_pb9 | 0.61 (22/41) | 0.68 (12/47) | 0.69 (6/47) | 0.69 (12/48) | **0.68** (10/47) |
  | depb9v6 | 0.51 (30/36) | 0.54 (15/28) | 0.62 (5/31) | 0.67 (11/44) | **0.60** (11/31) |
  | meanDC(发散臂) | 0.41 (35/29) | 0.19 (69/11) | 0.21 (76/6) | 0.29 (53/10) | **0.22** (68/10) |

  FWHM 膨胀比中位 1.0-1.35(≤3px 档最大:v14 1.34)。retention_vs_drizzle 中位:v14 0.59 / v19 0.33 / de_pb9 0.80 / depb9v6 0.71 / meanDC 0.27。
- **光学真值子集**(42 点落在光学足迹内,该区=密集周期人字纹,点=纹样暗极小):排序**反转**——depb9v6 0.97、de_pb9 0.87、v14 0.81、v19 0.72 均高,drizzle 0.58 反而低(小 σ 点 0.09-0.45)。
- **tile 带通 NCC**(dense/sparse/差): drizzle +0.013,v14 +0.143,v19 +0.193,depb9v6 +0.105,meanDC +0.294——但**有结构量混淆**(低点数 tile 均为低结构 tile,预注册的能量同档匹配无法满足),只作探索性参考。
- **对照组 sanity**: TGV 自身 retention 恒 1.0;20 随机非点的"深度"在噪声量级(比值不稳定属预期);v19/meanDC 非点处本底深度也只有 TGV 一半(全局细暗纹理压制)。

**判读**:
1. **owner 目视坐实且量化**: 所有神经臂衰减小暗缺陷。最重 = v19(58% erased)与 meanDC(68%,发散臂另案);v14 居中(retention 0.50);**DE-prox 臂最好**(de_pb9 0.68/10% erased,depb9v6 0.60/11%)。drizzle 0.84 证明**点在证据里是活的,是先验抹的**(唯 ≤3px 档 drizzle 也只 0.53——最小点部分证据受限)。
2. **机理读数**: 保留率排序 DE-prox > 强η(v14,0.5) > 弱η(v19,0.09)——DC 锚强则点活、先验主导则点死;DE 高通 prox 第三次证明其价值(带内细节/低频解耦之外再加 OOD 特征保留)。**该排序与真实 cross-FRC 排序(v19 最高 0.6705)反相关** → cross-FRC 对孤立点抹除近盲(点能量占带内比例小),现有主门探测不到这种失真;owner 目视再次跑赢仪器(ACL-061 同型)。**点保真需要成为独立评测轴/回归门**(0d 套件候选,与 extent 探针并列)。
3. **内容特异性 = A1 内容轴的真实域证据**: 周期纹样区(光学足迹)神经臂 retention 反而最高(depb9v6 0.97)且 drizzle 低——神经在**分布内**周期结构上有真 SR 增益;失真集中在**孤立**小暗点上(v6 motif 全是周期/结构化几何,孤立点=噪声身份)。isolated-vs-structured 分层在跑,待回填。
4. 全局增益 +15-18%(五臂一致)= 过锐化签名的定量:先验放大分布内结构、同时抹除分布外特征——同一枚硬币两面。

**下一步**: (a) isolated/structured 分层回填(在跑);(b) A1-dots 试点池(生成器审计判 config-only 可跑,`research_log/ood_generator_support_audit_draft.md`;对比度/软边缺口 30-40 行修复待 owner 拍板);(c) 点保真探针候选纳入 0d 常驻回归门;(d) v7 池吸收小暗点缺陷族(半径下探 1-3px、密度上调、若批准加 hole_depth_range);(e) champion 配方权衡新增一轴:v19 的 cross-FRC 纪录建立在最重的点抹除上,DE-prox 系配方在"不删真实缺陷"上占优。

**回填(isolated/structured 分层)**: 分类器 = 邻域检出数(r14px)+带通自相关旁瓣比,光学子集校准命中 100%(42/42,一次预注册内调整:加"≥4 邻居即 structured"分支,isolated 定义未动);计数 structured 2446 / ambiguous 794 / **isolated 322**(集中 3-8px;≤3px 档 N=1——最小点几乎全活在密集纹理区)。产物 `summary_by_arm_isolation.csv`、`board_crops_isolated.png`(目检:isolated 点在 drizzle/TGV 清晰、v14 减半、v19/meanDC 近乎平掉、DE-prox 臂保留)。
1. **isolated 组 = 纯先验抹除的干净证据**: drizzle erased **0%**、中位 0.83、preserved 92.5%——孤立点在证据里全活,无任何证据受限借口。神经臂:v19 erased **68.6%**(比其 structured 组 54.9% 更狠)、meanDC 78.6%;v14 0.48/6.8%(很少全抹但一律减半);**de_pb9 0.73/2.5%/vdrz 0.85 最佳**,depb9v6 0.62/4.7%。
2. **修正判读3**: 各臂中位 retention 在 isolated vs structured 间差 ≤0.05——"神经在结构极小上普遍更好"**不成立**;光学足迹的反转排序是那块强周期高对比区的局部现象,不能推广到全部 structured 点。真正的组间差是 **erased% 两极化**:弱η臂在孤立点上更易"全抹"(v19 68.6 vs 54.9),DC 锚强/DE-prox 臂在孤立点上几乎不全抹(2.5-6.8%)但仍减深。修正后的机理句:点的存亡 = 先验是否支持该内容 × DC/DE 路径能否把证据顶回来;弱 DC 下孤立点没有任何保护。
3. v7 含义更新:dots 池应同时含**孤立点**与**嵌入密集结构的小点**(≤3px 真实点几乎都嵌在纹理里,那一档 drizzle 也只 0.53——评测时两类分开计分)。

**涉及文件**: `algos/ep07_unet_sr/scripts/probe_dot_retention.py`(新);产物 `output/dot_probe/`;数据快照 `remote_inbox/20260713_dotprobe/`(不入 git)。

---

### [ACL-062] 2026-07-08（深夜）— v6 单变量拆解：臂A'(de_pb9→v6)证实优势来自配方非v5池、9-phase-bin 实锤；臂B(mean-DC@eta0.5)合成病态=DC被稀释16×,归一化需配 eta 重标定

**问题诊断**: de_pb9 八轴齐变(ACL-061),赢因待拆。开两个 30k/save5k 单变量臂(v6 池,可信仪器,锚 v14_50k/v19/de_pb9;TGV×drz 对照复现 0.7017/0.3558/23.03,管线可信):
- 臂B `solver_v21_meanDC_30k` = v14 + 仅 `--solver-dc-normalize mean`(单变量,smoke config-diff 证实);
- 臂A' `solver_v21_depb9v6_30k` = de_pb9 配方(9-bin ontf/DE-prox σ4/aligned_mean/m12/dc_weight0/no_drizzle False)搬到 v6 池(纯"v14+9bin"因 phasebin_ontf 与 no_drizzle 代码互斥做不出,`config.py:243`;A' 隔离 v5 池混淆并使 de_pb9 与 v14/v19 同池可比)。

**实验记录**(final,band=25-40µm FRC-vs-GT):

| 臂 | band_FRC | band_rmse | range_excursion | synth frc@30 | 真实 cross-FRC@30 |
|---|---|---|---|---|---|
| **depb9v6(A')** | **0.830** | 0.0182 | **2.70** | 0.819 | 0.6611 |
| meanDC(B) | 0.206(病态) | 0.239 | 10261(爆) | 0.274 | 0.6567 |
| v14 | 0.870 | 0.0149 | 33.58 | 0.855 | 0.649 |
| v19_etaB | 0.779 | 0.0246 | 1.33 | 0.763 | 0.6705 |
| de_pb9 | 0.813 | 0.0203 | 2.73 | 0.798 | 0.667 |
| tgv_oracle | 0.765 | 0.0274 | 2.39 | 0.758 | (TGV 0.702) |

**判读**:
1. **臂A' = 干净成功,de_pb9 优势来自配方而非 v5 池**:band_FRC 0.830 ≥ de_pb9 的 0.813(v5),range_excursion 2.70 ≈ de_pb9 2.73(低频干净),换 v6 完整保住甚至微升。**9-phase-bin(+DE-prox)输入端配方是带内细节+干净低频的真来源**,Stage 2a"输入聚合抹平相位"假设获正面支持。真实 cross-FRC 0.661 与 de_pb9/v19 同档(~0.66),未破 v19 纪录 0.6705,亦低于 TGV 0.702。
2. **臂B = 失败但有信息**:mean 归一化把 DC 梯度除以 M,eta 仍 0.5(为单变量未改)→ 有效 DC 步 = sum 版的 **1/16**,DC 太弱→低频完全不受锚→合成 range_excursion 爆到 10261、band_FRC 坍到 0.206(与 v20 坏快照同型,48/48 景一致)。真实 cross-FRC 0.6567 却"正常"——因 cross-FRC 带限+去均值,**对低频垃圾免疫**,看不见该臂的病(合成↔真实指标对该臂强烈矛盾正是此故;真实指标单独看会被骗)。**结论:mean 归一化非坏主意,但必须配 eta 重标定(eta_mean ≈ 16×eta_sum)**;臂B 证明"改归一化不改 eta"是错配。DE-prox(臂A'/de_pb9)才是当前可用的低频修复;mean-DC 需 eta 跟进臂再判。
3. **所有神经臂真实 cross-FRC 仍聚 ~0.66,未破 v19 0.6705 / TGV 0.702**:配方改善(合成 band 0.83)未转化为真实域增益——真实域天花板由配方之外的因素(算子误差/域差)设,呼应 η 跨域反转与 A3 算子误差轴。

**下一步**: (a) mean-DC + eta 重标定跟进臂(eta_mean 4-8 或小扫,配 mean 归一化的原则化步长);(b) depb9v6 regime 内纯 9-vs-4 bin 隔离(现可做,phase_bin_channels 单变量);(c) 收敛曲线(阶段2,5k checkpoint)确认两臂 30k 收敛性、并诊断臂B 病态是训练发散还是结构性(band_FRC-vs-step 曲线)。champion 候选 = depb9v6 配方 + DC 修复(eta 标定后)+ 更长训练。

**训练结果(回填,收敛曲线阶段2,5k checkpoint × 6,N=96,48/48 景全格,`output/v21_eval/v21_convergence_table.csv`)**:
1. **depb9v6 = 已收敛(平台)**: band_FRC 0.680(5k)→0.784(10k)→0.814(15k)→0.817(20k)→0.829(25k)→0.830(30k),25k→30k 增量 +0.001;range_excursion 全程 2.0-2.7(低频稳定)。配方训练稳定,30k final 可作 champion 配方基线。
2. **meanDC = step≥15k 训练期低频发散,非结构病/非加载配准问题**: band_FRC 峰值 0.754@10k 后崩塌(0.678@15k→0.218@25k→0.206@30k);range_excursion 1.29(5k)/1.36(10k)→**251(15k)→187(20k)→9703(25k)→10261(30k)**,mean_offset 同步跑飞(−0.06→−364)。10k 快照健康、网格门 ok → ACL-062 判读坐实:eta 错配(有效 DC 步 = sum 版 1/16)下低频无锚,训练动力学缓慢漂移至发散,发散起点 15k。**含义**: (a) eta 重标定跟进臂(eta_mean 4-8)必要性坐实,且新臂应监控 range_excursion-vs-step(发散在 band_FRC 之前先出现在 range_excursion,15k 处 band_FRC 仍 0.68 而 range 已 251——低频哨兵更灵敏);(b) 该臂 TB loss 高企是真发散,不是"没训完";跨臂聚合比较 raw loss 本就不可比(dc_weight/训练池/batch/输入通道逐臂不同,且 synth loss↔真实质量反相关有案,ACL-032/053)。
3. TGV 管线自检逐位复现(tgv_x_drz 0.7017/0.3558/23.03µm)→ 仪器可信。
4. 产物: 远端 `output/stage2b_bench/`(final+10 checkpoint 全表)、`output/v21_eval/`(曲线 png+csv,本地已同步)、`stageDE_v21_{leaderboard,recons,corrected,offset_probe}/`;TB run `algos/ep07_unet_sr/outputs/v21_convergence/`;辅助脚本留远端 `render_v21_{real_halves,convergence}.py`。

**涉及文件**: 无代码变更(实验记录;两臂用现有 harness 训练+评测)。

---

### [ACL-061] 2026-07-08（夜）— de_pb9 复活评测：owner 目视选中的臂过可信仪器 = "最均衡臂"（v14 的带内细节 + v19 的干净低频），DE 高通 prox 被证实为解耦机制；真实域打平纪录未破；细线为轻度过锐化而非幻觉

**问题诊断**: owner 目视 `solver_v14_de_pb9`（ACL-042/043 当年判"负"）细线最细、分离最好。核查（探查代理）发现:该臂**从未跑过偏移校正 cross-FRC**，死因仅是旧 synth PSNR 32.47（而 synth PSNR 与真实质量反相关，ACL-032）——被指向反方向的仪器杀掉。它与标准 v14 差 8 个轴(DE 高通 prox σ4、9 phase bins/in_ch14、warmstart=aligned_mean、dc_weight0、m12、no_drizzle=False、batch12、**训练池 v5 而非 v6**、40k)。DE prox 机制(`unroll.py:180`):prox 网络的 delta 加回前先减自身高斯模糊 → 只能注入高频、锁死低频。

**实验记录**(可信仪器复评,产物 `output/{depb9_eval,stageDE_*}` + TB `board/real_with_optical_depb9`;smoke 过=9-phase-bin/14通道路径渲染正常):
- **合成基准(48 景 N=96,带限 FRC-vs-GT,de_pb9 对 bench48 完全 held-out——它训练在 v5)**:

  | 臂 | band_FRC(25-40) | band_rmse | range_excursion | 真实 cross-FRC@30 |
  |---|---|---|---|---|
  | v14 | 0.8698 | 0.0149 | 33.58 | 0.649 |
  | **de_pb9** | **0.8133** | 0.0203 | **2.73** | **0.667** |
  | v19_etaB | 0.7792 | 0.0246 | 1.33 | 0.6705 |
  | tgv_oracle | 0.7650 | 0.0274 | 2.39 | — |

- **判读**:(1) de_pb9 带内 FRC 居 v14 与 v19 之间(0.813),band_rmse 略高于 v14 → 相对 v14 是"带内 FRC↓+band_rmse↑"= **轻度过锐化**,但绝对量小、远优于 v19/tgv,非幻觉(带内 FRC 高=与 GT 真结构强相关)。(2) **决定性优点:低频漂移被 DE prox 治好**——range_excursion 2.73 vs v14 的 33.58(12×)、fullband_rmse 0.21 vs 5.43(26×),接近 v19 水平。(3) de_pb9 = "v14 的带内细节(大部分)+ v19 的干净低频",**首个两者兼得的臂**——正是 owner 目视选中的性质。(4) 真实域 cross-FRC 0.667 **打平 v19 纪录(0.6705)但未破**;经典 TGV 0.702 仍领先。(5) OOD 信号:de_pb9 训练在 v5、评在 v6,却带内 FRC 超过域内 v19——跨分布泛化更强,呼应 OOD 主题。
- **光学第六列**:board/real_with_optical_depb9(optical|drizzle|TGV|v19|v14|de_pb9),de_pb9 偏移校正残余 0.100 HR px(与 v14/v19 同级),渲染相干、锐度与 v14/v19 相当。

**裁决 / 下一步**: de_pb9 证实 **DE 高通 prox 是"带内细节与低频漂移解耦"的可用机制**(与 mean-DC 归一化是殊途同归的两条路)。但它八轴齐变,赢因未拆:带内细节来自 9-phase-bin(Stage 2a 输入相位假设)? DE prox? 还是 v5 sharp 池?下一步 = v6 池单变量拆解臂(9-phase-bin 单独 / mean-DC+η / DE-prox 单独),成对锚定 v14,20k,可信仪器评测。owner 目视判断再次跑赢反相关旧指标——记忆 [[record-process-artifacts]] 的价值实证。v5 池已删,纯池归因留空(如需重生成再议)。

**涉及文件**: 无代码变更(实验记录条目;评测复用现有 harness)。

---

### [ACL-060] 2026-07-08（夜）— OOD 套件 A3 轴（算子误差）旋钮交付：stage2b harness 新增 DC σ 覆盖与 DC shift 抖动（默认关＝字节级旧行为）；A3 扫描档位预注册

**问题诊断**: OOD 泛化套件设计稿（`research_log/ood_generalization_suite_design.md`）A3 轴 = 算子误差鲁棒性——σ 不可自标定（ACL-059）后，σ 处理转为"鲁棒带"；η 跨域反转（ACL-054）的机理验证也需要"对算子撒谎"的对照实验。A3 零生成成本（复用 bench48 现有 GT/burst，只扰动喂给重建器的算子参数），排 OOD 五轴之首。

**修改内容**（`algos/ep07_unet_sr/scripts/run_stage2b_synth_benchmark.py`）:
1. **`--dc-sigma-override <float>`**：把喂给 DC/重建器的 PSF σ 替换为固定各向同性高斯（渲染/GT 不动，只骗算子）。神经路径＝替换 psf_override 内容（新纯函数 `resolve_neural_psf_params`，默认分支逐字段复现旧 ScenePSF 构造）；经典路径＝替换 oracle/portable 传给 TGV/MAP-TV 的 psf_sigma（新纯函数 `resolve_dc_sigma`，覆盖优先于两条件）。
2. **`--dc-shift-jitter-std-px <float>` + `--dc-shift-jitter-seed`（默认 20260708）**：给喂给重建器的每帧 shift 加零均值高斯抖动（LR px）。确定性种子 =（seed, crc32(scene_id)）→ 神经与经典两阶段（不同进程）对同景同帧施加**完全相同**的扰动；抖动施加于**全帧数组、先于前缀选择**，故帧 i 的扰动与所选 N 无关（成对阶梯不破坏）。std≤0 原样直通（默认关合同）。
3. **臂命名与分组**：扰动参数烧进臂名（`v14__dcsig0p25`、`tgv__oracle__jit0p1`、可复合 `__dcsig0p4__jit0p05`），重建落独立 recons/ 目录，gate/metrics 自动当独立臂分组，与未扰动行永不混淆；manifest 行记录 dc_sigma_override / jitter std+seed / base_arm。
4. **测试**（`tests/test_stage2b_benchmark.py` 新增 5 项，ep07 全套 **111 passed + 3 skipped**）：后缀命名、抖动确定性/跨阶段一致性/前缀配对性/默认直通、σ 解析两路径、psf 参数默认分支逐字段复现旧构造、CLI 默认全关。

**预期效果（A3 预注册档位，跑前立此为据）**: 池 = bench48 现有 GT/burst，N=96 单档先跑；评测臂 = 冻结 v14、v19_etaB + TGV oracle/portable 对照；**σ 覆盖档 σ_DC ∈ {0.10, 0.25, 0.40}**（对照 = 每景真 σ 的现有 oracle 行）；**shift 抖动档 ∈ {0.05, 0.10, 0.20} px**（对照 = 现有未扰动行；种子固定 20260708）。判读：各臂指标 vs 扰动强度的退化曲线；预期若 η 反转解读正确，v19（η0.09）对算子误差的退化应显著缓于 v14（η0.5）——这是 ACL-054 机理的直接检验。评测指标与 gate 流程照旧（帧级扰动为亚像素量级，网格门衡量的常数全局偏移不应显著移动；若某扰动臂 gate abort，如实报，不改门限）。

**训练结果（回填,夜间自治）**: 预注册神经扫描(v14/v19_etaB × 6 单扰动档)日间已跑完(`stage2b_bench/logs/a3_neural_session.log` 全 rc=0);depb9v6_30k 同档为 **non-prereg 探索性补充**(夜间队列阶段A);经典 TGV oracle 完成 dcsig0.10/0.25(0.40 被 SIGTERM,jitter 三档排队列阶段D,dcsig 档 portable≡oracle 属代码等价故只跑 oracle——`resolve_dc_sigma` 下 override 压过两条件)。band_FRC(25-40) 均值(Δ=vs 各自 N=96 未扰动基线):

| 臂 | 基线 | σ_DC 0.40 | σ_DC 0.25 | σ_DC 0.10 | jit 0.05px | jit 0.10px | jit 0.20px |
|---|---|---|---|---|---|---|---|
| v14(η0.5) | 0.870 | 0.834(−.036) | 0.802(−.067) | **0.712(−.157)** | 0.869(−.000) | 0.864(−.006) | 0.829(−.041) |
| v19_etaB(η0.09) | 0.779 | 0.776(−.004) | 0.778(−.001) | **0.775(−.004)** | 0.778(−.001) | 0.774(−.006) | 0.750(−.029) |
| depb9v6*(non-prereg) | 0.830 | 0.841(+.011) | 0.804(−.025) | 0.727(−.103) | 0.830(±.000) | 0.817(−.012) | 0.782(−.048) |
| tgv_oracle | 0.765 | (待补,SIGTERM) | 0.665(−.100) | **0.593(−.172)** | 待D | 待D | 待D |

**判读**:
1. **σ 轴:η 反转机理证实(预注册判据达成)**——v19 对 σ 谎言在 0.10-0.40 全程平坦(|Δ|≤0.004),v14 随谎言增大单调陡降(σ0.1 处 −0.157),TGV 比 v14 更陡(−0.172)。弱 η = DC 步小,算子误差传不进重建;ACL-054 的机理解读拿到直接实验支持。v19 在 σ0.1 档(0.775)反超 v14(0.712)与 TGV(0.593)——"portable 政权"下神经弱η臂优势成立。
2. **jitter 轴在现实量级无害**: 0.05-0.10px 三臂 |Δ|≤0.012(真实精修残余正是 0.05-0.1px)→ **当前对齐质量下 shift 误差不是真实域主凶**;0.20px 全臂 −0.03~−0.05。
3. **与 ACL-063 合读(核心权衡,champion 的新坐标系)**: 同一机理的两面——η 低=靠先验,对算子误差鲁棒,但先验抹 OOD 内容(v19 抹 68.6% 孤立点);η 高=证据忠实(点活),但算子谎言全吃(σ0.1 −0.157)。**鲁棒性不能靠弱 DC 买**。可行路线:DE-prox 系(depb9v6 σ0.25/0.40 近平、点保留 0.60-0.73,两轴均居中偏优)+ v7 训练期算子随机化(把 σ/shift 谎言变成分布内)+ σ 硬件标定(ACL-059 可选项,直接消灭主谎言)。
4. **真实域含义**: σ 在真实靶不可自标定(ACL-059)、占位 0.5 疑偏大(最锐边 ~0.28)→ v14 类强 DC 臂在真实域的损失很可能主要是 σ 错配,这给 0.66 天花板提供了**具体的第一嫌疑与可干预路径**(σ 标定/鲁棒化的预期收益上界 ~0.07-0.16 band 点,按 σ 谎言幅度)。
5. 方法论警示再确认: v14 的 psnr/fullband 列在全部行都病态(低频漂移,ACL-054 已知),band 指标不受影响——读 A3 表只用 band_FRC/band_rmse。

**涉及文件**: `algos/ep07_unet_sr/scripts/run_stage2b_synth_benchmark.py`、`algos/ep07_unet_sr/tests/test_stage2b_benchmark.py`。

---


### [ACL-059] 2026-07-08（傍晚）— σ 线收口：Step 2 真实数据 = 估计器合法拒绝（0/8 边过质量门）；结论 = 系统 σ 在本靶上不可自标定；σ 处理策略从"点校准"转向"鲁棒带"

**实验记录**（产物 `remote_inbox/20260712_sigma/`）:
1. **Step 2（真实 248 帧，ESF 内核，detector_box 孔径）：exit 4 合法拒绝**——检出 8 条候选直边，0 条过预注册质量门（r²≥0.90 且 amp_SNR≥5.0）：6 条 SNR 良好（8.7-13.7）但 r² 仅 0.67-0.77（剖面非参数化阶跃形），2 条双双不过。仅作描述（无效数字）：逐边原始 σ̂ 散布 **0.277-0.942**——与"边缘软度由各处热扩散决定、逐边不同"一致，实证了设计稿 §2.5 警示与 owner"靶上无遮挡型边缘"的判断。程序拒绝给数而非硬给，通用性纪律成立。
2. **σ 简并证伪 6 景补跑（E1/E2 归档）**：FAIL 复现——median |rel err| 50.4%（gaussian-only 56.0%），逐景散布 0.21-1.30 无系统性，三分位符号乱跳（+0.373/−0.438/+0.034）。ACL-056 简并结论在独立子样上归档完毕。
3. **σ 线总结论**：自监督类（E1/E2）原理性简并；场景先验类（ESF）合成台验证通过（ACL-058 修正后 median 3-4%）但本真实靶无净阶跃边而合法拒绝。**系统 σ 在本靶上不可自标定**。描述性参考（不作校准）：最锐边 ~0.28 提示占位 0.5 偏大、EP09 路线 1.129 基本排除。
4. **策略转向（与 owner OOD 方向合流）**：σ 不再追求点校准——(a) 把 σ 当**不确定带 [~0.1, ~0.4]**：DC 喂偏 σ/shift 的鲁棒性退化曲线成为 OOD 套件正式轴（冻结臂纯推理可测）；v7 池的算子误差 DR（σ 扰动，从未试过）动机升级。(b) 可选硬件路线留 owner：一次已知锐边标定采集（加热刀口/狭缝,经典热 MTF 法）可把 σ 钉死；有采集窗口值得做，无则不阻塞主线。
5. ESF 程序保留为**通用交付物**：任何含净阶跃边的 burst（含未来标定采集）可直接跑；合成台验证已过，程序冻结。

**涉及文件**: 无代码变更（实验与裁决条目）。

---


### [ACL-058] 2026-07-08（傍晚）— Step 1 高估 root-cause = bench 真值又缺一项（isothermal 场景 edge_sigma 软化，属 x 不属 A）；真值改二次和后对原始数据离线重判 **PASS**（median 4.1%）；Step 2 解读警示预注册落盘；<6 景优雅跳过修复

**问题诊断**:
1. Step 1 FAIL（+28~31% 系统性高估）与交付抽查（干净斜边 −6.7%~−0.4%）矛盾。本地诊断（`tmp/analysis_esf_fail_diagnosis.py`，四假设逐一检验）：旋转角/边曲率/孔径选型/噪声相关性全不显著；真因 = `generate_training_pool.py` 的 isothermal 分支经 `render_isothermal_field`（`tcforge/realism.py`）对场景电平场**无条件施加 edge_sigma=0.6 HR px 高斯软化**（config 键 `temperature_isothermal.edge_sigma`，生成器默认 1.4）——它属于场景 x 而非算子 A，但 ESF 测的是总过渡宽度。"淬取项" sqrt(σ̂²−σ_true²) 中位 0.279 LR px、gaussian-only CV 3.8%（常数状），与 0.6 HR 的离散有效换算 0.2965 LR 吻合；把它按二次和代回真值，诊断内重算 median 0.3139→0.047。
2. **叙事线**：估计器两次精确暴露的都是真值标签缺口——先是"标称 vs 有效 σ"（ACL-057 发现 B），再是场景 edge_sigma。E3 测的始终是帧里物理存在的模糊，错的都是标签；这正是"程序通用、标签要核"的教科书案例。
3. 二级残差（不进真值，文档化）：旋转内容 SSAA/box 足迹的角度相关几何反走样宽度 ~0.16-0.30 LR px（order-0/1 皆然，非字面"插值"机制）；修正后剩余 ~4% 已含此项，在 15% 线内。

**修改内容**:
1. **bench 真值 = sqrt(σ_psf_eff² + σ_scene_edge²)**（`scene_edge_sigma_lr`/`combined_true_sigma_lr`，edge 项同样过离散有效换算；来源优先级 `--edge-sigma-hr` > `--pool-config` > 池目录内 config json，找不到 → 0 并显式 log，绝不猜值）；bench 行新增三列 `sigma_true_psf_only`/`sigma_scene_edge`/`sigma_true_total`（判定基准）。
2. **`--reverdict-from`**：对既有 bench_rows.csv 离线重判（兼容旧 schema，旧 `sigma_true` 视为 psf-only），不重跑估计。
3. **`safe_evaluate_prereg`**：<6 有效景优雅 SKIPPED（CLI exit 5，逐景产物照常落盘），修复 σ 简并证伪归档的 ValueError 崩溃路径；e1e2/esf 两内核 bench 驱动共用。
4. 设计稿新增 **§2.5 Step 2 解读警示**（预注册）：σ̂_real 是系统 σ 的**上界**（真实热场边缘未必理想阶跃）；DC 算子的 σ 不含场景软度（属 x 不属 A）；内检 = 多边缘一致性 + EP09 路线对照；**禁止**收缩系数式数据集特异调校（owner 标准规则）。
5. 测试 ep09 **26 passed**（新增 3：真值合成语义、离线重判（旧 schema + 零 edge 对照）、不足景跳过路径）。

**训练结果**（对远端 Step 1 原始 bench_rows.csv 离线重判，产物 `output/sigma_esf_bench_reverdict/`）:
- **Step 1 修正后 = PASS**：median |rel err| 0.3139→**0.0409**（27 可评景）／ gaussian-only 0.2688→**0.0305**；systematic_bias=false（三分位中位 −0.041/−0.016/−0.041，中段 CI 过零）；scene_edge_sigma_lr=0.2965（source=pool_config）。21/48 无边景不变（可评性限制如实保留）。
- **程序按预注册冻结，Step 2（真实 248 帧）解锁**：材料已备（`output/real248_burst.npy` 在远端 + shifts/manifest 已拉回），执行与解读必须过设计稿 §2.5。

**涉及文件**: `algos/ep09_psf_calibration/src/psf_calibration/esf_selfcal.py`、`src/psf_calibration/sigma_selfcal.py`（safe_evaluate_prereg）、`scripts/sigma_selfcal.py`（--reverdict-from/--pool-config/--edge-sigma-hr、exit 5）、`tests/test_esf_selfcal.py`、`research_log/sigma_selfcal_prereg_design.md`（状态块 + §2.5）。

---


### [ACL-057] 2026-07-08（午后）— σ 自校准换代内核 E3（多帧投影 ESF）交付：参数化场景先验打破 E1/E2 简并；附带两项校准层发现（双线性核不可当常数 tent；**池渲染"标称 σ vs 有效 σ"缺口**）

**问题诊断**: ACL-056 证伪 E1/E2 后的换代路线——自监督目标测不了 σ，因为补偿场景总能复现观测；E3 引入**参数化场景先验**（场景含直边阶跃，热靶通用结构，非数据集特异）：直边法向的观测过渡宽度 = 阶跃 ⊗ Gauss(σ) ⊗ 已知孔径，补偿者无法再自由，σ 可辨识。

**修改内容**（`algos/ep09_psf_calibration/src/psf_calibration/esf_selfcal.py` + CLI `--kernel esf`）:
1. **管线**: 认证参考 scatter 的快速 SAA → 梯度脊 + 方向一致 RANSAC + TLS 精化的直边检测（无合格直边 → 显式拒绝输出 σ̂，通用模式 exit 4，no silent fallback）→ 全帧像素中心经精修 shifts 投影到边法向（多帧亚像素相位 = 超分密采样的 1D 剖面）→ soft-L1 稳健 erf 拟合 → 逐边 σ̂ 中位数聚合 + bootstrap CI（≥4 边按边重采样，否则按帧）+ 边间 spread 预警（各向异性 PSF / 阶跃假设破坏）。
2. **孔径模型（零自由参数，全部可解析推导）**: `pool_block_average` = scale² 离散点栅 **× 每帧双线性节点对 {−f,1−f}（权 {1−f,f}，f=frac(scale·shift) 为每帧已知常数）** 的精确复合，光栅化 box（方差 1/12 HR²）以匹配方差高斯并入 extra_var；`detector_box` = 连续 1 LR px box 的 Gauss–Legendre 求积（真实探测器）；`pool_point`/`point` 对应点采样。`--aperture auto` 按池 metadata 的 forward_mode 选 preset。
3. **校准层发现 A**: 把双线性重采样当常数 tent（方差 1/6 HR²）从 σ 中扣除，在平滑场下过扣——实测 σ=0.2 处 σ̂ 偏 −30%；tent 观仅对粗糙场成立。→ 改为上面 2 的每帧精确建模。完全忽略孔径则 σ=0.4 处偏 +18%。两种简化均被实测否决。
4. **校准层发现 B（对全项目有含义）**: **池渲染的"标称 σ"在小 σ 端高估实际施加的模糊**——scipy 截断离散高斯核的实际方差不足：标称 0.15 LR px → 有效 **0.044**；0.20 → **0.142**；≥0.35 缺口 <1%（新助手 `discrete_gaussian_effective_sigma`，scipy 核构造逐式一致）。估计器测到的是帧里物理存在的有效模糊，与该式精确吻合（σ̂=0.041/0.136 vs 有效 0.044/0.142）。**bench 真值改用有效 σ 对比（对比层修复，ACL-049 网格偏移同险类；nominal 列保留）**。含义待查：凡把小标称 σ 喂给离散高斯的环节（池渲染、DC 算子、σ̂_real 与 EP09 路线值的解读）都存在此语义缺口。
5. **抽查（合成斜边，池渲染管线独立生成，角度 25/30/−40/65°）**: σ̂ vs 有效真值相对误差 **−6.7%/−4.2%/−0.8%/−0.4%**（标称 0.15/0.20/0.35/0.55）——全程在预注册 15% 线内，小 σ 端绝对差仅 0.003 LR px。
6. **测试**: ep09 套件 **23 passed**（新增 7：孔径 preset/有效 σ 助手/纯拟合数学找回/池渲染斜边找回/box-σ 分离（σ_true≈0 时 σ̂≤0.15，孔径未被吸收）/无边拒绝/CLI esf 通用模式）。σ 下界命中 = "PSF 小于可分辨"合法结论保留并打标，仅上界命中作废。

**预期效果**: Step 1 = bench48 全景验证（判定线不变：median |rel err|≤15% + 噪声三分位无偏置；椭圆/airy 景标注退化拟合不剔除，gaussian-only 中位数并报）；通过则程序冻结、原样上真实 248 帧（`--aperture auto`→detector_box）。

**训练结果**（2026-07-08 午后回填，产物 `remote_inbox/20260711_stage2b/sigma_esf_bench/`）:
- **Step 1 = FAIL（合法负结果，预注册线未过）**: median |rel err| = 0.3139（全体）/ **0.2688（gaussian-only）** vs 线 0.15；27/48 景可评（21 景无合格直边）；**系统性偏正 +0.28~0.32（噪声三分位中位数全正）**。与交付时抽查（干净斜边 −6.7%~−0.4%）显著矛盾 → 高估是 bench 场景内容相关的，root-cause 诊断进行中（头号嫌疑 H_rot：v6 池 360° 旋转的 ndimage 插值 + SSAA 使 GT 边缘本身非理想阶跃，插值展宽被 σ̂ 吸收；备选：边缘曲率/孔径 preset/噪声偏置）。诊断结论回填至下一条目。
- **算子一致性核查（发现 B 的"待查项"关闭）**: 渲染（scipy round 路径）与 DC（torch ceil 路径，`forward_torch.py`）的离散高斯核在 σ∈[0.10,1.00] 全程数值恒等（≤1e-16）——有效 σ 缺口两侧同步存在，**训练内部 render-vs-DC 无错配**；real_eval 占位 σ=0.5 处缺口 <0.01%。缺口只是"标称 vs 物理"的元数据语义问题（bench 对比层已改用有效 σ）。脚注：`dataset.py:598-601` 的 None 强转使 DC 快速 round 路径成为死代码（无数值后果）。分析脚本 `tmp/analysis_sigma_kernel_check.py`（保留）。
- **σ 简并证伪归档偏差**: 按 ACL-056 建议命令 `--scene-limit 3` 跑崩（预注册判定需 ≥6 景，ValueError 而非优雅 FAIL）；3 景 E(σ) 平坦曲线 csv/png 已归档（`sigma_selfcal_falsify/scenes/`），≥6 景补跑待排。CLI 的 min-scenes 校验应前置（小修待办）。

**涉及文件**: `algos/ep09_psf_calibration/src/psf_calibration/esf_selfcal.py`（新）、`scripts/sigma_selfcal.py`（--kernel/--aperture/--esf-*）、`tests/test_esf_selfcal.py`（新）、`research_log/sigma_selfcal_prereg_design.md`（状态块更新）。

---


### [ACL-056] 2026-07-08（日间）— σ 自校准估计器交付，但**核心负发现：设计稿的 E1/E2 自监督估计器对 σ 近乎简并**——一并解释 EP09 三路发散与 0a"测不了 σ"的旧账；48 景校准战役取消，预注册 Step 2/3 暂停待估计器换代

**问题诊断 / 实验记录**:
1. 按 `research_log/sigma_selfcal_prereg_design.md` 实现了 E1（留帧前向预测,bootstrap CI）+ E2（残差谱平坦度/自相关衰减）双估计器与 bench 验证驱动（预注册判定逻辑代码化:median |rel err|≤15% 且噪声三分位无系统偏置）。
2. **简并性发现（本条目主结果）**:该算子族中高斯 blur 与亚像素 shift、块降采样（有效地）交换——对任意假设 σ′,补偿重建 x̂=B_σ′⁻¹B_σx 可**精确**复现全部帧（连续极限严格,离散网格上高斯无零点、逐频可逆）→ 留帧预测与残差白化度**原理上无法辨识 σ**,仅边界效应（mode=constant)与噪声放大留下微弱信号。
3. **实测证据（160px 裁窗、48 帧、真 σ=0.4）**:CG 40 步时 σ̂=0.60（CI 窄=系统偏差）;CG 150-300 步、λ 减小后 E(σ) 曲线整体趋平且单调偏向大 σ（欠收敛伪信号消失后无判别力）;非负底板 PGD 变体在硬边缘结构场景上也只有噪声级区分。tiny 图单测能"找回"σ 纯系边界效应（测试注释已写明,保留作管线守护）。
4. **旧账贯通**:该简并性从机理上解释了 EP09 三路标定发散（0.119/0.2257/1.129——不同路线=不同隐式先验,各自打破简并的方式不同）与 0a"σ̂ 卡网格边缘、仪器测不了 σ"（ACL-044/046)。σ 不是"没测准",是**这一类自监督程序原理上测不了**。

**修改内容**（代码全部交付,测试 16/16 绿——11 旧 + 5 新）:
- `algos/ep09_psf_calibration/src/psf_calibration/sigma_selfcal.py`（新）:E1/E2 估计器、CG 均值法方程重建器（λ 为经验对角尺度的相对值,全 σ 网格恒定）、`crop_lr` 中心裁窗计算旋钮、bench 验证驱动 + `evaluate_prereg` 判定、曲线/汇总 csv+json+png 落盘;模块 docstring 顶部**显著标注简并性限制**。
- `algos/ep09_psf_calibration/scripts/sigma_selfcal.py`（新 CLI）:通用 burst 模式（burst npy + dx_px/dy_px CSV,分布无关）与 bench 模式;help 中警告"预期 Step 1 FAIL,仅作证伪检查（--scene-limit 3),勿跑 48 景校准战役"。
- `algos/ep09_psf_calibration/tests/test_sigma_selfcal.py`（新,5 项):找回(tiny,守管线)、白化度可分性、预注册判定三分支、bench 模式产物、CLI smoke。
- 设计稿 `sigma_selfcal_prereg_design.md` 加顶部状态块:E1/E2 被本条目证伪,Step 2/3 暂停。

**预期效果 / 下一步（主循环裁决）**: 可辨识的 σ 估计需要"补偿器无法满足的约束"。推荐换代方向:**多帧投影 ESF**——用精修 shifts（残余 0.01-0.07px,ACL-048）把全部帧的边缘剖面投影到公共超分辨 1D 轴,拟合 erf⊗box 参数模型（参数化边缘先验即打破简并的约束;`psf_calibration/esf_fitting.py` 的 erf_model/fit_esf_profile 可复用;bench 与真实芯片场景都富含直边,"无可用边缘"时程序显式报告而非硬给数）。预注册验收框架（bench 先行、median≤15%、三分位无偏置）**原样保留**,只换估计器内核。

**涉及文件**: `algos/ep09_psf_calibration/src/psf_calibration/sigma_selfcal.py`、`algos/ep09_psf_calibration/scripts/sigma_selfcal.py`、`algos/ep09_psf_calibration/tests/test_sigma_selfcal.py`、`research_log/sigma_selfcal_prereg_design.md`。

---

### [ACL-055] 2026-07-08（上午）— DC sum→mean 归一化选项 + 基准低频稳定性三指标（算子/DC 主攻线的代码部分；结果待回填）

**问题诊断**: ACL-054 统一假设——DC 梯度是 M 帧**未归一化求和**（`forward_torch.py` 的 DC 目标对 N 维求和），有效步长 = η×M×算子增益：(1) 步长与证据预算 M 耦合（m32 崩溃，ACL-051）；(2) 低频算子增益最大 → η=0.5 时低频越过迭代稳定界（v14 域内低频幅值漂移：range 膨胀 7-10×、mean offset +1.1°C，ACL-054 核实 B），而带限指标对此失明。

**修改内容**:
1. **`unroll.py`**：`UnrolledSolver` 新参 `dc_normalize`（"sum"=旧行为逐字节不变（默认，测试钉住）；"mean"=DC 梯度除以 `y_burst.shape[1]`，步长与 M 解耦）。等效换算：固定 M 下 η_mean = M×η_sum（M=16：旧 0.09 ≈ 新 1.44，旧 0.5 ≈ 新 8.0）——归一化后 η 重扫的起点。注：frame_mask 存在时分母仍取 M（常数尺度、分布无关；补齐重复帧语义与 ACL-051 一致）。
2. **`config.py`**：字段 `solver_dc_normalize: str = "sum"` + `validate()` 校验 + CLI `--solver-dc-normalize {sum,mean}`。
3. **`solver_train.py::build_solver`**：`getattr(config, "solver_dc_normalize", "sum")`——旧 checkpoint 的 config dict/对象缺该字段时必回落 sum（推理兼容,有测试）。
4. **`run_stage2b_synth_benchmark.py`**：metrics 阶段新增三列 `fullband_rmse` / `mean_offset` / `range_excursion`（相对每景自身 GT，分布无关），进 summary 聚合——v14 式低频漂移从此在体检表直接可见；已有 recons 重跑 metrics 即可补列（metrics 纯读 recons/）。
5. **测试** `tests/test_dc_normalize.py` 8 项（默认 sum 字节级等价、复制帧下 mean≡单帧 sum、mean 的 M 不变性、坏值拒绝、CLI 解析、validate 校验、旧 config dict 重建回落 sum、缺属性对象 build_solver 回落 sum）+ `test_stage2b_benchmark.py` 新增 lowfreq_stability 手工小例。**ep07 全套 106 passed + 3 skipped**。

**预期效果**: 为归一化后的 η 重扫提供工具（起点 η_mean ∈ {1.44, 8.0} 附近）；基准台可直接观测"重 DC 是否还有低频失稳"。是否用 mean 模式重训对照臂＝下一步实验决策（owner/主循环），本条目只交付机制。**owner 规则遵守**：无任何数据集特异性常数；默认行为逐字节不变。

**训练结果**: 待回填（无训练；归一化 η 重扫臂由后续条目记录）。

**涉及文件**: `algos/ep07_unet_sr/src/unet_sr/unroll.py`、`src/unet_sr/config.py`、`src/unet_sr/solver_train.py`、`scripts/run_stage2b_synth_benchmark.py`、`tests/test_dc_normalize.py`、`tests/test_stage2b_benchmark.py`。

---

### [ACL-054] 2026-07-08（凌晨）— Stage 2b 合成基准 harness 交付（H1 域差 vs H2 架构 判决实验；代码+测试，结果待回填）

**问题诊断**: ACL-053 冠军臂负结果第三次呈现 synth↑/real↓ 反相关；owner 裁决单数据集调参终止。差距根因二选一：H1（合成-真实域差，先验兑现不了）vs H2（架构把信息留在桌上）。判决所需的关键对照——**TGV/MAP-TV 在我们自己合成集上的表现——此前从未测过**。设计稿：`research_log/stage2b_synth_benchmark_design.md`。

**修改内容**:
1. **`configs/synthetic/pool_2x_v6_bench48.json`**（新）：v6 训练池配置逐字拷贝，仅改 6 键（num_scenes=48、seed=20260708 与训练池种子不相交、n_frames_per_scene=96 固定以支持成对 N 阶梯、output_dir、dataset 名、注释）。有测试强制该"仅 6 键差异"不变量。
2. **`algos/ep07_unet_sr/scripts/run_stage2b_synth_benchmark.py`**（新）：四阶段 harness——
   - `recon-classical`：TGV/MAP-TV ×（portable=真实数据同款 σ 占位 / oracle=每景真 σ）× N∈{24,48,96} 前缀阶梯；**景级 spawn 进程池**（owner 需求：`--workers` 默认 55，BLAS 四件套线程环境在父进程 spawn 前权威设置=1，子进程 fresh import 生效；`--workers>1` 时强制方法内部 workers=1 防过订阅）；单任务 >300s 打 SLOW_TASKS 标记供 owner 裁剪。TGV/MAP-TV 超参与 stage0h 真实基线逐字一致。
   - `recon-neural`：checkpoint 参数化多臂（TrainingConfig 按 dataclass 字段过滤重建 + build_solver 复用，兼容 fusion 臂），DC 用**每景真 PSF**（见 3），`output_grid=native`（GT 与训练同网格）。
   - `gate`：ACL-049 教训制度化——各臂 vs GT 迭代式相位相关偏移测量（单发抛物线峰值对亚像素分数系统低估、但为收缩映射，迭代 4 轮收敛至 <0.025 px，已验证），常数校正后残余中位 ≤0.15 px 才放行，>1.5 px 或校正失败 → exit 2 拒产表。
   - `metrics`：与现役排行榜同一套 `frc_curve_v2` 口径 → 带限 FRC-vs-GT（记 24/30µm 点值、25-40µm 带均值、1/7 与 half-bit cutoff）+ band RMSE（加窗 FFT 带通差值 RMS）+ PSNR；长表 + 按 N/噪声三分位/ΔT 三分位分层汇总 CSV；缺失重建显式 log（no silent skip）。
3. **`real_eval.infer_solver_from_burst_full_halo` 增加 `psf_override` 参数**（默认 None=历史真实评测路径字节不变，有测试钉住默认值）：合成域算子已知，DC 不该吃 σ=0.5 占位。
4. **测试** `tests/test_stage2b_benchmark.py`：7 项（配置不变量、前缀阶梯、gate 判决表、迭代偏移往返、带指标 sanity、三分位分箱、psf_override 默认关）。ep07 全套 **97 passed + 3 skipped**。

**预期效果**: 判决读法——神经（域内、oracle 条件）明显>TGV → H1，修数据（v7 池）；打平/落后 → H2，修网络（prox 架构轮在本基准台迭代，真实 248 帧只当期末考）。附带产出：v20-vs-v19 域内差 = 迁移损耗直接量化；η 跨域稳健性表（v17 全家 final 尚存）。

**训练结果**（2026-07-08 上午回填；产物 `output/stage2b_bench/`，η 附录与 batch 探测另补）:
- **执行过程**（本身有方法论价值）：冒烟单任务 T=172s（独占）；55-way 并行下内存带宽争抢使并发单任务膨胀至 N=24/48/96 → 2082/3205/4896s；按工作量模型（剩余任务构成 ÷ 55 worker）估 ETA 优于朴素累计吞吐。网格门全臂通过（v14 残余 0.004 px；v19 corrected 后达标）。经典 576/576、主力神经 432/432 全量完成，48 景 count 满格。
- **主表（带限 FRC-vs-GT 25-40µm 均值，N=24/48/96）**：MAP-TV oracle 0.412/0.527/0.648；TGV oracle 0.524/0.661/0.765；TGV portable 0.676/0.747/0.789；**v14(η0.5) 0.848/0.861/0.870**；v19_etaB(η0.09) 0.749/0.769/0.779；v20_champion 0.43/0.42/0.42。band RMSE 同序（v14 0.015 最优）。cutoff 列低频毛刺（950µm+）弃用，判读只认带内值（设计预注册）。
- **判决 = H1（域差），且比预期更彻底**：v14 用 N=24（0.848）压过 TGV-oracle 用 N=96（0.765）——先验价值巨大、容量不是瓶颈、**prox 架构轮正式停**。神经优势随帧数减少而放大（N=24 时 +0.32），符合"先验在数据弱时最值钱"。
- **η 跨域反转（本轮最深洞察）**：域内 v14(η0.5) ≫ v19(η0.09)，真实域反之——**低 η 在真实域的收益本质是补偿真实算子误差（σ=0.5 占位、shift 残余），不是先验权重更优**。真实域与 0.87 之间隔着的主要是"算子错了"。
- **核实 A（v20 崩溃鉴别）**：v20 在自己训练池 3 景带限 FRC ≈0.29（比 held-out 还差）→ 过拟合证伪；config 与正常的 v19 逐字段几乎相同、加载零丢弃、配准干净 → 加载 bug 排除。**v20_champion checkpoint 本身是坏快照（整个频谱不对），正式判死**；其真实域 0.647"仅小幅倒退"的表象=带内 cross-FRC 对此病变不敏感。流程新规：**每个新 checkpoint 先过合成基准体检**。产物 `output/stage2b_v20check/`。
- **核实 B（v14 负 PSNR 溯源）**：真实低频幅值漂移（输出 std 9.0 vs GT 0.94、均值 +1.1°C、range 膨胀 7-10×），带内不受影响（band RMSE 0.0149 全场最优）；三臂输出约定逐字段相同，非量纲/harness bug。PSNR 用每景 GT 动态范围（此景仅 3.5°C）故暴负。
- **统一假设（串起 4 条旧线索）**：DC 为未归一化 M 帧求和（`forward_torch.py:715`），有效步长=η×M×算子增益；低频增益最大 → η=0.5 时低频越过迭代稳定界（v14 漂移），高频被 PSF 衰减仍稳定且受强约束（v14 带内最优）；η=0.09 全带回稳但 DC 变弱（域内 0.78）。η 敏感性、m32 崩溃、v14 低频漂移、真实域 η 反转——同一病根：**DC 步长从未原则化**。
- **裁决后行动（owner 批准）**：(1) 主攻"算子/DC 线"——σ 自校准 + 测试时 shift 精化 + DC sum→mean 归一化与 η 重扫（合成台带 GT 即时反馈低频稳定性）；(2) v7 池第二优先级；(3) 架构轮停。**owner 两条新规**：过程产物（分析脚本/图/中间输出）一律保留、发现随时入档；**所有校准必须做成分布无关的通用方法**（自校准流程，换分布可复用），禁止对本数据的特异性常数调校。
- **终版附录回填（2026-07-08 午，产物 `remote_inbox/20260711_stage2b/`，160 文件）**：
  - **η 域内曲线（7 臂 20k 同参族，@30µm N=96）**：0.773（η0.0625）/0.772（0.09）/0.786（0.125）/0.813（0.1875）/**0.830（0.25，网格内峰）**/0.700（1.0）/0.475（2.0）；v14（η0.5，50k）0.855。**η*_synth ≥0.25 vs 真实域 η*=0.09——跨域反转坐实为整条曲线**，"低 η 的真实域收益=补偿算子误差"从推断升级为测量。
  - **低频稳定性三列（N=96：fullband_rmse/mean_offset/range_excursion）**：v19_etaB 最稳（0.136/−0.044/1.33×）；v14 带内王者但低频漂移显著（5.43/−0.149/33.6×）；v20_champion 低频爆炸（15202/−2160/104642×）——坏快照尸检闭环，也证明三新列作为体检项有效。
  - **batch 吞吐探测（GPU 独占）**：4/6/8/12/16 → 25.7/27.6/**28.4/28.4**/26.0 samples/s，VRAM 10.1/17.9/19.0/26.4/31.9G——吞吐平台在 batch 8-12，**新系列定 batch 8（19G）**，owner "~20G 最快"直觉与 ACL-053 探测双复现。
  - **过程教训**：55-way 并发下内存带宽饱和使单任务慢化 5-30×（独占 T=172s → 均值 2082-4896s），吞吐估算必须按带宽模型而非独占外推；夜间跑腿代理轮询链断裂致 GPU 空转 ~8.5h（01:30-10:47）——babysitter 需要自愈式双通道唤醒（教训入下次 runbook 模板）。SLOW_TASKS 539/540 属饱和预期；失败景 0；MISSING 0。

**涉及文件**: `configs/synthetic/pool_2x_v6_bench48.json`、`algos/ep07_unet_sr/scripts/run_stage2b_synth_benchmark.py`、`algos/ep07_unet_sr/tests/test_stage2b_benchmark.py`、`algos/ep07_unet_sr/src/unet_sr/real_eval.py`、`research_log/stage2b_synth_benchmark_design.md`。

---

### [ACL-053] 2026-07-07（晚，无人值守）— 组合臂 η0.09×band=0.6705@30µm（B 增量为正）；冠军流程启动：batch 8 重锚 + 成对 50k 过夜对决 TGV

**实验记录**:
1. **A×B 组合臂**（solver_v19_etaB_20k，η0.09 + band gate25_40 w0.5，v14 同参）：**0.6705@30µm** / 0.3450@24µm / cutoff 25.45，synth 32.79。相对 η0.09 单臂（0.6675/0.3425）band loss 增量 +0.0030/+0.0025——正但小，纳入冠军配置。TGV 对照第 7 次逐位复现。与 TGV 差距缩至 **0.031**（今日开盘 0.053）。
2. **batch-VRAM 探测**（owner ~20GB 指引）：batch 6/8/12 → 14.6/19.0/26.3 GB，22.8/23.6/25.9 samples/s，无 OOM。选 **batch 8**（正中 20GB 档）。
3. **冠军对决已开**（tmux 0:champion，串行过夜 ~9.5h）：冠军臂 solver_v20_champion_50k（η0.09+band，batch 8，50k）先跑，锚臂 solver_v20_anchor_50k（η0.5 旧默认，batch 8，50k）随后——成对重锚满足单变量纪律（batch 效应两臂共担）。
4. **终局判据（预注册）**：冠军 vs 锚 vs TGV（0.702/0.356/23.03，偏移校正 cross-FRC 管线）。冠军 ≥TGV@30µm 或（cutoff ≤23.03 且 @30µm ≥0.69）→ 项目主目标"神经追平/超越经典"首次达成；否则如实记录残差与收敛性，方向转向下一批候选（更深 η×unroll 交互、E2 IRLS 鲁棒 DC、D per-pixel LN）。

**训练结果**(2026-07-07 深夜回填;产物 `remote_inbox/20260710_champion/`):
- **冠军臂 solver_v20_champion_50k = 0.6470@30µm / 0.3308@24µm / cutoff 25.45µm —— 预注册判据未达成**(需 ≥0.702@30µm 或 cutoff≤23.03 且 ≥0.69)。TGV 对照第 8 次逐位复现(0.7017/0.3558/23.03);偏移探针残余 0.080/0.083 HR px(centered 正常);管线有效,数字可信。
- **关键负结果:50k+batch8 较 20k+batch4 组合臂(0.6705)倒退 −0.023**(超出 ±0.01 噪声带);synth 却升至 33.82(组合臂 20k 为 32.79)——**synth↑/real↓ 的 ACL-032 反相关签名再现:在当前合成分布上训得越久,真实域越差**,直接支持"合成-真实域差"假设。神经最佳纪录仍为 20k 组合臂 0.6705(gap 0.031)。
- 锚臂(η0.5,batch8,50k)按 owner 指令在启动后即掐除(方向已转,方法学配菜让路)——batch8 与 50k 两变量因此未隔离,倒退归因(batch-LR 未重标定 / 低 η 长训下先验漂移 / 交互)不再单独开臂回答,并入 Stage 2b 框架。
- **裁决与转向(owner 2026-07-07 夜决策)**:η 校准线到此收官;单数据集调参终止。下一步 = **Stage 2b 合成基准套件**:TGV/MAP-TV 首次在 v6 合成集上做对照、按难度分层(噪声×帧数 N24/48/96×对比度,单列 real-like 组)、指标为带限 FRC-vs-GT(25-40µm)+band RMSE,判决 H1(域差→修数据/v7 池)vs H2(架构上限→prox 架构轮,合成台开发、真实 248 帧只当期末考);附带 η 跨域稳健性表(合成域算子精确,若 η*_synth 明显更高 = 低 η 的真实增益本质是补偿算子误差)。Stage 2b 优先级高于 σ 校准。
- §2.5 多分半材料已备一半:drizzle 分半参考 seeds 42/123/456 + odd_even 在 `output/stage2p5_multisplit/`(rc=0);TGV/神经新分半未渲染。判据未达标,多分半核验降级为非阻塞事项。
- 基础设施(同夜):v5 池 + 357 个已结案中间 checkpoint 删除,WSL 内释放 267G;checkpoint 策略今后 `--save-every 10000` 且必须显式 `--real-eval-every 1000 --synth-eval-every 1000`(评测节奏默认耦合 save_every,`solver_train.py:223`);WSL vhdx(542G,C 盘)由 owner 以 diskpart compact 原地压缩(拒绝迁往 SATA D 盘,保 I/O)。

**涉及文件**: 无代码变更（实验条目）。

---


### [ACL-052] 2026-07-07（日间，无人值守）— 方向 A（DC 权重校准 / eta 扫描）出利好：FRC@30µm 随 eta 单调改善，eta=0.125 cutoff 首次追平 TGV 的 23.03µm；向下细扫进行中

**问题诊断 / 实验记录**（owner 指令：ABCD 挨个试，A 利好则顺 A 深挖；产物 `remote_inbox/20260709_stage0k/eta_sweep/`）:
1. **A 的设计修正**：原拟"DC sum→mean + eta 可学"；核查发现 learnable eta 已被 ACL-026 证伪（优化器绕过 DC 步），且固定 M=16 下 mean 归一 ≡ eta 常数重标定 → **A = 冻结 eta 的取值扫描**（历史值 0.5 从未校准过），零新代码。
2. **粗扫（eta 0.125/0.25/1.0/2.0，各 20k、与 v14 严格同参同 seed）**：
   - synth PSNR：33.09 / **33.56** / 26.26 / 21.87（基线 eta0.5=31.26）——低 eta 大幅利好，高 eta 坍塌；synth 最优在 0.25。
   - **真实 cross-FRC@30µm 单调**：0.663（0.125）> 0.658（0.25）> 0.649（基线）> 0.629（1.0）> 0.609（2.0）；@24µm 前三者 0.343/0.345/0.335 微升。
   - **cutoff：eta0.125 = 23.03µm——与 TGV 持平（v6 池臂首次）**；0.25/基线 25.45，1.0→25.75，2.0→26.28。
   - TGV 对照行第 5 次逐位复现；八半幅 centered 残余 0.026–0.087 HR px。
3. **判读**：历史 eta=0.5 系统性偏大（DC 压重、prox 被挤压）；FRC 趋势在扫描下边界（0.125）未拐头 → 真最优在更低处；@30µm 0.663 距预注册 0.67 线差 0.007，但 cutoff 追平 TGV 属质变信号，按 owner"A 利好则跑 A"指令**判 A 利好、深挖**。
4. **风险预注册**：eta 过低 → DC 太弱 → 先验主导/幻觉风险。防线=判据本身（偏移校正 cross-FRC vs drizzle 对幻觉免疫，self-FRC 才会被骗）+ OOB/artifact 监控；预期 FRC 峰非单调，细扫用于定位拐点。

**修改内容**: 无代码变更（A 纯超参）。细扫三臂已开（eta 0.0625/0.09/0.1875，串行 ~3h，tmux 0:etafine，输出 solver_v17_eta{0p0625,0p09,0p1875}_20k）。B（band-gated loss）已实现就绪（commit 8c956f5，90+11 测试，Fourier 径向软掩膜、默认关闭字节级等价）；细扫收敛后排"最优 eta × B"组合臂。C（E1@50k）/D（per-pixel LN）在 A 线索穷尽前暂缓。

**预期效果**: 细扫定位 eta* 后，冠军配置 = eta* (+ band loss 若组合臂有效)，按 owner 的 ~20GB 显存指引重锚 batch 后训 50k 冠军臂，目标 TGV 0.702@30µm / 23.03µm。

**训练结果**: 2026-07-07 傍晚回填（细扫完成，产物 `remote_inbox/20260709_stage0k/eta_fine/`）
- 细扫三臂 @30µm：η0.0625=0.6648（cutoff 23.77）、**η0.09=0.6675（八点曲线峰）**、η0.1875=0.6454；synth 32.74/32.93/33.39。TGV 对照第 6 次逐位复现。
- **八点 η 曲线（@30µm）**：0.665 / **0.667** / 0.663 / 0.645 / 0.658 / 0.649 / 0.629 / 0.609（η=0.0625→2.0）——峰值平台 η≈0.0625–0.125，η*=0.09。仅此一个超参把与 TGV 的差距从 0.053 缩到 **0.035**（~35%）。0.1875/0.25 处的非单调为 ±0.01 噪声级。
- 低 η 幻觉防线未触发：OOB 正常、偏移校正后 sign changes 全 0、cross-FRC（幻觉免疫）为判据。
- **组合臂已开**：solver_v19_etaB_20k（η*=0.09 × band loss gate25_40 w0.5，v14 同参，~1h）。若 ≥0.68@30µm 或 cutoff <23.03 → 进冠军流程（~20GB batch 重锚、基线+冠军成对 50k 对决 TGV 0.702/23.03）。

**涉及文件**: 无（实验记录条目）。

---


### [ACL-051] 2026-07-07（凌晨，主智能体夜间自主执行）— Stage 0k 消融三箭齐平：便宜路线（迭代深度/证据预算）无肉；m32 变长 burst collate bug 修复；E0 推理期扩帧为负；裁决 = Stage 2a E1 逐帧融合立项开工

**问题诊断 / 实验记录**（owner 睡前授权整夜自主；所有远端任务 tmux+log，产物 `remote_inbox/20260709_stage0k/`）:
1. **unroll4 臂**（`solver_v15_unroll4_20k`，K=2→4，其余与 v14/C 同参同 seed）：synth PSNR 31.8（+0.55dB vs C）**但真实 cross-FRC 平**——0.660@30µm / 0.335@24µm / cutoff 25.75µm（v14 基线 0.649/0.335/25.45）。迭代深度不是瓶颈；synth 增益再次与真实域脱钩（ACL-032 模式重现）。
2. **m32 臂**（`solver_v15_m32_20k`，训练期 DC 帧 16→32）：首启崩溃——`stack expects each tensor to be equal size, [26,…] vs [32,…]`。根因：v3+ 池每景帧数随机（最小 24），`m = min(m_frames, n)` 使 M 随景变化破坏 collate 常量假设（m=16 时从未触发）。**修复 commit 72ed441**：`_select_m_indices` 不足时带放回补齐（重复帧只重加权 DC 证据），+回归测试，ep07 18 passed。重训后：0.636@30µm / 0.345@24µm / 25.45µm——**平**；且 synth PSNR 掉到 26.47（vs C 31.26）。**待查发现**：帧数翻倍 + 冻结 eta=0.5 疑似 DC 梯度随 M 缩放（sum vs mean 未归一），已交 E1 实现者核查现状（只报告不改动，checkpoint 兼容性优先）。
3. **E0 零训练消融**（v14_50k checkpoint 推理期 m 16→64）：**负**——0.616@30µm / 0.269@24µm（基线 0.649/0.335）。网络按 m=16 的 DC 统计收敛，推理期改证据预算破坏平衡；其 cutoff=229µm 系低频毛刺误触发 first-crossing（曲线非单调），点值可信、cutoff 弃用。
4. **对照行核验**：三次独立跑的 tgv_x_drz 均复现 0.702/0.356/23.03，判据管线稳定。centered 出口残余偏移 0.06–0.11 HR px（符合 ~0.05 LR px 预期）。

**裁决**（预注册规则：任一臂 @30µm ≥0.67 才算有肉）:
- unroll4 +0.011、m32 −0.013、E0 −0.033 → **便宜路线证伪**。差距（TGV 0.702 vs ~0.65 @30µm、23.03 vs 25.45µm cutoff）不是训练成熟度（0j）、不是迭代预算、不是证据帧数——指向**输入端聚合抹平子像素相位**（Stage 2a 设计稿的核心论断）。
- **Stage 2a E1（逐帧融合）立项**：后台 Fable 按 `research_log/stage2a_perframe_fusion_design_draft.md` 实现中（fusion.py + 接线 + 5 测试，默认关闭=字节级旧行为），完成即在 5090 开 `solver_v16_e1fusion_20k`（v6 池、20k、seed 42）。

**预期效果**: E1 若在校正 cross-FRC 上突破 0.67@30µm（+0.02 判据），逐帧融合方向确认，续接 E2（IRLS 鲁棒 DC）/E3（band-gated loss）；若平，按设计稿风险清单逐项排查（容量→per-pixel LN 候选 E1'）。

**训练结果**: 2026-07-07 07:5x 回填（E1 训练+后处理完成，产物 `remote_inbox/20260709_stage0k/e1/`，CSV 已核）
- **E1（v16_e1fusion_20k，perframe 融合 E=16/+48ch/chunk8）**：synth PSNR 31.26（与基线持平，融合无害）；**真实 cross-FRC 0.641@30µm / 0.332@24µm / cutoff 25.45µm——平**（v14 0.649/0.335/25.45），低于 0.67 及格线。TGV 对照行第 4 次逐位复现（0.702/0.356/23.03）。
- **夜间总记分板**（@30µm）：unroll4 0.660 / E1 0.641 / m32 0.636 / E0-推理 0.616，全部落在 v14±0.015 的高原内；TGV 0.702 的差距对"深度、证据帧数、输入端逐帧融合"三个正交方向全部免疫。
- **DC=sum 确认**（`forward_torch.py:715`，E1 实现时只读核查）：DC 对 M 帧求和不归一 → m32 的 synth 坍塌（31.26→26.47）系 DC 梯度随 M 翻倍所致；这也意味着 **DC 权重（eta 冻结 0.5 × 任意的 sum 尺度）从未被原则性设定**——列为下一批候选实验的头号议题。
- **E1 裁决**：逐帧融合（此实现、20k 步）不解决差距。按预注册规则不自动开新训练；晨间决策候选（按性价比排序）：(a) E3 band-gated loss（设计稿已备、从未试过、与 25–40µm 实测带对齐）；(b) E1@50k 成熟度补测（融合臂参数更多，20k 收敛假设未必成立）；(c) DC 归一化 + eta 解冻的原则性重设（动训练约定，需评估 checkpoint 兼容面）；(d) E1' per-pixel LN。

**涉及文件**: `algos/ep07_unet_sr/src/unet_sr/dataset.py`（72ed441）、`algos/ep07_unet_sr/tests/test_dataset.py`；E1 实现文件清单见后续回填。

---


### [ACL-050] 2026-07-06 — 对比层永久修正（real_eval 中心网格选项）+ 文档纠偏（速览块/旧结论软化）+ inbox 传输硬规则 + Stage 0j：v6+E3 主线 50k 成熟度臂

**问题诊断**:
1. v4 裁决后（ACL-049）有三笔欠账：偏移校正只存在于事后探针，渲染源头没有中心网格出口；repo 文档里大量已推翻的结论（神经破坏带内信息、34µm 频带、split-half FRC 主判据）会误导未来的智能体；远端把 84 个 inbox 产物 commit 进了 git（8e4467c，违反 e532574 惯例）。
2. v4 暴露的最便宜杠杆：V11（50k, v5 池）cutoff 23.03µm 优于 C（20k, v6 池）的 25.45µm——**v6+E3 主线臂没训到成熟度**，在判架构/容量之前必须先补齐这 30k 步。owner 确认：v6 池 + E3 主线（operator DR 全默认关）就是当前最新的数据集与架构；V11 只作历史基准，不重训。
3. DR 关案后 Stage 1a 的"D 臂"分支正式废弃；本轮起训练臂命名 v14。

**修改内容**:
1. **real_eval 中心网格**：新增 `to_center_grid(image, scale)`（Fourier 平移 −(scale−1)/2 px/轴）与 `infer_solver_from_burst_full_halo(..., output_grid="native"|"centered")`。默认 `native` 保历史 in-training eval_real 可比性；**一切与经典方法的对比一律用 `centered`**（残余 ~0.05px 内容相关偏移需精确时仍用 probe 逐半校正）。测试 `test_center_grid.py` 2 条，ep07 19 passed。
2. **文档纠偏**：changelog 顶部新增"✅ 当前有效结论速览"块（8 条当前真值，旧条目冲突以速览为准）；ACL-046/047/048 头部加 ⚠️ 已推翻标注；proposal/roadmap 状态块补充 v4 翻案与偏移校正要求。目的：未来智能体第一眼看到正确结论。
3. **inbox 硬规则**：`git rm -r --cached remote_inbox/`（84 文件，ce29f74，磁盘保留）；AGENTS.md 新增"remote_inbox 传输规则"硬规则段（只走 rsync/scp/SSH 管道增量，严禁 git，违反即回滚）。
4. **Stage 0j 任务包** `handoff/stage0j_remote_tasks.md`：Task 1=v6+E3 主线 50k 训练（`solver_v14_v6_nodr_50k`，与 C 完全同参仅步数 20k→50k、同 seed——前 20k 顺带复现 C 作 sanity）；Task 2=step 20k/50k 两个 checkpoint 用 `output_grid=centered` 渲染分半 → 残余偏移探针 → 排行榜 v5 对 TGV/MAP-TV/drizzle。

**预期效果**:
- v5 排行榜回答："成熟度补齐后 v6+E3 距 TGV（0.702@30µm / 23.03µm）还差多少"。追平/超过 → 路径明确（更长训练/loss band 微调）；仍差且 50k 曲线已平 → 容量/先验问题坐实，才立项 Stage 2 架构（逐帧融合）。
- 20k checkpoint 的 eval_synth 应≈C 的 31.26 PSNR（同 seed 同参），偏差大=复现性问题优先排查。

**训练结果**: 2026-07-06 回填（Stage 0j，产物 `remote_inbox/20260708_stage0j/`，已本地核验 CSV；⚠️ 远端 REPORT.md 的散文数字第三次出错——把 `__xb_ya` 单方向曲线的 @30µm 值当 @24µm 报，以下为 mean 曲线核验值）
- **复现性 PASS**：v14_20k×drz = 0.641@30µm / 0.343@24µm / cutoff 25.45µm，与 C（0.649/0.342/25.45）吻合。
- **成熟度不是差距来源**：50k vs 20k 几乎零提升（0.649 vs 0.641 @30µm，cutoff 同 25.45µm）——v6+E3 主线在 20k 已收敛。V11(v5 池) cutoff 23.03 的优势是池差异故事，另案。
- **主裁决**：v14_50k 0.649@30µm / 25.45µm，仍落后 TGV（0.702 / 23.03µm）约 0.05@30µm 与 2.4µm cutoff；v14×TGV 一致性 0.871/0.849（两者重建内容高度相同，TGV 对 drizzle 锚多抽出一点真实逐半信息）。
- **centered 出口生效**：残余偏移 ~0.09-0.10 HR px（≈0.05 LR px 内容残差，符合预期）。
- **结论**：ACL-050 决策树走"仍差且曲线已平"分支 → Stage 2 架构工作正式立项；先跑两个廉价消融（unroll_steps 2→4、solver_m_frames 16→32，各 ~1h@20k）区分"优化/证据预算不足"与"架构上限"。

**涉及文件**: `algos/ep07_unet_sr/{src/unet_sr/real_eval.py,tests/test_center_grid.py}`、`AGENTS.md`、`research_log/{algorithm_changelog,solver_v2_redesign_proposal,network_upgrade_roadmap}.md`、`handoff/stage0j_remote_tasks.md`

---

### [ACL-049] 2026-07-05 — Stage 0h 根因裁决：神经输出网格 +0.5 HR px 角点约定（有文档记载）导致所有神经×经典对比系统性低估；"神经臂带内净破坏"基本翻案；决策=对比层配准修正，不改训练约定

**问题诊断 / 0h 已核验结果**（产物 `remote_inbox/20260706_stage0h/`，报告质量高、数字均标出处，已本地复核）:
1. **Task 1 推翻"只有 V11 偏移"的预期**：同半幅偏移探针测得 V11 0.693 / **C 0.621 / D 0.628 HR px**（vs drizzle；TGV/MAP-TV 锚 0.024/0.034≈0）。ACL-048 对 C/D 的"带内内容被先验替换"定罪基于未校正曲线，**撤销重审**。
2. **精修对齐重渲染后偏移不消失**（V11 0.711 / C 0.691 / D 0.711，均约 (+0.58,+0.40)）→ 不是对齐 CSV 的问题，是渲染/网格约定。**根因已在 repo 代码定位**：`algos/ep07_unet_sr/src/unet_sr/forward_torch.py:19-21` 文档明载 forward 采样 `scale*(i+d)+{0..scale-1}`、"constant +0.5 HR-px block-center offset (self-check T1) is implicit"——solver 的 DC 把解锚在角点网格（LR 中心 ↔ 2(i+d)+0.5），而 drizzle/SAA/TGV/MAP-TV 全部用 2(i+d) 中心网格。**训练内部自洽（池渲染与 DC 同约定），错在对比层从未对账**。实测 ≈(+0.5,+0.5)+warmstart/prox 小残差。V8 时代以来所有神经×经典对比（含 0f/0g cross-FRC 排行榜、历史"神经比经典软/糊"的视觉印象一部分）都带此配准误差。
3. **校正偏移后神经臂同半幅几乎追平经典**：v11/C/D vs drizzle @30µm 0.670/0.670/0.680（TGV 锚 0.745），v11 vs TGV @30µm=0.860、@24µm=0.808 → **"神经臂带内净破坏"基本翻案**，0f/0g 神经 cross 数字全部系统性低估，排行榜 v3 的神经×经典行作废待 v4。
4. **仍成立**：C vs D 零差（cutoff 均 26.67µm、@30µm 均 0.207）——DR@0.1px null 为三重仪器一致结论，正式关案；经典新基线 TGV×drz 23.03µm / MAP-TV×drz 24.62µm（精修对齐下 TGV self cutoff 达 20.0µm=孔径零点区，不作声明）；精修对齐增益不受影响。
5. 远端执行质量：命令/出处/exit code 齐全，欠账 iter2 JSON 已补交，中途 4 核经典重建太慢主动中止换 16 线程并留了废弃 log——纪律良好。

**修改内容**:
1. `probe_pair_offset.py` 新增 `--save-corrected-dir`：每对的偏移校正后 b 数组存为 `<name>_corrected.npy`，供下游配准公平比较。烟测：校正后残余偏移 0.003/0.006px。
2. **决策：不改训练约定**（角点约定贯穿池渲染+DC+全部 checkpoint，改=全部作废）；修正放在对比层（校正数组进排行榜）。后续可选小改：real_eval 保存对比用输出时统一重采样到中心网格（待 v4 结果后与 owner 定）。
3. Stage 0i 任务包 `handoff/stage0i_remote_tasks.md`：Task 1=对神经三方法 a/b 两半分别测偏移+存校正版（a/b 偏移应一致，验证渲染常数假设）；Task 2=排行榜 v4（校正后的神经 cross + 同半幅行，经典行同跑对照）。

**预期效果 / 决策树（v4 = 方向裁决）**:
- 校正后神经 cross 追平/接近经典（TGV×drz 23.03µm 档）→ 神经臂未坏，主战场="精修对齐+正确配准下打赢 TGV"，重训开绿灯（v6 池、精修 shift 已默认、DR 关或按 ~0.05px）。
- 仍显著低于经典 → 架构层讨论（逐帧融合/先验强度）才立项。
- 排行榜纪律不变：24–30µm 段 + 1/7 cutoff，20µm 孔径零点不采信。

**训练结果**: 2026-07-05 回填（Stage 0i 排行榜 v4，产物 `remote_inbox/20260707_stage0i/`，已本地复核 CSV）
- **偏移常数假设成立（带小残差）**：a/b 两半偏移 v11 (0.711/0.658)、C (0.691/0.650)、D (0.711/0.643)，同方法 a/b 差 0.042–0.075px（约 (+0.55,+0.39) 常数 + ~0.05px 内容相关残差）；v11_a 校正后复测残余 0.008px。v4 用逐半校正，干净。
- **v4 最终裁决（cross vs drizzle，24–30µm 可信段）**：v11 0.644/0.343/cutoff 23.03µm；C 0.649/0.342/25.45µm；D 0.655/0.335/25.45µm；TGV 0.702/0.356/23.03µm；MAP-TV 0.690/0.296/24.62µm。v11×tgv 旁证 0.851/0.787。**配准伪影曾吃掉神经臂 @30µm 约 0.44–0.49 个 FRC 点**（v3→v4 delta）。
- **结论**：(1) "神经臂带内净破坏"正式翻案——校正后神经与经典同档，V11 cutoff 与 TGV 打平；(2) 但神经并未打赢经典（@30µm 落后 0.05–0.06），"cleanly beat classical in-band"的原始目标现在第一次有了可信记分板且差距很小；(3) C(20k, v6) cutoff 25.45 vs V11(50k, v5) 23.03——训练时长/成熟度是显著变量，v6 臂尚未训到位；(4) C vs D 校正后仍零差（@30µm 差 0.007），DR@0.1px 关案维持。

---

### [ACL-048] 2026-07-05 — Stage 0g 判定：精修 shift = 真信息增益，权威频带 34.07→25.45µm；精修对齐升级为 repo 默认资产；V11 网格偏移嫌疑（C/D 不适用）；新增偏移探针 + 文档纠偏 + Stage 0h

> ⚠️ 本条对 C/D 的"带内内容被先验替换定罪成立"已被 ACL-049 推翻：C/D 同样带 ~0.65px 配准偏移，校正后神经与经典同档。以顶部速览块为准。

**问题诊断 / 0g 已核验结果**（产物 `remote_inbox/20260705_stage0g/`，数字均已对 CSV/JSON 复核）:
1. **Task 1 shift 反馈回路三信号全正**：(a) iter2 收敛——残余 delta 0.012/0.071px（iter1 0.288/0.447），每轴均值≈0，improvement 缩至 6.9%（剩余为 σ/形状项）；(b) drizzle 分半 cutoff：phase_stratified 29.67→26.28µm、odd_even 26.9→20.6µm；(c) **EP15 M2 权威频带 34.07→25.45µm（±0.73，3 seeds）**，带内负对照压死（24µm：main 0.197 vs shuffle 0.020 vs bicubic 0.002），且 **aperture dip 首次 visible（margin 0.205）**——FRC 现在能看到 20µm 探测器孔径零点的物理特征，独立佐证对齐质量。**判定：精修 shift 为真信息增益；真实可恢复带 ~25.5–40µm 周期。** 0e 的 34µm 作废（那是"坏对齐下的频带"）。
2. **Task 2 同半幅对照——判决拆开**（本地对曲线做了 45–21µm 符号翻转计数，远端 REPORT 只给了"定罪缓议"的整体结论）：tgv×drz 0 次翻转（锚，@30µm 0.590）；**c_nodr×drz 仅 2 次、平滑衰减但低（@30µm 0.185 < drizzle 跨半自洽 0.377）→ 对 C/D 的"带内内容被先验替换"定罪成立**；v11×drz 14 次、v11×tgv 41 次高幅振荡（0.53→0.22→0.43）→ **V11 输出网格有刚性亚像素偏移嫌疑，对 V11 定罪缓议**。
3. 重要缓刑理由：0f/0g 的神经臂分半重建 DC 喂的仍是旧对齐（0.29px 误差在毒害 DC）——公平重审需用精修对齐重渲染（Stage 0h Task 2/3）。
4. 上轮 inbox 欠账：stage0g_iter2 的 summary json 未随包传回（数字自 REPORT 转录），0h 补交。

**修改内容**:
1. **精修对齐升级为 repo 默认资产**：`configs/alignment/stage0f_refined_alignment.csv` 入库（由本地 build_refined_alignment 从 t1a 精修表构建，与远端产物同源同参数）；`configs/alignment/paths.json` schema 0.2——`contour_alignment_results_csv` 指向精修 CSV，旧 CSV 保留为 `contour_alignment_results_csv_pre_stage0f`，附 provenance。已验证 loader 默认路径取回的 248 帧 dx/dy 与精修值逐帧一致。**影响面**：所有 `load_alignment_shifts`/`default_contour_alignment_csv` 消费者（0a、FRC、M2、SAA、tcforge real-like constellation）默认吃精修 shift。
2. **偏移探针** `algos/ep15_info_limit/scripts/probe_pair_offset.py`：加窗相位相关（两遍迭代 + 抛物线亚像素）估计全局偏移 → Fourier 反移 → 前后 FRC/cutoff/符号翻转对比。合成验证：注入 (+0.7,−0.7)px 恢复 (+0.710,−0.678)，无偏移对照 |0.018|px。
3. **文档纠偏（防误导清理）**：删除已完成任务包 handoff/stage1a、stage0f、stage0g（git 历史保留）；`solver_v2_redesign_proposal.md` 与 `network_upgrade_roadmap.md` 顶部加状态更新块（split-half FRC 判据废止→cross-FRC、DR 降级、shift 主线、25.45µm 权威频带、σ=0.2257 不可信、0a 早期数字污染警示）。
4. Stage 0h 任务包 `handoff/stage0h_remote_tasks.md`：Task 1=偏移探针裁决 V11；Task 2=精修对齐重渲染 V11/C/D 分半（GPU 推理）；Task 3=全精修口径 cross-FRC 排行榜 v3（经典臂也重建）。

**预期效果 / 决策树**:
- 0h 回答三个问题：(a) 精修对齐下神经臂 cross-FRC 是否回升（回升=旧 shift 毒害 DC 为主因，重训方向=精修 shift 喂 DC；不回升=prox 先验替换实锤，Stage 2 改架构方向）；(b) C vs D 是否仍平；(c) 经典臂新基线。
- **重训数据集决策（owner 问）**：合成池 shift 本就是精确真值，精修对齐不影响池——**继续用 v6**（C 臂已是 v6 基线，单变量可比）。DR 若再开，量级按实测残余 ~0.05–0.07px，不是 0.1px。v7 仅当后续决定把训练频谱对准实测 25–40µm 带时才立项。
- dc_resid 解冻评估推迟到 0h 后（σ 仍未标定，但精修对齐下的 DC 残差首次具备相对比较意义）。

**训练结果**: 待 Stage 0h 远端结果回填。

**涉及文件**: `configs/alignment/{paths.json,stage0f_refined_alignment.csv}`、`algos/ep15_info_limit/scripts/probe_pair_offset.py`、`research_log/{solver_v2_redesign_proposal,network_upgrade_roadmap}.md`、`handoff/{stage0h_remote_tasks.md,-stage1a,-stage0f,-stage0g}`

---

### [ACL-047] 2026-07-04 — Stage 0f 结果核验：逐帧 shift 误差 ~0.29px 实锤为头号瓶颈；神经臂带内净破坏（cross-FRC）；权威频带 ≈34µm；新增 shift 反馈工具 + Stage 0g 任务

> ⚠️ 本条两个结论已被后续推翻："神经臂带内净破坏"系 +0.5 HR px 配准伪影（ACL-049 v4 翻案）；"权威频带 34µm"是坏对齐下的测量（ACL-048 修正为 25.45µm）。shift 误差 ~0.29px 的诊断仍然成立。以顶部速览块为准。

**问题诊断 / 0f 已核验结果**（产物 `remote_inbox/20260704_stage0f/`，远端 REPORT 有两处错误见文末勘误）:
1. **Task 1（0a centered 重跑 ×2）**：约定修复在真实数据生效——系统偏置从 −0.24px 归零（axis means +0.010/+0.007）。诚实改善 12.62% CI[11.77,13.49]（full）/ 12.44% CI[11.61,13.31]（split_half），比带 bug 的 20.07% 挤掉约 7.5 个点。**关键分布事实**（本地复核 t1a/t1b CSV）：精修量是均值为零、模长集中 0.25–0.40px 的环状分布（248 帧仅 17 帧 <0.05px；mean |Δ|=0.288，p95=0.447，max=0.566）；仅 42/248 落在 0.25 格点（非插值伪影）；full 与 split_half 两个独立 x̂ 逐帧几乎一致 → **真实逐帧 shift 误差 ~0.29px（约 0.2px/轴）**。对照 0b（0.2px→−3.5dB），不修 shift 则 2× 去混叠预算基本不存在——**shift 校正取代 DR 成为头号杠杆**。σ̂ 两个变体都卡 0.05 下边缘：SAA x̂ 自带模糊，本仪器测不了 σ（预期内）。
2. **Task 2（cross-FRC）修正读数**（可信段 24–30µm；20µm 恰在探测器孔径零点、数值不可用）：@30µm tgv×maptv 0.70 > tgv×drz 0.558 > drizzle self 0.377 ≈ maptv×drz 0.374 ≫ **V11 0.108 / C 0.098 / D 0.118**；@24µm 同序。神经臂在带内携带的真实可复现信息**低于 drizzle 本身**，self-FRC 0.99+ 全是共享先验复现。**C vs D 统计上平 → DR 0.1px null 在可信仪器下确认**。存疑点：神经×经典的 cross 可能被网格约定偏移衰减（0a 刚出过同类 bug）——0g Task 2 同半幅对照裁决。
3. **Task 3（0d 分布）**：仅 extent 探针可靠分离（nrmse 2.86×、p95 2.36×）；flat_roi/beading 分离边距仅 1.06–1.15× 且坏例 n=1（checkpoint 加载失败回退预渲染数组，出处降级已记）；seam 探针好坏不可分（坏例 0.857 反而低于好例 0.86–0.87）。阈值决定权在 owner；建议方向：extent 收紧为唯一硬门，flat/beading 趋势跟踪，seam 废弃或重设计。
4. **Task 0e（EP15 M1+M2 @20µm）**：M2 first-crossing 1/7 cutoff **34.07µm**（3 seeds std 1.61µm）；20µm 处 main FRC 0.897 但负对照同高（shuffle 0.867/drift 0.887）→ 高频 re-rise 是伪影。M1 相位覆盖 stage lattice 25/25 但 detector bin 仅 11/25。**权威可恢复频带 ≈34µm**，与 drizzle 分半 28–30µm、cross cutoff 26–30µm 三角互证：真实增益带在 ~30–40µm 周期，2× 的"20µm"是重建网格不是信息声明（与 AGENTS.md 契约一致）。远端顺手修的 M2 aperture-zero 10→20µm（commit 0fc51c8，越权 push 但 diff 已审、正确）。
5. **REPORT.md 勘误**：(a) Task 2 全部"FRC@20µm"数字实为 CSV `frc_at_30um` 列、"@16µm"实为 24µm 列——排行榜必须按 24–30µm 读，本条 §2 为正确读数；(b) Task 3"建议阈值可直接替换硬编码"过头，仅 extent 成立。

**修改内容**:
1. 新增 `psf_calibration/refined_alignment.py` + 薄壳 `scripts/build_refined_alignment.py`：把 stage0a 逐帧精修 shift 合并回 contour 对齐 CSV schema（`refined_align_dx_px/dy_px`），下游全部 `--alignment-csv` 消费者（0a/run_real_split_frc_v2/run_m2_frc/SAA）即插即用。守卫：initial 必须与 base 对齐逐帧一致（防止喂错源精修）、248 行契约、重复 file 拒绝。已用真实 t1a 数据端到端验证（248 行，applied delta mean 0.2878/p95 0.4472，loader 往返 dx/dy 精确一致）。
2. 测试 `test_build_refined_alignment_merges_and_guards`（合并正确性 + 错源拒绝），ep09 11 passed。
3. Stage 0g 任务包 `handoff/stage0g_remote_tasks.md`：Task 1=shift 反馈回路（build → 0a iter2 收敛检查 → drizzle 分半 FRC → M2 重跑，全 CPU）；Task 2=同半幅跨方法对照（裁决"神经带内破坏 vs 网格约定偏移"）。

**预期效果**:
- 若 drizzle cutoff 从 29.7µm 下移、M2 cutoff 从 34µm 下移且负对照差距保持，精修 shift 升级为新对齐资产，所有方法输入受益；0a iter2 的残余 delta 给出收敛性与剩余误差。
- 自拟合风险纪律：精修 shift 对共享 SAA x̂ 拟合，下游 FRC 增益必须对照 M2 负对照（shuffle/drift）解读，不得单独作为证据。
- 训练臂继续冻结；若 0g 成立，下一个训练动作是把精修 shift 喂进 DC / 提前 proposal 2c 的 test-time shift refinement，而不是调 DR。

**训练结果**: 2026-07-03 回填（Stage 0g remote inbox 本地复核，产物 `remote_inbox/20260705_stage0g/`）
- **Task 1 / shift 反馈回路**: 精修对齐资产生成 248 行，applied delta `mean=0.2878px`, `p95=0.4472px`, axis means `(+0.0103,+0.0069)`。Iter2 残余显著收敛：train delta norm `mean=0.0123px`, `p95=0.0707px`; val `mean=0.00924px`, `p95=0.0500px`; 新对齐 baseline 下 val band-MSE improvement `6.897%`，bootstrap 95% CI `[6.590%, 7.179%]`。这说明一轮精修后没有继续追逐同量级 shift，剩余项更像模型/σ/score floor。
- **Task 1 / FRC 证据**: drizzle split-half 1/7 cutoff 下移：phase_stratified `29.67µm -> 26.276µm`，odd_even `26.9µm -> 20.598µm`。M2 权威重跑 cutoff `34.07µm -> 25.455µm`，3 seeds std `0.730µm`。负对照审计：24µm band table `main=0.1975 > drift=0.1852 > shuffle=0.0200`，20µm 处 `main=-0.285` 而 shift-shuffle `0.917`，仍标记为 aperture/伪影区；在 24–40µm sampled rings 中 shift-shuffle 全部低于 main，drift control 大体低于 main，但有 3/149 个近 24.15–24.35µm 的小幅反超（最差 margin `-0.0104`）。因此结论应是**精修 shift 具备升级为新对齐资产的候选资格**，不是"彻底排除伪影"；升级前应保留负对照 caveat。
- **Task 2 / 同半幅跨方法对照**: 经典锚点正常：`tgv_a_vs_drz_a` @30/@24µm = `0.590/0.197`。神经臂同半幅仍低：`v11_a_vs_drz_a` = `0.113/0.097`，`c_nodr_a_vs_drz_a` = `0.185/0.103`，`d_dr01_a_vs_drz_a` = `0.199/0.103`。但 `v11_a_vs_drz_a` 与 `c_nodr_a_vs_drz_a` 在 36–20µm 曲线中呈强非单调规律振荡（本地复核约 110 次一阶差分变号），不像单纯信息单调丢失。**结论：神经臂带内破坏定罪缓议，网格约定/配准偏移嫌疑需要单独定位。**
- **归档/流程复核**: remote inbox 中的 summary/curve CSV 与本地 `output/` 对应文件逐字节一致；`configs/` 与 tracked code 无 diff。流程问题：远端 REPORT 日期与本机日期不一致，长任务使用 `setsid nohup` 而非 AGENTS.md 要求的 Tmux 0 窗口；remote inbox 未包含 1.2 全量 summary/log 源产物副本。后续任务包需修正为 Tmux 运行并归档 stdout log、summary JSON 和关键输入 CSV。

**涉及文件**: `algos/ep09_psf_calibration/{src/psf_calibration/refined_alignment.py,scripts/build_refined_alignment.py,tests/test_psf_calibration.py}`、`handoff/stage0g_remote_tasks.md`

---

### [ACL-046] 2026-07-03 — Stage 1a 判定=仪表失效（inconclusive）；0a 半像素约定 bug 实锤；新增 Stage 0f 仪表修复（0a centered forward + split-source x̂、cross-method FRC、0d 分布对比工具）

> ⚠️ 本条 §4 对神经方法 split-half FRC 的解读方向正确（self-FRC 无效），但由此展开的"神经臂破坏带内信息"叙事已被 ACL-049 推翻（配准伪影）。以顶部速览块为准。

**问题诊断**（对 20260703 remote run 的本地复核，产物见 `remote_inbox/20260703_stage1a/`）:
1. **0a 的"真实算子误差 ≥0.4px"结论撤回——大头是 0a 打分器自己的半像素约定 bug**。Task A（radius 0.8）248 帧精修分布：mean Δ=(dx −0.237, dy −0.242) px、每轴 sd≈0.22、0 帧碰 0.8 边界。−0.24≈−0.25 LR px=0.5 HR px：`stage0a_mvp.forward_block_average_shifted` 采样 `scale*(i+shift)+{0..scale-1}`（block 中心 +0.5 HR px），而打分对象 x̂ 来自 `saa.reconstruct_saa`，其 scatter 在 `scale*(i+shift)`（无 +0.5）。两套独立实现差恰好半个 HR 像素，精修一直在补这个常数。此前所有"饱和"现象（MVP p95=0.05√2、full-grid p95=0.4√2 顶角）全部由此解释。**ep07 solver 本身不受影响**（forward_roundtrip_selfcheck 认证过其 adjoint 内部一致）。
2. **20.07% [18.16, 22.06] 的改善被该 bug 污染**：sweep 分解显示 σ=0.5 只做 shift 精修已得 +14.0%，换 σ=0.1 只再加 ~6%（且 σ 部分本就受 x̂ 同源退化偏低影响）。"模型误差天花板可修"的真实幅度待修约定后重测；剩余每轴 0.22px 散布还双峰聚在 0/−0.5 格点，疑似插值 landscape 伪影，真实逐帧对齐误差可能显著小于 0.2px。
3. **0d 回归套件无判别力**（Task B）：好例 V11 与 PromptA 在 flat-ROI/seam/beading 与已知坏例几乎同分布（seam autocorr 0.86 vs 0.87），仅 extent 探针双双 pass。判据 #2 当前不可用。
4. **split-half FRC 对神经方法结构性失效**（Task E）：经典方法物理表现正常（TGV cutoff 20.0µm、MAP-TV 21.2µm、drizzle 29.67µm、FRC@20µm 全为负），V11/C/D 却 FRC@20µm=0.93–0.9998 永不截止——确定性网络把同一先验高频复现进两个独立半幅，**FRC 奖励可复现的幻觉**；且 D(0.9975)>C(0.9310) 更可能表示 DR 臂更先验主导，方向可能反向。判据 #1 对神经候选失效。
5. **C vs D 复判**：synth −0.033dB、real 基本平——在 0.1px 档 DR 无害无益；但由于 1、4 两条，正确结论是 **inconclusive（没有可用仪表判定），不是"DR 无效"**。DR 的正确量级也依赖修好 0a 后的真实误差测量。

**修改内容**（全部本地已测，Stage 0f 仪表修复）:
1. **0a centered forward**：`forward_block_average_shifted` 新增 `block_convention`，默认 `centered`（block 采样点 −(scale−1)/2，使 LR 像素中心落在 `scale*(i+shift)`，与 SAA scatter 网格一致）；`legacy_corner` 保留旧行为仅供对照。贯穿 `_score_shift/_score_one_frame/score_candidate/run_stage0a_mvp`，CLI `--forward-convention`，summary 记录 `scoring.forward_convention`。
2. **0a split-source x̂**：`run_stage0a_mvp(xhat_source="split_half")` / CLI `--xhat-source split_half`：偶/奇 parity 各建一张 SAA，每帧只对**对侧**半幅的 x̂ 打分，消除"帧参与自身 x̂"的同源退化（σ̂ 偏低的主因之一）。`score_candidate` 改为 `hr_images + frame_xhat_idx` 路由。
3. **cross-method FRC**：`run_real_split_frc_v2.py` 新增 `--cross-pair name:x_a:x_b:y_a:y_b[:scale[:note]]`，对称化计算 FRC(X_a,Y_b) 与 FRC(X_b,Y_a) 及均值曲线/cutoff（cryo-EM map-vs-model 思路：共享先验幻觉在跨方法半幅间不相关，只有真实带内信息相关；仍属相对排名，两方法共享对齐）。另加 `--methods none`（跳过加载真数据，纯 artifact/cross 评估）。合成烟测：自带幻觉的"神经"self-FRC@20µm=0.999 不截止，cross-FRC 立即塌回真实信号带（@20µm≈−0.10，cutoff 与诚实方法一致）——机制符合设计。
4. **0d 分布对比工具**：新增 `algos/ep07_unet_sr/scripts/compare_regression_distributions.py`，从多个 suite JSON 抽 7 个探针标量，输出好/坏分布、当前阈值命中、可分性与建议阈值（geomean）。**只报数据，阈值仍由 owner 决定**。
5. 测试：ep09 新增 3 条——ramp 上 centered 与 SAA 网格严格一致且 legacy 恰差 +0.5 HR px；SAA 回环回归（centered 最优精修≈0，legacy 出 −0.25 对角偏置，直接编码本 bug）；split-source 路由正确性（对侧路由 band_mse≈0、错配路由显著非 0）。ep09 10 passed。

**预期效果 / 下一步（Stage 0f 远端任务，见 `handoff/stage0f_remote_tasks.md`）**:
- 0a 用 centered + full 与 centered + split_half 重跑：读修正后的真实改善幅度、per-frame 散布（决定 DR 正确量级与 test-time shift refinement 优先级）、split-source 下 σ 是否离开网格下边缘。
- cross-FRC 排行榜：V11/C/D 各自对 drizzle 半幅交叉，得到神经候选第一个可信的真实域相对排名；C vs D 复判凭它。
- 0d：远端补 V8/K4 坏例 JSON 后跑 compare 工具，分布表交 owner 定阈值。
- dc_resid 继续冻结；在 0f 仪表可用前不再增开训练臂。

**训练结果**: 待 Stage 0f 远端结果回填。

**涉及文件**: `algos/ep09_psf_calibration/{src/psf_calibration/stage0a_mvp.py,scripts/run_stage0a_mvp.py,tests/test_psf_calibration.py}`、`algos/ep15_info_limit/scripts/run_real_split_frc_v2.py`、`algos/ep07_unet_sr/scripts/compare_regression_distributions.py`、`handoff/stage0f_remote_tasks.md`

---

### [ACL-045] 2026-07-02 — Stage 1a operator DR 实现 + 0a bootstrap CI + TCForge 默认 pitch 修正

**问题诊断**:
- roadmap 风险 #1（shift 误差）的缓解措施从未实现：训练里渲染 y_i 的 shift 与喂给 A_i 的 shift 逐字节相同（`dataset.py _add_burst_to_sample` "EXACT shifts"），solver 从没见过"算子不准"。Stage 0 已实测真实算子误差显著（见 ACL-044：0a 精修在 ±0.4 px 盒仍边界饱和；0b 表明 0.1 px 误差吃掉 1.4 dB oracle）。
- 0a 的 `val_band_mse_improvement_pct` 无置信区间（`utils.py bootstrap_curve_minima` 现成未用），小样本下无法断言显著性。
- `tcforge/geometry.py DEFAULT_PIXEL_SIZE_UM=10.0` 是旧 2× 标尺误读残留；正式 pool 均显式传 20.0 不受影响，但裸调用会拿到错误物理标尺。

**修改内容**:
1. **Stage 1a operator DR**（proposal §1"求解器中间步"）：`dataset.py` 新增 `dc_shift_jitter_std_px` / `dc_psf_sigma_jitter_frac` / `dc_psf_angle_jitter_deg`（默认全 0=旧行为）。`_add_burst_to_sample` 里只扰动喂给 DC 的 shift（每帧 N(0,σ)）与 PSF（双轴同因子 ×U(1±frac)、角度 N(0,deg)）；**burst 像素保持真值渲染**——受控的 render-vs-DC 失配。同一 (seed,index,epoch) 下确定性（resume 安全）。
2. `config.py`：对应 `solver_dc_*` 字段 + CLI `--solver-dc-shift-jitter-std-px` 等三旗标 + 校验；`solver_train.py` 只给**训练** dataset 透传（synth_eval loader 不扰动，DR/no-DR 臂指标可比），banner 打印 `operator_DR=...`。
3. **0a bootstrap CI**：`stage0a_mvp.py` 新增 `_bootstrap_val_improvement_ci`（val 帧成对重采样 2000 次，95% CI + 显著性判定；best 在 train 上选出，val CI 不受选择污染），写入 `dc_resid_floor_probe.val_band_mse_improvement_bootstrap`，runner 打印。
4. **TCForge**：`DEFAULT_PIXEL_SIZE_UM` 10.0 → 20.0（LR pitch 语义，HR pitch=÷scale；含注释）。tcforge 测试全部显式传 pitch，不受影响。
5. 测试：`test_dataset.py::test_operator_dr_jitters_dc_params_but_not_burst`（burst 不变/参数确变/各向异性比保持/确定性）、`test_psf_calibration.py::test_stage0a_bootstrap_ci_detects_real_and_null_improvement`（真改善显著、纯噪声不显著）。本地 ep07 39 passed、ep09 7 passed、tcforge 12 passed。

**预期效果**:
- 1a DR 臂据此起跑：`--solver-dc-shift-jitter-std-px 0.1`（0b 定标），PSF jitter 首轮不开（单变量纪律）。
- 风险：DR 过强会教 prox 忽略多帧证据（V11 式保守化）——0.2 px 档已被 0b 证明破坏性大，不作起点。

**训练结果**: 2026-07-03 回填（Stage 1a remote run audit）
- **Task A / 0a 去饱和重跑**: `output/ep09_psf_calibration/stage0a_fullgrid_r08`。248 clean 帧、198/50 train/val、36 candidates × 2 splits。best=`sigma_x_lr_px=0.1, sigma_y_lr_px=0.1, angle_deg=0.0`；val band-MSE improvement `20.074%`，bootstrap 95% CI `[18.164%, 22.063%]`，显著。解释仍限于 floor probe，不作为物理 PSF σ̂。
- **Task C / v6 no-DR control**: `algos/ep07_unet_sr/outputs/solver_v13_v6_nodr_ctrl`，`solver_step_020000.pt` 与 `solver_final.pt` 均为 `step=20000`，`solver_dc_shift_jitter_std_px=0.0`。TensorBoard final: synth `psnr=31.2635`, `region_rmse=0.1718`, `boundary_f1=0.8245`, `out_of_band_ratio=0.04293`; real `artifact_score=0.4253`, `out_of_band_ratio=0.001691`, `dc_resid_band=1.2409`, `dc_resid_full=1.5268`。
- **Task D / v6 DR 0.1**: `algos/ep07_unet_sr/outputs/solver_v13_v6_dr01`，`solver_step_020000.pt` 与 `solver_final.pt` 均为 `step=20000`，`solver_dc_shift_jitter_std_px=0.1`。TensorBoard final: synth `psnr=31.2306`, `region_rmse=0.1767`, `boundary_f1=0.8246`, `out_of_band_ratio=0.04191`; real `artifact_score=0.4195`, `out_of_band_ratio=0.001701`, `dc_resid_band=1.2406`, `dc_resid_full=1.5281`。
- **Task E / split-half FRC leaderboard**: `output/stage0c_frc_leaderboard`，single phase-stratified `seed=42` split，6 methods all `success`。FRC @20µm / cutoff: drizzle `-0.429 / 29.67µm`, MAP-TV `-0.620 / 21.23µm`, TGV `-0.403 / 20.0µm`, V11 `0.9998 / no cutoff`, C no-DR `0.9311 / no cutoff`, D DR 0.1 `0.9975 / no cutoff`。Neural methods 的 A/B 半幅高度相关，不采信为真实 20µm 高频增益。
- **结论**: DR 0.1 相对 v6 no-DR final synth PSNR 仅 `-0.033 dB`，artifact score final 点小幅下降 `0.4253 -> 0.4195`，但 real FRC 与 reconstruction 差异未显示可信正收益；Stage 1a 不应判为成功，只能记为“训练完成、DR 未带来可采信收益”。原远端 `REPORT.md` 的 Task C 数字错误，已在 `remote_inbox/20260703_stage1a/REPORT.md` 校正。

**涉及文件**: `algos/ep07_unet_sr/src/unet_sr/{dataset,config,solver_train}.py`、`algos/ep07_unet_sr/tests/test_dataset.py`、`algos/ep09_psf_calibration/{src/psf_calibration/stage0a_mvp.py,scripts/run_stage0a_mvp.py,tests/test_psf_calibration.py}`、`tcforge/src/tcforge/geometry.py`

---

### [ACL-044] 2026-07-02 — Stage 0 gate review：已验证运行数字落盘 + 判据决策（进入 1a）

**问题诊断**: Stage 0 四个 scaffold（41b4732/fa45253/d0566bc/12fa269）只提交了代码，运行数字散落在 5090 的 `output/`，changelog 无记录（违反"截断必 log"）。本条把 owner review + 当晚补跑的全部实测数字落盘。

**已验证数字**（全部跑在 5090，20µm pitch 契约，248 真实帧）:
1. **0a MVP**（`output/ep09_psf_calibration/stage0a_mvp/`）：实际参数远小于默认——train 8 / val 4 帧、σ 网格 {0.2,0.5}、精修半径 0.05。val band-MSE 改善 **6.39%**（4 帧，无 CI，不可单独引用）。关键副产品：shift 精修 p95=0.0707=0.05√2 **顶到边界盒角**。
2. **0a full-grid**（`stage0a_fullgrid/`，198 train/50 val 全帧、σ 0.15-0.5、半径 0.4 步长 0.1、81 offsets、2738s）：val band-MSE 改善 **19.34%**（0.01231→0.00993，RMS 0.1110→0.0996）。best=σ0.15 iso——**又在网格边缘**；shift 精修 mean 0.35-0.40 px、p95=0.5657=0.4√2 **再次边界饱和** → 真实算子误差（shift+x̂ 系统差合并）≥0.4 px 量级，远超 2× 去混叠的 0.25 px 要求。**警告**：x̂ 是同源 SAA（含一次 PSF），σ 估计天然偏低，0.15 不作物理 σ̂ 认定；本数字只作"模型误差可修"的 floor probe 证据。
3. **0b 全量**（`output/ep15_info_limit/stage0b_info_budget2/`，288 runs，6 scene×3 error-seed）：248 帧档 alias-oracle PSNR delta vs 零误差 = **+0.27 dB @0.05px（噪声级）/ −1.39 dB @0.1px / −3.54 dB @0.2px**。16 帧档全正 delta=conditioning 伪影，弃用。
4. **0c 多 seed**（`output/stage0c_real_split_frc_v2_multiseed/`，5 seeds×both split）：drizzle 1/7 cutoff = **27.1–29.7 µm（mean≈28.2，±1.3µm）**，odd_even 26.9 一致 → 指标可复现。aligned_mean 全 seed"到 20µm 不过阈"且 FRC@20µm≈0.9995 非单调回升 = **插值核相关伪影，aligned_mean cutoff 一律不采信**。已知未缓解偏差：两半共享全帧拟合的 contour 对齐、splat/zoom 核相关、FPN 共模 → cutoff 绝对值偏乐观，**只作方法间相对比较**。
5. **0d**（`output/ep07_solver_regression_suite/local_baseline_report.json`，已知坏例 V8/K4 数组）：4/4 probe FAIL——坏例判别力成立。seam autocorr 全候选 0.74-0.86（阈 0.35），疑对共享结构过敏，**好例（V11/E2）校准完成前不作硬门**。

**判据决策**:
- Gate：**条件通过，进入 1a**。DR σ_shift 起点 **0.1 px**（0b 定标；0.05 无压力、0.2 破坏性大）。
- 主判据换轨确认：0c 以 **drizzle 锚点（≈28.2µm）为参照的相对 FRC** + 0d 套件（好例校准后）+ synth 不塌；below-20µm 只作 audit。
- **池变更纪律**：合成池已从 v3→v5_sharp→v6_cpu（真实 248 帧未变）。1a 在 v6 上训练，但历史 V11 checkpoint 早于 v6 → **必须先在 v6 上重训 no-DR 控制臂**，否则池+DR 两变量混杂。
- dc_resid 继续冻结为非判据，直到 0a 非退化标定（交替更新 x̂ 或联合优化）给出可信 σ̂。

**涉及文件**: 无代码变更（记录条目）。执行任务包：`handoff/stage1a_remote_tasks.md`。

---

### [ACL-043] 2026-07-01 — 输入侧多帧增强：on-the-fly phase-bin drizzle（可调 bins，免重生成数据）

**问题诊断**:
- 用户核心诉求:把多帧真正用到输入侧(当前 no_drizzle 只喂 aligned_mean 一张聚合图,hybrid 也只有固定 4-bin drizzle)。
- 离线信息预算(`outputs/ep07_solver_diag/info_budget2.py`)证明多帧亚像素信号真实(~+1~+1.5 dB @2×,微扫描采样均匀),更细的相位分桶能保留更多去混叠结构。但原 hybrid 路径从磁盘读**固定 4-bin** 预计算 drizzle,想加 bins 就得重生成整个 pool。

**修改内容**:
1. `dataset.py`: 新增 `phasebin_ontf` / `phase_bin_channels`。开启时在 `_load_cached` 里用保存的 `lr_burst`+`shifts` 现场 `phase_bin_drizzle(..., n_bins=phase_bin_channels)`,替代读磁盘 4-bin。**复用已测的 tcforge drizzle,不重生成数据。**
2. `config.py`: 新增 `solver_phasebin_ontf`(默认 False)与 CLI `--solver-phasebin-ontf` / `--phase-bin-channels`;校验:ontf 需 hybrid(禁与 --solver-no-drizzle 合用)、`phase_bin_channels` 必须是完全平方数(4/9/16,phase_bin_drizzle 要求),非 4 必须配 ontf。
3. `synth_eval.py` / `solver_train.py`: 训练与合成 GT 评测两路 dataset 都透传 ontf+bins,保证 cond 通道一致;`real_eval.py` 本就按 `phase_bin_channels` 现场算,天然对齐。
4. 测试: `tests/test_config.py` 解析 + 非平方数/与 no_drizzle 互斥的拒绝回归。

**预期效果**:
- 让 prox 看到更细相位分桶的多帧亚像素结构(4→9→16 bins),验证"输入侧多帧"是否兑现 +dB。**与 ACL-042(D-E,给容量抽信号)组合最有意义**——先有容量,再喂更丰富的多帧。
- 风险:bins 越多 cond 通道越多(in_ch=5+bins)、算力/显存略升;on-the-fly drizzle 有 CPU 成本(每 scene 一次,LRU 缓存摊薄)。**先 2k smoke 确认不崩再长训。** 默认 off,不影响既有实验。

**推荐参数**:
```bash
# 输入侧多帧(9-bin)+ D-E 高频残差,K2/noGN/noSE/p384,40k
--input-mode hybrid_drizzle2x --scale 2 --unroll-steps 2 \
--solver-phasebin-ontf --phase-bin-channels 9 \
--solver-prox-highpass-residual --solver-prox-highpass-sigma-hr 5 \
--synth-eval-holdout 500 --synth-eval-every 2500 --total-steps 40000
```

**训练结果**: 2026-07-02 回填
- 输出目录:
  - E4 9-bin: `algos/ep07_unet_sr/outputs/solver_v14_de_pb9`
  - E5 16-bin: `algos/ep07_unet_sr/outputs/solver_v14_de_pb16`
- 关键指标:
  - E4 40k: synth PSNR 32.47, region RMSE 0.1208, boundary F1 0.864, synth OOB 0.0103, real artifact 0.436, real OOB 0.0018.
  - E5 5k: synth PSNR 31.997, region RMSE 0.1342, boundary F1 0.858, synth OOB 0.0127, real artifact 0.398, real OOB 0.0013;因 5k 早期指标无改善且显存/时间成本更高而中止。
- 视觉效果: E4 的 real proxy 更干净,但合成 GT fidelity 低于 D-E/no-drizzle winner(E2:PSNR 32.53/RMSE 0.1191/F1 0.864);E5 早期未显示追赶趋势。
- 结论: on-the-fly richer phase-bin input 没有在 D-E 基础上兑现额外可用锐度。继续加 phase bins 不是主线。

**涉及文件**:
- `algos/ep07_unet_sr/src/unet_sr/dataset.py`, `config.py`, `synth_eval.py`, `solver_train.py`
- `algos/ep07_unet_sr/tests/test_config.py`

---

### [ACL-042] 2026-07-01 — D-E: solver prox 高频残差(low-freq anchor)，在不恢复 SE/GN 的前提下追回锐度

**问题诊断**:
- ACL-041 的 E3 主线(noSE+noGN)成功压住 extent 漂移(网格/絮状),但代价是 prox 变保守、偏糊:synth PSNR 从 V9(SE+GN) 的 ~37.5 掉到 V11 的 ~35,视觉更糊,后期训练也追不回。
- 离线信息预算模拟(`outputs/ep07_solver_diag/info_budget2.py`,用真实常数 scale 2 / PSF σ≈0.5 / 噪声 0.0724°C)确认:多帧亚像素信号真实存在(约 +1~+1.5 dB,微扫描采样均匀),它**已经在 phase-bin drizzle 条件通道里**——Prompt A(重加 drizzle)之所以 null,不是输入缺多帧,而是**保守的 noGN/noSE prox 抽不出被噪声埋着的去混叠高频**。瓶颈是 prox 容量,不是输入。
- 直接恢复 GN/SE 会把 extent 伪影带回来(EP07 诊断 §3:SE 全局池化 + GroupNorm 全局归一化是 extent 耦合路径)。因此需要"在结构上只给高频容量、不给低/中频自由度"的改法。

**修改内容**:
1. `unroll.py`: `UnrolledSolver` 新增 `prox_highpass_residual`(bool)与 `prox_highpass_sigma_hr`(HR px)。开启时 `_prox` 把学习到的修正做高通:`delta <- delta - gaussian_blur_2d(delta, sigma_hr)`,再 `x <- x + delta`。→ prox 只能注入高频细节;低/中频(extent 网格 glow / 背景抬升 / 絮状所在的频段)钉死到 warm-start + DC,structurally 无法被 prox 移动。
2. `config.py`: 新增 `solver_prox_highpass_residual`(默认 False)与 `solver_prox_highpass_sigma_hr`(默认 5.0),CLI `--solver-prox-highpass-residual` / `--solver-prox-highpass-sigma-hr`,并写入 checkpoint config。
3. `solver_train.py` / `scripts/render_checkpoint_evolution_drop.py`: 透传新字段(渲染侧从 checkpoint config 读取,默认 False 向后兼容)。
4. 测试: `tests/test_config.py` 解析回归;`tests/test_model_losses.py` 单元测试证明常数(纯低频)修正被高通抹掉、plain 残差原样相加。

**预期效果**:
- 在保持 E3 extent 稳定性的前提下追回锐度:prox 有容量把 drizzle/phase-bin 里的去混叠高频从噪声里抽出来,而不能再制造低/中频方框/絮状/背景抬升。
- 兑现离线预算里的 ~+1~+1.5 dB(2× 的物理天花板;不解锁 4×,噪声封顶)。
- 风险:sigma 太大→细节被当低频锚掉(欠锐);太小→仍可能残留中频漂移。sigma 需 smoke 扫(建议 4/5/6/8 HR px)。默认 off,不影响既有实验。

**推荐参数**:
```bash
# 在 E3 主线(noSE/noGN/K2/p384/full_halo96)上开高频残差,评测用合成 GT 全帧(不要用被物理地板钉死的 dc_resid 当判据)
--input-mode hybrid_drizzle2x --scale 2 --unroll-steps 2 \
--solver-prox-highpass-residual --solver-prox-highpass-sigma-hr 5 \
--synth-eval-holdout 500 --synth-eval-every 2500
# 建议 sigma smoke: 4 / 5 / 6 / 8
```

**训练结果**: 2026-07-02 回填
- 输出目录:
  - E1 sigma=5: `algos/ep07_unet_sr/outputs/solver_v13_de_s5`
  - E2 sigma=4: `algos/ep07_unet_sr/outputs/solver_v13_de_s4`
  - E3 sigma=8: `algos/ep07_unet_sr/outputs/solver_v13_de_s8`
  - E6 width control: `algos/ep07_unet_sr/outputs/solver_v15_de_wide`
- 关键指标:
  - V11 reference: synth PSNR 35.17, region RMSE 0.0859, boundary F1 0.858, real artifact 0.457.
  - E1 40k: PSNR 32.51, RMSE 0.1191, F1 0.863, real artifact 0.461.
  - E2 40k: PSNR 32.53, RMSE 0.1191, F1 0.864, real artifact 0.460.
  - E3 40k: PSNR 32.50, RMSE 0.1189, F1 0.863, real artifact 0.458.
  - E6 40k (`base_channels=96`, `batch_size=8`): PSNR 32.58, RMSE 0.1176, F1 0.863, real artifact 0.465.
- 视觉效果: D-E variants stayed broadly clean but did not become materially sharper than the V11 noSE/noGN baseline; E6 gave a tiny PSNR/RMSE bump at the cost of slightly worse real artifact.
- 结论: D-E 高频残差没有追回 V9/V11 合成 fidelity。推荐若保留 D-E,使用 E2 `sigma_hr=4` 作为当前最平衡配置;但 ACL-042 的主要结论是负面的:简单 high-pass residual 与加宽 prox 都不是当前主瓶颈,下一步应转向采集/噪声/alignment 或更大的低频锚定建模改动。

**涉及文件**:
- `algos/ep07_unet_sr/src/unet_sr/unroll.py`
- `algos/ep07_unet_sr/src/unet_sr/config.py`
- `algos/ep07_unet_sr/src/unet_sr/solver_train.py`
- `algos/ep07_unet_sr/scripts/render_checkpoint_evolution_drop.py`
- `algos/ep07_unet_sr/tests/test_config.py`, `algos/ep07_unet_sr/tests/test_model_losses.py`
- 依据: `outputs/ep07_solver_diag/info_budget2.py`(离线信息预算), `research_log/literature/2026_info_budget_and_why_phone_4x.md`

---

### [ACL-041] 2026-06-30 — 将 solver 主线默认切到 noSE + noGN + full_halo96 real-eval

**问题诊断**:
- E3 smoke 对比显示 `noSE+noGN` 的 tiled 与 `full_halo96` 真实推理几乎一致,而 `noSE+GN` 在 halo 下 artifact/out-of-band 更高,符合 GroupNorm/SE 范围分布漂移诊断。
- 既有 `SE+GN` 结构在合成指标上收敛更快,但真实图上更容易出现 halo 絮状、细线膨胀与推理范围敏感性;当前主目标是工业真实热图干净稳定,不是单纯 synthetic PSNR。
- 因此 50k 长训不应再靠手动追加 E3 flag,而应把 E3 组合提升为 solver 主线默认,减少命令遗漏风险。

**修改内容**:
1. `config.py` 默认值调整:
   - `solver_prox_use_se=False`
   - `solver_prox_norm="none"`
   - `real_eval_solver_mode="full_halo"`
   - `real_eval_solver_halo_hr=96`
2. `config.py` 新增显式恢复旧结构的 CLI:
   - `--solver-prox-use-se`
   - `--solver-prox-norm group`
   - `--real-eval-solver-mode tiled --real-eval-solver-halo-hr 0`
3. `tests/test_config.py` 增加两项回归:
   - 新 solver 主线默认确认为 `noSE/noGN/full_halo96`;
   - 旧 `SE+GN/tiled` 可通过 CLI 显式恢复。

**预期效果**:
- 50k solver 长训默认使用范围更稳定的 prox,降低 halo/tiled 分布差导致的絮状与边界位移风险。
- TensorBoard real-eval 默认直接显示 `full_halo96`,避免 tiled patch 边界方框成为视觉误导。
- 风险:去掉 GN/SE 后 synthetic PSNR 可能更低、收敛更慢;如果 10k-15k 真实图明显欠锐,优先考虑加宽 prox 或调 loss,不要直接恢复 GN/SE。

**推荐参数**:
```bash
# 新默认已经包含 noSE + noGN + full_halo96;主跑无需再显式写这几个 flag。
--input-mode hybrid_drizzle2x --scale 2 --unroll-steps 2 \
--solver-no-drizzle --patch-size-hr 384 \
--real-eval-solver-mode full_halo --real-eval-solver-halo-hr 96
```

**训练结果**: _(2026-06-30 中途诊断回填;训练仍在继续到 `--total-steps 60000`)_
- 输出目录: `outputs/solver_v11_k2_p384_nogn_halo96_50k`。实际运行命令使用 `--synth-eval-holdout 500`, `--num-workers 12`, `--total-steps 60000`;截至本次诊断已有 checkpoint 到 `solver_step_040000.pt`,训练 scalar 到 42.5k。
- 当前指标:
  - synth @42.5k: PSNR `35.17`, region RMSE `0.0859`, boundary F1 `0.858`, out-of-band `0.01044`。
  - real @40k (`full_halo96`): artifact `0.4566`, out-of-band `0.00205`, DC band residual `1.215`, DC full residual `1.509`。
  - real artifact 最低出现在 10k 左右 (`0.420`),之后缓慢升高到 40k;real DC residual 基本死平在 `1.21` 左右。
- 与关键对照:
  - 旧 `solver_v9_k2_p384_nodrizzle` (`SE+GN`, p384) @16k synth PSNR `37.52`, region RMSE `0.0757`,但 @15k real artifact `0.4806`, real OOB `0.00241`。
  - `v10_v5_sharp` plain UNet @50k synth PSNR `38.81`, visual 更锐,但 real artifact `0.4634`, real OOB `0.00219`。
  - `v10_v4_acl027` smooth UNet @40k real artifact `0.2597`, real OOB `1.56e-5`,但 synth PSNR/视觉锐度明显低。
- 视觉效果: 用户目视确认 v11 相比旧 solver/V10 sharp/纯 UNet 更糊,后期锐度提升有限。当前数字支持该判断:20k 后 synth PSNR 只从 `34.73` 增至 `35.17`,region RMSE 只从 `0.0906` 改至约 `0.085`,real artifact 反而升高。
- 结论: `noSE+noGN+full_halo96` 成功降低了 halo/tiled 范围漂移风险,但牺牲了拟合/锐化能力;它是"干净但偏保守"的解,不是继续训练即可追上旧 `SE+GN` 或 V10 sharp 的解。后续若要追回锐度,优先考虑在不恢复 GN/SE 的前提下加 prox 容量、改高频残差/低频锚定结构或重新平衡 loss;不建议简单加长训练到 60k 作为主要改进手段。
- Prompt A 后续 A/B _(2026-07-01,各 5k smoke,固定 seed=42)_:
  - A 基线 `outputs/solver_v12_promptA_A_nodrizzle_5k`:保持 E3 noSE/noGN/p384/full_halo96,`--solver-no-drizzle`。5k 指标:PSNR `34.0004`,region RMSE `0.103536`,boundary F1 `0.861061`,synth OOB `0.009335`,real artifact `0.414338`,real OOB `0.001536`,DC band `1.21752`,DC full `1.50482`。
  - B hybrid `outputs/solver_v12_promptA_B_hybrid_aligned_5k`:去掉 `--solver-no-drizzle`,使用 9ch hybrid cond(5 fused + 4 phase-bin drizzle),并设置 `--solver-warmstart aligned_mean` 去 ACL-032 波纹。5k 指标:PSNR `33.8948`,region RMSE `0.102609`,boundary F1 `0.861721`,synth OOB `0.010675`,real artifact `0.400323`,real OOB `0.001317`,DC band `1.21787`,DC full `1.50479`。
  - 判定:B 没有满足"同时提升 synth PSNR/RMSE/F1、real artifact/OOB 不升、DC band residual 下降"的胜出条件。B 的 real artifact/OOB 略好,region RMSE/F1 略好,但 PSNR 略低、synth OOB 更高,且关键 `real dc_resid_band` 未下降(反而 `1.21752 -> 1.21787`)。结论:给 prox 喂回 phase-bin drizzle 通道没有让 solver 更有效利用真实多帧观测;不把 hybrid 9ch 设为新主线。

**涉及文件**:
- `algos/ep07_unet_sr/src/unet_sr/config.py`
- `algos/ep07_unet_sr/tests/test_config.py`

---

### [ACL-040] 2026-06-30 — E3 solver prox 架构消融:可关闭 SE 与 GroupNorm 以测试 extent-invariance

**问题诊断**:
- K4 shared prox 已确认会递归放大 patch 边界响应;K2+p384 2k smoke 虽降低方格,但 15k 后真实评估 `artifact_score` 单调恶化,出现边界位移与白色絮状/细线膨胀。
- full-halo 能消除显式方框,但会引入更强絮状物和线宽膨胀;结合远场/extent 诊断,最可疑耦合路径是 prox 内部依赖整幅特征统计的模块:SE 的 global average pooling 与 GroupNorm 的 per-sample/channel 归一化。
- 因此下一步不再继续调 patch size 或 halo,而是做 E3:让 solver prox 结构本身对推理范围/上下文大小更不敏感。

**修改内容**:
1. `model.py`: `ThermalSRUNet`/`ConvBlock` 新增 `use_se` 与 `norm` 参数;默认仍为历史 `use_se=True,norm="group"`。`norm="none"` 时用 `Identity` 替代 GroupNorm,`use_se=False` 时用 `Identity` 替代 SEBlock。
2. `unroll.py`: `UnrolledSolver` 新增 `prox_use_se` 与 `prox_norm`,只作用于 solver 的 prox UNet;普通 UNet 训练默认不变。
3. `config.py` / `solver_train.py`:新增 CLI `--solver-prox-no-se` 与 `--solver-prox-norm {group,none}`,并写入 checkpoint config。
4. `scripts/render_checkpoint_evolution_drop.py`:读取 checkpoint config 中的 E3 prox 架构字段,保证 E3 checkpoint 后续可正常渲染。
5. `tests/test_config.py` / `tests/test_model_losses.py`:补 CLI 解析与无 GroupNorm/无 SE forward 回归测试。

**预期效果**:
- No-SE 版本用于验证 SE global pooling 是否是 full-halo 絮状和细线膨胀的主要来源;风险较低,因为只去掉通道注意力。
- No-SE + No-GN 版本用于验证 GroupNorm 尺度分布漂移假设;若有效,应表现为 tiled/full-halo 输出更接近、背景絮状下降、边界位移减少。
- 风险:去掉归一化后训练稳定性可能下降,需要先 2k smoke 再决定是否长训;旧 checkpoint 不能加载到显式 E3 结构,但默认结构仍完全兼容旧实验。

**推荐参数**:
```bash
# 低风险 E3a:只关 SE
--solver-prox-no-se

# 强消融 E3b:关 SE + 去 GroupNorm
--solver-prox-no-se --solver-prox-norm none
```

建议仍保持 K2/no-drizzle/p384 做 2k smoke 对照,不要直接开 50k 长训;若 E3b 不稳定,优先回退到 E3a 或降低 `boundary_boost` 后再测。

**训练结果**: _(待 2k/15k smoke 回填)_
- 输出目录: 待回填。
- 视觉效果: 重点观察真实 TensorBoard 中背景絮状、细线膨胀、边界位移、tiled vs halo 差异。
- 结论: 待回填。

**涉及文件**:
- `algos/ep07_unet_sr/src/unet_sr/model.py`
- `algos/ep07_unet_sr/src/unet_sr/unroll.py`
- `algos/ep07_unet_sr/src/unet_sr/config.py`
- `algos/ep07_unet_sr/src/unet_sr/solver_train.py`
- `algos/ep07_unet_sr/scripts/render_checkpoint_evolution_drop.py`
- `algos/ep07_unet_sr/tests/test_config.py`
- `algos/ep07_unet_sr/tests/test_model_losses.py`

---

### [ACL-039] 2026-06-30 — 固定 real-eval Matplotlib 为 Agg 后端,避免 Tk 线程析构中断训练

**问题诊断**:
- solver 训练在中途偶发 `Tcl_AsyncDelete: async handler deleted by the wrong thread` 后进程退出。该错误来自 Tcl/Tk GUI 后端的线程/析构问题,不是 CUDA OOM 的典型报错。
- 当前 `ep07_unet_sr` 环境默认 Matplotlib backend 为 `tkagg`;`real_eval.py` 在 checkpoint real-eval 保存温度 PNG 时局部 import `matplotlib.pyplot`,会创建 Tk 相关对象。对象析构可延后发生,所以崩溃步数不一定正好落在 `save_every` 节点。

**修改内容**:
1. `real_eval.py` 在模块初始化时强制 `matplotlib.use("Agg", force=True)`,让训练期间保存 PNG/TensorBoard 图都走非交互后端。
2. 不改变 solver forward、loss、optimizer、scheduler、halo 推理数学或 TensorBoard scalar/image tag。

**预期效果**:
- 消除由 Tk GUI 后端造成的随机训练中断。
- 不影响训练精度;仅影响 Matplotlib PNG 渲染后端。
- 风险:若有人在训练进程内期待弹出交互式 Matplotlib 窗口,该路径会变为只保存图。训练脚本本来不应弹窗,因此风险可接受。

**推荐参数**:
```bash
# 无需新增 CLI 参数;代码固定使用 Agg 后端。
```

**训练结果**: _(待下一次长训观察)_
- 输出目录: 待回填。
- 视觉效果: 不应改变模型输出;若 halo real-eval 仍出现白色絮状背景,应按 solver/数据/色标问题继续诊断,不归因于 Matplotlib 后端。
- 结论: 待长训验证不再出现 `Tcl_AsyncDelete`。

**涉及文件**: `algos/ep07_unet_sr/src/unet_sr/real_eval.py`

---

### [ACL-038] 2026-06-30 — solver real-eval 支持 full-frame outer halo,避免 prox 边界方框进入可视 FOV

**问题诊断**:
- no-drizzle solver 的真实图中仍可见规则方框。进一步 TensorBoard 分解显示: `x0/aligned_mean` 无方框,`prox1` 立即产生强方框,`dc1` 只能削弱;`prox2` 再次产生,`dc2` 再次削弱。
- 同一可视 flat ROI 上,保持输出区域不变但扩大求解上下文: `halo0` 有方框,`halo64` 明显减轻但目视仍不够干净,`halo96` 可压住可视方框,`halo128` 相比 `halo96` 无明显收益。说明方框主要是 learned prox 的 patch-local 边界响应,不是 alignment、drizzle 或最终 stitch window。
- full-frame outer halo 诊断显示 K2 checkpoint 在 RTX 5090 上可跑 `halo_hr=64/96/128`,峰值显存约 11.6-16.0GB 量级,作为 real-eval 路径可行。

**修改内容**:
1. `real_eval.py` 新增 `_solver_conditioning_from_burst()` 复用 solver 条件构建逻辑,避免 tiled/full-halo 两条路径重复。
2. `real_eval.py` 新增 `infer_solver_from_burst_full_halo()`:
   - 在 LR burst 上做 outer reflect halo;
   - halo 后构建 no-drizzle/hybrid 条件通道;
   - 对 enlarged field 做一次 full-frame solver;
   - 最后 crop 回原始 detector FOV。
3. `RealEvalConfig` / `TrainingConfig` / CLI 新增:
   - `--real-eval-solver-mode {tiled,full_halo}`
   - `--real-eval-solver-halo-hr N`
4. `solver_train.py` 和 `train.py` 将新增 real-eval 配置传入 `RealEvalConfig`。默认仍为 `tiled`,保持老实验口径;显式开启 `full_halo` 后才改变 solver real eval。
5. `tests/test_real_eval.py` 添加 full-halo crop-back contract 测试;`tests/test_config.py` 添加 CLI 解析/校验测试。

**预期效果**:
- 训练期间 TensorBoard `eval_real/*` 可直接展示 full-frame outer-halo solver 输出,避免 tiled eval 在真实图中引入 prox 边界方框。
- 不改变训练 loss 或 checkpoint 权重;这是 eval/inference 路径修复。若视觉确认有效,后续再考虑把 full-halo 作为默认 solver real-eval 口径。
- 风险: full-frame solver eval 比 tiled eval 单次显存峰值更高;当前 K2 checkpoint 通过,但 K4/更大模型仍需实测。`--real-eval-solver-halo-hr` 必须能被 `--scale` 整除。

**推荐参数**:
```bash
--real-eval-solver-mode full_halo --real-eval-solver-halo-hr 96
```

完整对标命令见本条训练结果/结论中的建议;训练本身仍建议从头跑 no-drizzle,不从旧中断 checkpoint resume。

**训练结果**: _(待 50K 主线训练后回填)_
- 输出目录: 推荐 `outputs/solver_v8_nodrizzle_fullhalo_eval`
- 视觉效果: 已在 `outputs/solver_v7_k2_nodrizzle_flat005_smoke/tb_logs` 追加 `eval_dense_tile_halo/*` 诊断图:原 `p192/o160` 细密 tiled 配置加 per-tile halo 后,可直接对比 `aligned | dense no-halo | dense halo64 | halo96 | halo128`。用户目视结论: `halo64` 仍不够,`halo96` 已可抑制,`halo128` 与 `halo96` 无明显差异。
- 关键指标: full-frame outer halo 诊断在 K2 smoke checkpoint 上通过 `halo_hr=0/64/96/128`;实测峰值约 `11.6/14.5/16.0/9.4GB`(CUDA allocator 影响较大,作相对参考)。
- 结论: eval 侧优先使用 `full_halo + halo_hr=96` 查看真实图;不再把 64 作为主推荐,128 仅在后续出现新型边界残留时再试。

**涉及文件**: `algos/ep07_unet_sr/src/unet_sr/real_eval.py`, `algos/ep07_unet_sr/src/unet_sr/config.py`, `algos/ep07_unet_sr/src/unet_sr/solver_train.py`, `algos/ep07_unet_sr/src/unet_sr/train.py`, `algos/ep07_unet_sr/tests/test_real_eval.py`, `algos/ep07_unet_sr/tests/test_config.py`

---

### [ACL-037] 2026-06-29 — K4 shared solver 的 30px 方块来自 recurrent prox 残差累积,主线改为 K2

**问题诊断**:
- `outputs/solver_v5_nodrizzle` 的 no-drizzle 真实图仍出现约 30 display-px 量级的暗/亮块状纹理;该现象不是 phase-bin drizzle 独有,也不是 `patch_size_hr=192` tile seam。
- 诊断方式:固定同一真实主 session 中心 patch,不经过 full-frame tiled blending,展开 `x0 -> prox1 -> dc1 -> ... -> prox4 -> dc4`,并在平坦区统计相对 `x0` 的残差 RMS 与 lag=15/30/32 HR-px 自相关。
- 关键发现:
  - `x0=aligned_mean` 和 5 个 no-drizzle condition channel 本身没有 15/29/30 HR-px 周期峰。
  - 已训练 `solver_v5_nodrizzle/solver_step_005000.pt` 中,`prox1` 后新增 15 HR-px 残差相关;`prox2/prox4` 后出现最强 29-30 HR-px 残差;DC 步会压低一部分但不会消除。
  - 只反复跑同一个 shared prox、不跑 DC,第 2/3 次 prox 已出现 32/30 HR-px best lag,说明源头是 `x <- x + prox([x, cond])` 的 shared recurrent residual loop。
  - 同一主 checkpoint 若停在 `dc2`,平坦区 `delta_rms=0.0969`、`corr30=0.421`、`edge_grad=0.784`;继续到 `dc4` 后 `delta_rms=0.1180`、`corr30=0.485`、`edge_grad=0.779`。后两轮没有带来边缘收益,反而放大 30px 残差。
- 机制判断:solver prox 是 HR 同分辨率 UNet,没有 `ConvTranspose2d`/PixelShuffle 棋盘源;但三层 pooling + shared recurrent residual 会引入相位敏感的固定模式,多次迭代把弱残差积分成可见块状纹理。普通 V10/UNet 只做一次映射,因此没有同样的 recurrent 放大。

**修改内容**:
1. 训练策略调整:主线不再使用 `--unroll-steps 4` 的 shared prox 配置,优先改为 `--unroll-steps 2`。
2. 暂不把 `--no-solver-share-weights` 作为主线:200-step smoke 中 K4 unshared 参数量从 7.45M 增至 29.78M,早期中间状态不稳定,训练成本和泛化风险更高。
3. 若 K2 长训后仍有残留,下一步代码级修复才考虑 prox residual damping (`x <- x + alpha * delta`,如 `alpha=0.5`) 或 flat-region residual 正则;不优先做后处理。

**预期效果**:
- 减少 shared prox recurrent loop 对 30px 残差模式的放大,同时保留前两轮带来的边缘增强和 terminal DC consistency。
- 速度提升:200-step smoke 中 K2 shared 约 `219 ms/step`,K4 shared 约 `370 ms/step`;预计 K2 训练吞吐明显优于 K4。
- 风险:K2 的物理/learned alternating 步数更少,若少数结构需要更多 refinement 可能略降锐度;需用真实图和 `eval_real/dc_resid_*` 做主判据。

**推荐参数**:
```bash
uv run python -m unet_sr.solver_train \
  --training-pool-dir ../../data/synthetic/pool_2x_v5_5k \
  --input-mode hybrid_drizzle2x --scale 2 --unroll-steps 2 \
  --solver-m-frames 12 --solver-band-sigma 5 \
  --solver-prior-anneal-steps 0 --solver-dc-weight 0 \
  --boundary-boost 4.0 --flatness-weight 0.0 \
  --synth-eval-holdout 200 --synth-eval-every 2500 \
  --batch-size 20 --patch-size-hr 192 \
  --num-workers 14 --compile --log-every 1000 --save-every 5000 \
  --solver-no-drizzle \
  --output-dir outputs/solver_v6_k2_nodrizzle --total-steps 50000
```

**训练结果**: _(K2 主线训练后回填)_
- 输出目录: `outputs/solver_v6_k2_nodrizzle`
- 视觉效果: 待训练后填写。
- 关键指标: 诊断基线来自 `outputs/solver_v5_nodrizzle/solver_step_005000.pt`: `dc2` vs `dc4` 同 patch 对比显示 `delta_rms 0.0969 -> 0.1180`, `corr30 0.421 -> 0.485`, `edge_grad 0.784 -> 0.779`。
- 结论: 先用 K2 替代 K4 shared 作为最低风险修复;不从 K4 checkpoint resume,需要从头训练。

**涉及文件**: `research_log/algorithm_changelog.md`

---

### [ACL-036] 2026-06-29 — DC forward plan: 预计算 PSF/gather 常量以减少 solver kernel 图开销

**问题诊断**:
- `solver_train.py` 的 K-step solver 稳态 profiling 显示 DataLoader/H2D 已不是瓶颈(`~0.2/0.8 ms`),主要耗时在 prox + DC + backward。
- DC 路径里 `forward_burst()` 每个 unroll step、以及 custom VJP backward 中都会重复构造同一 batch 的 PSF grouped-conv 权重、shifted block-average gather 索引/权重。它们只依赖 batch 的 PSF/shifts/patch shape,不依赖当前估计 `x`,因此重复构造是纯开销。
- 完整 CUDA Graph replay 直接套 DC 不安全: per-batch PSF shape/sigma/kernel radius 和 shifts 会改变图结构/常量;先把这些动态常量显式 plan 化,是更稳的前置优化。

**修改内容**:
1. `forward_torch.py` 新增 `ForwardBurstPlan` 与 `prepare_forward_burst_plan()`:
   - 预计算 per-batch PSF separable/2D grouped-conv 权重与恢复顺序;
   - 预计算 shifted detector block-average 的 gather index、插值权重和 validity mask;
   - `forward_burst(..., plan=...)` 使用 planned path,数学公式和原 fast path 等价。
2. `data_consistency_grad()` 与 `_DCGradLinearVJP` 接收并保存 plan,使训练 forward 与 custom VJP backward 都复用同一批常量。
3. `unroll.py` 在每次 solver forward 开头构建一次 plan,供 K 次 DC step 复用;新增环境变量 `TL_SOLVER_FORWARD_PLAN=0` 可回退旧路径。
4. `tests/test_forward_torch.py` 新增 planned path 等价测试,覆盖 forward、autograd adjoint `A^T`、DC gradient `g` 和训练用二阶 VJP `d/dx<g,c>`。

**预期效果**:
- 不改变训练语义、不改变 loss、不改变 forward operator;只减少每个 step 中重复的 kernel/index 构造和小 kernel 图开销。
- 预期提升 solver 吞吐,尤其在 K=4 且 `--compile` 后 prox 已较快时更明显。
- 风险: planned constants 与 batch shape/dtype/device 强绑定;如遇驱动/边界问题可用 `TL_SOLVER_FORWARD_PLAN=0` 回退。

**推荐参数**: 保持当前 solver 命令;默认启用 planned path。若要禁用: `TL_SOLVER_FORWARD_PLAN=0 uv run python -m unet_sr.solver_train ...`

**训练结果**:
- 输出目录: 未启动长训;本轮为 microbenchmark + 等价测试。
- 视觉效果: 未评估视觉,该改动应为数学等价优化。
- 关键指标:
  - B24/K4/M12/base64/`--compile`: no-plan `441.1 ms/step, 54.4 samples/s`; planned `345.3 ms/step, 69.5 samples/s`。
  - B20/K4/M12/base64/`--compile`: planned `290.1 ms/step, 68.9 samples/s`。
  - planned path DC forward 段约 `70.0 ms -> 9.7 ms`(B24),总吞吐约 `+27.7%`。
  - 测试: `tests/test_config.py tests/test_forward_torch.py tests/test_gate_c_smoke.py tests/test_real_eval.py` 共 22 项通过; planned path forward/A^T/DC/VJP 与 unplanned fp64 对齐 `<1e-12`。
- 结论: 值得合入;B20 planned 与 B24 planned 吞吐接近,考虑 OOM 风险可继续用 B20,追吞吐可用 B24。

**涉及文件**: `algos/ep07_unet_sr/src/unet_sr/forward_torch.py`, `algos/ep07_unet_sr/src/unet_sr/unroll.py`, `algos/ep07_unet_sr/tests/test_forward_torch.py`

### [ACL-035] 2026-06-29 — solver 真实数据评估改为 tile batching

**问题诊断**:
- `maybe_log_solver_real_eval()` 在每个 checkpoint 会对 248 帧真实主 session 做 EP11-style 推理;默认 `patch_size_hr=192`、`overlap=128` 时约 234 个 tile。
- 旧实现逐 tile 以 batch=1 调用 solver,每个 tile 都重复 K-step prox + DC 物理步,大量 GPU launch 被拆碎;这会让 checkpoint 后验/真实评估明显拖慢,尤其是新增 `eval_real/dc_resid_*` 后整体评估更重。

**修改内容**:
1. `real_eval.py` 的 `infer_solver_from_burst()` 新增 `tile_batch_size`,把多个 full-frame tile 堆叠成一个 batch 送入 solver;PSF、burst、shift 和 frame mask 按 batch 广播/构造。
2. `config.py` 新增 CLI 参数 `--real-eval-tile-batch`(默认 16),并在 `train.py` / `solver_train.py` 的 `RealEvalConfig` 中传递。
3. 保持输出 blending、window 加权和温度/高通指标口径不变;只减少 checkpoint eval 的小 batch 调用开销。

**预期效果**:
- checkpoint real_eval 墙钟时间下降,尤其是 GPU 上的 solver tile inference;训练主 step 数学不变。
- 风险: tile batch 过大可能增加显存峰值;显存不足时把 `--real-eval-tile-batch` 降到 8 或 4。

**推荐参数**: `--real-eval-tile-batch 16`；若显存充足可试 `24` 或 `32`,若 checkpoint eval OOM 则降到 `8`。

**训练结果**: _(训练后填写)_
- 输出目录: `outputs/solver_v5_nodrizzle`
- 视觉效果:
- 关键指标:
- 结论:

**涉及文件**: `algos/ep07_unet_sr/src/unet_sr/real_eval.py`, `algos/ep07_unet_sr/src/unet_sr/config.py`, `algos/ep07_unet_sr/src/unet_sr/train.py`, `algos/ep07_unet_sr/src/unet_sr/solver_train.py`

### [ACL-034] 2026-06-29 — solver_train 支持完整 resume 与 TensorBoard purge 续写

**问题诊断**:
- `solver_train.py` 虽然继承了通用 `--resume` CLI 参数,但此前没有实际读取 `config.resume_from`;中断后重跑会从 step 0 开始,并在同一 TensorBoard logdir 写入重叠 step,导致曲线混乱。
- 旧 solver checkpoint 只保存 `step`、`model_state_dict`、`config`,缺少 AdamW 动量与 LR scheduler 状态;从旧 checkpoint 只能做权重 warm restart,不能做到数学意义上的无缝续训。

**修改内容**:
1. `solver_train.py` 新增 `_save_solver_checkpoint()` / `_load_resume_checkpoint()`,新 checkpoint 保存并恢复 `optimizer_state_dict` 与 `scheduler_state_dict`。
2. 恢复时将训练步数、tqdm 初始值和日志 global step 设为 checkpoint step;旧 checkpoint 缺少 optimizer 时明确提示 fresh AdamW moments,缺少 scheduler 时按 step 快进 scheduler。
3. TensorBoard writer 在 resume 时使用 `purge_step=start_step + 1`,保留 checkpoint step 及之前曲线,隐藏中断尾部的脏事件。
4. 修复 resume step 已达到 `--total-steps` 时仍多跑一步的边界问题;直接保存 `solver_final.pt` 并退出。

**预期效果**:
- 新 checkpoint 起可正常中断/恢复 optimizer、scheduler 和训练步数;TensorBoard 曲线从旧 step 后自然续写。
- 从历史 `solver_step_010000.pt` 恢复仍是 warm restart,AdamW 动量无法补回;但保留 10K 权重并重建 scheduler 位置,优于从头训练。
- 风险: 旧 checkpoint 的 optimizer 动量缺失可能造成恢复后短期 loss 抖动;后续新 checkpoint 不再有这个问题。

**推荐参数**:
```bash
uv run python -m unet_sr.solver_train \
  --training-pool-dir ../../data/synthetic/pool_2x_v5_5k \
  --input-mode hybrid_drizzle2x --scale 2 --unroll-steps 4 \
  --solver-m-frames 12 --solver-band-sigma 5 \
  --solver-prior-anneal-steps 0 --solver-dc-weight 0 \
  --boundary-boost 4.0 --flatness-weight 0.0 \
  --synth-eval-holdout 200 --synth-eval-every 2500 \
  --batch-size 20 --patch-size-hr 192 \
  --num-workers 14 --compile --log-every 1000 --save-every 5000 \
  --solver-no-drizzle \
  --output-dir outputs/solver_v5_nodrizzle --total-steps 50000 \
  --resume outputs/solver_v5_nodrizzle/solver_step_010000.pt
```

**训练结果**: _(训练后填写)_
- 输出目录: `outputs/solver_v5_nodrizzle`
- 视觉效果:
- 关键指标:
- 结论:

**涉及文件**: `algos/ep07_unet_sr/src/unet_sr/solver_train.py`

### [ACL-033] 2026-06-29 — 物理前向算子向量化(去掉 per-sample 循环,grouped-conv 批处理)

**问题诊断**:
- solver 每 step ~727ms 且 **GPU 利用率跑不满**。热路径在 `forward_torch.forward_burst`:它对 batch 用 **`for b in range(B)` 的 Python 循环**——每个场景单独建 PSF kernel、单独做大图卷积 + 重采样(因为 65% 场景是 elliptical/airy,PSF 逐场景不同)。该 `forward_burst` 在 unroll 里每步 DC 调 K=4 次,反向求 `A^T`(autograd)再遍历,训练 double-backward(fp32,create_graph)再遍历 → 一个 step ~4×B 次**串行**小 kernel,GPU 做一点就停下等下一次 Python 迭代 → launch-bound、利用率低。prox UNet(批处理 + compiled)不是瓶颈。
- 关键约束:`forward_torch` 是 **Gate-A 认证算子**(fp64 下证了线性性 + autograd 伴随 `A^T` 的精确性),任何改写必须**数值等价**且重过认证。

**修改内容**:
1. `forward_torch.py`:新增向量化前向 `_forward_burst_fast`——把 per-sample 大图卷积换成 **grouped convolution** 一次 launch 处理整个 batch 的异构 PSF:
   - 按路径分组:各样本仍走它在 `_blur_one` 里**本来的路径**(各向同性高斯走可分离 fast path、其余走 2D kernel path),所以数值不变;
   - 组内不同核**零填充到统一半径**(尾部权重 + 多出的输入 padding 都是 0,精确等价),用 `conv2d(groups=n)`(可分离用两次 grouped conv1d);
   - 重采样 `block_average_shifted_batched`:把原函数加一维 B 批处理(同样的 gather/数学,只是不再逐 b)。
2. `forward_burst(..., fast=None)`:默认走 fast(可用环境变量 `TL_SOLVER_FAST_FORWARD=0` 或 `fast=False` 回退到认证 loop);保留 `_forward_burst_loop` 作为参考/回退。
3. **DC 梯度的二阶 double-backward → 一阶 self-adjoint VJP**(`_DCGradLinearVJP`):剖析发现训练真正的瓶颈不是卷积,而是 `data_consistency_grad` 的 **autograd 二阶图**(实测 DC+double-bwd ≈ 175ms,占 step 绝大头,且 forward 优化对它几乎无效)。DC 目标 ½‖M·H(Ax−y)‖² 对 x 是**二次型**,梯度 g 仿射,Jacobian `J=dg/dx=AᵀHᵀM²HA` 是**固定自伴线性算子**——所以不需要二阶图,backward 直接给 `J·v = ∇_v(½‖M·H·A·v‖²)`,**一次一阶 autograd**(A^T 仍由 autograd 给,认证伴随不变)。Huber 时(非线性)自动回退二阶图。环境变量 `TL_SOLVER_FAST_DCGRAD=0` 或 `fast_vjp=False` 回退。
4. 认证:`tests/test_forward_torch.py` 新增 `[4] fast==loop`(fwd + `A^T` fp64 对齐)+ VJP 等价测试(g 与二阶梯度 `d/dx⟨g,c⟩` 对齐 + `torch.autograd.gradcheck`)。本地实测 **forward 逐元素 0.0、A^T 5.6e-16、VJP 二阶梯度逐元素 0.0、gradcheck PASS**,Gate-A 全过。
5. `scripts/bench_forward_fast.py`(新):fp64/fp32 正确性(含 VJP 训练梯度)+ forward 与 DC-grad(含 double-backward)的 old-vs-new 计时/加速比(远端 GPU 上跑出真实数字)。

**预期效果**:
- forward:per-sample 串行 → grouped-conv,**GPU 实测 2.73x**(5090,batch18)。
- 训练 DC 路径:消掉二阶图 → 一阶 VJP(CPU 1.3x;**GPU 上预期更大**,二阶 gather/scatter 图开销大);叠加拉满利用率。**数值与认证算子逐元素等价**(g、A^T、二阶梯度全 0.0 差),训练/收敛行为不变,纯提速。
- 风险可控:两个环境变量分别一键回退;Gate-A + VJP gradcheck 是护栏。

**验收标准**:`uv run python -m pytest tests/test_forward_torch.py` 全过 + `uv run python scripts/bench_forward_fast.py` 报两个 correctness OK 且 new 不慢于 old。
**训练结果**: 待填(远端 5090:forward 2.73x 已测;接入后实际 ms/step 待填)。

---

### [ACL-032] 2026-06-29 — 去波纹暖启动(de-waffle x0) + 真实数据物理一致性指标(dc_resid)

**问题诊断**:
- v5 hybrid solver(`solver_v5_sharp`)真实图视觉很强(线直、几乎无幻觉),但暗背景有一层很淡的 ~2 HR-px 周期棋盘格(深网格 + 浅方格,无硬边界)。
- 根因(读代码确认,非 loss 问题):hybrid 9ch 路径的暖启动 `x0 = obs[ch5] = phase_bin_drizzle[0]`。phase-bin drizzle 把每帧按亚像素相位分到 4 个 bin、各自 drizzle 到 2x 网格;平坦背景相位覆盖不均时,每个 2x2 块的 4 个子位置来自不同帧子集 → 2-HR-px(=1 pitch=20µm)的覆盖"波纹"。该波纹在 step 5000 就已存在(因为它在 `x0` 里),DC 步是带限(highpass σ5)且平坦区 `Aᵀ(Ax−y)≈0`、整片 MSE(0.2)又被高幅值芯片主导 → 平坦区无锚,波纹被原样保留。prox UNet 用 bilinear interpolate 上采样(`model.py`),**排除转置卷积棋盘伪影**。
- 评测反演:plain UNet(v10)synth 指标(psnr 38.8 / boundary_f1 0.865)反而高于 solver(36.2 / 0.771),但真实图更差(串珠 + 背景絮状物)。说明 synth PSNR/boundary_f1 奖励"拟合合成 GT 生成器",与真实质量反相关,**不能用来选 solver 超参**。缺一个无 GT 的真实物理判据。

**修改内容**:
1. `config.py`:新增 `--solver-warmstart {phasebin,aligned_mean}`(默认 `phasebin`=保持旧行为)。`aligned_mean` 把 hybrid 路径的 `x0` 从 ch5(波纹相位 bin)换成 ch0(平滑 fused aligned_mean),**保留全部 9 个 cond 通道**(prox 仍看得到相位 bin),只是种子去波纹。`--solver-no-drizzle` 下无效(该路径本就从 ch0 暖启动)。
2. `solver_train.py` / `real_eval.py`:`mean_ch` 按 warmstart 解析(`0 if no_drizzle or warmstart==aligned_mean else 5`),训练与真实推理两端一致。
3. `real_eval.py`:新增真实数据物理一致性指标 `eval_real/dc_resid_band` 与 `eval_real/dc_resid_full` —— 把整帧重建回代认证前向算子 `‖A(x)−y‖`,用**留出帧**(不在 DC 子集里,避免自洽)+ `solver.dc_residual_rms()`。这是唯一 grounded 在物理、而非合成生成器上的真实判据(注意真实 PSF 是单高斯 σ0.5、被错配,故只作**相对**比较)。失败时 try/except 跳过,不中断隔夜训练。
4. `scripts/diagnose_drizzle_waffle.py`(新,numpy/scipy-only,零 GPU):对池场景 FFT 对比 ch0(aligned_mean)/ ch5(phasebin x0)/ GT 的 out_of_band 与 Nyquist `grid_score`,证明波纹只在 ch5、ch0 干净,并可出平坦 ROI montage PNG。

**预期效果**:
- `--solver-warmstart aligned_mean`:背景棋盘格在源头消失(种子去波纹),线条直度/细节不受影响(由 DC + cond 驱动,而非种子)。
- `--solver-no-drizzle`:作为最干净的对照(全程无 phase-bin),既验证 drizzle 是格子来源,又补上 v5 上 hybrid-vs-nodrizzle 缺失的 A/B;若视觉等同则系统更简单(可弃整条 phase-bin 路径)。
- `dc_resid`:为下一步选 checkpoint/配置提供真实物理判据;预期 solver(end-on-DC) < plain UNet(无视前向)。

**验收标准(隔夜两跑,次晨对比)**:格子消失 + 线仍直 + 真实 `dc_resid_band` ≤ 当前 hybrid + 真实 `out_of_band` 仍在 [0.002, 0.005]。

**训练结果**: _(2026-06-29 部分回填;两跑均未作为最终主线继续)_
- Run B `outputs/solver_v5_dewaffle` (`--solver-warmstart aligned_mean`,保留 9ch phase-bin cond):已保存 5k/10k/15k/20k/25k/30k checkpoint。5k 真实指标 `out_of_band=0.002074`, `artifact_score=0.439135`, `dc_resid_band=1.22472`, `dc_resid_full=1.50165`;30k 真实指标 `out_of_band=0.002310`, `artifact_score=0.468315`, `dc_resid_band=1.22643`, `dc_resid_full=1.50240`。视觉上亮区/背景仍有云块状纹理和边缘旁条纹,随训练未变干净。
- Run A `outputs/solver_v5_nodrizzle` (`--solver-no-drizzle`,5ch cond):当前本地只保留 5k checkpoint。5k 真实指标 `out_of_band=0.001913`, `artifact_score=0.421701`, `dc_resid_band=1.22283`, `dc_resid_full=1.50202`;共同 5k 处比 dewaffle 更干净,dc residual 基本打平。
- 输入侧诊断(真实 248 帧,不经过网络):`aligned_mean_up2` 在 full-flat mask 上 `out_of_band=1.7e-5`,4 个 phase-bin drizzle channel 为 `0.00186-0.00235`;phase-bin mean 仍有 `out_of_band=1.12e-4`。这说明 phase-bin cond 在平坦区自带 SR-band 相位/覆盖伪纹理,即使 warmstart 改成 aligned_mean,prox 仍能从 cond 读到并放大这些伪高频。
- Synthetic 指标不作为主判据:dewaffle 30k 的 synthetic PSNR/region RMSE 继续改善,但真实 artifact 同步升高,再次说明 synth GT 指标不能单独用于选择 solver 超参。
- 结论:在 aligned-mean warm-start 与 no-drizzle 两条输入路径中,优先保留 `--solver-no-drizzle`。不过两者仍共享 ACL-037 诊断出的 K4 shared recurrent prox 方块问题;下一轮应从头跑 `--solver-no-drizzle --unroll-steps 2`,而不是继续 K4 dewaffle/nodrizzle 二选一长训。

---

### [ACL-031] 2026-06-28 — solver_train 提速:DC monitor 延迟计算 + `--compile` 编译 prox 子网

**问题诊断**:
- v5 5k solver 首训时 GPU 利用率波动,增加 `--num-workers` 后收益不明显。检查热路径发现 `--solver-dc-weight 0` 的推荐配置下,训练循环仍每个 step 额外调用一次 `terminal_dc_loss()` 只为记录 `loss/dc` monitor,等价于每步多跑一次物理 forward `A(x)`。
- `solver_train.py` 接收 `--compile`,但此前没有使用 `config.compile_model`;用户加 `--compile` 对 solver 路径实际无效。
- 整个 solver 不适合直接 `torch.compile(solver)`:unroll 内含 `autograd.grad`、per-scene PSF Python loop 和动态 `ScenePSF`,整图编译风险高且 checkpoint state_dict 可能变复杂。

**修改内容**:
1. `solver_train.py`:当 `--solver-dc-weight 0` 时,不再每步计算 `dc` monitor;仅在 `step==1` 或 `step % --log-every == 0` 时用 `torch.no_grad()` 计算并写入 TensorBoard。`solver_dc_weight>0` 时保持原行为,因为 DC loss 参与反传。
2. `solver_train.py`:实现 `--compile` 在 solver 路径的实际作用,只编译 learned prox UNet 子网络;物理 DC 路径保持 eager,避免 `autograd.grad`/PSF 分支图断裂。
3. `solver_train.py`:保存 checkpoint 时清理 compiled child module 的 `._orig_mod.` state_dict 前缀,保持 `solver_step_*.pt` / `solver_final.pt` 可被未编译模型读取。

**预期效果**:
- 推荐 solver 配置 (`--solver-dc-weight 0`) 下减少每个 step 一次额外物理 forward,预期提高吞吐并降低 GPU 小 kernel/同步开销。
- `--compile` 对 prox 卷积子网生效,可能进一步提升 5090 上的卷积吞吐;若编译开销或 Triton 编译器环境不稳定,可去掉 `--compile` 回到 eager。
- 风险: `loss/dc` 不再每步记录,只按 `log_every` 采样;不影响训练梯度、checkpoint real_eval 或 synth_eval。

**推荐参数**:
```bash
uv run python -m unet_sr.solver_train \
  --training-pool-dir ../../data/synthetic/pool_2x_v5_5k \
  --input-mode hybrid_drizzle2x --scale 2 --unroll-steps 4 \
  --solver-m-frames 12 --solver-band-sigma 5 \
  --solver-prior-anneal-steps 0 --solver-dc-weight 0 \
  --boundary-boost 4.0 --flatness-weight 0.0 \
  --synth-eval-holdout 200 --synth-eval-every 2500 \
  --total-steps 20000 --batch-size 16 --patch-size-hr 192 \
  --save-every 5000 --num-workers 8 --compile --log-every 1000 \
  --output-dir outputs/solver_v5_sharp
```

说明: v5 5k 已完整生成 `phase_bin_drizzle_2x.npy`(5000/5000 scene),主跑优先使用 hybrid drizzle 路径;仅当 drizzle 文件缺失或 IO 压力过大时才加 `--solver-no-drizzle` 回退到 5ch aligned-mean warm-start。

**训练结果**: _(训练后填写)_
- 输出目录: `outputs/solver_v5_sharp`
- 视觉效果:
- 关键指标:
- 结论:

**涉及文件**: `algos/ep07_unet_sr/src/unet_sr/solver_train.py`, `algos/ep07_unet_sr/tests/test_gate_c_smoke.py`

### [ACL-030] 2026-06-27 — 诊断"保真好/无幻觉/但很糊":blur 在 GT 不在 loss → v5 把 GT edge_sigma 1.4→0.8

**问题诊断**:
- v4 训出的 V10/solver 在真实数据上:保真好、无珠串幻觉,但糊、无细节。怀疑 loss 坏了。
- 实测远端 checkpoint 真实温度图的 `out_of_band_ratio`(ACL-027 新指标):V10 5K→40K、solver 2.5K→20K **全程死平 ≈0.00001**,p99 梯度也死平 —— 不是越训越糊,而是一上来就顶到天花板、再训无用。
- 直接量 v4 GT 本身(edge_sigma=1.4):`out_of_band(GT)=0.00008`,与模型真实输出 0.00001 同量级、都≈0。**即使完美模型精确复制该 GT 也是糊的 —— 天花板是 GT,不是模型/loss。**
- 根因:edge_sigma=1.4 把等温场边缘高斯糊到 ~36µm = 20µm pitch 的 1.8 倍,GT 里根本没有 SR 带细节可学。概念错误:**真实感(模糊)属于观测/前向**(PSF 已经把 GT→观测模糊了),**清晰属于 GT**;再给 GT 叠 edge_sigma 是双重模糊、抹掉学习目标 → 模型学成"别锐化"(≈恒等),迁移到真实数据就是糊。

**修改内容**:
1. 新增 `configs/synthetic/pool_2x_v5_sharp.json`:除 `temperature_isothermal.edge_sigma` 1.4→0.8 外与 v4 完全相同(seed 保持 940940 → 受控 A/B,同几何只变 GT 锐度;output_dir `data/synthetic/pool_2x_v5_5k`)。`edge_sigma` 本就是 config 旋钮,无需改代码。
2. 本地用 `render_isothermal_field` + `metrics.out_of_band_ratio` 实测 GT 的 SR 带能量随 edge_sigma 变化(下表),确认 0.8 把可恢复细节抬 ~40×。
3. `docs/REMOTE_ORDERS.md` 增 "DATA REGEN — v5 sharp GT" 段:先 300-scene smoke 重生 + 短训确认 `eval_synth/out_of_band_ratio` 抬离 ~0,再全量重生;V10/solver 训练池改指向 v5。

**edge_sigma → GT 的 out_of_band(本地实测,2x grid)**:

| edge_sigma | out_of_band(GT) | 说明 |
|---|---|---|
| 1.4 (v4) | 0.00008 | ≈ 模型真实输出 0.00001 = 没细节 |
| 1.0 | 0.00110 | 14× |
| **0.8 (v5)** | **0.00324** | **40×;边缘≈1 pitch = 可恢复极限** |
| 0.6 | 0.00967 | 120×(逼近 sub-pitch,谨慎) |

**预期效果**:
- GT 带回可恢复的 SR 带细节,模型这才有"模糊 obs → 锐 GT"的去卷积可学,真实输出变锐。
- **不要降到 ~0.6 以下**:GT 比 pitch 还细 → 模型只能幻觉去够 → 珠串/FM-1 回归(老版本死法 = ACL-023 诚实天花板)。0.8 = 诚实的最锐。
- 风险/待验:真实输出(0.00001)比 v4 GT(0.00008)还低 → 真实观测可能比合成 obs 还糊;GT 修锐后若合成 obs 的 PSF 比真实窄,真实迁移仍受限 —— 单独验 obs-PSF 真实性。loss 侧可选 `asymmetric_laplacian_weight~0.05`(只罚比 GT 糊,GT 锐之后才有意义)。

**训练结果**: _(训练后填写)_
- smoke: `outputs/v5_smoke`(看 eval_synth/out_of_band 是否抬离 0)
- 全量: `outputs/v10_v5_sharp` / `outputs/solver_v5_sharp`

**涉及文件**: configs/synthetic/pool_2x_v5_sharp.json(新), docs/REMOTE_ORDERS.md, scripts/verify_pool_sharpness.py(新);无算法代码改动(edge_sigma 为既有 config 旋钮)

**修正(2026-06-28,远端跑出 0.8 池后)**:
- 上表是在 **HARD mask** 上孤立测的;真实生成管线把**抗锯齿 SSAA coverage mask**喂给 render(`build_scene_mask_with_metadata(antialias=True, ssaa_factor=4)` → `render_isothermal_field`),coverage 自带 ~0.7 HR px 软化,所以 edge_sigma **不是唯一软化源**。
- 按真实管线重测(AA + defects,本地复现):1.4→0.00008、**0.8→0.0013(远端 5k 池实测 0.00122,吻合,验证 sweep 可信)**、**0.6→0.0039(边缘≈1 pitch=目标)**、0.5→0.0073、AA 地板(edge_sigma=0)→0.018。
- 结论:0.8 只到 **15× v4(仍糊)**;**v5 改用 `edge_sigma=0.6`**(~50× v4,边缘落在 pitch)。**别低于 ~0.5**(逼近 AA 地板 = sub-pitch → 幻觉/珠串)。"realism 属于观测、清晰属于 GT"不变,只是 AA 已占掉一部分软化预算。
- verify_pool_sharpness.py 修两个 bug:(1) 缺陷计数键 `geo_meta`→`geometry_metadata`(导致 0/96 假阴性,缺陷其实在 mask + metadata 里);(2) PASS 阈值按真实管线重标 0.0025–0.010(原 0.0015 来自 hard-mask 高估,把合格的 0.6 也会判错;0.8 应判 too-soft)。
- 远端流程:`rm -rf data/synthetic/pool_2x_v5_5k`(旧 0.8 部分池;否则 resume-skip 会混入旧 scene)→ 按 0.6 重生 → `verify_pool_sharpness.py` PASS → smoke 短训。

---

### [ACL-029] 2026-06-27 — solver_train 接入 V10 同款 checkpoint real-eval/PNG 自动出图

**问题诊断**:
- V10/plain-UNet 训练入口在每个 checkpoint 自动运行 `real_eval`，写 TensorBoard `eval_real/*` 并保存 `eval_real/unet_step*_center_zoom3x_temperature.png`，可直接观察真实主 session 上的演化。
- `solver_train.py` 只保存 `solver_step_*.pt` 和 held-out synthetic 指标，未接入真实数据自动出图；当前 `outputs/solver_v4_acl027` 训练到 5K 只能看合成指标，缺少和 V10 对齐的真实温度图演化。
- solver 不能直接复用 `maybe_log_real_eval(model=solver)`：普通推理只调用 `model(features)`，而 `UnrolledSolver.forward()` 需要 `x0, lr_burst, shifts, ScenePSF, cond, frame_mask`。

**修改内容**:
1. `real_eval.py` 新增 `infer_solver_from_burst()`：真实主 session 上构建与训练一致的 solver condition，`solver_no_drizzle=True` 时为 `5 fused↑2x`，否则为 `5 fused↑2x + 4 phase-bin drizzle@2x`；`x0` 分别取 ch0 / ch5。
2. `real_eval.py` 新增 `maybe_log_solver_real_eval()`：复用 EP11-style center-zoom 温度图、highpass TensorBoard panel、`out_of_band_ratio` / `artifact_score` 标量，并保存 `eval_real/solver_step*_center_zoom3x_temperature.png`。
3. solver real-eval 的 DC burst 使用确定性均匀子集，帧数为 `--solver-m-frames`，避免每个 tile 对 248 帧全量做 K-step DC 造成 checkpoint eval 过慢。
4. 真实数据无合成 scene 的 per-scene PSF metadata，因此 solver real-eval 明确使用配置标量 `forward_model_psf_sigma` 的 Gaussian PSF 作为监控假设；该输出用于 checkpoint 视觉演化/质量门控，不作为物理 GT。
5. `solver_train.py` 在 `save_every` 和 final 节点调用 solver real-eval，和 V10 一样自动写 TensorBoard/PNG；新增启动时 real-eval cadence 打印。
6. `tests/test_real_eval.py` 增加 solver adapter 回归测试，覆盖 5ch no-drizzle 与 9ch hybrid contract，以及 deterministic `solver_m_frames` 子集。

**预期效果**:
- solver 训练在 checkpoint 处自动生产真实主 session center-zoom 温度 PNG，可直接和 V10 的演化图对齐检查。
- 避免 old 8ch hybrid eval 回退；测试固定 9ch contract。
- 风险: solver real-eval 比 plain UNet 显著更慢，尤其 `patch_size_hr=192 / overlap=128` 会产生较多 tile；必要时用 `--real-eval-every` 降低频率或 `--real-eval-frame-limit` 做快速 smoke。

**推荐参数**:
```bash
uv run python -m unet_sr.solver_train \
  --training-pool-dir ../../data/synthetic/pool_2x_v4_5k \
  --input-mode hybrid_drizzle2x --scale 2 --unroll-steps 4 \
  --solver-no-drizzle --solver-m-frames 12 --solver-band-sigma 5 \
  --solver-prior-anneal-steps 0 --solver-dc-weight 0 \
  --boundary-boost 4.0 --flatness-weight 0.0 \
  --synth-eval-holdout 200 --synth-eval-every 2500 \
  --total-steps 20000 --batch-size 16 --patch-size-hr 192 \
  --save-every 2500 --log-every 1000 --num-workers 16 \
  --output-dir outputs/solver_v4_acl027
```

**训练结果**: _(训练后填写)_
- 输出目录: `outputs/solver_v4_acl027`
- 视觉效果:
- 关键指标:
- 结论:

**涉及文件**: `algos/ep07_unet_sr/src/unet_sr/real_eval.py`, `algos/ep07_unet_sr/src/unet_sr/solver_train.py`, `algos/ep07_unet_sr/tests/test_real_eval.py`

---

### [ACL-028] 2026-06-26 — 修复 hybrid real_eval 推理仍生成 8ch 输入导致 9ch 模型崩溃

**问题诊断**:
- ACL-023/027 后训练数据集的 `input_mode="hybrid_drizzle2x"` 已固定为 9ch: `5 fused↑2x + 4 phase-bin drizzle@2x`。
- 但 `infer_from_burst()` 的真实数据评估路径仍按旧 V9A contract 拼接 `5 fused↑2x + 3 scatter drizzle@2x`，在 checkpoint real_eval 时向 9ch 模型输入 8ch tensor，触发 `expected input ... to have 9 channels, but got 8 channels instead`。
- 该问题只影响推理/real_eval 入口，训练 batch 本身已经读取预计算 `phase_bin_drizzle_2x.npy`。
- 同轮检查发现 solver held-out synth eval 在 `torch.no_grad()` 下调用 unrolled DC step 时，内部 `autograd.grad(A^T(Ax-y))` 无局部梯度图，会在第一个 synth-eval 节点崩溃。

**修改内容**:
1. `inference.py`: hybrid 推理路径改用 `tcforge.classical_sr.phase_bin_drizzle(..., n_bins=4)`，与训练 dataset 的 9ch phase-bin contract 对齐。
2. `tests/test_inference.py`: hybrid inference 回归测试新增 9ch 检查，避免只用输出 shape 掩盖输入通道错误。
3. `forward_torch.py`: `data_consistency_grad()` 内部用 `torch.enable_grad()` 包住局部 DC 梯度计算，使 solver eval/no_grad 路径仍能计算 Aᵀ(Ax-y)，但外层不保留参数梯度。
4. `config.py` / `solver_train.py`: 更新 CLI help 和注释中的旧 8ch/3ch scatter 说法。
5. `tests/test_dataset.py` / `tests/test_forward_torch.py`: 测试 fixture 使用 `phase_bin_drizzle_2x.npy`，并新增 no_grad 下 DC-grad 回归测试。

**预期效果**:
- `train.py` 在 `save_every` / `real_eval` 节点不再因 8ch/9ch 不匹配中断。
- `solver_train.py` 在 `synth_eval_every` 节点不再因 no_grad 禁用 DC 内部 autograd 而中断。
- 真实数据 TensorBoard/PNG 推理输入与合成训练输入保持同一通道语义。
- 风险: real-data eval 仍需现场从 raw burst 计算 phase-bin drizzle；这会比旧 3ch scatter 略有计算成本，但只发生在 checkpoint eval。

**推荐参数**: 保持 ACL-027 命令不变；若只想快速越过训练节点，可临时加 `--real-eval-frame-limit 48` 降低 eval 成本。

**训练结果**: _(训练后填写)_
- 输出目录: `outputs/v10_v4_acl027`
- 视觉效果:
- 关键指标:
- 结论:

**涉及文件**: `algos/ep07_unet_sr/src/unet_sr/inference.py`, `algos/ep07_unet_sr/src/unet_sr/forward_torch.py`, `algos/ep07_unet_sr/src/unet_sr/config.py`, `algos/ep07_unet_sr/src/unet_sr/train.py`, `algos/ep07_unet_sr/src/unet_sr/solver_train.py`, `algos/ep07_unet_sr/src/unet_sr/unroll.py`, `algos/ep07_unet_sr/tests/test_inference.py`, `algos/ep07_unet_sr/tests/test_dataset.py`, `algos/ep07_unet_sr/tests/test_model_losses.py`, `algos/ep07_unet_sr/tests/test_forward_torch.py`

---

### [ACL-027] 2026-06-26 — Loss/评价指标重设计:thin/gap 线先验 → 几何无关 boundary 权重 + 等温 flatness;评测改用 held-out 合成 GT(out_of_band 取代 raw_control_corr)

**问题诊断**:
- 旧 loss/指标过度针对早期"完美矩形/细线"几何:`thin_boost=6 / gap_boost=4` 本质是直线探测器先验,在 v4 缺陷数据(孔/裂纹/断角,均 >pitch)上要么无意义要么帮倒忙(过度加权裂纹细发丝、欠加权大孔边界)。
- `real_eval` 的 `artifact_score` / `raw_control_corr` 拿"干净输出"与"退化原图(bicubic of raw mean)"对比:输出越干净越偏离参照 → 分数反而越差("但凡生成更干净的图,分数反而降低")。
- 结构性错位:有 GT 的合成集反而不记保真指标,没 GT 的真实集才记(被迫用"和退化图比"的代理)—— 正好反了。

**修改内容**:
1. `mask_weights.compute_boundary_weight_np`:`1 + boost·exp(-(dist/τ)²)`,从 mask 边界距离场算 —— 几何无关地强调每种边界(芯片外缘/孔壁/裂纹壁/缺口);细线=处处贴边界、窄缝=两侧贴边界,均作为同一距离场的特例自然涌现;对比无关 → 低 ΔT(hard/stress)缺陷也照样加权。替换 thin/gap(`--boundary-boost/--boundary-tau-px`)。
2. `ContourSRLoss`:thin/gap → 单一 `boundary_weight`(驱动 highpass + grad_vector);mse 回归全局 DC 锚;新增 `flatness` 项(在 GT 平坦处罚 `‖∇pred‖`,用对比归一化的 target 梯度做软掩膜,不与真实边界打架),编码近等温先验(默认关,v4 用)。
3. `metrics.py`(新,纯 numpy):`out_of_band_ratio`(GT-free/PSF-free,pitch 截止频率以上的谱能量=幻觉/珠串,本地验证 smooth/mid=0、Nyquist=1)、`psnr`、`region_rmse`(体内温度=等温保真)、`boundary_f1`(缺陷/边界保真:填孔↓recall、造边↓precision,本地验证填孔 recall 1.0→0.85)。
4. `real_eval`:删 `raw_control_corr`,改记 `out_of_band_ratio`;`artifact_score` 降级为 FM-1 cliff 监视器(只看跨 checkpoint 的相对跳变)。
5. `synth_eval.py`(新)+ 数据集 held-out 尾切片(`holdout_tail/holdout_role`,scene 目录数字序,训练自动排除尾部,零泄漏):有 GT 的合成集上记 `eval_synth/{psnr,region_rmse,boundary_f1,out_of_band_ratio}`,经 forward_fn 闭包同时接入 train.py(plain UNet)与 solver_train.py(unrolled solver)。

**预期效果**:
- 指标不再奖励"像退化图";真正的保真(区内 RMSE、缺陷 F1)与幻觉(带外能量)被分别诚实量化。
- 边界强调对新缺陷几何通用;线先验的经验作为特例保留。
- 风险:boundary/flatness 权重需调;`boundary_f1` 的梯度阈值是相对百分位 → 跨 run 看趋势,不看绝对值。

**预期效果验证(本地 CPU)**: boundary 权重(每种边界→5.0、内部→1.0、细线全程加权)、4 个指标、config 解析/校验全部本地通过;9 个 src + 3 个 test 文件 py_compile 通过;torch 部分(loss/dataset/synth_eval forward)留远端 pytest。

**推荐参数**: `--boundary-boost 4.0 --flatness-weight 0.0 --synth-eval-holdout 200 --synth-eval-every 2500`(第一跑 flatness 关;下一跑 `--flatness-weight 0.05` 做 A/B,用 eval_synth 判增益)

**训练结果**: _(训练后填写)_
- 输出目录: `outputs/v10_v4_acl027`(V10 baseline)/ `outputs/solver_v4_acl027`(solver)
- 关键指标: eval_synth/region_rmse↓, boundary_f1↑, eval_real/out_of_band_ratio 平
- 结论:

**涉及文件**: mask_weights.py, losses.py, metrics.py(新), real_eval.py, synth_eval.py(新), dataset.py, config.py, solver_train.py, train.py; tests: test_model_losses / test_dataset / test_config; docs/REMOTE_ORDERS.md

---

### [ACL-026] 2026-06-25 — solver 架构修正:end-on-DC + 冻结 eta(把"硬 DC"真正做硬;软锚定降级为监视)

> 接 ACL-025。v2(anneal 8000 + dc_weight 0.5)训练复盘 → 定位到 `unroll.py` 实现的结构缺陷,做架构级修正,而非再调权重。

**问题诊断**(v2 远端 log):
- DC 从平滑暖启地板 **0.021(step1)单调爬到 0.061(step2000)**,冲向 v1 同一高位;`eta` 0.500→0.477 继续下漏;`gnorm` 随 anneal 上升(0.17→0.52)说明梯度几乎全来自 struct、DC 项梯度可忽略。
- **DC 升到暖启地板之上 = 幻觉签名**:平滑暖启过 A 再 highpass,残差 ≈ −highpass(burst)(地板);网络加的高频与 burst 不相关时方差叠加,DC 升过地板 → 加的是**物理不符**的高频。struct(宽容、亚像素容忍)下降的同时 DC(严格、相位敏感)上升 = 网络找了个"GT 神似但前向不符"的解。
- **根因(unroll 实现缺陷,非加权)**:① 每轮 `DC→prox` 顺序,**prox 末位发言**,无约束残差 UNet 在 DC 步之后重新注入幻觉高频,输出 `x_K` 不被任何 DC 步修正;② **eta 可学**,优化器把 DC 步往 0 调、架构层面绕过约束;③ 软 DC loss 项本质是已证伪的 loss 侧软锚定(ACL-017/019),且 `A^T(highpass)` 梯度天生被转置上采样抹平,打不过 struct 尖锐的边缘/梯度向量梯度(`structure_boost=4`)。三者叠加 → 再调 dc_weight/anneal 都是给漏底设计打补丁。

**修改内容**(`unroll.py` / `config.py` / `solver_train.py`):
1. **end-on-DC 重排**:每轮改 `prox → DC`,循环以 DC 步收尾 → 输出 `x_K` **构造上落在数据一致方向**,prox 不再有最后发言权;struct loss 经终末 DC 步反传进 prox(autograd 双反传),把 prox 耦合到观测。
2. **冻结 eta**(`solver_learn_eta=False` 默认):eta 改为 buffer、不进 optimizer,DC 步每次满强度开火、不可被调 0 绕过;新增 `--solver-learn-eta` 可切回可学(A/B 用)。
3. **软 DC loss 降级为可选监视**:`solver_train` 当 `dc_weight=0` 时在 `no_grad` 下算 `dc` 仅作监视(TB `loss/dc` 仍可看),不建图、不进 loss;`>0` 时仅作弱次级正则,不再是主机制。
4. docstring 更正:删除原"prox 只能填零空间、结构上无法覆盖观测"的过度声称(实现并不成立),改为诚实描述 end-on-DC 的保证 + "单步梯度非完整投影"的 caveat。

**预期效果 / 验证次序**:
- 终末 DC 步把输出拉回数据一致,DC 不再升过暖启地板;`eta` 恒定(TB 一条平线=冻结生效)。
- 推荐下一轮用**纯架构**配置:`--solver-prior-anneal-steps 0 --solver-dc-weight 0`(total=struct,一致性全交给架构),盯 `loss/dc` 监视曲线是否被架构压住。
- **先 Gate B**(单 clean 场景,确认 end-on-DC + 冻结 eta 把 DC 残差驱→~0,即架构有效)PASS 再上池子。
- **判据**:纯架构下 DC 仍压不住 → 单步梯度太弱,升级为终末几步 **CG 硬投影**(docstring 已标 caveat 路径);DC 被压住但 struct 学不动 → 回调。

**训练结果**:_(待远端;Gate B 验证架构 → 池子重启 `outputs/solver_v3_arch`,纯架构 anneal0/dcw0)_。

---

### [ACL-025] 2026-06-25 — solver v1 首训诊断:DC 项被欠权重无视(anneal-off + dc_weight 0.1)→ 上 prior 退火 + 抬 DC 权重

> 本条是 unrolled solver(ACL-024)**首个 20K 训练**的中途诊断 + 训练策略修正。仅 CLI 旗标变更,无代码改动。

**问题诊断**(远端 5K log;配置 no-drizzle / K=4 / M=12 / patch192 / batch18):
- `struct` 正常下降:0.081(step1500)→ ~0.04(最低 0.023@4500)。监督路通。
- **`loss/dc` 卡死在 ~0.055–0.064,不降反微升**;且 `dc(0.057) > struct(0.042)`——DC 是个**绝对量级比 struct 还大、却被无视**的量,不是可忽略噪声。
- **`eta`(可学 DC 步长)单调下漏**:0.484 → 0.456(step1500→5000),方向是优化器在"关小"物理修正步(未塌到 0)。
- **机制**:`total = 1.0*struct + solver_dc_weight(0.1)*dc`。AdamW 对 loss 整体缩放近似不变,真正起作用的是**权重比 struct:dc = 10:1**,struct 从 step0 满权主导 → 优化器榨干 struct、无视 DC,并顺势调小 eta。
- **排除算子 bug**:Gate 0(hp-corr 0.999)/ Gate A(adjoint 0.0)/ **Gate B 在单 clean 场景把 DC 压到 ~0** 已认证算子+几何+数据;机器有能力降 DC。故此为**训练加权动力学**,非 FM-6 系统误差。DC 漂移=求解器在绕开物理锚点,退化成"贵的 plain UNet",违背架构反幻觉初衷。

**修改内容**(无代码改动,现成旗标;**从头重启**,因 anneal 是 step0 起的 schedule,续训会错过 DC 主导早期相):
1. `--solver-prior-anneal-steps 8000`:`total = anneal*struct + 0.5*dc`,早期 anneal≈0 → **只剩 DC** 先压数据一致性,prior 在 0–8K 线性 ramp 填零空间(正是为 FM-1 cliff 设计、ACL-024 第一条基线刻意先关的开关)。
2. `--solver-dc-weight 0.1 → 0.5`:稳态比 10:1 → **2:1**(DC 占 33%),保证 prior ramp 满后 DC 仍是硬锚不反弹。
3. **隔离变量**:patch192/batch18/M12/unroll4/clip 全部保持与 v1 一致;v1 5K ckpt 保留作对照。`gnorm` 持续 3–4 顶 clip=1.0 暂不动(留作下一变量)。

**预期效果**:
- DC 主导期(0–8K)`loss/dc` 明显下降(目标 0.057 → <0.03),`eta` 止跌回升。
- prior ramp 满(8K 后)DC 维持低位不反弹。
- **判据**:若 8K 后 DC 又爬 → 下轮 `dc_weight`→1.0;若连 0–8K 都压不动 DC → 停,提示更深问题(非纯加权)。

**训练结果**:_(待远端;重启为 `outputs/solver_v2_dcanneal`,盯 0–8K 的 dc/eta 曲线)_。

---

### [ACL-024] 2026-06-25 — 决策记录:不上 diffusion / 不用现成底子,承诺 unrolled solver(roadmap 落盘)

> 本条是**决策记录(ADR)**,非代码变更。完整 roadmap 见 `research_log/network_upgrade_roadmap.md`。

**决策点**:v3 数据(5K)生成完后,是否把 U-Net 升级为 diffusion / flow matching?是否拿别人训好的超分/扩散模型(Real-ESRGAN / SwinIR / StableSR / SD)当底子在我们数据上微调?远端 5090 有 32G。

**决策**:
1. **不**把主干换成 diffusion/flow 当**主**架构;**不**用现成 RGB 底子微调。
2. **承诺** physics-constrained **unrolled solver**(确定性)+ band-limited 监督(沿用 memory `thermal-lift-redesign-direction` 已定方向)。
3. 生成模型**只在最后**作 unrolled solver 里的 plug-in **不确定度先验**(DPS/ΠGDM 式后验采样)考虑,不替代主干。

**理由(grounded)**:① 计量要 data-consistent 恢复、不要生成式幻觉;② 我们 ACL-023 的 band-limited 原则本就是确定性恢复框架,diffusion 的多峰优势只在我们不追的 band 外;③ 现成 RGB 底子域差大、latent-diffusion 的 VAE 恰好毁掉要恢复的高频;④ 5000 scene 偏向数据高效的 solver,不偏向数据饥饿的 diffusion;⑤ **经验证据**:loss-side forward 锚定已被证伪(`losses.py:299` `forward_model_weight=0`,ACL-017/019)→ 下一步是把同一算子升级成**硬约束(unrolling)**;⑥ 5090/32G 应"把对的东西做大"(更多 unroll 迭代/更深 prox/更狠 randomization),不够从头训高分扩散。

**落盘的实现次序**(详见 roadmap):Step 0 远端跑 5K 生成 → Step 1 torch shift-aware 前向 `A_i` + autograd 转置(用 numpy `ObservationOperator` 做 adjoint dot-product 验证)→ Step 2 K 步 unroll(DC 步 + 现有 U-Net 当 prox,drizzle 暖启;V10 已是 1 步 unroll)→ Step 3 band-aware loss + 标定 σ → Step 4 eval(EP15 FRC,及格线=band 内打赢经典 TGV/MAP-TV);前置 Step 5 远端重跑 EP15 定 20µm 权威频带。

**load-bearing 约束**:① shift 精度是头号风险 → 训练加 shift-jitter randomization;② torch `A_iᵀ` 必须复刻 +0.499 HR-px block-center 偏移(self-check T1);③ 用标定 σ=0.2257 LR-px(T5),不是占位 0.5;④ EP15 未在 20µm 重跑,band 数字在此前不可信。

**硬规矩**:一次只动一个变量(先 solver 打赢经典,再谈生成先验);band gate 一切。

**涉及文件**:新增 `research_log/network_upgrade_roadmap.md`;复用 `algos/ep07_unet_sr/src/unet_sr/{losses.py,model.py,train.py}`、`tcforge/src/tcforge/_ep06_reference/forward.py`、`algos/ep15_info_limit/scripts/run_m2_frc.py`。

---

### [ACL-023] 2026-06-25 — 探测器 pitch 重标定(20µm)+ forward 算子认证 + v3 信息保存数据管线

**问题诊断**:
- ep01–ep22 主线结论是 `no GT-certifiable winner`:学习方法从未干净打过经典 TGV/MAP-TV,且训练后期合成先验"反吃"真实细节(V9A 保真悬崖,`hp_corr_input` 0.974→0.906)。两层根因:①输入端 5×1x 统计通道在进网络前坍缩了 248 帧亚像素相位;②训练分布"贴着标定"退化(旋转固定 47.6°±1.5°、复用真实那一组 248 shift、PSF 贴标定),且 GT 信息含量超过信息极限 → 模型被迫在零空间幻觉。
- 标定本身有错:EP03 由 BMP mm 标尺测得的 detector pitch `10 µm/pixel` 是 **2× 误读**(axis 与 contour cross-check 共用同一 BMP 标尺锚点故 lockstep 一致、无内部矛盾)。真实 pitch = **20 µm/pixel**,阵列仍 480×640(原始 TXT 实测确认)。系统从"2× 过采样"修正为"**临界采样**(分辨率≈pitch=20µm)",2× SR 的真实含义变成 **20µm→10µm 目标分辨率** —— 直接解释"2X 信息很少 / 放弃 4X"。

**修改内容**:
1. **Pitch 重标定(commit `1ae3177`,26 文件)**:configs(stage_calibration/coordinate_set/synthetic×7)、core 默认值(ep03/ep04/ep07_cache/displacement)、docs(dataset_description/AGENTS/fresh_start_guide)全部 10→20µm;派生数重算(FOV 6.4×4.8→12.8×9.6mm、40µm 命令位移 4.0→2.0px、PSF σ µm 2.26→4.5µm,σ in LR px=0.226 不变);EP03 pitch 标 SUPERSEDED。DO-NOT-TOUCH:对齐 CSV、PSF σ(px)、scale/canvas、paper/(FRC µm 数值待远端 EP15 重跑)。
2. **forward 算子认证(commit `14110b5`)**:新增 `scripts/forward_roundtrip_selfcheck.py`,5 项全 PASS —— 约定无 dx/dy 交换(存在恒定 +0.5 HR-px block-center 偏移,求解器 adjoint 必须复刻)、in-band 可逆(shift-and-add/drizzle vs PSF-blurred GT corr 0.997)、box 采样物理性混叠 ~3%(realistic,需建模)、360° 旋转在内切圆内质量守恒、FRC 频带代理。**证伪了"原始生成代码本身是错的"这一担忧**。
3. **v3 信息保存 2× 数据管线(commit `14110b5`)**:tcforge 新增 `geometry.inscribe_disc`(360° 全随机旋转不裁角)、`shifts.random_constellation`/`build_scene_shifts`(每 scene 随机相位星座,good/medium/poor 覆盖,15% real-like 域匹配)、`classical_sr.phase_bin_drizzle`(4 个亚像素相位通道,显式暴露相位);`generate_training_pool.py` 改为每 scene 随机 N(24–96)+ 均匀 360° + phase-bin + stress SSAA=6 + 裸 f16 burst(**不压缩**,因生成 CPU-bound,zstd 只会加 CPU);config `configs/synthetic/pool_2x_v3.json`(5000 scenes)。tcforge 测试 75 passed。

**预期效果**:
- 输入保相位(burst+shifts+phase-bin)+ 分布全随机化(360°/星座/PSF)打破朝向与单一星座过拟合,GT 信息卡在诚实可恢复频带 → 不再把幻觉规模化。
- 认证过的 forward + 下一步物理约束展开式求解器(硬 data-consistency)把"防幻觉"从 loss 侧(ACL-017/019 已证伪)移到**架构侧**:模型结构上只能在 forward 零空间填先验,无法覆盖观测。

**核心设计原则(本轮确立)**:随机化"干扰参数"放很宽,但 GT 信息含量必须卡在诚实可恢复频带之上;label 是监督信号,只在 band 内追精度,band 外匹配 label = 幻觉 = 重蹈保真悬崖。

**远端就绪 / 数据量 / 耗时**:`data/synthetic/pool_2x_v3_5k` 已 symlink 到 5090 的 `/mnt/d`(1.6T 空闲);实测每 scene ~45–65MB(burst f16 ~53MB@91帧 + phase-bin 9.4MB + obs 1.3MB),5K 估 **~230–330GB**;smoke 6s/scene,因 2× + N~60 比旧 4×/248 帧轻很多,全 5K 预计几十分钟级(以 tqdm ETA 为准)。64 worker(RAM 限)。生成命令由用户在远端手动贴。

**训练结果**:_(本条为标定/数据管线/工具变更,无新增训练)_。数据生成待用户远端启动;下一步 Step 4 物理约束展开式求解器(data-consistency 硬约束 + band-aware loss/eval)。

**遗留 / caveat**:① 真实数据 PSF/对齐重标定仍需远端跑 `data/`(本地已拉 263 帧 txt 备用于此);② contour-alignment refined shifts 不能干净拟合单一刚性 (θ,pitch)(dx/dy 行给出不一致 θ、残差 p95≈0.79px on ~3px signal)—— stage command 仅 prior、对齐噪声/局部,展开式求解器对 shift 精度敏感需留意;③ `hr_mask_4x.png` 是误导性 legacy 文件名(内容实为 2× 960×1280),待清理;④ paper/ 的 FRC µm 数值与 null-space sinc 推导需远端 EP15 重跑后才能诚实更新。

**涉及文件**:commits `1ae3177`(重标定)、`14110b5`(v3 管线 + self-check);新增 `configs/synthetic/pool_2x_v3.json`、`scripts/forward_roundtrip_selfcheck.py`;tcforge `geometry.py`/`shifts.py`/`classical_sr.py`/`storage.py`/`__init__.py`、`scripts/generate_training_pool.py`。

---

### [ACL-022] 2026-06-14 — Task E 论文证据硬化：TGV actual split/FRC + F5b ROI2 + D.7 第二窗

**问题诊断**:
- 统一 harness ACL-021 的 TGV `split_half_nrmse` / `frc_*` 列仍使用 EP16 同子集 drizzle proxy，虽然已在表注声明，但审稿人可质疑 TGV 自身 split/FRC 是否一致。
- F5 主视觉只使用中心梳齿 ROI，容易被质疑为 cherry-pick。
- D.7 零训练融合 baseline 的 λ 在单一 fine-window 上选择，存在 selection-on-test 风险。

**修改内容**:
1. 新增 `algos/ep11_dl_benchmark/scripts/run_tgv_split_frc.py`：CPU-only 编排脚本，复用 EP16 `run_tgv_child.py` 子进程和 EP10 TGV 实现，对 full / split-A / split-B 分别运行各向异性 coverage-weighted TGV；full run 与 `output/ep10_tgv_sr/best_hr_highpass.npy` 做相对 L2 self-check，随后在 actual TGV half-set highpass 图上计算 split NRMSE 和 FRC。
2. 修改 `algos/ep11_dl_benchmark/scripts/run_unified_harness_t1_t2.py`：新增可选 `--tgv-split-json`，若 JSON 成功则优先读取 actual TGV split/FRC；否则默认退回 EP16 drizzle proxy。`--only` 刷新时合并已有 arm rows，避免只刷新 TGV 时截断 T1/T2 全表。
3. 新增 `scripts/paper_figures/fig05b_roi2_holdout.py`：固定分数 ROI2（rows `[0.270,0.415)`, cols `[0.530,0.685)`）生成 F5b 双域视觉图，并在第二 ROI 上重算 `lattice_score`、`sharp_p95`、profile zigzag proxies。
4. 新增 `algos/ep07_unet_sr/scripts/v9_review/run_fusion_window2.py`：固定第二验证窗 rows `542:676`, cols `478:674`，复用缓存全幅预测，对 D.7 融合 baseline 重新选择 λ，并比较 V10 工作点跨窗位置。
5. 更新 `docs/paper/07_experiments.md`、`docs/paper/09_figures_tables_assets.md`、`docs/paper/supp/D_full_results.md`：回填 actual TGV split/FRC、F5b ROI2 结论、D.7 第二窗结果，并保留 proxy / ROI / no-GT caveat。

**预期效果**:
- T1 的 TGV split/FRC 列不再依赖 drizzle proxy，可自洽回应 “TGV 自己的 split/FRC 呢”。
- F5 主视觉从单一中心 ROI 扩展到预声明 held-out ROI，降低 cherry-pick 风险。
- D.7 从单窗 λ 选择升级为双窗稳定性检查，结论限定为局部 proxy-frontier 压力测试，不升级为方法胜负。

**推荐参数**:

```bash
# E1 actual TGV split/FRC, CPU-only
CUDA_VISIBLE_DEVICES= uv run python algos/ep11_dl_benchmark/scripts/run_tgv_split_frc.py \
  --workers 4 --tgv-workers 4

# Refresh only the TGV row while preserving cached rows for other methods
cd algos/ep11_dl_benchmark
CUDA_VISIBLE_DEVICES= uv run python scripts/run_unified_harness_t1_t2.py \
  --only tgv \
  --tgv-split-json ../../output/ep11_unified_harness/tgv_split_frc.json \
  --device cpu --workers 4 --skip-f5

# E2 / E3 CPU-only checks
CUDA_VISIBLE_DEVICES= uv run python scripts/paper_figures/fig05b_roi2_holdout.py
cd algos/ep07_unet_sr
CUDA_VISIBLE_DEVICES= uv run python scripts/v9_review/run_fusion_window2.py
```

**训练结果**: _(本条为评估/论文证据硬化，无新增训练；2026-06-14 回填)_
- E1 TGV actual split/FRC: full-run self-check relative L2 = **0.0** vs EP10 submitted highpass anchor；split A/B 各 124 帧；total runtime 1248 s；TGV child backend status `aniso_forced_fallback`（预期 CPU anisotropic path）。Actual TGV split NRMSE = **0.03164**；FRC@20/16/14/12 µm = **0.978 / 0.975 / 0.969 / 0.955**；`frc_10um` 为 NaN（频带边界缺失），cutoff field = 10.0 µm 且 `crossed=False`，只作 split-consistency proxy。
- Harness TGV 行已刷新：`output/ep11_unified_harness/t1_metrics.csv` 与 `all_arm_metrics.csv` 全 arm success；TGV `split_half_source` / `frc_source` 指向 `output/ep11_unified_harness/tgv_split_frc.json`。
- E2 ROI2: F5b 资产生成到 `output/paper_figures/fig05b_main_visual_roi2.{png,pdf}`；ROI2 `lattice` 排序与中心 ROI 一致（drizzle < TGV < V10 < V9A60），但 `sharp_p95` 与 profile zigzag 排序部分不一致，因此只报告为 held-out visual/proxy audit。
- E3 第二窗: TGV×V9A60 λ 原窗 0.2、第二窗 0.1，λ 本身不完全稳定；第二窗 λ=0.1 为 `hp_corr_input=0.9643`, `hp_corr_tgv=0.9985`, `sharp_p95=0.5082`, `lattice=0.0126`，通过本窗 proxy-frontier gate。V10 λ=1.2@15K 第二窗为 `hp_corr_input=0.9199`, `sharp_p95=0.5008`, `lattice=0.0130`，仍低于本窗 TGV fidelity reference，保持“低 grain / 较锐但保真不足”的 proxy 位置判断。
- 结论: 三项加固均不改变 C1–C4 settle，不支持“学习方法打败 TGV”“更干净/更保真”或物理分辨率声明；所有新增证据限定为 2x contour-level、split-consistency、ROI-level visual/proxy。

**涉及文件**: `algos/ep11_dl_benchmark/scripts/run_tgv_split_frc.py`, `algos/ep11_dl_benchmark/scripts/run_unified_harness_t1_t2.py`, `scripts/paper_figures/fig05b_roi2_holdout.py`, `algos/ep07_unet_sr/scripts/v9_review/run_fusion_window2.py`, `docs/paper/07_experiments.md`, `docs/paper/09_figures_tables_assets.md`, `docs/paper/supp/D_full_results.md`

### [ACL-021] 2026-06-14 — 论文 T1/T2/F5 统一真实数据 harness

**问题诊断**:
- 论文最终 T1/T2 需要单一 EP11/common.metrics 口径，不能把 TensorBoard `eval_real/*` 的 artifact scale 与 EP11 harness scale 混进同一表。
- 旧 EP11 横评脚本只覆盖早期 1x 输入变体，无法安全评估 V9A/V9C hybrid-drizzle 输入和 V10 residual-over-observation；若未透传 `residual_channel=5`，V10 会退化成裸 delta 输出（缓存均值接近 0°C）。
- EP10 TGV 的 `best_hr_temperature.npy` 为 highpass/centered 产物，直接读作温度会得到约 0.24°C，必须经 TGV helper 重建普通 Celsius 温度图后再进入视觉对比。

**修改内容**:
1. 新增 `algos/ep11_dl_benchmark/scripts/run_unified_harness_t1_t2.py`：一次性评估 bicubic / drizzle / MAP-TV / TGV / v6 / v8.1a / v8.1b / v9b / v9d / V9A / V9C / V10-lam120@15K，并额外缓存 V9A-60K 作为 F5 late-drift visual control。
2. 复用 EP06 数据读取和 `common.metrics`、EP10 drizzle、EP15 FRC/zigzag probes、EP07 `infer_from_burst`，避免重新实现核心指标。
3. 对 `input_mode="hybrid_drizzle2x"` 的 arm 使用 `model_scale=1`；对 V10 `residual_mode="drizzle2x"` 自动设置 `residual_channel=HYBRID_DRIZZLE_MEAN_CHANNEL`（ch5），并在输出行记录 full/split cache 的温度均值 sanity check。
4. 统一输出 `all_arm_metrics.csv`、`t1_metrics.csv`、`t2_metrics.csv`、`tb_vs_harness_scale_check.csv`、`run_manifest.json`，并生成 F5 双域视觉图 `fig05_main_visual.{png,pdf}`。
5. 在 manifest 与 source columns 中显式标注边界：MAP-TV 是预计算 5x anchor；TGV split/FRC 列目前复用 EP16 同子集/同 shifts 的 drizzle proxy；F5 是 task-level visual gate，不是保真或分辨率证据。

**预期效果**:
- 论文 T1/T2/F5 使用同一真实数据 harness 与同一 artifact scale，避免历史 TB-scale 数字污染最终横评表。
- hybrid/V10 推理路径有 23°C 温度均值自检，防止 residual base 漏加 bug 复发。
- 风险：TGV split/FRC 尚非独立 TGV split 重算，必须在表注和 manifest 中保留 proxy caveat；MAP-TV 5x 与 2x methods 不可隐式混同。

**推荐参数**:

```bash
cd algos/ep11_dl_benchmark
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_unified_harness_t1_t2.py \
  --device cuda:0 \
  --workers 4 \
  --output-dir ../../output/ep11_unified_harness
```

**训练结果**: _(本条为评估/数据管线变更，无新增训练；2026-06-14 回填)_
- 完整运行: 13/13 methods success，elapsed 166.6 s；GPU0 峰值观测约 385 MB，未占用第二张 GPU。
- 输出目录: `output/ep11_unified_harness/`；F5 资产: `output/paper_figures/fig05_main_visual.png`、`output/paper_figures/fig05_main_visual.pdf`。
- 23°C sanity check: V9A 10K mean 22.366°C，V9C 5K mean 22.985°C，V10 lam120@15K mean 23.288°C，V9A 60K mean 23.307°C；均非 0°C delta 场。
- T1 selected rows（harness scale，artifact↓ / corr↑）: drizzle 1.138 / 0.771；TGV 0.695 / 0.741；v9b@11K 1.766 / 0.777；V9A@10K 1.762 / 0.719；V9C@5K 1.669 / 0.718；V10 lam120@15K 2.726 / 0.711。
- TB-vs-harness scale check: v9b@11K TB artifact 0.3385 vs harness 1.7662；v8.1a@15K 0.3919 vs 1.9429；v6@8K 0.3302 vs 1.7891。该表确认两套 artifact scale 不能混用。
- 结论: 统一 harness 只提供 gate/select 与 task-level visual evidence；不支持“学习方法打败 TGV”“更干净”或“更保真”表述。

**涉及文件**: `algos/ep11_dl_benchmark/scripts/run_unified_harness_t1_t2.py`, `docs/paper/00_status_and_plan.md`, `docs/paper/01_outline.md`, `docs/paper/02_introduction.md`, `docs/paper/05_method.md`, `docs/paper/06b_experimental_setup.md`, `docs/paper/07_experiments.md`, `docs/paper/08_limitations_conclusion.md`, `docs/paper/09_figures_tables_assets.md`, `docs/paper/10_writing_handover.md`, `docs/paper/reframe_c4_claim3.md`, `docs/paper/supp/C_method_details.md`, `docs/paper/supp/D_full_results.md`, `docs/paper/supp/E_reproducibility.md`

---

### [ACL-020] 2026-06-12 — V10: residual-over-observation 参数化 + 残差幅度惩罚

**问题诊断**:
- V9A 证明 hybrid 2x drizzle 输入能让中心细 zigzag 的真实相位信息进入网络，但 30K 左右出现保真悬崖：`hp_corr_input` 从约 0.974 跌到约 0.906，并在后期平台上被合成结构先验主导。
- V9A 整条训练时间轴的 fidelity-sharpness 前沿被 EP10 TGV 工作点支配：TGV 约为 `hp_corr_input=0.960, sharp_p95=0.96`；UNet 20K 保真高但偏软，30K+ 锐度高但主要来自观测去相关的过冲/幻觉。
- V9B/V9D 表明 loss 侧 1x forward consistency 难以约束漂移方向；V10 需要把“保持观测”变成输出参数化的零成本默认，而不是训练后期靠隐式 checkpoint 早停。

**修改内容**:
1. `algos/ep07_unet_sr/src/unet_sr/config.py`: 新增 `residual_mode: str = "none"` 与 `residual_penalty_weight: float = 0.0`，CLI 对应 `--residual-mode {none,drizzle2x}` 和 `--residual-penalty-weight`。
2. `config.py`: 校验 `residual_mode="drizzle2x"` 仅允许与 `input_mode="hybrid_drizzle2x"`、`scale=2`、旧 `--residual` 关闭、`forward_model_weight=0` 同用，保证 V10 是单因子 residual-over-observation 实验。
3. `algos/ep07_unet_sr/src/unet_sr/train.py`: V10 路径中模型输出解释为 `delta`，训练预测为 `pred = obs[:, 5:6] + delta`，其中 ch5 是 hybrid drizzle mean @2x；loss 中新增 `residual_penalty_weight * mean(abs(delta))`，TensorBoard 记录 `loss/residual_penalty`、`residual/delta_mean`、`residual/delta_std`。
4. `algos/ep07_unet_sr/src/unet_sr/inference.py`: `infer_full_frame` 新增 `residual_channel` 参数，tile 推理时先把模型输出 delta 加回同一 input channel 再 overlap blend；`infer_from_burst` 在 hybrid 路径透传该参数。
5. `algos/ep07_unet_sr/src/unet_sr/real_eval.py`: 从 checkpoint 的 `training_config.residual_mode` 自动选择 residual channel，确保真实数据 eval 与训练使用同一 ch5 加法路径。
6. `algos/ep07_unet_sr/tests/`: 增加 config 合法/非法组合、zero-model residual-channel 推理、残差 L1 penalty 单调性、旧 direct-predict 推理路径回归测试。
7. `algos/ep07_unet_sr/scripts/run_v10.md`: 记录 smoke 与全量 lambda sweep 命令，供用户手动启动 GPU 批。

**预期效果**:
- `delta=0` 时输出严格等于观测域最保真的 drizzle mean ch5，让“保留观测”成为模型默认解。
- L1 残差惩罚把 fidelity-sharpness 权衡从训练步数显式转移到 `lambda`，让 V10 可以扫描受控 Pareto 前沿。
- 风险：过大的 `residual_penalty_weight` 会把模型锁死在软的 drizzle 输入；过小则可能退化回 V9A 的后期先验侵蚀。

**推荐参数**:

```bash
cd algos/ep07_unet_sr
CUDA_VISIBLE_DEVICES=<GPU_ID> uv run python -m unet_sr.train \
  --training-pool-dir ../../data/synthetic/training_pool_2x_aa_burst \
  --output-dir outputs/ep07_v10_residual_lam${LAMBDA_TAG} \
  --input-mode hybrid_drizzle2x \
  --residual-mode drizzle2x \
  --residual-penalty-weight <LAMBDA> \
  --scale 2 \
  --batch-size 128 \
  --num-workers 8 \
  --total-steps 25000 \
  --save-every 2500 \
  --log-every 100 \
  --compile \
  --mse-loss-weight 0.3 \
  --highpass-loss-weight 0.8 \
  --structure-boost 2.0 \
  --grad-vector-weight 0.15 \
  --thin-boost 3.0 \
  --gap-boost 2.0
```

**训练结果**: _(2026-06-13 回填；含一处评估 bug 的发现与修正)_

- 输出目录: `outputs/ep07_v10_resid_lam002/`、`outputs/ep07_v10_resid_lam005/`、`outputs/ep07_v10_resid_lam015/`（各 25K，5K/.../25K checkpoint）。
- 代码验证: `cd algos/ep07_unet_sr && CUDA_VISIBLE_DEVICES= uv run pytest -q tests/test_config.py tests/test_inference.py tests/test_model_losses.py tests/test_real_eval.py` → 36 passed, 3 skipped（CUDA AMP 测试因 GPU 不可见被跳过）。

- **⚠️ 实际运行偏离推荐配置（两处混杂）**：
  1. **batch_size=64**（推荐/权威 `run_v10.md` 为 **128**）——重新引入了本实验本应顺带排除的 bs 混杂（见 `docs/next_move_plan.md` §8 caveat）。
  2. **λ 取 0.02/0.05/0.15**（而非 `run_v10.md` 标定的 0.0203/0.0406/0.0609）。但 `run_v10.md` 的标定公式本身有误：它把「未乘 λ 的 `mean(|delta|)` / total ≈ 24.6%」当成 λ=0.05 时的损失占比，实际占比是 `λ·penalty/total`，λ=0.05 时仅 ~1.2%、λ=0.15 时仅 ~3.7%。**残差惩罚在 0.02–0.15 区间对总损失贡献可忽略**，等于没有真正约束 delta 幅值——这解释了为何三档 λ 结果几乎重合。

- **🔴 评估 bug（已修复）**：fine-window Pareto 评估脚本 `scripts/v9_review/common.py::infer_checkpoint_cached` 调 `infer_from_burst` 时**未传 `residual_channel`**（只传了旧 `residual` 旗标，V10 下为 False），导致 V10 缓存里存的是**裸 delta**（mean ≈ -0.008°C 残差场），而非 `drizzle_mean(ch5) + delta` 的真实输出。`real_eval.py` 的漂移曲线路径无此问题（line 219-220 已读 `residual_mode` 并传 `residual_channel`）。
  - 错误判定（基于裸 delta）：hp_corr_input≈0.46、lattice≈0.08 → 曾被记为「Claim 4 灾难性失败、超标 16×」。**此判定作废。**
  - 修复：`common.py` 现从 `cfg["residual_mode"]` 派生 `residual_channel=5` 并透传；单 checkpoint `--force` 重推理验证通过（`v10_lam005_25k` 修复后 0.8804，与后处理 `delta+base` 的 0.880 逐行一致）。
  - 修正版产物：`output/ep07_v9_review/v10_pareto/{v9a_pareto_metrics.csv, v9a_pareto_scatter.png, v9a_checkpoint_strip.png}`（15 checkpoint 全部重算）。

- **关键指标（修正后；中心细线窗口，hp_corr_input=保真↑，sharp_p95=锐度↑，lattice↓）**：

  | 对象 | hp_corr_input | sharp_p95 | lattice | hp_corr_tgv |
  |---|---|---|---|---|
  | drizzle 输入（观测域上限） | 1.000 | 0.503 | 0.0015 | 0.960 |
  | **EP10 TGV（经典参照）** | **0.960** | **0.959** | 0.0169 | 1.000 |
  | V10 λ=0.02 @5K（最保真点） | 0.908 | 1.190 | 0.0170 | 0.915 |
  | V10 λ=0.05 @25K | 0.880 | 1.367 | 0.0236 | 0.892 |
  | V10 三档 @25K 区间 | 0.880–0.884 | 1.30–1.37 | 0.0218–0.0243 | 0.892–0.895 |

- **视觉/读数**：修正后 V10 落在 **V9A 后期同一区域**（V9A 40–60K：hp_in≈0.906、sharp≈1.2、lattice≈0.015）——比 TGV 锐（sharp 1.2–1.37 > 0.96），但保真不及 TGV（hp_in 0.88–0.91 < 0.96），lattice 与 TGV 同量级。**V10 不支配 TGV**，是「用保真换锐度」的 V9A 式折中点，不是干净的伪影/幻觉灾难。

- **结论**：
  1. 残差参数化 + 当前 λ 区间（0.02–0.15）**未能把输出绑在 drizzle base 附近**（惩罚损失占比仅 ~1–4%），模型仍漂到 V9A 的 fidelity≈0.88 折中区 → **Claim 4 不是正结果**（未支配 TGV）。
  2. 但本轮**也不能作为 Claim 4 的干净反证**：λ 区间太弱、未探到高保真端，且 bs=64 混杂未消除。要把 Claim 4 做成「即使显式残差控制也无法越过经典前沿」的铁案，需补**高-λ（bs=128、25K）扫描**把 fidelity↑/sharp↓ 的折中曲线探完整（见 `algos/ep07_unet_sr/scripts/run_v10_highlam.md`）。**✅ 已于 2026-06-14 完成，结论见下「高-λ sweep 结果」。**
  3. 当前「学习能微微越过 TGV」的最干净证据仍是**零训练 fusion baseline**（`TGV + 0.1·V9A60 delta` 支配 TGV），而非 V10。

---

#### 🆕 高-λ sweep 结果（bs=128 / patch=192 / 25K × 4 个变体；2026-06-14 回填）

> 闭环上文 conclusion 第 2 点：高-λ（λ∈{0.2,0.5,1.2,3.0}）扫描已完成，bs=64 混杂已消除（统一 bs=128），评估口径自检通过。**结论从「凌乱负结果」升级为「干净的可控权衡 + 有价值工作点」**，但仍**不写成「打败 TGV」**（保真全程 < TGV）。

- **运行**: 4 个变体 λ∈{0.2,0.5,1.2,3.0}，bs=128 / patch=192（3090 OOM 从 256 降，全变体统一记 caveat）/ 25K / `--save-every 2500`；每个变体 ~4.3–5.3 h，双 GPU 两两并行，wall ≈ 12 h。输出 `outputs/ep07_v10_resid_hl_lam{020,050,120,300}/`（各 10 个 checkpoint + `model_final.pt`）。
- **残差自检通过**: `output/ep07_v9_review/cache/v10hl_*_temperature.npy` 20 个文件均值全部 ≈ **23.29°C**（不是 ≈0）→ 本条主记录里的「漏加 base」评估 bug **未复现**，本轮 fine-window 数字可信。
- **fine-window Pareto（修复后 harness；hp_corr_input=保真↑，sharp_p95=锐度↑但不可单用，lattice=grain/HF↓）**:

  | 对象 | hp_corr_input | sharp_p95 | lattice |
  |---|---|---|---|
  | drizzle（观测软上限） | 1.000 | 0.503 | 0.0015 |
  | **EP10 TGV（经典基准）** | **0.960** | **0.959** | **0.0169** |
  | λ=0.2 @5K→25K | 0.915→0.882 | 1.047→1.264 | 0.0170→0.0242 |
  | λ=0.5 @5K→25K | 0.918→0.886 | 1.039→1.224 | 0.0155→0.0228 |
  | λ=1.2 @5K | 0.941 | 0.891 | 0.0098 |
  | **⭐ λ=1.2 @15K（最佳折中）** | **0.922** | **0.987** | **0.0141** |
  | λ=1.2 @25K | 0.904 | 1.090 | 0.0180 |
  | λ=3.0 @5K（塌回 drizzle） | 1.000 | 0.510 | 0.0015 |
  | λ=3.0 @10K | 0.956 | 0.801 | 0.0080 |
  | λ=3.0 @25K | 0.934 | 0.931 | 0.0115 |

- **达标 checkpoint**: 7 个满足 `hp_corr_input≥0.92 ∧ lattice≤0.0169`（λ=1.2 的 5K/10K/15K + λ=3.0 的 10K/15K/20K/25K；已排除 λ=3.0@5K 这个 `sharp_p95=0.51` 塌回 drizzle 的退化点）。
- **最佳折中点 = λ=1.2 @15K**: (hp_corr_input, sharp_p95, lattice) = **(0.922, 0.987, 0.0141)** —— 锐度 ≈ TGV(+3%)、grain 比 TGV 低 17%、保真 0.922 刚过门控（仍 < TGV 0.960）。
- **TB-scale 漂移端点（eval_real，artifact↓/corr↑；与 fine-window 是不同口径，绝不混表）**: λ 越大漂移越小、corr 越稳——λ=0.2: 0.70/0.67→0.73/0.658；λ=1.2: 0.34/0.726→0.64/0.695；λ=3.0: **0.29/0.721→0.59/0.722（corr 基本不掉）**。
- **结论（更新）**:
  1. 残差约束把 *fidelity–sharpness–grain* 三维折中变成**可调的 λ 旋钮**：大 λ（3.0）能同时拿高保真+低 grain（牺牲锐度趋向 drizzle）；λ=1.2@15K 拿到「锐而不 grain、保真刚过门控」的折中 → **存在「有价值工作点」**（`docs/paper/reframe_c4_claim3.md` §7 判据满足）。
  2. **但所有 checkpoint 保真仍 < TGV(0.960)**（最佳点 0.922）；维持 reframe 诚实裁决「no GT-certifiable winner」，**不写成「打败 TGV」**；报锐度必并报 lattice + 视觉。
  3. Phase 2 精化（λ=1.2 二分 / 第二 seed）**已决定跳过**（非支配关系，预算转给统一口径 harness T1/T2 重跑）。
- **产物**: `output/ep07_v9_review/v10_highlam/{v9a_pareto_metrics.csv, v9a_pareto_scatter.png, v9a_checkpoint_strip.png}`、`output/ep07_v9_review/ep07_eval_real_metrics.csv`（含 V10 四个变体 + V9C/V9D 漂移）。

**涉及文件**: `algos/ep07_unet_sr/src/unet_sr/config.py`, `algos/ep07_unet_sr/src/unet_sr/dataset.py`, `algos/ep07_unet_sr/src/unet_sr/train.py`, `algos/ep07_unet_sr/src/unet_sr/inference.py`, `algos/ep07_unet_sr/src/unet_sr/real_eval.py`, `algos/ep07_unet_sr/scripts/v9_review/common.py`（评估 bug 修复）, `algos/ep07_unet_sr/tests/test_config.py`, `algos/ep07_unet_sr/tests/test_inference.py`, `algos/ep07_unet_sr/tests/test_model_losses.py`, `algos/ep07_unet_sr/scripts/run_v10.md`

---

### [ACL-019] 2026-06-11 — V9C: hybrid 输入下用合法 1x lr_obs 启用 forward consistency

**问题诊断**:
- ACL-016 为 V9A 引入 `hybrid_drizzle2x` 后，模型输入的第 0 通道变成了 `aligned_mean` 的 2x 上采结果，不再是 forward consistency 所需的合法 1x LR 观测。因此 ACL-016 在 config 中临时禁止 `hybrid_drizzle2x + forward_model_weight>0`。
- V9B 证明 highpass-band forward consistency 在旧 1x 输入上无法压平真实数据漂移，但仍不能回答「hybrid 输入保留 2x 相位信息后，合法 1x 观测锚是否变得有效」。V9C 需要在 hybrid 输入下给 loss 单独提供原始 1x `aligned_mean` patch。
- 风险点参考 ACL-014：crop origin 若不按完整倍率对齐，会让 1x/2x/target patch 出现半个 LR 像素错位；flip/rot90 若未同步到辅助观测，会让 forward 项使用错位 anchor。

**修改内容**:
1. `algos/ep07_unet_sr/src/unet_sr/dataset.py`: hybrid 路径缓存原始 1x `obs_features`，在 5ch 1x obs 上采为 2x 输入前保留 `obs[0:1]`；`__getitem__` 对 hybrid sample 额外返回 `lr_obs`，shape `(1, patch_size_hr//2, patch_size_hr//2)`。
2. `dataset.py`: hybrid crop origin 在 2x 输入网格上强制偶数对齐，`lr_obs` 用 `(y//2, x//2)` 和半边长裁剪；`_augment` 扩展为同步变换 `lr_obs`，确保 flip/rot90 后观测锚与 target/pred 几何一致。
3. `algos/ep07_unet_sr/src/unet_sr/losses.py`: `ContourSRLoss.forward` 新增可选 `lr_obs` 参数；当 `lr_obs is not None` 时，forward-model 项以 `lr_obs` 为观测参照，并显式使用 downsample scale=2，不复用 hybrid 模型的 scale=1。旧 `lr_observation` 路径保持兼容。
4. `algos/ep07_unet_sr/src/unet_sr/train.py`: batch 含 `lr_obs` 且 `forward_model_weight>0` 时传入 loss；旧 `input_mode="lr"` 仍回退到 `obs[:, 0:1]`。
5. `algos/ep07_unet_sr/src/unet_sr/config.py`: 解除 `hybrid_drizzle2x + forward_model_weight>0` 的禁令；改为要求该组合显式使用 `--scale 2`，由 dataset/train 走 `lr_obs` 路径。
6. 测试覆盖 `lr_obs` shape/crop 对应关系、增广同步、偶数 origin、hybrid+AMP loss 有限、config 新校验，以及旧 LR 模式不返回 `lr_obs` 的回归。
7. `algos/ep07_unet_sr/scripts/run_v9.md`: 补 V9C/V9D smoke/full 命令，并说明 V9C 的合法 1x anchor 与 hybrid 输入第 0 通道不同。

**预期效果**:
- V9C 可以在保留 hybrid 2x drizzle 输入相位信息的同时启用 highpass forward consistency，避免把上采 mean 当作物理观测导致非法锚定。
- 如果 V9C 能压平后期 artifact/corr 漂移，说明 forward 锚在 hybrid 输入下开始可见漂移方向；若仍失败，则支持「loss-side anchor 仍不足」的结论。
- 风险：forward 项仍只约束 1x 可见频段，高频 hallucination 仍可能落在 forward operator 零空间；若出现 ACL-005 式震荡，先降 `forward_model_weight` 到 0.05。

**推荐参数**:

```bash
cd algos/ep07_unet_sr
CUDA_VISIBLE_DEVICES=1 uv run python -m unet_sr.train \
  --training-pool-dir ../../data/synthetic/training_pool_2x_aa_burst \
  --output-dir outputs/ep07_v9c_hybrid_legal_fwd \
  --input-mode hybrid_drizzle2x \
  --scale 2 \
  --batch-size 128 \
  --num-workers 8 \
  --total-steps 60000 \
  --save-every 5000 \
  --log-every 100 \
  --compile \
  --mse-loss-weight 0.3 \
  --highpass-loss-weight 0.8 \
  --structure-boost 2.0 \
  --grad-vector-weight 0.15 \
  --thin-boost 3.0 \
  --gap-boost 2.0 \
  --forward-model-weight 0.1 \
  --forward-model-band highpass
```

**训练结果**: _(2026-06-14 回填；60K 完成)_
- 输出目录: `outputs/ep07_v9c_hybrid_legal_fwd`（60K + `model_final.pt`）
- 代码验证: `cd algos/ep07_unet_sr && uv run pytest -q` → 47 passed。
- 关键指标（TB-scale `eval_real`，artifact↓ / raw_control_corr↑）: 0.516/0.714 @10K → **0.695/0.669 @60K**，与 1x 锚定变体端点（v9b 0.655/0.688、v9d 0.677/0.677、v8.1a 0.643/0.689）落到同一 ≈0.65–0.70 / ≈0.67–0.69 平台。
- 结论: **合法 1x 观测锚在 hybrid 输入下同样无法压平后期漂移**。这驳倒了「之前锚定失败只是因为 hybrid 第 0 通道不是合法 1x 观测」的反对意见——即使给 loss 单独喂合法 1x `aligned_mean` patch，漂移曲线仍与无锚/带限/全频锚定变体几乎重合。**与 V9B/V9D 合并：loss 侧 forward 锚定路线（band / full / legal × hybrid 全变体）正式、彻底关闭**；漂移是先验驱动、零空间驻留，只能从输入端（V9A hybrid 输入）或输出参数化（V10 residual）侧解决。落 `docs/paper/07_experiments.md` §6.2/§6.3 input×anchor 矩阵。

**涉及文件**: `algos/ep07_unet_sr/src/unet_sr/dataset.py`, `algos/ep07_unet_sr/src/unet_sr/losses.py`, `algos/ep07_unet_sr/src/unet_sr/train.py`, `algos/ep07_unet_sr/src/unet_sr/config.py`, `algos/ep07_unet_sr/tests/test_dataset.py`, `algos/ep07_unet_sr/tests/test_model_losses.py`, `algos/ep07_unet_sr/tests/test_config.py`, `algos/ep07_unet_sr/scripts/run_v9.md`

---

### [ACL-018] 2026-06-11 — V9A 数据管线：池侧预计算 drizzle 变体，修复首 batch 卡死 + 主机 OOM

**问题诊断**:
- ACL-016 的 hybrid_drizzle2x 把 drizzle 留在 DataLoader worker 内现场计算（ACL-016 已预警「DataLoader 吞吐可能成为瓶颈」），实测后果远超预期：
  - `drizzle_features`（248 帧全幅 480×640 → 960×1280 scatter）实测 **2.7 s/scene**，且每 scene 每 epoch 重算一次；
  - `_load_cached` 把 lr_burst cast 成 float32（**305 MB/scene**）整条塞进 LRU 缓存，16 scene/worker × 8 workers ≈ 47 GB（机器 60 GB）；
  - 复现实验（bs=64 / 8 workers / prefetch=4）：**首 batch 162.6 s**，loader 进程树 RSS 37.8 GB、swap 8.5 GB 打满，batch 8 时 worker 被系统 OOM killer 杀掉。叠加 `--compile` 首步编译，表现为「第一步都跑不动」。

**修改内容**:
1. `scripts/precompute_drizzle_variants.py`（新增）: 池侧离线预计算。每 scene 生成 K=4 个 drizzle 变体存 `drizzle_variants_2x.npy`（(K,3,960,1280) float16，~30 MB/scene）。variant 0 = 全帧无噪声（canonical，与推理口径一致）；variant 1–3 = 随机抽 60–100% 帧 + shifts 加 σ=0.05 px 噪声，与原 `_select_burst` 增广分布一致。ProcessPool 并行、原子写、可断点续跑。
2. `tcforge/storage.py`: `load_scene_compact` 发现 `drizzle_variants_{scale}x.npy` 时以 `mmap_mode="r"` 挂载到 `"drizzle_variants"` key。
3. `dataset.py`: hybrid 模式优先用预计算变体——每 (seed, epoch, scene) 确定性抽 1 个变体（mmap 切片，不全量物化）；增广从「连续随机」降为「K 选 1 离散随机」（叠加随机裁剪/翻转/旋转仍在）。无变体文件时 fallback 到现场 drizzle，但 lr_burst 不再 cast float32，直接持有 float16 mmap，`_select_burst` 只物化抽中的子集。`obs_up`（5ch 1x↑2x）缓存复用，epoch 重建只做变体切片 + concat。
4. 测试: 变体优先于 burst、无 lr_burst 也可跑、变体选择可复现、fallback 缓存保持 float16 mmap；原 burst 路径 5 个测试不变全过。

**预期效果**（已实测验证）:
- 冷加载 2.8 s → **0.52 s**/scene；epoch 重建 2.7 s → **0.006 s**；缓存 RAM ~370 MB → ~100 MB/scene（burst 不再驻留内存）。
- variant 0 与全帧 drizzle 最大误差 0.0118（float16 量化步长内），变体间 mean abs diff ~0.01（增广有效）。
- 风险：增广多样性从连续降为 K=4 离散；若过拟合迹象明显可加 `--num-variants` 重新预计算。

**推荐参数**: 训练前先跑 `uv run python scripts/precompute_drizzle_variants.py --pool-dir data/synthetic/training_pool_2x_aa_burst --num-variants 4 --workers 14`（~25 min，磁盘 +59 GB）；训练 CLI 与 ACL-016 一致无变化。

**训练结果**: _(2026-06-12 回填)_
- 输出目录: `outputs/ep07_v9a_hybrid_drizzle`
- 管线表现: 预计算变体方案有效——V9A 全程无首 batch 卡死、无 OOM kill，~49 min/5K steps 稳定推进至 60K（中途一次人为中断，35K checkpoint 续跑）。V9C 复用同一管线同样稳定。
- 算法结果见 ACL-016 回填；本条仅覆盖数据管线。

**涉及文件**: `scripts/precompute_drizzle_variants.py`, `tcforge/src/tcforge/storage.py`, `algos/ep07_unet_sr/src/unet_sr/dataset.py`, `algos/ep07_unet_sr/tests/test_dataset.py`, `algos/ep07_unet_sr/scripts/run_v9.md`

---

### [ACL-017] 2026-06-11 — V9B: highpass-band forward consistency loss

**问题诊断**:
- v8.1a conservative loss 把 `forward_model_weight` 降到 0 后，loss 中没有任何一项把输出锚定到观测；synthetic loss 收敛后，网络在结构 loss 先验驱动下持续把边画亮、对比拉大（real_eval artifact 0.39→0.64，corr 0.76→0.69 单调漂移）。
- 全频 forward model 的梯度经过 PSF 低通对所有频段施加「拉回模糊解」的力，与 highpass/grad_vector 在同一像素上方向冲突（v6 ACL-005 3K 步震荡平台的根因）。

**修改内容**:
1. `losses.py`: `forward_model_loss` 新增 `band: str = "full"` 和 `band_sigma_lr_px: float = 5.0` 参数。`band="highpass"` 时，对 downsampled pred 和 obs 各减去 σ=band_sigma 的高斯模糊后再算 MSE，放过低频段。高斯模糊复用 ACL-011 `custom_fwd` 包装。
2. `ContourSRLoss`: 透传 `forward_model_band` / `forward_model_band_sigma`。
3. `config.py`: 新增 `--forward-model-band {full,highpass}`（默认 full）和 `--forward-model-band-sigma`（默认 5.0）CLI 参数。
4. `train.py`: 传递新参数到 ContourSRLoss。
5. 测试: band=full 回归（与旧实现数值一致）、highpass DC 不变性、HF 扰动检测、AMP 有限性。

**预期效果**:
- 带限版只在边缘亮度/宽度所在的高频段施加观测一致性，低频由 synthetic MSE 锚定，两者不打架。
- 在合成域中该项不增加信息量，但改变函数族——惩罚重投影偏离输入观测的解，使迁移到真实数据后抑制 40K+ 漂移。
- 风险：若出现震荡平台或整体变钝，先降 `forward_model_weight` 到 0.05 重跑。

**推荐参数**: `--forward-model-weight 0.1 --forward-model-psf-sigma 0.5 --forward-model-band highpass --forward-model-band-sigma 5.0`（其余与 v8.1a 一致）

**训练结果**: _(2026-06-11 回填)_
- 输出目录: `outputs/ep07_v9b_fwd_consistency`
- 视觉效果: 与 v8.1a 同体感：膨胀/亮边相比 v8 时代明显收敛，膨胀-抑制达到可接受平衡，无 v8.1b 式条纹伪影；斜边残留均匀台阶状锯齿；中心最细 zigzag 线仍模糊（与 v8.1 A/B 不变性一致，输入信息瓶颈所致，非 loss 问题）。
- 关键指标（real_eval 248 帧 zoom3x，与 EP11 同口径）:
  - `artifact_score`: 10K 0.369 → 25K 0.605 → 40K 0.640 → 60K 0.655（对照 v8.1a: 0.390 → 0.551 → 0.627 → 0.643）
  - `raw_control_corr`: 10K 0.758 → 25K 0.711 → 40K 0.697 → 60K 0.688（对照 v8.1a: 0.756 → 0.717 → 0.698 → 0.689）
  - 40K→60K 漂移: artifact +0.0145 / corr −0.0082，与 v8.1a（+0.016 / −0.009）基本重合 → **漂移未压平，run_v9.md 验收标准未达成**
  - `loss/forward_model` 自 10K 起躺平在 0.004–0.009 地板，同期 artifact 持续上爬 → 漂移方向位于 forward 算子（shift→PSF→下采样→带限 highpass）的零空间，观测一致性对该方向不可见
- 结论: 带限 forward consistency（weight 0.1, band=highpass）单因子归因失败，对真实数据漂移无可测影响。结合 v8.1a / v8.1b / v9b 三个变体漂移曲线几乎重合，确认漂移是「合成先验在真实分布上无监督外推」的结构性矛盾，loss 侧旋钮已证伪。后续处置：artifact/corr 降级为 checkpoint 选择器（Pareto + 视觉门控，不默认取 60K）；观测锚定若要有效需让锚可见漂移方向（如 hybrid 输入下以合法 LR 观测构造 forward 项），或从输入端解决（V9A, ACL-016）。
- **V9D 补充** _(2026-06-12 回填，`outputs/ep07_v9d_fwd_fullband`)_: full-band anchor（`--forward-model-band full`，其余同 V9B）60K 跑完，artifact 0.677 / corr 0.677，漂移同样未压平且终点比 V9B（0.655/0.688）更差；1K–28K 阶段 artifact/corr 大幅震荡（如 20K artifact 0.575/corr 0.642 后又回弹），与 ACL-005 全频低通梯度冲突一致。V9B+V9D 合并证据：**loss 侧 forward 锚定路线（无论 band）正式关闭**。

**涉及文件**: `losses.py`, `config.py`, `train.py`, `mask_weights.py`, `scripts/run_v9.md`, `scripts/run_training.md`, `tests/test_model_losses.py`, `tests/test_config.py`

---

### [ACL-016] 2026-06-11 — V9A: hybrid 2x drizzle 输入模式

**问题诊断**:
- v8.1a/b 两个变体中心最细线模糊完全相同，与 loss 温度和 HR head 无关。根因：5 个输入通道全部是 1x 网格统计量（aligned_mean/median/coverage/variance/highpass），248 帧的亚像素相位信息在进网络前已坍缩。
- EP15 M4 证明 2x 网格经典方法能恢复 12 µm 频带信息（bare/MAP-TV split-half FRC 0.575→0.947）——信息在数据里，只是没喂给网络。

**修改内容**:
1. `configs/synthetic/training_pool_2x_burst.json`: 新增训练池配置，`save_lr_burst=true`、`compute_classical_sr=false`。场景几何/物理参数与旧池一致。
2. `dataset.py`: 新增 `input_mode="hybrid_drizzle2x"` 模式。加载 `lr_burst + shifts`；每 (scene, epoch) 随机保留 60-100% 帧（`min_burst_frames=30`）并加 `shift_noise_std_px=0.05` 高斯噪声；调 `drizzle_features(scale=2, kernel="bilinear")` 得 3 通道 @ 960×1280；5ch 1x obs 上采到 2x；拼成 8ch @ 2x。Effective scale=1，同坐标裁剪 patch。
3. `config.py`: 新增 `--input-mode {lr,hybrid_drizzle2x}` CLI 参数。hybrid 自动设 `in_channels=8`。校验禁止 hybrid + `forward_model_weight>0`（obs[:, 0:1] 不是合法 1x LR 观测）。
4. `model.py`: 无改动（`scale=1` + `in_channels=8` 由构造参数覆盖）。
5. `train.py`: 根据 `input_mode` 推导 `model_scale`；传递 `input_mode` 到 dataset。
6. `inference.py`: `infer_from_burst` 新增 hybrid 路径：1x fused↑2x + scatter drizzle@2x → 8ch → tile 推理（scale=1）。
7. `real_eval.py`: 从 `training_config.input_mode` 自动走对应推理路径。
8. 测试: hybrid 样本形状 (8,256,256)、增强同步、burst 子集可复现/下限生效、epoch 切换产生不同 burst、旧路径 `input_mode="lr"` 回归。

**预期效果**:
- 中心最细线：亚像素信息直接进网络，不再依赖合成先验猜测。
- 边缘锯齿：2x drizzle 通道编码边缘亚像素位置，网络不必把边吸附到 1x 网格。
- 风险：drizzle lattice 格纹可能被网络当结构学习；DataLoader 吞吐可能成为瓶颈。

**推荐参数**: `--input-mode hybrid_drizzle2x --training-pool-dir data/synthetic/training_pool_2x_aa_burst`（其余与 v8.1a 一致，`forward_model_weight=0`）

**训练结果**: _(2026-06-12 回填，60K 完成)_
- 输出目录: `outputs/ep07_v9a_hybrid_drizzle`（35K 处中断后以 batch_size=64 续跑至 60K）
- 视觉效果: 中心最细 zigzag「梯子」区呈现明显的训练阶段 Pareto 权衡——10K/20K 时梯子内部条纹部分可分辨（与输入 drizzle 通道、EP10 TGV 一致）；60K 时粗 zigzag 线最锐利、对比最大，但中心梯子重新糊成橙色团块，粗细线交融，与 v8.1a 60K 体感相同。
- 关键指标（real_eval 248 帧, contour_refined）:
  - `artifact_score` / `raw_control_corr`: 10K 0.446/0.719 → 20K 0.514/0.702 → 30K 0.660/0.663 → 60K 0.646/0.669。
	  - **漂移在 30K 后压平甚至轻微回头**（30K→60K artifact −0.014 / corr +0.007），是 v8.1a/v9b/v9d 中唯一不单调恶化的变体；但平台位置 corr 0.669 低于 v8.1a 60K 的 0.689，run_v9.md「corr 上升」验收标准在 60K 不达成。
  - 中心细线窗口 highpass corr（vs TGV | vs 输入 drizzle 通道，诊断脚本 `algos/ep07_unet_sr/scripts/v9_review/`（原 `tmp/v9a_review/`，已迁移））: **10K 0.966/0.973 → 60K 0.935/0.925**，v8.1a 60K 为 0.936/0.926 → hybrid 输入在 10K 时几乎完整透传了中心细线信息，60K 时被合成结构先验抹回 v8.1a 水平。
- 结论: **输入瓶颈假设证实，但训练后期合成先验会重新吃掉输入里的真实细节**。中心细线信息确实存在于 drizzle 输入通道中（ACL-015 推断正确），V9A 早期 checkpoint 能保留它；漂移机制现在精确定位为「结构 loss 先验逐步覆盖观测保真，把真实细纹理当模糊清理掉」。处置：① V9A 最终 checkpoint 不取 60K，在 10K–25K 区间做 Pareto + 视觉联合选优；② 下一个单因子实验方向是结构权重后期退火或 residual-to-drizzle 参数化，而不是更长训练。

**涉及文件**: `configs/synthetic/training_pool_2x_burst.json`, `dataset.py`, `config.py`, `train.py`, `inference.py`, `real_eval.py`, `mask_weights.py`, `scripts/run_v9.md`, `scripts/run_training.md`, `tests/test_dataset.py`, `tests/test_config.py`, `tests/test_inference.py`

---

### [ACL-015] 2026-06-10 — EP07 v8.1 A/B 归因实验：loss 降温 vs PixelShuffle head

**问题诊断**:
- v8 AA 训练池消除了部分二值 target 锯齿，但真实数据 `eval_real` 在 30K 之后出现更醒目的亮边、边缘膨胀和细密 2x 相位网格。
- 当前现象不是单点问题：final HR head 使用 bilinear upsample 后接带 GroupNorm 的 3x3 refine，容易放大 2x 相位纹理；同时 `structure_boost=4`、`grad_vector=0.3`、`thin_boost=6`、`gap_boost=4`、`laplacian=0.1`、全频 `forward_model=0.1` 叠加后，边缘区域 loss 权重过热，鼓励网络把边缘画亮画宽。
- 直接把 PixelShuffle、forward consistency、warmup 和多项权重一起改，会导致下一轮仍无法判断主因。

**修改内容**:
1. 设计两条并行训练线：
   - `V8_1A`: 保留现有 bilinear HR head，只把 loss 降温，用于验证膨胀是否主要来自结构权重过热。
   - `V8_1B`: 使用 PixelShuffle + ICNR + 1 个无归一化 HR residual block，并使用与 A 完全相同的 conservative loss，用于隔离 final upsampler/head 的 2x 相位伪影贡献。
2. `model.py`: 为 `ThermalSRUNet` 增加 `hr_upsampler={bilinear,pixelshuffle}` 和 `hr_res_blocks` 参数；默认保持旧 bilinear 行为，PixelShuffle 分支显式启用。
3. `config.py` / `train.py`: 将 HR upsampler 参数纳入 CLI、config.json 和模型构造，保证训练产物可复现。
4. `scripts/run_training.md`: 写入 V8_1A / V8_1B 推荐命令。
5. 测试覆盖默认 bilinear 输出尺寸、PixelShuffle 输出尺寸和无 BatchNorm 约束。

**预期效果**:
- 若 A 已明显压住亮边/膨胀，说明 loss 过热是主因；若 B 相比 A 进一步减少 2x 网格，说明 PixelShuffle head 对 final 相位纹理有效。
- 若 A/B 都变软，说明 conservative loss 降温过度，下一轮再考虑 highpass-only forward model 或结构权重 warmup。
- 风险: PixelShuffle 分支不能加载旧 bilinear checkpoint；必须通过 config 中的 `hr_upsampler` 重建模型。

**推荐参数**:
- `V8_1A`: `--hr-upsampler bilinear --mse-loss-weight 0.3 --highpass-loss-weight 0.8 --structure-boost 2.0 --grad-vector-weight 0.15 --laplacian-weight 0.0 --forward-model-weight 0.0 --thin-boost 3.0 --gap-boost 2.0`
- `V8_1B`: `--hr-upsampler pixelshuffle --hr-res-blocks 1` 加同一套 V8_1A loss 参数。

**训练结果**: _(2026-06-11 回填)_
- 输出目录: `outputs/ep07_v8_1a_loss_cooldown`, `outputs/ep07_v8_1b_pixelshuffle`
- 视觉效果:
  - **V8_1A (bilinear + conservative loss)**: 10K 中心棋盘纹最重（类似 v8 初期），20K 后除中心外棋盘消失、无膨胀；30–40K 棋盘减小、边框由糊变清晰但开始膨胀；60K 对比度最大、边框最膨胀。中心区域只有最细的几条线仍糊，稍粗的线锐利可辨结构；边缘锯齿相对 1B 有改善。
  - **V8_1B (PixelShuffle + 同一 loss)**: 10K 有晕染但比 v8 初版轻；20K 提亮、对比度增强、边框膨胀；30K 边框收回变清晰；至 60K 对比度持续增强。边缘锯齿未改善，且中等边框之间出现条纹状亮色伪影；中心区域同样模糊。
- 关键指标（synthetic loss 两个变体均在 ~40–45K 收敛后平坦；real_eval 248 帧 zoom3x）:
  - `eval_real/artifact_score`（越小越好）随训练**单调上升**: 1A 0.390(10K)→0.643(60K)，1B 0.413→0.709；1B 全程高于 1A。
  - `eval_real/raw_control_corr`（与 raw bicubic 控制图的 highpass 相关，反映观测保真）**单调下降**: 1A 0.756→0.689，1B 0.747→0.667。对照 EP10 TGV 的 0.916 / artifact 0.695，UNet 输出对真实观测的锚定明显偏弱。
- 结论:
  1. **PixelShuffle head 归因失败**: 1B 未减轻锯齿，反而引入新的条纹伪影且 artifact_score 全程更高 → final upsampler 不是 2x 相位伪影/锯齿主因，后续保留 bilinear head，放弃 PixelShuffle 分支。
  2. **Loss 降温部分有效**: 1A 锯齿改善、早期无膨胀，说明结构权重过热确实贡献了亮边/膨胀；但 40K 后 synthetic loss 已平坦而真实数据上对比度/膨胀仍持续漂移（artifact ↑ / corr ↓），说明**缺失观测一致性约束**（`forward_model_weight=0`）使无约束方向在合成先验驱动下继续漂移。
  3. **中心最细线模糊对两个变体完全不变** → 与 loss 温度和 HR head 无关，指向前端输入信息瓶颈：5 个输入通道全部是 1x 网格统计量，248 帧的亚像素相位信息在进网络前已被坍缩；而 EP10/EP15 证明 2x 网格经典方法可恢复 12 µm 频带信息（FRC 0.575→0.947）。下一步主攻方向为 2x-grid drizzle/classical-SR 输入通道 + 温和 highpass-band forward consistency。

**涉及文件**: `algos/ep07_unet_sr/src/unet_sr/model.py`, `algos/ep07_unet_sr/src/unet_sr/config.py`, `algos/ep07_unet_sr/src/unet_sr/train.py`, `algos/ep11_dl_benchmark/scripts/run_unet_vs_drizzle_2x.py`, `algos/ep07_unet_sr/tests/test_model_losses.py`, `algos/ep07_unet_sr/tests/test_config.py`, `algos/ep07_unet_sr/scripts/run_training.md`

---

### [ACL-014] 2026-06-10 — 修复 EP12 4x 增强后 drizzle/coverage 错位并默认启用 burst augmentation

**问题诊断**:
- EP12 Dataset 在执行 flip/rot90 后，只同步变换了 `obs_features` / `hr_target` / `hr_edge`，但 loss 使用的 `drizzle_mean` / `coverage` 仍来自未增强的 `obs_drz_patch`。这会让 coverage 加权 HF/NLL 和 forward consistency 在大部分增强 patch 上使用错位权重与观测约束。
- `_crop_origin()` 只按 `scale / drizzle_scale = 2` 对齐，导致 `y % 4 == 2` 或 `x % 4 == 2` 时 1x 上下文通道相对 4x target 偏移 2 HR px，即 0.5 LR px。
- EP12 重训若继续默认 `burst_augment=False`，会固定在单一 detector-axis coverage 图上训练，不能覆盖 M1 暴露出的 detector 轴相位覆盖坍缩风险。

**修改内容**:
1. `algos/ep12_4x_sr/src/sr4x/dataset.py`: `drizzle_mean` / `coverage` 改为从增强后的 drizzle 特征通道切片得到；普通路径和 `defer_1x_upsample` 路径均使用同一几何变换。
2. `dataset.py`: `_crop_origin()` 改为按完整 `self.scale` 对齐，保证 drizzle crop 与 1x context crop 都落在整数源像素边界。
3. `algos/ep12_4x_sr/src/sr4x/config.py`: `TrainingConfig.burst_augment` 默认改为 `True`，CLI 改为 `--burst-augment/--no-burst-augment`，保留 legacy fixed-drizzle pool 的关闭入口。
4. 新增 dataset/config 回归测试，分别锁定增强后 loss 辅助通道同步、crop 原点 4x 对齐和 legacy 关闭开关。
5. 更新 EP12 训练文档与 benchmark README，明确 4x 只作为学习型正则化/轮廓定位网格，采纳必须通过 M4 MAP-TV 与 EP07 2x x2up gate。

**预期效果**:
- 消除 EP12 训练中的系统性错位梯度，尤其是 forward consistency 用翻转/旋转后的预测去解释未翻转/未旋转观测的问题。
- 1x 上下文、2x drizzle 和 4x target 在 patch crop 上保持整数网格一致。
- 后续 4x 重训默认启用相位/帧子集扰动，降低对单一覆盖图的过拟合风险。
- 风险: 默认 burst augmentation 要求训练池包含 `lr_burst.npy` 和 `shifts.npy`；旧预计算池必须显式传 `--no-burst-augment`。

**推荐参数**:

```bash
cd algos/ep12_4x_sr
CUDA_VISIBLE_DEVICES=0 uv run python -m sr4x.train \
  --training-pool-dir ../../data/synthetic/training_pool_4x_aa_2000 \
  --output-dir outputs/ep12_hybrid_v2_guarded \
  --scale 4 \
  --drizzle-scale 2 \
  --burst-augment
```

**训练结果**:
- 代码验证: `cd algos/ep12_4x_sr && uv run pytest -q` → 13 passed。
- 真实重训结果: _(EP12 Hybrid v2 训练后填写)_

**涉及文件**: `algos/ep12_4x_sr/src/sr4x/dataset.py`, `algos/ep12_4x_sr/src/sr4x/config.py`, `algos/ep12_4x_sr/tests/test_dataset.py`, `algos/ep12_4x_sr/tests/test_config.py`, `algos/ep12_4x_sr/README.md`, `algos/ep12_4x_sr/run_training.md`, `algos/ep12_4x_sr/scripts/run_training.md`, `research_log/episodes/ep12_4x_benchmark/README.md`

---

### [ACL-013] 2026-06-10 — 4x v8 AA 训练池入口统一与 EP12 soft mask 接入

**问题诊断**:
- 4x 训练池文档同时存在旧 full-frame 入口、旧 same-grid drizzle 预计算入口和新 `generate_training_pool.py` compact 入口，容易误导执行流程。
- 当前 EP12 Hybrid 已改为 `drizzle_scale=2` + PixelShuffle 2x，不再需要离线 `obs_features_4x.npz`；但文档仍把 `build_4x_features.py` 写成训练前必需。
- EP12 Dataset 读取 `hr_mask_4x.png` 时使用 `>0` 二值化，会把 v8 AA soft coverage mask 重新变硬，导致 HR target 丢失抗锯齿覆盖率语义。

**修改内容**:
1. 统一 4x pool 生成入口为 `scripts/generate_training_pool.py`，删除旧入口 `generate_thermal_chip_phantom.py`、旧 smoke 检查脚本、旧 drizzle 预计算脚本及对应测试。
2. 更新 `docs/windows_4x_generation.md`：明确 2000 scenes / 16 workers 命令、metadata 检查字段、当前 EP12 Hybrid 从 `lr_burst.npy + shifts.npy` 按需计算 2x drizzle。
3. `algos/ep12_4x_sr/src/sr4x/dataset.py`：`hr_mask_4x.png` 改为按 `uint8/255.0` 读取 soft coverage，再用 `reconstruct_hr_temperature()` 重建 target。
4. 更新 EP12 README、训练命令文档和测试，使训练池契约变为 `obs_features_1x.npz + lr_burst.npy + shifts.npy + soft hr_mask_4x.png + metadata.json`。
5. 将 EP14 loss-atlas 中旧 same-grid 4x drizzle 文案标记为 legacy，避免与当前 Hybrid 训练路径混淆。

**预期效果**:
- 4x AA 数据生成只保留一个可复现入口，减少 Windows/Linux 迁移和本地大规模生成时的操作歧义。
- EP12 Hybrid 训练真正消费 v8 AA soft target，避免把抗锯齿训练池退化回硬边界 target。
- 风险: Dataset 仍支持旧可选预计算文件的兼容读取；后续若要完全移除 legacy 兼容，需要单独评估历史 checkpoint/notebook 复现需求。

**推荐参数**:

```bash
uv run python scripts/generate_training_pool.py \
  --config configs/synthetic/training_pool_4x.json \
  --output-dir data/synthetic/training_pool_4x_aa_2000 \
  --pool-size 2000 \
  --workers 16
```

**训练结果**: _(EP12 Hybrid 重新训练后填写)_

**涉及文件**: `docs/windows_4x_generation.md`, `algos/ep12_4x_sr/src/sr4x/dataset.py`, `algos/ep12_4x_sr/tests/test_dataset.py`, `algos/ep12_4x_sr/tests/test_train_smoke.py`, `algos/ep12_4x_sr/tests/test_model_losses.py`, `algos/ep12_4x_sr/README.md`, `algos/ep12_4x_sr/run_training.md`, `algos/ep12_4x_sr/scripts/run_training.md`, `scripts/generate_thermal_chip_phantom.py`, `scripts/smoke_test_thermal_chip_phantom.py`, `scripts/build_4x_features.py`, `scripts/precompute_drizzle_2x.py`

---

### [ACL-012] 2026-06-10 — EP15 M4 GPU MAP-TV 去卷积基准重跑

**问题诊断**:
- EP06 旧 MAP-TV 结果不可作为 4x baseline：`psf_sigma=1.0` 已超出 M3 支持的可信区间 `0.2-0.5 LR px`，`max_iter=4` 远未收敛，lambda 只取单点，forward model 没有包含探测器孔径 box integration。
- EP12 4x 网络没有显示真实增益，后续网络方法需要一个经典、可复现、必须超越的 “baseline to beat”。

**修改内容**:
1. 新增 `algos/ep15_info_limit/scripts/run_m4_deconv_anchor.py`：用 PyTorch GPU batch 实现 `BatchForwardModel`，一次处理 248 帧 shift / Gaussian PSF / detector box downsample；`adjoint()` 反向执行 upsample / PSF / reverse shift 并累加梯度。
2. 默认 forward model 改为 `HR -> shift -> Gaussian PSF -> avg_pool2d detector box -> LR`，`--no-box` 仅作为消融开关。
3. MAP-TV 主循环使用 FISTA + smoothed TV gradient，full run `max_iter=150`，输出 `iteration,data_rmse,tv_value,objective,relative_update` 收敛曲线。
4. 参数网格改为 `sigma={0.2,0.3,0.4,0.5} LR px`、`lambda={3e-4,1e-3,3e-3}`；每个 sigma 先用 odd/even split-half NRMSE + artifact/std proxy 选 lambda，再用全 248 帧跑 full reconstruction。
5. 新增四方法视觉对比、sigma=5 highpass 对比、zigzag 定量剖面、split-half FRC 复验和全参数选择 CSV。

**预期效果**:
- 给后续 UNet/Transformer 建立经典算法及格线：如果网络不能同时超过 MAP-TV 的 FRC 与 zigzag 指标，则没有采纳价值。
- 直接回答客户关心的 zigzag 细线是否变清楚，并给未来训练 target 的锐度水平作预演。
- 风险: MAP-TV FRC 上升主要是 split-half 一致性 proxy，不是独立光学 ground truth；hardcoded zigzag 剖面只覆盖当前 ROI；去卷积可能引入点状伪影。

**推荐参数**:

```bash
cd algos/ep15_info_limit
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_m4_deconv_anchor.py --smoke --chunk-size 8
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_m4_deconv_anchor.py --chunk-size 32
```

**训练结果**:
- 输出目录: `output/ep15_info_limit/m4_deconv_anchor/`
- 视觉效果: MAP-TV 相比 bare drizzle 减轻 lattice/coverage 伪影，center zigzag highpass 轮廓更集中，但仍有点状去卷积伪影；EP07 v6 仍表现为更强的 learned regularization 对照。
- 关键指标: 选择 `sigma=0.2 LR px, lambda=1e-3`；zigzag median FWHM **114 -> 100 µm**，median dip depth **0.929 -> 0.934**；bare/MAP-TV FRC 在 12 µm 为 **0.575 -> 0.947**；全量耗时 **4563 s**，full-run relative update 约 **0.005**，达到平台期但未触发 `tol=1e-5`。
- 结论: M4 是有限正向、但不强阳性的经典基准。后续 4x 网络必须同时优于 MAP-TV 的 FRC 频带一致性和 zigzag FWHM / dip 指标，否则不予采纳。

**涉及文件**: `algos/ep15_info_limit/scripts/run_m4_deconv_anchor.py`

---

### [ACL-011] 2026-06-10 — SSIM / Gaussian blur 改用 custom_fwd，消除 AMP 气泡

**问题诊断**:
- `ContourSRLoss` 中 `ssim()` 与 `gaussian_blur_2d()` 在 AMP 训练步内使用 `autocast(enabled=False)` + 手动 `.float()`，会在 fp16 主通路中插入 fp32 子图，造成 dtype 切换气泡并降低 GPU 吞吐。
- 原注释指出 fp16 Gaussian 统计会 NaN，因此不能简单删掉 fp32 保护。

**修改内容**:
1. `losses.py`: 抽取 `_ssim_float32` / `_gaussian_blur_2d_float32` 核心实现。
2. CUDA 路径改用 `@torch.amp.custom_fwd(cast_inputs=torch.float32)` 包装 `_ssim_cuda` / `_gaussian_blur_2d_cuda`，在 AMP 上下文内一次性 cast 到 fp32，避免嵌套 `autocast(False)`。
3. CPU 路径保持显式 `.float()` fallback。
4. 新增 `test_contour_sr_loss_finite_under_cuda_amp` 回归测试。

**预期效果**:
- 保持 SSIM / highpass 统计数值稳定（无 fp16 NaN），同时减少 AMP 气泡、略降 loss 段 kernel 切换开销。

**推荐参数**: 无需 CLI 变更。

**训练结果**: _(v8 重训后填写)_

**涉及文件**: `algos/ep07_unet_sr/src/unet_sr/losses.py`, `algos/ep07_unet_sr/tests/test_model_losses.py`

---

### [ACL-010] 2026-06-10 — thin/gap 权重图下沉 DataLoader worker，消除主进程 CPU 瓶颈

**问题诊断**:
- v8 在 `train.py` 主进程对整 batch 调用 `_make_mask_loss_weights`（逐样本 scipy EDT），batch=128 时约 600 ms/step，GPU 利用率降至 ~50%。
- 增加 `num_workers` 无效：瓶颈发生在 DataLoader 返回之后的主线程串行段，与 worker 并行无关。

**修改内容**:
1. 新增 `mask_weights.py`：抽取单 patch 与 batch 版权重图计算逻辑。
2. `dataset.py`：当 `thin_boost>1` 或 `gap_boost>1` 时，在 `__getitem__` 内对每个 patch 预计算 `thin_weight` / `gap_weight`，由 DataLoader worker 并行执行。
3. `train.py`：删除主进程 batch 级权重计算，直接消费 batch 中预计算张量。

**预期效果**:
- mask 权重 CPU 成本分散到 worker，并与 GPU 步通过 prefetch 重叠，恢复 v6 水平 GPU 利用率，同时保留 thin/gap loss 语义。
- 单 patch EDT（256×256）比 batch=128 主进程循环更轻；总 CPU 算力需求不变，但不再阻塞 GPU。

**推荐参数**: 沿用 v8 `--thin-boost 6 --gap-boost 4`；`num_workers` 保持 v6 水平（6–8）即可。

**训练结果**: _(v8 重训后填写)_

**涉及文件**: `algos/ep07_unet_sr/src/unet_sr/mask_weights.py`, `algos/ep07_unet_sr/src/unet_sr/dataset.py`, `algos/ep07_unet_sr/src/unet_sr/train.py`, `algos/ep07_unet_sr/tests/test_dataset.py`, `algos/ep07_unet_sr/tests/test_model_losses.py`

---

### [ACL-009] 2026-06-10 — 细结构感知与窄缝保护的温和 loss 加权

**问题诊断**:
- v6/v7 hybrid 配置整体视觉效果满意，但中心单像素细线仍容易被抹平，并与大块结构粘连。
- 根因 1: 1 px 细线在 256×256 patch 中占比极小，普通 MSE/highpass/grad_vector 对其话语权不足。
- 根因 2: 历史 ACL-001 的 skeleton_boost=30 / gap_boost=15 已证明过激权重会导致整体糊化和振铃，本轮只能采用个位数温和增强。

**修改内容**:
1. `losses.py` / `train.py` / `config.py`: 基于 batch 中 `hr_mask >= 0.5` 生成可选权重图，宽度 ≤3 HR px 的结构内细线在 highpass / grad_vector 项乘以 `thin_boost`，窄背景缝隙在 mse / highpass 项乘以 `gap_boost`。
2. `ContourSRLoss.forward` 新增可选 `thin_weight`、`gap_weight` 参数；默认 `None` 时保持旧行为，兼容旧 checkpoint 和旧调用。
3. CLI 新增 `--thin-boost` (默认 6) 与 `--gap-boost` (默认 4)，并在训练 config 中记录。

**预期效果**:
- 提升亚像素/单像素细线和窄缝在 loss 中的占比，减少抹平和粘连。
- 风险: boost 过大可能重现 ACL-001 的振铃与糊化，因此默认控制在个位数。

**推荐参数**: `--thin-boost 6 --gap-boost 4`

**训练结果**: _(v8 训练后填写)_
- 输出目录: `outputs/ep07_v8_aa`
- 视觉效果: _TODO_
- 关键指标: _TODO_
- 结论: _TODO_

**涉及文件**: `algos/ep07_unet_sr/src/unet_sr/losses.py`, `algos/ep07_unet_sr/src/unet_sr/train.py`, `algos/ep07_unet_sr/src/unet_sr/config.py`, `algos/ep07_unet_sr/tests/test_model_losses.py`

---

### [ACL-008] 2026-06-10 — TCForge 抗锯齿覆盖率渲染训练池

**问题诊断**:
- v6/v7 视觉结果中的大块结构边缘有 1-2 px 楼梯锯齿；诊断确认根因在 TCForge target，而不是 UNet 结构。
- `build_scene_mask_with_metadata` 用最近邻旋转二值 mask 并重新二值化，`render_temperature_field` 再把二值 mask 渲染为硬阶跃温度，导致 HR target 本身锯齿化。
- 5 µm HR 像素跨结构边界时真实辐射应按覆盖率加权；1 px 细线也不应拥有满幅值 `delta_T`。

**修改内容**:
1. `geometry.py`: `build_scene_mask_with_metadata(..., antialias=True)` 默认使用 4× SSAA 绘制、order=1 float 旋转、4×4 block-average 回落到 HR，输出 `[0,1]` soft coverage mask；`antialias=False` 保留旧二值路径。
2. `physics.py`: `render_temperature_field` 在无多温度 label 时接受 `[0,1]` float coverage mask，按 `T_bg + delta_T * coverage` 渲染。
3. `storage.py`: compact scene 的 `hr_mask_4x.png` 改为 0-255 覆盖率 PNG，读取时恢复为 `float32 [0,1]`；`hr_edge` 仍保持二值契约。
4. `scripts/generate_training_pool.py`: `hr_edge` 改由 `hr_mask >= 0.5` 生成，并同步提高 worker 内存估算。

**预期效果**:
- 大块旋转边缘成为 1-2 px 平滑覆盖率过渡，训练 target 不再强迫网络复现楼梯锯齿。
- 亚像素细线以覆盖率幅值进入 HR 温度 target，使目标更符合 20 µm PSF 后的可实现信号。
- 风险: soft mask 存为 uint8 PNG 有约 1/255 覆盖率量化误差，足够用于训练但不应用于计量级边界面积分析。

**推荐参数**: `--training-pool-dir ../../data/synthetic/training_pool_2x_aa`

**训练结果**: _(v8 训练后填写)_
- 输出目录: `data/synthetic/training_pool_2x_aa`（1000 scenes，训练池生成后补充证据图路径）
- 视觉效果: _TODO_
- 关键指标: _TODO_
- 结论: _TODO_

**涉及文件**: `tcforge/src/tcforge/geometry.py`, `tcforge/src/tcforge/physics.py`, `tcforge/src/tcforge/storage.py`, `scripts/generate_training_pool.py`, `tcforge/tests/test_geometry.py`, `tcforge/tests/test_physics.py`, `tcforge/tests/test_storage.py`

---

### [ACL-007] 2026-06-10 — TGV 各向异性正则化 + 覆盖率加权数据项（修复横向条纹伪影）

**问题诊断**:
- TGV 重建结果出现明显的横向条纹伪影
- 根因 1: raster 扫描的各向异性数据覆盖 — 行内 X 方向 acquisition gap=1（时间连续），行间 Y 方向 gap≈16（隔了一整行 X 扫描），导致 forward model 数据约束在 X 方向远强于 Y 方向
- 根因 2: scatter adjoint 的 bilinear splatting 在位移 Y 分量接近整数倍时，权重集中到固定 HR 行，产生行间不均匀覆盖

**修改内容**:
1. **各向异性 TGV 正则化** (`tgv.py`):
   - 新增 `_project_vector_ball_aniso()` 和 `_project_sym_ball_aniso()` — 椭球 dual projection
   - `_tgv_denoise_fallback()` 新增 `aniso_ratio_y` 参数，Y 方向正则化半径 × aniso_ratio_y
   - `tgv_denoise()` 在 `aniso_ratio_y != 1.0` 时绕过 CCPi 直接使用 Chambolle-Pock fallback
   - `reconstruct_map_tgv()` 透传 `aniso_ratio_y` 参数
2. **覆盖率加权数据梯度** (`tgv.py`):
   - 新增 `_compute_coverage_map()` — 预计算每个 HR 像素的帧覆盖率（仅依赖 shifts，入口算一次）
   - 新增 `_data_gradient_and_loss_coverage()` — 数据梯度除以 per-pixel coverage 而非统一 N_frames
   - `reconstruct_map_tgv()` 新增 `coverage_weighted` 参数
3. **CLI 集成** (`run_tgv_sr.py`):
   - `--aniso-ratio-y` (默认 1.5)
   - `--coverage-weighted` (默认启用)
   - config dict、result row、run_summary.json 均记录新参数

**预期效果**:
- 各向异性正则化直接抑制 Y 方向行间不连续（横向条纹）
- 覆盖率加权避免低覆盖区域被稀疏帧残差过度拟合
- 两者叠加应显著改善 TGV 横向伪影

**推荐参数**: `--aniso-ratio-y 1.5 --coverage-weighted`

**训练结果**:
- 输出目录: `output/ep10_tgv_sr/`
- 视觉效果: 横向条纹伪影几乎完全消除，中心 ROI highpass 轮廓更清晰、背景更干净
- 关键指标: artifact_score 3.8701 → **0.6950** (-82.0%)，raw_control_corr 0.9021 → **0.9162** (+0.014)
- 结论: 各向异性正则化 + 覆盖率加权数据项双管齐下有效解决了 raster 扫描各向异性导致的 TGV 横向条纹伪影；耗时 30.8 分钟（CPU Chambolle-Pock fallback，因 CCPi 不支持各向异性）

**涉及文件**: `algos/ep10_tgv_sr/src/ep10_tgv_sr/tgv.py`, `algos/ep10_tgv_sr/scripts/run_tgv_sr.py`

---

### [ACL-006] 2026-06-10 — 修复 real_eval 真实数据评估基准图尺寸不匹配错误

**问题诊断**:
- 训练在进入评估节点（如每 2000 步）时，调用 `maybe_log_real_eval` 会因为 `all the input array dimensions except for the concatenation axis must match exactly` 报错崩溃。
- 根因分析: 4x SR 预测输出（如 EP12）的尺寸为 1920×2560，但使用的默认基准图（EP10 的 2x 结构）为 960×1280。进行中心裁剪及 `zoom_center` 缩放后，两者的绝对尺寸在行方向上差了 2 倍，因而无法横向合并拼接导致报错。

**修改内容**:
1. **自动缩放基准图**: 在 `real_eval.py` 的 `maybe_log_real_eval` 中，若 `baseline_hp` 存在且其尺寸与预测图 `unet_hp` 不一致，则先使用 `ndimage.zoom` 对其按比例进行双线性上采样对齐尺寸，然后再进行后续的裁剪与拼接。

**预期效果**:
- 消除尺寸不匹配引发的拼接报错崩溃，使训练能连续越过 2000/4000/6000 步等所有评估节点并保存 checkpoint。

**推荐参数**: 保持原训练参数。

**训练结果**:
- 输出目录: `outputs/ep07_v6_physics`
- 视觉效果: 60k 步 hybrid 配置整体满意；真实数据 center zoom3x 视图比早期版本更稳定，未再因 real_eval baseline 尺寸不匹配中断。
- 关键指标: 本条主要修复评估崩溃，无独立训练指标；评估口径为 248 clean main-session frames、zoom3x，与 EP11 保持一致。
- 结论: real_eval 尺寸兼容修复有效；该轮训练遗留大块边缘锯齿、中心细线抹平/粘连、镂空中心紫色 vs 片外黑色三项视觉现象，其中前两项触发 ACL-008/009，第三项已确认是真实数据分布，不改可视化行为。

**涉及文件**: `real_eval.py`

---

### [ACL-005] 2026-06-10 — 用梯度向量匹配 loss 替换 laplacian + PSF forward_model

**问题诊断**:
- v6_physics (ACL-003) 在 ~3000 步后 highpass/edge/laplacian 进入震荡平台，无法继续下降
- 根因分析: 6 个 loss 目标互相竞争梯度方向
  - **highpass/laplacian 推"更锐"** vs **forward_model 通过 PSF 低通要求"更钝"** → 方向冲突
  - forward_model 本质缺陷: PSF 模糊后丢失高频信息，对低频与 MSE 重复，对高频无约束
  - MSE 和 forward_model 的 batch 间方差达 26×~80×，grad_norm 在 12~263 间剧烈波动
- 现有 edge loss 只比较梯度幅值 `|∇pred| - |∇target|`，漏检了梯度方向变化（膨胀场景下幅值相同但方向偏转）

**修改内容**:
1. **新增 `sobel_edges_xy()`**: 返回 Sobel 梯度分量 `(gx, gy)` 而非仅幅值；`sobel_edges()` 重构为调用它
2. **新增 `grad_vector_loss()`**: 比较完整梯度向量 `(gx, gy)` 的 L1 距离，加 target 梯度幅值加权
   - 是现有 edge loss（仅幅值对比）的**严格超集**: 既包含幅值信息（捕获断连/粘连），又包含方向信息（捕获膨胀/扭曲）
   - 一个 loss 同时覆盖: 膨胀、粘连、断连、幻觉四种结构缺陷
3. **`ContourSRLoss` 简化为 5 个 loss**: mse + highpass + edge + ssim + grad_vector
   - 移除 `laplacian_weight` 和 `forward_model_weight`（保留 kwargs 默认 0 兼容旧 checkpoint）
   - 新增 `grad_vector_weight` (默认 0.3)
4. **Config/CLI**: `config.py` 新增 `--grad-vector-weight`

**预期效果**:
- 消除 highpass 与 forward_model 的梯度冲突 → 训练曲线应稳定下降
- grad_vector 直接在 HR 空间操作，不经过 PSF 丢信息 → 保持锐度的同时约束结构形态
- Loss 数量 6→5，且不存在"一推一拉"的对立对 → grad_norm 方差应显著降低

**推荐参数**: `--grad-vector-weight 0.3 --laplacian-weight 0 --forward-model-weight 0`

**训练结果**:
- 输出目录: `outputs/ep07_v6_physics`（实际执行为 hybrid 配置: `grad_vector=0.3 + laplacian=0.1 + forward_model=0.1`，并非纯 ACL-005 v7 gradvec）
- 视觉效果: 60k 步结果整体满意，主结构轮廓和真实数据 center zoom3x 推演稳定；仍有大块结构边缘锯齿、中心 1 px 细线抹平/粘连，以及镂空中心紫色 vs 片外黑色三项现象。
- 关键指标: 本轮以真实数据视觉 sanity 与训练稳定性为主；real_eval 使用 248 clean main-session frames、zoom3x。
- 结论: grad_vector 方向有效，但本次训练实际保留了 laplacian/forward_model 的 hybrid 约束；遗留锯齿根因在 TCForge 二值旋转 target，细线/粘连需温和宽度感知权重处理，镂空中心紫色属于真实数据温度分布，不作为算法 bug。

**涉及文件**: `losses.py`, `config.py`, `train.py`

---

### [ACL-004] 2026-06-09 — Checkpoint 推演改为 EP11 真实数据 3× 温度图


**问题诊断**:
- 原先 `--tb-image-every` 默认等于 `save_every`，checkpoint 时记录的是 **TCForge 合成训练 batch** 的 pred/target，不是用户关心的真实数据视觉结论
- `real_eval` 默认仅 48 帧、overlap=32，与 EP11 benchmark（248 帧、overlap=128、center_zoom3x 温度 PNG）口径不一致

**修改内容**:
1. **`tb_image_every` 默认 0**：不再自动在 checkpoint 记录 TCForge 合成图；需显式 `--tb-image-every N` 才开启
2. **`real_eval` 对齐 EP11**：默认 248 帧、overlap=128、display zoom=3.0；温度图用 inferno + 1–99 percentile，与 EP11 `save_unet_temperature_view` 一致
3. **PNG 落盘**：每次 checkpoint 写入 `{output_dir}/eval_real/unet_step{N}_center_zoom3x_temperature.png`

**预期效果**:
- 每 2000 step / save checkpoint 可直接在 TensorBoard 或磁盘看到与 EP11 notebook 同口径的真实数据温度 sanity 图
- Checkpoint 推演耗时增加（248 帧全量 inference）；可用 `--real-eval-frame-limit 48` 加速调试

**推荐参数**: 保持 `--save-every 2000`；不要加 `--tb-image-every` 除非需要看合成域 loss 可视化

**训练结果**: _(待训练后填写)_

**涉及文件**: `real_eval.py`, `config.py`, `train.py`, `scripts/run_training.md`, `pyproject.toml`

---

### [ACL-003] 2025-06-09 — 新增 Laplacian 锐度 + PSF 前向模型 loss

**问题诊断**:
- v5_no_split (22k steps): 锐度比 v4 有所回升，但出现**细线结构变粗**（1px 细线被预测为 2-3px），中心区域仍有轻微模糊
- 根因分析: `skeleton_boost=30` 的权重地图让网络发现「把细线画宽也能降 loss」；缺少物理约束导致网络自由度过大

**修改内容**:
1. **Laplacian 锐度 loss（非对称）**: `losses.py` 新增 `laplacian()` 函数 + `ContourSRLoss` 添加 `laplacian_weight` 参数
   - 原理: 计算 pred 和 target 的 Laplacian 幅度，只惩罚 `|Lap_target| > |Lap_pred|` 的位置（即 pred 比 target 更模糊处）
   - 不惩罚更锐的方向，避免抑制合理的锐化
2. **PSF 前向模型一致性 loss**: `ContourSRLoss` 添加 `forward_model_weight` + `forward_model_psf_sigma` 参数
   - 原理: `HR_pred → Gaussian blur(σ=PSF) → downsample → 应匹配 LR aligned_mean`
   - 利用已知 PSF（σ=0.5 LR px = 1.0 HR px）构建物理约束
   - 天然防止虚假细节和线条增粗（变粗后 forward model 不匹配 LR 观测）
3. **配置 / CLI**: `config.py` 新增 `--laplacian-weight`、`--forward-model-weight`、`--forward-model-psf-sigma`
4. **训练管线**: `train.py` 从 `obs[:, 0:1]` 提取 `aligned_mean` 传入 loss 函数

**预期效果**:
- Laplacian loss 应直接惩罚细线变粗（Laplacian 幅度下降 = 边缘变钝）
- Forward model loss 提供物理一致性锚定，限制网络生成与 LR 观测矛盾的细节
- 两者配合应在保持锐度的同时防止 structure bloat

**推荐参数**: `--laplacian-weight 0.1 --forward-model-weight 0.1 --forward-model-psf-sigma 0.5`

**训练结果**: _(待训练后填写)_
- 输出目录: `outputs/ep07_v6_physics`
- 视觉效果: _TODO_
- 关键指标: _TODO_
- 结论: _TODO_

**涉及文件**: `losses.py`, `config.py`, `train.py`, `scripts/run_training.md`

---

### [ACL-002] 2025-06-0x — v5_no_split: 回归 gradient-based ContourSRLoss + base_channels=64

**问题诊断**:
- v4 (balance_edge / large_bucket) 使用 skeleton/gap/anti-merge loss 成功解决了粘连问题
- 但 skeleton_boost=30 / gap_boost=15 的极端权重导致整体糊 + 振铃效应
- mse_weight=0.02 太低，DC 锚定不足

**修改内容**:
1. 回归 gradient-based `ContourSRLoss`（移除 skeleton/gap/anti-merge 分支）
2. 提升 `base_channels` 从 48 → 64（增加模型容量）
3. 保留 `structure_boost=4.0`、`mse_weight=0.2`

**训练结果**:
- 锐度比 v4 有所恢复
- 但出现细线结构变粗 → 触发 ACL-003 改进

---

### [ACL-001] 2025-06-0x — v4 (balance_edge / large_bucket): 抗粘连 loss 实验

**问题诊断**:
- v3 (ep07_run, 40k steps) 视觉锐度不错，但中心细节扭曲，存在结构粘连
- 需要专门的 anti-merge 和 skeleton/gap-aware 权重来解决

**修改内容**:
1. 新增 `skeleton_boost=30, gap_boost=15, mask_boost=5` 精细权重地图
2. 新增 `anti_merge_weight=0.5` 惩罚不同结构间的粘连
3. 降低 `mse_weight=0.02`（让 structure loss 主导）
4. 降低 `base_channels=48`（减小模型以降低过拟合风险）

**训练结果**:
- ✅ 粘连问题解决
- ❌ 整体锐度下降、出现振铃效应
- ❌ mse_weight 太低导致 DC 锚定不足
- 结论: skeleton/gap 精细权重方向正确但参数过激进；需要更温和的约束方式

---

## 模板

```markdown
### [ACL-XXX] YYYY-MM-DD — 简短标题

**问题诊断**:
- 上一版本的什么问题触发了本次修改？

**修改内容**:
1. 具体改了什么（文件、函数、参数）
2. 原理是什么

**预期效果**:
- 预期改善什么
- 可能的风险

**推荐参数**: `--key value ...`

**训练结果**: _(训练后填写)_
- 输出目录:
- 视觉效果:
- 关键指标:
- 结论:

**涉及文件**: file1.py, file2.py, ...
```
