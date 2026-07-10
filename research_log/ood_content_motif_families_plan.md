# OOD content 轴新 motif 族实施计划（organic_blobs / text_serial / concentric_rings / voronoi_cells）

> 2026-07-10，主循环设计，fork agent 在 worktree 实现。背景：`research_log/ood_generalization_suite_design.md` §2 A1
> 设计的 4 个新 motif 族在首轮 OOD 套件（seeds 20260920-28，2026-07-10 在跑）中因生成器代码缺口被挡在外
> （`research_log/ood_generator_support_audit_draft.md` 任务1 A1：新族需要新代码分支），content 轴退而用了 4 档
> config 可达阶梯（texroles/xlmerge/legacymix/tracebus）。本计划补上代码缺口，产出 content 轴第二批（真正
> out-of-grammar 的新几何语法）。**eval-only：冻结 checkpoint 纯推理，永不训练（预注册规则 1）。**

## 0. 集成方式（定死）

- 新分支加在 `tcforge/src/tcforge/geometry.py::_compose_cpu_scene` 的 family 派发链（geometry.py:1436 起），
  与现有 6 族并列；族名注册进 `CPU_SCENE_FAMILIES`（geometry.py:901）。
- config 触达：legacy composer 路径（config 不带 scene_composer）+ `geometry.motif_weights` 单键 1.0，
  与首轮 legacymix/tracebus 池同一模式。
- **字节级兼容红线**：`family = _pick_weighted(...)` 之前不得新增任何 rng 抽样；现有 6 族分支代码一字不动。
  现有配置的 motif_weights 不含新键 → 现有池逐字节不变。
- **新增校验**（当前缺口：未知 motif 键静默落到 else/generic）：`_compose_cpu_scene` 入口校验
  `motif_weights` 的键 ⊆ `CPU_SCENE_FAMILIES`，未知键 raise ValueError。旧配置全部键合法，不受影响。

## 1. 公共硬约束（每族都要）

- 频带诚实下限（ACL-023/030，与现有族同一常量）：最细结构特征宽度 ≥ FLOOR = det*1.4 = 28µm；
  周期性 pitch ≥ pitch_floor = det*1.6 = 32µm。禁止任何亚 pitch 幻觉诱饵。
- 画布/坐标：µm 坐标经 `common` 走现有 `make_*` 原语，或直接在 draw 网格（SSAA 细网格）上光栅化后 `_add`/`_sub`；
  直接光栅化时用 float32/分块，控制内存（draw_shape 可达 5760×7680）。
- 每个结构 append `primitives` 元数据 dict（type + 关键几何参数），返回 family 名与 density（现有签名不变）。
- density 标量已在派发前采出，可用于计数/密度调制（_dcount/_lerp 模式照抄现有族）。

## 2. 族规格

### organic_blobs（有机斑块——非制造几何、平滑闭曲线语法）
- K = _dcount(3, 9) 个斑块；中心 _rpos；基半径 r ∈ [90, 260]µm。
- 边界 = 低阶谐波调制的径向轮廓（3-7 阶，总幅度 ≤ 0.35r），平滑有机外形；允许重叠取并。
- 最窄颈约束：r*(1−Σamp) ≥ 1.5*FLOOR（42µm），保证任何局部不产生亚 floor 细颈。
- ~25% 斑块带一个内孔（孔径 ≥ 2*FLOOR，环壁 ≥ FLOOR）。

### text_serial（文字-序列号——笔画语法）
- 7 段式字形（非点阵：28µm 点 @32µm pitch 几乎粘连，不诚实）。字形高 ∈ [140, 260]µm，
  笔画宽 ∈ [FLOOR, 1.3*FLOOR]，字符 pitch ≥ 字宽 + FLOOR。
- 每景 1-3 行，每行 n = _dcount(4, 10) 字符，每字符随机非空段子集；行向水平/垂直随机。
- 两种模式各 ~50%：raised（笔画直接 _add）/ engraved（先 _add 一块底板，再 _sub 笔画，刻槽宽 ≥ FLOOR）。

### concentric_rings（同心环——周期径向语法）
- 1-2 个环系（_dcount）；中心近画布中心 ± 抖动；可选椭圆比 ∈ [0.7, 1.0]。
- 径向周期 p ≥ max(pitch_floor, w + FLOOR)，环宽 w ∈ [FLOOR, p − FLOOR]（环与间隙都 ≥ FLOOR）；
  环数由最大半径（~0.35*min(canvas_w,canvas_h)）截断；~50% 带中央实心盘。

### voronoi_cells（Voronoi 胞元——胞元镶嵌语法）
- K = _dcount(6, 16) 个种子点，最小间距 ≥ 5*FLOOR（拒绝采样/抖动网格）；
  作用域 = 画布中央 ~0.60-0.75 区域（矩形或大圆）。
- 逐像素最近/次近种子距离 d1/d2，边界带 (d2 − d1) ≤ 通道宽 → 通道，其余 = 胞元（热）；
  通道宽 ∈ [FLOOR, 1.6*FLOOR]。实现须增量维护 d1/d2 两个 float32 数组（勿一次性 K 张距离图）。

## 3. 测试（tcforge/tests，与现有 suite 并跑全绿）

每族：motif_weights={族:1.0} 走 build_scene_mask_with_metadata → mask 非空、dtype/shape 正确、
元数据 family 正确；同 seed 两次生成逐字节相同（确定性）；未知族名 raise。
回归：现有全部 tcforge 测试不动、全绿（新校验不得破坏现有配置路径）。

## 4. 池配置（4 个，实现验证后落地）

`pool_2x_ood_content2_{organicblobs,textserial,rings,voronoi}.json`：逐字拷贝 pool_2x_v8_bench48.json，
仅改 _comment/dataset/seed/output_dir + 删 scene_composer/composer_params（走 legacy 路径，同 legacymix 模式）
+ motif_weights → 单键 1.0。Seeds = 20260930/31/32/33（已 grep 全配置确认不相交；20260920-28 被首轮 OOD 占用）。
48 景 n=96，与首轮同规格（ID 锚沿用 v8_bench48 已有 stage2b 行）。

## 5. 验收（主循环亲自做，不信 agent 报告）

1. worktree 内 tcforge 全测试绿。
2. 本地小样生成（每族 6 景，SSAA 同生产）+ PNG 预览目测（ACL-066 教训：占位参数没做目测 = 5000 景池报废）。
3. 最细特征抽查：预览里量最窄笔画/通道/环宽，≥ 28µm（HR 网格 ≥ 2.8px @10µm/px）。
4. 合并 → 全仓测试 → push → ACL 记录。池生成排在远端 oodchain 完成之后（CPU 分钟级/池）。
