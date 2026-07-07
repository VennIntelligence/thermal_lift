# v7 构图器（panel_cluster_v7）+ 缺陷体系进 tcforge —— 实现方案

> 2026-07-08，规划代理产出（只读调研，落点均经逐行核实）。状态：**待实现**。
> 配套方案：research_log/v7_noise_operator_integration_plan.md（噪声+σ-DR，另一代理产出）。
> 前置闸门：scripts/audit_v7_demo_gates.py 的 G1-G8 全过后才动工。

## 0. 范围与不动项

**范围**：①r4 构图器（`scripts/v7_composer_demo.py`，owner 已认可形态）按仓库 RNG 纪律接入 tcforge；②缺陷体系升级（dataset.md §1：统一标注 schema + HR int16 实例掩膜落盘、hot_spot 族、断线查表刻缝、erosion 旋钮/贴边档/自适应、dots 终档 config）；③config 新键 + 金样本/schema/no-silent-caps 测试。

**明确不在本方案内**（dataset.md §3/§4，另立任务）：σ-DR（训练管线侧）、PSF 椭圆比暴露、四项噪声升级（见配套方案）。

**红线**（dataset.md §0、§5）：FLOOR=28µm / pitch_floor=32µm（`geometry.py:993-995`）不动；所有新旋钮默认值 = 旧行为字节不变（含 RNG 流），金样本钉死（`tcforge/tests/data/defects_golden_v1.npz` + 8f863c9 手法）；v7/pilot/bench48 seed 与 v6(960960)、dots_pilot(20260714) 全部不相交。

---

## 1. 新场景族接入：`geometry.scene_composer = "panel_cluster_v7"`

### 1.1 接入点与默认缺省语义

不复用 `motif_weights` 塞新键（会把 v7 逻辑缠进 949-1562 行的 `_compose_cpu_scene`，且 v7 有自己的 tier 体系而非 motif 权重体系）。改为独立分派键：

- **新模块** `tcforge/src/tcforge/composer_v7.py`（~500 行，独立模块使 diff 干净、v6 代码零触碰）。
- `build_scene_mask_with_metadata`（`geometry.py:1613`）签名追加两个 keyword：

```python
def build_scene_mask_with_metadata(
    difficulty, seed, *,
    ...,                                  # 现有参数全部不动
    motif_weights: dict[str, float] | None = None,
    scene_composer: str | None = None,          # 新增；None => 旧行为
    composer_params: dict[str, object] | None = None,   # 新增；v7 专用
) -> tuple[np.ndarray, dict[str, object]]:
```

- 分派位置：`rng = np.random.default_rng(seed)`（geometry.py:1648）之后、`levels` 检查之前插入：

```python
if scene_composer == "panel_cluster_v7":
    from tcforge.composer_v7 import compose_panel_cluster_scene
    v7_mask, v7_prims, v7_extra = compose_panel_cluster_scene(
        rng, params=(composer_params or {}), common=common,
        draw_shape=draw_shape, canvas_w_um=canvas_w_um, canvas_h_um=canvas_h_um,
        detector_pitch_um=float(pixel_size_um))
    cov, meta = _finalize_scene_mask(v7_mask, v7_prims, "panel_cluster_v7",
        rng=rng, rotation_deg_center=rotation_deg_center, ...)   # 复用 1565 起的共享尾部
    meta.update(v7_extra)          # scene_tier / panels / traces / zones / secondary_part
    return cov, meta
if scene_composer not in (None, "legacy"):
    raise ValueError(...)
# 以下 motif_weights / legacy 路径一行不改（1664-1694 分支原样）
```

**RNG 安全性论证**：v6/v5 路径的 rng 在 1648 行创建后第一笔消耗在 `_compose_cpu_scene` 的 `density` 采样（1005 行）或 legacy 的 n_blocks（1728 行）；v7 分派发生在任何消耗之前且仅当新键出现才走，因此 **v6/v5 的 RNG 流逐字节不动**。`_compose_cpu_scene`、`_finalize_scene_mask` 本体零修改（`_finalize_scene_mask` 已支持 `density=None`，v7 传 None 即可；旋转、inscribe_disc、AA 下采样、metadata 组装全部复用，全向旋转由 config `rotation_deg_center: {uniform, 0, 360}` 走 v6 现成机制，`geometry.py:1588-1590`）。

### 1.2 金样本钉死策略（geometry 侧，实施前第一步）

现状 geometry 无金样本（test_geometry.py 只有 seed 复现测试）。在**动任何代码之前**从当前 HEAD 生成 `tcforge/tests/data/geometry_golden_v1.npz`：

- 小画布（如 canvas_shape=(240,320)、ssaa_factor=2）钉三条：legacy 路径（无 motif_weights）× 1 seed、v6 motif 路径（pool_2x_v6_cpu 的 motif_weights）× 2 seed（覆盖不同 family）。
- 存 `mask.tobytes()` + dtype + `json.dumps(metadata, sort_keys=True)`。
- 生成脚本放 scratchpad（8f863c9 先例）；fixture 需 `git add -f`（tests/data 被宽泛的 data/ ignore 误伤——8f863c9 提交信息已记录此坑）。
- 新测试 `test_geometry.py::test_scene_mask_legacy_and_v6_paths_match_golden`；另加 `test_scene_composer_default_none_is_inert`（显式传 `scene_composer=None` 与不传结果 tobytes 相等）。

---

## 2. r4 构图器 → tcforge 图元映射

### 2.1 直接复用（demo 组件 → tcforge 图元）

| demo 组件 | tcforge 落点 | 备注 |
|---|---|---|
| `Canvas.add/sub`、`final = add & ~sub` | v6 同语义：局部 `mask`/`subtract` uint8 + `_add/_sub` 闭包（`geometry.py:974-984`），收尾 `mask & ~subtract`（1561） | 直接照 v6 写法 |
| `stamp_rect` | `make_rectangle`（geometry.py:116） | ⚠️ 参数序翻转：demo `(cy, cx, h, w)` vs tcforge `(cx_um, cy_um, w_um, h_um)`——移植逐处核对 |
| `stamp_disc` | `make_circle_pad`（geometry.py:180） | 直径参数一致 |
| `well_with_elements` | v6 已有等价手法：`_place`/on-die moat（geometry.py:1182-1188, 1095-1104）。实现：`elements` 累积到 scratch uint8 画布 → `_sub(well & ~elements)`；`_add(elements)` | r4 回字嵌套岛 bug 修复的关键语义 |
| `_comb_patch`、`_lined_panel` 刻线 | `_sub(make_rectangle(...))` 循环（同 v6 `_carve_routing` 写法） | |
| `_fin_zone` | elements 循环 + well_with_elements（demo 有逐 fin 抖动和 6% 缺元，`make_pin_array` 不支持 → 循环 stamp） | |
| `_pad_zone` / `_edge_pad_row` | 逐 pad 循环 `make_rectangle`/`make_circle_pad`（per-pad 正态抖动 + drop，`make_pad_grid` present 掩膜表达不了抖动；pad ~30-100 个循环成本可接受） | |
| 收尾（disc/旋转/AA/metadata） | `_finalize_scene_mask`（geometry.py:1565）原样复用 | |

### 2.2 需要新写（都在 composer_v7.py 内）

1. **边邻接生长 + 25% 贴合 + 40-160µm 暗街**（demo `_cluster_scene`）：纯几何 + `_boxes_clear` + 内切圆 0.7×外接半径准入——逐行移植。
2. **整簇居中回移**（scale 阶梯 1.0/0.7/0.4/0.2/0.0）。
3. **`_void_with_traces`**：镂空腔 1-3 条 rim-to-rim 跨腔线（粗 60-150µm / 细 28-60µm）+ 可选嵌套岛，走 well_with_elements；**每条跨腔线记入 traces 表**。
4. **桥接/细总线/长程 L-trace** + traces 表记录。
5. **次要斜置部件**：scratch add/sub 画布组装，窗口化 `ndimage.rotate(order=0, reshape=False)` 粘回；traces 记录带 `angle_deg` 字段（供断线刻缝复合旋转变换）；其 zones 不记（省掉旋转 zone 栅格化复杂度，文档注明）。
6. **panel 角色系统**（lined/frame/windowed/textured 0.30/0.20/0.25/0.25，texture ≤1/3，XL 强制 lined/textured，<0.9mm 小块才许素面）。
7. **clutter** 移植。
8. **demo `_mask_defects`（notch/断线）不移植进 composer**：全部缺陷统一挪到 realism 阶段（§3），composer 只产出 traces/panels 查表数据。理由：缺陷标注/实例掩膜必须在单一阶段、post-rotation HR 坐标系统一落盘，避免在 SSAA 画布上多旋转一张 int16 标签图（~39MB/景）。demo notch（union-of-discs bites）与现有 `apply_defects` notch（`irregular_blob` irregularity=0.7）形态等价。

### 2.3 composer 输出契约

```python
def compose_panel_cluster_scene(rng, *, params, common, draw_shape,
                                canvas_w_um, canvas_h_um, detector_pitch_um
    ) -> tuple[np.ndarray, list[dict], dict]:
```

```python
extra_meta = {
  "scene_tier": "mid|high|xl",
  "panels": [{"cy_um","cx_um","h_um","w_um","role"}...],
  "traces": [{"cy_um","cx_um","h_um","w_um","angle_deg":0.0,"kind":"void_span|bridge|bus|l_trace"}...],
  "zones":  [{"cy_um","cx_um","h_um","w_um","kind":"pads|fins|edge_pads"}...],
  "secondary_part": bool,
}
```

FLOOR/pitch_floor 从 `detector_pitch_um` 推导（×1.4/×1.6，同 geometry.py:993-995），不引入新常数。demo SSAA=3 换 config ssaa_factor=4。栅格化圆整差异与 v6 同源，v6 已过 band 门，接受。

---

## 3. 缺陷体系升级（realism.py + 调用编排）

统一原则：**所有默认关的能力通过"不调用/不额外抽取"实现字节不变**（8f863c9 no-extra-draw 手法）。

### 3.1 (a) 统一标注 schema + HR int16 实例掩膜落盘

**主开关** `defects.record_instances`（默认 `false` = 现状；v7 置 true）。开启后**所有类型**（hole/notch/crack/broken_trace/hot_spot/dark_blob）无条件记实例——默认路径 meta dict 一个键都不多，`defects_golden_v1.npz` 原样保留、零重生成。

**实例掩膜画法**：各 realism 函数增加可选参 `label_map: np.ndarray | None = None, next_id: int = 1`；非 None 时在**刻蚀当下**把形状画进 int16 图（软边 blob 取 ≥0.5 水平集；coverage 类缺陷裁到刻蚀前 struct 内）。后画覆盖先画（文档写死）。不重栅格化。**画标签不消耗 RNG**。

`apply_defects`（realism.py:78）签名扩展（全部默认=旧行为）：

```python
def apply_defects(coverage, rng, *, severity_range=(0.3,1.0),
    hole_radius_px=(4,13), notch_radius_px=(3,10),
    crack_len_px=(60,260), crack_width_px=(2,4),
    max_holes=6, max_notches=6, max_cracks=4,
    hole_depth_range=(1.0,1.0), hole_edge_softness_px=0.0, min_holes=0,
    # ── 新增 ──
    hole_margin_px=8,              # §3.4 erosion 旋钮（默认 8 = _disk(8) 现状, realism.py:100）
    hole_margin_adaptive=False,    # §3.4 域空时确定性阶梯收缩
    hole_edge_fraction=0.0,        # §3.4 贴边档（v7 0.10-0.15）
    record_instances=False,        # §3.1 统一标注
    label_map=None, next_id=1,     # §3.1 实例掩膜（调用方持有）
) -> tuple[np.ndarray, dict]      # record_instances 时 meta 增 "instances" 与 next_id 回传
```

notch/crack 的位置/几何本来就已抽取（realism.py:136-148），record_instances 只是记录——零额外 draw。

**实例 schema（metadata.json 内嵌，`schema_version: 2`）**：

```json
{"id": 1, "type": "hole", "stage": "coverage",
 "center_yx_hr": [512, 640], "radius_px": 2.3, "depth_or_amplitude": 0.55,
 "edge_softness_px": 1.0, "length_px": null, "width_px": null, "gap_px": null,
 "context": "isolated|embedded|edge|background", "trace_index": null,
 "area_px": 14}
```

`context` 事后计算（参考 `probe_dot_retention.py:427-472` isolation 分类器口径），**无 RNG**。

**落盘**：`storage.save_scene_compact`（storage.py:22）加可选参 `defect_instances` → `np.savez_compressed(root/"defect_instances_2x.npz", labels=int16)`（压缩后 ~KB 级）；`load_scene_compact`（storage.py:96）存在则读入。`COMPACT_SCENE_FILES` 不动（可选文件，同 classical_sr 先例）。

**兼容**：`geo_meta["defects"]` legacy 计数键（含 8f863c9 `hole_centers_yx/hole_radii/hole_depths`，realism.py:150-163）原样保留。

### 3.2 (b) hot_spot 族（温度层，§1.2）

新函数（realism.py 追加，不改 `render_isothermal_field` 默认路径）：

```python
def apply_thermal_defects(field, coverage, rng, *, t_bg_c, delta_t_c,
    hot_spot_count=(2,8), hot_radius_px=(1.0,4.0), hot_amp_frac=(0.3,1.0),
    hot_edge_softness_px=1.0, hot_on_structure_p=0.7,
    dark_blob_p=0.6, dark_blob_count=(1,2), dark_blob_radius_px=(8.0,16.0),
    dark_blob_depth=(0.15,0.40), dark_blob_edge_softness_px=3.0,
    record_instances=False, label_map=None, next_id=1,
) -> tuple[np.ndarray, list[dict], int]:
```

物理语义照 demo `temp_render`：hot_spot `T += amp*ΔT*soft_disc`（结构 70%/背景 30%）；dark_blob `T -= depth*max(T-T_bg,0)*w`。经 PSF 模糊 = 场景层真实特征，与 `detector_defects`（探测器层单像素）严格区分。软盘函数移植 demo `_soft_disc`。

**RNG 纪律**：由 pool 脚本按 config 块 `thermal_defects.enabled` 门控——默认块缺失 → 函数不被调用 → 零位移。

**调用位置**：`generate_training_pool.py` isothermal 渲染（422-439）之后、`hr_edge`（440）之前。hot_spot 进 `hr_temperature_2x.npy` GT，detectability 评测直接可用。

### 3.3 (c) 断线缺陷：composer traces 查表刻缝（§1.3）

```python
def carve_trace_breaks(coverage, traces, rng, *, scene_rotation_deg,
    canvas_center_yx, hr_pitch_um,
    break_p=0.7, count_range=(1,2), gap_px=(6.0,20.0), width_pad_px=1.0,
    record_instances=False, label_map=None, next_id=1,
) -> tuple[np.ndarray, list[dict], int]
```

从 `geo_meta["traces"]`（um、pre-rotation + `angle_deg`）抽 1-2 条 → gap 中心（±0.35 长度）→ 经场景旋转变换映到 post-rotation HR 坐标：`p' = C + R(θ)·(p−C)`（θ 与 `_rotate_coverage_mask` 符号约定**用单元测试标定**，勿凭记忆写死）→ 以 `trace.angle_deg + rotation_deg` 取向解析栅格化旋转矩形置零。gap 6-20 HR px = 60-200µm。次要部件复合角由 `angle_deg` 字段覆盖。

**调用位置**：pool 脚本几何之后、`apply_defects` 之前（415 前插）；仅当 `defects.broken_traces.enabled` 且有 traces 表——默认缺失不调用。刻缝先于 apply_defects，使 hr_edge 与 notch/crack 作用在断线后 coverage 上，与 demo 语序等价。

### 3.4 (d) erosion 旋钮 + 贴边档 + 宽度自适应（§1.5 P0）

改造 realism.py:100 `interior = binary_erosion(struct, _disk(8))`：

- `hole_margin_px=8`：`_disk(int(hole_margin_px))`，默认 8 字节不变（erosion 无 RNG）。
- `hole_margin_adaptive=False`：True 且 interior 空时按**确定性阶梯** `[margin, max(margin//2,3), 2, 1]` 收缩直到非空；`meta["hole_margin_effective_px"]` 记录（仅激活时写键）。修复 pilot 副发现（24 景 11 景 0 点，realism.py:119-121 静默击穿）。⚠️ 域从空变非空改变 draw 次数——仅 adaptive=True（非默认）时发生，合规。
- `hole_edge_fraction=0.0`：>0 时每 hole 先判 `rng.random() < fraction` 决定从贴边环带 `struct & ~erosion(margin)` 采样（`if fraction > 0:` 包住 draw，默认零额外 draw）；实例 `context="edge"`。v7 取 0.10-0.15（owner 裁决 §7-5）。
- **no-silent-caps**：`min_holes` 因域空未达标时 meta 记 `holes_shortfall`；配套池级验收（§6 测试 15）。

### 3.5 (e) dots 终档（config-only，零代码）

v7 config defects 块照抄 dots pilot 实测档：`severity_range [1,1]`、`hole_radius_px [1,4]`、`hole_depth_range [0.3,1.0]`、`hole_edge_softness_px 1.0`、`min_holes 20`、`max_holes 50`；`max_notches 6 / max_cracks 4` 恢复 v6 值。孤立/嵌入两类由 §3.4 context 标注实现分类；定向投放分层（`hole_isolated_fraction`）留 pilot 目检后决定，减少一次变量。

### 3.6 温度层 zones 分组

`render_isothermal_field`（realism.py:166）加可选参 `zones=None, zone_rotation_deg=0.0, zone_level_jitter=0.03`：zones 非空时在现有 `per = rng.uniform(...)` 之后再抽 zone 基准电平并覆写 zone 内连通域（zone 栅格 order=0 旋转）。`zones=None` 路径逐字节一致。配套金样本 `isothermal_golden_v1.npz`（实施前生成）。

---

## 4. generate_training_pool.py 编排改动（锚点）

| 位置 | 改动 |
|---|---|
| `geometry_cfg` 解析（392 附近） | 读 `scene_composer`、`composer_params` |
| 几何调用（400-412） | 透传两个新参 |
| defects 段（415-419） | ①`defects.broken_traces.enabled` 且有 traces → `carve_trace_breaks`；②`apply_defects` 传新旋钮 + `record_instances` + 共享 `label_map/next_id` |
| 温度段（422-439） | isothermal 传 `zones`/`zone_rotation_deg`（仅 `temperature_isothermal.zones_enabled`）；后接 `thermal_defects.enabled` → `apply_thermal_defects` |
| metadata（594-646） | 新增 `"defect_annotations"`（schema_version 2、instances 合并三阶段、label_map_file、counts_by_type、hole_margin_effective_px、holes_shortfall）；`"occupancy_hr"`；geometry_metadata 自动带 scene_tier/panels/traces/zones |
| 落盘（648-666） | `save_scene_compact(..., defect_instances=...)`（`storage.save_defect_instances` 门控） |
| `MANIFEST_FIELDS`（51） | 不动 |

`label_map` 由 `_generate_one_scene` 创建（int16，仅 record_instances 时），`next_id` 跨三阶段串联保 id 全局唯一。

---

## 5. config 新键清单（全部默认=旧行为）与新配置文件

```jsonc
"geometry": {
  "scene_composer": "panel_cluster_v7",        // 缺省 null => legacy/motif 路径
  "composer_params": {
    "tier_weights": {"mid": 0.45, "high": 0.45, "xl": 0.10},
    "secondary_part_p": 0.25,
    "merge_p": 0.25, "street_gap_um": [40.0, 160.0], "bridge_p": 0.55,
    "role_weights": {"lined": 0.30, "frame": 0.20, "windowed": 0.25, "textured": 0.25},
    "texture_panel_max_frac": 0.334
  }
},
"defects": {
  "hole_margin_px": 8, "hole_margin_adaptive": true, "hole_edge_fraction": 0.12,
  "record_instances": true,
  "broken_traces": {"enabled": true, "break_p": 0.7, "count_range": [1,2], "gap_px": [6.0,20.0]}
},
"thermal_defects": {
  "enabled": true,
  "hot_spots": {"count": [2,8], "radius_px": [1.0,4.0], "amp_frac": [0.3,1.0],
                "edge_softness_px": 1.0, "on_structure_p": 0.7},
  "dark_blobs": {"prob": 0.6, "count": [1,2], "radius_px": [8.0,16.0],
                 "depth": [0.15,0.40], "edge_softness_px": 3.0}
},
"temperature_isothermal": {
  "level_min": 0.60, "edge_sigma": 0.6,
  "zones_enabled": true, "zone_level_jitter": 0.03
},
"storage": { "save_defect_instances": true }
```

新配置文件：`pool_2x_v7_pilot.json`（24 景试点，seed 与 960960/20260714/bench48 不相交）、`pool_2x_v7_cpu.json`（5000 景主池，owner go 前不跑）、bench48 v7 版。

---

## 6. 测试清单

**金样本（实施前从 pre-change HEAD 生成，`git add -f`）**
1. `test_scene_mask_legacy_and_v6_paths_match_golden`（新 geometry_golden_v1.npz：legacy + v6 motif 小画布）
2. `test_apply_defects_default_path_matches_golden`——原样保留不改
3. `test_isothermal_default_path_matches_golden`（新 isothermal_golden_v1.npz）

**RNG 流不变性（sentinel 手法）**
4. `test_record_instances_consumes_no_rng`：on/off 数组 tobytes 相等 + sentinel 相等
5. `test_hole_margin_default_matches_golden`

**新能力正确性**
6. `test_hole_margin_adaptive_fills_thin_line_scenes`（3px 细线域空 → adaptive 达标 + effective 记录 / off 时 shortfall 记录）
7. `test_hole_edge_fraction_places_boundary_band_holes`
8. `test_carve_trace_breaks_rotation_mapping`（标定 ndimage.rotate 符号约定；含 secondary 复合角 case）
9. `test_apply_thermal_defects_signs_and_bounds`
10. `test_defect_instance_schema_and_label_map`（id 稠密、label⊆instances、center 落支持域、counts 与 legacy 一致）
11. `test_isothermal_zones_group_levels`
12. `test_defect_instances_roundtrip`（int16 保真、缺失不报错）
13. `test_panel_cluster_v7_contract`（seed 复现；occupancy tier 带；trace 宽 ≥28µm、无 ≥0.9mm 素面板、每 void ≥1 跨线、XL 角色 ∈ {lined,textured}、簇在内切圆内）

**池级验收（脚本）**
14. `audit_v7_demo_gates.py --engine tcforge`：对 tcforge composer 复跑 G1-G8
15. `audit_generated_pool.py` 增缺陷 schema 抽查 + 零点景比例=0 检查

## 7. 实现顺序

```
步骤0（阻塞一切）: 金样本 fixture 三件套从当前 HEAD 生成 + 测试1/2/3
   ├─ 步骤A（并行）: composer_v7.py + geometry 分派 + 测试1/13
   │                验证: pytest + audit_v7_demo_gates --engine tcforge
   └─ 步骤B（并行）: realism.py 五项 + 测试4-11（金样本2/3持续绿 = 字节不变证明）
步骤C（依赖A+B）: storage 落盘 + 测试12；generate_training_pool.py 编排
                   验证: --num-scenes 3 冒烟，v6 config 同 seed 输出逐字节 diff
步骤D（依赖C）: v7 pilot/主池/bench48-v7 config + 全链路 preview 脚本
步骤E: 24景 pilot → audit gates → 缺陷特写 sheet → owner 目检
步骤F: owner go 判定点（删v6/4h/60核/350GB 确认；绝不自主启动）
```

每步 commit 独立，提交信息记录默认字节不变证据。
