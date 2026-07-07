# [DRAFT] 生成器支持面审计 — OOD 套件 + A1-dots 试点池可表达性

> 状态：DRAFT，只读代码审计，未 commit。对照 `research_log/ood_generalization_suite_design.md` §0/§2。
> 范围：`scripts/generate_training_pool.py`、`tcforge/src/tcforge/{geometry,realism,physics,shifts}.py`、
> `configs/synthetic/pool_2x_v6_{cpu,bench48}.json`。只报事实 + file:line，不做设计决策。

## 任务1：逐轴支持面审计（A1-A5）

### A1 场景内容

| 项 | 现状 | config 键 | 硬编码点 |
|---|---|---|---|
| motif 族清单 | `pga_grid/die_bga/multi_die/trace_bus/heat_spreader/generic`，6族，权重可调 | `geometry.motif_weights`（`pool_2x_v6_cpu.json:23-30`） | 族本身的构图逻辑（每族的部件排布、尺寸带、密度→计数映射）在 `_compose_cpu_scene`（`geometry.py:949-1562`），新增族需要新代码分支 |
| 参数化程度 | 每族内部：die/pad尺寸比例、pitch带(`p_lo/p_hi`)、pad形状(round/square)、stagger、keepout、drop_p、连续 `density∈[0,1]` 标量驱动计数 — 全部由 `rng` 在函数内采样，**无 config 键**能单独调这些内部比例 | 无（族选择之外全硬编码） | `FLOOR = det*1.4`(28µm)、`pitch_floor = det*1.6`(32µm) 硬编码于 `geometry.py:993-995`（"band-honesty floor"，ACL-023） |
| 15% real-like 成分 | 这是**shift 星座**层面的real-like，不是场景内容层面：`shift_constellation.include_real_like_fraction`（`pool_2x_v6_cpu.json:84`）→ `shift_module.build_scene_shifts`（`shifts.py:210-211`）以该概率从真实EP05轮廓对齐CSV子采样星座，**不影响几何/温度/motif** | `shift_constellation.include_real_like_fraction` | — |
| 缺陷模型全参数透传 | `apply_defects` 签名 8 个参数：`severity_range, hole_radius_px, notch_radius_px, crack_len_px, crack_width_px, max_holes, max_notches, max_cracks`（`realism.py:68-71`）；调用点 `generate_training_pool.py:415-419` 用 `{k:v for k,v in defects_cfg.items() if k!="enabled"} ` **原样 **kwargs 透传，未做任何键过滤/白名单 — 因此 apply_defects 未来新增的任何 kwarg，只要 JSON 加同名键即可透传，无需改 `generate_training_pool.py` | `defects.*`（`pool_2x_v6_cpu.json:33-43`） | 内部 erosion 结构元 `_disk(8)`(interior)、`_disk(2)`(boundary) 硬编码于 `realism.py:77-78`，不透传 config（见任务2） |

### A2 噪声

| 项 | 现状 | config 键 | 硬编码点 |
|---|---|---|---|
| 噪声族切换 | `noise_model.model` 二选一路径：(a) `field_vignette_stripe`→`_realism.field_noise_burst`（vignette+列条纹FPN+逐帧细粒噪声，`realism.py:135-156`）；(b) 其余 6 族走 `physics.add_noise`→`make_noise`：`iid_gaussian/fpn_lowfreq/column_stripe/spatial_correlated/mixed/detector_realistic`（`physics.py:12-19,260-322`） | `noise_model.model` + 各族专属键（`vignette_c/stripe_c/stripe_col_sigma/grain_c` 或 `fpn_sigma_px/stripe_sigma_c`） | 各族内部混合系数硬编码，如 `mixed` 的 `0.55*fpn+…+0.35*iid`（`physics.py:304-310`）、`detector_realistic` 的 `0.45/0.30/0.25`（`physics.py:311-318`）——族内混合比例不可调，只能调总幅值 |
| 行/列条纹 | 已支持：`column_stripe`/`mixed`/`detector_realistic`（physics.py）+ v6 生产路径 `field_vignette_stripe` 的 `stripe_c`（列方向FPN，`realism.py:150-153`） | 已有 | 条纹方向硬编码为**列**（沿列固定跨帧），无"行条纹"选项——微测辐射热计常见的行/列两种FPN方向目前只实现列方向；`_column_stripes`（`physics.py:240-244`）也只做列 |
| 1/f | 近似支持：`fpn_lowfreq`/`mixed`/`detector_realistic` 用高斯平滑随机场(`_lowfreq_fpn`, `physics.py:233-237`) 模拟低频FPN，**非真实1/f功率谱**（是高斯核平滑，不是1/f滤波器），标注为"疑似固定族"的合理性成立 | `noise_model.fpn_sigma_px` 控制平滑尺度 | 平滑核形状固定为高斯；无可配置的功率谱指数 |
| 热像素 | 已支持代码路径：`add_detector_defects`（`physics.py:325-372`），`mode="offset"`(热/冷点偏移) 或 `"stuck"`(卡死像素) | `detector_defects.{enabled,defect_rate,mode,hot_delta_c,cold_delta_c}` | v6 pool 当前**关闭**：`detector_defects.enabled=false`（`pool_2x_v6_cpu.json:94`，注释：真实帧无死点尖峰，开启会引入不真实<0°C冷点）——重新开启只需改 JSON，非代码gap |

### A3 算子（仅确认耦合点，推理期旋钮跳过）

渲染 σ（`psf_sigma_lr_px`，PSF 域）与 DC σ（`noise_sigma_c`，探测器噪声域）在**生成端是两个独立采样、无耦合**的量：
- 渲染 σ：`_uniform_range(rng, physics["psf_sigma_lr_px"], ...)` 或 `sample_psf_parameters(...)`（`generate_training_pool.py:361-380`），喂入 `generate_lr_burst`。
- DC σ：`noise_sigma_c = _uniform_range(rng, physics["noise_sigma_c"], ...)`（`generate_training_pool.py:380`），喂入 `add_noise`（非 field_vignette_stripe 路径）——两者共享同一个 `rng` 流的**先后关系**（先抽PSF、再抽噪声，`generate_training_pool.py:361` vs `380`），但数值上无函数耦合，只是共享随机数发生器的抽样顺序。A3 的 DC 偏置扰动旋钮属推理期（`--dc-sigma-override`），生成端不需要改。

### A4 PSF

| 项 | 现状 | config 键 | 硬编码点 |
|---|---|---|---|
| 支持形状 | `gaussian / elliptical_gaussian / airy_disk` 三种（`PsfShape` Literal, `physics.py:20`；核函数 `make_psf_kernel`, `physics.py:375-421`） | `psf_randomization.{sigma_range,elliptical_probability,airy_probability}` + `psf_sigma_y_lr_px`/`psf_angle_deg`（由 `sample_psf_parameters`, `physics.py:470-511` 采样） | 无第4种形状（如超高斯/离焦环）；airy 核用 `2*J1(z)/z` 精确 Airy 函数(`physics.py:410-414`)，非近似 |
| σ 范围键 | `psf_randomization.sigma_range`（覆盖 `physics_ranges.psf_sigma_lr_px`），v6 当前 `[0.15,0.55]` LR px | 是 | 核半径 `kernel_radius_sigma=4.0` 硬编码（`physics.py:382`），大 σ 外推（设计稿 A4 档 σ∈[0.6,0.9]）时核半径仍按 4σ 缩放，config 无需改——但更大 σ 意味着更大卷积核，性能而非可表达性问题 |
| 各向异性 | `psf_sigma_y_lr_px`(独立y-σ) + `psf_angle_deg`(旋转角) 全由 `sample_psf_parameters` 采样，`elliptical_probability` 控制出现概率，y/x比范围硬编码 `rng.uniform(0.80,1.20)`（`physics.py:502-503`，"强椭圆比"外推需改这行） | 部分（概率可调，比例范围不可调） | 强椭圆比（比如 y/x=2:1）目前不可达——`0.80-1.20` 范围硬编码 |

### A5 帧预算/星座

| 项 | 现状 | config 键 | 硬编码点 |
|---|---|---|---|
| N 范围 | `n_frames_per_scene`：整数或 `{"dist":"randint","low":...,"high":...}`（`generate_training_pool.py:198-220`），v6 训练池 `[24,96]`，bench48 固定 96 | `n_frames_per_scene` | N=12（A5 档位草案）无需改代码，纯改 JSON |
| shift 星座类型 | `shift_constellation.mode` 目前**只有隐式二选一**（非显式mode枚举）：`build_scene_shifts`（`shifts.py:185-276`）用 `include_real_like_fraction` 概率二选：(a) `random_phase`→`random_constellation`（相位网格+coverage质量 good/medium/poor + jitter，`shifts.py:107-182`）；(b) `real_like`→从真实CSV子采样 | `shift_constellation.{phase_steps_choices,coverage_quality_weights,jitter_std_px,include_real_like_fraction}` | **无第三种"漂移型轨迹(非随机)"模式**——`build_scene_shifts` 只实现随机相位网格 or 真实子采样两条路径，没有"确定性漂移/光栅扫描"星座生成器。设计稿 A5 档"漂移型轨迹(非随机星座)"是**代码缺口**，需要在 `shifts.py` 新增一个 constellation 生成函数 |
| "漂移轨迹"键语义警示 | `drift_distribution`/`drift_parameters`（`generate_training_pool.py:264,512-541`）语义是**逐帧强度/温度漂移**（`apply_drift`：`none/scalar_offset/lowfreq/gain_offset/temporal_trend`，`physics.py:11,615-643`），**不是**几何shift位置的漂移轨迹——与设计稿 A5 "漂移型轨迹(非随机星座)"字面撞名但语义不同（一个改像素值随时间变化，一个改子像素采样位置）。给 owner 的关键澄清点 | — | — |

### 温度模型

| 项 | 现状 | config 键 |
|---|---|---|
| `temperature_model` | `"standard"`（默认）→ `render_temperature_field`：2 级(`t_bg_c` + `coverage*delta_t_c`) + 可选低频背景噪声（`physics.py:123-172`）；`"isothermal"` → `_realism.render_isothermal_field`：每个连通结构独立采样等温水平 `∈[level_min,1.0]`，`edge_sigma` 高斯软化边缘，同样可加低频背景（`realism.py:111-132`） | `temperature_model` + `temperature_isothermal.{level_min,edge_sigma}` |
| coverage→温度映射位置 | isothermal: `lvl = lut[lbl]*cov` 然后 `ndimage.gaussian_filter(lvl, edge_sigma)`，`field = t_bg_c + delta_t_c*lvl`（`realism.py:119-126`）。coverage 值本身（0..1）直接乘温度水平，无非线性 | — |

## 任务2：A1-dots 试点池可表达性深查

### 2.1 缺陷半径单位与网格 — **关键澄清**

`apply_defects` 的调用点是 `generate_training_pool.py:400-419`：
```
hr_mask, geo_meta = build_scene_mask_with_metadata(...)      # 已经是"降采样后"的coverage
...
hr_mask, defect_meta = _realism.apply_defects(hr_mask, rng, **defect_params)
```
`build_scene_mask_with_metadata` 内部先在 SSAA 细网格 `draw_shape = hr_shape*ssaa_factor` 上栅格化几何（`geometry.py:1641`），旋转后再 `_downsample_coverage(rotated, aa_factor)` 块平均回 `hr_shape`（`geometry.py:1608-1609,1682`即`_finalize_scene_mask`）。**`apply_defects` 拿到的是已经降采样完成的 HR 网格**（`hr_shape=[960,1280]`，即 `lr_shape*scale`），不是 SSAA 细网格，也不是 LR 网格。

单位换算（`pixel_size_um=20.0, scale=2` → `_hr_pitch_um = pixel_size_um/scale = 10.0µm/px`，`geometry.py:35-42`）：
- **`hole_radius_px` 的 1 单位 = HR 网格像素 = 10µm**（2×SR、20µm探测器 pitch 下）。
- 当前 v6 `hole_radius_px=[4,13]` → 半径 40–130µm（直径 80–260µm）。
- owner 想要的 1–4 HR px 半径 → 直径 20–80µm，对应约 **1–4 LR px 直径的 1/2**（LR pitch=20µm，HR pitch=10µm，故 1 HR px = 0.5 LR px）。

### 2.2 `irregular_blob` 在 r=1-2px 时的行为 — **有质量缺口，非可用性缺口**

`irregular_blob`（`realism.py:27-43`）：`boundary` 在 `[0.3*radius_px, ~1.45*radius_px]` 间按谐波扰动，中心 `rr=0` 恒 `<= boundary(>0)`，**永不退化为空**——最小是单像素或十字/菱形状 3-5px 簇（不会消失）。

但两个真实质量问题：
1. **无抗锯齿**：因为 `apply_defects` 在**降采样之后**才调用（见2.1），SSAA(4/6) 完全不覆盖缺陷生成路径——`irregular_blob` 返回硬布尔掩膜（`rr <= boundary`），`apply_defects:106` 用 `cov*(1.0-defect.astype(float32))` 相乘，`defect` 是 0/1，无灰度过渡。r=1-2px 时会是**硬边块状/十字形**，不是圆润小点。这不是"SSAA 4/6 是否够"的问题——SSAA 对缺陷**从不生效**，是架构上的盲区，不是参数不够大。
2. 由此推论：即使把 `hole_radius_px` 调到 `[1,4]`，纯 config 生成的点确实"存在"且"可见"，但外观是**锐利硬边**，与真实小黑点的软过渡不一致——是外观/真实感缺口，不是"能否生成"缺口。

### 2.3 密度上限

`max_holes`/`severity` 耦合公式（`realism.py:86`）：
```python
count = int(rng.integers(0, round(max_holes * sev) + 1))
```
`sev = rng.uniform(*severity_range)`（`realism.py:84`，只控制**计数**，从不影响半径或深度——见2.4）。

- 要 20-50 点/景，把 `max_holes` 调到 ~50、`severity_range` 提高（如 `[0.8,1.0]`）：**config 可以直接给**，count 上限跟着 `max_holes*sev` 走，无需改代码。
- **但要注意**：`rng.integers(0, N+1)` 是 **[0, N] 均匀分布，包含 0**——config 只能推高计数的**上限**，不能设一个**下限**保证"每景至少 20 个"。以 `max_holes=50, severity_range=[1.0,1.0]` 为例，count ~ Uniform{0,...,50}，期望 25，但标准差 ≈14.4，某些景抽到个位数点数的概率不可忽略。若 owner 需要"每景硬性下限"，这是需要小改代码的点（见2.6 #3）。
- interior erosion `disk(8)`（`realism.py:77`）：把候选点限制在结构内部离边缘 ≥ 8 HR px = **80µm** 以上（`ndimage.binary_erosion(struct, _disk(8))`）。细特征（FLOOR=28µm 走线）腐蚀后基本无内部候选点，因此点缺陷天然只会落在大块 die body / 大 pad 内部，不会落在细走线/pad边缘——按 owner 预判"对芯片小黑点语义可接受"（是），本审计确认此限制**硬编码不可配置**，但语义上与真实小黑点（多出现在大平面区域）一致，不构成阻塞项。

### 2.4 对比度可控性 — **关键缺口**

`apply_defects:106`：
```python
defected = (cov * (1.0 - defect.astype(np.float32))).astype(np.float32)
```
`defect` 是**硬布尔**掩膜（0/1），因此洞内 `coverage` 被强制置为**精确 0**——无论 `severity`、`hole_radius_px` 怎么调，洞的"深度"永远是满量程（coverage 从原值直接砍到 0）。`severity_range` 只控制数量（2.3已确认），**没有任何参数控制深度/对比度**。

Coverage → 温度路径（isothermal 模型，v6 默认）：`lvl = lut[lbl]*cov`，`lbl` 是 `struct=cov>=0.5` 的连通域标签（`realism.py:118`）；洞内 `cov=0` → `struct=False` → `lbl=0` → `lvl=0`（洞内在标签前是纯背景水平），随后 `gaussian_filter(lvl, edge_sigma)` 才引入邻域"渗色"——v6 `edge_sigma=0.6` HR px（`pool_2x_v6_cpu.json:46`），远小于 4-13px 的默认洞半径，所以现有洞基本渲染成**几乎全深度冷点**（`field=t_bg_c`附近），只在洞边缘窄带因 `edge_sigma` 模糊出浅过渡。若把半径调到 1-4px（接近或小于 `edge_sigma`+PSF σ 的量级），观测到的"深度"会因物理blur自然变浅——但这是**PSF/edge_sigma 的副作用**，不是一个可独立调节的"对比度"旋钮。

**结论**：真实小黑点是"局部偏暗"（部分对比度），当前实现是"强制置零"（无对比度参数），severity 只管数量不管深度——**这是明确的 config 缺口**，纯 config 无法表达"浅色点"，只能通过半径足够小+物理blur侥幸获得类似浅色效果（不可控、不可预测）。

### 2.5 config-diff 模式与 dots 试点池键值草案

`pool_2x_v6_cpu.json` vs `pool_2x_v6_bench48.json` 逐键 diff（程序化比较，两文件其余全部键完全一致）：

| 键 | v6_cpu | bench48 |
|---|---|---|
| `_comment` | (设计说明) | (bench48说明) |
| `dataset` | `ThermalChipPhantom_v6_cpu` | `ThermalChipPhantom_v6_cpu_bench48` |
| `n_frames_per_scene` | `{"dist":"randint","low":24,"high":96}` | `96`（固定标量） |
| `num_scenes` | `5000` | `48` |
| `seed` | `960960` | `20260708` |
| `output_dir` | `data/synthetic/pool_2x_v6_5k` | `data/synthetic/pool_2x_v6_bench48` |

即真正影响生成行为的只有 3 键（`n_frames_per_scene`、`num_scenes`、`seed`）+ 1 路径键（`output_dir`）+ 2 纯文档键（`_comment`、`dataset`）——与 bench48 文件自身注释("VERBATIM copy … except exactly five fields")一致。

**dots 试点池建议键值草案**（沿用同一"逐字拷贝+改动键"模式，基于 `pool_2x_v6_cpu.json`；下列为建议起点，非最终决定）：

```jsonc
{
  // 文档/身份键（同 bench48 模式）
  "dataset": "ThermalChipPhantom_v6_cpu_a1dots_pilot",
  "num_scenes": 24,                 // 12-24 景档，取上限
  "seed": <新的不相交种子>,          // 不与 960960 / 20260708 冲突
  "output_dir": "data/synthetic/pool_2x_v6_a1dots_pilot",
  "n_frames_per_scene": 96,         // 或沿用 [24,96]——待定，非本审计范围

  // 唯一语义改动：defects 块
  "defects": {
    "enabled": true,
    "severity_range": [0.85, 1.0],   // 推高计数下限的期望（仍非硬下限，见2.3）
    "hole_radius_px": [1, 4],        // HR-grid px；物理直径 20-80µm（2×SR, 10µm/px）
    "notch_radius_px": [3, 10],      // 保留默认或...
    "crack_len_px": [60, 260],
    "crack_width_px": [2, 4],
    "max_holes": 50,                 // 密度上限 20-50/景的关键旋钮
    "max_notches": 0,                // 关闭 notch/crack，只保留点(hole)缺陷
    "max_cracks": 0
  }

  // 其余键（geometry/temperature_model/physics_ranges/noise_model/
  // shift_constellation/psf_randomization/detector_defects/features/storage/
  // obs_features_channels）逐字继承 pool_2x_v6_cpu.json，不改
}
```

若需要**对比度档**（浅/深点两档），当前无法通过键值表达——见2.4/2.6，需要代码改动才能加一个如 `hole_depth_range` 的键。

### 2.6 判定 + 最小代码改动清单（不实现）

**判定：纯 config 可以跑起来一个"点缺陷试点池"**（计数、半径范围、族选择、关闭 notch/crack 均可 config 表达，无需改代码即可生成一批"小洞"场景）。**但有 2 个真实感/精确度缺口 config 无法弥补**：

1. **对比度/深度**：`apply_defects` 硬编码全深度置零（`realism.py:106`），无深浅参数。
   - 改动：`realism.py::apply_defects` 增加可选参数 `hole_depth_range=(1.0,1.0)`（默认值保证旧配置字节级不变）；在洞累积处按每个洞独立抽样深度并做加权覆盖而非布尔OR，比如把 `defect`（bool累积）拆成一个 `float32` 深度累积数组，`defected = cov*(1.0 - depth_accum)`。预估 ~10-15 行，局限在 `apply_defects` 函数体（`realism.py:68-108`）。
2. **抗锯齿**：`apply_defects` 在 SSAA 降采样之后调用（`generate_training_pool.py:400 vs 415-419`），小半径(1-2px)洞是硬边。
   - 改动：`realism.py::irregular_blob`（或新增一个仅供洞用的软边变体）把 `rr <= boundary` 的硬阈值换成窄带过渡（如 `np.clip((boundary - rr)/aa_width, 0, 1)`，`aa_width≈0.5-1.0px`），返回 float 而非 bool；`apply_defects` 相应把 `defect |= mask`（布尔OR）改成 `np.maximum(defect, mask)`（float累积）。预估 ~15-20 行，局限在 `realism.py:27-108`，不涉及 `generate_training_pool.py` 或几何/SSAA管线。
3. **密度硬下限**（若 owner 需要"每景保证 ≥N 点"而不满足于"均匀分布、含0"）：
   - 改动：`realism.py::apply_defects` 计数公式 `int(rng.integers(0, round(max_holes*sev)+1))`（`realism.py:86`）加一个 `min_holes=0` 参数，改为 `rng.integers(min(min_holes, ceil), ceil+1)`。预估 ~3-5 行。

以上 3 项均**局限于 `tcforge/src/tcforge/realism.py` 内的 `apply_defects`/`irregular_blob`**，且都设计为新增可选参数、默认值保持旧行为字节级不变（因为 `generate_training_pool.py:417` 是原样 `**kwargs` 透传，新键只需加 JSON，不用碰调用点代码）。**不改动几何/SSAA/温度/融合管线**。

## 关键引用速查

- 缺陷调用点：`scripts/generate_training_pool.py:400-419`
- `apply_defects` 定义：`tcforge/src/tcforge/realism.py:68-108`
- `irregular_blob`：`tcforge/src/tcforge/realism.py:27-43`
- interior/boundary erosion：`tcforge/src/tcforge/realism.py:77-78`
- HR pitch 换算：`tcforge/src/tcforge/geometry.py:35-42`
- SSAA 降采样时点（downsample 早于 apply_defects）：`tcforge/src/tcforge/geometry.py:1607-1609,1682` + `scripts/generate_training_pool.py:400,415-419`
- isothermal coverage→温度：`tcforge/src/tcforge/realism.py:111-132`
- v6_cpu vs bench48 diff：程序化比较结果见 2.5（仅 `n_frames_per_scene`/`num_scenes`/`seed`/`output_dir`/`dataset`/`_comment`）
