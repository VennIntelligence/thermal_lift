# v7 噪声真实感升级 + 算子随机化 — 实现方案

> 2026-07-08，规划代理产出（只读调研，落点均经逐行核实）。依据：`todos/dataset.md` §3/§4
> （2026-07-07 实测表）、`scripts/audit_real_noise.py`（实测口径）、
> `tcforge/src/tcforge/{physics,realism}.py`、`scripts/generate_training_pool.py`、
> `algos/ep07_unet_sr/src/unet_sr/{dataset,config,solver_train}.py`。
> 状态：**待实现**（v7 集成期执行；顺序见 §6）。

## 0. 现状确认（调研结论，实现前提）

1. **v6 主线噪声路径**是 `noise_model.model == "field_vignette_stripe"`（`configs/synthetic/pool_2x_v6_cpu.json:87-93`），由 `generate_training_pool.py:488-497` 调 `realism.field_noise_burst`（`realism.py:190-211`）。burst 语义在函数内实现：`fixed = vignette + col_stripe`（(h,w)，广播到全 burst，多帧平均不掉），`grain = rng.normal(0, grain_c, size=(m,h,w))` 逐帧 i.i.d.。**四项噪声升级全部落在这条路径**。
2. `physics.make_noise` 的 `mixed`/`detector_realistic` 族内混合系数硬编码在 `physics.py:304-318`（0.55/0.25/0.75/0.35 与 0.45/0.30/0.50/0.25 等）；该族是 add_noise 路径（v6 主线不用，但 §4.2 要求顺带暴露）。
3. PSF 椭圆比硬编码 `physics.py:502-503`：`sigma * uniform(0.80, 1.20)`（x、y 各一次独立 draw）；airy 分支 496 行另有 `uniform(0.85, 1.20)`。
4. **RNG 流关键事实**（字节不变纪律的雷区）：
   - `generate_training_pool.py:479-487`：`noise_params_resolved` 对 **所有** 噪声模型（包括 field_vignette_stripe）都先用 `_uniform_range` 解析 `fpn_sigma_px`/`stripe_sigma_c`——这些 draw 消耗场景 rng。**这段一个字节都不能动**。
   - field 分支调用参数按 `vignette_c → stripe_c → (stripe_col_sigma 原样传入) → grain_c` 顺序 draw；函数内部按 `cy, cx → 2 个斜坡 uniform → csig → col normal(1,w) → grain normal(m,h,w)` 顺序 draw。
5. **σ-DR 的现成接线**：`TrainingConfig.solver_dc_psf_sigma_jitter_frac`（`config.py:125`，乘性 ±frac）→ CLI `--solver-dc-psf-sigma-jitter-frac`（`config.py:678-681`）→ `solver_train.py:252-254` 传给 `ThermalSRDataset(dc_psf_sigma_jitter_frac=...)` → `dataset.py:607-613` 在 `_add_burst_to_sample` 内对喂给 DC 的 `sample["psf_sigma_lr_px"/"psf_sigma_y_lr_px"]` 施加抖动，burst 像素保持真算子渲染；synth_eval loader 从不传抖动（`solver_train.py:250-251` 注释已声明）。**σ-DR 的绝对谎言完全复用这条链，只加档位不同的新旋钮**。benchmark 侧 `resolve_dc_sigma`/`resolve_neural_psf_params`（`run_stage2b_synth_benchmark.py:119-135`）是评测期整体 override 的语义参照（override 同时决定 x/y）。
6. 金样本模式参照：`tcforge/tests/test_realism.py:31-62`（golden npz + RNG sentinel draw + `tobytes()` 比对，8f863c9 先例）。**field_noise_burst 目前没有 golden——必须在改代码前先钉**。

---

## 1. 四项噪声升级（落点：`realism.field_noise_burst` + `generate_training_pool.py` field 分支）

### 1.1 新函数签名（realism.py）

```python
def field_noise_burst(burst, rng, *, vignette_c=0.13, stripe_c=0.028,
                      stripe_col_sigma=(2.5, 5.0), grain_c=0.10,
                      # v7 additions — all defaults OFF => zero extra rng draws,
                      # output AND stream bit-identical to legacy (golden-pinned):
                      row_stripe_c=0.0,            # (a) 行条纹 FPN 幅度, °C
                      stripe_row_sigma=(4.5, 6.5), # (a) 行向平滑 σ, px（仅 row_stripe_c>0 时 draw）
                      lowfreq_c=0.0,               # (b) 1/f^α 静态场幅度, °C
                      lowfreq_alpha=(1.7, 1.8),    # (b) 谱指数范围（仅 lowfreq_c>0 时 draw）
                      pixel_fpn_c=0.0,             # (d) 静态逐像素 FPN 幅度, °C
                      grain_ar1_rho=0.0):          # (c) grain 帧间 AR(1) lag-1 系数
```

**RNG 流保持策略**（硬约束）：每个新分量的 draw 都套在 `if <amp> > 0:` 守卫内；新 draw 统一插在 legacy 的 col-stripe draw 之后、grain draw 之前，固定顺序 **row → lowfreq → pixel_fpn**；AR(1) 不引入任何新 draw（见 1.4）。默认值下函数体执行路径与现行代码逐 draw 一致 → 输出与 rng 状态字节不变。

### 1.2 (a) 行条纹 FPN

- 实现（与列条纹对偶，burst 内固定）：
  ```python
  if row_stripe_c > 0:
      rsig = float(rng.uniform(*stripe_row_sigma)) if isinstance(stripe_row_sigma, (tuple, list)) else float(stripe_row_sigma)
      row = ndimage.gaussian_filter(rng.normal(size=(h, 1)).astype(np.float32), sigma=(rsig, 0))
      row_stripe = (row - row.mean()) / (row.std() + 1e-6)
      fixed += float(row_stripe_c) * row_stripe
  ```
- **单位/幅度对齐**：audit 口径中条纹幅度 = 平坦区 bg 残差（mean image 减 σ=25px 高斯背景）的行均值 profile 的 std。生成器中 unit-std profile × 系数 ⇒ profile std ≈ 系数，单位 °C 直接可比。实测行 0.049°C ⇒ v7 config 取 `[0.035, 0.06]`（宽域纪律，包夹实测 ±30%）。
- **相关长度对齐**：高斯平滑白噪的自相关 1/e 长度 = 2·σ_smooth（R(ℓ)∝exp(−ℓ²/4σ²)）。实测行 11px ⇒ σ_row=5.5 ⇒ `stripe_row_sigma=[4.5, 6.5]`（1/e 9–13px）。同一公式顺带校列条纹：实测 7px ⇒ σ_col=3.5，现默认 [2.5,5.0]（1/e 5–10px）保留为函数默认，v7 config 收窄至 `[2.5, 4.5]`；列幅度实测 0.065°C ⇒ v7 config `stripe_c=[0.05, 0.08]`（函数默认 0.028 不动）。

### 1.3 (b) 真 1/f^α 低频场

- **新 helper 落在 physics.py**（供 realism 与 make_noise 共用）：
  ```python
  def powerlaw_field(shape: tuple[int, int], alpha: float,
                     rng: np.random.Generator) -> np.ndarray:
      """Unit-std zero-mean 2D field with radial PSD ∝ f^-alpha (FFT synthesis:
      white noise → FFT → amplitude × f^(-alpha/2), DC bin zeroed → IFFT real part
      → _normalize_noise(·, 1.0)). float32."""
  ```
  一次 `rng.normal(size=shape)` draw（顺序可控）。频率网格用 `np.fft.fftfreq`，`f=sqrt(fx²+fy²)`，`f[0,0]` 置 0 幅。
- realism 侧接入（burst 内固定——它是 248 帧平均不掉的静态结构，audit 在 mean image 上量到）：
  ```python
  if lowfreq_c > 0:
      alpha = float(rng.uniform(*lowfreq_alpha)) if isinstance(lowfreq_alpha, (tuple, list)) else float(lowfreq_alpha)
      fixed += float(lowfreq_c) * physics.powerlaw_field((h, w), alpha, rng)
  ```
- **这是 opt-in 替换而非删除**：vignette（抛物碗+线性斜坡）保留为独立旋钮；v7 pilot 里若 lowfreq 与 vignette 低频功率重复计入，按 pilot 读数下调 `vignette_c`（候选 [0.06,0.12]），不在代码层耦合。
- **幅度对齐方式**：本地没有 5090 的 `summary.json`（含 PSD intercept），°C 幅度无法离线定死。预注册**校准程序**而非终值：pilot 生成 24 景，用 audit 同款 `radial_psd_slope`（fit 窗 [0.0125, 0.3] cyc/px，quadratic detrend + Hann）量合成 mean image，调 `lowfreq_c`（起始档 `[0.04, 0.10]`）使 α 落 [1.7,1.8] 且平坦区静态场总 std 对上实测。此项标记为**需 5090 summary.json 回填的开放参数**。
- 次级（可选，P2）：`make_noise` 加 `fpn_spectrum: Literal["gaussian","powerlaw"] = "gaussian"` + `fpn_alpha=1.77`，让 `_lowfreq_fpn` 走 powerlaw_field——非 v6 主线，可砍。

### 1.4 (c) 时序相关分量（AR(1) grain）

- **实现（零额外 draw，rho=0 走 legacy 分支保证 bit-exact）**：
  ```python
  grain = rng.normal(0.0, float(grain_c), size=burst.shape).astype(np.float32)  # legacy draw 原位不动
  if grain_ar1_rho > 0.0:
      rho = float(grain_ar1_rho); c = float(np.sqrt(1.0 - rho * rho))
      for t in range(1, m):
          grain[t] = rho * grain[t - 1] + c * grain[t]   # in-place，复用 legacy draws 作 innovations
  ```
  `e_0 = w_0` 已在稳态方差 ⇒ 全 burst 每帧边缘 std 恒 = `grain_c`（幅值语义不变，实测时序 σ=0.105°C 仍由 `grain_c=[0.08,0.12]` 覆盖），lag-1 自相关 = ρ。
- **burst 语义划分（最终状态，写进 docstring）**：
  - burst 内固定：vignette、列条纹、行条纹(新)、1/f^α 场(新)、逐像素 FPN(新)；
  - 逐帧：grain，帧间 AR(1) 相关(新)。多帧平均方差 ≈ σ²/M · (1+ρ)/(1−ρ)——这正是"直接影响多帧融合增益真实性"的机制，测试要验证（§5）。
- **档位**：实测 lag-1=0.63 ⇒ `grain_ar1_rho=[0.5, 0.7]`。注意 audit 的 lag-1 量在"减去逐像素时间均值的残差"上，drift（`apply_drift`，噪声之后施加）会推高全链路 lag-1；验收在**全链路生成的 pool burst** 上量（drift 分布照 v6），pilot 若量出 >0.7 再回调 ρ 下界，不动 drift。

### 1.5 (d) 静态逐像素 FPN

- 实现（白谱、无平滑、burst 内固定——按定义 248 帧平均不掉）：
  ```python
  if pixel_fpn_c > 0:
      fixed += float(pixel_fpn_c) * rng.normal(size=(h, w)).astype(np.float32)
  ```
- 单位直接对齐：实测"最平坦 patch 逐像素 std 0.05–0.09°C" ⇒ `pixel_fpn_c=[0.05, 0.09]`，v7 config 原样采用。
- 与 (b) 的边界：pixel FPN 是白谱高频端，powerlaw 是 1/f 段；两者叠加后 audit fit 窗内斜率仍应 ≈1.77（真实数据中两者共存且 R²=0.91，白底在 fit 窗内次要）——由 §5 验收测试钉住。

### 1.6 generate_training_pool.py 接线（field 分支，`:488-497`）

```python
if noise_model == "field_vignette_stripe":
    vignette = _uniform_range(rng, noise_cfg.get("vignette_c", 0.13), "noise_model.vignette_c")
    stripe   = _uniform_range(rng, noise_cfg.get("stripe_c", 0.028), "noise_model.stripe_c")
    grain    = _uniform_range(rng, noise_cfg.get("grain_c", 0.10), "noise_model.grain_c")
    extras: dict[str, Any] = {}
    if "row_stripe_c" in noise_cfg:      # 仅 config 显式给键才 draw —— 老 config 流不变
        extras["row_stripe_c"] = _uniform_range(rng, noise_cfg["row_stripe_c"], "noise_model.row_stripe_c")
        extras["stripe_row_sigma"] = noise_cfg.get("stripe_row_sigma", [4.5, 6.5])
    if "lowfreq_c" in noise_cfg:
        extras["lowfreq_c"] = _uniform_range(rng, noise_cfg["lowfreq_c"], "noise_model.lowfreq_c")
        extras["lowfreq_alpha"] = noise_cfg.get("lowfreq_alpha", [1.7, 1.8])
    if "pixel_fpn_c" in noise_cfg:
        extras["pixel_fpn_c"] = _uniform_range(rng, noise_cfg["pixel_fpn_c"], "noise_model.pixel_fpn_c")
    if "grain_ar1_rho" in noise_cfg:
        extras["grain_ar1_rho"] = _uniform_range(rng, noise_cfg["grain_ar1_rho"], "noise_model.grain_ar1_rho")
    lr_burst = _realism.field_noise_burst(
        lr_burst, rng, vignette_c=vignette, stripe_c=stripe,
        stripe_col_sigma=noise_cfg.get("stripe_col_sigma", [2.5, 5.0]),
        grain_c=grain, **extras)
    noise_params_resolved.update(extras)   # dict 更新不耗 rng；老 config 无键 → metadata 不变
```

顺序纪律：legacy 三个幅度先 draw（与现行参数求值顺序一致），extras 按固定键序后 draw。`stripe_row_sigma`/`lowfreq_alpha` 与 `stripe_col_sigma` 同款"范围直传、函数内 draw"。**479-487 行的 `noise_params_resolved` 预解析（含对 field 模型也 draw 的怪癖）原样保留。**

### 1.7 v7 config 预注册噪声块（写入未来 `pool_2x_v7.json`）

```json
"noise_model": {
  "model": "field_vignette_stripe",
  "vignette_c": [0.10, 0.16],
  "stripe_c": [0.05, 0.08],
  "stripe_col_sigma": [2.5, 4.5],
  "row_stripe_c": [0.035, 0.06],
  "stripe_row_sigma": [4.5, 6.5],
  "lowfreq_c": [0.04, 0.10],
  "lowfreq_alpha": [1.7, 1.8],
  "pixel_fpn_c": [0.05, 0.09],
  "grain_c": [0.08, 0.12],
  "grain_ar1_rho": [0.5, 0.7]
},
"detector_defects": {"enabled": false}
```

（注释版见正文：vignette pilot 后可能降 [0.06,0.12]；stripe/row/α/lag-1 各对应实测 0.065/0.049/1.77/0.63；lowfreq_c 为开放参数待 5090 intercept 回填；热像素保持关闭实证成立。）

---

## 2. 族内混合系数暴露（physics.py 304-318）

`make_noise`/`add_noise` 加参 `mix_weights: Mapping[str, float] | None = None`（默认 None = 现行常数，零行为差）。识别键与默认值（写进 docstring）：

| model | 键（默认） |
|---|---|
| `fpn_lowfreq` | `fpn_w=0.75, iid_w=0.25` |
| `column_stripe` | `stripe_default_scale=0.5, stripe_scale_cap=0.95` |
| `spatial_correlated` | `corr_w=0.70, iid_w=0.30` |
| `mixed` | `fpn_w=0.55, stripe_default_scale=0.25, stripe_scale_cap=0.75, iid_w=0.35` |
| `detector_realistic` | `fpn_w=0.45, corr_w=0.30, stripe_default_scale=0.20, stripe_scale_cap=0.50, iid_w=0.25, corr_sigma_factor=0.5, corr_sigma_min=1.0` |

实现：每分支 `mw = dict(defaults); mw.update(mix_weights or {})`，替换字面量。总残差仍被 `_normalize_noise(residual, sigma)` 锚到 `noise_sigma_c`，权重是相对量——docstring 必须写明。config 键：`noise_model.mix_weights`（dict），在 `generate_training_pool.py` 的 `add_noise` 分支（:499-510）透传 `mix_weights=noise_cfg.get("mix_weights")`。不改 draw 数量/顺序 ⇒ 流不变。

---

## 3. PSF 椭圆比暴露（physics.py 496/502-503 → config）

`sample_psf_parameters` 新签名：

```python
def sample_psf_parameters(*, seed=None, rng=None, sigma_range=(0.15, 0.55),
                          elliptical_probability=0.30, airy_probability=0.10,
                          elliptical_ratio_range=(0.80, 1.20),   # 现 502-503 硬编码
                          airy_ratio_range=(0.85, 1.20)):        # 现 496 硬编码
```

三处 `uniform(0.80,1.20)`/`uniform(0.85,1.20)` 改为 `uniform(*elliptical_ratio_range)`/`uniform(*airy_ratio_range)`（elliptical 分支 x、y 两次独立 draw 保持两次）。draw 次数与顺序不变 ⇒ 默认值下逐 bit 一致。

接线（`generate_training_pool.py:362-368`）：config 键 `psf_randomization.elliptical_ratio_range`（默认 `[0.80,1.20]`）、`psf_randomization.airy_ratio_range`（默认 `[0.85,1.20]`）。v7 训练范围维持默认（§3.2 只要求暴露）；σ∈[0.6,0.9] 及更宽椭圆比留 OOD 外推档，不进 v7 config。

---

## 4. σ-DR（训练管线侧，dataset.md §3.1，owner 裁决 #4：±[0.05,0.2]）

### 4.1 语义

渲染 σ 照常按景采样（`psf_randomization.sigma_range` 不动，生成器零改动）；**训练时**对喂给 DC 的 σ 加逐样本绝对谎言 δ：`|δ| ~ U(lo, hi)`（LR px），符号 Rademacher ±1 等概率，x/y 同一 δ（与 `--dc-sigma-override` 一样把"整体宽度信念"打偏；区别于现有乘性 `jitter_frac` 保各向异性比，加性谎言直接对齐 A3 剂量轴口径：ACL-060 在 σ 谎言 0.1 处 −0.157）。下限保护：`σ' = max(σ + δ, 0.05)`（场景 σ 最小 0.15，δ 最大 −0.2 会打穿零）。

### 4.2 落点（三个文件，完全平行于 Stage 1a jitter 的既有链路）

1. **`config.py`**（`TrainingConfig`，第 124-126 行旁）：
   ```python
   solver_dc_psf_sigma_lie_px: tuple[float, float] = (0.0, 0.0)
   ```
   CLI（678 行旁）：`--solver-dc-psf-sigma-lie-px LO HI`；`validate()`：`0 <= lo <= hi`；`hi > 0 and solver_dc_psf_sigma_jitter_frac > 0` 时报错（两种 σ 扰动互斥，保持剂量可解释）。
2. **`dataset.py`**：ctor 加 `dc_psf_sigma_lie_px: tuple[float, float] = (0.0, 0.0)`；`_add_burst_to_sample` 在 angle-jitter 块（614-615 行）之后、写 sample 键（618 行）之前插：
   ```python
   lie_lo, lie_hi = self.dc_psf_sigma_lie_px
   if lie_hi > 0:
       delta = float(rng.uniform(lie_lo, lie_hi)) * (1.0 if rng.integers(0, 2) else -1.0)
       sigma_x = max(sigma_x + delta, 0.05)
       sigma_y = max(sigma_y + delta, 0.05)
   ```
   复用同一 per-sample `rng`（590 行），守卫内才 draw ⇒ 默认路径流不变。
3. **`solver_train.py`**：`ThermalSRDataset(...)` 加 `dc_psf_sigma_lie_px=config.solver_dc_psf_sigma_lie_px`；`dr_label` off 判断补 `and config.solver_dc_psf_sigma_lie_px[1] == 0`，标签加 `psfσ-lie±[lo,hi]px`。**synth_eval loader 不传（保持精确算子，与 DR/no-DR 臂可比）**。

### 4.3 与 σ-hint 输入通道的边界（配套架构项，另立实验，不在本轮）

- 本轮 σ-DR **不加任何模型输入通道、不动 prox 架构**——只改 sample dict 里 DC 消费的标量。
- 边界约定（写入代码注释供后续实验遵守）：`sample["psf_sigma_lr_px"]` 语义 = "管线当前相信的 σ"（谎言后值）。未来 σ-hint 通道必须读**这同一个键**；真渲染 σ 仅存在于 `scene["metadata"]["psf_sigma_lr_px"]`，只许诊断用，禁止喂网络。
- 预注册档位：v7 复训臂 `--solver-dc-psf-sigma-lie-px 0.05 0.2`；对照臂不传。

## 5. 新测试清单

### 5.0 前置：测量函数下沉（让测试与审计同源）

从 `scripts/audit_real_noise.py` 提取纯函数到新模块 **`tcforge/src/tcforge/_noise_stats.py`**：`autocorr_1e_length`（63-76 行）、`radial_psd_slope`（79-126 行），另新增 `stripe_profiles(mean_img, bg_sigma_px=25.0)` 与 `lag1_autocorr_median(burst)`（照抄 audit main 口径）。`audit_real_noise.py` 改为 import（纯搬移；下次在 5090 重跑核对数字一致）。

### 5.1 金样本（改代码**前**生成，钉字节不变）

- `test_field_noise_default_path_matches_golden`：新 fixture `tcforge/tests/data/field_noise_golden_v1.npz`（全默认 + 显式 legacy kwargs 两 case，`tobytes()` + rng sentinel）。
- `test_make_noise_mix_weights_none_matches_golden`：`mixed`/`detector_realistic` 各一例。
- `test_sample_psf_parameters_default_ratio_matches_golden`：固定 seed 50 组参数序列一致。

### 5.2 逐项统计验收（tcforge/tests/test_realism.py 新增；全部用 §5.0 同源估计器）

| 测试 | 设置 | 断言 |
|---|---|---|
| `test_row_stripe_amplitude_and_corr_length` | 只开 `row_stripe_c=0.049, stripe_row_sigma=(5.5,5.5)` | row profile std=0.049±15%；1/e ∈[8,14]px；col 无泄漏 |
| `test_col_stripe_corr_length_formula` | 只开 `stripe_c`，σ=(3.5,3.5) | 1/e ∈[5,9]px（钉 2σ 映射） |
| `test_powerlaw_field_slope` | `powerlaw_field((160,160),1.77)` 多 seed | α=1.77±0.15、R²>0.85、std≈1 |
| `test_field_noise_composite_psd_slope` | v7 幅度全开 M=64 | α∈[1.5,2.0] |
| `test_grain_ar1_lag1_and_fusion_gain` | 只开 grain，ρ=0.63，M=248 | lag-1=0.63±0.05；逐帧 std=grain_c±10%；`std(mean)`≈`grain_c·sqrt((1+ρ)/((1−ρ)M))`±20% |
| `test_grain_ar1_rho_zero_bit_identical` | 同 seed，rho=0 vs 不传 | 逐字节一致 |
| `test_pixel_fpn_static_across_burst` | 只开 `pixel_fpn_c=0.07` M=32 | 时间 std≈0；空间 std=0.07±10%；平均不衰减 |
| `test_burst_semantics_partition` | 全开 | 固定/逐帧分量方差分解对账 |
| `test_make_noise_mix_weights_custom` | `mixed` 自定义权重 | 占比按权重移动；总 RMS 仍锚 noise_sigma_c |
| `test_psf_ratio_range_respected` | ratio=(0.5,0.6) 采 200 组 | 全落 [0.5,0.6]；draw 计数不变 |

### 5.3 ep07 侧（algos/ep07_unet_sr/tests/test_dataset.py 新增）

- `test_dc_sigma_lie_band_floor_and_default_off`：|δ|∈[0.05,0.2]、x/y 同 δ、floor 0.05 生效、burst/shifts 不变；lie=(0,0) 与现行逐值一致。
- `test_config_validate_lie_exclusive_with_jitter_frac`：互斥与 lo>hi 校验。

### 5.4 池级验收（pilot 阶段脚本，非 pytest）

新脚本 `scripts/audit_synth_noise.py`：24 景噪声 pilot 池（`save_lr_burst=true`）逐景平坦区跑与 `audit_real_noise.py` **完全同款**估计器（import `tcforge._noise_stats`），输出对照表（列/行条纹幅度与 1/e、α、lag-1、静态逐像素 std vs 实测六行）。验收线 = 实测 ±30%。`lowfreq_c`/`vignette_c` 终值在此闭环定。

## 6. 实现顺序建议

1. **钉 golden**（§5.1 三 fixture，从未改动的 HEAD 生成）——先于一切代码改动。
2. **`_noise_stats.py` 下沉** + `audit_real_noise.py` 改 import（纯搬移）。
3. **physics.py**：`powerlaw_field` + `mix_weights` + PSF ratio ranges（三个独立小 diff）。
4. **realism.py**：`field_noise_burst` 四项升级（依赖 3）。
5. **generate_training_pool.py**：field 分支 extras 接线 + 两个 ratio 键 + mix_weights 透传。
6. **tcforge 统计测试**（§5.2）全绿。
7. **ep07 σ-DR**：config.py → dataset.py → solver_train.py + §5.3 测试（与 1-6 无耦合，可并行）。
8. **24 景噪声 pilot**：`configs/synthetic/pool_2x_noise_pilot.json`（v6_cpu 复制 + §1.7 噪声块 + `save_lr_burst=true`，seed 不相交），5090 跑 `audit_synth_noise.py` → 定 `lowfreq_c`/`vignette_c` 终值、复核全链路 lag-1。
9. 终值合入 v7 主 config（与内容轴 composer、缺陷轴改动汇合后走 dataset.md §6 owner go 流程）。

红线复述：`generate_training_pool.py:479-487` 预解析 draw 不动；所有新键"config 无键 ⇒ 零 draw ⇒ 老池字节可复现"；bench48 按 v7 配方独立重生成；热像素 `detector_defects.enabled=false` 保持。
