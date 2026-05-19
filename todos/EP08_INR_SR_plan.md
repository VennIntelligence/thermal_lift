# EP08 — 深度学习轮廓增强：WIRE / SIREN / Deep Decoder 2× SR

> 状态：计划文档 v2.1（v2.0 基础上修正 forward-highpass 链路、Deep Decoder 架构、VRAM 估算等）
> 目标：在 255 帧 LWIR 微扫描数据上验证 INR-based 2× 轮廓重建，产出可与 EP06 SAA/IBP/MAP-TV 对比的指标
> 四方对比：**EP06 MAP-TV（经典优化）| SIREN（正弦 INR）| WIRE（Gabor INR）| Deep Decoder（CNN decoder）**

---

## 1. 决策：为什么选这三个算法

### 1.1 科学问题分层

三个算法 + EP06 baseline 构成嵌套的对照实验，回答三个递进的科学问题：

```
问题 1: 深度隐式 prior > 经典 TV 正则？
         → SIREN / WIRE / Deep Decoder   vs   EP06 MAP-TV

问题 2: INR 连续表示 > CNN decoder 离散表示？
         → SIREN / WIRE   vs   Deep Decoder

问题 3: Gabor 边缘敏感激活 > 通用正弦激活？
         → WIRE   vs   SIREN（最纯净的 activation ablation）
```

### 1.2 方法选型

| 方法 | 核心优势 | 风险 | 选择理由 |
|------|----------|------|----------|
| **WIRE** | CVPR 2023，Gabor 激活天然边缘敏感，178★ 完整代码含 multi-image SR 示例 | 需将激活函数适配到 2D highpass 场景 | 主线：与芯片轮廓增强目标高度匹配 |
| **SIREN** | NeurIPS 2020，INR 鼻祖，2000+★，广泛验证 | 正弦激活可能过拟合高频噪声 | 与 WIRE 共享 99% 代码，只改激活函数，构成最纯净 ablation |
| **Deep Decoder** | ICLR 2019，低参数 decoder，101★ | 表达能力可能过弱 | 独立对照：回答"INR 是否真的比 CNN decoder 更好" |

**不做的算法**（已在调研报告中排除）：

| 方法 | 排除原因 |
|------|----------|
| SuperF | repo 仅 3★，方法论粗糙（ReLU MLP + boxcar PSF），面向卫星/手持 burst 而非 LWIR 物理逆问题；只借鉴 shared-INR 思想 |
| BurstM | 需要合成 HR-LR burst 训练集，EP07 合成数据尚未完成 |
| DiffPIR/DAPS | 需要合成 chip-specific diffusion prior，依赖 EP07 后续训练 |
| 自然图像预训练 SR | 不适合单通道 highpass LWIR 芯片结构图 |
| 监督式 Real-ESRGAN/SwinIR | 需要大量 paired HR-LR 训练集，本项目无此条件 |

### 1.3 WIRE vs SIREN 的工程共享

SIREN 和 WIRE 是同结构 MLP，**唯一区别是激活函数**：

| 维度 | SIREN | WIRE |
|------|-------|------|
| 网络结构 | MLP 5 层 × 256 hidden | 完全相同 |
| 激活函数 | `sin(ω₀ · Wx + b)` | `Gabor(Wx + b)` |
| 频率控制 | `ω₀` 标量 | `ω₀` + `σ` + 方向参数 |
| 初始化 | SIREN 专用初始化 | Gabor 专用初始化 |
| Trainer / Forward / Data / Eval | **完全共享** | **完全共享** |

因此 SIREN 的**额外工程量仅为一个 `SirenLayer` 类**。

---

## 2. 输入数据

| 数据 | 位置 | 说明 |
|------|------|------|
| 主 session LR burst | `data/processed/` 中主 session 255 帧 | 默认输入 |
| 位移配置 | `configs/stage_calibration.json` + EP05 refined shifts | 47.6° 旋转角（stage-pixel 旋转角，非 chip 主轴角） |
| Forward model 参考 | `algos/ep06_sr_poc/src/common/forward_model.py` | 必须逐参数复刻为 PyTorch |
| Highpass 参考 | `algos/ep06_sr_poc/src/common/data_loader.py` `highpass_preprocess()` | 注意 mode 与 forward model 不同 |
| EP06 baseline 结果 | EP06 输出的 SAA / IBP / MAP-TV 结果 | 作为参照 |

---

## 3. 目录结构

```
thermal_lift/
├── algos/
│   └── ep08_inr_sr/                          # 独立 UV 项目
│       ├── pyproject.toml                     # pytorch, numpy, scipy, matplotlib
│       ├── .python-version                    # >=3.11
│       ├── src/ep08/
│       │   ├── __init__.py
│       │   ├── forward.py                     # P0: PyTorch 版 forward operator
│       │   ├── highpass.py                    # P0: PyTorch 版 highpass 预处理（独立于 forward）
│       │   ├── models/
│       │   │   ├── siren.py                   # P0: SIREN INR (Sine activation)
│       │   │   ├── wire.py                    # P0: WIRE INR (Gabor activation)
│       │   │   └── deep_decoder.py            # P1: Deep Decoder CNN
│       │   ├── coords.py                      # P2: chip-frame 坐标旋转（可选实验）
│       │   ├── geometry_losses.py             # P2: binary / rectilinear / angle loss（可选实验）
│       │   ├── trainer.py                     # P0: 共享训练循环 + hold-out + early stopping
│       │   ├── metrics.py                     # P0: 统一评估指标
│       │   └── utils.py
│       ├── configs/
│       │   ├── siren.yaml
│       │   ├── wire.yaml
│       │   └── deep_decoder.yaml
│       ├── scripts/
│       │   ├── train_siren.py                 # P0
│       │   ├── train_wire.py                  # P0
│       │   ├── train_deep_decoder.py          # P1
│       │   └── eval_all.py                    # P1: 统一评估脚本
│       └── tests/
│           ├── test_forward.py                # 必须：与 EP06 NumPy 等价性
│           ├── test_highpass.py               # 必须：与 EP06 highpass 等价性
│           ├── test_siren.py
│           ├── test_wire.py
│           └── test_deep_decoder.py
│
├── notebooks/
│   └── ep08_inr_sr/
│       ├── fragments/
│       │   ├── manifest.txt
│       │   ├── 01_setup.py                    # 环境说明 + 数据加载
│       │   ├── 02_forward_validation.py       # PyTorch forward 等价性验证
│       │   ├── 03_siren_results.py            # SIREN 实验结果
│       │   ├── 04_wire_results.py             # WIRE 实验结果
│       │   ├── 05_deep_decoder_results.py     # Deep Decoder 结果
│       │   └── 06_four_way_comparison.py      # 四方法 + EP06 对比
│       └── ep08_inr_sr.ipynb                  # 构建产物（.gitignore）
```

---

## 4. 实施计划

> ⚠️ **核心原则**：串行门控——每个阶段必须通过验证门控后才进入下一阶段。不做并行铺开。

### 阶段 0：基础设施

**目标**：建立 EP08 的共同地基。三个算法共享同一套 forward model、highpass preprocessing、数据加载、hold-out split 和指标评估。

#### 4.0.1 PyTorch Forward Operator（P0）

从 EP06 `algos/ep06_sr_poc/src/common/forward_model.py` **逐参数复刻**核心逻辑。

```python
class ForwardOperator(nn.Module):
    """PyTorch 可微分 forward operator.

    对应 EP06 的 ObservationOperator。
    流程: Shift → Gaussian Blur (HR space) → Sample to LR grid
    注意: 不含 highpass，highpass 是独立的 LR 预处理步骤。
    """
    def __init__(self, hr_shape, lr_shape, shifts, psf_sigma=1.0, scale=2):
        # psf_sigma: LR 像素单位，内部转换为 HR 像素: sigma_hr = psf_sigma * scale
        # shifts: (N, 2) tensor，EP05 convention: (dx, dy) LR-pixel
        # mode: "constant"（与 EP06 forward_model.py L103 一致）
        pass

    def forward(self, x_hr, index):
        """预测第 index 帧的 raw LR 观测。"""
        # 1. Gaussian blur in HR space: sigma_hr = psf_sigma * scale
        # 2. Sample HR→LR: map_coordinates equivalent, mode="constant", cval=0.0
        pass

    def adjoint(self, y_residual, index):
        """将 LR 残差回投到 HR grid。"""
        pass
```

**关键约束（必须与 EP06 逐字匹配）**：

| 参数 | EP06 值 | EP08 必须 |
|------|---------|-----------|
| Gaussian blur mode | `mode="constant"` ([forward_model.py L112](file:///home/ujs/mycode/thermal_lift/algos/ep06_sr_poc/src/common/forward_model.py#L112)) | `mode="constant"` |
| Gaussian blur cval | `cval=0.0` | `cval=0.0` |
| PSF sigma 输入单位 | LR 像素（内部 `sigma_hr = psf_sigma * scale`）| 相同 |
| PSF sigma 默认值 | `1.0` LR px | `1.0` LR px |
| Sampling interpolation | bilinear (`order=1`) | bilinear |
| Sampling mode | `mode="constant"` | `mode="constant"` |
| Scale | `2`（硬编码） | `2` |

> ⚠️ **PyTorch 采样实现注意**：EP06 使用 `scipy.ndimage.map_coordinates`，坐标为绝对像素单位 `(row, col)`。PyTorch 的 `F.grid_sample` 使用归一化 `[-1, 1]` 坐标，且 `align_corners` 参数影响映射。**推荐不使用 `grid_sample`**，而是手写 bilinear interpolation（参考 EP06 `_sample_reference_to_lr` L49–56），以保证数值等价性更容易验证。
>
> EP06 采样坐标计算（关键）：
> ```python
> # shift (dx, dy) 是 LR-pixel 单位，乘 scale 后变为 HR-pixel 偏移
> yy = scale * (np.arange(h_lr) + dy)
> xx = scale * (np.arange(w_lr) + dx)
> ```

#### 4.0.2 Highpass Preprocessing（P0，独立模块）

从 EP06 `algos/ep06_sr_poc/src/common/data_loader.py` `highpass_preprocess()` 复刻。

```python
def highpass_preprocess(frames, sigma_bg=5.0, mode="nearest"):
    """LR 帧减去高斯背景，生成 signed structure map.

    注意: mode="nearest" 与 forward operator 的 mode="constant" 不同！
    这是 EP06 的设计选择：
    - forward operator 用 constant（边界外补零）
    - highpass 背景估计用 nearest（边界外延伸最近值）
    """
    # frame - gaussian_filter(frame, sigma=sigma_bg, mode=mode)
```

**关键约束**：

| 参数 | EP06 值 | EP08 必须 |
|------|---------|-----------|
| Highpass Gaussian mode | `mode="nearest"` ([data_loader.py L124](file:///home/ujs/mycode/thermal_lift/algos/ep06_sr_poc/src/common/data_loader.py#L124)) | `mode="nearest"` |
| sigma_bg | `5.0` LR px | `5.0` LR px |

> ⚠️ **前车之鉴**：EP08 v1.0 把 forward operator 和 highpass preprocessing 的 mode 混为一谈，全写成 `mode="nearest"`。这会导致与 EP06 的数值等价性测试失败。两者必须独立实现、独立配置。

#### 4.0.3 Offset Correction / Raw-Control Track（P0）

从 EP06 `data_loader.py` `offset_correction()` 复刻。所有 SR 实验必须同时产出两个 track：

| Track | 数据域 | 用途 |
|-------|--------|------|
| **Highpass track** | `frame - gaussian_bg` signed field | 主重建目标，边缘增强 |
| **Raw-control track** | `frame - median(frame)` offset-corrected | 验证中心区域、内部轮廓、排除"只增强边缘"假象 |

#### 4.0.4 Hold-out Split（P0）

```python
def build_train_val_split(frames, shifts, val_ratio=0.2, seed=42):
    """
    位移相位分层抽样 split（非简单随机抽样）：

    策略：
    1. 计算每帧位移的亚像素相位: phase = (shift % 1.0)，量化为 4×4 网格 bin
    2. 从每个 bin 中按比例抽取 val_ratio 帧
    3. 如果某 bin 帧数不足（<3），整 bin 放入 train 集

    这避免了简单随机抽样可能导致某些位移方向完全没有 hold-out 帧的问题。

    Returns:
        train_indices, val_indices, val_mask (bool array)
    """
```

默认：`val_ratio=0.2`（约 51 帧），后续可调。

#### 4.0.5 统一指标 Pipeline（P0）

所有方法必须通过同一套指标评估。**五项全过才算通过门控**：

| 指标 | 计算方式 | 方向 | 门控阈值 |
|------|----------|------|----------|
| **Hold-out residual** | `mean(‖A_k(x_hr) - y_k‖² / σ_n²)` over val frames | 越低越好 | ≤ EP06 MAP-TV × 1.1 |
| **Split-half NRMSE** | 两半帧分别重建，求像素级 NRMSE | 越低越好 | ≤ EP06 MAP-TV |
| **Artifact score** | pin 区域 Laplacian 能量（见下文定义） | 越低越好 | ≤ EP06 MAP-TV |
| **Raw-control agreement** | offset-corrected 域 ROI 对比 | 视觉一致 | 目视检查 + 结构 SSIM |
| **Pin 区域无伪线** | 目视检查 + gradient 方向分布 | 无非物理直线 | 目视检查 |

**Artifact Score 定量定义**：

```python
def artifact_score(x_hr, pin_mask):
    """pin 区域（已知应平坦）的 Laplacian 能量。

    物理依据：pin/slot 内部在 highpass 域应接近零，
    非物理高频纹理会使 Laplacian 能量异常升高。
    """
    laplacian = ndimage.laplace(x_hr)
    return np.mean(laplacian[pin_mask] ** 2)
```

> `pin_mask` 需在阶段 0 中手动标注（一次性工作），覆盖芯片 pin/slot 的平坦内部区域。

> **为什么不能只看 hold-out residual**：AGENTS.md 硬教训 #2 "回投残差 ≠ 清晰度"。一个方法可以 hold-out residual 很低但边缘没有真正变窄（过度平滑），也可以边缘很锐但 artifact 严重（过拟合噪声）。五项指标从不同维度交叉验证。

#### 阶段 0 验证门控

| 验证项 | 通过标准 | 阻断条件 |
|--------|----------|----------|
| Forward 等价性 | PyTorch forward vs EP06 NumPy forward，逐像素 max abs error < 1e-5 | 任何帧 error > 1e-4 |
| Highpass 等价性 | PyTorch highpass vs EP06 highpass，逐像素 max abs error < 1e-5 | 任何帧 error > 1e-4 |
| Hold-out split 可复现 | 固定 seed=42 后 split 结果 bit-exact | 不可复现 |
| 指标 pipeline smoke test | 对 EP06 MAP-TV 结果跑一遍，数值合理 | 任何指标 NaN 或明显异常 |
| Raw-control track | offset correction 与 EP06 输出一致 | 不一致 |

**阻断意味着：不解决就不进入阶段 1。**

#### 4.0.6 EP06 Baseline 数值锚定（P0）

所有门控阈值引用 EP06 MAP-TV 的精确数值，不使用硬编码倍率。在阶段 0 中必须对 EP06 MAP-TV 结果跑一遍五项指标 pipeline，记录到配置文件：

```json
// configs/ep06_baseline_metrics.json
{
  "holdout_residual": null,      // 阶段 0 填入实际值
  "split_half_nrmse": null,
  "artifact_score": null,
  "raw_control_ssim": null,
  "boundary_width_px": null,
  "source": "EP06 MAP-TV, psf_sigma=1.0, 255 frames"
}
```

后续所有门控条件（如 "≤ EP06 MAP-TV × 1.1"）从此文件读取 baseline 值，确保阈值有数据支撑。

---

### 阶段 1：WIRE + SIREN 小规模验证

**目标**：用 32 帧代表性子集 + 中心 256×256 patch，验证 INR 框架可行性，同时完成 WIRE vs SIREN activation ablation。

#### 4.1.1 SIREN 核心实现

```python
class SirenLayer(nn.Module):
    """正弦激活层 (Sitzmann et al., NeurIPS 2020)."""

    def __init__(self, in_features, out_features, omega_0=30.0, is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        # SIREN 初始化 (Sitzmann 2020, Appendix 1.5)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / in_features) / omega_0,
                    np.sqrt(6 / in_features) / omega_0
                )
            # bias 初始化：与 weight 相同范围（原论文实验中也很重要）
            if self.linear.bias is not None:
                bound = 1 / in_features if is_first else np.sqrt(6 / in_features) / omega_0
                self.linear.bias.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SirenINR(nn.Module):
    """SIREN-based shared INR for multi-frame SR."""

    def __init__(self, hidden_dim=256, n_layers=5, omega_0=10.0):
        super().__init__()
        self.net = nn.Sequential(
            SirenLayer(2, hidden_dim, omega_0=omega_0, is_first=True),
            *[SirenLayer(hidden_dim, hidden_dim, omega_0=omega_0) for _ in range(n_layers - 2)],
            nn.Linear(hidden_dim, 1)  # 最后一层无激活
        )

    def forward(self, coords):
        # coords: (N, 2) 归一化坐标
        return self.net(coords).squeeze(-1)
```

**关键超参**：`omega_0` 控制频率带宽。建议从 `omega_0=10`（保守）开始，逐步增到 30。过大的 `omega_0` 会导致高频过拟合。

#### 4.1.2 WIRE 核心实现

```python
class GaborLayer(nn.Module):
    """Gabor 小波激活层 (Saragadam et al., CVPR 2023)."""

    def __init__(self, in_features, out_features, omega_0=10.0, sigma_0=10.0,
                 is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.sigma_0 = sigma_0
        self.linear = nn.Linear(in_features, out_features)
        self.linear_freq = nn.Linear(in_features, out_features)
        # WIRE 初始化参考 official repo
        # ⚠️ 注意：原论文使用 complex Gabor: exp(-σ²‖Wx‖²/2) · exp(jω₀·Wx) 取实部
        # 以下为 official repo 的简化实现（实数版），实现前务必核对 wire repo 的精确公式

    def forward(self, x):
        lin = self.linear(x)
        freq = self.linear_freq(x)
        # 实数 Gabor: Gaussian envelope × sinusoidal carrier
        return torch.exp(-self.sigma_0 * lin ** 2) * torch.sin(self.omega_0 * freq)


class WireINR(nn.Module):
    """WIRE-based shared INR for multi-frame SR."""

    def __init__(self, hidden_dim=256, n_layers=5, omega_0=10.0, sigma_0=10.0):
        super().__init__()
        self.net = nn.Sequential(
            GaborLayer(2, hidden_dim, omega_0=omega_0, sigma_0=sigma_0, is_first=True),
            *[GaborLayer(hidden_dim, hidden_dim, omega_0=omega_0, sigma_0=sigma_0)
              for _ in range(n_layers - 2)],
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, coords):
        return self.net(coords).squeeze(-1)
```

#### 4.1.3 INR 输出语义与 Forward-Highpass 链路

> ⛔ **v2.0 勘误**：原 `forward_loss` 伪代码写成 `highpass(forward(x_hr))`，但 forward（shift+blur+downsample）与 highpass（减高斯背景）**不对易**（因边界条件 `mode=constant` vs `mode=nearest` 不同）。必须明确链路方向。

**INR 输出语义：HR highpass field（signed structure map）**。

理由：
1. INR 不需要学习帧间温度 offset（已被 highpass 去除），降低学习难度
2. 避免 forward-highpass 交换顺序引入的边界不一致
3. Forward loss 直接在 highpass 域比较，链路更简洁

**正确的 Forward Loss 链路**：

```python
def forward_loss(model, forward_op, train_hp_frames, coords_grid,
                 sigma_noise=0.0724):
    """
    共享 data loss，SIREN 和 WIRE 完全相同。

    链路：INR → HR highpass field → forward(shift, blur, downsample) → LR 预测
    对比：LR 预测 vs LR highpass 观测

    x_hr_hp = model(coords_grid)       # INR 输出 HR highpass field
    for each train frame k:
        y_pred_k = forward_op(x_hr_hp, shift_k)   # shift+blur+downsample
        loss += ||y_pred_k - y_hp_k||^2 / (2 * sigma^2)

    注意：
    - y_hp_k = highpass_preprocess(raw_frame_k) 在训练前预计算
    - forward_op 内部仅做 shift+blur+downsample，不含 highpass
    - highpass 与 forward 不交换顺序
    """
```

#### 4.1.4 渐进式扩展策略

```
Step 1: 32 帧 + 256×256 patch  → 验证框架可行性
Step 2: 64 帧 + 全分辨率       → 扩大覆盖
Step 3: 128 帧 + 全分辨率      → 正式结果候选
Step 4: 255 帧 + 全分辨率      → 最终提交
```

**阶段 1 只做 Step 1**，Step 2–4 在阶段 3 中按门控结果决定。

#### 阶段 1 验证门控

| 验证项 | 通过标准 | 阻断条件 |
|--------|----------|----------|
| WIRE hold-out residual | 不高于 EP06 MAP-TV × 1.1（在同等 patch 上） | > 1.5× MAP-TV |
| SIREN hold-out residual | 不高于 EP06 MAP-TV × 1.1 | > 1.5× MAP-TV |
| WIRE vs SIREN 对比 | 两者在 5 项指标上有可比较的结果 | 任一方法 NaN 或 diverge |
| 无明显伪线 | pin 区域视觉检查 | 出现不合物理的直线或纹理 |
| Raw-control 一致性 | offset-corrected 域 ROI 与 highpass 域结构一致 | 明显矛盾 |
| Split-half 稳定性 | 两半帧重建的强边缘 Chamfer < 0.5 px | > 1.0 px |
| 训练收敛 | loss 曲线单调下降且 hold-out 不发散 | hold-out 持续上升 |

**阻断意味着：回到阶段 0 检查 forward model，或调整 INR 超参后重试。**

---

### 阶段 2：Deep Decoder 对照

**目标**：在与 WIRE/SIREN 完全相同的配置下，用 Deep Decoder 回答"INR 连续表示 > CNN decoder 离散表示？"

#### 4.2.1 Deep Decoder 实现

> ⛔ **v2.0 勘误**：原实现使用一步 Upsample + 全 1×1 卷积，本质上是逐像素 MLP（无空间感受野），不是原论文 (Heckel & Hand 2019) 的逐级上采样 + 通道缩减结构。已修正为正确架构。

```python
class DeepDecoder(nn.Module):
    """逐级上采样 CNN decoder (Heckel & Hand, ICLR 2019).

    与 INR 的核心区别：
    - INR: 坐标 (u,v) → 像素值（连续函数）
    - Deep Decoder: 固定噪声 z → 逐级 2x 上采样 → 图像（CNN decoder）
    - 低参数量来自逐级上采样 + 通道瓶颈，天然防止过拟合噪声

    架构：z(C₀,H₀,W₀) → [Upsample2x → Conv1×1 → BN → ReLU] × N → Conv1×1 → (1,H,W)
    """

    def __init__(self, channels=(128, 128, 64, 32, 16), output_shape=(960, 1280)):
        super().__init__()
        self.output_shape = output_shape

        # 计算初始 feature map 尺寸：使得 N 次 2x 上采样后 ≥ output_shape
        n_ups = len(channels) - 1  # 上采样次数
        h0 = int(np.ceil(output_shape[0] / (2 ** n_ups)))
        w0 = int(np.ceil(output_shape[1] / (2 ** n_ups)))

        # 构建逐级上采样 decoder
        layers = []
        for i in range(n_ups):
            layers += [
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(channels[i], channels[i + 1], 1),  # 1×1 通道缩减
                nn.BatchNorm2d(channels[i + 1]),
                nn.ReLU(),
            ]
        layers.append(nn.Conv2d(channels[-1], 1, 1))  # 输出单通道
        self.decoder = nn.Sequential(*layers)

        # 固定随机输入（不参与优化）
        self.register_buffer('z', torch.randn(1, channels[0], h0, w0))

    def forward(self):
        x = self.decoder(self.z)
        # Crop to exact output_shape（上采样后可能比目标略大）
        return x[:, :, :self.output_shape[0], :self.output_shape[1]].squeeze(0).squeeze(0)
```

> **参数量估计**：channels=(128,128,64,32,16) + 4 级上采样 → 约 30K 参数，远低于 INR 的 ~330K 参数。这种不对称正是 Deep Decoder 的设计意图：用极低的表达能力天然防止过拟合。

#### 4.2.2 与 INR 的对比设计

两者必须在**完全相同的配置**下训练：

| 配置 | SIREN / WIRE | Deep Decoder | 是否相同 |
|------|-------------|--------------|:---:|
| 帧数 | 32 帧子集 | 32 帧子集 | ✅ |
| Patch | 256×256 中心 | 256×256 中心 | ✅ |
| Hold-out split | 同一 val_mask | 同一 val_mask | ✅ |
| Forward model | 同一 ForwardOperator | 同一 ForwardOperator | ✅ |
| Data loss | L2 / σ_n | L2 / σ_n | ✅ |
| Geometry loss | **关闭** | **关闭** | ✅ |
| 优化器 | Adam | Adam | ✅ |
| 随机种子 | 42 | 42 | ✅ |

**只改变的变量**：网络结构（INR MLP vs CNN Decoder）。

#### 4.2.3 需要回答的核心问题

| 问题 | 判断依据 |
|------|----------|
| INR > CNN decoder 吗？ | SIREN/WIRE hold-out residual & split-half NRMSE < Deep Decoder |
| Deep Decoder 是否比 U-Net DIP 更稳定？ | Deep Decoder artifact score < 可选 DIP U-Net 参考 |
| 哪个方法最不容易 hallucinate？ | pin 区域目视检查 + artifact score |

#### 阶段 2 验证门控

| 验证项 | 通过标准 | 阻断条件 |
|--------|----------|----------|
| Deep Decoder hold-out residual | 有合理数值（不要求最优） | diverge 或 NaN |
| 三方法可比 | SIREN/WIRE/DeepDecoder 在 5 项指标上有完整结果 | 任一方法无法产出结果 |
| 四方对比表 | 与 EP06 MAP-TV 放在同一指标表中 | 缺少 EP06 baseline |
| Activation ablation 结论 | WIRE vs SIREN 有明确优劣或平手结论 | 结果矛盾或不可解释 |
| INR vs CNN 结论 | INR (SIREN/WIRE) vs Deep Decoder 有明确优劣结论 | 结果矛盾或不可解释 |

**阶段 2 的输出是一份小规模四方对比报告**，决定哪个方法进入阶段 3 做全量扩展。

---

### 阶段 3：赢家扩展到全量

**目标**：只对阶段 2 选出的赢家方法做渐进扩展，从 32 帧 patch 到 255 帧全分辨率。

#### 4.3.1 扩展规则

**只扩展同时满足以下条件的方法**：

1. 阶段 2 五项指标全部通过门控
2. 在三个方法中综合排名第一（或与第一无显著差异）
3. 无 pin 区域伪线

如果多个方法同样优秀，**最多扩展两个**（避免资源分散）。

#### 4.3.2 渐进扩展步骤

```
Step 2: 64 帧 + 全分辨率 (640×480 → 1280×960)
    ├── 验证：hold-out residual 不上升
    ├── 验证：无新增 artifact
    └── 验证：raw-control 一致性

Step 3: 128 帧 + 全分辨率
    ├── 验证：同上
    ├── 额外：与 Step 2 对比边缘宽度
    └── 额外：split-half 稳定性

Step 4: 255 帧 + 全分辨率
    ├── 验证：同上 + 全部 5 项指标
    ├── 额外：多 seed (3 次) 稳定性
    └── 这是最终提交结果
```

**每个 Step 之间有检查点**：如果 hold-out residual 开始上升或 artifact score 恶化，停止扩展，分析原因。

#### 4.3.3 可选实验：Geometry Loss Ablation

**只在阶段 3 的 Step 4 完成且通过门控后**，才考虑加入 geometry loss：

| 几何 loss | 公式 | 用途 | 风险 |
|-----------|------|------|------|
| Binary loss | `mean(m * (1 - m))` | 鼓励二值分离 | 可能强迫模型生成"像芯片"的结构 |
| Sparse edge | `‖∇m‖₁` | 鼓励平坦区 + 稀疏边缘 | 可能过度平滑内部结构 |
| Angle loss | `sin²(2(φ_edge - φ_chip))` | 鼓励边缘方向集中 | 需要已知 chip 主轴角 |

**Geometry loss 的门控**：
- 加入后 hold-out residual 不能变差（> 1.05× 无 geometry loss 版本）
- 加入后 artifact score 不能变差
- 加入后 raw-control agreement 不能变差
- 先在 TCForge 合成数据上验证有效，再用于真实数据

#### 4.3.4 可选实验：坐标旋转

**47.6° 是 stage-pixel 旋转角，不是已验证的 chip 主轴角。** 光学和红外未配准（AGENTS.md），不能默认芯片内部结构轴等于 stage 轴。

如果要做坐标旋转实验：

1. 先在不旋转的情况下完成 Step 4 + 门控
2. 旋转 47.6° 作为对照实验
3. 如果旋转后 hold-out residual 改善 > 5%，说明结构确实接近 stage 轴对齐
4. 如果改善 < 5% 或变差，说明 47.6° 不是 chip 主轴角，不应使用
5. 可选：将旋转角作为可学习参数（但增加过拟合风险）

#### 阶段 3 验证门控

| 验证项 | 通过标准 | 阻断条件 |
|--------|----------|----------|
| 255 帧 hold-out residual | ≤ EP06 MAP-TV × 1.1 | > 1.2× MAP-TV |
| 255 帧 split-half NRMSE | ≤ EP06 MAP-TV | > 1.3× MAP-TV |
| 255 帧 artifact score | ≤ EP06 MAP-TV | > 1.2× MAP-TV |
| Raw-control agreement | 目视 + 结构 SSIM ≥ 0.9 | SSIM < 0.8 |
| Pin 区域无伪线 | 目视检查 + gradient 方向分布 | 出现非物理直线 |
| 多 seed 稳定性 | 3 次运行的 hold-out residual std < 5% | std > 15% |
| 边缘宽度 | 强外轮廓 ESF 10-90% 宽度 < bicubic | ≥ bicubic |

---

### 阶段 4：统一评估与报告

**目标**：产出可供同事审核的完整四方对比报告。

#### 4.4.1 最终四方对比表

| 指标 | EP06 MAP-TV | SIREN | WIRE | Deep Decoder | 方向 |
|------|:-----------:|:-----:|:----:|:------------:|:----:|
| Hold-out residual | baseline | — | — | — | ↓ |
| Split-half NRMSE | baseline | — | — | — | ↓ |
| Artifact score | baseline | — | — | — | ↓ |
| Boundary width (px) | baseline | — | — | — | ↓ |
| P95 gradient | baseline | — | — | — | 参考 |
| Raw-control SSIM | baseline | — | — | — | ↑ |
| 运行时间 | baseline | — | — | — | 参考 |
| 多 seed std | — | — | — | — | ↓ |

#### 4.4.2 必须回答的科学问题

| 问题 | 回答方式 |
|------|----------|
| 深度方法是否超过经典天花板？ | best(SIREN/WIRE/DeepDecoder) vs MAP-TV 五项指标 |
| INR > CNN decoder？ | SIREN/WIRE vs Deep Decoder |
| Gabor > Sine？ | WIRE vs SIREN（唯一变量：激活函数） |
| 深度方法是否引入新 artifact？ | artifact score + pin 区域视觉检查 |
| 结果是否可复现？ | 多 seed + 固定配置 + 代码完整 |

#### 4.4.3 Notebook 报告要求

遵循 AGENTS.md notebook 展示规范：

1. **每个方法至少 4 张 ROI 对比图**：
   - 强外轮廓区域（highpass + raw-control）
   - Pin/slot 内部区域（highpass + raw-control）
   - 边缘 ESF 剖面（每个方法叠在同一图上）
   - Residual map（验证残差无边缘结构）

2. **每个图表必须附解读**：
   - 图表说明：展示什么数据
   - 数据分布/模式：注意什么
   - 核心发现：得出什么结论

3. **图片显示规范**：
   - `save_fig` 使用 `savefig_academic(fig, path)`（默认 `close=True`）
   - 绝不 `close=False`，绝不在 `save_fig` 后调 `plt.show()`
   - Highpass 图必须说明白色=零变化、红蓝=正负响应

#### 阶段 4 验证门控

| 验证项 | 通过标准 | 阻断条件 |
|--------|----------|----------|
| 四方对比表完整 | 所有方法所有指标有数值 | 缺失指标 |
| 三个科学问题有明确回答 | 每个问题有数据支持的结论 | 无法判断 |
| Notebook 可独立执行 | `Restart & Run All` 通过 | 执行中断 |
| 所有图表有解读 | 每个输出后有 Markdown 解释 | 裸输出 |
| 代码可复现 | 固定 seed + 配置文件 + 所有依赖记录 | 不可复现 |

---

## 5. 超参数配置参考

### 5.1 SIREN 默认配置

```yaml
model:
  type: siren
  hidden_dim: 256
  n_layers: 5
  omega_0: 10.0           # 保守起步，可逐步增到 30
  # 初始化: SIREN 专用（见 Sitzmann et al. 2020）

training:
  optimizer: Adam
  lr: 0.0005
  lr_schedule: cosine_decay      # 从 lr 线性衰减到 min_lr
  warmup_iters: 200               # 前 200 iter 线性升温到 lr
  min_lr: 1.0e-6                  # cosine decay 最低学习率
  grad_clip: 1.0                  # 梯度裁剪，SIREN 的 ω₀ 可能导致梯度爆炸
  max_iter: 10000
  early_stopping_patience: 1000   # 10% of max_iter，避免 loss 振荡时过早停止
  early_stopping_metric: holdout_residual

loss:
  data_weight: 1.0
  geometry_weight: 0.0     # 第一版关闭

coords:
  normalization: "neg1_to_1"     # [-1, 1] 对应 [0, H-1] × [0, W-1]
  # omega_0=10 在此归一化下 ≈ 1.6 cycles/unit ≈ 766 cycles per 480px
  # 必须明确记录，否则 ω₀ 调参变成玄学

data:
  n_frames: 32             # 阶段 1 起步
  val_ratio: 0.2
  seed: 42
  highpass_sigma_bg_lr_px: 5.0    # highpass 背景高斯，mode=nearest
  noise_sigma_c: 0.0724

forward:
  psf_sigma_lr_px: 1.0           # 光学 PSF，内部 sigma_hr = psf_sigma * scale
  scale: 2
  blur_mode: "constant"           # forward operator Gaussian mode
  sample_mode: "constant"         # forward operator sampling mode
```

> ⚠️ **配置命名约定**：`psf_sigma_lr_px` 和 `highpass_sigma_bg_lr_px` 必须明确区分。前者是光学 PSF（影响 forward model），后者是高通背景估计半径（影响 LR 预处理）。不要混用。

### 5.2 WIRE 额外配置

```yaml
model:
  type: wire
  hidden_dim: 256
  n_layers: 5
  omega_0: 10.0           # Gabor 频率参数
  sigma_0: 10.0           # Gabor 包络宽度
  # 初始化: WIRE 官方 repo 实现

# training, loss, data, forward 与 SIREN 完全相同
```

### 5.3 Deep Decoder 默认配置

```yaml
model:
  type: deep_decoder
  channels: [128, 128, 64, 32, 16]   # 逐级通道缩减，4 级 2x 上采样
  input_type: fixed_random_noise      # z 在初始化后冻结
  output_shape: [960, 1280]           # 2x HR grid

training:
  optimizer: Adam
  lr: 0.0001              # Deep Decoder 参数少，梯度方差大，用更小学习率
  lr_schedule: cosine_decay
  warmup_iters: 100
  min_lr: 1.0e-6
  grad_clip: 1.0
  max_iter: 5000
  early_stopping_patience: 500
  early_stopping_metric: holdout_residual

# loss, data, forward, coords 与 SIREN 完全相同
```

---

## 6. 物理参数速查

> 以下参数来自 AGENTS.md ground truth 和 EP06 源码，EP08 实现时必须严格一致。

| 参数 | 值 | 来源 | 在 EP08 中的位置 |
|------|-----|------|-----------------|
| 探测器输出尺寸 | 640×480 px | AGENTS.md | LR input shape |
| HR output shape (2x) | 1280×960 px | scale=2 | HR grid |
| TXT 采样间距 | 10 µm/px | AGENTS.md | 物理解释 |
| 当前空间分辨率 | 20 µm | AGENTS.md | 物理解释 |
| 电动台-像素旋转角 θ | 47.6° | AGENTS.md | **仅作 prior/可选实验，非 chip 主轴角** |
| PSF σ | ~0.5 px（AGENTS.md）/ 1.0 px（EP06 默认）| 两处 | **用 EP06 默认值 1.0 保持可比性** |
| Noise floor | 0.0724°C | AGENTS.md | data loss 归一化 |
| 主扫描 session | 255 帧 | AGENTS.md | 输入帧数 |
| Highpass σ_bg | 5.0 LR px | EP06 data_loader.py | highpass 预处理 |
| Forward blur mode | `"constant"` | EP06 forward_model.py | forward operator |
| Forward sample mode | `"constant"` | EP06 forward_model.py | forward operator |
| Highpass mode | `"nearest"` | EP06 data_loader.py | highpass 预处理 |

> **PSF σ 说明**：AGENTS.md 记录 σ≈0.5 px（可能 0.2–0.5），但 EP06 forward_model.py 默认 `psf_sigma=1.0`。EP08 **必须使用 EP06 默认值 1.0** 以保证 baseline 可比。如果后续有更精确的 PSF 标定，可作为 ablation 实验。

---

## 7. 风险与缓解

| 风险 | 严重性 | 缓解策略 |
|------|:------:|----------|
| PyTorch forward 与 EP06 NumPy 不等价 | ⛔ 高 | 阶段 0 必须通过逐像素等价性测试（< 1e-5） |
| INR 高频过强，pin 区域产生伪线 | ⚠️ 中 | 从 ω₀=10 保守起步；early stopping by hold-out；artifact score 门控 |
| 255 帧全量优化显存过高 | ⛔ 高 | **见下方 VRAM 估算表**；阶段 3 Step 3+ 必须使用 mini-batch frame sampling |
| Deep Decoder 过拟合噪声 | ⚠️ 中 | 表达能力本身受限；early stopping by hold-out |
| shift 估计误差影响 forward loss | ⚠️ 中 | 冻结 EP05 shifts（第一版），不开放 ε_k 修正 |
| SIREN ω₀ 过大导致高频振铃 | ⚠️ 中 | 从 ω₀=10 开始，逐步增到 30，每步检查 artifact |
| 47.6° 不是 chip 主轴角 | ⚠️ 中 | 坐标旋转标为可选实验，不作为默认 |
| Geometry loss 生成幻觉 | ⚠️ 中 | 降为 P2，先在 synthetic 验证 |
| SIREN/WIRE 梯度爆炸 | ⚠️ 中 | grad_clip=1.0 + cosine decay + warmup |

### 7.1 VRAM 估算与 Mini-batch 策略

> ℹ️ 硬件约束：24 GB VRAM。以下为 INR (256 hidden, 5 layers) 在 float32 下的粗估。

| 阶段 | 配置 | INR 前向 (GB) | Forward model (GB) | 梯度 (GB) | 总预估 (GB) | 24 GB 余量 |
|------|------|:---:|:---:|:---:|:---:|:---:|
| Stage 1 Step 1 | 32 帧, 256×256 patch, HR=512×512 | ~1.0 | ~0.5 | ~3.0 | **~4.5** | ✅ 充裕 |
| Stage 3 Step 2 | 64 帧, 640×480, HR=960×1280 | ~6.0 | ~2.0 | ~16 | **~24** | ⚠️ 刚好 |
| Stage 3 Step 4 | 255 帧, 640×480, HR=960×1280 | ~6.0 | ~8.0 | ~28+ | **~42** | ❌ OOM |

**Mini-batch Frame Sampling 策略**（阶段 3 Step 3+ 必须启用）：

```python
# 每个训练 iteration 随机采样 K 帧计算 loss，而非全量
for iteration in range(max_iter):
    frame_indices = rng.choice(n_train, size=batch_k, replace=False)
    loss = sum(forward_loss(model, forward_op, hp_frames[k], k)
               for k in frame_indices) / batch_k
    loss.backward()
    optimizer.step()
```

| 配置 | batch_k | VRAM 预估 |
|------|:-------:|:----------:|
| 255 帧, batch_k=8 | 8 | ~8 GB ✅ |
| 255 帧, batch_k=16 | 16 | ~12 GB ✅ |
| 255 帧, batch_k=32 | 32 | ~18 GB ✅ |

默认 `batch_k=16`，在配置中显式暴露，方便根据实际 VRAM 调整。

---

## 8. 与 EP07（TCForge）的配合

| EP07 状态 | EP08 行动 |
|-----------|-----------|
| 尚未开始 | EP08 用真实数据开发，不等 EP07 |
| EP07 smoke test 通过后（5 个合成场景） | 用 TCForge 合成数据做 ground truth 指标（Chamfer、Hausdorff、boundary F1） |
| EP07 benchmark 完成后（60 个合成场景） | 在更多场景上验证方法稳定性 |
| EP07 合成数据可用后 | Geometry loss 先在合成数据上验证有效性，再考虑用于真实数据 |

---

## 9. 验收清单

### 阶段 0 验收

- [ ] PyTorch forward model 与 EP06 NumPy forward 等价性测试通过（max abs error < 1e-5）
- [ ] PyTorch 采样使用手写 bilinear（非 `F.grid_sample`），坐标约定与 EP06 一致
- [ ] PyTorch highpass 与 EP06 highpass 等价性测试通过
- [ ] Forward operator 使用 `mode="constant"`（与 EP06 一致）
- [ ] Highpass preprocessing 使用 `mode="nearest"`（与 EP06 一致）
- [ ] PSF sigma = 1.0 LR px（与 EP06 默认一致）
- [ ] Hold-out split 使用位移相位分层抽样，可复现（seed=42）
- [ ] 五项指标 pipeline 对 EP06 结果 smoke test 通过
- [ ] EP06 baseline 五项指标数值记录到 `configs/ep06_baseline_metrics.json`
- [ ] `pin_mask` 标注完成（artifact score 计算所需）
- [ ] Raw-control / offset correction track 实现并验证
- [ ] 坐标归一化范围确认为 `[-1, 1]` 并记录到配置

### 阶段 1 验收

- [ ] SIREN 在 32 帧 patch 上训练收敛（含 cosine decay + grad clip）
- [ ] WIRE 在 32 帧 patch 上训练收敛
- [ ] INR 输出为 HR highpass field，forward loss 链路正确（INR→forward→LR vs LR highpass）
- [ ] 两者 hold-out residual ≤ EP06 MAP-TV × 1.1
- [ ] 两者无明显伪线
- [ ] WIRE vs SIREN activation ablation 有可比结果
- [ ] Raw-control 一致性检查通过
- [ ] Split-half 稳定性检查通过

### 阶段 2 验收

- [ ] Deep Decoder 使用逐级上采样 + 通道缩减架构（非一步上采样 + 1×1 卷积）
- [ ] Deep Decoder 在 32 帧 patch 上训练收敛
- [ ] 三方法 + EP06 四方对比表完整
- [ ] INR vs CNN decoder 有明确结论
- [ ] Gabor vs Sine 有明确结论
- [ ] 选出 1-2 个赢家方法进入阶段 3

### 阶段 3 验收

- [ ] 赢家方法扩展到 255 帧全分辨率
- [ ] 255 帧使用 mini-batch frame sampling（batch_k=16），VRAM 实测 < 20 GB
- [ ] 255 帧 hold-out residual ≤ EP06 MAP-TV × 1.1（从 `ep06_baseline_metrics.json` 读取）
- [ ] 255 帧 split-half NRMSE ≤ EP06 MAP-TV
- [ ] 255 帧 artifact score ≤ EP06 MAP-TV
- [ ] Raw-control SSIM ≥ 0.9
- [ ] 多 seed (3 次) 稳定性 std < 5%
- [ ] 边缘宽度 < bicubic
- [ ] 可选：geometry loss ablation（如已做）
- [ ] 可选：坐标旋转 ablation（如已做）

### 阶段 4 验收

- [ ] 四方对比 Notebook 完整且可独立执行
- [ ] 所有图表附解读
- [ ] 三个科学问题有数据支持的结论
- [ ] 所有代码、配置、随机种子可复现
- [ ] Notebook 遵循 AGENTS.md 展示规范（CVPR 风格、无双重显示）

---

## 10. 参考资料

### INR 方法

- **SIREN**: Sitzmann et al., "Implicit Neural Representations with Periodic Activation Functions", NeurIPS 2020
  - GitHub: https://github.com/vsitzmann/siren（2000+★）
  - 论文: https://arxiv.org/abs/2006.09661

- **WIRE**: Saragadam et al., "WIRE: Wavelet Implicit Neural Representations", CVPR 2023
  - GitHub: https://github.com/vishwa91/wire（178★，含 multi-image SR 示例）
  - 论文: https://arxiv.org/abs/2301.05187

### CNN Decoder 方法

- **Deep Decoder**: Heckel & Hand, "Deep Decoder: Concise Image Representations from Untrained Non-convolutional Networks", ICLR 2019
  - GitHub: https://github.com/reinhardh/supplement_deep_decoder（101★）

- **DeepInverse**: 工业级成像逆问题库，含 DIP 官方教程
  - GitHub: https://github.com/deepinv/deepinv（735★）
  - DIP 教程: https://deepinv.github.io/deepinv/auto_examples/optimization/demo_dip.html

### 项目内参考

- Forward model: [algos/ep06_sr_poc/src/common/forward_model.py](file:///home/ujs/mycode/thermal_lift/algos/ep06_sr_poc/src/common/forward_model.py)
- Data loader: [algos/ep06_sr_poc/src/common/data_loader.py](file:///home/ujs/mycode/thermal_lift/algos/ep06_sr_poc/src/common/data_loader.py)
- 物理参数: [AGENTS.md](file:///home/ujs/mycode/thermal_lift/AGENTS.md)
- 详细方法论调研: [深度学习-生成模型方法调研.md](file:///home/ujs/mycode/thermal_lift/handoff/note/深度学习-生成模型方法调研.md)
- 筛选分析: [ep0708plan.md](file:///home/ujs/mycode/thermal_lift/handoff/note/ep0708plan.md)

### 已排除方法（参考）

- **SuperF**: https://github.com/sjyhne/superf（3★，ICLR 2026）— 场景匹配但 repo 过新、方法论粗糙（ReLU + boxcar PSF），只借鉴 shared-INR 思想
- **DiffPIR**: https://github.com/yuanzhi-zhu/DiffPIR — 需合成芯片 prior，EP09+ 再考虑
- **BurstM**: https://github.com/Egkang-Luis/BurstM — 需合成训练集，后期加速方案
