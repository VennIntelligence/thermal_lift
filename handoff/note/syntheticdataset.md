# 合成芯片热像数据集 — 设计文档

> **目的**：为 EP07/EP08 的 DL 方法提供定量评估基础和可选训练数据。
> **目标读者**：Codex 实现代理 + 项目成员审查。

---

## 1. 为什么需要自建合成数据

### 1.1 我们面临的根本困难

我们有 255 帧真实 LR 观测，但**没有 HR ground truth**。这意味着：

- 所有 SR 方法只能用 forward consistency / split-half / FRC 等间接指标评价
- 无法回答"重建出的线宽 X µm 是否真实"
- 无法区分"真的恢复了高频信息" vs "先验填充了合理但不真实的结构"
- 无法做 boundary F1 / Chamfer distance 的定量评估

合成数据**唯一的作用**就是：在一个**我们完全控制的、物理模型匹配的**环境里，给方法打分。

### 1.2 为什么不用现有数据集

| 数据集 | 不用的原因 |
|--------|----------|
| Real-IISR / FLIR-IISR | 自然场景红外，RGB 退化模型，不是多帧微扫描 |
| PBVS Challenge | 面向 ×8/×16 视觉 SR，不保温度物理量 |
| TFRD | 稀疏传感器插值问题，完全不同任务 |
| NTIRE BurstSR | 手机 RAW burst，Bayer mosaic，与我们单通道 float32 不匹配 |

**核心矛盾**：现有数据集的退化模型都不匹配我们的 `shift → PSF → block_average → noise` pipeline。在错误退化模型上跑出的 PSNR 没有意义。

### 1.3 合成数据的三个用途

| 用途 | 优先级 | 说明 |
|:---:|:---:|------|
| **定量 benchmark** | P0 | 在合成数据上用 PSNR / SSIM / boundary F1 / Chamfer 比较所有方法 |
| **方法调参** | P0 | 用合成 hold-out 选超参（INR 频率带宽、DIP 迭代次数、正则强度） |
| **denoiser 预训练** | P2 | EP08 如果要训小型 diffusion prior，需要 10k+ patches |

---

## 2. 物理模型规格 — 必须严格匹配真实系统

### 2.1 观测模型（已在 EP06 中实现和验证）

```
y_k = Downsample( GaussianBlur( Shift(x_hr, δ_k), σ_psf ) ) + n_k
```

| 参数 | 值 | 来源 |
|------|-----|------|
| HR 网格 | 960×1280 (scale=2) 或 1920×2560 (scale=4) | EP06 约定 |
| LR 网格 | 480×640 | 探测器物理尺寸 |
| PSF σ | **0.5 px (LR)** = 1.0 px (2x HR) | EP06 合成验证最优值 |
| 噪声 | Gaussian, σ_n = **0.0724°C** | EP03 实测 noise floor |
| Downsample | **block average** (2×2 或 4×4) | 探测器像素积分 |
| Shift δ_k | 从真实 contour_refined alignment CSV 加载 | EP05 数据 |
| 帧数 | 255（主 session）或子集 32/64/128 | 按实验需求 |

### 2.2 温度场模型

芯片不是"图像"，是 piecewise-constant 温度场：

```
T(x, y) = T_bg + ΔT · mask(x, y) + low_freq_background(x, y)
```

| 参数 | 典型值 | 随机化范围 |
|------|--------|----------|
| T_bg (背景温度) | 21.0°C | 19–24°C |
| ΔT (前景-背景温差) | 1.5°C | 0.5–3.0°C |
| mask | 二值 {0, 1} | 几何随机生成 |
| low_freq_background | 缓慢热梯度 | 2D 低阶多项式, 幅度 0–0.5°C |

**为什么 piecewise-constant**：真实芯片是金属 vs 空气/基板，温度跳变是阶跃型的，只在 PSF 模糊后才变成连续 ESF。生成器应该在 HR 级别生成锐利二值 mask，让 forward model 自然产生模糊。

---

## 3. 几何生成器设计

### 3.1 几何原语

所有结构都是**芯片坐标系下的轴对齐矩形组合**，然后旋转到探测器坐标系。

| 原语 | 描述 | 参数 |
|------|------|------|
| **Rectangle** | 基本矩形 | (cx, cy, w, h) |
| **Pin / Slot** | 窄长矩形，模拟针脚 | (cx, cy, w, h)，w << h 或 h << w |
| **L-shape** | 两个矩形组合 | 两个 rect 的 union |
| **Cross / T-shape** | 两个矩形交叉 | 两个 rect 的 union |
| **Parallel lines** | 等间距窄矩形阵列 | (n, gap, width, length) |
| **Frame / Border** | 大矩形减去内部矩形 | outer_rect - inner_rect |

### 3.2 场景组合规则

一个合成场景由以下层次组成：

```
Scene
├── Chip outline (大矩形 frame，模拟芯片外边界)
│   ├── Cutout 1: Pin array (镂空针脚阵列)
│   ├── Cutout 2: L-shape slot
│   ├── Cutout 3: Cross pattern
│   └── Cutout 4: Parallel grooves
└── Background (芯片外区域)
```

**芯片是前景**（高温金属），**镂空是背景**（低温空气/基板）。mask=1 表示芯片材料，mask=0 表示镂空/背景。

### 3.3 尺度规格

所有几何尺寸在 HR 像素单位下定义（1 HR pixel = 5 µm @2x）：

| 结构 | HR 像素范围 | 物理尺寸 | 说明 |
|------|-----------|---------|------|
| 芯片外框宽度 | 600–900 px | 3–4.5 mm | 占据视场大部分 |
| Pin 宽度 | 2–8 px | 10–40 µm | **关键：2 px = 10 µm 是当前分辨率极限** |
| Pin 长度 | 20–80 px | 100–400 µm | |
| Pin 间距 | 4–16 px | 20–80 µm | |
| L/T 臂宽 | 3–10 px | 15–50 µm | |
| 平行线宽度 | 2–6 px | 10–30 µm | |
| 平行线间距 | 4–12 px | 20–60 µm | |

### 3.4 旋转

所有几何在芯片坐标系（轴对齐）中生成，然后**整体旋转 47.6° ± jitter** 到探测器坐标系：

```python
theta = 47.6 + np.random.uniform(-1.0, 1.0)  # 度
# 使用 scipy.ndimage.rotate 或仿射变换
```

### 3.5 难度分级

建议生成三个难度级别，用于 ablation：

| 级别 | 结构复杂度 | 最小特征 | 温差 | 噪声 |
|:---:|---------|---------|------|------|
| **Easy** | 单个大矩形 + 2–3 个宽 pin | ≥ 6 px (30 µm) | 2.0°C | 0.0724°C |
| **Medium** | 芯片框 + pin 阵列 + L 形 | ≥ 3 px (15 µm) | 1.5°C | 0.0724°C |
| **Hard** | 完整芯片 + 密集 pin + 十字 + 平行线 | ≥ 2 px (10 µm) | 0.8°C | 0.0724°C |

---

## 4. 数据集规模与分片

### 4.1 全图数据（用于 SR benchmark）

| 项目 | 数量 | 说明 |
|------|------|------|
| 场景数 | **20–50 个** | 每个场景是一个独立的芯片几何 |
| 每场景帧数 | 255 (或子集 32/64) | 使用真实 shift pattern |
| HR GT 尺寸 | 960×1280 (2x) / 1920×2560 (4x) | |
| LR 帧尺寸 | 480×640 | |
| 存储估算 | ~50 场景 × (HR 5MB + 255帧 LR 300MB) ≈ **15 GB** | 可以按需生成，不必全存 |

### 4.2 Patch 数据（用于 denoiser 训练，P2 优先级）

| 项目 | 数量 | 说明 |
|------|------|------|
| HR patch 数 | **10k–50k** | 128×128 或 256×256 random crop |
| 对应 LR patch | 64×64 或 128×128 | |
| 用途 | 训练小型 DDPM denoiser / diffusion prior | |
| 存储 | ~50k × 128×128 × 4B ≈ **3 GB** | 可以在线生成 |

---

## 5. 输出格式

### 5.1 每个场景的产物

```
data/synthetic/scenes/scene_0001/
├── metadata.json              ← 几何参数、物理参数、随机种子
├── hr_mask_2x.npy             ← 960×1280, float32, {0.0, 1.0}
├── hr_temperature_2x.npy      ← 960×1280, float32, 单位 °C
├── hr_highpass_2x.npy          ← 960×1280, float32, highpass 后
├── hr_edge_map_2x.npy          ← 960×1280, float32, |∇mask| 膨胀后
├── hr_mask_4x.npy             ← 1920×2560 (可选)
├── hr_temperature_4x.npy      ← 1920×2560 (可选)
├── lr_burst.npy                ← (255, 480, 640), float32, 模拟 LR 帧
├── lr_burst_highpass.npy       ← (255, 480, 640), float32, highpass 后
├── shifts.npy                  ← (255, 2), float64, 使用的 δ_k
└── edge_segments.csv           ← 边界线段坐标 (用于 Chamfer/F1)
```

### 5.2 metadata.json 示例

```json
{
  "scene_id": "scene_0001",
  "seed": 42,
  "scale": 2,
  "geometry": {
    "chip_frame": {"cx": 480, "cy": 640, "w": 700, "h": 500},
    "cutouts": [
      {"type": "pin_array", "n": 8, "width": 4, "height": 40, "gap": 8, "cx": 300, "cy": 400},
      {"type": "L_shape", "arm1": [200, 300, 6, 50], "arm2": [200, 300, 40, 6]},
      {"type": "cross", "cx": 500, "cy": 500, "arm_w": 4, "arm_h": 30}
    ],
    "rotation_deg": 47.3
  },
  "physics": {
    "T_bg": 21.0,
    "delta_T": 1.5,
    "psf_sigma_lr": 0.5,
    "noise_sigma": 0.0724,
    "low_freq_amplitude": 0.2,
    "shift_source": "real_contour_refined"
  }
}
```

---

## 6. 评估指标（合成数据独有）

有了 HR GT，可以计算真实数据上无法算的指标：

### 6.1 全图指标

| 指标 | 公式 / 工具 | 评价什么 |
|------|-----------|---------|
| **PSNR** | `10 * log10(max² / MSE)` | 整体重建保真度 |
| **SSIM** | `skimage.metrics.structural_similarity` | 结构保持 |
| **NRMSE** | `||x_sr - x_gt|| / ||x_gt||` | 归一化误差 |

### 6.2 边界指标（最重要）

| 指标 | 说明 | 评价什么 |
|------|------|---------|
| **Boundary F1** | 检测到的边界 vs GT 边界，阈值 1 px | 轮廓检测完整性 |
| **Chamfer distance** | 检测边界到 GT 边界的平均距离 | 定位精度 |
| **Hausdorff distance** | 最大偏差 | 最坏情况 |
| **Edge width (ESF σ)** | 在 GT 边界处拟合 ESF | 锐化程度 |
| **Pin gap accuracy** | 恢复的 pin 间距 vs GT 间距 | 细结构可分辨性 |
| **Topology F1** | 连通分量数目一致性 | 是否产生断裂/合并 |

### 6.3 稳定性指标

| 指标 | 说明 |
|------|------|
| **帧子集稳定性** | 用 32/64/128/255 帧重建，检查结果方差 |
| **噪声 seed 稳定性** | 同一几何、不同噪声实例，重建差异 |
| **PSF 敏感性** | σ = 0.3/0.5/0.7/1.0 时的指标变化 |

---

## 7. 实现规划

### 7.1 代码结构

```
data/synthetic/
├── generator.py               ← 主生成器脚本
├── geometry.py                ← 几何原语（Rectangle, Pin, L-shape, Cross, ...）
├── physics.py                 ← 温度场 + 低频背景 + highpass
├── forward_model.py           ← 调用 EP06 的 forward model（或 PyTorch 版）
├── evaluate.py                ← 合成 benchmark 评估脚本
├── configs/
│   ├── easy.yaml              ← 简单难度配置
│   ├── medium.yaml            ← 中等难度配置
│   └── hard.yaml              ← 困难难度配置
└── scenes/                    ← 生成的场景数据（gitignore）
```

### 7.2 依赖

**只用已有依赖**，不引入新包：

- `numpy` — 几何生成、数组操作
- `scipy.ndimage` — 旋转、高斯模糊
- `matplotlib` — 可视化验证
- EP06 `forward_model.py` — 复用 shift + PSF + downsample

### 7.3 实现步骤

| 步骤 | 内容 | 预估时间 | 产物 |
|:---:|------|:---:|------|
| **S1** | 几何原语：矩形、pin 阵列、L 形、十字、平行线 | 0.5 天 | `geometry.py` |
| **S2** | 场景组合：随机芯片生成 + 旋转 + 温度场 | 0.5 天 | `physics.py` |
| **S3** | LR burst 生成：调用 forward model + 加噪声 | 0.5 天 | `forward_model.py` |
| **S4** | HR GT 产物：mask + temperature + highpass + edge map | 0.5 天 | `generator.py` |
| **S5** | 评估脚本：PSNR / Chamfer / F1 / ESF | 1 天 | `evaluate.py` |
| **S6** | Smoke test：生成 5 个场景，跑 EP06 SAA/IBP/MAP-TV，验证指标合理 | 0.5 天 | 验证报告 |
| **总计** | | **~3 天** | |

### 7.4 关键设计决策

| 决策 | 推荐 | 原因 |
|------|------|------|
| Shift 来源 | **直接加载真实 contour_refined shifts** | 避免统计模型与真实分布不匹配 |
| HR 分辨率 | 先做 2x (960×1280)，后加 4x | 与 EP06 一致 |
| Highpass | 使用 `thermal_core` 中的同一 highpass 函数 | 保证退化链一致 |
| 随机 seed | **全局 seed + 场景 ID** 确定性生成 | 可复现 |
| 存储 | 按需生成 vs 预存 | 全图预存（慢），patch 在线生成（快） |

---

## 8. 预期产出与验证

### 8.1 Smoke test 预期结果

在 Easy 级别（大特征、高温差）上：

| 方法 | 预期 PSNR (2x) | 预期 Boundary F1 |
|------|:---:|:---:|
| Bicubic (single frame) | ~25–28 dB | ~0.5 |
| SAA uniform (255 帧) | ~30–33 dB | ~0.7 |
| IBP (255 帧) | ~31–34 dB | ~0.75 |
| MAP-TV (255 帧) | ~31–35 dB | ~0.75–0.80 |
| WIRE INR (EP07) | **~33–37 dB?** | **~0.80–0.90?** |

如果 WIRE 在合成数据上显著优于经典方法，就证明了 DL 方法的价值。如果不优于，说明 DL 的隐式正则在我们的场景下不比 TV 更好——这也是有价值的结论。

### 8.2 最终交付

| 交付物 | 描述 |
|--------|------|
| **生成器代码** | `data/synthetic/` 下的完整 pipeline，可一键生成任意数量场景 |
| **预生成 benchmark** | 20–50 个场景，覆盖 Easy/Medium/Hard，存在 `data/synthetic/scenes/` |
| **评估脚本** | 输入 SR 结果 .npy + 场景目录 → 输出 PSNR/Chamfer/F1/ESF 表格 |
| **Smoke test 报告** | 验证 EP06 方法在合成数据上的基线指标 |
| **论文 Figure** | HR GT vs LR frame vs SR result 的并排对比 + 指标表 |

---

## 9. 开放问题（需讨论）

1. **Shift 来源**：只用真实 shifts，还是也生成随机 shift pattern？
   - 真实 shifts：完全匹配实验条件，但只有一组 pattern
   - 随机 shifts：可以测试不同覆盖率，但可能引入不真实的分布

2. **4x HR GT 的生成方式**：
   - 方案 A：直接在 1920×2560 网格生成几何 → 最精确
   - 方案 B：先在 960×1280 生成，再 upsample → 有锯齿
   - 推荐方案 A

3. **低频背景模型**：
   - 简单：2D 线性梯度
   - 中等：2D 二次多项式
   - 复杂：从真实数据拟合的低频模板
   - 推荐从简单开始

4. **是否模拟温度漂移**：
   - 真实数据有帧间 ~0.0724°C 级别的慢漂移
   - 如果用 highpass 预处理，漂移已被消除
   - 推荐：主线不模拟漂移（因为我们用 highpass），但可选开关

5. **数据集命名与论文表述**：
   - 建议名称：**ThermalChipPhantom** 或 **TCPBench**
   - 强调与现有数据集的区别：面向工业芯片 LWIR 微扫描、rectilinear 几何、物理退化链匹配
