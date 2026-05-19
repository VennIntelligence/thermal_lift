# EP07 / EP08 方案分析：从调研中筛选最有潜力的方法

> 基于两份调研报告（`reposeEP07.md` 762 行 + `深度学习-生成模型方法调研.md` 847 行），结合我们的硬件约束、数据现实和已有代码基础，筛选出 EP07 / EP08 应该做什么。

---

## 0. 当前状态速览

| 维度 | 现状 |
|------|------|
| **已完成** | EP01-EP06：数据审计 → 位移标定 → 理论极限 → 轮廓锚定 → 对齐基线 → 经典 2x SR POC |
| **EP06 经典方法天花板** | SAA/IBP/MAP-TV 视觉上拉不开差距；MAP-TV λ=0.01 修正后 artifact 降至 0.144，但 gradient 0.136 反而低于 SAA 0.161 |
| **已有 forward model** | NumPy/SciPy 实现，`shift → PSF blur → downsample`，仅支持 scale=2，**非 PyTorch** |
| **对齐质量** | data-driven contour refined held-out Chamfer 0.1341 px，2x phase 4/4 占满，4x 只有 4/16 |
| **硬件** | **2 × RTX 3090 24GB**，无大规模预训练能力 |
| **数据** | 255 帧 640×480 float32 温度矩阵，单通道，无 HR GT |

---

## 1. 调研方案大筛选：淘汰 + 保留

### 1.1 淘汰的方法（及原因）

| 方法 | 调研推荐分 | **淘汰原因** |
|------|:---:|------|
| **SuperF** (ICLR 2026) | 10.0 | 原论文在 **H100 80GB** 上跑，255 帧 640×480 全图优化显存不可控；虽可 patch，但工程改造量大且原 repo 很新(3★)，社区验证不足 |
| **DiffPIR / DAPS / DPS** | 7.6–8.0 | 需要**领域 prior 预训练**；自然图像 prior 直接用会生成伪纹理；需在合成芯片数据上重训 diffusion model，这是 EP09+ 的事 |
| **Flower / PnP-Flow** | 7.0–7.4 | 同上：需训练 flow matching prior，当前无合成数据、无 prior |
| **BurstM** (ECCV 2024) | 6.5 | 需要合成 HR-LR burst 监督训练，依赖 pseudo-GT，是后期加速方案 |
| **HDR-DSP-SR** | 6.7 | 面向卫星 push-frame，场景迁移成本高 |
| **Real-IISR / CIDIS / VGTSR** | — | 自然/城市/UAV 红外数据集，与芯片镂空结构差异太大，不能直接用 |
| **GeoSR / ECBSR / EFDN** | — | 小众 SISR，需 paired 训练数据，只能作为 loss 思想参考 |

### 1.2 保留的方法（3 + 1 条线路）

| 优先级 | 方法 | 调研分 | **保留理由** |
|:---:|------|:---:|------|
| **🥇 P0** | **WIRE-style INR** + 物理 forward model | 9.3 | **无需训练数据、无需 prior、单通道 MLP 8GB 即可跑、有 multi-image SR 示例、适合强边缘场景** |
| **🥈 P1** | **DIP / Deep Decoder** + 物理 forward model | 8.4–9.0 | **零训练集、24GB 绰绰有余、DIP 是经典 baseline、Deep Decoder 更不易拟合噪声** |
| **🥉 P2** | **合成芯片几何数据 + 小型 denoiser** | — | 为 EP08 4x 探索做准备；128×128 单通道 diffusion 在 3090 上 **5–12h 即可训完** |
| **辅助** | **多尺度检测** (DoG/LoG/top-hat/ridge) | — | 快速提升内轮廓覆盖率，与 SR 正交，可并行 |

---

## 2. 关键方法深入分析

### 2.1 🥇 WIRE-style INR（EP07 主攻）

**为什么它是最佳选择：**

| 维度 | 评估 |
|------|------|
| 场景匹配 | 原生支持多帧 + 亚像素位移 + 无 HR GT + 连续 HR 函数 + 强边缘 |
| 显存 | MLP 3–5 层 × 256 hidden，单通道输出，**8GB 即可**；patch-based 训练可进一步压缩 |
| 代码 | [vishwa91/wire](https://github.com/vishwa91/wire) 178★，含 multi-image SR 示例脚本 |
| 物理约束 | 可直接把 `shift → PSF → downsample` 写进 loss |
| 几何先验 | 可在芯片坐标系 `R(-47.6°)` 旋转后输入坐标，让模型「看到」横平竖直结构 |
| 优势 vs MAP-TV | Gabor/wavelet 激活天然边缘友好、无 TV staircasing、无需手调 λ |

**与 SuperF 的关系：** SuperF 的核心思想（shared INR + test-time optimization + frame-specific alignment）可以用 WIRE 的 backbone 实现，不需要 SuperF 的完整框架。WIRE 本身就有 multi-image SR 代码，我们只需要：
1. 把 WIRE 的 MLP 作为 HR 场景表示
2. 把 EP06 的 forward model 从 NumPy 翻译到 PyTorch（一天工作量）
3. Loss = Σ_k || forward(INR(coords), shift_k) - y_k ||²

**关键风险：** INR 高频能力过强可能产生伪线。缓解：先用 32 帧子集 → 逐步扩到 255 帧，用 hold-out residual 监控 hallucination。

### 2.2 🥈 DIP / Deep Decoder（EP07 对照线）

**角色：快速验证「深度结构 prior 是否超过经典方法天花板」**

| 维度 | DIP (U-Net) | Deep Decoder |
|------|:---:|:---:|
| 参数量 | 多（over-parameterized） | 少（under-parameterized） |
| 显存 | ~4–8GB | ~2–4GB |
| 过拟合风险 | 中高，需 early stopping | 低，天然屏障 |
| 边缘锐度 | 较高 | 较保守 |
| 适合场景 | 探索上限 | 保守 baseline |

**实现路径：**
- 复用 [deepinv/deepinv](https://github.com/deepinv/deepinv) (735★)，它内置 DIP 示例 + 自定义 operator 接口
- 或直接自建：`net(z_fixed) → x_hr → forward_model → loss`
- 两者都不需要训练数据，完全 test-time optimization

**与 WIRE 的对比价值：**
- WIRE 是连续函数表示（坐标 → 像素值）
- DIP 是卷积网络表示（固定噪声 → 图像）
- 对比两者可以判断哪种隐式正则更适合我们的数据

### 2.3 🥉 合成数据 + 小型 denoiser（EP08 准备）

**为什么现在要开始准备：**

4x 需要结构先验，而先验需要合成数据来训练。2 × 3090 24GB 的约束下：

| 方案 | 可行性 |
|------|--------|
| 128×128 单通道 DDPM U-Net，10k–50k 合成 patch | ✅ **5–12h 训完** |
| 256×256 单通道，100k patch | ✅ **12–24h** |
| 全尺寸 640×480 直接训 | ⚠️ 勉强可以，batch size 受限 |

**合成数据生成器设计：**
1. 用 Python 生成 HR rectilinear mask：矩形、针脚、L 形、十字、平行线
2. 旋转 47.6° ± jitter
3. 加前景/背景温差 + 低频热背景
4. 用我们的 forward model 生成 LR burst
5. 输出：HR GT + LR burst + edge GT → 可定量评估

这个生成器本身就是有价值的论文贡献（procedural thermal chip phantom）。

---

## 3. EP07 / EP08 具体方案

### EP07：深度结构先验 2x SR（~2–3 周）

```
EP07: DL 方法 2x SR + 多尺度检测
├── 7.1 PyTorch forward model 翻译       [2 天]
│     把 EP06 NumPy forward model 翻译成 differentiable PyTorch
│     包含：shift interpolation + Gaussian PSF + block downsample
│     验证：与 NumPy 版逐像素对比，误差 < 1e-6
│
├── 7.2 WIRE INR 2x 主实验               [5–7 天]
│     ├── 基础版：32 帧子集，冻结 shift，WIRE MLP → forward → L2 loss
│     ├── 扩展版：255 帧全量，mini-batch frame sampling
│     ├── 加入几何 loss：L_bin + L_sparse_edge（芯片坐标系）
│     ├── 加入 hold-out 51 帧验证
│     └── 与 EP06 SAA/IBP/MAP-TV 同指标并排对比
│
├── 7.3 DIP / Deep Decoder 对照实验       [3–5 天]
│     ├── DIP (U-Net) + 同一 forward model
│     ├── Deep Decoder + 同一 forward model
│     ├── 两者都做 hold-out + split-half
│     └── 对比：DIP vs Deep Decoder vs WIRE vs SAA/IBP/MAP-TV
│
├── 7.4 多尺度结构检测                    [2–3 天, 可与 7.2 并行]
│     ├── DoG / LoG / top-hat / ridge / structure tensor
│     ├── 在 2x SR 输出上做候选检测
│     ├── 覆盖率统计：与 EP04 Otsu baseline 对比
│     └── Split-half 一致性过滤
│
└── 7.5 Notebook + 报告                   [2 天]
      ├── 方法消融对比表 + ROI 可视化
      ├── Forward consistency + FRC + edge-MTF
      └── EP07 结论：DL 是否超越经典天花板
```

### EP08：4x 探索 + 合成验证（~2–3 周）

```
EP08: 4x SR 探索 + 合成 benchmark
├── 8.1 合成芯片数据生成器                 [3–5 天]
│     ├── Python procedural generator
│     ├── 10k–50k HR patches (128×128 或 256×256)
│     ├── 用真实 δ_k 分布 + PSF + noise 生成 LR bursts
│     └── 输出 HR GT + LR burst + edge GT
│
├── 8.2 合成数据定量验证                   [2–3 天]
│     ├── 在合成数据上跑 WIRE / DIP / SAA / MAP-TV
│     ├── 有 GT：PSNR / SSIM / boundary F1 / Chamfer
│     └── 确认哪个方法在有 GT 时最优
│
├── 8.3 WIRE / DIP 4x 真实数据探索         [3–5 天]
│     ├── 输出 4x 网格 (1920×2560)
│     ├── FRC / edge-MTF / split-half 验证
│     ├── 多 seed 稳定性检查
│     └── Uncertainty map 输出
│
├── 8.4 小型 denoiser 训练 (可选)          [3–5 天]
│     ├── 在合成数据上训单通道 DDPM
│     ├── 用 DiffPIR 框架接入
│     ├── 在合成 benchmark 上验证
│     └── 如果好于 WIRE/DIP → 用于 4x 真实数据
│
└── 8.5 报告 + 物理极限结论                [2 天]
      ├── 4x 是否有真实新增信息（FRC 判断）
      ├── 如果有 → 作为论文主结果
      └── 如果没有 → 清楚报告物理极限，4x 标注为 "prior-assisted visualization"
```

---

## 4. 显存预估

| 实验 | 输入规模 | 模型 | 预估显存 | 3090 24GB |
|------|---------|------|---------|:---------:|
| WIRE 2x, 32 帧 | 32×480×640 | MLP 5层×256 | ~3–5 GB | ✅ |
| WIRE 2x, 255 帧 mini-batch | 8 帧/batch | MLP 5层×256 | ~4–6 GB | ✅ |
| DIP U-Net 2x | 1×960×1280 output | ~2M params | ~6–10 GB | ✅ |
| Deep Decoder 2x | 1×960×1280 output | ~100K params | ~2–4 GB | ✅ |
| WIRE 4x | 1×1920×2560 output | MLP 5层×256 | ~6–10 GB | ✅ |
| 小型 DDPM 训练 | 128×128, batch=32 | U-Net ~5M params | ~8–12 GB | ✅ |
| DiffPIR 推理 4x | 1920×2560 | 预训练 denoiser | ~10–16 GB | ✅ |

> **结论：所有方案都在 24GB 显存范围内，无需排除任何因显存不足。** 两块 3090 可以并行跑不同实验。

---

## 5. 已有代码的复用计划

| 现有资产 | 复用方式 |
|---------|---------|
| `algos/ep06_sr_poc/src/common/forward_model.py` | 翻译成 PyTorch 版；保留 NumPy 版做 cross-validation |
| `algos/ep06_sr_poc/src/common/alignment.py` | 直接复用 shift 加载逻辑 |
| `algos/ep06_sr_poc/src/common/data_loader.py` | 直接复用帧加载、session 筛选 |
| `algos/ep06_sr_poc/src/common/metrics.py` | 扩展：加 FRC、edge-MTF |
| `core/src/thermal_core/plotting.py` | 继续用 CVPR 学术风格出图 |
| EP06 evaluation pipeline | 复用 split-half / Chamfer / gradient / artifact score |

---

## 6. 对调研方案的最终评价

### 「深度学习-生成模型方法调研」(847 行)

**价值很高的部分：**
- §2 问题重新定义 → 精准，forward model 公式化
- §5.1 SuperF/WIRE INR 实现蓝图 → 直接可用（几何 loss 设计）
- §5.2 DIP/Deep Decoder → 作为我们的 P1 对照
- §9 评估指标 → forward consistency + contour stability 框架直接采用

**需要跳过的部分：**
- §5.3 DiffPIR/DAPS/Flower → 需要 prior 预训练，是 EP09+ 的事
- §7 合成数据渲染工具（ThRend/Mitsuba/pbrt）→ 对我们来说 Python procedural generator 就够了
- §8 现有数据集 → 与芯片场景差距太大
- §10 8 周计划 → 过于线性，我们需要更紧凑的 2–3 周迭代

### 「reposeEP07」(762 行)

**价值很高的部分：**
- §2.2–2.3 多尺度边缘检测流程 → EP07.4 直接用
- §2.4D ridge/double-erf 模型 → 内部镂空结构的正确物理模型
- §4.3–4.5 FRC + split-half + edge-MTF → 无 GT 评价金标准
- §3.5 风险评估表 → 已验证（MAP-TV 过锐化被预警到了）

**需要跳过的部分：**
- §1.2–1.3 传统多帧反卷积 / IBP / POCS → EP06 已做过，天花板已知
- §6 论文表述建议 → 太早，先出结果

---

## 7. 需要你确认的决策

> [!IMPORTANT]

1. **EP07 起手是 WIRE 还是 DIP？**
   - 推荐：**WIRE 先行**（场景匹配度最高），DIP 作对照
   - 如果你更偏好先做简单的：可以先 DIP（3 天出结果），再做 WIRE

2. **EP07 是否同时做多尺度检测？**
   - 推荐：**并行做**（在另一块 3090 上跑，或者纯 CPU 任务）
   - 如果只想集中火力：先做 WIRE/DIP，多尺度推到 EP07 后半段

3. **合成数据生成器放 EP07 还是 EP08？**
   - 推荐：**EP08 开头做**（EP07 先用真实数据验证 DL 方法可行性）
   - 如果你想更稳：EP07 就开始做简单版生成器，用于 WIRE/DIP 的 smoke test

4. **4x 是什么定位？**
   - 推荐：**EP08 做探索，如果 FRC 不支持就标注为 prior-assisted visualization**
   - 你接受这个定位吗？还是 4x 必须是硬性目标？
