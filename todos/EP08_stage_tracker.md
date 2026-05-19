# EP08 Stage Tracker — 全阶段任务记忆

> **本文件是 EP08 的持久进度追踪文件。** 每个 Stage 完成后必须更新本文件。
> 任何参与 EP08 的智能体在开始工作前必须先读本文件，了解当前进度和上下文。

---

## EP07/TCForge 依赖

**EP07 (ThermalChipPhantom / TCForge)** 是 EP08 的合成数据供应方。TCForge 位于 `tcforge/`，是独立 UV 项目，提供：
- **可控 HR ground truth**: 合成芯片几何结构 + 已知温度分布
- **EP06 兼容 forward model**: shift → PSF blur → downsample，边界约定与 EP06 一致
- **LR burst 生成**: 按 raster 扫描路径生成 multi-frame 观测
- **Highpass 等价性验证**: 内置 EP06 forward reference 交叉检查
- **评估 API**: `evaluate_scene()` / `evaluate_dataset()` 用于有 GT 的定量评估

**使用场景**:
- Stage 2: 新方法（Deep Decoder / DeepInverse-DIP）先在 TCForge 合成数据上 sanity check，确认算法实现正确后再跑真实数据
- Stage 3: Geometry loss ablation 等可选实验**必须先在 TCForge 上验证**
- Stage 4: 合成数据上的定量指标（有 GT 的 PSNR/SSIM）作为补充证据

---

## 总览

| Stage | 目标 | 状态 | 输出位置 |
|:-----:|------|:----:|----------|
| 0 | P0 数值地基：forward/highpass 等价 + 三模型骨架 | ✅ 完成 | `algos/ep08_inr_sr/` |
| 1 | 小规模 SIREN/WIRE 训练（32帧×256²） | ✅ 完成 | `output/ep08_inr_sr/{siren,wire}_stage1/` |
| 2 | Deep Decoder + DeepInverse-DIP 对照 → 小规模四方报告 | 🔜 待执行 | `output/ep08_inr_sr/{deep_decoder,deepinv_dip}_stage2/` |
| 3 | 赢家扩展到全量（64→128→255帧 + 全分辨率） | ⬜ 未开始 | — |
| 4 | 统一评估 + 最终四方对比报告 | ⬜ 未开始 | — |

---

## Stage 0: P0 数值地基 ✅

**目标**: 建立 PyTorch forward/highpass 与 EP06 NumPy 的逐参数数值等价性；实现三模型骨架。

**完成内容**:
- `forward.py`: PyTorch forward operator，mode=constant，与 EP06 max_abs_error < 1e-5
- `highpass.py`: mode=nearest 背景估计，与 EP06 等价
- `splits.py`: 位移相位分层 hold-out split
- `metrics.py`: 五项指标（holdout_residual, split_half_nrmse, artifact_score, raw_control_agreement, p95_gradient）
- `models/`: SIREN + WIRE + Deep Decoder
- `trainer.py`: INRTrainer + TrainConfig
- `data.py`: 真实数据加载（复用 EP06 loader + EP05 shifts）
- 31 个单元测试全过；validate_p0.py 通过

**审计发现（已在 Stage 1 修复）**:
- 🟡 WIRE 原为单线性 Gabor → Stage 1 修正为双线性
- 🔵 训练脚本不从 YAML 读配置 → Stage 1 修正

---

## Stage 1: 小规模 INR 训练 ✅

**目标**: 32 帧 × 256×256 LR patch 的 SIREN + WIRE 正式训练，验证 INR 框架可行性。

**配置**: seed=42, 27 train / 5 val, lr=5e-4, max_iter=10000, batch_k=8, warmup=200, early_stop_patience=1000

**Stage 1 指标**:

| 指标 | SIREN | WIRE | 赢家 |
|------|------:|-----:|:---:|
| Hold-out residual | 3.67 | **2.97** | WIRE |
| Split-half NRMSE | **0.29** | 0.49 | SIREN |
| Artifact score | **0.22** | 2.24 | SIREN (10×) |
| Raw-control agreement | **0.21** | 0.09 | SIREN |
| P95 gradient | 0.93 | 1.61 | — |
| Best step / Final step | 1630/2630 | 2989/3989 | — |

**关键发现**:
1. 两个 INR 方法均成功训练并收敛，框架可行
2. SIREN 综合更优（4/5 指标），特别是 artifact + 稳定性
3. WIRE 边缘更锐但 artifact 10× 更高，split-half 不稳定
4. SIREN omega_0=10 保守配置表现良好
5. 两者 HR highpass 图都清晰展示了芯片内部结构轮廓

**重要修改**:
- WIRE 修正为 carrier/envelope 双线性 Gabor (wire.py)
- 新增 `stage1.py` 统一训练模块（717行），三个训练脚本变为薄 wrapper
- Trainer 增加 best_state restore、train/val indices 注入、early_stop_min_steps

---

## Stage 2: Deep Decoder + DeepInverse-DIP 对照 🔜

**目标**: 在与 SIREN/WIRE 完全相同的配置下训练 Deep Decoder 和 DeepInverse-DIP，回答"INR 连续表示 > CNN decoder 离散表示？"，并用 DeepInverse 成熟实现交叉验证我们的 Deep Decoder 是否写对了。

**科学问题**:
1. INR (SIREN/WIRE) > CNN decoder (Deep Decoder)？
2. 我们的 Deep Decoder 实现是否与成熟库（DeepInverse）一致？
3. 哪个方法最不容易 hallucinate？

**必须共享的配置**: 同一 32 帧、同一 256×256 patch、同一 train/val split（seed=42）、同一 forward operator、同一五项指标。

**TCForge sanity check**: 在真实数据训练前，先在 TCForge 合成数据（小场景，如 `lr_shape=(64,96)`, `n_frames=16`）上跑一遍 Deep Decoder 和 DeepInverse-DIP，确认算法实现正确（有 HR GT 可算 PSNR）。

**门控**:
- Deep Decoder + DeepInverse-DIP 都产出有效结果
- 四方对比表完整（SIREN/WIRE/DeepDecoder/DeepInverse-DIP）
- EP06 baseline 开始填入（至少 hold-out residual）

**输出**:
- `output/ep08_inr_sr/deep_decoder_stage2/` — 我们的 Deep Decoder
- `output/ep08_inr_sr/deepinv_dip_stage2/` — DeepInverse ConvDecoder-DIP
- `output/ep08_inr_sr/stage2_comparison.csv` — 四方对比表
- notebook fragments 更新
- research log 更新

---

## Stage 3: 赢家扩展到全量 ⬜

**目标**: 只对 Stage 2 综合排名第一的方法做渐进扩展（最多两个方法）。

**扩展路径**:
```
Step 2: 64 帧 + 全分辨率 (640×480 → 1280×960)
Step 3: 128 帧 + 全分辨率
Step 4: 255 帧 + 全分辨率（最终提交）
```

**每步检查点**: hold-out residual 不上升、无新增 artifact、raw-control 一致性。

**门控**（参考计划 §4.3）:
- 255 帧 hold-out residual ≤ EP06 MAP-TV × 1.1
- 255 帧 split-half NRMSE ≤ EP06 MAP-TV
- 255 帧 artifact score ≤ EP06 MAP-TV
- Raw-control SSIM ≥ 0.9（需要引入真正的窗口化 SSIM）
- Pin 区域无伪线
- 多 seed（3 次）稳定性 std < 5%
- 边缘宽度 ESF 10-90% < bicubic

**可选实验**（Stage 3 Step 4 通过后才考虑）:
- Geometry loss ablation（binary/sparse edge/angle loss）
- 坐标旋转实验（47.6°）
- 需先在 TCForge 合成数据上验证

**VRAM 注意**: 255 帧 + 全分辨率 (960×1280 HR) 需要 batch_k=16 帧采样策略。

**TCForge 前置**: Geometry loss 等可选实验必须先在 TCForge 合成数据上验证有效再用于真实数据。

---

## Stage 4: 统一评估与报告 ⬜

**目标**: 产出完整四方对比报告，回答三个科学问题。

**最终四方对比表**:

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

**必须回答的科学问题**:
1. 深度方法是否超过经典天花板？→ best(INR/DD) vs MAP-TV
2. INR > CNN decoder？→ SIREN/WIRE vs Deep Decoder
3. Gabor > Sine？→ WIRE vs SIREN

**Notebook 报告要求**: 每方法至少 4 张 ROI 对比图 + 每图表附解读。

---

## 遗留技术债

| 问题 | 优先级 | 影响阶段 |
|------|:------:|:--------:|
| `raw_control_agreement` 非窗口化 SSIM | 🟡 中 | Stage 3 门控 |
| `stage1.py` 717 行偏大 | 🔵 低 | 可维护性 |
| `data.py` 的 sys.path 注入脆弱 | 🔵 低 | 鲁棒性 |
| EP06 baseline metrics 仍为 null | 🟡 中 | Stage 2 四方对比 |
