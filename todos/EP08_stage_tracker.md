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
- Stage 2: 新方法（Deep Decoder / DeepInverse-DIP）先在 TCForge 合成数据上 benchmark，确认算法实现正确后再跑真实数据
- Stage 3: Geometry loss ablation 等可选实验**必须先在 TCForge 上验证**
- Stage 4: 合成数据上的定量指标（有 GT 的 PSNR/SSIM）作为补充证据

---

## 总览

| Stage | 目标 | 状态 | 输出位置 |
|:-----:|------|:----:|----------|
| 0 | P0 数值地基：forward/highpass 等价 + 三模型骨架 | ✅ 完成 | `algos/ep08_inr_sr/` |
| 1 | 小规模 SIREN/WIRE 训练（32帧×256²） | ✅ 完成 | `output/ep08_inr_sr/{siren,wire}_stage1/` |
| 2 | Deep Decoder + DeepInverse-DIP 对照 → 小规模四方报告 | ✅ 完成 | `output/ep08_inr_sr/{deep_decoder,deepinv_dip}_stage2/` |
| 3 | 赢家扩展到全量（64→128→255帧 + 全分辨率） | 🟡 基础设施完成，长训练未启动 | `algos/ep08_inr_sr/scripts/stage4_controller.py` |
| 4 | 统一评估 + 最终四方对比报告 | 🟡 远程 runbook 完成，等待训练产物 | `research_log/episodes/ep08_inr_sr/stage4_remote_operation_guide.md` |

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
- 34 个单元测试全过；validate_p0.py 通过

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

## Stage 2: Deep Decoder + DeepInverse-DIP 对照 ✅

**目标**: 在与 SIREN/WIRE 完全相同的配置下训练 Deep Decoder 和 DeepInverse-DIP，回答"INR 连续表示 > CNN decoder 离散表示？"，并用 DeepInverse 成熟实现交叉验证我们的 Deep Decoder 是否写对了。

**科学问题**:
1. INR (SIREN/WIRE) > CNN decoder (Deep Decoder)？
2. 我们的 Deep Decoder 实现是否与成熟库（DeepInverse）一致？
3. 哪个方法最不容易 hallucinate？

**必须共享的配置**: 同一 32 帧、同一 256×256 patch、同一 train/val split（seed=42）、同一 forward operator、同一五项指标。

**TCForge benchmark**: 在真实数据训练前，先在 TCForge 合成数据（默认 `lr_shape=(256,256)`, `n_frames=32`）上跑 SIREN、WIRE、Deep Decoder 和 DeepInverse-DIP，确认四个方法在 HR highpass GT 下闭环正确。

**完成内容**:
- Deep Decoder 正式训练完成，复用 Stage 1 基础设施，且 `split_indices.json` 与 SIREN Stage 1 bit-exact 一致。
- DeepInverse-DIP 集成完成：保留 `deepinv==0.4.0` 的 `ConvDecoder` backbone，使用 EP08 自定义训练循环、固定 latent 和 hold-out early stopping；split-half 使用相同初始化的短预算数据稳定性检查。
- TCForge benchmark 完成，四个方法在同一 HR highpass GT 场景上均达到 PSNR ≥ 12 dB。
- EP06 MAP-TV patch-level baseline 已按 EP08 32 帧、256×256、seed=42 split 生成五项指标。
- 五方对比表与柱状图已生成：`output/ep08_inr_sr/stage2_comparison.csv/png`。

**TCForge benchmark 指标**:

| Method | Highpass PSNR (dB) | Global SSIM proxy |
|---|---:|---:|
| SIREN | 24.6634 | 0.7520 |
| WIRE | 24.4126 | 0.7325 |
| Deep Decoder | 19.8896 | 0.2068 |
| DeepInverse-DIP | 18.4729 | 0.4447 |

**Stage 2 真实数据指标**:

| Method | Hold-out residual | Split-half NRMSE | Artifact score | Raw-control agreement | P95 gradient | Best step | 结论 |
|---|---:|---:|---:|---:|---:|---:|---|
| SIREN | 3.6729 | 0.2916 | 0.2178 | 0.2085 | 0.9289 | 1630 | 综合最均衡 |
| WIRE | 2.9665 | 0.4899 | 2.2434 | 0.0868 | 1.6098 | 2989 | 高频强但 artifact 高 |
| Deep Decoder | 3.6258 | 0.4406 | 0.0355 | 0.3227 | 0.4885 | 6685 | 原生 512 输出后更均衡 |
| DeepInverse-DIP | 2.3051 | 0.4562 | 3.0867 | 0.0608 | 1.2143 | 9900 | forward consistency 强但 artifact 高 |
| EP06 MAP-TV | 2.2439 | 0.1743 | 0.3978 | 0.3102 | 0.8667 | 100 | 同协议 classic baseline |

**EP06 patch baseline 来源**:
- `output/ep08_inr_sr/ep06_patch_baseline/metrics.json`
- protocol: `ep08_32_frame_256_patch_seed42_split`
- five metrics are computed through the EP08 metrics pipeline.

**Stage 2 结论**:
1. 四方法 TCForge benchmark 均通过 highpass PSNR ≥ 12 dB，说明实现和 forward wrapper 没有明显几何/shape 错位。
2. DeepInverse-DIP 的 hold-out residual 很低，但 artifact score 最高、raw-control agreement 最低；它是强拟合上限参照，不是 Stage 3 首选。
3. Deep Decoder 修复后不再是 64×64 插值色块，artifact 最低且 raw-control agreement 较高，但 split-half 和 hold-out 仍不优于 SIREN。
4. SIREN 与 EP06 MAP-TV 是 Stage 3 的主要对照轴；Deep Decoder 可作为低 artifact 深度 decoder 对照。

**门控**:
- Deep Decoder + DeepInverse-DIP 都产出有效结果：通过
- 五方对比表完整：通过
- EP06 baseline 开始填入且不伪造不可用指标：通过

**输出**:
- `output/ep08_inr_sr/deep_decoder_stage2/` — 我们的 Deep Decoder
- `output/ep08_inr_sr/deepinv_dip_stage2/` — DeepInverse ConvDecoder-DIP
- `output/ep08_inr_sr/tcforge_benchmark/` — TCForge HR-GT benchmark
- `output/ep08_inr_sr/stage2_comparison.csv` — 五方对比表
- notebook fragments 更新
- research log 更新

---

## Stage 3: 赢家扩展到全量 🟡

**目标**: 将 Stage 2 候选推进到 64→128→255 帧 full-frame progressive 训练。当前执行计划包含 SIREN、WIRE、DeepInv-DIP 与 EP06 MAP-TV；Deep Decoder 已在 Stage 2 作为低 artifact 对照完成，不进入默认 full-frame 长训练。

**扩展路径**:
```
Step 2: 64 帧 + 全分辨率 (640×480 → 1280×960)
Step 3: 128 帧 + 全分辨率
Step 4: 255 帧 + 全分辨率（最终提交）
```

**每步检查点**: hold-out residual 不上升、无新增 artifact、raw-control 一致性。

**远程编排基础设施**:
- `algos/ep08_inr_sr/scripts/stage4_controller.py`
- `research_log/episodes/ep08_inr_sr/stage4_remote_operation_guide.md`
- Controller 默认每次 `tick` 启动当前阶段可安全运行的后台任务：MAP-TV(cpu)、SIREN(cuda:1)、DeepInv-DIP(cuda:0)，WIRE 等 SIREN 完成后在 cuda:1 串行启动。
- Controller 只做 `metrics.json` 数值健康门控和日志/pid 追踪；不启动 notebook，不替代视觉检查，不做训练代码修改。

**门控**（参考计划 §4.3，按当前可用 baseline 修正）:
- hold-out residual：使用 EP08 自身协议的阶梯比较（32帧→64帧→128帧→255帧），每步不应恶化超过 10%
- split-half NRMSE：同上阶梯比较
- artifact score ≤ EP06 MAP-TV full-frame proxy（0.1437）
- raw-control agreement：窗口化 SSIM ≥ 0.9（Stage 3 引入）
- pin 区域无伪线
- 多 seed（3次）std < 5%
- ESF 10-90% 边缘宽度 < bicubic
- EP06 MAP-TV 只作 full-frame proxy；如需同协议 classic baseline，先重算 EP06 在 EP08 split 下的指标（参考 Fix 5）

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
| EP06 full-frame proxy 已填入；patch-level same-protocol baseline 待 Fix 5 | 🟡 中 | Stage 3/4 对比 |
