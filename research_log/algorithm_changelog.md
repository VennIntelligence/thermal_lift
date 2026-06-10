# 算法变更日志 (Algorithm Changelog)

> **用途**: 每次对 SR 算法（网络架构、Loss 设计、训练策略、数据管线）做出修改时，必须在此记录变更条目。  
> **格式**: 按时间倒序排列，每条变更必须包含：问题诊断、修改内容、预期效果、训练结果。  
> **规则**: 见 `AGENTS.md` 中「算法变更日志规则」。

---

## 变更记录

### [ACL-012] 2026-06-10 — EP15 M4 GPU MAP-TV 去卷积锚重跑

**问题诊断**:
- EP06 旧 MAP-TV 结果不可作为 4x baseline：`psf_sigma=1.0` 已超出 M3 支持的可信区间 `0.2-0.5 LR px`，`max_iter=4` 远未收敛，lambda 只取单点，forward model 没有包含探测器孔径 box integration。
- EP12 4x 网络没有显示真实增益，后续网络方法需要一个经典、可复现、必须超越的 “baseline to beat”。

**修改内容**:
1. 新增 `algos/ep15_info_limit/scripts/run_m4_deconv_anchor.py`：用 PyTorch GPU batch 实现 `BatchForwardModel`，一次处理 248 帧 shift / Gaussian PSF / detector box downsample；`adjoint()` 反向执行 upsample / PSF / reverse shift 并累加梯度。
2. 默认 forward model 改为 `HR -> shift -> Gaussian PSF -> avg_pool2d detector box -> LR`，`--no-box` 仅作为消融开关。
3. MAP-TV 主循环使用 FISTA + smoothed TV gradient，full run `max_iter=150`，输出 `iteration,data_rmse,tv_value,objective,relative_update` 收敛曲线。
4. 参数网格改为 `sigma={0.2,0.3,0.4,0.5} LR px`、`lambda={3e-4,1e-3,3e-3}`；每个 sigma 先用 odd/even split-half NRMSE + artifact/std proxy 选 lambda，再用全 248 帧跑 full reconstruction。
5. 新增四臂视觉对比、sigma=5 highpass 对比、zigzag 定量剖面、split-half FRC 复验和全参数选择 CSV。

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
