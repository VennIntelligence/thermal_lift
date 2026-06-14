# Supplementary C — 方法实现细节（中文草稿）

> **角色**: technical appendix 的方法块，支撑主文 §4；目标是让审稿人/复现者不读代码也能还原管线。
> **语言**: 中文草稿（2026-06-12 决策），迁 LaTeX 时翻译为英文。
> **对应清单**: `10_writing_handover.md` §3.C（C.1/C.2/C.4/C.5 ★；C.3 原标「等 V9C」，其 config.json 已落盘故差异表可填，仅训练结果待跑完）。

---

## C.1 TCForge 物理匹配合成平台（对应主文 §4.2）

### C.1.1 模块结构

| 模块（`tcforge/src/tcforge/`） | 职责 |
|---|---|
| `geometry.py` | IC 布局 coverage mask 生成 + 4×SSAA |
| `physics.py` | 温度渲染、PSF、噪声、漂移 |
| `forward.py` | LR burst 退化（观测重放） |
| `fusion.py` | 1x 观测特征融合（5 通道） |
| `classical_sr.py` | drizzle / shift-and-add 特征（2x 证据通道） |
| `storage.py` | scene 紧凑存取 |

### C.1.2 场景几何分布

`build_scene_mask_with_metadata(difficulty, seed)`（`geometry.py` L639–1009）：8 层随机原语（大块 → 中宽柱 → L 形走线（25% 粗/75% 细）→ 框 → 细部/引脚阵列 → 圆/via → 对边 pin → 减通道），全部 seed 确定性。难度由特征尺度 (major, minor, line)/µm 控制：

| difficulty | major | minor | line |
|---|---:|---:|---:|
| easy | 80 | 60 | 45 |
| medium | 55 | 42 | 30 |
| hard | 35 | 28 | 22 |
| stress | 22 | 16 | 12 |

场景整体旋转 θ = 47.6° + U(−1.5°, +1.5°)——围绕实测 stage 角度抖动，匹配真实几何取向。

### C.1.3 4×SSAA coverage 抗锯齿

默认 `antialias=True, ssaa_factor=4`：在 4× 画布绘制 → order=1 旋转 → 4× 块均值降采得 coverage ∈ [0,1]（`geometry.py` L1006–1008）。**设计意图**：HR 目标不含阶梯锯齿、亚像素细线携带 coverage 缩放后的幅度——网络不被训练去复制锯齿（EP08/EP11 时代锯齿目标的教训）。

### C.1.4 温度渲染与物理随机化

T = T_bg + ΔT·coverage + A·smooth_noise（`physics.py::render_temperature_field`）。默认随机化（`DEFAULT_PHYSICS_RANDOMIZATION`）：

| 参数 | 分布 |
|---|---|
| ΔT | U(0.5, 5.0) °C（覆盖实测内/外轮廓对比度 1.94/2.49 °C，supp A.1.4） |
| PSF σ | U(0.15, 0.55) LR px（覆盖采纳区间 [0.2,0.5]，supp A.5.3）；30% 椭圆 + 10% Airy |
| 噪声 | lognormal，均值 0.0724 °C（= 实测噪声底）；模式 `detector_realistic` |
| 漂移幅度 | U(0, 0.3) °C |
| shift | `real_default_contour_refined` profile 重放（248 帧/scene）+ jitter σ=0.02 px |

**Forward 模式**：`physical_block_average`（HR 一次 PSF blur → 各 shift 分块双线性均值，scale 2/4）为训练池默认；`exact_ep06_point` 为 EP06 点采样对照。

### C.1.5 训练池与 burst 变体

生成：`scripts/generate_training_pool.py --config configs/synthetic/training_pool_2x*.json`。

| 池 | 规模 | 用途 |
|---|---|---|
| `training_pool_2x_aa` | 1000 scenes × 248 帧；难度 easy 200/medium 400/hard 300/stress 100；LR 480×640 → HR 960×1280 | 1x 统计输入各臂（v8.1a/b、v9b/d） |
| `training_pool_2x_aa_burst` | 同上 + `save_lr_burst=true`（~152 GB） | hybrid 输入各臂（V9A/V9C/V10） |

**K=4 drizzle 变体**（`scripts/precompute_drizzle_variants.py`，ACL-016）：k=0 全 248 帧无 shift 噪声（canonical，与推理一致）；k=1..3 保留 60–100% 帧（≥30）+ shift N(0, 0.05 px)。输出 `drizzle_variants_2x.npy`（K,3,H,W float16，bilinear kernel）；训练时按 [seed, epoch, scene, 0xBEEF] 确定性抽样（`dataset.py` L330）——帧子集与 shift 噪声增广「烘焙」进变体，避免训练时现场 drizzle 卡死 dataloader。

**存储 schema**（`storage.py`）：`hr_mask_4x.png`、`hr_edge_4x.png`、`obs_features_1x.npz`、`shifts.npy`、`metadata.json`（+ 可选 `lr_burst.npy`、`drizzle_variants_2x.npy`）。

**C.1 资产依赖**：`tcforge/src/tcforge/`；`configs/synthetic/training_pool_2x*.json`；`scripts/{generate_training_pool.py, precompute_drizzle_variants.py}`；S-T3（合成参数全表 = 本节表格汇总）。
**C.1 待回填**：无——可成稿。

---

## C.2 网络与损失（对应主文 §4.3）

### C.2.1 UNet 骨干（各臂固定）

`ThermalSRUNet`（`algos/ep07_unet_sr/src/unet_sr/model.py`）：编码 3×（MaxPool + ConvBlock[2×3×3 Conv + GroupNorm + SiLU + SE]），通道 c/2c/4c/8c（c=base_channels=64）；解码双线性上采 + skip concat + ConvBlock。HR 头两种：

- **bilinear**（默认交付）：双线性 ×scale → 3×3 Conv；
- **pixelshuffle**（v8.1b 消融臂）：ICNR 初始化 sub-pixel conv + PixelShuffle + HRResBlock——**负结果**，条纹伪影 + proxy 全程更差（supp D.4.1），保留为 head 归因对照。

hybrid 输入模式下网络在 2x 网格上等效 scale=1 运行（输入已是 2x，无最终上采样）。参数量由 `train.py` 运行时打印（base 64、8ch 输入约个位数 M）。

### C.2.2 ContourSRLoss 总式

L = w_m·MSE + w_hp·HP + w_e·Edge + w_s·(1−SSIM) + w_gv·GV + w_lap·Lap + w_fm·FM + w_r·R

| 项 | 定义（`losses.py`） | conservative 权重 |
|---|---|---:|
| MSE | E[(pred−target)²]，可乘 gap_weight | 0.3 |
| HP highpass 结构 | ‖HP(pred)−HP(target)‖₁，HP = x − G_{σ=5}(x)，乘 structure/thin/gap 权重图 | 0.8 |
| Edge | ‖|∇pred|−|∇target|‖₁ + 0.25×半分辨率版 | 0.05 |
| SSIM | 1−SSIM（11×11 Gaussian 窗） | 0.15 |
| GV grad-vector | ‖gx_p−gx_t‖₁+‖gy_p−gy_t‖₁（全向量，捕捉边缘膨胀/畸变——幅值型 edge loss 看不见的方向） | 0.15 |
| Lap | ReLU(|Δtarget|−|Δpred|) 均值（只罚「比目标钝」） | 0（仅 v6 用 0.1） |
| FM forward consistency | MSE(AvgPool_s(G_{σ_PSF·s}(pred)), obs)，band=full / highpass（σ_band=5 LR px 两侧同减低通） | 0（锚定臂 0.1） |
| R residual penalty（V10） | w_r·mean(|δ|)，δ = pred − 输入 ch5（drizzle mean@2x） | 0（V10 扫 λ） |

**权重图机制**（`mask_weights.py`）：structure_boost——HP/GV 权重 ×(1 + b·‖∇target‖/max)；thin_boost——结构像素且宽度 ≤3 HR px ×倍率；gap_boost——两侧有结构的 ≤3 px 背景窄缝 ×倍率。

### C.2.3 hot vs conservative 权重对照（损失演化史的浓缩）

| 参数 | hot（CLI 默认） | conservative（v8.1a 壳，V9 全系沿用） |
|---|---:|---:|
| mse_loss_weight | 0.2 | 0.3 |
| highpass_loss_weight | 1.0 | 0.8 |
| structure_boost | 4.0 | 2.0 |
| grad_vector_weight | 0.3 | 0.15 |
| thin_boost | 6.0 | 3.0 |
| gap_boost | 4.0 | 2.0 |

历史动机一句话：skeleton boost 30 时代 → 振铃；loss cooldown A/B 证明降温不解决中心细线模糊（输入瓶颈，主文 §6.3）→ conservative 作为公平 loss 壳冻结，v6（hot+Lap+full FM）保留为漂移放大参照臂。

### C.2.4 hybrid 8 通道输入（V9A/V9C/V10）

`dataset.py::_build_hybrid_obs` / 推理 `inference.py::infer_from_burst`：

```
ch 0–4: 1x fused 统计（aligned_mean/median/coverage/variance/highpass_fused）双线性 ↑2x
ch 5:   drizzle mean @2x      ← 观测域证据主通道（V10 残差基准）
ch 6:   drizzle coverage @2x  ← 空洞可识别（未观测 bin mean 填全局均值、coverage=0）
ch 7:   drizzle variance @2x
```

V9C 合法锚配套：hybrid 的 ch0 是上采样均值（不是合法 1x 观测），故 forward 锚另走数据管线携带原生 1x aligned-mean patch（偶数 2x origin 裁剪、增广同步）。

**C.2 资产依赖**：`algos/ep07_unet_sr/src/unet_sr/{model.py, losses.py, mask_weights.py, dataset.py, config.py}`；`algos/ep07_unet_sr/scripts/run_v9.md`（conservative 壳定义）。
**C.2 待回填**：⬜ 参数量精确数字（跑 `train.py --help`/启动日志摘录，1 分钟）。

---

## C.3 训练 config 对照表（七臂全字段差异）

**共同字段**：scale=2、base_channels=64、patch_size_hr=256、total_steps=60000、lr=2e-4、AMP、compile、edge/ssim/coarse = 0.05/0.15/0.25、highpass_sigma=5、real_eval：248 帧 + contour_refined + center-1/3 + zoom3。差异字段（各 `outputs/*/config.json`）：

| 字段 | v6 | v8.1a | v8.1b | v9b | v9d | V9A | V9C |
|---|---|---|---|---|---|---|---|
| 训练池 | 2x | 2x_aa | 2x_aa | 2x_aa | 2x_aa | 2x_aa_burst | 2x_aa_burst |
| input_mode / in_ch | lr / 5 | lr / 5 | lr / 5 | lr / 5 | lr / 5 | **hybrid / 8** | **hybrid / 8** |
| HR 头 | bilinear | bilinear | **pixelshuffle+res1** | bilinear | bilinear | bilinear | bilinear |
| loss 壳 | **hot+Lap0.1** | cons. | cons. | cons. | cons. | cons. | cons. |
| forward 锚 | full 0.1 | 无 | 无 | **highpass 0.1** | **full 0.1** | 无 | **legal-1x highpass 0.1** |
| batch_size | 128 | 128 | 128 | 128 | 128 | **64**（35K 中断后续跑） | **64** |
| save_every | 2000 | 5000 | 5000 | **1000** | **1000** | 5000 | 5000 |

**T2 消融矩阵直读**（input × anchor）：{1x, hybrid} × {none, band, full, legal} = v8.1a / v9b / v9d / V9A / V9C（v6、v8.1b 为 loss 温度与 head 的额外归因臂）。

**可比性 caveat（必写）**：V9A 前 35K bs=128、之后与 V9C 全程 bs=64——hybrid 双臂与 1x 各臂训练动力学不完全同条件；V9A 30K 保真悬崖与 bs 切换重合，混杂未排除（V10 恒定 bs 顺带检验，`docs/next_move_plan.md` §8）。

**C.3 资产依赖**：`algos/ep07_unet_sr/outputs/ep07_{v6_physics, v8_1a_loss_cooldown, v8_1b_pixelshuffle, v9a_hybrid_drizzle, v9b_fwd_consistency, v9c_hybrid_legal_fwd, v9d_fwd_fullband}/config.json`。
**C.3 待回填**：⬜ V9C 训练结果行（60K 完整曲线 + 选点，等今晚训练自然结束后 C4 收尾流程）；⬜ V10 三臂行（等 ACL-020 实验落地，含 residual_mode/residual_penalty_weight 字段）。

---

## C.4 经典方法实现细节（对应主文 §4.1）

### C.4.1 Drizzle（EP10）

STScI `drizzle.resample.Drizzle`（`algos/ep10_drizzle/src/ep10_drizzle/drizzle_sr.py`）：pixfrac 默认 0.7（sweep {1.0, 0.8, 0.7, 0.6, 0.5}）、kernel `square`、输出网格 (H×2, W×2)、pixmap HR 坐标 = 2×(col+dx, row+dy)（shift 约定 [dx,dy] LR px 观测→参考系）、coverage = out_wht，<1.0 置 NaN。TCForge 训练侧 scatter 用 bilinear kernel（C.1.5）——两种 kernel 的差异在 D.7 fine-window 指标中以「drizzle 输入通道」参照点体现。

### C.4.2 各向异性 coverage 加权 TGV（EP10，实用经典交付）

外层 FISTA：x_{k+1} = TGV_prox(z_k − η∇D(z_k))（`algos/ep10_tgv_sr/src/ep10_tgv_sr/tgv.py`）。

- **各向异性对偶投影**：TGV 内层（Chambolle-Pock 路径）一阶对偶球改椭圆 {(a,b): (a/r_a)²+(b/r_b)² ≤ 1}，r_a = α₁·r_y、r_b = α₁（r_y = aniso_ratio_y = 1.5）；二阶对称张量同构造（r_yy = α₀r_y、r_xx = α₀、r_yx = √(α₀²r_y)）。动机 = raster 各向异性（supp B.3.1）。
- **coverage 加权数据梯度**：每 HR 像素梯度除以预计算 coverage（bilinear splat + PSF adjoint 得到），替代统一除帧数 N——修掉 bilinear scatter 把权重集中在固定 HR 行导致的水平条纹。
- **参数**（`run_tgv_sr.py` sweep）：λ_tv ∈ {3e-4, 1e-3, 3e-3} × σ_PSF ∈ {0.18, 0.50}，α_ratio=2.0（α₁=λ, α₀=λ/2），max_iter=100、inner 80。
- **修复效果**：artifact 3.870 → **0.695**（−82%），raw-control corr 0.902 → **0.916**，30.8 min CPU（ACL-007；TB-scale，TGV 口径注意 A.3.1 红线）。

### C.4.3 MAP-TV 去卷积锚（EP15 M4，验收 gate）

`algos/ep15_info_limit/scripts/run_m4_deconv_anchor.py`：forward = shift → Gaussian PSF（σ LR px）→ avg_pool box；GPU FISTA + smoothed-TV 梯度，150 iter（relative update ~0.005 收敛）。

- **网格**：σ ∈ {0.2, 0.3, 0.4, 0.5} × λ ∈ {3e-4, 1e-3, 3e-3}；
- **选择规则**：每 σ 内 split-half（偶/奇帧）proxy = split_half_nrmse + 0.05×artifact + 0.08×std_excess 取最小，再跨 σ 选全局最优 → **σ=0.2, λ=1e-3**；
- **成本**：4563 s GPU；
- **角色**：验收锚——学习臂必须在 FRC-band 一致性与 zigzag 轮廓指标上同时不输它才可采纳（主文 §5.5）。结果数字（FWHM 114→100 µm 等）归档在 supp D.2/D.5。

**C.4 资产依赖**：`algos/ep10_drizzle/`、`algos/ep10_tgv_sr/src/`、`algos/ep15_info_limit/scripts/run_m4_deconv_anchor.py`；`output/ep10_tgv_sr/{sweep_results.csv, best_hr_temperature.npy, run_summary.json}`；`research_log/algorithm_changelog.md` ACL-007/ACL-012。
**C.4 待回填**：无——可成稿。⬜ TGV 对偶投影推导若要给完整公式块（含步长/收敛条件），迁 LaTeX 时从 `tgv.py` 注释整理（半页内）。

---

## C.5 checkpoint 选择协议（对应主文 §4.3 末段 + §6.6）

### C.5.1 规则（`algos/ep07_unet_sr/scripts/plot_checkpoint_selection.py`）

每臂在其全部 eval checkpoint 上：

1. 归一化 proxy 对：a_norm = (artifact − min)/(max − min)；c_norm = (max_corr − corr)/(max_corr − min_corr)（corr 反向）；
2. 理想点距离 d = √(a_norm² + c_norm²)，升序遍历；
3. **5K 窗口去重**：每 checkpoint 归属 window = step // 5000，同窗只留最先入选者 → 最多 3 个候选；
4. **末端对照**：60K 若未入选则强制追加为 drift reference（不参与选优）；
5. rank-1 为 canonical；
6. **视觉 gate**：每候选拼 `eval_real/unet_step{K}_center_zoom3x_temperature.png` 三联 panel（温度域，不只 highpass）标注 proxy 值，人工过门后定稿。

### C.5.2 伪代码（主文引用版，5 行）

```
cands = top3_by_ideal_distance(normalize(artifact↓, corr↑), min_gap=5K)
cands += {60K}                      # always carry drift reference
canonical = argmin_ideal_distance(cands)
panels = render_temperature_panels(cands)
deliver(canonical if visual_gate(panels) else next_candidate)
```

### C.5.3 已执行选点结果（TB-scale，EP11 四臂）

| 臂 | canonical | artifact / corr | 60K 端点 | 端点惩罚 |
|---|---|---|---|---|
| v6 | 8K | 0.330 / 0.774 | 0.883 / 0.648 | artifact ×2.7 |
| v8.1a | 15K | 0.392 / 0.758 | 0.643 / 0.689 | — |
| v8.1b | 5K | 0.370 / 0.739 | 0.709 / 0.667 | — |
| v9b | 11K | 0.339 / 0.777 | 0.655 / 0.688 | — |

（`output/ep11_dl_benchmark/checkpoint_selection/checkpoint_candidates.csv`；TGV Pareto 参考点 (0.695, 0.916)。）主文金句的数据基础：**按端点上报会让每个臂交出最差 checkpoint**。

**C.5 资产依赖**：`algos/ep07_unet_sr/scripts/{extract_checkpoint_metrics.py, plot_checkpoint_selection.py}`；`output/ep11_dl_benchmark/checkpoint_selection/{checkpoint_metrics.csv, checkpoint_candidates.csv, fig_pareto.png, panel_*.png}`；F4/S-F4 同源。
**C.5 待回填**：⬜ V9A/V9C 选点行（V9A 可立即跑选点脚本补入；V9C 等 60K）；⬜ V10 三臂选点（等训练）；hybrid 臂与 1x 臂的 proxy **不跨列横比**（A.3.4 推论 2）要在表注重申。
