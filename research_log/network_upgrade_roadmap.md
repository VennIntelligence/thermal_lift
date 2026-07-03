# Network Upgrade Roadmap — post-v3 数据生成之后怎么走

> ⚠️ **2026-07-05 状态更新（权威记录见 ACL-046～048，与本文冲突处以 ACL 为准）**：
> - **Step 5 已完成且数字更新两次**：EP15 M2 在 20µm 契约下的权威频带，旧对齐下为 34.07µm；换用精修对齐（现 repo 默认 `configs/alignment/stage0f_refined_alignment.csv`）后为 **25.45µm ±0.73**（负对照保持、aperture dip 可见）。band gate 应引用 25.45µm。
> - **风险 #1（shift 精度）已实测并大部分解决**：真实逐帧误差 ~0.29px（"p95≈0.79px"为旧对齐旧口径），0a 精修一轮收敛，残余 0.012/0.071px。其"缓解=训练时 shift-jitter DR"已被实测判 null（DR@0.1px 两臂无差异）；**主线改为精修对齐直接喂 DC + 测试时联合精化**。
> - **§4 的"PSF σ=0.2257 标定值"不可信**：EP09 三路发散未解决，0a 类仪器测不了 σ。σ 仍是开放校准项，loss/DC 中勿当"钉死值"引用。
> - **Step 4 的"synthetic split-half FRC 快 sanity"照旧，但真实域主 gate 改为 cross-method FRC**（self split-half FRC 对神经方法结构性失效，奖励可复现幻觉）。
>
> 写于 2026-06-25。目的:把"数据生成完后该不该把 U-Net 换成 diffusion / flow matching、要不要拿现成超分底子微调"这个决策**落盘**,并给出 cold-reopen 也能直接续上的实现次序。
> 配套:决策记录见 `research_log/algorithm_changelog.md` ACL-024;跨会话记忆见 memory `thermal-lift-redesign-direction`。

---

## 0. 决策(headline)

- **不**把主干换成 diffusion / flow matching 当**主**架构。
- **不**拿别人训练好的 RGB 超分/扩散模型(Real-ESRGAN / SwinIR / StableSR / SD)当底子微调。
- **承诺方向**:physics-constrained **unrolled solver**(确定性)+ band-limited 监督。这是 memory `thermal-lift-redesign-direction` 已定的方向,本 roadmap 把它具体到现有代码。
- 生成模型(diffusion/flow)**只在最后**、**只**作为 unrolled solver 里的 plug-in **不确定度先验**考虑,不是替代主干。见 §5。

---

## 1. 为什么不上 diffusion / 不用现成底子(grounded)

1. **计量 ≠ 自然图超分**。diffusion/flow 的核心优势是建模多峰后验、**采样**出合理纹理——自然图要的就是这个"幻觉"。我们做的是热成像**计量**,幻觉出来的高频看着锐、置信高,却不是真信号,且事后分不清信号 vs 幻觉。对测量仪器是主动有害。
2. **我们自己的 band-limited 原则就否决了它**。ACL-023:"GT 信息卡在可恢复频带内,label 只在 band 内追精度"——这是**确定性恢复**框架:信息在测量里就存在,只需解卷积/解混叠。diffusion 的多峰优势只在**频带之外**(我们明确不追的幻觉区)才有意义。越坚持 band-limited,diffusion 的边际价值越接近零。
3. **现成 RGB 底子域差太大 + VAE 致命**。底子都在 RGB 自然照片上训;我们是单通道热红外,噪声统计/空间频谱/先验全不同,先验大半是错的偏置。尤其 SD 系 latent diffusion 在 8× 下采样、RGB 上训的 VAE 隐空间里跑,**恰在我们要恢复的高频段**有不可逆重建误差——SR 上限被 VAE 保真度卡死。
4. **数据效率**。diffusion 先验数据饥饿(自然图要百万级,我们 5000 scene 塞牙缝);unrolled solver 数据高效(物理算子干大部分活,网络只学 proximal 一步)。5000 scene + 已认证 forward 算子,**强烈**偏向求解器。
5. **经验证据**:loss-side forward 锚定**已被证伪**(`losses.py:299` `forward_model_loss` 默认 `forward_model_weight=0`,ACL-017/019 发现软 loss 不够)。下一步的正确动作是把同一个算子从**软惩罚**升级成**硬约束**——也就是 unrolling。
6. **5090/32G 的正确用法是把对的东西做大**:更多 unroll 迭代、更深 prox 网络、更狠 domain randomization、训得更久——不是换范式。32G 不够从头训高分扩散,微调 latent diffusion 又回到第 3 条的 VAE 坑。

---

## 2. 现有资产(已经在仓库里,直接复用)

| 资产 | 位置 | 在新方案里的角色 |
|---|---|---|
| 可微 torch forward `A` | `algos/ep07_unet_sr/src/unet_sr/losses.py:299` (`forward_model_loss` = `gaussian_blur_2d`→`avg_pool2d`);`gaussian_blur_2d` 在 `:289` | 升级成**硬 data-consistency 层**;转置 `Aᵀ` 由 autograd 免费给(conv+avg_pool 的反传) |
| numpy 匹配 forward+adjoint | `tcforge/src/tcforge/_ep06_reference/forward.py`:`forward` `:97`、`adjoint` `:116`、`ObservationOperator` `:138`、`build_observation_operator` `:179` | **adjoint 参照真值**:用 dot-product test 验证 torch `Aᵀ` 正确 |
| forward self-check(5 PASS) | `scripts/forward_roundtrip_selfcheck.py` + `output/forward_selfcheck/selfcheck_summary.json` | 提供两个 load-bearing 常数(见 §4) |
| 现有 U-Net | `algos/ep07_unet_sr/src/unet_sr/model.py:93` `ThermalSRUNet`(~5M params,scale=1 同分辨率 refiner) | 直接当 unroll 里的 **proximal 网络** `R_θ` |
| V10 = 已经是 1 步 unroll | `train.py:227`(`model_scale=1`)、`:348-351`(`pred = drizzle_mean(obs ch5) + UNet_delta`,L1 罚 delta) | 升级 = 把这 1 步**重复 K 次 + 中间插 DC 步**,不是重写 |
| v3 数据 + 配置 | `scripts/generate_training_pool.py`、`configs/synthetic/pool_2x_v3.json`(5000 scenes,输出 `data/synthetic/pool_2x_v3_5k`) | 训练数据 |
| 权威 FRC | `algos/ep15_info_limit/scripts/run_m2_frc.py:273` `frc_curve`(1/7 与 half-bit 阈值) | band gate + eval |
| 要打败的经典基线 | MAP-TV `algos/ep06_sr_poc/src/map_tv/map_tv.py`、TGV `algos/ep10_tgv_sr/src/ep10_tgv_sr/tgv.py`、IBP `algos/ep06_sr_poc/src/ibp/ibp.py` | "real SR gain" 的及格线:band 内干净超过它们 |

---

## 3. 实现次序(cold-reopen 可直接续)

> 数学:HR 估计 `x`(2× 网格,单通道);第 i 帧观测 `y_i`,已知 sub-pixel shift `s_i`、PSF σ。每帧前向 `A_i x = avg_pool( gaussian_blur( shift(x, s_i) ), scale )`。求 `min_x ½ Σ_i ‖A_i x − y_i‖² + R(x)`,用 **learned proximal-gradient / HQS 展开 K 步**。

**Step 0 — 跑 5K 数据生成(远端 5090)**
- 用户贴生成命令(5K,64 workers,~hours,~350GB)。配置已就绪 `pool_2x_v3.json`,远端 `data/synthetic/pool_2x_v3_5k` 已 symlink 到 `/mnt/d`(memory `thermal-lift-machines`)。
- 这是其它步骤的前置。

**Step 1 — torch、batched、shift-aware 前向 `A_i` + autograd 转置**
- 现有 `forward_model_loss` 只 blur+pool 单张 `lr_observation`,**没有 per-frame shift**。要新写一个含 `shift(x, s_i)` 的批量算子(可微 grid_sample / FFT 相移)。
- **必须验证**:dot-product/adjoint test `⟨A_i x, y⟩ ≈ ⟨x, A_iᵀ y⟩`,以 §2 的 numpy `ObservationOperator` 为真值对照;并核对 §4 的 +0.5 偏置与标定 σ。
- 落点:新模块 `algos/ep07_unet_sr/src/unet_sr/forward_torch.py`(建议)。

**Step 2 — K 步 unrolled solver(复用现有 U-Net 当 prox)**
- `x₀` = 经典 drizzle 暖启(现有 hybrid 通道,`dataset.py` 的 `HYBRID_DRIZZLE_MEAN_CHANNEL=5`)。
- 每步 k:① **DC 步**(硬物理锚)`x ← x − η_k · Σ_i A_iᵀ(A_i x − y_i)`,`η_k` 可学;② **prox 步** `x ← R_θ(x)`,`R_θ` = 现有 `ThermalSRUNet`(权重可共享或 per-step)。
- K=3–8 起步。这就是 V10 的 `x = drizzle + UNet_delta` **重复 K 次、中间多了 DC 步**——增量演进。

**Step 3 — band-aware loss + 标定 σ**
- 终端 `x_K` 用现有 `ContourSRLoss`,但**加 band gate**:残差只在 EP15 认证的可恢复频带内计 loss(Fourier 带通加权,或 `forward_model_loss(band="highpass")` 思路推广),兑现"label 只在 band 内追精度"。
- 保留一个终端 DC 项 `‖A x_K − y‖²` 保持诚实。
- 前向算子的 σ 用**标定值 0.2257 LR-px(§4)**,不是占位 0.5。

**Step 4 — eval / 及格线**
- in-loop:synthetic split-half FRC(快 sanity)。
- 权威:离线 EP15 FRC(`run_m2_frc.py`)在**真实 248 帧**上。
- **及格线**:band 内 FRC 干净超过经典 TGV/MAP-TV(§2)。达不到就别声称 real SR gain。

**Step 5(应在 Step 3/4 之前先做的前置)— 远端重跑 EP15 定 20µm 权威频带**
- 当前 EP15 输出是旧 10µm 标尺下的,20µm 重标定后**尚未重跑**。band gate 与 paper 的 FRC µm 数都依赖它。先跑,再信 band-limited 数字。

---

## 4. 风险 / load-bearing 约束(别踩)

1. **shift 精度(头号风险)**。solver 在 `A_i` 里用**假定**的 `s_i`;真实 refined shift 噪声大(拟合不出单一刚性 (θ,pitch),残差 p95≈0.79px on ~3px 信号,ACL-023)。`s_i` 错则 DC 与真值打架。
   - 缓解:训练时做 **shift-jitter domain randomization**(渲染 `y_i` 用的 shift 与喂给 `A_i` 的 shift 加扰),让 solver 学会 shift-鲁棒。
   - 进阶(先 defer):把 shift 设为可 refine 的变量做联合估计。
2. **+0.499 HR-px block-center 偏置**(self-check T1)。torch `A_iᵀ` 必须复现这个常数偏置,否则反投影错位。以 numpy adjoint 为准对齐。
3. **PSF σ = 0.2257 LR-px**(self-check T5 标定值),不是 loss 里占位的 0.5。20µm 下的权威 σ 应由 EP09 PSF 标定 / EP15 复核钉死。
4. **box 采样有 ~3% 真实混叠**(T3,> 0.9 Nyquist);是物理,不是 bug——别假设理想 band-limiting。最细旋转线可能要把 SSAA 调高(T3 note)。
5. **EP15 未在 20µm 重跑**(见 Step 5)。band gate 与 paper FRC µm 数在重跑前不可信。

---

## 5. 生成模型——只在最后、只有一个正确姿势

如果确定性 solver 干净打赢经典后,还想要(a)标定的不确定度,或(b)觉得 band 内细节没吃干净,**才**考虑加生成先验:

- **做法**:DPS / ΠGDM / DDRM 式**后验采样**——diffusion/flow 当**正则先验**,物理 `A_i` 每步**强制 data-consistency**。和 unrolled solver 是**组合**,不是替代。
- **真正的回报是不确定度图,不是更锐的点估计**:采样多张 HR,逐像素方差 = "哪块细节被数据约束(方差小=真) vs 先验编的(方差大=可疑)"。对测量仪器这张图是金子,正好补 solver 没有的东西。
- 先验在**自己的热数据**上训一个**小**模型,**绝不**碰 RGB 底子、**绝不**用 latent-diffusion VAE(§1.3)。
- 真要走生成,**flow matching > DDPM**(训练更稳、采样步数少)。

---

## 6. 硬规矩

- **一次只动一个变量**:先把确定性 unrolled solver 做到打赢经典,**再**谈生成先验。两个一起上 → 增益归因不清。
- **band gate 一切**:不追 band 外精度(那是幻觉)。
- **任何上限/截断都要 log**:别让 silent truncation 读成"全覆盖"。
