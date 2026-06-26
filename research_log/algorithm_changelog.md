# 算法变更日志 (Algorithm Changelog)

> **用途**: 每次对 SR 算法（网络架构、Loss 设计、训练策略、数据管线）做出修改时，必须在此记录变更条目。  
> **格式**: 按时间倒序排列，每条变更必须包含：问题诊断、修改内容、预期效果、训练结果。  
> **规则**: 见 `AGENTS.md` 中「算法变更日志规则」。

---

## 变更记录

### [ACL-028] 2026-06-26 — 修复 hybrid real_eval 推理仍生成 8ch 输入导致 9ch 模型崩溃

**问题诊断**:
- ACL-023/027 后训练数据集的 `input_mode="hybrid_drizzle2x"` 已固定为 9ch: `5 fused↑2x + 4 phase-bin drizzle@2x`。
- 但 `infer_from_burst()` 的真实数据评估路径仍按旧 V9A contract 拼接 `5 fused↑2x + 3 scatter drizzle@2x`，在 checkpoint real_eval 时向 9ch 模型输入 8ch tensor，触发 `expected input ... to have 9 channels, but got 8 channels instead`。
- 该问题只影响推理/real_eval 入口，训练 batch 本身已经读取预计算 `phase_bin_drizzle_2x.npy`。

**修改内容**:
1. `inference.py`: hybrid 推理路径改用 `tcforge.classical_sr.phase_bin_drizzle(..., n_bins=4)`，与训练 dataset 的 9ch phase-bin contract 对齐。
2. `tests/test_inference.py`: hybrid inference 回归测试新增 9ch 检查，避免只用输出 shape 掩盖输入通道错误。
3. `config.py` / `solver_train.py`: 更新 CLI help 和注释中的旧 8ch/3ch scatter 说法。
4. `tests/test_dataset.py`: 测试 fixture 使用 `phase_bin_drizzle_2x.npy`，覆盖默认 hybrid 与 `provide_burst=True` solver 路径。

**预期效果**:
- `train.py` 在 `save_every` / `real_eval` 节点不再因 8ch/9ch 不匹配中断。
- 真实数据 TensorBoard/PNG 推理输入与合成训练输入保持同一通道语义。
- 风险: real-data eval 仍需现场从 raw burst 计算 phase-bin drizzle；这会比旧 3ch scatter 略有计算成本，但只发生在 checkpoint eval。

**推荐参数**: 保持 ACL-027 命令不变；若只想快速越过训练节点，可临时加 `--real-eval-frame-limit 48` 降低 eval 成本。

**训练结果**: _(训练后填写)_
- 输出目录: `outputs/v10_v4_acl027`
- 视觉效果:
- 关键指标:
- 结论:

**涉及文件**: `algos/ep07_unet_sr/src/unet_sr/inference.py`, `algos/ep07_unet_sr/src/unet_sr/config.py`, `algos/ep07_unet_sr/src/unet_sr/train.py`, `algos/ep07_unet_sr/src/unet_sr/solver_train.py`, `algos/ep07_unet_sr/src/unet_sr/unroll.py`, `algos/ep07_unet_sr/tests/test_inference.py`, `algos/ep07_unet_sr/tests/test_dataset.py`, `algos/ep07_unet_sr/tests/test_model_losses.py`

---

### [ACL-027] 2026-06-26 — Loss/评价指标重设计:thin/gap 线先验 → 几何无关 boundary 权重 + 等温 flatness;评测改用 held-out 合成 GT(out_of_band 取代 raw_control_corr)

**问题诊断**:
- 旧 loss/指标过度针对早期"完美矩形/细线"几何:`thin_boost=6 / gap_boost=4` 本质是直线探测器先验,在 v4 缺陷数据(孔/裂纹/断角,均 >pitch)上要么无意义要么帮倒忙(过度加权裂纹细发丝、欠加权大孔边界)。
- `real_eval` 的 `artifact_score` / `raw_control_corr` 拿"干净输出"与"退化原图(bicubic of raw mean)"对比:输出越干净越偏离参照 → 分数反而越差("但凡生成更干净的图,分数反而降低")。
- 结构性错位:有 GT 的合成集反而不记保真指标,没 GT 的真实集才记(被迫用"和退化图比"的代理)—— 正好反了。

**修改内容**:
1. `mask_weights.compute_boundary_weight_np`:`1 + boost·exp(-(dist/τ)²)`,从 mask 边界距离场算 —— 几何无关地强调每种边界(芯片外缘/孔壁/裂纹壁/缺口);细线=处处贴边界、窄缝=两侧贴边界,均作为同一距离场的特例自然涌现;对比无关 → 低 ΔT(hard/stress)缺陷也照样加权。替换 thin/gap(`--boundary-boost/--boundary-tau-px`)。
2. `ContourSRLoss`:thin/gap → 单一 `boundary_weight`(驱动 highpass + grad_vector);mse 回归全局 DC 锚;新增 `flatness` 项(在 GT 平坦处罚 `‖∇pred‖`,用对比归一化的 target 梯度做软掩膜,不与真实边界打架),编码近等温先验(默认关,v4 用)。
3. `metrics.py`(新,纯 numpy):`out_of_band_ratio`(GT-free/PSF-free,pitch 截止频率以上的谱能量=幻觉/珠串,本地验证 smooth/mid=0、Nyquist=1)、`psnr`、`region_rmse`(体内温度=等温保真)、`boundary_f1`(缺陷/边界保真:填孔↓recall、造边↓precision,本地验证填孔 recall 1.0→0.85)。
4. `real_eval`:删 `raw_control_corr`,改记 `out_of_band_ratio`;`artifact_score` 降级为 FM-1 cliff 监视器(只看跨 checkpoint 的相对跳变)。
5. `synth_eval.py`(新)+ 数据集 held-out 尾切片(`holdout_tail/holdout_role`,scene 目录数字序,训练自动排除尾部,零泄漏):有 GT 的合成集上记 `eval_synth/{psnr,region_rmse,boundary_f1,out_of_band_ratio}`,经 forward_fn 闭包同时接入 train.py(plain UNet)与 solver_train.py(unrolled solver)。

**预期效果**:
- 指标不再奖励"像退化图";真正的保真(区内 RMSE、缺陷 F1)与幻觉(带外能量)被分别诚实量化。
- 边界强调对新缺陷几何通用;线先验的经验作为特例保留。
- 风险:boundary/flatness 权重需调;`boundary_f1` 的梯度阈值是相对百分位 → 跨 run 看趋势,不看绝对值。

**预期效果验证(本地 CPU)**: boundary 权重(每种边界→5.0、内部→1.0、细线全程加权)、4 个指标、config 解析/校验全部本地通过;9 个 src + 3 个 test 文件 py_compile 通过;torch 部分(loss/dataset/synth_eval forward)留远端 pytest。

**推荐参数**: `--boundary-boost 4.0 --flatness-weight 0.0 --synth-eval-holdout 200 --synth-eval-every 2500`(第一跑 flatness 关;下一跑 `--flatness-weight 0.05` 做 A/B,用 eval_synth 判增益)

**训练结果**: _(训练后填写)_
- 输出目录: `outputs/v10_v4_acl027`(V10 baseline)/ `outputs/solver_v4_acl027`(solver)
- 关键指标: eval_synth/region_rmse↓, boundary_f1↑, eval_real/out_of_band_ratio 平
- 结论:

**涉及文件**: mask_weights.py, losses.py, metrics.py(新), real_eval.py, synth_eval.py(新), dataset.py, config.py, solver_train.py, train.py; tests: test_model_losses / test_dataset / test_config; docs/REMOTE_ORDERS.md

---

### [ACL-026] 2026-06-25 — solver 架构修正:end-on-DC + 冻结 eta(把"硬 DC"真正做硬;软锚定降级为监视)

> 接 ACL-025。v2(anneal 8000 + dc_weight 0.5)训练复盘 → 定位到 `unroll.py` 实现的结构缺陷,做架构级修正,而非再调权重。

**问题诊断**(v2 远端 log):
- DC 从平滑暖启地板 **0.021(step1)单调爬到 0.061(step2000)**,冲向 v1 同一高位;`eta` 0.500→0.477 继续下漏;`gnorm` 随 anneal 上升(0.17→0.52)说明梯度几乎全来自 struct、DC 项梯度可忽略。
- **DC 升到暖启地板之上 = 幻觉签名**:平滑暖启过 A 再 highpass,残差 ≈ −highpass(burst)(地板);网络加的高频与 burst 不相关时方差叠加,DC 升过地板 → 加的是**物理不符**的高频。struct(宽容、亚像素容忍)下降的同时 DC(严格、相位敏感)上升 = 网络找了个"GT 神似但前向不符"的解。
- **根因(unroll 实现缺陷,非加权)**:① 每轮 `DC→prox` 顺序,**prox 末位发言**,无约束残差 UNet 在 DC 步之后重新注入幻觉高频,输出 `x_K` 不被任何 DC 步修正;② **eta 可学**,优化器把 DC 步往 0 调、架构层面绕过约束;③ 软 DC loss 项本质是已证伪的 loss 侧软锚定(ACL-017/019),且 `A^T(highpass)` 梯度天生被转置上采样抹平,打不过 struct 尖锐的边缘/梯度向量梯度(`structure_boost=4`)。三者叠加 → 再调 dc_weight/anneal 都是给漏底设计打补丁。

**修改内容**(`unroll.py` / `config.py` / `solver_train.py`):
1. **end-on-DC 重排**:每轮改 `prox → DC`,循环以 DC 步收尾 → 输出 `x_K` **构造上落在数据一致方向**,prox 不再有最后发言权;struct loss 经终末 DC 步反传进 prox(autograd 双反传),把 prox 耦合到观测。
2. **冻结 eta**(`solver_learn_eta=False` 默认):eta 改为 buffer、不进 optimizer,DC 步每次满强度开火、不可被调 0 绕过;新增 `--solver-learn-eta` 可切回可学(A/B 用)。
3. **软 DC loss 降级为可选监视**:`solver_train` 当 `dc_weight=0` 时在 `no_grad` 下算 `dc` 仅作监视(TB `loss/dc` 仍可看),不建图、不进 loss;`>0` 时仅作弱次级正则,不再是主机制。
4. docstring 更正:删除原"prox 只能填零空间、结构上无法覆盖观测"的过度声称(实现并不成立),改为诚实描述 end-on-DC 的保证 + "单步梯度非完整投影"的 caveat。

**预期效果 / 验证次序**:
- 终末 DC 步把输出拉回数据一致,DC 不再升过暖启地板;`eta` 恒定(TB 一条平线=冻结生效)。
- 推荐下一轮用**纯架构**配置:`--solver-prior-anneal-steps 0 --solver-dc-weight 0`(total=struct,一致性全交给架构),盯 `loss/dc` 监视曲线是否被架构压住。
- **先 Gate B**(单 clean 场景,确认 end-on-DC + 冻结 eta 把 DC 残差驱→~0,即架构有效)PASS 再上池子。
- **判据**:纯架构下 DC 仍压不住 → 单步梯度太弱,升级为终末几步 **CG 硬投影**(docstring 已标 caveat 路径);DC 被压住但 struct 学不动 → 回调。

**训练结果**:_(待远端;Gate B 验证架构 → 池子重启 `outputs/solver_v3_arch`,纯架构 anneal0/dcw0)_。

---

### [ACL-025] 2026-06-25 — solver v1 首训诊断:DC 项被欠权重无视(anneal-off + dc_weight 0.1)→ 上 prior 退火 + 抬 DC 权重

> 本条是 unrolled solver(ACL-024)**首个 20K 训练**的中途诊断 + 训练策略修正。仅 CLI 旗标变更,无代码改动。

**问题诊断**(远端 5K log;配置 no-drizzle / K=4 / M=12 / patch192 / batch18):
- `struct` 正常下降:0.081(step1500)→ ~0.04(最低 0.023@4500)。监督路通。
- **`loss/dc` 卡死在 ~0.055–0.064,不降反微升**;且 `dc(0.057) > struct(0.042)`——DC 是个**绝对量级比 struct 还大、却被无视**的量,不是可忽略噪声。
- **`eta`(可学 DC 步长)单调下漏**:0.484 → 0.456(step1500→5000),方向是优化器在"关小"物理修正步(未塌到 0)。
- **机制**:`total = 1.0*struct + solver_dc_weight(0.1)*dc`。AdamW 对 loss 整体缩放近似不变,真正起作用的是**权重比 struct:dc = 10:1**,struct 从 step0 满权主导 → 优化器榨干 struct、无视 DC,并顺势调小 eta。
- **排除算子 bug**:Gate 0(hp-corr 0.999)/ Gate A(adjoint 0.0)/ **Gate B 在单 clean 场景把 DC 压到 ~0** 已认证算子+几何+数据;机器有能力降 DC。故此为**训练加权动力学**,非 FM-6 系统误差。DC 漂移=求解器在绕开物理锚点,退化成"贵的 plain UNet",违背架构反幻觉初衷。

**修改内容**(无代码改动,现成旗标;**从头重启**,因 anneal 是 step0 起的 schedule,续训会错过 DC 主导早期相):
1. `--solver-prior-anneal-steps 8000`:`total = anneal*struct + 0.5*dc`,早期 anneal≈0 → **只剩 DC** 先压数据一致性,prior 在 0–8K 线性 ramp 填零空间(正是为 FM-1 cliff 设计、ACL-024 第一条基线刻意先关的开关)。
2. `--solver-dc-weight 0.1 → 0.5`:稳态比 10:1 → **2:1**(DC 占 33%),保证 prior ramp 满后 DC 仍是硬锚不反弹。
3. **隔离变量**:patch192/batch18/M12/unroll4/clip 全部保持与 v1 一致;v1 5K ckpt 保留作对照。`gnorm` 持续 3–4 顶 clip=1.0 暂不动(留作下一变量)。

**预期效果**:
- DC 主导期(0–8K)`loss/dc` 明显下降(目标 0.057 → <0.03),`eta` 止跌回升。
- prior ramp 满(8K 后)DC 维持低位不反弹。
- **判据**:若 8K 后 DC 又爬 → 下轮 `dc_weight`→1.0;若连 0–8K 都压不动 DC → 停,提示更深问题(非纯加权)。

**训练结果**:_(待远端;重启为 `outputs/solver_v2_dcanneal`,盯 0–8K 的 dc/eta 曲线)_。

---

### [ACL-024] 2026-06-25 — 决策记录:不上 diffusion / 不用现成底子,承诺 unrolled solver(roadmap 落盘)

> 本条是**决策记录(ADR)**,非代码变更。完整 roadmap 见 `research_log/network_upgrade_roadmap.md`。

**决策点**:v3 数据(5K)生成完后,是否把 U-Net 升级为 diffusion / flow matching?是否拿别人训好的超分/扩散模型(Real-ESRGAN / SwinIR / StableSR / SD)当底子在我们数据上微调?远端 5090 有 32G。

**决策**:
1. **不**把主干换成 diffusion/flow 当**主**架构;**不**用现成 RGB 底子微调。
2. **承诺** physics-constrained **unrolled solver**(确定性)+ band-limited 监督(沿用 memory `thermal-lift-redesign-direction` 已定方向)。
3. 生成模型**只在最后**作 unrolled solver 里的 plug-in **不确定度先验**(DPS/ΠGDM 式后验采样)考虑,不替代主干。

**理由(grounded)**:① 计量要 data-consistent 恢复、不要生成式幻觉;② 我们 ACL-023 的 band-limited 原则本就是确定性恢复框架,diffusion 的多峰优势只在我们不追的 band 外;③ 现成 RGB 底子域差大、latent-diffusion 的 VAE 恰好毁掉要恢复的高频;④ 5000 scene 偏向数据高效的 solver,不偏向数据饥饿的 diffusion;⑤ **经验证据**:loss-side forward 锚定已被证伪(`losses.py:299` `forward_model_weight=0`,ACL-017/019)→ 下一步是把同一算子升级成**硬约束(unrolling)**;⑥ 5090/32G 应"把对的东西做大"(更多 unroll 迭代/更深 prox/更狠 randomization),不够从头训高分扩散。

**落盘的实现次序**(详见 roadmap):Step 0 远端跑 5K 生成 → Step 1 torch shift-aware 前向 `A_i` + autograd 转置(用 numpy `ObservationOperator` 做 adjoint dot-product 验证)→ Step 2 K 步 unroll(DC 步 + 现有 U-Net 当 prox,drizzle 暖启;V10 已是 1 步 unroll)→ Step 3 band-aware loss + 标定 σ → Step 4 eval(EP15 FRC,及格线=band 内打赢经典 TGV/MAP-TV);前置 Step 5 远端重跑 EP15 定 20µm 权威频带。

**load-bearing 约束**:① shift 精度是头号风险 → 训练加 shift-jitter randomization;② torch `A_iᵀ` 必须复刻 +0.499 HR-px block-center 偏移(self-check T1);③ 用标定 σ=0.2257 LR-px(T5),不是占位 0.5;④ EP15 未在 20µm 重跑,band 数字在此前不可信。

**硬规矩**:一次只动一个变量(先 solver 打赢经典,再谈生成先验);band gate 一切。

**涉及文件**:新增 `research_log/network_upgrade_roadmap.md`;复用 `algos/ep07_unet_sr/src/unet_sr/{losses.py,model.py,train.py}`、`tcforge/src/tcforge/_ep06_reference/forward.py`、`algos/ep15_info_limit/scripts/run_m2_frc.py`。

---

### [ACL-023] 2026-06-25 — 探测器 pitch 重标定(20µm)+ forward 算子认证 + v3 信息保存数据管线

**问题诊断**:
- ep01–ep22 主线结论是 `no GT-certifiable winner`:学习方法从未干净打过经典 TGV/MAP-TV,且训练后期合成先验"反吃"真实细节(V9A 保真悬崖,`hp_corr_input` 0.974→0.906)。两层根因:①输入端 5×1x 统计通道在进网络前坍缩了 248 帧亚像素相位;②训练分布"贴着标定"退化(旋转固定 47.6°±1.5°、复用真实那一组 248 shift、PSF 贴标定),且 GT 信息含量超过信息极限 → 模型被迫在零空间幻觉。
- 标定本身有错:EP03 由 BMP mm 标尺测得的 detector pitch `10 µm/pixel` 是 **2× 误读**(axis 与 contour cross-check 共用同一 BMP 标尺锚点故 lockstep 一致、无内部矛盾)。真实 pitch = **20 µm/pixel**,阵列仍 480×640(原始 TXT 实测确认)。系统从"2× 过采样"修正为"**临界采样**(分辨率≈pitch=20µm)",2× SR 的真实含义变成 **20µm→10µm 目标分辨率** —— 直接解释"2X 信息很少 / 放弃 4X"。

**修改内容**:
1. **Pitch 重标定(commit `1ae3177`,26 文件)**:configs(stage_calibration/coordinate_set/synthetic×7)、core 默认值(ep03/ep04/ep07_cache/displacement)、docs(dataset_description/AGENTS/fresh_start_guide)全部 10→20µm;派生数重算(FOV 6.4×4.8→12.8×9.6mm、40µm 命令位移 4.0→2.0px、PSF σ µm 2.26→4.5µm,σ in LR px=0.226 不变);EP03 pitch 标 SUPERSEDED。DO-NOT-TOUCH:对齐 CSV、PSF σ(px)、scale/canvas、paper/(FRC µm 数值待远端 EP15 重跑)。
2. **forward 算子认证(commit `14110b5`)**:新增 `scripts/forward_roundtrip_selfcheck.py`,5 项全 PASS —— 约定无 dx/dy 交换(存在恒定 +0.5 HR-px block-center 偏移,求解器 adjoint 必须复刻)、in-band 可逆(shift-and-add/drizzle vs PSF-blurred GT corr 0.997)、box 采样物理性混叠 ~3%(realistic,需建模)、360° 旋转在内切圆内质量守恒、FRC 频带代理。**证伪了"原始生成代码本身是错的"这一担忧**。
3. **v3 信息保存 2× 数据管线(commit `14110b5`)**:tcforge 新增 `geometry.inscribe_disc`(360° 全随机旋转不裁角)、`shifts.random_constellation`/`build_scene_shifts`(每 scene 随机相位星座,good/medium/poor 覆盖,15% real-like 域匹配)、`classical_sr.phase_bin_drizzle`(4 个亚像素相位通道,显式暴露相位);`generate_training_pool.py` 改为每 scene 随机 N(24–96)+ 均匀 360° + phase-bin + stress SSAA=6 + 裸 f16 burst(**不压缩**,因生成 CPU-bound,zstd 只会加 CPU);config `configs/synthetic/pool_2x_v3.json`(5000 scenes)。tcforge 测试 75 passed。

**预期效果**:
- 输入保相位(burst+shifts+phase-bin)+ 分布全随机化(360°/星座/PSF)打破朝向与单一星座过拟合,GT 信息卡在诚实可恢复频带 → 不再把幻觉规模化。
- 认证过的 forward + 下一步物理约束展开式求解器(硬 data-consistency)把"防幻觉"从 loss 侧(ACL-017/019 已证伪)移到**架构侧**:模型结构上只能在 forward 零空间填先验,无法覆盖观测。

**核心设计原则(本轮确立)**:随机化"干扰参数"放很宽,但 GT 信息含量必须卡在诚实可恢复频带之上;label 是监督信号,只在 band 内追精度,band 外匹配 label = 幻觉 = 重蹈保真悬崖。

**远端就绪 / 数据量 / 耗时**:`data/synthetic/pool_2x_v3_5k` 已 symlink 到 5090 的 `/mnt/d`(1.6T 空闲);实测每 scene ~45–65MB(burst f16 ~53MB@91帧 + phase-bin 9.4MB + obs 1.3MB),5K 估 **~230–330GB**;smoke 6s/scene,因 2× + N~60 比旧 4×/248 帧轻很多,全 5K 预计几十分钟级(以 tqdm ETA 为准)。64 worker(RAM 限)。生成命令由用户在远端手动贴。

**训练结果**:_(本条为标定/数据管线/工具变更,无新增训练)_。数据生成待用户远端启动;下一步 Step 4 物理约束展开式求解器(data-consistency 硬约束 + band-aware loss/eval)。

**遗留 / caveat**:① 真实数据 PSF/对齐重标定仍需远端跑 `data/`(本地已拉 263 帧 txt 备用于此);② contour-alignment refined shifts 不能干净拟合单一刚性 (θ,pitch)(dx/dy 行给出不一致 θ、残差 p95≈0.79px on ~3px signal)—— stage command 仅 prior、对齐噪声/局部,展开式求解器对 shift 精度敏感需留意;③ `hr_mask_4x.png` 是误导性 legacy 文件名(内容实为 2× 960×1280),待清理;④ paper/ 的 FRC µm 数值与 null-space sinc 推导需远端 EP15 重跑后才能诚实更新。

**涉及文件**:commits `1ae3177`(重标定)、`14110b5`(v3 管线 + self-check);新增 `configs/synthetic/pool_2x_v3.json`、`scripts/forward_roundtrip_selfcheck.py`;tcforge `geometry.py`/`shifts.py`/`classical_sr.py`/`storage.py`/`__init__.py`、`scripts/generate_training_pool.py`。

---

### [ACL-022] 2026-06-14 — Task E 论文证据硬化：TGV actual split/FRC + F5b ROI2 + D.7 第二窗

**问题诊断**:
- 统一 harness ACL-021 的 TGV `split_half_nrmse` / `frc_*` 列仍使用 EP16 同子集 drizzle proxy，虽然已在表注声明，但审稿人可质疑 TGV 自身 split/FRC 是否一致。
- F5 主视觉只使用中心梳齿 ROI，容易被质疑为 cherry-pick。
- D.7 零训练融合 baseline 的 λ 在单一 fine-window 上选择，存在 selection-on-test 风险。

**修改内容**:
1. 新增 `algos/ep11_dl_benchmark/scripts/run_tgv_split_frc.py`：CPU-only 编排脚本，复用 EP16 `run_tgv_child.py` 子进程和 EP10 TGV 实现，对 full / split-A / split-B 分别运行各向异性 coverage-weighted TGV；full run 与 `output/ep10_tgv_sr/best_hr_highpass.npy` 做相对 L2 self-check，随后在 actual TGV half-set highpass 图上计算 split NRMSE 和 FRC。
2. 修改 `algos/ep11_dl_benchmark/scripts/run_unified_harness_t1_t2.py`：新增可选 `--tgv-split-json`，若 JSON 成功则优先读取 actual TGV split/FRC；否则默认退回 EP16 drizzle proxy。`--only` 刷新时合并已有 arm rows，避免只刷新 TGV 时截断 T1/T2 全表。
3. 新增 `scripts/paper_figures/fig05b_roi2_holdout.py`：固定分数 ROI2（rows `[0.270,0.415)`, cols `[0.530,0.685)`）生成 F5b 双域视觉图，并在第二 ROI 上重算 `lattice_score`、`sharp_p95`、profile zigzag proxies。
4. 新增 `algos/ep07_unet_sr/scripts/v9_review/run_fusion_window2.py`：固定第二验证窗 rows `542:676`, cols `478:674`，复用缓存全幅预测，对 D.7 融合 baseline 重新选择 λ，并比较 V10 工作点跨窗位置。
5. 更新 `docs/paper/07_experiments.md`、`docs/paper/09_figures_tables_assets.md`、`docs/paper/supp/D_full_results.md`：回填 actual TGV split/FRC、F5b ROI2 结论、D.7 第二窗结果，并保留 proxy / ROI / no-GT caveat。

**预期效果**:
- T1 的 TGV split/FRC 列不再依赖 drizzle proxy，可自洽回应 “TGV 自己的 split/FRC 呢”。
- F5 主视觉从单一中心 ROI 扩展到预声明 held-out ROI，降低 cherry-pick 风险。
- D.7 从单窗 λ 选择升级为双窗稳定性检查，结论限定为局部 proxy-frontier 压力测试，不升级为方法胜负。

**推荐参数**:

```bash
# E1 actual TGV split/FRC, CPU-only
CUDA_VISIBLE_DEVICES= uv run python algos/ep11_dl_benchmark/scripts/run_tgv_split_frc.py \
  --workers 4 --tgv-workers 4

# Refresh only the TGV row while preserving cached rows for other methods
cd algos/ep11_dl_benchmark
CUDA_VISIBLE_DEVICES= uv run python scripts/run_unified_harness_t1_t2.py \
  --only tgv \
  --tgv-split-json ../../output/ep11_unified_harness/tgv_split_frc.json \
  --device cpu --workers 4 --skip-f5

# E2 / E3 CPU-only checks
CUDA_VISIBLE_DEVICES= uv run python scripts/paper_figures/fig05b_roi2_holdout.py
cd algos/ep07_unet_sr
CUDA_VISIBLE_DEVICES= uv run python scripts/v9_review/run_fusion_window2.py
```

**训练结果**: _(本条为评估/论文证据硬化，无新增训练；2026-06-14 回填)_
- E1 TGV actual split/FRC: full-run self-check relative L2 = **0.0** vs EP10 submitted highpass anchor；split A/B 各 124 帧；total runtime 1248 s；TGV child backend status `aniso_forced_fallback`（预期 CPU anisotropic path）。Actual TGV split NRMSE = **0.03164**；FRC@20/16/14/12 µm = **0.978 / 0.975 / 0.969 / 0.955**；`frc_10um` 为 NaN（频带边界缺失），cutoff field = 10.0 µm 且 `crossed=False`，只作 split-consistency proxy。
- Harness TGV 行已刷新：`output/ep11_unified_harness/t1_metrics.csv` 与 `all_arm_metrics.csv` 全 arm success；TGV `split_half_source` / `frc_source` 指向 `output/ep11_unified_harness/tgv_split_frc.json`。
- E2 ROI2: F5b 资产生成到 `output/paper_figures/fig05b_main_visual_roi2.{png,pdf}`；ROI2 `lattice` 排序与中心 ROI 一致（drizzle < TGV < V10 < V9A60），但 `sharp_p95` 与 profile zigzag 排序部分不一致，因此只报告为 held-out visual/proxy audit。
- E3 第二窗: TGV×V9A60 λ 原窗 0.2、第二窗 0.1，λ 本身不完全稳定；第二窗 λ=0.1 为 `hp_corr_input=0.9643`, `hp_corr_tgv=0.9985`, `sharp_p95=0.5082`, `lattice=0.0126`，通过本窗 proxy-frontier gate。V10 λ=1.2@15K 第二窗为 `hp_corr_input=0.9199`, `sharp_p95=0.5008`, `lattice=0.0130`，仍低于本窗 TGV fidelity reference，保持“低 grain / 较锐但保真不足”的 proxy 位置判断。
- 结论: 三项加固均不改变 C1–C4 settle，不支持“学习方法打败 TGV”“更干净/更保真”或物理分辨率声明；所有新增证据限定为 2x contour-level、split-consistency、ROI-level visual/proxy。

**涉及文件**: `algos/ep11_dl_benchmark/scripts/run_tgv_split_frc.py`, `algos/ep11_dl_benchmark/scripts/run_unified_harness_t1_t2.py`, `scripts/paper_figures/fig05b_roi2_holdout.py`, `algos/ep07_unet_sr/scripts/v9_review/run_fusion_window2.py`, `docs/paper/07_experiments.md`, `docs/paper/09_figures_tables_assets.md`, `docs/paper/supp/D_full_results.md`

### [ACL-021] 2026-06-14 — 论文 T1/T2/F5 统一真实数据 harness

**问题诊断**:
- 论文最终 T1/T2 需要单一 EP11/common.metrics 口径，不能把 TensorBoard `eval_real/*` 的 artifact scale 与 EP11 harness scale 混进同一表。
- 旧 EP11 横评脚本只覆盖早期 1x 输入变体，无法安全评估 V9A/V9C hybrid-drizzle 输入和 V10 residual-over-observation；若未透传 `residual_channel=5`，V10 会退化成裸 delta 输出（缓存均值接近 0°C）。
- EP10 TGV 的 `best_hr_temperature.npy` 为 highpass/centered 产物，直接读作温度会得到约 0.24°C，必须经 TGV helper 重建普通 Celsius 温度图后再进入视觉对比。

**修改内容**:
1. 新增 `algos/ep11_dl_benchmark/scripts/run_unified_harness_t1_t2.py`：一次性评估 bicubic / drizzle / MAP-TV / TGV / v6 / v8.1a / v8.1b / v9b / v9d / V9A / V9C / V10-lam120@15K，并额外缓存 V9A-60K 作为 F5 late-drift visual control。
2. 复用 EP06 数据读取和 `common.metrics`、EP10 drizzle、EP15 FRC/zigzag probes、EP07 `infer_from_burst`，避免重新实现核心指标。
3. 对 `input_mode="hybrid_drizzle2x"` 的 arm 使用 `model_scale=1`；对 V10 `residual_mode="drizzle2x"` 自动设置 `residual_channel=HYBRID_DRIZZLE_MEAN_CHANNEL`（ch5），并在输出行记录 full/split cache 的温度均值 sanity check。
4. 统一输出 `all_arm_metrics.csv`、`t1_metrics.csv`、`t2_metrics.csv`、`tb_vs_harness_scale_check.csv`、`run_manifest.json`，并生成 F5 双域视觉图 `fig05_main_visual.{png,pdf}`。
5. 在 manifest 与 source columns 中显式标注边界：MAP-TV 是预计算 5x anchor；TGV split/FRC 列目前复用 EP16 同子集/同 shifts 的 drizzle proxy；F5 是 task-level visual gate，不是保真或分辨率证据。

**预期效果**:
- 论文 T1/T2/F5 使用同一真实数据 harness 与同一 artifact scale，避免历史 TB-scale 数字污染最终横评表。
- hybrid/V10 推理路径有 23°C 温度均值自检，防止 residual base 漏加 bug 复发。
- 风险：TGV split/FRC 尚非独立 TGV split 重算，必须在表注和 manifest 中保留 proxy caveat；MAP-TV 5x 与 2x methods 不可隐式混同。

**推荐参数**:

```bash
cd algos/ep11_dl_benchmark
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_unified_harness_t1_t2.py \
  --device cuda:0 \
  --workers 4 \
  --output-dir ../../output/ep11_unified_harness
```

**训练结果**: _(本条为评估/数据管线变更，无新增训练；2026-06-14 回填)_
- 完整运行: 13/13 methods success，elapsed 166.6 s；GPU0 峰值观测约 385 MB，未占用第二张 GPU。
- 输出目录: `output/ep11_unified_harness/`；F5 资产: `output/paper_figures/fig05_main_visual.png`、`output/paper_figures/fig05_main_visual.pdf`。
- 23°C sanity check: V9A 10K mean 22.366°C，V9C 5K mean 22.985°C，V10 lam120@15K mean 23.288°C，V9A 60K mean 23.307°C；均非 0°C delta 场。
- T1 selected rows（harness scale，artifact↓ / corr↑）: drizzle 1.138 / 0.771；TGV 0.695 / 0.741；v9b@11K 1.766 / 0.777；V9A@10K 1.762 / 0.719；V9C@5K 1.669 / 0.718；V10 lam120@15K 2.726 / 0.711。
- TB-vs-harness scale check: v9b@11K TB artifact 0.3385 vs harness 1.7662；v8.1a@15K 0.3919 vs 1.9429；v6@8K 0.3302 vs 1.7891。该表确认两套 artifact scale 不能混用。
- 结论: 统一 harness 只提供 gate/select 与 task-level visual evidence；不支持“学习方法打败 TGV”“更干净”或“更保真”表述。

**涉及文件**: `algos/ep11_dl_benchmark/scripts/run_unified_harness_t1_t2.py`, `docs/paper/00_status_and_plan.md`, `docs/paper/01_outline.md`, `docs/paper/02_introduction.md`, `docs/paper/05_method.md`, `docs/paper/06b_experimental_setup.md`, `docs/paper/07_experiments.md`, `docs/paper/08_limitations_conclusion.md`, `docs/paper/09_figures_tables_assets.md`, `docs/paper/10_writing_handover.md`, `docs/paper/reframe_c4_claim3.md`, `docs/paper/supp/C_method_details.md`, `docs/paper/supp/D_full_results.md`, `docs/paper/supp/E_reproducibility.md`

---

### [ACL-020] 2026-06-12 — V10: residual-over-observation 参数化 + 残差幅度惩罚

**问题诊断**:
- V9A 证明 hybrid 2x drizzle 输入能让中心细 zigzag 的真实相位信息进入网络，但 30K 左右出现保真悬崖：`hp_corr_input` 从约 0.974 跌到约 0.906，并在后期平台上被合成结构先验主导。
- V9A 整条训练时间轴的 fidelity-sharpness 前沿被 EP10 TGV 工作点支配：TGV 约为 `hp_corr_input=0.960, sharp_p95=0.96`；UNet 20K 保真高但偏软，30K+ 锐度高但主要来自观测去相关的过冲/幻觉。
- V9B/V9D 表明 loss 侧 1x forward consistency 难以约束漂移方向；V10 需要把“保持观测”变成输出参数化的零成本默认，而不是训练后期靠隐式 checkpoint 早停。

**修改内容**:
1. `algos/ep07_unet_sr/src/unet_sr/config.py`: 新增 `residual_mode: str = "none"` 与 `residual_penalty_weight: float = 0.0`，CLI 对应 `--residual-mode {none,drizzle2x}` 和 `--residual-penalty-weight`。
2. `config.py`: 校验 `residual_mode="drizzle2x"` 仅允许与 `input_mode="hybrid_drizzle2x"`、`scale=2`、旧 `--residual` 关闭、`forward_model_weight=0` 同用，保证 V10 是单因子 residual-over-observation 实验。
3. `algos/ep07_unet_sr/src/unet_sr/train.py`: V10 路径中模型输出解释为 `delta`，训练预测为 `pred = obs[:, 5:6] + delta`，其中 ch5 是 hybrid drizzle mean @2x；loss 中新增 `residual_penalty_weight * mean(abs(delta))`，TensorBoard 记录 `loss/residual_penalty`、`residual/delta_mean`、`residual/delta_std`。
4. `algos/ep07_unet_sr/src/unet_sr/inference.py`: `infer_full_frame` 新增 `residual_channel` 参数，tile 推理时先把模型输出 delta 加回同一 input channel 再 overlap blend；`infer_from_burst` 在 hybrid 路径透传该参数。
5. `algos/ep07_unet_sr/src/unet_sr/real_eval.py`: 从 checkpoint 的 `training_config.residual_mode` 自动选择 residual channel，确保真实数据 eval 与训练使用同一 ch5 加法路径。
6. `algos/ep07_unet_sr/tests/`: 增加 config 合法/非法组合、zero-model residual-channel 推理、残差 L1 penalty 单调性、旧 direct-predict 推理路径回归测试。
7. `algos/ep07_unet_sr/scripts/run_v10.md`: 记录 smoke 与全量 lambda sweep 命令，供用户手动启动 GPU 批。

**预期效果**:
- `delta=0` 时输出严格等于观测域最保真的 drizzle mean ch5，让“保留观测”成为模型默认解。
- L1 残差惩罚把 fidelity-sharpness 权衡从训练步数显式转移到 `lambda`，让 V10 可以扫描受控 Pareto 前沿。
- 风险：过大的 `residual_penalty_weight` 会把模型锁死在软的 drizzle 输入；过小则可能退化回 V9A 的后期先验侵蚀。

**推荐参数**:

```bash
cd algos/ep07_unet_sr
CUDA_VISIBLE_DEVICES=<GPU_ID> uv run python -m unet_sr.train \
  --training-pool-dir ../../data/synthetic/training_pool_2x_aa_burst \
  --output-dir outputs/ep07_v10_residual_lam${LAMBDA_TAG} \
  --input-mode hybrid_drizzle2x \
  --residual-mode drizzle2x \
  --residual-penalty-weight <LAMBDA> \
  --scale 2 \
  --batch-size 128 \
  --num-workers 8 \
  --total-steps 25000 \
  --save-every 2500 \
  --log-every 100 \
  --compile \
  --mse-loss-weight 0.3 \
  --highpass-loss-weight 0.8 \
  --structure-boost 2.0 \
  --grad-vector-weight 0.15 \
  --thin-boost 3.0 \
  --gap-boost 2.0
```

**训练结果**: _(2026-06-13 回填；含一处评估 bug 的发现与修正)_

- 输出目录: `outputs/ep07_v10_resid_lam002/`、`outputs/ep07_v10_resid_lam005/`、`outputs/ep07_v10_resid_lam015/`（各 25K，5K/.../25K checkpoint）。
- 代码验证: `cd algos/ep07_unet_sr && CUDA_VISIBLE_DEVICES= uv run pytest -q tests/test_config.py tests/test_inference.py tests/test_model_losses.py tests/test_real_eval.py` → 36 passed, 3 skipped（CUDA AMP 测试因 GPU 不可见被跳过）。

- **⚠️ 实际运行偏离推荐配置（两处混杂）**：
  1. **batch_size=64**（推荐/权威 `run_v10.md` 为 **128**）——重新引入了本实验本应顺带排除的 bs 混杂（见 `docs/next_move_plan.md` §8 caveat）。
  2. **λ 取 0.02/0.05/0.15**（而非 `run_v10.md` 标定的 0.0203/0.0406/0.0609）。但 `run_v10.md` 的标定公式本身有误：它把「未乘 λ 的 `mean(|delta|)` / total ≈ 24.6%」当成 λ=0.05 时的损失占比，实际占比是 `λ·penalty/total`，λ=0.05 时仅 ~1.2%、λ=0.15 时仅 ~3.7%。**残差惩罚在 0.02–0.15 区间对总损失贡献可忽略**，等于没有真正约束 delta 幅值——这解释了为何三档 λ 结果几乎重合。

- **🔴 评估 bug（已修复）**：fine-window Pareto 评估脚本 `scripts/v9_review/common.py::infer_checkpoint_cached` 调 `infer_from_burst` 时**未传 `residual_channel`**（只传了旧 `residual` 旗标，V10 下为 False），导致 V10 缓存里存的是**裸 delta**（mean ≈ -0.008°C 残差场），而非 `drizzle_mean(ch5) + delta` 的真实输出。`real_eval.py` 的漂移曲线路径无此问题（line 219-220 已读 `residual_mode` 并传 `residual_channel`）。
  - 错误判定（基于裸 delta）：hp_corr_input≈0.46、lattice≈0.08 → 曾被记为「Claim 4 灾难性失败、超标 16×」。**此判定作废。**
  - 修复：`common.py` 现从 `cfg["residual_mode"]` 派生 `residual_channel=5` 并透传；单 checkpoint `--force` 重推理验证通过（`v10_lam005_25k` 修复后 0.8804，与后处理 `delta+base` 的 0.880 逐行一致）。
  - 修正版产物：`output/ep07_v9_review/v10_pareto/{v9a_pareto_metrics.csv, v9a_pareto_scatter.png, v9a_checkpoint_strip.png}`（15 checkpoint 全部重算）。

- **关键指标（修正后；中心细线窗口，hp_corr_input=保真↑，sharp_p95=锐度↑，lattice↓）**：

  | 对象 | hp_corr_input | sharp_p95 | lattice | hp_corr_tgv |
  |---|---|---|---|---|
  | drizzle 输入（观测域上限） | 1.000 | 0.503 | 0.0015 | 0.960 |
  | **EP10 TGV（经典参照）** | **0.960** | **0.959** | 0.0169 | 1.000 |
  | V10 λ=0.02 @5K（最保真点） | 0.908 | 1.190 | 0.0170 | 0.915 |
  | V10 λ=0.05 @25K | 0.880 | 1.367 | 0.0236 | 0.892 |
  | V10 三档 @25K 区间 | 0.880–0.884 | 1.30–1.37 | 0.0218–0.0243 | 0.892–0.895 |

- **视觉/读数**：修正后 V10 落在 **V9A 后期同一区域**（V9A 40–60K：hp_in≈0.906、sharp≈1.2、lattice≈0.015）——比 TGV 锐（sharp 1.2–1.37 > 0.96），但保真不及 TGV（hp_in 0.88–0.91 < 0.96），lattice 与 TGV 同量级。**V10 不支配 TGV**，是「用保真换锐度」的 V9A 式折中点，不是干净的伪影/幻觉灾难。

- **结论**：
  1. 残差参数化 + 当前 λ 区间（0.02–0.15）**未能把输出绑在 drizzle base 附近**（惩罚损失占比仅 ~1–4%），模型仍漂到 V9A 的 fidelity≈0.88 折中区 → **Claim 4 不是正结果**（未支配 TGV）。
  2. 但本轮**也不能作为 Claim 4 的干净反证**：λ 区间太弱、未探到高保真端，且 bs=64 混杂未消除。要把 Claim 4 做成「即使显式残差控制也无法越过经典前沿」的铁案，需补**高-λ（bs=128、25K）扫描**把 fidelity↑/sharp↓ 的折中曲线探完整（见 `algos/ep07_unet_sr/scripts/run_v10_highlam.md`）。**✅ 已于 2026-06-14 完成，结论见下「高-λ sweep 结果」。**
  3. 当前「学习能微微越过 TGV」的最干净证据仍是**零训练 fusion baseline**（`TGV + 0.1·V9A60 delta` 支配 TGV），而非 V10。

---

#### 🆕 高-λ sweep 结果（bs=128 / patch=192 / 25K × 4 个变体；2026-06-14 回填）

> 闭环上文 conclusion 第 2 点：高-λ（λ∈{0.2,0.5,1.2,3.0}）扫描已完成，bs=64 混杂已消除（统一 bs=128），评估口径自检通过。**结论从「凌乱负结果」升级为「干净的可控权衡 + 有价值工作点」**，但仍**不写成「打败 TGV」**（保真全程 < TGV）。

- **运行**: 4 个变体 λ∈{0.2,0.5,1.2,3.0}，bs=128 / patch=192（3090 OOM 从 256 降，全变体统一记 caveat）/ 25K / `--save-every 2500`；每个变体 ~4.3–5.3 h，双 GPU 两两并行，wall ≈ 12 h。输出 `outputs/ep07_v10_resid_hl_lam{020,050,120,300}/`（各 10 个 checkpoint + `model_final.pt`）。
- **残差自检通过**: `output/ep07_v9_review/cache/v10hl_*_temperature.npy` 20 个文件均值全部 ≈ **23.29°C**（不是 ≈0）→ 本条主记录里的「漏加 base」评估 bug **未复现**，本轮 fine-window 数字可信。
- **fine-window Pareto（修复后 harness；hp_corr_input=保真↑，sharp_p95=锐度↑但不可单用，lattice=grain/HF↓）**:

  | 对象 | hp_corr_input | sharp_p95 | lattice |
  |---|---|---|---|
  | drizzle（观测软上限） | 1.000 | 0.503 | 0.0015 |
  | **EP10 TGV（经典基准）** | **0.960** | **0.959** | **0.0169** |
  | λ=0.2 @5K→25K | 0.915→0.882 | 1.047→1.264 | 0.0170→0.0242 |
  | λ=0.5 @5K→25K | 0.918→0.886 | 1.039→1.224 | 0.0155→0.0228 |
  | λ=1.2 @5K | 0.941 | 0.891 | 0.0098 |
  | **⭐ λ=1.2 @15K（最佳折中）** | **0.922** | **0.987** | **0.0141** |
  | λ=1.2 @25K | 0.904 | 1.090 | 0.0180 |
  | λ=3.0 @5K（塌回 drizzle） | 1.000 | 0.510 | 0.0015 |
  | λ=3.0 @10K | 0.956 | 0.801 | 0.0080 |
  | λ=3.0 @25K | 0.934 | 0.931 | 0.0115 |

- **达标 checkpoint**: 7 个满足 `hp_corr_input≥0.92 ∧ lattice≤0.0169`（λ=1.2 的 5K/10K/15K + λ=3.0 的 10K/15K/20K/25K；已排除 λ=3.0@5K 这个 `sharp_p95=0.51` 塌回 drizzle 的退化点）。
- **最佳折中点 = λ=1.2 @15K**: (hp_corr_input, sharp_p95, lattice) = **(0.922, 0.987, 0.0141)** —— 锐度 ≈ TGV(+3%)、grain 比 TGV 低 17%、保真 0.922 刚过门控（仍 < TGV 0.960）。
- **TB-scale 漂移端点（eval_real，artifact↓/corr↑；与 fine-window 是不同口径，绝不混表）**: λ 越大漂移越小、corr 越稳——λ=0.2: 0.70/0.67→0.73/0.658；λ=1.2: 0.34/0.726→0.64/0.695；λ=3.0: **0.29/0.721→0.59/0.722（corr 基本不掉）**。
- **结论（更新）**:
  1. 残差约束把 *fidelity–sharpness–grain* 三维折中变成**可调的 λ 旋钮**：大 λ（3.0）能同时拿高保真+低 grain（牺牲锐度趋向 drizzle）；λ=1.2@15K 拿到「锐而不 grain、保真刚过门控」的折中 → **存在「有价值工作点」**（`docs/paper/reframe_c4_claim3.md` §7 判据满足）。
  2. **但所有 checkpoint 保真仍 < TGV(0.960)**（最佳点 0.922）；维持 reframe 诚实裁决「no GT-certifiable winner」，**不写成「打败 TGV」**；报锐度必并报 lattice + 视觉。
  3. Phase 2 精化（λ=1.2 二分 / 第二 seed）**已决定跳过**（非支配关系，预算转给统一口径 harness T1/T2 重跑）。
- **产物**: `output/ep07_v9_review/v10_highlam/{v9a_pareto_metrics.csv, v9a_pareto_scatter.png, v9a_checkpoint_strip.png}`、`output/ep07_v9_review/ep07_eval_real_metrics.csv`（含 V10 四个变体 + V9C/V9D 漂移）。

**涉及文件**: `algos/ep07_unet_sr/src/unet_sr/config.py`, `algos/ep07_unet_sr/src/unet_sr/dataset.py`, `algos/ep07_unet_sr/src/unet_sr/train.py`, `algos/ep07_unet_sr/src/unet_sr/inference.py`, `algos/ep07_unet_sr/src/unet_sr/real_eval.py`, `algos/ep07_unet_sr/scripts/v9_review/common.py`（评估 bug 修复）, `algos/ep07_unet_sr/tests/test_config.py`, `algos/ep07_unet_sr/tests/test_inference.py`, `algos/ep07_unet_sr/tests/test_model_losses.py`, `algos/ep07_unet_sr/scripts/run_v10.md`

---

### [ACL-019] 2026-06-11 — V9C: hybrid 输入下用合法 1x lr_obs 启用 forward consistency

**问题诊断**:
- ACL-016 为 V9A 引入 `hybrid_drizzle2x` 后，模型输入的第 0 通道变成了 `aligned_mean` 的 2x 上采结果，不再是 forward consistency 所需的合法 1x LR 观测。因此 ACL-016 在 config 中临时禁止 `hybrid_drizzle2x + forward_model_weight>0`。
- V9B 证明 highpass-band forward consistency 在旧 1x 输入上无法压平真实数据漂移，但仍不能回答「hybrid 输入保留 2x 相位信息后，合法 1x 观测锚是否变得有效」。V9C 需要在 hybrid 输入下给 loss 单独提供原始 1x `aligned_mean` patch。
- 风险点参考 ACL-014：crop origin 若不按完整倍率对齐，会让 1x/2x/target patch 出现半个 LR 像素错位；flip/rot90 若未同步到辅助观测，会让 forward 项使用错位 anchor。

**修改内容**:
1. `algos/ep07_unet_sr/src/unet_sr/dataset.py`: hybrid 路径缓存原始 1x `obs_features`，在 5ch 1x obs 上采为 2x 输入前保留 `obs[0:1]`；`__getitem__` 对 hybrid sample 额外返回 `lr_obs`，shape `(1, patch_size_hr//2, patch_size_hr//2)`。
2. `dataset.py`: hybrid crop origin 在 2x 输入网格上强制偶数对齐，`lr_obs` 用 `(y//2, x//2)` 和半边长裁剪；`_augment` 扩展为同步变换 `lr_obs`，确保 flip/rot90 后观测锚与 target/pred 几何一致。
3. `algos/ep07_unet_sr/src/unet_sr/losses.py`: `ContourSRLoss.forward` 新增可选 `lr_obs` 参数；当 `lr_obs is not None` 时，forward-model 项以 `lr_obs` 为观测参照，并显式使用 downsample scale=2，不复用 hybrid 模型的 scale=1。旧 `lr_observation` 路径保持兼容。
4. `algos/ep07_unet_sr/src/unet_sr/train.py`: batch 含 `lr_obs` 且 `forward_model_weight>0` 时传入 loss；旧 `input_mode="lr"` 仍回退到 `obs[:, 0:1]`。
5. `algos/ep07_unet_sr/src/unet_sr/config.py`: 解除 `hybrid_drizzle2x + forward_model_weight>0` 的禁令；改为要求该组合显式使用 `--scale 2`，由 dataset/train 走 `lr_obs` 路径。
6. 测试覆盖 `lr_obs` shape/crop 对应关系、增广同步、偶数 origin、hybrid+AMP loss 有限、config 新校验，以及旧 LR 模式不返回 `lr_obs` 的回归。
7. `algos/ep07_unet_sr/scripts/run_v9.md`: 补 V9C/V9D smoke/full 命令，并说明 V9C 的合法 1x anchor 与 hybrid 输入第 0 通道不同。

**预期效果**:
- V9C 可以在保留 hybrid 2x drizzle 输入相位信息的同时启用 highpass forward consistency，避免把上采 mean 当作物理观测导致非法锚定。
- 如果 V9C 能压平后期 artifact/corr 漂移，说明 forward 锚在 hybrid 输入下开始可见漂移方向；若仍失败，则支持「loss-side anchor 仍不足」的结论。
- 风险：forward 项仍只约束 1x 可见频段，高频 hallucination 仍可能落在 forward operator 零空间；若出现 ACL-005 式震荡，先降 `forward_model_weight` 到 0.05。

**推荐参数**:

```bash
cd algos/ep07_unet_sr
CUDA_VISIBLE_DEVICES=1 uv run python -m unet_sr.train \
  --training-pool-dir ../../data/synthetic/training_pool_2x_aa_burst \
  --output-dir outputs/ep07_v9c_hybrid_legal_fwd \
  --input-mode hybrid_drizzle2x \
  --scale 2 \
  --batch-size 128 \
  --num-workers 8 \
  --total-steps 60000 \
  --save-every 5000 \
  --log-every 100 \
  --compile \
  --mse-loss-weight 0.3 \
  --highpass-loss-weight 0.8 \
  --structure-boost 2.0 \
  --grad-vector-weight 0.15 \
  --thin-boost 3.0 \
  --gap-boost 2.0 \
  --forward-model-weight 0.1 \
  --forward-model-band highpass
```

**训练结果**: _(2026-06-14 回填；60K 完成)_
- 输出目录: `outputs/ep07_v9c_hybrid_legal_fwd`（60K + `model_final.pt`）
- 代码验证: `cd algos/ep07_unet_sr && uv run pytest -q` → 47 passed。
- 关键指标（TB-scale `eval_real`，artifact↓ / raw_control_corr↑）: 0.516/0.714 @10K → **0.695/0.669 @60K**，与 1x 锚定变体端点（v9b 0.655/0.688、v9d 0.677/0.677、v8.1a 0.643/0.689）落到同一 ≈0.65–0.70 / ≈0.67–0.69 平台。
- 结论: **合法 1x 观测锚在 hybrid 输入下同样无法压平后期漂移**。这驳倒了「之前锚定失败只是因为 hybrid 第 0 通道不是合法 1x 观测」的反对意见——即使给 loss 单独喂合法 1x `aligned_mean` patch，漂移曲线仍与无锚/带限/全频锚定变体几乎重合。**与 V9B/V9D 合并：loss 侧 forward 锚定路线（band / full / legal × hybrid 全变体）正式、彻底关闭**；漂移是先验驱动、零空间驻留，只能从输入端（V9A hybrid 输入）或输出参数化（V10 residual）侧解决。落 `docs/paper/07_experiments.md` §6.2/§6.3 input×anchor 矩阵。

**涉及文件**: `algos/ep07_unet_sr/src/unet_sr/dataset.py`, `algos/ep07_unet_sr/src/unet_sr/losses.py`, `algos/ep07_unet_sr/src/unet_sr/train.py`, `algos/ep07_unet_sr/src/unet_sr/config.py`, `algos/ep07_unet_sr/tests/test_dataset.py`, `algos/ep07_unet_sr/tests/test_model_losses.py`, `algos/ep07_unet_sr/tests/test_config.py`, `algos/ep07_unet_sr/scripts/run_v9.md`

---

### [ACL-018] 2026-06-11 — V9A 数据管线：池侧预计算 drizzle 变体，修复首 batch 卡死 + 主机 OOM

**问题诊断**:
- ACL-016 的 hybrid_drizzle2x 把 drizzle 留在 DataLoader worker 内现场计算（ACL-016 已预警「DataLoader 吞吐可能成为瓶颈」），实测后果远超预期：
  - `drizzle_features`（248 帧全幅 480×640 → 960×1280 scatter）实测 **2.7 s/scene**，且每 scene 每 epoch 重算一次；
  - `_load_cached` 把 lr_burst cast 成 float32（**305 MB/scene**）整条塞进 LRU 缓存，16 scene/worker × 8 workers ≈ 47 GB（机器 60 GB）；
  - 复现实验（bs=64 / 8 workers / prefetch=4）：**首 batch 162.6 s**，loader 进程树 RSS 37.8 GB、swap 8.5 GB 打满，batch 8 时 worker 被系统 OOM killer 杀掉。叠加 `--compile` 首步编译，表现为「第一步都跑不动」。

**修改内容**:
1. `scripts/precompute_drizzle_variants.py`（新增）: 池侧离线预计算。每 scene 生成 K=4 个 drizzle 变体存 `drizzle_variants_2x.npy`（(K,3,960,1280) float16，~30 MB/scene）。variant 0 = 全帧无噪声（canonical，与推理口径一致）；variant 1–3 = 随机抽 60–100% 帧 + shifts 加 σ=0.05 px 噪声，与原 `_select_burst` 增广分布一致。ProcessPool 并行、原子写、可断点续跑。
2. `tcforge/storage.py`: `load_scene_compact` 发现 `drizzle_variants_{scale}x.npy` 时以 `mmap_mode="r"` 挂载到 `"drizzle_variants"` key。
3. `dataset.py`: hybrid 模式优先用预计算变体——每 (seed, epoch, scene) 确定性抽 1 个变体（mmap 切片，不全量物化）；增广从「连续随机」降为「K 选 1 离散随机」（叠加随机裁剪/翻转/旋转仍在）。无变体文件时 fallback 到现场 drizzle，但 lr_burst 不再 cast float32，直接持有 float16 mmap，`_select_burst` 只物化抽中的子集。`obs_up`（5ch 1x↑2x）缓存复用，epoch 重建只做变体切片 + concat。
4. 测试: 变体优先于 burst、无 lr_burst 也可跑、变体选择可复现、fallback 缓存保持 float16 mmap；原 burst 路径 5 个测试不变全过。

**预期效果**（已实测验证）:
- 冷加载 2.8 s → **0.52 s**/scene；epoch 重建 2.7 s → **0.006 s**；缓存 RAM ~370 MB → ~100 MB/scene（burst 不再驻留内存）。
- variant 0 与全帧 drizzle 最大误差 0.0118（float16 量化步长内），变体间 mean abs diff ~0.01（增广有效）。
- 风险：增广多样性从连续降为 K=4 离散；若过拟合迹象明显可加 `--num-variants` 重新预计算。

**推荐参数**: 训练前先跑 `uv run python scripts/precompute_drizzle_variants.py --pool-dir data/synthetic/training_pool_2x_aa_burst --num-variants 4 --workers 14`（~25 min，磁盘 +59 GB）；训练 CLI 与 ACL-016 一致无变化。

**训练结果**: _(2026-06-12 回填)_
- 输出目录: `outputs/ep07_v9a_hybrid_drizzle`
- 管线表现: 预计算变体方案有效——V9A 全程无首 batch 卡死、无 OOM kill，~49 min/5K steps 稳定推进至 60K（中途一次人为中断，35K checkpoint 续跑）。V9C 复用同一管线同样稳定。
- 算法结果见 ACL-016 回填；本条仅覆盖数据管线。

**涉及文件**: `scripts/precompute_drizzle_variants.py`, `tcforge/src/tcforge/storage.py`, `algos/ep07_unet_sr/src/unet_sr/dataset.py`, `algos/ep07_unet_sr/tests/test_dataset.py`, `algos/ep07_unet_sr/scripts/run_v9.md`

---

### [ACL-017] 2026-06-11 — V9B: highpass-band forward consistency loss

**问题诊断**:
- v8.1a conservative loss 把 `forward_model_weight` 降到 0 后，loss 中没有任何一项把输出锚定到观测；synthetic loss 收敛后，网络在结构 loss 先验驱动下持续把边画亮、对比拉大（real_eval artifact 0.39→0.64，corr 0.76→0.69 单调漂移）。
- 全频 forward model 的梯度经过 PSF 低通对所有频段施加「拉回模糊解」的力，与 highpass/grad_vector 在同一像素上方向冲突（v6 ACL-005 3K 步震荡平台的根因）。

**修改内容**:
1. `losses.py`: `forward_model_loss` 新增 `band: str = "full"` 和 `band_sigma_lr_px: float = 5.0` 参数。`band="highpass"` 时，对 downsampled pred 和 obs 各减去 σ=band_sigma 的高斯模糊后再算 MSE，放过低频段。高斯模糊复用 ACL-011 `custom_fwd` 包装。
2. `ContourSRLoss`: 透传 `forward_model_band` / `forward_model_band_sigma`。
3. `config.py`: 新增 `--forward-model-band {full,highpass}`（默认 full）和 `--forward-model-band-sigma`（默认 5.0）CLI 参数。
4. `train.py`: 传递新参数到 ContourSRLoss。
5. 测试: band=full 回归（与旧实现数值一致）、highpass DC 不变性、HF 扰动检测、AMP 有限性。

**预期效果**:
- 带限版只在边缘亮度/宽度所在的高频段施加观测一致性，低频由 synthetic MSE 锚定，两者不打架。
- 在合成域中该项不增加信息量，但改变函数族——惩罚重投影偏离输入观测的解，使迁移到真实数据后抑制 40K+ 漂移。
- 风险：若出现震荡平台或整体变钝，先降 `forward_model_weight` 到 0.05 重跑。

**推荐参数**: `--forward-model-weight 0.1 --forward-model-psf-sigma 0.5 --forward-model-band highpass --forward-model-band-sigma 5.0`（其余与 v8.1a 一致）

**训练结果**: _(2026-06-11 回填)_
- 输出目录: `outputs/ep07_v9b_fwd_consistency`
- 视觉效果: 与 v8.1a 同体感：膨胀/亮边相比 v8 时代明显收敛，膨胀-抑制达到可接受平衡，无 v8.1b 式条纹伪影；斜边残留均匀台阶状锯齿；中心最细 zigzag 线仍模糊（与 v8.1 A/B 不变性一致，输入信息瓶颈所致，非 loss 问题）。
- 关键指标（real_eval 248 帧 zoom3x，与 EP11 同口径）:
  - `artifact_score`: 10K 0.369 → 25K 0.605 → 40K 0.640 → 60K 0.655（对照 v8.1a: 0.390 → 0.551 → 0.627 → 0.643）
  - `raw_control_corr`: 10K 0.758 → 25K 0.711 → 40K 0.697 → 60K 0.688（对照 v8.1a: 0.756 → 0.717 → 0.698 → 0.689）
  - 40K→60K 漂移: artifact +0.0145 / corr −0.0082，与 v8.1a（+0.016 / −0.009）基本重合 → **漂移未压平，run_v9.md 验收标准未达成**
  - `loss/forward_model` 自 10K 起躺平在 0.004–0.009 地板，同期 artifact 持续上爬 → 漂移方向位于 forward 算子（shift→PSF→下采样→带限 highpass）的零空间，观测一致性对该方向不可见
- 结论: 带限 forward consistency（weight 0.1, band=highpass）单因子归因失败，对真实数据漂移无可测影响。结合 v8.1a / v8.1b / v9b 三个变体漂移曲线几乎重合，确认漂移是「合成先验在真实分布上无监督外推」的结构性矛盾，loss 侧旋钮已证伪。后续处置：artifact/corr 降级为 checkpoint 选择器（Pareto + 视觉门控，不默认取 60K）；观测锚定若要有效需让锚可见漂移方向（如 hybrid 输入下以合法 LR 观测构造 forward 项），或从输入端解决（V9A, ACL-016）。
- **V9D 补充** _(2026-06-12 回填，`outputs/ep07_v9d_fwd_fullband`)_: full-band anchor（`--forward-model-band full`，其余同 V9B）60K 跑完，artifact 0.677 / corr 0.677，漂移同样未压平且终点比 V9B（0.655/0.688）更差；1K–28K 阶段 artifact/corr 大幅震荡（如 20K artifact 0.575/corr 0.642 后又回弹），与 ACL-005 全频低通梯度冲突一致。V9B+V9D 合并证据：**loss 侧 forward 锚定路线（无论 band）正式关闭**。

**涉及文件**: `losses.py`, `config.py`, `train.py`, `mask_weights.py`, `scripts/run_v9.md`, `scripts/run_training.md`, `tests/test_model_losses.py`, `tests/test_config.py`

---

### [ACL-016] 2026-06-11 — V9A: hybrid 2x drizzle 输入模式

**问题诊断**:
- v8.1a/b 两个变体中心最细线模糊完全相同，与 loss 温度和 HR head 无关。根因：5 个输入通道全部是 1x 网格统计量（aligned_mean/median/coverage/variance/highpass），248 帧的亚像素相位信息在进网络前已坍缩。
- EP15 M4 证明 2x 网格经典方法能恢复 12 µm 频带信息（bare/MAP-TV split-half FRC 0.575→0.947）——信息在数据里，只是没喂给网络。

**修改内容**:
1. `configs/synthetic/training_pool_2x_burst.json`: 新增训练池配置，`save_lr_burst=true`、`compute_classical_sr=false`。场景几何/物理参数与旧池一致。
2. `dataset.py`: 新增 `input_mode="hybrid_drizzle2x"` 模式。加载 `lr_burst + shifts`；每 (scene, epoch) 随机保留 60-100% 帧（`min_burst_frames=30`）并加 `shift_noise_std_px=0.05` 高斯噪声；调 `drizzle_features(scale=2, kernel="bilinear")` 得 3 通道 @ 960×1280；5ch 1x obs 上采到 2x；拼成 8ch @ 2x。Effective scale=1，同坐标裁剪 patch。
3. `config.py`: 新增 `--input-mode {lr,hybrid_drizzle2x}` CLI 参数。hybrid 自动设 `in_channels=8`。校验禁止 hybrid + `forward_model_weight>0`（obs[:, 0:1] 不是合法 1x LR 观测）。
4. `model.py`: 无改动（`scale=1` + `in_channels=8` 由构造参数覆盖）。
5. `train.py`: 根据 `input_mode` 推导 `model_scale`；传递 `input_mode` 到 dataset。
6. `inference.py`: `infer_from_burst` 新增 hybrid 路径：1x fused↑2x + scatter drizzle@2x → 8ch → tile 推理（scale=1）。
7. `real_eval.py`: 从 `training_config.input_mode` 自动走对应推理路径。
8. 测试: hybrid 样本形状 (8,256,256)、增强同步、burst 子集可复现/下限生效、epoch 切换产生不同 burst、旧路径 `input_mode="lr"` 回归。

**预期效果**:
- 中心最细线：亚像素信息直接进网络，不再依赖合成先验猜测。
- 边缘锯齿：2x drizzle 通道编码边缘亚像素位置，网络不必把边吸附到 1x 网格。
- 风险：drizzle lattice 格纹可能被网络当结构学习；DataLoader 吞吐可能成为瓶颈。

**推荐参数**: `--input-mode hybrid_drizzle2x --training-pool-dir data/synthetic/training_pool_2x_aa_burst`（其余与 v8.1a 一致，`forward_model_weight=0`）

**训练结果**: _(2026-06-12 回填，60K 完成)_
- 输出目录: `outputs/ep07_v9a_hybrid_drizzle`（35K 处中断后以 batch_size=64 续跑至 60K）
- 视觉效果: 中心最细 zigzag「梯子」区呈现明显的训练阶段 Pareto 权衡——10K/20K 时梯子内部条纹部分可分辨（与输入 drizzle 通道、EP10 TGV 一致）；60K 时粗 zigzag 线最锐利、对比最大，但中心梯子重新糊成橙色团块，粗细线交融，与 v8.1a 60K 体感相同。
- 关键指标（real_eval 248 帧, contour_refined）:
  - `artifact_score` / `raw_control_corr`: 10K 0.446/0.719 → 20K 0.514/0.702 → 30K 0.660/0.663 → 60K 0.646/0.669。
	  - **漂移在 30K 后压平甚至轻微回头**（30K→60K artifact −0.014 / corr +0.007），是 v8.1a/v9b/v9d 中唯一不单调恶化的变体；但平台位置 corr 0.669 低于 v8.1a 60K 的 0.689，run_v9.md「corr 上升」验收标准在 60K 不达成。
  - 中心细线窗口 highpass corr（vs TGV | vs 输入 drizzle 通道，诊断脚本 `algos/ep07_unet_sr/scripts/v9_review/`（原 `tmp/v9a_review/`，已迁移））: **10K 0.966/0.973 → 60K 0.935/0.925**，v8.1a 60K 为 0.936/0.926 → hybrid 输入在 10K 时几乎完整透传了中心细线信息，60K 时被合成结构先验抹回 v8.1a 水平。
- 结论: **输入瓶颈假设证实，但训练后期合成先验会重新吃掉输入里的真实细节**。中心细线信息确实存在于 drizzle 输入通道中（ACL-015 推断正确），V9A 早期 checkpoint 能保留它；漂移机制现在精确定位为「结构 loss 先验逐步覆盖观测保真，把真实细纹理当模糊清理掉」。处置：① V9A 最终 checkpoint 不取 60K，在 10K–25K 区间做 Pareto + 视觉联合选优；② 下一个单因子实验方向是结构权重后期退火或 residual-to-drizzle 参数化，而不是更长训练。

**涉及文件**: `configs/synthetic/training_pool_2x_burst.json`, `dataset.py`, `config.py`, `train.py`, `inference.py`, `real_eval.py`, `mask_weights.py`, `scripts/run_v9.md`, `scripts/run_training.md`, `tests/test_dataset.py`, `tests/test_config.py`, `tests/test_inference.py`

---

### [ACL-015] 2026-06-10 — EP07 v8.1 A/B 归因实验：loss 降温 vs PixelShuffle head

**问题诊断**:
- v8 AA 训练池消除了部分二值 target 锯齿，但真实数据 `eval_real` 在 30K 之后出现更醒目的亮边、边缘膨胀和细密 2x 相位网格。
- 当前现象不是单点问题：final HR head 使用 bilinear upsample 后接带 GroupNorm 的 3x3 refine，容易放大 2x 相位纹理；同时 `structure_boost=4`、`grad_vector=0.3`、`thin_boost=6`、`gap_boost=4`、`laplacian=0.1`、全频 `forward_model=0.1` 叠加后，边缘区域 loss 权重过热，鼓励网络把边缘画亮画宽。
- 直接把 PixelShuffle、forward consistency、warmup 和多项权重一起改，会导致下一轮仍无法判断主因。

**修改内容**:
1. 设计两条并行训练线：
   - `V8_1A`: 保留现有 bilinear HR head，只把 loss 降温，用于验证膨胀是否主要来自结构权重过热。
   - `V8_1B`: 使用 PixelShuffle + ICNR + 1 个无归一化 HR residual block，并使用与 A 完全相同的 conservative loss，用于隔离 final upsampler/head 的 2x 相位伪影贡献。
2. `model.py`: 为 `ThermalSRUNet` 增加 `hr_upsampler={bilinear,pixelshuffle}` 和 `hr_res_blocks` 参数；默认保持旧 bilinear 行为，PixelShuffle 分支显式启用。
3. `config.py` / `train.py`: 将 HR upsampler 参数纳入 CLI、config.json 和模型构造，保证训练产物可复现。
4. `scripts/run_training.md`: 写入 V8_1A / V8_1B 推荐命令。
5. 测试覆盖默认 bilinear 输出尺寸、PixelShuffle 输出尺寸和无 BatchNorm 约束。

**预期效果**:
- 若 A 已明显压住亮边/膨胀，说明 loss 过热是主因；若 B 相比 A 进一步减少 2x 网格，说明 PixelShuffle head 对 final 相位纹理有效。
- 若 A/B 都变软，说明 conservative loss 降温过度，下一轮再考虑 highpass-only forward model 或结构权重 warmup。
- 风险: PixelShuffle 分支不能加载旧 bilinear checkpoint；必须通过 config 中的 `hr_upsampler` 重建模型。

**推荐参数**:
- `V8_1A`: `--hr-upsampler bilinear --mse-loss-weight 0.3 --highpass-loss-weight 0.8 --structure-boost 2.0 --grad-vector-weight 0.15 --laplacian-weight 0.0 --forward-model-weight 0.0 --thin-boost 3.0 --gap-boost 2.0`
- `V8_1B`: `--hr-upsampler pixelshuffle --hr-res-blocks 1` 加同一套 V8_1A loss 参数。

**训练结果**: _(2026-06-11 回填)_
- 输出目录: `outputs/ep07_v8_1a_loss_cooldown`, `outputs/ep07_v8_1b_pixelshuffle`
- 视觉效果:
  - **V8_1A (bilinear + conservative loss)**: 10K 中心棋盘纹最重（类似 v8 初期），20K 后除中心外棋盘消失、无膨胀；30–40K 棋盘减小、边框由糊变清晰但开始膨胀；60K 对比度最大、边框最膨胀。中心区域只有最细的几条线仍糊，稍粗的线锐利可辨结构；边缘锯齿相对 1B 有改善。
  - **V8_1B (PixelShuffle + 同一 loss)**: 10K 有晕染但比 v8 初版轻；20K 提亮、对比度增强、边框膨胀；30K 边框收回变清晰；至 60K 对比度持续增强。边缘锯齿未改善，且中等边框之间出现条纹状亮色伪影；中心区域同样模糊。
- 关键指标（synthetic loss 两个变体均在 ~40–45K 收敛后平坦；real_eval 248 帧 zoom3x）:
  - `eval_real/artifact_score`（越小越好）随训练**单调上升**: 1A 0.390(10K)→0.643(60K)，1B 0.413→0.709；1B 全程高于 1A。
  - `eval_real/raw_control_corr`（与 raw bicubic 控制图的 highpass 相关，反映观测保真）**单调下降**: 1A 0.756→0.689，1B 0.747→0.667。对照 EP10 TGV 的 0.916 / artifact 0.695，UNet 输出对真实观测的锚定明显偏弱。
- 结论:
  1. **PixelShuffle head 归因失败**: 1B 未减轻锯齿，反而引入新的条纹伪影且 artifact_score 全程更高 → final upsampler 不是 2x 相位伪影/锯齿主因，后续保留 bilinear head，放弃 PixelShuffle 分支。
  2. **Loss 降温部分有效**: 1A 锯齿改善、早期无膨胀，说明结构权重过热确实贡献了亮边/膨胀；但 40K 后 synthetic loss 已平坦而真实数据上对比度/膨胀仍持续漂移（artifact ↑ / corr ↓），说明**缺失观测一致性约束**（`forward_model_weight=0`）使无约束方向在合成先验驱动下继续漂移。
  3. **中心最细线模糊对两个变体完全不变** → 与 loss 温度和 HR head 无关，指向前端输入信息瓶颈：5 个输入通道全部是 1x 网格统计量，248 帧的亚像素相位信息在进网络前已被坍缩；而 EP10/EP15 证明 2x 网格经典方法可恢复 12 µm 频带信息（FRC 0.575→0.947）。下一步主攻方向为 2x-grid drizzle/classical-SR 输入通道 + 温和 highpass-band forward consistency。

**涉及文件**: `algos/ep07_unet_sr/src/unet_sr/model.py`, `algos/ep07_unet_sr/src/unet_sr/config.py`, `algos/ep07_unet_sr/src/unet_sr/train.py`, `algos/ep11_dl_benchmark/scripts/run_unet_vs_drizzle_2x.py`, `algos/ep07_unet_sr/tests/test_model_losses.py`, `algos/ep07_unet_sr/tests/test_config.py`, `algos/ep07_unet_sr/scripts/run_training.md`

---

### [ACL-014] 2026-06-10 — 修复 EP12 4x 增强后 drizzle/coverage 错位并默认启用 burst augmentation

**问题诊断**:
- EP12 Dataset 在执行 flip/rot90 后，只同步变换了 `obs_features` / `hr_target` / `hr_edge`，但 loss 使用的 `drizzle_mean` / `coverage` 仍来自未增强的 `obs_drz_patch`。这会让 coverage 加权 HF/NLL 和 forward consistency 在大部分增强 patch 上使用错位权重与观测约束。
- `_crop_origin()` 只按 `scale / drizzle_scale = 2` 对齐，导致 `y % 4 == 2` 或 `x % 4 == 2` 时 1x 上下文通道相对 4x target 偏移 2 HR px，即 0.5 LR px。
- EP12 重训若继续默认 `burst_augment=False`，会固定在单一 detector-axis coverage 图上训练，不能覆盖 M1 暴露出的 detector 轴相位覆盖坍缩风险。

**修改内容**:
1. `algos/ep12_4x_sr/src/sr4x/dataset.py`: `drizzle_mean` / `coverage` 改为从增强后的 drizzle 特征通道切片得到；普通路径和 `defer_1x_upsample` 路径均使用同一几何变换。
2. `dataset.py`: `_crop_origin()` 改为按完整 `self.scale` 对齐，保证 drizzle crop 与 1x context crop 都落在整数源像素边界。
3. `algos/ep12_4x_sr/src/sr4x/config.py`: `TrainingConfig.burst_augment` 默认改为 `True`，CLI 改为 `--burst-augment/--no-burst-augment`，保留 legacy fixed-drizzle pool 的关闭入口。
4. 新增 dataset/config 回归测试，分别锁定增强后 loss 辅助通道同步、crop 原点 4x 对齐和 legacy 关闭开关。
5. 更新 EP12 训练文档与 benchmark README，明确 4x 只作为学习型正则化/轮廓定位网格，采纳必须通过 M4 MAP-TV 与 EP07 2x x2up gate。

**预期效果**:
- 消除 EP12 训练中的系统性错位梯度，尤其是 forward consistency 用翻转/旋转后的预测去解释未翻转/未旋转观测的问题。
- 1x 上下文、2x drizzle 和 4x target 在 patch crop 上保持整数网格一致。
- 后续 4x 重训默认启用相位/帧子集扰动，降低对单一覆盖图的过拟合风险。
- 风险: 默认 burst augmentation 要求训练池包含 `lr_burst.npy` 和 `shifts.npy`；旧预计算池必须显式传 `--no-burst-augment`。

**推荐参数**:

```bash
cd algos/ep12_4x_sr
CUDA_VISIBLE_DEVICES=0 uv run python -m sr4x.train \
  --training-pool-dir ../../data/synthetic/training_pool_4x_aa_2000 \
  --output-dir outputs/ep12_hybrid_v2_guarded \
  --scale 4 \
  --drizzle-scale 2 \
  --burst-augment
```

**训练结果**:
- 代码验证: `cd algos/ep12_4x_sr && uv run pytest -q` → 13 passed。
- 真实重训结果: _(EP12 Hybrid v2 训练后填写)_

**涉及文件**: `algos/ep12_4x_sr/src/sr4x/dataset.py`, `algos/ep12_4x_sr/src/sr4x/config.py`, `algos/ep12_4x_sr/tests/test_dataset.py`, `algos/ep12_4x_sr/tests/test_config.py`, `algos/ep12_4x_sr/README.md`, `algos/ep12_4x_sr/run_training.md`, `algos/ep12_4x_sr/scripts/run_training.md`, `research_log/episodes/ep12_4x_benchmark/README.md`

---

### [ACL-013] 2026-06-10 — 4x v8 AA 训练池入口统一与 EP12 soft mask 接入

**问题诊断**:
- 4x 训练池文档同时存在旧 full-frame 入口、旧 same-grid drizzle 预计算入口和新 `generate_training_pool.py` compact 入口，容易误导执行流程。
- 当前 EP12 Hybrid 已改为 `drizzle_scale=2` + PixelShuffle 2x，不再需要离线 `obs_features_4x.npz`；但文档仍把 `build_4x_features.py` 写成训练前必需。
- EP12 Dataset 读取 `hr_mask_4x.png` 时使用 `>0` 二值化，会把 v8 AA soft coverage mask 重新变硬，导致 HR target 丢失抗锯齿覆盖率语义。

**修改内容**:
1. 统一 4x pool 生成入口为 `scripts/generate_training_pool.py`，删除旧入口 `generate_thermal_chip_phantom.py`、旧 smoke 检查脚本、旧 drizzle 预计算脚本及对应测试。
2. 更新 `docs/windows_4x_generation.md`：明确 2000 scenes / 16 workers 命令、metadata 检查字段、当前 EP12 Hybrid 从 `lr_burst.npy + shifts.npy` 按需计算 2x drizzle。
3. `algos/ep12_4x_sr/src/sr4x/dataset.py`：`hr_mask_4x.png` 改为按 `uint8/255.0` 读取 soft coverage，再用 `reconstruct_hr_temperature()` 重建 target。
4. 更新 EP12 README、训练命令文档和测试，使训练池契约变为 `obs_features_1x.npz + lr_burst.npy + shifts.npy + soft hr_mask_4x.png + metadata.json`。
5. 将 EP14 loss-atlas 中旧 same-grid 4x drizzle 文案标记为 legacy，避免与当前 Hybrid 训练路径混淆。

**预期效果**:
- 4x AA 数据生成只保留一个可复现入口，减少 Windows/Linux 迁移和本地大规模生成时的操作歧义。
- EP12 Hybrid 训练真正消费 v8 AA soft target，避免把抗锯齿训练池退化回硬边界 target。
- 风险: Dataset 仍支持旧可选预计算文件的兼容读取；后续若要完全移除 legacy 兼容，需要单独评估历史 checkpoint/notebook 复现需求。

**推荐参数**:

```bash
uv run python scripts/generate_training_pool.py \
  --config configs/synthetic/training_pool_4x.json \
  --output-dir data/synthetic/training_pool_4x_aa_2000 \
  --pool-size 2000 \
  --workers 16
```

**训练结果**: _(EP12 Hybrid 重新训练后填写)_

**涉及文件**: `docs/windows_4x_generation.md`, `algos/ep12_4x_sr/src/sr4x/dataset.py`, `algos/ep12_4x_sr/tests/test_dataset.py`, `algos/ep12_4x_sr/tests/test_train_smoke.py`, `algos/ep12_4x_sr/tests/test_model_losses.py`, `algos/ep12_4x_sr/README.md`, `algos/ep12_4x_sr/run_training.md`, `algos/ep12_4x_sr/scripts/run_training.md`, `scripts/generate_thermal_chip_phantom.py`, `scripts/smoke_test_thermal_chip_phantom.py`, `scripts/build_4x_features.py`, `scripts/precompute_drizzle_2x.py`

---

### [ACL-012] 2026-06-10 — EP15 M4 GPU MAP-TV 去卷积基准重跑

**问题诊断**:
- EP06 旧 MAP-TV 结果不可作为 4x baseline：`psf_sigma=1.0` 已超出 M3 支持的可信区间 `0.2-0.5 LR px`，`max_iter=4` 远未收敛，lambda 只取单点，forward model 没有包含探测器孔径 box integration。
- EP12 4x 网络没有显示真实增益，后续网络方法需要一个经典、可复现、必须超越的 “baseline to beat”。

**修改内容**:
1. 新增 `algos/ep15_info_limit/scripts/run_m4_deconv_anchor.py`：用 PyTorch GPU batch 实现 `BatchForwardModel`，一次处理 248 帧 shift / Gaussian PSF / detector box downsample；`adjoint()` 反向执行 upsample / PSF / reverse shift 并累加梯度。
2. 默认 forward model 改为 `HR -> shift -> Gaussian PSF -> avg_pool2d detector box -> LR`，`--no-box` 仅作为消融开关。
3. MAP-TV 主循环使用 FISTA + smoothed TV gradient，full run `max_iter=150`，输出 `iteration,data_rmse,tv_value,objective,relative_update` 收敛曲线。
4. 参数网格改为 `sigma={0.2,0.3,0.4,0.5} LR px`、`lambda={3e-4,1e-3,3e-3}`；每个 sigma 先用 odd/even split-half NRMSE + artifact/std proxy 选 lambda，再用全 248 帧跑 full reconstruction。
5. 新增四方法视觉对比、sigma=5 highpass 对比、zigzag 定量剖面、split-half FRC 复验和全参数选择 CSV。

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

### [ACL-011] 2026-06-10 — SSIM / Gaussian blur 改用 custom_fwd，消除 AMP 气泡

**问题诊断**:
- `ContourSRLoss` 中 `ssim()` 与 `gaussian_blur_2d()` 在 AMP 训练步内使用 `autocast(enabled=False)` + 手动 `.float()`，会在 fp16 主通路中插入 fp32 子图，造成 dtype 切换气泡并降低 GPU 吞吐。
- 原注释指出 fp16 Gaussian 统计会 NaN，因此不能简单删掉 fp32 保护。

**修改内容**:
1. `losses.py`: 抽取 `_ssim_float32` / `_gaussian_blur_2d_float32` 核心实现。
2. CUDA 路径改用 `@torch.amp.custom_fwd(cast_inputs=torch.float32)` 包装 `_ssim_cuda` / `_gaussian_blur_2d_cuda`，在 AMP 上下文内一次性 cast 到 fp32，避免嵌套 `autocast(False)`。
3. CPU 路径保持显式 `.float()` fallback。
4. 新增 `test_contour_sr_loss_finite_under_cuda_amp` 回归测试。

**预期效果**:
- 保持 SSIM / highpass 统计数值稳定（无 fp16 NaN），同时减少 AMP 气泡、略降 loss 段 kernel 切换开销。

**推荐参数**: 无需 CLI 变更。

**训练结果**: _(v8 重训后填写)_

**涉及文件**: `algos/ep07_unet_sr/src/unet_sr/losses.py`, `algos/ep07_unet_sr/tests/test_model_losses.py`

---

### [ACL-010] 2026-06-10 — thin/gap 权重图下沉 DataLoader worker，消除主进程 CPU 瓶颈

**问题诊断**:
- v8 在 `train.py` 主进程对整 batch 调用 `_make_mask_loss_weights`（逐样本 scipy EDT），batch=128 时约 600 ms/step，GPU 利用率降至 ~50%。
- 增加 `num_workers` 无效：瓶颈发生在 DataLoader 返回之后的主线程串行段，与 worker 并行无关。

**修改内容**:
1. 新增 `mask_weights.py`：抽取单 patch 与 batch 版权重图计算逻辑。
2. `dataset.py`：当 `thin_boost>1` 或 `gap_boost>1` 时，在 `__getitem__` 内对每个 patch 预计算 `thin_weight` / `gap_weight`，由 DataLoader worker 并行执行。
3. `train.py`：删除主进程 batch 级权重计算，直接消费 batch 中预计算张量。

**预期效果**:
- mask 权重 CPU 成本分散到 worker，并与 GPU 步通过 prefetch 重叠，恢复 v6 水平 GPU 利用率，同时保留 thin/gap loss 语义。
- 单 patch EDT（256×256）比 batch=128 主进程循环更轻；总 CPU 算力需求不变，但不再阻塞 GPU。

**推荐参数**: 沿用 v8 `--thin-boost 6 --gap-boost 4`；`num_workers` 保持 v6 水平（6–8）即可。

**训练结果**: _(v8 重训后填写)_

**涉及文件**: `algos/ep07_unet_sr/src/unet_sr/mask_weights.py`, `algos/ep07_unet_sr/src/unet_sr/dataset.py`, `algos/ep07_unet_sr/src/unet_sr/train.py`, `algos/ep07_unet_sr/tests/test_dataset.py`, `algos/ep07_unet_sr/tests/test_model_losses.py`

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
