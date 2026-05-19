# EP08 — INR-based 2x Contour SR

> 状态：Stage 2 小规模四/五方对照已完成。SIREN/WIRE、Deep Decoder、DeepInverse-DIP 均完成 32 帧、256x256 LR patch 训练；EP06 MAP-TV proxy 指标已登记。
>
> Stage 4 远程编排 runbook: `research_log/episodes/ep08_inr_sr/stage4_remote_operation_guide.md`

## 目标

EP08 验证三类深度 prior 是否能在主 session 255 帧 LWIR 微扫描数据上带来可复现的 2x contour-level SR 增益，并与 EP06 classic baseline 做同一框架下的对比：

| Track | 方法 | 作用 |
|---|---|---|
| Classic baseline | EP06 MAP-TV | 经典优化参照，不作为光学真值 |
| INR baseline | SIREN | 正弦隐式表示，作为 WIRE 的 activation ablation |
| INR main | WIRE | Gabor 激活，重点检查边缘/轮廓表达 |
| CNN decoder control | Deep Decoder | 非 INR 深度 prior 对照 |

本 Episode 的交付目标仍是工业检测视角下的轮廓可见性和稳定性，不声明 5 um 计量级温度分辨率。

## 串行门控

EP08 必须按串行门控推进，不能在基础验证未过时并行铺开结论：

1. P0 forward/highpass 等价性：PyTorch forward operator 与 EP06 NumPy forward operator 逐参数等价；highpass preprocessing 独立匹配 EP06 `mode="nearest"` 约定。
2. P0 数据与 split：只使用主 session=2 的 255 帧；hold-out split 要保留位移相位覆盖，不能简单随机破坏相位分布。
3. P0 单方法训练门控：SIREN / WIRE / Deep Decoder 各自必须产出 highpass track 和 raw-control track，并通过数值健康检查。
4. P0/P1 四方对比门控：同一指标表中对比 EP06 MAP-TV、SIREN、WIRE、Deep Decoder；只有同时满足 hold-out、split-half、artifact、raw-control、pin 区域目视检查，才可报告候选增益。

## 位移与真值边界

- Stage command 只能作为位移 prior / 初始化 / 正则约束，不能当作 alignment ground truth。
- EP04 localization 是 alignment anchor / quality gate，不是最终 SR 成功证明。
- EP06 MAP-TV 是工程 baseline，不是外部光学真值。
- Highpass 图用于突出边缘和局部结构；raw-control track 必须同时展示，用于避免把纯边缘增强误解为温度计量提升。

## 当前完成状态

当前已完成 EP08 的代码/验证地基：

- 已建立独立 UV 算法项目 `algos/ep08_inr_sr/`，与根环境和其他算法目录隔离。
- 已实现 PyTorch forward/adjoint、highpass/offset correction、phase-stratified split、真实主 session 数据入口和统一指标模块。
- 已实现 SIREN、WIRE、Deep Decoder 三类模型与轻量 CLI；Deep Decoder 使用 `Upsample2x -> Conv1x1 -> BatchNorm -> ReLU` 的逐级 decoder 架构。
- 已建立研究日志入口、报告入口和 notebook fragment 骨架，并完成 notebook 构建执行 smoke。
- 已更新 `algos/ep08_inr_sr/configs/ep06_baseline_metrics.json`：填入 EP06 MAP-TV full-frame proxy 指标，并明确 hold-out residual / split-half NRMSE 在 EP06 协议下不可用。
- Notebook fragment 只读取已有产物并展示门控状态；SIREN / WIRE fragments 已更新为展示训练历史、收敛曲线、HR highpass、raw-control 和 split-half 差异图。
- 本轮验证：`uv run pytest -q` 为 34 passed；`uv run python scripts/validate_p0.py` 已写出 passed forward/highpass/split validation artifacts；真实数据 2 帧 16x16 patch CPU 一步 SIREN smoke 通过；GPU0 SIREN/WIRE synthetic smoke 通过；GPU1 Deep Decoder synthetic smoke 通过。
- Stage 1 训练：SIREN 与 WIRE 均使用同一 seed=42 train/val split（27 train / 5 val），32 帧主 session，中心 256x256 LR patch，HR 512x512，`batch_k=8`，`lr=5e-4`，`warmup_steps=200`，`early_stop_patience=1000`。
- Stage 2 训练：Deep Decoder 与 DeepInverse-DIP 使用相同 32 帧、相同 center crop、相同 seed=42 split 和同一 EP08 `ForwardOperator`。DeepInverse-DIP 保留 `deepinv==0.4.0` 的 `ConvDecoder` backbone，但使用 EP08 自定义训练循环、固定 latent 和 hold-out early stopping；split-half 使用相同初始化的短预算数据稳定性检查。
- Stage 4 基础设施：已新增 `algos/ep08_inr_sr/scripts/stage4_controller.py`，用于远程端按小时 tick 推进 64→128→255 full-frame progressive 训练；controller 只做后台启动、pid/log 追踪和 `metrics.json` 数值健康门控，不在 notebook 中启动训练，也不替代人工视觉检查。

## Stage 1 指标

| Method | Hold-out residual | Split-half NRMSE | Artifact score | Raw-control agreement | P95 gradient | Best step | Final step |
|---|---:|---:|---:|---:|---:|---:|---:|
| SIREN | 3.6729 | 0.2916 | 0.2178 | 0.2085 | 0.9289 | 1630 | 2630 |
| WIRE | 2.9665 | 0.4899 | 2.2434 | 0.0868 | 1.6098 | 2989 | 3989 |

Stage 1 结论：两者均完成训练且未在 500 iter 前早停，说明 INR 框架可训练。WIRE 的 hold-out residual 更低，边缘响应更强，但 split-half NRMSE、artifact score 和 raw-control agreement 均弱于 SIREN，提示 Gabor 激活带来更强高频响应的同时也提高了 artifact / 不稳定风险。Stage 2 已补入 EP06 MAP-TV patch-level same-protocol baseline。

## Stage 2 指标

TCForge benchmark 用 `lr_shape=(256,256)`, `n_frames=32`, `scale=2` 的有 GT 合成场景确认四个方法在 HR highpass 域闭环正确：

| Method | Highpass PSNR (dB) | Global SSIM proxy |
|---|---:|---:|
| SIREN | 24.6634 | 0.7520 |
| WIRE | 24.4126 | 0.7325 |
| Deep Decoder | 19.8896 | 0.2068 |
| DeepInverse-DIP | 18.4729 | 0.4447 |

真实 32 帧 patch 五方对照：

| Method | Hold-out residual | Split-half NRMSE | Artifact score | Raw-control agreement | P95 gradient | Best step | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| SIREN | 3.6729 | 0.2916 | 0.2178 | 0.2085 | 0.9289 | 1630 | complete |
| WIRE | 2.9665 | 0.4899 | 2.2434 | 0.0868 | 1.6098 | 2989 | complete |
| Deep Decoder | 3.6258 | 0.4406 | 0.0355 | 0.3227 | 0.4885 | 6685 | complete |
| DeepInverse-DIP | 2.3051 | 0.4562 | 3.0867 | 0.0608 | 1.2143 | 9900 | complete |
| EP06 MAP-TV | 2.2439 | 0.1743 | 0.3978 | 0.3102 | 0.8667 | 100 | complete |

Deep Decoder vs DeepInverse-DIP 交叉验证结论：
- 四方法在 TCForge HR highpass GT benchmark 上 PSNR 均超过 18 dB，说明实现和 forward wrapper 没有明显 shape / geometry 错位。
- 真实数据上 DeepInverse-DIP hold-out residual 低，但 artifact score 和 raw-control agreement 最差，说明 DIP 仍是强拟合风险参照，不进入默认扩展。
- Deep Decoder 修复后不再是 64×64 插值色块，artifact 最低且 raw-control agreement 较高，但 split-half 和 hold-out 仍不优于 SIREN。
- 综合推荐：Stage 3 主方法选 SIREN；EP06 MAP-TV patch baseline 和 Deep Decoder 分别作为 classic / low-artifact 对照。

EP06 MAP-TV baseline 说明：`output/ep08_inr_sr/ep06_patch_baseline/metrics.json` 已按 EP08 32 帧、256×256、seed=42 split 生成同协议五项指标。旧 full-frame proxy 仍只作为 Stage 3 artifact proxy 的历史参照。

## WIRE 实现修正

`algos/ep08_inr_sr/src/ep08/models/wire.py` 已从单线性投影修正为 carrier 与 envelope 两个独立 `nn.Linear` 投影：

- `linear`: sinusoidal carrier projection
- `linear_scale`: Gaussian envelope projection
- 两者使用相同初始化范围，并在 `test_wire.py` 中检查 shape、backward 和两组参数梯度。

这与 official WIRE real-valued Gabor layer 的双投影结构对齐，避免 carrier/envelope 被迫共享方向。

## 待完成

| Gate | 待办 | 状态 |
|---|---|---|
| P0-forward | PyTorch forward/highpass 等价性单测 | 已完成（34 个项目测试全过） |
| P0-data | 主 session 真实数据加载 + EP05 shifts 接入 | 已完成 smoke |
| P0-SIREN | SIREN 训练与 hold-out 指标 | 已完成 Stage 1 正式训练与五项指标 |
| P0-WIRE | WIRE 训练与 hold-out 指标 | 已完成 Stage 1 正式训练与五项指标；artifact 风险高于 SIREN |
| P1-DeepDecoder | Deep Decoder 训练与对照指标 | 已完成 Stage 2 正式训练与五项指标 |
| P1-DeepInverse | DeepInverse-DIP 成熟库交叉验证 | 已完成自定义训练循环、TCForge benchmark 和真实数据指标 |
| P1-report | 五方对比结论与 Stage 2 门控判定 | 已完成；推荐 SIREN 进入 Stage 3，Deep Decoder 作对照 |
| P2-Stage4-controller | 远程 full-frame progressive 训练编排与健康检查 | 已完成；见 Stage 4 runbook，尚未启动长训练 |
