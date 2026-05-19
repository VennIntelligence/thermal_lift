# EP08 — INR-based 2x Contour SR

> 状态：Stage 1 小规模 INR 训练已完成。SIREN 与修正后的双投影 WIRE 均完成 32 帧、256x256 LR patch、GPU 正式训练，并产出五项指标与中间训练产物。

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
- 已建立 `algos/ep08_inr_sr/configs/ep06_baseline_metrics.json` 占位，用于后续填入 EP06 baseline 指标来源。
- Notebook fragment 只读取已有产物并展示门控状态；SIREN / WIRE fragments 已更新为展示训练历史、收敛曲线、HR highpass、raw-control 和 split-half 差异图。
- 本轮验证：`uv run pytest -q` 为 32 passed；`uv run python scripts/validate_p0.py` 已写出 passed forward/highpass/split validation artifacts；真实数据 2 帧 16x16 patch CPU 一步 SIREN smoke 通过；GPU0 SIREN/WIRE synthetic smoke 通过；GPU1 Deep Decoder synthetic smoke 通过。
- Stage 1 训练：SIREN 与 WIRE 均使用同一 seed=42 train/val split（27 train / 5 val），32 帧主 session，中心 256x256 LR patch，HR 512x512，`batch_k=8`，`lr=5e-4`，`warmup_steps=200`，`early_stop_patience=1000`。

## Stage 1 指标

| Method | Hold-out residual | Split-half NRMSE | Artifact score | Raw-control agreement | P95 gradient | Best step | Final step |
|---|---:|---:|---:|---:|---:|---:|---:|
| SIREN | 3.6729 | 0.2916 | 0.2178 | 0.2085 | 0.9289 | 1630 | 2630 |
| WIRE | 2.9665 | 0.4899 | 2.2434 | 0.0868 | 1.6098 | 2989 | 3989 |

Stage 1 结论：两者均完成训练且未在 500 iter 前早停，说明 INR 框架可训练。WIRE 的 hold-out residual 更低，边缘响应更强，但 split-half NRMSE、artifact score 和 raw-control agreement 均弱于 SIREN，提示 Gabor 激活带来更强高频响应的同时也提高了 artifact / 不稳定风险。本阶段 EP06 baseline metrics 仍为 null，因此不做绝对阈值对比。

## WIRE 实现修正

`algos/ep08_inr_sr/src/ep08/models/wire.py` 已从单线性投影修正为 carrier 与 envelope 两个独立 `nn.Linear` 投影：

- `linear`: sinusoidal carrier projection
- `linear_scale`: Gaussian envelope projection
- 两者使用相同初始化范围，并在 `test_wire.py` 中检查 shape、backward 和两组参数梯度。

这与 official WIRE real-valued Gabor layer 的双投影结构对齐，避免 carrier/envelope 被迫共享方向。

## 待完成

| Gate | 待办 | 状态 |
|---|---|---|
| P0-forward | PyTorch forward/highpass 等价性单测 | 已完成（31 个项目测试全过） |
| P0-data | 主 session 真实数据加载 + EP05 shifts 接入 | 已完成 smoke |
| P0-SIREN | SIREN 训练与 hold-out 指标 | 已完成 Stage 1 正式训练与五项指标 |
| P0-WIRE | WIRE 训练与 hold-out 指标 | 已完成 Stage 1 正式训练与五项指标；artifact 风险高于 SIREN |
| P1-DeepDecoder | Deep Decoder 训练与对照指标 | 脚本 smoke 已完成；正式指标未完成 |
| P1-report | 四方对比结论与 P0 门控判定 | Stage 1 SIREN/WIRE 对比已记录；Deep Decoder 与 EP06 baseline 待补 |
