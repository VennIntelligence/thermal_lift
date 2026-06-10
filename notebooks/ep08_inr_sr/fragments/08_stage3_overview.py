# %% [markdown]
# ## 7. Stage 3 Full-Frame Progressive Expansion
#
# Stage 3 把 EP08 从 32 帧、中心 patch 的方法筛选推进到主 session 全帧 POC。核心决策是：SIREN、WIRE 和 DeepInverse-DIP 都按同一 forward/metric 协议扩展到 full frame；Deep Decoder 在 Stage 2 已作为低 artifact 保守对照完成使命，不再进入长时间 full-frame 训练；EP06 MAP-TV 作为 classic optimization baseline 保留在 Stage 3 对比中。
#
# | Decision area | Stage 3 setting | Why it matters |
# |---|---|---|
# | Methods | SIREN / WIRE / DeepInverse-DIP scale to full frame; Deep Decoder retired | 保留 INR 与 DIP 两类可扩展候选，同时避免把 Stage 2 已显示欠表达的 decoder prior 消耗到全帧训练预算。 |
# | Baseline | EP06 MAP-TV included under the same Stage 3 loader/split where possible | MAP-TV 提供非学习型 classic baseline，帮助区分 INR 增强和 TV 正则带来的保守轮廓。 |
# | Geometry | LR `480x640` maps to HR `960x1280` at 2x | Stage 3 直接覆盖主 session 全视场，不再只评估中心 `256x256` patch。 |
# | Coordinates | `preserve` is default; `stretch` is kept as an ablation | `preserve` 保持矩形视场的物理长宽比，`stretch` 复查 Stage 1/2 legacy 坐标归一化是否引入方向性偏置。 |
# | Progressive plan | 64 -> 128 -> 248 clean frames | 先用较小帧数检查训练健康、内存和 artifact，再推进到 248 clean-frame 主 session full run。 |
#
# Stage 3 的验收仍然是 contour-level POC：看芯片内部结构/形状轮廓是否更清楚、更稳定，而不是宣称 5 um 计量级温度分辨率。Stage command 继续只作为位移 prior / 初始化 / 正则约束；alignment 质量和 split-half 稳定性必须由数据证据约束。
