# %% [markdown]
# ## 1. 数据集生成架构
#
# EP07 的核心产物不是“几张合成图”，而是一条可复现的数据生成与验收链路。下面的 Mermaid 图把后台实际做的事展开：配置定义物理边界，TCForge 生成 HR 真值与 LR burst，manifest/smoke/evaluate 把文件契约和数值检查写成可复查产物，最后才供 SR 算法消费。
#
# ```mermaid
# flowchart LR
#     subgraph C[Config / prior]
#         C1["phantom_smoke.json<br/>scale, shape, PSF, noise, highpass"]
#         C2["shift_profiles.json<br/>real EP05 shifts / ideal phase grid / stage prior"]
#     end
#
#     subgraph G[TCForge generator]
#         G1["geometry<br/>binary chip mask"]
#         G2["physics<br/>HR temperature + low-frequency background"]
#         G3["shifts<br/>LR-to-reference alignment shifts"]
#         G4["forward<br/>exact_ep06_point or physical_block_average"]
#         G5["noise / drift<br/>detector noise and optional P1 drift"]
#         G6["highpass<br/>EP06-compatible structure maps"]
#     end
#
#     subgraph A[Scene artifact]
#         A1["hr_mask_2x.npy<br/>sharp structure ground truth"]
#         A2["hr_temperature_2x.npy<br/>smooth physical temperature ground truth"]
#         A3["hr_edge_map_2x.npy<br/>contour proxy"]
#         A4["lr_burst_raw.npy<br/>ordinary LR temperature observations"]
#         A5["lr_burst_highpass.npy<br/>structure-response observations"]
#         A6["shifts.npy<br/>alignment prior/control"]
#         A7["metadata.json<br/>physical contract and provenance"]
#     end
#
#     subgraph V[Validation]
#         V1["manifest.csv/json<br/>scene index and hashes"]
#         V2["smoke_test_report.json<br/>file, shape, dtype, finite, highpass parity"]
#         V3["evaluation_summary.csv/json<br/>coverage, ranges, shift norms, highpass reference"]
#     end
#
#     subgraph U[Consumers]
#         U1["SR algorithm regression<br/>compare reconstruction to HR truth"]
#         U2["Notebook/report<br/>show generation logic and smoke evidence"]
#     end
#
#     C1 --> G1
#     C1 --> G2
#     C1 --> G4
#     C1 --> G5
#     C1 --> G6
#     C2 --> G3
#     G1 --> G2
#     G1 --> A1
#     G1 --> A3
#     G2 --> G4
#     G3 --> G4
#     G4 --> G5
#     G5 --> A4
#     A4 --> G6
#     G6 --> A5
#     G2 --> A2
#     G3 --> A6
#     G1 --> A7
#     G2 --> A7
#     G3 --> A7
#     G4 --> A7
#     A1 --> V1
#     A2 --> V1
#     A3 --> V1
#     A4 --> V1
#     A5 --> V1
#     A6 --> V1
#     A7 --> V1
#     V1 --> V2
#     V1 --> V3
#     A1 --> U1
#     A2 --> U1
#     A3 --> U1
#     A4 --> U1
#     A5 --> U1
#     A6 --> U1
#     V2 --> U2
#     V3 --> U2
# ```
#
# > **图表说明**: 这是 EP07 数据集生成逻辑的 Markdown/Mermaid 架构图，节点对应仓库中的配置文件、TCForge 模块、落盘数组和验收产物。
# >
# > **怎么看**: 左到右是数据流。`hr_mask_2x.npy` 是几何结构真值，`hr_temperature_2x.npy` 是经过物理渲染后的连续温度真值，`lr_burst_raw.npy` 和 `lr_burst_highpass.npy` 才是算法输入侧观测。
# >
# > **正常/异常**: `shifts.npy` 是合成 forward 的已知控制量，但这不允许外推到真实数据，把真实 stage command 当 alignment ground truth。`highpass` 图是结构响应，不是绝对温度图。
# >
# > **核心发现**: EP07 的工作量主要在“数据契约”和“可复查验收”上；notebook 必须展示这条链路，否则读者只会看到几张孤立图片。

# %% [markdown]
# ### 模块契约与使用方式
#
# | 层级 | 主要入口 | 生成或检查什么 | 结果怎么用 | 不能怎么用 |
# |---|---|---|---|---|
# | Geometry | `tcforge.geometry.build_scene_mask()` | 二值芯片结构、引脚、开窗、沟槽、旋转扰动 | 作为 contour/shape ground truth | 不能把 mask 当温度场 |
# | Physics | `tcforge.physics.render_temperature_field()` | 背景温度、结构温升、低频热背景、噪声/漂移 | 作为 HR temperature truth | 不能要求边缘像 mask 一样无限锐利 |
# | Shifts | `tcforge.shifts.load_shift_profile()` | ideal phase grid、EP05 refined shifts、stage prior control | 作为 synthetic observation 的 alignment 控制 | 不能替代真实数据 alignment 真值 |
# | Forward | `tcforge.forward.generate_lr_burst()` | HR -> LR burst，支持 `exact_ep06_point` / `physical_block_average` | 生成算法输入和 forward-model regression | 两种 forward mode 不能混在一个指标表里 |
# | Highpass | `tcforge.highpass.highpass_preprocess()` | EP06 风格结构响应：`float32`、spatial-only sigma、`nearest` | 边缘/结构输入或诊断图 | 不能解释为绝对升温/降温 |
# | Manifest | `scripts/generate_thermal_chip_phantom.py` | `metadata.json`、`manifest.csv/json`、scene 文件索引 | 保证数据包可重建、可复查 | 不能只保存图片而丢失 metadata |
# | Smoke | `scripts/smoke_test_thermal_chip_phantom.py` | 文件、shape、dtype、finite、mask、shift、高通独立复算 | P0 硬门控 | 不能替代算法质量评估 |
# | Evaluate | `scripts/evaluate_thermal_chip_phantom.py` / `tcforge.evaluate` | scene 统计、coverage、shift norm、highpass reference diff、可选 SR 指标 | 回归摘要和后续算法对照 | 没有 SR 输出时不能报告 SR 成功 |
#
# > **数据说明**: 这张表把每个模块的输入输出契约和使用边界写在 notebook 正文里，避免把生成脚本里的工程细节藏在后台。
# >
# > **怎么看**: “结果怎么用”说明后续 SR 算法应消费的证据；“不能怎么用”说明 EP07 不作真实数据结论的边界。
# >
# > **异常是否正常**: HR 温度场比 mask 模糊是正常物理渲染；highpass 正负响应同时出现也是正常背景扣除结果。
# >
# > **核心发现**: EP07 可以作为 SR 算法的可控回归基准，但它必须先通过数据包自洽检查，再谈方法比较。
