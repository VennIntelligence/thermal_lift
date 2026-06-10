# %% [markdown]
# ## A5. 紧凑存盘：training_pool 里到底有什么？
#
# 每个 `scene_xxxx/` 目录（compact 格式）典型文件：
#
# | 文件 | 是否存储 | 训练时用途 |
# |------|----------|------------|
# | `hr_mask_2x.png` | ✅ | 重建 GT 温度 + loss 加权 mask |
# | `hr_edge_2x.png` | ✅ | 可选边缘监督 |
# | `obs_features_1x.npz` | ✅ | **UNet 输入** |
# | `metadata.json` | ✅ | 温度渲染参数 |
# | `shifts.npy` | ✅ | 记录生成位移 |
# | 完整 LR burst (248帧) | ❌ | 仅离线生成阶段存在 |
# | HR temperature .npy | ❌ | 训练时在线 `reconstruct_hr_temperature()` |

# %%
save_fig("06_compact_storage_schematic.png")

# %% [markdown]
# ## A6. DataLoader 裁剪 + UNet 输入输出
#
# **Step 7** `ThermalSRDataset` 随机裁 HR 对齐 patch（默认 256×256）  
# **Step 8** `ThermalSRUNet`: **5ch @ 1×LR → 1ch @ 2×HR**（scale=2 时）
#
# 下图：全场景 obs ch0、裁出的 LR patch、对应的 HR target patch（中心 128×128 示例）。

# %%
save_fig("07_patch_and_unet_io.png")

# %% [markdown]
# > **图表说明**: 展示从全场景到 patch 级训练样本；右图 HR target 来自 TCForge 中心裁剪，几何保持旋转。
# >
# > **怎么看**: 左→中是下采样分辨率，右是 2× 超分目标分辨率。
# >
# > **核心发现**: 进入 loss 之前，数据已经过 **离线融合 + 在线裁 patch + 在线重建 GT** 三道工序。
