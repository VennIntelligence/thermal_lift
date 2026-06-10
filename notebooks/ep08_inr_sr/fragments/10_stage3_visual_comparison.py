# %% [markdown]
# ## 9. Stage 3 — 248 Clean-Frame 全量结果 (Center Crop Detail)
#
# 以下展示四种方法在 **248 clean-frame 全量训练**下的 full-frame 结果。为清晰呈现芯片内部
# 结构，所有图像**裁剪到芯片中心区域**（约 2.5× 放大）。裁剪区域根据 MAP-TV
# highpass 能量分布自动定位。
#
# | 列 | 含义 |
# |---|---|
# | **HR Highpass** | 各方法输出的 highpass 重建，观察轮廓是否清晰、几何正确 |
# | **Raw-control** | bicubic 上采样参照（无 SR 增强），对比基线 |
# | **Split-half A − B** | 输入帧随机分两半独立重建后取差，越白（零）越稳定 |

# %%
TARGET_FRAMES = FULL_CLEAN_FRAMES
runs_found = []
for method in ["ep06_map_tv", "siren", "wire", "deepinv_dip"]:
    d = STAGE3_DIR / f"{method}_{TARGET_FRAMES:03d}_full_preserve"
    if (d / "metrics.json").exists():
        runs_found.append(method)
        print(f"✅ {method}: metrics at {relative(d / 'metrics.json')}")

if not runs_found:
    print(f"\nPending: no {TARGET_FRAMES}-clean-frame results under {relative(STAGE3_DIR)}.")
else:
    show_optional_fig(
        "stage3_visual_comparison.png",
        "Pending: run `uv run python scripts/build_ep08_cache.py` to build the visual comparison figure.",
    )

# %% [markdown]
# Figure 2: Stage 3 visual comparison for 248 clean frames. Center crops compare MAP-TV, SIREN, WIRE, and DeepInverse-DIP with raw-control and split-half views.

# %% [markdown]
# > **图表说明**: 四行分别对应 MAP-TV、SIREN、WIRE、DeepInverse-DIP 在 248 clean-frame 全量训练下的 full-frame 结果。所有图像裁剪到芯片中心区域（约 2.5× 放大），右侧标注四项关键指标。
# >
# > **怎么看**: HR highpass 中芯片轮廓应呈现清晰的**直线和直角**几何结构——工业芯片的物理边缘就是直的。Raw-control 是 bicubic 上采样的温度参照。Split-half 差异图中白色表示两个独立帧子集重建结果一致（稳定），彩色信号表示不稳定区域。
# >
# > **核心发现**:
# > - **MAP-TV**: 轮廓最干净、几何最正确、split-half 最白（最稳定），artifact 最低
# > - **SIREN/WIRE**: 芯片平直边缘被扭曲成波浪形，split-half 差异图中出现明显结构残留
# > - **DeepInverse-DIP**: 背景区域充满噪声幻觉纹理，split-half 差异几乎全是高频噪点，说明 CNN decoder prior 主要拟合了噪声
