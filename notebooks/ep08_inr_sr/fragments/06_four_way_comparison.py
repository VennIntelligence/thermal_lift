# %% [markdown]
# ## 5. Stage 1 Comparison
#
# Stage 1 只比较 SIREN 与 WIRE 的 32 帧 256x256 patch 结果。EP06 baseline 仍为 null 占位，因此本表不做绝对阈值判定。

# %%
comparison_path = OUTPUT_DIR / "stage1_comparison.csv"
comparison = read_csv_if_exists(comparison_path)
if comparison.empty:
    comparison = pd.DataFrame(
        [
            {
                "method": "SIREN",
                "family": "inr_sine",
                "holdout_residual": None,
                "split_half_nrmse": None,
                "artifact_score": None,
                "raw_control_agreement": None,
                "p95_gradient": None,
                "stage1_gate": "missing",
            },
            {
                "method": "WIRE",
                "family": "inr_gabor",
                "holdout_residual": None,
                "split_half_nrmse": None,
                "artifact_score": None,
                "raw_control_agreement": None,
                "p95_gradient": None,
                "stage1_gate": "missing",
            },
        ]
    )
display(comparison)

# %% [markdown]
# > **数据说明**: 表格读取 `stage1_comparison.csv`，汇总 SIREN 与 WIRE 的 hold-out residual、split-half NRMSE、artifact score、raw-control agreement 和 P95 gradient。
# >
# > **怎么看**: Hold-out residual、split-half NRMSE 和 artifact score 通常越低越好；raw-control agreement 越高越好；P95 gradient 只是边缘响应强度 proxy，较大也可能来自噪声或振铃。
# >
# > **正常/异常**: 空值是当前阶段的诚实占位，不是零、不是失败分数，也不是通过。EP06 baseline 指标仍为 null，因此本阶段只检查 INR 自身是否收敛并产出合理结构。
# >
# > **核心发现**: Stage 1 的核心问题是 WIRE 与 SIREN 是否都能在相同 split 下稳定训练，并给出可比较的五项指标；不是宣布最终优于 EP06。

# %%
display(show_png_if_exists(OUTPUT_DIR / "stage1_comparison.png"))

# %% [markdown]
# > **图表说明**: 若存在，`stage1_comparison.png` 应把 SIREN 与 WIRE 放在同一 ROI 和同一显示尺度下比较。
# >
# > **怎么看**: 先看 raw-control 是否保持结构一致，再看 highpass 中轮廓是否更连贯；不要用展示倍率或单项锐度指标判断 SR 成功。
# >
# > **正常/异常**: 如果不同方法使用不同 ROI、不同 color scale 或不同输入 frame set，图像不能作为公平对比证据。
# >
# > **核心发现**: 当前 notebook 已固定 Stage 1 对比展示位置；后续四方对比要等 Deep Decoder 与 EP06 baseline 指标补齐。
