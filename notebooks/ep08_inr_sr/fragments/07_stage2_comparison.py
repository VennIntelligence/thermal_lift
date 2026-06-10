# %% [markdown]
# ## 6. Stage 2 Five-Way Comparison
#
# Stage 2 将 SIREN、WIRE、Deep Decoder、DeepInverse-DIP 和 EP06 MAP-TV 放入同一汇总表。EP06 优先读取同一 32 帧、256×256、seed=42 split 的 patch-level MAP-TV 指标；如果没有生成，则保留并显式标注旧 full-frame proxy。

# %%
comparison_path = OUTPUT_DIR / "stage2_comparison.csv"
comparison = read_csv_if_exists(comparison_path)
if not comparison.empty:
    display(comparison.round(6))

ep06_patch_dir = OUTPUT_DIR / "ep06_patch_baseline"
ep06_patch_metrics = read_json_if_exists(ep06_patch_dir / "metrics.json")
baseline_config = read_json_if_exists(BASELINE_CONFIG)
display(
    pd.DataFrame(
        [
            {
                "source": "EP06 patch baseline",
                "path": relative(ep06_patch_dir / "metrics.json"),
                "exists": bool(ep06_patch_metrics),
                "stage_gate": ep06_patch_metrics.get("stage_gate"),
                "protocol": ep06_patch_metrics.get("protocol"),
            },
            {
                "source": "legacy full-frame proxy",
                "path": relative(BASELINE_CONFIG),
                "exists": bool(baseline_config),
                "stage_gate": baseline_config.get("source", {}).get("status"),
                "protocol": baseline_config.get("map_tv", {}).get("metric_source_protocol"),
            },
        ]
    )
)
if ep06_patch_metrics:
    display(pd.DataFrame([ep06_patch_metrics]))
else:
    display(pd.DataFrame([baseline_config.get("map_tv", {})]))

# %% [markdown]
# 此五方方法对比表（Stage 2 Five-Way Comparison）汇总了 SIREN、WIRE、Deep Decoder、DeepInverse-DIP 以及经典 MAP-TV 重建方法在同一空间分块及数据划分协议下的量化性能。
# 对比所依托的物理维度包括：泛化重投误差（`holdout_residual`），用以表征前向模型的一致性；子集重构一致性（`split_half_nrmse`），表征重建对于时间帧选择的物理不敏感性；一阶伪影得分（`artifact_score`），表征高频虚假细节的引入概率；以及原始通道一致性（`raw_control_agreement`），量化复原结构与传统热像分布的几何对齐精度。在处理经典 MAP-TV 算法指标时，若其来源为历史全图代理（`ep06_fullframe_proxy`），则其泛化残差与子集差异由于计算域及边界填充的差异而设为空值，以避免将其误读为与分块协议直接可比的性能，从而保证物理基线比对的科学性与无偏性。

# %%
display(show_png_if_exists(OUTPUT_DIR / "stage2_comparison.png"))

# %% [markdown]
# 该五分支量化指标柱状图直观反映了各方法在各项性能评估维度上的相对优劣。
# 在多指标联合判定体系中，单一维度的绝对极值不足以直接构成算法优劣的定论。泛化误差、子集 NRMSE 及伪影得分作为惩罚项应当保持在安全阈值以下；原始通道一致性则作为物理契合度指标，越高越好。若某一神经网络在平均梯度上表现突出，但在 NRMSE 曲线或伪影审计中呈现发散，应当在物理结论中界定为过拟合或引入空间噪声幻觉的高风险分支，进而排除在后续主推重建方案之外。

# %%
display(show_png_if_exists(ep06_patch_dir / "hr_highpass.png"))
display(show_png_if_exists(ep06_patch_dir / "split_half_difference.png"))

# %% [markdown]
# 当分块级的 EP06 MAP-TV 经典重建基线存在时，上方图表并列展示了其高分辨率高通结构响应与子集重构一致性差异图。
# MAP-TV 重建图像的高通响应为评估经典算法在无模型幻觉干扰下的几何边缘重建界限提供了定量比照；而子集差异图空间梯度的均匀程度，则直接表征了正则化参数对于帧序列中随机热噪声及配准波动的压制效能。将分块级经典基线引入对比，完成了经典凸优化算法与深度学习先验方法在相同物理条件、空间范围与评估框架下的联合标定。

# %% [markdown]
# ### Stage 2 Scientific Questions
#
# | Question | Preliminary answer |
# |---|---|
# | INR 是否优于 CNN decoder? | 在真实 32 帧 patch 上，SIREN 比 Deep Decoder 有更好的 hold-out residual 和更强轮廓响应，但 Deep Decoder 的 artifact 与 split-half 更保守；结论是 INR 更适合作为主增强候选，Deep Decoder 是稳定性对照。 |
# | 我们的 Deep Decoder 是否实现正确? | TCForge benchmark 在同一 HR highpass GT 场景下检查 SIREN、WIRE、Deep Decoder、DeepInverse-DIP 四方法；Deep Decoder 使用 native HR 输出尺寸，若尺寸或 forward wrapper 错位会在 benchmark 层暴露。 |
# | 哪个方法最不容易 hallucinate? | Deep Decoder 最保守、artifact 最低；SIREN 在轮廓强度和稳定性之间更均衡；DeepInverse-DIP 和 WIRE 的高频/伪纹理风险更高。 |
# | Stage 3 进入方法 | 推荐 SIREN 作为主方法进入 64/128/248 clean-frame 扩展；Deep Decoder 可作为低 artifact 对照。 |
