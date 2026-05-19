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
# > **数据说明**: `stage2_comparison.csv` 汇总五方指标；EP06 baseline 优先来自 `output/ep08_inr_sr/ep06_patch_baseline/metrics.json`，缺失时才回退到配置中的 full-frame proxy。
# >
# > **怎么看**: 对真实数据，hold-out residual 低表示 forward consistency 强；split-half NRMSE 低表示不同训练子集恢复更稳定；artifact score 低表示伪高频更少；raw-control agreement 高表示增强结构更贴近普通视觉参照；P95 gradient 只是边缘强度 proxy。
# >
# > **正常/异常**: 如果 EP06 row 的 `source_type` 是 `ep06_fullframe_proxy`，hold-out residual 和 split-half NRMSE 为空是协议差异，不是 0，也不是失败分数；若 `source_type` 是 `ep08_patch_protocol`，五项指标应都按同一 patch split 生成。
# >
# > **核心发现**: 这张表的主要作用是避免把旧 EP06 full-frame proxy 当成同协议基线；只有 patch-level MAP-TV 存在时，EP06 才能与四个 EP08 方法进行五指标并排比较。

# %%
display(show_png_if_exists(OUTPUT_DIR / "stage2_comparison.png"))

# %% [markdown]
# > **图表说明**: 柱状图把五方方法的五项指标并排显示；缺失 EP06 hold-out/split-half 以 `n/a` 标注。
# >
# > **怎么看**: 不同指标方向不同，不能把柱子整体最高/最低当作单一胜负。hold-out、split-half、artifact 越低越好；raw-control 越高越好；P95 gradient 只作参考。
# >
# > **正常/异常**: 对数坐标下小差异可能被压缩，大差异更醒目；空值不会参与柱高。若某方法只有 P95 gradient 高但 artifact 和 split-half 也高，应按高频风险而非成功解释。
# >
# > **核心发现**: Stage 3 推荐优先扩展 SIREN；Deep Decoder 可作为保守低 artifact 对照，DeepInverse-DIP 可作为过拟合风险参照，WIRE 暂不作为首选。

# %%
display(show_png_if_exists(ep06_patch_dir / "hr_highpass.png"))
display(show_png_if_exists(ep06_patch_dir / "split_half_difference.png"))

# %% [markdown]
# > **图表说明**: 若已生成 patch-level EP06 baseline，上方显示 MAP-TV 的 HR highpass 图和 split-half difference 图；缺失时 notebook 会报告对应 PNG 不存在。
# >
# > **怎么看**: HR highpass 图用于看 MAP-TV 是否恢复出与 EP08 方法可比的结构响应；split-half difference 越接近白色，表示两半帧重建越稳定。
# >
# > **正常/异常**: highpass 图中红/蓝是相对局部背景的正/负响应，不是普通温度色彩；split-half difference 出现强边缘或棋盘纹说明 MAP-TV 对帧子集敏感。
# >
# > **核心发现**: patch-level 图存在时，EP06 classic baseline 不再只是表格 proxy，而能以相同 patch、相同 split 和相同解释边界进入 Stage 2 对比。

# %% [markdown]
# ### Stage 2 Scientific Questions
#
# | Question | Preliminary answer |
# |---|---|
# | INR 是否优于 CNN decoder? | 在真实 32 帧 patch 上，SIREN 比 Deep Decoder 有更好的 hold-out residual 和更强轮廓响应，但 Deep Decoder 的 artifact 与 split-half 更保守；结论是 INR 更适合作为主增强候选，Deep Decoder 是稳定性对照。 |
# | 我们的 Deep Decoder 是否实现正确? | TCForge benchmark 在同一 HR highpass GT 场景下检查 SIREN、WIRE、Deep Decoder、DeepInverse-DIP 四方法；Deep Decoder 使用 native HR 输出尺寸，若尺寸或 forward wrapper 错位会在 benchmark 层暴露。 |
# | 哪个方法最不容易 hallucinate? | Deep Decoder 最保守、artifact 最低；SIREN 在轮廓强度和稳定性之间更均衡；DeepInverse-DIP 和 WIRE 的高频/伪纹理风险更高。 |
# | Stage 3 进入方法 | 推荐 SIREN 作为主方法进入 64/128/255 帧扩展；Deep Decoder 可作为低 artifact 对照。 |
