# %% [markdown]
# ## Step 1 — 运行或加载 EP04 Anchor Gate 结果
#
# 本节生成外轮廓与内轮廓的 `segment x scanline` 质量门控表。若 CSV 已存在，则直接加载缓存；若缺失或 `FORCE_RERUN=True`，则并行重跑。

# %%
outer_results_path = OUTPUT_DIR / "segment_validation_results.csv"
outer_summary_path = OUTPUT_DIR / "segment_summary.csv"
outer_global_path = OUTPUT_DIR / "global_summary.json"

if FORCE_RERUN or not (outer_results_path.exists() and outer_summary_path.exists() and outer_global_path.exists()):
    outer_started = time.perf_counter()
    outer_results = run_all_segments(
        OUTER_SEGMENTS_CSV,
        EP01_OUTPUT_DIR / "frame_audit.csv",
        DATA_DIR,
        OUTPUT_DIR,
        n_jobs=N_JOBS,
        theta_deg=THETA_DEG,
        pixel_size_um=PIXEL_SIZE_UM,
        noise_floor_c=NOISE_SIGMA,
        show_progress=True,
        save_outputs=False,
    )
    outer_runtime_seconds = time.perf_counter() - outer_started
    outer_segment_summary, outer_global_summary = save_validation_outputs(
        outer_results,
        OUTPUT_DIR,
        extra_summary={
            "runtime_seconds": float(outer_runtime_seconds),
            "n_jobs": int(N_JOBS),
            "theta_deg": THETA_DEG,
            "pixel_size_um": PIXEL_SIZE_UM,
            "noise_floor_c": NOISE_SIGMA,
            "segments_csv": str(OUTER_SEGMENTS_CSV),
            "frame_audit_csv": str(EP01_OUTPUT_DIR / "frame_audit.csv"),
            "data_dir": str(DATA_DIR),
            "output_dir": str(OUTPUT_DIR),
        },
    )
else:
    outer_results = pd.read_csv(outer_results_path)
    outer_segment_summary = pd.read_csv(outer_summary_path)
    with open(outer_global_path, encoding="utf-8") as f:
        outer_global_summary = json.load(f)

inner_results_path = INNER_OUTPUT_DIR / "inner_segment_validation_results.csv"
inner_summary_path = INNER_OUTPUT_DIR / "inner_segment_summary.csv"
inner_global_path = INNER_OUTPUT_DIR / "inner_global_summary.json"

if FORCE_RERUN or not (inner_results_path.exists() and inner_summary_path.exists() and inner_global_path.exists()):
    inner_started = time.perf_counter()
    inner_results = run_all_inner_segments(
        INNER_SEGMENTS_CSV,
        EP01_OUTPUT_DIR / "frame_audit.csv",
        DATA_DIR,
        INNER_OUTPUT_DIR,
        n_jobs=N_JOBS,
        theta_deg=THETA_DEG,
        pixel_size_um=PIXEL_SIZE_UM,
        noise_floor_c=NOISE_SIGMA,
        show_progress=True,
        save_outputs=False,
    )
    inner_runtime_seconds = time.perf_counter() - inner_started
    inner_segment_summary, inner_global_summary = save_validation_outputs(
        inner_results,
        INNER_OUTPUT_DIR,
        output_prefix="inner_",
        extra_summary={
            "runtime_seconds": float(inner_runtime_seconds),
            "n_jobs": int(N_JOBS),
            "theta_deg": THETA_DEG,
            "pixel_size_um": PIXEL_SIZE_UM,
            "noise_floor_c": NOISE_SIGMA,
            "segments_csv": str(INNER_SEGMENTS_CSV),
            "frame_audit_csv": str(EP01_OUTPUT_DIR / "frame_audit.csv"),
            "data_dir": str(DATA_DIR),
            "output_dir": str(INNER_OUTPUT_DIR),
        },
    )
else:
    inner_results = pd.read_csv(inner_results_path)
    inner_segment_summary = pd.read_csv(inner_summary_path)
    with open(inner_global_path, encoding="utf-8") as f:
        inner_global_summary = json.load(f)

anchor_summary_table = combined_anchor_summary_table(outer_segment_summary, inner_segment_summary)
display(anchor_summary_table)

# %% [markdown]
# > **数据说明**: 表格按外轮廓、内轮廓和合并口径汇总 segment 数、A-class 数、anchor 通过数、split-half、CRB ratio 与 SNR。
# > **读法**: 关注 `n_segments` 和 `n_pass` 的关系，而不是只看通过率。`split_half` 越小表示奇偶帧子集给出的边缘位置越一致；`CRB ratio` 是实测 split-half 相对理论噪声下限的倍数，接近合理范围说明定位稳定性与噪声模型大体一致；`SNR` 只说明热对比是否足够。
# > **正常/异常理解**: 外轮廓通过更多是正常现象，因为外边界通常更强、更简单；内轮廓低通过率不等于内部结构不存在，而是说明局部热边缘在当前 localization 模型下不够稳定。若某类段 SNR 很高但仍 fail，应优先怀疑局部模型、NCC 相位覆盖或 split-half 稳定性，而不是直接否定该结构。
# > **对本 Episode 的意义**: EP04 给 EP06 提供的是“哪些局部边界可用于配准”的 benchmark；内轮廓低通过率表示 localization-only gate 覆盖不足，不表示内部结构 SR 不值得做。

# %%
global_table = pd.concat(
    [
        pd.Series(outer_global_summary, name="outer").rename_axis("metric"),
        pd.Series(inner_global_summary, name="inner").rename_axis("metric"),
    ],
    axis=1,
).reset_index()
display(global_table)

# %% [markdown]
# > **数据说明**: 该表保留外/内轮廓批处理的全局元数据，包括输入 CSV、运行时间、噪声和 stage prior 配置。
# > **读法**: 这张表不是性能排行榜，而是复现实验账本。应检查 outer/inner 是否使用同一 `frame_audit_csv`、同一 `data_dir`、同一 noise floor、同一 θ 和 pixel size。
# > **正常/异常理解**: 正常情况是外/内轮廓共享同一个主 session 和同一套 highpass-NCC + joint ESF + split-half/CRB gate；如果两列配置不同，后续 pass/fail 就不能直接比较。运行时间差异本身不代表质量差异，只反映段数、缓存和并行状态。
# > **对本 Episode 的意义**: 后续报告可以复现实验输入与运行配置；位移 prior 只记录配置背景，不作为对齐真值。实际对齐证据来自红外帧之间的数据一致性，而不是 stage command 本身。
