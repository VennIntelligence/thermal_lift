# %% [markdown]
# ## 4. SR 数据基础汇总
#
# 这一节把 EP01 的机器可读输出收束为后续 SR agent 可以直接继承的输入规则：
# raw TXT/BMP 完整，主 session 是默认重建输入，跨 session 温度段不能混合。

# %%
with open(PROJECT_ROOT / "configs" / "noise_floor.json") as f:
    noise_config = json.load(f)
noise_floor_c = float(noise_config["noise_floor_celsius"])

main_df = df_sorted[df_sorted["is_main_session"]].copy()
main_coords = set(zip(main_df["X"].astype(int), main_df["Y"].astype(int)))
all_coords = set(zip(df_sorted["X"].astype(int), df_sorted["Y"].astype(int)))
main_mean_range = float(main_df["T_mean"].max() - main_df["T_mean"].min())
frame_shape = (
    int(df_sorted["rows"].mode().iat[0]),
    int(df_sorted["cols"].mode().iat[0]),
)

summary_table = make_ep01_summary_table(
    n_txt=pairing["n_txt"],
    n_bmp=pairing["n_bmp"],
    n_paired=pairing["n_paired"],
    frame_shape=frame_shape,
    coord_count=len(all_coords),
    main_session=main_session,
    main_frames=len(main_df),
    main_coord_count=len(main_coords),
    boundary_jumps=model.boundary_jumps,
    main_mean_range=main_mean_range,
    noise_floor_c=noise_floor_c,
)
summary_table.to_csv(OUTPUT_DIR / "sr_data_basis_summary.csv", index=False)
display(summary_table)

# %% [markdown]
# > **数据说明**: 这张汇总表把 EP01 的文件数量、矩阵尺寸、主 session 帧数、温度跳变量级和坐标覆盖收束到一处。
# > 它是后续 Episode 读取原始数据前应先查看的“输入合同”。
# >
# > **怎么读**: 文件数量和矩阵尺寸回答“数据是否完整且可计算”；
# > `main_session`、`main_frames` 和 `main_coord_count` 回答“默认 SR 输入是哪一段”；
# > session 边界跳变与 noise floor 的对比回答“为什么不能跨 session 混合”。
# >
# > **数据分布**: 全部 263 个 TXT 都有 BMP 伴随文件，矩阵尺寸统一为 480×640；
# > 采集顺序下最大温度段为 session 2，共 255 帧，覆盖全部 253 个实际存在的坐标。
# >
# > **正常/异常理解**: 正常汇总应同时满足三点：矩阵尺寸一致、主 session 足够大、主 session 覆盖二维坐标网格。
# > 如果只满足帧数多但空间覆盖不足，不能直接用于二维 SR；如果空间覆盖好但跨 session 温度跳变大，也不能直接合并。
# >
# > **核心发现**: 后续 SR 的默认输入是主 session 255 帧。
# > 跨 session 的温度跳变达到数十倍噪声底，不能为了增加帧数而直接混合到同一次重建中。

# %%
# Save machine-readable audit products for later episodes.
df_sorted.to_csv(OUTPUT_DIR / "frame_audit.csv", index=False)
print(f"审计数据: output/ep01_data_processing/frame_audit.csv ({len(df_sorted)} frames)")
print("SR 汇总: output/ep01_data_processing/sr_data_basis_summary.csv")

def markdown_table(table: pd.DataFrame) -> str:
    headers = list(table.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in table.iterrows():
        values = [str(row[col]).replace("\n", " ") for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)

main_row = session_summary.loc[session_summary["session"].eq(main_session)].iloc[0]
missing_coords = sorted(
    set((x, y) for x in sorted(VALID_COORDS) for y in sorted(VALID_COORDS)) - all_coords
)
boundary_text = markdown_table(model.boundary_jumps.round(3)) if not model.boundary_jumps.empty else "No detected boundaries."
summary_md = markdown_table(summary_table)
coord_repeat_md = markdown_table(coord_repeat_summary)
r_distribution_md = markdown_table(r_distribution)

report = f"""# EP01 — SR Data Basis and Main-Session Model

## Scope

EP01 audits the raw LWIR TXT/BMP dataset and turns it into a reproducible input model for micro-scan super-resolution. The goal is not to decide SR success or failure; it is to define which frames can be used together, what time order they have, and how session-level temperature drift constrains reconstruction.

## Executable Summary

{summary_md}

## Frame Inventory

All `{pairing['n_txt']}` TXT thermal matrices are readable `{frame_shape[0]} x {frame_shape[1]}` arrays with no NaN/Inf frames, and all have matching BMP companions. TXT remains the numerical input for SR; BMP is retained as same-name visual reference only.

Coordinate/repeat coverage:

{coord_repeat_md}

Repeat-ID distribution:

{r_distribution_md}

The dataset contains `{len(all_coords)}/256` actual coordinates. Missing coordinates are `{missing_coords}`. These gaps are coordinate-level absences, not merely missing `R=0` repeats.

## Acquisition Order and Sessions

Filename order is not acquisition order. Sorting by renamed filename produces `{int(model.filename_session_ids.max()) + 1}` apparent temperature sessions because repeat and early frames are interleaved with the raster grid. Sorting by file modification time recovers `{n_sessions}` physical temperature segments:

{markdown_table(session_summary.round(3))}

Boundary jumps in acquisition order:

{boundary_text}

The main session is session `{main_session}` with `{int(main_row['n_frames'])}` frames. It spans acquisition orders `{int(main_row['first_order'])}` to `{int(main_row['last_order'])}`, covers `{len(main_coords)}/256` coordinates, and has a mean-temperature span of `{main_mean_range:.3f}` deg C within the session.

## SR Input Rule

Downstream SR should inherit `frame_audit.csv` and use `acquisition_order`, `session`, and `is_main_session` as the frame-selection contract. The default 2x contour-level SR POC input is the `{len(main_df)}`-frame main session. Stage/filename coordinates are useful as command priors for initialization or regularization, but actual alignment must be constrained by image data and later EP04 localization quality gates.

Cross-session frames should not be mixed into one reconstruction pass. The detected session-boundary jumps are `{model.boundary_jumps['abs_delta_mean_C'].median():.2f}` deg C median and `{model.boundary_jumps['abs_delta_mean_C'].max():.2f}` deg C max, which are about `{model.boundary_jumps['abs_delta_mean_C'].median() / noise_floor_c:.0f}x` and `{model.boundary_jumps['abs_delta_mean_C'].max() / noise_floor_c:.0f}x` the `{noise_floor_c:.4f}` deg C noise floor.

## Output Files

- `frame_audit.csv`
- `acquisition_order_audit.csv`
- `sr_data_basis_summary.csv`
- `coordinate_coverage_map.png`
- `frame_temperature_statistics.png`
- `robust_temperature_timeline.png`
- `order_comparison.png`
- `session_detection.png`
- `session_coordinate_coverage.png`
"""
(REPORT_DIR / "audit_report.md").write_text(report, encoding="utf-8")
print("审计报告: reports/ep01_data_processing/audit_report.md")

# %% [markdown]
# > **数据说明**: `frame_audit.csv` 是 EP01 给后续 Episode 的主要机器可读产物，
# > 包含 `mtime`、`filename_order`、`acquisition_order`、`session`、`is_main_session` 以及均温/中位温/稳健温度统计。
# > `audit_report.md` 则把同样结论写成可读报告，便于在不同 Episode 间传递上下文。
# >
# > **怎么读**: 后续代码应优先读取 `frame_audit.csv`，并用 `is_main_session` 过滤默认 SR 输入。
# > 如果需要复查某个坐标或某条扫描线，再查看 `acquisition_order_audit.csv`。
# >
# > **数据分布**: 输出文件同时保留逐帧审计、主 session 选择、坐标覆盖和温度段边界。
# > 报告文件将这些结果写成下游 SR 可直接引用的输入规则。
# >
# > **正常/异常理解**: 正常交付物应同时有人可读报告和机器可读 CSV。
# > 如果后续 Episode 重新从文件名字母序推断时序，或绕过 `frame_audit.csv` 重新混合全部帧，就会重复 EP01 已经排除的排序错误。
# >
# > **核心发现**: EP01 的最终交付是主 session 模型：255 帧、253 个实际坐标、真实采集顺序和 session 内重建规则。
# > 后续 SR 不需要重新解释原始文件名顺序，只需继承这里生成的审计产物。
