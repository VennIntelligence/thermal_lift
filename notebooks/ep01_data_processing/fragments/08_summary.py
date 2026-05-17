# %% [markdown]
# ## 审计总结
#
# | 指标 | 值 |
# |------|------|
# | TXT 帧数 | 263 |
# | 矩阵尺寸 | 480 × 640 |
# | 唯一坐标 | 253 |
# | NaN / Inf | 0 / 0 |
# | 温度范围 | [18.21, 26.80] °C |
# | 采集顺序 session 数 | 3 |
# | 主扫描 session | 最大温度段，255 帧 |
#
# > **数据说明**: 这张汇总表把 EP01 对原始 TXT 数据、坐标覆盖、重命名核查和采集顺序 session 的审计结果收束到一处。
# >
# > **数据分布**: 数据矩阵完整、坐标覆盖接近完整；真实采集顺序显示一个 255 帧主扫描温度段，
# > 以及少量早期低温/补采帧。R=0 主网格按 Y 行、X 列顺序采集，坐标解码与采集顺序一致。
# >
# > **核心发现**: EP01 已确认数据可用于后续算法，但后续 Episode 必须使用 `acquisition_order` 和主扫描 session。
# > 文件名字母序产生的 13 sessions 是排序伪影，不应继续作为标定依据。

# %%
# Save the audit CSV and report for later episodes.
df_sorted.to_csv(OUTPUT_DIR / "frame_audit.csv", index=False)
print(f"💾 审计数据: output/ep01_data_processing/frame_audit.csv ({len(df_sorted)} 帧)")

main_row = session_summary.loc[session_summary["session"].eq(main_session)].iloc[0]
report = f"""# EP01 Data Audit Report

## Scope

This report validates the renamed TXT thermal matrices, their BMP companions, coordinate decoding, and acquisition-order session structure before super-resolution or displacement calibration work.

## Main Result

The TXT data are readable and internally complete. Coordinate decoding is consistent with the original naming rules and with the acquisition-time scan order. The previous 13-session interpretation came from sorting by renamed filename rather than by acquisition time.

| Metric | Value |
|---|---:|
| TXT frames | {len(df_sorted)} |
| BMP files | {pairing['n_bmp']} |
| TXT/BMP paired files | {pairing['n_paired']} |
| Frame shape | {int(df_sorted['rows'].mode().iat[0])} x {int(df_sorted['cols'].mode().iat[0])} |
| NaN / Inf frames | {int(df_sorted['has_nan'].sum())} / {int(df_sorted['has_inf'].sum())} |
| Unique coordinates | {len(coord_map)} |
| Missing coordinates | {256 - len(coord_map)} |
| 3-repeat coordinates | {sum(1 for repeats in coord_map.values() if len(repeats) == 3)} |
| 2-repeat coordinates | {sum(1 for repeats in coord_map.values() if len(repeats) == 2)} |
| Temperature range | [{df_sorted['T_min'].min():.2f}, {df_sorted['T_max'].max():.2f}] deg C |
| Mean-temperature range | [{df_sorted['T_mean'].min():.2f}, {df_sorted['T_mean'].max():.2f}] deg C |
| Filename-order sessions | {int(filename_session_ids.max()) + 1} |
| Acquisition-order sessions | {n_sessions} |
| Main acquisition session | {main_session} ({int(main_row['n_frames'])} frames) |
| Main session mean temperature | {main_row['mean_temp']:.3f} deg C |
| R=0 row-order mismatches | {len(row_order_mismatches)} |

## Interpretation

All 263 TXT frames are valid 480 x 640 matrices with no NaN or Inf values, and all TXT files have matching BMP companions. Coordinate coverage is high at 253 / 256 positions, with three full coordinate gaps: (14,6), (16,6), and (16,16).

The renamed filename order is not the acquisition order. Sorting by filename produces 13 apparent sessions by inserting repeat and warm-up frames into the middle of the main scan. Sorting by file modification time recovers the physical acquisition order: a few early low-temperature/repeat frames followed by a 255-frame main scan near 23.3 deg C.

R=0 acquisition order follows the expected scan pattern: Y increases from 0 to 40 um, and X increases within each Y row. This supports the current coordinate decoding and points to ordering, not renaming, as the source of the earlier session artifact.

## Downstream Rule

EP02 and later SR work should use `session`, `is_main_session`, and `acquisition_order` from `frame_audit.csv`. The main displacement calibration should use only the main acquisition session unless there is a specific reason to analyze warm-up or repeat frames separately.

## Output Files

- `frame_audit.csv`
- `acquisition_order_audit.csv`
- `coordinate_coverage_map.png`
- `frame_temperature_statistics.png`
- `temperature_timeline.png`
- `session_detection.png`
"""
(REPORT_DIR / "audit_report.md").write_text(report, encoding="utf-8")
print("💾 审计报告: reports/ep01_data_processing/audit_report.md")

# %% [markdown]
# > **数据说明**: `frame_audit.csv` 是 EP01 给后续 Episode 的主要机器可读产物，
# > 现在包含 `source_file`、`mtime`、`filename_order`、`acquisition_order`、`session` 和 `is_main_session`。
# >
# > **数据分布**: CSV 保留每一帧的坐标、repeat、温度统计和采集时序；
# > 报告保留重命名核查、采集顺序核查、session 修正和后续使用规则。
# >
# > **核心发现**: EP01 的修正结论已经足够支撑 EP02 重跑：
# > 坐标重命名可继续使用，但主分析必须排除早期低温/补采帧，只用主扫描 session。
