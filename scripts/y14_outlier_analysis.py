"""
Y=14 µm outlier analysis for EP02 displacement calibration.

Questions:
1. Is Y=14 a significant noise outlier in the Y-coordinate measurements?
2. Does removing Y=14 improve the 4/2 projection ratio or RMS?
3. What does the gradient_ncc (contour-only) method show vs raw_ncc?
"""

import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR = Path("output/ep02_displacement_calibration")

# ── 1. Load Y-coordinate method measurements ────────────────────────
y_df = pd.read_csv(OUT_DIR / "y_coordinate_method_measurements.csv")
print(f"Total Y-coordinate measurements: {len(y_df)}")
print(f"Columns: {list(y_df.columns)}")
print(f"Methods: {y_df['method_label'].unique()}")
print(f"Scan axes: {y_df['scan_axis'].unique()}")
print(f"Fixed coords: {sorted(y_df['fixed_coord_um'].unique())}")
print()

# ── 2. Focus on Y-scan pairs (delta_Y != 0) ─────────────────────────
y_scan = y_df[y_df['delta_Y_um'] != 0].copy()
print(f"Y-scan pairs (delta_Y != 0): {len(y_scan)}")
print(f"delta_Y_um values: {sorted(y_scan['delta_Y_um'].unique())}")
print()

# ── 3. Per-fixed_coord statistics for each method ────────────────────
for method in ['raw_ncc', 'highpass_ncc', 'gradient_ncc']:
    sub = y_scan[y_scan['method_label'] == method].copy()
    if len(sub) == 0:
        print(f"[{method}] — no data")
        continue

    print(f"═══ {method} ═══")

    # Group by fixed_coord and delta_Y
    for delta_y in sorted(sub['delta_Y_um'].unique()):
        dsub = sub[sub['delta_Y_um'] == delta_y]
        print(f"\n  delta_Y = {delta_y} µm:")

        by_coord = dsub.groupby('fixed_coord_um').agg(
            n=('ref_residual_px', 'count'),
            median_residual=('ref_residual_px', 'median'),
            median_parallel=('parallel_px', 'median'),
            median_perp=('perpendicular_px', 'median'),
            median_peak_ncc=('peak_ncc', 'median'),
            median_mag=('measured_mag_px', 'median'),
        ).round(4)
        print(by_coord.to_string())

    # Overall stats with and without specific fixed_coords
    for delta_y in sorted(sub['delta_Y_um'].unique()):
        dsub = sub[sub['delta_Y_um'] == delta_y]

        all_resid = dsub['ref_residual_px'].values
        all_parallel = dsub['parallel_px'].values

        rms_all = np.sqrt(np.mean(all_resid**2))
        med_parallel_all = np.median(all_parallel)

        # Identify which fixed_coords are outliers
        by_coord = dsub.groupby('fixed_coord_um')['ref_residual_px'].median()
        overall_median = by_coord.median()
        overall_mad = np.median(np.abs(by_coord - overall_median))

        print(f"\n  delta_Y={delta_y}: overall RMS={rms_all:.4f}, median_parallel={med_parallel_all:.4f}")
        print(f"  Per-coord residual median: overall_median={overall_median:.4f}, MAD={overall_mad:.4f}")

        # Flag outliers (> 3*MAD from median)
        threshold = overall_median + 3 * overall_mad if overall_mad > 0 else overall_median * 1.5
        outliers = by_coord[by_coord > threshold]
        if len(outliers) > 0:
            print(f"  ⚠️  Outlier coords (residual > {threshold:.4f}): {dict(outliers.round(4))}")
        else:
            print(f"  No outlier coords detected (threshold={threshold:.4f})")

    print()

# ── 4. Specific Y=14 analysis ───────────────────────────────────────
print("═══ Y=14 SPECIFIC ANALYSIS ═══")
print()

# Check if Y=14 appears as fixed_coord or as Y_a/Y_b
y14_as_fixed = y_scan[y_scan['fixed_coord_um'] == 14]
y14_as_ya = y_scan[y_scan['Y_a'] == 14]
y14_as_yb = y_scan[y_scan['Y_b'] == 14]

print(f"Y=14 as fixed_coord: {len(y14_as_fixed)} rows")
print(f"Y=14 as Y_a: {len(y14_as_ya)} rows")
print(f"Y=14 as Y_b: {len(y14_as_yb)} rows")
print()

# The Y-scan pairs: fixed_coord is the X coordinate, and Y_a/Y_b vary
# So Y=14 as Y_a or Y_b means the pair involves the Y=14 coordinate line
y14_involved = y_scan[(y_scan['Y_a'] == 14) | (y_scan['Y_b'] == 14)]
print(f"Pairs involving Y=14 (either Y_a or Y_b): {len(y14_involved)} rows")
if len(y14_involved) > 0:
    print(y14_involved[['file_a', 'file_b', 'delta_Y_um', 'method_label',
                         'ref_residual_px', 'parallel_px', 'peak_ncc', 'measured_mag_px']].to_string())
print()

# ── 5. Impact of removing Y=14 involved pairs ───────────────────────
print("═══ IMPACT OF REMOVING Y=14 ═══")
print()

for method in ['raw_ncc', 'highpass_ncc', 'gradient_ncc']:
    sub = y_scan[y_scan['method_label'] == method]
    if len(sub) == 0:
        continue

    sub_no14 = sub[(sub['Y_a'] != 14) & (sub['Y_b'] != 14)]

    print(f"--- {method} ---")
    for delta_y in sorted(sub['delta_Y_um'].unique()):
        d_all = sub[sub['delta_Y_um'] == delta_y]
        d_no14 = sub_no14[sub_no14['delta_Y_um'] == delta_y]

        if len(d_all) == 0:
            continue

        rms_all = np.sqrt(np.mean(d_all['ref_residual_px'].values**2))
        med_par_all = np.median(d_all['parallel_px'].values)

        rms_no14 = np.sqrt(np.mean(d_no14['ref_residual_px'].values**2)) if len(d_no14) > 0 else np.nan
        med_par_no14 = np.median(d_no14['parallel_px'].values) if len(d_no14) > 0 else np.nan

        print(f"  delta_Y={delta_y}: ALL n={len(d_all)}, RMS={rms_all:.4f}, parallel={med_par_all:.4f}")
        print(f"  delta_Y={delta_y}: NO14 n={len(d_no14)}, RMS={rms_no14:.4f}, parallel={med_par_no14:.4f}")

    # 4/2 ratio
    d2_all = sub[sub['delta_Y_um'] == 2]['parallel_px'].median()
    d4_all = sub[sub['delta_Y_um'] == 4]['parallel_px'].median()
    d2_no14 = sub_no14[sub_no14['delta_Y_um'] == 2]['parallel_px'].median() if len(sub_no14[sub_no14['delta_Y_um'] == 2]) > 0 else np.nan
    d4_no14 = sub_no14[sub_no14['delta_Y_um'] == 4]['parallel_px'].median() if len(sub_no14[sub_no14['delta_Y_um'] == 4]) > 0 else np.nan

    ratio_all = d4_all / d2_all if d2_all != 0 else np.nan
    ratio_no14 = d4_no14 / d2_no14 if d2_no14 != 0 else np.nan

    print(f"  4/2 ratio: ALL={ratio_all:.4f} (2um={d2_all:.4f}, 4um={d4_all:.4f})")
    print(f"  4/2 ratio: NO14={ratio_no14:.4f} (2um={d2_no14:.4f}, 4um={d4_no14:.4f})")
    print(f"  Expected 4/2 ratio: 2.0000")
    print()

# ── 6. Time-adjacent X-step: does Y=14 row show anomaly? ────────────
print("═══ TIME-ADJACENT X-STEP: Y=14 ROW CHECK ═══")
print()

ta_df = pd.read_csv(OUT_DIR / "time_adjacent_method_measurements.csv")
ta_xstep = ta_df[(ta_df['move_type'] == 'x_step') & (ta_df['method_label'] == 'highpass_ncc')]

by_y = ta_xstep.groupby('Y_a').agg(
    n=('ref_residual_px', 'count'),
    rms_residual=('ref_residual_px', lambda x: np.sqrt(np.mean(x**2))),
    median_residual=('ref_residual_px', 'median'),
    median_peak=('peak_ncc', 'median'),
    median_mag=('measured_mag_px', 'median'),
).round(4)

print("Per-Y-row X-step quality (highpass_ncc):")
print(by_y.to_string())
print()

# Flag Y=14 specifically
if 14 in by_y.index:
    y14_rms = by_y.loc[14, 'rms_residual']
    other_rms = by_y.drop(14)['rms_residual']
    print(f"Y=14 row RMS: {y14_rms:.4f}")
    print(f"Other rows RMS: median={other_rms.median():.4f}, range=[{other_rms.min():.4f}, {other_rms.max():.4f}]")

    # Is Y=14 an outlier in X-step quality?
    med = other_rms.median()
    mad = np.median(np.abs(other_rms - med))
    z_score = (y14_rms - med) / mad if mad > 0 else 0
    print(f"Y=14 MAD z-score: {z_score:.2f} (>3 = strong outlier)")

# ── 7. AVI y14um vs others ──────────────────────────────────────────
print()
print("═══ AVI Y14UM VS OTHER Y-SCAN AVIS ═══")
print()

avi_df = pd.read_csv(OUT_DIR / "avi_direction_summary.csv")
y_avis = avi_df[avi_df['scan_axis'] == 'y'].copy()

cols = ['avi', 'path_straightness', 'median_angle_row_down_deg', 'median_magnitude_px', 'magnitude_robust_cv', 'median_peak_ncc']
print(y_avis[cols].to_string(index=False))
print()

y14_straight = y_avis[y_avis['avi'] == 'y14um.avi']['path_straightness'].values
other_straight = y_avis[y_avis['avi'] != 'y14um.avi']['path_straightness'].values
if len(y14_straight) > 0:
    print(f"y14um straightness: {y14_straight[0]:.4f}")
    print(f"Others straightness: median={np.median(other_straight):.4f}, min={np.min(other_straight):.4f}")

print("\n═══ DONE ═══")
