#!/usr/bin/env python3
"""Diagnose stage prior (old_stage_model) outliers using only standard library.

用法: uv run python scripts/diagnose_stage_outliers.py（无参数）
输入: EP05 contour alignment CSV（default_contour_alignment_csv()）、configs/stage_calibration.json
输出: 仅终端报告（stage prior 残差统计 + Top-20 离群帧），不写文件
关联: EP05
"""

import csv
import json
import math
from pathlib import Path

def project_root() -> Path:
    root = Path.cwd()
    while root != root.parent and not (root / "AGENTS.md").exists():
        root = root.parent
    return root

def coordinate_to_shift_scalar(x_um, y_um, theta_deg, pixel_size_um):
    theta = math.radians(theta_deg)
    dx = (x_um * math.cos(theta) + y_um * math.sin(theta)) / pixel_size_um
    dy = (-x_um * math.sin(theta) + y_um * math.cos(theta)) / pixel_size_um
    return dx, dy

def main():
    root = project_root()
    alignment_csv = default_contour_alignment_csv(project_root_path=root)
    stage_config_json = root / "configs" / "stage_calibration.json"
    
    if not alignment_csv.exists():
        raise FileNotFoundError(alignment_csv)
    if not stage_config_json.exists():
        raise FileNotFoundError(stage_config_json)
        
    with open(stage_config_json, encoding="utf-8") as f:
        stage_config = json.load(f)
        
    theta_deg = float(stage_config["theta_deg"])
    pixel_size_um = float(stage_config["pixel_size_um"])
    
    # Read rows from alignment CSV
    rows = []
    ref_file = None
    with open(alignment_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['success'].lower() in ('true', '1', 'yes'):
                rows.append(row)
                if not ref_file and row.get('reference_file'):
                    ref_file = row['reference_file']

    if not ref_file:
        # Fallback if reference_file column is empty in some rows
        for row in rows:
            if float(row['refined_align_dx_px']) == 0.0 and float(row['refined_align_dy_px']) == 0.0:
                ref_file = row['file']
                break
        if not ref_file:
            ref_file = "10_16_0.txt" # typical reference

    # Find reference row coordinates
    ref_row = None
    for row in rows:
        if row['file'] == ref_file:
            ref_row = row
            break
    if not ref_row:
        ref_row = rows[0] # fallback

    ref_X = float(ref_row['X'])
    ref_Y = float(ref_row['Y'])

    # Compute residuals
    processed_rows = []
    res_norms = []
    res_norms_r0 = []
    res_norms_r1 = []
    res_norms_r2 = []

    for row in rows:
        X = float(row['X'])
        Y = float(row['Y'])
        R = int(row['R'])
        
        dx_s, dy_s = coordinate_to_shift_scalar(
            X - ref_X,
            Y - ref_Y,
            theta_deg,
            pixel_size_um
        )
        
        # Predicted shift based on run_ep05_edge_line_overlay.py logic
        pred_dx = -dx_s
        pred_dy = dy_s
        
        ref_dx = float(row['refined_align_dx_px'])
        ref_dy = float(row['refined_align_dy_px'])
        
        res_dx = ref_dx - pred_dx
        res_dy = ref_dy - pred_dy
        res_norm = math.hypot(res_dx, res_dy)
        res_norms.append(res_norm)
        
        if R == 0:
            res_norms_r0.append(res_norm)
        elif R == 1:
            res_norms_r1.append(res_norm)
        elif R == 2:
            res_norms_r2.append(res_norm)
            
        processed_rows.append({
            'file': row['file'],
            'acquisition_order': int(row['acquisition_order']),
            'X': X,
            'Y': Y,
            'R': R,
            'refined_align_dx_px': ref_dx,
            'refined_align_dy_px': ref_dy,
            'stage_pred_dx': pred_dx,
            'stage_pred_dy': pred_dy,
            'res_norm': res_norm
        })

    # Stats helper
    def stats(arr):
        if not arr:
            return "N/A"
        arr_sorted = sorted(arr)
        n = len(arr)
        med = arr_sorted[n // 2]
        p90 = arr_sorted[int(n * 0.9)] if int(n * 0.9) < n else arr_sorted[-1]
        mean = sum(arr) / n
        return f"n={n}, median={med:.6f} px, p90={p90:.6f} px, max={arr_sorted[-1]:.6f} px"

    print("Stage Prior Model Residuals (Relative to Contour-Refined Shifts):")
    print(f"Overall:   {stats(res_norms)}")
    print(f"Repeat R=0: {stats(res_norms_r0)}")
    print(f"Repeat R=1: {stats(res_norms_r1)}")
    print(f"Repeat R=2: {stats(res_norms_r2)}")
    
    # Sort and get top 20 outliers
    processed_rows.sort(key=lambda x: x['res_norm'], reverse=True)
    
    print("\nTop 20 Outliers for Stage Prior Model:")
    print("{:<12} {:<10} {:<4} {:<4} {:<2} {:<12} {:<12} {:<12} {:<12} {:<12}".format(
        "file", "acq_order", "X", "Y", "R", "refined_dx", "refined_dy", "pred_dx", "pred_dy", "res_norm"
    ))
    for r in processed_rows[:20]:
        print("{:<12} {:<10d} {:<4g} {:<4g} {:<2d} {:<12.6f} {:<12.6f} {:<12.6f} {:<12.6f} {:<12.6f}".format(
            r['file'], r['acquisition_order'], r['X'], r['Y'], r['R'],
            r['refined_align_dx_px'], r['refined_align_dy_px'],
            r['stage_pred_dx'], r['stage_pred_dy'], r['res_norm']
        ))

if __name__ == "__main__":
    main()
