#!/usr/bin/env python3
"""Reverse-engineer true stage coordinates of repeat frames using drift-decomposed contour shifts.

用 R=0 帧拟合线性热漂移模型，再从 R=2 补采帧的实测 contour 对齐位移中扣除漂移，
反推其真实电动台坐标（诊断 R=2 帧的名义坐标是否可信）。

用法（项目根目录，无 CLI 参数）::

    uv run python scripts/solve_true_coordinates.py

输入依赖: EP05 contour-refined 对齐 CSV（由 configs/alignment/paths.json 解析）、
    configs/stage_calibration.json
输出: 仅 stdout 打印（漂移模型系数 + R=2 帧真实坐标表），不写文件

关联: EP05
"""

import csv
import json
import math
from pathlib import Path

from thermal_core.alignment_paths import default_contour_alignment_csv

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

def shift_to_coordinate_scalar(dx, dy, theta_deg, pixel_size_um):
    theta = math.radians(theta_deg)
    # Inverse rotation matrix
    x_um = pixel_size_um * (dx * math.cos(theta) - dy * math.sin(theta))
    y_um = pixel_size_um * (dx * math.sin(theta) + dy * math.cos(theta))
    return x_um, y_um

def fit_linear_drift(t_list, val_list):
    # Fit y = m * t + c
    n = len(t_list)
    sum_t = sum(t_list)
    sum_val = sum(val_list)
    sum_t2 = sum(t**2 for t in t_list)
    sum_tval = sum(t*val for t, val in zip(t_list, val_list))
    
    denom = (n * sum_t2 - sum_t**2)
    if abs(denom) < 1e-9:
        return 0.0, sum_val / n
    m = (n * sum_tval - sum_t * sum_val) / denom
    c = (sum_val * sum_t2 - sum_t * sum_tval) / denom
    return m, c

def main():
    root = project_root()
    alignment_csv = default_contour_alignment_csv(project_root_path=root)
    stage_config_json = root / "configs" / "stage_calibration.json"
    
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
        for row in rows:
            if float(row['refined_align_dx_px']) == 0.0 and float(row['refined_align_dy_px']) == 0.0:
                ref_file = row['file']
                break

    ref_row = next(row for row in rows if row['file'] == ref_file)
    ref_X = float(ref_row['X'])
    ref_Y = float(ref_row['Y'])

    # Analyze R=0 frames to fit the time-dependent drift
    t_r0 = []
    drift_dx = []
    drift_dy = []
    
    for row in rows:
        if int(row['R']) == 0:
            X = float(row['X'])
            Y = float(row['Y'])
            t = int(row['acquisition_order'])
            
            # Theoretical shift (stage prior)
            dx_prior, dy_prior = coordinate_to_shift_scalar(
                X - ref_X,
                Y - ref_Y,
                theta_deg,
                pixel_size_um
            )
            pred_dx = -dx_prior
            pred_dy = dy_prior
            
            # Actual shift
            act_dx = float(row['refined_align_dx_px'])
            act_dy = float(row['refined_align_dy_px'])
            
            # Drift = Actual - Theoretical
            drift_x = act_dx - pred_dx
            drift_y = act_dy - pred_dy
            
            t_r0.append(t)
            drift_dx.append(drift_x)
            drift_dy.append(drift_y)

    # Fit linear drift over time
    m_dx, c_dx = fit_linear_drift(t_r0, drift_dx)
    m_dy, c_dy = fit_linear_drift(t_r0, drift_dy)
    
    print("Fitted linear thermal drift model (per acquisition step):")
    print(f"  Drift dx(t) = {m_dx:+.6f} * t + {c_dx:+.6f} px")
    print(f"  Drift dy(t) = {m_dy:+.6f} * t + {c_dy:+.6f} px")
    
    # Let's inspect the 6 repeat frames R=2
    print("\nReverse-engineering true coordinates for R=2 frames:")
    print("{:<12} {:<10} {:<6} {:<6} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}".format(
        "file", "acq_order", "nom_X", "nom_Y", "act_dx", "act_dy", "drift_dx", "drift_dy", "true_X", "true_Y"
    ))
    
    for row in rows:
        if int(row['R']) == 2:
            nom_X = float(row['X'])
            nom_Y = float(row['Y'])
            t = int(row['acquisition_order'])
            
            # Measured shifts
            act_dx = float(row['refined_align_dx_px'])
            act_dy = float(row['refined_align_dy_px'])
            
            # Extrapolate drift at time t
            pred_drift_x = m_dx * t + c_dx
            pred_drift_y = m_dy * t + c_dy
            
            # Spatial shift = Measured - Drift
            spatial_dx = act_dx - pred_drift_x
            spatial_dy = act_dy - pred_drift_y
            
            # Map spatial shift back to relative coordinates (dx = -dx_s, dy = dy_s)
            # So dx_s = -spatial_dx, dy_s = spatial_dy
            dx_s = -spatial_dx
            dy_s = spatial_dy
            
            rel_X, rel_Y = shift_to_coordinate_scalar(dx_s, dy_s, theta_deg, pixel_size_um)
            
            # True absolute coordinates
            true_X = rel_X + ref_X
            true_Y = rel_Y + ref_Y
            
            print("{:<12} {:<10d} {:<6g} {:<6g} {:<12.6f} {:<12.6f} {:<12.6f} {:<12.6f} {:<12.6f} {:<12.6f}".format(
                row['file'], t, nom_X, nom_Y, act_dx, act_dy, pred_drift_x, pred_drift_y, true_X, true_Y
            ))

if __name__ == "__main__":
    main()
