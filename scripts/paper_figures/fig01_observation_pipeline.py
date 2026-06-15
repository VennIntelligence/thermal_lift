#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# Use non-interactive backend
os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "tcforge" / "src"))

import tcforge
from tcforge.physics import render_temperature_field

from thermal_core.plotting import setup_academic_style, savefig_academic, METHOD_COLORS

OUTPUT_DIR = ROOT / "output" / "paper_figures"
PNG_PATH = OUTPUT_DIR / "fig01_observation_pipeline.png"
PDF_PATH = OUTPUT_DIR / "fig01_observation_pipeline.pdf"
SVG_PATH = OUTPUT_DIR / "fig01_observation_pipeline.svg"

def main():
    setup_academic_style()
    np.random.seed(42)
    
    # 1. Generate TCForge Latent Field
    mask = tcforge.build_multi_temp_mask("hard", 42)
    # Ensure temperature_offsets_c matches the unique labels in the mask
    max_label = mask.max()
    offsets = np.linspace(0.0, 15.0, max_label + 1).tolist()
    x_full = render_temperature_field(mask, t_bg_c=25.0, temperature_offsets_c=offsets, low_freq_amplitude_c=1.0, seed=42)
    
    # 2. Parameters for extremely obvious degradation (as requested)
    dx, dy = 60.5, 40.5    # Massive shift (HR pixels)
    sigma = 8.0            # Huge blur
    box_size = 8           # 8x8 HR pixels per LR pixel (massive pixelation)
    factor = 8
    noise_sigma = 1.2      # Huge noise
    
    # 3. Apply Pipeline to Full Image
    Sx_full = ndi.shift(x_full, (dy, dx), order=1)
    HSx_full = ndi.gaussian_filter(Sx_full, sigma)
    BHSx_full = ndi.uniform_filter(HSx_full, size=box_size)
    
    h_f, w_f = BHSx_full.shape
    DBHSx_full = BHSx_full.reshape(h_f//factor, factor, w_f//factor, factor).mean(axis=(1, 3))
    y_obs_full = DBHSx_full + np.random.randn(*DBHSx_full.shape) * noise_sigma
    
    # 4. Crop a fixed window to show the effects (16:9 aspect ratio)
    hr_h = 360
    hr_w = 640
    
    # Choose a centered interesting spot
    cy = h_f // 2 - hr_h // 2
    cx = w_f // 2 - hr_w // 2
    
    # Ensure bounds
    cy = max(0, min(cy, h_f - hr_h))
    cx = max(0, min(cx, w_f - hr_w))
    
    lr_h = hr_h // factor
    lr_w = hr_w // factor
    cy_lr = cy // factor
    cx_lr = cx // factor
    
    x_crop = x_full[cy:cy+hr_h, cx:cx+hr_w]
    Sx_crop = Sx_full[cy:cy+hr_h, cx:cx+hr_w]
    HSx_crop = HSx_full[cy:cy+hr_h, cx:cx+hr_w]
    BHSx_crop = BHSx_full[cy:cy+hr_h, cx:cx+hr_w]
    DBHSx_crop = DBHSx_full[cy_lr:cy_lr+lr_h, cx_lr:cx_lr+lr_w]
    y_obs_crop = y_obs_full[cy_lr:cy_lr+lr_h, cx_lr:cx_lr+lr_w]
    
    images = [x_crop, Sx_crop, HSx_crop, BHSx_crop, DBHSx_crop, y_obs_crop]
    titles = [
        r"$x$" + "\nLatent Field",
        r"$S_{t_k} x$" + "\nShifted",
        r"$H S_{t_k} x$" + "\nBlurred",
        r"$B H S_{t_k} x$" + "\nBox Integrated",
        r"$D B H S_{t_k} x$" + "\nDownsampled",
        r"$y_k$" + "\nNoisy Obs."
    ]
    
    operators = [
        ("$S_{t_k}$", "Stage Shift"),
        ("$H$", "Optical PSF Blur"),
        ("$B$", "Detector Aperture"),
        ("$D$", "Sensor Sampling"),
        ("$+ n_k$", "Readout Noise")
    ]
    
    fig = plt.figure(figsize=(7.2, 4.05)) # 16:9 overall figure
    fig.patch.set_facecolor('white')
    
    # Layout dimensions for 16:9 subplots
    w_ax = 0.26
    h_ax = 0.26 # By making them equal, physical ratio inherits figure ratio 16:9
    w_space = 0.08
    h_space = 0.15
    left_margin = 0.04
    top_margin = 0.10
    
    col_x = [left_margin + j*(w_ax + w_space) for j in range(3)]
    row_y = [1.0 - top_margin - h_ax, 1.0 - top_margin - 2*h_ax - h_space]
    
    # Snake layout
    positions = [
        (col_x[0], row_y[0]), # 0: x
        (col_x[1], row_y[0]), # 1: Sx
        (col_x[2], row_y[0]), # 2: HSx
        (col_x[2], row_y[1]), # 3: BHSx
        (col_x[1], row_y[1]), # 4: DBHSx
        (col_x[0], row_y[1]), # 5: y
    ]
    
    axes = []
    for (px, py) in positions:
        ax = fig.add_axes([px, py, w_ax, h_ax])
        axes.append(ax)
        
    for i, (ax, img, title) in enumerate(zip(axes, images, titles)):
        # Calculate dynamic range for this patch to look good
        vmin, vmax = np.percentile(img, [1, 99])
        im = ax.imshow(img, cmap="inferno", origin="upper", vmin=22.0, vmax=42.0)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('#555555')
            spine.set_linewidth(0.8)
            
        ax.text(0.5, 1.05, title, transform=ax.transAxes, ha='center', va='bottom', fontsize=9, color='#111111', linespacing=1.2)
        
    arrow_color = '#000000' # Black arrows
    
    def draw_arrow(start, end, op_text, src_text, orientation):
        arrow = FancyArrowPatch(start, end, transform=fig.transFigure,
                                arrowstyle='-|>', mutation_scale=12,
                                linewidth=1.5, color=arrow_color)
        fig.patches.append(arrow)
        
        mid_x = (start[0] + end[0]) / 2.0
        mid_y = (start[1] + end[1]) / 2.0
        
        # Increase text size slightly
        if orientation == 'H':
            fig.text(mid_x, mid_y + 0.015, op_text, ha='center', va='bottom', fontsize=11, color=arrow_color, fontweight='bold')
            fig.text(mid_x, mid_y - 0.015, src_text, ha='center', va='top', fontsize=8, color='#333333')
        elif orientation == 'V':
            fig.text(mid_x - 0.015, mid_y, op_text, ha='right', va='center', fontsize=11, color=arrow_color, fontweight='bold')
            fig.text(mid_x + 0.015, mid_y, src_text, ha='left', va='center', fontsize=8, color='#333333')
            
    # 0 -> 1
    draw_arrow((col_x[0] + w_ax + 0.005, row_y[0] + h_ax/2), 
               (col_x[1] - 0.005, row_y[0] + h_ax/2), 
               operators[0][0], operators[0][1], 'H')
               
    # 1 -> 2
    draw_arrow((col_x[1] + w_ax + 0.005, row_y[0] + h_ax/2), 
               (col_x[2] - 0.005, row_y[0] + h_ax/2), 
               operators[1][0], operators[1][1], 'H')
               
    # 2 -> 3 (Down)
    draw_arrow((col_x[2] + w_ax/2, row_y[0] - 0.005), 
               (col_x[2] + w_ax/2, row_y[1] + h_ax + 0.005), 
               operators[2][0], operators[2][1], 'V')
               
    # 3 -> 4 (Left)
    draw_arrow((col_x[2] - 0.005, row_y[1] + h_ax/2), 
               (col_x[1] + w_ax + 0.005, row_y[1] + h_ax/2), 
               operators[3][0], operators[3][1], 'H')
               
    # 4 -> 5 (Left)
    draw_arrow((col_x[1] - 0.005, row_y[1] + h_ax/2), 
               (col_x[0] + w_ax + 0.005, row_y[1] + h_ax/2), 
               operators[4][0], operators[4][1], 'H')

    # Main equation at the bottom (no box), text slightly larger
    eq_text = r"Observation Model: $\quad y_k \;=\; D\,B\,H\,S_{t_k}\,x \,+\, n_k$"
    fig.text(0.5, 0.06, eq_text, ha='center', va='center', fontsize=14, color='#111111')
             
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    savefig_academic(fig, PNG_PATH, dpi=300, close=False)
    savefig_academic(fig, PDF_PATH, dpi=300, close=False)
    savefig_academic(fig, SVG_PATH, dpi=300, close=True)
    
    print(f"Saved figure to {PNG_PATH}")
    print(f"Saved figure to {SVG_PATH}")

if __name__ == "__main__":
    main()
