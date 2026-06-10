# %% [markdown]
# ## Temperature Map Comparison
#
# 前面所有 highpass 图展示的是减去低频背景后的结构响应。本节把低频背景加回 TGV SR highpass 输出，还原为温度图（°C），与 bicubic baseline 温度图直接对比。
#
# 温度还原方法：取 offset-corrected 原始参考帧 → bicubic 2x 上采样 → Gaussian 低通提取低频背景 → 加上 TGV highpass HR。

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display, Markdown
from scipy import ndimage

from thermal_core.plotting import COLORMAPS, FIGURE_SIZES, savefig_academic, setup_academic_style

setup_academic_style()

tgv_dir = EP10_DIRS.get("tgv", PROJECT_ROOT / "output" / "ep10_tgv_sr")
tgv_temp_path = tgv_dir / "best_hr_temperature.npy"
tgv_hp_path = tgv_dir / "best_hr_highpass.npy"

# Load or compute temperature map
if tgv_temp_path.exists():
    tgv_temp = np.load(tgv_temp_path)
    print(f"Loaded TGV temperature map: {tgv_temp.shape}, range [{tgv_temp.min():.2f}, {tgv_temp.max():.2f}] °C")
else:
    print(f"Missing: {tgv_temp_path}")
    print("Run: algos/ep10_tgv_sr/scripts/run_tgv_quick.py to generate temperature output")
    tgv_temp = None

# %%
if tgv_temp is not None:
    # Build bicubic baseline temperature from cached inputs
    cache_dir = tgv_dir / "cache"
    ref_hp_path = cache_dir / "ref_hp_hr.npy"
    hp_frames_path = cache_dir / "hp_frames.npy"

    # We need the raw reference HR for the bicubic baseline.
    # Reconstruct from ref_hp_hr + low-freq background, or load from raw data.
    # Simplest: load one raw frame, bicubic upsample.
    try:
        from common.data_loader import load_main_session_frames, bicubic_upsample, offset_correction

        DATA_DIR = PROJECT_ROOT / "data" / "data_raw" / "infrared_avi"
        AUDIT_CSV = PROJECT_ROOT / "output" / "ep01_data_processing" / "frame_audit.csv"
        raw_frames, _meta = load_main_session_frames(DATA_DIR, AUDIT_CSV, workers=8, dtype=np.float32)
        raw_frames = offset_correction(raw_frames)
        ref_lr = raw_frames[len(raw_frames) // 2]
        bicubic_hr = bicubic_upsample(ref_lr, scale=2).astype(np.float64)
        has_bicubic = True
    except Exception as exc:
        print(f"Could not load raw frames for bicubic baseline: {exc}")
        has_bicubic = False

    if has_bicubic:
        def center_crop(img, fraction=1.0 / 3.0):
            h, w = img.shape
            ch = max(1, int(round(h * fraction)))
            cw = max(1, int(round(w * fraction)))
            y0 = (h - ch) // 2
            x0 = (w - cw) // 2
            return img[y0: y0 + ch, x0: x0 + cw]

        crops_bic = center_crop(bicubic_hr)
        crops_tgv = center_crop(tgv_temp.astype(np.float64))
        crops_diff = crops_tgv - crops_bic

        all_vals = np.concatenate([crops_bic.ravel(), crops_tgv.ravel()])
        vmin = float(np.nanpercentile(all_vals, 1))
        vmax = float(np.nanpercentile(all_vals, 99))

        fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZES["double_col"], constrained_layout=True)

        im0 = axes[0].imshow(crops_bic, cmap=COLORMAPS["temperature"], vmin=vmin, vmax=vmax, interpolation="nearest")
        axes[0].set_title("Bicubic baseline (°C)", fontsize=8.5, pad=2)

        axes[1].imshow(crops_tgv, cmap=COLORMAPS["temperature"], vmin=vmin, vmax=vmax, interpolation="nearest")
        axes[1].set_title("TGV aniso+cov SR (°C)", fontsize=8.5, pad=2)

        dlim = float(np.nanpercentile(np.abs(crops_diff), 99))
        axes[2].imshow(crops_diff, cmap="RdBu_r", vmin=-dlim, vmax=dlim, interpolation="nearest")
        axes[2].set_title("TGV − Bicubic (°C)", fontsize=8.5, pad=2)

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(
            "EP10 Temperature Map: TGV aniso+cov vs Bicubic (center 1/3 crop)",
            fontsize=10, fontweight="bold",
        )
        temp_fig_path = OUTPUT_DIR / "temperature_comparison.png"
        savefig_academic(fig, temp_fig_path)
        from IPython.display import Image as NotebookImage
        display(NotebookImage(filename=str(temp_fig_path), retina=True))
    else:
        display(Markdown("**Bicubic baseline unavailable** — cannot generate temperature comparison."))

# %% [markdown]
# > **图表说明**: 左图是 offset-corrected 原始参考帧的 bicubic 2x 上采样温度图；中图是 TGV aniso+cov SR 重建的温度图（highpass + 低频背景）；右图是两者差值。
# >
# > **怎么看**: 温度图直接显示芯片表面的绝对温度分布（°C）。与 highpass 图不同，这里保留了低频热场渐变。SR 改善体现在中图的结构边缘是否比左图更锐、轮廓是否更连续。右图差值中的红/蓝表示 SR 相对 bicubic 增强/抑制的局部温度，边缘处应有对称的增强信号。
# >
# > **核心发现**: 温度图验证 TGV SR 在还原温度域后仍保持结构增强效果，且未引入全局温度偏移或大面积伪影。
