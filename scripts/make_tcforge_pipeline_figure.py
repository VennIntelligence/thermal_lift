"""Generate a TCForge physics-matched synthesis pipeline figure (paper §4.3).

Renders authentic thumbnails directly from the tcforge library so the panel
faithfully shows the four synthesis stages described in the method section:

    1. Scene geometry   — 8-layer IC primitives, rotate theta=47.6 +/-1.5deg,
                          4x SSAA -> coverage in [0, 1]
    2. Temperature field — T = T_bg + dT * coverage + drift noise
    3. Forward degradation — Gaussian PSF -> block-average down x2,
                          sub-pixel shift burst (measured 248-frame profile)
    4. LR burst -> training pair — 1000 scenes, hybrid K=4 drizzle variants

The output is meant as a self-contained "material" panel to be cropped /
re-assembled into the hand-drawn main figure.

Run:
    uv run python scripts/make_tcforge_pipeline_figure.py

输入: 无外部数据，仅 tcforge/src 合成引擎 + core/src 学术绘图样式
输出: paper/figures/fig_tcforge_pipeline.png / .pdf / .svg
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tcforge" / "src"))
sys.path.insert(0, str(ROOT / "core" / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from thermal_core.plotting import setup_academic_style  # noqa: E402

import tcforge as tf  # noqa: E402

# ── physics constants (must mirror paper §4.3) ───────────────────────────────
THETA_DEG = 47.6
THETA_JITTER = 1.5
DELTA_T_C = 3.0          # representative draw from U(0.5, 5.0)
PSF_SIGMA_LR = 0.40      # representative draw from U(0.15, 0.55)
NOISE_SIGMA_C = 0.0724
SCALE = 2
SEED = 7

# Modest canvas keeps SSAA rotation fast while staying faithful (HR 480x640).
HR_SHAPE = (480, 640)
N_BURST = 24

OUT_DIR = ROOT / "paper" / "figures"
STEM = "fig_tcforge_pipeline"


def build_assets() -> dict[str, np.ndarray]:
    """Run the real forward pipeline once and collect display thumbnails."""

    coverage = tf.build_scene_mask(
        "medium",
        SEED,
        rotation_deg_center=THETA_DEG,
        rotation_jitter_deg=THETA_JITTER,
        canvas_shape=HR_SHAPE,
        scale=SCALE,
        antialias=True,
        ssaa_factor=4,
    )

    temp_hr = tf.render_temperature_field(
        coverage,
        t_bg_c=21.0,
        delta_t_c=DELTA_T_C,
        low_freq_amplitude_c=0.25,
        seed=SEED,
    )

    # Sub-pixel shift profile: prefer the measured contour-refined profile,
    # fall back to an ideal phase grid + jitter if the CSV is unavailable.
    try:
        shifts, _ = tf.load_shift_profile(
            "real_default_contour_refined", n_frames=N_BURST
        )
    except Exception:
        shifts = tf.ideal_phase_grid(
            n_frames=N_BURST, scale=SCALE, jitter_std_px=0.05, seed=SEED
        )

    burst = tf.generate_lr_burst(
        temp_hr,
        shifts,
        forward_mode="physical_block_average",
        psf_sigma_lr_px=PSF_SIGMA_LR,
        scale=SCALE,
    )
    burst = tf.add_noise(burst, noise_sigma_c=NOISE_SIGMA_C, seed=SEED)

    psf = tf.make_psf_kernel(psf_sigma_lr_px=PSF_SIGMA_LR, scale=SCALE)

    return {
        "coverage": coverage,
        "temp_hr": temp_hr,
        "lr": burst[0],
        "burst": burst,
        "psf": psf,
        "shifts": np.asarray(shifts),
    }


CROP_FRAC = 0.58  # show a zoomed-in central window, not the whole canvas


def crop_center(img: np.ndarray, frac: float = CROP_FRAC) -> np.ndarray:
    """Return the central ``frac`` window of a 2D image (zoom-in crop)."""
    h, w = img.shape
    ch, cw = max(1, int(round(h * frac))), max(1, int(round(w * frac)))
    r0, c0 = (h - ch) // 2, (w - cw) // 2
    return img[r0:r0 + ch, c0:c0 + cw]


def _imshow(ax, data, cmap, *, frame_color="#222222", lw=1.0):
    ax.imshow(data, cmap=cmap, aspect="equal", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(frame_color)
        spine.set_linewidth(lw)


def _arrow(fig, x0, x1, y):
    fig.patches.append(
        FancyArrowPatch(
            (x0, y),
            (x1, y),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=15,
            lw=1.7,
            color="#1a1a1a",
            zorder=5,
        )
    )


def make_figure(assets: dict[str, np.ndarray]) -> plt.Figure:
    setup_academic_style()
    plt.rcParams["figure.constrained_layout.use"] = False

    fig_w, fig_h = 11.6, 4.9
    fig = plt.figure(figsize=(fig_w, fig_h))
    sq = fig_w / fig_h  # multiply a width fraction to get an equal-display height

    # Column layout (figure coords) — packed tight so arrows stay short.
    panel_w, panel_h = 0.185, 0.42
    gap = 0.045
    panel_y = 0.34
    span = 4 * panel_w + 3 * gap
    left0 = (1.0 - span) / 2.0
    lefts = [left0 + i * (panel_w + gap) for i in range(4)]
    centers = [lf + panel_w / 2 for lf in lefts]

    titles = [
        "1. Scene geometry",
        "2. Temperature field",
        "3. Forward degradation",
        "4. LR burst \u2192 training pair",
    ]
    captions = [
        "8-layer IC primitives\n"
        r"rotate $\theta=47.6\degree\!\pm\!1.5\degree$" "\n"
        r"4$\times$ SSAA $\to$ coverage $\in[0,1]$",
        r"$T = T_\mathrm{bg} + \Delta T\cdot\mathrm{cov} + \mathrm{drift}$" "\n"
        r"$\Delta T\sim U(0.5,5.0)\,\degree$C" "\n"
        "smooth low-freq drift",
        r"Gaussian PSF $\sigma\sim U(0.15,0.55)$" "\n"
        r"$\to$ block-avg downsample ($\times2$)" "\n"
        "sub-pixel shifts (248 profile)\n"
        r"$+\,0.02$px jitter, noise $0.0724\,\degree$C",
        r"1000 scenes  $480\!\times\!640 \to 960\!\times\!1280$" "\n"
        r"hybrid input: $K{=}4$ drizzle variants" "\n"
        "(LR burst, shifts) paired to HR",
    ]
    cmaps = ["viridis", "inferno", "inferno", "inferno"]
    datas = [crop_center(assets["coverage"]), crop_center(assets["temp_hr"]),
             crop_center(assets["lr"]), None]

    axes = []
    for i, cx in enumerate(centers):
        if i < 3:
            ax = fig.add_axes([cx - panel_w / 2, panel_y, panel_w, panel_h])
            if i in (1, 2):
                d = datas[i]
                vmin, vmax = np.percentile(d, [1, 99])
                ax.imshow(d, cmap=cmaps[i], aspect="equal",
                          interpolation="nearest", vmin=vmin, vmax=vmax)
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor("#222222")
                    spine.set_linewidth(1.0)
            else:
                _imshow(ax, datas[i], cmaps[i])
            axes.append(ax)
        else:
            # Stage 4: stacked burst montage to convey "burst of frames".
            burst = assets["burst"]
            offs = [(-0.018, 0.030), (0.0, 0.0), (0.018, -0.030)]
            idxs = [min(len(burst) - 1, 6), 0, min(len(burst) - 1, 12)]
            for (ox, oy), fi in zip(offs, idxs):
                axb = fig.add_axes(
                    [cx - panel_w / 2 + ox, panel_y + oy, panel_w, panel_h]
                )
                _imshow(axb, crop_center(burst[fi]), "inferno",
                        frame_color="#444444", lw=0.9)
            axes.append(axb)

    # PSF kernel inset on stage 3 (upper-left corner of the panel).
    cx3 = centers[2]
    psf_w = 0.060
    ax_psf = fig.add_axes(
        [cx3 - panel_w / 2 + 0.006,
         panel_y + panel_h - psf_w * sq - 0.020,
         psf_w, psf_w * sq]
    )
    ax_psf.imshow(assets["psf"], cmap="magma", interpolation="nearest")
    ax_psf.set_xticks([])
    ax_psf.set_yticks([])
    for spine in ax_psf.spines.values():
        spine.set_edgecolor("white")
        spine.set_linewidth(0.8)
    ax_psf.set_title("PSF", fontsize=7.5, pad=1.5, color="white")

    # Sub-pixel shift scatter inset on stage 3 (lower-left corner).
    sh_w = 0.066
    ax_sh = fig.add_axes(
        [cx3 - panel_w / 2 + 0.006, panel_y + 0.012, sh_w, sh_w * sq]
    )
    sh = assets["shifts"]
    ax_sh.scatter(sh[:, 0], sh[:, 1], s=7, c="#55A868", edgecolors="none", alpha=0.9)
    ax_sh.set_xticks([])
    ax_sh.set_yticks([])
    ax_sh.set_facecolor("white")
    for spine in ax_sh.spines.values():
        spine.set_edgecolor("#888888")
        spine.set_linewidth(0.6)
    ax_sh.set_title("sub-px shifts", fontsize=7.0, pad=1.5)

    # Stage titles + captions.
    for cx, title, cap in zip(centers, titles, captions):
        fig.text(cx, panel_y + panel_h + 0.018, title, ha="center", va="bottom",
                 fontsize=10.5, fontweight="bold", color="#1a1a1a")
        fig.text(cx, panel_y - 0.030, cap, ha="center", va="top",
                 fontsize=9.2, color="#333333", linespacing=1.45)

    # Arrows between stages (short, packed between panel edges).
    for cx_l, cx_r in zip(centers[:-1], centers[1:]):
        _arrow(fig, cx_l + panel_w / 2 + 0.004, cx_r - panel_w / 2 - 0.004,
               panel_y + panel_h / 2)

    # Header band.
    fig.text(0.5, 0.975,
             "TCForge: physics-matched synthesis from measured forward model",
             ha="center", va="top", fontsize=12.5, fontweight="bold",
             color="#1a1a1a")
    fig.text(0.5, 0.930,
             "synthetic HR ground truth  \u2192  calibrated degradation  "
             "\u2192  realistic LR training bursts",
             ha="center", va="top", fontsize=8.5, style="italic", color="#555555")

    return fig


def main() -> None:
    assets = build_assets()
    fig = make_figure(assets)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{STEM}.png"
    pdf = OUT_DIR / f"{STEM}.pdf"
    svg = OUT_DIR / f"{STEM}.svg"
    fig.savefig(png, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(svg, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"saved: {png}")
    print(f"saved: {pdf}")
    print(f"saved: {svg}")


if __name__ == "__main__":
    main()
