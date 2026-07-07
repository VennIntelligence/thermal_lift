"""Real detector noise audit from real IR burst frames (flat-region statistics).

Answers research_log/ood_generalization_suite_design.md §4 ("A2 噪声族的物理参数最好来自真实 burst
的噪声实测(平坦区统计)——待排一个小分析任务"): measure column/row stripe FPN amplitude, low-freq
spatial spectral slope, temporal noise level + lag-1 correlation, and hot/dead pixel prevalence
from the 248-frame real SR-acquisition burst, restricted to flat (structure-free) image regions.
Read-only w.r.t. the raw data; writes only under output/real_noise_audit/.

Burst selection: frame_audit.csv rows with frame_role=="sr_default" & is_sr_usable=="True"
(248 frames, single clean acquisition session). These are NOT a fixed-scene temporal burst —
X/Y are sub-pixel stage shifts (<= detector pitch) for multi-frame SR — so all measurements are
confined to flat (no visible scene structure) regions, which are insensitive to those small shifts.
"Temporal order" for the lag-1 autocorrelation and the pixel-mean/residual split uses the CSV's
acquisition_order (recovered from file mtime), not filename order.

Usage:
    cd /home/ujs/thermal_lift && uv run python scripts/audit_real_noise.py

Outputs:
    output/real_noise_audit/summary.json   - all measured numbers (see docstring items below)
    output/real_noise_audit/diagnostics.png - mean image / flat mask / col&row profiles / PSD fit

Units: the raw .txt frames are already calibrated temperature in degrees C (per frame_audit.csv
T_min/T_max/T_mean columns, ~18-24 C range) -- NOT raw sensor DN. All amplitudes below are in C.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

# §5.0 sink: the flat-region estimators now live in tcforge._noise_stats so the synthetic
# audit and the realism tests measure with byte-for-byte the same code. Import them here
# (tcforge/src is added to sys.path the same way the other repo scripts do).
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TCFORGE_SRC = _PROJECT_ROOT / "tcforge" / "src"
if _TCFORGE_SRC.exists() and str(_TCFORGE_SRC) not in sys.path:
    sys.path.insert(0, str(_TCFORGE_SRC))
from tcforge._noise_stats import autocorr_1e_length, radial_psd_slope  # noqa: E402

RAW_DIR = Path("data/data_raw/infrared_avi")
FRAME_AUDIT_CSV = Path("output/ep01_data_processing/frame_audit.csv")
OUT_DIR = Path("output/real_noise_audit")

BORDER_PX = 10          # exclude this many px from every frame edge
BG_SIGMA_PX = 25.0      # gaussian smoothing scale for the "smooth background" (item 1/2)
LOCAL_ROUGHNESS_WIN_PX = 9  # window for the local-std "roughness" measure used by the flat mask
EROSION_RADIUS_PX = 2   # shrink the raw (roughness < median) flat mask by this much
SPECTRAL_CROP = 160     # side length (px) of the crop used for the PSD slope fit (item 3)
HOT_Z_THRESH = 6.0      # |z| threshold for hot/dead pixel candidates (item 5)
HOT_MAX_CLUSTER = 2     # connected-component size <= this counts as "isolated" hot pixel


def _disk(rad: int) -> np.ndarray:
    yy, xx = np.ogrid[-rad:rad + 1, -rad:rad + 1]
    return (yy * yy + xx * xx) <= rad * rad


def load_burst_manifest() -> tuple[list[str], list[int], list[tuple[float, float]]]:
    rows = list(csv.DictReader(open(FRAME_AUDIT_CSV)))
    sel = [r for r in rows if r["frame_role"] == "sr_default" and r["is_sr_usable"] == "True"]
    sel.sort(key=lambda r: int(r["acquisition_order"]))
    files = [r["file"] for r in sel]
    acq_order = [int(r["acquisition_order"]) for r in sel]
    xy = [(float(r["X"]), float(r["Y"])) for r in sel]
    return files, acq_order, xy


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files, acq_order, xy = load_burst_manifest()
    n = len(files)
    xs, ys = [p[0] for p in xy], [p[1] for p in xy]
    print(f"[burst] frame_role==sr_default & is_sr_usable==True -> {n} frames")
    print(f"[burst] X range [{min(xs)}, {max(xs)}] um, Y range [{min(ys)}, {max(ys)}] um "
          f"(sub-pixel stage shifts, 20um pitch)")

    burst = np.empty((n, 480, 640), dtype=np.float32)
    for i, f in enumerate(files):
        burst[i] = np.loadtxt(RAW_DIR / f, dtype=np.float32)
    h, w = burst.shape[1], burst.shape[2]

    mean_img = burst.mean(axis=0)
    resid = burst - mean_img[None]  # per-frame temporal residual (filename order for now)

    # ---- flat-region mask ----
    # NOTE: a raw single-pixel central-difference gradient (the naive "local gradient" reading of
    # the spec) turned out to be dominated by a STATIC per-pixel fixed-pattern texture present
    # even in genuinely structure-free areas (see script docstring / final report: pixel-to-pixel
    # std in the smoothest patch found is ~0.05-0.09 C, ~7-10x the ~0.0074 C expected from pure
    # per-frame temporal noise averaged over 248 frames) -- that raw-gradient mask is close to
    # spatially-independent salt-and-pepper and collapses to ~1 surviving pixel under ANY erosion.
    # Using a windowed local-std "roughness" measure instead (which is itself already spatially
    # smoothed over the window) gives contiguous flat blobs that survive erosion.
    m1 = ndimage.uniform_filter(mean_img, LOCAL_ROUGHNESS_WIN_PX)
    m2 = ndimage.uniform_filter(mean_img.astype(np.float64) ** 2, LOCAL_ROUGHNESS_WIN_PX)
    local_roughness = np.sqrt(np.clip(m2 - m1 ** 2, 0, None))
    grad_thresh = float(np.median(local_roughness))
    flat_raw = local_roughness < grad_thresh
    flat_eroded = ndimage.binary_erosion(flat_raw, structure=_disk(EROSION_RADIUS_PX))
    border_mask = np.zeros_like(flat_eroded)
    border_mask[BORDER_PX:h - BORDER_PX, BORDER_PX:w - BORDER_PX] = True
    flat = flat_eroded & border_mask
    flat_frac = float(flat.mean())
    print(f"[flat-mask] local_roughness(win={LOCAL_ROUGHNESS_WIN_PX}px)_median_threshold={grad_thresh:.5f}, "
          f"flat_frac(post-erosion,excl border)={flat_frac:.4f}")

    # ---- items 1 & 2: column / row stripe FPN amplitude ----
    smooth_bg = ndimage.gaussian_filter(mean_img, BG_SIGMA_PX)
    bg_resid = mean_img - smooth_bg

    masked = np.where(flat, bg_resid, np.nan)
    with np.errstate(invalid="ignore"):
        col_profile = np.nanmean(masked, axis=0)
        row_profile = np.nanmean(masked, axis=1)
        col_count = np.sum(flat, axis=0)
        row_count = np.sum(flat, axis=1)
    col_valid = col_count > 0
    row_valid = row_count > 0
    col_amp = float(np.nanstd(col_profile[col_valid]))
    row_amp = float(np.nanstd(row_profile[row_valid]))
    row_over_col_ratio = row_amp / col_amp if col_amp > 0 else float("nan")

    # temporal-noise-only expected floor for these column/row means (item 2 significance check):
    # each mean_img pixel has temporal variance temporal_sigma^2 / n; averaging over k flat
    # pixels along a column/row reduces variance further by ~1/k (independent grain assumption).
    tstd_full = resid.std(axis=0, ddof=1)
    temporal_sigma_prelim = float(np.median(tstd_full[flat]))
    k_col_avg = float(np.mean(col_count[col_valid]))
    k_row_avg = float(np.mean(row_count[row_valid]))
    col_noise_floor = temporal_sigma_prelim / np.sqrt(n * max(k_col_avg, 1.0))
    row_noise_floor = temporal_sigma_prelim / np.sqrt(n * max(k_row_avg, 1.0))
    col_snr = col_amp / col_noise_floor if col_noise_floor > 0 else float("nan")
    row_snr = row_amp / row_noise_floor if row_noise_floor > 0 else float("nan")

    # ---- item 6: stripe correlation length (1/e) ----
    col_corr_len = autocorr_1e_length(col_profile, col_valid)
    row_corr_len = autocorr_1e_length(row_profile, row_valid)

    # ---- item 3: low-frequency spectral slope on largest flat crop ----
    step = 16
    best = (-1.0, BORDER_PX, BORDER_PX)
    for y0 in range(BORDER_PX, h - BORDER_PX - SPECTRAL_CROP, step):
        for x0 in range(BORDER_PX, w - BORDER_PX - SPECTRAL_CROP, step):
            frac = flat[y0:y0 + SPECTRAL_CROP, x0:x0 + SPECTRAL_CROP].mean()
            if frac > best[0]:
                best = (frac, y0, x0)
    crop_flat_frac, y0, x0 = best
    print(f"[spectral] crop origin=({y0},{x0}) size={SPECTRAL_CROP} flat_frac_in_crop={crop_flat_frac:.3f}")
    crop = mean_img[y0:y0 + SPECTRAL_CROP, x0:x0 + SPECTRAL_CROP].astype(np.float64)
    spec = radial_psd_slope(crop, SPECTRAL_CROP)
    print(f"[spectral] alpha={spec['alpha']:.3f} r2={spec['r2_loglog_fit']:.4f} over freq "
          f"[{spec['fit_freq_min_cyc_px']:.4f},{spec['fit_freq_max_cyc_px']:.4f}] cyc/px, "
          f"n_bins={spec['n_fit_bins']}")

    # ---- item 4: temporal noise level + lag-1 autocorrelation (acquisition-time order) ----
    temporal_sigma = temporal_sigma_prelim
    order_idx = np.argsort(acq_order)
    resid_sorted = resid[order_idx]
    r0 = resid_sorted[:-1]
    r1 = resid_sorted[1:]
    r0m = r0 - r0.mean(axis=0, keepdims=True)
    r1m = r1 - r1.mean(axis=0, keepdims=True)
    num = (r0m * r1m).sum(axis=0)
    den = np.sqrt((r0m ** 2).sum(axis=0) * (r1m ** 2).sum(axis=0)) + 1e-12
    lag1 = num / den
    lag1_median = float(np.median(lag1[flat]))
    print(f"[temporal] median flat-region temporal sigma={temporal_sigma:.5f} C, "
          f"median lag-1 autocorr (acquisition order)={lag1_median:.4f}")

    # ---- item 5: hot / dead pixel count ----
    med5 = ndimage.median_filter(mean_img, size=5)
    hp_resid = mean_img - med5
    mad = float(np.median(np.abs(hp_resid[flat] - np.median(hp_resid[flat]))))
    robust_sigma = 1.4826 * mad
    z = hp_resid / (robust_sigma + 1e-9)
    outlier = (np.abs(z) > HOT_Z_THRESH) & border_mask
    lbl, nlbl = ndimage.label(outlier)
    sizes = ndimage.sum(outlier, lbl, index=np.arange(1, nlbl + 1)) if nlbl > 0 else np.array([])
    isolated_labels = np.where(sizes <= HOT_MAX_CLUSTER)[0] + 1
    isolated_count = int(len(isolated_labels))
    cluster_count = int(np.sum(sizes > HOT_MAX_CLUSTER))
    isolated_pixel_mask = np.isin(lbl, isolated_labels) if isolated_count else np.zeros_like(lbl, dtype=bool)
    isolated_mag = np.abs(hp_resid[isolated_pixel_mask])
    n_pixels_scanned = int(border_mask.sum())
    # tail-shape check at stricter thresholds + absolute-magnitude counts (is this "heavy noise
    # tail" or genuine multi-degree stuck/dead pixels the way detector_defects models them?)
    tail_by_z = {}
    for zt in (6.0, 8.0, 10.0, 15.0):
        out_zt = (np.abs(z) > zt) & border_mask
        lbl_zt, nlbl_zt = ndimage.label(out_zt)
        sizes_zt = ndimage.sum(out_zt, lbl_zt, index=np.arange(1, nlbl_zt + 1)) if nlbl_zt > 0 else np.array([])
        tail_by_z[zt] = int(np.sum(sizes_zt <= HOT_MAX_CLUSTER))
    abs_delta_counts = {
        thr: int(np.sum(isolated_mag > thr)) for thr in (0.2, 0.5, 1.0, 2.0)
    }
    print(f"[hot-px] robust_sigma={robust_sigma:.5f} C, |z|>{HOT_Z_THRESH} isolated={isolated_count} "
          f"(of {n_pixels_scanned} scanned px), larger-cluster(likely real structure, not defects)={cluster_count}")
    print(f"[hot-px] isolated |delta_C| stats: median={np.median(isolated_mag) if isolated_count else float('nan'):.4f} "
          f"mean={np.mean(isolated_mag) if isolated_count else float('nan'):.4f} "
          f"max={np.max(isolated_mag) if isolated_count else float('nan'):.4f}")
    print(f"[hot-px] isolated count vs |z| threshold: {tail_by_z}")
    print(f"[hot-px] isolated count vs abs delta_C threshold: {abs_delta_counts}")

    # ---- summary.json ----
    summary = {
        "burst_selection": {
            "rule": "frame_audit.csv: frame_role=='sr_default' & is_sr_usable=='True'",
            "n_frames": n,
            "x_range_um": [min(xs), max(xs)],
            "y_range_um": [min(ys), max(ys)],
            "note": "sub-pixel-shift SR acquisition, not a fixed-scene temporal burst; all stats "
                    "restricted to flat/structure-free regions to be insensitive to the shifts",
        },
        "flat_mask": {
            "method": (f"local_std(mean_img, win={LOCAL_ROUGHNESS_WIN_PX}px) < median -> "
                       f"erode(disk r={EROSION_RADIUS_PX}) -> exclude {BORDER_PX}px border "
                       "(raw single-pixel gradient collapsed under erosion -- see docstring)"),
            "local_roughness_median_threshold_c": grad_thresh,
            "flat_fraction_of_frame": flat_frac,
        },
        "units": "degrees C (calibrated raw .txt temperature, not sensor DN)",
        "1_column_stripe_fpn": {
            "amplitude_c_std": col_amp,
            "temporal_noise_floor_c": col_noise_floor,
            "snr_vs_noise_floor": col_snr,
            "bg_smoothing_sigma_px": BG_SIGMA_PX,
        },
        "2_row_stripe_fpn": {
            "amplitude_c_std": row_amp,
            "temporal_noise_floor_c": row_noise_floor,
            "snr_vs_noise_floor": row_snr,
            "row_over_col_amplitude_ratio": row_over_col_ratio,
        },
        "3_low_freq_spectral_slope": {
            "alpha": spec["alpha"],
            "r2_loglog_fit": spec["r2_loglog_fit"],
            "fit_freq_range_cyc_px": [spec["fit_freq_min_cyc_px"], spec["fit_freq_max_cyc_px"]],
            "n_fit_bins": spec["n_fit_bins"],
            "crop_origin_yx": [y0, x0],
            "crop_size_px": SPECTRAL_CROP,
            "crop_flat_fraction": crop_flat_frac,
        },
        "4_temporal_noise": {
            "median_sigma_c": temporal_sigma,
            "median_lag1_autocorr_acquisition_order": lag1_median,
        },
        "5_hot_dead_pixels": {
            "z_threshold": HOT_Z_THRESH,
            "isolated_count": isolated_count,
            "n_pixels_scanned": n_pixels_scanned,
            "larger_cluster_count_excluded_as_structure": cluster_count,
            "robust_sigma_c": robust_sigma,
            "isolated_abs_delta_c": {
                "median": float(np.median(isolated_mag)) if isolated_count else None,
                "mean": float(np.mean(isolated_mag)) if isolated_count else None,
                "max": float(np.max(isolated_mag)) if isolated_count else None,
            },
            "isolated_count_vs_z_threshold": {str(k): v for k, v in tail_by_z.items()},
            "isolated_count_vs_abs_delta_c_threshold": {str(k): v for k, v in abs_delta_counts.items()},
        },
        "6_stripe_correlation_length_px": {
            "column_profile_1e_length_px": col_corr_len,
            "row_profile_1e_length_px": row_corr_len,
        },
    }
    with open(OUT_DIR / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[write] {OUT_DIR / 'summary.json'}")

    # ---- diagnostics.png ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    im0 = axes[0, 0].imshow(mean_img, cmap="inferno")
    axes[0, 0].set_title("temporal mean (248 frames)")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    axes[0, 1].imshow(flat, cmap="gray")
    axes[0, 1].add_patch(plt.Rectangle((x0, y0), SPECTRAL_CROP, SPECTRAL_CROP,
                                        edgecolor="red", facecolor="none", lw=1.5))
    axes[0, 1].set_title(f"flat mask (frac={flat_frac:.3f}); red=spectral crop")

    im2 = axes[0, 2].imshow(bg_resid, cmap="coolwarm", vmin=-np.nanstd(bg_resid) * 3,
                             vmax=np.nanstd(bg_resid) * 3)
    axes[0, 2].set_title(f"mean - gaussian(sigma={BG_SIGMA_PX}) residual")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    axes[1, 0].plot(col_profile, lw=0.8, label="column profile")
    axes[1, 0].axhline(0, color="k", lw=0.5)
    axes[1, 0].set_title(f"col profile (flat-masked), std={col_amp:.4f} C, 1/e_len={col_corr_len}px")
    axes[1, 0].set_xlabel("column index")

    axes[1, 1].plot(row_profile, lw=0.8, color="tab:orange", label="row profile")
    axes[1, 1].axhline(0, color="k", lw=0.5)
    axes[1, 1].set_title(f"row profile (flat-masked), std={row_amp:.4f} C, 1/e_len={row_corr_len}px")
    axes[1, 1].set_xlabel("row index")

    fsel = spec["fit_sel"]
    axes[1, 2].loglog(spec["freq"], spec["radial_psd"], ".", ms=3, color="gray", label="radial PSD")
    axes[1, 2].loglog(spec["freq"][fsel], spec["radial_psd"][fsel], ".", ms=4, color="tab:red",
                       label=f"fit window (alpha={spec['alpha']:.2f})")
    fit_line = 10 ** (spec["intercept"]) * spec["freq"][fsel] ** (-spec["alpha"])
    axes[1, 2].loglog(spec["freq"][fsel], fit_line, "-", color="tab:red", lw=1.5)
    axes[1, 2].set_title(f"radial PSD (flat crop), alpha={spec['alpha']:.2f} r2={spec['r2_loglog_fit']:.3f}")
    axes[1, 2].set_xlabel("spatial freq (cyc/px)")
    axes[1, 2].legend(fontsize=8)

    fig.suptitle(f"Real IR burst noise audit (n={n} frames, flat_frac={flat_frac:.3f})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "diagnostics.png", dpi=140)
    print(f"[write] {OUT_DIR / 'diagnostics.png'}")


if __name__ == "__main__":
    main()
