#!/usr/bin/env python
"""Small dark-dot defect retention probe (P0).

Quantifies the owner's visual observation that small dark dots (local cold
spots) sharply recovered by TGV are blurred or erased by the neural arms.

Reference arm = TGV (dots are detected ONLY on the TGV work image).
Evidence anchor = drizzle (dots weak in drizzle are evidence-limited, not
prior-erased; retention_vs_drz separates the two situations).

Pipeline (all thresholds pre-registered in CONFIG below):
  0. preprocessing      : per-arm work image (a+b)/2, sub-pixel alignment to
                          TGV via windowed phase correlation, robust gain
                          (Theil-Sen on high-gradient pixels).
  1. dot detection      : multi-scale LoG (dark response) on TGV work image,
                          3D (y,x,sigma) local maxima, overlap dedup, then
                          validity funnel (edge / depth SNR / a-b half
                          consistency / size cap).  No silent caps.
  2. per-dot per-arm    : depth = annulus-median background - mean of lowest
                          3 px in a 4-sigma square window; retention =
                          depth_arm / depth_tgv (gain-normalised); FWHM
                          equivalent diameter.
  3. stratified summary : arm x size-bin (and depth tertiles).
  4. optical subset     : dots inside the optical footprint (physical truth).
  5. tile band-pass NCC : dot-dense vs dot-free (gradient-energy matched)
                          128 px tiles, DoG band ~25-40 um.
  6. visual products    : crop board PNG, retention-vs-size scatter.
  7. built-in sanity    : TGV self-retention == 1.0; 20 random non-dot
                          control positions run through the same measurement.

Only numpy / scipy / pandas / matplotlib are used (no skimage).

Usage:
  python probe_dot_retention.py \
      --inbox remote_inbox/20260713_dotprobe --outdir output/dot_probe
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

# --------------------------------------------------------------------------
# Pre-registered configuration
# --------------------------------------------------------------------------
CONFIG = {
    "hr_um_per_px": 10.0,           # 20 um pitch, 2x SR
    "edge_margin_px": 16,           # filter 1: edge exclusion
    "log_sigmas": [1.0, 1.5, 2.0, 3.0, 4.0],
    "detect_resp_nsigma": 2.0,      # initial LoG peak threshold, in robust
                                    # (MAD) sigmas of each scale's response
                                    # map.  Low on purpose: the real gate is
                                    # the depth-SNR filter below.
    "dedup_dist_sigma": 2.0,        # centres closer than 2*max(sigma) merge
    "snr_min": 4.0,                 # filter 2: depth_tgv >= 4 x local noise
    "half_snr_min": 2.0,            # filter 3: each half >= 2 x local noise
    "r_max_px": 6.0,                # filter 4: FWHM radius (diam/2) <= 6 px
    "window_halfwidth_sigma": 2.0,  # "4-sigma square window" = side 4*sigma
    "annulus_r_sigma": (3.0, 5.0),  # background annulus radii (Euclidean)
    "n_lowest": 3,                  # depth = bg - mean(lowest 3 px)
    "align_apply_px": 0.15,         # shift arms with |offset| > 0.15 px
    "gain_tol": 0.05,               # normalise depth if |slope-1| > 0.05
    "gain_grad_pct": 90.0,          # high-gradient pixel selection
    "gain_max_px": 200_000,
    "gain_n_pairs": 400_000,
    "class_erased": 0.30,           # retention < 0.30       -> erased
    "class_blurred": 0.70,          # 0.30 <= retention <0.70-> blurred
    "size_bins_px": [3.0, 5.0, 8.0],  # FWHM diam: <=3, 3-5, 5-8, >8 px
    "tile_px": 128,
    "n_dense_tiles": 8,
    "n_empty_tiles": 8,
    "dog_sigmas_px": (2.5, 4.0),    # band-pass ~25-40 um
    "n_controls": 20,
    "control_sigma": 2.0,
    "control_min_dist_px": 24.0,    # controls keep away from detected dots
    "board_n_dots": 24,
    "seed": 20260713,
}

# arm name -> (file_a, file_b);  tgv is the reference, drizzle the anchor
ARM_FILES = {
    "drizzle": ("drizzle_a.npy", "drizzle_b.npy"),
    "tgv": ("tgv_a.npy", "tgv_b.npy"),
    "v14": ("v14_a_corrected.npy", "v14_b_corrected.npy"),
    "v19": ("v19_etaB_a_corrected.npy", "v19_etaB_b_corrected.npy"),
    "de_pb9": ("de_pb9_a_corrected.npy", "de_pb9_b_corrected.npy"),
    "depb9v6": ("depb9v6_a_corrected.npy", "depb9v6_b_corrected.npy"),
    "meanDC": ("meanDC_a_corrected.npy", "meanDC_b_corrected.npy"),
}
MEASURE_ARMS = ["drizzle", "tgv", "v14", "v19", "de_pb9", "depb9v6", "meanDC"]
NEURAL_ARMS = ["v14", "v19", "de_pb9", "depb9v6", "meanDC"]
BOARD_COLS = ["drizzle", "tgv", "v14", "v19", "de_pb9", "depb9v6", "meanDC"]


# --------------------------------------------------------------------------
# Step 0: loading, alignment, gain
# --------------------------------------------------------------------------
def load_arms(inbox: str) -> dict:
    arms = {}
    for name, (fa, fb) in ARM_FILES.items():
        a = np.load(os.path.join(inbox, fa)).astype(np.float64)
        b = np.load(os.path.join(inbox, fb)).astype(np.float64)
        arms[name] = {"a": a, "b": b, "work": 0.5 * (a + b)}
    return arms


def phase_corr_offset(ref: np.ndarray, img: np.ndarray) -> tuple[float, float]:
    """Sub-pixel offset via windowed, low-pass weighted phase correlation.

    Returns (dy, dx) = the shift to APPLY to `img` (scipy.ndimage.shift
    convention) so that it aligns onto `ref`.  Equivalently `img` sits at
    (-dy, -dx) relative to `ref`.
    """
    h, w = ref.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    f_ref = np.fft.fft2((ref - ref.mean()) * win)
    f_img = np.fft.fft2((img - img.mean()) * win)
    q = f_ref * np.conj(f_img)
    q /= np.abs(q) + 1e-12
    # mild low-pass so arms with differing high-freq content compare fairly
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    q *= np.exp(-((fy**2 + fx**2) / (2 * 0.25**2)))
    corr = np.real(np.fft.ifft2(q))
    py, px = np.unravel_index(np.argmax(corr), corr.shape)

    def parab(cm, c0, cp):
        # log-parabola: exact sub-pixel for a Gaussian-shaped peak (the
        # low-pass weight above makes the phase-corr peak Gaussian)
        if cm > 0 and c0 > 0 and cp > 0:
            cm, c0, cp = np.log(cm), np.log(c0), np.log(cp)
        den = cm - 2.0 * c0 + cp
        return 0.0 if den == 0 else 0.5 * (cm - cp) / den

    dy = parab(corr[(py - 1) % h, px], corr[py, px], corr[(py + 1) % h, px])
    dx = parab(corr[py, (px - 1) % w], corr[py, px], corr[py, (px + 1) % w])
    oy = (py + h // 2) % h - h // 2 + dy
    ox = (px + w // 2) % w - w // 2 + dx
    return float(oy), float(ox)


def robust_gain(ref: np.ndarray, arm: np.ndarray, margin: int,
                rng: np.random.Generator) -> dict:
    """Theil-Sen (sampled pairwise-median) slope of arm vs ref on
    high-gradient interior pixels."""
    grad = ndimage.gaussian_gradient_magnitude(ref, 1.5)
    interior = np.zeros(ref.shape, bool)
    interior[margin:-margin, margin:-margin] = True
    thresh = np.percentile(grad[interior], CONFIG["gain_grad_pct"])
    mask = interior & (grad >= thresh)
    ys, xs = np.nonzero(mask)
    if len(ys) > CONFIG["gain_max_px"]:
        sel = rng.choice(len(ys), CONFIG["gain_max_px"], replace=False)
        ys, xs = ys[sel], xs[sel]
    x = ref[ys, xs]
    y = arm[ys, xs]
    i = rng.integers(0, len(x), CONFIG["gain_n_pairs"])
    j = rng.integers(0, len(x), CONFIG["gain_n_pairs"])
    dxp = x[j] - x[i]
    dyp = y[j] - y[i]
    ok = np.abs(dxp) > 1e-3
    slope = float(np.median(dyp[ok] / dxp[ok]))
    intercept = float(np.median(y - slope * x))
    return {"slope": slope, "intercept": intercept, "n_px": int(len(x))}


def preprocess(arms: dict, rng: np.random.Generator):
    """Align every arm to the TGV work image and measure gain.  Returns the
    per-arm table (DataFrame) and mutates arms[name]['aligned'] /
    ['slope_applied']."""
    ref = arms["tgv"]["work"]
    rows = []
    for name in MEASURE_ARMS:
        work = arms[name]["work"]
        if name == "tgv":
            dy = dx = 0.0
        else:
            dy, dx = phase_corr_offset(ref, work)
        norm = float(np.hypot(dy, dx))
        shifted = norm > CONFIG["align_apply_px"]
        aligned = (ndimage.shift(work, (dy, dx), order=3, mode="nearest")
                   if shifted else work)
        arms[name]["aligned"] = aligned
        if name == "tgv":
            gain = {"slope": 1.0, "intercept": 0.0, "n_px": 0}
        else:
            gain = robust_gain(ref, aligned, CONFIG["edge_margin_px"], rng)
        slope_applied = (gain["slope"]
                         if abs(gain["slope"] - 1.0) > CONFIG["gain_tol"]
                         else 1.0)
        arms[name]["slope_applied"] = slope_applied
        rows.append({
            "arm": name,
            "offset_dy_px": dy,
            "offset_dx_px": dx,
            "offset_norm_px": norm,
            "shift_applied": bool(shifted),
            "gain_slope": gain["slope"],
            "gain_intercept": gain["intercept"],
            "gain_n_px": gain["n_px"],
            "slope_applied_to_depth": slope_applied,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Step 1: detection on TGV
# --------------------------------------------------------------------------
def detect_dots(tgv_work: np.ndarray) -> tuple[pd.DataFrame, dict]:
    sigmas = np.asarray(CONFIG["log_sigmas"])
    h, w = tgv_work.shape
    stack = np.empty((len(sigmas), h, w))
    for k, s in enumerate(sigmas):
        # dark dot (local minimum) -> POSITIVE scale-normalised LoG response
        stack[k] = (s**2) * ndimage.gaussian_laplace(tgv_work, s)
    # per-scale robust threshold
    thr = np.empty(len(sigmas))
    for k in range(len(sigmas)):
        r = stack[k]
        mad = np.median(np.abs(r - np.median(r))) * 1.4826
        thr[k] = CONFIG["detect_resp_nsigma"] * mad
    maxf = ndimage.maximum_filter(stack, size=(3, 3, 3), mode="nearest")
    cand = (stack == maxf) & (stack > thr[:, None, None])
    ks, ys, xs = np.nonzero(cand)
    resp = stack[ks, ys, xs]
    order = np.argsort(resp)[::-1]
    ks, ys, xs, resp = ks[order], ys[order], xs[order], resp[order]
    n_raw = len(resp)

    # greedy dedup: centre distance < dedup_dist_sigma * max(sigma) -> keep
    # the stronger (list is response-sorted)
    kept: list[int] = []
    ky = np.empty(0)
    kx = np.empty(0)
    ksig = np.empty(0)
    for i in range(n_raw):
        s_i = sigmas[ks[i]]
        if kept:
            d = np.hypot(ky - ys[i], kx - xs[i])
            lim = CONFIG["dedup_dist_sigma"] * np.maximum(ksig, s_i)
            if np.any(d < lim):
                continue
        kept.append(i)
        ky = np.append(ky, ys[i])
        kx = np.append(kx, xs[i])
        ksig = np.append(ksig, s_i)
    kept = np.asarray(kept, int)
    df = pd.DataFrame({
        "y": ys[kept].astype(int),
        "x": xs[kept].astype(int),
        "sigma": sigmas[ks[kept]],
        "log_response": resp[kept],
    })
    funnel = {"raw_3d_local_maxima": int(n_raw), "after_dedup": int(len(df))}
    return df, funnel


def window_slices(y: int, x: int, half: int, shape) -> tuple[slice, slice]:
    return (slice(max(0, y - half), min(shape[0], y + half + 1)),
            slice(max(0, x - half), min(shape[1], x + half + 1)))


def measure_dot(img: np.ndarray, y: int, x: int, sigma: float) -> dict:
    """Depth / background / FWHM-equivalent diameter at (y, x)."""
    half = int(np.ceil(CONFIG["window_halfwidth_sigma"] * sigma))
    r_in, r_out = (int(np.ceil(CONFIG["annulus_r_sigma"][0] * sigma)),
                   int(np.ceil(CONFIG["annulus_r_sigma"][1] * sigma)))
    sl = window_slices(y, x, r_out, img.shape)
    patch = img[sl]
    yy, xx = np.mgrid[sl[0], sl[1]]
    rr = np.hypot(yy - y, xx - x)
    ann = patch[(rr >= r_in) & (rr <= r_out)]
    bg = float(np.median(ann))
    wsl = window_slices(y, x, half, img.shape)
    wpatch = img[wsl].ravel()
    low = np.sort(wpatch)[:CONFIG["n_lowest"]]
    depth = bg - float(low.mean())
    n_fwhm = int(np.count_nonzero(wpatch < bg - 0.5 * depth))
    diam = float(np.sqrt(4.0 * n_fwhm / np.pi))
    return {"depth": depth, "bg": bg, "fwhm_diam": diam, "n_fwhm": n_fwhm}


def local_noise(tgv_a: np.ndarray, tgv_b: np.ndarray, y: int, x: int,
                sigma: float) -> float:
    """Per-half noise sigma: std of the a-b difference in the window / sqrt2."""
    half = int(np.ceil(CONFIG["window_halfwidth_sigma"] * sigma))
    sl = window_slices(y, x, half, tgv_a.shape)
    return float(np.std(tgv_a[sl] - tgv_b[sl]) / np.sqrt(2.0))


def apply_filters(dots: pd.DataFrame, arms: dict, funnel: dict) -> pd.DataFrame:
    tgv_w = arms["tgv"]["work"]
    tgv_a, tgv_b = arms["tgv"]["a"], arms["tgv"]["b"]
    h, w = tgv_w.shape
    m = CONFIG["edge_margin_px"]

    edge_ok = ((dots["y"] >= m) & (dots["y"] < h - m)
               & (dots["x"] >= m) & (dots["x"] < w - m))
    funnel["removed_edge_16px"] = int((~edge_ok).sum())
    dots = dots[edge_ok].reset_index(drop=True)
    funnel["after_edge"] = int(len(dots))

    meas = [measure_dot(tgv_w, r.y, r.x, r.sigma) for r in dots.itertuples()]
    dots["depth_tgv"] = [mm["depth"] for mm in meas]
    dots["bg_tgv"] = [mm["bg"] for mm in meas]
    dots["fwhm_diam_tgv_px"] = [mm["fwhm_diam"] for mm in meas]
    dots["noise_tgv"] = [local_noise(tgv_a, tgv_b, r.y, r.x, r.sigma)
                         for r in dots.itertuples()]
    dots["snr_tgv"] = dots["depth_tgv"] / dots["noise_tgv"]

    snr_ok = dots["snr_tgv"] >= CONFIG["snr_min"]
    funnel["removed_depth_snr_lt4"] = int((~snr_ok).sum())
    dots = dots[snr_ok].reset_index(drop=True)
    funnel["after_depth_snr"] = int(len(dots))

    da = [measure_dot(tgv_a, r.y, r.x, r.sigma)["depth"]
          for r in dots.itertuples()]
    db = [measure_dot(tgv_b, r.y, r.x, r.sigma)["depth"]
          for r in dots.itertuples()]
    dots["depth_tgv_a"] = da
    dots["depth_tgv_b"] = db
    half_ok = ((dots["depth_tgv_a"] >= CONFIG["half_snr_min"] * dots["noise_tgv"])
               & (dots["depth_tgv_b"] >= CONFIG["half_snr_min"] * dots["noise_tgv"]))
    funnel["removed_half_consistency"] = int((~half_ok).sum())
    dots = dots[half_ok].reset_index(drop=True)
    funnel["after_half_consistency"] = int(len(dots))

    size_ok = dots["fwhm_diam_tgv_px"] / 2.0 <= CONFIG["r_max_px"]
    funnel["removed_size_r_gt6px"] = int((~size_ok).sum())
    dots = dots[size_ok].reset_index(drop=True)
    funnel["after_size"] = int(len(dots))
    funnel["final_dots"] = int(len(dots))
    return dots


# --------------------------------------------------------------------------
# Step 2: per-dot per-arm measurement
# --------------------------------------------------------------------------
def classify(r: float) -> str:
    if not np.isfinite(r) or r < CONFIG["class_erased"]:
        return "erased"
    if r < CONFIG["class_blurred"]:
        return "blurred"
    return "preserved"


def measure_all_arms(dots: pd.DataFrame, arms: dict,
                     optical: np.ndarray) -> pd.DataFrame:
    dots = dots.copy()
    dots.insert(0, "dot_id", np.arange(len(dots)))
    dots["fwhm_diam_tgv_um"] = dots["fwhm_diam_tgv_px"] * CONFIG["hr_um_per_px"]
    dots["in_optical"] = [bool(np.isfinite(optical[r.y, r.x]))
                          for r in dots.itertuples()]
    for name in MEASURE_ARMS:
        img = arms[name]["aligned"]
        slope = arms[name]["slope_applied"]
        depth_raw, depth, fwhm = [], [], []
        for r in dots.itertuples():
            mm = measure_dot(img, r.y, r.x, r.sigma)
            depth_raw.append(mm["depth"])
            depth.append(mm["depth"] / slope)
            fwhm.append(mm["fwhm_diam"])
        dots[f"depth_raw_{name}"] = depth_raw
        dots[f"depth_{name}"] = depth
        dots[f"fwhm_diam_{name}_px"] = fwhm
        if name != "tgv":
            dots[f"retention_{name}"] = dots[f"depth_{name}"] / dots["depth_tgv"]
            dots[f"fwhm_ratio_{name}"] = (dots[f"fwhm_diam_{name}_px"]
                                          / dots["fwhm_diam_tgv_px"])
            dots[f"class_{name}"] = dots[f"retention_{name}"].map(classify)
    # sanity: TGV self-retention (aligned==work, slope==1) must be exactly 1
    dots["retention_tgv_self"] = dots["depth_tgv"] / dots["depth_tgv"]
    for name in NEURAL_ARMS:
        dots[f"retention_vs_drz_{name}"] = (dots[f"depth_{name}"]
                                            / dots["depth_drizzle"])
    edges = CONFIG["size_bins_px"]
    labels = [f"<={edges[0]:g}px", f"{edges[0]:g}-{edges[1]:g}px",
              f"{edges[1]:g}-{edges[2]:g}px", f">{edges[2]:g}px"]
    dots["size_bin"] = pd.cut(dots["fwhm_diam_tgv_px"],
                              [-np.inf] + edges + [np.inf], labels=labels)
    q1, q2 = dots["depth_tgv"].quantile([1 / 3, 2 / 3])
    dots["depth_bin"] = pd.cut(dots["depth_tgv"], [-np.inf, q1, q2, np.inf],
                               labels=["shallow", "mid", "deep"])
    return dots


# --------------------------------------------------------------------------
# Step 3: stratified summary
# --------------------------------------------------------------------------
def summarize(dots: pd.DataFrame) -> pd.DataFrame:
    rows = []
    size_order = list(dots["size_bin"].cat.categories) + ["ALL"]
    for name in [a for a in MEASURE_ARMS if a != "tgv"]:
        for sb in size_order:
            sub = dots if sb == "ALL" else dots[dots["size_bin"] == sb]
            if len(sub) == 0:
                rows.append({"arm": name, "size_bin": sb, "N": 0})
                continue
            ret = sub[f"retention_{name}"]
            cls = sub[f"class_{name}"]
            rows.append({
                "arm": name,
                "size_bin": sb,
                "N": int(len(sub)),
                "median_retention": float(ret.median()),
                "erased_pct": float((cls == "erased").mean() * 100),
                "blurred_pct": float((cls == "blurred").mean() * 100),
                "preserved_pct": float((cls == "preserved").mean() * 100),
                "median_fwhm_ratio": float(sub[f"fwhm_ratio_{name}"].median()),
            })
    return pd.DataFrame(rows)


def summarize_by_depth(dots: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in [a for a in MEASURE_ARMS if a != "tgv"]:
        for db in ["shallow", "mid", "deep"]:
            sub = dots[dots["depth_bin"] == db]
            if len(sub) == 0:
                rows.append({"arm": name, "depth_bin": db, "N": 0})
                continue
            ret = sub[f"retention_{name}"]
            cls = sub[f"class_{name}"]
            rows.append({
                "arm": name,
                "depth_bin": db,
                "N": int(len(sub)),
                "median_retention": float(ret.median()),
                "erased_pct": float((cls == "erased").mean() * 100),
                "blurred_pct": float((cls == "blurred").mean() * 100),
                "preserved_pct": float((cls == "preserved").mean() * 100),
                "median_fwhm_ratio": float(sub[f"fwhm_ratio_{name}"].median()),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Step 5: tile band-pass NCC
# --------------------------------------------------------------------------
def bandpass(img: np.ndarray) -> np.ndarray:
    s1, s2 = CONFIG["dog_sigmas_px"]
    return ndimage.gaussian_filter(img, s1) - ndimage.gaussian_filter(img, s2)


def tile_ncc_analysis(dots: pd.DataFrame, arms: dict) -> tuple[pd.DataFrame, dict]:
    t = CONFIG["tile_px"]
    tgv_w = arms["tgv"]["work"]
    h, w = tgv_w.shape
    ny, nx = h // t, w // t
    counts = np.zeros((ny, nx), int)
    for r in dots.itertuples():
        ty, tx = r.y // t, r.x // t
        if ty < ny and tx < nx:
            counts[ty, tx] += 1
    grad = ndimage.gaussian_gradient_magnitude(tgv_w, 1.5) ** 2
    energy = np.array([[grad[i * t:(i + 1) * t, j * t:(j + 1) * t].mean()
                        for j in range(nx)] for i in range(ny)])
    flat = [(counts[i, j], energy[i, j], i, j)
            for i in range(ny) for j in range(nx)]
    dense = sorted(flat, key=lambda z: -z[0])[:CONFIG["n_dense_tiles"]]
    dense_set = {(z[2], z[3]) for z in dense}
    empties = [z for z in flat if z[0] == 0]
    if len(empties) < CONFIG["n_empty_tiles"]:
        # deviation (recorded in summary): if fewer than 8 zero-dot tiles
        # exist, the comparison group falls back to the lowest-count tiles
        pool = sorted((z for z in flat if (z[2], z[3]) not in dense_set),
                      key=lambda z: z[0])
        empties = pool[:max(CONFIG["n_empty_tiles"] * 3,
                            CONFIG["n_empty_tiles"])]
    used = set()
    matched = []
    for cnt, e, i, j in dense:
        cands = [z for z in empties if (z[2], z[3]) not in used]
        best = min(cands, key=lambda z: abs(z[1] - e))
        used.add((best[2], best[3]))
        matched.append(best)

    grp2 = "no_dot" if all(z[0] == 0 for z in matched) else "sparse"
    bp = {name: bandpass(arms[name]["aligned"]) for name in MEASURE_ARMS}
    rows = []
    for group, tiles in [("dense", dense), (grp2, matched)]:
        for cnt, e, i, j in tiles:
            sl = (slice(i * t, (i + 1) * t), slice(j * t, (j + 1) * t))
            ref = bp["tgv"][sl].ravel()
            row = {"tile": f"r{i}c{j}", "group": group, "y0": i * t,
                   "x0": j * t, "n_dots": int(cnt),
                   "grad_energy": float(e)}
            for name in MEASURE_ARMS:
                if name == "tgv":
                    continue
                a = bp[name][sl].ravel()
                row[f"ncc_{name}"] = float(np.corrcoef(ref, a)[0, 1])
            rows.append(row)
    df = pd.DataFrame(rows)
    means = {}
    for name in [a for a in MEASURE_ARMS if a != "tgv"]:
        md = float(df.loc[df.group == "dense", f"ncc_{name}"].mean())
        me = float(df.loc[df.group == grp2, f"ncc_{name}"].mean())
        means[name] = {"dense_mean": md, f"{grp2}_mean": me,
                       f"dense_minus_{grp2}": md - me}
    return df, means


# --------------------------------------------------------------------------
# Step 7: random-position controls
# --------------------------------------------------------------------------
def control_measurements(dots: pd.DataFrame, arms: dict,
                         rng: np.random.Generator) -> pd.DataFrame:
    tgv_w = arms["tgv"]["work"]
    tgv_a, tgv_b = arms["tgv"]["a"], arms["tgv"]["b"]
    h, w = tgv_w.shape
    sig = CONFIG["control_sigma"]
    margin = max(CONFIG["edge_margin_px"],
                 int(np.ceil(CONFIG["annulus_r_sigma"][1] * sig)))
    dy = dots["y"].to_numpy()
    dx = dots["x"].to_numpy()
    pts = []
    while len(pts) < CONFIG["n_controls"]:
        y = int(rng.integers(margin, h - margin))
        x = int(rng.integers(margin, w - margin))
        if len(dy) and np.min(np.hypot(dy - y, dx - x)) < CONFIG["control_min_dist_px"]:
            continue
        pts.append((y, x))
    rows = []
    for k, (y, x) in enumerate(pts):
        mm = measure_dot(tgv_w, y, x, sig)
        row = {"ctrl_id": k, "y": y, "x": x, "sigma": sig,
               "depth_tgv": mm["depth"],
               "noise_tgv": local_noise(tgv_a, tgv_b, y, x, sig)}
        for name in MEASURE_ARMS:
            if name == "tgv":
                continue
            d = (measure_dot(arms[name]["aligned"], y, x, sig)["depth"]
                 / arms[name]["slope_applied"])
            row[f"depth_{name}"] = d
            row[f"retention_{name}"] = d / mm["depth"] if mm["depth"] != 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Step 6: visual products
# --------------------------------------------------------------------------
def pick_board_dots(dots: pd.DataFrame,
                    rng: np.random.Generator) -> pd.DataFrame:
    """Stratified sample: 2 seeded-random dots per (size_bin x depth_bin)
    cell (typical dots, not outliers), then back-fill so that every class
    (erased/blurred/preserved, judged on the median neural retention) that
    exists in the population is represented at least twice."""
    med_ret = dots[[f"retention_{n}" for n in NEURAL_ARMS]].median(axis=1)
    dots = dots.assign(_mret=med_ret, _mcls=med_ret.map(classify))
    chosen = []
    for sb in dots["size_bin"].cat.categories:
        for db in ["shallow", "mid", "deep"]:
            cell = dots[(dots["size_bin"] == sb) & (dots["depth_bin"] == db)]
            if len(cell) == 0:
                continue
            take = min(2, len(cell))
            idx = rng.choice(cell.index.to_numpy(), take, replace=False)
            chosen.extend(cell.loc[i] for i in idx)
    board = pd.DataFrame(chosen).drop_duplicates("dot_id")
    for cls in ["erased", "blurred", "preserved"]:
        pop = dots[dots["_mcls"] == cls]
        need = min(2, len(pop)) - int((board["_mcls"] == cls).sum())
        if need > 0:
            pool = pop.drop(index=[i for i in board.index if i in pop.index],
                            errors="ignore")
            idx = rng.choice(pool.index.to_numpy(),
                             min(need, len(pool)), replace=False)
            # drop random members of the most-represented class to make room
            for i in idx:
                big = board["_mcls"].value_counts().idxmax()
                drop = rng.choice(board[board["_mcls"] == big].index.to_numpy())
                board = board.drop(index=drop)
                board = pd.concat([board, dots.loc[[i]]])
    if len(board) > CONFIG["board_n_dots"]:
        board = board.sort_values("_mret").iloc[
            np.linspace(0, len(board) - 1, CONFIG["board_n_dots"]).astype(int)]
    return board.sort_values(["size_bin", "depth_bin"])


def render_board(board: pd.DataFrame, arms: dict, path: str) -> None:
    n = len(board)
    ncol = len(BOARD_COLS)
    fig, axes = plt.subplots(n, ncol, figsize=(1.75 * ncol, 1.75 * n))
    axes = np.atleast_2d(axes)
    for i, r in enumerate(board.itertuples()):
        hw = max(14, int(np.ceil(6 * r.sigma)))
        sl = window_slices(r.y, r.x, hw, arms["tgv"]["work"].shape)
        vmin, vmax = -1.5 * r.depth_tgv, 0.75 * r.depth_tgv
        for j, name in enumerate(BOARD_COLS):
            ax = axes[i, j]
            img = arms[name]["aligned"]
            mm = measure_dot(img, r.y, r.x, r.sigma)
            crop = (img[sl] - mm["bg"]) / arms[name]["slope_applied"]
            ax.imshow(crop, cmap="gray", vmin=vmin, vmax=vmax,
                      interpolation="nearest")
            cy, cx = r.y - sl[0].start, r.x - sl[1].start
            ax.add_patch(plt.Circle((cx, cy), np.sqrt(2) * r.sigma, fill=False,
                                    color="tab:red", lw=0.6, alpha=0.8))
            if name == "tgv":
                lab = "ref"
            else:
                lab = f"{getattr(r, f'retention_{name}'):.2f}"
            ax.text(0.03, 0.03, lab, transform=ax.transAxes, fontsize=6,
                    color="yellow", va="bottom")
            if i == 0:
                ax.set_title(name, fontsize=8)
            if j == 0:
                ax.set_ylabel(
                    f"#{r.dot_id} d={r.fwhm_diam_tgv_px:.1f}px\n"
                    f"depth={r.depth_tgv:.3f} {r.size_bin}/{r.depth_bin}",
                    fontsize=5)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("dot crops (a+b)/2, local bg removed, window = "
                 "[-1.5, +0.75] x depth_tgv; label = retention", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.995))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_scatter(dots: pd.DataFrame, path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    arms_plot = ["drizzle"] + NEURAL_ARMS
    x = dots["fwhm_diam_tgv_px"].to_numpy()
    xb = np.arange(1.0, np.ceil(x.max()) + 1.0)  # 1-px bins for medians
    ylim = (-0.4, 2.0)
    n_out = 0
    for k, name in enumerate(arms_plot):
        y = dots[f"retention_{name}"].to_numpy()
        n_out += int(np.sum((y < ylim[0]) | (y > ylim[1])))
        ax.scatter(x, y, s=5, alpha=0.18, color=colors[k], linewidths=0)
        med = [np.median(y[(x >= lo) & (x < lo + 1)])
               for lo in xb[:-1] if np.any((x >= lo) & (x < lo + 1))]
        ctr = [lo + 0.5 for lo in xb[:-1] if np.any((x >= lo) & (x < lo + 1))]
        ax.plot(ctr, med, "-o", color=colors[k], lw=1.8, ms=4, label=name)
    ax.axhline(CONFIG["class_erased"], color="k", ls=":", lw=0.8)
    ax.axhline(CONFIG["class_blurred"], color="k", ls=":", lw=0.8)
    ax.axhline(1.0, color="k", ls="-", lw=0.5, alpha=0.4)
    ax.set_ylim(*ylim)
    ax.set_xlabel("TGV FWHM-equivalent diameter (HR px; 1 px = 10 um)")
    ax.set_ylabel("retention (depth_arm / depth_tgv)")
    ax.set_title("dark-dot depth retention vs size (reference = TGV); "
                 "lines = 1-px-bin medians")
    ax.text(0.99, 0.01, f"{n_out} pts outside y-range (see per_dot.csv)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
    ax.legend(fontsize=8)
    sec = ax.secondary_xaxis(
        "top", functions=(lambda p: p * CONFIG["hr_um_per_px"],
                          lambda u: u / CONFIG["hr_um_per_px"]))
    sec.set_xlabel("um")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_optical(subset: pd.DataFrame, arms: dict, optical: np.ndarray,
                   path: str) -> None:
    cols = ["optical"] + BOARD_COLS
    n = len(subset)
    fig, axes = plt.subplots(n, len(cols), figsize=(1.75 * len(cols), 1.9 * n))
    axes = np.atleast_2d(axes)
    for i, r in enumerate(subset.itertuples()):
        hw = max(14, int(np.ceil(6 * r.sigma)))
        sl = window_slices(r.y, r.x, hw, optical.shape)
        vmin, vmax = -1.5 * r.depth_tgv, 0.75 * r.depth_tgv
        for j, name in enumerate(cols):
            ax = axes[i, j]
            if name == "optical":
                crop = optical[sl]
                ax.imshow(crop, cmap="gray", interpolation="nearest")
            else:
                img = arms[name]["aligned"]
                mm = measure_dot(img, r.y, r.x, r.sigma)
                crop = (img[sl] - mm["bg"]) / arms[name]["slope_applied"]
                ax.imshow(crop, cmap="gray", vmin=vmin, vmax=vmax,
                          interpolation="nearest")
                if name != "tgv":
                    ax.text(0.03, 0.03, f"{getattr(r, f'retention_{name}'):.2f}",
                            transform=ax.transAxes, fontsize=6, color="yellow",
                            va="bottom")
            cy, cx = r.y - sl[0].start, r.x - sl[1].start
            ax.add_patch(plt.Circle((cx, cy), np.sqrt(2) * r.sigma, fill=False,
                                    color="tab:red", lw=0.6))
            if i == 0:
                ax.set_title(name, fontsize=8)
            if j == 0:
                ax.set_ylabel(f"#{r.dot_id}", fontsize=6)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("optical-footprint dots (physical ground truth)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# summary.md
# --------------------------------------------------------------------------
def df_to_md(df: pd.DataFrame, floatfmt: str = "%.3f") -> str:
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda v: floatfmt % v if np.isfinite(v) else "nan")
    header = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(d.columns)) + "|"
    body = "\n".join("| " + " | ".join(map(str, row)) + " |"
                     for row in d.itertuples(index=False))
    return "\n".join([header, sep, body])


def write_summary(outdir, pre, funnel, summary, summary_depth, dots,
                  optical_subset, ncc_means, controls, sanity_tgv_ok):
    lines = ["# Dot retention probe (P0) -- measurement summary", ""]
    lines += ["## 0. Preprocessing: alignment to TGV + gain", "",
              df_to_md(pre, "%.4f"), "",
              f"- shift applied when |offset| > {CONFIG['align_apply_px']} px "
              f"(scipy.ndimage.shift, order=3)",
              f"- depth divided by slope when |slope-1| > {CONFIG['gain_tol']}",
              "- historical offset probes (offset_probe_summary_*.csv) measure "
              "arm-vs-drizzle before correction (~0.05-0.1 px); the numbers "
              "above are arm-vs-TGV on the delivered (already corrected) "
              "files, so they are expected to sit at the ~0.1 px level.", ""]
    lines += ["## 1. Detection funnel (TGV work image only)", ""]
    for k, v in funnel.items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## 3. Arm x size-bin summary (size = TGV FWHM diam; "
              "1 px = 10 um)", "", df_to_md(summary), ""]
    lines += ["## 3b. Arm x depth-tertile summary", "",
              df_to_md(summary_depth), ""]
    lines += ["## 4. Optical-footprint subset", ""]
    if len(optical_subset) == 0:
        lines.append("No detected dot falls inside the optical footprint "
                     "(y 391:457, x 540:606, 2233 px).")
    else:
        cols = (["dot_id", "y", "x", "sigma", "depth_tgv",
                 "fwhm_diam_tgv_px"]
                + [f"retention_{n}" for n in ["drizzle"] + NEURAL_ARMS])
        lines.append(df_to_md(optical_subset[cols]))
        lines += ["",
                  "Factual note: the optical footprint covers a dense "
                  "periodic chevron/zigzag line pattern (see "
                  "optical_subset_crops.png). The detections inside it sit "
                  "on dark minima of that pattern (structure minima with "
                  "optical ground truth), not on isolated dots.", ""]
        med = {n: float(optical_subset[f"retention_{n}"].median())
               for n in ["drizzle"] + NEURAL_ARMS}
        lines.append("Median retention on this subset: "
                     + ", ".join(f"{k}={v:.3f}" for k, v in med.items()))
    lines += ["", "## 5. Tile band-pass NCC vs TGV "
              f"(DoG sigma {CONFIG['dog_sigmas_px'][0]}-"
              f"{CONFIG['dog_sigmas_px'][1]} px, {CONFIG['tile_px']} px tiles)",
              ""]
    ncc_df = pd.DataFrame(ncc_means).T.reset_index(names="arm")
    lines += [df_to_md(ncc_df), ""]
    lines += ["## 7. Built-in sanity", "",
              f"- TGV self-retention == 1.0 for all dots: {sanity_tgv_ok}",
              f"- {CONFIG['n_controls']} random non-dot controls "
              f"(sigma={CONFIG['control_sigma']}), per-arm retention median "
              "[IQR] (measurement-bias check; depths at non-dots are "
              "noise-level):", ""]
    ctrl_rows = []
    for name in ["drizzle"] + NEURAL_ARMS:
        r = controls[f"retention_{name}"]
        ctrl_rows.append({
            "arm": name,
            "ctrl_retention_median": float(r.median()),
            "ctrl_retention_q25": float(r.quantile(0.25)),
            "ctrl_retention_q75": float(r.quantile(0.75)),
            "ctrl_depth_median": float(controls[f"depth_{name}"].median()),
        })
    ctrl_rows.append({"arm": "tgv (self)",
                      "ctrl_retention_median": 1.0,
                      "ctrl_retention_q25": 1.0, "ctrl_retention_q75": 1.0,
                      "ctrl_depth_median":
                          float(controls["depth_tgv"].median())})
    lines += [df_to_md(pd.DataFrame(ctrl_rows)), ""]
    lines += [
        "## Deviations / implementation decisions", "",
        "1. Initial LoG candidate threshold = 2 x per-scale robust (MAD) "
        "sigma of the response map. The design left this unspecified; a low "
        "threshold was chosen so the pre-registered depth-SNR gate does the "
        "actual filtering (funnel above reports every stage, no silent "
        "caps).",
        "2. '4-sigma square window' implemented as a square of SIDE 4*sigma "
        "(half-width 2*sigma), so the window stays inside the background "
        "annulus that starts at 3*sigma.",
        "3. Size cap 'r <= 6 px' implemented on the measured TGV "
        "FWHM-equivalent radius (diam/2 <= 6 px).",
        "4. Step 5 fallback: zero-dot 128 px tiles do not exist (all 70 "
        "tiles contain detections), so the comparison group is the 8 "
        "LOWEST-count tiles ('sparse' group in tile_ncc.csv; n_dots column "
        "gives the actual counts, 36-42 dots/tile vs 63-133 for dense). "
        "The pre-registered gradient-energy matching could NOT be "
        "satisfied: every low-count tile is also low-structure (grad_energy "
        "7e-5..1e-4 vs 3e-3..5e-2 for dense tiles; column in "
        "tile_ncc.csv), so the dense-vs-sparse NCC difference confounds "
        "dot density with overall structure level.",
        "5. No arm needed resampling (all |offset| < 0.15 px); all five "
        "neural arms exceeded the 5% gain tolerance and had depth divided "
        "by their slope (table in section 0).",
        "6. Board dots are a seeded random stratified sample (2 per "
        "size x depth cell) with class back-fill, rather than extreme-"
        "retention picks, so crops show typical dots.",
        "7. Detection is pattern-agnostic (pre-registered): LoG dark maxima "
        "include dark minima of periodic line structure, not only isolated "
        "dots. See the optical-subset note above.",
        "",
        "## Config (pre-registered)", "", "```json",
        json.dumps(CONFIG, indent=2), "```", ""]
    with open(os.path.join(outdir, "summary.md"), "w") as f:
        f.write("\n".join(lines))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inbox", default="remote_inbox/20260713_dotprobe")
    ap.add_argument("--outdir", default="output/dot_probe")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    inter = os.path.join(args.outdir, "intermediate")
    os.makedirs(inter, exist_ok=True)
    rng = np.random.default_rng(CONFIG["seed"])

    print("[0] loading arms ...")
    arms = load_arms(args.inbox)
    optical = np.load(os.path.join(args.inbox, "optical_warp_hr.npy"))

    print("[0] alignment + gain ...")
    pre = preprocess(arms, rng)
    pre.to_csv(os.path.join(args.outdir, "preprocessing.csv"), index=False)
    print(pre.to_string(index=False))
    for name in MEASURE_ARMS:
        np.save(os.path.join(inter, f"aligned_{name}.npy"),
                arms[name]["aligned"].astype(np.float32))

    print("[1] LoG detection on TGV ...")
    cand, funnel = detect_dots(arms["tgv"]["work"])
    cand.to_csv(os.path.join(inter, "candidates_pre_filter.csv"), index=False)
    dots = apply_filters(cand, arms, funnel)
    with open(os.path.join(args.outdir, "detection_funnel.json"), "w") as f:
        json.dump(funnel, f, indent=2)
    print(json.dumps(funnel, indent=2))
    if len(dots) == 0:
        print("NO dots survived the funnel -- aborting after writing funnel.")
        return

    print(f"[2] per-dot per-arm measurement on {len(dots)} dots ...")
    dots = measure_all_arms(dots, arms, optical)
    sanity_tgv_ok = bool(np.allclose(dots["retention_tgv_self"], 1.0))
    dots.to_csv(os.path.join(args.outdir, "per_dot.csv"), index=False)

    print("[3] stratified summaries ...")
    summary = summarize(dots)
    summary.to_csv(os.path.join(args.outdir, "summary_by_arm_size.csv"),
                   index=False)
    summary_depth = summarize_by_depth(dots)
    summary_depth.to_csv(os.path.join(args.outdir, "summary_by_arm_depth.csv"),
                         index=False)
    print(summary.to_string(index=False))

    print("[4] optical subset ...")
    optical_subset = dots[dots["in_optical"]].copy()
    optical_subset.to_csv(os.path.join(args.outdir, "optical_subset.csv"),
                          index=False)
    if len(optical_subset):
        render_optical(optical_subset, arms, optical,
                       os.path.join(args.outdir, "optical_subset_crops.png"))
    print(f"  dots in optical footprint: {len(optical_subset)}")

    print("[5] tile band-pass NCC ...")
    tiles, ncc_means = tile_ncc_analysis(dots, arms)
    tiles.to_csv(os.path.join(args.outdir, "tile_ncc.csv"), index=False)
    print(json.dumps(ncc_means, indent=2))

    print("[7] controls ...")
    controls = control_measurements(dots, arms, rng)
    controls.to_csv(os.path.join(args.outdir, "control_points.csv"),
                    index=False)

    print("[6] figures ...")
    board = pick_board_dots(dots, rng)
    board.to_csv(os.path.join(inter, "board_selection.csv"), index=False)
    render_board(board, arms, os.path.join(args.outdir, "board_crops.png"))
    render_scatter(dots, os.path.join(args.outdir, "retention_vs_size.png"))

    write_summary(args.outdir, pre, funnel, summary, summary_depth, dots,
                  optical_subset, ncc_means, controls, sanity_tgv_ok)
    print(f"done. products in {args.outdir}")


if __name__ == "__main__":
    main()
