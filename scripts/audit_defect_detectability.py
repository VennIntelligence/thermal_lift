#!/usr/bin/env python3
"""Zero-training defect-detectability audit for a synthetic training pool (gate layer 1).

Motivation (ACL-063/065/066/067): the v7 pool's dense small dark-dot defect family
(hole radius 1-4 HR px, 20-50 per scene, partial depth 0.3-1.0, 1 px soft edge) was
built to FIX small-dot erasure, but training on it made erasure catastrophically
worse. Locked-in hypothesis to test here: those dots are already physically
undetectable in the NOISY LR INPUT (1-4 HR px = 0.5-2 LR px, further hit by PSF,
partial depth, soft edges and structured noise), so the Bayes-optimal behaviour the
network learns is "small dark dot == unrecoverable -> output the prior mean (erase)".
Control: v6 holes (4-13 px, full depth, hard edge, <=6/scene) should be far above
any plausible detectability threshold.

For every annotated hole instance this script computes two input-side CNRs:

  cnr_analytic   -- white-noise matched-filter CNR. The hole's HR-domain
                    perturbation template is rebuilt with realism.py semantics
                    (depth_frac x local structure level, linear soft edge of
                    hole_edge_softness_px, isothermal edge_sigma smoothing), then
                    pushed through the POOL GENERATOR'S OWN forward operators
                    (tcforge.physics.apply_psf_blur with the scene's recorded PSF +
                    tcforge.forward._block_average_from_blurred with each frame's
                    recorded shift; local patches aligned to the HR block grid so
                    the +0.5px block-center convention is inherited, not re-derived).
                    CNR_f = ||template_f||_2 / sigma_white ; multi-frame
                    CNR = sqrt(sum_f CNR_f^2). sigma_white is the per-frame grain
                    sigma estimated from well-separated frame differences (burst-
                    fixed FPN/vignette/1/f cancel; AR(1) decorrelates at large lag).
                    cnr_analytic_ar1 additionally applies the AR(1) multi-frame
                    penalty sqrt((1-rho)/(1+rho)) using metadata grain_ar1_rho.

  cnr_empirical  -- measured on the actual noisy lr_burst: crops around the hole's
                    per-frame LR position (block-center convention, see
                    hole_lr_position()) are registered and averaged; depth = local
                    annulus-median background minus core mean; noise = robust
                    (MAD) std of the SAME averaged crop's annulus -- so burst-fixed
                    structured noise (row/col stripe FPN, 1/f field, pixel FPN,
                    vignette curvature) that survives averaging is counted, which
                    the white-noise analytic CNR cannot see.

Extra per-hole diagnostics: amp_fit (matched-filter amplitude of the projected
template against the averaged data crop; ~1.0 when the modelled template matches
what is physically in the burst -- a live guard against coordinate/convention
bugs) and reg_dy/dx_lr_px (sub-pixel offset between the data dip and the template,
computed only where the dip is strong enough to localise).

Reads holes exactly like algos/ep07_unet_sr/scripts/probe_dot_retention_gt.py
(geometry_metadata.defects hole_centers_yx/hole_radii/hole_depths; falls back to
defect_annotations.instances type=="hole") and reuses its measure_dot() convention
for the GT-side depth/background measurement.

Third mode, --simulate-from-config: legacy-recipe pools (v6 bench48 / v6_cpu) never
record hole coordinates (realism.py only writes hole_centers_yx when a non-default
hole knob is active). For those, pass a pool config (or bare defects block) and the
script SAMPLES hypothetical holes from that distribution (radius/depth/softness
draws mirroring apply_defects; centres drawn from the interior-eroded structure
mask, hole_margin_px semantics), places them on the audited pool's real scenes
(real background level, PSF, shifts, noise) and pushes them through the same
forward chain. The analytic CNR needs nothing else; for the empirical CNR the
projected template is INJECTED into the real noisy LR crops (dip subtracted from
the measured burst), so the identical measurement runs on physically real noise.
Summaries are schema-identical across modes and directly comparable.

Outputs: per_hole.csv, summary.json, cnr_hist_by_radius.png,
detectability_heatmap.png, hand_check_crops.png.

Usage (repo root, pure CPU):
  uv run python scripts/audit_defect_detectability.py \
      --pool-dir data/synthetic/pool_2x_v7_5k \
      --max-scenes 200 --outdir output/defect_detectability_v7 \
      --cnr-thresholds 1,2,3,5 --seed 20260709
  # v6 reference (no recorded instances -> simulate from the v6 defects block):
  uv run python scripts/audit_defect_detectability.py \
      --pool-dir data/synthetic/pool_2x_v6_bench48 \
      --simulate-from-config configs/synthetic/pool_2x_v6_bench48.json \
      --outdir output/defect_detectability_v6ref --seed 20260709
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TCFORGE_SRC = PROJECT_ROOT / "tcforge" / "src"
if TCFORGE_SRC.exists() and str(TCFORGE_SRC) not in sys.path:
    sys.path.insert(0, str(TCFORGE_SRC))

from tcforge.forward import _block_average_from_blurred  # noqa: E402  (pool generator's own op)
from tcforge.physics import apply_psf_blur  # noqa: E402  (pool generator's own op)

# GT-side measurement convention: identical to probe_dot_retention_gt.py
WINDOW_HALFWIDTH_SIGMA = 2.0
ANNULUS_R_SIGMA = (3.0, 5.0)
N_LOWEST = 3


# ---------------------------------------------------------------------------
# measure_dot: ported verbatim from probe_dot_retention_gt.py (same口径)
# ---------------------------------------------------------------------------
def window_slices(y: int, x: int, half: int, shape: tuple[int, int]) -> tuple[slice, slice]:
    return (
        slice(max(0, y - half), min(shape[0], y + half + 1)),
        slice(max(0, x - half), min(shape[1], x + half + 1)),
    )


def measure_dot(img: np.ndarray, y: int, x: int, sigma: float) -> dict[str, Any]:
    """Depth / background at a KNOWN (y, x); window/annulus scaled from sigma (= hole radius_px)."""
    half = int(np.ceil(WINDOW_HALFWIDTH_SIGMA * sigma))
    r_in = int(np.ceil(ANNULUS_R_SIGMA[0] * sigma))
    r_out = int(np.ceil(ANNULUS_R_SIGMA[1] * sigma))
    sl = window_slices(y, x, r_out, img.shape)
    patch = img[sl]
    yy, xx = np.mgrid[sl[0], sl[1]]
    rr = np.hypot(yy - y, xx - x)
    ann = patch[(rr >= r_in) & (rr <= r_out)]
    edge_clipped = ann.size == 0 or (sl[0].stop - sl[0].start) < (2 * r_out + 1) or (
        sl[1].stop - sl[1].start
    ) < (2 * r_out + 1)
    bg = float(np.median(ann)) if ann.size else float("nan")
    wsl = window_slices(y, x, half, img.shape)
    wpatch = img[wsl].ravel()
    n_take = min(N_LOWEST, wpatch.size)
    low = np.sort(wpatch)[:n_take]
    depth = bg - float(low.mean()) if n_take else float("nan")
    return {"depth": depth, "bg": bg, "edge_clipped": bool(edge_clipped)}


# ---------------------------------------------------------------------------
# Scene IO
# ---------------------------------------------------------------------------
def list_scene_dirs(pool_dir: Path) -> list[Path]:
    manifest = pool_dir / "manifest.csv"
    if manifest.exists():
        df = pd.read_csv(manifest)
        return [pool_dir / str(d) for d in df["scene_dir"]]
    return sorted(p for p in pool_dir.glob("scene_*") if p.is_dir())


def load_burst(scene_dir: Path) -> np.ndarray | None:
    p_npy = scene_dir / "lr_burst.npy"
    p_npz = scene_dir / "lr_burst.npz"
    if p_npy.exists():
        return np.load(p_npy).astype(np.float32)
    if p_npz.exists():
        with np.load(p_npz) as data:
            return data["lr_burst"].astype(np.float32)
    return None


def read_holes(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Hole instances. Primary: geometry_metadata.defects hole_centers_yx / hole_radii /
    hole_depths (probe_dot_retention_gt.py口径). Contexts joined from
    defect_annotations.instances (same append order in realism.apply_defects).
    Fallback: defect_annotations.instances type=='hole' alone."""
    defects = (metadata.get("geometry_metadata") or {}).get("defects") or {}
    ann = metadata.get("defect_annotations") or {}
    hole_insts = [i for i in (ann.get("instances") or []) if i.get("type") == "hole"]

    centers = defects.get("hole_centers_yx")
    if centers:
        radii = defects.get("hole_radii") or []
        depths = defects.get("hole_depths") or [1.0] * len(centers)
        softness = float(defects.get("hole_edge_softness_px", 0.0) or 0.0)
        if not (len(centers) == len(radii) == len(depths)):
            raise ValueError("hole_centers_yx/hole_radii/hole_depths length mismatch")
        contexts = [i.get("context") for i in hole_insts] if len(hole_insts) == len(centers) \
            else [None] * len(centers)
        return [
            {"y": int(c[0]), "x": int(c[1]), "radius_px": float(r), "depth_frac": float(d),
             "softness_px": softness, "context": ctx}
            for c, r, d, ctx in zip(centers, radii, depths, contexts)
        ]
    # fallback: instance records only (e.g. legacy-recipe pool with record_instances)
    return [
        {"y": int(i["center_yx_hr"][0]), "x": int(i["center_yx_hr"][1]),
         "radius_px": float(i["radius_px"]),
         "depth_frac": float(i.get("depth_or_amplitude") or 1.0),
         "softness_px": float(i.get("edge_softness_px") or 0.0),
         "context": i.get("context")}
        for i in hole_insts
    ]


def _disk(rad: float) -> np.ndarray:
    """Boolean disc structuring element (same as realism._disk)."""
    r = int(max(1, round(rad)))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    return (yy * yy + xx * xx) <= r * r


def load_defects_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text())
    return dict(cfg.get("defects", cfg))


def simulate_holes(defects_cfg: dict[str, Any], scene_dir: Path, gt: np.ndarray,
                   meta: dict[str, Any], rng: np.random.Generator,
                   n_holes: int) -> list[dict[str, Any]]:
    """Sample hypothetical holes from a defects config on a real scene.

    Mirrors realism.apply_defects semantics: radius ~ U(hole_radius_px),
    depth ~ U(hole_depth_range) (default full depth), softness =
    hole_edge_softness_px (default hard), centre drawn from the structure
    interior eroded by hole_margin_px. Placement uses the saved hr_mask
    coverage (post-defect -- close enough for an input-side audit); falls
    back to a GT-level threshold when the mask is missing/empty."""
    radius_range = tuple(defects_cfg.get("hole_radius_px", (4, 13)))
    depth_range = tuple(defects_cfg.get("hole_depth_range", (1.0, 1.0)))
    softness = float(defects_cfg.get("hole_edge_softness_px", 0.0) or 0.0)
    margin = int(defects_cfg.get("hole_margin_px", 8))
    interior = None
    mask_path = scene_dir / "hr_mask_4x.png"
    if mask_path.exists():
        m = plt.imread(mask_path)
        if m.ndim == 3:
            m = m[..., 0]
        interior = ndimage.binary_erosion(m >= 0.5, _disk(margin))
        if not interior.any():
            interior = None
    if interior is None:
        t_bg = float(meta["T_bg_c"])
        dt = float(meta.get("delta_T_c", 1.0))
        interior = ndimage.binary_erosion(gt >= t_bg + 0.4 * dt, _disk(max(margin // 2, 2)))
        if not interior.any():
            return []
    iy, ix = np.where(interior)
    holes = []
    for _ in range(int(n_holes)):
        j = int(rng.integers(len(iy)))
        holes.append({"y": int(iy[j]), "x": int(ix[j]),
                      "radius_px": float(rng.uniform(*radius_range)),
                      "depth_frac": float(rng.uniform(*depth_range)),
                      "softness_px": softness, "context": "simulated"})
    return holes


# ---------------------------------------------------------------------------
# Noise estimation
# ---------------------------------------------------------------------------
def estimate_white_sigma(burst: np.ndarray) -> float:
    """Per-frame white (grain) sigma from well-separated frame differences.

    Burst-fixed components (vignette, stripes, 1/f, pixel FPN) cancel in the
    difference; AR(1) grain decorrelates at lag >= ~8 (rho<=0.7 -> rho^8 ~ 6%,
    < 3% sigma bias). Robust MAD estimator."""
    n = burst.shape[0]
    if n < 2:
        return float("nan")
    lag = max(1, min(n - 1, max(8, n // 3)))
    n_pairs = min(4, n - lag)
    diffs = [burst[i + lag] - burst[i] for i in range(n_pairs)]
    d = np.concatenate([x.ravel() for x in diffs])
    mad = np.median(np.abs(d - np.median(d)))
    return float(1.4826 * mad / np.sqrt(2.0))


def robust_std(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


# ---------------------------------------------------------------------------
# Analytic template chain (reuses the generator's own forward ops)
# ---------------------------------------------------------------------------
def hole_unit_profile(y0: int, x0: int, shape: tuple[int, int], hy: int, hx: int,
                      radius: float, softness: float, edge_sigma: float) -> np.ndarray:
    """Unit-amplitude HR perturbation profile of one hole on the patch grid
    [y0, y0+shape[0]) x [x0, x0+shape[1]) (global HR index coords).

    realism.py semantics: irregular_blob soft mask value = clip((r - d)/soft + 0.5, 0, 1)
    (0.5 ON the nominal boundary; hard disc when softness == 0), i.e. the idealized
    (harmonic-free) blob; the coverage drop depth_frac*w propagates through
    render_isothermal_field's gaussian_filter(edge_sigma) into the temperature field."""
    yy = (np.arange(y0, y0 + shape[0], dtype=np.float64) - hy)[:, None]
    xx = (np.arange(x0, x0 + shape[1], dtype=np.float64) - hx)[None, :]
    d = np.hypot(yy, xx)
    if softness > 0:
        w = np.clip((radius - d) / softness + 0.5, 0.0, 1.0)
    else:
        w = (d <= radius).astype(np.float64)
    if edge_sigma > 0:
        w = ndimage.gaussian_filter(w, float(edge_sigma))
    return w.astype(np.float32)


def hole_lr_position(y_hr: float, x_hr: float, dx: float, dy: float, scale: int) -> tuple[float, float]:
    """Fractional LR index of an HR position in one frame.

    Generator convention (tcforge.forward._block_average_from_blurred): LR pixel i of a
    frame with shift (dx, dy) averages HR samples scale*(i+dy)+{0..scale-1} (rows), so its
    effective block center sits at HR index scale*(i+dy) + (scale-1)/2. Inverting:
    i = (y_hr - (scale-1)/2)/scale - dy. (This is the +0.5 HR px block-center convention
    of ACL-049 -- inherited from the generator, not re-derived.)"""
    half = (scale - 1) / 2.0
    return (y_hr - half) / scale - dy, (x_hr - half) / scale - dx


class HoleTemplate:
    """HR patch of one hole's perturbation, aligned to the HR block grid, projected
    per-frame through the generator's PSF + shifted block-average."""

    def __init__(self, hy: int, hx: int, radius: float, softness: float, amp_c: float,
                 edge_sigma: float, psf_kwargs: dict[str, Any], scale: int,
                 shifts: np.ndarray, hr_shape: tuple[int, int]):
        self.scale = int(scale)
        sig_hr = max(float(psf_kwargs["psf_sigma_lr_px"]),
                     float(psf_kwargs.get("psf_sigma_y_lr_px") or 0.0)) * scale
        max_shift = float(np.max(np.abs(shifts))) if len(shifts) else 0.0
        r_reach = radius + softness / 2.0 + 4.0 * edge_sigma + 4.0 * max(sig_hr, 1.0)
        pad = int(np.ceil(r_reach + scale * (max_shift + 1.0))) + 2
        y0 = (int(hy) - pad) // scale * scale
        x0 = (int(hx) - pad) // scale * scale
        y1 = ((int(hy) + pad) // scale + 1) * scale
        x1 = ((int(hx) + pad) // scale + 1) * scale
        self.y0, self.x0 = y0, x0
        self.border_clipped = bool(y0 < 0 or x0 < 0 or y1 > hr_shape[0] or x1 > hr_shape[1])
        prof = hole_unit_profile(y0, x0, (y1 - y0, x1 - x0), int(hy), int(hx),
                                 radius, softness, edge_sigma)
        template_hr = (float(amp_c) * prof).astype(np.float64)
        self.peak_hr_c = float(template_hr.max())
        # The generator's own PSF operator (same params the scene was rendered with).
        self.blurred = apply_psf_blur(template_hr, scale=scale, mode="constant", cval=0.0,
                                      **psf_kwargs).astype(np.float64)

    def project(self, dx: float, dy: float) -> np.ndarray:
        """LR template patch for one frame (LR rows y0/scale .., generator's op)."""
        return _block_average_from_blurred(self.blurred, (dx, dy), scale=self.scale)

    def sample_at(self, t_lr: np.ndarray, iy: np.ndarray, ix: np.ndarray) -> np.ndarray:
        """Bilinear-sample a projected LR patch at global fractional LR coords."""
        coords = np.stack([iy - self.y0 / self.scale, ix - self.x0 / self.scale])
        return ndimage.map_coordinates(t_lr, coords, order=1, mode="constant", cval=0.0)


# ---------------------------------------------------------------------------
# Per-hole audit
# ---------------------------------------------------------------------------
def audit_hole(hole: dict[str, Any], gt: np.ndarray, burst: np.ndarray, shifts: np.ndarray,
               meta: dict[str, Any], sigma_white: float, edge_sigma: float,
               inject: bool = False) -> dict[str, Any]:
    """inject=True (simulate mode): the hole does not exist in the burst; its
    projected LR dip is subtracted from the real noisy crops before the
    empirical measurement (injected-recovery on physically real noise)."""
    scale = int(meta["scale"])
    hy, hx = int(hole["y"]), int(hole["x"])
    radius = float(hole["radius_px"])
    softness = float(hole["softness_px"])
    depth_frac = float(hole["depth_frac"])
    t_bg_c = float(meta["T_bg_c"])
    psf_kwargs = {
        "psf_sigma_lr_px": float(meta["psf_sigma_lr_px"]),
        "psf_shape": str(meta.get("psf_shape", "gaussian")),
        "psf_sigma_y_lr_px": (None if meta.get("psf_sigma_y_lr_px") is None
                              else float(meta["psf_sigma_y_lr_px"])),
        "psf_angle_deg": float(meta.get("psf_angle_deg", 0.0)),
    }
    rho = float(((meta.get("noise_parameters") or {}).get("grain_ar1_rho")) or 0.0)

    # GT-side measurement (probe口径) -> local structure level for the template amplitude.
    m_gt = measure_dot(gt, hy, hx, radius)
    bg_gt = m_gt["bg"]
    amp_c = depth_frac * max((bg_gt - t_bg_c) if np.isfinite(bg_gt) else 0.0, 0.0)

    row: dict[str, Any] = {
        "y": hy, "x": hx, "radius_px": radius, "depth_frac": depth_frac,
        "softness_px": softness, "context": hole.get("context"),
        "bg_gt_c": bg_gt, "depth_gt_meas_c": m_gt["depth"],
        "gt_edge_clipped": m_gt["edge_clipped"],
        "amp_model_c": amp_c, "sigma_white_c": sigma_white,
        "grain_ar1_rho": rho, "n_frames": int(burst.shape[0]),
    }

    tmpl = HoleTemplate(hy, hx, radius, softness, amp_c, edge_sigma, psf_kwargs, scale,
                        shifts, gt.shape)
    row["template_peak_hr_c"] = tmpl.peak_hr_c
    row["analytic_border_clipped"] = tmpl.border_clipped

    # Empirical crop geometry (LR px).
    psf_lr = max(float(psf_kwargs["psf_sigma_lr_px"]),
                 float(psf_kwargs.get("psf_sigma_y_lr_px") or 0.0))
    r_eff_lr = float(np.sqrt((radius / scale) ** 2 + psf_lr ** 2
                             + (edge_sigma / scale) ** 2 + (softness / (2 * scale)) ** 2))
    r_core = max(1.0, r_eff_lr)
    r_in, r_out = r_core + 2.0, r_core + 6.0
    hw = int(np.ceil(r_out)) + 1
    u = np.arange(-hw, hw + 1, dtype=np.float64)
    uu, vv = np.meshgrid(u, u, indexing="ij")
    rr = np.hypot(uu, vv)
    core_m, ann_m = rr <= r_core, (rr >= r_in) & (rr <= r_out)

    # Annulus structure mask: keep only annulus pixels whose GT neighbourhood is at
    # the hole's structure level (median over the LR block footprint). Guards the
    # empirical background/noise estimate against off-structure pixels for large or
    # edge holes (GT used for MASKING only, never for the measured values).
    if np.isfinite(bg_gt):
        gy = np.clip(hy + scale * uu, 0, gt.shape[0] - 1)
        gx = np.clip(hx + scale * vv, 0, gt.shape[1] - 1)
        gt_lvl = ndimage.uniform_filter(gt.astype(np.float64), size=scale)[
            gy.astype(int), gx.astype(int)]
        on_struct = gt_lvl >= t_bg_c + 0.5 * max(bg_gt - t_bg_c, 0.0)
        ann_struct = ann_m & on_struct
        if ann_struct.sum() >= max(12, int(0.25 * ann_m.sum())):
            ann_m = ann_struct

    h_lr, w_lr = burst.shape[1], burst.shape[2]
    cnr2_sum, cnr_singles = 0.0, []
    tpeak_lr, tl2_lr = [], []
    crop_sum = np.zeros_like(uu, dtype=np.float64)
    tmpl_sum = np.zeros_like(uu, dtype=np.float64)
    n_used = 0
    for f in range(burst.shape[0]):
        dx, dy = float(shifts[f, 0]), float(shifts[f, 1])
        t_lr = tmpl.project(dx, dy)
        l2 = float(np.sqrt(np.sum(t_lr.astype(np.float64) ** 2)))
        tl2_lr.append(l2)
        tpeak_lr.append(float(t_lr.max()))
        if np.isfinite(sigma_white) and sigma_white > 0:
            cnr_f = l2 / sigma_white
            cnr_singles.append(cnr_f)
            cnr2_sum += cnr_f * cnr_f
        # empirical: aligned crop around this frame's hole position
        iy_c, ix_c = hole_lr_position(hy, hx, dx, dy, scale)
        if not (hw <= iy_c <= h_lr - 1 - hw and hw <= ix_c <= w_lr - 1 - hw):
            continue
        coords = np.stack([iy_c + uu, ix_c + vv])
        data = ndimage.map_coordinates(burst[f], coords, order=1)
        sampled = tmpl.sample_at(t_lr, iy_c + uu, ix_c + vv)
        crop_sum += (data - sampled) if inject else data
        tmpl_sum += sampled
        n_used += 1

    row["cnr_analytic"] = float(np.sqrt(cnr2_sum)) if cnr_singles else float("nan")
    row["cnr_analytic_single_med"] = float(np.median(cnr_singles)) if cnr_singles else float("nan")
    row["cnr_analytic_ar1"] = row["cnr_analytic"] * float(np.sqrt((1 - rho) / (1 + rho)))
    row["template_peak_lr_c"] = float(np.mean(tpeak_lr)) if tpeak_lr else float("nan")
    row["template_l2_lr_mean"] = float(np.mean(tl2_lr)) if tl2_lr else float("nan")
    row["n_frames_used_emp"] = n_used

    if n_used == 0:
        row.update({"depth_emp_c": np.nan, "noise_emp_c": np.nan, "cnr_empirical": np.nan,
                    "amp_fit": np.nan, "reg_dy_lr_px": np.nan, "reg_dx_lr_px": np.nan})
        return row

    crop = crop_sum / n_used                    # aligned multi-frame mean (deg C)
    tbar = tmpl_sum / n_used                    # mean projected template on same grid
    bg_emp = float(np.median(crop[ann_m]))
    depth_emp = bg_emp - float(np.mean(crop[core_m]))
    noise_emp = robust_std(crop[ann_m])
    row["depth_emp_c"] = depth_emp
    row["noise_emp_c"] = noise_emp
    row["cnr_empirical"] = depth_emp / noise_emp if noise_emp > 0 else float("nan")

    dip = bg_emp - crop                          # positive where the hole darkens
    denom = float(np.sum(tbar * tbar))
    tmpl_meaningful = denom > 1e-12 and float(tbar.max()) > 1e-4  # >0.1 mC peak
    row["amp_fit"] = float(np.sum(dip * tbar) / denom) if tmpl_meaningful else float("nan")

    # sub-pixel registration offset data-dip vs template (only when localisable)
    if np.isfinite(row["cnr_empirical"]) and row["cnr_empirical"] >= 5.0 and tmpl_meaningful:
        best, best_v = (0, 0), -np.inf
        for oy in range(-3, 4):
            for ox in range(-3, 4):
                v = float(np.sum(np.roll(np.roll(dip, oy, 0), ox, 1) * tbar))
                if v > best_v:
                    best_v, best = v, (oy, ox)

        def _para(vm, v0, vp):
            d = vm - 2 * v0 + vp
            return 0.0 if abs(d) < 1e-12 else float(np.clip(0.5 * (vm - vp) / d, -1, 1))

        oy, ox = best
        cy = [float(np.sum(np.roll(np.roll(dip, oy + k, 0), ox, 1) * tbar)) for k in (-1, 0, 1)]
        cx = [float(np.sum(np.roll(np.roll(dip, oy, 0), ox + k, 1) * tbar)) for k in (-1, 0, 1)]
        # dip rolled by +o matches template <=> dip is at -o relative to template
        row["reg_dy_lr_px"] = -(oy + _para(*cy))
        row["reg_dx_lr_px"] = -(ox + _para(*cx))
    else:
        row["reg_dy_lr_px"], row["reg_dx_lr_px"] = np.nan, np.nan
    row["_crop"] = crop - bg_emp
    row["_tbar"] = tbar
    return row


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def tertile_bins(values: np.ndarray) -> list[str]:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0 or np.nanmax(finite) - np.nanmin(finite) < 1e-9:
        return ["all"] * len(arr)
    q1, q2 = np.nanquantile(arr, [1.0 / 3.0, 2.0 / 3.0])
    out = []
    for v in arr:
        if not np.isfinite(v):
            out.append("nan")
        elif v <= q1:
            out.append("lo")
        elif v <= q2:
            out.append("mid")
        else:
            out.append("hi")
    return out


def _frac_below(series: pd.Series, thresholds: list[float]) -> dict[str, float | None]:
    v = series.dropna()
    if len(v) == 0:
        return {f"{t:g}": None for t in thresholds}
    return {f"{t:g}": float((v < t).mean()) for t in thresholds}


def _cell(df: pd.DataFrame, thresholds: list[float]) -> dict[str, Any]:
    return {
        "n": int(len(df)),
        "median_cnr_analytic": (float(df["cnr_analytic"].median())
                                if df["cnr_analytic"].notna().any() else None),
        "median_cnr_analytic_ar1": (float(df["cnr_analytic_ar1"].median())
                                    if df["cnr_analytic_ar1"].notna().any() else None),
        "median_cnr_empirical": (float(df["cnr_empirical"].median())
                                 if df["cnr_empirical"].notna().any() else None),
        "median_amp_fit": (float(df["amp_fit"].median()) if df["amp_fit"].notna().any() else None),
        "frac_below_analytic": _frac_below(df["cnr_analytic"], thresholds),
        "frac_below_empirical": _frac_below(df["cnr_empirical"], thresholds),
    }


def build_summary(df: pd.DataFrame, thresholds: list[float], pool_dir: Path,
                  n_scenes: int, args_echo: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "pool_dir": str(pool_dir),
        "n_scenes_audited": n_scenes,
        "n_holes": int(len(df)),
        "cnr_thresholds": thresholds,
        "args": args_echo,
        "sigma_white_median_c": float(df["sigma_white_c"].median()) if len(df) else None,
        "overall": _cell(df, thresholds) if len(df) else None,
        "by_radius_bin": {},
        "by_radius_depth": {},
    }
    for rb, g in df.groupby("radius_bin_px"):
        summary["by_radius_bin"][str(int(rb))] = _cell(g, thresholds)
    for (rb, dt), g in df.groupby(["radius_bin_px", "depth_tertile"]):
        summary["by_radius_depth"][f"r{int(rb)}|{dt}"] = _cell(g, thresholds)
    return summary


# ---------------------------------------------------------------------------
# Figures (sequential single-hue ramp for the ordered radius bins / magnitude)
# ---------------------------------------------------------------------------
GRID_KW = {"color": "0.85", "lw": 0.6}


def _bin_colors(bins: list[int]) -> dict[int, Any]:
    cmap = plt.get_cmap("Blues")
    pos = np.linspace(0.45, 0.95, max(len(bins), 2))
    return {b: cmap(p) for b, p in zip(bins, pos)}


def fig_cnr_hist(df: pd.DataFrame, thresholds: list[float], path: Path) -> None:
    bins_r = sorted(df["radius_bin_px"].dropna().unique().astype(int))
    colors = _bin_colors(bins_r)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, col, title in zip(axes, ["cnr_analytic", "cnr_empirical"],
                              ["Analytic matched-filter CNR (multi-frame)",
                               "Empirical CNR (aligned mean, local annulus noise)"]):
        vals = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        vals = vals[vals > 0]
        if len(vals) == 0:
            ax.set_title(title + " (no data)")
            continue
        lo = min(0.1, float(vals.min()))
        hi = max(float(vals.max()), max(thresholds) * 2)
        edges = np.logspace(np.log10(max(lo, 1e-3)), np.log10(hi), 36)
        for rb in bins_r:
            v = df.loc[df["radius_bin_px"] == rb, col].dropna()
            v = v[v > 0]
            ax.hist(v, bins=edges, histtype="step", lw=1.8, color=colors[rb],
                    label=f"r≈{rb}px (n={len(v)})")
        for t in thresholds:
            ax.axvline(t, color="0.55", lw=0.9, ls="--", zorder=0)
            ax.text(t, ax.get_ylim()[1], f" {t:g}", color="0.45", fontsize=7,
                    ha="left", va="top")
        ax.set_xscale("log")
        ax.set_xlabel("CNR (log scale)")
        ax.set_title(title, fontsize=10)
        ax.grid(True, which="major", **GRID_KW)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("holes")
    axes[0].legend(fontsize=8, frameon=False, title="HR hole radius bin")
    fig.suptitle("Input-side hole detectability, stratified by radius (dashed = CNR thresholds)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_detectability_heatmap(df: pd.DataFrame, thresholds: list[float], path: Path) -> None:
    bins_r = sorted(df["radius_bin_px"].dropna().unique().astype(int))
    tert_order = [t for t in ["lo", "mid", "hi", "all"] if t in set(df["depth_tertile"])]
    n_t = len(thresholds)
    fig, axes = plt.subplots(2, n_t, figsize=(3.1 * n_t, 2.6 + 0.42 * len(bins_r)),
                             squeeze=False)
    for row_i, col in enumerate(["cnr_analytic", "cnr_empirical"]):
        for col_i, thr in enumerate(thresholds):
            ax = axes[row_i][col_i]
            mat = np.full((len(bins_r), len(tert_order)), np.nan)
            nmat = np.zeros_like(mat)
            for i, rb in enumerate(bins_r):
                for j, dt in enumerate(tert_order):
                    g = df[(df["radius_bin_px"] == rb) & (df["depth_tertile"] == dt)][col].dropna()
                    if len(g):
                        mat[i, j] = float((g >= thr).mean())
                        nmat[i, j] = len(g)
            im = ax.imshow(mat, vmin=0, vmax=1, cmap="Blues", aspect="auto")
            for i in range(len(bins_r)):
                for j in range(len(tert_order)):
                    if np.isfinite(mat[i, j]):
                        ax.text(j, i, f"{mat[i, j]:.2f}\nn={int(nmat[i, j])}",
                                ha="center", va="center", fontsize=7,
                                color="white" if mat[i, j] > 0.6 else "0.2")
            ax.set_xticks(range(len(tert_order)), tert_order, fontsize=8)
            ax.set_yticks(range(len(bins_r)), [f"r{b}" for b in bins_r], fontsize=8)
            if row_i == 0:
                ax.set_title(f"CNR ≥ {thr:g}", fontsize=9)
            if col_i == 0:
                ax.set_ylabel({"cnr_analytic": "analytic", "cnr_empirical": "empirical"}[col]
                              + "\nradius bin", fontsize=9)
            if row_i == 1:
                ax.set_xlabel("depth tertile", fontsize=8)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="detectable fraction")
    fig.suptitle("Detectable fraction (CNR ≥ threshold) vs radius bin × depth tertile", fontsize=11)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_hand_check(picked: list[dict[str, Any]], gt_cache: dict[str, np.ndarray], path: Path) -> None:
    """Side-by-side GT crop / aligned-mean LR data dip / mean projected template.
    Coordinate-convention guard: the data dip must sit at the template position
    (amp_fit ~ 1, |reg| < ~0.3 LR px for strong holes)."""
    if not picked:
        return
    fig, axes = plt.subplots(len(picked), 3, figsize=(7.2, 2.4 * len(picked)), squeeze=False)
    for i, row in enumerate(picked):
        gt = gt_cache[row["scene_id"]]
        hw_hr = max(10, int(np.ceil(3 * row["radius_px"] + 8)))
        sl = window_slices(int(row["y"]), int(row["x"]), hw_hr, gt.shape)
        gt_crop = gt[sl] - row["bg_gt_c"]
        vmax = max(row["amp_model_c"], 1e-3)
        panels = [
            (gt_crop, f"GT (HR) r={row['radius_px']:.1f} d={row['depth_frac']:.2f}"),
            (-row["_crop"], f"data dip (LR mean) CNRe={row['cnr_empirical']:.1f}"),
            (row["_tbar"], f"template CNRa={row['cnr_analytic']:.1f} fit={row['amp_fit']:.2f}"),
        ]
        for j, (img, title) in enumerate(panels):
            ax = axes[i][j]
            ax.imshow(-img if j == 0 else img, cmap="gray_r", vmin=-0.25 * vmax, vmax=vmax,
                      interpolation="nearest")
            cy, cx = ((int(row["y"]) - sl[0].start, int(row["x"]) - sl[1].start) if j == 0
                      else (img.shape[0] // 2, img.shape[1] // 2))
            r_draw = row["radius_px"] if j == 0 else max(1.0, row["radius_px"] / 2.0)
            ax.add_patch(plt.Circle((cx, cy), r_draw, fill=False, color="tab:red",
                                    lw=0.7, alpha=0.85))
            ax.set_title(title, fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                reg = (f"reg=({row['reg_dy_lr_px']:.2f},{row['reg_dx_lr_px']:.2f})px"
                       if np.isfinite(row.get("reg_dy_lr_px", np.nan)) else "reg=n/a")
                ax.set_ylabel(f"{row['scene_id']}#{row['hole_id']}\n{reg}", fontsize=6)
    fig.suptitle("Hand-check: GT hole vs actual LR dip vs projected template (red = hole radius)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool-dir", type=Path, required=True, help="pool root (contains scene_*/)")
    ap.add_argument("--max-scenes", type=int, default=None,
                    help="audit at most this many scenes (random sample; default all)")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--cnr-thresholds", default="1,2,3,5",
                    help="comma list of CNR thresholds (default 1,2,3,5)")
    ap.add_argument("--seed", type=int, default=20260709)
    ap.add_argument("--edge-sigma-hr", type=float, default=0.6,
                    help="isothermal-render edge_sigma in HR px (config "
                         "temperature_isothermal.edge_sigma; 0.6 for v6/v7 pools; "
                         "not recorded in metadata)")
    ap.add_argument("--hand-check-per-bin", type=int, default=2,
                    help="holes per radius bin on the hand-check crop board")
    ap.add_argument("--simulate-from-config", type=Path, default=None,
                    help="pool config json (or bare defects block) to SAMPLE hypothetical "
                         "holes from, instead of reading recorded instances -- for "
                         "legacy-recipe pools (v6) that never recorded hole coordinates. "
                         "Analytic CNR is exact; empirical CNR measures the template "
                         "injected into the real noisy burst.")
    ap.add_argument("--simulate-holes-per-scene", type=int, default=12,
                    help="hypothetical holes sampled per scene in simulate mode")
    args = ap.parse_args()

    thresholds = [float(t) for t in args.cnr_thresholds.split(",") if t.strip()]
    rng = np.random.default_rng(args.seed)
    scene_dirs = list_scene_dirs(args.pool_dir)
    if not scene_dirs:
        raise SystemExit(f"no scenes found under {args.pool_dir}")
    if args.max_scenes is not None and args.max_scenes < len(scene_dirs):
        idx = rng.choice(len(scene_dirs), size=args.max_scenes, replace=False)
        scene_dirs = [scene_dirs[i] for i in sorted(idx)]

    simulate = args.simulate_from_config is not None
    defects_cfg = load_defects_config(args.simulate_from_config) if simulate else None

    t0 = time.time()
    rows: list[dict[str, Any]] = []
    gt_cache: dict[str, np.ndarray] = {}
    n_no_holes = n_no_burst = n_no_gt = 0
    for k, scene_dir in enumerate(scene_dirs):
        meta = json.loads((scene_dir / "metadata.json").read_text())
        gt_path = scene_dir / "hr_temperature_2x.npy"
        if not gt_path.exists():
            n_no_gt += 1
            continue
        burst = load_burst(scene_dir)
        if burst is None:
            n_no_burst += 1
            continue
        gt = np.load(gt_path).astype(np.float32)
        if simulate:
            holes = simulate_holes(defects_cfg, scene_dir, gt, meta, rng,
                                   args.simulate_holes_per_scene)
        else:
            holes = read_holes(meta)
        if not holes:
            n_no_holes += 1
            continue
        shifts = np.load(scene_dir / "shifts.npy").astype(np.float64)  # columns [dx, dy]
        if shifts.shape[0] != burst.shape[0]:
            raise ValueError(f"{scene_dir.name}: shifts/burst frame count mismatch")
        sigma_white = estimate_white_sigma(burst)
        gt_cache[scene_dir.name] = gt
        for hole_id, hole in enumerate(holes):
            row = audit_hole(hole, gt, burst, shifts, meta, sigma_white, args.edge_sigma_hr,
                             inject=simulate)
            row.update({"scene_id": scene_dir.name, "hole_id": hole_id,
                        "mode": "simulated" if simulate else "instance",
                        "psf_sigma_lr_px": float(meta["psf_sigma_lr_px"]),
                        "psf_shape": str(meta.get("psf_shape", "gaussian")),
                        "noise_sigma_c_meta": float(meta.get("noise_sigma_c", np.nan)),
                        "delta_T_c": float(meta.get("delta_T_c", np.nan))})
            rows.append(row)
        if (k + 1) % 20 == 0:
            print(f"[audit] {k + 1}/{len(scene_dirs)} scenes, {len(rows)} holes, "
                  f"{time.time() - t0:.0f}s", flush=True)

    if not rows:
        raise SystemExit("no holes audited -- pool has no hole annotations "
                         "(need geometry_metadata.defects.hole_centers_yx or "
                         "defect_annotations.instances; for legacy-recipe pools "
                         "use --simulate-from-config)")

    df = pd.DataFrame(rows)
    crops = df[["_crop", "_tbar"]]
    df = df.drop(columns=["_crop", "_tbar"])
    df["radius_bin_px"] = df["radius_px"].round().clip(1, 16).astype(int)
    df["depth_tertile"] = tertile_bins(df["depth_frac"].to_numpy())

    args.outdir.mkdir(parents=True, exist_ok=True)
    front = ["scene_id", "hole_id", "y", "x", "radius_px", "radius_bin_px", "depth_frac",
             "depth_tertile", "softness_px", "context", "cnr_analytic", "cnr_analytic_ar1",
             "cnr_empirical", "amp_fit", "n_frames", "n_frames_used_emp"]
    df = df[front + [c for c in df.columns if c not in front]]
    df.to_csv(args.outdir / "per_hole.csv", index=False)

    n_scenes = df["scene_id"].nunique()
    args_echo = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    summary = build_summary(df, thresholds, args.pool_dir, n_scenes, args_echo)
    summary["mode"] = "simulated_from_config" if simulate else "recorded_instances"
    summary["scenes_skipped"] = {"no_holes": n_no_holes, "no_gt": n_no_gt,
                                 "no_burst": n_no_burst}
    summary["runtime_s"] = round(time.time() - t0, 1)
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    fig_cnr_hist(df, thresholds, args.outdir / "cnr_hist_by_radius.png")
    fig_detectability_heatmap(df, thresholds, args.outdir / "detectability_heatmap.png")

    # hand-check board: strongest empirical holes per radius bin (localisable),
    # plus the per-bin median hole, so weak bins are also eyeballed.
    picked: list[dict[str, Any]] = []
    df_c = df.join(crops)
    for rb in sorted(df_c["radius_bin_px"].unique()):
        cell = df_c[(df_c["radius_bin_px"] == rb) & df_c["cnr_empirical"].notna()]
        if not len(cell):
            continue
        take = cell.sort_values("cnr_empirical", ascending=False).head(args.hand_check_per_bin)
        picked.extend(take.to_dict("records"))
    fig_hand_check(picked, gt_cache, args.outdir / "hand_check_crops.png")

    ov = summary["overall"]
    print(f"[audit] pool={args.pool_dir}")
    print(f"[audit] {n_scenes} scenes, {len(df)} holes; skipped: {summary['scenes_skipped']}")
    print(f"[audit] median CNR analytic={ov['median_cnr_analytic']:.2f} "
          f"(AR1-corr {ov['median_cnr_analytic_ar1']:.2f}) "
          f"empirical={ov['median_cnr_empirical']:.2f} amp_fit={ov['median_amp_fit']:.2f}")
    for t in thresholds:
        print(f"[audit] frac below CNR {t:g}: analytic={ov['frac_below_analytic'][f'{t:g}']:.3f} "
              f"empirical={ov['frac_below_empirical'][f'{t:g}']:.3f}")
    print(f"[audit] outputs in {args.outdir} ({summary['runtime_s']}s)")


if __name__ == "__main__":
    main()
