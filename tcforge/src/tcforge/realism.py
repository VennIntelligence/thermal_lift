"""Realism augmentations for the synthetic training pool.

Three structure-aware degradations validated against real IR frames
(data/data_raw/infrared_avi) on 2026-06-26 — see research_log/synthetic_data_realism.md:

  - apply_defects:           irregular holes / broken edges / jagged cracks (> pitch => recoverable)
  - render_isothermal_field: near-isothermal per-structure temperature (connected metal => one
                             level; NO thickness->temperature coupling)
  - field_noise_burst:       detector noise = smooth vignette + smooth column-stripe FPN (BOTH
                             fixed across the burst) + per-frame fine grain.

Parameters are meant to be drawn from WIDE ranges (cross-detector generality), not pinned to one
local calibration.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def _disk(rad: float) -> np.ndarray:
    r = int(max(1, round(rad)))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    return (yy * yy + xx * xx) <= r * r


def irregular_blob(shape, cy, cx, radius_px, rng, irregularity=0.45, n_harm=5,
                   edge_softness_px=0.0) -> np.ndarray:
    """Mask of a smooth irregular closed shape (radial-harmonic boundary).
    Used for drilled holes, broken corners and edge fragments (subtractive).

    Default (`edge_softness_px=0`): hard boolean mask `rr <= boundary` (legacy, bit-exact).
    `edge_softness_px > 0`: float32 mask with a linear radial transition of total width
    `edge_softness_px` centred on the nominal boundary (value 1 inside, 0.5 ON the
    boundary, 0 outside) — the 0.5-level set stays at the hard-edge boundary, so
    softness does not shrink the blob. RNG draws are identical in both modes."""
    rows, cols = shape
    ks = rng.integers(2, 7, size=n_harm).astype(np.float64)
    amps = rng.uniform(0.0, irregularity, size=n_harm) / ks
    phs = rng.uniform(0.0, 2 * np.pi, size=n_harm)
    yy, xx = np.ogrid[:rows, :cols]
    dy = yy - cy
    dx = xx - cx
    theta = np.arctan2(dy, dx)
    rr = np.sqrt(dy * dy + dx * dx)
    boundary = np.full(shape, float(radius_px), dtype=np.float32)
    for a, k, p in zip(amps, ks, phs):
        boundary = boundary + radius_px * a * np.sin(k * theta + p)
    np.clip(boundary, 0.3 * radius_px, None, out=boundary)
    if edge_softness_px and float(edge_softness_px) > 0.0:
        soft = (boundary - rr) / float(edge_softness_px) + 0.5
        return np.clip(soft, 0.0, 1.0).astype(np.float32)
    return rr <= boundary


def jagged_crack(shape, start, length_px, width_px, rng, jaggedness=0.45, branch_p=0.25) -> np.ndarray:
    """Boolean mask of a jagged thin crack (random-walk centerline, dilated, occasional branch)."""
    rows, cols = shape
    line = np.zeros(shape, dtype=bool)

    def _walk(y, x, ang, ln):
        for _ in range(int(ln)):
            iy, ix = int(round(y)), int(round(x))
            if 0 <= iy < rows and 0 <= ix < cols:
                line[iy, ix] = True
            else:
                return
            ang += rng.normal(0.0, jaggedness)
            y += np.sin(ang)
            x += np.cos(ang)
            if rng.random() < branch_p / max(1.0, ln):
                _walk(y, x, ang + rng.uniform(-1.0, 1.0), ln * 0.4)

    _walk(float(start[0]), float(start[1]), rng.uniform(0, 2 * np.pi), length_px)
    return ndimage.binary_dilation(line, structure=_disk(width_px * 0.5))


def apply_defects(coverage, rng, *, severity_range=(0.3, 1.0),
                  hole_radius_px=(4, 13), notch_radius_px=(3, 10),
                  crack_len_px=(60, 260), crack_width_px=(2, 4),
                  max_holes=6, max_notches=6, max_cracks=4,
                  hole_depth_range=(1.0, 1.0), hole_edge_softness_px=0.0,
                  min_holes=0):
    """Subtract irregular holes (interior), broken edges (boundary) and jagged cracks from a
    coverage mask. All defects are > pitch (recoverable). Per-scene `severity` randomizes counts.
    Returns (defected_coverage, meta).

    Hole-only knobs (ACL-063, A1-dots; notches/cracks untouched):
      - hole_depth_range: per-hole depth d ~ U(range); coverage drops by d*mask instead of the
        legacy full-depth zeroing. Default (1.0, 1.0) => no depth draw, legacy behaviour.
      - hole_edge_softness_px: > 0 => soft (float) hole edges, linear transition of this total
        width centred on the blob boundary (see irregular_blob). Default 0.0 => hard boolean.
      - min_holes: lower bound on the hole count draw, rng.integers(min_holes, ceil+1);
        clamped to ceil (= round(max_holes*severity)) when larger, clamp recorded in meta.
    All three are orthogonal to `severity` (which only scales count ceilings). With all three
    at defaults the output AND the RNG stream are bit-identical to the legacy implementation
    (no extra rng draws on the default path) — pinned by the golden-sample test."""
    cov = np.asarray(coverage, dtype=np.float32)
    struct = cov > 0.5
    interior = ndimage.binary_erosion(struct, _disk(8))
    boundary = struct & ~ndimage.binary_erosion(struct, _disk(2))
    iy, ix = np.where(interior)
    by, bx = np.where(boundary)
    sy, sx = np.where(struct)
    defect = np.zeros(cov.shape, dtype=bool)                 # notches + cracks (hard, legacy)
    hole_removal = np.zeros(cov.shape, dtype=np.float32)     # holes: depth-weighted, maybe soft
    counts = {"holes": 0, "notches": 0, "cracks": 0}
    sev = float(rng.uniform(*severity_range))

    depth_lo, depth_hi = float(hole_depth_range[0]), float(hole_depth_range[1])
    depth_active = not (depth_lo == 1.0 and depth_hi == 1.0)
    softness = float(hole_edge_softness_px)
    holes_custom = depth_active or softness > 0.0 or int(min_holes) != 0
    hole_ceil = int(round(max_holes * sev))
    hole_floor = max(0, min(int(min_holes), hole_ceil))
    hole_depths: list[float] = []
    hole_radii: list[float] = []
    hole_centers_yx: list[list[int]] = []
    for _ in range(int(rng.integers(hole_floor, hole_ceil + 1))):
        if len(iy) == 0:
            break
        j = rng.integers(len(iy))
        radius = float(rng.uniform(*hole_radius_px))
        # Depth draw ONLY when non-default, so the default RNG stream is unchanged.
        depth = float(rng.uniform(depth_lo, depth_hi)) if depth_active else 1.0
        blob = irregular_blob(cov.shape, iy[j], ix[j], radius, rng,
                              edge_softness_px=softness)
        np.maximum(hole_removal, depth * blob.astype(np.float32), out=hole_removal)
        counts["holes"] += 1
        hole_radii.append(round(radius, 4))
        # Center pixel (HR-grid coords, same frame as `coverage`/hr_mask) — no extra RNG
        # draw, so this is safe to record unconditionally within the holes_custom branch.
        hole_centers_yx.append([int(iy[j]), int(ix[j])])
        if depth_active:
            hole_depths.append(round(depth, 4))
    for _ in range(int(rng.integers(0, round(max_notches * sev) + 1))):
        if len(by) == 0:
            break
        j = rng.integers(len(by))
        defect |= irregular_blob(cov.shape, by[j], bx[j], rng.uniform(*notch_radius_px), rng, irregularity=0.7)
        counts["notches"] += 1
    for _ in range(int(rng.integers(0, round(max_cracks * sev) + 1))):
        if len(sy) == 0:
            break
        j = rng.integers(len(sy))
        defect |= jagged_crack(cov.shape, (sy[j], sx[j]), rng.uniform(*crack_len_px),
                               rng.uniform(*crack_width_px), rng)
        counts["cracks"] += 1

    removal = np.maximum(hole_removal, defect.astype(np.float32))
    defected = (cov * (1.0 - removal)).astype(np.float32)
    counts["severity"] = round(sev, 4)
    # New meta keys only when a non-default hole knob is active: keeps the default-path
    # meta dict (and hence the golden sample) exactly identical to the legacy one.
    if holes_custom:
        counts["hole_radii"] = hole_radii
        counts["hole_edge_softness_px"] = softness
        counts["min_holes"] = int(min_holes)
        counts["min_holes_effective"] = hole_floor
        counts["hole_centers_yx"] = hole_centers_yx
        if depth_active:
            counts["hole_depths"] = hole_depths
    return defected, counts


def render_isothermal_field(mask, rng, *, t_bg_c=22.0, delta_t_c=2.6, level_min=0.82,
                            edge_sigma=1.4, low_freq_amplitude_c=0.0, low_freq_sigma_px=96.0):
    """Near-isothermal temperature field: each connected structure at ~one level (modest spread
    `[level_min, 1.0]`), soft edges (coverage-weighted + small Gaussian). Optional gentle low-freq
    scene background. No thickness->temperature coupling — connected metal is ~isothermal."""
    cov = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    struct = cov >= 0.5
    lbl, n = ndimage.label(struct)
    lvl = np.zeros(cov.shape, dtype=np.float32)
    if n > 0:
        per = rng.uniform(float(level_min), 1.0, size=n).astype(np.float32)
        lut = np.zeros(n + 1, dtype=np.float32)
        lut[1:] = per
        lvl = lut[lbl] * cov                                   # soft edges preserved via coverage
    lvl = ndimage.gaussian_filter(lvl, float(edge_sigma))
    field = float(t_bg_c) + float(delta_t_c) * lvl
    if low_freq_amplitude_c and low_freq_amplitude_c > 0:
        nz = ndimage.gaussian_filter(rng.normal(size=cov.shape).astype(np.float32), float(low_freq_sigma_px))
        nz -= float(nz.mean())
        nz /= (float(np.max(np.abs(nz))) + 1e-6)
        field = field + float(low_freq_amplitude_c) * nz
    return field.astype(np.float32)


def field_noise_burst(burst, rng, *, vignette_c=0.13, stripe_c=0.028,
                      stripe_col_sigma=(2.5, 5.0), grain_c=0.10):
    """Detector noise on an LR burst (M,h,w): a smooth vignette + a smooth low-amplitude column
    stripe FPN (BOTH fixed across the burst, so multi-frame averaging does NOT remove them) plus
    per-frame fine Gaussian grain. Returns the noisy burst (same shape)."""
    burst = np.asarray(burst, dtype=np.float32)
    squeeze = burst.ndim == 2
    if squeeze:
        burst = burst[None]
    m, h, w = burst.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = rng.uniform(0.35, 0.65) * h, rng.uniform(0.35, 0.65) * w
    rad = np.hypot((yy - cy) / h, (xx - cx) / w)
    large = -(rad ** 2) + 0.4 * ((xx / w - 0.5) * rng.uniform(-1, 1) + (yy / h - 0.5) * rng.uniform(-1, 1))
    large = (large - large.mean()) / (large.std() + 1e-6)
    csig = float(rng.uniform(*stripe_col_sigma)) if isinstance(stripe_col_sigma, (tuple, list)) else float(stripe_col_sigma)
    col = ndimage.gaussian_filter(rng.normal(size=(1, w)).astype(np.float32), sigma=(0, csig))
    stripe = (col - col.mean()) / (col.std() + 1e-6)
    fixed = (float(vignette_c) * large + float(stripe_c) * stripe).astype(np.float32)    # (h,w), fixed
    grain = rng.normal(0.0, float(grain_c), size=burst.shape).astype(np.float32)         # per-frame
    out = (burst + fixed[None] + grain).astype(np.float32)
    return out[0] if squeeze else out
