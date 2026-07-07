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


def _soft_disc(shape, cy, cx, r_px, soft_px) -> np.ndarray:
    """Windowed soft disc (value 1 inside r_px, linear ramp of width soft_px, 0 outside).
    Ported verbatim from the v7 prototype (temperature-layer defect weight). No RNG."""
    out = np.zeros(shape, dtype=np.float32)
    R = int(np.ceil(r_px + soft_px)) + 2
    y0, y1 = max(cy - R, 0), min(cy + R + 1, shape[0])
    x0, x1 = max(cx - R, 0), min(cx + R + 1, shape[1])
    if y1 <= y0 or x1 <= x0:
        return out
    yy = np.arange(y0, y1)[:, None] - cy
    xx = np.arange(x0, x1)[None, :] - cx
    d = np.hypot(yy, xx)
    out[y0:y1, x0:x1] = np.clip((r_px + soft_px / 2 - d) / max(soft_px, 1e-3), 0.0, 1.0)
    return out


def _ndimage_rotate_point_forward(y, x, cy, cx, angle_deg):
    """Forward map (input point -> output location) of ``ndimage.rotate(reshape=False)``.

    ndimage.rotate samples output[o] = input[R@(o-c)+c] with R=[[c,s],[-s,c]] (rows,cols;
    c=cos, s=sin). Hence the forward map is o = c + R^{-1}@(i-c) with R^{-1}=[[c,-s],[s,c]].
    Sign convention pinned by test_carve_trace_breaks_rotation_mapping (calibrated against
    an actual ndimage.rotate of a delta image), NOT hand-derived from memory."""
    t = np.radians(float(angle_deg))
    c, s = float(np.cos(t)), float(np.sin(t))
    dy, dx = float(y) - float(cy), float(x) - float(cx)
    return float(cy) + c * dy - s * dx, float(cx) + s * dy + c * dx


def _ndimage_rotate_vec_forward(dy, dx, angle_deg):
    """Forward map of a (row, col) direction vector under ndimage.rotate (no translation)."""
    t = np.radians(float(angle_deg))
    c, s = float(np.cos(t)), float(np.sin(t))
    return c * float(dy) - s * float(dx), s * float(dy) + c * float(dx)


def _classify_context(struct: np.ndarray, y: int, x: int, win: int = 12) -> str:
    """Defect-context class {isolated|embedded|edge|background} at (y,x) on the pre-etch
    struct. Isolation口径 mirrors the dots-pilot / audit dot_strata classifier (win=12,
    background fraction < 5% => isolated-on-uniform). No RNG. NOTE: the historical
    scripts/probe_dot_retention.py referenced in the plan is not present in-repo; the
    equivalent classifier in scripts/audit_v7_demo_gates.py::dot_strata is used as the
    authoritative口径."""
    h, w = struct.shape
    yi, xi = int(y), int(x)
    if not (0 <= yi < h and 0 <= xi < w) or not bool(struct[yi, xi]):
        return "background"
    y0, y1 = max(yi - win, 0), min(yi + win + 1, h)
    x0, x1 = max(xi - win, 0), min(xi + win + 1, w)
    patch = struct[y0:y1, x0:x1]
    if patch.size == 0:
        return "background"
    return "isolated" if float((~patch).mean()) < 0.05 else "embedded"


def _di(rng, lo, hi) -> int:
    """Inclusive integer draw in [lo, hi] (mirrors the v7 prototype _di)."""
    return int(rng.integers(int(lo), int(hi) + 1))


def _oriented_rect_mask(shape, cy, cx, d_along, half_along, half_across) -> np.ndarray:
    """Boolean mask of a rotated rectangle centred at (cy,cx): extent +-half_along along the
    unit-ish direction d_along=(dy,dx), +-half_across across it. Windowed (no full-canvas cost)."""
    h, w = shape
    dy, dx = float(d_along[0]), float(d_along[1])
    norm = float(np.hypot(dy, dx)) or 1.0
    dy, dx = dy / norm, dx / norm
    py, px = -dx, dy                                    # perpendicular
    reach = int(np.ceil(max(float(half_along), float(half_across)))) + 2
    y0, y1 = max(int(round(cy)) - reach, 0), min(int(round(cy)) + reach + 1, h)
    x0, x1 = max(int(round(cx)) - reach, 0), min(int(round(cx)) + reach + 1, w)
    out = np.zeros(shape, dtype=bool)
    if y1 <= y0 or x1 <= x0:
        return out
    yy = np.arange(y0, y1)[:, None] - float(cy)
    xx = np.arange(x0, x1)[None, :] - float(cx)
    along = yy * dy + xx * dx
    across = yy * py + xx * px
    out[y0:y1, x0:x1] = (np.abs(along) <= float(half_along)) & (np.abs(across) <= float(half_across))
    return out


def _instance(iid, type_, stage, cy, cx, *, radius_px=None, depth_or_amplitude=None,
              edge_softness_px=None, length_px=None, width_px=None, gap_px=None,
              context=None, trace_index=None, area_px=0) -> dict:
    """One defect-annotation record (metadata schema_version 2, §3.1)."""
    return {
        "id": int(iid), "type": str(type_), "stage": str(stage),
        "center_yx_hr": [int(cy), int(cx)],
        "radius_px": None if radius_px is None else round(float(radius_px), 4),
        "depth_or_amplitude": None if depth_or_amplitude is None else round(float(depth_or_amplitude), 4),
        "edge_softness_px": None if edge_softness_px is None else round(float(edge_softness_px), 4),
        "length_px": None if length_px is None else round(float(length_px), 4),
        "width_px": None if width_px is None else round(float(width_px), 4),
        "gap_px": None if gap_px is None else round(float(gap_px), 4),
        "context": context, "trace_index": None if trace_index is None else int(trace_index),
        "area_px": int(area_px),
    }


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
                  min_holes=0,
                  # ── v7 integration §3.4/§3.1 ──
                  hole_margin_px=8, hole_margin_adaptive=False, hole_edge_fraction=0.0,
                  min_notches=0,
                  record_instances=False, label_map=None, next_id=1):
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

    v7 integration knobs (all default => legacy byte-identical, no extra RNG draws):
      - hole_margin_px: interior erosion radius (default 8 == legacy _disk(8)).
      - hole_margin_adaptive: when True AND the eroded interior is empty, shrink the margin
        along a deterministic ladder [margin, max(margin//2,3), 2, 1] until non-empty (fixes
        the silent-punch-through on thin-line scenes). Changes draw count ONLY when it fires,
        i.e. never on the default path. Effective margin recorded when it shrinks.
      - hole_edge_fraction: > 0 => each hole draws rng.random(); if < fraction the centre is
        sampled from the boundary band (struct & ~interior) and tagged context="edge". Guarded
        by `if fraction > 0`, so the default path draws nothing extra.
      - record_instances / label_map / next_id: when record_instances, meta gains "instances"
        (schema_version-2 records for every hole/notch/crack) and "next_id"; when label_map is
        not None the >=0.5 level set of each shape (clipped to pre-etch struct) is painted with a
        dense id (later shapes overwrite earlier). Painting consumes NO RNG.

    With every knob at its default the output, meta dict AND the RNG stream are bit-identical to
    the legacy implementation — pinned by the golden-sample test."""
    cov = np.asarray(coverage, dtype=np.float32)
    struct = cov > 0.5
    margin = int(hole_margin_px)
    interior = ndimage.binary_erosion(struct, _disk(margin))
    margin_effective = margin
    margin_shrunk = False
    if bool(hole_margin_adaptive) and (not interior.any()) and struct.any():
        # Deterministic shrink ladder (no RNG); only reached when adaptive=True (non-default).
        for m in (margin, max(margin // 2, 3), 2, 1):
            cand = ndimage.binary_erosion(struct, _disk(m))
            if cand.any():
                interior, margin_effective, margin_shrunk = cand, m, (m != margin)
                break
    boundary = struct & ~ndimage.binary_erosion(struct, _disk(2))
    iy, ix = np.where(interior)
    by, bx = np.where(boundary)
    sy, sx = np.where(struct)
    fraction = float(hole_edge_fraction)
    ey, ex = np.where(struct & ~interior) if fraction > 0.0 else (iy[:0], ix[:0])
    defect = np.zeros(cov.shape, dtype=bool)                 # notches + cracks (hard, legacy)
    hole_removal = np.zeros(cov.shape, dtype=np.float32)     # holes: depth-weighted, maybe soft
    counts = {"holes": 0, "notches": 0, "cracks": 0}
    instances: list[dict] = []
    nid = int(next_id)
    sev = float(rng.uniform(*severity_range))

    def _paint(shape_bool: np.ndarray) -> tuple[np.ndarray, int]:
        """Region (shape >=0.5, clipped to struct); paint into label_map if present."""
        region = shape_bool & struct
        area = int(region.sum())
        nonlocal nid
        this_id = nid
        if label_map is not None and area:
            label_map[region] = this_id
        nid += 1
        return region, area

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
        # Edge-band decision draw ONLY when hole_edge_fraction > 0 (guarded => default unchanged).
        if fraction > 0.0 and len(ey) > 0 and rng.random() < fraction:
            j = int(rng.integers(len(ey)))
            hy, hx = int(ey[j]), int(ex[j])
            ctx = "edge"
        else:
            j = int(rng.integers(len(iy)))
            hy, hx = int(iy[j]), int(ix[j])
            ctx = None
        radius = float(rng.uniform(*hole_radius_px))
        # Depth draw ONLY when non-default, so the default RNG stream is unchanged.
        depth = float(rng.uniform(depth_lo, depth_hi)) if depth_active else 1.0
        blob = irregular_blob(cov.shape, hy, hx, radius, rng, edge_softness_px=softness)
        np.maximum(hole_removal, depth * blob.astype(np.float32), out=hole_removal)
        counts["holes"] += 1
        hole_radii.append(round(radius, 4))
        # Center pixel (HR-grid coords, same frame as `coverage`/hr_mask) — no extra RNG
        # draw, so this is safe to record unconditionally within the holes_custom branch.
        hole_centers_yx.append([hy, hx])
        if depth_active:
            hole_depths.append(round(depth, 4))
        if record_instances:
            region, area = _paint(blob >= 0.5)
            if ctx is None:
                ctx = _classify_context(struct, hy, hx)
            instances.append(_instance(nid - 1, "hole", "coverage", hy, hx,
                                       radius_px=radius, depth_or_amplitude=depth,
                                       edge_softness_px=softness, context=ctx, area_px=area))
    # min_notches floor (default 0 => rng.integers(0, ...) == legacy => byte-identical;
    # v7 sets a floor so every scene carries >=1 edge notch, matching the prototype's 2-5).
    notch_ceil = round(max_notches * sev)
    notch_floor = max(0, min(int(min_notches), int(notch_ceil)))
    for _ in range(int(rng.integers(notch_floor, notch_ceil + 1))):
        if len(by) == 0:
            break
        j = rng.integers(len(by))
        nr = rng.uniform(*notch_radius_px)
        nblob = irregular_blob(cov.shape, by[j], bx[j], nr, rng, irregularity=0.7)
        defect |= nblob
        counts["notches"] += 1
        if record_instances:
            region, area = _paint(nblob)
            instances.append(_instance(nid - 1, "notch", "coverage", int(by[j]), int(bx[j]),
                                       radius_px=float(nr), depth_or_amplitude=1.0,
                                       context="edge", area_px=area))
    for _ in range(int(rng.integers(0, round(max_cracks * sev) + 1))):
        if len(sy) == 0:
            break
        j = rng.integers(len(sy))
        clen = rng.uniform(*crack_len_px)
        cwid = rng.uniform(*crack_width_px)
        cblob = jagged_crack(cov.shape, (sy[j], sx[j]), clen, cwid, rng)
        defect |= cblob
        counts["cracks"] += 1
        if record_instances:
            region, area = _paint(cblob)
            instances.append(_instance(nid - 1, "crack", "coverage", int(sy[j]), int(sx[j]),
                                       length_px=float(clen), width_px=float(cwid),
                                       depth_or_amplitude=1.0,
                                       context=_classify_context(struct, int(sy[j]), int(sx[j])),
                                       area_px=area))

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
        # no-silent-caps (§3.4): domain ran empty before the min_holes floor was met.
        if hole_floor > counts["holes"]:
            counts["holes_shortfall"] = int(hole_floor - counts["holes"])
    if margin_shrunk:
        counts["hole_margin_effective_px"] = int(margin_effective)
    if record_instances:
        counts["instances"] = instances
        counts["next_id"] = int(nid)
    return defected, counts


def _apply_zone_levels(rng, per, lbl, n, cov, zones, rot_deg, jitter, hr_pitch, level_min):
    """Overwrite the per-component level of components whose centroid falls inside a zone
    with a shared per-zone base (+ small jitter). Ports the v7 prototype level_render zone
    grouping. Consumes RNG (len(zones)+1 uniforms + one normal per zoned component); called
    ONLY when zones is non-empty."""
    h, w = cov.shape
    binm = cov >= 0.5
    zmap = np.zeros((h, w), dtype=np.int16)
    for zi, z in enumerate(zones, start=1):
        cy, cx = float(z["cy_um"]), float(z["cx_um"])
        zh, zw = float(z["h_um"]), float(z["w_um"])
        y0 = max(int((cy - zh / 2) / hr_pitch), 0)
        y1 = min(int((cy + zh / 2) / hr_pitch) + 1, h)
        x0 = max(int((cx - zw / 2) / hr_pitch), 0)
        x1 = min(int((cx + zw / 2) / hr_pitch) + 1, w)
        if y1 > y0 and x1 > x0:
            zmap[y0:y1, x0:x1] = zi
    if abs(float(rot_deg)) > 1e-6:
        zmap = ndimage.rotate(zmap, float(rot_deg), reshape=False, order=0, mode="constant")
    base = rng.uniform(float(level_min), 1.0, size=len(zones) + 1).astype(np.float32)
    cents = ndimage.center_of_mass(binm, lbl, index=np.arange(1, n + 1))
    out = per.copy()
    for li, (cyc, cxc) in enumerate(cents):        # per[li] <-> component (li+1)
        if np.isnan(cyc):
            continue
        zi = int(zmap[int(round(cyc)), int(round(cxc))])
        if zi:
            out[li] = float(np.clip(base[zi] + rng.normal(0.0, float(jitter)),
                                    float(level_min), 1.0))
    return out


def render_isothermal_field(mask, rng, *, t_bg_c=22.0, delta_t_c=2.6, level_min=0.82,
                            edge_sigma=1.4, low_freq_amplitude_c=0.0, low_freq_sigma_px=96.0,
                            zones=None, zone_rotation_deg=0.0, zone_level_jitter=0.03,
                            hr_pitch_um=None, stratified_anchor=False):
    """Near-isothermal temperature field: each connected structure at ~one level (modest spread
    `[level_min, 1.0]`), soft edges (coverage-weighted + small Gaussian). Optional gentle low-freq
    scene background. No thickness->temperature coupling — connected metal is ~isothermal.

    v7 integration §3.6: when `zones` (list of {cy_um,cx_um,h_um,w_um,...}) is non-empty, the
    components whose centroid falls in a zone are re-levelled to a shared per-zone base (rotate
    the zone raster by `zone_rotation_deg` — the scene rotation — to align with the already-rotated
    mask; `hr_pitch_um` converts zone um->px). `zones=None` (or empty) is byte-identical to the
    legacy path with zero extra RNG draws — pinned by isothermal_golden_v1.npz."""
    cov = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    struct = cov >= 0.5
    lbl, n = ndimage.label(struct)
    lvl = np.zeros(cov.shape, dtype=np.float32)
    if n > 0:
        per = rng.uniform(float(level_min), 1.0, size=n).astype(np.float32)
        if stratified_anchor:
            # Guarantee per-scene bright/dark spread (v7 G7 / prototype level_render): among
            # sizeable components pin one to the dark end and one to the bright end. Gated =>
            # default draws nothing extra. per[k] <-> component (k+1).
            areas = np.bincount(lbl.ravel(), minlength=n + 1)
            big = np.where(areas[1:] >= 50)[0]
            if len(big) >= 2:
                pick = rng.choice(len(big), size=2, replace=False)
                per[big[pick[0]]] = float(level_min) + float(rng.uniform(0.0, 0.05))
                per[big[pick[1]]] = 1.0 - float(rng.uniform(0.0, 0.05))
        if zones:
            if hr_pitch_um is None:
                raise ValueError("render_isothermal_field: hr_pitch_um required when zones given")
            per = _apply_zone_levels(rng, per, lbl, n, cov, zones,
                                     zone_rotation_deg, zone_level_jitter,
                                     float(hr_pitch_um), float(level_min))
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


def carve_trace_breaks(coverage, traces, rng, *, scene_rotation_deg, canvas_center_yx,
                       hr_pitch_um, break_p=0.7, count_range=(1, 2), gap_px=(6.0, 20.0),
                       width_pad_px=1.0, record_instances=False, label_map=None, next_id=1):
    """Carve 1-2 broken-trace gaps into a POST-rotation HR coverage using the composer trace
    look-up table (§3.3). Each trace is stored in pre-rotation um with an `angle_deg` (0 for the
    main cluster, the part angle for the rotated secondary). A gap centre is chosen along the
    trace's long axis (+-0.35 length), mapped through the scene rotation about `canvas_center_yx`
    (HR px) — `_ndimage_rotate_point_forward` sign convention pinned by test — and an oriented
    gap rectangle (gap_px along, trace_width+2*pad across) is zeroed. Returns
    (coverage, instances, next_id). Gated by the caller: default config never calls this."""
    cov = np.asarray(coverage, dtype=np.float32).copy()
    instances: list[dict] = []
    nid = int(next_id)
    if not traces:
        return cov, instances, nid
    if rng.random() >= float(break_p):
        return cov, instances, nid
    h, w = cov.shape
    cy0, cx0 = float(canvas_center_yx[0]), float(canvas_center_yx[1])
    hr = float(hr_pitch_um)
    for _ in range(_di(rng, *count_range)):
        j = int(rng.integers(0, len(traces)))
        t = traces[j]
        h_um, w_um = float(t["h_um"]), float(t["w_um"])
        cy_um, cx_um = float(t["cy_um"]), float(t["cx_um"])
        ang = float(t.get("angle_deg", 0.0))
        length_um = max(h_um, w_um)
        width_um = min(h_um, w_um)
        ldir = (0.0, 1.0) if w_um >= h_um else (1.0, 0.0)   # long axis in pre-rotation (row,col)
        if abs(ang) > 1e-9:                                  # part (secondary) composite angle
            ldir = _ndimage_rotate_vec_forward(ldir[0], ldir[1], ang)
        s = length_um * float(rng.uniform(-0.35, 0.35))
        gcy_um = cy_um + s * ldir[0]
        gcx_um = cx_um + s * ldir[1]
        oy, ox = _ndimage_rotate_point_forward(gcy_um / hr, gcx_um / hr, cy0, cx0,
                                               scene_rotation_deg)
        d_along = _ndimage_rotate_vec_forward(ldir[0], ldir[1], scene_rotation_deg)
        gp = float(rng.uniform(*gap_px))
        width_px = width_um / hr
        region = _oriented_rect_mask((h, w), oy, ox, d_along, gp / 2.0,
                                     (width_px + 2.0 * float(width_pad_px)) / 2.0)
        cov[region] = 0.0
        if record_instances:
            if label_map is not None and region.any():
                label_map[region] = nid
            instances.append(_instance(nid, "broken_trace", "trace_break",
                                       int(round(oy)), int(round(ox)), gap_px=gp,
                                       width_px=width_px, length_px=gp, trace_index=j,
                                       context="edge", area_px=int(region.sum())))
            nid += 1
    return cov, instances, nid


def apply_thermal_defects(field, coverage, rng, *, t_bg_c, delta_t_c,
                          hot_spot_count=(2, 8), hot_radius_px=(1.0, 4.0), hot_amp_frac=(0.3, 1.0),
                          hot_edge_softness_px=1.0, hot_on_structure_p=0.7,
                          dark_blob_p=0.6, dark_blob_count=(1, 2), dark_blob_radius_px=(8.0, 16.0),
                          dark_blob_depth=(0.15, 0.40), dark_blob_edge_softness_px=3.0,
                          record_instances=False, label_map=None, next_id=1):
    """Temperature-layer defects on the HR GT field (§3.2), ported from the prototype temp_render:
      - dark blobs (on structure): T -= depth * max(T - T_bg, 0) * soft_disc  (never below T_bg).
      - hot spots (70% on structure / 30% background): T += amp * dT * soft_disc.
    These are SCENE-layer features (blurred by the PSF, land in hr_temperature GT), strictly
    distinct from detector_defects (single-pixel, detector layer). Returns (field, instances,
    next_id). Called ONLY when config thermal_defects.enabled — default absent => never invoked
    => zero RNG displacement."""
    out = np.asarray(field, dtype=np.float32).copy()
    cov = np.asarray(coverage, dtype=np.float32)
    struct = cov >= 0.5
    h, w = out.shape
    tbg = float(t_bg_c)
    on = np.argwhere(cov > 0.7)
    instances: list[dict] = []
    nid = int(next_id)

    def _record(kind, y, x, r_px, amp_or_depth, soft, weight):
        nonlocal nid
        if not record_instances:
            return
        reg = weight >= 0.5
        if label_map is not None and reg.any():
            label_map[reg] = nid
        instances.append(_instance(nid, kind, "thermal", int(y), int(x), radius_px=r_px,
                                   depth_or_amplitude=amp_or_depth, edge_softness_px=soft,
                                   context=_classify_context(struct, int(y), int(x)),
                                   area_px=int(reg.sum())))
        nid += 1

    # dark blobs first (matches prototype temp_render order), gated by dark_blob_p on structure
    if len(on) and rng.random() < float(dark_blob_p):
        for _ in range(_di(rng, *dark_blob_count)):
            y, x = on[int(rng.integers(0, len(on)))]
            r_px = float(rng.uniform(*dark_blob_radius_px))
            depth = float(rng.uniform(*dark_blob_depth))
            weight = _soft_disc(out.shape, int(y), int(x), r_px, float(dark_blob_edge_softness_px))
            out -= depth * np.maximum(out - tbg, 0.0) * weight
            _record("dark_blob", y, x, r_px, depth, float(dark_blob_edge_softness_px), weight)

    for _ in range(_di(rng, *hot_spot_count)):
        if len(on) and rng.random() < float(hot_on_structure_p):
            y, x = on[int(rng.integers(0, len(on)))]
        else:
            y = int(rng.integers(h // 5, 4 * h // 5))
            x = int(rng.integers(w // 5, 4 * w // 5))
        r_px = float(rng.uniform(*hot_radius_px))
        amp = float(rng.uniform(*hot_amp_frac))
        weight = _soft_disc(out.shape, int(y), int(x), r_px, float(hot_edge_softness_px))
        out += amp * float(delta_t_c) * weight
        _record("hot_spot", int(y), int(x), r_px, amp, float(hot_edge_softness_px), weight)

    return out, instances, nid


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
