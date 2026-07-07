#!/usr/bin/env python3
"""v7 content-axis demo — prototype composer, round 4 (owner review).

r3 -> r4 changes (owner verdict 2026-07-08): hollows are RIGHT, but —
  * Voids must not be EMPTY: every hollow-frame interior / window cutout gets
    1-3 spanning traces (thick 60-150um AND thin 28-60um mixed) crossing the
    void, connecting rim to rim (and to the nested island when present).
    (This also fixes a latent r3 bug: nested islands were killed by the
    add&~sub semantics; islands now go through well_with_elements.)
  * NO large plain-solid panels — "全实心训练的就是没有". The "solid" role is
    replaced by "lined": 2-6 carved lines of varied width/length; only panels
    smaller than ~0.9mm may stay plain. XL high-occ scenes use lined/textured
    panels (thin carvings keep occupancy high without plain slabs).

Kept from r3: connected multi-panel assembly (varied-size rectangles grown by
edge adjacency; ~25% merged, rest separated by 40-160um dark streets; thick
bridges / thin buses across ~55% of streets; a few long L-traces; unbridged
panels stay separate isothermal components => bright/dark variety).
Texture panels (comb/pads/fins) capped to <= 1/3 of panels.

Kept from r2: no low tier (mid + high only), defects rendered (small dark dots
r1-4px/depth.3-1, hot spots, shallow dark blobs, irregular edge notches,
broken traces), FULL-CHAIN sheets (isothermal levels U(0.6,1) -> PSF blur
0.15-0.55 LR px -> 2x block average -> vignette + column stripes + grain),
~25% secondary part at independent orientation, full U(0,360) rotation,
28um/32um floors, audit-metric occupancy.

Prototype only: local rasteriser + lightweight forward, no tcforge integration,
no RNG discipline. After sign-off this becomes a tcforge motif family behind
config knobs (golden-pinned defaults) and the pilot runs the REAL pipeline.

Usage: uv run python scripts/v7_composer_demo.py \
           [--out research_log/assets/v7_planning/composer_demo_r4]
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

# ── geometry constants (match the training pool) ─────────────────────────────
HR_SHAPE = (960, 1280)          # HR px, 10um/px (pixel 20um, scale 2)
LR_SHAPE = (480, 640)
PIXEL_SIZE_UM = 20.0
SCALE = 2
SSAA = 3                        # demo AA (pool uses 4; enough for eyeballing)
UPD = PIXEL_SIZE_UM / (SCALE * SSAA)     # um per draw px
UM_PER_HR = PIXEL_SIZE_UM / SCALE        # 10um per HR px
DRAW_SHAPE = (HR_SHAPE[0] * SSAA, HR_SHAPE[1] * SSAA)
W_UM = HR_SHAPE[1] * PIXEL_SIZE_UM / SCALE   # 12800
H_UM = HR_SHAPE[0] * PIXEL_SIZE_UM / SCALE   # 9600
DISC_R_UM = min(W_UM, H_UM) / 2.0            # inscribe-disc radius

FLOOR = 28.0                    # finest feature (um), red line
PITCH_FLOOR = 32.0              # array pitch floor (um), red line
LEVEL_MIN = 0.60                # owner-approved bright/dark widening (v6: 0.82)


# ── low-level stamps (windowed, um coordinates, axis-aligned) ─────────────────
def _window(cy: float, cx: float, h: float, w: float):
    y0 = max(int((cy - h / 2) / UPD) - 1, 0)
    y1 = min(int((cy + h / 2) / UPD) + 2, DRAW_SHAPE[0])
    x0 = max(int((cx - w / 2) / UPD) - 1, 0)
    x1 = min(int((cx + w / 2) / UPD) + 2, DRAW_SHAPE[1])
    if y1 <= y0 or x1 <= x0:
        return None
    yy = (np.arange(y0, y1, dtype=np.float32) + 0.5) * UPD - cy
    xx = (np.arange(x0, x1, dtype=np.float32) + 0.5) * UPD - cx
    return (slice(y0, y1), slice(x0, x1)), yy[:, None], xx[None, :]


def stamp_rect(target: np.ndarray, cy, cx, h, w) -> None:
    r = _window(cy, cx, h, w)
    if r is None:
        return
    win, yy, xx = r
    target[win] |= ((np.abs(yy) <= h / 2) & (np.abs(xx) <= w / 2)).astype(np.uint8)


def stamp_disc(target: np.ndarray, cy, cx, d) -> None:
    r = _window(cy, cx, d, d)
    if r is None:
        return
    win, yy, xx = r
    target[win] |= ((yy ** 2 + xx ** 2) <= (d / 2) ** 2).astype(np.uint8)


@dataclass
class Canvas:
    """add/sub composition; final = add & ~sub (v6 semantics). Bright elements
    inside carved wells must be EXCLUDED from sub, never re-added."""
    add: np.ndarray = field(default_factory=lambda: np.zeros(DRAW_SHAPE, dtype=np.uint8))
    sub: np.ndarray = field(default_factory=lambda: np.zeros(DRAW_SHAPE, dtype=np.uint8))

    def rect(self, cy, cx, h, w, mode="add"):
        stamp_rect(self.add if mode == "add" else self.sub, cy, cx, h, w)

    def disc(self, cy, cx, d, mode="add"):
        stamp_disc(self.add if mode == "add" else self.sub, cy, cx, d)

    def well_with_elements(self, cy, cx, zh, zw, elements: np.ndarray):
        well = np.zeros(DRAW_SHAPE, dtype=np.uint8)
        stamp_rect(well, cy, cx, zh, zw)
        self.sub |= well & (1 - elements)
        self.add |= elements

    def final(self) -> np.ndarray:
        return (self.add & ~self.sub).astype(np.uint8)


# ── composer helpers ──────────────────────────────────────────────────────────
def _u(rng, lo, hi):
    return float(rng.uniform(lo, hi))


def _di(rng, lo, hi):
    return int(rng.integers(lo, hi + 1))


@dataclass
class SceneMeta:
    tier: str
    zones: list = field(default_factory=list)      # (cy, cx, h, w, kind) um
    sparse_pts: list = field(default_factory=list)  # panel centres um
    blocks: list = field(default_factory=list)      # panel bboxes (y, x, h, w) um
    traces: list = field(default_factory=list)      # bright trace rects (y,x,h,w)
    notch_pts: list = field(default_factory=list)   # carved edge-bite centres um
    break_pts: list = field(default_factory=list)   # broken-trace gap centres um
    secondary: bool = False


def _pad_zone(rng, cv: Canvas, cy, cx, zh, zw, *, on_die: bool,
              w_floor: float = FLOOR):
    """Jittered pad field; on-panel pads sit in a carved cool well."""
    pad = _u(rng, w_floor, 46.0)
    pitch = pad + _u(rng, FLOOR, 46.0)             # gap >= FLOOR by construction
    jit = pitch * _u(rng, 0.02, 0.07)
    drop = _u(rng, 0.04, 0.16)
    shape_round = rng.random() < 0.5
    nr = max(int((zh - pad) // pitch), 2)
    nc = max(int((zw - pad) // pitch), 2)
    y0 = cy - (nr - 1) * pitch / 2
    x0 = cx - (nc - 1) * pitch / 2
    elements = np.zeros(DRAW_SHAPE, dtype=np.uint8) if on_die else None
    target = elements if on_die else cv.add
    for i in range(nr):
        for j in range(nc):
            if rng.random() < drop:
                continue
            py = y0 + i * pitch + rng.normal(0.0, jit)
            px = x0 + j * pitch + rng.normal(0.0, jit)
            if shape_round:
                stamp_disc(target, py, px, pad)
            else:
                stamp_rect(target, py, px, pad, pad)
    if on_die:
        cv.well_with_elements(cy, cx, zh, zw, elements)


def _comb_patch(rng, cv: Canvas, cy, cx, zh, zw, horiz: bool,
                w_floor: float = FLOOR):
    """Interlocking carved routing comb over a region (panel stays connected).
    Channel width mixes thin and THICK (up to ~80um)."""
    cw = _u(rng, w_floor, 44.0) if rng.random() < 0.7 else _u(rng, 44.0, 80.0)
    pitch = cw + _u(rng, FLOOR + 6, 96.0)
    span, run = (zh, zw) if horiz else (zw, zh)
    n = max(int(span // pitch), 2)
    offs = (np.arange(n) - (n - 1) / 2) * pitch
    for i, off in enumerate(offs):
        frac = _u(rng, 0.6, 0.95)
        side = -1.0 if i % 2 == 0 else 1.0
        if horiz:
            cv.rect(cy + off, cx + side * (run - run * frac) / 2, cw, run * frac, mode="sub")
        else:
            cv.rect(cy + side * (run - run * frac) / 2, cx + off, run * frac, cw, mode="sub")


def _fin_zone(rng, cv: Canvas, cy, cx, zh, zw, w_floor: float = FLOOR):
    fw = _u(rng, w_floor, 46.0)
    pitch = fw + _u(rng, FLOOR, 52.0)
    vertical = rng.random() < 0.5
    span = zw if vertical else zh
    n = max(int((span - fw) // pitch), 2)
    offs = (np.arange(n) - (n - 1) / 2) * pitch
    elements = np.zeros(DRAW_SHAPE, dtype=np.uint8)
    for off in offs:
        if rng.random() < 0.06:
            continue
        o = off + rng.normal(0.0, pitch * 0.03)
        ln = (zh if vertical else zw) * _u(rng, 0.82, 0.94)
        if vertical:
            stamp_rect(elements, cy, cx + o, ln, fw)
        else:
            stamp_rect(elements, cy + o, cx, fw, ln)
    cv.well_with_elements(cy, cx, zh, zw, elements)


def _edge_pad_row(rng, cv: Canvas, meta: "SceneMeta", panel, edge: int):
    """Single carved pad row hugging one panel edge (bond-pad ring look)."""
    dy, dx, dh, dw = panel
    pad = _u(rng, 34.0, 56.0)
    pitch = pad + _u(rng, FLOOR, 44.0)
    strip = pad + 2 * FLOOR
    inset = strip / 2 + FLOOR
    elements = np.zeros(DRAW_SHAPE, dtype=np.uint8)
    if edge in (0, 1):
        cy = dy + (dh / 2 - inset) * (1 if edge else -1)
        run = dw * 0.86
        n = max(int(run // pitch), 3)
        for j in range(n):
            if rng.random() < 0.05:
                continue
            px = dx - (n - 1) * pitch / 2 + j * pitch + rng.normal(0, pitch * 0.03)
            stamp_rect(elements, cy, px, pad, pad)
        cv.well_with_elements(cy, dx, strip, run, elements)
        meta.zones.append((cy, dx, strip, run, "edge_pads"))
    else:
        cx = dx + (dw / 2 - inset) * (1 if edge == 3 else -1)
        run = dh * 0.86
        n = max(int(run // pitch), 3)
        for j in range(n):
            if rng.random() < 0.05:
                continue
            py = dy - (n - 1) * pitch / 2 + j * pitch + rng.normal(0, pitch * 0.03)
            stamp_rect(elements, py, cx, pad, pad)
        cv.well_with_elements(dy, cx, run, strip, elements)
        meta.zones.append((dy, cx, run, strip, "edge_pads"))


def _boxes_clear(y, x, bh, bw, boxes, frac=0.25) -> bool:
    for (oy, ox, oh, ow) in boxes:
        iy = max(0.0, min(y + bh / 2, oy + oh / 2) - max(y - bh / 2, oy - oh / 2))
        ix = max(0.0, min(x + bw / 2, ox + ow / 2) - max(x - bw / 2, ox - ow / 2))
        if iy * ix > frac * min(bh * bw, oh * ow):
            return False
    return True


# ── the r3/r4 core: connected multi-panel assembly ────────────────────────────
PANEL_ROLES = ("lined", "frame", "windowed", "textured")


def _trace_w(rng, w_floor: float = FLOOR):
    """Mixed-thickness trace width: 60% thin (28-60um), 40% thick (60-150um)."""
    return _u(rng, w_floor, 60.0) if rng.random() < 0.6 else _u(rng, 60.0, 150.0)


def _void_with_traces(rng, cv: Canvas, meta: SceneMeta, vy, vx, vh, vw,
                      overhang: float, island: bool, w_floor: float = FLOOR):
    """Carve a void of (vh, vw) but keep 1-3 spanning traces (thick+thin mixed)
    crossing it — rim-to-rim connections so no void is ever empty (owner r3
    verdict). Optional nested island (回-shape), wired through the same path."""
    elements = np.zeros(DRAW_SHAPE, dtype=np.uint8)
    if island:
        ih = vh * _u(rng, 0.28, 0.50)
        iw = vw * _u(rng, 0.28, 0.50)
        stamp_rect(elements, vy, vx, ih, iw)
    for _ in range(_di(rng, 1, 3)):
        tw_ = _trace_w(rng, w_floor)
        if rng.random() < 0.5:                     # horizontal crossing
            ty = vy + vh * _u(rng, -0.35, 0.35)
            stamp_rect(elements, ty, vx, tw_, vw + 2 * overhang)
            meta.traces.append((ty, vx, tw_, vw))
        else:                                      # vertical crossing
            tx = vx + vw * _u(rng, -0.35, 0.35)
            stamp_rect(elements, vy, tx, vh + 2 * overhang, tw_)
            meta.traces.append((vy, tx, vh, tw_))
    cv.well_with_elements(vy, vx, vh, vw, elements)


def _lined_panel(rng, cv: Canvas, y, x, h, w, w_floor: float = FLOOR):
    """Structured 'solid' panel: carved lines of varied width/length, count
    scaled with panel area so big panels never read as plain slabs (owner:
    全实心训练的就是没有). Same-orientation lines keep >= FLOOR bright
    separation (gate G1: random offsets were pinching sub-28um slivers)."""
    n_lines = int(np.clip(round(h * w / 2.0e6 * _u(rng, 2.0, 4.0)), 3, 14))
    placed_h: list[tuple[float, float]] = []       # (offset_um, halfwidth_um)
    placed_v: list[tuple[float, float]] = []
    for _ in range(n_lines):
        sw = _u(rng, w_floor, 90.0)
        horiz = rng.random() < 0.5
        placed = placed_h if horiz else placed_v
        span = (h if horiz else w)
        for _try in range(8):
            off = span * _u(rng, -0.38, 0.38)
            if all(abs(off - o) >= hw + sw / 2 + FLOOR for o, hw in placed):
                break
        else:
            continue
        placed.append((off, sw / 2))
        if horiz:
            cv.rect(y + off, x + w * _u(rng, -0.20, 0.20),
                    sw, w * _u(rng, 0.35, 0.90), mode="sub")
        else:
            cv.rect(y + h * _u(rng, -0.20, 0.20), x + off,
                    h * _u(rng, 0.35, 0.90), sw, mode="sub")
    if min(h, w) > 1600.0 and rng.random() < 0.6:  # big panel: add a window
        wh, ww = h * _u(rng, 0.18, 0.30), w * _u(rng, 0.18, 0.30)
        cv.rect(y + h * _u(rng, -0.25, 0.25), x + w * _u(rng, -0.25, 0.25),
                wh, ww, mode="sub")


def _apply_panel_role(rng, cv: Canvas, meta: SceneMeta, y, x, h, w, role: str,
                      w_floor: float = FLOOR):
    """Render one panel. Hollow voids keep spanning traces; texture confined;
    every large panel carries internal structure. w_floor > FLOOR is used by
    the rotated secondary part (order-0 rotation staircase must not push a
    28um feature below the band floor)."""
    cv.rect(y, x, h, w, mode="add")
    small = min(h, w) < 900.0                      # small blocks may stay plain
    if role == "frame":                            # bright ring + crossed VOID
        b = min(_u(rng, 50.0, 130.0), min(h, w) * 0.25)
        iv_h, iv_w = h - 2 * b, w - 2 * b
        if iv_h > 3 * FLOOR and iv_w > 3 * FLOOR:
            island = rng.random() < 0.45 and min(iv_h, iv_w) > 6 * FLOOR
            _void_with_traces(rng, cv, meta, y, x, iv_h, iv_w, b, island,
                              w_floor=w_floor)
    elif role == "windowed":                       # 1-2 BIG crossed windows
        placed_w: list[tuple[float, float, float, float]] = []
        for _ in range(_di(rng, 1, 2)):
            wh = h * _u(rng, 0.30, 0.52)
            ww = w * _u(rng, 0.30, 0.52)
            for _try in range(8):                  # keep >= FLOOR bright strip
                wy = y + (h - wh) * _u(rng, -0.35, 0.35)
                wx = x + (w - ww) * _u(rng, -0.35, 0.35)
                if all(abs(wy - oy) >= (wh + oh) / 2 + FLOOR
                       or abs(wx - ox) >= (ww + ow) / 2 + FLOOR
                       for oy, ox, oh, ow in placed_w):
                    break
            else:
                continue
            placed_w.append((wy, wx, wh, ww))
            _void_with_traces(rng, cv, meta, wy, wx, wh, ww, 40.0,
                              island=bool(rng.random() < 0.15
                                          and min(wh, ww) > 6 * FLOOR),
                              w_floor=w_floor)
        if rng.random() < 0.5 and not small:       # plus a couple of lines
            _lined_panel(rng, cv, y, x, h, w, w_floor=w_floor)
    elif role == "textured":                       # regional texture, ONE panel
        zh, zw = h * _u(rng, 0.70, 0.85), w * _u(rng, 0.70, 0.85)
        kind = rng.choice(["comb", "pads", "fins"], p=[0.5, 0.3, 0.2])
        if kind == "comb":
            _comb_patch(rng, cv, y, x, zh, zw, horiz=bool(rng.random() < 0.5),
                        w_floor=w_floor)
        elif kind == "pads":
            _pad_zone(rng, cv, y, x, zh, zw, on_die=True, w_floor=w_floor)
            meta.zones.append((y, x, zh, zw, "pads"))
        else:
            _fin_zone(rng, cv, y, x, zh, zw, w_floor=w_floor)
            meta.zones.append((y, x, zh, zw, "fins"))
    else:                                          # lined (replaces plain solid)
        if not small or rng.random() < 0.5:
            _lined_panel(rng, cv, y, x, h, w, w_floor=w_floor)


def _cluster_scene(rng, cv: Canvas, meta: SceneMeta, *, tier: str,
                   force_xl: bool = False):
    """Connected multi-panel assembly — the r3 scene core."""
    if force_xl:                                   # G3 capability tier: occ>0.40
        K, s_lo, s_hi, g_hi = _di(rng, 10, 14), 0.22, 0.40, 50.0
    elif tier == "high":
        K, s_lo, s_hi, g_hi = _di(rng, 6, 9), 0.14, 0.34, 160.0
    else:
        K, s_lo, s_hi, g_hi = _di(rng, 3, 5), 0.08, 0.20, 160.0

    shrink = 1.0

    def _size():
        return (H_UM * _u(rng, s_lo, s_hi) * shrink,
                W_UM * _u(rng, s_lo, s_hi) * shrink)

    panels: list[tuple] = []
    adjac: list[tuple] = []                        # (i, j, gap, side)
    if force_xl:                                   # anchor XL with one big slab
        h0, w0 = H_UM * _u(rng, 0.60, 0.70), W_UM * _u(rng, 0.60, 0.70)
    else:
        h0, w0 = _size()
    panels.append((H_UM / 2 + H_UM * _u(rng, -0.06, 0.06),
                   W_UM / 2 + W_UM * _u(rng, -0.06, 0.06), h0, w0))
    fails = 0
    while len(panels) < K and fails < 8:
        placed = False
        for _try in range(30):
            i = int(rng.integers(0, len(panels)))
            py, px, ph, pw = panels[i]
            hh, ww = _size()
            # touch/merge (tiny overlap), else a dark street between panels;
            # XL merges more often (footprint capability, G3)
            merge_p = 0.50 if force_xl else 0.25
            gap = (-10.0 if rng.random() < merge_p else _u(rng, 40.0, g_hi))
            side = int(rng.integers(0, 4))
            if side in (0, 1):                     # above / below
                off_max = (pw + ww) / 2 - max(0.3 * min(pw, ww), 200.0)
                ny = py + ((ph + hh) / 2 + gap) * (-1 if side == 0 else 1)
                nx = px + _u(rng, -off_max, off_max)
            else:                                  # left / right
                off_max = (ph + hh) / 2 - max(0.3 * min(ph, hh), 200.0)
                nx = px + ((pw + ww) / 2 + gap) * (-1 if side == 2 else 1)
                ny = py + _u(rng, -off_max, off_max)
            # allow mild disc clipping of panel corners (0.7x circumradius),
            # same acceptance as v6's big dies
            if np.hypot(ny - H_UM / 2, nx - W_UM / 2) + 0.7 * np.hypot(hh, ww) / 2 \
                    > DISC_R_UM * 0.95:
                continue
            others = ([p for k2, p in enumerate(panels) if k2 != i]
                      if gap < 0 else panels)
            if not _boxes_clear(ny, nx, hh, ww, others, frac=0.02):
                continue
            panels.append((ny, nx, hh, ww))
            adjac.append((i, len(panels) - 1, gap, side))
            placed = True
            break
        if placed:
            fails = 0
            shrink = 1.0
        else:                                      # retry smaller, don't give up
            fails += 1
            shrink = max(shrink * 0.85, 0.45)

    # recentre the grown cluster on the canvas (adjacency growth wanders),
    # capped so every panel stays inside the inscribe disc
    cy_m = float(np.mean([p[0] for p in panels]))
    cx_m = float(np.mean([p[1] for p in panels]))
    sy, sx = H_UM / 2 - cy_m, W_UM / 2 - cx_m
    for scale in (1.0, 0.7, 0.4, 0.2, 0.0):
        ok = all(np.hypot(p[0] + sy * scale - H_UM / 2,
                          p[1] + sx * scale - W_UM / 2)
                 + 0.7 * np.hypot(p[2], p[3]) / 2
                 <= DISC_R_UM * 0.97 for p in panels)
        if ok:
            panels = [(p[0] + sy * scale, p[1] + sx * scale, p[2], p[3])
                      for p in panels]
            break

    roles = [str(rng.choice(PANEL_ROLES, p=[0.30, 0.20, 0.25, 0.25]))
             for _ in panels]
    if not any(r in ("frame", "windowed") for r in roles):
        roles[int(rng.integers(0, len(roles)))] = "windowed"
    if "textured" not in roles:                    # G5: >=1 dense-texture panel
        lined = [k for k, r in enumerate(roles) if r == "lined"]
        pick = (lined[int(rng.integers(0, len(lined)))] if lined
                else int(rng.integers(0, len(roles))))
        roles[pick] = "textured"
    tex_idx = [k for k, r in enumerate(roles) if r == "textured"]
    for k in tex_idx[max(1, len(panels) // 3):]:   # texture = minority accent
        roles[k] = "lined"
    if force_xl:                                   # XL high-occ: lined/textured
        # only (thin carvings keep area high; NO plain slabs — owner r3 verdict)
        roles = [(r if r == "textured" else "lined") for r in roles]
    for (py, px, ph, pw), role in zip(panels, roles):
        _apply_panel_role(rng, cv, meta, py, px, ph, pw, role)
        meta.blocks.append((py, px, ph, pw))
        meta.sparse_pts.append((py, px))

    # bridges / thin buses across the dark streets (thick AND thin lines).
    # Deliberately NOT all pairs: unbridged street => separate isothermal
    # component => bright/dark variety survives (owner: some panels dark)
    for (i, j, gap, side) in adjac:
        if gap <= 0 or rng.random() > 0.55:
            continue
        yi, xi, hi_, wi = panels[i]
        yj, xj, hj, wj = panels[j]
        span = gap + 60.0                          # reach 30um into both panels
        if side in (0, 1):                         # street runs horizontally
            lo = max(xi - wi / 2, xj - wj / 2)
            hi = min(xi + wi / 2, xj + wj / 2)
            if hi - lo < 120.0:
                continue
            cy_s = (yi + (hi_ / 2) * (-1 if side == 0 else 1)
                    + yj + (hj / 2) * (1 if side == 0 else -1)) / 2
            if rng.random() < 0.6:                 # one thick bridge
                bw_ = _u(rng, 60.0, 180.0)
                bx = _u(rng, lo + bw_ / 2, hi - bw_ / 2)
                cv.rect(cy_s, bx, span, bw_, mode="add")
                meta.traces.append((cy_s, bx, span, bw_))
            else:                                  # thin bus, 2-4 traces
                n_b = _di(rng, 2, 4)
                tw_ = _u(rng, FLOOR, 50.0)
                for b in range(n_b):
                    bx = lo + (b + 1) * (hi - lo) / (n_b + 1)
                    cv.rect(cy_s, bx, span, tw_, mode="add")
                    meta.traces.append((cy_s, bx, span, tw_))
        else:                                      # street runs vertically
            lo = max(yi - hi_ / 2, yj - hj / 2)
            hi = min(yi + hi_ / 2, yj + hj / 2)
            if hi - lo < 120.0:
                continue
            cx_s = (xi + (wi / 2) * (-1 if side == 2 else 1)
                    + xj + (wj / 2) * (1 if side == 2 else -1)) / 2
            if rng.random() < 0.6:
                bw_ = _u(rng, 60.0, 180.0)
                by = _u(rng, lo + bw_ / 2, hi - bw_ / 2)
                cv.rect(by, cx_s, bw_, span, mode="add")
                meta.traces.append((by, cx_s, bw_, span))
            else:
                n_b = _di(rng, 2, 4)
                tw_ = _u(rng, FLOOR, 50.0)
                for b in range(n_b):
                    by = lo + (b + 1) * (hi - lo) / (n_b + 1)
                    cv.rect(by, cx_s, tw_, span, mode="add")
                    meta.traces.append((by, cx_s, tw_, span))

    # long-range L-traces between distant panels — thick and thin mixed
    # (few: every trace merges its endpoints into one isothermal component)
    for _ in range(_di(rng, 0, 3)):
        if len(panels) < 2:
            break
        i, j = rng.choice(len(panels), size=2, replace=False)
        yi, xi, *_r1 = panels[int(i)]
        yj, xj, *_r2 = panels[int(j)]
        if (abs(yi - yj) + abs(xi - xj)) < 300.0:
            continue
        tw_ = _u(rng, FLOOR, 60.0) if rng.random() < 0.6 else _u(rng, 60.0, 150.0)
        cv.rect(yi, (xi + xj) / 2, tw_, abs(xj - xi) + tw_, mode="add")
        cv.rect((yi + yj) / 2, xj, abs(yj - yi) + tw_, tw_, mode="add")
        meta.traces.append((yi, (xi + xj) / 2, tw_, abs(xj - xi) + tw_))
        meta.traces.append(((yi + yj) / 2, xj, abs(yj - yi) + tw_, tw_))

    if tier == "high" and rng.random() < 0.4:      # bond-pad row, largest panel
        big_p = max(panels, key=lambda p: p[2] * p[3])
        _edge_pad_row(rng, cv, meta, big_p, edge=int(rng.integers(0, 4)))


def _secondary_part(rng, main: Canvas, meta: SceneMeta):
    """ONE smaller sub-assembly at an independent orientation: two connected
    mini-panels (r3 style), rotated as a whole about its own centre."""
    part = Canvas()
    pmeta = SceneMeta(tier="sec")
    fr = _u(rng, 0.10, 0.16)
    ph, pw = H_UM * fr, W_UM * fr * _u(rng, 0.8, 1.2)
    ref = meta.blocks if meta.blocks else [(H_UM / 2, W_UM / 2, 0, 0)]
    ref_y = float(np.mean([b[0] for b in ref]))
    ref_x = float(np.mean([b[1] for b in ref]))
    phi = np.arctan2(H_UM / 2 - ref_y, W_UM / 2 - ref_x) + _u(rng, -0.5, 0.5)
    half_diag = np.hypot(pw * 2.2, ph) / 2
    rad = min(DISC_R_UM * _u(rng, 0.55, 0.72), DISC_R_UM * 0.94 - half_diag)
    py = H_UM / 2 + rad * np.sin(phi)
    px = W_UM / 2 + rad * np.cos(phi)

    # w_floor 34um: order-0 rotation staircase shaves ~1 draw px off feature
    # width; 34um pre-rotation keeps every feature >= 28um post-rotation
    # (gate G1 diagnosis: rotated 28um lead stubs measured 27um => opened away)
    SEC_FLOOR = 34.0
    gap = _u(rng, 40.0, 120.0)
    ph2, pw2 = ph * _u(rng, 0.6, 0.9), pw * _u(rng, 0.5, 0.8)
    x2 = px + (pw + pw2) / 2 + gap
    _apply_panel_role(rng, part, pmeta, py, px, ph, pw,
                      str(rng.choice(["lined", "windowed", "textured"])),
                      w_floor=SEC_FLOOR)
    _apply_panel_role(rng, part, pmeta, py, x2, ph2, pw2,
                      str(rng.choice(["lined", "frame"])), w_floor=SEC_FLOOR)
    bw_ = _u(rng, 50.0, 140.0)
    part.rect(py, (px + x2) / 2, bw_, abs(x2 - px) + bw_, mode="add")

    ang = _u(rng, 18.0, 72.0) * (1 if rng.random() < 0.5 else -1)
    half = int((half_diag + 600.0) / UPD)
    cy_px, cx_px = int(py / UPD), int((px + x2) / 2 / UPD)
    y0, y1 = max(cy_px - half, 0), min(cy_px + half, DRAW_SHAPE[0])
    x0, x1 = max(cx_px - half, 0), min(cx_px + half, DRAW_SHAPE[1])
    for src, dst in ((part.add, main.add), (part.sub, main.sub)):
        wnd = src[y0:y1, x0:x1]
        rot = ndimage.rotate(wnd, ang, reshape=False, order=0, mode="constant")
        dst[y0:y1, x0:x1] |= rot
    meta.secondary = True


def _clutter(rng, cv: Canvas, meta: SceneMeta):
    for _ in range(_di(rng, 0, 3)):
        cv.disc(H_UM * _u(rng, 0.15, 0.85), W_UM * _u(rng, 0.12, 0.88),
                _u(rng, 34.0, 80.0), mode="add")
    for _ in range(_di(rng, 0, 2)):
        y, x = H_UM * _u(rng, 0.15, 0.85), W_UM * _u(rng, 0.12, 0.88)
        tl = _u(rng, 60.0, 120.0)
        twd = _u(rng, FLOOR, 70.0)
        gap = twd + _u(rng, FLOOR, 60.0)
        if rng.random() < 0.5:
            cv.rect(y, x - gap / 2, tl, twd, mode="add")
            cv.rect(y, x + gap / 2, tl, twd, mode="add")
        else:
            cv.rect(y - gap / 2, x, twd, tl, mode="add")
            cv.rect(y + gap / 2, x, twd, tl, mode="add")


def _mask_defects(rng, cv: Canvas, meta: SceneMeta):
    """Mask-level defects (carved BEFORE rotation): irregular edge notches
    (union-of-discs bites) + broken traces (gap cut across a trace)."""
    if meta.blocks:
        for _ in range(_di(rng, 2, 5)):
            y, x, bh, bw = meta.blocks[int(rng.integers(0, len(meta.blocks)))]
            edge = int(rng.integers(0, 4))
            if edge == 0:
                ny, nx = y - bh / 2, x + bw * _u(rng, -0.4, 0.4)
            elif edge == 1:
                ny, nx = y + bh / 2, x + bw * _u(rng, -0.4, 0.4)
            elif edge == 2:
                ny, nx = y + bh * _u(rng, -0.4, 0.4), x - bw / 2
            else:
                ny, nx = y + bh * _u(rng, -0.4, 0.4), x + bw / 2
            r0 = _u(rng, 40.0, 140.0)
            for _b in range(_di(rng, 2, 4)):
                cv.disc(ny + rng.normal(0, r0 * 0.4), nx + rng.normal(0, r0 * 0.4),
                        r0 * _u(rng, 0.6, 1.3), mode="sub")
            meta.notch_pts.append((ny, nx))
    if meta.traces and rng.random() < 0.7:
        for _ in range(_di(rng, 1, 2)):
            ty, tx, th, tw_ = meta.traces[int(rng.integers(0, len(meta.traces)))]
            gap = _u(rng, 60.0, 200.0)              # 6-20 HR px
            if tw_ >= th:                            # horizontal trace
                gx = tx + tw_ * _u(rng, -0.35, 0.35)
                cv.rect(ty, gx, th + 8.0, gap, mode="sub")
                meta.break_pts.append((ty, gx))
            else:
                gy = ty + th * _u(rng, -0.35, 0.35)
                cv.rect(gy, tx, gap, tw_ + 8.0, mode="sub")
                meta.break_pts.append((gy, tx))


def compose_scene(rng: np.random.Generator, tier: str,
                  force_xl: bool = False) -> tuple[Canvas, SceneMeta]:
    cv = Canvas()
    meta = SceneMeta(tier=tier)
    want_secondary = not force_xl and rng.random() < 0.25
    _cluster_scene(rng, cv, meta, tier=tier, force_xl=force_xl)
    if want_secondary:
        _secondary_part(rng, cv, meta)
    _clutter(rng, cv, meta)
    _mask_defects(rng, cv, meta)
    return cv, meta


# ── rendering: coverage -> levels -> temperature defects -> forward chain ─────
_DISC = None


def _disc_mask():
    global _DISC
    if _DISC is None:
        yy, xx = np.mgrid[:DRAW_SHAPE[0], :DRAW_SHAPE[1]]
        r = min(DRAW_SHAPE) / 2
        _DISC = (((yy - DRAW_SHAPE[0] / 2) ** 2 + (xx - DRAW_SHAPE[1] / 2) ** 2)
                 <= r ** 2)
    return _DISC


def render_coverage(cv: Canvas, angle: float) -> np.ndarray:
    m = (cv.final().astype(bool) & _disc_mask()).astype(np.float32)
    if angle:
        m = ndimage.rotate(m, angle, reshape=False, order=1, mode="constant")
    hr = m.reshape(HR_SHAPE[0], SSAA, HR_SHAPE[1], SSAA).mean(axis=(1, 3))
    return np.clip(hr, 0.0, 1.0)


def level_render(hr: np.ndarray, rng: np.random.Generator,
                 zones: list | None = None, angle: float = 0.0,
                 level_min: float = LEVEL_MIN) -> np.ndarray:
    """Coverage x per-component isothermal level; zone members share a base."""
    binm = hr >= 0.5
    lab, n = ndimage.label(binm)
    if n == 0:
        return hr * 0.0
    lv = rng.uniform(level_min, 1.0, size=n + 1).astype(np.float32)
    lv[0] = 0.0
    # Stratified anchoring (gate G7): among sizeable components, pin one to the
    # dark end and one to the bright end so per-scene 明暗 spread is guaranteed
    # (iid draws left ~24% of scenes with spread < 0.15).
    areas = np.bincount(lab.ravel(), minlength=n + 1)
    big = np.where(areas[1:] >= 50)[0] + 1
    if len(big) >= 2:
        pick = rng.choice(len(big), size=2, replace=False)
        lv[big[pick[0]]] = level_min + float(rng.uniform(0.0, 0.05))
        lv[big[pick[1]]] = 1.0 - float(rng.uniform(0.0, 0.05))
    if zones:
        zmap = np.zeros(HR_SHAPE, dtype=np.int16)
        for zi, (zy, zx, zh, zw, _k) in enumerate(zones, start=1):
            y0 = max(int((zy - zh / 2) / UM_PER_HR), 0)
            y1 = min(int((zy + zh / 2) / UM_PER_HR) + 1, HR_SHAPE[0])
            x0 = max(int((zx - zw / 2) / UM_PER_HR), 0)
            x1 = min(int((zx + zw / 2) / UM_PER_HR) + 1, HR_SHAPE[1])
            zmap[y0:y1, x0:x1] = zi
        if angle:
            zmap = ndimage.rotate(zmap, angle, reshape=False, order=0,
                                  mode="constant")
        base = rng.uniform(level_min, 1.0, size=len(zones) + 1).astype(np.float32)
        cents = ndimage.center_of_mass(binm, lab, index=np.arange(1, n + 1))
        for li, (cy, cx) in enumerate(cents, start=1):
            if np.isnan(cy):
                continue
            zi = int(zmap[int(round(cy)), int(round(cx))])
            if zi:
                lv[li] = float(np.clip(base[zi] + rng.normal(0.0, 0.03),
                                       level_min, 1.0))
    support = hr > 0.02
    _, (iy, ix) = ndimage.distance_transform_edt(lab == 0, return_indices=True)
    lab_f = lab[iy, ix]
    return np.where(support, hr * lv[lab_f], 0.0)


def _soft_disc(shape, cy, cx, r_px, soft_px):
    out = np.zeros(shape, dtype=np.float32)
    R = int(np.ceil(r_px + soft_px)) + 2
    y0, y1 = max(cy - R, 0), min(cy + R + 1, shape[0])
    x0, x1 = max(cx - R, 0), min(cx + R + 1, shape[1])
    if y1 <= y0 or x1 <= x0:
        return out
    yy = np.arange(y0, y1)[:, None] - cy
    xx = np.arange(x0, x1)[None, :] - cx
    d = np.hypot(yy, xx)
    out[y0:y1, x0:x1] = np.clip((r_px + soft_px / 2 - d) / max(soft_px, 1e-3),
                                0.0, 1.0)
    return out


@dataclass
class TempDefects:
    dots: list = field(default_factory=list)        # (y, x, r_px, depth) HR px
    hots: list = field(default_factory=list)        # (y, x, r_px, amp)
    darks: list = field(default_factory=list)       # (y, x, r_px, depth)


def temp_render(rng, cov_hr, lvl_img, *, add_defects=True):
    """Temperature + temperature-layer defects (dots / hot spots / dark blobs).
    Dots follow the decided pilot tiers: r 1-4 HR px, depth 0.3-1.0, soft edge
    ~1 HR px, 20-50 per scene, placed on structure."""
    T_bg = _u(rng, 19.0, 23.0)
    dT = _u(rng, 1.0, 3.0)
    T = T_bg + dT * lvl_img
    dfx = TempDefects()
    if not add_defects:
        return T, T_bg, dT, dfx
    on = np.argwhere(cov_hr > 0.7)
    if len(on):
        for _ in range(_di(rng, 20, 50)):
            y, x = on[int(rng.integers(0, len(on)))]
            r_px = _u(rng, 1.0, 4.0)
            depth = _u(rng, 0.3, 1.0)
            w = _soft_disc(T.shape, int(y), int(x), r_px, 1.0)
            T -= depth * np.maximum(T - T_bg, 0.0) * w
            dfx.dots.append((int(y), int(x), r_px, depth))
        if rng.random() < 0.6:
            for _ in range(_di(rng, 1, 2)):
                y, x = on[int(rng.integers(0, len(on)))]
                r_px = _u(rng, 8.0, 16.0)
                depth = _u(rng, 0.15, 0.4)
                w = _soft_disc(T.shape, int(y), int(x), r_px, 3.0)
                T -= depth * np.maximum(T - T_bg, 0.0) * w
                dfx.darks.append((int(y), int(x), r_px, depth))
    for _ in range(_di(rng, 2, 8)):
        if len(on) and rng.random() < 0.7:
            y, x = on[int(rng.integers(0, len(on)))]
        else:
            y = int(rng.integers(HR_SHAPE[0] // 5, 4 * HR_SHAPE[0] // 5))
            x = int(rng.integers(HR_SHAPE[1] // 5, 4 * HR_SHAPE[1] // 5))
        r_px = _u(rng, 1.0, 4.0)
        amp = _u(rng, 0.3, 1.0)
        w = _soft_disc(T.shape, int(y), int(x), r_px, 1.0)
        T += amp * dT * w
        dfx.hots.append((int(y), int(x), r_px, amp))
    return T, T_bg, dT, dfx


def forward_chain(rng, T_hr):
    """PSF blur -> 2x physical block average -> vignette + col stripes + grain."""
    sigma_lr = _u(rng, 0.15, 0.55)
    T_blur = ndimage.gaussian_filter(T_hr, sigma=sigma_lr * SCALE, mode="nearest")
    lr = T_blur.reshape(LR_SHAPE[0], SCALE, LR_SHAPE[1], SCALE).mean(axis=(1, 3))
    yy, xx = np.mgrid[:LR_SHAPE[0], :LR_SHAPE[1]].astype(np.float32)
    cy = LR_SHAPE[0] * _u(rng, 0.4, 0.6)
    cx = LR_SHAPE[1] * _u(rng, 0.4, 0.6)
    sig = _u(rng, 0.5, 0.8) * min(LR_SHAPE)
    g = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sig * sig)))
    lr = lr + _u(rng, 0.10, 0.16) * (g - float(g.mean()))
    col = rng.normal(0.0, 1.0, size=LR_SHAPE[1]).astype(np.float32)
    col = ndimage.gaussian_filter1d(col, sigma=_u(rng, 2.5, 5.0))
    col /= max(float(col.std()), 1e-6)
    lr = lr + (_u(rng, 0.020, 0.035) * col)[None, :]
    lr = lr + rng.normal(0.0, _u(rng, 0.08, 0.12), size=LR_SHAPE)
    return lr.astype(np.float32), sigma_lr


def interior_crop(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    side = int(min(h, w) / np.sqrt(2.0)) & ~1
    cy, cx = h // 2, w // 2
    return img[cy - side // 2: cy + side // 2, cx - side // 2: cx + side // 2]


def _pct(img, lo=1.0, hi=99.0):
    return float(np.percentile(img, lo)), float(np.percentile(img, hi))


# ── sheets ────────────────────────────────────────────────────────────────────
def _make_scene(seed, tier, force_xl=False, angle_mode="random"):
    rng = np.random.default_rng(seed)
    cv, meta = compose_scene(rng, tier, force_xl=force_xl)
    angle = _u(rng, 0.0, 360.0) if angle_mode == "random" else 0.0
    cov = render_coverage(cv, angle)
    occ = float((cov > 0.5).mean())
    lvl = level_render(cov, rng, zones=meta.zones, angle=angle)
    T, T_bg, dT, dfx = temp_render(rng, cov, lvl)
    lr, sigma_lr = forward_chain(rng, T)
    return dict(meta=meta, cov=cov, T=T, lr=lr, occ=occ, dT=dT,
                sigma=sigma_lr, dfx=dfx, angle=angle)


SHEET1_PLAN = [("mid", False), ("mid", False), ("mid", False), ("high", False),
               ("high", False), ("high", False), ("high", False), ("high", True)]


def sheet_fullchain(out: Path, seed0: int) -> None:
    """8 scenes x (HR GT temperature | LR frame): the full forward chain."""
    fig, axes = plt.subplots(4, 4, figsize=(4.4 * 4, 3.6 * 4))
    fig.suptitle(
        "v7 composer demo r4 — connected multi-panel assemblies (merged/streets/"
        "traces, no plain-solid slabs (lined panels), minority texture. "
        "FULL CHAIN: HR GT | LR frame. occ = full-frame audit metric.",
        fontsize=12.5, y=0.995)
    occs = []
    for k, (tier, xl) in enumerate(SHEET1_PLAN):
        s = _make_scene(seed0 + k, tier, force_xl=xl)
        occs.append((tier + (" XL" if xl else ""), s["occ"]))
        gt = interior_crop(s["T"])
        lr = interior_crop(s["lr"])
        r, c = divmod(k, 2)
        ax_gt, ax_lr = axes[r][2 * c], axes[r][2 * c + 1]
        lo, hi = _pct(gt)
        ax_gt.imshow(gt, cmap="inferno", vmin=lo, vmax=hi, interpolation="nearest")
        m = s["meta"]
        nd, nh = len(s["dfx"].dots), len(s["dfx"].hots)
        ax_gt.set_title(
            f"GT {tier}{' XL' if xl else ''}  occ={s['occ']:.2f}  ΔT={s['dT']:.1f}°C  "
            f"panels={len(m.blocks)}  dots={nd} hot={nh}"
            f"{'  +SEC' if m.secondary else ''}  s={seed0 + k}",
            fontsize=8)
        llo, lhi = _pct(lr)
        ax_lr.imshow(lr, cmap="inferno", vmin=llo, vmax=lhi, interpolation="nearest")
        ax_lr.set_title(f"LR 480x640  σ={s['sigma']:.2f}px + FPN/grain", fontsize=8)
        for ax in (ax_gt, ax_lr):
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=125, bbox_inches="tight")
    plt.close(fig)
    print("  occ:", " ".join(f"{t}={o:.3f}" for t, o in occs))
    print("wrote", out.resolve())


def sheet_defect_zoom(out: Path, seed0: int) -> None:
    """Defect close-ups, GT | LR pairs (unrotated scenes for exact crops)."""
    rows = []
    for k in range(14):
        tier = "high" if k % 3 != 2 else "mid"
        s = _make_scene(seed0 + 300 + k, tier, angle_mode="zero")
        m, d = s["meta"], s["dfx"]
        if len(rows) < 2 and d.dots:
            y, x, *_ = d.dots[int(len(d.dots) // 2)]
            rows.append(("small dark dots (r1-4px, depth.3-1)", y, x, s))
        elif len(rows) == 2 and d.hots:
            y, x, *_ = d.hots[0]
            rows.append(("bright hot spot", y, x, s))
        elif len(rows) == 3 and d.darks:
            y, x, *_ = d.darks[0]
            rows.append(("shallow dark blob", y, x, s))
        elif len(rows) == 4 and m.notch_pts:
            ny, nx = m.notch_pts[0]
            rows.append(("irregular edge notch", ny / UM_PER_HR, nx / UM_PER_HR, s))
        elif len(rows) == 5 and m.break_pts:
            by, bx = m.break_pts[0]
            rows.append(("broken trace", by / UM_PER_HR, bx / UM_PER_HR, s))
        if len(rows) == 6:
            break

    side = 180                                       # HR px = 1.8mm crop
    fig, axes = plt.subplots(len(rows), 2, figsize=(9.4, 4.5 * len(rows)))
    fig.suptitle("v7 demo r4 — defect close-ups: HR GT | LR frame (1.8mm crops, "
                 "nearest; per-crop scale)", fontsize=12.5, y=0.998)
    axes = np.atleast_2d(axes)
    for r, (title, cy, cx, s) in enumerate(rows):
        cy = int(np.clip(cy, side // 2, HR_SHAPE[0] - side // 2))
        cx = int(np.clip(cx, side // 2, HR_SHAPE[1] - side // 2))
        gt_c = s["T"][cy - side // 2: cy + side // 2, cx - side // 2: cx + side // 2]
        ly, lx, ls = cy // 2, cx // 2, side // 2
        ly = int(np.clip(ly, ls // 2, LR_SHAPE[0] - ls // 2))
        lx = int(np.clip(lx, ls // 2, LR_SHAPE[1] - ls // 2))
        lr_c = s["lr"][ly - ls // 2: ly + ls // 2, lx - ls // 2: lx + ls // 2]
        lo, hi = _pct(gt_c, 0.5, 99.5)
        axes[r, 0].imshow(gt_c, cmap="inferno", vmin=lo, vmax=hi,
                          interpolation="nearest")
        axes[r, 0].set_title(f"GT — {title}", fontsize=9.5)
        llo, lhi = _pct(lr_c, 0.5, 99.5)
        axes[r, 1].imshow(lr_c, cmap="inferno", vmin=llo, vmax=lhi,
                          interpolation="nearest")
        axes[r, 1].set_title("LR (same region)", fontsize=9.5)
        for c in (0, 1):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out, dpi=125, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.resolve())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("research_log/assets/v7_planning/composer_demo_r4"))
    ap.add_argument("--seed0", type=int, default=1010001)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sheet_fullchain(args.out / "sheet1_fullchain.png", args.seed0)
    sheet_defect_zoom(args.out / "sheet2_defect_zoom.png", args.seed0)


if __name__ == "__main__":
    main()
