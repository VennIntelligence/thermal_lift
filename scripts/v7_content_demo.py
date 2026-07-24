#!/usr/bin/env python
"""v7 content demo — owner eyeball gate before the destructive v6->v7 regen.

Renders PROTOTYPE coverage for the new v7 content families plus mid-density
baselines, pushes everything through the REAL pool rendering path
(apply_defects + render_isothermal_field), and tiles contact sheets.

RULING UPDATE r1 (owner visual review, 2026-07-07): text_serial, conc_rings,
voronoi_cells, organic_patch are ALL REJECTED — they leave the chip domain.

RULING UPDATE r2 (owner visual review, 2026-07-07): quad_meander (real-
structure) also NOT kept. v5 must not be copied verbatim — e.g. the fixed
~47.6 deg rotation center is wrong for v7: scenes must rotate in ALL
directions. Take v5's TRAITS (rect blocks of mixed sizes arranged/combined/
subtracted, bright-and-dark blocks, thick/thin short traces) and REDESIGN.

r3 answer: `make_chip_scene` — a v7 "chip composer" prototype. Manhattan
rect/line composition inside a die region; element repertoire = big blocks
with rect cutouts (消融), small-pad arrays, buses/single traces, hollow
frames; occupancy targeted to the density-audit tiers (low/mid/high) with a
separate fragmentation knob (few-big-blocks vs many-small-pads — the audit's
second axis); per-scene brightness spread via level_min; whole-scene rotation
uniform in [0, 360). Rejected prototype functions KEPT below for the record,
no longer called.

Families in demo r1 (superseded):
  quad_meander  — real-structure motif measured from data/optical micrographs
                  (scale bars: 2.jpg 100um/446px, 21.jpg 50um/576px,
                  0021.jpg 20um/456px; finger stacks cross-checked between
                  magnifications, agreement < 0.2um):
                    * central cross: 4 metal fingers w=4.4um g=4.0-5.0um
                      (pitch ~8.7um, SUB-FLOOR -> rendered merged, arm ~31um)
                    * quadrant nested-L (greek key) meanders: bar 33-55um,
                      gap 20-55um (gaps floored to 28um for band honesty)
                    * end pads 136-261um; unit cell ~500um
  text_serial   — laser-mark style alphanumerics (raised strokes or etched
                  into a pad), stroke >= FLOOR
  conc_rings    — concentric annuli, ring/gap >= FLOOR
  voronoi_cells — metal cells separated by dark walls >= FLOOR
  organic_patch — smoothed-noise blobs (contamination-like), opened at FLOOR

Defect showcase (v7 tiers): pilot dots (depth 0.3-1.0, soft edge 1px,
r 1-4 HR px), prototype hot spots (temperature-level bright bumps),
prototype edge-adjacent dots (erosion constraint relaxed).

Everything here is a LOOK prototype: wiring into tcforge with config keys,
RNG discipline and golden pinning happens after owner sign-off.
Band-honesty floor: FLOOR = 28um = 2.8 HR px (HR pitch 10um), pitch >= 32um.

Usage (repo root, no CLI args): uv run python scripts/v7_content_demo.py
Inputs: tcforge/src 渲染管线（apply_defects + render_isothermal_field），无外部数据
Output: research_log/assets/v7_demo/  (tiles + 2 contact sheets)
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TCFORGE_SRC = PROJECT_ROOT / "tcforge" / "src"
if TCFORGE_SRC.exists() and str(TCFORGE_SRC) not in sys.path:
    sys.path.insert(0, str(TCFORGE_SRC))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.spatial import cKDTree

from tcforge.geometry import _binary, _downsample_coverage, build_scene_mask_with_metadata
from tcforge.realism import apply_defects, irregular_blob, render_isothermal_field

HR = (960, 1280)           # HR grid, 10 um/px (pixel_size 20 um, scale 2)
UM_PER_HRPX = 10.0
FLOOR_PX = 2.8             # 28 um band-honesty floor in HR px
SSAA = 4
OUT = Path(__file__).resolve().parent.parent / "research_log" / "assets" / "v7_demo"
OUT.mkdir(parents=True, exist_ok=True)


def um(v: float) -> float:
    """um -> HR px"""
    return v / UM_PER_HRPX


# ---------------------------------------------------------------------------
# soft-edge helpers (analytic AA via signed distance where possible)
# ---------------------------------------------------------------------------
def _soft(d: np.ndarray, w: float = 0.7) -> np.ndarray:
    """distance-to-boundary (positive inside) -> coverage in [0,1]"""
    return np.clip(0.5 + d / (2.0 * w), 0.0, 1.0).astype(np.float32)


def _downsample_from_ssaa(draw: np.ndarray) -> np.ndarray:
    return _downsample_coverage(_binary(draw), SSAA).astype(np.float32)


def _shrink(cov: np.ndarray, f: float, dy: float = 0.0) -> np.ndarray:
    """Zoom a coverage canvas down by factor f and paste centered (+dy down)."""
    small = ndimage.zoom(cov, f, order=1)
    out = np.zeros_like(cov)
    oy = int((cov.shape[0] - small.shape[0]) / 2 + dy * cov.shape[0])
    ox = (cov.shape[1] - small.shape[1]) // 2
    out[oy:oy + small.shape[0], ox:ox + small.shape[1]] = np.clip(small, 0, 1)
    return out


# ---------------------------------------------------------------------------
# quad_meander — real-structure motif (measured, see module docstring)
# ---------------------------------------------------------------------------
def quad_meander_cell(cell_px: int, bar: float, gap: float, arm_w: float,
                      cross_ch: float, ss: int) -> np.ndarray:
    """One 4-quadrant greek-key cell drawn at ss x supersampling (binary uint8)."""
    n = cell_px * ss
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    half = n / 2.0
    ch = cross_ch * ss / 2.0          # half-width of the dark cross channel
    per = (bar + gap) * ss
    cell = np.zeros((n, n), dtype=np.uint8)
    # distance from the cell's outer edges, per quadrant, mirrored
    dx = np.minimum(x, n - 1 - x)      # distance to nearest vertical outer edge
    dy = np.minimum(y, n - 1 - y)
    dmin = np.minimum(dx, dy)          # nested-L level set hugging outer corner
    band = (dmin % per) < bar * ss
    # keep quadrants only (carve the cross channel between them)
    in_quad = (np.abs(x - half) > ch) & (np.abs(y - half) > ch)
    cell[band & in_quad] = 1
    # central cross arms (merged sub-floor finger stacks -> single arm ~31um)
    aw = arm_w * ss / 2.0
    arm = (np.abs(y - half) < aw) | (np.abs(x - half) < aw)
    cell[arm] = 1
    return cell


def make_quad_meander(rng: np.random.Generator, rot_deg: float = 0.0) -> np.ndarray:
    bar = um(rng.uniform(33, 55))
    gap = max(FLOOR_PX, um(rng.uniform(28, 55)))
    arm_w = um(31)
    cross_ch = max(FLOOR_PX, um(rng.uniform(30, 42)))
    cell_px = int(round(um(rng.uniform(420, 560))))
    ss = 2  # cell tiles are axis-aligned; 2x supersample suffices
    cell = quad_meander_cell(cell_px, bar, gap, arm_w, cross_ch, ss)
    # tile over a die region covering most of the canvas, leave background margin
    die_h = int(HR[0] * rng.uniform(0.62, 0.8))
    die_w = int(HR[1] * rng.uniform(0.62, 0.8))
    ny = max(1, die_h // cell_px)
    nx = max(1, die_w // cell_px)
    tile = np.tile(cell, (ny, nx))
    cov_die = _downsample_coverage(tile, ss).astype(np.float32)
    cov = np.zeros(HR, dtype=np.float32)
    oy = (HR[0] - cov_die.shape[0]) // 2
    ox = (HR[1] - cov_die.shape[1]) // 2
    cov[oy:oy + cov_die.shape[0], ox:ox + cov_die.shape[1]] = cov_die
    if abs(rot_deg) > 0.05:
        cov = ndimage.rotate(cov, rot_deg, reshape=False, order=1, mode="constant", cval=0.0)
        cov = np.clip(cov, 0.0, 1.0)
    return cov


# ---------------------------------------------------------------------------
# text_serial — laser-mark alphanumerics
# ---------------------------------------------------------------------------
_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]
_CHARS = "ABCDEFGHJKLMNPRSTUVWXYZ0123456789"


def _serial(rng: np.random.Generator, n_lo=5, n_hi=10) -> str:
    n = int(rng.integers(n_lo, n_hi + 1))
    s = "".join(_CHARS[rng.integers(len(_CHARS))] for _ in range(n))
    if rng.random() < 0.5 and n >= 6:
        k = int(rng.integers(2, n - 2))
        s = s[:k] + "-" + s[k:]
    return s


def make_text_serial(rng: np.random.Generator, etched: bool) -> np.ndarray:
    n = (HR[0] * SSAA, HR[1] * SSAA)
    im = Image.new("L", (n[1], n[0]), 0)
    dr = ImageDraw.Draw(im)
    font_path = _FONTS[int(rng.integers(len(_FONTS)))]
    n_lines = int(rng.integers(1, 4))
    char_h_um = rng.uniform(450, 1000)                # bold stroke ~ h/8 >= 56um > FLOOR
    size = int(um(char_h_um) * SSAA)
    try:
        font = ImageFont.truetype(font_path, size=size)
    except OSError:
        font = ImageFont.truetype(_FONTS[0], size=size)
    # writing area: full canvas, or the pad interior for the etched variant
    if etched:
        ph = int(HR[0] * rng.uniform(0.5, 0.75))
        pw = int(HR[1] * rng.uniform(0.55, 0.8))
        oy = (HR[0] - ph) // 2
        ox = (HR[1] - pw) // 2
        box = (oy + int(0.06 * ph), ox + int(0.05 * pw),
               oy + ph - int(0.06 * ph), ox + pw - int(0.05 * pw))  # y0,x0,y1,x1 HR px
    else:
        box = (int(0.08 * HR[0]), int(0.05 * HR[1]), int(0.92 * HR[0]), int(0.95 * HR[1]))
    by0, bx0, by1, bx1 = (v * SSAA for v in box)
    line_h = size * 1.35
    n_lines = max(1, min(n_lines, int((by1 - by0) / line_h)))
    y0 = rng.uniform(by0, max(by0 + 1, by1 - n_lines * line_h))
    for i in range(n_lines):
        txt = _serial(rng)
        w = dr.textlength(txt, font=font)
        while w > (bx1 - bx0) and len(txt) > 3:
            txt = txt[:-1]
            w = dr.textlength(txt, font=font)
        x0 = rng.uniform(bx0, max(bx0 + 1, bx1 - w))
        dr.text((x0, y0 + i * line_h), txt, fill=255, font=font)
    text_cov = _downsample_from_ssaa((np.asarray(im) > 128).astype(np.uint8))
    if not etched:
        return text_cov
    pad = np.zeros(HR, dtype=np.float32)
    pad[box[0] - int(0.06 * ph):box[2] + int(0.06 * ph),
        box[1] - int(0.05 * pw):box[3] + int(0.05 * pw)] = 1.0
    return np.clip(pad - text_cov, 0.0, 1.0)


# ---------------------------------------------------------------------------
# conc_rings — concentric annuli
# ---------------------------------------------------------------------------
def make_conc_rings(rng: np.random.Generator) -> np.ndarray:
    cov = np.zeros(HR, dtype=np.float32)
    y, x = np.mgrid[0:HR[0], 0:HR[1]].astype(np.float32)
    n_groups = int(rng.integers(1, 4))
    for _ in range(n_groups):
        cy = rng.uniform(0.2, 0.8) * HR[0]
        cx = rng.uniform(0.2, 0.8) * HR[1]
        r = rng.uniform(um(60), um(400))
        ring_w = max(FLOOR_PX, um(rng.uniform(28, 100)))
        ring_g = max(FLOOR_PX, um(rng.uniform(28, 100)))
        n_rings = int(rng.integers(4, 13))
        d = np.hypot(y - cy, x - cx)
        for _k in range(n_rings):
            cov = np.maximum(cov, _soft(np.minimum(d - r, r + ring_w - d)))
            r += ring_w + ring_g
        # solid center pad half the time
        if rng.random() < 0.5:
            r0 = max(FLOOR_PX, rng.uniform(um(40), um(120)))
            cov = np.maximum(cov, _soft(r0 - np.hypot(y - cy, x - cx)))
    return cov


# ---------------------------------------------------------------------------
# voronoi_cells — metal cells with dark walls
# ---------------------------------------------------------------------------
def make_voronoi(rng: np.random.Generator) -> np.ndarray:
    n_pts = int(rng.integers(15, 45))
    pts = np.column_stack([rng.uniform(0, HR[0], n_pts), rng.uniform(0, HR[1], n_pts)])
    y, x = np.mgrid[0:HR[0], 0:HR[1]]
    grid = np.column_stack([y.ravel(), x.ravel()]).astype(np.float32)
    tree = cKDTree(pts)
    dd, ii = tree.query(grid, k=2, workers=-1)
    wall_w = max(FLOOR_PX, um(rng.uniform(28, 60)))
    margin = (dd[:, 1] - dd[:, 0]).reshape(HR)          # 0 on cell boundary
    lab = ii[:, 0].reshape(HR)
    keep = rng.random(n_pts) < rng.uniform(0.6, 0.95)   # fill only a subset of cells
    cov = _soft(margin - wall_w) * keep[lab]
    return cov.astype(np.float32)


# ---------------------------------------------------------------------------
# organic_patch — contamination-like smooth blobs
# ---------------------------------------------------------------------------
def make_organic(rng: np.random.Generator) -> np.ndarray:
    field = ndimage.gaussian_filter(rng.normal(size=HR).astype(np.float32),
                                    rng.uniform(18, 40))
    thr = np.percentile(field, rng.uniform(72, 88))
    blobs = field > thr
    blobs = ndimage.binary_opening(blobs, _disk(int(round(FLOOR_PX / 2))))
    lab, n = ndimage.label(blobs)
    if n:
        sizes = ndimage.sum(blobs, lab, index=np.arange(1, n + 1))
        small = np.flatnonzero(sizes < (FLOOR_PX * 3) ** 2) + 1
        blobs &= ~np.isin(lab, small)
    return ndimage.gaussian_filter(blobs.astype(np.float32), 0.7)


def _disk(r: int) -> np.ndarray:
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (yy * yy + xx * xx) <= r * r


# ---------------------------------------------------------------------------
# defect prototypes on top of the real pipeline
# ---------------------------------------------------------------------------
PILOT_DOTS = dict(severity_range=[1.0, 1.0], hole_radius_px=[1, 4],
                  notch_radius_px=[3, 10], crack_len_px=[60, 260], crack_width_px=[2, 4],
                  max_holes=50, max_notches=0, max_cracks=0, min_holes=20,
                  hole_depth_range=[0.3, 1.0], hole_edge_softness_px=1.0)


def add_zone_dots(cov: np.ndarray, rng: np.random.Generator, zone: np.ndarray,
                  n_dots: int = 10):
    """Prototype: dots placed anywhere in `zone` (relaxed-erosion / edge placement)."""
    zy, zx = np.where(zone)
    removal = np.zeros(HR, dtype=np.float32)
    centers = []
    for _ in range(min(n_dots, len(zy))):
        j = rng.integers(len(zy))
        r = float(rng.uniform(1, 3))
        depth = float(rng.uniform(0.4, 1.0))
        blob = irregular_blob(HR, zy[j], zx[j], r, rng, edge_softness_px=1.0)
        np.maximum(removal, depth * blob.astype(np.float32), out=removal)
        centers.append((int(zy[j]), int(zx[j])))
    return np.clip(cov * (1.0 - removal), 0.0, 1.0), centers


def add_hot_spots(field: np.ndarray, cov: np.ndarray, rng: np.random.Generator,
                  delta_t_c: float, n_spots: int = 6):
    """Prototype hot_spot family: small scene-level warm bumps (short/leakage/ESD)."""
    out = field.copy()
    y, x = np.mgrid[0:HR[0], 0:HR[1]].astype(np.float32)
    struct = cov > 0.5
    sy, sx = np.where(struct)
    by, bx = np.where(~struct)
    centers = []
    for i in range(n_spots):
        on_struct = i % 2 == 0
        yy0, xx0 = (sy, sx) if on_struct else (by, bx)
        j = rng.integers(len(yy0))
        cy, cx = float(yy0[j]), float(xx0[j])
        sig = float(rng.uniform(1.2, 2.4))
        amp = float(rng.uniform(0.4, 1.0)) * delta_t_c
        out += amp * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * sig * sig))
        centers.append((int(cy), int(cx), on_struct))
    return out, centers


# ---------------------------------------------------------------------------
# rendering + sheets
# ---------------------------------------------------------------------------
def render(cov: np.ndarray, rng: np.random.Generator, delta_t_c: float = 3.0,
           level_min: float = 0.82):
    return render_isothermal_field(cov, rng, t_bg_c=21.0, delta_t_c=delta_t_c,
                                   level_min=level_min, edge_sigma=0.6,
                                   low_freq_amplitude_c=0.10, low_freq_sigma_px=96.0)


OCC_TIERS = {"low": (0.02, 0.08), "mid": (0.10, 0.25), "high": (0.28, 0.50)}


# ---------------------------------------------------------------------------
# r4 drawing primitives — rotated / diagonal / curved elements, bbox-limited
# ---------------------------------------------------------------------------
def draw_rot_rect(cov, cy, cx, h, w, ang_deg, val=1.0):
    th = np.deg2rad(ang_deg)
    R = 0.5 * float(np.hypot(h, w)) + 3
    y0, y1 = int(max(0, cy - R)), int(min(cov.shape[0], cy + R))
    x0, x1 = int(max(0, cx - R)), int(min(cov.shape[1], cx + R))
    if y1 <= y0 or x1 <= x0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    u = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
    v = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
    m = _soft(np.minimum(w / 2 - np.abs(u), h / 2 - np.abs(v)))
    if val >= 0.5:
        cov[y0:y1, x0:x1] = np.maximum(cov[y0:y1, x0:x1], m)
    else:
        cov[y0:y1, x0:x1] = np.clip(cov[y0:y1, x0:x1] - m, 0.0, 1.0)


def draw_ellipse(cov, cy, cx, a, b, ang_deg, val=1.0):
    th = np.deg2rad(ang_deg)
    R = max(a, b) + 3
    y0, y1 = int(max(0, cy - R)), int(min(cov.shape[0], cy + R))
    x0, x1 = int(max(0, cx - R)), int(min(cov.shape[1], cx + R))
    if y1 <= y0 or x1 <= x0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    u = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
    v = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
    r = np.sqrt((u / a) ** 2 + (v / b) ** 2)
    m = _soft((1.0 - r) * min(a, b))
    if val >= 0.5:
        cov[y0:y1, x0:x1] = np.maximum(cov[y0:y1, x0:x1], m)
    else:
        cov[y0:y1, x0:x1] = np.clip(cov[y0:y1, x0:x1] - m, 0.0, 1.0)


def draw_segment(cov, p0, p1, width):
    (ay, ax), (by, bx) = p0, p1
    m = width / 2 + 3
    y0 = int(max(0, min(ay, by) - m)); y1 = int(min(cov.shape[0], max(ay, by) + m))
    x0 = int(max(0, min(ax, bx) - m)); x1 = int(min(cov.shape[1], max(ax, bx) + m))
    if y1 <= y0 or x1 <= x0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    vy, vx = by - ay, bx - ax
    L2 = vy * vy + vx * vx + 1e-9
    t = np.clip(((yy - ay) * vy + (xx - ax) * vx) / L2, 0.0, 1.0)
    d = np.hypot(yy - (ay + t * vy), xx - (ax + t * vx))
    cov[y0:y1, x0:x1] = np.maximum(cov[y0:y1, x0:x1], _soft(width / 2 - d))


def draw_polyline(cov, pts, width):
    for p0, p1 in zip(pts[:-1], pts[1:]):
        draw_segment(cov, p0, p1, width)


def bezier_pts(p0, pc, p1, n=14):
    t = np.linspace(0, 1, n)[:, None]
    arr = ((1 - t) ** 2 * np.array(p0) + 2 * (1 - t) * t * np.array(pc)
           + t ** 2 * np.array(p1))
    return [tuple(q) for q in arr]


def make_chip_scene2(rng: np.random.Generator, tier: str, frag: float | None = None):
    """v7 chip composer r4 — answers the r3 owner review AND the accumulated
    dataset-design lessons in one design:

    * patch-level information (owner r3): elements at MANY orientations inside
      one scene — independently rotated blocks/arrays, diagonal & 45-dogleg
      connectors between blocks, ellipse/circle pads, curved bond-wire arcs.
      A patch no longer sees only one H/V grid.
    * anti-overfit on dense gratings (owner r3 + FM-1/珠串 lesson): array pitch
      floor raised to 36um (margin above the 32um honesty floor), per-array
      pitch/size jitter + 5-20% random missing elements (kills exact
      periodicity, doubles as missing-pad realism), and a global cap on the
      area fraction of fine texture (arrays + thin buses).
    * density-audit lessons kept from r3: measured-occupancy targeting
      (knob-vs-outcome disconnect fix), fragmentation as an explicit second
      axis, block separation for per-component brightness levels, whole-scene
      rotation uniform in [0,360).

    Returns (coverage, info) — info carries the routed segments so defect
    stages can carve line breaks exactly on traces.
    """
    lo, hi = OCC_TIERS[tier]
    target = float(rng.uniform(lo, hi))
    frag = float(rng.uniform(0.0, 1.0)) if frag is None else float(frag)
    cov = np.zeros(HR, dtype=np.float32)
    fine = np.zeros(HR, dtype=np.float32)          # tracks fine-texture coverage
    dh = int(HR[0] * rng.uniform(0.62, 0.9)); dw = int(HR[1] * rng.uniform(0.62, 0.9))
    oy = (HR[0] - dh) // 2 + int(rng.uniform(-0.04, 0.04) * HR[0])
    ox = (HR[1] - dw) // 2 + int(rng.uniform(-0.04, 0.04) * HR[1])
    oy = max(0, min(HR[0] - dh, oy)); ox = max(0, min(HR[1] - dw, ox))
    blocks: list[tuple[float, float]] = []          # centers for connectivity
    segments: list[tuple[tuple, tuple, float]] = []  # routed traces (for breaks)
    n_px = HR[0] * HR[1]

    def occupied() -> float:
        return float(np.count_nonzero(cov > 0.5)) / n_px

    def fine_frac() -> float:
        return float(np.count_nonzero(fine > 0.5)) / n_px

    def free_center(h: float, w: float, max_overlap=0.15):
        for _ in range(8):
            cy = rng.uniform(oy + h / 2, oy + dh - h / 2)
            cx = rng.uniform(ox + w / 2, ox + dw - w / 2)
            y0, y1 = int(cy - h / 2), int(cy + h / 2)
            x0, x1 = int(cx - w / 2), int(cx + w / 2)
            if float((cov[y0:y1, x0:x1] > 0.5).mean()) <= max_overlap:
                return cy, cx
        return None

    if rng.random() < 0.6:   # die guard ring / seal ring at the die boundary
        t = max(FLOOR_PX, rng.uniform(2.8, 10))
        draw_rot_rect(cov, oy + dh / 2, ox + dw / 2, dh, dw, 0.0, 1.0)
        draw_rot_rect(cov, oy + dh / 2, ox + dw / 2, dh - 2 * t, dw - 2 * t, 0.0, 0.0)

    guard = 0
    while occupied() < target and guard < 700:
        guard += 1
        r = rng.random()
        if r < 0.32 * (1.0 - frag) + 0.06:
            # macro block: axis-aligned rect / rotated rect / ellipse, w/ cutouts
            h = rng.uniform(60, 320); w = rng.uniform(60, 320)
            p = free_center(h, w)
            if p is None:
                continue
            cy, cx = p
            kind = rng.random()
            ang = 0.0 if rng.random() < 0.7 else float(rng.uniform(0, 180))
            if kind < 0.85:
                draw_rot_rect(cov, cy, cx, h, w, ang, 1.0)
                for _ in range(int(rng.integers(0, 4))):     # 消融 cutouts
                    ch = rng.uniform(2 * FLOOR_PX, 0.5 * h)
                    cw = rng.uniform(2 * FLOOR_PX, 0.5 * w)
                    dyc = rng.uniform(-0.3, 0.3) * h; dxc = rng.uniform(-0.3, 0.3) * w
                    cang = ang if rng.random() < 0.7 else float(rng.uniform(0, 180))
                    draw_rot_rect(cov, cy + dyc, cx + dxc, ch, cw, cang, 0.0)
            else:
                draw_ellipse(cov, cy, cx, w / 2, h / 2, ang, 1.0)
                if rng.random() < 0.5:
                    draw_ellipse(cov, cy, cx, w / 4, h / 4, ang, 0.0)
            blocks.append((cy, cx))
        elif r < 0.32 * (1.0 - frag) + 0.06 + 0.34 * frag and fine_frac() < 0.10:
            # pad array: pitch >= 36um, size/pitch jitter, 5-20% dropout,
            # optionally rotated as a whole
            e = rng.uniform(3.0, 8.0)
            pitch = max(3.6, e + rng.uniform(3.6, 9.0) - 3.0 + 3.0)  # >=36um
            nx_, ny_ = int(rng.integers(2, 11)), int(rng.integers(2, 11))
            aw, ah = nx_ * pitch, ny_ * pitch
            if aw > 260 or ah > 260:      # cap single-grating span (texture cap)
                continue
            p = free_center(ah, aw, max_overlap=0.05)
            if p is None:
                continue
            cy, cx = p
            aang = 0.0 if rng.random() < 0.6 else float(rng.uniform(0, 180))
            th = np.deg2rad(aang)
            drop = rng.uniform(0.05, 0.20)
            for iy in range(ny_):
                for ix in range(nx_):
                    if rng.random() < drop:
                        continue
                    ly = (iy - (ny_ - 1) / 2) * pitch + rng.uniform(-0.4, 0.4)
                    lx = (ix - (nx_ - 1) / 2) * pitch + rng.uniform(-0.4, 0.4)
                    gy = cy + ly * np.cos(th) - lx * np.sin(th)
                    gx = cx + ly * np.sin(th) + lx * np.cos(th)
                    ee = e * rng.uniform(0.85, 1.15)
                    draw_rot_rect(cov, gy, gx, ee, ee, aang, 1.0)
                    draw_rot_rect(fine, gy, gx, ee, ee, aang, 1.0)
        elif r < 0.80:
            # routing: connector between two blocks (diag / dogleg / 45-dogleg)
            # or a free stub at an arbitrary angle
            w_ = float(max(FLOOR_PX, rng.uniform(2.8, 9)))
            if len(segments) > 14:
                continue
            if len(blocks) >= 2 and rng.random() < 0.75:
                i, j = rng.choice(len(blocks), size=2, replace=False)
                (ay, ax), (by, bx) = blocks[i], blocks[j]
                style = rng.random()
                if style < 0.4:
                    pts = [(ay, ax), (by, bx)]
                elif style < 0.7:
                    pts = [(ay, ax), (ay, bx), (by, bx)]
                else:
                    myy = ay + np.sign(by - ay) * abs(bx - ax)
                    pts = [(ay, ax), (myy, bx), (by, bx)]
                draw_polyline(cov, pts, w_)
                for p0, p1 in zip(pts[:-1], pts[1:]):
                    segments.append((p0, p1, w_))
                if w_ < 6:
                    draw_polyline(fine, pts, w_)
            else:
                length = rng.uniform(50, 250)
                ang = rng.uniform(0, np.pi)
                cy = rng.uniform(oy, oy + dh); cx = rng.uniform(ox, ox + dw)
                p0 = (cy - length / 2 * np.sin(ang), cx - length / 2 * np.cos(ang))
                p1 = (cy + length / 2 * np.sin(ang), cx + length / 2 * np.cos(ang))
                draw_segment(cov, p0, p1, w_)
                segments.append((p0, p1, w_))
                if w_ < 6:
                    draw_segment(fine, p0, p1, w_)
        elif r < 0.90:
            # hollow frame (guard ring), sometimes rotated
            h = rng.uniform(80, 380); w = rng.uniform(80, 380)
            t = max(FLOOR_PX, rng.uniform(2.8, 12))
            p = free_center(h, w, max_overlap=0.25)
            if p is None:
                continue
            cy, cx = p
            ang = 0.0 if rng.random() < 0.6 else float(rng.uniform(0, 180))
            draw_rot_rect(cov, cy, cx, h, w, ang, 1.0)
            draw_rot_rect(cov, cy, cx, h - 2 * t, w - 2 * t, ang, 0.0)
        else:
            # bond-wire arc: thin curved trace from a block toward the die edge
            if not blocks:
                continue
            ay, ax = blocks[int(rng.integers(len(blocks)))]
            by = rng.uniform(oy, oy + dh); bx = rng.uniform(ox, ox + dw)
            mid = ((ay + by) / 2 + rng.uniform(-80, 80),
                   (ax + bx) / 2 + rng.uniform(-80, 80))
            w_ = float(max(FLOOR_PX, rng.uniform(2.8, 5)))
            pts = bezier_pts((ay, ax), mid, (by, bx))
            draw_polyline(cov, pts, w_)

    angle = float(rng.uniform(0.0, 360.0))
    cov = np.clip(ndimage.rotate(cov, angle, reshape=False, order=1,
                                 mode="constant", cval=0.0), 0.0, 1.0)
    th = np.deg2rad(-angle)   # rotate segment coords along with the scene
    cy0, cx0 = (HR[0] - 1) / 2, (HR[1] - 1) / 2
    def rot_pt(p):
        y, x = p
        return (cy0 + (y - cy0) * np.cos(th) - (x - cx0) * np.sin(th),
                cx0 + (y - cy0) * np.sin(th) + (x - cx0) * np.cos(th))
    segments = [(rot_pt(p0), rot_pt(p1), w_) for p0, p1, w_ in segments]
    occ = float(np.count_nonzero(cov > 0.5)) / n_px
    return cov, {"tier": tier, "occ": occ, "frag": frag, "angle": angle,
                 "segments": segments}


# ---------------------------------------------------------------------------
# r5 — FLOORPLAN composer. r4 verdict (owner): random scatter is NOT 排列组合 —
# lines through blocks, floating donuts and randomly rotated blocks are not a
# chip. Real layout logic: recursive partition of the die into aligned regions
# separated by routing channels; each region holds one element type; traces
# run IN the channels; non-parallel content comes from a SECOND component at
# its own orientation + board-level traces + sparse 45-deg routing — the
# physically real sources of mixed orientation in a thermal image.
# ---------------------------------------------------------------------------
def _rect_local(canvas, y0, x0, y1, x1, val=1.0):
    y0i, x0i = int(max(0, round(y0))), int(max(0, round(x0)))
    y1i = int(min(canvas.shape[0], round(y1))); x1i = int(min(canvas.shape[1], round(x1)))
    if y1i <= y0i or x1i <= x0i:
        return
    if val >= 0.5:
        canvas[y0i:y1i, x0i:x1i] = 1.0
    else:
        canvas[y0i:y1i, x0i:x1i] = 0.0


DIE_FRAC = {"low": (0.35, 0.6), "mid": (0.55, 0.8), "high": (0.65, 0.9)}
P_EMPTY = {"low": 0.65, "mid": 0.30, "high": 0.06}


def compose_die(rng, H, W, tier, frag):
    """One die floorplan on a local HxW canvas -> (canvas, routed segments)."""
    die = np.zeros((H, W), dtype=np.float32)
    segs: list[tuple[tuple, tuple, float]] = []
    ring_t = 0.0
    if rng.random() < 0.6:                              # guard/seal ring
        ring_t = max(FLOOR_PX, rng.uniform(2.8, 9))
        _rect_local(die, 0, 0, H, W, 1.0)
        _rect_local(die, ring_t, ring_t, H - ring_t, W - ring_t, 0.0)
    # peripheral IO pad rows (between ring and core)
    pad_zone = ring_t + 3
    if rng.random() < 0.5:
        e = rng.uniform(5, 10); pitch = e + rng.uniform(4, 9)
        for edge in range(4):
            if rng.random() < 0.75:
                n = int(((W if edge < 2 else H) - 2 * (pad_zone + 10)) // pitch)
                for i in range(max(0, n)):
                    c = pad_zone + 10 + i * pitch
                    if edge == 0:
                        _rect_local(die, pad_zone, c, pad_zone + e, c + e, 1.0)
                    elif edge == 1:
                        _rect_local(die, H - pad_zone - e, c, H - pad_zone, c + e, 1.0)
                    elif edge == 2:
                        _rect_local(die, c, pad_zone, c + e, pad_zone + e, 1.0)
                    else:
                        _rect_local(die, c, W - pad_zone - e, c + e, W - pad_zone, 1.0)
        pad_zone += e + 6
    inset = pad_zone + rng.uniform(4, 12)
    leaves: list[tuple[float, float, float, float]] = []
    channels: list[tuple[str, float, float, float]] = []

    def split(y0, x0, y1, x1, depth):
        h, w = y1 - y0, x1 - x0
        if depth > 0 and min(h, w) > 110 and rng.random() < 0.8:
            ch = rng.uniform(6, 16)
            if w >= h:
                xs = x0 + rng.uniform(0.35, 0.65) * w
                channels.append(("v", xs, y0, y1))
                split(y0, x0, y1, xs - ch / 2, depth - 1)
                split(y0, xs + ch / 2, y1, x1, depth - 1)
            else:
                ys = y0 + rng.uniform(0.35, 0.65) * h
                channels.append(("h", ys, x0, x1))
                split(y0, x0, ys - ch / 2, x1, depth - 1)
                split(ys + ch / 2, x0, y1, x1, depth - 1)
        else:
            leaves.append((y0, x0, y1, x1))

    split(inset, inset, H - inset, W - inset, int(rng.integers(2, 5)))

    p_empty = P_EMPTY[tier]
    for (y0, x0, y1, x1) in leaves:
        h, w = y1 - y0, x1 - x0
        if min(h, w) < 18 or rng.random() < p_empty:
            continue
        m = rng.uniform(3, 10)
        yy0, xx0, yy1, xx1 = y0 + m, x0 + m, y1 - m, x1 - m
        lh, lw = yy1 - yy0, xx1 - xx0
        r = rng.random()
        if r < 0.45 * (1.0 - frag) + 0.10:
            # macro block with axis-aligned slots (排列里的消融)
            _rect_local(die, yy0, xx0, yy1, xx1, 1.0)
            for _ in range(int(rng.integers(0, 4))):
                if rng.random() < 0.5:
                    sh = rng.uniform(2 * FLOOR_PX, 0.22 * lh + 2 * FLOOR_PX)
                    sy = rng.uniform(yy0 + 4, max(yy0 + 5, yy1 - sh - 4))
                    sx0 = rng.uniform(xx0 + 4, xx0 + 0.5 * lw)
                    _rect_local(die, sy, sx0, sy + sh, rng.uniform(sx0 + 10, xx1 - 4), 0.0)
                else:
                    sw = rng.uniform(2 * FLOOR_PX, 0.22 * lw + 2 * FLOOR_PX)
                    sx = rng.uniform(xx0 + 4, max(xx0 + 5, xx1 - sw - 4))
                    sy0 = rng.uniform(yy0 + 4, yy0 + 0.5 * lh)
                    _rect_local(die, sy0, sx, rng.uniform(sy0 + 10, yy1 - 4), sx + sw, 0.0)
        elif r < 0.45 * (1.0 - frag) + 0.10 + 0.40 * frag:
            # pad/via array: pitch-jittered, 5-20% dropout, bounded by its leaf
            e = rng.uniform(3.0, 8.0)
            pitch = e + max(FLOOR_PX, rng.uniform(2.8, 9.0))
            nx_ = min(12, int(lw // pitch)); ny_ = min(12, int(lh // pitch))
            if nx_ < 2 or ny_ < 2:
                continue
            drop = rng.uniform(0.05, 0.20)
            for iy in range(ny_):
                for ix in range(nx_):
                    if rng.random() < drop:
                        continue
                    ee = e * rng.uniform(0.85, 1.15)
                    py = yy0 + iy * pitch + rng.uniform(-0.4, 0.4)
                    px = xx0 + ix * pitch + rng.uniform(-0.4, 0.4)
                    _rect_local(die, py, px, py + ee, px + ee, 1.0)
        elif r < 0.45 * (1.0 - frag) + 0.10 + 0.40 * frag + 0.25:
            # bus bundle along the leaf's long axis (recorded for line breaks)
            k = int(rng.integers(2, 7))
            w_ = float(max(FLOOR_PX, rng.uniform(2.8, 7)))
            if lw >= lh:
                sp = lh / k
                if sp < w_ + FLOOR_PX:
                    k = max(1, int(lh // (w_ + FLOOR_PX))); sp = lh / max(k, 1)
                for i in range(k):
                    y = yy0 + (i + 0.5) * sp
                    draw_segment(die, (y, xx0), (y, xx1), w_)
                    segs.append(((y, xx0), (y, xx1), w_))
            else:
                sp = lw / k
                if sp < w_ + FLOOR_PX:
                    k = max(1, int(lw // (w_ + FLOOR_PX))); sp = lw / max(k, 1)
                for i in range(k):
                    x = xx0 + (i + 0.5) * sp
                    draw_segment(die, (yy0, x), (yy1, x), w_)
                    segs.append(((yy0, x), (yy1, x), w_))
        else:
            # round pads / ring pad (aligned, inside its leaf)
            for _ in range(int(rng.integers(1, 4))):
                a = rng.uniform(8, max(9, min(lh, lw) / 3))
                cy = rng.uniform(yy0 + a, yy1 - a); cx = rng.uniform(xx0 + a, xx1 - a)
                draw_ellipse(die, cy, cx, a, a * rng.uniform(0.7, 1.0), 0.0, 1.0)
                if rng.random() < 0.4:
                    draw_ellipse(die, cy, cx, a * 0.45, a * 0.45, 0.0, 0.0)

    # routing in the partition channels (traces BETWEEN blocks, not through)
    for orient, c, a, b in channels:
        for _ in range(int(rng.integers(0, 3))):
            off = rng.uniform(-3.5, 3.5)
            w_ = float(max(FLOOR_PX, rng.uniform(2.8, 5.5)))
            lo_ = a + rng.uniform(0.0, 0.25) * (b - a)
            hi_ = b - rng.uniform(0.0, 0.25) * (b - a)
            p0 = (lo_, c + off) if orient == "v" else (c + off, lo_)
            p1 = (hi_, c + off) if orient == "v" else (c + off, hi_)
            draw_segment(die, p0, p1, w_)
            segs.append((p0, p1, w_))
    # sparse 45-deg routing (0-2)
    for _ in range(int(rng.integers(0, 3))):
        y0 = rng.uniform(0.15, 0.85) * H; x0 = rng.uniform(0.15, 0.85) * W
        L = rng.uniform(50, 180) / 1.414; sgn = 1 if rng.random() < 0.5 else -1
        w_ = float(max(FLOOR_PX, rng.uniform(2.8, 5)))
        p0, p1 = (y0, x0), (y0 + L * sgn, x0 + L)
        draw_segment(die, p0, p1, w_)
        segs.append((p0, p1, w_))
    # bilateral floorplan symmetry (real dies often mirror halves)
    if rng.random() < 0.3:
        half = W // 2
        die[:, W - half:] = die[:, :half][:, ::-1]
    return die, segs


def make_chip_scene3(rng: np.random.Generator, tier: str, frag: float | None = None):
    """r5 scene = main-die floorplan + optional second component at its own
    orientation + board-level traces between them + whole-scene rotation."""
    lo, hi = OCC_TIERS[tier]
    frag = float(rng.uniform(0.0, 1.0)) if frag is None else float(frag)
    n_px = HR[0] * HR[1]
    best = None
    for _attempt in range(3):
        cov = np.zeros(HR, dtype=np.float32)
        segments: list[tuple[tuple, tuple, float]] = []
        f0, f1 = DIE_FRAC[tier]
        dh = int(HR[0] * rng.uniform(f0, f1)); dw = int(HR[1] * rng.uniform(f0, f1))
        oy = (HR[0] - dh) // 2 + int(rng.uniform(-0.08, 0.08) * HR[0])
        ox = (HR[1] - dw) // 2 + int(rng.uniform(-0.08, 0.08) * HR[1])
        oy = max(0, min(HR[0] - dh, oy)); ox = max(0, min(HR[1] - dw, ox))
        die, segs = compose_die(rng, dh, dw, tier, frag)
        cov[oy:oy + dh, ox:ox + dw] = np.maximum(cov[oy:oy + dh, ox:ox + dw], die)
        segments += [((p0[0] + oy, p0[1] + ox), (p1[0] + oy, p1[1] + ox), w_)
                     for p0, p1, w_ in segs]
        # second component at its own orientation (physical source of
        # non-parallel structure in one thermal frame)
        if rng.random() < 0.55:
            sh = int(dh * rng.uniform(0.28, 0.45)); sw = int(dw * rng.uniform(0.28, 0.45))
            die2, _ = compose_die(rng, sh, sw, tier, frag)
            ang2 = float(rng.uniform(0, 360))
            die2r = np.clip(ndimage.rotate(die2, ang2, reshape=True, order=1), 0, 1)
            h2, w2 = die2r.shape
            corners = [(2, 2), (2, HR[1] - w2 - 2), (HR[0] - h2 - 2, 2),
                       (HR[0] - h2 - 2, HR[1] - w2 - 2)]
            corners = [(y, x) for y, x in corners if y >= 0 and x >= 0]
            if corners:
                y2, x2 = min(corners, key=lambda c: float(
                    cov[c[0]:c[0] + h2, c[1]:c[1] + w2].sum()))
                cov[y2:y2 + h2, x2:x2 + w2] = np.maximum(
                    cov[y2:y2 + h2, x2:x2 + w2], die2r)
                # board-level traces linking the two components
                for _ in range(int(rng.integers(1, 4))):
                    p0 = (oy + rng.uniform(0.2, 0.8) * dh, ox + rng.uniform(0.2, 0.8) * dw)
                    p1 = (y2 + rng.uniform(0.3, 0.7) * h2, x2 + rng.uniform(0.3, 0.7) * w2)
                    w_ = float(max(FLOOR_PX, rng.uniform(2.8, 6)))
                    draw_segment(cov, p0, p1, w_)
                    segments.append((p0, p1, w_))
        occ0 = float(np.count_nonzero(cov > 0.5)) / n_px
        cand = (abs(np.clip(occ0, lo, hi) - occ0), cov, segments)
        if best is None or cand[0] < best[0]:
            best = cand
        if cand[0] == 0.0:
            break
    _, cov, segments = best
    # top-up: if still under the tier floor (dropout-capped arrays can't fill),
    # add aligned filler macros inside the existing die footprint
    occ0 = float(np.count_nonzero(cov > 0.5)) / n_px
    m = cov > 0.5
    if occ0 < lo and m.any():
        ys, xs = np.where(m)
        by0, by1, bx0, bx1 = ys.min(), ys.max(), xs.min(), xs.max()
        g = 0
        while occ0 < lo and g < 80:
            g += 1
            h = rng.uniform(40, 170); w = rng.uniform(40, 170)
            if by1 - by0 < h + 2 or bx1 - bx0 < w + 2:
                continue
            y0i = int(rng.uniform(by0, by1 - h)); x0i = int(rng.uniform(bx0, bx1 - w))
            y1i, x1i = int(y0i + h), int(x0i + w)
            if float((cov[y0i:y1i, x0i:x1i] > 0.5).mean()) > 0.35:
                continue
            _rect_local(cov, y0i, x0i, y1i, x1i, 1.0)
            if rng.random() < 0.6:   # aligned slot cutout
                if rng.random() < 0.5:
                    _rect_local(cov, y0i + 0.35 * h, x0i + 6, y0i + 0.65 * h, x1i - 6, 0.0)
                else:
                    _rect_local(cov, y0i + 6, x0i + 0.35 * w, y1i - 6, x0i + 0.65 * w, 0.0)
            occ0 = float(np.count_nonzero(cov > 0.5)) / n_px
    angle = float(rng.uniform(0.0, 360.0))
    cov = np.clip(ndimage.rotate(cov, angle, reshape=False, order=1,
                                 mode="constant", cval=0.0), 0.0, 1.0)
    th = np.deg2rad(-angle)
    cy0, cx0 = (HR[0] - 1) / 2, (HR[1] - 1) / 2

    def rot_pt(p):
        y, x = p
        return (cy0 + (y - cy0) * np.cos(th) - (x - cx0) * np.sin(th),
                cx0 + (y - cy0) * np.sin(th) + (x - cx0) * np.cos(th))

    segments = [(rot_pt(p0), rot_pt(p1), w_) for p0, p1, w_ in segments]
    occ = float(np.count_nonzero(cov > 0.5)) / n_px
    return cov, {"tier": tier, "occ": occ, "frag": frag, "angle": angle,
                 "segments": segments}


def carve_line_breaks(cov, rng, segments, frac=0.3):
    """Carve short gaps (断线) across a fraction of routed traces."""
    breaks = []
    for p0, p1, w_ in segments:
        if rng.random() > frac:
            continue
        t = rng.uniform(0.2, 0.8)
        by = p0[0] + t * (p1[0] - p0[0]); bx = p0[1] + t * (p1[1] - p0[1])
        if not (20 < by < HR[0] - 20 and 20 < bx < HR[1] - 20):
            continue
        gap = rng.uniform(6, 18)
        seg_ang = float(np.degrees(np.arctan2(p1[0] - p0[0], p1[1] - p0[1])))
        draw_rot_rect(cov, by, bx, w_ + 6, gap, seg_ang, 0.0)
        breaks.append((int(by), int(bx)))
    return cov, breaks


def make_chip_scene(rng: np.random.Generator, tier: str, frag: float | None = None):
    """v7 chip composer prototype (r3). Returns (coverage, info dict).

    Manhattan-geometry composition (axis-aligned before one global rotation):
      big blocks w/ rect cutouts | small-pad arrays | buses & single traces |
      hollow frames. Element mix steered by `frag` in [0,1] (0 = few big
      blocks, 1 = many small pads — the density audit's fragmentation axis);
      elements added until measured occupancy hits the tier target.
    """
    lo, hi = OCC_TIERS[tier]
    target = float(rng.uniform(lo, hi))
    frag = float(rng.uniform(0.0, 1.0)) if frag is None else float(frag)
    cov = np.zeros(HR, dtype=np.float32)
    dh = int(HR[0] * rng.uniform(0.6, 0.9))
    dw = int(HR[1] * rng.uniform(0.6, 0.9))
    # center the die (small jitter) so the whole-scene rotation clips little
    oy = (HR[0] - dh) // 2 + int(rng.uniform(-0.04, 0.04) * HR[0])
    ox = (HR[1] - dw) // 2 + int(rng.uniform(-0.04, 0.04) * HR[1])
    oy = max(0, min(HR[0] - dh, oy)); ox = max(0, min(HR[1] - dw, ox))

    def rand_yx(h: int, w: int, max_overlap: float | None = None) -> tuple[int, int] | None:
        """Random placement inside the die; if max_overlap is set, rejection-sample
        so blocks stay mostly separate (separate components -> distinct brightness
        levels under the isothermal per-component renderer)."""
        if h >= dh or w >= dw:
            return None
        for _try in range(8):
            y = int(rng.uniform(oy, oy + dh - h)); x = int(rng.uniform(ox, ox + dw - w))
            if max_overlap is None:
                return y, x
            if float((cov[y:y + h, x:x + w] > 0.5).mean()) <= max_overlap:
                return y, x
        return None

    n_px = HR[0] * HR[1]
    guard = 0
    while float(np.count_nonzero(cov > 0.5)) / n_px < target and guard < 800:
        guard += 1
        r = rng.random()
        if r < 0.30 * (1.0 - frag):
            # big block, optionally with rect cutouts (排列/组合/消融)
            h = int(rng.uniform(60, 340)); w = int(rng.uniform(60, 340))
            p = rand_yx(h, w, max_overlap=0.15)
            if p is None:
                continue
            y, x = p
            cov[y:y + h, x:x + w] = 1.0
            for _ in range(int(rng.integers(0, 4))):
                ch = int(rng.uniform(2 * FLOOR_PX, max(2 * FLOOR_PX + 1, 0.55 * h)))
                cw = int(rng.uniform(2 * FLOOR_PX, max(2 * FLOOR_PX + 1, 0.55 * w)))
                cy = int(rng.uniform(y, y + h - ch)); cx = int(rng.uniform(x, x + w - cw))
                cov[cy:cy + ch, cx:cx + cw] = 0.0
        elif r < 0.30 + 0.40 * frag:
            # small-pad array (pitch >= 32um honesty floor)
            e = int(rng.uniform(3, 9))
            gap = int(max(FLOOR_PX, rng.uniform(3, 9)))
            nx_ = int(rng.integers(2, 13)); ny_ = int(rng.integers(2, 13))
            p = rand_yx(ny_ * (e + gap), nx_ * (e + gap), max_overlap=0.05)
            if p is None:
                continue
            y, x = p
            for iy in range(ny_):
                for ix in range(nx_):
                    cov[y + iy * (e + gap):y + iy * (e + gap) + e,
                        x + ix * (e + gap):x + ix * (e + gap) + e] = 1.0
        elif r < 0.82:
            # bus (parallel traces) or single trace, thick/thin, long/short
            k = 1 if rng.random() < 0.5 else int(rng.integers(3, 9))
            w_ = int(round(max(FLOOR_PX, rng.uniform(2.8, 10 if k == 1 else 6))))
            gap = int(round(max(FLOOR_PX, rng.uniform(2.8, 9))))
            length = int(rng.uniform(60, 900))
            if rng.random() < 0.5:   # horizontal
                p = rand_yx(k * (w_ + gap), length)
                if p is None:
                    continue
                y, x = p
                for i in range(k):
                    cov[y + i * (w_ + gap):y + i * (w_ + gap) + w_, x:x + length] = 1.0
            else:                    # vertical
                p = rand_yx(length, k * (w_ + gap))
                if p is None:
                    continue
                y, x = p
                for i in range(k):
                    cov[y:y + length, x + i * (w_ + gap):x + i * (w_ + gap) + w_] = 1.0
        else:
            # hollow frame (guard ring / seal ring look)
            h = int(rng.uniform(80, 400)); w = int(rng.uniform(80, 400))
            t = int(round(max(FLOOR_PX, rng.uniform(2.8, 12))))
            p = rand_yx(h, w, max_overlap=0.25)
            if p is None:
                continue
            y, x = p
            cov[y:y + h, x:x + w] = 1.0
            cov[y + t:y + h - t, x + t:x + w - t] = 0.0
    angle = float(rng.uniform(0.0, 360.0))
    cov = np.clip(ndimage.rotate(cov, angle, reshape=False, order=1,
                                 mode="constant", cval=0.0), 0.0, 1.0)
    occ = float(np.count_nonzero(cov > 0.5)) / n_px
    return cov, {"tier": tier, "occ": occ, "frag": frag, "angle": angle}


def make_broken_traces(rng: np.random.Generator):
    """Chip-like trace bundle (thick/thin, long/short) with line-break defects."""
    cov = np.zeros(HR, dtype=np.float32)
    breaks: list[tuple[int, int]] = []
    n_lines = int(rng.integers(14, 26))
    for _ in range(n_lines):
        horiz = rng.random() < 0.65
        w = max(FLOOR_PX, float(rng.uniform(2.8, 9)))
        length = float(rng.uniform(150, 950))
        if horiz:
            y = rng.uniform(0.06, 0.94) * HR[0]
            x0 = rng.uniform(0.03, max(0.04, 0.95 - length / HR[1])) * HR[1]
            seg = np.zeros(HR, dtype=np.float32)
            yy, xx = np.mgrid[0:HR[0], 0:HR[1]].astype(np.float32)
            d = np.minimum.reduce([yy - (y - w / 2), (y + w / 2) - yy,
                                   xx - x0, (x0 + length) - xx])
            seg = _soft(d)
        else:
            x = rng.uniform(0.06, 0.94) * HR[1]
            y0 = rng.uniform(0.03, max(0.04, 0.95 - length / HR[0])) * HR[0]
            yy, xx = np.mgrid[0:HR[0], 0:HR[1]].astype(np.float32)
            d = np.minimum.reduce([xx - (x - w / 2), (x + w / 2) - xx,
                                   yy - y0, (y0 + length) - yy])
            seg = _soft(d)
        # carve 0-3 breaks (缺线)
        for _b in range(int(rng.integers(0, 4))):
            gap = float(rng.uniform(6, 20))
            if horiz:
                bx = x0 + rng.uniform(0.15, 0.85) * length
                yy2, xx2 = np.mgrid[0:HR[0], 0:HR[1]].astype(np.float32)
                cut = _soft(np.minimum(xx2 - (bx - gap / 2), (bx + gap / 2) - xx2))
                by, bxc = int(y), int(bx)
            else:
                by0 = y0 + rng.uniform(0.15, 0.85) * length
                yy2, xx2 = np.mgrid[0:HR[0], 0:HR[1]].astype(np.float32)
                cut = _soft(np.minimum(yy2 - (by0 - gap / 2), (by0 + gap / 2) - yy2))
                by, bxc = int(by0), int(x)
            seg = np.clip(seg - cut, 0, 1)
            breaks.append((by, bxc))
        cov = np.maximum(cov, seg)
    return cov, breaks


def main() -> None:
    tiles: list[tuple[str, np.ndarray]] = []   # (label, HR temperature field)
    master = np.random.default_rng(20260707)

    # r4: chip composer v2 — multi-orientation elements, connectivity routing,
    #     anti-periodic arrays, fine-texture cap; tiers + frag + all-dir rotation.
    lineup = [
        ("low", 0.15, 101), ("low", 0.55, 102), ("low", 0.9, 103),
        ("mid", 0.1, 201), ("mid", 0.4, 202), ("mid", 0.7, 203), ("mid", 0.95, 204),
        ("high", 0.15, 301), ("high", 0.5, 302), ("high", 0.85, 303),
    ]
    scenes = {}
    for tier, frag, seed in lineup:
        rng = np.random.default_rng(seed)
        cov, info = make_chip_scene3(rng, tier, frag=frag)
        scenes[seed] = (cov, info)
        lvl = float(rng.uniform(0.35, 0.85))
        tiles.append((f"{tier} occ={info['occ']:.2f} frag={frag:.2f} "
                      f"rot={info['angle']:.0f}° lvl={lvl:.2f}",
                      render(cov, rng, level_min=lvl)))

    # r4 integrated preview: full v7 defect set — dark dots (incl. relaxed
    # placement), MORE + BIGGER irregular notches, line breaks carved on the
    # actual routed traces, a mid-scale irregular dark patch (die-attach void /
    # delamination), and bright hot spots.
    for tier, seed in [("mid", 401), ("high", 402)]:
        rng = np.random.default_rng(seed)
        cov, info = make_chip_scene3(rng, tier)
        cov, _bks = carve_line_breaks(cov, rng, info["segments"], frac=0.35)
        dparams = {**PILOT_DOTS, "max_holes": 30, "min_holes": 10, "max_notches": 12,
                   "notch_radius_px": [4, 16]}
        cov_d, _meta = apply_defects(cov, rng, **dparams)
        struct = cov_d > 0.5
        inner = ndimage.binary_erosion(struct, _disk(2))
        cov_d, _c1 = add_zone_dots(cov_d, rng, inner, n_dots=10)
        # mid-scale irregular dark patch (delamination-like, shallow, big blob)
        sy, sx = np.where(struct)
        if len(sy):
            j = rng.integers(len(sy))
            blob = irregular_blob(HR, sy[j], sx[j], float(rng.uniform(14, 28)), rng,
                                  irregularity=0.6, edge_softness_px=2.0)
            cov_d = np.clip(cov_d - float(rng.uniform(0.3, 0.5)) * blob.astype(np.float32),
                            0.0, 1.0)
        lvl = float(rng.uniform(0.4, 0.7))
        field = render(cov_d, rng, level_min=lvl)
        field, _c2 = add_hot_spots(field, cov_d, rng, delta_t_c=3.0, n_spots=5)
        tiles.append((f"INTEGRATED {tier} occ={info['occ']:.2f} rot={info['angle']:.0f}° "
                      "(dots+notches+breaks+patch+hotspots)", field))

    # --- sheet 3: patch-information check — random 128px patches from 2 scenes;
    #     the r3 owner critique was "after patching everything looks the same";
    #     multi-orientation content should make these visibly diverse.
    rngp = np.random.default_rng(777)
    patches = []
    for seed in (202, 302):
        cov, _ = scenes[seed]
        f = render(cov, np.random.default_rng(seed + 5000),
                   level_min=float(rngp.uniform(0.4, 0.7)))
        for _ in range(8):
            py = int(rngp.uniform(0, HR[0] - 128)); px = int(rngp.uniform(0, HR[1] - 128))
            patches.append((f"s{seed} ({py},{px})", f[py:py + 128, px:px + 128]))
    fig, axes = plt.subplots(2, 8, figsize=(2.1 * 8, 2.4 * 2), dpi=110)
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, (label, p) in zip(np.ravel(axes), patches):
        ax.imshow(p, cmap="inferno")
        ax.set_title(label, fontsize=6)
    fig.suptitle("v7 r4 — random 128px training patches from two scenes "
                 "(orientation/content diversity check)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "sheet3_patch_diversity.png", bbox_inches="tight")
    plt.close(fig)

    # --- sheet 1: content families -----------------------------------------
    n = len(tiles)
    ncols, nrows = 4, int(np.ceil(n / 4))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.35 * nrows), dpi=110)
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, (label, field) in zip(np.ravel(axes), tiles):
        ax.imshow(field[::2, ::2], cmap="inferno")
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    fig.suptitle("v7 content demo — prototype families through the real render path "
                 "(HR GT temperature, inferno)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "sheet1_content_families.png", bbox_inches="tight")
    plt.close(fig)

    # --- sheet 2: defect showcase (zoom crops) ------------------------------
    crops: list[tuple[str, np.ndarray]] = []

    # (a) pilot dots via the REAL apply_defects on a wide-structure host
    #     (quad_meander bars are 3-5 px < erosion(8) support -> zero dots there;
    #      that IS the §1.5 finding, demonstrated below with the relaxed zone)
    rngd = np.random.default_rng(60)
    host, _m = build_scene_mask_with_metadata(
        "medium", 7042, canvas_shape=HR, pixel_size_um=20.0, scale=2,
        antialias=True, ssaa_factor=4, motif_weights={"die_bga": 1.0},
        rotation_deg_center=47.6, rotation_jitter_deg=1.0)
    host = host.astype(np.float32)
    cov_d, meta = apply_defects(host, rngd, **PILOT_DOTS)
    field = render(cov_d, rngd)
    centers = meta.get("hole_centers_yx") or []
    depths = meta.get("hole_depths") or []
    if len(depths) >= 3:
        order = np.argsort(depths)
        for k, tag in [(order[0], "shallow"), (order[len(order) // 2], "mid"), (order[-1], "deep")]:
            cy, cx = centers[k]
            cy = int(np.clip(cy, 96, HR[0] - 96)); cx = int(np.clip(cx, 96, HR[1] - 96))
            crops.append((f"pilot dot {tag} d={depths[k]:.2f} (real pipeline)",
                          field[cy - 96:cy + 96, cx - 96:cx + 96]))

    # (b) relaxed-erosion dots ON the real-structure meander (§1.5 fix preview)
    rnge = np.random.default_rng(63)
    cov_e = make_quad_meander(rnge)
    struct = cov_e > 0.5
    inner = ndimage.binary_erosion(struct, _disk(2))
    cov_e2, icenters = add_zone_dots(cov_e, rnge, inner, n_dots=14)
    rim = struct & ~ndimage.binary_erosion(struct, _disk(3))
    cov_e3, ecenters = add_zone_dots(cov_e2, rnge, rim, n_dots=8)
    fielde = render(cov_e3, rnge)
    for cy, cx in icenters[:2]:
        cy = int(np.clip(cy, 96, HR[0] - 96)); cx = int(np.clip(cx, 96, HR[1] - 96))
        crops.append(("dots on meander, erosion2 (proto)", fielde[cy - 96:cy + 96, cx - 96:cx + 96]))
    for cy, cx in ecenters[:2]:
        cy = int(np.clip(cy, 96, HR[0] - 96)); cx = int(np.clip(cx, 96, HR[1] - 96))
        crops.append(("edge/rim dot (proto)", fielde[cy - 96:cy + 96, cx - 96:cx + 96]))

    # (c) line breaks (缺线) on the trace bundle
    rngb = np.random.default_rng(9301)
    cov_bt2, bt_breaks = make_broken_traces(rngb)
    fbt = render(cov_bt2, rngb)
    for by, bx in bt_breaks[:2]:
        by = int(np.clip(by, 96, HR[0] - 96)); bx = int(np.clip(bx, 96, HR[1] - 96))
        crops.append(("line break (proto)", fbt[by - 96:by + 96, bx - 96:bx + 96]))

    # (d) irregular notches via the REAL generator family on a v5 legacy host
    rngn = np.random.default_rng(65)
    hostn, _mn = build_scene_mask_with_metadata(
        "medium", 9022, canvas_shape=HR, pixel_size_um=20.0, scale=2,
        antialias=True, ssaa_factor=4, motif_weights=None,
        rotation_deg_center=47.6, rotation_jitter_deg=1.0)
    hostn = hostn.astype(np.float32)
    notch_params = {**PILOT_DOTS, "max_holes": 0, "min_holes": 0,
                    "max_notches": 8, "notch_radius_px": [4, 12]}
    covn, _metan = apply_defects(hostn, rngn, **notch_params)
    fieldn = render(covn, rngn)
    diffm = (hostn > 0.5) & ~(covn > 0.5)
    labn, nn = ndimage.label(diffm)
    if nn:
        sizes = ndimage.sum(diffm, labn, index=np.arange(1, nn + 1))
        for k in np.argsort(sizes)[::-1][:2]:
            cy, cx = ndimage.center_of_mass(diffm, labn, int(k) + 1)
            cy = int(np.clip(cy, 96, HR[0] - 96)); cx = int(np.clip(cx, 96, HR[1] - 96))
            crops.append(("irregular notch (real pipeline)",
                          fieldn[cy - 96:cy + 96, cx - 96:cx + 96]))

    # hot spots
    rngh = np.random.default_rng(64)
    cov_h = make_quad_meander(rngh)
    fh = render(cov_h, rngh)
    fh2, hcenters = add_hot_spots(fh, cov_h, rngh, delta_t_c=3.0, n_spots=6)
    for cy, cx, on_s in hcenters[:2]:
        cy = int(np.clip(cy, 128, HR[0] - 128)); cx = int(np.clip(cx, 128, HR[1] - 128))
        crops.append((f"hot_spot proto ({'on-structure' if on_s else 'background'})",
                      fh2[cy - 96:cy + 96, cx - 96:cx + 96]))

    ncols = 4
    nrows = int(np.ceil(len(crops) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.6 * nrows), dpi=110)
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, (label, crop) in zip(np.ravel(axes), crops):
        ax.imshow(crop, cmap="inferno")
        ax.set_title(label, fontsize=9)
    fig.suptitle("v7 defect showcase — 192x192 HR px zooms (1.92mm field)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "sheet2_defect_showcase.png", bbox_inches="tight")
    plt.close(fig)

    for i, (label, field) in enumerate(tiles):
        Image.fromarray(
            (plt.get_cmap("inferno")((field - field.min()) / (np.ptp(field) + 1e-9))[..., :3] * 255
             ).astype(np.uint8)).save(OUT / f"tile_{i:02d}.png")
    print(f"[demo] {len(tiles)} tiles, {len(crops)} crops -> {OUT}")


if __name__ == "__main__":
    main()
