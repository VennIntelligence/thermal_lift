"""Fig 60 — Optical ground truth registered onto the HR thermal grid.

The optical microscope view of the board (0.211 um/px) was registered onto
the 10 um/px HR reconstruction grid by optical_register.py (similarity
transform, theta=225.2 deg, NCC peak 0.985, residual NCC 0.951 after
blur-1.5; see remote_inbox/20260713_dotprobe/optical_register_result.json).
The warped optical footprint is a ~46 px diamond covering a serpentine-
trace region.

Top row: the warped optical patch (gray) next to the same crop from four
reconstruction arms (shared temperature scale, inferno). Bottom row: the
optical trace boundary overlaid on each arm — geometric agreement between
reconstructed hot traces and the optically-true trace layout is directly
visible; disagreement would show as contour lines cutting through thermal
structure.

The trace boundary is extracted on the ORIGINAL full-resolution microscope
view (data/optical/2.jpg.jpg, 2592x1944 @ 0.211 um/px — the registered
view per the register json), NOT on the blurred HR warp: iso-contour at
the mid-gray threshold between the trace (dark) and substrate (light)
modes, with the burned-in 100-um scale-bar box masked out. Contour vertex
coordinates are then mapped to the HR grid through the register-json
similarity transform p_hr = s*R(theta)@(p_opt - c_opt) + t (convention
verified against optical_warp_hr.npy: NCC 0.93 for R, 0.03 for R.T).

Data: remote_inbox/20260713_dotprobe/optical_warp_hr.npy (warped optical,
display only), optical_register_result.json (transform),
data/optical/2.jpg.jpg (native optical, contour source),
remote_inbox/20260710_expab/*_a*.npy (reconstructions, registration-
corrected onto the drizzle grid). ACL context: optical registration era
(stage0h dot probe); champion arms of ACL-071/076.
Run:  uv run python docs/publication_figures/scripts/fig60_optical_registration.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np
from contourpy import LineType, contour_generator
from PIL import Image
from scipy.ndimage import gaussian_filter

from pubfig_style import REPO_ROOT, W_DOUBLE, save_fig, setup_academic_style

setup_academic_style()

DOTPROBE = REPO_ROOT / "remote_inbox/20260713_dotprobe"
OPT = np.load(DOTPROBE / "optical_warp_hr.npy")
REG = json.loads((DOTPROBE / "optical_register_result.json").read_text())
OPT_RAW = np.asarray(
    Image.open(REPO_ROOT / "data/optical" / REG["main"]["view"]),
    dtype=np.float64)
EXPAB = REPO_ROOT / "remote_inbox/20260710_expab"
ARMS = [
    ("drizzle_a.npy", "Drizzle"),
    ("tgv_a.npy", "TGV"),
    ("depb9v6_a_corrected.npy", "Ours, v6 pool"),
    ("depb9v9_3k_a_corrected.npy", "Ours, v9 3k"),
]

Y0, Y1, X0, X1 = 391, 457, 540, 606  # footprint bbox from the register json


def crop(a):
    return a[Y0:Y1, X0:X1]


opt = crop(OPT).astype(float)
mask = np.isfinite(opt) & (opt > 0)
opt_m = np.where(mask, opt, np.nan)


def trace_contours_hr():
    """Iso-contours of the native-resolution optical view, mapped to HR
    crop coordinates via the register-json similarity transform."""
    # mask the burned-in "100um" scale-bar box (white box, bottom right)
    white = OPT_RAW >= 250
    rows = np.nonzero(white.sum(axis=1) > 50)[0]
    cols = np.nonzero(white.sum(axis=0) > 20)[0]
    bar_mask = np.zeros(OPT_RAW.shape, bool)
    bar_mask[rows.min() - 15:, cols.min() - 15:] = True

    # mid-gray threshold between trace (dark) and substrate (light) modes
    vals = OPT_RAW[~bar_mask]
    thr = 0.5 * (np.percentile(vals, 15) + np.percentile(vals, 85))

    # Smooth before contouring (sigma 16 opt px = 3.4 um = 0.34 HR px).
    # A symmetric blur does not displace straight-edge contours, so trace
    # boundaries stay accurately localized; it only suppresses structures
    # below the thermally-resolvable scale (JPEG noise, the 2-3 um wires of
    # the central micro-serpentine) that would render as solid ink at the
    # 66-px panel scale. Contour on a 4x-decimated grid: vertex spacing
    # ~0.08 HR px, still far finer than the HR grid.
    sm = gaussian_filter(OPT_RAW, sigma=16.0)
    ds = 4
    z = np.ma.array(sm[::ds, ::ds], mask=bar_mask[::ds, ::ds])
    xs = np.arange(z.shape[1], dtype=float) * ds
    ys = np.arange(z.shape[0], dtype=float) * ds
    gen = contour_generator(x=xs, y=ys, z=z, line_type=LineType.Separate)

    m = REG["main"]
    s, theta = m["s_hr_per_opt"], np.radians(m["theta_deg"])
    t, c = np.array(m["t_hr"]), np.array(m["o_c"])
    ct, st = np.cos(theta), np.sin(theta)
    R = np.array([[ct, -st], [st, ct]])  # y-down (x,y); verified vs warp

    polys = []
    for line in gen.lines(thr):
        if len(line) < 25:  # drop dust specks (<~20 um perimeter, sub-HR-px)
            continue
        p_hr = s * (R @ (line - c).T).T + t
        polys.append(p_hr - [X0, Y0])  # into crop coordinates
    return polys


CONTOURS_HR = trace_contours_hr()


def draw_contours(ax):
    for poly in CONTOURS_HR:
        ax.plot(poly[:, 0], poly[:, 1], color="#00C2C7", lw=0.55)
    ax.set_xlim(-0.5, X1 - X0 - 0.5)
    ax.set_ylim(Y1 - Y0 - 0.5, -0.5)

# arms carry different DC offsets -> compare median-removed crops on a
# single shared scale
arm_crops = {lab: crop(np.load(EXPAB / f)) for f, lab in ARMS}
arm_crops = {lab: c - np.median(c) for lab, c in arm_crops.items()}
allv = np.concatenate([c.ravel() for c in arm_crops.values()])
vmin, vmax = np.percentile(allv, [1, 99.5])

# layout="none" at creation time: set_layout_engine("none") after the fact
# leaves a placeholder engine that silently blocks subplots_adjust.
fig, axes = plt.subplots(2, 5, figsize=(W_DOUBLE, 3.55), layout="none")

# ── top row: optical + temperature crops ─────────────────────────────
ax = axes[0, 0]
ax.imshow(opt_m, cmap="gray", interpolation="nearest")
ax.set_title("Optical (warped)", fontsize=8)
for (f, lab), ax in zip(ARMS, axes[0, 1:]):
    ax.imshow(arm_crops[lab], cmap="inferno", vmin=vmin, vmax=vmax,
              interpolation="nearest")
    ax.set_title(lab, fontsize=8)

# ── bottom row: optical trace contour over each arm ──────────────────
ax = axes[1, 0]
ax.imshow(opt_m, cmap="gray", interpolation="nearest")
draw_contours(ax)
# ylabel (not title): the inter-row gap is occupied by the scale bar
ax.set_ylabel("+ trace contour", fontsize=7.5, color="#008B8F")
for (f, lab), ax in zip(ARMS, axes[1, 1:]):
    ax.imshow(arm_crops[lab], cmap="inferno", vmin=vmin, vmax=vmax,
              interpolation="nearest")
    draw_contours(ax)

for ax in axes.ravel():
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

# scale bar: 30 HR px = 300 um on top-left panel
# scale bar just below the optical panel (footprint is rotated -> no clean
# in-panel corner)
axes[0, 0].plot([1, 31], [70, 70], color="#222222", lw=2,
                solid_capstyle="butt", clip_on=False)
axes[0, 0].annotate("300 $\\mu$m", (35, 70.5), ha="left", va="center",
                    fontsize=6.5, color="#222222", annotation_clip=False)
axes[0, 0].set_xlim(-0.5, X1 - X0 - 0.5)
axes[0, 0].set_ylim(Y1 - Y0 - 0.5, -0.5)

axes[1, 0].annotate(
    "contour from native optical (0.211 $\\mu$m/px), mapped by the "
    "similarity transform: $\\theta$=225.2$^\\circ$, NCC 0.985 "
    "(residual 0.951)",
    (0.0, -0.10), xycoords="axes fraction", ha="left", va="top", fontsize=6.5,
    color="#555555", annotation_clip=False)

fig.suptitle("Registered optical ground truth vs reconstructed traces "
             "(serpentine region)", x=0.01, ha="left", fontsize=9)
fig.subplots_adjust(top=0.87, bottom=0.075, wspace=0.06, hspace=0.14,
                    left=0.01, right=0.99)

paths = save_fig(fig, "fig60_optical_registration")
print("\n".join(str(p) for p in paths))
