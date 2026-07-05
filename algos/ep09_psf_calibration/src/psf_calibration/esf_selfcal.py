"""E3 sigma self-calibration kernel: multi-frame projected ESF (ACL-057).

Why this kernel is identifiable where E1/E2 were not (ACL-056): the E1/E2
degeneracy exists because ANY assumed sigma' admits a compensating scene that
reproduces all frames — self-supervised objectives cannot pick sigma. E3 breaks
the degeneracy with a PARAMETRIC SCENE PRIOR: it assumes the scene contains at
least one straight step edge. Under that constraint the compensator is no
longer free — the observed transition width along the edge normal is exactly
step (x) Gaussian(sigma) (x) known-aperture, so sigma becomes identifiable.
The prior is about scene STRUCTURE (straight hot/cold boundaries — generic for
thermal targets), not about any particular dataset: no dataset-specific
constants anywhere.

Pipeline (per burst):
  1. quick SAA reconstruction on the HR grid (certified reference scatter);
  2. straight-edge detection: gradient ridge + orientation-coherent RANSAC +
     total-least-squares refinement (explicit refusal when nothing qualifies);
  3. for each edge, project ALL frames' pixel centers (refined shifts) onto the
     edge normal -> dense super-resolved 1D scatter profile;
  4. robust fit of  baseline + amp * mean_p Phi((d - c + p.n)/sigma_eff)
     where p runs over the KNOWN aperture sampling pattern and
     sigma_eff^2 = sigma^2 + extra_var (see below);
  5. aggregate per-edge sigmas by median; bootstrap CI (edges if >=4, else
     frames); inter-edge spread reported as a step-assumption/anisotropy check.

Aperture models (known, zero free parameters; LR px units):
  - "pool_block_average": the tcforge `physical_block_average` renderer takes
    scale x scale bilinear point samples of the sigma-blurred HR canvas per LR
    pixel. Modelled EXACTLY: discrete offset grid m in {0..scale-1} per axis,
    COMPOSED with the frame's bilinear node pair {-f, 1-f} (weights {1-f, f},
    f = frac(scale*shift) — a known per-frame constant, so kernels are built
    per frame). Only the HR rasterization box (var 1/12 HR^2) is folded into
    extra_var as a matched Gaussian. Modelling bilinear as a constant tent
    (var 1/6 HR^2) instead was measured to bias sigma_hat -30% at sigma=0.2
    (the tent view only holds for rough fields); ignoring the aperture
    entirely biases +18% at sigma=0.4. Both rejected empirically.
  - "pool_point": tcforge `exact_ep06_point` (one bilinear point sample);
    same per-frame bilinear treatment, no offset grid.
  - "detector_box": continuous 1 LR px box integration (real detector,
    fill-factor ~1) via Gauss-Legendre quadrature; extra_var = 0.
  - "point": pure point sample (unit tests of the fitter math).

Anisotropic PSFs: E3 measures the DIRECTIONAL sigma along each edge normal.
For elliptical/airy scenes the per-edge spread will be genuinely high and the
comparison against a scalar metadata sigma is a documented degraded fit — such
scenes are labelled, never silently dropped.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.special import erf

from psf_calibration.sigma_selfcal import (
    PREREG_MEDIAN_REL_ERR_TOL,
    _plot_bench,
    _write_rows_csv,
    evaluate_prereg,
    load_pool_scene,
)

APERTURES = ("pool_block_average", "pool_point", "detector_box", "point")

# Rasterization box (1 HR px, var 1/12 HR^2) folded into extra_var as a matched
# Gaussian for pool-rendered bursts. The bilinear resampling kernel is NOT a
# constant tent for smooth fields (treating it as var 1/6 HR^2 over-corrects and
# biased sigma_hat by -30% at sigma=0.2) — it is instead modelled EXACTLY per
# frame: the sampler reads grid nodes at offsets {-f, 1-f} HR px with weights
# {1-f, f}, where f = frac(scale*shift) is a KNOWN per-frame constant.
_POOL_RASTER_VAR_HR2 = 1.0 / 12.0


@dataclass
class EsfSelfCalConfig:
    scale: int = 2
    aperture: str = "detector_box"
    # --- edge detection ---
    smooth_sigma_hr: float = 1.0
    grad_quantile: float = 0.985
    ransac_iters: int = 300
    ransac_tol_hr: float = 1.2
    orient_tol_deg: float = 15.0
    min_edge_len_hr: float = 16.0
    min_edge_len_frac: float = 0.12
    max_edges: int = 8
    border_erode_hr: int = 3
    # --- profile collection ---
    half_width_lr: float = 5.0
    tangent_margin_lr: float = 1.0
    max_points_per_edge: int = 20000
    min_samples: int = 150
    min_side_samples: int = 30
    # --- fitting / gates ---
    sigma_bounds: tuple[float, float] = (0.02, 3.0)
    min_r2: float = 0.90
    min_amp_snr: float = 5.0
    box_quad_points: int = 8
    # --- aggregation ---
    rel_spread_warn: float = 0.35
    bootstrap_rounds: int = 500
    frame_bootstrap_rounds: int = 200
    seed: int = 0
    extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SAA + edge detection
# ---------------------------------------------------------------------------


def quick_saa(burst: np.ndarray, shifts: np.ndarray, scale: int) -> tuple[np.ndarray, np.ndarray]:
    """Weighted shift-and-add on the HR grid using the certified reference scatter."""

    from tcforge._ep06_reference.forward import _scatter_lr_to_reference  # noqa: PLC0415

    hr_shape = (burst.shape[1] * scale, burst.shape[2] * scale)
    num = np.zeros(hr_shape, dtype=np.float64)
    den = np.zeros(hr_shape, dtype=np.float64)
    for k in range(len(burst)):
        frame = np.asarray(burst[k], dtype=np.float64)
        finite = np.isfinite(frame)
        num += _scatter_lr_to_reference(np.where(finite, frame, 0.0), shifts[k], hr_shape=hr_shape, scale=scale)
        den += _scatter_lr_to_reference(finite.astype(np.float64), shifts[k], hr_shape=hr_shape, scale=scale)
    valid = den > 0.25
    saa = num / np.maximum(den, 1e-9)
    if valid.any():
        saa[~valid] = float(np.median(saa[valid]))
    return saa, valid


@dataclass
class Edge:
    nx: float
    ny: float
    rho_hr: float
    t_lo_hr: float
    t_hi_hr: float
    n_inliers: int
    length_hr: float
    straightness_rms_hr: float
    mean_grad: float


def _wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def detect_edges(saa: np.ndarray, valid: np.ndarray, cfg: EsfSelfCalConfig, rng: np.random.Generator) -> list[Edge]:
    """Straight-edge segments via gradient ridge + orientation-coherent RANSAC + TLS refine."""

    sm = ndimage.gaussian_filter(saa, cfg.smooth_sigma_hr)
    gy, gx = np.gradient(sm)
    mag = np.hypot(gx, gy)
    core = ndimage.binary_erosion(valid, iterations=max(1, cfg.border_erode_hr))
    if not core.any():
        return []
    m = getattr(cfg, "_border_margin_hr", 0)
    if m > 0 and 2 * m < min(core.shape):
        core[:m, :] = False
        core[-m:, :] = False
        core[:, :m] = False
        core[:, -m:] = False
    if not core.any():
        return []
    thr = float(np.quantile(mag[core], cfg.grad_quantile))
    if thr <= 0:
        return []
    sel = core & (mag >= thr)
    ys, xs = np.nonzero(sel)
    if len(xs) < 8:
        return []
    phi = np.arctan2(gy[ys, xs], gx[ys, xs])  # gradient (= edge normal) direction
    pts_x = xs.astype(np.float64)
    pts_y = ys.astype(np.float64)
    grad_w = mag[ys, xs]

    min_len = max(cfg.min_edge_len_hr, cfg.min_edge_len_frac * float(min(saa.shape)))
    min_inliers = max(20, int(0.8 * min_len))
    orient_tol = np.deg2rad(cfg.orient_tol_deg)

    remaining = np.ones(len(xs), dtype=bool)
    edges: list[Edge] = []
    for _ in range(int(cfg.max_edges)):
        if remaining.sum() < min_inliers:
            break
        best_idx: np.ndarray | None = None
        cand = np.nonzero(remaining)[0]
        anchors = rng.choice(cand, size=min(cfg.ransac_iters, len(cand)), replace=False)
        for a in anchors:
            n = np.array([np.cos(phi[a]), np.sin(phi[a])])
            rho = n[0] * pts_x[a] + n[1] * pts_y[a]
            d = n[0] * pts_x + n[1] * pts_y - rho
            ok = remaining & (np.abs(d) <= cfg.ransac_tol_hr) & (np.abs(_wrap_angle(phi - phi[a])) <= orient_tol)
            if best_idx is None or ok.sum() > len(best_idx):
                best_idx = np.nonzero(ok)[0]
        if best_idx is None or len(best_idx) < min_inliers:
            break
        # TLS refinement on the inlier cloud
        px, py = pts_x[best_idx], pts_y[best_idx]
        cx, cy = float(px.mean()), float(py.mean())
        cov = np.cov(np.vstack([px - cx, py - cy]))
        evals, evecs = np.linalg.eigh(cov)
        tangent = evecs[:, int(np.argmax(evals))]
        normal = np.array([-tangent[1], tangent[0]])
        g_mean = np.array([np.mean(gx[py.astype(int), px.astype(int)]), np.mean(gy[py.astype(int), px.astype(int)])])
        if float(normal @ g_mean) < 0:
            normal = -normal
            tangent = -tangent
        rho = float(normal[0] * cx + normal[1] * cy)
        d_ref = normal[0] * pts_x + normal[1] * pts_y - rho
        phi_n = float(np.arctan2(normal[1], normal[0]))
        refined = remaining & (np.abs(d_ref) <= cfg.ransac_tol_hr) & (np.abs(_wrap_angle(phi - phi_n)) <= orient_tol)
        idx = np.nonzero(refined)[0]
        remaining[idx] = False  # consume regardless of acceptance (prevents re-finding the same structure)
        if len(idx) < min_inliers:
            continue
        # canonical tangent (-ny, nx): profile collection recomputes it the same way,
        # so the stored t-bounds must use this convention (sign bugs here cost samples)
        tangent = np.array([-normal[1], normal[0]])
        t = tangent[0] * pts_x[idx] + tangent[1] * pts_y[idx]
        length = float(t.max() - t.min())
        if length < min_len:
            continue
        edges.append(
            Edge(
                nx=float(normal[0]),
                ny=float(normal[1]),
                rho_hr=rho,
                t_lo_hr=float(t.min()),
                t_hi_hr=float(t.max()),
                n_inliers=int(len(idx)),
                length_hr=length,
                straightness_rms_hr=float(np.std(normal[0] * pts_x[idx] + normal[1] * pts_y[idx] - rho)),
                mean_grad=float(np.mean(grad_w[idx])),
            )
        )
    return edges


# ---------------------------------------------------------------------------
# Aperture model
# ---------------------------------------------------------------------------


def aperture_projection(
    aperture: str,
    scale: int,
    nx: float,
    ny: float,
    quad_points: int,
    *,
    frac_dx: float = 0.0,
    frac_dy: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Projected sampling kernel for a unit edge normal -> (offsets [LR px], weights, extra_var [LR px^2]).

    Pool apertures are frame-dependent: `frac_dx/frac_dy` = frac(scale*shift)
    select the exact bilinear node pair per axis. Constant mean offsets are
    absorbed by the fitted edge center, so no centering is applied.
    """

    if aperture not in APERTURES:
        raise ValueError(f"aperture must be one of {APERTURES}")
    if aperture in ("pool_block_average", "pool_point"):
        m = np.arange(scale, dtype=np.float64) if aperture == "pool_block_average" else np.zeros(1)
        fy = float(frac_dy) % 1.0
        fx = float(frac_dx) % 1.0
        node_y = np.array([-fy, 1.0 - fy])
        wgt_y = np.array([1.0 - fy, fy])
        node_x = np.array([-fx, 1.0 - fx])
        wgt_x = np.array([1.0 - fx, fx])
        py = (m[:, None] + node_y[None, :]).ravel()  # HR px along rows
        pwy = np.tile(wgt_y, len(m)) / len(m)
        px = (m[:, None] + node_x[None, :]).ravel()
        pwx = np.tile(wgt_x, len(m)) / len(m)
        offs = ((py[:, None] * ny + px[None, :] * nx) / scale).ravel()
        w = (pwy[:, None] * pwx[None, :]).ravel()
        keep = w > 1e-12
        return offs[keep], w[keep] / w[keep].sum(), _POOL_RASTER_VAR_HR2 / (scale * scale)
    if aperture == "detector_box":
        q, w1 = np.polynomial.legendre.leggauss(int(quad_points))
        q = 0.5 * q  # nodes on [-0.5, 0.5]
        w1 = 0.5 * w1
        offs = np.add.outer(q * nx, q * ny).ravel()
        w = np.outer(w1, w1).ravel()
        return offs, w / w.sum(), 0.0
    return np.zeros(1), np.ones(1), 0.0  # "point"


def frame_kernels(
    aperture: str, scale: int, nx: float, ny: float, quad_points: int, shifts: np.ndarray
) -> dict[int, tuple[np.ndarray, np.ndarray]] | tuple[np.ndarray, np.ndarray]:
    """Per-frame projected kernels for pool apertures; a single shared kernel otherwise."""

    if aperture in ("pool_block_average", "pool_point"):
        out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for k in range(len(shifts)):
            offs, w, _ = aperture_projection(
                aperture, scale, nx, ny, quad_points,
                frac_dx=float(scale * shifts[k, 0]), frac_dy=float(scale * shifts[k, 1]),
            )
            out[k] = (offs, w)
        return out
    offs, w, _ = aperture_projection(aperture, scale, nx, ny, quad_points)
    return (offs, w)


def aperture_extra_var(aperture: str, scale: int) -> float:
    return _POOL_RASTER_VAR_HR2 / (scale * scale) if aperture in ("pool_block_average", "pool_point") else 0.0


def discrete_gaussian_effective_sigma(sigma_lr: float, scale: int, truncate: float = 4.0) -> float:
    """Std (LR px) of the DISCRETE Gaussian kernel scipy actually applies at HR.

    Pool renders blur with `ndimage.gaussian_filter(sigma=sigma_lr*scale)` whose
    truncated, grid-sampled kernel carries LESS variance than nominal at small
    sigma (e.g. nominal 0.15 LR px @scale=2 -> effective 0.044; 0.20 -> 0.142;
    >=0.35 the gap shrinks below ~1%). The ESF estimator measures the blur that
    is PHYSICALLY IN THE FRAMES, i.e. the effective value — bench ground truth
    must therefore be compared in effective terms (comparison-layer fix, same
    hazard class as the ACL-049 grid-offset lesson).
    """

    sd = float(sigma_lr) * scale
    if sd <= 0:
        return 0.0
    radius = int(truncate * sd + 0.5)  # scipy _gaussian_kernel1d convention
    if radius < 1:
        return 0.0
    k = np.arange(-radius, radius + 1, dtype=np.float64)
    w = np.exp(-0.5 * (k / sd) ** 2)
    w /= w.sum()
    return float(np.sqrt(np.sum(w * k * k)) / scale)


def esf_model(d: np.ndarray, a: float, c: float, sigma: float, b: float, offs: np.ndarray, w: np.ndarray, extra_var: float) -> np.ndarray:
    sig_eff = float(np.sqrt(max(sigma, 1e-6) ** 2 + extra_var))
    z = (d[:, None] - c + offs[None, :]) / (sig_eff * np.sqrt(2.0))
    return b + a * ((0.5 * (1.0 + erf(z))) @ w)


Kernels = dict[int, tuple[np.ndarray, np.ndarray]] | tuple[np.ndarray, np.ndarray]


def esf_model_grouped(
    d: np.ndarray,
    frame_ids: np.ndarray,
    kernels: Kernels,
    a: float,
    c: float,
    sigma: float,
    b: float,
    extra_var: float,
) -> np.ndarray:
    """ESF model with per-frame sampling kernels (pool bilinear phases are frame constants)."""

    if isinstance(kernels, tuple):
        offs, w = kernels
        return esf_model(d, a, c, sigma, b, offs, w, extra_var)
    out = np.empty_like(d)
    for k in np.unique(frame_ids):
        offs, w = kernels[int(k)]
        sel = frame_ids == k
        out[sel] = esf_model(d[sel], a, c, sigma, b, offs, w, extra_var)
    return out


# ---------------------------------------------------------------------------
# Profile collection + robust fit
# ---------------------------------------------------------------------------


def collect_profile(
    burst: np.ndarray, shifts: np.ndarray, edge: Edge, cfg: EsfSelfCalConfig, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project every frame's pixel centers onto the edge normal -> (d, value, frame_idx)."""

    scale = cfg.scale
    rho_lr = edge.rho_hr / scale
    t_lo = edge.t_lo_hr / scale + cfg.tangent_margin_lr
    t_hi = edge.t_hi_hr / scale - cfg.tangent_margin_lr
    tx, ty = -edge.ny, edge.nx
    h, w = burst.shape[1], burst.shape[2]
    jj, ii = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    # LR pixels near the frame border can sample outside the source canvas
    # (pool renders) or carry partial-aperture readout (real sensors) — drop them
    m_lr = int(np.ceil(np.abs(shifts).max())) + 1 if len(shifts) else 1
    interior = (ii >= m_lr) & (ii < h - m_lr) & (jj >= m_lr) & (jj < w - m_lr)
    d_all: list[np.ndarray] = []
    v_all: list[np.ndarray] = []
    f_all: list[np.ndarray] = []
    for k in range(len(burst)):
        x = jj + float(shifts[k, 0])
        y = ii + float(shifts[k, 1])
        d = edge.nx * x + edge.ny * y - rho_lr
        t = tx * x + ty * y
        frame = np.asarray(burst[k], dtype=np.float64)
        mask = (np.abs(d) <= cfg.half_width_lr) & (t >= t_lo) & (t <= t_hi) & np.isfinite(frame) & interior
        if not mask.any():
            continue
        d_all.append(d[mask])
        v_all.append(frame[mask])
        f_all.append(np.full(int(mask.sum()), k, dtype=np.int32))
    if not d_all:
        return np.empty(0), np.empty(0), np.empty(0, dtype=np.int32)
    d_cat = np.concatenate(d_all)
    v_cat = np.concatenate(v_all)
    f_cat = np.concatenate(f_all)
    if len(d_cat) > cfg.max_points_per_edge:
        pick = rng.choice(len(d_cat), size=cfg.max_points_per_edge, replace=False)
        d_cat, v_cat, f_cat = d_cat[pick], v_cat[pick], f_cat[pick]
    return d_cat, v_cat, f_cat


def fit_edge_profile(
    d: np.ndarray,
    v: np.ndarray,
    cfg: EsfSelfCalConfig,
    kernels: Kernels,
    extra_var: float,
    frame_ids: np.ndarray | None = None,
    point_weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """Robust ESF fit; returns params + quality + validity verdict."""

    out: dict[str, Any] = {"valid": False, "fail_reason": "insufficient_samples"}
    left = v[d < -1.0]
    right = v[d > 1.0]
    if len(d) < cfg.min_samples or len(left) < cfg.min_side_samples or len(right) < cfg.min_side_samples:
        return out
    if frame_ids is None:
        frame_ids = np.zeros(len(d), dtype=np.int32)
    b0 = float(np.median(left))
    a0 = float(np.median(right)) - b0
    if a0 == 0.0:
        a0 = 1e-3
    far = np.abs(d) > 2.5
    resid0 = v[far] - np.where(d[far] < 0, b0, b0 + a0)
    f_scale = max(1.4826 * float(np.median(np.abs(resid0 - np.median(resid0)))) if far.sum() >= 8 else 0.0, 1e-6)
    sw = np.sqrt(point_weights) if point_weights is not None else None

    def resid(p: np.ndarray) -> np.ndarray:
        r = esf_model_grouped(d, frame_ids, kernels, p[0], p[1], p[2], p[3], extra_var) - v
        return r * sw if sw is not None else r

    lo, hi = cfg.sigma_bounds
    span = max(float(np.max(np.abs(d))), 1.0)
    try:
        res = least_squares(
            resid,
            x0=[a0, 0.0, 0.5, b0],
            bounds=([-np.inf, -0.5 * span, lo, -np.inf], [np.inf, 0.5 * span, hi, np.inf]),
            loss="soft_l1",
            f_scale=f_scale,
            max_nfev=400,
        )
    except Exception as exc:  # pragma: no cover - scipy failure path
        out["fail_reason"] = type(exc).__name__
        return out
    a, c, sigma, b = (float(t) for t in res.x)
    pred = esf_model_grouped(d, frame_ids, kernels, a, c, sigma, b, extra_var)
    r = v - pred
    weights_eff = point_weights if point_weights is not None else np.ones_like(v)
    ss_res = float(np.sum(weights_eff * r * r))
    mu = float(np.average(v, weights=weights_eff))
    ss_tot = float(np.sum(weights_eff * (v - mu) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    resid_sigma = 1.4826 * float(np.median(np.abs(r - np.median(r))))
    amp_snr = abs(a) / max(resid_sigma, 1e-9)
    # Upper bound hit = fit blew up -> invalid. Lower bound hit = "PSF unresolvably
    # small given aperture + extras" -> a legitimate metrological statement, kept
    # valid but flagged so aggregation/report can surface it.
    at_upper = sigma >= hi * 0.95
    at_lower = sigma <= lo * 1.05
    valid = bool(np.isfinite([a, c, sigma, b]).all() and r2 >= cfg.min_r2 and amp_snr >= cfg.min_amp_snr and not at_upper)
    out.update(
        {
            "amplitude": a,
            "center_lr_px": c,
            "sigma_hat": abs(sigma),
            "baseline": b,
            "r2": float(r2),
            "amp_snr": float(amp_snr),
            "resid_sigma": float(resid_sigma),
            "n_points": int(len(d)),
            "at_upper_bound": bool(at_upper),
            "at_lower_bound": bool(at_lower),
            "valid": valid,
            "fail_reason": "pass" if valid else ("sigma_at_upper_bound" if at_upper else "quality_gate"),
        }
    )
    return out


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------


def run_esf_selfcal(
    burst: np.ndarray,
    shifts: np.ndarray,
    cfg: EsfSelfCalConfig,
    *,
    out_dir: Path | None = None,
    label: str = "burst",
    make_plot: bool = True,
) -> dict[str, Any]:
    burst = np.asarray(burst, dtype=np.float64)
    shifts = np.asarray(shifts, dtype=np.float64)
    if burst.ndim != 3 or shifts.shape != (len(burst), 2):
        raise ValueError("burst must be (N,H,W) and shifts (N,2) [dx,dy] LR px")
    rng = np.random.default_rng(cfg.seed)

    saa, valid_mask = quick_saa(burst, shifts, cfg.scale)
    # exclude the border band: shift-truncation / partial-coverage artifacts there
    # produce spurious high-gradient lines (both in pool renders and real SAA)
    max_shift = float(np.abs(shifts).max()) if len(shifts) else 0.0
    cfg._border_margin_hr = int(np.ceil(max_shift * cfg.scale)) + int(np.ceil(3 * cfg.smooth_sigma_hr)) + cfg.scale  # type: ignore[attr-defined]
    edges = detect_edges(saa, valid_mask, cfg, rng)
    edge_rows: list[dict[str, Any]] = []
    profiles: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    extra_var = aperture_extra_var(cfg.aperture, cfg.scale)
    for e_i, edge in enumerate(edges):
        kernels = frame_kernels(cfg.aperture, cfg.scale, edge.nx, edge.ny, cfg.box_quad_points, shifts)
        d, v, f = collect_profile(burst, shifts, edge, cfg, rng)
        fit = fit_edge_profile(d, v, cfg, kernels, extra_var, frame_ids=f)
        row = {
            "edge_id": e_i,
            "nx": edge.nx,
            "ny": edge.ny,
            "angle_deg": float(np.degrees(np.arctan2(edge.ny, edge.nx))),
            "rho_hr": edge.rho_hr,
            "length_hr": edge.length_hr,
            "n_inliers": edge.n_inliers,
            "straightness_rms_hr": edge.straightness_rms_hr,
            "mean_grad": edge.mean_grad,
            **fit,
        }
        edge_rows.append(row)
        profiles.append((d, v, f))

    valid_rows = [r for r in edge_rows if r.get("valid")]
    sigma_hats = np.array([r["sigma_hat"] for r in valid_rows], dtype=np.float64)
    warnings: list[str] = []
    if len(sigma_hats) == 0:
        summary: dict[str, Any] = {
            "label": label,
            "kernel": "esf",
            "status": "no_usable_edges",
            "sigma_hat": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "n_edges_detected": len(edges),
            "n_edges_valid": 0,
            "rel_spread": float("nan"),
            "warnings": warnings,
            "aperture": cfg.aperture,
            "edges": edge_rows,
        }
        _persist(summary, edge_rows, profiles, cfg, out_dir, label, make_plot=False)
        return summary

    sigma_hat = float(np.median(sigma_hats))
    if len(sigma_hats) >= 3:
        q1, q3 = np.percentile(sigma_hats, [25, 75])
        rel_spread = float((q3 - q1) / max(sigma_hat, 1e-9))
    else:
        rel_spread = float("nan")
    if np.isfinite(rel_spread) and rel_spread > cfg.rel_spread_warn:
        warnings.append("edge_sigma_spread_high (anisotropic PSF or step-assumption violation)")

    if len(sigma_hats) >= 4:
        boots = np.array(
            [np.median(rng.choice(sigma_hats, size=len(sigma_hats), replace=True)) for _ in range(cfg.bootstrap_rounds)]
        )
    else:
        boots = _frame_bootstrap(profiles, edge_rows, shifts, cfg, rng)
    ci_lo = float(np.percentile(boots, 2.5)) if len(boots) else float("nan")
    ci_hi = float(np.percentile(boots, 97.5)) if len(boots) else float("nan")

    summary = {
        "label": label,
        "kernel": "esf",
        "status": "ok",
        "sigma_hat": sigma_hat,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "n_edges_detected": len(edges),
        "n_edges_valid": int(len(valid_rows)),
        "rel_spread": rel_spread,
        "warnings": warnings,
        "aperture": cfg.aperture,
        "edges": edge_rows,
    }
    _persist(summary, edge_rows, profiles, cfg, out_dir, label, make_plot=make_plot)
    return summary


def _frame_bootstrap(
    profiles: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    edge_rows: list[dict[str, Any]],
    shifts: np.ndarray,
    cfg: EsfSelfCalConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bootstrap over frames when too few edges exist for an edge bootstrap."""

    valid_ids = [i for i, r in enumerate(edge_rows) if r.get("valid")]
    if not valid_ids:
        return np.asarray([], dtype=float)
    extra_var = aperture_extra_var(cfg.aperture, cfg.scale)
    kernel_cache = {
        i: frame_kernels(cfg.aperture, cfg.scale, edge_rows[i]["nx"], edge_rows[i]["ny"], cfg.box_quad_points, shifts)
        for i in valid_ids
    }
    n_frames = len(shifts)
    out: list[float] = []
    for _ in range(int(cfg.frame_bootstrap_rounds)):
        counts = np.bincount(rng.integers(0, n_frames, size=n_frames), minlength=n_frames).astype(np.float64)
        sigs: list[float] = []
        for i in valid_ids:
            d, v, f = profiles[i]
            pw = counts[f]
            keep = pw > 0
            if keep.sum() < cfg.min_samples:
                continue
            fit = fit_edge_profile(
                d[keep], v[keep], cfg, kernel_cache[i], extra_var, frame_ids=f[keep], point_weights=pw[keep]
            )
            if fit.get("valid"):
                sigs.append(fit["sigma_hat"])
        if sigs:
            out.append(float(np.median(sigs)))
    return np.asarray(out, dtype=float)


def _persist(
    summary: dict[str, Any],
    edge_rows: list[dict[str, Any]],
    profiles: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    cfg: EsfSelfCalConfig,
    out_dir: Path | None,
    label: str,
    *,
    make_plot: bool,
) -> None:
    if out_dir is None:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slim = {k: v for k, v in summary.items() if k != "edges"}
    slim["edges"] = summary["edges"]
    (out_dir / f"{label}_esf_summary.json").write_text(json.dumps(slim, indent=2, default=float), encoding="utf-8")
    if edge_rows:
        import csv  # noqa: PLC0415

        keys = sorted({k for r in edge_rows for k in r})
        with (out_dir / f"{label}_esf_edges.csv").open("w", newline="", encoding="utf-8") as fh:
            wtr = csv.DictWriter(fh, fieldnames=keys, restval="")
            wtr.writeheader()
            wtr.writerows(edge_rows)
    if make_plot:
        _plot_profile(out_dir / f"{label}_esf_profile.png", summary, edge_rows, profiles, cfg)


def _plot_profile(
    path: Path,
    summary: dict[str, Any],
    edge_rows: list[dict[str, Any]],
    profiles: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    cfg: EsfSelfCalConfig,
) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    valid = [(i, r) for i, r in enumerate(edge_rows) if r.get("valid")]
    if not valid:
        return
    best_i, best = max(valid, key=lambda ir: ir[1]["amp_snr"])
    d, v, _ = profiles[best_i]
    order = np.argsort(d)
    sub = order[:: max(1, len(order) // 3000)]
    # dense display curve: frame-averaged kernel (mid-phase) is adequate for viz
    offs, w, _ = aperture_projection(cfg.aperture, cfg.scale, best["nx"], best["ny"], cfg.box_quad_points, frac_dx=0.5, frac_dy=0.5)
    extra_var = aperture_extra_var(cfg.aperture, cfg.scale)
    dense = np.linspace(float(d.min()), float(d.max()), 500)
    curve = esf_model(dense, best["amplitude"], best["center_lr_px"], best["sigma_hat"], best["baseline"], offs, w, extra_var)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(d[sub], v[sub], ".", ms=1.5, alpha=0.35, color="tab:blue", label="projected samples")
    ax1.plot(dense, curve, color="tab:orange", lw=1.5, label=f"fit sigma={best['sigma_hat']:.3f}")
    ax1.set_xlabel("signed distance to edge (LR px)")
    ax1.set_ylabel("value")
    ax1.set_title(f"best edge #{best['edge_id']} (r2={best['r2']:.3f})")
    ax1.legend(fontsize=8)
    sig = [r["sigma_hat"] for _, r in valid]
    ax2.errorbar(range(len(sig)), sig, fmt="o", color="tab:blue")
    ax2.axhline(summary["sigma_hat"], color="tab:orange", ls="--", label=f"median={summary['sigma_hat']:.3f}")
    if np.isfinite(summary["ci_lo"]):
        ax2.axhspan(summary["ci_lo"], summary["ci_hi"], color="tab:orange", alpha=0.12)
    ax2.set_xlabel("valid edge index")
    ax2.set_ylabel("sigma_hat (LR px)")
    ax2.legend(fontsize=8)
    fig.suptitle(f"ESF self-cal: {summary['label']} [{cfg.aperture}]")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Bench validation (prereg Step 1, E3 kernel)
# ---------------------------------------------------------------------------


def resolve_aperture(requested: str, forward_mode: str | None) -> str:
    """'auto' -> aperture preset from the pool's recorded forward_mode (default detector_box)."""

    if requested != "auto":
        return requested
    if forward_mode == "physical_block_average":
        return "pool_block_average"
    if forward_mode == "exact_ep06_point":
        return "pool_point"
    return "detector_box"


def _esf_bench_scene_task(args: dict[str, Any]) -> dict[str, Any]:
    scene = load_pool_scene(Path(args["scene_dir"]))
    md = scene["metadata"]
    cfg = EsfSelfCalConfig(**args["cfg_kwargs"])
    cfg.seed = int(args["seed"])
    # pool scenes without a recorded forward_mode default to the block-average render
    cfg.aperture = resolve_aperture(args["aperture"], md.get("forward_mode") or "physical_block_average")
    summary = run_esf_selfcal(
        np.asarray(scene["burst"], dtype=np.float64),
        scene["shifts"],
        cfg,
        out_dir=Path(args["out_dir"]) / "scenes",
        label=scene["scene_id"],
        make_plot=bool(args["scene_plots"]),
    )
    sigma_nominal = float(md.get("psf_sigma_lr_px", float("nan")))
    # ground truth in EFFECTIVE terms: pool blurs with a truncated discrete kernel
    # whose variance undershoots the nominal sigma at the small end (see helper)
    sigma_true = (
        discrete_gaussian_effective_sigma(sigma_nominal, int(md.get("scale", cfg.scale)))
        if np.isfinite(sigma_nominal)
        else float("nan")
    )
    hat = float(summary["sigma_hat"])
    rel = (hat - sigma_true) / sigma_true if sigma_true > 0 else float("nan")
    return {
        "scene_id": scene["scene_id"],
        "kernel": "esf",
        "status": summary["status"],
        "sigma_true_nominal": sigma_nominal,
        "sigma_true": sigma_true,
        "sigma_hat_esf": hat,
        "sigma_hat_e1": hat,  # key alias so evaluate_prereg/_plot_bench are reused unchanged
        "ci_lo": summary["ci_lo"],
        "ci_hi": summary["ci_hi"],
        "rel_err_signed": float(rel),
        "abs_rel_err": float(abs(rel)),
        "n_edges_valid": summary["n_edges_valid"],
        "rel_spread": summary["rel_spread"],
        "warnings": ";".join(summary["warnings"]),
        "psf_shape": str(md.get("psf_shape", "gaussian")),
        "noise_sigma_c": float(md.get("noise_sigma_c", float("nan"))),
        "delta_T_c": float(md.get("delta_T_c", float("nan"))),
        "aperture": cfg.aperture,
    }


def run_esf_bench_validation(
    pool_dir: Path,
    cfg: EsfSelfCalConfig,
    out_dir: Path,
    *,
    aperture: str = "auto",
    workers: int = 1,
    scene_limit: int | None = None,
    scene_plots: bool = True,
    median_tol: float = PREREG_MEDIAN_REL_ERR_TOL,
) -> dict[str, Any]:
    import os  # noqa: PLC0415

    pool_dir = Path(pool_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_dirs = sorted(p for p in pool_dir.iterdir() if p.is_dir() and (p / "metadata.json").exists())
    if scene_limit is not None:
        scene_dirs = scene_dirs[:scene_limit]
    if not scene_dirs:
        raise FileNotFoundError(f"no scene directories under {pool_dir}")

    cfg_kwargs = {k: getattr(cfg, k) for k in EsfSelfCalConfig.__dataclass_fields__ if k not in ("extras", "aperture", "seed")}
    cfg_kwargs["aperture"] = cfg.aperture  # placeholder; resolved per scene from metadata when 'auto'
    tasks = [
        {
            "scene_dir": str(d),
            "cfg_kwargs": cfg_kwargs,
            "seed": cfg.seed + i,
            "out_dir": str(out_dir),
            "scene_plots": scene_plots,
            "aperture": aperture,
        }
        for i, d in enumerate(scene_dirs)
    ]

    if workers > 1:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[var] = "1"
        import multiprocessing as mp  # noqa: PLC0415

        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            rows = list(pool.map(_esf_bench_scene_task, tasks))
    else:
        rows = [_esf_bench_scene_task(t) for t in tasks]

    verdict = evaluate_prereg(rows, median_tol=median_tol)
    verdict["n_scenes_total"] = len(rows)
    verdict["n_no_edge_scenes"] = int(sum(1 for r in rows if r["status"] == "no_usable_edges"))
    verdict["kernel"] = "esf"
    _write_rows_csv(out_dir / "bench_rows.csv", rows)
    (out_dir / "bench_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    _plot_bench(out_dir / "bench_summary.png", rows, verdict)
    return {"rows": rows, "verdict": verdict}


def make_config(base: EsfSelfCalConfig | None = None, **overrides: Any) -> EsfSelfCalConfig:
    """Small helper for CLI/test construction."""

    cfg = base or EsfSelfCalConfig()
    return replace(cfg, **overrides)
