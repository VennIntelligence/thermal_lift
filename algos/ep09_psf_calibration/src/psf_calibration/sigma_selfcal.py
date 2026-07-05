"""Self-supervised PSF sigma calibration (ACL-056, prereg design: research_log/sigma_selfcal_prereg_design.md).

Two mutually-checking estimators over a candidate sigma grid, both distribution-agnostic
and free of any learned prior (the reconstructor is a linear CG solve on the certified
EP06 numpy operator; Tikhonov lambda is a pure numerical stabilizer, constant across the
sigma grid so it cannot bias the argmin):

- E1 leave-frames-out forward prediction: reconstruct from a random frame subset,
  predict held-out frames through the candidate-sigma forward model, score MSE.
  Bootstrap rounds over random splits give sigma_hat with a percentile CI.
- E2 residual whiteness: full-pool reconstruction residuals should be noise-like
  under the correct operator; scored by spectral flatness (primary) and
  autocorrelation decay length (secondary).

Agreement between E1 and E2 is reported, never forced. Bench mode validates the
whole procedure against known true sigma on a synthetic pool BEFORE it may be
trusted on real data (Step 1 of the prereg protocol).

KNOWN LIMITATION (ACL-056 finding — read before trusting any output):
for this operator family the Gaussian blur commutes with the sub-pixel shift and
(effectively) the block downsampling, so for any assumed sigma' there exists a
compensating reconstruction x_hat = B_sigma'^{-1} B_sigma x that reproduces ALL
frames — held-out prediction (E1) and residual whiteness (E2) are therefore
near-DEGENERATE in sigma at realistic image sizes (measured: E1 curves flatten
as CG converges; tiny-image recovery in the unit tests works only through
border effects). This module is retained as the prereg Step-1 falsification
instrument and as scaffolding; an identifiable estimator needs a constraint the
compensator cannot satisfy (e.g. parametric edge/ESF fitting across frames).
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from psf_calibration.utils import parabolic_minimum

DEFAULT_SIGMA_GRID: tuple[float, ...] = tuple(
    float(f"{v:.4f}") for v in np.geomspace(0.05, 1.3, 12)
)
PREREG_MEDIAN_REL_ERR_TOL = 0.15  # prereg acceptance line (design doc §2 Step 1)


# ---------------------------------------------------------------------------
# Linear reconstructor (no learned prior)
# ---------------------------------------------------------------------------


def _operator(shifts: np.ndarray, sigma: float, hr_shape: tuple[int, int], scale: int):
    from tcforge._ep06_reference.forward import build_observation_operator  # noqa: PLC0415

    return build_observation_operator(hr_shape, shifts=shifts, psf_sigma=float(sigma), scale=scale)


def _mean_normal_matvec(op, frame_idx: np.ndarray, x: np.ndarray) -> np.ndarray:
    total = np.zeros_like(x)
    for j in frame_idx:
        total += op.adjoint(op.forward(x, int(j)), int(j))
    return total / float(len(frame_idx))


def _mean_backprojection(op, frame_idx: np.ndarray, burst: np.ndarray) -> np.ndarray:
    total = np.zeros(op.hr_shape, dtype=np.float64)
    for j in frame_idx:
        frame = np.where(np.isfinite(burst[j]), burst[j], 0.0)
        total += op.adjoint(frame, int(j))
    return total / float(len(frame_idx))


def cg_reconstruct(
    burst: np.ndarray,
    shifts: np.ndarray,
    frame_idx: np.ndarray,
    sigma: float,
    *,
    scale: int,
    lam_eff: float,
    cg_iters: int,
    x0: np.ndarray | None = None,
    rtol: float = 1e-6,
) -> np.ndarray:
    """Solve (1/M sum A^T A + lam_eff I) x = 1/M sum A^T y by conjugate gradients."""

    hr_shape = (burst.shape[1] * scale, burst.shape[2] * scale)
    op = _operator(shifts, sigma, hr_shape, scale)
    b = _mean_backprojection(op, frame_idx, burst)
    x = b.copy() if x0 is None else x0.astype(np.float64, copy=True)

    def matvec(v: np.ndarray) -> np.ndarray:
        return _mean_normal_matvec(op, frame_idx, v) + lam_eff * v

    r = b - matvec(x)
    p = r.copy()
    rs = float(np.vdot(r, r).real)
    b_norm = float(np.linalg.norm(b)) or 1.0
    for _ in range(int(cg_iters)):
        if np.sqrt(rs) / b_norm < rtol:
            break
        ap = matvec(p)
        denom = float(np.vdot(p, ap).real)
        if denom <= 0:
            break
        alpha = rs / denom
        x += alpha * p
        r -= alpha * ap
        rs_new = float(np.vdot(r, r).real)
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x


def _lambda_scale(burst: np.ndarray, shifts: np.ndarray, sigma_mid: float, scale: int) -> float:
    """Empirical diagonal scale g of the mean-normal operator (self-scaling, data-agnostic)."""

    hr_shape = (burst.shape[1] * scale, burst.shape[2] * scale)
    op = _operator(shifts, sigma_mid, hr_shape, scale)
    idx = np.arange(min(len(shifts), 8))
    ones = np.ones(hr_shape, dtype=np.float64)
    return float(np.mean(_mean_normal_matvec(op, idx, ones)))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _crop_margin_lr_px(shifts: np.ndarray) -> int:
    return int(np.ceil(np.abs(shifts).max())) + 2 if len(shifts) else 2


def _crop(frame: np.ndarray, margin: int) -> np.ndarray:
    if margin <= 0 or 2 * margin >= min(frame.shape):
        return frame
    return frame[margin:-margin, margin:-margin]


def _masked_mse(pred: np.ndarray, obs: np.ndarray, margin: int) -> float:
    p = _crop(pred, margin)
    o = _crop(obs, margin)
    mask = np.isfinite(o)
    diff = (p - np.where(mask, o, 0.0)) * mask
    denom = int(mask.sum())
    return float((diff**2).sum() / denom) if denom else float("nan")


# ---------------------------------------------------------------------------
# E1: leave-frames-out forward prediction
# ---------------------------------------------------------------------------


@dataclass
class SelfCalConfig:
    sigma_grid: tuple[float, ...] = DEFAULT_SIGMA_GRID
    rounds: int = 8
    holdout_frac: float = 0.25
    cg_iters: int = 40
    lam: float = 1e-3
    max_frames: int | None = 48
    crop_lr: int | None = None  # centered LR crop (compute knob; same crop for every sigma → cannot bias the argmin)
    seed: int = 0
    scale: int = 2
    agree_tol_grid_steps: float = 1.5
    extras: dict[str, Any] = field(default_factory=dict)


def _center_crop_burst(burst: np.ndarray, crop_lr: int | None, scale: int) -> np.ndarray:
    """Deterministic centered crop of every LR frame (dims kept multiples of scale)."""

    if crop_lr is None:
        return burst
    n, h, w = burst.shape
    ch = min(int(crop_lr), h)
    cw = min(int(crop_lr), w)
    ch -= ch % scale
    cw -= cw % scale
    if ch <= 0 or cw <= 0:
        return burst
    top = (h - ch) // 2
    left = (w - cw) // 2
    return burst[:, top : top + ch, left : left + cw]


def _frame_pool(n_frames: int, cfg: SelfCalConfig, rng: np.random.Generator) -> np.ndarray:
    if cfg.max_frames is not None and n_frames > cfg.max_frames:
        return np.sort(rng.choice(n_frames, size=cfg.max_frames, replace=False))
    return np.arange(n_frames)


def e1_leave_frames_out(
    burst: np.ndarray, shifts: np.ndarray, cfg: SelfCalConfig, rng: np.random.Generator
) -> dict[str, Any]:
    grid = np.asarray(cfg.sigma_grid, dtype=np.float64)
    pool = _frame_pool(len(burst), cfg, rng)
    margin = _crop_margin_lr_px(shifts[pool])
    lam_eff = cfg.lam * _lambda_scale(burst, shifts, float(np.median(grid)), cfg.scale)
    hr_shape = (burst.shape[1] * cfg.scale, burst.shape[2] * cfg.scale)

    n_hold = max(2, int(round(cfg.holdout_frac * len(pool))))
    curves = np.full((cfg.rounds, len(grid)), np.nan)
    sigma_hats: list[float] = []
    for r in range(cfg.rounds):
        perm = rng.permutation(pool)
        hold, train = perm[:n_hold], np.sort(perm[n_hold:])
        x_warm: np.ndarray | None = None
        for k, sigma in enumerate(grid):
            x_hat = cg_reconstruct(
                burst, shifts, train, float(sigma),
                scale=cfg.scale, lam_eff=lam_eff, cg_iters=cfg.cg_iters, x0=x_warm,
            )
            x_warm = x_hat
            op = _operator(shifts, float(sigma), hr_shape, cfg.scale)
            errs = [
                _masked_mse(op.forward(x_hat, int(j)), np.asarray(burst[j], dtype=np.float64), margin)
                for j in hold
            ]
            curves[r, k] = float(np.nanmean(errs))
        sig, ok = parabolic_minimum(grid, curves[r])
        sigma_hats.append(float(sig) if ok else float(grid[int(np.nanargmin(curves[r]))]))

    hats = np.asarray(sigma_hats, dtype=np.float64)
    return {
        "grid": grid.tolist(),
        "curves": curves.tolist(),
        "curve_mean": np.nanmean(curves, axis=0).tolist(),
        "curve_std": np.nanstd(curves, axis=0).tolist(),
        "sigma_hats": hats.tolist(),
        "sigma_hat": float(np.median(hats)),
        "ci_lo": float(np.percentile(hats, 2.5)),
        "ci_hi": float(np.percentile(hats, 97.5)),
        "n_frames_used": int(len(pool)),
        "n_holdout": int(n_hold),
        "crop_margin_lr_px": int(margin),
    }


# ---------------------------------------------------------------------------
# E2: residual whiteness
# ---------------------------------------------------------------------------


def _radial_bins(shape: tuple[int, int], n_bins: int = 16) -> tuple[np.ndarray, int]:
    yy = np.fft.fftfreq(shape[0])[:, None]
    xx = np.fft.fftfreq(shape[1])[None, :]
    rr = np.sqrt(yy**2 + xx**2)
    edges = np.linspace(0.0, 0.5, n_bins + 1)
    which = np.clip(np.digitize(rr, edges) - 1, 0, n_bins - 1)
    return which, n_bins


def spectral_flatness(residual: np.ndarray, n_bins: int = 16) -> float:
    """Flatness in (0,1]; 1 = white. Geometric/arithmetic mean of radial power bins (DC bin excluded).

    Bin count adapts to small images; empty bins are dropped rather than poisoning the mean.
    """

    res = residual - float(np.mean(residual))
    nb_eff = max(4, min(int(n_bins), min(res.shape) // 2))
    wy = np.hanning(res.shape[0])[:, None]
    wx = np.hanning(res.shape[1])[None, :]
    power = np.abs(np.fft.fft2(res * wy * wx)) ** 2
    which, nb = _radial_bins(res.shape, nb_eff)
    sums = np.bincount(which.ravel(), weights=power.ravel(), minlength=nb)
    counts = np.bincount(which.ravel(), minlength=nb)
    with np.errstate(invalid="ignore"):
        bins = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    bins = bins[1:]  # drop the DC-dominated bin
    bins = bins[np.isfinite(bins)]
    if len(bins) < 2:
        return float("nan")
    bins = np.maximum(bins, 1e-30)
    return float(np.exp(np.mean(np.log(bins))) / np.mean(bins))


def autocorr_decay_length(residual: np.ndarray) -> float:
    """First radius (px) where the radially-averaged normalized autocorrelation drops below 1/e."""

    res = residual - float(np.mean(residual))
    power = np.abs(np.fft.fft2(res)) ** 2
    ac = np.fft.ifft2(power).real
    ac /= max(ac[0, 0], 1e-30)
    h, w = ac.shape
    r_max = min(h, w) // 2
    yy = np.minimum(np.arange(h), h - np.arange(h))[:, None]
    xx = np.minimum(np.arange(w), w - np.arange(w))[None, :]
    rr = np.sqrt(yy**2 + xx**2)
    prof = np.array([ac[(rr >= r - 0.5) & (rr < r + 0.5)].mean() for r in range(r_max)])
    thresh = 1.0 / np.e
    below = np.where(prof < thresh)[0]
    if len(below) == 0:
        return float(r_max)
    i = int(below[0])
    if i == 0:
        return 0.0
    a, b = prof[i - 1], prof[i]
    frac = (a - thresh) / (a - b) if a != b else 0.0
    return float(i - 1 + frac)


def e2_residual_whiteness(
    burst: np.ndarray, shifts: np.ndarray, cfg: SelfCalConfig, rng: np.random.Generator
) -> dict[str, Any]:
    grid = np.asarray(cfg.sigma_grid, dtype=np.float64)
    pool = _frame_pool(len(burst), cfg, rng)
    margin = _crop_margin_lr_px(shifts[pool])
    lam_eff = cfg.lam * _lambda_scale(burst, shifts, float(np.median(grid)), cfg.scale)
    hr_shape = (burst.shape[1] * cfg.scale, burst.shape[2] * cfg.scale)

    flatness = np.zeros(len(grid))
    decay = np.zeros(len(grid))
    x_warm: np.ndarray | None = None
    for k, sigma in enumerate(grid):
        x_hat = cg_reconstruct(
            burst, shifts, pool, float(sigma),
            scale=cfg.scale, lam_eff=lam_eff, cg_iters=cfg.cg_iters, x0=x_warm,
        )
        x_warm = x_hat
        op = _operator(shifts, float(sigma), hr_shape, cfg.scale)
        f_vals, d_vals = [], []
        for j in pool:
            obs = np.asarray(burst[j], dtype=np.float64)
            resid = _crop(op.forward(x_hat, int(j)) - np.where(np.isfinite(obs), obs, 0.0), margin)
            f_vals.append(spectral_flatness(resid))
            d_vals.append(autocorr_decay_length(resid))
        flatness[k] = float(np.mean(f_vals))
        decay[k] = float(np.mean(d_vals))

    return {
        "grid": grid.tolist(),
        "flatness": flatness.tolist(),
        "decay_len_px": decay.tolist(),
        "sigma_hat_flatness": _safe_extremum(grid, -flatness),
        "sigma_hat_decay": _safe_extremum(grid, decay),
        "n_frames_used": int(len(pool)),
    }


def _safe_extremum(grid: np.ndarray, values: np.ndarray) -> float:
    """Parabolic-refined argmin with NaN defence (NaN in → NaN out, never a crash)."""

    vals = np.asarray(values, dtype=np.float64)
    if not np.isfinite(vals).any():
        return float("nan")
    sig, _ok = parabolic_minimum(grid, vals)
    return float(sig)


# ---------------------------------------------------------------------------
# Combined run + persistence
# ---------------------------------------------------------------------------


def _local_grid_step(grid: np.ndarray, sigma: float) -> float:
    k = int(np.clip(np.searchsorted(grid, sigma), 1, len(grid) - 1))
    return float(grid[k] - grid[k - 1])


def run_selfcal(
    burst: np.ndarray,
    shifts: np.ndarray,
    cfg: SelfCalConfig,
    *,
    out_dir: Path | None = None,
    label: str = "burst",
    make_plot: bool = True,
) -> dict[str, Any]:
    burst = np.asarray(burst, dtype=np.float64)
    shifts = np.asarray(shifts, dtype=np.float64)
    if burst.ndim != 3 or shifts.shape != (len(burst), 2):
        raise ValueError("burst must be (N,H,W) and shifts (N,2)")
    burst = _center_crop_burst(burst, cfg.crop_lr, cfg.scale)
    rng = np.random.default_rng(cfg.seed)
    e1 = e1_leave_frames_out(burst, shifts, cfg, rng)
    e2 = e2_residual_whiteness(burst, shifts, cfg, rng)

    grid = np.asarray(cfg.sigma_grid, dtype=np.float64)
    tol = cfg.agree_tol_grid_steps * _local_grid_step(grid, e1["sigma_hat"])
    agreement = "agree" if abs(e1["sigma_hat"] - e2["sigma_hat_flatness"]) <= tol else "diverge"

    summary: dict[str, Any] = {
        "label": label,
        "sigma_hat_e1": e1["sigma_hat"],
        "ci_lo": e1["ci_lo"],
        "ci_hi": e1["ci_hi"],
        "sigma_hat_e2_flatness": e2["sigma_hat_flatness"],
        "sigma_hat_e2_decay": e2["sigma_hat_decay"],
        "agreement": agreement,
        "agree_tol": tol,
        "config": {
            "sigma_grid": list(cfg.sigma_grid),
            "rounds": cfg.rounds,
            "holdout_frac": cfg.holdout_frac,
            "cg_iters": cfg.cg_iters,
            "lam": cfg.lam,
            "max_frames": cfg.max_frames,
            "crop_lr": cfg.crop_lr,
            "seed": cfg.seed,
            "scale": cfg.scale,
        },
        "e1": e1,
        "e2": e2,
    }
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_curve_csv(out_dir / f"{label}_curves.csv", e1, e2)
        (out_dir / f"{label}_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        if make_plot:
            _plot_curves(out_dir / f"{label}_curves.png", label, e1, e2)
    return summary


def _write_curve_csv(path: Path, e1: dict[str, Any], e2: dict[str, Any]) -> None:
    import csv  # noqa: PLC0415

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sigma", "e1_mse_mean", "e1_mse_std", "e2_flatness", "e2_decay_len_px"])
        for i, sigma in enumerate(e1["grid"]):
            w.writerow([sigma, e1["curve_mean"][i], e1["curve_std"][i], e2["flatness"][i], e2["decay_len_px"][i]])


def _plot_curves(path: Path, label: str, e1: dict[str, Any], e2: dict[str, Any]) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    grid = np.asarray(e1["grid"])
    mean = np.asarray(e1["curve_mean"])
    std = np.asarray(e1["curve_std"])
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(grid, mean, "o-", color="tab:blue", label="E1 heldout MSE")
    ax1.fill_between(grid, mean - std, mean + std, alpha=0.2, color="tab:blue")
    ax1.axvline(e1["sigma_hat"], color="tab:blue", ls="--", lw=1)
    ax1.axvspan(e1["ci_lo"], e1["ci_hi"], color="tab:blue", alpha=0.08)
    ax1.set_xlabel("sigma (LR px)")
    ax1.set_ylabel("E1 heldout MSE", color="tab:blue")
    ax1.set_xscale("log")
    ax2 = ax1.twinx()
    ax2.plot(grid, e2["flatness"], "s-", color="tab:orange", label="E2 flatness")
    ax2.axvline(e2["sigma_hat_flatness"], color="tab:orange", ls=":", lw=1)
    ax2.set_ylabel("E2 spectral flatness", color="tab:orange")
    fig.suptitle(f"sigma self-cal: {label}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Bench validation (prereg Step 1)
# ---------------------------------------------------------------------------


def load_pool_scene(scene_dir: Path) -> dict[str, Any]:
    """Plain-numpy compact-scene loader (no tcforge.storage dependency)."""

    scene_dir = Path(scene_dir)
    burst_path = scene_dir / "lr_burst.npy"
    if not burst_path.exists():
        raise FileNotFoundError(f"{scene_dir} has no lr_burst.npy (pool must be generated with save_lr_burst)")
    metadata = json.loads((scene_dir / "metadata.json").read_text(encoding="utf-8"))
    return {
        "scene_id": scene_dir.name,
        "burst": np.load(burst_path, mmap_mode="r"),
        "shifts": np.load(scene_dir / "shifts.npy").astype(np.float64),
        "metadata": metadata,
    }


def _bench_scene_task(args: dict[str, Any]) -> dict[str, Any]:
    scene = load_pool_scene(Path(args["scene_dir"]))
    md = scene["metadata"]
    cfg = SelfCalConfig(**args["cfg_kwargs"])
    cfg.seed = int(args["seed"])
    out_dir = Path(args["out_dir"]) / "scenes"
    summary = run_selfcal(
        np.asarray(scene["burst"], dtype=np.float64),
        scene["shifts"],
        cfg,
        out_dir=out_dir,
        label=scene["scene_id"],
        make_plot=bool(args["scene_plots"]),
    )
    sigma_true = float(md.get("psf_sigma_lr_px", float("nan")))
    hat = float(summary["sigma_hat_e1"])
    rel = (hat - sigma_true) / sigma_true if sigma_true > 0 else float("nan")
    return {
        "scene_id": scene["scene_id"],
        "sigma_true": sigma_true,
        "sigma_hat_e1": hat,
        "ci_lo": summary["ci_lo"],
        "ci_hi": summary["ci_hi"],
        "sigma_hat_e2_flatness": summary["sigma_hat_e2_flatness"],
        "sigma_hat_e2_decay": summary["sigma_hat_e2_decay"],
        "agreement": summary["agreement"],
        "rel_err_signed": float(rel),
        "abs_rel_err": float(abs(rel)),
        "psf_shape": str(md.get("psf_shape", "gaussian")),
        "noise_sigma_c": float(md.get("noise_sigma_c", float("nan"))),
        "delta_T_c": float(md.get("delta_T_c", float("nan"))),
        "n_frames_used": summary["e1"]["n_frames_used"],
    }


def evaluate_prereg(
    rows: list[dict[str, Any]],
    *,
    median_tol: float = PREREG_MEDIAN_REL_ERR_TOL,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Prereg Step 1 verdict: median |rel err| <= tol AND no systematic noise-tertile bias.

    Systematic bias = the median SIGNED relative error has the same sign in all three
    noise tertiles AND every tertile's bootstrap 95% CI excludes zero.
    """

    rel = np.array([r["rel_err_signed"] for r in rows], dtype=np.float64)
    noise = np.array([r["noise_sigma_c"] for r in rows], dtype=np.float64)
    ok = np.isfinite(rel) & np.isfinite(noise)
    rel, noise = rel[ok], noise[ok]
    if len(rel) < 6:
        raise ValueError("need at least 6 scenes with finite rel_err and noise for the prereg verdict")

    median_abs = float(np.median(np.abs(rel)))
    q1, q2 = np.quantile(noise, [1 / 3, 2 / 3])
    tertile_of = np.digitize(noise, [q1, q2])
    rng = np.random.default_rng(seed)
    tertiles: list[dict[str, Any]] = []
    for t in range(3):
        vals = rel[tertile_of == t]
        boots = np.array([np.median(rng.choice(vals, size=len(vals), replace=True)) for _ in range(n_boot)])
        tertiles.append(
            {
                "tertile": t,
                "n": int(len(vals)),
                "median_signed": float(np.median(vals)),
                "ci_lo": float(np.percentile(boots, 2.5)),
                "ci_hi": float(np.percentile(boots, 97.5)),
            }
        )
    signs = [np.sign(t["median_signed"]) for t in tertiles]
    excludes_zero = [t["ci_lo"] > 0 or t["ci_hi"] < 0 for t in tertiles]
    systematic_bias = bool(len(set(signs)) == 1 and signs[0] != 0 and all(excludes_zero))

    gauss = [r for r in rows if r.get("psf_shape") == "gaussian" and np.isfinite(r["rel_err_signed"])]
    median_abs_gauss = float(np.median([abs(r["rel_err_signed"]) for r in gauss])) if gauss else float("nan")

    return {
        "n_scenes": int(len(rel)),
        "median_abs_rel_err": median_abs,
        "median_abs_rel_err_gaussian_only": median_abs_gauss,
        "median_tol": float(median_tol),
        "median_ok": bool(median_abs <= median_tol),
        "noise_tertiles": tertiles,
        "systematic_bias": systematic_bias,
        "prereg_pass": bool(median_abs <= median_tol and not systematic_bias),
    }


def run_bench_validation(
    pool_dir: Path,
    cfg: SelfCalConfig,
    out_dir: Path,
    *,
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

    cfg_kwargs = {
        "sigma_grid": tuple(cfg.sigma_grid),
        "rounds": cfg.rounds,
        "holdout_frac": cfg.holdout_frac,
        "cg_iters": cfg.cg_iters,
        "lam": cfg.lam,
        "max_frames": cfg.max_frames,
        "crop_lr": cfg.crop_lr,
        "scale": cfg.scale,
        "agree_tol_grid_steps": cfg.agree_tol_grid_steps,
    }
    tasks = [
        {
            "scene_dir": str(d),
            "cfg_kwargs": cfg_kwargs,
            "seed": cfg.seed + i,
            "out_dir": str(out_dir),
            "scene_plots": scene_plots,
        }
        for i, d in enumerate(scene_dirs)
    ]

    if workers > 1:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[var] = "1"
        import multiprocessing as mp  # noqa: PLC0415

        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            rows = list(pool.map(_bench_scene_task, tasks))
    else:
        rows = [_bench_scene_task(t) for t in tasks]

    verdict = evaluate_prereg(rows, median_tol=median_tol)
    _write_rows_csv(out_dir / "bench_rows.csv", rows)
    (out_dir / "bench_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    _plot_bench(out_dir / "bench_summary.png", rows, verdict)
    return {"rows": rows, "verdict": verdict}


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv  # noqa: PLC0415

    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _plot_bench(path: Path, rows: list[dict[str, Any]], verdict: dict[str, Any]) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    shapes = sorted({r["psf_shape"] for r in rows})
    for shape in shapes:
        sub = [r for r in rows if r["psf_shape"] == shape]
        ax1.scatter(
            [r["sigma_true"] for r in sub],
            [r["sigma_hat_e1"] for r in sub],
            label=shape,
            alpha=0.8,
        )
    lims = [
        min(min((r["sigma_true"] for r in rows), default=0.05), 0.05),
        max(max((r["sigma_true"] for r in rows), default=1.3), 1.3),
    ]
    ax1.plot(lims, lims, "k--", lw=1)
    ax1.set_xlabel("sigma_true (LR px)")
    ax1.set_ylabel("sigma_hat E1 (LR px)")
    ax1.legend(fontsize=8)
    ax2.hist([r["rel_err_signed"] for r in rows if np.isfinite(r["rel_err_signed"])], bins=24)
    ax2.axvline(0, color="k", lw=1)
    ax2.set_xlabel("signed relative error")
    status = "PASS" if verdict["prereg_pass"] else "FAIL"
    fig.suptitle(
        f"sigma self-cal bench validation — {status} "
        f"(median |rel err| = {verdict['median_abs_rel_err']:.3f}, tol {verdict['median_tol']:.2f})"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
