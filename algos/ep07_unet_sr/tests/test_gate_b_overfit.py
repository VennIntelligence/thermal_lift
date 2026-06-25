"""Gate B — single clean scene, end-to-end geometry + trainability check (remote GPU).

Gate A certified the operator in isolation. Gate B is the END-TO-END FM-6 catch that Gate A
cannot see: on ONE clean synthetic scene (no noise/drift/defects) it verifies
  (1) data-fitting min ||A x - y||^2 drives the DC residual toward ~0  -> the assumed shifts/
      PSF/offset really do explain the observed burst (a half-pixel/sign bug plateaus here);
  (2) the recovered x correlates with the GT field -> reconstruction is geometrically aligned;
  (3) the full UnrolledSolver runs forward+backward with finite grads -> the unroll is trainable
      (double-backward through the autograd DC step works).

Usage on the remote:  python algos/ep07_unet_sr/tests/test_gate_b_overfit.py
Exit 0 = PASS (proceed to Gate C / smoke); nonzero = STOP, geometry/plumbing bug.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
for p in (_ROOT / "tcforge" / "src", _ROOT / "algos" / "ep07_unet_sr" / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tcforge import build_scene_mask_with_metadata, generate_lr_burst, render_temperature_field  # noqa: E402
from tcforge import shifts as shift_module  # noqa: E402
from unet_sr.forward_torch import ScenePSF, forward_burst  # noqa: E402
from unet_sr.unroll import UnrolledSolver  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_clean_scene(scale=2, H=128, W=128, M=16, sigma=0.25):
    """One CLEAN scene (no noise/drift/defects): GT field + clean burst + shifts + psf."""
    hr_mask, _ = build_scene_mask_with_metadata(
        "hard", 7, rotation_deg_center=31.0, rotation_jitter_deg=0.0,
        canvas_shape=(H, W), pixel_size_um=20.0, scale=scale, antialias=True,
        ssaa_factor=6, inscribe_disc=True,
    )
    gt = render_temperature_field(hr_mask, t_bg_c=21.0, delta_t_c=3.0,
                                  low_freq_amplitude_c=0.2, low_freq_sigma_px=96.0, seed=123)
    shifts, _ = shift_module.build_scene_shifts(
        7, M, {"mode": "random", "coverage_quality": "good", "include_real_like_fraction": 0.0,
               "n_phase_bins": 4, "seed_offset": 0}, scale=scale)
    burst = generate_lr_burst(gt.astype(np.float32), shifts, forward_mode="physical_block_average",
                              psf_sigma_lr_px=sigma, scale=scale)  # CLEAN forward, no nuisances
    return gt.astype(np.float32), burst.astype(np.float32), shifts.astype(np.float32), sigma


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-20))


def main() -> int:
    scale = 2
    gt_np, burst_np, shifts_np, sigma = make_clean_scene(scale=scale)
    H, W = gt_np.shape
    M = burst_np.shape[0]
    print(f"Gate B on {DEVICE}: clean scene {H}x{W}, M={M} frames, gaussian sigma={sigma}")

    gt = torch.tensor(gt_np, device=DEVICE).view(1, 1, H, W)
    burst = torch.tensor(burst_np, device=DEVICE).view(1, M, H // scale, W // scale)
    shifts = torch.tensor(shifts_np, device=DEVICE).view(1, M, 2)
    psf = ScenePSF(sigma_lr_px=torch.tensor([sigma], device=DEVICE), shape=["gaussian"],
                   sigma_y_lr_px=[None], angle_deg=torch.tensor([0.0], device=DEVICE))

    # (1)/(2) data-fitting: Adam on x minimizing ||A x - y||^2 (full band), warm-start = up(mean)
    x0 = torch.nn.functional.interpolate(burst.mean(1, keepdim=True), scale_factor=scale,
                                         mode="bilinear", align_corners=False)
    x = x0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=0.05)

    def resid_rms(xx):
        with torch.no_grad():
            return float((forward_burst(xx, shifts, psf, scale) - burst).pow(2).mean().sqrt())

    r0 = resid_rms(x)
    for i in range(400):
        opt.zero_grad()
        loss = (forward_burst(x, shifts, psf, scale) - burst).pow(2).mean()
        loss.backward()
        opt.step()
        if i in (49, 199, 399):
            print(f"  step {i + 1:>4}: DC residual RMS = {resid_rms(x):.3e}   corr(x,GT) = {corr(x, gt):.4f}")
    rf = resid_rms(x)
    c = corr(x, gt)

    # (3) solver forward+backward smoke (trainability / double-backward)
    solver = UnrolledSolver(n_steps=3, cond_channels=8, base_channels=32, scale=scale,
                            band_highpass_sigma_lr_px=5.0).to(DEVICE).train()
    cond = torch.zeros(1, 8, H, W, device=DEVICE)
    pred = solver(x0.detach(), burst, shifts, psf, cond)
    sloss = (pred - gt).pow(2).mean()
    sloss.backward()
    grads_finite = all(p.grad is not None and torch.isfinite(p.grad).all()
                       for p in solver.parameters() if p.requires_grad)

    print(f"\n[1] DC residual: {r0:.3e} -> {rf:.3e}   (drop {r0 / max(rf, 1e-30):.1f}x; want >10x)")
    print(f"[2] corr(recovered x, GT) = {c:.4f}   (want > 0.90: geometric alignment OK)")
    print(f"[3] solver fwd+bwd finite grads = {grads_finite}   (double-backward through unroll OK)")
    ok = (rf < 0.1 * r0) and (c > 0.90) and grads_finite
    print("\nGATE B:", "PASS — proceed to Gate C / smoke" if ok else "FAIL — STOP, geometry/plumbing bug")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
