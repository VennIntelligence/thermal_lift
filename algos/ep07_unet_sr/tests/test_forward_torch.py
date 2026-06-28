"""Gate A — certify the torch forward operator BEFORE any long training run.

Runs on the remote GPU box (needs torch + tcforge).  Three checks:
  1. forward parity: torch forward_burst == numpy generate_lr_burst (gaussian/elliptical/airy)
  2. linearity: A(a*u + b*v) == a*A(u) + b*A(v)
  3. autograd transpose identity: <A u, v> == <u, A^T v>, where A^T v := grad_u <A u, v>
     (this certifies the autograd-derived adjoint used by the DC step IS the exact transpose,
      which the hand-written numpy/ep08/ep15 adjoints are NOT — see forward_torch.py docstring)

Usage on the remote:  python algos/ep07_unet_sr/tests/test_forward_torch.py
Exit code 0 = PASS (proceed to Gate B); nonzero = STOP and fix before training.
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

from tcforge.forward import generate_lr_burst  # noqa: E402
from unet_sr.forward_torch import ScenePSF, data_consistency_grad, forward_burst  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DT = torch.float64  # double precision so parity tolerances are tight and unambiguous


def _scene_psf(sigma, shape, sigma_y, angle):
    return ScenePSF(
        sigma_lr_px=torch.tensor([sigma], dtype=DT, device=DEVICE),
        shape=[shape],
        sigma_y_lr_px=[sigma_y],
        angle_deg=torch.tensor([angle], dtype=DT, device=DEVICE),
    )


def check_forward_parity() -> float:
    rng = np.random.default_rng(0)
    scale = 2
    worst = 0.0
    cases = [
        dict(shape="gaussian", sigma_y=None, angle=0.0),
        dict(shape="elliptical_gaussian", sigma_y=0.45, angle=37.0),
        dict(shape="airy_disk", sigma_y=None, angle=0.0),
    ]
    for case in cases:
        for _ in range(6):
            H, W = 64, 96
            x = rng.standard_normal((H, W)).astype(np.float64)
            sigma = float(rng.uniform(0.15, 0.55))
            shifts = rng.uniform(-2.0, 2.0, size=(10, 2)).astype(np.float64)
            ref = generate_lr_burst(
                x.astype(np.float32), shifts.astype(np.float32),
                forward_mode="physical_block_average",
                psf_sigma_lr_px=sigma, psf_shape=case["shape"],
                psf_sigma_y_lr_px=case["sigma_y"], psf_angle_deg=case["angle"], scale=scale,
            )
            xt = torch.tensor(x, dtype=DT, device=DEVICE).view(1, 1, H, W)
            st = torch.tensor(shifts, dtype=DT, device=DEVICE).view(1, 10, 2)
            psf = _scene_psf(sigma, case["shape"], case["sigma_y"], case["angle"])
            got = forward_burst(xt, st, psf, scale)[0].cpu().numpy()
            d = float(np.abs(got - ref).max())
            worst = max(worst, d)
            print(f"  parity {case['shape']:>18} sigma={sigma:.3f}: max|torch-numpy|={d:.2e}")
    return worst


def check_linearity() -> float:
    torch.manual_seed(1)
    scale = 2
    psf = _scene_psf(0.35, "elliptical_gaussian", 0.5, 20.0)
    u = torch.randn(1, 1, 64, 96, dtype=DT, device=DEVICE)
    v = torch.randn(1, 1, 64, 96, dtype=DT, device=DEVICE)
    s = torch.empty(1, 8, 2, dtype=DT, device=DEVICE).uniform_(-2, 2)
    a, b = 2.3, -1.7
    lhs = forward_burst(a * u + b * v, s, psf, scale)
    rhs = a * forward_burst(u, s, psf, scale) + b * forward_burst(v, s, psf, scale)
    return float((lhs - rhs).abs().max())


def check_adjoint_identity() -> float:
    """<A u, v> == <u, A^T v>, A^T v computed by autograd."""
    torch.manual_seed(2)
    scale = 2
    psf = _scene_psf(0.3, "gaussian", None, 0.0)
    u = torch.randn(1, 1, 64, 96, dtype=DT, device=DEVICE, requires_grad=True)
    s = torch.empty(1, 12, 2, dtype=DT, device=DEVICE).uniform_(-2, 2)
    Au = forward_burst(u, s, psf, scale)            # (1,12,h,w)
    v = torch.randn_like(Au)
    ip_lr = (Au * v).sum()                          # <A u, v>
    (ATv,) = torch.autograd.grad(ip_lr, u)          # A^T v
    ip_hr = (u * ATv).sum()                         # <u, A^T v>
    rel = float(((ip_lr - ip_hr).abs() / ip_lr.abs().clamp_min(1e-30)).detach())
    # also exercise the DC-grad entry point (Huber + highpass) to ensure it builds a graph
    g, r = data_consistency_grad(
        u.detach().requires_grad_(True), v.detach() * 0 + Au.detach(), s, psf, scale,
        band_highpass_sigma_lr_px=0.0, huber_delta=None, create_graph=False,
    )
    assert g.shape == u.shape and torch.isfinite(g).all()
    return rel


def check_fast_equivalence() -> float:
    """ACL-033: the vectorized (grouped-conv) forward == the certified per-sample loop, fp64.

    Mixed batch of all PSF shapes (incl. zero-sigma identity) so both the separable group and the
    2D-kernel group are exercised, plus shifts that fall partly out of bounds (validity mask)."""
    torch.manual_seed(3)
    scale = 2
    sigma = torch.tensor([0.7, 1.3, 0.9, 1.1, 0.0, 1.6], dtype=DT, device=DEVICE)
    psf = ScenePSF(
        sigma_lr_px=sigma,
        shape=["gaussian", "gaussian", "elliptical_gaussian", "airy_disk", "gaussian", "elliptical_gaussian"],
        sigma_y_lr_px=[None, None, 1.4, 0.8, None, 2.0],
        angle_deg=torch.tensor([0.0, 0.0, 20.0, -35.0, 0.0, 60.0], dtype=DT, device=DEVICE),
    )
    x = torch.randn(6, 1, 48, 64, dtype=DT, device=DEVICE, requires_grad=True)
    shifts = (torch.rand(6, 9, 2, dtype=DT, device=DEVICE) - 0.5) * 3.0
    y_loop = forward_burst(x, shifts, psf, scale, fast=False)
    y_fast = forward_burst(x, shifts, psf, scale, fast=True)
    fwd = float((y_loop - y_fast).detach().abs().max())
    r = torch.randn_like(y_loop)
    (g_loop,) = torch.autograd.grad((y_loop * r).sum(), x, retain_graph=True)
    (g_fast,) = torch.autograd.grad((y_fast * r).sum(), x)
    adj = float((g_loop - g_fast).abs().max())
    return max(fwd, adj)


def test_fast_forward_equals_certified_loop() -> None:
    assert check_fast_equivalence() < 1e-9


def test_data_consistency_grad_works_inside_no_grad() -> None:
    """Solver eval runs under no_grad, but its DC step still needs local A^T gradients."""
    scale = 2
    psf = _scene_psf(0.3, "gaussian", None, 0.0)
    x = torch.randn(1, 1, 32, 48, dtype=DT, device=DEVICE)
    shifts = torch.empty(1, 4, 2, dtype=DT, device=DEVICE).uniform_(-1, 1)
    y = forward_burst(x, shifts, psf, scale).detach()

    with torch.no_grad():
        g, r = data_consistency_grad(
            x,
            y,
            shifts,
            psf,
            scale,
            band_highpass_sigma_lr_px=0.0,
            huber_delta=None,
            create_graph=False,
        )

    assert g.shape == x.shape
    assert r.shape == y.shape
    assert torch.isfinite(g).all()
    assert torch.isfinite(r).all()


def main() -> int:
    print(f"Gate A on {DEVICE} (dtype={DT}) — certify torch forward operator A")
    fwd = check_forward_parity()
    lin = check_linearity()
    adj = check_adjoint_identity()
    fast = check_fast_equivalence()
    print(f"\n[1] forward parity (gauss/ellip/airy)   worst max-abs = {fwd:.2e}   (want < 1e-5)")
    print(f"[2] linearity            A(au+bv)=aAu+bAv  max-abs = {lin:.2e}   (want < 1e-9)")
    print(f"[3] adjoint identity     <Au,v>=<u,A^T v>  rel-err = {adj:.2e}   (want < 1e-9)")
    print(f"[4] fast==loop (fwd+A^T) vectorized parity  max-abs = {fast:.2e}   (want < 1e-9)")
    ok = (fwd < 1e-5) and (lin < 1e-9) and (adj < 1e-9) and (fast < 1e-9)
    print("\nGATE A:", "PASS — proceed to Gate B" if ok else "FAIL — STOP, fix the operator")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
