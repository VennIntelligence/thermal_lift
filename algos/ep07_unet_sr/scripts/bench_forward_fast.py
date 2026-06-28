#!/usr/bin/env python
"""Verify + benchmark the vectorized forward operator (ACL-033) against the certified loop.

The solver's physics forward A used a per-sample `for b in range(B)` Python loop (one small
PSF conv per scene, run K=4x per step + autograd A^T + double-backward) — serialized work that
starved the GPU. `forward_burst(..., fast=True)` replaces it with grouped convolutions that
process the whole batch's heterogeneous per-scene PSFs in one launch.

This script:
  1. CORRECTNESS — at fp64, asserts fast == loop for the forward AND the autograd adjoint A^T
     (the property Gate A certifies), on a mixed batch of all PSF shapes.
  2. SPEED — at fp32 on the training shapes, times loop vs fast for both the bare forward and the
     full data_consistency_grad WITH create_graph=True (what training actually pays), with CUDA
     warmup + synchronize. Reports per-call ms and speedup.

    cd algos/ep07_unet_sr && uv run python scripts/bench_forward_fast.py
    uv run python scripts/bench_forward_fast.py --batch 18 --frames 12 --hr 192 --iters 50
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from unet_sr.forward_torch import ScenePSF, data_consistency_grad, forward_burst  # noqa: E402


def _make_psf(B: int, device, dtype) -> ScenePSF:
    """A realistic mix: ~35% isotropic gaussian (separable path), ~65% elliptical/airy (2D path),
    matching the v3/v5 scene PSF distribution."""
    shapes, sigy, ang, sig = [], [], [], []
    for b in range(B):
        m = b % 3
        if m == 0:
            shapes.append("gaussian"); sigy.append(None); ang.append(0.0); sig.append(0.45 + 0.05 * (b % 4))
        elif m == 1:
            shapes.append("elliptical_gaussian"); sigy.append(0.6 + 0.05 * (b % 3))
            ang.append(15.0 * (b % 5)); sig.append(0.5)
        else:
            shapes.append("airy_disk"); sigy.append(None); ang.append(0.0); sig.append(0.4 + 0.03 * (b % 4))
    return ScenePSF(
        sigma_lr_px=torch.tensor(sig, dtype=dtype, device=device),
        shape=shapes,
        sigma_y_lr_px=sigy,
        angle_deg=torch.tensor(ang, dtype=dtype, device=device),
    )


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time(fn, iters: int, device) -> float:
    for _ in range(3):           # warmup (cudnn autotune / compile)
        fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    return (time.perf_counter() - t0) * 1000.0 / iters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=18)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--hr", type=int, default=192)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, N, H, scale = args.batch, args.frames, args.hr, args.scale

    # 1) Correctness at fp64
    dt64 = torch.float64
    psf64 = _make_psf(B, device, dt64)
    x64 = torch.randn(B, 1, 64, 64, dtype=dt64, device=device, requires_grad=True)
    sh64 = (torch.rand(B, N, 2, dtype=dt64, device=device) - 0.5) * 3.0
    yl = forward_burst(x64, sh64, psf64, scale, fast=False)
    yf = forward_burst(x64, sh64, psf64, scale, fast=True)
    fwd_err = (yl - yf).abs().max().item()
    r = torch.randn_like(yl)
    (gl,) = torch.autograd.grad((yl * r).sum(), x64, retain_graph=True)
    (gf,) = torch.autograd.grad((yf * r).sum(), x64)
    adj_err = (gl - gf).abs().max().item()
    print(f"device={device}  batch={B} frames={N} hr={H} scale={scale}")
    print(f"[correctness fp64] forward max|diff|={fwd_err:.2e}   adjoint A^T max|diff|={adj_err:.2e}"
          f"   -> {'OK' if max(fwd_err, adj_err) < 1e-9 else 'FAIL'}")

    # 2) Speed at fp32 on training shapes
    dt = torch.float32
    psf = _make_psf(B, device, dt)
    x = torch.randn(B, 1, H, H, dtype=dt, device=device, requires_grad=True)
    sh = (torch.rand(B, N, 2, dtype=dt, device=device) - 0.5) * 3.0
    y = forward_burst(x, sh, psf, scale, fast=True).detach()

    fwd_loop = _time(lambda: forward_burst(x, sh, psf, scale, fast=False), args.iters, device)
    fwd_fast = _time(lambda: forward_burst(x, sh, psf, scale, fast=True), args.iters, device)

    def dc(fast):
        xt = x.detach().requires_grad_(True)
        g, _ = data_consistency_grad(xt, y, sh, psf, scale, band_highpass_sigma_lr_px=5.0,
                                     create_graph=True)
        # mimic training: backprop the DC grad's norm so double-backward through A/A^T runs
        (g ** 2).sum().backward()

    import unet_sr.forward_torch as FT
    FT._FAST_FORWARD_DEFAULT = False
    dc_loop = _time(lambda: dc(False), max(10, args.iters // 3), device)
    FT._FAST_FORWARD_DEFAULT = True
    dc_fast = _time(lambda: dc(True), max(10, args.iters // 3), device)

    print(f"\n[forward only]            loop={fwd_loop:8.2f} ms   fast={fwd_fast:8.2f} ms"
          f"   speedup={fwd_loop / max(fwd_fast, 1e-9):5.2f}x")
    print(f"[DC grad + double-bwd]    loop={dc_loop:8.2f} ms   fast={dc_fast:8.2f} ms"
          f"   speedup={dc_loop / max(dc_fast, 1e-9):5.2f}x")
    print("\nNote: a training step runs the DC path K=4x; the fast path also raises GPU utilization "
          "(removes the per-sample launch bubbles), so wall-clock gain can exceed the per-call ratio.")
    return 0 if max(fwd_err, adj_err) < 1e-9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
