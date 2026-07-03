"""Stage 2a E1 per-frame fusion tests (design draft §5, tests ①-⑤)."""

from __future__ import annotations

import numpy as np
import torch

from unet_sr.forward_torch import ScenePSF
from unet_sr.fusion import PerFrameFusion, corner_grid_splat
from unet_sr.unroll import UnrolledSolver

SCALE = 2


def _burst(b: int = 1, m: int = 16, h: int = 24, w: int = 24, seed: int = 0):
    rng = np.random.default_rng(seed)
    burst = torch.from_numpy(rng.normal(size=(b, m, h, w)).astype(np.float32))
    shifts = torch.from_numpy(rng.uniform(-0.9, 0.9, size=(b, m, 2)).astype(np.float32))
    return burst, shifts


def test_fusion_permutation_invariant() -> None:  # ①
    torch.manual_seed(0)
    fusion = PerFrameFusion(feat_channels=8, frame_chunk=5).eval()
    burst, shifts = _burst(m=16)
    with torch.no_grad():
        out = fusion(burst, shifts, SCALE, (48, 48))
        perm = torch.randperm(16)
        out_perm = fusion(burst[:, perm], shifts[:, perm], SCALE, (48, 48))
    torch.testing.assert_close(out, out_perm, atol=1e-5, rtol=0)


def test_fusion_m_invariant_under_duplication() -> None:  # ②
    torch.manual_seed(0)
    fusion = PerFrameFusion(feat_channels=8, frame_chunk=8).eval()
    burst, shifts = _burst(m=16)
    with torch.no_grad():
        out16 = fusion(burst, shifts, SCALE, (48, 48))
        # Constant-M top-up duplicates frames (dataset._select_m_indices); every
        # pooled statistic must be unchanged.
        out32 = fusion(
            torch.cat([burst, burst], dim=1), torch.cat([shifts, shifts], dim=1), SCALE, (48, 48)
        )
    torch.testing.assert_close(out16, out32, atol=1e-5, rtol=0)


def test_corner_grid_splat_centroid_matches_dc_block_center() -> None:  # ③ (ACL-049)
    # An LR impulse at (iy, ix) with shift (dx, dy) must scatter its mass with
    # centroid at scale*(i+d) + 0.5 in each axis — the DC forward's corner-grid
    # block center. A half-pixel drift here would rebuild the ACL-049 bug
    # inside the network.
    iy, ix = 5, 7
    dx, dy = 0.30, -0.20
    h = w = 16
    y = torch.zeros(1, 1, h, w)
    y[0, 0, iy, ix] = 1.0
    shifts = torch.tensor([[[dx, dy]]], dtype=torch.float32)
    value, coverage = corner_grid_splat(y, shifts, SCALE, (h * SCALE, w * SCALE))
    mass = (value * coverage)[0, 0]  # undo the coverage normalization -> raw A^T y
    total = float(mass.sum())
    assert total > 0
    ys = torch.arange(h * SCALE, dtype=torch.float32)
    xs = torch.arange(w * SCALE, dtype=torch.float32)
    cy = float((mass.sum(dim=1) * ys).sum() / total)
    cx = float((mass.sum(dim=0) * xs).sum() / total)
    expect_cy = SCALE * (iy + dy) + 0.5
    expect_cx = SCALE * (ix + dx) + 0.5
    assert abs(cy - expect_cy) < 1e-4, (cy, expect_cy)
    assert abs(cx - expect_cx) < 1e-4, (cx, expect_cx)


def test_fusion_masked_frames_do_not_contribute() -> None:  # ④
    torch.manual_seed(0)
    fusion = PerFrameFusion(feat_channels=8, frame_chunk=4).eval()
    burst, shifts = _burst(m=12)
    junk = burst.clone()
    junk[:, 8:] = 1e3  # poison the masked frames
    mask = torch.ones(1, 12, 1, 1)
    mask[:, 8:] = 0.0
    with torch.no_grad():
        out_clean = fusion(burst[:, :8], shifts[:, :8], SCALE, (48, 48))
        out_masked = fusion(junk, shifts, SCALE, (48, 48), frame_mask=mask)
    torch.testing.assert_close(out_clean, out_masked, atol=1e-5, rtol=0)


def test_solver_fusion_none_is_legacy_identical_and_perframe_runs() -> None:  # ⑤ + smoke
    def _build(**kw):
        torch.manual_seed(123)
        return UnrolledSolver(
            n_steps=2, cond_channels=5, base_channels=8, scale=SCALE,
            prox_use_se=False, prox_norm="none",
            band_highpass_sigma_lr_px=1.0,  # tiny test frames; default 5.0 over-pads 16x16
            **kw,
        ).eval()

    legacy = _build()
    off = _build(fusion="none")
    assert not any(k.startswith("fusion") for k in off.state_dict())
    for (ka, va), (kb, vb) in zip(legacy.state_dict().items(), off.state_dict().items()):
        assert ka == kb
        torch.testing.assert_close(va, vb, atol=0, rtol=0)

    burst, shifts = _burst(m=6, h=16, w=16, seed=3)
    x0 = torch.zeros(1, 1, 32, 32)
    cond = torch.randn(1, 5, 32, 32)
    psf = ScenePSF(
        sigma_lr_px=torch.tensor([0.4]), shape=["gaussian"], sigma_y_lr_px=[None],
        angle_deg=torch.tensor([0.0]),
    )
    with torch.no_grad():
        out_legacy = legacy(x0, burst, shifts, psf, cond)
        out_off = off(x0, burst, shifts, psf, cond)
    torch.testing.assert_close(out_legacy, out_off, atol=0, rtol=0)

    on = _build(fusion="perframe", fusion_channels=4, fusion_frame_chunk=4)
    assert any(k.startswith("fusion") for k in on.state_dict())
    with torch.no_grad():
        out_on = on(x0, burst, shifts, psf, cond)
    assert out_on.shape == out_off.shape
    assert torch.isfinite(out_on).all()
