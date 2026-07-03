from __future__ import annotations

import numpy as np

from unet_sr.real_eval import to_center_grid


def test_to_center_grid_shifts_content_by_minus_half_px() -> None:
    rng = np.random.default_rng(5)
    freq = np.fft.fftfreq(64)
    # Smooth band-limited image so the fractional shift is well defined.
    spec = np.fft.fft2(rng.normal(size=(64, 64)))
    keep = (np.abs(freq)[:, None] < 0.2) & (np.abs(freq)[None, :] < 0.2)
    img = np.real(np.fft.ifft2(spec * keep)).astype(np.float32)

    out = to_center_grid(img, scale=2)

    # Reference: exact Fourier translation of the content by (-0.5, -0.5) px.
    fy = freq[:, None]
    fx = freq[None, :]
    ref = np.real(np.fft.ifft2(np.fft.fft2(img.astype(np.float64)) * np.exp(2j * np.pi * (fy + fx) * 0.5)))
    np.testing.assert_allclose(out, ref.astype(np.float32), atol=1e-5)
    # And it must actually move content: correlation peak of out vs img sits off zero.
    assert float(np.abs(out - img).max()) > 1e-3


def test_to_center_grid_scale1_is_identity() -> None:
    img = np.random.default_rng(0).normal(size=(16, 16)).astype(np.float32)
    np.testing.assert_allclose(to_center_grid(img, scale=1), img, atol=0)
