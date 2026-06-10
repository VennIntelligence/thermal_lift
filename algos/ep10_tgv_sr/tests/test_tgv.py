from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
EP06_SRC = Path(__file__).resolve().parents[2] / "ep06_sr_poc" / "src"
for path in (SRC, EP06_SRC):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from ep10_tgv_sr import reconstruct_map_tgv, tgv_denoise
from ep10_tgv_sr import tgv as tgv_module
from ep10_tgv_sr.tgv import _data_gradient_and_loss_cached
from ibp import forward
from map_tv.map_tv import _data_gradient_and_loss, tv_denoise_chambolle


def _scene(shape: tuple[int, int] = (48, 56)) -> np.ndarray:
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]]
    img = 0.18 * np.cos(2 * np.pi * y / shape[0])
    img += 0.35 * (x / max(1, shape[1] - 1))
    img += 0.45 * ((x > 28) & (y > 20))
    img -= img.min()
    img /= img.max()
    return img


def _shifts(n_frames: int) -> np.ndarray:
    phases = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
    return phases[np.arange(n_frames) % 4]


def _second_diff(img: np.ndarray) -> float:
    return float(np.mean(np.abs(np.diff(img[8:40, 4:24], n=2, axis=1))))


def test_tgv_denoise_preserves_linear_ramp_better_than_tv() -> None:
    rng = np.random.default_rng(10)
    y, x = np.mgrid[0:64, 0:64]
    clean = 0.2 + 0.6 * x / 63.0 + 0.1 * y / 63.0
    noisy = clean + rng.normal(scale=0.06, size=clean.shape)

    tv = tv_denoise_chambolle(noisy, weight=0.06, max_iter=100)
    tgv = tgv_denoise(noisy, weight=0.06, max_iter=100)

    assert tgv.shape == noisy.shape
    assert np.isfinite(tgv).all()
    assert np.mean((tgv - clean) ** 2) < np.mean((noisy - clean) ** 2)
    assert _second_diff(tgv) < 0.5 * _second_diff(tv)


def test_tgv_backend_provenance_records_ccpi_success(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_tgv(
        image: np.ndarray,
        lambda_par: float,
        alpha1: float,
        alpha0: float,
        max_iter: int,
        lipshitz: float,
        tolerance: float,
        *,
        out: np.ndarray,
        device: str,
    ) -> np.ndarray:
        calls.append(device)
        out[...] = image
        return out

    monkeypatch.setattr(tgv_module, "_load_ccpi_tgv", lambda: fake_tgv)
    monkeypatch.setattr(tgv_module, "_candidate_tgv_devices", lambda device: ["cpu"])

    image = np.ones((8, 8), dtype=np.float64)
    denoised = tgv_denoise(image, weight=0.01, max_iter=1, device="cpu")
    provenance = tgv_module.get_tgv_backend_provenance()

    assert np.allclose(denoised, image)
    assert calls == ["cpu"]
    assert provenance["backend"] == "ccpi"
    assert provenance["status"] == "success"
    assert provenance["requested_device"] == "cpu"
    assert provenance["selected_device"] == "cpu"
    assert provenance["candidate_devices"] == ["cpu"]
    assert provenance["error"] is None


def test_tgv_backend_provenance_records_import_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tgv_module, "_load_ccpi_tgv", lambda: None)
    monkeypatch.setattr(tgv_module, "_CCPI_IMPORT_ERROR", ImportError("missing ccpi"))

    image = np.ones((8, 8), dtype=np.float64)
    denoised = tgv_denoise(image, weight=0.01, max_iter=1, device="auto")
    provenance = tgv_module.get_tgv_backend_provenance()

    assert denoised.shape == image.shape
    assert np.isfinite(denoised).all()
    assert provenance["backend"] == "fallback"
    assert provenance["status"] == "ccpi_import_failed"
    assert provenance["requested_device"] == "auto"
    assert provenance["selected_device"] == "local_chambolle_pock"
    assert "missing ccpi" in str(provenance["error"])


def test_tgv_backend_provenance_records_runtime_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_tgv(
        image: np.ndarray,
        lambda_par: float,
        alpha1: float,
        alpha0: float,
        max_iter: int,
        lipshitz: float,
        tolerance: float,
        *,
        out: np.ndarray,
        device: str,
    ) -> np.ndarray:
        raise RuntimeError(f"{device} unavailable")

    monkeypatch.setattr(tgv_module, "_load_ccpi_tgv", lambda: failing_tgv)
    monkeypatch.setattr(tgv_module, "_candidate_tgv_devices", lambda device: ["gpu", "cpu"])
    monkeypatch.setattr(tgv_module, "_RUNTIME_FALLBACK_WARNED", True)

    image = np.ones((8, 8), dtype=np.float64)
    denoised = tgv_denoise(image, weight=0.01, max_iter=1, device="gpu")
    provenance = tgv_module.get_tgv_backend_provenance()

    assert denoised.shape == image.shape
    assert np.isfinite(denoised).all()
    assert provenance["backend"] == "fallback"
    assert provenance["status"] == "ccpi_runtime_failed"
    assert provenance["requested_device"] == "gpu"
    assert provenance["candidate_devices"] == ["gpu", "cpu"]
    assert provenance["selected_device"] == "local_chambolle_pock"
    assert "cpu unavailable" in str(provenance["error"])


def test_reconstruct_map_tgv_smoke() -> None:
    hr = _scene()
    shifts = _shifts(12)
    frames = np.stack([forward(hr, shift, psf_sigma=0.18) for shift in shifts])
    frames += np.random.default_rng(11).normal(scale=0.02, size=frames.shape)

    recon, records = reconstruct_map_tgv(
        frames,
        shifts,
        lambda_tv=0.002,
        alpha_ratio=2.0,
        max_iter=2,
        step_size=0.8,
        psf_sigma=0.18,
        workers=1,
        tol=0.0,
        tgv_inner_iter=20,
    )

    assert recon.shape == hr.shape
    assert np.isfinite(recon).all()
    assert len(records) == 2
    assert {"iteration", "objective_proxy", "relative_update"}.issubset(records[0])
    assert {
        "tgv_backend",
        "tgv_backend_status",
        "tgv_backend_device",
        "tgv_backend_error",
    }.issubset(records[0])



def test_cached_gradient_matches_ep06_gradient() -> None:
    hr = _scene((24, 28))
    shifts = _shifts(6)
    frames = np.stack([forward(hr, shift, psf_sigma=0.18) for shift in shifts])
    x0 = hr + np.random.default_rng(12).normal(scale=0.01, size=hr.shape)

    grad_ref, loss_ref = _data_gradient_and_loss(
        x0,
        frames,
        shifts,
        psf_sigma=0.18,
        scale=2,
        workers=1,
    )
    grad_cached, loss_cached = _data_gradient_and_loss_cached(
        x0,
        frames,
        shifts,
        psf_sigma=0.18,
        scale=2,
        workers=1,
    )

    assert np.allclose(loss_cached, loss_ref, rtol=1e-12, atol=1e-12)
    assert np.allclose(grad_cached, grad_ref, rtol=1e-10, atol=1e-10)
