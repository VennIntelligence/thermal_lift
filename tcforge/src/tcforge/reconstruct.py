"""Reconstruct HR temperature targets from compact scene parameters."""

from __future__ import annotations

import numpy as np

from .physics import render_temperature_field


def reconstruct_hr_temperature(
    hr_mask: np.ndarray,
    *,
    T_bg_c: float = 21.0,
    delta_T_c: float = 2.0,
    low_freq_amplitude_c: float = 0.2,
    low_freq_sigma_px: float = 96.0,
    seed: int | None = None,
) -> np.ndarray:
    """Rebuild the HR Celsius temperature field from a binary HR mask."""

    return render_temperature_field(
        hr_mask,
        t_bg_c=T_bg_c,
        delta_t_c=delta_T_c,
        low_freq_amplitude_c=low_freq_amplitude_c,
        low_freq_sigma_px=low_freq_sigma_px,
        seed=seed,
    )
