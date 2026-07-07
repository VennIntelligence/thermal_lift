"""Generate golden fixtures for the v7 noise/PSF byte-identity contract (§5.1).

MUST be run against the UNMODIFIED HEAD tcforge (before any v7 code change): it pins the
exact legacy output + RNG stream so the post-change default path can be proven bit-identical.

Writes three npz fixtures into tcforge/tests/data/:
  - field_noise_golden_v1.npz    (field_noise_burst: default + explicit-legacy kwargs)
  - make_noise_golden_v1.npz     (make_noise mixed / detector_realistic, mix_weights=None)
  - psf_params_golden_v1.npz     (sample_psf_parameters default ratio ranges, 50-draw sequence)

Construction constants are mirrored verbatim in the corresponding tests.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tcforge import physics, realism

DATA = Path(__file__).resolve().parents[1] / "tcforge" / "tests" / "data"
# When run from the worktree the scratchpad is outside the repo; resolve the repo data dir directly.
REPO_DATA = Path("/Users/ujs/mycode/thermal_lift/.claude/worktrees/agent-a4b1574848700fc6d/tcforge/tests/data")
OUT = REPO_DATA if REPO_DATA.exists() else DATA


def _burst() -> np.ndarray:
    # Non-flat burst so a vignette/stripe/grain pattern is genuinely exercised across a gradient.
    m, h, w = 8, 64, 80
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 20.0 + 0.01 * yy + 0.005 * xx
    return np.broadcast_to(base, (m, h, w)).astype(np.float32).copy()


def make_field_noise_golden() -> None:
    # Case 1: fully default kwargs.
    rng = np.random.default_rng(20260708)
    default_out = realism.field_noise_burst(_burst(), rng)
    default_sentinel = float(rng.random())

    # Case 2: explicit legacy kwargs (all current defaults, spelled out; none of the v7 knobs).
    rng = np.random.default_rng(77)
    legacy_out = realism.field_noise_burst(
        _burst(), rng,
        vignette_c=0.13, stripe_c=0.028, stripe_col_sigma=(2.5, 5.0), grain_c=0.10)
    legacy_sentinel = float(rng.random())

    np.savez(OUT / "field_noise_golden_v1.npz",
             default_out=default_out, default_sentinel=np.float64(default_sentinel),
             legacy_out=legacy_out, legacy_sentinel=np.float64(legacy_sentinel))
    print("field_noise_golden_v1.npz",
          default_out.shape, default_out.dtype, "sent", default_sentinel)


def make_make_noise_golden() -> None:
    shape = (3, 48, 60)
    mixed_out = physics.make_noise(shape, noise_sigma_c=0.08, seed=123,
                                   noise_model="mixed", fpn_sigma_px=5.0, stripe_sigma_c=None)
    detector_out = physics.make_noise(shape, noise_sigma_c=0.08, seed=456,
                                      noise_model="detector_realistic", fpn_sigma_px=5.0,
                                      stripe_sigma_c=0.03)
    np.savez(OUT / "make_noise_golden_v1.npz",
             mixed_out=mixed_out, detector_out=detector_out)
    print("make_noise_golden_v1.npz", mixed_out.shape, mixed_out.dtype)


def make_psf_params_golden() -> None:
    rng = np.random.default_rng(2024)
    seq = [physics.sample_psf_parameters(rng=rng) for _ in range(50)]
    params_json = json.dumps(seq, sort_keys=True)
    np.savez(OUT / "psf_params_golden_v1.npz",
             params_json=np.str_(params_json))
    print("psf_params_golden_v1.npz", len(seq), "draws")


if __name__ == "__main__":
    make_field_noise_golden()
    make_make_noise_golden()
    make_psf_params_golden()
    print("wrote goldens to", OUT)
