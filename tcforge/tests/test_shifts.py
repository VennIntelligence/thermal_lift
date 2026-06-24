from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import tcforge.shifts as shifts_mod


def _write_shift_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "acquisition_order,success,refined_align_dx_px,refined_align_dy_px\n"
        "2,True,0.2,0.4\n"
        "1,True,0.1,0.3\n",
        encoding="utf-8",
    )


def test_ideal_phase_grid_uses_quarter_phase_default_and_records_convention() -> None:
    shifts, metadata = shifts_mod.load_shift_profile("ideal_phase_grid", n_frames=16, scale=2)

    assert shifts.shape == (16, 2)
    assert shifts.dtype == np.float32
    assert np.allclose(np.unique(shifts[:, 0]), [0.0, 0.25, 0.5, 0.75])
    assert np.allclose(np.unique(shifts[:, 1]), [0.0, 0.25, 0.5, 0.75])
    assert metadata["convention"] == "LR-to-reference alignment shift"
    assert metadata["phase_steps"] == 4


def test_real_profile_can_be_trimmed_for_small_demo_if_source_exists() -> None:
    project_root = Path(__file__).resolve().parents[2]
    source = shifts_mod.default_contour_alignment_csv(project_root=project_root)
    if not source.exists():
        pytest.skip("Git-tracked contour alignment CSV is not available")

    shifts, metadata = shifts_mod.load_shift_profile("real_default_contour_refined", n_frames=4)

    assert shifts.shape == (4, 2)
    assert np.isfinite(shifts).all()
    assert metadata["profile"] == "real_default_contour_refined"


def test_real_profile_uses_explicit_path_without_project_layout(tmp_path: Path) -> None:
    source = tmp_path / "contour_alignment_results.csv"
    _write_shift_csv(source)

    shifts = shifts_mod.load_real_default_contour_refined(source, n_frames=2)

    assert shifts.dtype == np.float32
    assert np.allclose(shifts, [[0.1, 0.3], [0.2, 0.4]])


def test_real_profile_default_can_come_from_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "contour_alignment_results.csv"
    _write_shift_csv(source)
    monkeypatch.setenv("TCFORGE_REAL_SHIFT_CSV", str(source))

    shifts, metadata = shifts_mod.load_shift_profile("real_default_contour_refined", n_frames=3)

    assert shifts.shape == (3, 2)
    assert np.allclose(shifts[:2], [[0.1, 0.3], [0.2, 0.4]])
    assert metadata["profile"] == "real_default_contour_refined"


def _phase_cells_used(shifts: np.ndarray, phase_steps: int) -> set[tuple[int, int]]:
    frac_dx = np.mod(shifts[:, 0], 1.0)
    frac_dy = np.mod(shifts[:, 1], 1.0)
    cx = np.clip((frac_dx * phase_steps).astype(int), 0, phase_steps - 1)
    cy = np.clip((frac_dy * phase_steps).astype(int), 0, phase_steps - 1)
    return set(zip(cx.tolist(), cy.tolist()))


def test_random_constellation_shape_dtype_and_reproducible() -> None:
    a = shifts_mod.random_constellation(40, phase_steps=4, coverage="good", seed=7)
    b = shifts_mod.random_constellation(40, phase_steps=4, coverage="good", seed=7)

    assert a.shape == (40, 2)
    assert a.dtype == np.float32
    assert np.array_equal(a, b)


def test_random_constellation_good_covers_all_phase_cells() -> None:
    phase_steps = 4
    shifts = shifts_mod.random_constellation(
        phase_steps * phase_steps * 3, phase_steps=phase_steps, coverage="good", seed=3
    )
    used = _phase_cells_used(shifts, phase_steps)
    assert len(used) == phase_steps * phase_steps


def test_random_constellation_poor_and_medium_cover_fewer_cells() -> None:
    phase_steps = 4
    n = phase_steps * phase_steps * 3
    good = _phase_cells_used(
        shifts_mod.random_constellation(n, phase_steps=phase_steps, coverage="good", seed=11),
        phase_steps,
    )
    medium = _phase_cells_used(
        shifts_mod.random_constellation(n, phase_steps=phase_steps, coverage="medium", seed=11),
        phase_steps,
    )
    poor = _phase_cells_used(
        shifts_mod.random_constellation(n, phase_steps=phase_steps, coverage="poor", seed=11),
        phase_steps,
    )
    assert len(good) == phase_steps * phase_steps
    assert len(medium) < len(good)
    assert len(poor) <= len(medium)


def test_random_constellation_jitter_respected() -> None:
    no_jitter = shifts_mod.random_constellation(50, phase_steps=4, coverage="good", jitter_std_px=0.0, seed=5)
    jitter = shifts_mod.random_constellation(50, phase_steps=4, coverage="good", jitter_std_px=0.05, seed=5)
    # Without jitter the integer-offset + phase values are exact grid fractions.
    frac = np.mod(no_jitter, 1.0)
    on_grid = np.isclose(np.mod(frac * 4.0, 1.0), 0.0, atol=1e-4)
    assert on_grid.all()
    # Jitter moves shifts off the exact grid.
    assert not np.allclose(no_jitter, jitter)


def test_build_scene_shifts_random_phase_reproducible() -> None:
    cfg = {
        "phase_steps_choices": [2, 3, 4],
        "coverage_quality_weights": {"good": 0.5, "medium": 0.3, "poor": 0.2},
        "jitter_std_px": [0.0, 0.08],
        "include_real_like_fraction": 0.0,
    }
    s1, m1 = shifts_mod.build_scene_shifts(12345, 32, cfg, scale=2)
    s2, m2 = shifts_mod.build_scene_shifts(12345, 32, cfg, scale=2)

    assert s1.shape == (32, 2)
    assert np.array_equal(s1, s2)
    assert m1["constellation_mode"] == "random_phase"
    assert m1 == m2
    assert m1["coverage"] in {"good", "medium", "poor"}
    assert m1["phase_steps"] in {2, 3, 4}


def test_build_scene_shifts_real_like_when_fraction_is_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "contour_alignment_results.csv"
    _write_shift_csv(source)
    monkeypatch.setenv("TCFORGE_REAL_SHIFT_CSV", str(source))
    cfg = {
        "phase_steps_choices": [4],
        "coverage_quality_weights": {"good": 1.0},
        "jitter_std_px": [0.0, 0.0],
        "include_real_like_fraction": 1.0,
    }
    shifts, meta = shifts_mod.build_scene_shifts(999, 6, cfg, scale=2)

    assert shifts.shape == (6, 2)
    assert meta["constellation_mode"] == "real_like"
