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
