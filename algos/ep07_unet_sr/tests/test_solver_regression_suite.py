"""Unit tests for the Stage 0d solver regression suite."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "solver_regression_suite.py"
SPEC = importlib.util.spec_from_file_location("solver_regression_suite", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
suite = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = suite
SPEC.loader.exec_module(suite)


def _base_image(size: int = 128) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    base = 21.0 + 0.015 * np.sin(xx / 11.0) + 0.01 * np.cos(yy / 9.0)
    base[38:92, 42:88] += 1.2
    base[58:62, 30:100] += 0.4
    return base.astype(np.float32)


def _artifact_arrays() -> dict[str, np.ndarray]:
    base = _base_image()
    tiled = base.copy()
    tiled[:, ::32] += 0.25
    tiled[::32, :] -= 0.18
    tiled[86:118, 6:38] += 0.18
    for x in range(44, 88, 4):
        tiled[38:42, x : x + 2] += 0.45
    full = base + 0.22
    return {
        "aligned_mean": base,
        "tiled_p192_o128": tiled.astype(np.float32),
        "full_halo96": full.astype(np.float32),
    }


def test_empty_arrays_skip_each_case() -> None:
    thresholds = suite.SuiteThresholds()
    cases = [
        suite.flat_roi_probe({}, thresholds),
        suite.extent_consistency_probe({}, thresholds),
        suite.seam_spectrum_probe({}, thresholds, tile_step_hr=64),
        suite.beading_probe({}, thresholds),
    ]

    assert {case.status for case in cases} == {"skip"}
    assert all(case.skip_reason for case in cases)


def test_artifact_arrays_fail_regression_probes() -> None:
    thresholds = suite.SuiteThresholds(
        flat_lowpass_p95_delta_c=0.05,
        flat_lowpass_std_delta_c=0.02,
        extent_nrmse=0.10,
        extent_p95_abs_diff_c=0.08,
        seam_autocorr=0.20,
        bead_edge_ratio=2.0,
        bead_excess_p95_c=0.02,
    )
    arrays = _artifact_arrays()

    flat = suite.flat_roi_probe(arrays, thresholds, roi_size=32, lowpass_sigma=6.0)
    extent = suite.extent_consistency_probe(arrays, thresholds)
    seam = suite.seam_spectrum_probe(arrays, thresholds, tile_step_hr=32, highpass_sigma=4.0)
    bead = suite.beading_probe(arrays, thresholds)

    assert flat.status == "fail"
    assert extent.status == "fail"
    assert seam.status == "fail"
    assert bead.status == "fail"


def test_cli_writes_json_for_clean_arrays(tmp_path: Path) -> None:
    base = _base_image()
    arrays_path = tmp_path / "clean_arrays.npz"
    np.savez_compressed(
        arrays_path,
        aligned_mean=base,
        tiled_p192_o128=base + 0.001,
        full_halo96=base + 0.0015,
    )
    report_path = tmp_path / "report.json"

    code = suite.main(
        [
            "--arrays-npz",
            str(arrays_path),
            "--output-json",
            str(report_path),
            "--tile-step-hr",
            "32",
            "--flat-roi-size",
            "32",
        ]
    )

    assert code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["audit"]["detector_pitch_um_per_pixel"] == 20.0
    assert report["audit"]["spatial_resolution_um"] == 20.0
    assert "10um" in report["audit"]["historical_10um_policy"]
    assert report["suite_status"] in {"pass", "pass_with_skips"}
    assert report["summary"]["fail"] == 0
    assert set(report["inputs"]["labels"]) == {"aligned_mean", "tiled_p192_o128", "full_halo96"}


def test_explicit_missing_arrays_do_not_fall_back_to_local_defaults(tmp_path: Path) -> None:
    report_path = tmp_path / "missing_report.json"

    code = suite.main(
        [
            "--arrays-npz",
            str(tmp_path / "does_not_exist.npz"),
            "--output-json",
            str(report_path),
        ]
    )

    assert code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["suite_status"] == "skipped"
    assert report["summary"] == {"fail": 0, "pass": 0, "skip": 4}
    assert report["inputs"]["labels"] == []
    assert any("missing" in note for note in report["inputs"]["notes"])
