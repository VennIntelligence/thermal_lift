#!/usr/bin/env python3
"""Run P0 structural smoke checks for synthetic ThermalChipPhantom outputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
TCFORGE_SRC = PROJECT_ROOT / "tcforge" / "src"
if TCFORGE_SRC.exists() and str(TCFORGE_SRC) not in sys.path:
    sys.path.insert(0, str(TCFORGE_SRC))

from generate_thermal_chip_phantom import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SHIFT_CONFIG,
    REQUIRED_SCENE_FILES,
    _fallback_forward,
    _load_json,
    generate_dataset,
)


def _read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_metadata(scene_dir: Path) -> dict[str, Any]:
    with (scene_dir / "metadata.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _assert(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def _sample_frame_indices(n_frames: int, limit: int | None) -> np.ndarray:
    n = int(n_frames)
    if n <= 0:
        return np.asarray([], dtype=int)
    if limit is None or int(limit) <= 0 or int(limit) >= n:
        return np.arange(n, dtype=int)
    return np.unique(np.linspace(0, n - 1, int(limit), dtype=int))


def _reference_highpass(frames: np.ndarray, *, sigma_bg: float, mode: str = "nearest") -> np.ndarray:
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim == 2:
        bg = ndimage.gaussian_filter(arr, sigma=float(sigma_bg), mode=mode)
    elif arr.ndim == 3:
        bg = ndimage.gaussian_filter(arr, sigma=(0.0, float(sigma_bg), float(sigma_bg)), mode=mode)
    else:
        raise ValueError("frames must be 2D or 3D")
    return (arr - bg).astype(np.float32, copy=False)


def _check_point_source_forward(failures: list[str]) -> None:
    hr = np.zeros((8, 8), dtype=np.float32)
    hr[2, 3] = 1.0
    try:
        from tcforge._ep06_reference.forward import forward  # type: ignore

        lr = forward(hr, shift=(0.5, 0.0), psf_sigma=0.0, scale=2)
    except Exception:
        lr = _fallback_forward(hr, np.asarray([0.5, 0.0]), 0.0, scale=2)
    _assert(lr.shape == (4, 4), "point-source forward produced wrong LR shape", failures)
    _assert(float(lr[1, 1]) == 1.0, "point-source sign convention failed: expected LR(1,1)=1", failures)
    _assert(int(np.count_nonzero(lr)) == 1, "point-source forward should produce exactly one nonzero sample", failures)


def _check_scene(
    scene_dir: Path,
    failures: list[str],
    warnings: list[str],
    *,
    min_frames: int,
    highpass_check_frames: int | None,
) -> None:
    for name in REQUIRED_SCENE_FILES:
        _assert((scene_dir / name).exists(), f"{scene_dir}: missing {name}", failures)
    if any(not (scene_dir / name).exists() for name in REQUIRED_SCENE_FILES):
        return

    metadata = _load_metadata(scene_dir)
    scale = int(metadata["scale"])
    lr_shape = tuple(map(int, metadata["lr_shape"]))
    hr_shape = tuple(map(int, metadata["hr_shape"]))
    expected_hr = (lr_shape[0] * scale, lr_shape[1] * scale)
    _assert(hr_shape == expected_hr, f"{scene_dir}: metadata hr_shape is inconsistent with lr_shape*scale", failures)

    hr_temperature = np.load(scene_dir / "hr_temperature_2x.npy")
    mask = np.load(scene_dir / "hr_mask_2x.npy")
    edge_map = np.load(scene_dir / "hr_edge_map_2x.npy")
    raw = np.load(scene_dir / "lr_burst_raw.npy")
    hp = np.load(scene_dir / "lr_burst_highpass.npy")
    shifts = np.load(scene_dir / "shifts.npy")

    arrays = {
        "hr_temperature_2x.npy": hr_temperature,
        "hr_mask_2x.npy": mask,
        "hr_edge_map_2x.npy": edge_map,
        "lr_burst_raw.npy": raw,
        "lr_burst_highpass.npy": hp,
        "shifts.npy": shifts,
    }
    for name, arr in arrays.items():
        _assert(np.isfinite(arr).all(), f"{scene_dir}: {name} contains NaN/Inf", failures)

    _assert(hr_temperature.shape == hr_shape, f"{scene_dir}: hr_temperature shape mismatch", failures)
    _assert(mask.shape == hr_shape, f"{scene_dir}: mask shape mismatch", failures)
    _assert(edge_map.shape == hr_shape, f"{scene_dir}: edge map shape mismatch", failures)
    _assert(raw.ndim == 3 and raw.shape[1:] == lr_shape, f"{scene_dir}: raw burst shape mismatch", failures)
    _assert(hp.shape == raw.shape, f"{scene_dir}: highpass burst shape mismatch", failures)
    _assert(shifts.shape == (raw.shape[0], 2), f"{scene_dir}: shifts shape mismatch", failures)
    _assert(raw.shape[0] >= int(min_frames), f"{scene_dir}: synthetic LR burst must contain at least {min_frames} frames", failures)

    _assert(mask.dtype == np.uint8, f"{scene_dir}: mask dtype must be uint8", failures)
    mask_values = set(np.unique(mask).tolist())
    _assert(mask_values <= {0, 1}, f"{scene_dir}: mask values must be 0/1, got {sorted(mask_values)}", failures)
    _assert(np.asarray(hp).dtype == np.float32, f"{scene_dir}: highpass dtype should be float32", failures)

    physics = metadata["physics"]
    shift_meta = metadata.get("shifts", {})
    _assert(
        shift_meta.get("convention") == "LR-to-reference alignment shift",
        f"{scene_dir}: shift convention must be recorded as LR-to-reference alignment shift",
        failures,
    )
    t_bg = float(physics["T_bg_c"])
    delta_t = float(physics["delta_T_c"])
    low_amp = float(physics["low_freq_background_c"])
    noise = float(physics["noise_sigma_c"])
    lower = t_bg - low_amp - 3.5 * noise - 1e-4
    upper = t_bg + delta_t + low_amp + 3.5 * noise + 1e-4
    _assert(float(np.min(hr_temperature)) >= lower, f"{scene_dir}: HR temperature minimum outside configured range", failures)
    _assert(float(np.max(hr_temperature)) <= upper, f"{scene_dir}: HR temperature maximum outside configured range", failures)

    hp_indices = _sample_frame_indices(len(raw), highpass_check_frames)
    recomputed_hp = _reference_highpass(
        raw[hp_indices],
        sigma_bg=float(physics["highpass_sigma_lr_px"]),
        mode=str(physics.get("highpass_mode", "nearest")),
    )
    _assert(recomputed_hp.dtype == np.float32, f"{scene_dir}: reference highpass does not return float32", failures)
    _assert(
        np.allclose(hp[hp_indices], recomputed_hp, rtol=1e-5, atol=1e-5),
        f"{scene_dir}: stored highpass does not match independent reference highpass on {len(hp_indices)} sampled frames",
        failures,
    )

    margin = int(np.ceil(float(physics["highpass_sigma_lr_px"]) * 3.0 + float(np.max(np.abs(shifts))) + 2.0))
    if raw.shape[1] > 2 * margin + 8 and raw.shape[2] > 2 * margin + 8:
        hp_check = hp[:, margin:-margin, margin:-margin]
        hp_abs_max = float(np.max(np.abs(hp_check)))
        if metadata["difficulty"] == "easy":
            limit = delta_t + 3.0 * noise + 1e-4
            _assert(hp_abs_max <= limit, f"{scene_dir}: easy-scene interior highpass abs max {hp_abs_max:.3f} exceeds {limit:.3f} C", failures)
    else:
        hp_abs_max = float(np.max(np.abs(hp)))
        warnings.append(
            f"{scene_dir}: frame is too small for interior highpass range gate; full-frame abs max is {hp_abs_max:.3f} C"
        )

    norms = np.linalg.norm(shifts, axis=1)
    _assert(np.isfinite(norms).all(), f"{scene_dir}: shift norms contain NaN/Inf", failures)
    if np.max(norms) > 20.0:
        warnings.append(f"{scene_dir}: shift norm exceeds 20 LR px ({float(np.max(norms)):.3f})")


def run_smoke(
    output_root: Path,
    *,
    min_frames: int = 32,
    highpass_check_frames: int | None = 16,
) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    rows = _read_manifest(output_root / "manifest.csv")
    _assert(len(rows) > 0, "manifest.csv has no rows", failures)
    for row in rows:
        scene_dir = Path(row["scene_dir"])
        _assert(scene_dir.exists(), f"manifest scene_dir does not exist: {scene_dir}", failures)
        if scene_dir.exists():
            _check_scene(
                scene_dir,
                failures,
                warnings,
                min_frames=min_frames,
                highpass_check_frames=highpass_check_frames,
            )
    _check_point_source_forward(failures)
    return failures, warnings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--generate", action="store_true", help="generate the configured dataset before checking")
    parser.add_argument("--quick-demo", action="store_true", help="with --generate, override to one small scene for fast local checks")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--shift-config", type=Path, default=DEFAULT_SHIFT_CONFIG)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--n-frames", type=int, default=None)
    parser.add_argument("--lr-shape", default=None, help="optional generation shape override as ROWS,COLS")
    parser.add_argument(
        "--min-frames",
        type=int,
        default=None,
        help="minimum accepted burst length; defaults to 32, or 1 for --generate demo runs",
    )
    parser.add_argument(
        "--highpass-check-frames",
        type=int,
        default=16,
        help="number of evenly spaced frames to recompute with an independent highpass reference; <=0 checks all frames",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.generate:
        config = _load_json(args.config)
        config["_config_path"] = str(args.config)
        if args.quick_demo:
            config["allow_fallback_demo"] = True
            config["num_scenes"] = 1 if args.max_scenes is None else int(args.max_scenes)
            config["n_frames_per_scene"] = 8 if args.n_frames is None else int(args.n_frames)
            shape_text = "32,40" if args.lr_shape is None else args.lr_shape
            rows, cols = (int(v) for v in shape_text.lower().replace("x", ",").split(","))
            config["lr_shape"] = [rows, cols]
            config["hr_shape"] = [rows * int(config["scale"]), cols * int(config["scale"])]
        else:
            if args.max_scenes is not None:
                config["num_scenes"] = int(args.max_scenes)
            if args.n_frames is not None:
                config["n_frames_per_scene"] = int(args.n_frames)
            if args.lr_shape is not None:
                rows, cols = (int(v) for v in args.lr_shape.lower().replace("x", ",").split(","))
                config["lr_shape"] = [rows, cols]
                config["hr_shape"] = [rows * int(config["scale"]), cols * int(config["scale"])]
        if "difficulties" in config:
            config["difficulties"] = config["difficulties"][: int(config["num_scenes"])]
        if "seeds" in config:
            config["seeds"] = config["seeds"][: int(config["num_scenes"])]
        generate_dataset(config, args.output_root, args.shift_config)

    min_frames = int(args.min_frames) if args.min_frames is not None else (1 if args.quick_demo else 32)
    hp_limit = None if int(args.highpass_check_frames) <= 0 else int(args.highpass_check_frames)
    failures, warnings = run_smoke(args.output_root, min_frames=min_frames, highpass_check_frames=hp_limit)
    report = {"status": "fail" if failures else "pass", "failures": failures, "warnings": warnings}
    report_path = args.output_root / "smoke_test_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")

    for warning in warnings:
        print(f"WARNING: {warning}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        print(f"wrote smoke report: {report_path}", file=sys.stderr)
        return 1
    print(f"smoke passed: {args.output_root}")
    print(f"wrote smoke report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
