from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np


POOL = Path("data/synthetic/pool_2x_v3_5k")
EXPECTED_SCENES = 5000
LR_SHAPE = (480, 640)
HR_SHAPE = (960, 1280)
REQUIRED_FILES = {
    "metadata.json",
    "lr_burst.npy",
    "shifts.npy",
    "obs_features_1x.npz",
    "phase_bin_drizzle_2x.npy",
    "hr_mask_4x.png",
    "hr_edge_4x.png",
}


def fail(errors: list[str], scene: str, message: str) -> None:
    errors.append(f"{scene}: {message}")


def finite_array(errors: list[str], scene: str, name: str, arr: np.ndarray) -> None:
    if arr.size == 0:
        fail(errors, scene, f"{name} is empty")
        return
    if not np.isfinite(arr).all():
        fail(errors, scene, f"{name} contains NaN/Inf")


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    if not POOL.exists():
        print(f"Missing pool directory: {POOL}", file=sys.stderr)
        return 2

    scenes = sorted(p for p in POOL.iterdir() if p.is_dir() and p.name.startswith("scene_"))
    if len(scenes) != EXPECTED_SCENES:
        errors.append(f"expected {EXPECTED_SCENES} scene dirs, found {len(scenes)}")

    expected_names = [f"scene_{i:04d}" for i in range(EXPECTED_SCENES)]
    actual_names = [p.name for p in scenes]
    missing_names = sorted(set(expected_names) - set(actual_names))
    extra_names = sorted(set(actual_names) - set(expected_names))
    if missing_names:
        errors.append(f"missing scene dirs: {missing_names[:20]}{' ...' if len(missing_names) > 20 else ''}")
    if extra_names:
        errors.append(f"extra scene dirs: {extra_names[:20]}{' ...' if len(extra_names) > 20 else ''}")

    manifest = POOL / "manifest.csv"
    if not manifest.exists():
        errors.append("missing manifest.csv")
        manifest_ids: set[str] = set()
    else:
        with manifest.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        manifest_ids = {row.get("scene_id", "") for row in rows}
        if len(rows) != EXPECTED_SCENES:
            errors.append(f"manifest row count expected {EXPECTED_SCENES}, found {len(rows)}")
        if manifest_ids != set(expected_names):
            errors.append("manifest scene_id set does not exactly match expected scene IDs")

    for idx, scene_dir in enumerate(scenes, start=1):
        scene = scene_dir.name
        present = {p.name for p in scene_dir.iterdir() if p.is_file()}
        missing = REQUIRED_FILES - present
        if missing:
            fail(errors, scene, f"missing files: {sorted(missing)}")
            continue

        try:
            metadata = json.loads((scene_dir / "metadata.json").read_text(encoding="utf-8"))
        except Exception as exc:
            fail(errors, scene, f"metadata.json unreadable: {exc}")
            continue

        if metadata.get("scene_id") != scene:
            fail(errors, scene, f"metadata scene_id mismatch: {metadata.get('scene_id')!r}")
        if tuple(metadata.get("lr_shape", [])) != LR_SHAPE:
            fail(errors, scene, f"metadata lr_shape mismatch: {metadata.get('lr_shape')!r}")
        if tuple(metadata.get("hr_shape", [])) != HR_SHAPE:
            fail(errors, scene, f"metadata hr_shape mismatch: {metadata.get('hr_shape')!r}")

        n_frames = int(metadata.get("n_frames", -1))
        if n_frames <= 0:
            fail(errors, scene, f"invalid n_frames: {n_frames}")

        try:
            lr_burst = np.load(scene_dir / "lr_burst.npy", mmap_mode="r")
            if lr_burst.ndim != 3 or lr_burst.shape[1:] != LR_SHAPE:
                fail(errors, scene, f"lr_burst shape mismatch: {lr_burst.shape}")
            if n_frames > 0 and lr_burst.shape[0] != n_frames:
                fail(errors, scene, f"lr_burst frame count {lr_burst.shape[0]} != metadata n_frames {n_frames}")
            finite_array(errors, scene, "lr_burst", lr_burst)
        except Exception as exc:
            fail(errors, scene, f"lr_burst.npy unreadable: {exc}")

        try:
            shifts = np.load(scene_dir / "shifts.npy", mmap_mode="r")
            if shifts.ndim != 2 or shifts.shape[1] != 2:
                fail(errors, scene, f"shifts shape mismatch: {shifts.shape}")
            if n_frames > 0 and shifts.shape[0] != n_frames:
                fail(errors, scene, f"shifts frame count {shifts.shape[0]} != metadata n_frames {n_frames}")
            finite_array(errors, scene, "shifts", shifts)
        except Exception as exc:
            fail(errors, scene, f"shifts.npy unreadable: {exc}")

        try:
            phase = np.load(scene_dir / "phase_bin_drizzle_2x.npy", mmap_mode="r")
            if phase.shape != (4, *HR_SHAPE):
                fail(errors, scene, f"phase_bin_drizzle_2x shape mismatch: {phase.shape}")
            finite_array(errors, scene, "phase_bin_drizzle_2x", phase)
        except Exception as exc:
            fail(errors, scene, f"phase_bin_drizzle_2x.npy unreadable: {exc}")

        try:
            with np.load(scene_dir / "obs_features_1x.npz") as obs:
                if not obs.files:
                    fail(errors, scene, "obs_features_1x.npz has no arrays")
                for key in obs.files:
                    arr = obs[key]
                    if arr.ndim < 2:
                        fail(errors, scene, f"obs_features_1x[{key}] has unexpected ndim {arr.ndim}")
                    finite_array(errors, scene, f"obs_features_1x[{key}]", arr)
        except Exception as exc:
            fail(errors, scene, f"obs_features_1x.npz unreadable: {exc}")

        for png_name in ("hr_mask_4x.png", "hr_edge_4x.png"):
            image = cv2.imread(str(scene_dir / png_name), cv2.IMREAD_UNCHANGED)
            if image is None:
                fail(errors, scene, f"{png_name} unreadable")
            elif image.shape[:2] != HR_SHAPE:
                fail(errors, scene, f"{png_name} shape mismatch: {image.shape}")
            elif image.max() == image.min():
                warnings.append(f"{scene}: {png_name} is constant value {int(image.max())}")

        if idx % 250 == 0:
            print(f"checked {idx}/{len(scenes)} scenes")

    print(f"checked scenes: {len(scenes)}")
    print(f"errors: {len(errors)}")
    print(f"warnings: {len(warnings)}")

    if warnings:
        print("first warnings:")
        for warning in warnings[:20]:
            print(f"  WARN {warning}")

    if errors:
        print("first errors:")
        for error in errors[:50]:
            print(f"  ERROR {error}")
        return 1

    print("sanity check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
