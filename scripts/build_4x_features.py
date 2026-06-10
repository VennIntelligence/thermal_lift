#!/usr/bin/env python3
"""Build EP12 drizzle feature artifacts from compact TCForge scenes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

try:
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - exercised only in minimal tcforge envs
    def tqdm(iterable, **_: Any):
        return iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TCFORGE_SRC = PROJECT_ROOT / "tcforge" / "src"
if TCFORGE_SRC.exists() and str(TCFORGE_SRC) not in sys.path:
    sys.path.insert(0, str(TCFORGE_SRC))

from tcforge.classical_sr import drizzle_features  # noqa: E402


@dataclass(frozen=True)
class BuildOptions:
    scale: int = 4
    kernel: str = "bilinear"
    include_2x: bool = True
    precompute_1x_up4x: bool = True
    force: bool = False


def _scene_paths(pool_dir: str | Path) -> list[Path]:
    root = Path(pool_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"pool_dir not found: {root}")
    manifest = root / "manifest.csv"
    if manifest.exists():
        paths: list[Path] = []
        with manifest.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                value = row.get("scene_dir")
                if not value:
                    raise ValueError(f"{manifest} contains a row without scene_dir")
                path = Path(value).expanduser()
                paths.append(path if path.is_absolute() else root / path)
        return paths
    return sorted(path for path in root.iterdir() if path.is_dir() and (path / "metadata.json").exists())


def _load_metadata(scene_dir: Path) -> dict[str, Any]:
    with (scene_dir / "metadata.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_npz_features(path: Path) -> np.ndarray:
    with np.load(path) as data:
        if "obs_features" in data:
            features = data["obs_features"]
        elif "features" in data:
            features = data["features"]
        else:
            features = data[data.files[0]]
    return np.asarray(features, dtype=np.float32)


def _save_feature_npz(path: Path, features: np.ndarray, *, scale: int, source: str, kernel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        obs_features=np.asarray(features, dtype=np.float16),
        channel_names=np.asarray(["mean", "coverage", "variance"] if features.shape[0] == 3 else []),
        scale=np.asarray(scale, dtype=np.int16),
        source=np.asarray(source),
        kernel=np.asarray(kernel),
    )


def _upsample_features(features: np.ndarray, factor: int, output_shape: tuple[int, int]) -> np.ndarray:
    if factor == 1:
        out = np.asarray(features, dtype=np.float32)
    else:
        out = ndimage.zoom(np.asarray(features, dtype=np.float32), (1, factor, factor), order=1)
    rows, cols = map(int, output_shape)
    if out.shape[1:] != (rows, cols):
        out = out[:, :rows, :cols]
        if out.shape[1] < rows or out.shape[2] < cols:
            padded = np.zeros((out.shape[0], rows, cols), dtype=np.float32)
            padded[:, : out.shape[1], : out.shape[2]] = out
            out = padded
    return np.asarray(out, dtype=np.float32)


def build_scene_features(scene_dir: str | Path, options: BuildOptions = BuildOptions()) -> dict[str, Any]:
    """Build all requested EP12 feature artifacts for one scene."""

    root = Path(scene_dir)
    lr_path = root / "lr_burst.npy"
    shifts_path = root / "shifts.npy"
    if not lr_path.exists():
        return {"scene_dir": str(root), "status": "skipped", "reason": "missing lr_burst.npy"}
    if not shifts_path.exists():
        return {"scene_dir": str(root), "status": "skipped", "reason": "missing shifts.npy"}

    metadata = _load_metadata(root)
    scale = int(options.scale)
    lr_shape = tuple(map(int, metadata.get("lr_shape", [])))
    if len(lr_shape) != 2:
        lr_shape = tuple(map(int, np.load(lr_path, mmap_mode="r").shape[1:]))
    hr_shape = tuple(map(int, metadata.get("hr_shape", (lr_shape[0] * scale, lr_shape[1] * scale))))

    out_4x = root / "obs_features_4x.npz"
    out_2x = root / "obs_features_2x_up4x.npz"
    out_1x = root / "obs_features_1x_up4x.npz"
    requested = [out_4x]
    if options.include_2x:
        requested.append(out_2x)
    if options.precompute_1x_up4x:
        requested.append(out_1x)
    if not options.force and all(path.exists() for path in requested):
        return {"scene_dir": str(root), "status": "skipped", "reason": "outputs already exist"}

    # Load entirely into memory to avoid OS page faults during dense random-access scatter
    lr_burst = np.load(lr_path)
    shifts = np.load(shifts_path).astype(np.float32, copy=False)
    written: list[str] = []

    if options.force or not out_4x.exists():
        features_4x = drizzle_features(
            lr_burst,
            shifts,
            scale=scale,
            output_shape=hr_shape,
            kernel=options.kernel,
        )
        _save_feature_npz(out_4x, features_4x, scale=scale, source="lr_burst.npy", kernel=options.kernel)
        written.append(out_4x.name)

    if options.include_2x and (options.force or not out_2x.exists()):
        features_2x = drizzle_features(
            lr_burst,
            shifts,
            scale=2,
            output_shape=(lr_shape[0] * 2, lr_shape[1] * 2),
            kernel=options.kernel,
        )
        features_2x_up = _upsample_features(features_2x, scale // 2, hr_shape)
        _save_feature_npz(out_2x, features_2x_up, scale=scale, source="2x_drizzle_up4x", kernel=options.kernel)
        written.append(out_2x.name)

    if options.precompute_1x_up4x and (options.force or not out_1x.exists()):
        obs_1x_path = root / "obs_features_1x.npz"
        if obs_1x_path.exists():
            obs_1x = _load_npz_features(obs_1x_path)
            obs_1x_up = _upsample_features(obs_1x, scale, hr_shape)
            np.savez_compressed(
                out_1x,
                obs_features=np.asarray(obs_1x_up, dtype=np.float16),
                source=np.asarray("obs_features_1x.npz"),
                scale=np.asarray(scale, dtype=np.int16),
            )
            written.append(out_1x.name)

    return {"scene_dir": str(root), "status": "written" if written else "skipped", "written": written}


def build_pool_features(pool_dir: str | Path, *, workers: int, options: BuildOptions) -> list[dict[str, Any]]:
    scenes = _scene_paths(pool_dir)
    if not scenes:
        raise FileNotFoundError(f"no compact scene directories found under {pool_dir}")
    worker_count = max(1, int(workers))
    rows: list[dict[str, Any]] = []
    if worker_count == 1:
        for scene in tqdm(scenes, desc="Building EP12 features"):
            rows.append(build_scene_features(scene, options))
        return rows

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_scene = {executor.submit(build_scene_features, scene, options): scene for scene in scenes}
        for future in tqdm(as_completed(future_to_scene), total=len(future_to_scene), desc="Building EP12 features"):
            scene = future_to_scene[future]
            try:
                rows.append(future.result())
            except Exception:
                rows.append({"scene_dir": str(scene), "status": "failed", "traceback": traceback.format_exc()})
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", required=True, type=Path, help="Training pool with scene dirs containing lr_burst.npy.")
    parser.add_argument("--workers", type=int, default=1, help="Scene-level worker processes.")
    parser.add_argument("--scale", type=int, default=4, help="Target HR scale. EP12 defaults to 4.")
    parser.add_argument("--kernel", choices=["nearest", "bilinear"], default="bilinear", help="Scatter kernel.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing feature artifacts.")
    parser.add_argument("--no-2x", action="store_true", help="Do not write obs_features_2x_up4x.npz.")
    parser.add_argument("--no-1x-up4x", action="store_true", help="Do not write obs_features_1x_up4x.npz.")
    parser.add_argument("--summary-json", type=Path, default=None, help="Optional path for build summary JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    options = BuildOptions(
        scale=args.scale,
        kernel=args.kernel,
        include_2x=not args.no_2x,
        precompute_1x_up4x=not args.no_1x_up4x,
        force=args.force,
    )
    rows = build_pool_features(args.pool_dir, workers=args.workers, options=options)
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["status"])] = counts.get(str(row["status"]), 0) + 1
    print("Feature build summary:")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
