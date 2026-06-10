#!/usr/bin/env python3
"""Evaluate EP12 split-half drizzle consistency for a compact training pool."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from tqdm import tqdm

from sr4x.evaluate import split_half_consistency_scene


def _scene_paths(pool_dir: str | Path) -> list[Path]:
    root = Path(pool_dir).expanduser()
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


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {"scene_count": 0}
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and value == value
        }
    )
    out: dict[str, object] = {"scene_count": len(rows)}
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if values:
            out[f"{key}_mean"] = sum(values) / len(values)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", required=True, type=Path, help="Compact pool with scene dirs.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional CSV output path.")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--kernel", choices=["nearest", "bilinear"], default="bilinear")
    parser.add_argument("--min-even-coverage", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        split_half_consistency_scene(
            scene,
            scale=args.scale,
            kernel=args.kernel,
            min_even_coverage=args.min_even_coverage,
        )
        for scene in tqdm(_scene_paths(args.pool_dir), desc="Split-half")
    ]
    payload = {"scenes": rows, "aggregate": _aggregate(rows)}
    print(json.dumps(payload["aggregate"], indent=2, ensure_ascii=False))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.output_csv is not None and rows:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row})
        with args.output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
