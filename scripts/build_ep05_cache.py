#!/usr/bin/env python3
"""Validate EP05 output directories and write cache manifests for notebook display."""

from __future__ import annotations

import argparse
import time

from thermal_core.ep05_cache import build_ep05_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite cache_manifest.json even if artifacts already validated.",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    cache = build_ep05_cache(force=args.force)
    elapsed = time.perf_counter() - t0
    print(f"✅ EP05 cache validated: {cache.capacity_dir.parent}")
    print(f"   displacement: {cache.displacement_dir.name}")
    print(f"   capacity:     {cache.capacity_dir.name}")
    print(f"   contour:      {cache.contour_dir.name}")
    print(f"   overlay:      {cache.overlay_dir.name}")
    print(f"   main frames:  {cache.summary_json.get('n_main_frames_scored', 'unknown')}")
    print(f"   elapsed: {elapsed:.1f}s")
    print("   rebuild: uv run python scripts/build_ep05_cache.py")


if __name__ == "__main__":
    main()
