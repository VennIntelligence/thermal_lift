#!/usr/bin/env python3
"""Validate EP06 SR outputs and build 4x ROI cache figure."""

from __future__ import annotations

import argparse
import time

from thermal_core.ep06_cache import build_ep06_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild 4x ROI figure even if it already exists.",
    )
    parser.add_argument(
        "--skip-4x",
        action="store_true",
        help="Only validate 2x outputs and write manifest (skip 4x rebuild).",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    cache = build_ep06_cache(force=args.force, skip_4x=args.skip_4x)
    elapsed = time.perf_counter() - t0
    print(f"✅ EP06 cache ready: {cache.output_dir}")
    print(f"   4x figure built: {cache.manifest.get('four_x_built', False)}")
    print(f"   elapsed: {elapsed:.1f}s")
    print("   rebuild: uv run python scripts/build_ep06_cache.py")


if __name__ == "__main__":
    main()
