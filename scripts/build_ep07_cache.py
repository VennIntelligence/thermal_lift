#!/usr/bin/env python3
"""Build EP07 demo figures and table cache for notebook fragments."""

from __future__ import annotations

import argparse
import time

from thermal_core.ep07_cache import build_ep07_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if cache_manifest.json and all artifacts already exist.",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    cache = build_ep07_cache(force=args.force)
    elapsed = time.perf_counter() - t0
    print(f"✅ EP07 cache built: {cache.output_dir}")
    print(f"   demo_skipped: {cache.demo_skipped}")
    print(f"   figures: {0 if cache.demo_skipped else 9}")
    print(f"   elapsed: {elapsed:.1f}s")
    print("   rebuild: uv run python scripts/build_ep07_cache.py")


if __name__ == "__main__":
    main()
