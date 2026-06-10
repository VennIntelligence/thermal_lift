#!/usr/bin/env python3
"""Build EP02 CSV/PNG cache from raw frames and EP01 audit (run when data or EP02 logic changes)."""

from __future__ import annotations

import argparse
import time

from thermal_core.ep02_cache import build_ep02_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if cache_manifest.json and all artifacts already exist.",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    cache = build_ep02_cache(force=args.force)
    elapsed = time.perf_counter() - t0
    print(f"✅ EP02 cache built: {cache.output_dir}")
    print(f"   theta={cache.theta_deg:.1f} deg, pitch={cache.pixel_size_um:.1f} um/pixel")
    print(f"   frames: {cache.manifest.get('n_frames', len(cache.frame_audit))}")
    optional = cache.manifest.get("optional_artifacts", [])
    if optional:
        print(f"   optional AVI artifacts: {', '.join(optional)}")
    print(f"   elapsed: {elapsed:.1f}s")
    print("   rebuild: uv run python scripts/build_ep02_cache.py")


if __name__ == "__main__":
    main()
