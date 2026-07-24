#!/usr/bin/env python3
"""Build EP01 CSV/PNG cache from raw TXT/BMP (run when data or EP01 logic changes).

用法: uv run python scripts/build_ep01_cache.py [--force]
输入: data/data_raw/infrared_avi/ 原始 TXT/BMP 帧（经 thermal_core.ep01_cache 重算）
输出: output/ep01_data_processing/（frame_audit.csv 等 CSV/PNG + cache_manifest.json）
关联: EP01
"""

from __future__ import annotations

import argparse
import time

from thermal_core.ep01_cache import build_ep01_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if cache_manifest.json and all artifacts already exist.",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    cache = build_ep01_cache(force=args.force)
    elapsed = time.perf_counter() - t0
    print(f"✅ EP01 cache built: {cache.output_dir}")
    print(f"   frames: {cache.manifest.get('n_frames', len(cache.df))}")
    print(f"   elapsed: {elapsed:.1f}s")
    print("   rebuild: uv run python scripts/build_ep01_cache.py")


if __name__ == "__main__":
    main()
