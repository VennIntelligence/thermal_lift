#!/usr/bin/env python3
"""Build EP03 CSV/PNG cache from configs, EP01 audit, and reference frame.

用法: uv run python scripts/build_ep03_cache.py [--force]
输入: configs/ 物理参数、output/ep01_data_processing/frame_audit.csv、参考帧 TXT
输出: output/ep03_theoretical_limits/（CSV/PNG + cache_manifest.json）
关联: EP03
"""

from __future__ import annotations

import argparse
import time

from thermal_core.ep03_cache import build_ep03_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if cache_manifest.json and all artifacts already exist.",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    cache = build_ep03_cache(force=args.force)
    elapsed = time.perf_counter() - t0
    print(f"✅ EP03 cache built: {cache.output_dir}")
    print(
        f"   pitch={cache.detector_pitch_um:.1f} um/px, "
        f"resolution={cache.spatial_resolution_um:.1f} um, "
        f"noise={cache.noise_sigma_c:.4f} C"
    )
    print(f"   reference: {cache.reference_file} (order={cache.reference_order})")
    print(f"   elapsed: {elapsed:.1f}s")
    print("   rebuild: uv run python scripts/build_ep03_cache.py")


if __name__ == "__main__":
    main()
