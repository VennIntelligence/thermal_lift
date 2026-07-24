#!/usr/bin/env python3
"""Build EP04 CSV/PNG cache from segment validation (run when data or EP04 logic changes).

用法: uv run python scripts/build_ep04_cache.py [--force] [--force-segment-inputs] [--n-jobs N]
输入: output/ep01_data_processing/ 与 output/ep03_theoretical_limits/ 缓存、原始帧
输出: output/ep04_global_validation/（outer/inner 结果 CSV、PNG + cache_manifest.json）
关联: EP04
"""

from __future__ import annotations

import argparse
import time

from thermal_core.ep04_cache import build_ep04_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if cache_manifest.json and all artifacts already exist.",
    )
    parser.add_argument(
        "--force-segment-inputs",
        action="store_true",
        help="Regenerate EP04-local segment CSV copies even if they already exist.",
    )
    parser.add_argument("--n-jobs", type=int, default=None, help="Parallel workers (default: min(cpu, 16)).")
    args = parser.parse_args()

    t0 = time.perf_counter()
    cache = build_ep04_cache(
        force=args.force,
        force_segment_inputs=args.force_segment_inputs,
        n_jobs=args.n_jobs,
    )
    elapsed = time.perf_counter() - t0
    print(f"✅ EP04 cache built: {cache.output_dir}")
    print(f"   outer rows: {len(cache.outer_results)}  inner rows: {len(cache.inner_results)}")
    print(f"   reference: {cache.reference_file} (order={cache.reference_order})")
    print(f"   elapsed: {elapsed:.1f}s")
    print("   rebuild: uv run python scripts/build_ep04_cache.py")


if __name__ == "__main__":
    main()
