#!/usr/bin/env python3
"""Build EP10 three-algorithm comparison notebook figure cache.

用法: uv run python scripts/build_ep10_cache.py [--force]
输入: output/ep10_drizzle、ep10_drizzle_4x、ep10_map_tv_sweep、ep10_tgv_sr 各算法产物
输出: output/ep10_method_comparison/（对比图 PNG + cache_manifest.json）
关联: EP10
"""

from __future__ import annotations

import argparse
import time

from thermal_core.ep10_cache import EP10_FIGURE_ARTIFACTS, build_ep10_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if all cached figures already exist.",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    cache = build_ep10_cache(force=args.force)
    elapsed = time.perf_counter() - t0
    built = cache.manifest.get("figures_built", [])
    print(f"✅ EP10 cache: {cache.output_dir}")
    print(f"   figures built this run: {built or '(skipped — cache complete)'}")
    print(f"   artifacts: {', '.join(EP10_FIGURE_ARTIFACTS)}")
    print(f"   elapsed: {elapsed:.1f}s")
    print("   rebuild: uv run python scripts/build_ep10_cache.py")


if __name__ == "__main__":
    main()
