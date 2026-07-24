#!/usr/bin/env python3
"""Build EP07 demo figures and table cache for notebook fragments.

用法: uv run python scripts/build_ep07_cache.py [--force]
输入: tcforge/src 合成引擎（不可用时跳过 demo 生成并在 manifest 记录 demo_skipped）
输出: output/ep07_thermal_chip_phantom/（demo_dataset/ + 图表 + cache_manifest.json）
关联: EP07
"""

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
