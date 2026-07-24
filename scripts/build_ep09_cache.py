#!/usr/bin/env python3
"""Validate EP09 PSF calibration artifacts and write cache manifest.

用法: uv run python scripts/build_ep09_cache.py [--force]
输入: output/ep09_psf_calibration/ 已有标定产物（calibration_summary.json、sigma_*.json、
      route_sigma_summary.csv 等；缺失时报错并提示 algo 侧重建命令）
输出: 同目录 cache_manifest.json（只校验 + 写 manifest，不重算）
关联: EP09
"""

from __future__ import annotations

import argparse
import time

from thermal_core.ep09_cache import EP09_ARTIFACTS, build_ep09_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite manifest even if all artifacts already exist.",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    cache = build_ep09_cache(force=args.force)
    elapsed = time.perf_counter() - t0
    print(f"✅ EP09 cache validated: {cache.output_dir}")
    if cache.summary:
        print(f"   final sigma: {cache.summary['final_sigma_lr_px']:.4f} LR px")
        print(f"   4x verdict: {cache.summary['four_x_verdict']}")
    print(f"   artifacts: {len(EP09_ARTIFACTS) - 1} files + manifest")
    print(f"   elapsed: {elapsed:.1f}s")
    print("   rebuild manifest: uv run python scripts/build_ep09_cache.py")


if __name__ == "__main__":
    main()
