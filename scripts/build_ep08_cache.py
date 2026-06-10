#!/usr/bin/env python3
"""Build EP08 Stage 3 notebook figure cache (run when stage3 metrics change)."""

from __future__ import annotations

import argparse
import time

from thermal_core.ep08_cache import EP08_FIGURE_ARTIFACTS, build_ep08_cache


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if all cached figures already exist.",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    cache = build_ep08_cache(force=args.force)
    elapsed = time.perf_counter() - t0
    built = cache.manifest.get("figures_built", [])
    pending = cache.manifest.get("pending_figures", [])
    print(f"✅ EP08 cache: {cache.output_dir}")
    print(f"   stage3 runs: {cache.manifest.get('stage3_runs', '?')}")
    print(f"   stage3 runs in 248-frame contract: {cache.manifest.get('stage3_runs_in_contract', '?')}")
    print(f"   dropped stale runs: {cache.manifest.get('dropped_stale_runs', []) or '(none)'}")
    print(f"   figures built this run: {built or '(skipped — cache complete)'}")
    print(f"   pending figures: {pending or '(none)'}")
    print(f"   artifacts: {', '.join(EP08_FIGURE_ARTIFACTS)}")
    print(f"   elapsed: {elapsed:.1f}s")
    print("   rebuild: uv run python scripts/build_ep08_cache.py")


if __name__ == "__main__":
    main()
