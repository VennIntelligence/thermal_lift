#!/usr/bin/env python3
"""Build EP01–EP10 notebook caches in dependency order.

按依赖顺序依次调用 scripts/build_ep01_cache.py ... build_ep10_cache.py。
Git 迁移到新机器、或数据 / core 逻辑变更后先运行本脚本，再跑 build_all_notebooks.py。

用法:
    uv run python scripts/build_all_caches.py                   # 全部构建
    uv run python scripts/build_all_caches.py --only ep02,ep03  # 只构建部分
    uv run python scripts/build_all_caches.py --force           # 强制重建
    （--skip-missing: 跳过尚不存在的 builder 脚本）

输入: data/data_raw/ 原始数据及各 EP 上游产物（由各 builder 自行检查）
输出: output/ep01_* ... output/ep10_* 各缓存目录（CSV/PNG + cache_manifest.json）
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


# Dependency order: ep01 first, then ep02/ep03, ep04, ep05, ep06, ep07 independent, ep08/ep09/ep10
CACHE_BUILDERS: list[tuple[str, str]] = [
    ("ep01", "scripts/build_ep01_cache.py"),
    ("ep02", "scripts/build_ep02_cache.py"),
    ("ep03", "scripts/build_ep03_cache.py"),
    ("ep04", "scripts/build_ep04_cache.py"),
    ("ep05", "scripts/build_ep05_cache.py"),
    ("ep06", "scripts/build_ep06_cache.py"),
    ("ep07", "scripts/build_ep07_cache.py"),
    ("ep08", "scripts/build_ep08_cache.py"),
    ("ep09", "scripts/build_ep09_cache.py"),
    ("ep10", "scripts/build_ep10_cache.py"),
]


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated episode ids, e.g. ep02,ep03",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Pass --force to each cache builder.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip builders whose script file does not exist yet.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    only = {item.strip().lower() for item in args.only.split(",") if item.strip()}

    selected = [
        (ep_id, script)
        for ep_id, script in CACHE_BUILDERS
        if not only or ep_id in only
    ]

    if not selected:
        print("No cache builders selected.")
        sys.exit(1)

    failed: list[str] = []
    t_total = time.perf_counter()

    for ep_id, script_rel in selected:
        script = root / script_rel
        if not script.exists():
            if args.skip_missing:
                print(f"SKIP {ep_id}: {script_rel} not found")
                continue
            print(f"ERROR {ep_id}: missing {script_rel}")
            failed.append(ep_id)
            continue

        cmd = [sys.executable, str(script)]
        if args.force:
            cmd.append("--force")

        print(f"{'=' * 60}")
        print(f"Building {ep_id}: {' '.join(cmd)}")
        t0 = time.perf_counter()
        result = subprocess.run(cmd, cwd=root)
        elapsed = time.perf_counter() - t0

        if result.returncode != 0:
            failed.append(ep_id)
            print(f"FAILED {ep_id} ({elapsed:.1f}s)")
        else:
            print(f"OK {ep_id} ({elapsed:.1f}s)")

    total = time.perf_counter() - t_total
    print(f"{'=' * 60}")
    print(f"Done: {len(selected) - len(failed)}/{len(selected)} succeeded ({total:.1f}s total)")
    if failed:
        print(f"Failed: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
