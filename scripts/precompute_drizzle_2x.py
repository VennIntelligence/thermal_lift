#!/usr/bin/env python3
"""预计算 obs_features_2x.npz — 补全 EP12 训练缺失的 drizzle 特征文件。

对已有的 lr_burst.npy + shifts.npy 做 2x bilinear drizzle scatter-add，
保存为压缩 .npz，使 EP12 训练不再需要实时重算。

用法:
    uv run python scripts/precompute_drizzle_2x.py [--pool-dir PATH] [--workers N] [--scale S]
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# tcforge 在项目 editable install 中
from tcforge.classical_sr import drizzle_features


def process_scene(scene_dir: Path, scale: int = 2, kernel: str = "bilinear") -> str:
    """对单个场景预计算 drizzle 特征并保存为 .npz。"""
    out_name = f"obs_features_{scale}x.npz"
    out_path = scene_dir / out_name
    if out_path.exists():
        return f"SKIP {scene_dir.name} ({out_name} already exists)"

    burst_path = scene_dir / "lr_burst.npy"
    shifts_path = scene_dir / "shifts.npy"
    if not burst_path.exists() or not shifts_path.exists():
        return f"MISS {scene_dir.name} (no lr_burst.npy or shifts.npy)"

    t0 = time.monotonic()
    burst = np.load(burst_path, mmap_mode="r")
    shifts = np.load(shifts_path).astype(np.float32, copy=False)
    h_lr, w_lr = burst.shape[1], burst.shape[2]

    features = drizzle_features(
        burst,
        shifts,
        scale=scale,
        output_shape=(h_lr * scale, w_lr * scale),
        kernel=kernel,
    )
    np.savez_compressed(out_path, obs_features=features)
    elapsed = time.monotonic() - t0
    size_mb = out_path.stat().st_size / 1e6
    return f"OK   {scene_dir.name} → {out_name} {features.shape} {size_mb:.1f}MB ({elapsed:.1f}s)"


def main() -> None:
    parser = argparse.ArgumentParser(description="预计算 drizzle 特征 .npz")
    parser.add_argument(
        "--pool-dir",
        type=Path,
        default=Path("data/synthetic/training_pool_4x"),
        help="训练数据池目录 (default: data/synthetic/training_pool_4x)",
    )
    parser.add_argument("--scale", type=int, default=2, help="drizzle scale (default: 2)")
    parser.add_argument("--kernel", type=str, default="bilinear", help="drizzle kernel")
    parser.add_argument("--workers", type=int, default=8, help="并行进程数 (default: 8)")
    args = parser.parse_args()

    pool_dir = args.pool_dir.expanduser().resolve()
    if not pool_dir.exists():
        print(f"ERROR: pool dir not found: {pool_dir}", file=sys.stderr)
        sys.exit(1)

    scenes = sorted(
        p for p in pool_dir.iterdir()
        if p.is_dir() and (p / "metadata.json").exists()
    )
    print(f"Found {len(scenes)} scenes in {pool_dir}")
    print(f"Target: obs_features_{args.scale}x.npz  kernel={args.kernel}  workers={args.workers}")
    print()

    t_start = time.monotonic()
    ok = skip = miss = err = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_scene, s, args.scale, args.kernel): s
            for s in scenes
        }
        for i, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
            except Exception as e:
                result = f"ERR  {futures[future].name}: {e}"
                err += 1

            if result.startswith("OK"):
                ok += 1
            elif result.startswith("SKIP"):
                skip += 1
            elif result.startswith("MISS"):
                miss += 1

            # 每 50 个或最后一个打印进度
            if i % 50 == 0 or i == len(futures):
                elapsed = time.monotonic() - t_start
                print(f"[{i}/{len(futures)}] {elapsed:.0f}s elapsed | {result}")

    total_time = time.monotonic() - t_start
    print(f"\n{'='*60}")
    print(f"Done in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  OK:   {ok}")
    print(f"  SKIP: {skip}")
    print(f"  MISS: {miss}")
    print(f"  ERR:  {err}")

    if ok > 0:
        # 验证一个文件
        sample = pool_dir / scenes[0].name / f"obs_features_{args.scale}x.npz"
        if sample.exists():
            with np.load(sample) as data:
                arr = data[data.files[0]]
                print(f"\n验证: {sample.name} shape={arr.shape} dtype={arr.dtype}")


if __name__ == "__main__":
    main()
