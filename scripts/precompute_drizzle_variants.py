#!/usr/bin/env python3
"""Precompute augmented drizzle feature variants for a burst training pool.

V9A (hybrid_drizzle2x) 原本在 DataLoader worker 内对每个 scene 每个 epoch
现场跑 ``drizzle_features``（全幅 scatter，~2.7 s/scene），并把 float32 burst
缓存进内存（~305 MB/scene），导致首 batch 极慢且主机 OOM（见 ACL-018）。

本脚本把该计算移到池侧：对每个 scene 预生成 K 个固定增广版本的 drizzle
特征，保存为 ``drizzle_variants_{scale}x.npy``（shape ``(K, 3, H_hr, W_hr)``,
float16，与池内 obs_features 的存储 dtype 一致）。

变体定义（与 ``unet_sr.dataset.ThermalSRDataset._select_burst`` 的增广分布一致）：

* variant 0 — 全部 burst 帧、无 shift 噪声（canonical，与推理口径一致）
* variant 1..K-1 — 随机抽取 60–100% 帧（≥ min-frames）+ shift 高斯噪声

训练时 dataset 每 epoch 按 (seed, epoch, scene) 确定性抽取一个变体，
不再读取 lr_burst。

用法（项目根目录）::

    uv run python scripts/precompute_drizzle_variants.py \
        --pool-dir data/synthetic/training_pool_2x_aa_burst \
        --num-variants 4 --workers 14

可断点续跑：已存在且 shape 正确的变体文件默认跳过（--overwrite 强制重算）。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TCFORGE_SRC = PROJECT_ROOT / "tcforge" / "src"
if TCFORGE_SRC.exists() and str(TCFORGE_SRC) not in sys.path:
    sys.path.insert(0, str(TCFORGE_SRC))

from tcforge.classical_sr import drizzle_features  # noqa: E402


def _scene_dirs(pool_dir: Path) -> list[Path]:
    manifest = pool_dir / "manifest.csv"
    if manifest.exists():
        dirs: list[Path] = []
        with manifest.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                value = row["scene_dir"]
                path = Path(value)
                dirs.append(path if path.is_absolute() else pool_dir / path)
        return dirs
    return sorted(p for p in pool_dir.iterdir() if p.is_dir() and (p / "metadata.json").exists())


def _build_variants(
    scene_dir: Path,
    *,
    num_variants: int,
    seed: int,
    keep_frac_min: float,
    keep_frac_max: float,
    min_frames: int,
    shift_noise_std_px: float,
    overwrite: bool,
) -> tuple[str, str, float]:
    """Compute and save drizzle variants for one scene. Returns (scene_id, status, seconds)."""

    t0 = time.perf_counter()
    metadata = json.loads((scene_dir / "metadata.json").read_text(encoding="utf-8"))
    scale = int(metadata.get("scale", 2))
    out_path = scene_dir / f"drizzle_variants_{scale}x.npy"

    lr_burst = np.load(scene_dir / "lr_burst.npy", mmap_mode="r")
    shifts = np.load(scene_dir / "shifts.npy").astype(np.float32, copy=False)
    n_frames, h_lr, w_lr = lr_burst.shape
    expected_shape = (num_variants, 3, h_lr * scale, w_lr * scale)

    if out_path.exists() and not overwrite:
        existing = np.load(out_path, mmap_mode="r")
        if existing.shape == expected_shape and existing.dtype == np.float16:
            return scene_dir.name, "skipped", time.perf_counter() - t0

    scene_key = int(metadata.get("scene_index", zlib.crc32(scene_dir.name.encode())))
    variants = np.empty(expected_shape, dtype=np.float16)
    for k in range(num_variants):
        rng = np.random.default_rng([seed, scene_key, k])
        if k == 0:
            indices = np.arange(n_frames)
            shifts_k = shifts
        else:
            keep_frac = rng.uniform(keep_frac_min, keep_frac_max)
            n_keep = min(n_frames, max(min_frames, int(round(n_frames * keep_frac))))
            indices = rng.choice(n_frames, size=n_keep, replace=False)
            indices.sort()
            shifts_k = shifts[indices] + rng.normal(
                0, shift_noise_std_px, size=(len(indices), 2)
            ).astype(np.float32)
        burst_k = np.asarray(lr_burst[indices], dtype=np.float32)
        drz = drizzle_features(burst_k, shifts_k, scale=scale, kernel="bilinear")
        variants[k] = drz.astype(np.float16)

    # Atomic write: np.save appends ".npy" to bare names, so write via handle.
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    with tmp_path.open("wb") as f:
        np.save(f, variants)
    tmp_path.rename(out_path)
    return scene_dir.name, "built", time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--num-variants", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--keep-frac-min", type=float, default=0.6)
    parser.add_argument("--keep-frac-max", type=float, default=1.0)
    parser.add_argument("--min-frames", type=int, default=30)
    parser.add_argument("--shift-noise-std-px", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    pool_dir = args.pool_dir.expanduser().resolve()
    scenes = _scene_dirs(pool_dir)
    if not scenes:
        raise SystemExit(f"no scenes found under {pool_dir}")

    meta = {
        "generator": "scripts/precompute_drizzle_variants.py",
        "num_variants": args.num_variants,
        "seed": args.seed,
        "variant_0": "full burst, no shift noise (canonical)",
        "augmentation": {
            "keep_frac": [args.keep_frac_min, args.keep_frac_max],
            "min_frames": args.min_frames,
            "shift_noise_std_px": args.shift_noise_std_px,
        },
        "kernel": "bilinear",
        "dtype": "float16",
    }
    (pool_dir / "drizzle_variants_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    kwargs = dict(
        num_variants=args.num_variants,
        seed=args.seed,
        keep_frac_min=args.keep_frac_min,
        keep_frac_max=args.keep_frac_max,
        min_frames=args.min_frames,
        shift_noise_std_px=args.shift_noise_std_px,
        overwrite=args.overwrite,
    )

    counts = {"built": 0, "skipped": 0, "failed": 0}
    t_start = time.perf_counter()
    if args.workers <= 1:
        iterator = (
            _build_variants(scene, **kwargs) for scene in scenes
        )
        for scene_id, status, dt in tqdm(iterator, total=len(scenes), desc="drizzle variants"):
            counts[status] += 1
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_build_variants, scene, **kwargs): scene for scene in scenes}
            for future in tqdm(as_completed(futures), total=len(futures), desc="drizzle variants"):
                scene = futures[future]
                try:
                    _, status, _ = future.result()
                    counts[status] += 1
                except Exception as exc:  # noqa: BLE001 — 单 scene 失败不中断全池
                    counts["failed"] += 1
                    print(f"[FAILED] {scene.name}: {exc!r}", file=sys.stderr)

    elapsed = time.perf_counter() - t_start
    print(
        f"done in {elapsed/60:.1f} min: built={counts['built']} "
        f"skipped={counts['skipped']} failed={counts['failed']} (pool={pool_dir})"
    )
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
