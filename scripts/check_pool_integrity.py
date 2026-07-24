"""合成训练池全量结构完整性校验。

对 tcforge 生成的训练池做逐场景硬校验（区别于 `audit_generated_pool.py` 的抽样
视觉审计）：场景目录数量与命名连续性、manifest.csv 行数与 scene_id 集合、
metadata 形状字段、各数组文件的形状/帧数一致性与 NaN/Inf、PNG 可读性。
lr_burst.npy 与 phase_bin_drizzle_*.npy 是按池配置可选的伴生文件——以首个
场景的文件集合为基准，要求全池一致。

典型用法（根 UV 环境）：

    # 校验 v9 2x 生产池（场景数从 manifest 推断）
    uv run python scripts/check_pool_integrity.py --pool data/synthetic/pool_2x_v9

    # 显式指定期望场景数与倍率
    uv run python scripts/check_pool_integrity.py \\
        --pool data/synthetic/pool_2x_v3_5k --expected-scenes 5000 --scale 2

退出码：0 = 通过；1 = 存在校验错误；2 = 池目录不存在。
关联：tcforge COMPACT_SCENE_FILES 契约（tcforge/README.md）、ACL-023 信息保存管线。
（由收尾前的临时脚本 tmp_check_pool_sanity.py 参数化收编而来，2026-07-24。）
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

CORE_FILES = {
    "metadata.json",
    "shifts.npy",
    "obs_features_1x.npz",
    "hr_mask_4x.png",
    "hr_edge_4x.png",
}
OPTIONAL_FILES_TEMPLATE = ("lr_burst.npy", "phase_bin_drizzle_{scale}x.npy")


def fail(errors: list[str], scene: str, message: str) -> None:
    errors.append(f"{scene}: {message}")


def finite_array(errors: list[str], scene: str, name: str, arr: np.ndarray) -> None:
    if arr.size == 0:
        fail(errors, scene, f"{name} is empty")
        return
    if not np.isfinite(arr).all():
        fail(errors, scene, f"{name} contains NaN/Inf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pool", type=Path, required=True, help="池根目录，如 data/synthetic/pool_2x_v9")
    parser.add_argument(
        "--expected-scenes",
        type=int,
        default=None,
        help="期望场景数（缺省时以 manifest.csv 行数为准）",
    )
    parser.add_argument("--scale", type=int, default=2, help="SR 倍率，决定 HR 形状与 phase 文件名（默认 2）")
    parser.add_argument(
        "--lr-shape",
        type=str,
        default="480,640",
        help="LR 帧形状 ROWS,COLS（默认 480,640）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pool: Path = args.pool
    lr_shape = tuple(int(v) for v in args.lr_shape.split(","))
    hr_shape = tuple(v * args.scale for v in lr_shape)
    phase_name = OPTIONAL_FILES_TEMPLATE[1].format(scale=args.scale)
    optional_files = {OPTIONAL_FILES_TEMPLATE[0], phase_name}

    errors: list[str] = []
    warnings: list[str] = []

    if not pool.exists():
        print(f"Missing pool directory: {pool}", file=sys.stderr)
        return 2

    scenes = sorted(p for p in pool.iterdir() if p.is_dir() and p.name.startswith("scene_"))

    manifest = pool / "manifest.csv"
    manifest_ids: set[str] = set()
    if not manifest.exists():
        errors.append("missing manifest.csv")
    else:
        with manifest.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        manifest_ids = {row.get("scene_id", "") for row in rows}

    expected_scenes = args.expected_scenes if args.expected_scenes is not None else len(manifest_ids)
    if expected_scenes <= 0:
        errors.append("cannot determine expected scene count (no --expected-scenes and empty manifest)")
    else:
        if len(scenes) != expected_scenes:
            errors.append(f"expected {expected_scenes} scene dirs, found {len(scenes)}")
        expected_names = [f"scene_{i:04d}" for i in range(expected_scenes)]
        actual_names = [p.name for p in scenes]
        missing_names = sorted(set(expected_names) - set(actual_names))
        extra_names = sorted(set(actual_names) - set(expected_names))
        if missing_names:
            errors.append(f"missing scene dirs: {missing_names[:20]}{' ...' if len(missing_names) > 20 else ''}")
        if extra_names:
            errors.append(f"extra scene dirs: {extra_names[:20]}{' ...' if len(extra_names) > 20 else ''}")
        if manifest_ids and manifest_ids != set(expected_names):
            errors.append("manifest scene_id set does not exactly match expected scene IDs")

    required_files = set(CORE_FILES)
    if scenes:
        first_present = {p.name for p in scenes[0].iterdir() if p.is_file()}
        required_files |= optional_files & first_present
        skipped_optional = optional_files - first_present
        if skipped_optional:
            print(f"note: optional files absent in first scene, skipped pool-wide: {sorted(skipped_optional)}")

    for idx, scene_dir in enumerate(scenes, start=1):
        scene = scene_dir.name
        present = {p.name for p in scene_dir.iterdir() if p.is_file()}
        missing = required_files - present
        if missing:
            fail(errors, scene, f"missing files: {sorted(missing)}")
            continue

        try:
            metadata = json.loads((scene_dir / "metadata.json").read_text(encoding="utf-8"))
        except Exception as exc:
            fail(errors, scene, f"metadata.json unreadable: {exc}")
            continue

        if metadata.get("scene_id") != scene:
            fail(errors, scene, f"metadata scene_id mismatch: {metadata.get('scene_id')!r}")
        if tuple(metadata.get("lr_shape", [])) != lr_shape:
            fail(errors, scene, f"metadata lr_shape mismatch: {metadata.get('lr_shape')!r}")
        if tuple(metadata.get("hr_shape", [])) != hr_shape:
            fail(errors, scene, f"metadata hr_shape mismatch: {metadata.get('hr_shape')!r}")

        n_frames = int(metadata.get("n_frames", -1))
        if n_frames <= 0:
            fail(errors, scene, f"invalid n_frames: {n_frames}")

        if OPTIONAL_FILES_TEMPLATE[0] in required_files:
            try:
                lr_burst = np.load(scene_dir / "lr_burst.npy", mmap_mode="r")
                if lr_burst.ndim != 3 or lr_burst.shape[1:] != lr_shape:
                    fail(errors, scene, f"lr_burst shape mismatch: {lr_burst.shape}")
                if n_frames > 0 and lr_burst.shape[0] != n_frames:
                    fail(errors, scene, f"lr_burst frame count {lr_burst.shape[0]} != metadata n_frames {n_frames}")
                finite_array(errors, scene, "lr_burst", lr_burst)
            except Exception as exc:
                fail(errors, scene, f"lr_burst.npy unreadable: {exc}")

        try:
            shifts = np.load(scene_dir / "shifts.npy", mmap_mode="r")
            if shifts.ndim != 2 or shifts.shape[1] != 2:
                fail(errors, scene, f"shifts shape mismatch: {shifts.shape}")
            if n_frames > 0 and shifts.shape[0] != n_frames:
                fail(errors, scene, f"shifts frame count {shifts.shape[0]} != metadata n_frames {n_frames}")
            finite_array(errors, scene, "shifts", shifts)
        except Exception as exc:
            fail(errors, scene, f"shifts.npy unreadable: {exc}")

        if phase_name in required_files:
            try:
                phase = np.load(scene_dir / phase_name, mmap_mode="r")
                if phase.ndim != 3 or phase.shape[1:] != hr_shape:
                    fail(errors, scene, f"{phase_name} shape mismatch: {phase.shape}")
                finite_array(errors, scene, phase_name, phase)
            except Exception as exc:
                fail(errors, scene, f"{phase_name} unreadable: {exc}")

        try:
            with np.load(scene_dir / "obs_features_1x.npz") as obs:
                if not obs.files:
                    fail(errors, scene, "obs_features_1x.npz has no arrays")
                for key in obs.files:
                    arr = obs[key]
                    if arr.ndim < 2:
                        fail(errors, scene, f"obs_features_1x[{key}] has unexpected ndim {arr.ndim}")
                    finite_array(errors, scene, f"obs_features_1x[{key}]", arr)
        except Exception as exc:
            fail(errors, scene, f"obs_features_1x.npz unreadable: {exc}")

        for png_name in ("hr_mask_4x.png", "hr_edge_4x.png"):
            image = cv2.imread(str(scene_dir / png_name), cv2.IMREAD_UNCHANGED)
            if image is None:
                fail(errors, scene, f"{png_name} unreadable")
            elif image.shape[:2] != hr_shape:
                fail(errors, scene, f"{png_name} shape mismatch: {image.shape}")
            elif image.max() == image.min():
                warnings.append(f"{scene}: {png_name} is constant value {int(image.max())}")

        if idx % 250 == 0:
            print(f"checked {idx}/{len(scenes)} scenes")

    print(f"checked scenes: {len(scenes)}")
    print(f"errors: {len(errors)}")
    print(f"warnings: {len(warnings)}")

    if warnings:
        print("first warnings:")
        for warning in warnings[:20]:
            print(f"  WARN {warning}")

    if errors:
        print("first errors:")
        for error in errors[:50]:
            print(f"  ERROR {error}")
        return 1

    print("sanity check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
