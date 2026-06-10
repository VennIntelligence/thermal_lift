"""Dataset for EP12 4x drizzle feature scenes."""

from __future__ import annotations

import csv
import json
import multiprocessing as mp
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.ndimage import zoom
from torch.utils.data import Dataset, Sampler

from tcforge.classical_sr import drizzle_features
from tcforge.reconstruct import reconstruct_hr_temperature
from tcforge.storage import _read_png_gray8


def _scene_paths(training_pool_dir: str | Path) -> list[Path]:
    root = Path(training_pool_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"training_pool_dir not found: {root}")

    manifest = root / "manifest.csv"
    if manifest.exists():
        paths: list[Path] = []
        with manifest.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                value = row.get("scene_dir")
                if not value:
                    raise ValueError(f"{manifest} contains a row without scene_dir")
                scene_dir = Path(value).expanduser()
                paths.append(scene_dir if scene_dir.is_absolute() else root / scene_dir)
        if not paths:
            raise ValueError(f"{manifest} contains no scenes")
        return paths

    paths = sorted(path for path in root.iterdir() if path.is_dir() and (path / "metadata.json").exists())
    if not paths:
        raise FileNotFoundError(f"no manifest.csv or scene directories found under {root}")
    return paths


def _npz_array(path: Path) -> np.ndarray:
    with np.load(path) as data:
        for key in ("obs_features", "features", "arr_0"):
            if key in data:
                return np.asarray(data[key], dtype=np.float32)
        if len(data.files) == 1:
            return np.asarray(data[data.files[0]], dtype=np.float32)
    raise ValueError(f"{path} does not contain a feature array")


def _as_feature_array(array: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"{name} must have shape (C,H,W), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return arr


def _upsample_to_shape(features: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    arr = _as_feature_array(features, "features")
    hr_h, hr_w = map(int, output_shape)
    zoom_y = hr_h / arr.shape[-2]
    zoom_x = hr_w / arr.shape[-1]
    out = zoom(arr, (1, zoom_y, zoom_x), order=1).astype(np.float32, copy=False)
    if out.shape[-2:] != (hr_h, hr_w):
        raise ValueError(f"upsampled features shape mismatch: {out.shape[-2:]} vs {(hr_h, hr_w)}")
    return out


def _metadata_float(metadata: dict[str, Any], key: str, default: float) -> float:
    return float(metadata.get(key, default))


def _reconstruct_target(mask: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    seed = metadata.get("low_freq_seed", metadata.get("seed"))
    return reconstruct_hr_temperature(
        mask,
        T_bg_c=_metadata_float(metadata, "T_bg_c", 21.0),
        delta_T_c=_metadata_float(metadata, "delta_T_c", 2.0),
        low_freq_amplitude_c=_metadata_float(metadata, "low_freq_amplitude_c", 0.2),
        low_freq_sigma_px=_metadata_float(metadata, "low_freq_sigma_px", 96.0),
        seed=None if seed is None else int(seed),
    ).astype(np.float32, copy=False)


class SceneInterleavedSampler(Sampler[int]):
    """Cache-friendly sampling with worker-scene affinity.

    When ``num_workers > 1``, scenes are partitioned across workers so that
    each DataLoader worker only accesses its own scene subset, keeping LRU
    cache hit rate near 100%.  This mirrors EP07's proven strategy.

    Within each worker's partition, scenes are grouped into buckets of
    ``scenes_per_bucket``.  Patches are emitted in round-robin order across
    scenes within a bucket: each scene contributes ``patches_per_fetch``
    consecutive patches before the sampler moves on.
    """

    def __init__(
        self,
        n_scenes: int,
        patches_per_scene: int,
        *,
        scenes_per_bucket: int = 16,
        patches_per_fetch: int = 8,
        seed: int = 42,
        num_workers: int = 0,
        batch_size: int = 1,
    ) -> None:
        super().__init__()
        self.n_scenes = int(n_scenes)
        self.patches_per_scene = int(patches_per_scene)
        self.scenes_per_bucket = int(scenes_per_bucket)
        self.patches_per_fetch = int(patches_per_fetch)
        self.seed = int(seed)
        self.num_workers = max(1, int(num_workers)) if num_workers > 0 else 1
        self.batch_size = max(1, int(batch_size))
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _generate_bucket_indices(
        self, scene_order: np.ndarray, rng: np.random.Generator,
    ) -> list[int]:
        """Generate cache-friendly bucket-interleaved patch indices."""
        indices: list[int] = []
        bucket_sz = self.scenes_per_bucket
        for bstart in range(0, len(scene_order), bucket_sz):
            bucket_scenes = scene_order[bstart : bstart + bucket_sz]
            scene_patches: list[list[int]] = []
            for s in bucket_scenes:
                base = int(s) * self.patches_per_scene
                perm = rng.permutation(self.patches_per_scene)
                scene_patches.append((base + perm).tolist())
            cursors = [0] * len(bucket_scenes)
            while True:
                any_remaining = False
                for si in range(len(bucket_scenes)):
                    patches = scene_patches[si]
                    c = cursors[si]
                    if c < len(patches):
                        any_remaining = True
                        end = min(c + self.patches_per_fetch, len(patches))
                        indices.extend(patches[c:end])
                        cursors[si] = end
                if not any_remaining:
                    break
        return indices

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        scene_order = rng.permutation(self.n_scenes)

        W = self.num_workers
        if W <= 1:
            # Single-process or 1 worker: no affinity needed.
            return iter(self._generate_bucket_indices(scene_order, rng))

        # --- Worker-scene affinity ---
        # PyTorch DataLoader sends batch b to worker (b % W) via round-robin.
        # We partition scenes across workers and interleave the output
        # batch-by-batch so each worker only sees its own scene partition,
        # keeping LRU cache hit rate near 100%.
        B = self.batch_size

        # Round-robin partition for balance (worker w gets scenes w, w+W, …)
        worker_scene_lists = [scene_order[w::W] for w in range(W)]

        # Generate cache-friendly bucket indices per worker
        worker_indices: list[list[int]] = []
        for w in range(W):
            w_rng = np.random.default_rng([self.seed, self._epoch, w])
            worker_indices.append(
                self._generate_bucket_indices(worker_scene_lists[w], w_rng)
            )

        # Pad each worker's indices to the same length, rounded to batch_size
        max_len = max((len(wi) for wi in worker_indices), default=0)
        padded_len = ((max_len + B - 1) // B) * B

        for w in range(W):
            wi = worker_indices[w]
            orig_len = len(wi)
            if orig_len == 0:
                worker_indices[w] = [0] * padded_len
            elif orig_len < padded_len:
                repeats = (padded_len + orig_len - 1) // orig_len
                worker_indices[w] = (wi * repeats)[:padded_len]

        # Interleave batch-by-batch: emit [w0_batch0, w1_batch0, …, w0_batch1, …]
        result: list[int] = []
        n_batches_per_worker = padded_len // B if B > 0 else 0
        for b_idx in range(n_batches_per_worker):
            start = b_idx * B
            for w in range(W):
                result.extend(worker_indices[w][start : start + B])

        return iter(result)

    def __len__(self) -> int:
        return self.n_scenes * self.patches_per_scene


class ThermalSR4xDataset(Dataset[dict[str, Any]]):
    """Random same-grid 4x patches over precomputed drizzle feature scenes."""

    def __init__(
        self,
        training_pool_dir: str | Path,
        *,
        patch_size: int = 256,
        scale: int = 4,
        drizzle_scale: int = 2,
        seed: int = 42,
        patches_per_scene: int = 64,
        max_scene_cache: int = 8,
        include_multiscale: bool = False,
        burst_augment: bool = False,
        burst_keep_range: tuple[float, float] = (0.6, 1.0),
        min_burst_frames: int = 30,
        shift_noise_std_px: float = 0.05,
        drizzle_kernel: str = "bilinear",
        defer_1x_upsample: bool = False,
        return_metadata: bool = True,
    ) -> None:
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if scale <= 0:
            raise ValueError("scale must be positive")
        if patches_per_scene <= 0:
            raise ValueError("patches_per_scene must be positive")
        if max_scene_cache <= 0:
            raise ValueError("max_scene_cache must be positive")
        keep_low, keep_high = map(float, burst_keep_range)
        if keep_low <= 0.0 or keep_high <= 0.0 or keep_low > keep_high or keep_high > 1.0:
            raise ValueError("burst_keep_range must satisfy 0 < low <= high <= 1")
        if min_burst_frames <= 0:
            raise ValueError("min_burst_frames must be positive")
        if shift_noise_std_px < 0:
            raise ValueError("shift_noise_std_px must be >= 0")
        self.scene_paths = _scene_paths(training_pool_dir)
        self.patch_size = int(patch_size)
        self.scale = int(scale)
        self.drizzle_scale = int(drizzle_scale)
        if self.scale % self.drizzle_scale != 0:
            raise ValueError("scale must be divisible by drizzle_scale")
        self.seed = int(seed)
        self.patches_per_scene = int(patches_per_scene)
        self.max_scene_cache = int(max_scene_cache)
        self.include_multiscale = bool(include_multiscale)
        self.burst_augment = bool(burst_augment)
        self.burst_keep_range = (keep_low, keep_high)
        self.min_burst_frames = int(min_burst_frames)
        self.shift_noise_std_px = float(shift_noise_std_px)
        self.drizzle_kernel = str(drizzle_kernel)
        self.defer_1x_upsample = bool(defer_1x_upsample)
        self.return_metadata = bool(return_metadata)
        self._cache: OrderedDict[tuple[int, int], dict[str, Any]] = OrderedDict()
        self._shared_epoch = mp.Value("i", 0)

    @property
    def in_channels(self) -> int:
        return 11 if self.include_multiscale else 8

    def __len__(self) -> int:
        return len(self.scene_paths) * self.patches_per_scene

    def set_epoch(self, epoch: int) -> None:
        self._shared_epoch.value = int(epoch)

    @property
    def _epoch(self) -> int:
        return self._shared_epoch.value

    def _select_burst(
        self,
        lr_burst: np.ndarray,
        shifts: np.ndarray,
        *,
        scene_index: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        frames = np.asarray(lr_burst, dtype=np.float32)
        shift_arr = np.asarray(shifts, dtype=np.float32)
        if frames.ndim != 3 or shift_arr.shape != (frames.shape[0], 2):
            raise ValueError("lr_burst/shifts shape mismatch")
        rng = np.random.default_rng(self.seed + scene_index * 1_000_003 + self._epoch * 17_171)
        keep_ratio = float(rng.uniform(*self.burst_keep_range))
        n_frames = frames.shape[0]
        n_keep = min(n_frames, max(int(round(n_frames * keep_ratio)), min(self.min_burst_frames, n_frames), 1))
        indices = np.sort(rng.choice(n_frames, n_keep, replace=False))
        selected_shifts = shift_arr[indices].copy()
        if self.shift_noise_std_px > 0:
            selected_shifts += rng.normal(0.0, self.shift_noise_std_px, size=selected_shifts.shape).astype(np.float32)
        return (
            frames[indices],
            selected_shifts.astype(np.float32, copy=False),
            {
                "burst_augmented": True,
                "burst_keep_ratio": keep_ratio,
                "burst_frames_kept": int(n_keep),
                "shift_noise_std_px": self.shift_noise_std_px,
            },
        )

    def _drizzle_from_burst(
        self,
        lr_burst: np.ndarray,
        shifts: np.ndarray,
        *,
        hr_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray | None]:
        frames = np.asarray(lr_burst, dtype=np.float32)
        obs_drz = drizzle_features(
            frames,
            shifts,
            scale=self.drizzle_scale,
            output_shape=hr_shape,
            kernel=self.drizzle_kernel,
        )
        obs_2x_up = None
        if self.include_multiscale:
            obs_2x = drizzle_features(
                frames,
                shifts,
                scale=2,
                output_shape=(frames.shape[1] * 2, frames.shape[2] * 2),
                kernel=self.drizzle_kernel,
            )
            obs_2x_up = _upsample_to_shape(obs_2x, hr_shape)
        return obs_drz, obs_2x_up

    def _load_cached(self, scene_index: int) -> dict[str, Any]:
        cache_key = (int(scene_index), int(self._epoch if self.burst_augment else -1))
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            return cached

        root = self.scene_paths[scene_index]
        required = ["obs_features_1x.npz", "hr_mask_4x.png", "hr_edge_4x.png", "metadata.json"]
        has_burst_inputs = (root / "lr_burst.npy").exists() and (root / "shifts.npy").exists()
        drz_name = f"obs_features_{self.drizzle_scale}x.npz"
        if not self.burst_augment and not (root / drz_name).exists() and not has_burst_inputs:
            required.append(drz_name)
        if self.burst_augment:
            required.extend(["lr_burst.npy", "shifts.npy"])
        missing = [name for name in required if not (root / name).exists()]
        if missing:
            raise FileNotFoundError(f"{root} missing EP12 scene files: {missing}")

        obs_1x = _npz_array(root / "obs_features_1x.npz").astype(np.float32, copy=False)
        if obs_1x.ndim != 3:
            raise ValueError(f"{root}/obs_features_1x.npz must have shape (C,H,W)")
        with (root / "metadata.json").open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        hr_mask = (_read_png_gray8(root / "hr_mask_4x.png").astype(np.float32) / 255.0).astype(np.float32)
        hr_edge = (_read_png_gray8(root / "hr_edge_4x.png") > 0).astype(np.float32)
        hr_target = _reconstruct_target(hr_mask, metadata)
        hr_h, hr_w = map(int, hr_target.shape)
        model_up = self.scale // self.drizzle_scale
        drz_h, drz_w = hr_h // model_up, hr_w // model_up

        augmentation = {"burst_augmented": False}
        obs_2x_up = None
        if self.burst_augment:
            burst = np.load(root / "lr_burst.npy", mmap_mode="r")
            shifts = np.load(root / "shifts.npy").astype(np.float32, copy=False)
            burst_sub, shifts_sub, augmentation = self._select_burst(burst, shifts, scene_index=scene_index)
            obs_drz, obs_2x_up = self._drizzle_from_burst(burst_sub, shifts_sub, hr_shape=(drz_h, drz_w))
        else:
            drz_name = f"obs_features_{self.drizzle_scale}x.npz"
            if (root / drz_name).exists():
                obs_drz = _npz_array(root / drz_name).astype(np.float32, copy=False)
            elif (root / "lr_burst.npy").exists() and (root / "shifts.npy").exists():
                obs_drz, obs_2x_up = self._drizzle_from_burst(
                    np.load(root / "lr_burst.npy", mmap_mode="r"),
                    np.load(root / "shifts.npy").astype(np.float32, copy=False),
                    hr_shape=(drz_h, drz_w),
                )
            else:
                raise FileNotFoundError(f"{root} has no {drz_name} or lr_burst.npy")

        if obs_drz.ndim != 3 or obs_drz.shape[0] < 3:
            raise ValueError(f"{root} drizzle features must have shape (>=3,H,W), got {obs_drz.shape}")
        if self.include_multiscale and obs_2x_up is None:
            if (root / "obs_features_2x_up4x.npz").exists():
                obs_2x_up = _npz_array(root / "obs_features_2x_up4x.npz").astype(np.float32, copy=False)
            elif (root / "lr_burst.npy").exists() and (root / "shifts.npy").exists():
                _, obs_2x_up = self._drizzle_from_burst(
                    np.load(root / "lr_burst.npy", mmap_mode="r"),
                    np.load(root / "shifts.npy").astype(np.float32, copy=False),
                    hr_shape=(drz_h, drz_w),
                )
            else:
                raise FileNotFoundError(f"{root} has no obs_features_2x_up4x.npz or lr_burst.npy")

        # --- Cache compact representation to save RAM ---
        packed = {
            "scene_dir": root,
            "metadata": metadata,
            "augmentation": augmentation,
            "obs_drz": _as_feature_array(obs_drz, "obs_features_drz"),
            "obs_1x": obs_1x,
            "obs_2x_up": _as_feature_array(obs_2x_up, "obs_features_2x_up") if obs_2x_up is not None else None,
            "hr_target": hr_target,
            "hr_edge": hr_edge,
        }
        self._cache[cache_key] = packed
        self._cache.move_to_end(cache_key)
        while len(self._cache) > self.max_scene_cache:
            self._cache.popitem(last=False)
        return packed

    def _crop_origin(self, index: int, shape: tuple[int, int]) -> tuple[int, int]:
        rows, cols = shape
        if self.patch_size > rows or self.patch_size > cols:
            raise ValueError(f"patch_size={self.patch_size} larger than scene shape {shape}")
        rng = np.random.default_rng(self.seed + int(index) + self._epoch * len(self))
        max_y = rows - self.patch_size
        max_x = cols - self.patch_size
        y = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
        x = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0
        # Align to the full SR scale so both drizzle and 1x context crops
        # map to integer source pixels.
        if self.scale > 1:
            y = (y // self.scale) * self.scale
            x = (x // self.scale) * self.scale
        return y, x

    def _augment(
        self,
        obs: np.ndarray,
        target: np.ndarray,
        edge: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if rng.random() < 0.5:
            obs = obs[:, :, ::-1].copy()
            target = target[:, ::-1].copy()
            edge = edge[:, ::-1].copy()
        if rng.random() < 0.5:
            obs = obs[:, ::-1, :].copy()
            target = target[::-1, :].copy()
            edge = edge[::-1, :].copy()
        k = int(rng.integers(0, 4))
        if k:
            obs = np.rot90(obs, k, axes=(1, 2)).copy()
            target = np.rot90(target, k, axes=(0, 1)).copy()
            edge = np.rot90(edge, k, axes=(0, 1)).copy()
        return obs, target, edge

    def _augment_deferred(
        self,
        obs_hr: np.ndarray,
        obs_lr: np.ndarray,
        target: np.ndarray,
        edge: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if rng.random() < 0.5:
            obs_hr = obs_hr[:, :, ::-1].copy()
            obs_lr = obs_lr[:, :, ::-1].copy()
            target = target[:, ::-1].copy()
            edge = edge[:, ::-1].copy()
        if rng.random() < 0.5:
            obs_hr = obs_hr[:, ::-1, :].copy()
            obs_lr = obs_lr[:, ::-1, :].copy()
            target = target[::-1, :].copy()
            edge = edge[::-1, :].copy()
        k = int(rng.integers(0, 4))
        if k:
            obs_hr = np.rot90(obs_hr, k, axes=(1, 2)).copy()
            obs_lr = np.rot90(obs_lr, k, axes=(1, 2)).copy()
            target = np.rot90(target, k, axes=(0, 1)).copy()
            edge = np.rot90(edge, k, axes=(0, 1)).copy()
        return obs_hr, obs_lr, target, edge

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index = len(self) + index
        if index < 0 or index >= len(self):
            raise IndexError(index)
        scene_index = index // self.patches_per_scene
        scene = self._load_cached(scene_index)
        obs_drz = scene["obs_drz"]
        obs_1x = scene["obs_1x"]
        obs_2x_up = scene.get("obs_2x_up")
        target = scene["hr_target"]
        edge = scene["hr_edge"]

        # Crop origin at HR grid; aligned to the full SR scale.
        y, x = self._crop_origin(index, tuple(map(int, target.shape)))
        p = self.patch_size
        model_up = self.scale // self.drizzle_scale
        p_drz = p // model_up

        # Crop target/edge at full (4x) resolution
        target_patch = target[y : y + p, x : x + p]
        edge_patch = edge[y : y + p, x : x + p]

        # Crop drizzle features at drizzle resolution
        y_drz = y // model_up
        x_drz = x // model_up
        obs_drz_patch = obs_drz[:, y_drz : y_drz + p_drz, x_drz : x_drz + p_drz]

        # Crop 1x features at LR coordinates
        scale = self.scale
        y_lr = y // scale
        x_lr = x // scale
        p_lr = p // scale
        y_lr_end = min(y_lr + p_lr, obs_1x.shape[1])
        x_lr_end = min(x_lr + p_lr, obs_1x.shape[2])
        obs_1x_crop = obs_1x[:, y_lr:y_lr_end, x_lr:x_lr_end]

        drz_parts = [obs_drz_patch]
        if self.include_multiscale and obs_2x_up is not None:
            drz_parts.append(obs_2x_up[:, y_drz : y_drz + p_drz, x_drz : x_drz + p_drz])
        aug_rng = np.random.default_rng(self.seed + int(index) + self._epoch * len(self) + 8)

        if self.defer_1x_upsample:
            obs_hr_patch = np.concatenate(drz_parts, axis=0).astype(np.float32, copy=False)
            obs_hr_patch, obs_1x_crop, target_patch, edge_patch = self._augment_deferred(
                obs_hr_patch,
                obs_1x_crop,
                target_patch,
                edge_patch,
                aug_rng,
            )
            obs_drz_loss_patch = obs_hr_patch[:2]
            sample = {
                "obs_features_hr": torch.from_numpy(obs_hr_patch.astype(np.float32, copy=False)),
                "obs_features_1x_lr": torch.from_numpy(obs_1x_crop.astype(np.float32, copy=False)),
            }
        else:
            upsample_factor = self.drizzle_scale  # 1x → drizzle grid
            obs_1x_up_patch = zoom(obs_1x_crop, (1, upsample_factor, upsample_factor), order=1).astype(np.float32, copy=False)
            # Handle edge rounding: ensure exact drizzle patch size
            if obs_1x_up_patch.shape[1] != p_drz or obs_1x_up_patch.shape[2] != p_drz:
                obs_1x_up_patch = obs_1x_up_patch[:, :p_drz, :p_drz]
            parts = [*drz_parts, obs_1x_up_patch]
            obs_patch = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
            obs_patch, target_patch, edge_patch = self._augment(obs_patch, target_patch, edge_patch, aug_rng)
            obs_drz_loss_patch = obs_patch[:2]
            sample = {
                "obs_features": torch.from_numpy(obs_patch.astype(np.float32, copy=False)),
            }

        sample.update({
            "hr_target": torch.from_numpy(target_patch[None, :, :].astype(np.float32, copy=False)),
            "hr_edge": torch.from_numpy(edge_patch[None, :, :].astype(np.float32, copy=False)),
            "drizzle_mean": torch.from_numpy(obs_drz_loss_patch[0:1].astype(np.float32, copy=False)),
            "coverage": torch.from_numpy(obs_drz_loss_patch[1:2].astype(np.float32, copy=False)),
        })

        if self.return_metadata:
            metadata = scene["metadata"]
            augmentation = scene.get("augmentation", {})
            sample["metadata"] = {
                "scene_id": str(metadata.get("scene_id", self.scene_paths[scene_index].name)),
                "scene_dir": str(scene["scene_dir"]),
                "patch_y_4x": int(y),
                "patch_x_4x": int(x),
                "scale": int(metadata.get("scale", self.scale)),
                "burst_augmented": bool(augmentation.get("burst_augmented", False)),
                "burst_frames_kept": int(augmentation.get("burst_frames_kept", 0)),
            }
        return sample
