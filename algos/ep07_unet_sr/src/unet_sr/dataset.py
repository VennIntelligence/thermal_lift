"""Dataset for compact TCForge training scenes."""

from __future__ import annotations

import csv
import multiprocessing as mp
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from tcforge.reconstruct import reconstruct_hr_temperature
from tcforge.storage import load_scene_compact





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
        raise FileNotFoundError(f"no manifest.csv or compact scene directories found under {root}")
    return paths


def _metadata_float(metadata: dict[str, Any], key: str, default: float) -> float:
    return float(metadata.get(key, default))


def _reconstruct_target(scene: dict[str, Any]) -> np.ndarray:
    metadata = scene["metadata"]
    seed = metadata.get("low_freq_seed", metadata.get("seed"))
    return reconstruct_hr_temperature(
        scene["hr_mask"],
        T_bg_c=_metadata_float(metadata, "T_bg_c", 21.0),
        delta_T_c=_metadata_float(metadata, "delta_T_c", 2.0),
        low_freq_amplitude_c=_metadata_float(metadata, "low_freq_amplitude_c", 0.2),
        low_freq_sigma_px=_metadata_float(metadata, "low_freq_sigma_px", 96.0),
        seed=None if seed is None else int(seed),
    ).astype(np.float32, copy=False)


class SceneInterleavedSampler(Sampler[int]):
    """Cache-friendly sampling with batch-level scene diversity.

    Divides scenes into buckets of ``scenes_per_bucket`` scenes.  Within each
    bucket, patches are emitted in round-robin order across scenes: each scene
    contributes ``patches_per_fetch`` consecutive patches before the sampler
    moves to the next scene in the bucket.  This means:

    * **Cache-friendly**: a DataLoader worker only needs to keep
      ``scenes_per_bucket`` scenes in its LRU cache at once.
    * **Diverse batches**: every batch naturally spans multiple scenes,
      keeping gradient variance healthy.

    Supports DDP by partitioning scenes across ranks.
    """

    def __init__(
        self,
        n_scenes: int,
        patches_per_scene: int,
        *,
        scenes_per_bucket: int = 16,
        patches_per_fetch: int = 8,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        num_workers: int = 0,
        batch_size: int = 1,
    ) -> None:
        super().__init__()
        self.n_scenes = int(n_scenes)
        self.patches_per_scene = int(patches_per_scene)
        self.scenes_per_bucket = int(scenes_per_bucket)
        self.patches_per_fetch = int(patches_per_fetch)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.num_workers = max(1, int(num_workers)) if num_workers > 0 else 1
        self.batch_size = max(1, int(batch_size))
        self._epoch: int = 0

        # Pad scene count to be evenly divisible by world_size.
        if self.world_size > 1:
            self._n_scenes_padded = (
                (self.n_scenes + self.world_size - 1) // self.world_size
            ) * self.world_size
        else:
            self._n_scenes_padded = self.n_scenes

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for deterministic cross-epoch shuffling."""
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

        # DDP: pad to equal size per rank, then slice.
        if self.world_size > 1:
            if len(scene_order) < self._n_scenes_padded:
                extra = scene_order[: self._n_scenes_padded - len(scene_order)]
                scene_order = np.concatenate([scene_order, extra])
            per_rank = self._n_scenes_padded // self.world_size
            scene_order = scene_order[
                self.rank * per_rank : (self.rank + 1) * per_rank
            ]

        W = self.num_workers
        if W <= 1:
            # Original behaviour: single-process or 1 worker.
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
        if self.world_size > 1:
            per_rank = self._n_scenes_padded // self.world_size
            return per_rank * self.patches_per_scene
        return self.n_scenes * self.patches_per_scene


class ThermalSRDataset(Dataset[dict[str, Any]]):
    """Random HR-aligned patch dataset over compact TCForge scenes.

    The compact pool stores only binary HR geometry and 1x fused observation
    features. HR temperature targets are reconstructed from scene metadata so
    the dataset stays compatible with Task 3's disk contract.
    """

    def __init__(
        self,
        training_pool_dir: str | Path,
        *,
        patch_size_hr: int = 256,
        scale: int = 4,
        seed: int = 42,
        patches_per_scene: int = 64,
        max_scene_cache: int = 4,
        residual: bool = False,
        return_metadata: bool = True,
    ) -> None:
        if scale <= 0:
            raise ValueError("scale must be positive")
        if patches_per_scene <= 0:
            raise ValueError("patches_per_scene must be positive")
        if max_scene_cache <= 0:
            raise ValueError("max_scene_cache must be positive")

        self.residual = bool(residual)
        self.data_scale = int(scale)

        # In residual mode, input and output are both at HR resolution
        effective_scale = 1 if self.residual else int(scale)
        if patch_size_hr <= 0 or patch_size_hr % effective_scale != 0:
            raise ValueError("patch_size_hr must be positive and divisible by effective scale")

        self.scene_paths = _scene_paths(training_pool_dir)
        self.patch_size_hr = int(patch_size_hr)
        self.patch_size_lr = int(patch_size_hr // effective_scale)
        self.scale = effective_scale
        self.seed = int(seed)
        self.patches_per_scene = int(patches_per_scene)
        self.max_scene_cache = int(max_scene_cache)
        self.return_metadata = bool(return_metadata)
        self._cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        # Shared epoch counter: survives fork into persistent DataLoader workers.
        # Main process writes via set_epoch(); workers read in __getitem__.
        self._shared_epoch = mp.Value("i", 0)

    def __len__(self) -> int:
        return len(self.scene_paths) * self.patches_per_scene

    def set_epoch(self, epoch: int) -> None:
        """Called by the training loop to diversify random crops by epoch.

        Uses a multiprocessing.Value so that persistent DataLoader workers
        (which hold forked copies of this dataset) see the updated epoch
        when computing crop positions and augmentation seeds.
        """
        self._shared_epoch.value = int(epoch)

    @property
    def _epoch(self) -> int:
        """Current epoch, readable from any worker process."""
        return self._shared_epoch.value

    def _load_cached(self, scene_index: int) -> dict[str, Any]:
        cached = self._cache.get(scene_index)
        if cached is not None:
            self._cache.move_to_end(scene_index)
            return cached

        scene = load_scene_compact(self.scene_paths[scene_index])
        metadata = scene["metadata"]
        scene_scale = int(metadata.get("scale", self.data_scale))
        if scene_scale != self.data_scale:
            raise ValueError(f"scene {scene['scene_dir']} scale={scene_scale}, expected {self.data_scale}")

        obs = np.asarray(scene["obs_features"], dtype=np.float32)
        hr_target = _reconstruct_target(scene)
        hr_edge = np.asarray(scene["hr_edge"], dtype=np.float32)
        if obs.ndim != 3:
            raise ValueError(f"obs_features must have shape (C,H,W), got {obs.shape}")
        if hr_target.shape != hr_edge.shape:
            raise ValueError(f"hr_target/hr_edge shape mismatch: {hr_target.shape} vs {hr_edge.shape}")

        if self.residual:
            # Upsample obs features from LR to HR, concat classical SR
            from scipy.ndimage import zoom
            obs_hr = zoom(obs, (1, self.data_scale, self.data_scale), order=1).astype(np.float32)
            classical_sr = scene.get("classical_sr")
            if classical_sr is None:
                raise ValueError(
                    f"scene {scene['scene_dir']} has no classical_sr; "
                    "residual mode requires compute_classical_sr=true in pool config"
                )
            classical_sr = np.asarray(classical_sr, dtype=np.float32)
            obs = np.concatenate([obs_hr, classical_sr[None, :, :]], axis=0)

        if hr_target.shape != (obs.shape[1] * self.scale, obs.shape[2] * self.scale):
            raise ValueError(
                f"scene {scene['scene_dir']} has incompatible obs/target shapes: "
                f"{obs.shape} -> {hr_target.shape} at effective scale {self.scale}"
            )
        hr_mask = np.asarray(scene["hr_mask"], dtype=np.float32)
        packed = {
            "scene_dir": scene["scene_dir"],
            "metadata": metadata,
            "obs_features": obs,
            "hr_target": hr_target,
            "hr_edge": hr_edge,
            "hr_mask": hr_mask,
        }
        self._cache[scene_index] = packed
        self._cache.move_to_end(scene_index)
        while len(self._cache) > self.max_scene_cache:
            self._cache.popitem(last=False)
        return packed

    def _crop_origin_lr(self, index: int, lr_shape: tuple[int, int]) -> tuple[int, int]:
        rows, cols = lr_shape
        if self.patch_size_lr > rows or self.patch_size_lr > cols:
            raise ValueError(
                f"patch_size_hr={self.patch_size_hr} requires LR patch {self.patch_size_lr}, "
                f"larger than scene LR shape {lr_shape}"
            )
        max_y = rows - self.patch_size_lr
        max_x = cols - self.patch_size_lr
        rng = np.random.default_rng(self.seed + int(index) + self._epoch * len(self))
        y_lr = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
        x_lr = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0
        return y_lr, x_lr

    def _augment(
        self,
        obs: np.ndarray,
        target: np.ndarray,
        edge: np.ndarray,
        mask: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Random flip + 90-degree rotation for C,H,W obs and H,W target/edge/mask."""
        if rng.random() < 0.5:
            obs = obs[:, :, ::-1].copy()
            target = target[:, ::-1].copy()
            edge = edge[:, ::-1].copy()
            mask = mask[:, ::-1].copy()
        if rng.random() < 0.5:
            obs = obs[:, ::-1, :].copy()
            target = target[::-1, :].copy()
            edge = edge[::-1, :].copy()
            mask = mask[::-1, :].copy()
        k = int(rng.integers(0, 4))
        if k > 0:
            obs = np.rot90(obs, k, axes=(1, 2)).copy()
            target = np.rot90(target, k, axes=(0, 1)).copy()
            edge = np.rot90(edge, k, axes=(0, 1)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()
        return obs, target, edge, mask

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index = len(self) + index
        if index < 0 or index >= len(self):
            raise IndexError(index)

        scene_index = index // self.patches_per_scene
        scene = self._load_cached(scene_index)
        obs = scene["obs_features"]
        target = scene["hr_target"]
        edge = scene["hr_edge"]
        mask = scene["hr_mask"]

        y_lr, x_lr = self._crop_origin_lr(index, tuple(map(int, obs.shape[1:])))
        y_hr = y_lr * self.scale
        x_hr = x_lr * self.scale
        p_lr = self.patch_size_lr
        p_hr = self.patch_size_hr

        obs_patch = obs[:, y_lr : y_lr + p_lr, x_lr : x_lr + p_lr]
        target_patch = target[y_hr : y_hr + p_hr, x_hr : x_hr + p_hr]
        edge_patch = edge[y_hr : y_hr + p_hr, x_hr : x_hr + p_hr]
        mask_patch = mask[y_hr : y_hr + p_hr, x_hr : x_hr + p_hr]
        aug_rng = np.random.default_rng(self.seed + int(index) + self._epoch * len(self) + 8)
        obs_patch, target_patch, edge_patch, mask_patch = self._augment(
            obs_patch, target_patch, edge_patch, mask_patch, aug_rng,
        )

        sample = {
            "obs_features": torch.from_numpy(obs_patch.astype(np.float32, copy=False)),
            "hr_target": torch.from_numpy(target_patch[None, :, :].astype(np.float32, copy=False)),
            "hr_edge": torch.from_numpy(edge_patch[None, :, :].astype(np.float32, copy=False)),
            "hr_mask": torch.from_numpy(mask_patch[None, :, :].astype(np.float32, copy=False)),
        }

        if self.return_metadata:
            metadata = scene["metadata"]
            sample["metadata"] = {
                "scene_id": str(metadata.get("scene_id", self.scene_paths[scene_index].name)),
                "scene_dir": str(scene["scene_dir"]),
                "patch_y_hr": int(y_hr),
                "patch_x_hr": int(x_hr),
                "scale": int(metadata.get("scale", self.scale)),
            }
        return sample
