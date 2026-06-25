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

from tcforge.classical_sr import drizzle_features
from tcforge.reconstruct import reconstruct_hr_temperature
from tcforge.storage import load_scene_compact

from .mask_weights import compute_mask_loss_weights_np


HYBRID_DRIZZLE_MEAN_CHANNEL = 5




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
        input_mode: str = "lr",
        return_metadata: bool = True,
        thin_boost: float = 1.0,
        gap_boost: float = 1.0,
        loss_weight_max_width_px: int = 3,
        min_burst_frames: int = 30,
        shift_noise_std_px: float = 0.05,
        provide_burst: bool = False,
        solver_m_frames: int = 16,
        solver_no_drizzle: bool = False,
    ) -> None:
        if scale <= 0:
            raise ValueError("scale must be positive")
        if patches_per_scene <= 0:
            raise ValueError("patches_per_scene must be positive")
        if max_scene_cache <= 0:
            raise ValueError("max_scene_cache must be positive")
        if input_mode not in ("lr", "hybrid_drizzle2x"):
            raise ValueError(f"input_mode must be 'lr' or 'hybrid_drizzle2x', got {input_mode!r}")

        self.residual = bool(residual)
        self.input_mode = str(input_mode)
        self.data_scale = int(scale)
        self.min_burst_frames = int(min_burst_frames)
        self.shift_noise_std_px = float(shift_noise_std_px)
        self.provide_burst = bool(provide_burst)
        self.solver_m_frames = int(solver_m_frames)
        self.solver_no_drizzle = bool(solver_no_drizzle)
        if self.provide_burst and input_mode != "hybrid_drizzle2x":
            raise ValueError("provide_burst (unrolled solver) requires input_mode='hybrid_drizzle2x'")
        if self.input_mode == "hybrid_drizzle2x" and patch_size_hr % self.data_scale != 0:
            raise ValueError("hybrid_drizzle2x requires patch_size_hr divisible by the 2x data scale")

        effective_scale = 1 if (self.residual or self.input_mode == "hybrid_drizzle2x") else int(scale)
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
        self.thin_boost = float(thin_boost)
        self.gap_boost = float(gap_boost)
        self.loss_weight_max_width_px = int(loss_weight_max_width_px)
        self._need_loss_weights = self.thin_boost > 1.0 or self.gap_boost > 1.0
        self._cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
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

    def _select_burst(
        self,
        lr_burst: np.ndarray,
        shifts: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Randomly subsample burst frames and add shift noise for augmentation."""

        n_frames = lr_burst.shape[0]
        keep_frac = rng.uniform(0.6, 1.0)
        n_keep = max(self.min_burst_frames, int(round(n_frames * keep_frac)))
        n_keep = min(n_keep, n_frames)
        indices = rng.choice(n_frames, size=n_keep, replace=False)
        indices.sort()
        burst_sub = np.asarray(lr_burst[indices], dtype=np.float32)
        shifts_sub = shifts[indices].copy()
        if self.shift_noise_std_px > 0:
            shifts_sub += rng.normal(0, self.shift_noise_std_px, size=shifts_sub.shape).astype(np.float32)
        return burst_sub, shifts_sub

    def _build_hybrid_obs(
        self,
        obs_up: np.ndarray,
        entry: dict[str, Any],
        epoch: int,
        scene_index: int,
    ) -> np.ndarray:
        """Build 8-channel 2x observation: 5ch fused↑2x + 3ch drizzle@2x.

        Prefers precomputed ``drizzle_variants`` (one variant drawn per epoch,
        see ``scripts/precompute_drizzle_variants.py``); falls back to on-the-fly
        scatter drizzle from ``lr_burst`` for pools without variants.
        """

        rng = np.random.default_rng([self.seed, epoch, scene_index, 0xBEEF])
        variants = entry.get("_drz_variants")
        if variants is not None:
            k = int(rng.integers(0, variants.shape[0]))
            drz = np.asarray(variants[k], dtype=np.float32)
        else:
            burst_sub, shifts_sub = self._select_burst(entry["_lr_burst"], entry["_shifts"], rng)
            drz = drizzle_features(burst_sub, shifts_sub, scale=self.data_scale, kernel="bilinear")
        return np.concatenate([obs_up, drz], axis=0).astype(np.float32, copy=False)

    def _load_cached(self, scene_index: int) -> dict[str, Any]:
        cached = self._cache.get(scene_index)
        if cached is not None:
            self._cache.move_to_end(scene_index)
            if self.input_mode == "hybrid_drizzle2x" and not self.solver_no_drizzle:
                epoch = self._epoch
                if cached.get("_hybrid_epoch") != epoch:
                    cached["obs_features"] = self._build_hybrid_obs(
                        cached["_obs_up"], cached, epoch, scene_index,
                    )
                    cached["_hybrid_epoch"] = epoch
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

        hr_mask = np.asarray(scene["hr_mask"], dtype=np.float32)

        if self.input_mode == "hybrid_drizzle2x":
            from scipy.ndimage import zoom as scipy_zoom

            obs_up = scipy_zoom(obs, (1, self.data_scale, self.data_scale), order=1).astype(np.float32)
            packed = {
                "scene_dir": scene["scene_dir"],
                "metadata": metadata,
                "hr_target": hr_target,
                "hr_edge": hr_edge,
                "hr_mask": hr_mask,
                "_obs_1x": obs,
                "_obs_up": obs_up,
            }
            epoch = self._epoch
            if self.solver_no_drizzle:
                # Lean solver input: 5ch upsampled fused only. The DC term (raw burst) carries the
                # multi-frame SR signal, so no drizzle is built — skips BOTH the on-the-fly scatter
                # cost AND the precomputed-variants disk. x0 = upsampled aligned_mean (obs ch0).
                packed["obs_features"] = obs_up
            else:
                variants = scene.get("drizzle_variants")
                if variants is not None:
                    # float16 mmap, sliced per epoch — never fully materialised.
                    packed["_drz_variants"] = variants
                else:
                    lr_burst = scene.get("lr_burst")
                    if lr_burst is None:
                        raise ValueError(
                            f"scene {scene['scene_dir']} has no drizzle_variants and no lr_burst; "
                            "hybrid_drizzle2x requires precomputed variants "
                            "(scripts/precompute_drizzle_variants.py) or save_lr_burst=true"
                        )
                    # Keep the float16 mmap as-is: _select_burst casts only the
                    # sampled subset to float32, avoiding ~305 MB/scene RAM.
                    packed["_lr_burst"] = lr_burst
                    packed["_shifts"] = np.asarray(scene["shifts"], dtype=np.float32)
                packed["obs_features"] = self._build_hybrid_obs(obs_up, packed, epoch, scene_index)
                packed["_hybrid_epoch"] = epoch
            if self.provide_burst and "_lr_burst" not in packed:
                lr_burst = scene.get("lr_burst")
                if lr_burst is None:
                    raise ValueError(
                        f"scene {scene['scene_dir']} has no lr_burst; provide_burst requires save_lr_burst=true"
                    )
                packed["_lr_burst"] = lr_burst
                packed["_shifts"] = np.asarray(scene["shifts"], dtype=np.float32)
        elif self.residual:
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
            packed = {
                "scene_dir": scene["scene_dir"],
                "metadata": metadata,
                "obs_features": obs,
                "hr_target": hr_target,
                "hr_edge": hr_edge,
                "hr_mask": hr_mask,
            }
        else:
            packed = {
                "scene_dir": scene["scene_dir"],
                "metadata": metadata,
                "obs_features": obs,
                "hr_target": hr_target,
                "hr_edge": hr_edge,
                "hr_mask": hr_mask,
            }

        if packed["hr_target"].shape != (packed["obs_features"].shape[1] * self.scale, packed["obs_features"].shape[2] * self.scale):
            raise ValueError(
                f"scene {scene['scene_dir']} has incompatible obs/target shapes: "
                f"{packed['obs_features'].shape} -> {packed['hr_target'].shape} at effective scale {self.scale}"
            )
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
        if self.input_mode == "hybrid_drizzle2x" and self.data_scale > 1:
            y_lr = (y_lr // self.data_scale) * self.data_scale
            x_lr = (x_lr // self.data_scale) * self.data_scale
        return y_lr, x_lr

    def _augment(
        self,
        obs: np.ndarray,
        target: np.ndarray,
        edge: np.ndarray,
        mask: np.ndarray,
        lr_obs: np.ndarray | None,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Random flip + 90-degree rotation for C,H,W obs and H,W target/edge/mask."""
        if rng.random() < 0.5:
            obs = obs[:, :, ::-1].copy()
            if lr_obs is not None:
                lr_obs = lr_obs[:, :, ::-1].copy()
            target = target[:, ::-1].copy()
            edge = edge[:, ::-1].copy()
            mask = mask[:, ::-1].copy()
        if rng.random() < 0.5:
            obs = obs[:, ::-1, :].copy()
            if lr_obs is not None:
                lr_obs = lr_obs[:, ::-1, :].copy()
            target = target[::-1, :].copy()
            edge = edge[::-1, :].copy()
            mask = mask[::-1, :].copy()
        k = int(rng.integers(0, 4))
        if k > 0:
            obs = np.rot90(obs, k, axes=(1, 2)).copy()
            if lr_obs is not None:
                lr_obs = np.rot90(lr_obs, k, axes=(1, 2)).copy()
            target = np.rot90(target, k, axes=(0, 1)).copy()
            edge = np.rot90(edge, k, axes=(0, 1)).copy()
            mask = np.rot90(mask, k, axes=(0, 1)).copy()
        return obs, target, edge, mask, lr_obs

    def _add_burst_to_sample(
        self, sample: dict[str, Any], scene: dict[str, Any], index: int,
        y_lr: int, x_lr: int, p_hr: int,
    ) -> None:
        """Crop M burst frames at the SAME scale-aligned LR ROI as the HR patch, with EXACT
        shifts (no noise) + per-scene PSF, for the unrolled solver's data-consistency term.
        Scale-aligned integer crop => forward_burst(HR_patch) == burst_patch in the interior
        (validated; the DC term masks an ~8 LR-px rim). M is fixed (collation needs constant M;
        every v3 scene has >= 24 frames)."""
        burst = scene["_lr_burst"]
        sc_shifts = scene["_shifts"]
        n = int(burst.shape[0])
        m = min(self.solver_m_frames, n)
        rng = np.random.default_rng(self.seed + int(index) + self._epoch * len(self) + 17)
        sel = np.sort(rng.choice(n, size=m, replace=False))
        ds = self.data_scale
        y1, x1, p1 = y_lr // ds, x_lr // ds, p_hr // ds
        burst_patch = np.asarray(burst[sel][:, y1 : y1 + p1, x1 : x1 + p1], dtype=np.float32)
        if burst_patch.shape != (m, p1, p1):
            raise ValueError(f"burst patch shape {burst_patch.shape} != {(m, p1, p1)}")
        md = scene["metadata"]
        sy = md.get("psf_sigma_y_lr_px")
        sample["lr_burst_patch"] = torch.from_numpy(burst_patch)
        sample["burst_shifts"] = torch.from_numpy(np.asarray(sc_shifts[sel], dtype=np.float32))
        sample["psf_sigma_lr_px"] = float(md["psf_sigma_lr_px"])
        sample["psf_sigma_y_lr_px"] = float(sy) if sy is not None else float(md["psf_sigma_lr_px"])
        sample["psf_angle_deg"] = float(md.get("psf_angle_deg", 0.0))
        sample["psf_shape"] = str(md.get("psf_shape", "gaussian"))

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
        lr_obs_patch: np.ndarray | None = None
        if self.input_mode == "hybrid_drizzle2x":
            y_1x = y_lr // self.data_scale
            x_1x = x_lr // self.data_scale
            p_1x = p_hr // self.data_scale
            lr_obs_patch = scene["_obs_1x"][0:1, y_1x : y_1x + p_1x, x_1x : x_1x + p_1x]
            if lr_obs_patch.shape != (1, p_1x, p_1x):
                raise ValueError(
                    f"lr_obs crop shape mismatch: got {lr_obs_patch.shape}, expected {(1, p_1x, p_1x)}"
                )
        # The unrolled solver consumes the raw burst + EXACT shifts, so it must NOT see flipped/
        # rotated patches (the v3 generator already randomizes orientation + constellation, and
        # augmenting here would require transforming the shift vectors — an FM-6 footgun we skip).
        if not self.provide_burst:
            aug_rng = np.random.default_rng(self.seed + int(index) + self._epoch * len(self) + 8)
            obs_patch, target_patch, edge_patch, mask_patch, lr_obs_patch = self._augment(
                obs_patch, target_patch, edge_patch, mask_patch, lr_obs_patch, aug_rng,
            )

        sample = {
            "obs_features": torch.from_numpy(obs_patch.astype(np.float32, copy=False)),
            "hr_target": torch.from_numpy(target_patch[None, :, :].astype(np.float32, copy=False)),
            "hr_edge": torch.from_numpy(edge_patch[None, :, :].astype(np.float32, copy=False)),
            "hr_mask": torch.from_numpy(mask_patch[None, :, :].astype(np.float32, copy=False)),
        }
        if lr_obs_patch is not None:
            sample["lr_obs"] = torch.from_numpy(lr_obs_patch.astype(np.float32, copy=False))
        if self.provide_burst:
            self._add_burst_to_sample(sample, scene, index, y_lr, x_lr, p_hr)

        if self._need_loss_weights:
            thin_np, gap_np = compute_mask_loss_weights_np(
                mask_patch,
                thin_boost=self.thin_boost,
                gap_boost=self.gap_boost,
                max_width_px=self.loss_weight_max_width_px,
            )
            if thin_np is not None:
                sample["thin_weight"] = torch.from_numpy(thin_np.astype(np.float32, copy=False))
            if gap_np is not None:
                sample["gap_weight"] = torch.from_numpy(gap_np.astype(np.float32, copy=False))

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
