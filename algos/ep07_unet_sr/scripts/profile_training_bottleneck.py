#!/usr/bin/env python3
"""Profile EP07 UNet training bottlenecks: data loading vs GPU compute."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from tcforge.storage import load_scene_compact
from unet_sr.dataset import ThermalSRDataset, _reconstruct_target
from unet_sr.losses import ThermalSRLoss
from unet_sr.model import ThermalSRUNet


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000.0:8.1f} ms"


def _fmt_gb(bytes_val: int) -> str:
    return f"{bytes_val / (1024 ** 3):.2f} GB"


def _gpu_mem(device: torch.device) -> dict[str, int]:
    if device.type != "cuda":
        return {}
    idx = device.index if device.index is not None else torch.cuda.current_device()
    return {
        "allocated": torch.cuda.memory_allocated(idx),
        "reserved": torch.cuda.memory_reserved(idx),
        "max_allocated": torch.cuda.max_memory_allocated(idx),
    }


def _profile_scene_load(scene_path: Path, *, residual: bool, scale: int) -> dict[str, float]:
    t0 = time.perf_counter()
    scene = load_scene_compact(scene_path)
    t_load = time.perf_counter()

    hr_target = _reconstruct_target(scene)
    t_recon = time.perf_counter()

    obs = np.asarray(scene["obs_features"], dtype=np.float32)
    if residual:
        from scipy.ndimage import zoom

        obs_hr = zoom(obs, (1, scale, scale), order=1).astype(np.float32)
        classical_sr = np.asarray(scene["classical_sr"], dtype=np.float32)
        obs = np.concatenate([obs_hr, classical_sr[None, :, :]], axis=0)
    t_residual = time.perf_counter()

    hr_edge = np.asarray(scene["hr_edge"], dtype=np.float32)
    bytes_per_scene = obs.nbytes + hr_target.nbytes + hr_edge.nbytes
    if residual and "classical_sr" in scene:
        bytes_per_scene += np.asarray(scene["classical_sr"]).nbytes

    return {
        "load_disk_ms": (t_load - t0) * 1000.0,
        "reconstruct_hr_ms": (t_recon - t_load) * 1000.0,
        "residual_prep_ms": (t_residual - t_recon) * 1000.0,
        "total_ms": (t_residual - t0) * 1000.0,
        "bytes_per_scene": float(bytes_per_scene),
        "obs_shape": obs.shape,
        "hr_shape": hr_target.shape,
    }


def _profile_dataset(
    dataset: ThermalSRDataset,
    *,
    n_samples: int,
    shuffle_indices: bool,
) -> dict[str, float | int]:
    indices = list(range(min(n_samples, len(dataset))))
    if shuffle_indices:
        rng = np.random.default_rng(42)
        rng.shuffle(indices)

    cold_times: list[float] = []
    warm_times: list[float] = []
    dataset._cache.clear()

    for i, idx in enumerate(indices):
        t0 = time.perf_counter()
        dataset[idx]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if i < 20 or len(dataset._cache) < dataset.max_scene_cache:
            cold_times.append(elapsed_ms)
        else:
            warm_times.append(elapsed_ms)

    return {
        "n_samples": len(indices),
        "cold_mean_ms": statistics.mean(cold_times) if cold_times else 0.0,
        "cold_p95_ms": float(np.percentile(cold_times, 95)) if cold_times else 0.0,
        "warm_mean_ms": statistics.mean(warm_times) if warm_times else 0.0,
        "warm_p95_ms": float(np.percentile(warm_times, 95)) if warm_times else 0.0,
        "cache_size": len(dataset._cache),
        "max_scene_cache": dataset.max_scene_cache,
    }


def _profile_dataloader(
    dataset: ThermalSRDataset,
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    device: torch.device,
    n_batches: int,
) -> dict[str, float]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    batch_times: list[float] = []
    h2d_times: list[float] = []
    iterator = iter(loader)
    for _ in range(n_batches):
        t0 = time.perf_counter()
        batch = next(iterator)
        t_batch = time.perf_counter()

        if device.type == "cuda":
            t_h2d0 = time.perf_counter()
            _ = batch["obs_features"].to(device=device, non_blocking=True)
            _ = batch["hr_target"].to(device=device, non_blocking=True)
            _ = batch["hr_edge"].to(device=device, non_blocking=True)
            torch.cuda.synchronize(device)
            h2d_times.append((time.perf_counter() - t_h2d0) * 1000.0)

        batch_times.append((t_batch - t0) * 1000.0)

    return {
        "batch_mean_ms": statistics.mean(batch_times),
        "batch_p95_ms": float(np.percentile(batch_times, 95)),
        "h2d_mean_ms": statistics.mean(h2d_times) if h2d_times else 0.0,
        "throughput_samples_per_s": batch_size * 1000.0 / statistics.mean(batch_times),
    }


def _make_batch(device: torch.device, *, batch_size: int, in_channels: int, patch_size_hr: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    obs = torch.randn(batch_size, in_channels, patch_size_hr, patch_size_hr, device=device, dtype=torch.float32)
    target = torch.randn(batch_size, 1, patch_size_hr, patch_size_hr, device=device, dtype=torch.float32)
    edge = torch.rand(batch_size, 1, patch_size_hr, patch_size_hr, device=device, dtype=torch.float32)
    return obs, target, edge


def _profile_gpu_step(
    *,
    device: torch.device,
    batch_size: int,
    in_channels: int,
    patch_size_hr: int,
    residual: bool,
    use_amp: bool,
    use_compile: bool,
    n_warmup: int,
    n_timed: int,
) -> dict[str, float | bool | dict[str, int]]:
    model = ThermalSRUNet(in_channels=in_channels, out_channels=1, base_channels=48, scale=1 if residual else 2).to(device)
    compiled = False
    if use_compile and hasattr(torch, "compile"):
        model = torch.compile(model)
        compiled = True

    criterion = ThermalSRLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scaler = GradScaler(enabled=use_amp and device.type == "cuda")
    obs, target, edge = _make_batch(device, batch_size=batch_size, in_channels=in_channels, patch_size_hr=patch_size_hr)

    def one_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp and device.type == "cuda"):
            pred = model(obs)
            if residual:
                pred = obs[:, -1:, :, :] + pred
            losses = criterion(pred, target, edge_mask=edge)
            total = losses["total"]
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    for _ in range(n_warmup):
        one_step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    step_times: list[float] = []
    for _ in range(n_timed):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        one_step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_times.append((time.perf_counter() - t0) * 1000.0)

    mem = _gpu_mem(device)
    return {
        "step_mean_ms": statistics.mean(step_times),
        "step_p95_ms": float(np.percentile(step_times, 95)),
        "steps_per_s": 1000.0 / statistics.mean(step_times),
        "compile_enabled": compiled,
        "gpu_mem": mem,
    }


def _estimate_worker_ram(
    *,
    bytes_per_scene: float,
    max_scene_cache: int,
    num_workers: int,
) -> float:
    # Each DataLoader worker is a separate process with its own LRU cache.
    return bytes_per_scene * max_scene_cache * max(1, num_workers)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile EP07 UNet training bottlenecks.")
    parser.add_argument("--training-pool-dir", default="../../data/synthetic/training_pool_2x")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-scene-cache", type=int, default=50)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--patch-size-hr", type=int, default=256)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--residual", action="store_true", default=True)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--n-scene-load", type=int, default=5)
    parser.add_argument("--n-dataset-samples", type=int, default=80)
    parser.add_argument("--n-dataloader-batches", type=int, default=20)
    parser.add_argument("--n-gpu-warmup", type=int, default=5)
    parser.add_argument("--n-gpu-timed", type=int, default=20)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        index = 0 if device.index is None else int(device.index)
        torch.cuda.set_device(index)
        device = torch.device(f"cuda:{index}")
        print(f"Using GPU {index}: {torch.cuda.get_device_name(index)}")

    pool_dir = Path(args.training_pool_dir).expanduser().resolve()
    scene_paths = sorted(p for p in pool_dir.iterdir() if p.is_dir() and (p / "metadata.json").exists())
    if not scene_paths:
        raise FileNotFoundError(f"No scenes under {pool_dir}")

    in_channels = 6 if args.residual else 5
    print("\n=== Training config (mirrors your CLI) ===")
    print(f"pool={pool_dir}")
    print(f"batch_size={args.batch_size}, num_workers={args.num_workers}, max_scene_cache={args.max_scene_cache}")
    print(f"prefetch_factor={args.prefetch_factor}, patch_size_hr={args.patch_size_hr}, residual={args.residual}")
    print(f"amp={args.amp}, compile={args.compile}")

    print("\n=== 1) Per-scene CPU cost (disk + HR reconstruct + residual prep) ===")
    scene_stats = [_profile_scene_load(scene_paths[i], residual=args.residual, scale=args.scale) for i in range(args.n_scene_load)]
    load_ms = statistics.mean(s["load_disk_ms"] for s in scene_stats)
    recon_ms = statistics.mean(s["reconstruct_hr_ms"] for s in scene_stats)
    residual_ms = statistics.mean(s["residual_prep_ms"] for s in scene_stats)
    total_ms = statistics.mean(s["total_ms"] for s in scene_stats)
    bytes_per_scene = statistics.mean(s["bytes_per_scene"] for s in scene_stats)
    print(f"  load_scene_compact      {_fmt_ms(load_ms / 1000.0)}")
    print(f"  reconstruct_hr_temp     {_fmt_ms(recon_ms / 1000.0)}")
    print(f"  residual zoom+concat    {_fmt_ms(residual_ms / 1000.0)}")
    print(f"  total per cold scene    {_fmt_ms(total_ms / 1000.0)}")
    print(f"  cached scene RAM        {_fmt_gb(int(bytes_per_scene))}  shape obs={scene_stats[0]['obs_shape']} hr={scene_stats[0]['hr_shape']}")
    worker_ram = _estimate_worker_ram(
        bytes_per_scene=bytes_per_scene,
        max_scene_cache=args.max_scene_cache,
        num_workers=args.num_workers,
    )
    print(f"  est. worker cache RAM   {_fmt_gb(int(worker_ram))}  ({args.num_workers} workers × cache {args.max_scene_cache})")

    dataset = ThermalSRDataset(
        pool_dir,
        patch_size_hr=args.patch_size_hr,
        scale=args.scale,
        residual=args.residual,
        patches_per_scene=64,
        max_scene_cache=args.max_scene_cache,
    )

    print("\n=== 2) Dataset __getitem__ (single process, cache cold vs warm) ===")
    ds_stats = _profile_dataset(dataset, n_samples=args.n_dataset_samples, shuffle_indices=True)
    print(f"  cold mean/p95           {_fmt_ms(ds_stats['cold_mean_ms'] / 1000.0)} / {_fmt_ms(ds_stats['cold_p95_ms'] / 1000.0)}")
    print(f"  warm mean/p95           {_fmt_ms(ds_stats['warm_mean_ms'] / 1000.0)} / {_fmt_ms(ds_stats['warm_p95_ms'] / 1000.0)}")
    print(f"  cache                   {ds_stats['cache_size']}/{ds_stats['max_scene_cache']}")

    print("\n=== 3) DataLoader end-to-end (workers + batch collate) ===")
    dl_stats = _profile_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        device=device,
        n_batches=args.n_dataloader_batches,
    )
    print(f"  batch wait mean/p95     {_fmt_ms(dl_stats['batch_mean_ms'] / 1000.0)} / {_fmt_ms(dl_stats['batch_p95_ms'] / 1000.0)}")
    print(f"  H2D transfer mean       {_fmt_ms(dl_stats['h2d_mean_ms'] / 1000.0)}")
    print(f"  throughput              {dl_stats['throughput_samples_per_s']:.1f} samples/s")

    print("\n=== 4) GPU train step (forward + loss + backward + optim) ===")
    gpu_stats = _profile_gpu_step(
        device=device,
        batch_size=args.batch_size,
        in_channels=in_channels,
        patch_size_hr=args.patch_size_hr,
        residual=args.residual,
        use_amp=args.amp,
        use_compile=args.compile,
        n_warmup=args.n_gpu_warmup,
        n_timed=args.n_gpu_timed,
    )
    print(f"  step mean/p95           {_fmt_ms(gpu_stats['step_mean_ms'] / 1000.0)} / {_fmt_ms(gpu_stats['step_p95_ms'] / 1000.0)}")
    print(f"  GPU steps/s             {gpu_stats['steps_per_s']:.2f}")
    print(f"  torch.compile           {gpu_stats['compile_enabled']}")
    mem = gpu_stats["gpu_mem"]
    if mem:
        print(f"  VRAM allocated/reserved {_fmt_gb(mem['allocated'])} / {_fmt_gb(mem['reserved'])}")
        print(f"  VRAM peak allocated     {_fmt_gb(mem['max_allocated'])}")

    batch_s = dl_stats["batch_mean_ms"] / 1000.0
    gpu_s = gpu_stats["step_mean_ms"] / 1000.0
    data_bound_s = max(0.0, batch_s - gpu_s)
    total_est_s = max(batch_s, gpu_s)

    print("\n=== 5) Diagnosis summary ===")
    print(f"  Est. step time (data-bound)   {_fmt_ms(total_est_s)}  (limited by slower of data vs GPU)")
    print(f"  DataLoader batch time         {_fmt_ms(batch_s)}")
    print(f"  GPU compute time              {_fmt_ms(gpu_s)}")
    if batch_s > gpu_s * 1.2:
        bottleneck = "CPU DataLoader / scene cache misses"
    elif gpu_s > batch_s * 1.2:
        bottleneck = "GPU compute (model + SSIM/edge loss)"
    else:
        bottleneck = "balanced (minor overlap between data and compute)"
    print(f"  Primary bottleneck            {bottleneck}")

    theoretical_util = min(100.0, 100.0 * gpu_s / total_est_s) if total_est_s > 0 else 0.0
    print(f"  Implied GPU duty cycle        ~{theoretical_util:.0f}%  (explains low util if data-bound)")

    if batch_s > gpu_s:
        needed_workers = max(1, int(np.ceil(batch_s / max(gpu_s, 1e-6))))
        print(f"  Suggested num_workers (rough) >= {needed_workers} to hide GPU latency")
        print("  Quick wins to try:")
        print("    - lower max_scene_cache (e.g. 8-16) to cut worker RAM / cache thrash")
        print("    - precompute hr_target + residual obs at pool generation time")
        print("    - reduce batch_size if VRAM-bound (frees memory for more prefetch)")
        print("    - disable --compile for first N steps (compile stalls look like idle GPU)")


if __name__ == "__main__":
    main()
