from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from sr4x.config import TrainingConfig
from sr4x.train import train
from tcforge.storage import _write_png_gray8


def _make_scene(root: Path) -> Path:
    scene = root / "scene_0000"
    scene.mkdir(parents=True)
    hr_h, hr_w = 32, 32
    lr_h, lr_w = 8, 8
    yy, xx = np.mgrid[:hr_h, :hr_w]
    mask = ((yy > 8) & (yy < 24) & (xx > 10) & (xx < 22)).astype(np.uint8)
    edge = np.zeros_like(mask)
    edge[8:24, 10] = 1
    edge[8:24, 21] = 1
    edge[8, 10:22] = 1
    edge[23, 10:22] = 1
    target_like = 21.0 + 2.0 * mask.astype(np.float32)
    obs_1x = np.zeros((5, lr_h, lr_w), dtype=np.float32)
    obs_1x[0] = target_like.reshape(lr_h, 4, lr_w, 4).mean(axis=(1, 3))
    obs_1x[1] = obs_1x[0]
    obs_1x[2] = 1.0
    obs_1x[3] = 0.01
    obs_1x[4] = 0.0

    rng = np.random.default_rng(0)
    lr_base = obs_1x[0].astype(np.float32)
    lr_burst = np.stack([lr_base + 0.01 * rng.normal(size=lr_base.shape).astype(np.float32) for _ in range(4)])
    shifts = np.asarray([[0.0, 0.0], [0.25, 0.0], [0.0, 0.25], [0.25, 0.25]], dtype=np.float32)

    np.savez_compressed(scene / "obs_features_1x.npz", obs_features=obs_1x)
    np.save(scene / "lr_burst.npy", lr_burst.astype(np.float16))
    np.save(scene / "shifts.npy", shifts)
    _write_png_gray8(scene / "hr_mask_4x.png", mask * np.uint8(255))
    _write_png_gray8(scene / "hr_edge_4x.png", edge * np.uint8(255))
    (scene / "metadata.json").write_text(
        '{"scene_id": "smoke", "scale": 4, "T_bg_c": 21.0, "delta_T_c": 2.0, "low_freq_amplitude_c": 0.0}',
        encoding="utf-8",
    )
    with (root / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scene_dir"])
        writer.writeheader()
        writer.writerow({"scene_dir": "scene_0000"})
    return scene


def test_train_cpu_smoke(tmp_path: Path) -> None:
    pool = tmp_path / "pool"
    pool.mkdir()
    _make_scene(pool)
    output = tmp_path / "out"
    cfg = TrainingConfig(
        training_pool_dir=str(pool),
        output_dir=str(output),
        device="cpu",
        total_steps=2,
        batch_size=2,
        patch_size=16,
        base_channels=4,
        unet_depth=3,
        num_workers=0,
        patches_per_scene=4,
        log_every=1,
        save_every=10,
        sigma_lf=2.0,
        tb_image_every=10,
    )

    path = train(cfg)

    assert path.exists()
