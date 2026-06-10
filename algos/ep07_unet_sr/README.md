# EP07v2 UNet Thermal SR

Regression-style UNet for the EP07v2 4x thermal super-resolution POC. The model learns from compact TCForge synthetic scenes: 1x fused observation features go in, reconstructed 4x Celsius temperature patches come out.

This is an independent UV project under `algos/ep07_unet_sr/`; do not run it from the repository root virtualenv.

## Environment

```bash
cd algos/ep07_unet_sr
uv sync
```

`tcforge` is installed from `../../tcforge` as an editable path dependency. CPU smoke tests are supported. Formal training should use a CUDA PyTorch build when available; if CUDA is unavailable, pass `--device cpu`.

## Generate A Smoke Pool

Run this from the repository root:

```bash
uv run python scripts/generate_training_pool.py \
  --config configs/synthetic/training_pool_4x.json \
  --num-scenes 5 \
  --output-dir /tmp/smoke_test_pool \
  --workers 1
```

The pool stores compact scenes only:

```text
scene_0000/
├── hr_mask_4x.png
├── hr_edge_4x.png
├── obs_features_1x.npz
├── shifts.npy
└── metadata.json
```

It does not store the raw 248-frame LR burst or the HR temperature field. `ThermalSRDataset` reads `obs_features_1x.npz` and reconstructs the HR target from `hr_mask_4x.png` plus `metadata.json` fields such as `T_bg_c`, `delta_T_c`, `low_freq_amplitude_c`, `low_freq_sigma_px`, and `low_freq_seed`.

## Training

```bash
# Single-GPU AMP training
cd algos/ep07_unet_sr
uv run python -m unet_sr.train \
  --training-pool-dir /path/to/pool \
  --output-dir outputs/run1 \
  --total-steps 50000 \
  --device cuda \
  --amp
```

```bash
# Two-GPU DDP + AMP training
cd algos/ep07_unet_sr
uv run torchrun --nproc_per_node=2 -m unet_sr.train \
  --training-pool-dir /path/to/pool \
  --output-dir outputs/run1 \
  --total-steps 50000 \
  --device cuda \
  --amp
```

```bash
# CPU fallback. AMP and DDP are disabled on CPU.
cd algos/ep07_unet_sr
uv run python -m unet_sr.train \
  --training-pool-dir /tmp/smoke_test_pool \
  --output-dir outputs/smoke \
  --total-steps 10 \
  --batch-size 2 \
  --patch-size-hr 64 \
  --num-workers 0 \
  --device cpu
```

Expected smoke behavior: the run prints finite `total`, `mse`, and `edge` losses and writes `outputs/smoke/model_final.pt`.

## Inference

```python
import torch
from unet_sr.inference import infer_full_frame, infer_from_burst
from unet_sr.model import ThermalSRUNet

model = ThermalSRUNet(scale=4)
checkpoint = torch.load("outputs/smoke/model_final.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])

hr = infer_full_frame(model, obs_features, scale=4, patch_size_hr=256, overlap=32, device="cpu")
hr_from_burst = infer_from_burst(model, lr_burst, shifts, scale=4, device="cpu")
```

`infer_full_frame` tiles in LR coordinates and blends predicted HR patches in HR coordinates. `infer_from_burst` first calls `tcforge.fusion.fuse_burst_to_features()`.

## Tests

```bash
cd algos/ep07_unet_sr
uv run pytest -q
```

## Git And Data Rules

Training pools, checkpoints, logs, `outputs/`, `.venv/`, and generated data must not be committed. Source code and tests are the reproducible part.

## Current Limits

This is synthetic-only pretraining. Applying it to the real 248-frame clean SR set still requires alignment quality gates and feature validation. A visually sharper 4x output is not evidence for 5 um temperature metrology or true optical resolution recovery.
