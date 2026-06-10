# EP14 — 4x Drizzle-informed SR Loss Atlas

## Goal

Visual tutorial for EP12 training: Drizzle-informed training input pipeline (8-channel) + ThermalSR4xLoss term-by-term breakdown.

## Build

```bash
uv run python scripts/build_ep14_cache.py --force
uv run python scripts/build_notebook.py notebooks/ep14_4x_loss_atlas --execute
```

## Data source

- Geometry/temperature/burst/fusion: `tcforge` (including `drizzle_features` scale=4 and `fuse_burst_to_features` scale=1)
- Demo aligns with `configs/synthetic/training_pool_4x.json` defaults: **248 frames/scene**, scale=4, real refined shifts, realistic detector noise.
- Demo shows 16 LR frames for sub-sampled figures; obs_features fused from full 248-frame burst.
- Loss patch: TCForge center crop with synthetic ringing + simulated log_var uncertainty on pred (teaching only).

## Input & Output Specification

### Input Tensor (8 Channels, 4x Scale)
The input to the 4x UNet model is a concatenated tensor of size `(8, H_hr, W_hr)` containing:
- **ch0-ch2**: 4x Drizzle features splatted directly onto the 4x grid (mean, coverage, variance).
- **ch3-ch7**: 1x observation features (aligned mean, median, coverage, variance, highpass) bilinearly upsampled to 4x grid.

### Output Tensor (2 Channels, 4x Scale)
The model outputs two channels of size `(2, H_hr, W_hr)`:
- **ch0**: Predicted 4x HR temperature (`pred`).
- **ch1**: Predicted log-variance representing heteroscedastic uncertainty (`log_var`).

## Loss breakdown (ThermalSR4xLoss)

1. **LF Loss (low-frequency L1)**: Gaussian blur (sigma=8.0) to anchor global calibration and DC bias.
2. **HF Loss (high-frequency L1)**: Highpass error weighted by coverage map to guide main structure restoration.
3. **Edge Loss (Sobel L1)**: Fine edge loss (weighted by edge mask boost) + 2x coarse scale downsampled edge loss to enforce edge connectivity.
4. **Forward Consistency Loss (physical projection)**: Blur 4x prediction by PSF and average pool back to 1x LR grid to verify against observed drizzle mean (data fidelity term).
5. **NLL Loss (heteroscedastic Gaussian)**: Negative Log-Likelihood utilizing `log_var` to adaptively suppress high-uncertainty regions (e.g. boundaries).
6. **HF Detail Loss (inverse coverage weight)**: Highpass error weighted by inverse coverage to boost weak edges in sparse-coverage regions.

## Status

- TCForge-integrated pipeline figures (00-06) + 4x loss atlas figures (08-16) ready.
