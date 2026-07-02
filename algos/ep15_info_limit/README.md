# EP15 Info Limit

Independent UV project for EP15 first-principles information-limit checks.

## Setup

```bash
cd algos/ep15_info_limit
uv sync
uv pip install -e ../../core
```

M4 uses the CUDA PyTorch wheel in this isolated UV environment:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv run python -c "import torch; print(torch.cuda.is_available())"
```

## M1 Phase Structure

```bash
uv run python scripts/run_m1_phase_structure.py
```

Outputs are written to `output/ep15_info_limit/m1_phase_structure/`.

## M2 FRC Information Cutoff

```bash
uv run python scripts/run_m2_frc.py
```

Outputs are written to `output/ep15_info_limit/m2_frc/`.

## M3 Sigma Arbitration

```bash
uv run python scripts/run_m3_sigma_arbitration.py
```

Outputs are written to `output/ep15_info_limit/m3_sigma/`.

## M4 MAP-TV Deconvolution Anchor

Run a smoke pass first. Select one idle GPU explicitly; during the 2026-06-10 run GPU 0 was used because GPU 1 was occupied by EP07 training.

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_m4_deconv_anchor.py --smoke --chunk-size 8
```

Run the full single-GPU scan:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_m4_deconv_anchor.py --chunk-size 32
```

Default M4 settings use the M1 grid scale, 248 clean main-session frames, contour-refined shifts, PSF sigma scan `0.2,0.3,0.4,0.5`, lambda scan `3e-4,1e-3,3e-3`, FISTA MAP-TV with 150 iterations, and detector-aperture box integration via `avg_pool2d`. `--no-box` is only for ablation and skips the strict EP06 box-model smoke comparison.

Outputs are written to `output/ep15_info_limit/m4_deconv_anchor/`. `parameter_selection.csv` records every sigma/lambda split-half selection run plus per-sigma full-run timings; `convergence_curves.csv` records the per-sigma full selected-lambda runs. The four-arm comparison uses the EP07 v6 checkpoint at `../ep07_unet_sr/outputs/ep07_v6_physics/model_final.pt` when available.

## Stage 0b Info Budget2 Shift Sweep

Synthetic replacement scaffold for the historical `info_budget2.py` artifact:

```bash
uv run python scripts/run_stage0b_info_budget2_shift_sweep.py
```

Useful MVP smoke:

```bash
uv run python scripts/run_stage0b_info_budget2_shift_sweep.py \
  --lr-size 64 \
  --scene-seeds 11,23 \
  --frame-budgets 16,64 \
  --shift-error-seeds 401
```

Outputs are written to `output/ep15_info_limit/stage0b_info_budget2/`.
The script reads the corrected 20 um detector pitch from
`configs/stage_calibration.json`, accepts `--psf-sigmas-lr-px`, and sweeps
`--shift-error-grid` over LR-pixel Gaussian shift errors for Stage 1a DR
calibration. It reports both a spatial drizzle/Wiener baseline and a Fourier
alias ridge oracle (`alias_multiframe_wiener_*`). For DR calibration, prefer
the full or near-full frame-budget oracle delta columns; tiny budgets can be
conditioning diagnostics rather than monotonic robustness curves.
