# EP11 — UNet 2x@4000 vs TGV 2x Benchmark

This episode runs a quick real-data visual benchmark on the 248 clean main-session
thermal frames. It compares the EP07 residual UNet step-4000 checkpoint against
the existing EP10 TGV best 2x artifact in the same highpass domain,
same center ROI, and same colormap range.

## Scope

- Input: EP06 clean main session, 248 raw temperature frames.
- Alignment: EP05 `contour_refined` shifts.
- UNet checkpoint: `../ep07_unet_sr/outputs/ep07_unet_sr_v4/checkpoint_step_004000.pt`.
- Baseline artifact: `../../output/ep10_tgv_sr/best_hr_highpass.npy`.
- Baseline metrics: `../../output/ep10_tgv_sr/sweep_results.csv` plus `run_summary.json`.
- Highpass sigma: `5.0`, matching EP10 TGV.
- Default device: `cuda:1`. Bare `--device cuda` is also resolved to `cuda:1` when two or more CUDA devices are visible. `cuda:0` is protected unless `--allow-cuda0` is explicitly passed.

## Run

```bash
cd algos/ep11_dl_benchmark
uv sync
uv run python scripts/run_unet_vs_drizzle_2x.py \
  --checkpoint ../ep07_unet_sr/outputs/ep07_unet_sr_v4/checkpoint_step_004000.pt \
  --baseline-hr ../../output/ep10_tgv_sr/best_hr_highpass.npy \
  --baseline-sweep ../../output/ep10_tgv_sr/sweep_results.csv \
  --baseline-summary ../../output/ep10_tgv_sr/run_summary.json \
  --baseline-name "TGV best 2x" \
  --output-dir ../../output/ep11_dl_benchmark \
  --zoom 3.0 \
  --center-fraction 0.3333333 \
  --device cuda:1
```

Smoke test:

```bash
uv run python scripts/run_unet_vs_drizzle_2x.py \
  --checkpoint ../ep07_unet_sr/outputs/ep07_unet_sr_v4/checkpoint_step_004000.pt \
  --baseline-hr ../../output/ep10_tgv_sr/best_hr_highpass.npy \
  --baseline-sweep ../../output/ep10_tgv_sr/sweep_results.csv \
  --baseline-summary ../../output/ep10_tgv_sr/run_summary.json \
  --baseline-name "TGV best 2x" \
  --output-dir /tmp/ep11_smoke \
  --limit 16 \
  --device cuda:1
```

The script writes:

- `unet_step4000_hr_temp.npy`
- `unet_step4000_hr_highpass.npy`
- `raw_mean_control_2x_hr_temp.npy`
- `raw_mean_control_2x_hr_highpass.npy`
- `unet_vs_tgv_2x_center_zoom3x_highpass.png`
- `unet_step4000_center_zoom3x_temperature.png`
- `comparison_summary.csv`
- `comparison_notes.md`
- `run_manifest.json`

## Interpretation Boundary

This benchmark is a contour-level visual comparison, not a 5 um metrology,
temperature-accuracy conclusion, or 3x SR claim. The reconstruction grid is
still EP07 2x; `--zoom 3.0` is only display magnification for the center ROI.
The fair side-by-side view is the highpass figure. The raw-temperature figure is
a UNet-only sanity view; the raw-mean control is used only for correlation, not
as an algorithm comparison.

The UNet checkpoint is trained on synthetic data and `checkpoint_step_004000.pt`
is a mid-training checkpoint, not the final 25000-step model. Treat any real
inner-contour improvement as domain-gap-sensitive until split-half consistency,
artifact behavior, and raw-control agreement are stable. Tenengrad or sharpness
alone must not be used to declare a winner.
