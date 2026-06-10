# EP12 4x — UNet@2000 vs Bare Drizzle Benchmark

## Goal

Compare the EP12 drizzle-informed 4x UNet step-2000 checkpoint against bare tcforge
scatter-add drizzle mean on the real 248 clean main-session frames.

## Inputs

- Raw input: EP06 clean main session, 248 frames.
- Alignment: EP05 `contour_refined` shifts.
- UNet checkpoint: `algos/ep12_4x_sr/outputs/ep12_large_bucketv2/checkpoint_step_002000.pt`.
- Baseline: bare drizzle mean (`drizzle_features` channel 0).
- Highpass sigma: `5.0`.

## Artifacts

- Script: `algos/ep12_4x_benchmark/scripts/run_ep12_vs_drizzle_4x.py`.
- Notebook: `notebooks/ep12_4x_benchmark/`.
- Output: `output/ep12_4x_benchmark/`.

## Boundary

Contour-level visual benchmark only. Checkpoint is synthetic-pretrained at step 2000;
real-data gains carry domain-gap risk. 3x is display zoom; reconstruction grid is 4x.

## EP07 2x x2up vs EP12 4x Gate — 2026-06-10

### Inputs

- EP07 2x model: `algos/ep07_unet_sr/outputs/ep07_v6_physics/model_final.pt` (step 60000).
- EP12 4x checkpoint: `algos/ep12_4x_sr/outputs/ep12_hybrid_v1/checkpoint_step_048000.pt`.
- Real data: EP06 clean main session, 248 frames.
- Alignment: `contour_refined` shifts.

### Command

```bash
cd algos/ep12_4x_benchmark
uv run python scripts/run_ep07x2up_vs_ep12_4x.py --device cuda:1
```

### Outputs

- Output directory: `output/ep12_4x_benchmark/ep07x2up_vs_ep12/`.
- Temperature comparison: `ep07x2up_vs_ep12_center_zoom3x_temperature.png`.
- Highpass comparison: `ep07x2up_vs_ep12_center_zoom3x_highpass.png`.
- Center zigzag ROI temperature: `ep07x2up_vs_ep12_zigzag_roi_temperature.png`.
- Center zigzag ROI highpass: `ep07x2up_vs_ep12_zigzag_roi_highpass.png`.
- Metrics CSV: `metrics_summary.csv`.

### Visual Conclusion

EP12 48k 4x does not show a visible gain over EP07 2x x2up in contour clarity or center zigzag-line separability. EP07x2up is sharper but has stronger highpass over/undershoot; EP12 is smoother and closer to bare drizzle, while proxy metrics also do not support EP12 as the better arm.
