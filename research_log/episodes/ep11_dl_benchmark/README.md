# EP11 — UNet 2x@4000 vs TGV 2x Visual Benchmark

## Goal

Compare the EP07 residual UNet step-4000 checkpoint against the existing EP10
TGV best 2x artifact on the real 248 clean main-session frames.
The comparison is same-domain highpass, same center-third ROI, same 3x display
zoom, and same diverging colormap range.

## Inputs

- Raw input: EP06 clean main session, 248 frames.
- Alignment: EP05 `contour_refined` shifts.
- UNet checkpoint: `algos/ep07_unet_sr/outputs/ep07_unet_sr_v4/checkpoint_step_004000.pt`.
- TGV highpass: `output/ep10_tgv_sr/best_hr_highpass.npy`.
- Highpass sigma: `5.0`.

## Artifacts

- Script: `algos/ep11_dl_benchmark/scripts/run_unet_vs_drizzle_2x.py`.
- Notebook: `notebooks/ep11_dl_benchmark/`.
- Output: `output/ep11_dl_benchmark/`.

## Boundary

This episode is a quick contour-level visual benchmark. It does not retrain
UNet, does not rerun the full TGV sweep, and does not claim 5 um metrology,
temperature accuracy, or 3x SR. The 3x setting is center-ROI display zoom only;
the reconstruction grid remains EP07 2x. UNet@4000 is a synthetic-pretrained
mid-training checkpoint, so any real-data advantage must be interpreted with
domain-gap risk.

## Progress

- 2026-06-11: Added EP07 four-arm checkpoint-selection整理 for v6 / v8.1a / v8.1b / v9b. Scripts: `algos/ep07_unet_sr/scripts/extract_checkpoint_metrics.py`, `algos/ep07_unet_sr/scripts/plot_checkpoint_selection.py`; report: `paper/reports/ep11_dl_benchmark/unet_checkpoint_selection.md`; generated artifacts under `output/ep11_dl_benchmark/checkpoint_selection/`, including GPU 1 unified EP11 reruns for the four recommended canonical checkpoints.
