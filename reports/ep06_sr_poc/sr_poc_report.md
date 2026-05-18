# EP06 — 2x Contour-Level SR POC Report

## Scope

EP06 evaluates a 2x contour-level super-resolution POC on the 255-frame main TXT session. The goal is to test whether multi-frame micro-scanning improves visible chip/internal contours relative to LR and bicubic references.

This report does **not** claim 4x SR, 5 um actual spatial resolution, or absolute-temperature SR. The highpass track is a structure-map reconstruction. The raw-temperature track is a control track used to test whether the highpass preprocessing creates unsupported structure.

## Inputs

- Frames: `output/ep01_data_processing/frame_audit.csv`, main session only.
- Data: `data/data_raw/infrared_avi/*.txt`.
- Alignment: `output/ep05_contour_alignment/contour_alignment_results.csv`.
- Shift convention: `align_dx_px/align_dy_px` and `refined_align_dx_px/refined_align_dy_px` move each LR frame into the reference coordinate system, in LR pixels.
- Evaluation anchors: `output/ep04_global_validation/segment_summary.csv` as a contour proxy, not optical ground truth.

## Reproduction

```bash
uv run python algos/ep06_sr_poc/scripts/run_saa.py
uv run python algos/ep06_sr_poc/scripts/run_ibp.py --max-iter 8
uv run python algos/ep06_sr_poc/scripts/run_map_tv.py --max-iter 12 --step-size 0.5 --lambda-grid 0.00001,0.0001,0.0003,0.001
uv run python algos/ep06_sr_poc/scripts/run_evaluation.py
uv run python scripts/build_notebook.py notebooks/ep06_sr_poc --execute
```

## Methods

| Method | Role | Notes |
|---|---|---|
| LR reference | Detector-grid baseline | Direct input reference, displayed on the HR grid only for comparison |
| Bicubic reference | Display interpolation baseline | No new information; controls for visual upsampling |
| SAA-uniform | Multi-frame baseline | Uses EP05 positive LR-to-reference shifts directly for 2x backfill |
| SAA-weighted | Quality-gated baseline | Uses EP05 alignment quality proxy weights |
| IBP | Forward-model baseline | Receives the same shifts; prediction applies inverse shift internally |
| MAP-TV | Regularized baseline | Lambda selected by split-half consistency proxy |

## Required Outputs

- Arrays: `saa_uniform_highpass.npy`, `saa_weighted_highpass.npy`, `saa_uniform_raw.npy`, `saa_weighted_raw.npy`, `ibp_highpass.npy`, `ibp_raw.npy`, `map_tv_highpass.npy`, `map_tv_raw.npy`.
- References: `lr_reference.npy`, `bicubic_reference.npy`, `lr_raw_reference.npy`, `bicubic_raw_reference.npy`.
- Validation and convergence: `saa_synthetic_validation.json`, `ibp_synthetic_validation.json`, `ibp_convergence.csv`, `map_tv_synthetic_validation.json`, `map_tv_lambda_selection.csv`, `map_tv_convergence.csv`.
- Evaluation: `evaluation_summary.csv`.
- Direct visual comparisons: `comparison_fullview.png`, `comparison_roi_1.png`, `comparison_roi_2.png`, `comparison_roi_3.png`, `comparison_control_track.png`, `comparison_center_raw_temperature.png`, `gradient_magnitude_comparison.png`, `split_half_consistency.png`, `artifact_audit.png`.

## Evaluation Criteria

Primary evidence:

1. Direct highpass full-view comparison across LR, bicubic, SAA, IBP, and MAP-TV.
2. Three ROI comparisons with the same method ordering.
3. Highpass main track versus raw-temperature control track.

Supporting metrics:

- Mean and P95 gradient magnitude.
- EP04 segment-point Chamfer proxy in LR pixels.
- MAP-TV split-half consistency over the lambda grid.
- Artifact score based on high-frequency residual and Laplacian energy.
- Correlation and NRMSE to bicubic, used only as reference stability checks.

## Results Summary

All real-data arrays are finite and have the expected 2x shape `(960, 1280)`.

Synthetic validation:

| Method | PSNR [dB] | Acceptance note |
|---|---:|---|
| SAA | 28.43 | Passes the 25 dB smoke threshold |
| IBP | 28.78 | Improves over SAA |
| MAP-TV | 29.03 | Improves over IBP after 12 iterations and small-lambda split-half selection |

Highpass main-track metrics:

| Method | Mean gradient | P95 gradient | Artifact score | Chamfer proxy [LR px] | Notes |
|---|---:|---:|---:|---:|---|
| Bicubic | 0.4521 | 0.8844 | 0.3049 | 0.0379 | Display-only baseline; no new multi-frame information |
| SAA-uniform | 0.1613 | 0.4420 | 0.1414 | 0.0111 | Smoothest multi-frame output; validates phase coverage |
| SAA-weighted | 0.1610 | 0.4427 | 0.1410 | 0.0111 | Nearly identical to uniform; quality weights do not dominate this dataset |
| IBP | 0.1788 | 0.4987 | 0.1455 | 0.0111 | Adds forward-model sharpening with modest artifact increase |
| MAP-TV | 0.2226 | 0.6592 | 0.1631 | 0.0000 | Sharpest contour proxy, but also highest artifact score |

Raw-control visual highpass metrics:

| Method | Mean gradient | P95 gradient | Artifact score | Chamfer proxy [LR px] |
|---|---:|---:|---:|---:|
| SAA-weighted raw | 0.1620 | 0.4431 | 0.1449 | 0.0111 |
| IBP raw | 0.1803 | 0.5000 | 0.1489 | 0.0111 |
| MAP-TV raw | 0.2299 | 0.6827 | 0.1671 | 0.0000 |

MAP-TV split-half selection chose `lambda=0.0001` for the highpass track and `lambda=0.00001` for the raw track. Larger lambdas improved split-half NRMSE slightly but raised the artifact penalty enough that the combined proxy selected smaller regularization.

Direct visual comparison:

- `comparison_fullview.png` now uses a center 3x visual crop, showing that SAA suppresses frame noise and background drift relative to LR/bicubic, while IBP and MAP-TV recover sharper central chip contours.
- `comparison_roi_*.png` now keeps the same chip-center location and increases zoom across the ROI series, exposing the main tradeoff: MAP-TV has the crispest edges, but it also amplifies narrow highpass lobes more than SAA/IBP.
- `comparison_control_track.png` shows the same center 3x crop for the highpass main track and raw-temperature control track, reducing the risk that highpass input preprocessing alone created the structures.
- `comparison_center_raw_temperature.png` adds an ordinary raw-temperature center crop with LR raw and bicubic raw references so the central pin/internal regions can be checked without the highpass red/blue edge response.

## Interpretation Rules

- A sharper image is not sufficient evidence if split-half consistency degrades or artifact score rises sharply.
- Raw-control agreement is required before presenting highpass features as data-supported structure.
- Segment Chamfer is a proxy from EP04 localization anchors, not a final metrology claim.
- 2x grid output is a reconstruction grid. It is not a claim that the thermal system now resolves 5 um structure.

## Current Risks

- The fallback implementations in the run scripts are intentionally conservative orchestration fallbacks. The current run used the algorithm modules under `algos/ep06_sr_poc/src`.
- MAP-TV is the sharpest candidate but has the largest artifact score. It should not be reported as strictly better without showing the ROI and raw-control panels.
- EP04 segment anchors are useful for contour proxy scoring but are not independent optical ground truth.

## EP07 Handoff

Use EP06 to choose a single candidate reconstruction path for deeper validation. EP07 should add stricter MTF/edge-transfer checks, cross-ROI stability analysis, and customer-facing examples while preserving the distinction between contour visibility and actual spatial resolution.
