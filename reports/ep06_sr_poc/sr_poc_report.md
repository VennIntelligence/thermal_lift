# EP06 — 2x Contour-Level SR POC Report

## Scope

EP06 evaluates a 2x contour-level super-resolution POC on 248 clean SR-usable frames selected from the historical 255-frame main TXT acquisition segment. The goal is to test whether multi-frame micro-scanning improves visible chip/internal contours relative to LR and bicubic references.

This report does **not** claim 4x SR, 5 um actual spatial resolution, or absolute-temperature SR. The highpass track is a structure-map reconstruction. The raw-temperature track is a control track used to test whether the highpass preprocessing creates unsupported structure.

## Inputs

- Frames: 248 clean SR-usable rows from `output/ep01_data_processing/frame_audit.csv`, selected with `is_sr_usable` when available and aligned one-to-one with EP05 contour results. The raw 255-frame main segment is historical acquisition context only.
- Data: `data/data_raw/infrared_avi/*.txt`.
- Alignment: default `output/ep05_contour_alignment/contour_alignment_results.csv` (248 rows). Ablation also uses `data_driven_ncc_init` from the same CSV and filename/stage controls from EP05 capacity outputs. Tuned refined alignment is optional/pending unless a 248-frame candidate under `output/ep05_alignment_tuning_study/` is explicitly supplied.
- Shift convention: `align_dx_px/align_dy_px` and `refined_align_dx_px/refined_align_dy_px` move each LR frame into the reference coordinate system, in LR pixels.
- Evaluation anchors: `output/ep04_global_validation/segment_summary.csv` as a contour proxy, not optical ground truth.

## Reproduction

```bash
uv run python algos/ep06_sr_poc/scripts/run_saa.py --alignment-method data_driven_contour_refined --psf-sigma 0.5 --output-dir output/ep06_sr_poc_data_driven_align_sweep/default_contour_refined_psf05
uv run python algos/ep06_sr_poc/scripts/run_ibp.py --alignment-method data_driven_contour_refined --max-iter 8 --psf-sigma 0.5 --output-dir output/ep06_sr_poc_data_driven_align_sweep/default_contour_refined_psf05
uv run python algos/ep06_sr_poc/scripts/run_map_tv.py --alignment-method data_driven_contour_refined --max-iter 8 --step-size 0.25 --psf-sigma 0.5 --no-fista --lambda-grid 0.0003,0.001,0.003,0.01 --output-dir output/ep06_sr_poc_data_driven_align_sweep/default_contour_refined_psf05
uv run python algos/ep06_sr_poc/scripts/run_evaluation.py --output-dir output/ep06_sr_poc_data_driven_align_sweep/default_contour_refined_psf05 --center-roi-sizes 160,112,80
uv run python scripts/run_ep06_alignment_ablation.py
uv run python scripts/summarize_ep06_alignment_sweep.py --sweep-root output/ep06_sr_poc_data_driven_align_sweep --baseline-dir output/ep06_sr_poc
uv run python scripts/build_notebook.py notebooks/ep06_sr_poc --execute
```

For the full alignment sweep, repeat the first four commands with these changes:

- `tuned_contour_refined_psf05`: optional/pending. Only add `--alignment-csv <validated 248-frame CSV under output/ep05_alignment_tuning_study/> --alignment-method data_driven_contour_refined` after EP05 tuning has produced a full 248-frame candidate.
- `ncc_init_psf05`: add `--alignment-method data_driven_ncc_init` and write to `output/ep06_sr_poc_data_driven_align_sweep/ncc_init_psf05`.

## Methods

| Method | Role | Notes |
|---|---|---|
| LR reference | Detector-grid baseline | Direct input reference, displayed on the HR grid only for comparison |
| Bicubic reference | Display interpolation baseline | No new information; controls for visual upsampling |
| SAA-uniform | Multi-frame baseline | Uses EP05 positive LR-to-reference shifts directly for 2x backfill |
| SAA-weighted | Quality-gated baseline | Uses EP05 alignment quality proxy weights |
| IBP | Forward-model baseline | Receives the same shifts; prediction applies inverse shift internally |
| MAP-TV | Regularized baseline | Lambda selected by split-half consistency plus artifact/std penalty |

## Required Outputs

- Arrays: `saa_uniform_highpass.npy`, `saa_weighted_highpass.npy`, `saa_uniform_raw.npy`, `saa_weighted_raw.npy`, `ibp_highpass.npy`, `ibp_raw.npy`, `map_tv_highpass.npy`, `map_tv_raw.npy`.
- References: `lr_reference.npy`, `bicubic_reference.npy`, `lr_raw_reference.npy`, `bicubic_raw_reference.npy`.
- Validation and convergence: `saa_synthetic_validation.json`, `ibp_synthetic_validation.json`, `ibp_convergence.csv`, `map_tv_synthetic_validation.json`, `map_tv_lambda_selection.csv`, `map_tv_convergence.csv`.
- Evaluation: `evaluation_summary.csv`.
- Direct visual comparisons: `comparison_fullview.png`, `comparison_roi_1.png`, `comparison_roi_2.png`, `comparison_roi_3.png`, `comparison_control_track.png`, `comparison_center_raw_temperature.png`, `gradient_magnitude_comparison.png`, `split_half_consistency.png`, `artifact_audit.png`.
- Alignment sweep: current baseline directories are `output/ep06_sr_poc_data_driven_align_sweep/{default_contour_refined_psf05,ncc_init_psf05}/`; tuned refined remains optional/pending until backed by a validated 248-frame CSV.
- Sweep summary: `output/ep06_sr_poc_data_driven_align_sweep/summary/sweep_method_metrics.csv`, `sweep_map_tv_lambda.csv`, `sweep_validation_summary.csv`, `sweep_delta_vs_baseline.csv`, `sweep_summary.json`, `sweep_metric_bars.png`, `sweep_map_tv_lambda_selection.png`, `sweep_delta_vs_baseline.png`.

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
| SAA | 30.25 | Passes the 25 dB smoke threshold in the psf=0.5 sweep |
| IBP | 30.50 | Beats the SAA synthetic initialization |
| MAP-TV | 30.38 | Beats the SAA synthetic initialization, but is not the sharpest real-data candidate |

Highpass main-track metrics from `default_contour_refined_psf05`:

| Method | Std | Mean gradient | P95 gradient | Artifact score | Chamfer proxy [LR px] | Notes |
|---|---:|---:|---:|---:|---:|---|
| Bicubic | 0.2057 | 0.4521 | 0.8844 | 0.3049 | 0.0379 | Display-only baseline; no new multi-frame information |
| SAA-uniform | 0.1923 | 0.1613 | 0.4420 | 0.1413 | 0.0111 | Smoothest multi-frame output; validates phase coverage |
| SAA-weighted | 0.1926 | 0.1610 | 0.4427 | 0.1410 | 0.0111 | Nearly identical to uniform; quality weights do not dominate this dataset |
| IBP | 0.1982 | 0.1719 | 0.4483 | 0.1531 | 0.0111 | psf=0.5 is more conservative than old psf=1.0 in gradient/std, but artifact still rises |
| MAP-TV | 0.1951 | 0.1363 | 0.4323 | 0.1437 | 0.0157 | Strong regularization after split-half selection; smoother than IBP/SAA in gradient metrics |

Raw-control visual highpass metrics:

| Method | Std | Mean gradient | P95 gradient | Artifact score | Chamfer proxy [LR px] |
|---|---:|---:|---:|---:|---:|
| SAA-weighted raw | 0.1926 | 0.1620 | 0.4431 | 0.1449 | 0.0111 |
| IBP raw | 0.1983 | 0.1739 | 0.4575 | 0.1569 | 0.0111 |
| MAP-TV raw | 0.1957 | 0.1438 | 0.4398 | 0.1433 | 0.0157 |

## Alignment Ablation

EP06 now treats alignment as an explicit experimental factor, not as a hidden preprocessing detail. The current fast ablation uses SAA-weighted highpass outputs because SAA is cheap enough to compare multiple alignment fields while preserving the same frame set, highpass sigma, weights convention, and 2x grid.

Current ablation outputs:

- `output/ep06_alignment_ablation/strategy_metrics.csv`
- `output/ep06_alignment_ablation/split_half_metrics.csv`
- `output/ep06_alignment_ablation/phase_coverage.csv`
- `output/ep06_alignment_ablation/strategy_split_half_nrmse.png`
- `output/ep06_alignment_ablation/strategy_gradient_artifact.png`
- `output/ep06_alignment_ablation/difference_to_default.png`
- `output/ep06_alignment_ablation/phase_coverage_2x.png`
- `output/ep06_alignment_ablation/difference_to_default_panels.png`

SAA alignment comparison:

The tuned refined row below is a historical deprecated-path sensitivity result and is excluded from the current EP06 baseline until rerun from a validated 248-frame tuning-study CSV.

| Alignment | Mean gradient | P95 gradient | Artifact score | Split-half NRMSE | NRMSE to default | 2x phase |
|---|---:|---:|---:|---:|---:|---:|
| default contour refined | `0.0373` | `0.2008` | `1.5094` | `0.0217` | `0.0000` | `4/4` |
| NCC init | `0.0386` | `0.1892` | `1.6971` | `0.0237` | `0.0302` | `4/4` |
| tuned contour refined | `0.0375` | `0.2019` | `1.4710` | `0.0251` | `0.0210` | `4/4` |
| filename affine fit | `0.0379` | `0.1996` | `1.6721` | `0.0272` | `0.0349` | `4/4` |

Interpretation:

- Default contour refined is the most stable SAA alignment in this center-ROI ablation by split-half NRMSE.
- The historical tuned refined row improves EP05 held-out Chamfer and has the lowest artifact proxy here, but it came through a deprecated path and its split-half NRMSE is worse than default refined. It remains excluded/pending for the current baseline.
- NCC init preserves continuous phase coverage and the strongest EP05 gradient correlation, but its artifact proxy is higher in this SAA ablation. It remains an important phase-prior control.
- Filename affine remains a strong prior/control, but it is not the best SAA ablation strategy and should not be promoted to alignment truth.

MAP-TV split-half selection chose `lambda=0.01` for both the highpass and raw tracks in the current run. Larger regularization reduces split-half NRMSE and suppresses gradient energy, so MAP-TV should now be described as a stronger regularized candidate rather than the sharpest method.

## Data-Driven Alignment Sweep

The historical full sweep re-ran SAA-weighted, IBP, and MAP-TV under three data-driven alignment variants, all with `psf_sigma=0.5` for the forward-model methods. Results that use `tuned_contour_refined_psf05` are deprecated-path sensitivity records, not current EP06 baseline evidence. The current baseline should be rerun with 248-frame default contour refined and NCC-init variants; tuned refined can be added only after EP05 produces a validated 248-frame tuning-study CSV.

Sweep outputs:

- `output/ep06_sr_poc_data_driven_align_sweep/default_contour_refined_psf05/`
- `output/ep06_sr_poc_data_driven_align_sweep/tuned_contour_refined_psf05/` (historical deprecated-path output; excluded/pending)
- `output/ep06_sr_poc_data_driven_align_sweep/ncc_init_psf05/`
- `output/ep06_sr_poc_data_driven_align_sweep/summary/sweep_metric_bars.png`
- `output/ep06_sr_poc_data_driven_align_sweep/summary/sweep_map_tv_lambda_selection.png`
- `output/ep06_sr_poc_data_driven_align_sweep/summary/sweep_delta_vs_baseline.png`

Highpass sweep metrics:

| Alignment | Method | Std | Mean gradient | P95 gradient | Artifact score | Chamfer proxy [LR px] |
|---|---|---:|---:|---:|---:|---:|
| default contour refined | SAA-weighted | 0.192552 | 0.160952 | 0.442653 | 0.141004 | 0.011111 |
| default contour refined | IBP | 0.198153 | 0.171898 | 0.448343 | 0.153128 | 0.011111 |
| default contour refined | MAP-TV | 0.195114 | 0.136266 | 0.432344 | 0.143728 | 0.015713 |
| tuned contour refined | SAA-weighted | 0.192400 | 0.162785 | 0.443990 | 0.141700 | 0.011111 |
| tuned contour refined | IBP | 0.197529 | 0.172596 | 0.448763 | 0.153043 | 0.011111 |
| tuned contour refined | MAP-TV | 0.194602 | 0.136244 | 0.432641 | 0.143867 | 0.015713 |
| NCC init | SAA-weighted | 0.194832 | 0.167308 | 0.443325 | 0.156153 | 0.011111 |
| NCC init | IBP | 0.200905 | 0.179577 | 0.449252 | 0.169303 | 0.011111 |
| NCC init | MAP-TV | 0.197679 | 0.140189 | 0.432341 | 0.157196 | 0.015713 |

MAP-TV highpass lambda selection:

| Alignment | Selected lambda | Split-half NRMSE | Artifact score | Selection proxy |
|---|---:|---:|---:|---:|
| default contour refined | 0.01 | 0.015298 | 0.140477 | 0.022863 |
| tuned contour refined | 0.01 | 0.020330 | 0.142266 | 0.027903 |
| NCC init | 0.01 | 0.014472 | 0.153473 | 0.022738 |

Delta versus the old baseline:

- Default SAA and MAP-TV are unchanged by the sweep. Default IBP with `psf_sigma=0.5` lowers std by `-0.006953` and P95 gradient by `-0.050307` versus the old IBP, but artifact rises by `+0.007619`.
- Historical tuned refined stays close to default in the deprecated-path sweep. Relative to the old baseline, artifact changes are SAA `+0.000696`, IBP `+0.007534`, and MAP-TV `+0.000139`; these numbers should not be used as current baseline evidence until rerun with a validated 248-frame input.
- NCC init preserves the phase-prior control, but raises artifact across all full-SR methods: SAA `+0.015149`, IBP `+0.023794`, and MAP-TV `+0.013468`.

Decision from the sweep:

- Default contour refined remains the preferred EP06 main alignment. The earlier SAA ablation had the best split-half NRMSE for default refined, and the full MAP/IBP sweep does not overturn that decision.
- Tuned contour refined is useful as a sensitivity candidate, but its metric shifts do not change the MAP-TV or IBP conclusion.
- NCC init should remain a control. It preserves the phase prior, but its higher artifact proxy in the full SR sweep argues against promoting it to final alignment.
- MAP-TV should not be called best or sharpest. Under every data-driven alignment variant it selects `lambda=0.01` and remains smoother/lower-gradient than SAA/IBP; it is a conservative regularized candidate and a sharpness/regularization diagnostic.
- IBP with `psf_sigma=0.5` is more conservative than the old `psf_sigma=1.0` in gradient/std, but the artifact proxy still rises, so it is not a clear winner.

Direct visual comparison:

- `comparison_fullview.png` now uses a center 6x visual crop, showing that SAA suppresses frame noise and background drift relative to LR/bicubic, while IBP tests forward-model sharpening and MAP-TV tests stronger regularized smoothing.
- `comparison_roi_*.png` now keeps the same chip-center location and increases zoom across the ROI series, exposing the main tradeoff between IBP edge response, MAP-TV regularization, and SAA stability.
- `comparison_control_track.png` shows the same center 6x crop for the highpass main track and raw-temperature control track, reducing the risk that highpass input preprocessing alone created the structures.
- `comparison_center_raw_temperature.png` adds an ordinary raw-temperature center crop with LR raw and bicubic raw references so the central pin/internal regions can be checked without the highpass red/blue edge response.

## Interpretation Rules

- A sharper image is not sufficient evidence if split-half consistency degrades or artifact score rises sharply.
- Raw-control agreement is required before presenting highpass features as data-supported structure.
- Segment Chamfer is a proxy from EP04 localization anchors, not a final metrology claim.
- 2x grid output is a reconstruction grid. It is not a claim that the thermal system now resolves 5 um structure.
- Stage commands and filename-derived affine fits remain priors/controls, not alignment ground truth.
- Highpass outputs are structure maps. Red/blue highpass lobes indicate local positive/negative structure response and should not be read as direct absolute temperature.

## Current Risks

- The fallback implementations in the run scripts are intentionally conservative orchestration fallbacks. The current run used the algorithm modules under `algos/ep06_sr_poc/src`.
- MAP-TV is no longer described as the sharpest candidate in the current run; it is a regularized candidate selected by split-half stability and must still be shown with ROI and raw-control panels.
- The full data-driven sweep does not promote NCC init, IBP, or MAP-TV over default contour refined as the EP06 recommendation. Tuned refined remains excluded/pending until rerun from a validated 248-frame tuning-study CSV.
- EP04 segment anchors are useful for contour proxy scoring but are not independent optical ground truth.

## EP07 Handoff

Use EP06 to choose a single candidate reconstruction path for deeper validation. EP07 should add stricter MTF/edge-transfer checks, cross-ROI stability analysis, and customer-facing examples while preserving the distinction between contour visibility and actual spatial resolution.
