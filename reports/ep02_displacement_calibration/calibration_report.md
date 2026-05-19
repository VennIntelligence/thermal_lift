# EP02 Raster Path, Stage Prior, and Alignment Evidence

## Scope

EP02 reconstructs the main-session raster acquisition path, maps filename/stage coordinates into detector-space stage priors, and records which displacement evidence can be used for downstream alignment. It does not treat stage command as alignment truth and does not use small 2 um adjacent-step diagnostics to judge multi-frame SR feasibility.

The downstream contract is: EP02 provides acquisition path plus coordinate prior; EP04/EP05-style data-driven contour/NCC alignment provides the alignment anchor and quality gate before EP06 2x contour-level SR.

## Key Results

| Item | Result | Interpretation |
|---|---:|---|
| Main session frames | 255 | session=2 only |
| R=0 raster frames | 248 | primary step-and-shoot grid |
| Within-row X transitions | 232 | acquisition-time adjacent |
| Row transitions | 15 | Y advance plus X reset |
| Unique coordinates | 253 | filename coordinate coverage |
| Stage prior theta | 47.6 deg | configured prior |
| Detector pitch | 10.0 um/pixel | detector sampling pitch |
| Stage-prior dx/dy span | 5.65 px / 5.65 px | global detector-space prior coverage |
| 2x phase bins | 4/4 non-empty | prior covers all half-pixel bins |

2x phase-bin frame counts from the configured prior:

| phase-y bin \\ phase-x bin | 0 | 1 |
|---|---:|---:|
| 0 | 66 | 63 |
| 1 | 64 | 62 |

## Small-Step Diagnostics

| Diagnostic | Value | Use |
|---|---:|---|
| X coordinate-neighbor acquisition gap median | 1 frame | valid local smoke test |
| Y coordinate-neighbor acquisition gap median | 16 frames | not time-adjacent |
| X 2 um visible NCC projection | 0.0936 px | local image response |
| X 2 um stage-prior magnitude | 0.2000 px | nominal prior |
| X 4/2 visible projection ratio | 2.05 | short-time linearity check |
| Y 4/2 visible projection ratio | 0.64 | fails calibration monotonicity |

The X result supports local direction and short-time linearity under this ROI/preprocess choice. It is not an absolute stage-amplitude validation. The Y coordinate-neighbor result is a raster-path failure diagnostic: fixed-X Y neighbors are separated by about one row of acquisition time, so thermal evolution contaminates NCC.

### Time-Adjacent Method Comparison

`time_adjacent_method_summary.csv` is now interpreted as method evidence, not as a replacement calibration. X-step rows are true acquisition-neighbor smoke tests; row-transition rows combine Y advance with X reset and are not clean Y-only calibration samples.

| Method | Window | Projection ratio | RMS vs prior (px) | Interpretation |
|---|---|---:|---:|---|
| raw NCC | X-step | 0.506 | 0.1501 | strongest local visible X response |
| high-pass NCC | X-step | 0.475 | 0.1584 | robust local direction smoke test |
| gradient NCC | X-step | 0.451 | 0.1658 | contour-weighted local response |
| phase correlation | X-step | 0.000 | 0.2886 | degenerates on tiny subpixel steps |
| raw NCC | row transition | 0.545 | 1.8748 | not a clean Y-only pair |
| high-pass NCC | row transition | 0.523 | 1.9597 | not a clean Y-only pair |
| gradient NCC | row transition | 0.526 | 1.9404 | not a clean Y-only pair |
| phase correlation | row transition | 0.443 | 2.3179 | not a clean Y-only pair |

### Y Coordinate-Neighbor Failure Across Preprocessing

| Method | 2um projection/prior | 4um projection/prior | Visible 4um/2um | Expected 4um/2um | RMS 2um / 4um (px) |
|---|---:|---:|---:|---:|---:|
| raw NCC | 1.724 | 0.550 | 0.638 | 2.0 | 0.1810 / 0.1921 |
| high-pass NCC | 1.601 | 0.511 | 0.638 | 2.0 | 0.1571 / 0.2027 |
| gradient NCC | 1.589 | 0.511 | 0.643 | 2.0 | 0.1586 / 0.2063 |

The stable 4/2 ratio near 0.64 is the key failure signature. High-pass and gradient preprocessing do not fix the non-monotonic response, so these Y-only coordinate-neighbor pairs remain invalid for quantitative Y displacement calibration.

### AVI Direction Evidence

`avi_theta_summary.csv` is retained as auxiliary direction validation. The selected row is gradient NCC combined across 16 AVI files:

| Item | Value |
|---|---:|
| Gradient combined theta mean | 47.14 deg |
| Gradient combined theta median | 47.11 deg |
| 95% CI | [46.36, 47.92] deg |
| Reference theta | 47.60 deg |
| Reference inside CI | Yes |
| Samples | 16 AVI files |

This supports the configured 47.6 deg direction, but does not replace `configs/stage_calibration.json`. AVI files are rendered 8-bit videos, contain many duplicate frames, and show an X/Y subgroup split of about 3 deg under gradient NCC.

### AVI-TXT Line Match

| AVI group | Expected mapping | Correct-axis diff | Rejected-axis diff | Acquisition-gap implication |
|---|---|---:|---:|---|
| X-scan AVI | `xN.avi -> TXT fixed Y=N` | median 18.41 deg contour / 14.19 deg high-pass / 11.36 deg gradient | median 77.20 deg contour / 74.77 deg high-pass / 80.79 deg gradient | correct fixed-Y TXT rows have median gap 1 |
| Y-scan AVI | `yN.avi -> TXT fixed X=N` | median 1.86 deg contour / 5.28 deg high-pass / 6.24 deg gradient | median 82.48 deg contour / 85.78 deg high-pass / 81.66 deg gradient | correct fixed-X TXT lines have median gap 16 |

The naming rule is therefore consistent: AVI prefix names the moving axis, and the number is the fixed orthogonal coordinate. This rules out a broad x/y naming inversion. The Y-only TXT failure is primarily explained by raster acquisition gap and thermal-field evolution.

### Historical NCC Failure Audit

The original coordinate-adjacent NCC result is preserved only as a failure audit:

| Diagnostic | Value | Current interpretation |
|---|---:|---|
| Coordinate-adjacent NCC theta | 34.05 deg y-up, CI [33.23, 34.83] deg | contaminated coordinate-neighbor model |
| 47.6 deg inside old CI | No | do not update theta from this fit |
| Single-rotation RMS residual | 0.1567 px | local residual, not an SR threshold |
| Projection linearity R2 | 0.0001 | old global coordinate-neighbor model failed |
| Valid repeat pairs | 0 / 2 | no usable repeatability calibration |
| Y high-pass visible 4um/2um | 0.638 | Y-only coordinate neighbors fail monotonicity |

These values do not reinstate the old theta-success/failure storyline, do not update theta, and do not constitute a no-go conclusion for contour-level SR.

## Data-Driven Alignment Evidence

Existing EP05 alignment scores were available and were read as downstream alignment evidence:

| Method | Median holdout Chamfer (px) | Median gradient corr |
|---|---:|---:|
| No alignment | 0.3813 | 0.7023 |
| Stage prior only | 0.2402 | 0.8817 |
| Filename affine prior | 0.1708 | 0.9551 |
| Data-driven NCC init | 0.1563 | 0.9668 |
| Data-driven contour refined | 0.1341 | 0.9487 |

Data-driven contour refinement reduces median holdout Chamfer by **44.2%** relative to stage-prior-only alignment, and by **64.8%** relative to no alignment. This is the quantitative reason EP02 treats stage/filename coordinates as priors while using data-driven contour/NCC metrics to support alignment evidence, anchor selection, and quality gating.

## Decision Table

| Evidence | Use for | Do not use for |
|---|---|---|
| Stage/filename coordinate prior | coverage planning, initialization, regularization | alignment truth or success metric |
| Data-driven contour/NCC alignment | alignment anchor and quality gate before 2x contour-level SR | replacing the physical coordinate system |
| X time-adjacent small steps | local direction and short-time linearity smoke test | global SR feasibility claim or absolute stage-amplitude truth |
| Y coordinate-adjacent pairs | raster-path failure diagnosis and coordinate metadata | Y displacement calibration |
| AVI continuous scans | auxiliary direction and naming sanity check | SR input or high-precision theta replacement |

## Outputs

Primary notebook figures:

- `output/ep02_displacement_calibration/ep02_raster_acquisition_path.png`
- `output/ep02_displacement_calibration/ep02_stage_prior_coverage.png`
- `output/ep02_displacement_calibration/ep02_small_step_smoke_tests.png`
- `output/ep02_displacement_calibration/ep02_data_driven_alignment_comparison.png`
- `output/ep02_displacement_calibration/avi_theta_forest_plot.png`

Primary tables:

- `output/ep02_displacement_calibration/time_adjacent_method_summary.csv`
- `output/ep02_displacement_calibration/time_adjacent_method_measurements.csv`
- `output/ep02_displacement_calibration/y_coordinate_method_summary.csv`
- `output/ep02_displacement_calibration/y_coordinate_method_measurements.csv`
- `output/ep02_displacement_calibration/avi_theta_summary.csv`
- `output/ep02_displacement_calibration/avi_txt_xline_match_summary.csv`
- `output/ep02_displacement_calibration/avi_txt_yline_match_summary.csv`
- `output/ep02_displacement_calibration/ep02_data_driven_alignment_comparison.csv`
- `output/ep02_displacement_calibration/ep02_alignment_evidence_decision_table.csv`

## Conclusion

EP02 provides raster path reconstruction, detector-space coordinate prior coverage, bounded small-step diagnostics, and auxiliary AVI/TXT consistency checks. The correct next step is data-driven alignment on the main session, followed by quality-gated 2x contour-level SR.
