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

## Data-Driven Alignment Evidence

Existing EP05 alignment scores were available and were read as downstream alignment evidence:

| Method | Median holdout Chamfer (px) | Median gradient corr |
|---|---:|---:|
| No alignment | 0.3813 | 0.7023 |
| Stage prior only | 0.2402 | 0.8817 |
| Filename affine prior | 0.1708 | 0.9551 |
| Data-driven NCC init | 0.1563 | 0.9668 |
| Data-driven contour refined | 0.1341 | 0.9487 |

Data-driven contour refinement reduces median holdout Chamfer by **44.2%** relative to stage-prior-only alignment, and by **64.8%** relative to no alignment. This is the quantitative reason EP02 treats stage/filename coordinates as priors while reserving alignment truth for data-driven contour/NCC metrics.

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

Primary tables:

- `output/ep02_displacement_calibration/time_adjacent_method_measurements.csv`
- `output/ep02_displacement_calibration/y_coordinate_method_measurements.csv`
- `output/ep02_displacement_calibration/ep02_data_driven_alignment_comparison.csv`
- `output/ep02_displacement_calibration/ep02_alignment_evidence_decision_table.csv`

## Conclusion

EP02 provides raster path reconstruction, detector-space coordinate prior coverage, and bounded small-step diagnostics. The correct next step is data-driven alignment on the main session, followed by quality-gated 2x contour-level SR.
