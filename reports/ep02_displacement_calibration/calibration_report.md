# EP02 Calibration Report

## Scope

This report uses same-session adjacent TXT thermal frames from EP01 and estimates image displacement with NCC plus quadratic peak fitting.
The calibration set is restricted to EP01's main acquisition session (`session=2`), not filename-sorted pseudo-sessions.

## Main Result

The current NCC measurements do **not** validate the previous theta value of 47.6 deg.

| Metric | Value |
|---|---:|
| Main-session adjacent frame pairs | 463 |
| NCC fit-ok rate | 1.000 |
| Median NCC peak | 0.99545 |
| Image-row theta fit | 145.690 deg |
| Y-up diagnostic theta fit | 34.053 deg |
| Y-up 95% bootstrap CI | [33.226, 34.829] deg |
| Reference theta inside CI | False |
| Rotation-model RMS residual | 0.1567 px |
| Valid repeatability pairs | 0 / 2 main-session pairs |
| Repeatability median error (valid only) | n/a |
| Repeatability p95 error (valid only) | n/a |
| Linearity projection R2 | 0.0001 |

## Interpretation

NCC produces high correlation peaks, but the measured displacement field is inconsistent with a single rigid rotation and a fixed 20.0 um/px scale. The clearest remaining failure mode is the Y-scan: 2 um and 4 um command steps do not scale linearly in the measured displacement.

Therefore EP02 should be treated as a failed independent validation of theta rather than a replacement calibration. The global `configs/stage_calibration.json` should remain unchanged until the displacement measurement method is improved or independently checked.

## Acquisition-Order and Motion Direction Check

EP01 found that the earlier 13-session interpretation was a filename-sorting artifact. Using acquisition time collapses the data into a short warm-up/repeat segment and one 255-frame main scan. EP02 now uses only that main scan.

Within the main scan, X-scan median directions span -35.24 deg to -31.54 deg in y-up coordinates, and Y-scan median directions span 52.35 deg to 54.26 deg. The larger anomaly is magnitude: Y-scan 2 um steps have median displacement 0.3515 px, while Y-scan 4 um steps have median displacement 0.2262 px. A 4 um command should not produce a smaller displacement than a 2 um command.

This means the low-temperature jump frames were not the main cause of the Y-scan failure. The next check should focus on Y-scan frame-pair construction, scan reversal/backlash, axis sign conventions, or NCC bias under thermal-field changes.

## SR Impact

The fitted model RMS residual is 0.1567 px. This exceeds the 0.1 px practical threshold used for 2x SR feasibility and is above the 0.05 px target for 4x SR. Repeatability is also weakly constrained because only 0 main-session repeat pairs avoid boundary peaks. Current displacement evidence is insufficient for reliable SR reconstruction.

## Output Files

- `frame_pairs.csv`
- `displacement_measurements.csv`
- `motion_direction_diagnostic.csv`
- `theta_estimate.json`
- `repeatability.csv`
- `linearity.csv`
- `displacement_vector_field.png`
- `dx_dy_vs_coordinates.png`
- `motion_direction_diagnostic.png`
- `theta_bootstrap.png`
- `theta_residuals.png`
- `repeatability_boxplot.png`
- `linearity_regression.png`
- `sr_impact_summary.png`
