# EP02 AVI theta Independent Validation Addendum

## Scope

This addendum records Phase 9 of EP02. It reinterprets the AVI continuous-motion direction measurements from Phase 6 as independent estimates of the stage rotation angle theta.

The AVI files are still not valid SR inputs: they are rendered 8-bit videos, contain about 67% repeated frames, and do not preserve the raw temperature matrix. Their value here is narrower: they provide continuous-motion direction evidence from 16 separate AVI scans.

## Inputs

- High-pass NCC direction summary: `output/ep02_displacement_calibration/avi_direction_summary.csv`
- Gradient NCC direction summary: `output/ep02_displacement_calibration/avi_gradient_check/avi_direction_summary.csv`
- High-pass NCC frame-pair table: `output/ep02_displacement_calibration/avi_registration_pairs.csv`
- Gradient NCC frame-pair table: `output/ep02_displacement_calibration/avi_gradient_check/avi_registration_pairs.csv`

## Physical Model

- X-scan AVI: stage moves along X axis, so image direction = theta
- Y-scan AVI: stage moves along Y axis, so image direction = theta + 90 deg
- Therefore: theta_from_Y = Y-scan direction - 90 deg

The analysis uses the row-down image angle convention from the AVI registration scripts. Per-AVI confidence intervals in the forest plot are estimated from the robust angular spread within each motion segment. Summary confidence intervals are normal-approximation 95% confidence intervals for the mean across AVI-level theta estimates.

## Outputs

- Script: `scripts/avi_theta_estimation.py`
- Estimates: `output/ep02_displacement_calibration/avi_theta_estimates.csv`
- Summary: `output/ep02_displacement_calibration/avi_theta_summary.csv`
- JSON result: `output/ep02_displacement_calibration/avi_theta_result.json`
- Forest plot: `output/ep02_displacement_calibration/avi_theta_forest_plot.png`

## Summary Results

| Method | Source | Mean theta | Median theta | 95% CI | Range | N | Covers 47.6 deg |
|--------|--------|-----------:|-------------:|--------|-------|--:|-----------------|
| gradient | X-scan only | 48.67 deg | 48.70 deg | [48.59, 48.76] deg | 48.49-48.82 deg | 8 | No |
| gradient | Y-scan only | 45.61 deg | 45.63 deg | [45.54, 45.67] deg | 45.47-45.73 deg | 8 | No |
| gradient | combined | 47.14 deg | 47.11 deg | [46.36, 47.92] deg | 45.47-48.82 deg | 16 | Yes |
| highpass | X-scan only | 51.60 deg | 51.67 deg | [51.43, 51.77] deg | 51.15-51.84 deg | 8 | No |
| highpass | Y-scan only | 41.03 deg | 41.07 deg | [40.82, 41.24] deg | 40.56-41.37 deg | 8 | No |
| highpass | combined | 46.32 deg | 46.26 deg | [43.64, 49.00] deg | 40.56-51.84 deg | 16 | Yes |

The selected Phase 9 result is the gradient combined estimate:

| Item | Value |
|------|------:|
| Best theta estimate | 47.14 deg |
| Median theta estimate | 47.11 deg |
| 95% CI | [46.36, 47.92] deg |
| Reference theta | 47.60 deg |
| Absolute difference from reference | 0.46 deg |
| Samples | 16 AVI files |
| 47.6 deg inside CI | Yes |

## Per-AVI Gradient Estimates

| AVI | Scan axis | Direction-derived theta | Cumulative-slope theta |
|-----|-----------|------------------------:|-----------------------:|
| x.avi | X | 48.53 deg | 48.53 deg |
| x2um.avi | X | 48.49 deg | 48.58 deg |
| x4um.avi | X | 48.64 deg | 48.65 deg |
| x6um.avi | X | 48.76 deg | 48.72 deg |
| x8um.avi | X | 48.59 deg | 48.73 deg |
| x10um.avi | X | 48.79 deg | 48.77 deg |
| x12um.avi | X | 48.82 deg | 48.78 deg |
| x14um.avi | X | 48.76 deg | 48.81 deg |
| y0um.avi | Y | 45.62 deg | 45.77 deg |
| y2um.avi | Y | 45.73 deg | 45.69 deg |
| y4um.avi | Y | 45.62 deg | 45.66 deg |
| y6um.avi | Y | 45.67 deg | 45.68 deg |
| y8um.avi | Y | 45.64 deg | 45.65 deg |
| y10um.avi | Y | 45.65 deg | 45.60 deg |
| y12um.avi | Y | 45.47 deg | 45.54 deg |
| y14um.avi | Y | 45.47 deg | 45.63 deg |

## Interpretation

The gradient NCC result is the cleanest Phase 9 evidence because it tracks edge/contour structure instead of broad thermal texture. Its combined mean is 47.14 deg, only 0.46 deg away from the established 47.6 deg configuration. This is the first independent directional validation of theta in EP02.

The X-scan and Y-scan estimates differ by about 3.06 deg: gradient X-only mean is 48.67 deg, while gradient Y-only mean is 45.61 deg. The two groups bracket 47.6 deg. This pattern is consistent with a small systematic bias in the AVI-derived geometry rather than random registration failure, because each group is internally very tight: 0.13 deg standard deviation for X-scan and 0.09 deg for Y-scan.

High-pass NCC is directionally stable within each scan group, but it has a much larger X/Y split: high-pass X-only median is 51.67 deg and high-pass Y-only median is 41.07 deg. That method is useful as a diagnostic cross-check, but the Phase 9 decision should rely on gradient NCC.

The cumulative-slope estimates agree closely with the median direction estimates for the gradient method. This indicates that the per-frame direction statistic is not being dominated by isolated noisy frame pairs; the integrated motion path points in the same direction.

## Decision

Theta = 47.6 deg retains its status in `configs/stage_calibration.json`.

The AVI validation supports the current theta configuration, but it should not replace it. AVI files are rendered videos rather than raw temperature matrices, and the X/Y split of about 3 deg indicates residual systematic error in the video-derived geometry or registration method. Phase 9 should therefore be recorded as auxiliary independent evidence, not as a higher-precision configuration source.

## Reproduction

Run:

```bash
uv run python scripts/avi_theta_estimation.py
```

Expected key stdout:

```text
method/source: gradient / combined
theta mean: 47.141 deg
95% CI: [46.364, 47.918] deg
n: 16
47.6 deg within CI: True
```
