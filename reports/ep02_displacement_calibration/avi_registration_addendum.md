# EP02 AVI Registration Addendum

## Scope

This addendum analyzes the continuous-motion AVI files as diagnostic evidence for stage motion. It does not test X/Y stage orthogonality, which is treated as an instrument guarantee. The AVI frames are cropped from 839x560 rendered frames to the 640x480 thermal image area before registration.

## Method

- Script: `scripts/avi_y_direction_check.py`
- Crop: `x=50, y=45, width=640, height=480`
- Duplicate removal: consecutive-frame mean absolute difference threshold = 0.3 gray levels
- Main registration: high-pass NCC, search radius = 40 px
- Contour check: gradient NCC, written to `output/ep02_displacement_calibration/avi_gradient_check/`

## High-Pass NCC Results

| Metric | X-scan AVI | Y-scan AVI |
|---|---:|---:|
| AVI count | 8 | 8 |
| Median duplicate-frame rate | 66.9% | 67.2% |
| Median motion pairs per AVI | 202 | 198 |
| Median direction, row-down | 51.67 deg | 131.07 deg |
| Direction range, row-down | 51.15-51.84 deg | 130.56-131.37 deg |
| Median frame-to-frame magnitude | 0.0884 px | 0.0758 px |
| Minimum path straightness | 0.9997 | 0.9847 |
| Median NCC peak | 0.9120 | 0.9099 |

## Gradient NCC Contour Check

| Metric | X-scan AVI | Y-scan AVI |
|---|---:|---:|
| Median direction, row-down | 48.70 deg | 135.63 deg |
| Direction range, row-down | 48.49-48.82 deg | 135.47-135.73 deg |
| Median frame-to-frame magnitude | 0.1103 px | 0.0943 px |
| Minimum path straightness | 0.9997 | 0.9826 |

## Interpretation

The AVI sequences contain clean continuous motion segments. Both X-scan and Y-scan videos show stable direction, stable speed, no search-boundary hits, and nearly straight cumulative paths.

The absolute direction differs by preprocessing method: raw/high-pass tracks thermal texture, while gradient emphasizes contours. Therefore the AVI direction angles should not directly replace `configs/stage_calibration.json`. Their main value is diagnostic: Y-scan AVI motion is coherent, so the TXT Y-only coordinate-pair failure is more consistent with non-time-adjacent frame selection and thermal-field evolution than with a completely unusable Y axis.

`y14um.avi` is the mild outlier, with lower path straightness than the other Y videos, but it does not change the overall conclusion.

## Outputs

- `output/ep02_displacement_calibration/avi_direction_summary.csv`
- `output/ep02_displacement_calibration/avi_registration_pairs.csv`
- `output/ep02_displacement_calibration/avi_direction_comparison.png`
- `output/ep02_displacement_calibration/avi_cumulative_motion_paths.png`
- `output/ep02_displacement_calibration/avi_y0um_displacement_timeseries.png`
- `output/ep02_displacement_calibration/avi_gradient_check/`
