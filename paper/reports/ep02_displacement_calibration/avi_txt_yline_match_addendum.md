# EP02 AVI-TXT Y-Line Match Addendum

## Scope

This addendum checks whether the y-scan AVI file names can be matched back to TXT coordinate lines. The test distinguishes two possible interpretations:

- `yNum.avi -> TXT fixed X=N`, then TXT frames vary along Y.
- `yNum.avi -> TXT fixed Y=N`, then TXT frames vary along X.

The comparison uses central contour centroids and high-pass/gradient NCC directions. Axis direction is compared modulo 180 degrees because the AVI motion segment may run opposite to increasing TXT coordinate order.

## Main Result

| Mapping hypothesis | Contour axis difference to AVI | High-pass NCC axis difference | Gradient NCC axis difference | Acquisition gap |
|---|---:|---:|---:|---:|
| `yNum.avi -> TXT fixed X=N` | median 1.86 deg | median 5.28 deg | median 6.24 deg | median 16 frames |
| `yNum.avi -> TXT fixed Y=N` | median 82.48 deg | median 85.78 deg | median 81.66 deg | median 1 frame |

Therefore `yNum.avi` matches TXT fixed-X lines, not TXT fixed-Y rows. This supports the intended interpretation that `y0um.avi`, `y2um.avi`, ..., `y14um.avi` are continuous Y scans at fixed X positions.

## TXT Fixed-X Line Behavior

TXT fixed-X contour paths are monotonic with Y coordinate for all tested AVI names:

| AVI | TXT line | First TXT | Last TXT | Contour axis difference to AVI | Median acquisition gap |
|---|---|---|---|---:|---:|
| `y0um.avi` | `X=0` | `0_2_0.txt` | `0_40_0.txt` | 1.86 deg | 16 |
| `y2um.avi` | `X=2` | `2_2_0.txt` | `2_40_0.txt` | 2.96 deg | 16 |
| `y4um.avi` | `X=4` | `4_2_0.txt` | `4_40_0.txt` | 1.17 deg | 16 |
| `y6um.avi` | `X=6` | `6_2_0.txt` | `6_40_0.txt` | 2.69 deg | 16 |
| `y8um.avi` | `X=8` | `8_2_0.txt` | `8_40_0.txt` | 2.08 deg | 16 |
| `y10um.avi` | `X=10` | `10_0_0.txt` | `10_40_0.txt` | 1.17 deg | 16 |
| `y12um.avi` | `X=12` | `12_0_0.txt` | `12_40_0.txt` | 1.85 deg | 16 |
| `y14um.avi` | `X=14` | `14_0_0.txt` | `14_40_0.txt` | 1.68 deg | 16, max 30 |

Known missing R=0 grid points are `(14, 6, 0)`, `(16, 6, 0)`, and `(16, 16, 0)`. The missing `14_6_0.txt` explains the larger max acquisition gap on the `X=14` fixed-X line.

## Interpretation

The AVI-TXT match rules out a broad Y naming inversion or row/column interpretation error. The TXT fixed-X lines have stable outer-contour motion along the same image axis as the corresponding y-scan AVI files.

The Y-only TXT NCC failure is therefore not a filename sorting problem. The failure comes from the raster acquisition pattern: adjacent Y coordinates on a fixed-X TXT line are separated by about one full row of acquisition time. During that gap, the thermal field changes enough to bias intensity-based NCC magnitude, even though the contour direction remains stable.

This means TXT Y coordinates remain valuable as commanded-position and ordering metadata. They should not be used as direct quantitative Y displacement measurements from coordinate-adjacent TXT NCC pairs.

## Outputs

- `scripts/avi_txt_yline_match_check.py`
- `output/ep02_displacement_calibration/avi_txt_yline_match_summary.csv`
- `output/ep02_displacement_calibration/avi_txt_yline_pair_measurements.csv`
- `output/ep02_displacement_calibration/avi_txt_yline_contour_paths.csv`
- `output/ep02_displacement_calibration/avi_txt_yline_axis_match.png`
- `output/ep02_displacement_calibration/avi_txt_yline_projection_monotonicity.png`
- `output/ep02_displacement_calibration/avi_txt_yline_contour_paths.png`
