# EP02 AVI-TXT X-Line Match Addendum

## Scope

This addendum is the X-axis counterpart of `avi_txt_yline_match_addendum.md`. It checks whether x-scan AVI names match TXT fixed-Y coordinate rows.

Two hypotheses are tested:

- `xNum.avi -> TXT fixed Y=N`, then TXT frames vary along X.
- `xNum.avi -> TXT fixed X=N`, then TXT frames vary along Y.

For `x.avi`, `N=0` is assumed.

## Main Result

| Mapping hypothesis | Contour axis difference to AVI | High-pass NCC axis difference | Gradient NCC axis difference | Acquisition gap |
|---|---:|---:|---:|---:|
| `xNum.avi -> TXT fixed Y=N` | median 18.41 deg | median 14.19 deg | median 11.36 deg | median 1 frame |
| `xNum.avi -> TXT fixed X=N` | median 77.20 deg | median 74.77 deg | median 80.79 deg | median 16 frames |

Therefore `xNum.avi` matches TXT fixed-Y rows much better than TXT fixed-X columns. This is the symmetric naming rule: the AVI prefix names the moving axis, while the number identifies the fixed orthogonal coordinate.

## Important Difference From Y

The X-side correspondence is less clean than the Y-side correspondence. TXT fixed-Y rows have contour directions around 33-35 deg row-down, while x-scan AVI motion directions are around 51-52 deg row-down. This leaves a systematic axis difference of roughly 11-18 deg depending on method.

This does not invalidate the TXT X result. Unlike TXT fixed-X Y lines, TXT fixed-Y X rows are acquisition-time adjacent with median acquisition gap 1 frame. That is why X-step TXT NCC remains the strongest short-time motion diagnostic in EP02.

## Interpretation

The X AVI files support the expected classification:

- `x.avi` behaves like an X scan at fixed `Y=0`.
- `x2um.avi`, `x4um.avi`, ..., `x14um.avi` behave like X scans at fixed `Y=2/4/.../14`.

However, because AVI and TXT X directions differ by a systematic angle, x-scan AVI should not replace the TXT time-adjacent X diagnostic. It is useful for file-name interpretation and line classification, not for overwriting the TXT X displacement direction.

## Outputs

- `scripts/avi_txt_xline_match_check.py`
- `output/ep02_displacement_calibration/avi_txt_xline_match_summary.csv`
- `output/ep02_displacement_calibration/avi_txt_xline_pair_measurements.csv`
- `output/ep02_displacement_calibration/avi_txt_xline_contour_paths.csv`
- `output/ep02_displacement_calibration/avi_txt_xline_axis_match.png`
- `output/ep02_displacement_calibration/avi_txt_xline_projection_monotonicity.png`
