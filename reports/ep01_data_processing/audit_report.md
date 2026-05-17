# EP01 Data Audit Report

## Scope

This report validates the renamed TXT thermal matrices, their BMP companions, coordinate decoding, and acquisition-order session structure before super-resolution or displacement calibration work.

## Main Result

The TXT data are readable and internally complete. Coordinate decoding is consistent with the original naming rules and with the acquisition-time scan order. The previous 13-session interpretation came from sorting by renamed filename rather than by acquisition time.

| Metric | Value |
|---|---:|
| TXT frames | 263 |
| BMP files | 263 |
| TXT/BMP paired files | 263 |
| Frame shape | 480 x 640 |
| NaN / Inf frames | 0 / 0 |
| Unique coordinates | 253 |
| Missing coordinates | 3 |
| 3-repeat coordinates | 4 |
| 2-repeat coordinates | 2 |
| Temperature range | [18.21, 26.80] deg C |
| Mean-temperature range | [19.69, 23.85] deg C |
| Filename-order sessions | 13 |
| Acquisition-order sessions | 3 |
| Main acquisition session | 2 (255 frames) |
| Main session mean temperature | 23.292 deg C |
| R=0 row-order mismatches | 0 |

## Interpretation

All 263 TXT frames are valid 480 x 640 matrices with no NaN or Inf values, and all TXT files have matching BMP companions. Coordinate coverage is high at 253 / 256 positions, with three full coordinate gaps: (14,6), (16,6), and (16,16).

The renamed filename order is not the acquisition order. Sorting by filename produces 13 apparent sessions by inserting repeat and warm-up frames into the middle of the main scan. Sorting by file modification time recovers the physical acquisition order: a few early low-temperature/repeat frames followed by a 255-frame main scan near 23.3 deg C.

R=0 acquisition order follows the expected scan pattern: Y increases from 0 to 40 um, and X increases within each Y row. This supports the current coordinate decoding and points to ordering, not renaming, as the source of the earlier session artifact.

## Downstream Rule

EP02 and later SR work should use `session`, `is_main_session`, and `acquisition_order` from `frame_audit.csv`. The main displacement calibration should use only the main acquisition session unless there is a specific reason to analyze warm-up or repeat frames separately.

## Output Files

- `frame_audit.csv`
- `acquisition_order_audit.csv`
- `coordinate_coverage_map.png`
- `frame_temperature_statistics.png`
- `temperature_timeline.png`
- `session_detection.png`
