# EP01 — SR Data Basis and Main-Session Model

## Scope

EP01 audits the raw LWIR TXT/BMP dataset and turns it into a reproducible input model for micro-scan super-resolution. The goal is not to decide SR success or failure; it is to define which frames can be used together, what time order they have, and how session-level temperature drift constrains reconstruction.

## Executable Summary

| Metric | Value | SR use |
| --- | --- | --- |
| Raw TXT/BMP files | 263 TXT, 263 BMP, 263 paired | Use TXT as SR input; BMP is visual reference. |
| Matrix size | 480 x 640 | All frames share one detector grid. |
| Total frames | 263 | Full audit population before session gating. |
| Main session frames | session 2: 255 | Default input set for micro-scan SR. |
| Session temperature jump scale | 2.91 deg C median, 4.16 deg C max (40x / 57x noise floor) | Cross-session frames should not be mixed. |
| Main-session drift span | 0.62 deg C | SR alignment should model frames within this temperature band. |
| Coordinate coverage | 253/256 total; 253/256 in main session | Main session covers the full usable coordinate grid. |

## Frame Inventory

All `263` TXT thermal matrices are readable `480 x 640` arrays with no NaN/Inf frames, and all have matching BMP companions. TXT remains the numerical input for SR; BMP is retained as same-name visual reference only.

Coordinate/repeat coverage:

| frames_per_coordinate | n_coordinates |
| --- | --- |
| 3 | 4 |
| 2 | 2 |
| 1 | 247 |

Repeat-ID distribution:

| R | n_frames | n_unique_coordinates |
| --- | --- | --- |
| 0 | 253 | 253 |
| 1 | 4 | 4 |
| 2 | 6 | 6 |

The dataset contains `253/256` actual coordinates. Missing coordinates are `[(14, 6), (16, 6), (16, 16)]`. These gaps are coordinate-level absences, not merely missing `R=0` repeats.

## Acquisition Order and Sessions

Filename order is not acquisition order. Sorting by renamed filename produces `13` apparent temperature sessions because repeat and early frames are interleaved with the raster grid. Sorting by file modification time recovers `3` physical temperature segments:

| session | n_frames | first_order | last_order | first_file | last_file | mean_temp | median_temp | min_mean_temp | max_mean_temp | frame_median_temp | robust_temp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 0 | 0 | 0_0_1.txt | 0_0_1.txt | 21.757 | 21.757 | 21.757 | 21.757 | 21.54 | 21.673 |
| 1 | 7 | 1 | 7 | 0_0_0.txt | 8_0_0.txt | 19.862 | 19.852 | 19.69 | 20.095 | 19.607 | 19.756 |
| 2 | 255 | 8 | 262 | 8_0_1.txt | 10_0_2.txt | 23.292 | 23.263 | 23.226 | 23.847 | 23.028 | 23.174 |

Boundary jumps in acquisition order:

| boundary_after_order | from_session | to_session | from_file | to_file | delta_mean_C | abs_delta_mean_C |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 1 | 0_0_1.txt | 0_0_0.txt | -1.662 | 1.662 |
| 7 | 1 | 2 | 8_0_0.txt | 8_0_1.txt | 4.157 | 4.157 |

The main session is session `2` with `255` frames. It spans acquisition orders `8` to `262`, covers `253/256` coordinates, and has a mean-temperature span of `0.621` deg C within the session.

## SR Input Rule

Downstream SR should inherit `frame_audit.csv` and use `acquisition_order`, `session`, and `is_main_session` as the frame-selection contract. The default 2x contour-level SR POC input is the `255`-frame main session. Stage/filename coordinates are useful as command priors for initialization or regularization, but actual alignment must be constrained by image data and later EP04 localization quality gates.

Cross-session frames should not be mixed into one reconstruction pass. The detected session-boundary jumps are `2.91` deg C median and `4.16` deg C max, which are about `40x` and `57x` the `0.0724` deg C noise floor.

## Output Files

- `frame_audit.csv`
- `acquisition_order_audit.csv`
- `sr_data_basis_summary.csv`
- `coordinate_coverage_map.png`
- `frame_temperature_statistics.png`
- `robust_temperature_timeline.png`
- `order_comparison.png`
- `session_detection.png`
- `session_coordinate_coverage.png`
