# EP01 — SR Data Basis and Main-Session Model

## Scope

EP01 audits the raw LWIR TXT/BMP dataset and turns it into a reproducible input model for micro-scan super-resolution. The goal is not to decide SR success or failure; it is to define which frames can be used together, what time order they have, and how session-level temperature drift constrains reconstruction.

## Executable Summary

| Metric | Value | SR use |
| --- | --- | --- |
| Raw TXT/BMP files | 263 TXT, 263 BMP, 263 paired | Use TXT as SR input; BMP is visual reference. |
| Matrix size | 480 x 640 | All frames share one detector grid. |
| Total frames | 263 | Full audit population before session gating. |
| Raw main session frames | session 2: 255 | Physical temperature segment before repeat exclusion. |
| Clean SR input frames | 248 | Default input set for micro-scan SR after repeat exclusion. |
| Session temperature jump scale | 2.91 deg C median, 4.16 deg C max (40x / 57x noise floor) | Cross-session frames should not be mixed. |
| Main-session drift span | 0.54 deg C | Default input frames stay within this temperature band. |
| Coordinate coverage | 253/256 total; 253/256 raw main; 248/256 clean SR | Clean SR input covers the usable coordinate grid after repeat exclusion. |

## Frame Inventory

All `263` TXT thermal matrices are readable `480 x 640` arrays with no NaN/Inf frames, and all have matching BMP companions. TXT remains the numerical input for SR; BMP is retained as same-name visual reference only.

TXT/BMP pairing and rename provenance:

| Check | Value | Downstream meaning |
| --- | --- | --- |
| TXT matrices | 263 | Numerical LWIR temperature inputs for downstream audit. |
| BMP previews | 263 | Same-name visual references; not SR numerical input. |
| Paired stems | 263 | Frames with both TXT and BMP companions. |
| TXT without BMP | none | Still numerically readable, but harder to inspect visually. |
| BMP without TXT | none | Cannot be used as raw temperature input. |
| Rename provenance | missing: /home/ujs/mycode/thermal_lift/output/ep01_data_processing/rename_mapping.csv | Record provenance gap only; do not reconstruct raw names by guesswork. |

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

Repeat-frame exclusion summary:

| frame_role | repeat_exclusion_reason | sr_exclusion_reason | n_frames |
| --- | --- | --- | --- |
| other_session |  | not_raw_main_session | 5 |
| repeat_diagnostic | post_main_repeat | repeat_frame | 6 |
| repeat_diagnostic | prewarm_or_main_start_repeat | not_raw_main_session | 3 |
| repeat_diagnostic | prewarm_or_main_start_repeat | repeat_frame | 1 |
| sr_default |  |  | 248 |

The dataset contains `253/256` actual coordinates. Missing coordinates are `[(14, 6), (16, 6), (16, 16)]`. These gaps are coordinate-level absences, not merely missing `R=0` repeats.

Explicit missing-coordinate table:

| X_um | Y_um | all_frames | main_session_frames | configured_known_missing | status | downstream_handling |
| --- | --- | --- | --- | --- | --- | --- |
| 14 | 6 | 0 | 0 | yes | known coordinate-level absence | Exclude from usable coordinate grid; do not synthesize a frame. |
| 16 | 6 | 0 | 0 | yes | known coordinate-level absence | Exclude from usable coordinate grid; do not synthesize a frame. |
| 16 | 16 | 0 | 0 | yes | known coordinate-level absence | Exclude from usable coordinate grid; do not synthesize a frame. |

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

Boundary jumps compared with the `0.0724` deg C noise floor:

| boundary_after_order | to_order | transition | from_file | to_file | delta_mean_C | abs_delta_mean_C | noise_floor_C | abs_delta_over_noise_floor | diagnostic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | S0 -> S1 | 0_0_1.txt | 0_0_0.txt | -1.662 | 1.662 | 0.072 | 22.954 | thermal-state jump; keep sessions isolated |
| 7 | 8 | S1 -> S2 | 8_0_0.txt | 8_0_1.txt | 4.157 | 4.157 | 0.072 | 57.416 | thermal-state jump; keep sessions isolated |

The raw physical main session is session `2` with `255` frames. It spans acquisition orders `8` to `262` and covers `253/256` coordinates before repeat exclusion. The clean SR input set excludes all `R != 0` repeat frames and contains `248` frames across `248/256` coordinates. Its acquisition-order span is `9` to `256` and its mean-temperature span is `0.544` deg C.

R=0 raster row-order diagnostic:

| Y_um | first_order | last_order | gap_from_previous_row | n_r0_frames | first_X_um | last_X_um | missing_X_um | X_monotonic_in_acquisition | matches_expected_after_known_missing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 19 | start | 16 | 0 | 40 | none | True | True |
| 2 | 20 | 35 | 1 | 16 | 0 | 40 | none | True | True |
| 4 | 36 | 51 | 1 | 16 | 0 | 40 | none | True | True |
| 6 | 52 | 65 | 1 | 14 | 0 | 40 | 14, 16 | True | True |
| 8 | 66 | 81 | 1 | 16 | 0 | 40 | none | True | True |
| 10 | 82 | 97 | 1 | 16 | 0 | 40 | none | True | True |
| 12 | 98 | 113 | 1 | 16 | 0 | 40 | none | True | True |
| 14 | 114 | 129 | 1 | 16 | 0 | 40 | none | True | True |
| 16 | 130 | 144 | 1 | 15 | 0 | 40 | 16 | True | True |
| 18 | 145 | 160 | 1 | 16 | 0 | 40 | none | True | True |
| 20 | 161 | 176 | 1 | 16 | 0 | 40 | none | True | True |
| 24 | 177 | 192 | 1 | 16 | 0 | 40 | none | True | True |
| 28 | 193 | 208 | 1 | 16 | 0 | 40 | none | True | True |
| 32 | 209 | 224 | 1 | 16 | 0 | 40 | none | True | True |
| 36 | 225 | 240 | 1 | 16 | 0 | 40 | none | True | True |
| 40 | 241 | 256 | 1 | 16 | 0 | 40 | none | True | True |

## SR Input Rule

Downstream SR should inherit `frame_audit.csv` and use `acquisition_order` plus `is_sr_usable == True` as the frame-selection contract. `session == 2` remains the raw physical temperature segment (`255` frames), while `is_sr_usable` / the compatibility alias `is_main_session` define the repeat-excluded clean SR input (`248` frames). Stage/filename coordinates are useful as command priors for initialization or regularization, but actual alignment must be constrained by image data and later EP04 localization quality gates.

Cross-session frames should not be mixed into one reconstruction pass. The detected session-boundary jumps are `2.91` deg C median and `4.16` deg C max, which are about `40x` and `57x` the `0.0724` deg C noise floor.

`frame_audit.csv` downstream contract:

| Column | Meaning | Downstream use | Contract status |
| --- | --- | --- | --- |
| file | Standard X_Y_R TXT filename. | Join key for raw TXT/BMP files and later per-frame metrics. | required |
| X, Y, R | Stage command coordinate and repeat ID parsed from filename. | Command prior and grid bookkeeping only; not alignment truth. | required |
| filename_order | Alphabetical order after renaming. | Audit/debug only; do not use for session detection or timelines. | required |
| mtime | Filesystem modification time used to recover capture order. | Provenance for acquisition_order. | required |
| acquisition_order | Zero-based frame order sorted by mtime then filename. | Canonical time axis for sessions, raster diagnostics, and frame selection. | required |
| rows, cols | Detector matrix shape. | Validate that all TXT frames share the 480 x 640 grid. | required |
| T_min, T_max, T_mean, T_std | Basic per-frame temperature statistics. | Detect bad frames and session-level thermal jumps. | required |
| T_q05, T_median, T_q95, T_robust | Robust per-frame temperature statistics. | Check that session conclusions are not driven by extreme pixels. | required |
| session | Temperature segment ID detected in acquisition order. | Physical thermal-state diagnostic; do not use alone for SR input after repeat exclusion. | required |
| session_source | Method used to assign the session field. | Provenance guard against filename-order session artifacts. | required |
| is_raw_main_session | Boolean flag for the largest acquisition-order temperature segment before repeat exclusion. | Preserves the 255-frame physical session-2 definition for diagnostics. | required |
| is_repeat_frame | Boolean flag for nonzero repeat IDs (R != 0). | Exclude from downstream SR inputs; keep only for repeat diagnostics. | required |
| has_repeat_sibling | Whether the same (X, Y) coordinate has more than one repeat ID in the full audit. | Use for repeat-acquisition diagnostics and provenance checks. | required |
| repeat_exclusion_reason | Reason a repeat frame is excluded from clean SR input. | Documents prewarm/main/post-main repeat handling without deleting audit rows. | required |
| is_clean_main_session | Boolean flag for raw main-session frames after repeat-frame exclusion. | Clean thermal baseline for later alignment and SR work. | required |
| is_sr_usable | Boolean flag for the default downstream SR input set. | Primary frame-selection gate for reconstruction code. | required |
| sr_input_index | Zero-based index within is_sr_usable frames sorted by acquisition_order. | Stable per-frame index for clean SR inputs; blank for excluded frames. | required |
| sr_exclusion_reason | Reason a frame is outside the default SR input set. | Audit trail for repeat, non-main-session, and invalid-frame exclusion. | required |
| frame_role | Human-readable role assigned by EP01. | Quickly separates sr_default, repeat_diagnostic, and non-main diagnostic frames. | required |
| is_main_session | Compatibility alias for is_sr_usable after repeat exclusion. | Legacy loaders that filter is_main_session now receive the clean SR input set. | required |

## Output Files

Rebuild cache with `uv run python scripts/build_ep01_cache.py`.

- `frame_audit.csv`
- `acquisition_order_audit.csv`
- `sr_data_basis_summary.csv`
- `coordinate_coverage_map.png`
- `frame_temperature_statistics.png`
- `robust_temperature_timeline.png`
- `order_comparison.png`
- `repeat_exclusion_order_comparison.png`
- `session_detection_a.png`
- `session_detection_b.png`
- `session_coordinate_coverage.png`
