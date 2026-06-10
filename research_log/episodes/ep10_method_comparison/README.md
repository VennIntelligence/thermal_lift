# EP10 — MAP-TV Parameter Sweep / Method Comparison

## Goal

Compare Drizzle, MAP-TV, and MAP-TGV candidates for the 2x highpass-domain
contour-level SR POC, with MAP-TV retaining a systematic CPU sweep over
`lambda_tv` and Gaussian PSF `sigma`.

## Scope

- Reuse EP06 code without modifying EP06 files.
- Real input is the EP06 248 clean-frame highpass stack plus EP05
  `contour_refined` alignment shifts.
- Validate each parameter with split-half NRMSE, holdout highpass-domain
  forward residual, artifact score, and raw-control agreement.

## Artifacts

- Script: `algos/ep10_map_tv_sweep/scripts/run_sweep.py`
- Drizzle script: `algos/ep10_drizzle/scripts/run_drizzle.py`
- TGV script: `algos/ep10_tgv_sr/scripts/run_tgv_sr.py`
- Output: `output/ep10_map_tv_sweep/`
- Cross-method notebook: `notebooks/ep10_method_comparison/`

## Notes

The full 28-point sweep is compute-heavy because each parameter includes one
248-frame reconstruction, five split-half pairs, and one 80% holdout reconstruction.
The script writes partial CSV rows after each completed parameter so runs can
resume safely.

2026-05-21 quality-review fixes:

- MAP-TV sweep logic was moved into importable `src/ep10_map_tv_sweep/` modules.
- MAP-TV now has local tests and writes per-split/per-holdout detail CSVs on
  future runs.
- Input caches now include parameter/file signatures to prevent silent stale
  reuse when alignment or highpass settings change.
- Drizzle future runs write artifact scores without the LR overshoot component,
  keeping the overshoot-inclusive score only as a debug column.
- TGV records backend provenance so CCPi and fallback paths are auditable.
- The EP10 notebook now builds as a three-algorithm visual comparison and writes
  CVPR-style comparison figures to `output/ep10_method_comparison/`.

2026-05-21 follow-up:

- Project plotting now keeps executed notebook inline figures at 300 dpi instead
  of downsampling previews to 100 dpi.
- Drizzle now supports exploratory `--scale 4`; the 4x run writes to
  `output/ep10_drizzle_4x/`.
- The EP10 notebook now includes intermediate diagnostics and a 2x-vs-4x
  center 1/3 crop comparison. The 4x view is explicitly treated as contour
  oversampling / visualization support, not a 2.5 um physical-resolution claim.
