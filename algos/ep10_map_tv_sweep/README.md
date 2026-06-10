# EP10 MAP-TV Sweep

Joint CPU sweep for MAP-TV regularization strength and Gaussian PSF sigma.

## Method

This experiment evaluates a 2x highpass-domain MAP-TV reconstruction on the
EP10 248 clean-frame real input set. The data term uses the EP06 matrix-free forward model:
shift the HR estimate into each LR frame, apply a Gaussian PSF, and compare
against the highpass-preprocessed LR observation. The prior is isotropic total
variation, applied as a proximal step inside the same FISTA-style outer loop as
EP06.

The optimized proxy objective is:

```text
0.5 * mean_i || A_i x - y_i ||_2^2 + lambda_tv * TV(x)
```

where `A_i` contains the contour-refined shift and Gaussian PSF for frame `i`.
The stage/alignment information is an input prior/anchor; holdout residual and
split-half consistency are quality proxies, not independent optical truth and
not proof of 5 um temperature metrology.

The reusable sweep logic lives in `src/ep10_map_tv_sweep/`; `scripts/run_sweep.py`
is a thin CLI wrapper.

```bash
cd /home/ujs/mycode/thermal_lift/algos/ep10_map_tv_sweep
uv sync
uv run python scripts/run_sweep.py --workers 1 --map-tv-workers 1
```

The default grid is:

- `lambda_tv`: `0.0001,0.0003,0.0005,0.001,0.002,0.005,0.01`
- `psf_sigma`: `0.10,0.18,0.30,0.50`

Outputs are written to `../../output/ep10_map_tv_sweep/`. The script writes
partial `sweep_results.csv` rows as each parameter finishes, so interrupted
runs can be resumed with the default `--resume` behavior.

Key outputs:

- `sweep_results.csv`: one row per lambda/sigma pair, including summary
  split-half, holdout, artifact, raw-control, runtime, and detail-file paths.
- `details/split_half_*.csv`: per split diagnostics for each parameter pair.
- `details/holdout_*.csv`: per holdout frame forward-residual diagnostics.
- `split_half_details.csv` and `holdout_details.csv`: concatenated detail tables.
- `cache/full_hr_*.npy`: ignored full HR highpass caches for every completed
  parameter pair; these are reproducible products, not Git-tracked source.
- `best_params.json`: Pareto frontier and top candidate metadata.
- `sweep_heatmap.png`: CVPR-style parameter heatmap.

The input cache records a signature of `data_dir`, `frame_audit_csv`,
`alignment_csv`, `alignment_method`, and highpass settings. If any of those
change, the cached input arrays are rebuilt instead of being silently reused.

For an auditable detached run:

```bash
cd /home/ujs/mycode/thermal_lift
setsid bash -c 'echo $$ > output/ep10_map_tv_sweep/full_run.pid; trap "rm -f output/ep10_map_tv_sweep/full_run.pid" EXIT; exec algos/ep10_map_tv_sweep/.venv/bin/python -u algos/ep10_map_tv_sweep/scripts/run_sweep.py --output-dir output/ep10_map_tv_sweep --workers 1 --map-tv-workers 1' \
  > output/ep10_map_tv_sweep/full_run.log 2>&1 < /dev/null &
```
