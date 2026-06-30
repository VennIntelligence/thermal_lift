# EP07 Solver V8 K4 Full-Halo Eval Archive

Date: 2026-06-30

This archive preserves the meaningful V8/K4 real-eval images before deleting
`algos/ep07_unet_sr/outputs/`.

## Source Run

- Output dir: `algos/ep07_unet_sr/outputs/solver_v8_k4_nodrizzle_fullhalo_eval`
- Checkpoints observed: `solver_step_005000.pt`, `solver_step_010000.pt`
- Key config from `solver_step_010000.pt`:
  - `unroll_steps=4`
  - `patch_size_hr=192`
  - `real_eval_overlap=128`
  - `real_eval_solver_mode=full_halo`
  - `real_eval_solver_halo_hr=96`
  - `solver_no_drizzle=True`

## Preserved Evidence

- `figures/direct_solver_step*_full_halo96_center_zoom3x_temperature.png`:
  direct PNGs saved by `real_eval.py`.
- `figures/tb_step*_eval_real_*.png`:
  image summaries exported from the original TensorBoard event file.
- `figures/compare_step10000_*.png`:
  same `solver_step_010000.pt` checkpoint rendered under multiple real-eval
  modes to separate checkpoint quality from rendering/inference口径:
  - `aligned_mean`: no solver, alignment baseline.
  - `tiled_p192_o128`: legacy/default tiled solver real-eval.
  - `dense_p192_o160`: denser small-grid tiled diagnostic口径.
  - `full_halo96`: full-frame outer-halo solver eval.
  Difference panels subtract `tiled_p192_o128`.
- `scalars/*.csv`:
  scalar series exported from the original TensorBoard event file.

## TensorBoard Tags

The comparison images were also written back to:

`algos/ep07_unet_sr/outputs/solver_v8_k4_nodrizzle_fullhalo_eval/tb_logs`

with tag prefix:

`eval_compare_v8k4_step10000/`

Important tags:

- `eval_compare_v8k4_step10000/temp_aligned_tiled_dense_fullhalo`
- `eval_compare_v8k4_step10000/highpass_aligned_tiled_dense_fullhalo`
- `eval_compare_v8k4_step10000/temp_diff_vs_tiled`
- `eval_compare_v8k4_step10000/highpass_diff_vs_tiled`

## Interpretation Note

The V8/K4 checkpoint was trained with the intended K4/no-drizzle parameters.
The visible TensorBoard result differs from older tiled views because the
real-eval renderer was explicitly switched to `full_halo96`. This changes the
evaluation/inference context, not the training batch, loss, optimizer, or saved
weights.
