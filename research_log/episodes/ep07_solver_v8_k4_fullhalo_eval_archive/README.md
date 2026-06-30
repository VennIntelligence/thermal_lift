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
- `figures/candidate_step10000_degrid_*.png`:
  post-render low-frequency graft candidates. These keep the tiled/dense
  high-frequency detail and replace only a Gaussian lowpass component with the
  aligned-mean lowpass.
- `figures/candidate_step10000_tile_halo_*.png`:
  per-tile outer-halo candidates. Each legacy `p192/o128` tile is solved with
  halo `32/64/96`, cropped back to the central 192 HR patch, then stitched.
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
- `eval_compare_v8k4_step10000/degrid_candidates_temp`
- `eval_compare_v8k4_step10000/degrid_candidates_highpass`
- `eval_compare_v8k4_step10000/tile_halo_candidates_temp`
- `eval_compare_v8k4_step10000/tile_halo_candidates_highpass`

## Interpretation Note

The V8/K4 checkpoint was trained with the intended K4/no-drizzle parameters.
The visible TensorBoard result differs from older tiled views because the
real-eval renderer was explicitly switched to `full_halo96`. This changes the
evaluation/inference context, not the training batch, loss, optimizer, or saved
weights.

User visual read after the first comparison:

- `aligned_mean` is the original blurry baseline.
- Legacy `tiled_p192_o128` is clear but has visible grid artifacts.
- Dense `p192/o160` looks close to the legacy tiled output.
- `full_halo96` removes the grid, but fine structural lines become swollen and
  show flocculent texture; this is visually unacceptable.

Mechanism hypothesis:

- The prox UNet is trained and supervised on 192 HR patches.
- It contains GroupNorm and SE attention, both of which depend on spatial
  statistics. Running the shared K4 prox on the entire full-frame halo field is
  therefore a distribution shift, not a neutral boundary-only change.
- K4 recurrent reuse can amplify this shifted prox response, explaining why
  `full_halo96` removes the tile boundary while worsening thin structures.

Decision implication: do not use `full_halo96` alone as the main K4 real-eval
selection view. Prefer tiled views for checkpoint quality judgment, then test
degrid/tile-halo candidates separately for removing the grid without changing
the learned structure prior too much.

Follow-up visual read after the tile-halo comparison:

- `tile_halo_single_temp/full_halo96`: worst flocculent texture; grid is gone,
  but thin structures swell and become visually poor.
- `tile_halo_single_temp/tile_halo96`: less flocculent than full-frame halo,
  but still worse than the gridded tiled render.
- `tile_halo_single_temp/tile_halo64`: least flocculent among halo variants,
  but a faint grid returns and the result is still not clean.
- `tile_halo_single_temp/tiled_p192_o128`: sharpest/cleanest structures, but
  with the strongest visible grid.

This establishes a clear tradeoff rather than a root fix: increasing solver
context suppresses the grid but moves the prox further from its 192-patch
training distribution; decreasing context preserves structure but keeps the
grid. The current evidence should be reviewed by a second agent before adding
more ad-hoc inference postprocessing.

## Second-agent root-cause diagnosis (2026-06-30)

See `diagnosis_20260630/DIAGNOSIS.md`. Offline analysis of the render arrays plus
an architecture extent-invariance test (`diag_extent.py`, no checkpoint needed)
confirms and quantifies the mechanism:

- The grid is a prox **patch-boundary + stitch** artifact: a seam comb present in
  every tiled variant, absent in `full_halo`, suppressed monotonically by larger
  solve context (NOT in the target; not the stitch window or DC rim).
- The flocculence/swelling/background-lift is the **GroupNorm + SE extent
  distribution shift** (user hypothesis CONFIRMED): a far-field perturbation 272 px
  away changes a window's prediction by ~1.4σ with GroupNorm+SE but **exactly 0**
  for a pure-conv UNet; SE and GroupNorm contribute ~equally. K4 recurrence
  amplifies it ~linearly (composes with ACL-037).
- `halo` is a single tradeoff knob (small=sharp+grid, large=clean-bg+shifted),
  not a fix. Prioritized experiments E1–E5 (offset-averaged tiling first; then
  multi-scale training / drop-SE / K2+damping / lowpass-anchor split) are in the
  report with pass/fail criteria. Reproducible metrics: `diag_arrays*.py`.
