# Stage 0h Remote Report

## Executive Summary

Stage 0h completed with refined alignment as the repo default. Step 0 self-check passed: `git pull --ff-only` returned `Already up to date`, EP09 tests reported `11 passed`, and `default_contour_alignment_csv()` printed `/home/ujs/thermal_lift/configs/alignment/stage0f_refined_alignment.csv` (source: `logs/step0_self_check_clean.log`).

**Main result:** Task 1 did not match the expected "V11 only" offset pattern. C and D also showed large same-half offsets against drizzle: C `0.621 HR px`, D `0.628 HR px`, both above the `0.2 px` alert line (source: `summary/key_task1_offset_probe.csv`). This means the earlier neural-arm content-destruction verdict must be reopened for all V11/C/D, not only V11.

After refined-alignment rerender, the neural outputs still show a rigid offset: V11 `0.711 HR px`, C `0.691 HR px`, D `0.711 HR px`; TGV/MAP-TV anchors are near zero at `0.005`/`0.007 HR px` (source: `summary/key_task3_rerender_offset_probe.csv`). Therefore the neural offset is not fixed merely by switching the real-data DC input to refined shifts; it likely lives in the neural render/grid convention and needs a separate fix.

On the official v3 leaderboard, neural cross-FRC did recover relative to Stage 0f, but remains far below the classical arm in the trusted 24-30um band. C and D remain effectively tied. Classical new baselines are TGV x drizzle cutoff `23.03um`, MAP-TV x drizzle cutoff `24.62um`, and TGV x MAP-TV has high 24-30um agreement (`@30um=0.881`, `@24um=0.730`), while 20um remains aperture-zero audit only (source: `summary/key_stage0f_vs_stage0h_cross.csv`, `summary/key_classic_stage0h_baseline.csv`).

No thresholds were changed, no configs were changed, no training was run, and nothing was pushed. Long-running stdout logs are included under `logs/`. The Stage 0g欠账 JSON is included at `backfill/stage0g_iter2_stage0a_summary.json`.

## Commands And Logs

| Step | Command / runner | Output | Stdout log | Final exit code |
|---|---|---|---|---:|
| Step 0 | `git pull --ff-only`; `uv run --with pytest pytest algos/ep09_psf_calibration/tests/ -q`; `default_contour_alignment_csv()` check | `Already up to date`; `11 passed`; `stage0f_refined_alignment.csv` | `logs/step0_self_check_clean.log` | `0` |
| Task 1 | `algos/ep15_info_limit/scripts/probe_pair_offset.py` on 0g same-half arrays, six pairs exactly as in task packet | `summary/task1_offset_probe_summary.csv`; `curves/task1_offset_probe_curves_long.csv` | `logs/task1_offset_probe.log` | `0` |
| Task 2 | `scripts/stage0h_reconstruct_halves.py --methods v11,C_nodr,D_dr01` in `algos/ep07_unet_sr` UV env | `output/stage0h_frc_recons/{v11,C_nodr,D_dr01}_{a,b}.npy`; manifest copied to `manifests/stage0h_render_manifest.json` | `logs/task2_neural_recons.log` | `0` |
| Task 3 classic | `scripts/stage0h_reconstruct_halves.py --methods tgv,maptv`, explicit 16-thread CPU env | `output/stage0h_frc_recons/{tgv,maptv}_{a,b}.npy`; manifest copied to `manifests/stage0h_render_manifest.json` | `logs/task3_classic_recons.log` | `0` |
| Task 3 leaderboard | Task packet v3 `run_real_split_frc_v2.py --methods none ...` command | `summary/task3_leaderboard_method_summary.csv`; `curves/task3_leaderboard_frc_curves_long.csv` | `logs/task3_leaderboard_v3.log` | `0` |
| Rerender offset probe | `probe_pair_offset.py` on new Stage 0h recons | `summary/task3_rerender_offset_probe_summary.csv`; `curves/task3_rerender_offset_probe_curves_long.csv` | `logs/task3_offset_probe_rerender.log` | `0` |

Resource note: the first classic run was started with `classic_workers=4` but was stopped before any half output because the CPU runtime looked too long; that partial stdout is preserved in `logs/task3_classic_recons_aborted_4core.log` and is not used in any reported number. The final classic run used explicit `OMP_NUM_THREADS=16`, `OPENBLAS_NUM_THREADS=16`, `MKL_NUM_THREADS=16`, `NUMEXPR_NUM_THREADS=16`, with `torch_threads=16` (source: `logs/task3_classic_recons.log`). All final long/background commands completed in tmux panes and printed `[tmux done] exit_code=0` in the retained pane output; the stdout captured by `tee` is copied under `logs/`.

## Task 1 Offset Probe

Source: `summary/key_task1_offset_probe.csv` derived from `summary/task1_offset_probe_summary.csv`.

| Pair | Offset norm (HR px) | Offset (dx,dy HR px) | Cutoff before -> after (um) | Sign changes before -> after | FRC@30 after | FRC@24 after | Read |
|---|---:|---:|---:|---:|---:|---:|---|
| V11 vs drizzle | `0.693` | `(+0.600,+0.347)` | `30.07 -> 25.82` | `14 -> 0` | `0.535` | `0.179` | Significant offset; oscillation collapses |
| C no-DR vs drizzle | `0.621` | `(+0.538,+0.310)` | `26.67 -> 25.82` | `2 -> 0` | `0.525` | `0.186` | <span style="color:red">RED FLAG: >=0.2px offset, C/D verdict must be reopened</span> |
| D DR0.1 vs drizzle | `0.628` | `(+0.531,+0.336)` | `26.83 -> 25.82` | `4 -> 0` | `0.548` | `0.186` | <span style="color:red">RED FLAG: >=0.2px offset, C/D verdict must be reopened</span> |
| TGV vs drizzle | `0.024` | `(-0.018,-0.015)` | `25.45 -> 25.45` | `0 -> 0` | `0.592` | `0.197` | Anchor near zero |
| MAP-TV vs drizzle | `0.034` | `(+0.030,-0.016)` | `26.28 -> 26.28` | `10 -> 12` | `0.509` | `0.068` | Anchor near zero in displacement |
| V11 vs TGV | `0.784` | `(+0.741,+0.255)` | `30.07 -> 20.00` | `41 -> 0` | `0.827` | `0.755` | Strong V11 grid offset signature |

Task 1 conclusion: the offset probe convicts a rigid grid offset for V11 and unexpectedly also for C/D. The classical anchors stay near zero against drizzle, so this is not a global probe failure.

## Task 2/3 Rerender Provenance

Neural rerender source: `logs/task2_neural_recons.log` and `manifests/stage0h_render_manifest.json`.

- Actual alignment CSV printed by the scratch renderer: `/home/ujs/thermal_lift/configs/alignment/stage0f_refined_alignment.csv`.
- V11 checkpoint: `algos/ep07_unet_sr/outputs/solver_v11_k2_p384_nogn_halo96_50k/solver_step_040000.pt`.
- C checkpoint: `algos/ep07_unet_sr/outputs/solver_v13_v6_nodr_ctrl/solver_final.pt`.
- D checkpoint: `algos/ep07_unet_sr/outputs/solver_v13_v6_dr01/solver_final.pt`.
- Neural render times: V11 A/B `6.56/5.45 sec`, C A/B `5.03/5.17 sec`, D A/B `5.17/5.37 sec` (source: `logs/task2_neural_recons.log`).

Classic rerender source: `logs/task3_classic_recons.log` and `manifests/stage0h_render_manifest.json`.

- Actual alignment CSV printed by the scratch renderer: `/home/ujs/thermal_lift/configs/alignment/stage0f_refined_alignment.csv`.
- TGV params: `lambda_tv=0.003`, `alpha_ratio=2.0`, `psf_sigma=0.5`, `max_iter=100`, `step_size=1.0`, `tgv_inner_iter=80`, `aniso_ratio_y=1.5`, `coverage_weighted=True`, `workers=16`.
- MAP-TV params: `lambda_tv=0.001`, `psf_sigma=0.2`, `max_iter=150`, `step_size=0.1`, `workers=16`.
- Classic render times: TGV A/B `534.12/544.20 sec`, MAP-TV A/B `163.59/170.36 sec` (source: `logs/task3_classic_recons.log`).

## Rerender Offset Probe

Source: `summary/key_task3_rerender_offset_probe.csv` derived from `summary/task3_rerender_offset_probe_summary.csv`.

| Pair | Offset norm (HR px) | Offset (dx,dy HR px) | Cutoff before -> after (um) | Sign changes before -> after | FRC@30 after | FRC@24 after | Read |
|---|---:|---:|---:|---:|---:|---:|---|
| V11 new vs drizzle | `0.711` | `(+0.587,+0.400)` | `29.97 -> 22.80` | `26 -> 0` | `0.670` | `0.385` | Offset persists after rerender |
| C new vs drizzle | `0.691` | `(+0.564,+0.398)` | `26.67 -> 25.38` | `21 -> 0` | `0.670` | `0.390` | Offset persists after rerender |
| D new vs drizzle | `0.711` | `(+0.583,+0.407)` | `26.67 -> 25.45` | `23 -> 0` | `0.680` | `0.385` | Offset persists after rerender |
| TGV new vs drizzle | `0.005` | `(-0.004,+0.004)` | `21.38 -> 21.38` | `0 -> 0` | `0.745` | `0.417` | Anchor remains near zero |
| MAP-TV new vs drizzle | `0.007` | `(+0.004,-0.005)` | `20.00 -> 20.00` | `0 -> 0` | `0.908` | `0.729` | Anchor remains near zero |
| V11 new vs TGV | `0.717` | `(+0.646,+0.309)` | `29.97 -> 20.00` | `28 -> 0` | `0.860` | `0.808` | Neural grid offset persists against TGV too |

Rerender conclusion: the neural offset did not disappear with refined-alignment rerender. This points to a neural output grid/render convention issue rather than the old real-data alignment CSV alone. TGV/MAP-TV against drizzle are effectively zero-offset controls.

## Leaderboard V3

Official v3 summary source: `summary/key_task3_leaderboard.csv`, original full table at `summary/task3_leaderboard_method_summary.csv`. Stage 0f comparison source: `summary/key_stage0f_vs_stage0h_cross.csv`.

### Neural Cross Recovery

| Method | Stage 0f FRC@30 | Stage 0h FRC@30 | Delta@30 | Stage 0f FRC@24 | Stage 0h FRC@24 | Delta@24 | Stage 0h cutoff (um) | Source |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| V11 x drizzle | `0.108` | `0.155` | `+0.047` | `0.107` | `0.203` | `+0.095` | `26.59` | `summary/key_stage0f_vs_stage0h_cross.csv` |
| C no-DR x drizzle | `0.098` | `0.207` | `+0.108` | `0.113` | `0.206` | `+0.092` | `26.67` | `summary/key_stage0f_vs_stage0h_cross.csv` |
| D DR0.1 x drizzle | `0.118` | `0.207` | `+0.089` | `0.109` | `0.203` | `+0.094` | `26.67` | `summary/key_stage0f_vs_stage0h_cross.csv` |

Answer to question (a): neural cross-FRC does recover under refined alignment, especially C/D at 30um, but the recovery is modest and still confounded by the persistent `~0.7 HR px` neural grid offset. Do not treat these uncorrected neural cross numbers as a final content verdict.

### C vs D

Source: `summary/key_stage0f_vs_stage0h_cross.csv`.

- C no-DR x drizzle: cutoff `26.67um`, FRC@30 `0.207`, FRC@24 `0.206`.
- D DR0.1 x drizzle: cutoff `26.67um`, FRC@30 `0.207`, FRC@24 `0.203`.

Answer to question (b): C vs D remains flat within this instrument. DR 0.1 still has no measurable real-domain advantage in the trusted 24-30um band.

### Classical New Baseline

Source: `summary/key_classic_stage0h_baseline.csv`.

| Method | 1/7 cutoff (um) | FRC@30 | FRC@24 | FRC@20 | Read |
|---|---:|---:|---:|---:|---|
| TGV self | `20.00` | `0.980` | `0.978` | `-0.201` | Self only; 20um aperture region not a resolution claim |
| MAP-TV self | `22.74` | `0.854` | `0.638` | `-0.966` | Self only |
| TGV same-half vs drizzle | `21.38` | `0.745` | `0.417` | `0.718` | Same-half control; 20um ignored |
| TGV x drizzle | `23.03` | `0.702` | `0.356` | `-0.720` | Main classical-vs-drizzle baseline |
| MAP-TV x drizzle | `24.62` | `0.690` | `0.296` | `-0.976` | Main classical-vs-drizzle baseline |
| TGV x MAP-TV | `20.00` | `0.881` | `0.730` | `-0.666` | Strong classical agreement at 24-30um; 20um ignored |

Answer to question (c): the refined-alignment classical baseline is now roughly `23-24.6um` cutoff against drizzle, with `FRC@30 ~= 0.69-0.70` and `FRC@24 ~= 0.30-0.36`. TGV x MAP-TV agreement is much stronger at 24-30um (`0.881/0.730`) but its nominal `20.00um` cutoff should not be read as a 20um claim because 20um is the detector aperture-zero/audit region.

For context, the refined drizzle phase-stratified self baseline is cutoff `26.28um`, FRC@30 `0.559`, FRC@24 `-0.001` (source: `summary/stage0g_refined_drizzle_method_summary.csv`).

## Decision Notes

1. **Task 1 RED FLAG:** C/D old same-half arrays also have `>=0.2px` offsets. The previous C/D "content replacement" conviction is not clean.
2. **Rerender result:** refined real-data DC input does not remove the neural grid offset. The next actionable item is a neural render/output-grid convention audit before using neural-vs-classical cross-FRC as a final content verdict.
3. **Neural cross:** refined alignment improves neural cross-FRC versus Stage 0f, but not enough to catch classical methods in the trusted 24-30um band.
4. **C vs D:** still flat; DR 0.1 remains null under this real-domain readout.
5. **Classical arm:** refined TGV/MAP-TV are the current baseline to beat: TGV x drizzle `23.03um`, MAP-TV x drizzle `24.62um`, and TGV x MAP-TV high agreement in 24-30um.
6. **20um discipline:** all 20um values are recorded but not used for decisions.

## Included Artifacts

- `summary/`: key CSVs, full method summaries, split balance/indices, Stage 0f baseline summary, Stage 0g refined drizzle summary.
- `curves/`: long FRC curve CSVs and per-pair curve CSV folders.
- `manifests/`: run manifests for offset probes, leaderboard, render provenance, and Stage 0g refined drizzle.
- `logs/`: stdout logs for every stage, including the aborted 4-core classic attempt.
- `backfill/stage0g_iter2_stage0a_summary.json`: required Stage 0g欠账 JSON.
- `scripts/stage0h_reconstruct_halves.py`: scratch script used for rerender provenance.
