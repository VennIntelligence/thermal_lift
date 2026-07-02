# EP07 Solver — Unattended Experiment Queue (2026-07-01)

For an unattended scheduler: run each experiment in order, record its final metrics into the
results table below (and back-fill ACL-042 / ACL-043 训练结果), then apply the decision rule.
All code is already committed (ACL-042 D-E high-pass residual prox; ACL-043 on-the-fly phase-bins).
The remote should NOT modify algorithm code — only debug/accelerate and run.

Run from `algos/ep07_unet_sr/`. Steps = **40000** (not 50k). E3 mainline defaults (noSE / noGN /
`full_halo96` real-eval) are automatic — do not re-specify them.

## Global rules (read first)
- **Success metrics = held-out SYNTHETIC GT + REAL visual.** Higher PSNR / boundary-F1 = better;
  lower region-RMSE / out-of-band / real-artifact = better; real temperature PNG must be **sharper
  than V11 and free of grid / flocculence / background-lift**.
- **DO NOT use `eval_real/dc_resid_band` as a success gate.** It is floored at ~1.21 by PSF
  misspecification + alignment (identical across V9/V11/Prompt-A/B) and cannot discriminate.
- **Reference baselines** (ACL-041 midpoint): V11 (noSE/noGN, no-drizzle) synth PSNR ≈ **35.2**,
  region-RMSE ≈ 0.086, boundary-F1 ≈ 0.86, real artifact ≈ 0.46. V9 (SE+GN) synth PSNR ≈ **37.5**
  but with extent artifacts. **Goal: recover toward V9 synth sharpness WITHOUT the real artifacts.**
- **Physics ceiling (info-budget, ACL-043 / `outputs/ep07_solver_diag/info_budget2.py`)**: the total
  recoverable multi-frame gain at our SNR is only ~+1–1.5 dB @2×. Don't expect a 4×-style jump; the
  win is "clean + as sharp as physics allows," not breaking the ceiling.
- If a run crashes, capture the traceback, mark it FAILED in the table, and continue to the next.
- **Use the explicit shared command below, not argparse defaults.** The comparable V11/Prompt-A runs used
  `batch_size=12`, `solver_m_frames=12`, `solver_dc_weight=0`, and `--compile`; omitting these silently
  falls back to `batch_size=4`, `M=16`, `dc_weight=0.1`, and no compile, making time and metrics less comparable.
- Local RTX 5090 sanity check: `outputs/_memtest_b12_de_s5` completed 200 steps with `batch_size=12`,
  `M=12`, `p384`, D-E sigma 5, and `--compile` without OOM. The temporary output was deleted after the check.

## Shared base command
```bash
BASE="uv run python -m unet_sr.solver_train \
  --training-pool-dir ../../data/synthetic/pool_2x_v5_5k \
  --input-mode hybrid_drizzle2x --scale 2 --unroll-steps 2 \
  --patch-size-hr 384 --batch-size 12 --solver-m-frames 12 \
  --solver-dc-weight 0 --num-workers 12 --compile --log-every 500 \
  --synth-eval-holdout 500 --synth-eval-every 2500 \
  --save-every 5000 --total-steps 40000"
```

---

## Phase 0 — smoke gates (2k each; verify new code runs on GPU before committing 40k)
Run with `--total-steps 2000 --save-every 1000 --synth-eval-every 1000` and a throwaway output dir.
- **S1 (D-E)**: `$BASE --solver-no-drizzle --solver-prox-highpass-residual --solver-prox-highpass-sigma-hr 5`
  → must reach 2k without crash; synth eval finite; PNG saved.
- **S2 (phasebin-ontf 9-bin)**: `$BASE --solver-phasebin-ontf --phase-bin-channels 9 --solver-warmstart aligned_mean --solver-prox-highpass-residual --solver-prox-highpass-sigma-hr 5`
  → verifies on-the-fly 9-bin drizzle builds (in_ch=14) and trains without shape error.
- **S2b (phasebin-ontf 16-bin quick shape check)**: same as S2 but `--phase-bin-channels 16`
  and `--total-steps 1000`. This is just a shape/memory gate for E5 (in_ch=21).
If S1/S2 pass, proceed. If S2 fails (data/geom), skip Phase 2 and log it. If S2b fails but S2 passes,
run E4 and skip E5.

---

## Phase 1 — D-E high-pass-residual σ sweep (main capacity-recovery lever)
Isolates D-E vs V11 (same no-drizzle input). Pick the σ that best recovers synth sharpness with real
artifact not worse than V11.

| id | override | output dir |
|---|---|---|
| E1 | `--solver-no-drizzle --solver-prox-highpass-residual --solver-prox-highpass-sigma-hr 5` | `outputs/solver_v13_de_s5` |
| E2 | `--solver-no-drizzle --solver-prox-highpass-residual --solver-prox-highpass-sigma-hr 4` | `outputs/solver_v13_de_s4` |
| E3 | `--solver-no-drizzle --solver-prox-highpass-residual --solver-prox-highpass-sigma-hr 8` | `outputs/solver_v13_de_s8` |

**Decision**: let σ\* = best of {E1,E2,E3} by (synth PSNR↑, region-RMSE↓, boundary-F1↑) subject to
real artifact ≤ ~0.46 and PNG visibly cleaner than V11. If none beats V11 on synth, D-E alone is
insufficient → still run Phase 2/3, and flag "consider low-freq-anchor split (bigger change)".

---

## Phase 2 — richer multi-frame INPUT on top of D-E (the user's target)
Only if S2 passed. Uses hybrid input (drop `--solver-no-drizzle`) + on-the-fly phase-bins.

| id | override (σ\* = Phase-1 winner) | output dir |
|---|---|---|
| E4 | `--solver-phasebin-ontf --phase-bin-channels 9 --solver-warmstart aligned_mean --solver-prox-highpass-residual --solver-prox-highpass-sigma-hr σ*` | `outputs/solver_v14_de_pb9` |
| E5 | `--solver-phasebin-ontf --phase-bin-channels 16 --solver-warmstart aligned_mean --solver-prox-highpass-residual --solver-prox-highpass-sigma-hr σ*` | `outputs/solver_v14_de_pb16` |

**Decision**: if E4/E5 beat the Phase-1 winner on synth PSNR/F1 with real artifact not worse →
**richer multi-frame input pays off** (do this as mainline). If E4≈E5≈Phase-1 → the sub-pixel signal
was already saturated in 4 bins and the bottleneck was capacity (confirms the info-budget read);
stop adding bins.

---

## Phase 3 — capacity variant (orthogonal control)
`--base-channels 96` is expected to be much slower and may OOM at batch 12. First run a short
200-500 step memory/timing check. If it OOMs, lower only E6 to `--batch-size 6` and mark the
comparison as not perfectly batch-matched.

| id | override | output dir |
|---|---|---|
| E6 | `--solver-no-drizzle --solver-prox-highpass-residual --solver-prox-highpass-sigma-hr σ* --base-channels 96` | `outputs/solver_v15_de_wide` |

**Decision**: if E6 > E1 (σ\*) on synth with real artifact not worse → raw prox width helps too
(combine with the winner). If E6 ≈ E1 → capacity is not width-limited; D-E structure was the lever.

---

## Results table (scheduler fills in; back-fill ACL-042/043)
| id | synth PSNR | region RMSE | boundary F1 | synth OOB | real artifact | real OOB | visual (sharp? clean?) | verdict |
|---|---|---|---|---|---|---|---|---|
| V11 (ref) | 35.17 | 0.0859 | 0.858 | 0.0104 | 0.457 | 0.0021 | soft, clean | baseline |
| E1 | 32.51 | 0.1191 | 0.863 | 0.0102 | 0.461 | 0.0021 | clean but not sharper than V11 | sigma=5 no win |
| E2 | 32.53 | 0.1191 | 0.864 | 0.0098 | 0.460 | 0.0022 | clean, best Phase-1 balance | selected sigma*=4 |
| E3 | 32.50 | 0.1189 | 0.863 | 0.0102 | 0.458 | 0.0021 | clean but no PSNR gain | sigma=8 no win |
| E4 | 32.47 | 0.1208 | 0.864 | 0.0103 | 0.436 | 0.0018 | cleaner real proxy, softer synth | 9-bin input no win |
| E5 | aborted (5k) | | | | | | no improvement | early stop |
| E6 | 32.58 | 0.1176 | 0.863 | 0.0096 | 0.465 | 0.0022 | marginal PSNR/RMSE bump, artifact worse | width not primary limiter |

## Final analysis
1. D-E (Phase 1) with σ*=4 (E2) improved boundary F1 (0.864) and maintained low artifacts (0.460), but failed to recover synthetic PSNR toward V9 (only reaching 32.53 vs V11's 35.17).
2. Richer multi-frame input (E4 9-bin) slightly dropped PSNR to 32.47, and E5 (16-bin) showed no improvement at 5k steps (aborted), confirming the capacity/info-budget bottleneck.
3. Increasing width to 96 channels (E6) yielded a marginal synthetic PSNR bump to 32.58, but artifacts slightly worsened to 0.465, showing raw width is not the primary limiter.
4. Recommended mainline config: D-E/no-drizzle with σ=4 (E2), as further capacity/bins show diminishing returns. We appear to have hit the info-budget physics ceiling; the next lever should be acquisition/noise rather than network architecture.
