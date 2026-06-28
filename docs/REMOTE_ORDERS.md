# REMOTE ORDERS — Physics-Constrained Unrolled SR Solver (staged, gated)

> For the remote agent on the 5090 box. Philosophy: **gated** — each stage has a PASS bar;
> if a gate fails, STOP and report, do not burn GPU hours on the next stage. This exists to
> avoid repeating the project's documented failure modes (see "Hard rules" below).
>
> Author context: local agent (Mac). Full design: `research_log/network_upgrade_roadmap.md`;
> decision record: `research_log/algorithm_changelog.md` ACL-024.

## 0. Hard rules (do NOT repeat past mistakes)
- **Anti-hallucination is structural (hard DC), NOT loss-side.** Loss-side forward anchoring
  is FALSIFIED (ACL-017/019). Do not "fix" drift with a soft forward-model weight.
- **Stop before the ~30K fidelity cliff.** Select checkpoints in the 10K–25K window; never
  default to the last step (ACL-016/020).
- **Band-gate everything.** Supervise/eval only inside the honest recoverable band; matching
  labels out-of-band = hallucination.
- **One variable at a time.** Hold batch size constant (bs confound bit V9A, ACL-020).
- **Residual params: add the base back on EVERY eval path** (ACL-020 false-failure bug).
- **One scale per eval table** (ACL-021 scale-contamination).
- **Sharpness alone is NOT success.** A win = in-band split-half FRC ≥ classical TGV/MAP-TV.

## 1. Environment
- Repo on the box: **`~/thermal_lift`** (`/home/ujs/thermal_lift` in WSL — moved off the slow
  `/mnt/c` Windows mount for I/O). `cd ~/thermal_lift && git pull` to the latest `main`.
- Pool: `data/synthetic/pool_2x_v3_5k`. Run python via the project env (`uv run python ...`).
  All commands below are repo-root-relative.

---

## GATE 0 — Data audit  ✅ RUNNABLE NOW (even on a partially-generated pool)
Verifies the generated data is self-consistent and honest (does NOT need the solver code).
```
uv run python scripts/audit_generated_pool.py --pool data/synthetic/pool_2x_v3_5k --k 64 --out output/pool_audit.json
```
PASS bar:
- `median hp-corr (clean A(target) vs saved burst) > 0.90` — the save/reconstruct/metadata/
  shift/PSF roundtrip is sound. (If LOW: a convention/shift/PSF bug in the data → STOP, paste
  the per-scene table; do not train.)
- `scenes with structural problems = 0`.
- `median GT unrecoverable-band energy < ~0.05` — data is honest (not stuffed with detail the
  PSF kills). Slightly higher is OK but report it.
Report: the full table + the three summary numbers.

## GATE A — Forward operator certification  ✅ RUNNABLE NOW
Certifies the torch forward `A` matches the data-gen forward and that autograd gives the exact
transpose `A^T` (the DC gradient). This is the make-or-break correctness check (FM-6).
```
uv run python algos/ep07_unet_sr/tests/test_forward_torch.py
```
PASS bar (printed): forward parity `< 1e-5` (gaussian/elliptical/airy), linearity `< 1e-9`,
adjoint identity `< 1e-9`. If FAIL: STOP, paste output — the operator/convention is wrong and
every downstream step would inherit a systematic error.

> Run GATE 0 and GATE A now and report results. They are independent of the training code below
> and validate the two foundations (data + operator) while generation finishes.

---

## GATE B — single-scene overfit  ✅ RUNNABLE NOW (`unroll.py` landed)
End-to-end FM-6 catch that Gate A (operator-only) cannot see: on ONE clean synthetic scene,
data-fitting `min ||A x - y||^2` must drive the DC residual toward ~0 and recover a GT-aligned
image; plus the full `UnrolledSolver` runs forward+backward with finite grads.
```
uv run python algos/ep07_unet_sr/tests/test_gate_b_overfit.py
```
PASS bar (printed): DC residual drops `>10x`, `corr(recovered x, GT) > 0.90`, solver grads finite.
If the residual plateaus or corr is low: a half-pixel/sign/PSF bug — STOP, paste output.

> Run GATES 0, A, B now and report. They need only the code already pushed (no dataset plumbing).

---

## GATE C — training smoke on the real pool  ✅ RUNNABLE NOW
Validates the full solver training plumbing end-to-end on real scenes: the dataset delivers
burst+shifts+PSF at the scale-aligned crop, ScenePSF builds from the batch, and the
`UnrolledSolver` runs forward+backward without NaN.
```
uv run python algos/ep07_unet_sr/tests/test_gate_c_smoke.py --pool data/synthetic/pool_2x_v3_5k
```
PASS bar (printed): shapes/plumbing OK, all losses+grads finite. If FAIL: STOP, paste output.

## DRIZZLE — pick ONE input path (the on-the-fly drizzle is what made Gate C slow)
The solver needs the raw `lr_burst` (DC term — already saved) regardless. The *drizzle* is only a
warm-start/conditioning, and computing it on-the-fly per sample is slow. Two ways:
- **(A, RECOMMENDED) Lean / no-drizzle** — `--solver-no-drizzle`. Drops the drizzle entirely: no
  on-the-fly cost, no precompute, no extra disk. Warm-start = upsampled aligned_mean; cond = 5ch.
  The DC term carries the multi-frame SR signal. Fastest path to a baseline.
- **(B) Hybrid + precomputed drizzle** — keep the 8ch hybrid warm-start, but precompute the drizzle
  once so it's a fast mmap (not on-the-fly):
  ```
  uv run python scripts/precompute_drizzle_variants.py --pool-dir data/synthetic/pool_2x_v3_5k \
    --num-variants 1 --workers 14        # ~15-20 min, ~37 GB (num-variants 1 is enough for the solver)
  ```
  Then run WITHOUT `--solver-no-drizzle`. After precompute the dataset auto-uses the variants.

## DATA REGEN — v5 sharp GT (ACL-030)  — DO THIS FIRST; the v4 blur was in the TARGET, not the loss
> ACL-030: the v4 GT (`edge_sigma=1.4`) carried ~zero super-resolution-band energy —
> `out_of_band(GT)=0.00008`, matching the trained model's real-data output `0.00001`. The model
> faithfully reproduced a detail-free target → "good fidelity, no hallucination, but blurry / no
> detail". **The loss is fine.** **NOTE (measured through the real pipeline):** the GT is rendered
> from an antialiased SSAA coverage mask that *itself* softens edges ~0.7 HR px, so `edge_sigma` is
> not the only softening — `edge_sigma` 1.4→0.00008, **0.8→0.0013 (still too soft)**, **0.6→0.0039
> (target = edges at ~1 pitch)**, ≤0.4→≥0.013 (sub-pitch). v5 therefore uses **`edge_sigma=0.6`**
> (a first 0.8 attempt only reached 15× v4). Do **NOT** go below ~0.5 (approaches the AA floor →
> sub-pitch → re-invites FM-1 beading). Realism/blur belongs in the forward path (PSF); sharpness in
> the GT. seed kept = v4 → controlled A/B. `edge_sigma` is a config knob, no code change.

```
# (a) SMOKE first — 300 scenes, ~minutes — cheap proof before the overnight full regen
uv run --with tqdm python scripts/generate_training_pool.py \
  --config configs/synthetic/pool_2x_v5_sharp.json \
  --num-scenes 300 --output-dir data/synthetic/pool_2x_v5_smoke_300 --workers 14

# VERIFY the GT landed: PASS = median out_of_band ~0.003-0.005 (~40x v4) + defects present
uv run python scripts/verify_pool_sharpness.py --pool data/synthetic/pool_2x_v5_smoke_300 --k 96

# short V10 train on the smoke pool — watch eval_synth/out_of_band_ratio LIFT off ~0
uv run python -m unet_sr.train \
  --training-pool-dir data/synthetic/pool_2x_v5_smoke_300 \
  --input-mode hybrid_drizzle2x --scale 2 --loss-type contour_sr \
  --boundary-boost 4.0 --flatness-weight 0.0 \
  --synth-eval-holdout 48 --synth-eval-every 1000 \
  --total-steps 5000 --batch-size 24 --patch-size-hr 256 \
  --num-workers 12 --save-every 1000 --no-real-eval --output-dir outputs/v5_smoke
# PASS bar: eval_synth/out_of_band_ratio rises from ~0 toward the GT's ~0.004, region_rmse stays
# low, and the center-zoom temperature visibly sharpens. If it does NOT lift → STOP and report
# (the model isn't picking up the new band; then it's an obs-PSF / capacity issue, not the GT).

# (b) FULL regen — 5000 scenes, overnight — ONLY after the smoke confirms.
# rm first: a prior partial run at this path is KEPT by the resume-skip (per-scene metadata.json)
# and would mix old-edge_sigma scenes into the pool.
rm -rf data/synthetic/pool_2x_v5_5k
uv run --with tqdm python scripts/generate_training_pool.py \
  --config configs/synthetic/pool_2x_v5_sharp.json --workers 14
uv run python scripts/verify_pool_sharpness.py --pool data/synthetic/pool_2x_v5_5k --k 96
```

## TRAINING — V10 baseline (plain UNet) on v5 + ACL-027 loss/metrics  — the headline run
> ACL-027 reworks the loss + eval for the v4 defect data. **Loss**: the thin/gap line priors are
> replaced by one geometry-agnostic `boundary_weight` (chip outline + hole/crack/notch rims,
> contrast-independent) + an optional isothermal `flatness` term (OFF here; A/B next). **Eval**: a
> held-out synthetic GT split now reports the metrics we actually care about —
> `eval_synth/{psnr, region_rmse, boundary_f1, out_of_band_ratio}` — and on real data
> `eval_real/out_of_band_ratio` (PSF-free) **replaces the deleted `raw_control_corr`**, which had
> correlated the clean output against a bicubic blur and so rewarded *not* restoring.
```
uv run python -m unet_sr.train \
  --training-pool-dir data/synthetic/pool_2x_v5_5k \
  --input-mode hybrid_drizzle2x --scale 2 --loss-type contour_sr \
  --boundary-boost 4.0 --flatness-weight 0.0 \
  --synth-eval-holdout 200 --synth-eval-every 2500 \
  --total-steps 50000 --batch-size 24 --patch-size-hr 256 \
  --num-workers 12 --save-every 2500 --output-dir outputs/v10_v5_sharp
```
- `in_channels` auto = 9 (5 fused↑2x + 4 phase-bin drizzle); both inputs are precomputed on disk.
- Headline tracking: `eval_synth/region_rmse`↓ and `eval_synth/boundary_f1`↑ (real objective, vs GT);
  `eval_real/out_of_band_ratio` should stay **flat** — a jump = beading / FM-1 cliff onset.
- **A/B the isothermal prior next**: same command with `--flatness-weight 0.05`; compare eval_synth.
- The 200-scene tail is auto-excluded from training (no leakage). Tune batch/patch/workers to the GPU;
  real_eval stays on (needs the real frames + ep10 baseline).

## TRAINING — K-step unrolled solver (the real run)  — RUN after Gates 0/A/B/C all PASS
> **ACL-026 architecture** (current): each unroll step is `prox → DC` and the solver **ENDS on the
> DC step**, so the output `x_K` is forward-consistent by construction; **eta is FROZEN by default**
> (`--solver-learn-eta` to unfreeze) so the optimizer can't bypass the DC step. Anti-hallucination is
> now **architectural**, not the (falsified) soft DC loss. History: v1 (ACL-025) and v2 both saw
> `loss/dc` climb ABOVE the smooth warm-start floor while `struct` fell = hallucination; root cause
> was the old `DC→prox` order (prox had the last word) + a learnable eta bleeding to 0.

**Recommended first run = PURE ARCHITECTURE** (no soft DC term, no anneal — let the architecture
enforce consistency; `loss/dc` is now just a MONITOR in TB):
```
uv run python -m unet_sr.solver_train \
  --training-pool-dir data/synthetic/pool_2x_v5_5k \
  --input-mode hybrid_drizzle2x --scale 2 --unroll-steps 4 --solver-no-drizzle \
  --solver-m-frames 12 --solver-band-sigma 5 \
  --solver-prior-anneal-steps 0 --solver-dc-weight 0 \
  --boundary-boost 4.0 --flatness-weight 0.0 \
  --synth-eval-holdout 200 --synth-eval-every 2500 \
  --total-steps 20000 --batch-size 18 --patch-size-hr 192 \
  --save-every 2500 --num-workers 8 --output-dir outputs/solver_v5_sharp
```
**Before the pool run, RE-RUN GATE B** — its check [4] now certifies the ACL-026 architecture: after
overfitting the SOLVER on one clean scene with structural supervision only, the terminal DC residual
must drop BELOW the warm-start floor (not climb above it like v1/v2), with corr>0.85 and eta frozen.
If [4] fails (DC climbs above the floor, or eta moved), STOP: the single gradient step is too weak →
escalate to a terminal CG projection (see the unroll.py caveat).
- `eta` in TB should be a **flat line** (frozen working); `loss/dc` should **NOT climb above the
  step-1 warm-start floor** (the v1/v2 failure). If it still climbs → the architecture isn't holding,
  report before burning 20K.
- Runs in **fp32 (no AMP)** for double-backward stability. If OOM: lower `--batch-size` /
  `--patch-size-hr` / `--unroll-steps`. Memory is dominated by the K-step double-backward.
- **Hold batch constant** across all solver ablations (V9A/ACL-020 confound). **Default stop ~20K**
  (FM-1 cliff); checkpoints every 2500 — select in the 10K–25K window.
- If DC holds but the result is too soft, add a **small** `--solver-dc-weight 0.1` as a weak
  regularizer (NOT the mechanism) — do not go back to the anneal-driven soft anchoring.
- Eval (offline, separate harness): synthetic split-half FRC + EP11/EP15 real-data; bar = in-band
  FRC ≥ classical TGV/MAP-TV. Add the warm-start base on every eval path.
- ACL-027: thin/gap line priors are REPLACED by the geometry-agnostic `--boundary-boost` (enabled
  here at 4.0; covers chip outline + hole/crack/notch rims, contrast-independent). `--flatness-weight`
  (isothermal prior) is OFF for this first run — A/B it next. The solver path now also logs the
  held-out `eval_synth/*` GT metrics (the headline) the same way as V10.

## PARALLEL (not a training gate) — re-run EP15 at 20µm
Re-derive the authoritative real-data recoverable band at the recalibrated pitch; needed before
trusting real-data FRC numbers / paper claims. Independent of the training run.

---
## Reporting protocol
After each gate: paste the numbers + PASS/FAIL. On any FAIL: stop, summarize the failing case,
wait for the local agent. Do not "push through" a failed gate.
