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
  --training-pool-dir data/synthetic/pool_2x_v3_5k \
  --input-mode hybrid_drizzle2x --scale 2 --unroll-steps 4 --solver-no-drizzle \
  --solver-m-frames 12 --solver-band-sigma 5 \
  --solver-prior-anneal-steps 0 --solver-dc-weight 0 \
  --total-steps 20000 --batch-size 18 --patch-size-hr 192 \
  --save-every 2500 --num-workers 8 --output-dir outputs/solver_v3_arch
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
- NOTE: thin/gap loss-weighting is disabled for this run (a shape-contract mismatch with
  ContourSRLoss); re-enable once the reshape is confirmed.

## PARALLEL (not a training gate) — re-run EP15 at 20µm
Re-derive the authoritative real-data recoverable band at the recalibrated pitch; needed before
trusting real-data FRC numbers / paper claims. Independent of the training run.

---
## Reporting protocol
After each gate: paste the numbers + PASS/FAIL. On any FAIL: stop, summarize the failing case,
wait for the local agent. Do not "push through" a failed gate.
