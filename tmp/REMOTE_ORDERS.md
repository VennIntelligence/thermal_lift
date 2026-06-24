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
- Repo on the box; `git pull` to the latest `main`. Pool: `data/synthetic/pool_2x_v3_5k`
  (symlink → `/mnt/d/thermal_lift_data/pool_2x_v3_5k`).
- Run python via the project env (e.g. `uv run python ...` or the activated `.venv`). All
  commands below are repo-root-relative.

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

## [PENDING — awaiting next code drop from local: `unroll.py` + dataset plumbing + train integration]

The following stages need the unrolled-solver training code, which the local agent is writing
next. Specs are given so you know what's coming; do NOT attempt to run them until the code lands
and this file is updated.

## GATE B — single-scene overfit (minutes, GPU)
Overfit the K-step solver to ONE clean synthetic scene (no noise/drift/defects). The DC residual
must drive ≈ 0. If it cannot, there is a geometry/adjoint/plumbing bug — STOP. This is the
end-to-end FM-6 catch that Gate A (operator-only) cannot see.

## GATE C — smoke (50–100 steps on the real pool)
Finite loss, no NaNs, prediction/target/observation visually aligned (no half-pixel shift),
DC term decreasing. Hold bs constant with the real run.

## TRAINING — K-step unrolled solver (the real run)
- K-step: drizzle warm-start → [DC gradient step (autograd A^T, per-scene PSF, highpass band,
  Huber) + UNet prox] × K. UNet = existing `ThermalSRUNet` as the proximal net.
- Loss: band-aware (per-scene PSF sets the band) + terminal DC; structure terms from
  `ContourSRLoss`. Late-anneal the prior weight to fight the cliff.
- Checkpoint every 2500; **default stop ~20K** (revisit only with FRC evidence).
- Eval each checkpoint: synthetic split-half FRC (fast) + the EP11/EP15 real-data harness.
  Bar = in-band FRC ≥ classical TGV/MAP-TV. Add the drizzle base on every eval path.

## PARALLEL (not a training gate) — re-run EP15 at 20µm
Re-derive the authoritative real-data recoverable band at the recalibrated pitch; needed before
trusting real-data FRC numbers / paper claims. Independent of the training run.

---
## Reporting protocol
After each gate: paste the numbers + PASS/FAIL. On any FAIL: stop, summarize the failing case,
wait for the local agent. Do not "push through" a failed gate.
