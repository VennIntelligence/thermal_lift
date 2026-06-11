# 6. Experiments (draft — numbers verified where stated; ⬜ = pending)

> Final-table rule: every number in T1/T2 must come from ONE unified harness re-run
> (`run_unet_vs_drizzle_2x.py` lineage). TensorBoard `eval_real/*` values and EP11-harness values
> use different artifact scales and must never be mixed in a table. Trajectory figures may use
> TB scales (consistent within-figure).

## 6.1 Main comparison (T1 + F5)

Arms: bicubic / drizzle / MAP-TV anchor / anisotropic coverage-weighted TGV / UNet best
(selected checkpoint per protocol) / [⬜ V9A hybrid selected]. Columns: split-half NRMSE,
artifact score, raw-control corr, FRC@{16,14,12} µm, zigzag median FWHM & dip, runtime.

Verified so far (mixed scales; unify before T1):
- TGV (TB-scale): artifact 0.695, corr 0.916; stripes resolved (was 3.870 / 0.902); 30.8 min CPU.
- MAP-TV anchor: zigzag FWHM 114→100 µm, dip 0.929→0.934 (per-profile mixed: 2 wider / 1 much
  narrower → "limited contour enhancement"); FRC table in §5.1; 4563 s GPU (150 iter).
- UNet canonical (EP11 harness): v9b@11K split-half 0.0517, artifact 1.732, corr 0.7776 — best
  proxy compromise among the four 1x-input arms; v6@8K 0.0497/1.786/0.7738; v8.1a@15K
  0.0687/1.987/0.7572; v8.1b@5K 0.0479/1.759/0.7414 (failed arm, kept as control).
- ⬜ V9A/V9C selected checkpoints through the same harness.

Current verdict line (to be finalized): classical TGV/MAP-TV remain the trustworthy deliverable;
1x-input UNet arms trade observation fidelity for stylization; evidence-injected arms ⬜.

## 6.2 Null-space drift (F3) — the paper's core negative-mechanism figure

Per-arm trajectories of (artifact, corr) vs step, with forward-loss inset:

| arm (1x input) | anchor | artifact 10K→60K | corr 10K→60K | fwd-loss behavior |
|---|---|---|---|---|
| hot loss (v6) | full 0.1 (hybrid cfg) | 0.339@2K → 0.883 | 0.773 → 0.648 | — (confounded, drift reference) |
| conservative (v8.1a) | none | 0.390 → 0.643 | 0.756 → 0.689 | n/a |
| conservative (v8.1b, PixelShuffle) | none | 0.413 → 0.709 | 0.747 → 0.667 | n/a (failed head, control) |
| conservative (v9b) | band-limited 0.1 | 0.369 → 0.655 | 0.758 → 0.688 | **flat 0.004–0.009 from 10K** |
| conservative (V9D) | full-band 0.1 | ⬜ | ⬜ | ⬜ (predicted: floor or oscillation) |

Key measured facts: v9b's 40K→60K drift (+0.0145 / −0.0082) coincides with v8.1a's
(+0.016 / −0.009) — anchoring changed nothing; its forward loss sat at floor throughout the
drift. Loss-side knobs across three arms produce near-identical drift curves → the drift is
prior-driven and null-space-resident. V9D completes the claim against the "band too narrow"
objection. ⬜ V9C tests anchoring *with* evidence-injected input.

## 6.3 Input-mode ablation (T2 + F6)

Matrix (input × anchor): {1x stats, hybrid drizzle} × {none, band-limited, full-band, legal}.
Filled: 1x×none (v8.1a), 1x×band (v9b), 1x×full (⬜ V9D), hybrid×none (⬜ V9A, training),
hybrid×legal-band (⬜ V9C). Readouts: selected-checkpoint harness metrics + center-thin-line and
edge-staircase visual crops + per-arm trajectories.

Two-arm attribution already established (loss-cooldown vs PixelShuffle): the finest-line blur is
invariant to loss temperature and HR head → input information bottleneck. V9A early evidence
(step 5K): center thin lines more resolved, edge staircase reduced, corr 0.708 already above all
1x arms' 60K endpoints (cross-mode caution applies; trajectory pending).

## 6.4 Frame budget (EP16 classical CPU complete; learned/GPU arms pending)

EP16 completed the classical CPU subset matrix for drizzle and TGV:
`N={31,62,124,248}` with phase-stratified sampling, 17 frame-budget rows, all success
(`output/ep16_budget_robustness/frame_budget.csv`). The EP16 table is an inference-time
stability study, not part of the unified T1 harness.

Drizzle shows the clearest information-budget trend: raw-control corr rises from
0.747±0.032 at N=31 to 0.772 at N=248, while split-half NRMSE falls 0.0715→0.0306,
artifact falls 1.649→1.145, and FRC@16 µm rises 0.109→0.479. Most raw-control gain appears
by N=62; the FRC/split-half proxies continue improving with more phase coverage.

TGV has lower artifact than drizzle throughout the same matrix (0.946±0.003 at N=31 to
0.708 at N=248), but raw-control corr is non-monotonic (0.728±0.053, 0.754±0.009,
0.735±0.013, 0.741). TGV split-half/FRC columns use the same drizzle proxy on each identical
subset and shift set to keep the overnight run budget at 17 full TGV reconstructions; TGV
artifact/raw-control/zigzag columns are measured on the actual TGV HR image. MAP-TV and learned
arms remain pending for a GPU-available follow-up.

## 6.5 Robustness (EP16 classical CPU complete; learned/GPU arms pending)

EP16 completed the shift perturbation and alignment-source classical matrices
(`shift_robustness.csv`: 20 rows all success; `alignment_source.csv`: 4 rows all success).
Shift perturbation adds Gaussian noise to measured contour-refined shifts; it is a pressure
test, not a calibrated estimate of true alignment error.

For drizzle, raw-control corr is nearly flat from σ=0 to 0.2 px (0.771→0.770), but artifact
worsens 1.145→1.434 and FRC@16 µm drops 0.479→0.340. For TGV, raw-control corr also remains
stable in this proxy (0.741→0.744), while the shared FRC proxy follows the same decline because
it is computed with the same subset and shifts. This supports the robustness narrative as
metric-specific: raw-control agreement is stable under these small perturbations, while
coverage/FRC-style proxies are more sensitive.

The alignment-source ablation is stronger. Replacing command-prior shifts with contour-refined
shifts improves drizzle raw-control corr 0.662→0.771 and FRC@16 µm 0.0166→0.479; TGV raw-control
corr improves 0.642→0.741 under the same source swap. This is the end-to-end evidence that
data-driven alignment refinement matters. Command shifts remain priors, not ground truth.

## 6.6 Checkpoint-selection protocol in action (F4)

Pareto scatter per arm with TGV reference point (0.695, 0.916 TB-scale); selected checkpoints
(v6@8K, v8.1a@15K, v8.1b@5K, v9b@11K) vs 60K endpoints; visual panels confirming selection.
Message: endpoint reporting (the field default) would have reported the *worst* checkpoints of
every arm.

## 6.7 Negative results (kept, one short subsection)

PixelShuffle HR head (stripes, worse proxies); 4x network (no real gain; consistent with MTF
bound — 4x Nyquist MTF ≤ 0.042); loss-side anchoring against null-space drift (6.2); rendered
AVI as SR input (8-bit, 67% duplicates — excluded by audit, direction check only).

> TODO(§6): unify harness & fill T1/T2; freeze N-subset and perturbation seeds; final visual
> crop coordinates (center zigzag ROI + one block-edge ROI); per-figure captions drafted in
> `09_figures_tables_assets.md`.
