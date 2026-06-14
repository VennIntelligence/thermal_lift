# 6. Experiments (draft — numbers verified where stated)

> Final-table rule: every number in T1/T2 must come from ONE unified harness re-run
> (`algos/ep11_dl_benchmark/scripts/run_unified_harness_t1_t2.py`). TensorBoard `eval_real/*` values and EP11-harness values
> use different artifact scales and must never be mixed in a table. Trajectory figures may use
> TB scales (consistent within-figure).

## 6.1 Main comparison (T1 + F5)

Final T1 numbers come from the unified paper harness
(`output/ep11_unified_harness/t1_metrics.csv`, manifest in the same directory). Artifact values in
this table use the EP11/common.metrics harness scale; TensorBoard `eval_real/*` artifact values are
only used for within-arm trajectory figures (§6.2) and are not mixed here.

| arm | step | split NRMSE↓ | artifact↓ | corr↑ | FRC16↑ | FRC14↑ | FRC12↑ | lattice↓ | sharp P95↑ | FWHM µm↓ | dip↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bicubic | — | 0.047 | 1.388 | 1.000 | 0.951 | 0.945 | 0.941 | 0.001 | 0.344 | 60 | 1.000 |
| drizzle | — | 0.024 | 1.138 | 0.771 | 0.439 | 0.564 | 0.765 | 0.001 | 0.481 | 45 | 0.970 |
| MAP-TV (5x) | — | 0.024 | 2.333 | 0.438 | 0.965 | 0.955 | 0.948 | 0.002 | 0.352 | 42 | 1.000 |
| TGV | — | 0.031 | 0.695 | 0.741 | 0.479 | 0.556 | 0.610 | 0.011 | 0.999 | 40 | 0.973 |
| v6 hot loss | 8K | 0.051 | 1.789 | 0.774 | 0.752 | 0.761 | 0.217 | 0.001 | 0.656 | 40 | 1.000 |
| v8.1a conservative | 15K | 0.073 | 1.943 | 0.758 | 0.712 | 0.660 | 0.429 | 0.001 | 0.864 | 40 | 1.000 |
| v9b band anchor | 11K | 0.054 | 1.766 | 0.777 | 0.744 | 0.741 | 0.197 | 0.001 | 0.698 | 40 | 1.000 |
| v9d full anchor | 7K | 0.054 | 1.726 | 0.771 | 0.744 | 0.708 | 0.150 | 0.001 | 0.697 | 40 | 1.000 |
| V9A hybrid | 10K | 0.054 | 1.762 | 0.719 | 0.945 | 0.870 | 0.190 | 0.001 | 0.683 | 40 | 1.000 |
| V9C hybrid legal anchor | 5K | 0.064 | 1.669 | 0.718 | 0.891 | 0.769 | 0.195 | 0.001 | 0.766 | 40 | 1.000 |
| V10 residual λ=1.2 | 15K | 0.041 | 2.726 | 0.711 | 0.986 | 0.984 | 0.979 | 0.013 | 0.968 | 40 | 1.000 |

Notes: MAP-TV is the precomputed EP15 5x deconvolution anchor, so `output_grid_scale` is explicit
in the CSV. TGV split/FRC columns reuse the EP16 drizzle split proxy on identical subsets/shifts;
TGV artifact/corr/zigzag columns are measured on the actual TGV image. Hybrid/V10 cache means pass
the 23°C sanity check (V10 λ=1.2@15K mean 23.288°C), confirming that residual-over-drizzle inference
added back channel 5 rather than reporting a near-zero delta field.

Current verdict line (reframed 2026-06-13; see `reframe_c4_claim3.md`): in this no-GT regime there
is no certifiable single winner. The unified harness splits the evidence rather than producing a
winner: TGV has the lowest 2x-anchor artifact score and strong contour width, but it carries the
known TV staircase/beading caveat; drizzle and some learned rows score higher on raw-control corr or
FRC-like consistency but are either soft or carry unverifiable learned high-frequency content. The
high-λ V10 fine-window sweep shows that residual-over-observation makes the
fidelity–sharpness–grain trade-off tunable: λ=1.2@15K reaches sharpness ≈ TGV with lower fine-window
grain (`lattice` 0.014 < TGV 0.0169) but lower observation fidelity (`hp_corr_input` 0.922 < TGV
0.960). F5 (`output/paper_figures/fig05_main_visual.{png,pdf}`) is therefore reported as a
dual-domain task-level visual gate, not as fidelity or resolution evidence.
A predeclared upper-right hold-out structure ROI (F5b,
`output/paper_figures/fig05b_main_visual_roi2.{png,pdf}`) preserves the same `lattice` ordering as
the center crop, while `sharp_p95` and profile-based zigzag proxies change order; this supports only
ROI-level visual/proxy auditing, not method dominance.

## 6.2 Null-space drift (F3) — the paper's core negative-mechanism figure

Per-arm trajectories of (artifact, corr) vs step, with forward-loss inset:

| arm (1x input) | anchor | artifact 10K→60K | corr 10K→60K | fwd-loss behavior |
|---|---|---|---|---|
| hot loss (v6) | full 0.1 (hybrid cfg) | 0.339@2K → 0.883 | 0.773 → 0.648 | — (confounded, drift reference) |
| conservative (v8.1a) | none | 0.390 → 0.643 | 0.756 → 0.689 | n/a |
| conservative (v8.1b, PixelShuffle) | none | 0.413 → 0.709 | 0.747 → 0.667 | n/a (failed head, control) |
| conservative (v9b) | band-limited 0.1 | 0.369 → 0.655 | 0.758 → 0.688 | **flat 0.004–0.009 from 10K** |
| conservative (v9d) | full-band 0.1 | 0.379 → 0.677 | 0.758 → 0.677 | oscillates 1–28K (e.g. 0.575/0.642 @20K) then settles |
| hybrid (v9c) | legal 1x 0.1 | 0.516 → 0.695 | 0.714 → 0.669 | legal anchor, drift not flattened |

Key measured facts: v9b's 40K→60K drift (+0.0145 / −0.0082) coincides with v8.1a's
(+0.016 / −0.009) — anchoring changed nothing; its forward loss sat at floor throughout the
drift. Loss-side knobs now span four anchor variants — none (v8.1a), band-limited (v9b),
full-band (v9d), and a *legal* 1x anchor under evidence-injected hybrid input (v9c) — and all
converge to the same ≈0.65–0.70 artifact / ≈0.67–0.69 corr plateau. Near-identical drift curves
across every loss-side variant → the drift is prior-driven and null-space-resident; the loss-side
anchoring route is fully closed (v9d defeats the "band too narrow" objection; v9c defeats the
"anchor was illegal under hybrid input" objection).

## 6.3 Input-mode ablation (T2 + F6)

Matrix (input × anchor): {1x stats, hybrid drizzle} × {none, band-limited, full-band, legal}.
All arms trained (60K): 1x×none (v8.1a), 1x×band (v9b), 1x×full (v9d), hybrid×none (v9a),
hybrid×legal-band (v9c); a fifth learned variant (V10) adds a residual-over-observation
parametrization on top of the hybrid input with a tunable penalty `λ` (the Claim-4 knob, §6.1).
Readouts: selected-checkpoint harness metrics + center-thin-line and edge-staircase visual crops
+ per-arm trajectories. T2 selected-checkpoint rows come from
`output/ep11_unified_harness/t2_metrics.csv`:

| arm | input | anchor / parameterization | step | split↓ | artifact↓ | corr↑ | FRC16↑ | FRC14↑ | FRC12↑ | lattice↓ | sharp P95↑ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v8.1a | 1x stats | none | 15K | 0.073 | 1.943 | 0.758 | 0.712 | 0.660 | 0.429 | 0.001 | 0.864 |
| v8.1b | 1x stats | none + PixelShuffle control | 5K | 0.050 | 1.782 | 0.739 | 0.734 | 0.692 | 0.392 | 0.001 | 0.682 |
| v9b | 1x stats | band-limited 0.1 | 11K | 0.054 | 1.766 | 0.777 | 0.744 | 0.741 | 0.197 | 0.001 | 0.698 |
| v9d | 1x stats | full-band 0.1 | 7K | 0.054 | 1.726 | 0.771 | 0.744 | 0.708 | 0.150 | 0.001 | 0.697 |
| V9A | hybrid drizzle2x | none | 10K | 0.054 | 1.762 | 0.719 | 0.945 | 0.870 | 0.190 | 0.001 | 0.683 |
| V9C | hybrid drizzle2x | legal 1x anchor | 5K | 0.064 | 1.669 | 0.718 | 0.891 | 0.769 | 0.195 | 0.001 | 0.766 |
| V10 | hybrid drizzle2x | residual λ=1.2 | 15K | 0.041 | 2.726 | 0.711 | 0.986 | 0.984 | 0.979 | 0.013 | 0.968 |

Two-arm attribution already established (loss-cooldown vs PixelShuffle): the finest-line blur is
invariant to loss temperature and HR head → input information bottleneck. V9A early evidence
(step 5K/10K in the fine-window diagnosis): center thin lines recover more phase structure than the
1x-stat input can expose, but final harness metrics show this input evidence is not a free fidelity
win. The anchor axis is null across both inputs: neither a 1x-input anchor (v9b/v9d) nor a legal
anchor under hybrid input (v9c) flattens the drift (§6.2). V10 changes the trade-off by
parameterizing output as residual-over-observation, but its selected row remains a sharp/high-FRC,
lower-corr/high-artifact point. Thus the input path can expose sub-pixel evidence, whereas loss-side
anchors do not make synthetic-prior drift observable; no T2 row certifies learned fidelity.

## 6.4 Frame budget (EP16 classical CPU complete)

EP16 completed the classical CPU subset matrix for drizzle and TGV:
`N={31,62,124,248}` with phase-stratified sampling, 17 frame-budget rows, all success
(`output/ep16_budget_robustness/frame_budget.csv`). The EP16 table is an inference-time
stability study, not part of the unified T1 harness.

Drizzle shows the clearest information-budget trend: raw-control corr rises from
0.747±0.032 at N=31 to 0.771 at N=248, while split-half NRMSE falls 0.0715→0.0306,
artifact falls 1.649→1.145, and FRC@16 µm rises 0.109→0.479. Most raw-control gain appears
by N=62; the FRC/split-half proxies continue improving with more phase coverage.

TGV has lower artifact than drizzle throughout the same matrix (0.946±0.003 at N=31 to
0.708 at N=248), but raw-control corr is non-monotonic (0.728±0.053, 0.754±0.009,
0.735±0.013, 0.741). TGV split-half/FRC columns use the same drizzle proxy on each identical
subset and shift set to keep the overnight run budget at 17 full TGV reconstructions; TGV
artifact/raw-control/zigzag columns are measured on the actual TGV HR image. MAP-TV and learned
learned arms are left as follow-up and are not a main-text gate.

## 6.5 Robustness (EP16 classical CPU complete)

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

> TODO(§6): final caption polish for F5 and decide whether a second held-out fine-window check is
> worth adding before LaTeX migration.
