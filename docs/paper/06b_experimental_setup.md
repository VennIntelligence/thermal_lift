# Experimental Setup

*This section consolidates the setup-class details scattered across §3 (problem / calibrated
forward model), §4 (method), and §5 (GT-free protocol) into one place; §6 references it directly
and does not re-derive these settings. Every quantity is tagged with its provenance (episode or
config file).*

## Datasets

**Real data.** All real-data results use the **248-frame clean main session**: of 263 step-and-shoot
raster frames, acquisition order — established from file mtimes, not filenames — splits them into
three thermal sessions; the 255-frame main session minus repeat-acquisition frames (R≠0) leaves 248
(EP01, §3.2). Cross-session temperature jumps reach a median 3.55 °C (49× the noise floor), so frames
are **never mixed across sessions** (EP01). There is **no HR thermal ground truth** in this regime
(§3.5). Rendered AVI/BMP exports are excluded as numeric input (8-bit, ~67% duplicated frames; EP01)
and used only as an independent direction check.

**Synthetic data.** Learned arms train on the **TCForge physics-matched pool**, synthesized with the
*measured* forward model rather than adversarial degradations (§4.2): chip-like geometry rendered with
4× SSAA soft [0,1] coverage masks (so HR targets carry coverage-scaled, staircase-free edges),
temperature T_bg + ΔT·coverage, Gaussian PSF in the credible σ range, 10 µm detector-box integration,
the measured noise floor, and per-frame shifts replaying the actual raster command lattice. Pools:
1000 scenes (2x AA), plus a burst pool storing per-frame LR stacks/shifts with **K = 4** precomputed
drizzle variants per scene — variant 0 = canonical full-burst / no-noise (matching inference);
variants 1–3 bake in frame-subset and σ = 0.05 px shift-noise augmentation (ACL-018).

## Methods / arms compared

**Classical.** (i) **Bicubic** upsampling; (ii) **Drizzle**, flux-conserving sub-pixel fusion of the
248 aligned frames onto the 2x grid (mean zero-coverage ~27% on the 5x diagnostic grid; §4.1);
(iii) **MAP-TV** deconvolution — GPU batch forward model + FISTA, 150 iterations, with σ = 0.2 LR px
and λ = 1e-3 selected by split-half NRMSE + artifact proxies — used as the **acceptance anchor**: a
learned arm is adopted only if it beats MAP-TV on FRC-band consistency *and* contour-profile metrics
simultaneously (§4.1, §5.5); (iv) **anisotropic coverage-weighted TGV** (elliptical dual-ball
projection, Y-axis regularization ratio 1.5, coverage-normalized data gradient; artifact 3.870→0.695,
raw-control corr 0.902→0.916; §4.1) — the practical classical deliverable.

**Learned (UNet).** The backbone is held fixed across arms; the experimental variables are the
**input mode** ∈ {1×-grid statistics, hybrid drizzle 2×} crossed with the **observation anchor** ∈
{none, band-limited (weight 0.1), full-band (weight 0.1), legal} (§4.3; ACL-016/017/019), plus a
**V10 residual-over-observation** variant with an L1 residual-magnitude penalty swept over λ
(ACL-020). Negative-result arms — the **PixelShuffle HR head** and the **4× network** — are retained,
not silently dropped (§4.3, §6.7; the 4× failure is consistent with the MTF bound, 4x Nyquist MTF
≤ 0.042, §3.3).

## Implementation details

**Backbone.** Plain UNet, base width 64, bilinear HR head (§4.3). The 1×-stats input is 5 channels
(aligned mean / median / coverage / variance / high-pass on the 1x grid); the hybrid input is 8
channels (3 drizzle channels rendered on the 2x grid concatenated with the 5 upsampled 1x channels),
with the network run at scale 1 on the 2x grid (ACL-016).

**Contour-oriented loss (conservative setting).** MSE 0.3 + high-pass structure 0.8 (σ=5) + SSIM 0.15
+ gradient-vector matching 0.15 + edge 0.05, with thin-structure ×3 and gap ×2 weighting (§4.3;
structure-boost 2.0 in the run config, ACL-020).

**V10 parameterization.** The network output is interpreted as a residual δ over the hybrid
drizzle-mean channel (ch5): x̂ = ch5 + δ, penalized by λ·mean|δ| (ACL-020).

**Optimization / schedule.** V9-series arms train 60K steps (checkpoint every 5K; ACL-016/017/019).
The V10 high-λ sweep uses λ ∈ {0.2, 0.5, 1.2, 3.0}, batch size 128, HR patch 192, 25K steps with full
cosine annealing (checkpoint every 2.5K; `run_v10_highlam.md`). **Comparability caveat:** the V10
high-λ arms use HR patch 192 (forced down from 256 by RTX 3090 memory), whereas the older comparison
arms use patch 256, so cross-patch proxy comparison carries this caveat (`run_v10_highlam.md`).
Training runs on a **dual-GPU host** (~3.5–4.25 h per 25K-step arm; `run_v10_highlam.md`).

## Evaluation metrics & protocol

The protocol has four layers; adoption requires passing all (§5).

- **Information existence — phase-stratified split-half FRC** over 3 seeds: the 1/7 cutoff sits at
  **17.0 µm period** (std 0.50 µm; §5.1). The 10–12 µm rebound is flagged as coverage/lattice + drift
  *risk*, not resolution evidence; claims rest on the 17.0 µm cutoff only.
- **Observation-anchored proxy pair — artifact score (↓) and raw-control correlation (↑)**, both
  functionals of the same high-pass residual and therefore **constructively anti-correlated**: a drift
  thermometer and selection criterion, not two scores to be jointly maximized; cross-input-mode
  numeric comparison is invalid (§5.2).
- **Structure / grain.** Zigzag **apparent FWHM and valley-dip** on the client-relevant center traces
  (MAP-TV anchor: FWHM 114→100 µm, dip 0.929→0.934, per-profile mixed → "limited contour enhancement";
  §6.1) and a structural **grain proxy `lattice` (↓)** (reference window: drizzle 0.0015 / TGV 0.0169 /
  evidence-injected learned ≈ 0.024; `reframe_c4_claim3.md`). Any sharpness reading (`sharp_p95`) is
  **reported only jointly** with `lattice` and the visual gate, never alone — it is blind to contour
  continuity and is inflated by both TV-staircase beading and grain.
- **Dual-domain visual gate.** High-pass structure maps (edge evidence) *and* plain temperature views
  (inferno, 1–99 percentile) — the latter catch the "only edges got brighter" failure (§5.4).
- **Checkpoint selection (mechanical).** Per arm, normalize the proxy pair to [0,1], take the 3 steps
  nearest the ideal point (≥5K-step separation), always carry the final step as a drift reference, and
  gate the choice with visual panels; the 60K endpoint is **never** the default deliverable (§4.3).

## Reproducibility / metric-scale discipline

Every number in the **T1/T2 final tables comes from a single unified harness**
(`algos/ep11_dl_benchmark/scripts/run_unified_harness_t1_t2.py`). **TensorBoard `eval_real/*` (TB-scale) and EP11-harness-scale
artifact values use different scales and are never mixed in the same table or figure** (trajectory
figures may use TB scale, consistently within one figure; §6). Random seeds are fixed (e.g., FRC over
3 fixed seeds; §5.1).

**Claim boundary.** The deliverable is **2x contour-level** visibility of chip-internal structure
(5 µm output sample, 10 µm Nyquist period; §3.5). A 5 µm output sample is **not** 5 µm spatial
resolution, and **no** temperature metrology or 4×/5× physical recovery is claimed. Task-level
contour-legibility preferences are reported separately from the verifiable proxies and are **not**
fidelity evidence (`reframe_c4_claim3.md`).

**Unified harness outputs.** Final T1/T2 rows for selected V9A/V9C/V9D/V10 checkpoints are in
`output/ep11_unified_harness/{t1_metrics.csv,t2_metrics.csv,all_arm_metrics.csv,run_manifest.json}`.
The V10 high-λ fine-window readout is λ=1.2@15K = (hp_corr_input 0.922, sharp_p95 0.987,
lattice 0.0141), while its final T1/T2 harness row is artifact/corr 2.726/0.711. These are different
metric domains and are reported separately.
