# 5. Evaluating Without Ground Truth (draft)

The protocol has four layers; no single layer proves success — adoption requires passing all,
and each layer's failure modes are reported with controls.

## 5.1 Information existence: phase-stratified split-half FRC with controls

Split the 248 frames into halves stratified by sub-pixel phase, reconstruct each half
independently (drizzle), and compute Fourier Ring Correlation between the halves; repeat over
3 seeds. Readings on the main session:

- **1/7 cutoff at 17.03 µm period (std 0.50 µm across seeds; half-bit criterion identical).**
  Coherent information beyond the 20 µm resolution exists, but less than the 11–14 µm
  theoretical hope.
- Band table (period → FRC): 20 µm 0.348, 16 µm 0.138, 14 µm 0.098, **12 µm 0.593, 11 µm 0.877,
  10 µm 0.935**, 9 µm 0.816, 8 µm 0.545.
- **Controls and the honest caveat.** The high-frequency rebound (10–12 µm) would be exciting if
  trustworthy, but the controls say otherwise: the bicubic positive control does not show the
  expected lower cutoff (13.58 µm), the shift-shuffle negative control retains median FRC 0.504
  in 8–12 µm instead of collapsing, the acquisition-order drift control degrades the cutoff to
  26.20 µm, and mean zero-coverage on the fine grid is 27%. We therefore attribute the rebound to
  coverage/lattice artifacts plus thermal drift, flag it as **risk rather than resolution
  evidence**, and base claims on the 17 µm cutoff only.

After MAP-TV deconvolution the split-half FRC at 20/16/14/12/10 µm reads
0.976/0.965/0.955/0.947/0.934 (bare drizzle: 0.319/0.088/0.053/0.575/0.893) — deconvolution
raises split-half *consistency*; this is stability evidence, not optical ground truth.

## 5.2 Observation-anchored proxy pair — and why it anti-correlates

On real data we track two proxies against a raw-bicubic control: an **artifact score**
(excess high-pass energy ratio) and **raw-control correlation** (high-pass correlation with the
control). Both are functionals of the same high-pass residual; movement along the
"synthetic-prior stylization" axis (edges brighter/wider than observations support) raises the
first and lowers the second **by construction**. Consequently: (i) the pair is a *drift
thermometer and selection criterion*, not a pair of independent scores to be jointly maximized;
(ii) cross-input-mode numerical comparison is invalid (evidence-injecting inputs legitimately
carry more high-frequency energy); (iii) trajectories over training steps are the informative
object, not endpoint values.

## 5.3 Null-space drift diagnosis

Decompose any reconstruction change as δx = δx_range + δx_null w.r.t. the observation operator A
(Sec. 3.4): A δx_null = 0, so no observation-side loss can see δx_null. The practical diagnosis
requires only two curves per training run: the forward-consistency loss (does it sit at its
floor?) and the real-data proxy trajectory (does it keep drifting?). **Floor + drift = the drift
is in the null space**, and stronger observation anchoring is provably unhelpful; the remedy must
come from the prior (training distribution), the input (evidence injection), or selection.
Measured instance in Sec. 6: forward loss flat at 0.004–0.009 from step 10K while artifact climbs
0.37 → 0.65.

## 5.4 Structure-level metrics and visual gates

- **Zigzag profile metrics** on the client-relevant center traces: per-profile apparent FWHM and
  valley dip depth across fixed cross-sections (median FWHM / dip reported with per-profile
  spread; mixed per-profile outcomes are reported as such).
- **Visual panels in two domains**: high-pass structure maps (edge evidence; red/blue =
  signed response relative to local background, white ≈ no change) *and* plain temperature
  views (inferno, 1–99 percentile) — the latter catch the "only edges got brighter" failure that
  high-pass views flatter.
- **Alignment-gate consistency** (EP04 roles): reconstructions must not degrade hold-out contour
  consistency on gated anchors.

## 5.5 Adoption rule

A learned arm is adopted iff, at its *selected* checkpoint (Sec. 4 rule): (a) FRC-band
consistency ≥ MAP-TV anchor; (b) zigzag FWHM/dip not worse than anchor; (c) proxy trajectory
shows no post-selection drift cliff; (d) visual panels pass in both domains. Otherwise the
classical anchor remains the deliverable. This rule is what licenses honest negative results
(PixelShuffle head; 4x network; loss-side anchoring) alongside positive ones.

> TODO(§5): formalize δx decomposition notation with Sec. 3.4 operator; decide whether
> null-space projection figure (A†A applied to checkpoint diffs) is feasible cheaply — if yes,
> it upgrades 5.3 from two-curve diagnosis to direct measurement [stretch goal, GPU-light].
