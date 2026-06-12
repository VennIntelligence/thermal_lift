# 4. Methods (draft)

The method section presents (i) classical anchored reconstruction, (ii) the physics-matched
synthetic training platform, (iii) the learned track with its input and anchoring variants —
organized so that the ablation matrix in Sec. 6 (input mode × observation anchor) reads directly
out of this section.

## 4.1 Classical anchors

**Drizzle.** Flux-conserving sub-pixel scatter of the 248 aligned frames onto the 2x grid;
minimal-assumption fusion baseline; exposes coverage/lattice artifacts that any fine-grid method
must manage (mean zero-coverage fraction ~27% on the 5x diagnostic grid).

**MAP-TV deconvolution anchor.** GPU batch forward model (shift → Gaussian PSF → detector box →
decimation) over all frames; FISTA with smoothed-TV gradient, 150 iterations; σ and λ selected by
split-half NRMSE + artifact/std proxies over σ ∈ {0.2,…,0.5} × λ ∈ {3e-4, 1e-3, 3e-3}
(selected σ = 0.2 LR px, λ = 1e-3). This is the **acceptance gate**: a learned method is adopted
only if it beats the anchor on FRC-band consistency *and* contour-profile metrics simultaneously.

**Anisotropic coverage-weighted TGV.** Raster acquisition makes the data constraint anisotropic
(within-line X gap 1 vs across-line Y gap ≈16) and bilinear scatter concentrates weight on fixed
HR rows, producing horizontal stripe artifacts under isotropic TGV. Two changes remove them:
elliptical dual-ball projections with Y-axis regularization ratio 1.5, and a data gradient
normalized by per-pixel frame coverage instead of frame count. Effect: artifact score 3.870 →
0.695 (−82%), raw-control corr 0.902 → 0.916. This is the practical classical deliverable.

## 4.2 Physics-matched synthetic platform (TCForge)

Training data are synthesized with the measured forward model, not adversarial degradations:
chip-like scene geometry rendered with 4× SSAA coverage anti-aliasing (soft [0,1] coverage masks,
so HR targets do not contain staircase edges and sub-pixel lines carry coverage-scaled amplitude);
temperature rendering T_bg + ΔT·coverage; Gaussian PSF in the credible σ range; detector box
integration; measured noise floor; shift distributions replaying the actual raster command
lattice. Pools: 1000 scenes (2x AA), plus a burst pool storing per-frame LR stacks and shifts
with K = 4 precomputed drizzle variants per scene for input-mode training (frame-subset and
shift-noise augmentation baked into variants; variant 0 is the canonical full-burst/no-noise
configuration matching inference).

## 4.3 Learned track

**Backbone (fixed across arms).** Plain UNet (base 64), bilinear HR head — a PixelShuffle head
was attribution-tested and rejected (it added stripe artifacts and worsened proxies without
reducing aliasing; reported as negative result). Architecture is deliberately held constant;
the experimental variables are the *input information path* and the *observation anchor*.

**Contour-oriented loss (conservative setting).** MSE 0.3 + high-pass structure 0.8 (σ=5) +
SSIM 0.15 + gradient-vector matching 0.15 (full (gx,gy) L1, capturing dilation/distortion that
magnitude-only edge losses miss) + edge 0.05, with mild thin-structure (×3) and gap (×2)
weighting; the earlier hot setting (structure ×4, thin ×6) is kept as a drift-amplification
reference arm. History (skeleton boost 30 era → ringing; loss cooldown A/B) is summarized in one
paragraph as motivation for conservative weights.

**Input modes.**
- *1x statistics input (baseline):* 5 channels on the 1x grid — aligned mean / median / coverage
  / variance / high-pass. This is the conventional featurization; we show it collapses the
  burst's sub-pixel phase information before the network sees it.
- *Evidence-injecting hybrid input:* 3 drizzle channels rendered on the 2x grid from the aligned
  burst (scatter mean / coverage / scatter variance) concatenated with the 5 upsampled 1x
  channels; the network runs at scale 1 on the 2x grid. Sub-pixel evidence enters as data, not
  as a learned prior. [V9A]

**Observation anchors (loss-side).**
- none / band-limited forward consistency (high-pass band of the reprojected residual, weight
  0.1) / full-band forward consistency (weight 0.1) [V9D pending] — all reproject the prediction
  through the measured operator and compare to the held 1x observation.
- *Legal anchor under hybrid input* [V9C pending]: the hybrid input's channel 0 is an upsampled
  mean (not a valid 1x observation), so the anchor consumes the original 1x aligned-mean patch
  carried separately through the data pipeline (even-origin crops; augmentation-synchronized).

**Checkpoint selection (part of the method, not an afterthought).** Per arm, normalize the proxy
pair (artifact score ↓, raw-control corr ↑) to [0,1], take the 3 steps closest to the ideal point
(≥5K-step separation), always carry the final step as a drift reference, and gate the mechanical
choice with visual panels (temperature view, not high-pass only). The 60K endpoint is *never*
the default deliverable.

> TODO(§4): finalize V9A/V9C details and the hybrid-anchor plumbing description after Codex
> lands the code; add loss-equation block; pseudo-code for selection rule (5 lines).
