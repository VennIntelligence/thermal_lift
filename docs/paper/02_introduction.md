# 1. Introduction (draft)

Long-wave infrared (LWIR) thermography is a standard tool for non-destructive inspection of
semiconductor devices: defects, shorts, and process variations manifest as sub-degree temperature
patterns over chip-internal structures. The optics, however, are diffraction- and cost-limited.
The system studied here samples a 6.4 × 4.8 mm field of view at a 10 µm detector pitch with a
calibrated spatial resolution of 20 µm — internal structures of interest (thin interconnect lines,
zigzag traces, gaps between plates) live at or just below this scale. Hardware upgrades (macro
optics at ~10 µm class resolution) were evaluated by the equipment owner and judged unsatisfactory;
the practical question is whether *computational* multi-frame enhancement can make chip-internal
contours visibly and reliably clearer on the existing instrument.

Micro-scanning provides the physical basis: the stage steps the scene by sub-pixel offsets
(2–4 µm commands on a 16-position raster line) and acquires hundreds of frames, so the burst
carries sub-pixel phase diversity that a single frame does not. This is the classical multi-frame
super-resolution (MFSR) setting — with one decisive complication: **there is no high-resolution
thermal ground truth, at acquisition time or ever.** No paired HR sensor exists for this modality
and scene; optical microscopy is not radiometrically comparable and is not registered to the
thermal axis. Every standard ingredient of learned SR — paired training data, full-reference
metrics, model selection on validation PSNR — is unavailable on the real data.

This regime is common in industrial and scientific imaging, yet the SR literature treats it
mostly by transplantation: train on synthetic or proxy-paired data, deploy on the target domain,
and showcase sharper-looking outputs. The danger is well known informally — *hallucination* — but
it is rarely measured. In this paper we measure it, characterize its mechanism, and build a
2x contour-level enhancement pipeline whose claims survive the measurement.

Our study is organized around three questions:

**(Q1) Does the data contain recoverable information beyond the single-frame grid?**
We bound the coherent information content of the 248-frame clean main session with
phase-stratified split-half Fourier Ring Correlation (FRC), including positive, negative, and
drift controls: the 1/7 cutoff sits at a 17.0 µm period — real but modest headroom beyond the
20 µm resolution, and short of the 11–14 µm theoretical hope. A fine-grid MAP-TV deconvolution
anchor converts this headroom into limited but genuine contour gains (zigzag trace median FWHM
114→100 µm) and sets the classical bar that any learned method must beat.

**(Q2) Does a learned model actually receive that information?**
A controlled two-arm attribution experiment shows that with conventional 1x-grid statistical
input channels (aligned mean/median/variance/coverage), the finest-structure failure mode is
*invariant* to loss design and decoder architecture: the burst's sub-pixel phase information
collapses in the input featurization, before the network can use it. Injecting the same
information as 2x-grid drizzle channels restores it [pending V9A; wording to be finalized on
results].

**(Q3) Is the model's added "detail" anchored to the observations?**
Here we report a finding we believe is broadly relevant: networks trained on synthetic targets
drift, on real data, along directions in the **null space of the observation operator**
(shift ∘ PSF ∘ detector integration ∘ decimation). The forward-consistency loss sits at its
floor while real-data artifact proxies degrade monotonically with training — the drift is
*invisible* to observation anchoring, whether band-limited or full-band [V9D pending]. Loss-side
anchoring therefore cannot fix it; what works is evidence injection at the input and a
checkpoint-selection protocol on the proxy Pareto front, gated by visual panels.

Contributions. (C1) A fully *measured* no-GT burst-SR problem — stage-to-pixel rotation
47.6°±0.1°, PSF σ arbitrated to 0.2–0.5 LR px, noise floor 0.0724 °C, raster acquisition
structure with session gating — plus a physics-matched synthetic training platform.
(C2) A GT-free evaluation protocol combining controlled FRC, a coupled observation-anchored
proxy pair (with an analysis of why the two proxies anti-correlate by construction), and a
mechanical checkpoint-selection rule. (C3) The null-space drift finding with its remedy.
(C4) An honest classical-vs-learned benchmark in which anisotropic coverage-weighted TGV and a
fine-grid MAP-TV anchor define the acceptance gate; learned arms are adopted only where they
beat it under the protocol.

We deliberately bound the claim: the deliverable is contour-level visibility at a 2x output
grid — not 5 µm resolution, not temperature metrology, not 4x recovery. We argue this bounded,
evidence-gated formulation is the transferable template for SR in no-GT industrial regimes.

> TODO(intro): finalize Q2/Q3 wording after V9A/V9C/V9D land; insert teaser figure reference
> (F5 crop: bicubic vs TGV vs evidence-injected UNet, center zigzag ROI).
