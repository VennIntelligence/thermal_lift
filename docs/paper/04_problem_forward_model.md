# 3. Problem Setting and Calibrated Forward Model (draft)

All numbers below are measured on the system; provenance in parentheses (episode/report).

## 3.1 Instrument and data

- LWIR camera, 8–14 µm band; detector 640 × 480; **detector pitch 10 µm/pixel**
  (BMP mm-axis measurement 10.000, TXT/BMP contour cross-check 9.980, outer-mask IoU 0.9938; EP03).
- **Calibrated spatial resolution 20 µm** — pitch and resolution are distinct quantities and the
  paper keeps them separate throughout (F1 includes the pitch/resolution/output-grid diagram).
- Raw output: temperature matrices (°C, TXT), not rendered imagery. Rendered AVI/BMP exports are
  excluded as numeric input (8-bit, ~67% duplicated frames; EP01) and used only as an independent
  direction check (below).
- Noise floor: **0.0724 °C** (smooth-region adjacent-coordinate MAE; EP01/EP03). Representative
  contrasts: inner-contour median |ΔT| 1.938 °C (SNR 26.8), outer 2.490 °C (SNR 34.4) (EP03).

## 3.2 Microscan acquisition and session structure

- Step-and-shoot raster: X steps within a line (16 coordinates per line,
  X,Y ∈ {0,2,…,20,24,28,32,36,40} µm), Y steps between lines. Within-line neighbors are
  temporally adjacent (acquisition gap 1); across-line neighbors are ~16 frames apart — an
  anisotropy that matters for both alignment validation and reconstruction (Sec. 4, TGV).
- 263 frames total split by acquisition order into 3 thermal sessions; the main session holds
  255 frames; removing repeat-acquisition frames (R≠0) leaves the **248-frame clean set** used
  throughout. Cross-session temperature jumps reach a median of 3.55 °C (49× noise floor), so
  frames are never mixed across sessions (EP01).
- Acquisition order is established from file mtimes, not filenames; filename coordinates are
  identifiers, not timestamps (EP01).

## 3.3 Calibration chain (measured, with uncertainty)

**Stage-to-pixel rotation.** θ = **47.6° ± 0.1°**, calibrated on day one (EP02); an independent
gradient-NCC estimate from continuous-scan videos gives θ ≈ 47.14° with 95% CI [46.36°, 47.92°],
covering the calibrated value (EP02 Phase 9). Stage commands map to pixel shifts via θ and the
10 µm pitch; a 40 µm command corresponds to 4.0 px. **Commands are priors, initializations, and
regularization targets — never alignment ground truth.**

**PSF.** Three estimation routes disagree beyond tolerance if applied naively
(forward-residual 0.226, ESF fitting 1.129, joint hold-out 0.119 LR px; spread 1.01 px; EP09),
which we resolve by arbitration (EP15 M3): multi-edge ESF shows outer-frame apparent widths
(median 1.015 LR px) are dominated by thermal/geometric edge width, while strong interior edges
give 0.747 and the sharpest single edge 0.546; FRC-shape fitting of a Gaussian-PSF × 10 µm
box-aperture MTF² prefers σ = 0.2 LR px and degrades monotonically toward 1.0. The credible
optical range adopted is **σ ∈ [0.2, 0.5] LR px**, and all PSF-dependent computations either scan
this range or report σ explicitly. The ESF-vs-forward discrepancy itself is reported as a finding:
apparent edge width = PSF ⊗ thermal edge width.

**MTF/SNR feasibility (necessary conditions, not success proofs).** At the 2x output grid's
Nyquist period (10 µm), Gaussian-PSF MTF is 0.454 / 0.089 / 0.007 for σ = 0.2 / 0.35 / 0.5 LR px;
at 4x it is ≤ 0.042 and effectively zero for σ ≥ 0.35 (EP03). Effective SNR = ΔT·MTF/noise marks
2x as conditionally feasible for the measured contrasts and 4x as out of reach for all but the
most optimistic σ — which is why the deliverable grid is 2x and finer grids appear only as
contour oversampling (EP12's 4x network failure is consistent with this bound).

**Alignment and gates.** Data-driven alignment (high-pass NCC initialization + contour
refinement; EP05) reduces held-out contour Chamfer distance from 0.381 px (no alignment) and
0.240 px (stage prior) to **0.134 px**, while preserving all four 2x phase bins. A localization
benchmark over 84 outer and 390 inner contour segments × 13 scanlines (EP04) grades anchors into
alignment-input / holdout-validation / SR-target-not-truth roles (A-class split-half median
~0.027 px); inner-contour regions failing the gate are retained as *targets* whose visibility the
reconstruction should improve — they are never used as truth.

## 3.4 Observation model

With HR image x on the 2x grid, frame k's observation is
y_k = D B H S_{t_k} x + n_k, where S shifts by the (refined) sub-pixel offset, H is the Gaussian
PSF (σ in the credible range), B is the 10 µm detector box integration, D decimates to the 1x
grid, and n_k has the measured noise floor. This operator family — its measured parameters, its
band attenuation, and crucially its **null space** — organizes the method (Sec. 4), the protocol
(Sec. 5), and the drift analysis (Sec. 6).

## 3.5 Claim boundary

The target is contour-level visibility of chip-internal structure on a 2x output grid (5 µm
sample, 10 µm Nyquist period). A 5 µm output sample is not 5 µm spatial resolution; no temperature
metrology is claimed; the finest interior lines (~1–2 px at 10 µm pitch) sit at the resolution
limit and full restoration is not on the physical menu — the question is whether their contours
become clearer and more stable than LR/bicubic under honest gates.

> TODO(§3): F1 composite (instrument schematic + raster pattern + calibration chain + grid
> diagram via `plot_sampling_resolution_diagram`); decide whether EP04 gate table goes to
> supplementary.
