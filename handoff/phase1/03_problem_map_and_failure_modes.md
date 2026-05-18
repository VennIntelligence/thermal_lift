# Problem Map and Failure Modes

## Current Technical Problem

The project has moved past "can we output a 2x image?" The current question is:

> Which reconstructed contours are supported by real thermal evidence, and which
> are artifacts, priors, or cross-modal hallucinations?

This matters because EP06 already produced visually stronger images, especially
with MAP-TV. The next risk is overclaiming.

## Known Failure Modes

### 1. Display Magnification Mistaken for SR

Bicubic and dense output grids can look sharper without adding information.

Mitigation:

- Always show LR, bicubic, SAA, IBP, MAP-TV side by side.
- Use forward-model consistency and held-out validation.
- Explicitly label grid pitch versus spatial resolution.

### 2. Back-Projection Residual Mistaken for Clarity

Low reconstruction residual can mean over-smoothing or fitting the measurement
model, not improved contour visibility.

Mitigation:

- Use residual only as a sanity check.
- Pair with edge transfer, split-half agreement, artifact score, and ROI visual
  audit.

### 3. Sharpness Metric Mistaken for Success

Gradient magnitude, Tenengrad, and P95 gradient can increase because of ringing,
noise, or false edges.

EP06 example:

- MAP-TV has the highest gradient and best Chamfer proxy.
- It also has the highest artifact score.

Mitigation:

- Report artifact score and split-half alongside sharpness.
- Use raw-control agreement.
- Use human optical audit to flag impossible geometry.

### 4. Optical Guidance Hallucination

Optical images can reveal chip geometry that is real in visible light but not
thermally resolved in LWIR. A guided model can transfer optical lines into the
thermal output even when the thermal measurements do not support them.

Mitigation:

- Separate "optical-supported" from "thermal-supported".
- Use optical only for audit/guidance, not as ground truth unless a registration
  and thermal-response model are built.
- Require held-out LR re-projection consistency.

### 5. Highpass Preprocessing Artifacts

Highpass inputs can create attractive red/blue lobe structures that are not
ordinary temperature contours.

EP06 mitigation already added:

- Raw-temperature control track.
- Output-side highpass comparison.

Future mitigation:

- Require both highpass and raw-control tracks for any new method.
- Add ordinary raw-temperature crop figures for central internal structures.

### 6. 4x/8x Claim Inflation

Many papers report 4x or 8x, but the meaning varies:

- Output-grid factor.
- Synthetic downsampling factor.
- RGB/visible-guided generation factor.
- Perceptual benchmark factor.
- True multi-frame physics recovery factor.

Our project cannot import those numbers without matching assumptions.

Mitigation:

- In real data, 4x is only an exploratory visualization/stress test.
- In synthetic data, 4x/8x can be measured against HR thermal truth.
- Separate "benchmark success" from "real LWIR system resolution".

### 7. Synthetic Benchmark Too Easy

EP06 synthetic tests are smoke tests. They are useful for code sanity, but they
are too simple to decide real-method value.

Current EP06 synthetic truth:

- Simple sinusoidal/shape field.
- Random subpixel shifts.
- Gaussian PSF.
- Additive Gaussian noise.

Needed improvement:

- Chip-like geometry.
- Material-dependent temperatures and emissivity.
- Thermal diffusion / broad edges.
- Drift and offset changes.
- Bad frames / patch-level local failures.
- PSF variation and detector sampling.
- 2x/4x/8x benchmark tracks.

### 8. External Dataset Domain Mismatch

Most public thermal SR datasets are scenes, humans, vehicles, or UAV images.
They may not resemble industrial chip inspection.

Mitigation:

- Use external datasets for algorithm screening and pretraining only.
- Use synthetic chip-like benchmark for controlled truth.
- Use real main session for final plausibility and failure audit.

## Current Bottleneck Diagnosis

The bottleneck is not simply algorithm power. It is validation:

- We need to know whether sharper edges are thermal-supported.
- We need a true-HR benchmark to compare 2x/4x/8x algorithms.
- We need optical-audit protocols to reject hallucinations.
- We need to understand which literature results are comparable to our physical
  setting and which are not.

## Practical Claim Levels

| Level | Evidence Needed | Claim |
|---|---|---|
| L0 | visually nicer than LR | exploratory visualization |
| L1 | split-half stable + raw-control agrees | thermal-supported contour enhancement |
| L2 | synthetic HR truth metrics pass | method works under modeled degradation |
| L3 | real data + optical audit + holdout + re-projection pass | credible 2x contour-level POC |
| L4 | independent thermal ground truth or calibrated target | spatial-resolution/metrology claim |

Current EP06 is around L2/L3 for 2x contour-level POC, depending on ROI. It is
not L4.
