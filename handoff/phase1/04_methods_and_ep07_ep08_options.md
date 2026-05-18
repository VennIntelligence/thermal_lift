# Methods and EP07/EP08 Options

## Current Implemented Methods

EP06 implemented classic 2x SR:

| Method | Role | Current Interpretation |
|---|---|---|
| LR reference | detector-grid baseline | real input, no SR |
| Bicubic | display interpolation baseline | controls for magnification |
| SAA-uniform | multi-frame baseline | validates phase diversity and denoising |
| SAA-weighted | quality-gated SAA | nearly same as uniform on current data |
| IBP | forward-model baseline | conservative sharpening candidate |
| MAP-TV | regularized baseline | sharpness upper-bound candidate with artifact risk |

## Candidate Method Families

### A. Drizzle / Variable-Pixel Linear Reconstruction

Why relevant:

- Designed for undersampled, dithered images.
- Can preserve photometry and use weights.
- Has coverage/weight maps that are useful for explaining unreliable regions.
- Very close to our micro-scanning setting.

How to adapt:

- Replace current SAA backfill with drizzle-style `pixfrac`.
- Use EP05 shifts and EP04/EP06 quality weights.
- Generate coverage, weight, and uncertainty maps.
- Run 2x and 4x visualization.

Expected value:

- Strong explainability.
- Low hallucination risk.
- Good bridge between classic EP06 and astronomy methods.

### B. Lucky / Quality-Gated Patch Fusion

Why relevant:

- Real thermal sequence has local drift and patch-specific alignment failures.
- Whole-frame weighting did not change SAA much, but patch-level weighting may.

How to adapt:

- Divide image into patches/ROIs.
- Estimate local quality from NCC peak, local edge consistency, residual, and
  EP04 anchor proximity.
- Select or downweight bad frames per patch.
- Fuse with drizzle/SAA/IBP.

Expected value:

- More robust than global weights.
- Natural for real acquisition artifacts.

### C. Improved Regularized Physical SR

Current MAP-TV is sharp but artifact-prone.

Alternatives:

- Huber-TV.
- TGV.
- Wavelet/sparse regularization.
- Bilateral/edge-aware regularization.
- MAP with explicit PSF uncertainty.
- Joint offset/gain/drift terms.

Expected value:

- Keep forward-model discipline.
- Reduce MAP-TV blockiness/ringing.

### D. Plug-and-Play / RED

Why relevant:

- Allows stronger image priors while keeping the measurement model.
- Safer than pure generative hallucination if re-projection is enforced.

Possible denoisers:

- BM3D/NLM for conservative baseline.
- DnCNN-style denoiser.
- Thermal-specific or synthetic-trained denoiser.

Guardrails:

- Must report LR re-projection error on held-out frames.
- Must report optical hallucination audit.

### E. Self-Supervised Single-Scene / Burst SR

Candidate approaches:

- Deep Image Prior / DeepIR-like factorization.
- Zero-shot SR.
- Internal patch recurrence.
- Self-supervised burst SR with held-out frame prediction.

Why relevant:

- Real data lacks HR thermal truth.
- We can optimize on the 255-frame sequence itself.

Risks:

- Can hallucinate stable-looking edges.
- Needs strong held-out re-projection validation.
- Slow and sensitive to hyperparameters.

### F. Synthetic-Trained Deep Burst SR

Why relevant:

- We can generate chip-like synthetic bursts with known HR truth.
- Literature has strong 4x burst SR methods in mobile photography.

Possible model families to investigate:

- Deep Burst Super-Resolution style optical-flow/attention fusion.
- BasicVSR / EDVR / recurrent video SR style models.
- Transformer-based burst/video SR.
- Implicit neural representation burst SR.

Guardrails:

- Train/validate first on synthetic thermal forward model.
- Test on real data with raw-control, split-half, and optical audit.
- Do not claim real 4x resolution from synthetic success alone.

### G. Optical-Guided Thermal SR

Why relevant:

- We have clearer optical imagery.
- Human can identify obvious hallucinations.
- Literature reports high factors like 8x/16x in guided thermal SR.

Risks:

- Injects optical edges into thermal output.
- May improve visual plausibility while failing thermal measurement consistency.

Recommended use:

- Keep as separate "guided visualization" track.
- Require feature classification:
  - optical-supported and thermal-supported;
  - optical-only;
  - thermal-only;
  - unsupported.

## Proposed EP07

EP07 should be a validation episode, not a new algorithm race.

Working title:

> EP07: Real-data SR credibility audit with optical reference, raw-control,
> split-half, and edge-transfer validation.

Tasks:

1. Freeze EP06 outputs: SAA-weighted, IBP, MAP-TV.
2. Add optical/BMP/manual ROI audit panels.
3. Build the four-way feature audit table:
   optical, raw-control, split-half, SR candidate.
4. Add edge-transfer metrics:
   ESF width, lobe width, ringing amplitude, edge localization stability.
5. Add even/odd or split-half SR reconstructions for SAA, IBP, MAP-TV.
6. Produce customer-facing examples with conservative captions.

Success:

- Clearer 2x contour-level result with credible non-hallucination evidence.
- Ranked recommendation: likely IBP conservative, MAP-TV sharp candidate.
- Explicit list of regions where current data does not support a claim.

## Proposed EP08

EP08 should be a benchmark and method exploration episode.

Working title:

> EP08: Synthetic thermal micro-scan benchmark and cross-domain SR method
> screening.

Tasks:

1. Build or import a synthetic HR thermal generator.
2. Generate micro-shifted LR bursts with known truth.
3. Evaluate 2x/4x/8x methods.
4. Compare classic, astronomy, PnP, self-supervised, and deep burst methods.
5. Identify which methods transfer to real data without hallucination.

Success:

- A benchmark table with true metrics.
- A short-list of methods worth implementing in real EP09 or later.
- A defensible answer to "why do papers show 4x/8x and what can we claim?"

## Recommended Near-Term Method Order

1. Drizzle at 2x and 4x visualization.
2. Patch-lucky drizzle/SAA.
3. Improved MAP regularization: Huber-TV/TGV.
4. PnP/BM3D or NLM prior.
5. Self-supervised HR latent with held-out LR prediction.
6. Synthetic-trained burst SR.
7. Optical-guided thermal SR as a clearly separated guided visualization track.
