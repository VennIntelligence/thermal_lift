# EP09 — PSF Sigma Calibration Report

## Executive Summary

EP09 estimates the Gaussian PSF sigma as **0.226 LR px** with route spread **1.010 LR px**. The 4x gate verdict is **not_cleared_inconsistent_routes**.

| Route | Sigma LR px | 95% CI LR px | N | Role |
|---|---:|---:|---:|---|
| A_forward | 0.226 | [0.208, 0.240] | 248 | primary |
| B_esf | 1.129 | [1.041, 1.215] | 33 | cross_check |
| C_joint | 0.119 | n/a | 32 | cross_check |

## Gate Results

- Route consistency within 0.05 px: **False**
- Forward residual single-minimum gate: **True**
- 95% CI width <= 0.10 px: **True** (width=0.032 px)
- ESF valid segment count >= 10: **True**
- Overall gate: **False**

## Route A: Forward-Model Residual

Route A uses the EP06 MAP-TV 2x highpass reconstruction as a pseudo-HR scene and sweeps sigma in the EP06 forward model. The score is the cropped LR highpass residual against the 248 clean main-session observations loaded from the EP06 SR metadata filter, split by acquisition order into train and validation frames.

- Sigma: **0.2257 LR px**
- EP06 pseudo-HR input: `output/ep06_sr_poc/map_tv_highpass.npy`
- Train/validation sigma delta: **0.0047 px**
- Relative residual depth vs best edge: **0.0010**
- Minimum at grid edge: **False**

## Route B: 1D ESF Fitting

Route B fits an error-function edge model on EP04 outer contour anchors after contrast/projection/R2 gates. It is an independent check, but it can include true thermal edge width in addition to the optical PSF.

- Median sigma: **1.1286 LR px**
- Valid segments: **33**

## Route C: Joint MAP-TV Hold-Out Sweep

Route C reconstructs a short-budget MAP-TV HR image for each candidate sigma on a deterministic frame subset, then scores held-out frames. It is intentionally lower budget than EP06 and is used as a cross-check rather than the primary estimate.

- Joint sigma: **0.1190 LR px**
- Grid minimum: **0.1000 LR px**

## 4x Decision

4x is not cleared because the independent calibration routes disagree beyond the EP09 gate.

The decision is conservative: 4x should not be promoted unless the calibrated sigma, route consistency, and confidence interval all clear the gates.
