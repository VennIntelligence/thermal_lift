# Dataset, Physics, and Validation Constraints

## Real Dataset

The project dataset is a micro-scanned LWIR chip-inspection sequence.

Raw inputs:

- TXT temperature matrices: 480 x 640, Celsius values.
- BMP renderings: same-name visual references only.
- AVI continuous scans: independent rendered 8-bit diagnostic videos, not SR
  inputs.

Main SR input:

- Session=2 only.
- 255 frames.
- Same acquisition temperature segment.
- Step-and-shoot raster path.

Coordinate set:

```text
{0,2,4,6,8,10,12,14,16,18,20,24,28,32,36,40} um
```

Important scan-path behavior:

- Row-wise X neighbors are truly time-adjacent.
- Fixed-X Y neighbors are separated by about a full row of acquisition time.
- Therefore Y-only coordinate-neighbor NCC is contaminated by thermal evolution
  and cannot be used as quantitative displacement truth.

## Coordinate Model

The project uses:

```python
THETA_DEG = 47.6
PIXEL_SIZE_UM = 10.0

def coordinate_to_shift(x_um, y_um):
    theta = np.radians(THETA_DEG)
    dx = (x_um * np.cos(theta) + y_um * np.sin(theta)) / PIXEL_SIZE_UM
    dy = (-x_um * np.sin(theta) + y_um * np.cos(theta)) / PIXEL_SIZE_UM
    return dx, dy
```

Rules:

- Stage command is a prior.
- Actual alignment must be data-constrained.
- Do not treat filename coordinates or stage command as ground truth shifts.

## Thermal and Optical Constraints

The physical hierarchy is:

1. Optical/LWIR PSF and thermal diffusion determine what information reaches the
   detector.
2. Detector sampling pitch samples that already blurred field.
3. Multi-frame micro-scanning can recover information only if it survived the
   PSF/noise floor and if phase diversity and alignment are good.
4. Deep or cross-modal methods can add plausible structure, but that structure
   must be audited for hallucination.

Key constants:

| Quantity | Value |
|---|---:|
| LWIR waveband | 8-14 um |
| Detector sampling pitch | 10 um/pixel |
| Current calibrated spatial resolution | 20 um |
| Noise floor | 0.0724 C |
| Approx PSF | Gaussian sigma about 0.5 px, maybe 0.2-0.5 px |

## Why 2x Is the Default

EP03 supports 2x because:

- 2x gives a 5 um output grid, which is a useful subpixel grid for contour
  representation.
- Main session phase bins are well populated at 2x.
- Many contour segments have high local SNR.
- Alignment anchors and holdout validation exist.

However:

- 2x grid does not mean 5 um true spatial resolution.
- 2x contour improvement must be shown through structure consistency, not display
  magnification.

## Why 4x/8x Is Different

4x and 8x are not forbidden as experiments. They are risky as claims.

Reasons:

- EP03 MTF at 4x Nyquist is extremely low under realistic PSF assumptions.
- The 4x grid is 2.5 um per sample, far below the calibrated 20 um spatial
  resolution.
- Any 4x/8x method can easily produce visually pleasing edges by prior injection
  or regularization.
- Optical guidance can make outputs look more correct but can also inject
  visible-only geometry into thermal images.

Correct 4x language:

- "4x contour visualization".
- "4x method stress test".
- "4x synthetic benchmark result".
- "4x optical-guided candidate, not pure thermal resolution proof".

Incorrect 4x language:

- "4x thermal spatial resolution achieved".
- "2.5 um thermal structure resolved".
- "5 um/2.5 um absolute-temperature metrology".

## Optical Reference Role

Optical reference is valuable, especially for human hallucination audit, but it
is not a registered thermal ground truth in the current project.

Recommended roles:

- Geometry prior: identify chip layout and plausible structure.
- Human audit: decide whether SR output invents impossible geometry.
- ROI selection: choose zones where optical structure exists and thermal signal
  should plausibly respond.
- Failure classification: distinguish "optical-only" edges from thermal-supported
  edges.

Not allowed without additional registration and physics:

- Forcing thermal SR to match optical boundaries.
- Counting all optical edges as thermal truth.
- Using optical agreement alone to claim thermal SR success.

## Four-Way Reality Audit

For each candidate output feature:

| Optical | Raw-control | Split-half | Interpretation |
|---|---|---|---|
| present | present | stable | strong candidate true structure |
| present | weak/absent | unstable | optical-supported but not thermal-supported |
| absent | present | stable | possible real thermal structure or optical mismatch |
| absent | absent | unstable | likely hallucination/artifact |

This table should become a core EP07 validation tool.
