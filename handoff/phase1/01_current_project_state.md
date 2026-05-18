# Current Project State

## Project Goal

`thermal_lift` is a LWIR micro-scanning super-resolution project for industrial
chip inspection. The practical goal is not to prove 5 um absolute-temperature
metrology. The current goal is to make chip internal structures and shape
contours clearer and more stable from the 255-frame main TXT temperature session.

The current official scope is:

- 2x contour-level SR POC on real LWIR temperature matrices.
- Use optical/BMP imagery for visual context and later hallucination audit, not
  as registered thermal ground truth.
- Use stage command only as displacement prior, initialization, or regularizer.
- Use data-driven alignment, split-half checks, raw-control tracks, and holdout
  contours to constrain claims.

## Core Facts

| Item | Current Value |
|---|---:|
| Detector output | 640 x 480 pixels |
| TXT sampling pitch | 10 um/pixel |
| Calibrated spatial resolution | 20 um |
| Waveband | LWIR 8-14 um |
| Main session | 255 frames, session=2 |
| Noise floor | 0.0724 C |
| Stage-to-pixel rotation | theta=47.6 deg |
| Scan coordinates | {0,2,4,6,8,10,12,14,16,18,20,24,28,32,36,40} um |
| Current SR output | 2x grid, 960 x 1280 |
| Current best classic method | MAP-TV as sharp candidate; IBP as conservative candidate |

The key distinction remains:

- `10 um/pixel` is detector sampling pitch.
- `20 um` is calibrated optical/spatial resolution.
- `5 um grid` is a 2x reconstruction grid, not a 5 um resolved thermal feature
  claim.
- `2.5 um grid` from 4x is an even denser display/reconstruction grid and needs
  much stronger evidence.

## EP01-EP06 Summary

### EP01: Data Processing and Audit

EP01 established the usable input set and the time/session model.

Results:

- All 263 TXT temperature matrices are readable 480 x 640 arrays.
- TXT is the numeric SR input; BMP is visual reference only.
- The correct session detection uses `acquisition_order` from file mtime, not
  renamed filename sorting.
- The true structure is 3 temperature sessions.
- Main SR input is session=2 with 255 frames.
- Cross-session mixing is forbidden because temperature jumps dominate noise.

Implication:

All later SR work should read `output/ep01_data_processing/frame_audit.csv` and
filter `is_main_session=True`.

### EP02: Displacement Calibration and Rotation

EP02 corrected the coordinate model and validated the direction logic.

Results:

- Stage coordinate to pixel coordinate requires theta=47.6 deg.
- AVI gradient NCC gave theta about 47.14 deg with 95% CI covering 47.6 deg.
- AVI files are rendered 8-bit videos with many duplicate frames and are not SR
  inputs.
- Y-only coordinate-neighbor TXT pairs are not valid quantitative displacement
  calibration because raster acquisition creates time gaps of about a row.

Implication:

Stage command is a useful prior, not alignment truth. Data-driven alignment is
mandatory.

### EP03: Theoretical Limits and Local Observability

EP03 established the physical risk boundary.

Results:

- 2x contour-level POC is physically reasonable.
- 4x is not rejected forever, but it is high risk and not a default delivery
  claim.
- Many local inner and outer contour segments have SNR far above the 0.0724 C
  noise floor.
- Local ESF/CRB can support alignment anchors and quality gates, but not final
  shape reconstruction claims.

Important MTF numbers at Nyquist:

| Grid | Frequency | MTF sigma=0.20 | MTF sigma=0.35 | MTF sigma=0.50 |
|---|---:|---:|---:|---:|
| 1x | 0.5 cyc/px | 0.821 | 0.546 | 0.291 |
| 2x | 1.0 cyc/px | 0.454 | 0.089 | 0.007 |
| 4x | 2.0 cyc/px | 0.042 | 0.000063 | 0.000000003 |

Implication:

4x can be a stress test or contour visualization ablation, not a clean physical
resolution claim without additional forward-model and validation evidence.

### EP04: Alignment Anchor Benchmark and Quality Gates

EP04 separated localization anchors from final SR claims.

Results:

- Outer contour gives more stable anchor coverage.
- Inner contour pass rate is lower, but passed inner segments can be as stable
  as outer ones.
- EP04 produced alignment input segments, holdout validation segments, and
  `sr_target_not_truth` regions.

Key EP06 role counts:

| Contour | alignment_input | holdout_validation | sr_target_not_truth |
|---|---:|---:|---:|
| outer | 26 | 19 | 39 |
| inner | 40 | 43 | 303 |

Implication:

Failed internal localization segments are not abandoned. They are exactly the
hard internal-structure SR targets, but they cannot be used as truth.

### EP05: 2x Capacity and Alignment Baseline

EP05 established that the main session has enough 2x phase coverage and a usable
alignment baseline.

Results:

- 2x phase bins are all occupied, about 58-69 frames per bin depending on method.
- Data-driven contour refinement gives the best held-out Chamfer median.
- NCC init and filename affine fit provide smoother phase priors.
- Refined Chamfer should not be used to claim 4x because local refinement can
  absorb offsets.

Representative alignment metrics:

| Method | Held-out Chamfer median | Gradient corr median | 2x bins |
|---|---:|---:|---:|
| no alignment | 0.3813 px | 0.7023 | 1/4 occupied |
| stage prior | 0.2402 px | 0.8817 | 4/4 occupied |
| filename affine | 0.1708 px | 0.9551 | 4/4 occupied |
| NCC init | 0.1563 px | 0.9668 | 4/4 occupied |
| contour refined | 0.1341 px | 0.9487 | 4/4 occupied |

Implication:

2x SR has a real data foundation. 4x remains a risk item.

### EP06: Classic 2x Contour-Level SR POC

EP06 implemented classic physical SR methods on the main session:

- LR reference.
- Bicubic reference.
- SAA-uniform.
- SAA-weighted.
- IBP.
- MAP-TV.

It used two tracks:

- Highpass-input main track: structure-map reconstruction.
- Raw-temperature control track: raw frame SR with output-side highpass
  visualization.

Results:

- All arrays are finite and have expected 2x shape `(960, 1280)`.
- Synthetic smoke test passed: SAA 28.43 dB, IBP 28.78 dB, MAP-TV 29.03 dB.
- SAA suppresses frame noise and drift.
- IBP adds modest forward-model sharpening.
- MAP-TV gives the sharpest contour proxy but also the highest artifact score.
- Raw-control outputs reproduce major chip structures after output highpass,
  reducing the risk that highpass preprocessing alone invented structures.

Highpass main-track metrics:

| Method | Mean gradient | P95 gradient | Artifact score | Chamfer proxy |
|---|---:|---:|---:|---:|
| Bicubic | 0.4521 | 0.8844 | 0.3049 | 0.0379 |
| SAA-uniform | 0.1613 | 0.4420 | 0.1414 | 0.0111 |
| SAA-weighted | 0.1610 | 0.4427 | 0.1410 | 0.0111 |
| IBP | 0.1788 | 0.4987 | 0.1455 | 0.0111 |
| MAP-TV | 0.2226 | 0.6592 | 0.1631 | 0.0000 |

Implication:

EP06 supports a 2x contour-level POC, not 4x SR, 5 um true spatial resolution, or
absolute-temperature SR. IBP is the conservative candidate; MAP-TV is the
sharpness upper-bound candidate that needs artifact controls.

## Current Decision Point

The next decision is not simply "try 4x". The better framing is:

1. Build a stronger validation layer using real data, split-half consistency,
   raw-control agreement, and optical human audit.
2. Build a synthetic benchmark with known HR thermal truth and realistic
   micro-scan degradation.
3. Use that benchmark to evaluate classic, cross-domain, and self-supervised
   methods at 2x/4x/8x.
4. Only then decide what EP07 and EP08 should claim.
