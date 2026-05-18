# Synthetic Benchmark Plan

## Motivation

The real dataset has no registered HR thermal truth. Optical imagery helps human
audit, but it is not thermal ground truth. A generated benchmark can provide true
HR thermal maps and exact subpixel motion, allowing rigorous comparison of 2x,
4x, and 8x methods.

This should become the backbone of EP08.

## Benchmark Principle

Start from a high-resolution thermal truth:

```text
HR thermal truth
  -> optical/thermal PSF
  -> detector integration/downsample
  -> subpixel micro-shifts
  -> noise + drift + offsets
  -> LR burst frames
  -> SR algorithms
  -> compare against HR truth
```

The benchmark should generate both:

- HR truth temperature fields.
- LR observed frames with known shifts and metadata.

It should also optionally generate:

- Optical-like structure map.
- Material mask.
- Emissivity map.
- Ground-truth contour labels.
- Frame-level and patch-level quality masks.

## Synthetic Scene Types

### 1. Minimal Analytical Chip Shapes

Purpose:

- Fast unit tests.
- Algorithm sanity checks.
- Known edges and simple geometry.

Features:

- Rectangles, traces, pads, vias, diagonals, rounded corners.
- Internal holes and narrow channels.
- Controlled edge widths and contrasts.

### 2. Chip-Like Procedural Thermal Fields

Purpose:

- Better mimic the real chip layout.
- Test contour-level SR without requiring a full physics renderer.

Features:

- Material regions with different temperature offsets.
- Smooth thermal diffusion around boundaries.
- Spatially varying background gradient.
- Local hot/cold spots.
- Edge widths drawn from realistic PSF/thermal diffusion ranges.

### 3. Physics Renderer / External Generator

Purpose:

- Use existing thermal rendering tools if available.
- Produce scenes closer to real radiometry and material/emissivity behavior.

Candidate tools to investigate:

- DIRSIG: physics-driven EO/IR synthetic image generation.
- ThRend: ray-tracing LWIR thermography renderer.
- ThermalSynth: Unity-based synthetic thermal scenario generator, likely more
  human/scenario-oriented than chip-oriented.
- Other thermal generation datasets/tools that the research agent should find.

### 4. Optical-Guided Paired Synthetic Data

Purpose:

- Evaluate guided thermal SR and hallucination.

Outputs:

- HR optical-like geometry.
- HR thermal truth that intentionally does not include every optical edge.
- LR thermal burst.

This lets us measure whether optical-guided methods inject optical-only
structures into thermal output.

## Degradation Parameters

Use project-calibrated defaults first:

| Parameter | Default |
|---|---:|
| LR shape | 480 x 640 or smaller crop for dev |
| HR scale | 2x, 4x, optional 8x |
| Detector pitch | 10 um/pixel |
| Spatial resolution | 20 um equivalent |
| PSF sigma | sweep 0.2, 0.35, 0.5 px |
| Noise sigma | 0.0724 C default, plus sweeps |
| Drift | slow per-frame offset/gain |
| Shifts | project-like 255-frame micro-scan path |
| Phase coverage | full 2x, stressed 4x, imperfect coverage cases |

Shift design:

- Use the real coordinate set and theta=47.6 deg.
- Add data-driven shift noise.
- Include patch-local residual shifts as a stress condition.

## Metrics

Full-image metrics:

- PSNR.
- SSIM.
- NRMSE.
- Frequency transfer / MTF recovery proxy.

Contour metrics:

- Chamfer distance to HR truth contours.
- Edge F1 at multiple tolerances.
- ESF width error.
- Ringing amplitude.
- False edge rate.
- Missed edge rate.

Thermal metrics:

- Temperature RMSE on smooth regions.
- Boundary contrast preservation.
- Bias under drift/offset correction.

Stability metrics:

- Split-half SR consistency.
- Held-out LR frame prediction.
- Re-projection residual on held-out frames.

Hallucination metrics:

- Optical-only edge injection rate in paired synthetic optical/thermal data.
- Unsupported-edge count after thresholding.
- Human-audit panels for representative failure cases.

## Method Matrix

Initial benchmark methods:

| Family | Methods |
|---|---|
| Baseline | LR, bicubic, Lanczos |
| Classic multi-frame | SAA, SAA-weighted, drizzle |
| Robust fusion | patch-lucky SAA/drizzle |
| Forward model | IBP |
| Regularized | MAP-TV, MAP-Huber, MAP-TGV |
| PnP | BM3D/NLM/DnCNN prior with forward model |
| Self-supervised | DIP/DeepIR-like, zero-shot burst SR |
| Deep burst | Deep Burst SR-like, EDVR/BasicVSR-like |
| Guided | optical-guided thermal SR with hallucination audit |

## Expected Decisions

The synthetic benchmark should answer:

1. Under our PSF/noise/shift conditions, what is the realistic 2x upper bound?
2. Does 4x recover true contour geometry or mostly sharpen priors?
3. Does 8x ever help under realistic LWIR PSF, or only under synthetic-easy cases?
4. Which algorithms maintain low false-edge rate?
5. Which methods transfer to the real EP06/EP07 data without inventing optical
   or regularizer-driven structures?

## Deliverables

Recommended EP08 outputs:

- `data/synthetic/ep08_benchmark_manifest.json`
- `output/ep08_synthetic_benchmark/benchmark_summary.csv`
- `output/ep08_synthetic_benchmark/method_by_scale_metrics.csv`
- `output/ep08_synthetic_benchmark/false_edge_audit.csv`
- `reports/ep08_synthetic_benchmark/synthetic_benchmark_report.md`
- `notebooks/ep08_synthetic_benchmark/fragments/`

## Minimum Viable EP08

If time is limited:

1. Procedural chip-like HR truth.
2. Real 255-frame shift path.
3. PSF sigma sweep: 0.2, 0.35, 0.5.
4. Methods: bicubic, SAA, drizzle, IBP, MAP-TV.
5. Scales: 2x and 4x.
6. Metrics: PSNR, SSIM, contour Chamfer, false-edge rate, split-half.

This is enough to decide whether deeper methods are worth the implementation
cost.
