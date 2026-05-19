# ThermalChipPhantom / TCForge Plan

> Status: planning document
> Benchmark name: **ThermalChipPhantom**
> Generation engine: **TCForge**
> Purpose: provide a controlled synthetic benchmark and optional training source for chip-like LWIR multi-frame microscan SR.

---

## 1. Core Decision

We will split the work into two named artifacts:

| Artifact | Role | Location |
|---|---|---|
| **TCForge** | Deterministic procedural generation engine | source code under `core/src/thermal_core/synthetic/` |
| **ThermalChipPhantom** | Generated benchmark dataset and manifests | generated artifacts under `data/synthetic/` and summaries under `output/` |

Code must not live under `data/` because `data/` is ignored by Git. Generated scenes, cached bursts, patch shards, and large arrays should live under `data/synthetic/` or `output/`, depending on whether they are reusable data or report products.

Recommended repo layout:

```text
core/src/thermal_core/synthetic/
├── __init__.py
├── geometry.py          # chip-frame geometry primitives and masks
├── physics.py           # thermal field, noise, drift, highpass helpers
├── forward.py           # synthetic forward model modes
├── manifest.py          # metadata schema, manifest writing, hashes
├── evaluate.py          # GT-aware synthetic metrics
└── visualization.py     # compact sanity-check plots

configs/synthetic/
├── phantom_smoke.json
├── phantom_benchmark.json
├── phantom_patches.json
└── shift_profiles.json

scripts/
├── generate_thermal_chip_phantom.py
├── evaluate_thermal_chip_phantom.py
└── smoke_test_thermal_chip_phantom.py

data/synthetic/
├── thermal_chip_phantom/
│   ├── manifest.csv
│   ├── dataset_metadata.json
│   └── scenes/
└── tcforge_cache/

output/thermal_chip_phantom/
├── smoke_test_report.md
├── baseline_metrics.csv
└── figures/
```

Use JSON configs first. Do not introduce YAML unless we explicitly add `pyyaml` to the root project.

---

## 2. Why This Dataset Exists

The real main session has 255 LWIR temperature frames but no HR ground truth. Real-data validation must therefore rely on forward consistency, split-half stability, FRC/edge-MTF proxies, and visual contour checks. ThermalChipPhantom fills a different role:

1. provide HR ground truth for method ranking under a known physical degradation chain;
2. quantify boundary recovery with PSNR, SSIM, Chamfer, Boundary F1, topology, and ESF width;
3. tune WIRE/DIP/Deep Decoder hyperparameters before spending time on real-data runs;
4. optionally generate patches for a narrow-domain denoiser or diffusion prior later.

This dataset must not be used to claim real chip resolution by itself. It is a controlled diagnostic environment, not optical ground truth for the real sample.

---

## 3. Physical Constants and Units

Use the confirmed project constants:

| Parameter | Value | Notes |
|---|---:|---|
| LR detector grid | `480 x 640` | rows x cols |
| Detector pitch | `10 um / LR px` | sampling pitch, not spatial resolution |
| Current spatial resolution | `20 um` | calibrated resolution |
| Main session frames | `255` | session 2 only |
| Noise floor | `0.0724 C` | Gaussian default noise sigma |
| Stage-to-pixel theta | `47.6 deg` | prior / geometry orientation, not alignment truth |
| Default PSF | `psf_sigma_lr_px = 0.5` | convert internally to HR sigma as `sigma_hr = sigma_lr * scale` |

All geometry generator parameters should be stored in physical micrometers first, then converted to HR pixels. This avoids confusing 2x HR pixel size with true optical resolution.

Important wording:

- `5 um` is the 2x HR grid pitch, not a resolution claim.
- `10 um` features are below current 20 um resolution and should be labeled stress cases.
- `20 um` features are near the current resolution limit.
- `30-80 um` features are safer contour-level cases.

---

## 4. Forward Model Modes

TCForge should support two forward-model modes from day one.

### 4.1 `exact_ep06_point`

This mode reproduces the current EP06 matrix-free forward convention as closely as possible:

```text
x_hr
  -> Gaussian blur with sigma = psf_sigma_lr_px * scale
  -> sample detector-center coordinates using shift=(dx, dy)
  -> add noise / drift / bad pixels
  -> optional LR highpass
```

The shift convention must match EP05/EP06:

- `shift=(dx, dy)` is in LR pixels.
- It is the displacement that moves an observed LR frame into the reference coordinate system.
- EP06 `forward()` predicts the raw observation by sampling reference HR scene at detector positions plus that alignment shift.

This mode is required for fair smoke tests against existing SAA/IBP/MAP-TV code.

### 4.2 `physical_block_average`

This mode models detector pixel integration more physically:

```text
x_hr
  -> subpixel shift in reference/observation convention
  -> Gaussian blur
  -> block average over scale x scale HR pixels
  -> add noise / drift / bad pixels
  -> optional LR highpass
```

This mode is useful for testing model mismatch and for later PyTorch forward work. It must be reported separately from `exact_ep06_point`; never mix metrics without labeling the mode.

### 4.3 Scale Policy

Initial implementation should support:

| Scale | Status | Purpose |
|---:|---|---|
| `2` | P0 | EP06-compatible benchmark |
| `4` | P1 | EP08 research / prior-assisted visualization |

EP06 code currently rejects scale other than 2. For 4x, TCForge can generate GT and LR bursts, but EP06 classic solvers cannot be reused unchanged.

---

## 5. Highpass and Raw Tracks

Highpass must be applied at the same conceptual place as real processing:

```text
HR raw temperature field
  -> forward model
  -> LR raw burst
  -> add noise / drift / bad pixels
  -> LR highpass burst
```

Save both raw and highpass tracks:

| File | Meaning |
|---|---|
| `hr_mask_{scale}x.npy` | binary material mask GT |
| `hr_temperature_{scale}x.npy` | raw HR thermal field GT |
| `hr_edge_map_{scale}x.npy` | edge GT derived from mask |
| `lr_burst_raw.npy` | simulated LR observations before highpass |
| `lr_burst_highpass.npy` | simulated LR structure maps after highpass |
| `shifts.npy` | exact shifts used, `(N, 2)` with columns `[dx, dy]` |

`hr_highpass_{scale}x.npy` may be saved only as an auxiliary visualization target. It should not be treated as the primary physical GT unless the metric explicitly says it is a highpass-domain metric.

---

## 6. Geometry Generator

Generate a sharp material mask in chip coordinates, then rotate it into detector coordinates.

P0 primitives:

| Primitive | Parameters |
|---|---|
| rectangle | center, width, height |
| frame / border | outer rectangle minus inner rectangle |
| pin / slot | narrow rectangle |
| pin array | count, pitch, width, length, orientation |
| L shape | union of two rectangles |
| T / cross shape | union of rectangles |
| parallel grooves | repeated narrow cutouts |

P1 refinements:

| Feature | Purpose |
|---|---|
| corner radius | avoid unrealistically perfect stair-step corners |
| edge roughness below 1 HR px | controlled realism stress test |
| missing / broken pins | topology tests |
| variable emissivity contrast | future physical realism |

Mask semantics:

- `mask = 1`: chip/material/foreground.
- `mask = 0`: background, air, substrate, or cutout.
- `temperature = T_bg + delta_T * mask + low_freq_background`.

Rotation:

```text
rotation_deg = 47.6 + jitter
default jitter = uniform(-1.0, 1.0) deg
```

The rotation is a geometry prior, not proof that stage command is alignment truth.

---

## 7. Difficulty Tiers

Use four tiers rather than three, so below-resolution features are explicitly labeled.

| Tier | Min feature | Contrast | Noise | Purpose |
|---|---:|---:|---:|---|
| `easy` | `40-80 um` | `2.0-3.0 C` | `0.0724 C` | sanity and method debugging |
| `medium` | `20-40 um` | `1.0-2.0 C` | `0.0724 C` | near project POC target |
| `hard` | `15-25 um` | `0.8-1.5 C` | `0.0724 C` | near/below optical limit stress |
| `stress` | `10-20 um` | `0.5-1.0 C` | `0.0724-0.15 C` | hallucination and uncertainty tests |

Do not present `stress` recovery as a real-data promise. It is deliberately difficult and may be physically unsupported.

---

## 8. Drift and Robustness

Decision: include drift, but separate it from the primary clean benchmark.

### 8.1 Dataset Tracks

| Track | Drift | Role |
|---|---|---|
| `clean` | off | core method ranking under matched forward model |
| `drift_scalar` | global per-frame offset | offset correction and highpass robustness |
| `drift_lowfreq` | spatial low-frequency frame drift | highpass / background removal stress test |
| `drift_gain_offset` | gain plus offset | raw-control robustness |

### 8.2 Drift Models

Apply drift after the forward model and before highpass:

```text
lr_raw_k = A_k(x_hr) + noise_k + drift_k
lr_highpass_k = highpass(lr_raw_k)
```

P0 drift models:

1. `scalar_offset_random_walk`
   - one scalar offset per frame;
   - random walk or low-pass filtered Gaussian sequence over acquisition order;
   - default amplitude `0.02-0.15 C`.

2. `spatial_lowfreq_gaussian`
   - one low-frequency field per frame;
   - generate Gaussian noise then blur with `sigma=40-120 LR px`;
   - default amplitude `0.02-0.20 C`.

3. `gain_plus_offset`
   - `y_k = gain_k * y_k + offset_k`;
   - default gain range `1 +/- 0.002` to `1 +/- 0.01`.

This directly tests whether our highpass and offset correction behave correctly. It also improves the project narrative because we can show the clean result and then the robustness result.

---

## 9. Shift Profiles

P0 shift sources:

| Profile | Source | Use |
|---|---|---|
| `real_default_contour_refined` | `output/ep05_contour_alignment/contour_alignment_results.csv` refined columns | primary benchmark |
| `real_ncc_init` | same CSV init columns | phase-prior control |
| `real_tuned_contour_refined` | `output/ep05_alignment_tuning/full_r360_e93_rad100_s0125/contour_alignment_results.csv` | sensitivity |

P1 shift profiles:

| Profile | Use |
|---|---|
| `stage_prior_affine` | prior/control, not truth |
| `jittered_real` | robustness to alignment perturbation |
| `ideal_phase_grid` | theoretical upper bound for phase coverage |

Every generated scene must save `shifts.npy` and record:

- profile name;
- source path;
- source file hash;
- columns used;
- shift units;
- shift convention.

---

## 10. Metadata Schema

Each scene should have:

```json
{
  "schema_version": "0.1",
  "dataset": "ThermalChipPhantom",
  "engine": "TCForge",
  "scene_id": "tcp_medium_0001",
  "seed": 1001,
  "split": "test",
  "difficulty": "medium",
  "scale": 2,
  "lr_shape": [480, 640],
  "hr_shape": [960, 1280],
  "pixel_size_um": 10.0,
  "spatial_resolution_um": 20.0,
  "geometry": {
    "units": "um",
    "rotation_deg": 47.3,
    "min_feature_um": 20.0,
    "primitives": []
  },
  "physics": {
    "T_bg_c": 21.0,
    "delta_T_c": 1.5,
    "low_freq_background_c": 0.2,
    "psf_sigma_lr_px": 0.5,
    "noise_sigma_c": 0.0724,
    "forward_mode": "exact_ep06_point",
    "highpass_sigma_lr_px": 5.0,
    "drift_model": "none"
  },
  "shifts": {
    "profile": "real_default_contour_refined",
    "source_path": "output/ep05_contour_alignment/contour_alignment_results.csv",
    "source_sha256": "...",
    "columns": ["refined_align_dx_px", "refined_align_dy_px"],
    "units": "LR pixels",
    "convention": "LR-to-reference alignment shift"
  },
  "provenance": {
    "generator_git_sha": "...",
    "created_at_utc": "...",
    "config_path": "configs/synthetic/phantom_benchmark.json"
  }
}
```

The dataset-level `manifest.csv` should include at least:

```text
scene_id, split, difficulty, scale, seed, forward_mode, drift_model,
min_feature_um, delta_T_c, psf_sigma_lr_px, noise_sigma_c,
shift_profile, scene_dir, metadata_sha256
```

---

## 11. Evaluation Metrics

Synthetic-only metrics:

| Metric | Domain | Direction | Notes |
|---|---|---|---|
| PSNR | raw or highpass | higher better | report data range explicitly |
| SSIM | raw or highpass | higher better | optional if dependency available; otherwise defer |
| NRMSE | raw or highpass | lower better | normalize by GT std for highpass |
| Boundary F1 | edge GT | higher better | threshold in HR px and um |
| Chamfer distance | edge GT | lower better | primary contour localization metric |
| Hausdorff distance | edge GT | lower better | sensitive worst-case metric |
| ESF width | raw GT boundary | lower better if no artifacts | must be paired with stability |
| Pin gap accuracy | geometry GT | lower error better | for pin arrays only |
| Topology F1 | connected components | higher better | catches merged/split structures |

Robustness metrics:

- clean vs drift metric degradation;
- frame subset stability for 32/64/128/255 frames;
- noise seed stability;
- PSF sensitivity for `sigma=0.3/0.5/0.7/1.0`;
- alignment perturbation sensitivity;
- hold-out reprojection residual for methods that optimize against synthetic bursts.

Do not rank a method on gradient/Tenengrad alone. Synthetic GT enables better contour metrics; use them.

---

## 12. Baseline Methods to Run

P0 baselines:

1. LR single-frame reference;
2. bicubic display interpolation;
3. SAA uniform;
4. SAA quality-weighted;
5. IBP;
6. MAP-TV.

P1 / EP07 methods:

1. WIRE-style INR;
2. DIP;
3. Deep Decoder.

P2 / EP08 methods:

1. small one-channel denoiser;
2. DiffPIR / DAPS with TCForge prior;
3. Flower / PnP-Flow only after a usable domain prior exists.

The initial smoke test should run only P0 baselines on 5 scenes.

---

## 13. Implementation Phases

### S0: Lock Specification

- Create this plan.
- Confirm TCForge name and source/artifact split.
- Convert open decisions into JSON config defaults.

### S1: Geometry and Metadata

- Implement rectangles, cutouts, pin arrays, L/T/cross, parallel grooves.
- Store geometry in um.
- Convert to HR grid deterministically.
- Save `metadata.json` and `manifest.csv`.

### S2: Physics and Tracks

- Build raw HR temperature field.
- Add low-frequency background.
- Generate edge map from mask.
- Implement highpass helper matching existing `highpass_preprocess` default semantics.

### S3: Forward Model

- Implement `exact_ep06_point`.
- Add tests against current EP06 `forward()` for several random scenes and shifts.
- Implement `physical_block_average` as a labeled secondary mode.
- Add shift sign tests with a point source.

### S4: Noise, Drift, and Bad Pixels

- Add Gaussian noise with fixed seed.
- Add scalar drift and low-frequency spatial drift.
- Add optional bad-pixel mask only after clean/drift tracks are stable.

### S5: Generator CLI

Example command:

```bash
uv run python scripts/generate_thermal_chip_phantom.py \
  --config configs/synthetic/phantom_smoke.json \
  --output-root data/synthetic/thermal_chip_phantom_smoke
```

The CLI should support:

- `--num-scenes`;
- `--difficulty`;
- `--scale`;
- `--forward-mode`;
- `--shift-profile`;
- `--drift-model`;
- `--seed`;
- `--overwrite` guarded explicitly.

### S6: Evaluation CLI

Example command:

```bash
uv run python scripts/evaluate_thermal_chip_phantom.py \
  --dataset-root data/synthetic/thermal_chip_phantom_smoke \
  --result-root output/thermal_chip_phantom/smoke
```

The evaluator should accept method outputs as `.npy` and write:

- `baseline_metrics.csv`;
- `boundary_metrics.csv`;
- `robustness_metrics.csv`;
- compact sanity-check figures.

### S7: Smoke Test

Generate 5 scenes:

- 2 easy;
- 2 medium;
- 1 hard;
- scale 2;
- `exact_ep06_point`;
- no drift first, then low-frequency drift copy.

Run SAA/IBP/MAP-TV and verify:

- arrays are finite;
- shapes are correct;
- shift convention is not inverted;
- easy scenes produce sensible metric ordering;
- drift track degrades raw more than highpass/offset-corrected track.

### S8: Full Benchmark

Recommended first full benchmark:

| Split | Scenes | Notes |
|---|---:|---|
| train | 30 | for denoiser/prior development only |
| val | 10 | hyperparameter tuning |
| test | 20 | final reporting |

For classical and test-time methods, train split is not required. For denoiser/diffusion work, patch generation should sample from train scenes only.

---

## 14. Tests Required

Minimum unit tests:

1. deterministic generation: same seed and config produce same hashes;
2. geometry bounds: masks are finite, binary, and within expected area range;
3. feature scale: generated min feature in pixels matches requested um within tolerance;
4. shift sign: point-source test matches EP06 convention;
5. forward equivalence: `exact_ep06_point` matches EP06 NumPy forward for scale 2;
6. PSF units: `psf_sigma_lr_px=0.5` becomes `1.0 HR px` at 2x;
7. highpass: constant image highpass is approximately zero;
8. drift: drift is applied before highpass and affects raw more than highpass;
9. manifest completeness: every scene listed exists and hashes match.

---

## 15. Reporting Rules

When reporting ThermalChipPhantom results:

- Always state `forward_mode`.
- Always state whether the track is raw or highpass.
- Always state whether drift is enabled.
- Always state min feature size in um and relation to 20 um spatial resolution.
- Do not describe 4x output as verified 5 um resolution.
- Keep `stress` cases explicitly labeled as stress/hallucination tests.
- Distinguish synthetic GT metrics from real-data forward consistency.

Recommended phrasing:

> TCForge generates ThermalChipPhantom, a procedural LWIR chip phantom benchmark that matches the project microscan geometry and degradation chain. It is used to rank algorithms under controlled HR ground truth, while real-data claims remain gated by hold-out reprojection, split-half stability, and contour consistency.

---

## 16. Acceptance Criteria

TCForge v0.1 is accepted when:

1. source code lives outside ignored data directories;
2. a smoke dataset can be regenerated from config with deterministic hashes;
3. `exact_ep06_point` passes equivalence tests against EP06 forward model;
4. generated scenes include raw HR, mask GT, edge GT, LR raw burst, LR highpass burst, shifts, metadata, and manifest;
5. drift can be toggled independently and is documented in metadata;
6. SAA/IBP/MAP-TV smoke baseline runs end to end on 5 scenes;
7. at least Boundary F1, Chamfer, NRMSE, and PSNR are written to CSV;
8. no generated `.npy`, `.csv`, or figure artifacts are staged for Git by default.

---

## 17. Default Initial Config

Use these defaults unless a later experiment changes them deliberately:

```json
{
  "dataset": "ThermalChipPhantom",
  "engine": "TCForge",
  "scale": 2,
  "lr_shape": [480, 640],
  "forward_mode": "exact_ep06_point",
  "psf_sigma_lr_px": 0.5,
  "noise_sigma_c": 0.0724,
  "highpass_sigma_lr_px": 5.0,
  "shift_profile": "real_default_contour_refined",
  "rotation_deg_center": 47.6,
  "rotation_jitter_deg": 1.0,
  "drift_tracks": ["clean", "drift_scalar", "drift_lowfreq"],
  "difficulties": ["easy", "medium", "hard", "stress"]
}
```

---

## 18. Open Decisions

Current recommended defaults are listed here; change only with explicit reason.

| Question | Default |
|---|---|
| Engine name | TCForge |
| Benchmark name | ThermalChipPhantom |
| Code location | `core/src/thermal_core/synthetic/` |
| Generated data location | `data/synthetic/` |
| Config format | JSON |
| Primary forward mode | `exact_ep06_point` |
| Secondary forward mode | `physical_block_average` |
| Primary scale | 2x |
| 4x status | P1 research / stress only |
| Drift | included as robustness track, not clean primary |
| Below-20 um features | stress cases only |
