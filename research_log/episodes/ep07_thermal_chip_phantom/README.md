# EP07 — ThermalChipPhantom / TCForge

## Goal

Build the report and tracking skeleton for TCForge, the synthetic ThermalChipPhantom generator used to validate contour-level SR algorithms under controlled HR ground truth, forward convention, highpass, manifest and smoke-test contracts.

EP07 is not a new real-data SR claim. It is an engineering episode for a reproducible synthetic data engine and its small demo notebook.

## Scope

- Notebook source fragments: `notebooks/ep07_thermal_chip_phantom/fragments/`
- Demo output target: `output/ep07_thermal_chip_phantom/demo_dataset/`
- Report skeleton: `reports/ep07_thermal_chip_phantom/`
- TCForge runtime expectation: independent UV project under `tcforge/`

## Environment

TCForge is expected to be an isolated UV package:

```bash
cd tcforge
uv sync --extra dev
```

The EP07 notebook is built from the project root:

```bash
uv run python scripts/build_notebook.py notebooks/ep07_thermal_chip_phantom --execute
```

Because notebook execution starts at the project root, `01_setup.py` temporarily adds `tcforge/src` to `sys.path`.

## Notebook Plan

| Fragment | Purpose |
|---|---|
| `01_setup.py` | Environment declaration, paths, TCForge import probe, small demo config |
| `02_architecture_principles.py` | TCForge module contracts as a Markdown table |
| `03_synthetic_scene.py` | Small HR scene demo and geometry/temperature visualization |
| `04_forward_highpass.py` | LR forward burst, EP06-like highpass and interpretation |
| `05_manifest_smoke.py` | metadata, manifest and smoke acceptance table |
| `06_demo_visualization.py` | Disk-based demo dataset overview |
| `07_conclusions.py` | Integration handoff and limitations |

## Current Status

- Implemented `tcforge/` as an independent UV project with core modules for geometry, physics, shifts, forward models, highpass, manifest, evaluation and visualization.
- Vendored the EP06 forward reference under `tcforge/src/tcforge/_ep06_reference/` and locked its sign convention with tests.
- Added root-level generator, smoke-test and evaluation CLIs.
- Added TCForge unit tests covering geometry, physics, highpass parity, forward conventions, shifts, drift and manifest helpers.
- Applied independent review follow-up: scene-level evaluation API, independent highpass verification in smoke/evaluate, portable real-shift CSV lookup, `_edge_map()` dead-code cleanup, shared worker utility and visualization tests.
- EP07 notebook uses `lr_shape=(64, 96)` and `n_frames=16` to demonstrate the architecture without generating full-frame GiB-scale output.
- Executed the full P0 smoke path locally on 2026-05-19: 5 full-frame scenes, 255 frames each, output under ignored `data/synthetic/thermal_chip_phantom_p0_smoke_tmp/`; smoke and evaluate CLIs passed.
- Benchmark/P1 config keys that are not yet materialized by the CLI (`drift_tracks`, split objects, crop ROI) fail fast instead of silently generating partial data.

## Independent Review Fixes

| Gap | Resolution |
|---|---|
| `evaluate.py` lacked a scene-level entry point | Added `evaluate_scene()`, `summarize_scene()`, `evaluate_dataset()` and `aggregate_scene_metrics()` with tests. |
| `shifts.py` assumed repo-local EP05 output | Added env/current-working-directory aware default lookup and explicit-path tests. |
| Smoke only checked generator highpass against itself | Smoke now uses independent scipy reference highpass and samples evenly across the burst, with `--highpass-check-frames 0` for all frames. |
| `visualization.py` had no tests | Added optional matplotlib save test plus validation-before-import test. |
| `_edge_map()` had unreachable fallback code | Split fallback into `_fallback_edge_map()` and made fallback explicit demo-only behavior. |
| Evaluate CLI did not recompute highpass | Evaluation summary now records `highpass_reference_check_frames`, `highpass_reference_max_abs_diff_c` and `highpass_reference_allclose`. |
| Worker resolver duplicated | Added shared `tcforge._utils.resolve_workers()`. |

## Required Follow-up

- Re-run full P0 smoke after any generator, forward, highpass or manifest contract change.
- Implement P1 benchmark materialization before running `phantom_benchmark.json`: drift tracks, split assignment, crop ROI storage and multi-forward-mode reporting.
- Keep `exact_ep06_point` and `physical_block_average` metrics separated in every report.
- Keep notebook fallback paths limited to local report editing; formal validation must use TCForge tests and CLI smoke.

## Guardrails

- Do not treat stage command or synthetic shifts as real-data alignment ground truth.
- Do not mix `exact_ep06_point` and `physical_block_average` forward modes in one metric table.
- Do not use highpass images as absolute temperature evidence.
- Do not generate full-frame multi-scene smoke data from the notebook.
- Do not commit `output/` demo artifacts.
