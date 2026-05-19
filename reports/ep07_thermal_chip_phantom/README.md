# EP07 — ThermalChipPhantom / TCForge Report

## Summary

EP07 documents the implemented P0 architecture and smoke-demo workflow for TCForge, an isolated synthetic data generator for ThermalChipPhantom experiments. The notebook is report-oriented and uses a small demo dataset to validate the data contract before any full-scale benchmark is generated.

The independent review follow-up has been applied: TCForge now has a scene-level evaluation API, smoke/evaluate highpass validation uses an independent scipy reference instead of reusing the generator helper, and the `_edge_map()` fallback path no longer contains unreachable code.

## Artifacts

- Notebook fragments: `notebooks/ep07_thermal_chip_phantom/fragments/`
- Demo output path: `output/ep07_thermal_chip_phantom/demo_dataset/`
- Research log: `research_log/episodes/ep07_thermal_chip_phantom/README.md`

## Evidence Included After Execution

- TCForge architecture contract table.
- HR mask / HR temperature / HR edge proxy visualization.
- LR raw and highpass frame comparison.
- Manifest and smoke acceptance table.
- Disk-based demo overview with phase coverage.
- Integration handoff table listing remaining P1 benchmark materialization work.
- Full P0 smoke was executed locally on 2026-05-19 under ignored `data/synthetic/thermal_chip_phantom_p0_smoke_tmp/`: 5 full-frame scenes, 255 frames each; smoke and evaluate CLIs passed.

## Independent Review Follow-up

| Item | Status | Resolution |
|---|---:|---|
| Scene-level evaluate API | Fixed | Added `tcforge.evaluate.evaluate_scene()`, `summarize_scene()`, `evaluate_dataset()` and aggregate helpers. |
| Shift CSV portability | Improved | Default real-shift lookup now honors `TCFORGE_REAL_SHIFT_CSV` and current working directory before falling back to repo-local paths. |
| Highpass smoke self-comparison | Fixed | Smoke recomputes sampled or all frames with an independent scipy reference; default samples 16 evenly spaced frames. |
| Visualization test gap | Improved | Added lazy-import/optional-matplotlib tests. |
| Generator/smoke coupling | Reduced | Smoke no longer imports generator `_highpass()` for validation. |
| `_edge_map()` dead code | Fixed | Added explicit fallback helper gated by `allow_fallback_demo`; formal runs still fail fast. |
| Evaluate CLI highpass check | Fixed | Evaluation summary records independent highpass check frame count, max diff and allclose status. |
| Duplicate worker resolver | Fixed | Shared `tcforge._utils.resolve_workers()` is used by forward and highpass modules. |

## Current Limitations

The P0 package, CLIs and tests are present, and the full 5-scene/full-frame P0 smoke run has passed locally. P1 benchmark features in `phantom_benchmark.json` intentionally fail fast until drift tracks, split assignment and crop ROI storage are fully materialized.
