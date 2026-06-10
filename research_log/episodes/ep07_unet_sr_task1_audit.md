# EP07v2 Task 1 Audit

Date: 2026-06-07

Scope: environment preparation, TCForge code audit, baseline test, and Task 2 readiness check for EP07v2. This task made no functional code changes.

## 1. Git Worktree Summary

Command used:

```bash
git status --ignored --short
```

The worktree was already dirty before this audit record was created. Notable pre-existing changes:

- `AGENTS.md` is modified, even though EP07v2 says not to modify it.
- `core/src/thermal_core/*`, many notebook fragments, reports, and scripts are modified or untracked.
- `tcforge/src/tcforge/evaluate.py` and `tcforge/src/tcforge/shifts.py` are modified.
- `tmp/` is untracked and contains the EP07v2 staged prompts.
- Ignored generated/runtime paths include `.venv/`, `.pytest_cache/`, `data/`, `output/`, built `.ipynb` notebooks, and Python caches.

This Task 1 run did not revert or edit those existing changes.

## 2. Baseline Test Result

Commands used:

```bash
cd tcforge
uv sync --extra dev
uv run pytest -v
```

Result:

- `uv sync --extra dev`: completed successfully.
- `pytest`: `29 passed, 1 skipped in 0.16s`.
- The skipped test was `tests/test_visualization.py::test_plot_and_save_figure_when_matplotlib_is_available`.

Baseline is healthy enough for Task 2, but Task 2 should re-run the same test suite after changes.

## 3. TCForge Structure

Current TCForge source lives under `tcforge/src/tcforge/`:

- `geometry.py`: binary HR chip mask generation. Defaults remain 2x: `DEFAULT_CANVAS_SHAPE=(960, 1280)`, `DEFAULT_SCALE=2`.
- `forward.py`: public LR burst forward modes: `exact_ep06_point` and `physical_block_average`.
- `_ep06_reference/forward.py`: EP06 matrix-free reference forward/adjoint. It currently rejects non-2x scale.
- `physics.py`: temperature rendering, edge map, Gaussian noise, and drift models.
- `manifest.py`: `SceneManifest`, JSON/CSV helpers, and required-file validation.
- `evaluate.py`: current evaluation path is tied to old full `.npy` 2x scenes.
- `shifts.py`: real and ideal shift profile loading/generation.
- `highpass.py`: highpass preprocessing.
- `visualization.py`: simple plotting helpers.

Tests exist for geometry, forward, physics, drift, highpass, shifts, manifest, evaluate, CLI config, and visualization. There are no tests yet for `fusion.py`, `storage.py`, or `reconstruct.py`.

## 4. Functions Already Supporting `scale`

These functions already accept a `scale` parameter or are scale-aware:

- `tcforge/src/tcforge/geometry.py:29` `_hr_pitch_um`
- `tcforge/src/tcforge/geometry.py:39` `_length_px`
- `tcforge/src/tcforge/geometry.py:46` `_coord_px`
- `tcforge/src/tcforge/geometry.py:66` `make_rectangle`
- `tcforge/src/tcforge/geometry.py:89` `make_frame`
- `tcforge/src/tcforge/geometry.py:127` `make_pin_array`
- `tcforge/src/tcforge/geometry.py:180` `make_cross`
- `tcforge/src/tcforge/geometry.py:215` `make_trenches`
- `tcforge/src/tcforge/geometry.py:270` `make_l_shape`
- `tcforge/src/tcforge/geometry.py:350` `build_scene_mask`
- `tcforge/src/tcforge/geometry.py:374` `build_scene_mask_with_metadata`
- `tcforge/src/tcforge/forward.py:40` `physical_block_average_forward`
- `tcforge/src/tcforge/forward.py:73` `generate_lr_burst`
- `tcforge/src/tcforge/shifts.py:64` `ideal_phase_grid`
- `tcforge/src/tcforge/shifts.py:139` `load_shift_profile`
- `tcforge/src/tcforge/manifest.py:12` `SceneManifest`
- `tcforge/src/tcforge/evaluate.py:281` `_infer_scale`

Important caveat: `generate_lr_burst(..., forward_mode="exact_ep06_point", scale=4)` currently fails because it delegates to `_ep06_reference/forward.py`, whose validator rejects `scale != 2`.

## 5. Hardcoded 2x or Old 2x File Format

Known locations:

- `tcforge/src/tcforge/geometry.py:10`: `DEFAULT_CANVAS_SHAPE = (960, 1280)`.
- `tcforge/src/tcforge/geometry.py:12`: `DEFAULT_SCALE = 2`.
- `tcforge/src/tcforge/_ep06_reference/forward.py:1`: docstring says locked 2x model.
- `tcforge/src/tcforge/_ep06_reference/forward.py:18`: `_validate_scale()` rejects non-2 scale.
- `tcforge/src/tcforge/_ep06_reference/forward.py:31`: docstring says `2x2 block averaging`.
- `tcforge/src/tcforge/manifest.py:21`: default `scale=2`.
- `tcforge/src/tcforge/manifest.py:23`: default `hr_shape=(960, 1280)`.
- `tcforge/src/tcforge/evaluate.py:14`: `REQUIRED_SCENE_FILES` requires `hr_temperature_2x.npy`, `hr_mask_2x.npy`, `hr_edge_map_2x.npy`, `lr_burst_raw.npy`, `lr_burst_highpass.npy`, `shifts.npy`, and `metadata.json`.
- `tcforge/src/tcforge/evaluate.py:23`: reconstruction names are still 2x-oriented.
- `tcforge/src/tcforge/evaluate.py:103`: `summarize_scene()` loads fixed old 2x `.npy` files.
- `tcforge/src/tcforge/evaluate.py:146`: `evaluate_scene()` is documented for optional 2x SR outputs.
- `tcforge/src/tcforge/evaluate.py:164`: `evaluate_scene()` loads fixed old 2x `.npy` files.
- `tcforge/src/tcforge/evaluate.py:215`: `evaluate_dataset()` defaults to `sr_temperature_2x.npy`.
- `scripts/generate_thermal_chip_phantom.py:33`: old required scene files are full 2x `.npy` artifacts.
- `scripts/generate_thermal_chip_phantom.py:442`: `_save_scene()` writes HR temperature and full LR bursts, which conflicts with EP07v2 compact storage requirements.
- `scripts/smoke_test_thermal_chip_phantom.py:94`: smoke checks expect the old full 2x scene files.

Per EP07v2 constraints, do not edit `scripts/generate_thermal_chip_phantom.py` or `scripts/smoke_test_thermal_chip_phantom.py` in Task 2.

## 6. Existing Reusable APIs

Likely reusable in Task 2:

- `tcforge.forward.physical_block_average_forward`: already scale-generic by loop over `range(scale)`.
- `tcforge.forward.generate_lr_burst`: usable once `_ep06_reference/forward.py` no longer rejects `scale=4`.
- `tcforge.geometry.build_scene_mask_with_metadata`: already accepts `canvas_shape` and `scale`.
- `tcforge.physics.render_temperature_field`: use directly for HR reconstruction parity.
- `tcforge.physics.edge_map`: use for `hr_edge_4x.png`.
- `tcforge.physics.add_noise` and `tcforge.physics.apply_drift`: reuse for generation.
- `tcforge.highpass.highpass_preprocess`: reuse inside burst fusion instead of rewriting highpass.
- `tcforge.shifts.load_shift_profile`: returns `(shifts, metadata)`, not just shifts.
- `tcforge.manifest.write_json`, `read_json`, `write_manifest_csv`, and `validate_file_list`: reusable for compact scene metadata and manifest support.
- `tcforge.evaluate` metric helpers (`mae`, `rmse`, `nrmse`, `psnr`, `binary_iou`, `boundary_f1`) are not inherently 2x-specific.

No implementation was found for `fuse_burst_to_features`, `_shift_and_accumulate`, `save_scene_compact`, `load_scene_compact`, `reconstruct_hr_temperature`, `obs_features_1x.npz` storage, `configs/synthetic/training_pool_4x.json`, or `scripts/generate_training_pool.py`.

## 7. Task 2 API Notes

Task 2 should account for these current API differences:

- `load_shift_profile()` returns `(shifts, metadata)`. The staged prompt pseudocode sometimes treats it as if it returns only shifts.
- `render_temperature_field()` uses parameter names `t_bg_c` and `delta_t_c`, while EP07v2 metadata names use `T_bg_c` and `delta_T_c`.
- `apply_drift()` accepts canonical models: `none`, `scalar_offset`, `lowfreq`, `gain_offset`.
- `edge_map()` returns `uint8`; old generator casts to `float32`, but compact PNG storage should keep binary `uint8`.
- Existing `evaluate.py` can summarize full old 2x scenes only. Compact 4x scenes need a new detection/loading path, while preserving old 2x support.
- `SceneManifest` has `files` and `extra` escape hatches but no explicit compact fields yet.
- `__init__.py` must export new Task 2 APIs after `fusion.py`, `storage.py`, and `reconstruct.py` are added.

## 8. Prompt vs Current Code Mismatches

Observed mismatches between EP07v2 and current code:

- EP07v2 says TCForge should support 4x, but the `exact_ep06_point` path still rejects non-2x scale.
- EP07v2 requires compact storage without HR temperature or full LR burst on disk; current generator/evaluator/smoke paths are old full `.npy` storage.
- EP07v2 says clean SR frames are 248; existing smoke config still uses `n_frames_per_scene=255`.
- EP07v2 mentions `psf_sigma_lr_px=0.226` from `configs/psf_calibration.json`, while existing smoke/benchmark configs still use `0.5`.
- The full prompt says `geometry.py` should add 4x constants but keep defaults unchanged; Task 2 should not change `DEFAULT_SCALE` or `DEFAULT_CANVAS_SHAPE`.
- The staged Task 1 output path is `research_log/episodes/ep07_unet_sr_task1_audit.md`; there is no existing EP07 UNet episode directory yet.

## 9. Task 2 Readiness

Task 2 can start. No blocking baseline failures were found.

Recommended Task 2 checklist:

1. Add 4x constants to `geometry.py` without changing defaults.
2. Relax `_ep06_reference/forward.py::_validate_scale()` to any positive integer and add 4x forward/adjoint tests.
3. Add `fusion.py` with `fuse_burst_to_features()` and `_shift_and_accumulate()`.
4. Add `storage.py` compact save/load using PNG mask/edge, compressed float16 `obs_features_1x.npz`, `shifts.npy`, and `metadata.json`.
5. Add `reconstruct.py` by directly reusing `render_temperature_field()` to guarantee parity.
6. Extend `evaluate.py` to recognize compact 4x scenes while preserving old 2x full-scene support.
7. Update `tcforge/src/tcforge/__init__.py` exports.
8. Add focused tests for scale=4 forward, fusion feature shape/coverage, compact roundtrip, reconstruction parity, and compact evaluation recognition.
9. Re-run `cd tcforge && uv run pytest -v`.
