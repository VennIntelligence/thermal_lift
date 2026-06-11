# EP16 Budget Robustness

## Scope

EP16 fills the classical CPU portion of paper Section 6.4-6.5:

- E1 frame budget: `N={31,62,124,248}` phase-stratified subsets.
- E2 shift robustness: `sigma={0,0.05,0.1,0.2}` LR-pixel noise on contour-refined shifts.
- E3 alignment source: `command_prior` vs `contour_refined` at full 248 frames.

Only drizzle and TGV are in scope. UNet and GPU MAP-TV are intentionally left
for a separate GPU-available task.

## Commands

```bash
cd algos/ep16_budget_robustness
CUDA_VISIBLE_DEVICES="" uv sync
CUDA_VISIBLE_DEVICES="" uv run python scripts/run_ep16_classical.py --arms drizzle --skip-tgv
CUDA_VISIBLE_DEVICES="" uv run python scripts/run_ep16_classical.py --arms both --run-tgv --tgv-parallel 2 --tgv-workers 6
```

Notebook build:

```bash
uv run python scripts/build_notebook.py notebooks/ep16_budget_robustness --execute
```

## Metric Policy

All reconstructions use highpass frames with `sigma_bg=5.0` in the 2x HR
domain. Raw-control correlation is Pearson correlation against
`highpass(bicubic(nanmean(raw subset)), sigma_bg=5.0)`.

Drizzle split-half and FRC are exact STScI-drizzle phase-stratified proxies with
five split-half repeats for NRMSE and three phase-stratified FRC seeds.

TGV full HR images are reconstructed by the EP10 CPU TGV implementation through
`conda run -p ../ep10_tgv_sr/.venv python scripts/run_tgv_child.py`.
To preserve the Task C overnight budget, TGV split-half/FRC columns use the same
drizzle information-budget proxy on the identical subset and shifts; TGV-specific
raw-control, artifact, and zigzag metrics are computed from the full TGV HR image.
The manifest records this policy in `metric_definitions`.

## Outputs

- `output/ep16_budget_robustness/frame_budget.csv`
- `output/ep16_budget_robustness/shift_robustness.csv`
- `output/ep16_budget_robustness/alignment_source.csv`
- `output/ep16_budget_robustness/run_manifest.json`
- `output/ep16_budget_robustness/hr/*_hr.npy`
- `output/ep16_budget_robustness/fig_frame_budget.png`
- `output/ep16_budget_robustness/fig_shift_robustness.png`
- `output/ep16_budget_robustness/fig_alignment_source.png`
- `output/paper_figures/fig07_budget_robustness.png`
- `output/paper_figures/fig07_budget_robustness.pdf`

## Current Status

Completed the classical CPU run on 2026-06-11.

Result matrix:

| CSV | Rows | Status |
|---|---:|---|
| `frame_budget.csv` | 17 | all success |
| `shift_robustness.csv` | 20 | all success |
| `alignment_source.csv` | 4 | all success |
| `run_manifest.json` | 37 unique runs | all success |

Run inventory:

- Drizzle: 20 unique HR runs, all success.
- TGV: 17 unique HR runs, all success.
- TGV summed child runtime: about 353.7 minutes; wall time was about 3.2 hours with two-way parallelism.
- HR outputs: 37 `.npy` files under `output/ep16_budget_robustness/hr/`.

Generated figure exports:

- `output/ep16_budget_robustness/fig_frame_budget.png`
- `output/ep16_budget_robustness/fig_shift_robustness.png`
- `output/ep16_budget_robustness/fig_alignment_source.png`
- `output/paper_figures/fig07_budget_robustness.png`
- `output/paper_figures/fig07_budget_robustness.pdf`

No failed runs were recorded. Run-level status remains authoritative in
`output/ep16_budget_robustness/run_manifest.json`.
