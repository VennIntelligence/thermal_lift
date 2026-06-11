# EP16 Budget Robustness

CPU-only classical arms for paper Section 6.4-6.5:

- E1 frame budget: `N={31,62,124,248}` phase-stratified subsets.
- E2 shift robustness: Gaussian noise on contour-refined shifts.
- E3 alignment source ablation: command prior vs contour-refined shifts.

The runner reconstructs all drizzle runs directly in this UV project. TGV runs
are launched as CPU-only child processes through the existing EP10 conda
environment:

```bash
cd algos/ep16_budget_robustness
CUDA_VISIBLE_DEVICES="" uv sync
CUDA_VISIBLE_DEVICES="" uv run python scripts/run_ep16_classical.py \
  --arms both \
  --run-tgv \
  --tgv-parallel 2 \
  --tgv-workers 6
```

Useful shorter commands:

```bash
# Drizzle only, useful before the overnight TGV queue.
CUDA_VISIBLE_DEVICES="" uv run python scripts/run_ep16_classical.py --arms drizzle --skip-tgv

# Rebuild figures and CSVs from completed run JSON files.
CUDA_VISIBLE_DEVICES="" uv run python scripts/run_ep16_classical.py --summarize-only
```

Outputs are written under `../../output/ep16_budget_robustness/` and are ignored
by Git. The script writes `run_manifest.json` after each run, so failed TGV
children remain visible and later invocations can resume successful runs.

Resource contract:

- `CUDA_VISIBLE_DEVICES` is forced to an empty string in parent and child.
- TGV parent parallelism defaults to 2.
- Each TGV child passes `workers<=6` to `reconstruct_map_tgv`.
- BLAS/OpenMP thread environment variables default to 1 unless already set.

