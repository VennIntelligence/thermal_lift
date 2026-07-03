# Stage 1a Inbox Manifest

This folder is the corrected audit package for the Stage 1a remote run.

## Files

- `REPORT.md`: corrected human-readable report.
- `artifacts/task_a/stage0a_summary.json`: Stage 0a full-grid summary.
- `artifacts/task_a/stage0a_model_sweep.csv`: Stage 0a candidate sweep table.
- `artifacts/task_a/stage0a_best_shift_refinements.csv`: Stage 0a best-candidate shift refinements.
- `artifacts/task_b/goodcase_v11_40k.json`: V11 regression suite output.
- `artifacts/task_b/goodcase_promptA_5k.json`: PromptA regression suite output.
- `artifacts/task_c_d/final_metrics.csv`: corrected C/D final scalar metrics from TensorBoard.
- `artifacts/task_c_d/checkpoint_audit.json`: C/D checkpoint/config audit summary.
- `artifacts/task_e/method_summary.csv`: Stage 0c FRC leaderboard summary.
- `artifacts/task_e/run_manifest.json`: Stage 0c FRC run manifest.
- `artifacts/task_e/split_balance.csv`: Stage 0c seed-42 phase split balance.

Large `.npy` reconstructions and model checkpoints are intentionally not copied into this inbox.
