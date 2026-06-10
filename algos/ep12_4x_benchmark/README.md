# EP12 4x — UNet@2000 vs Bare Drizzle Benchmark

Quick real-data visual benchmark on the 248 clean main-session frames. Compares the
EP12 drizzle-informed 4x UNet checkpoint against the bare tcforge scatter-add
drizzle mean on the same highpass domain, center ROI, and colormap range.

## Scope

- Input: EP06 clean main session, 248 raw temperature frames.
- Alignment: EP05 `contour_refined` shifts.
- UNet checkpoint: `../ep12_4x_sr/outputs/ep12_large_bucketv2/checkpoint_step_002000.pt`.
- Baseline: bare drizzle mean (`tcforge.classical_sr.drizzle_features` channel 0).
- Highpass sigma: `5.0`.
- Default device: `cuda:1`. `cuda:0` is protected unless `--allow-cuda0` is passed.

## Run

```bash
cd algos/ep12_4x_benchmark
uv sync
uv run python scripts/run_ep12_vs_drizzle_4x.py \
  --checkpoint ../ep12_4x_sr/outputs/ep12_large_bucketv2/checkpoint_step_002000.pt \
  --output-dir ../../output/ep12_4x_benchmark \
  --device cuda:1
```

Smoke test:

```bash
uv run python scripts/run_ep12_vs_drizzle_4x.py \
  --checkpoint ../ep12_4x_sr/outputs/ep12_large_bucketv2/checkpoint_step_002000.pt \
  --output-dir /tmp/ep12_4x_smoke \
  --limit 16 \
  --device cuda:1
```
