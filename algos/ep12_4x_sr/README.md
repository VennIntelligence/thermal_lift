# EP12 4x SR

Drizzle-informed 4x thermal restoration model. The model consumes sparse 4x
drizzle features plus 1x context upsampled to the same grid, then predicts a
4x temperature field and an optional log-variance confidence map.

Smoke command:

```bash
cd algos/ep12_4x_sr
uv run pytest
```

Training expects scene directories with:

- `obs_features_4x.npz`: 3 channels `(drizzle_mean, coverage, variance)`
- `obs_features_1x.npz`: 5 channels at LR
- `hr_mask_4x.png`
- `hr_edge_4x.png`
- `metadata.json`
