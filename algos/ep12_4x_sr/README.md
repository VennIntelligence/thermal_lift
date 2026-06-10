# EP12 4x SR

Hybrid drizzle-informed 4x thermal restoration model.

Current route:

```text
248 LR frames + shifts
  -> 2x drizzle features (computed by Dataset from lr_burst.npy)
  -> concat with 1x fused features upsampled to 2x
  -> UNet on 2x grid
  -> PixelShuffle 2x
  -> 4x temperature field
```

Training expects v8 AA compact scene directories from `scripts/generate_training_pool.py`:

- `hr_mask_4x.png`: soft coverage mask, loaded as `[0, 1]`
- `hr_edge_4x.png`
- `obs_features_1x.npz`: 5 LR channels
- `lr_burst.npy`: 248 LR frames, required for on-demand 2x drizzle
- `shifts.npy`
- `metadata.json`

`obs_features_4x.npz` is not part of the current training contract.
Training enables burst augmentation by default so each epoch rebuilds drizzle
features from a perturbed frame subset. Use `--no-burst-augment` only for
legacy pools that intentionally rely on fixed precomputed drizzle features.

Adoption boundary: 4x output is a presentation/regularization grid, not
evidence for new 10-14 um information. A 4x checkpoint is only useful if it
beats the M4 MAP-TV anchor on FRC/zigzag gates and is not worse than EP07 2x
x2up on artifact score and contour quality.

Smoke command:

```bash
cd algos/ep12_4x_sr
uv run pytest
```
