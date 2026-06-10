# EP10 Drizzle CPU SR

CPU 2x micro-scan reconstruction using STScI `drizzle`.

## Environment

```bash
cd algos/ep10_drizzle
uv sync
```

The experiment imports EP06 shared modules from `../ep06_sr_poc/src/common`.

## API Notes

`drizzle.resample.Drizzle` 2.2.0 exposes `pixfrac` on `Drizzle.add_image`, not on the constructor:

```python
Drizzle(kernel="square", out_shape=(960, 1280), disable_ctx=True)
driz.add_image(data, exptime=1.0, pixmap=pixmap, pixfrac=0.7, in_units="cps")
```

`out_img` is the weighted mean output image. `out_wht` is the accumulated output weight/count image and is used here as the coverage map.

## Run

```bash
cd algos/ep10_drizzle
uv run python scripts/run_drizzle.py
```

Outputs are written to `../../output/ep10_drizzle/`.

For an auditable detached run with a persistent log:

```bash
cd /home/ujs/mycode/thermal_lift
setsid bash -c 'echo $$ > output/ep10_drizzle/full_run.pid; trap "rm -f output/ep10_drizzle/full_run.pid" EXIT; exec algos/ep10_drizzle/.venv/bin/python -u algos/ep10_drizzle/scripts/run_drizzle.py --output-dir output/ep10_drizzle' \
  > output/ep10_drizzle/full_run.log 2>&1 < /dev/null &
```

`artifact_score` is written without the LR overshoot component so it is
comparable with MAP-TV and TGV. The legacy overshoot-inclusive value is kept as
`artifact_score_with_lr_overshoot` for debugging only.
