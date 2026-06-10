# EP10 MAP-TGV SR

CPU MAP super-resolution experiment using CCPi-Regularisation-Toolkit TGV as
the proximal operator in the EP06 MAP-TV FISTA loop.

## Environment

This algorithm intentionally uses a standalone conda environment instead of the
root UV environment. EP10 depends on `ccpi-regulariser`, whose supported binary
distribution is provided through conda channels (`conda-forge` + `ccpi`); using
that package manager keeps the CCPi CPU/CUDA backend reproducible without local
C/C++/CUDA builds. The local Chambolle-Pock implementation is only a diagnostic
fallback and is recorded in the run provenance when CCPi cannot be imported or
executed.

```bash
cd algos/ep10_tgv_sr
conda env create -p .venv -f environment.yml
conda run -p .venv pip install -e ../../core
```

The repository-local run used Miniforge/mamba because `conda` was not already
on PATH:

```bash
~/miniforge3/bin/mamba env create -p .venv -f environment.yml
~/miniforge3/bin/mamba run -p .venv pip install -e ../../core
```

## Run

```bash
~/miniforge3/bin/mamba run -p algos/ep10_tgv_sr/.venv \
  python algos/ep10_tgv_sr/scripts/run_tgv_sr.py
```

Detached full run used in this workspace:

```bash
setsid bash -c 'cd /home/ujs/mycode/thermal_lift; echo $$ > output/ep10_tgv_sr/full_run.pid; exec algos/ep10_tgv_sr/.venv/bin/python -u algos/ep10_tgv_sr/scripts/run_tgv_sr.py --output-dir output/ep10_tgv_sr --workers 4' \
  > output/ep10_tgv_sr/full_run.log 2>&1 < /dev/null &
```

The runner uses `--tgv-device auto` by default. On a CUDA-capable machine it
uses CCPi-RGL's official GPU backend and assigns one parameter worker per
detected GPU; on CPU-only machines it falls back to a single parameter worker.
Each reconstruction iteration records `tgv_backend`, `tgv_backend_status`,
`tgv_backend_device`, and `tgv_backend_error` in the returned records and output
CSV, so CCPi success and fallback paths are auditable after the run. The latest
proximal call can also be queried in Python with
`ep10_tgv_sr.tgv.get_tgv_backend_provenance()`.
Final comparison images and `best_hr_highpass.npy` are reconstructed from the
248 clean-frame real input set. Holdout reconstructions are used only for the
holdout MSE.

Monitor:

```bash
pid=$(cat output/ep10_tgv_sr/full_run.pid)
ps -p "$pid" -o pid,ppid,etime,stat,%cpu,%mem,cmd
tail -f output/ep10_tgv_sr/full_run.log
```

Outputs are written to ignored `output/ep10_tgv_sr/`:

- `sweep_results.csv`
- `best_hr_highpass.npy`
- `tgv_vs_tv_comparison.png`
- `synthetic_validation.png`

The script first runs a 64x64 piecewise-linear denoising gate. It stops before
real data if TGV does not reduce noise while keeping the ramp less staircased
than TV.
