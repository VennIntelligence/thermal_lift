# EP09 — PSF Sigma Calibration

CPU-only calibration of the Gaussian PSF sigma used by the Thermal Lift forward model.

## Environment

EP09 uses the project root UV environment. It does not define a separate algorithm venv.

```bash
cd /home/ujs/mycode/thermal_lift
uv sync
uv pip install -e core/
```

## Routes

```bash
uv run python algos/ep09_psf_calibration/scripts/run_forward_residual.py
uv run python algos/ep09_psf_calibration/scripts/run_esf_fitting.py
uv run python algos/ep09_psf_calibration/scripts/run_joint_estimation.py
uv run python algos/ep09_psf_calibration/scripts/summarize_calibration.py
```

Outputs are written to `output/ep09_psf_calibration/`. The summary script also updates
`configs/psf_calibration.json` and writes `reports/ep09_psf_calibration/psf_calibration_report.md`.

## Interpretation

All sigma values are reported in LR detector pixels unless the field name says `hr_px_at_2x`.
Route A is the primary estimate because it directly scores the EP06 forward model against held-out LR
observations. Routes B and C are independent cross-checks and gate diagnostics.
