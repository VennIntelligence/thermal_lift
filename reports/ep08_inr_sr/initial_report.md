# EP08 Initial Report — INR-based 2x Contour SR

## Scope

EP08 compares four methods under one contour-level SR evaluation frame:

| Method | Family | Role |
|---|---|---|
| EP06 MAP-TV | Classic optimization | Reference baseline |
| SIREN | INR / sine activation | Continuous-coordinate baseline |
| WIRE | INR / Gabor activation | Main edge-sensitive candidate |
| Deep Decoder | CNN decoder prior | Non-INR deep prior control |

The goal is not to claim 5 um metrology or treat display magnification as SR evidence. The target is clearer and more stable chip internal contour visibility on the 2x grid, constrained by forward consistency, split-half stability, artifact checks, and raw-control agreement.

## P0 Gate Status

| Gate | Required evidence | Current status |
|---|---|---|
| Forward equivalence | PyTorch forward matches EP06 NumPy ObservationOperator | Unit tests passed |
| Highpass equivalence | PyTorch highpass matches EP06 `sigma_bg=5.0`, `mode="nearest"` | Unit tests passed |
| Real-data input | Main session frames + EP05 shifts, no stage command truth use | CPU smoke passed |
| EP06 baseline provenance | MAP-TV metrics source recorded without fabricating values | Partial proxy metrics recorded; hold-out/split-half marked unavailable |
| SIREN training/eval | highpass + raw-control + metrics | Stage 1 complete, five metrics generated |
| WIRE training/eval | highpass + raw-control + metrics | Stage 1 complete with dual-projection Gabor; artifact risk higher than SIREN |
| Deep Decoder training/eval | highpass + raw-control + metrics | Stage 2 complete |
| DeepInverse-DIP training/eval | DeepInverse ConvDecoder-DIP with EP08 custom Physics | Stage 2 complete |
| Stage 2 comparison | same ROI, same scales, same split where applicable | Five-way comparison complete; EP06 proxy fields marked by protocol |

Current verification snapshot: `cd algos/ep08_inr_sr && uv run pytest -q` returns 32 passed, and `uv run python scripts/validate_p0.py` writes passed forward/highpass/split validation artifacts under `output/ep08_inr_sr/`. Smoke outputs under `output/ep08_inr_sr/` are generated artifacts and are not report conclusions.

## Stage 1 Results

Both INR methods were trained on the same 32 main-session frames, center 256x256 LR patch, HR 512x512 grid, seed=42 split, `batch_k=8`, `lr=5e-4`, `warmup_steps=200`, and `early_stop_patience=1000`. The shared split is 27 train frames and 5 hold-out frames.

| Method | Hold-out residual | Split-half NRMSE | Artifact score | Raw-control agreement | P95 gradient | Best step | Final step |
|---|---:|---:|---:|---:|---:|---:|---:|
| SIREN | 3.6729 | 0.2916 | 0.2178 | 0.2085 | 0.9289 | 1630 | 2630 |
| WIRE | 2.9665 | 0.4899 | 2.2434 | 0.0868 | 1.6098 | 2989 | 3989 |

Interpretation: SIREN and WIRE both converged and produced complete Stage 1 artifacts. WIRE obtains a lower hold-out residual and stronger P95 gradient, but its split-half NRMSE and artifact score are worse and its raw-control agreement is lower. This makes WIRE a higher-frequency but higher-risk result in this small run, not a clear winner.

Stage 1 artifacts are recorded under:

- `output/ep08_inr_sr/siren_stage1/`
- `output/ep08_inr_sr/wire_stage1/`
- `output/ep08_inr_sr/stage1_comparison.csv`

Each method directory contains `checkpoint.pt`, `hr_image.npy`, `metrics.json`, `metrics.csv`, `training_history.csv/json`, `training_curve.png`, `hr_highpass.png`, `hr_raw_control.png`, `split_half_difference.png`, `config_used.json`, and `split_indices.json`.

## Evidence Rules

- Stage command is a prior / initialization / regularizer only; it is not alignment ground truth.
- EP04 localization is an alignment anchor and quality gate, not a final SR proof.
- EP06 MAP-TV is a baseline for comparison, not external optical truth.
- Highpass output is a signed structure map. It must be paired with raw-control views before claiming contour improvement.
- Gradient/Tenengrad-style metrics are auxiliary sharpness proxies; they cannot independently prove SR success.

## Stage 2 Results

TCForge benchmark used a synthetic scene with HR ground truth (`lr_shape=(256,256)`, `n_frames=32`, `scale=2`). All four methods are evaluated in the HR highpass domain:

| Method | Highpass PSNR (dB) | Global SSIM proxy |
|---|---:|---:|
| SIREN | 24.6634 | 0.7520 |
| WIRE | 24.4126 | 0.7325 |
| Deep Decoder | 19.8896 | 0.2068 |
| DeepInverse-DIP | 18.4729 | 0.4447 |

Five-way Stage 2 comparison:

| Method | Hold-out residual | Split-half NRMSE | Artifact score | Raw-control agreement | P95 gradient | Best step | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| SIREN | 3.6729 | 0.2916 | 0.2178 | 0.2085 | 0.9289 | 1630 | Most balanced |
| WIRE | 2.9665 | 0.4899 | 2.2434 | 0.0868 | 1.6098 | 2989 | Sharper but artifact-prone |
| Deep Decoder | 3.6258 | 0.4406 | 0.0355 | 0.3227 | 0.4885 | 6685 | Low-artifact decoder control |
| DeepInverse-DIP | 2.3051 | 0.4562 | 3.0867 | 0.0608 | 1.2143 | 9900 | Strong fit, artifact-prone |
| EP06 MAP-TV | 2.2439 | 0.1743 | 0.3978 | 0.3102 | 0.8667 | 100 | Same-protocol classic baseline |

EP06 MAP-TV now has a patch-level same-protocol row generated at `output/ep08_inr_sr/ep06_patch_baseline/metrics.json`. The older full-frame proxy remains useful only as a historical artifact reference.

Stage 2 interpretation:

- DeepInverse-DIP confirms that a mature ConvDecoder backbone can fit held-out LR frames strongly, but its artifact score remains the worst in the table.
- Deep Decoder is the lowest-artifact learned result after the native 512x512 fix, but it still does not dominate SIREN on stability or forward consistency.
- WIRE increases high-frequency response, but Stage 1/2 evidence still shows higher artifact and lower stability than SIREN.
- SIREN is the best Stage 3 candidate because it is the most balanced across forward consistency, stability, artifact control, and visual structure.

## Stage 3 Recommendation

Proceed to Stage 3 with SIREN as the primary expansion method. Keep Deep Decoder as a low-artifact conservative control. Do not promote DeepInverse-DIP or WIRE to full-frame expansion unless a later ablation adds holdout-driven early stopping or stronger artifact control.
