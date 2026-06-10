# EP15 Info Limit

Independent UV project for EP15 first-principles information-limit checks.

## Setup

```bash
cd algos/ep15_info_limit
uv sync
uv pip install -e ../../core
```

## M1 Phase Structure

```bash
uv run python scripts/run_m1_phase_structure.py
```

Outputs are written to `output/ep15_info_limit/m1_phase_structure/`.

## M2 FRC Information Cutoff

```bash
uv run python scripts/run_m2_frc.py
```

Outputs are written to `output/ep15_info_limit/m2_frc/`.

## M3 Sigma Arbitration

```bash
uv run python scripts/run_m3_sigma_arbitration.py
```

Outputs are written to `output/ep15_info_limit/m3_sigma/`.
