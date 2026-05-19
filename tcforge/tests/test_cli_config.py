from __future__ import annotations

import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from generate_thermal_chip_phantom import _validate_supported_config


def test_generator_fails_fast_on_unsupported_benchmark_keys() -> None:
    config = {
        "num_scenes": 1,
        "forward_modes": ["exact_ep06_point"],
        "drift_tracks": ["clean", "drift_scalar"],
        "storage_strategy": "crop_roi",
        "split": {"train": 1},
    }

    with pytest.raises(ValueError, match="Unsupported benchmark/P1 keys"):
        _validate_supported_config(config)
