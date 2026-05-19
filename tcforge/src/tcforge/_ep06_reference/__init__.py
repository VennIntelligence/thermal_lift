"""Locked EP06 reference implementations vendored for TCForge."""

from .forward import (
    ObservationOperator,
    adjoint,
    build_observation_operator,
    downsample_block_average,
    forward,
    upsample_block_adjoint,
)

__all__ = [
    "ObservationOperator",
    "adjoint",
    "build_observation_operator",
    "downsample_block_average",
    "forward",
    "upsample_block_adjoint",
]
