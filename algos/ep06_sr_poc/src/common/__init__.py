"""Shared infrastructure for EP06 2x contour-level SR POC."""

from .alignment import load_alignment_shifts, load_alignment_table, load_quality_weights, validate_alignment_frame_count
from .data_loader import (
    bicubic_upsample,
    highpass_preprocess,
    load_main_session_frames,
    load_main_session_metadata,
    offset_correction,
)
from .forward_model import (
    ObservationOperator,
    adjoint,
    build_observation_operator,
    downsample_block_average,
    forward,
    upsample_block_adjoint,
)
from .metrics import (
    artifact_score,
    contour_chamfer,
    contour_chamfer_from_edges,
    gradient_magnitude,
    psnr,
    split_half_consistency,
)

__all__ = [
    "ObservationOperator",
    "adjoint",
    "artifact_score",
    "bicubic_upsample",
    "build_observation_operator",
    "contour_chamfer",
    "contour_chamfer_from_edges",
    "downsample_block_average",
    "forward",
    "gradient_magnitude",
    "highpass_preprocess",
    "load_alignment_shifts",
    "load_alignment_table",
    "load_main_session_frames",
    "load_main_session_metadata",
    "load_quality_weights",
    "offset_correction",
    "psnr",
    "split_half_consistency",
    "upsample_block_adjoint",
    "validate_alignment_frame_count",
]
