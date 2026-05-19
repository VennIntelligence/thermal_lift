"""TCForge core library for ThermalChipPhantom synthetic data generation."""

from __future__ import annotations

from .forward import FORWARD_MODES, generate_lr_burst, physical_block_average_forward
from .geometry import (
    build_scene_mask,
    build_scene_mask_with_metadata,
    composite,
    make_cross,
    make_frame,
    make_l_shape,
    make_pin_array,
    make_rectangle,
    make_trenches,
    rotate_mask,
)
from .highpass import highpass_preprocess
from .manifest import SceneManifest
from .evaluate import (
    aggregate_scene_metrics,
    binary_iou,
    boundary_f1,
    evaluate_dataset,
    evaluate_scene,
    finite_summary,
    mae,
    nrmse,
    psnr,
    rmse,
    summarize_scene,
)
from .physics import add_noise, apply_drift, edge_map, render_temperature_field
from .shifts import SHIFT_CONVENTION, ideal_phase_grid, load_shift_profile

__version__ = "0.1.0"

__all__ = [
    "FORWARD_MODES",
    "SHIFT_CONVENTION",
    "SceneManifest",
    "__version__",
    "add_noise",
    "aggregate_scene_metrics",
    "apply_drift",
    "binary_iou",
    "boundary_f1",
    "build_scene_mask",
    "build_scene_mask_with_metadata",
    "composite",
    "edge_map",
    "evaluate_dataset",
    "evaluate_scene",
    "finite_summary",
    "generate_lr_burst",
    "highpass_preprocess",
    "ideal_phase_grid",
    "load_shift_profile",
    "make_cross",
    "make_frame",
    "make_l_shape",
    "make_pin_array",
    "make_rectangle",
    "make_trenches",
    "mae",
    "nrmse",
    "physical_block_average_forward",
    "psnr",
    "render_temperature_field",
    "rmse",
    "rotate_mask",
    "summarize_scene",
]
