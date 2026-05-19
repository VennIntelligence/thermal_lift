#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parent
EP08_ROOT = SCRIPTS_DIR.parent
PROJECT_ROOT = EP08_ROOT.parent.parent
SRC = EP08_ROOT / "src"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ep08.stage1 import parse_patch_shape

PROJECT_OUTPUT = PROJECT_ROOT / "output" / "ep08_inr_sr"
STAGE3_OUTPUT = PROJECT_OUTPUT / "stage3"
STAGE3_DEFAULT_N_FRAMES = 64
STAGE3_COORD_ASPECT_MODE = "preserve"

Stage1Runner = Callable[[str, argparse.Namespace], None]
SingleRunner = Callable[[argparse.Namespace], None]


def normalize_method(method: str) -> str:
    value = method.strip().lower().replace("-", "_")
    if value == "map_tv":
        return "ep06_map_tv"
    if value in {"siren", "wire", "deepinv_dip", "ep06_map_tv"}:
        return value
    raise ValueError(f"unknown Stage 3 method: {method!r}")


def default_config_path(method: str) -> Path | None:
    resolved = normalize_method(method)
    if resolved in {"siren", "wire", "deepinv_dip"}:
        return EP08_ROOT / "configs" / f"{resolved}_stage3.yaml"
    return None


def _load_yaml_mapping(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"config must contain a mapping: {path}")
    return payload


def _config_default_n_frames(path: Path | None) -> int:
    data_cfg = _load_yaml_mapping(path).get("data", {})
    if isinstance(data_cfg, dict) and data_cfg.get("default_n_frames") is not None:
        return int(data_cfg["default_n_frames"])
    return STAGE3_DEFAULT_N_FRAMES


def _config_default_patch_shape(path: Path | None) -> int | tuple[int, int] | None:
    data_cfg = _load_yaml_mapping(path).get("data", {})
    if not isinstance(data_cfg, dict):
        return None
    if "patch_shape" in data_cfg:
        return parse_patch_shape(data_cfg["patch_shape"])
    return None


def patch_shape_label(patch_shape: int | tuple[int, int] | None) -> str:
    if patch_shape is None:
        return "full"
    if isinstance(patch_shape, tuple):
        return f"{int(patch_shape[0])}x{int(patch_shape[1])}"
    return str(int(patch_shape))


def default_output_dir(
    method: str,
    *,
    n_frames: int,
    patch_shape: int | tuple[int, int] | None,
    coord_aspect_mode: str,
) -> Path:
    resolved = normalize_method(method)
    patch = patch_shape_label(patch_shape)
    aspect = str(coord_aspect_mode).strip().lower()
    return STAGE3_OUTPUT / f"{resolved}_{int(n_frames):03d}_{patch}_{aspect}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unified EP08 Stage 3 launcher for full-frame progressive SIREN, "
            "WIRE, DeepInverse-DIP, and EP06 MAP-TV baselines."
        )
    )
    parser.add_argument("method", choices=["siren", "wire", "deepinv_dip", "ep06_map_tv", "map_tv"])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-mode", choices=["synthetic", "real"], default=None)
    parser.add_argument("--device", default=None, help="cuda:0, cuda:1, or cpu")
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--n-frames", type=int, default=None)
    parser.add_argument("--patch-shape", type=parse_patch_shape, default=argparse.SUPPRESS, help="LR patch shape: H,W, a single int, or full/None")
    parser.add_argument("--patch-size", type=int, default=None, help="Backward-compatible square LR patch size in pixels")
    parser.add_argument("--coord-aspect-mode", "--coordinate-aspect-mode", choices=["preserve", "stretch"], default=None)
    parser.add_argument("--batch-k", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--min-lr-factor", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None, dest="grad_clip_norm")
    parser.add_argument("--val-interval", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None, dest="early_stop_patience")
    parser.add_argument("--early-stopping-min-delta", type=float, default=None, dest="early_stop_min_delta")
    parser.add_argument("--early-stopping-min-steps", type=int, default=None, dest="early_stop_min_steps")
    parser.add_argument("--split-half-max-iter", type=int, default=None)
    parser.add_argument("--skip-split-half", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--alignment-method", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--frame-audit-path", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)

    parser.add_argument("--hidden-features", type=int, default=None)
    parser.add_argument("--hidden-layers", type=int, default=None)
    parser.add_argument("--omega-0", type=float, default=None)
    parser.add_argument("--first-omega-0", type=float, default=None)
    parser.add_argument("--hidden-omega-0", type=float, default=None)
    parser.add_argument("--sigma-0", type=float, default=None)
    parser.add_argument("--first-sigma-0", type=float, default=None)
    parser.add_argument("--hidden-sigma-0", type=float, default=None)

    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--in-spatial", default=None, help="HxW DeepInverse-DIP latent noise size, e.g. 32,32")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument("--lambda-tv", type=float, default=1.0e-3)
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--tol", type=float, default=1.0e-4)
    parser.add_argument("--tv-inner-iter", type=int, default=30)
    parser.add_argument("--highpass-sigma", type=float, default=8.0)
    parser.add_argument("--psf-sigma", type=float, default=1.0)
    parser.add_argument("--noise-sigma", type=float, default=0.0724)
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.method = normalize_method(args.method)
    if args.config is None:
        args.config = default_config_path(args.method)

    if args.n_frames is None:
        args.n_frames = _config_default_n_frames(args.config)

    if not hasattr(args, "patch_shape"):
        if args.patch_size is not None:
            args.patch_shape = int(args.patch_size)
        else:
            args.patch_shape = _config_default_patch_shape(args.config)

    if args.coord_aspect_mode is None:
        coordinates_cfg = _load_yaml_mapping(args.config).get("coordinates", {})
        if isinstance(coordinates_cfg, dict) and coordinates_cfg.get("aspect_mode") is not None:
            args.coord_aspect_mode = str(coordinates_cfg["aspect_mode"])
        else:
            args.coord_aspect_mode = STAGE3_COORD_ASPECT_MODE

    if args.output_dir is None:
        args.output_dir = default_output_dir(
            args.method,
            n_frames=int(args.n_frames),
            patch_shape=args.patch_shape,
            coord_aspect_mode=str(args.coord_aspect_mode),
        )
    return args


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return finalize_args(build_parser().parse_args(argv))


def dispatch_stage3(
    args: argparse.Namespace,
    *,
    stage1_runner: Stage1Runner | None = None,
    deepinv_runner: SingleRunner | None = None,
    map_tv_runner: SingleRunner | None = None,
) -> None:
    method = normalize_method(args.method)
    if method in {"siren", "wire"}:
        if stage1_runner is None:
            from ep08.stage1 import run_stage1_training

            stage1_runner = run_stage1_training
        stage1_runner(method, args)
        return

    if method == "deepinv_dip":
        if deepinv_runner is None:
            from train_deepinv_dip import run_deepinv_dip_training

            deepinv_runner = run_deepinv_dip_training
        deepinv_runner(args)
        return

    if map_tv_runner is None:
        from generate_ep06_stage3_baseline import run_map_tv_stage3_baseline

        map_tv_runner = run_map_tv_stage3_baseline
    map_tv_runner(args)


def main(argv: Sequence[str] | None = None) -> None:
    dispatch_stage3(parse_args(argv))


if __name__ == "__main__":
    main()


__all__ = [
    "STAGE3_DEFAULT_N_FRAMES",
    "default_config_path",
    "default_output_dir",
    "dispatch_stage3",
    "finalize_args",
    "normalize_method",
    "parse_args",
    "patch_shape_label",
]
