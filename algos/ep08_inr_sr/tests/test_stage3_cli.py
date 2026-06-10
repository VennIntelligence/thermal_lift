from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent.parent


def _load_script(name: str) -> ModuleType:
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage3_cli_modules_import() -> None:
    train_stage3 = _load_script("train_stage3")
    ep06_stage3 = _load_script("generate_ep06_stage3_baseline")

    assert hasattr(train_stage3, "dispatch_stage3")
    assert hasattr(ep06_stage3, "run_map_tv_stage3_baseline")


def test_stage3_default_config_paths_and_output_dirs() -> None:
    train_stage3 = _load_script("train_stage3")

    for method in ("siren", "wire", "deepinv_dip"):
        args = train_stage3.parse_args([method])

        assert args.config == ROOT / "configs" / f"{method}_stage3.yaml"
        assert args.n_frames == 64
        assert args.patch_shape is None
        assert args.coord_aspect_mode == "preserve"
        assert args.output_dir == PROJECT_ROOT / "output" / "ep08_inr_sr" / "stage3" / f"{method}_064_full_preserve"


def test_stage3_dispatch_helpers_do_not_train() -> None:
    train_stage3 = _load_script("train_stage3")
    calls: list[tuple[str, object]] = []

    siren_args = train_stage3.parse_args(["siren", "--data-mode", "real", "--n-frames", "32", "--patch-shape", "full", "--max-iter", "2"])
    train_stage3.dispatch_stage3(siren_args, stage1_runner=lambda method, args: calls.append((method, args)))

    deepinv_args = train_stage3.parse_args(["deepinv_dip", "--n-frames", "128", "--patch-shape", "full"])
    train_stage3.dispatch_stage3(deepinv_args, deepinv_runner=lambda args: calls.append(("deepinv_dip", args)))

    map_args = train_stage3.parse_args(["map_tv", "--n-frames", "248", "--patch-shape", "full"])
    train_stage3.dispatch_stage3(map_args, map_tv_runner=lambda args: calls.append(("ep06_map_tv", args)))

    assert [call[0] for call in calls] == ["siren", "deepinv_dip", "ep06_map_tv"]
    assert siren_args.output_dir.name == "siren_032_full_preserve"
    assert deepinv_args.output_dir.name == "deepinv_dip_128_full_preserve"
    assert map_args.method == "ep06_map_tv"
    assert map_args.output_dir.name == "ep06_map_tv_248_full_preserve"


def test_stage3_siren_args_are_stage1_override_compatible() -> None:
    train_stage3 = _load_script("train_stage3")
    from ep08.stage1 import apply_cli_overrides

    args = train_stage3.parse_args(["siren", "--n-frames", "64", "--hidden-features", "32", "--hidden-layers", "1"])
    cfg = apply_cli_overrides({"model": {"name": "siren"}}, args)

    assert cfg["data"]["default_n_frames"] == 64
    assert cfg["model"]["hidden_features"] == 32
    assert cfg["model"]["hidden_layers"] == 1
    assert "hidden_channels" not in cfg["model"]
    assert "latent_spatial" not in cfg["model"]


def test_map_tv_stage3_patch_shape_parsing() -> None:
    ep06_stage3 = _load_script("generate_ep06_stage3_baseline")

    full_args = ep06_stage3.parse_args(["--patch-shape", "full", "--n-frames", "128"])
    rectangular_args = ep06_stage3.parse_args(["--patch-shape", "240,320", "--n-frames", "64"])

    assert full_args.patch_shape is None
    assert full_args.output_dir.name == "ep06_map_tv_128_full_preserve"
    assert rectangular_args.patch_shape == (240, 320)
    assert rectangular_args.output_dir.name == "ep06_map_tv_064_240x320_preserve"
