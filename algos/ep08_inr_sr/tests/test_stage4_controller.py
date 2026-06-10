from __future__ import annotations

import importlib.util
import sys
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
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_stage4_controller_plan_defaults_do_not_train() -> None:
    controller = _load_script("stage4_controller")

    args = controller.parse_args(["plan", "--max-frame", "64"])
    tasks = controller.build_plan(args)

    assert [task.run_name for task in tasks] == [
        "ep06_map_tv_064_full_preserve",
        "siren_064_full_preserve",
        "wire_064_full_preserve",
        "deepinv_dip_064_full_preserve",
    ]
    assert tasks[0].resource == "cpu"
    assert tasks[1].resource == "cuda:1"
    assert tasks[2].resource == "cuda:1"
    assert tasks[3].resource == "cuda:0"
    assert "--output-dir" in tasks[1].command
    assert str(PROJECT_ROOT / "output" / "ep08_inr_sr" / "stage3" / "siren_064_full_preserve") in tasks[1].command
    assert tasks[0].command[-4:] == ("--max-iter", "50", "--workers", "1")


def test_stage4_controller_batch_override_is_neural_only() -> None:
    controller = _load_script("stage4_controller")

    args = controller.parse_args(["plan", "--max-frame", "64", "--batch-k", "4"])
    tasks = controller.build_plan(args)
    command_by_method = {task.method: task.command for task in tasks}

    assert "--batch-k" not in command_by_method["ep06_map_tv"]
    for method in ("siren", "wire", "deepinv_dip"):
        command = command_by_method[method]
        assert "--batch-k" in command
        assert command[command.index("--batch-k") + 1] == "4"


def test_stage4_controller_default_final_phase_is_clean_248() -> None:
    controller = _load_script("stage4_controller")

    args = controller.parse_args(["plan"])
    tasks = controller.build_plan(args)

    assert controller.parse_frame_list("64,128,all_clean") == [64, 128, 248]
    assert sorted({task.n_frames for task in tasks}) == [64, 128, 248]
    assert any(task.run_name == "ep06_map_tv_248_full_preserve" for task in tasks)
    assert any(task.run_name == "deepinv_dip_248_full_preserve" for task in tasks)


def test_stage4_health_gate_neural_best_step_and_map_tv_exception() -> None:
    controller = _load_script("stage4_controller")
    tasks = controller.build_plan(controller.parse_args(["plan", "--max-frame", "64"]))
    siren = next(task for task in tasks if task.method == "siren")
    map_tv = next(task for task in tasks if task.method == "ep06_map_tv")
    base_metrics = {
        "stage_gate": "complete",
        "n_frames": 64,
        "holdout_residual": 3.0,
        "split_half_nrmse": 0.2,
        "artifact_score": 0.5,
        "raw_control_agreement": 0.3,
        "p95_gradient": 0.9,
        "best_step": 700,
        "early_stopped_before_500": False,
    }

    assert controller.assess_metrics(siren, base_metrics, check_artifacts=False)["healthy"]

    early_metrics = dict(base_metrics, best_step=100)
    siren_health = controller.assess_metrics(siren, early_metrics, check_artifacts=False)
    assert not siren_health["healthy"]
    assert any("best_step" in item for item in siren_health["failures"])

    map_metrics = dict(base_metrics, method="ep06_map_tv", best_step=50)
    assert controller.assess_metrics(map_tv, map_metrics, check_artifacts=False)["healthy"]
