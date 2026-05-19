#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

SCRIPTS_DIR = Path(__file__).resolve().parent
EP08_ROOT = SCRIPTS_DIR.parent
PROJECT_ROOT = EP08_ROOT.parent.parent
SRC = EP08_ROOT / "src"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from train_stage3 import default_output_dir, normalize_method

PROJECT_OUTPUT = PROJECT_ROOT / "output" / "ep08_inr_sr"
STAGE3_OUTPUT = PROJECT_OUTPUT / "stage3"
CONTROL_DIR = PROJECT_OUTPUT / "stage4_controller"
LOG_DIR = CONTROL_DIR / "logs"
RUN_RECORD_DIR = CONTROL_DIR / "runs"
STATUS_JSON = CONTROL_DIR / "status.json"
LOCK_PATH = CONTROL_DIR / "tick.lock"

DEFAULT_FRAMES = (64, 128, 255)
METHOD_SEQUENCE = ("map_tv", "siren", "wire", "deepinv_dip")
NEURAL_METHODS = {"siren", "wire", "deepinv_dip"}
METHOD_LABELS = {
    "ep06_map_tv": "EP06 MAP-TV",
    "siren": "SIREN",
    "wire": "WIRE",
    "deepinv_dip": "DeepInv-DIP",
}


@dataclass(frozen=True)
class Task:
    method: str
    command_method: str
    n_frames: int
    coord_aspect_mode: str
    output_dir: Path
    command: tuple[str, ...]
    resource: str
    priority: int

    @property
    def run_name(self) -> str:
        return self.output_dir.name


@dataclass(frozen=True)
class HealthPolicy:
    holdout_min: float = 0.0
    holdout_max: float = 10.0
    split_half_max: float = 1.0
    artifact_warn: float = 5.0
    artifact_max: float = 20.0
    neural_min_best_step: int = 500


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def parse_frame_list(value: str | Sequence[int]) -> list[int]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        frames = [int(part) for part in parts]
    else:
        frames = [int(part) for part in value]
    if not frames:
        raise ValueError("at least one frame count is required")
    invalid = [frame for frame in frames if frame <= 0]
    if invalid:
        raise ValueError(f"frame counts must be positive: {invalid}")
    return sorted(dict.fromkeys(frames))


def finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def device_resource(device: str | None) -> str:
    if device is None:
        return "cpu"
    normalized = str(device).strip().lower()
    if not normalized or normalized == "cpu":
        return "cpu"
    return normalized


def task_command(
    *,
    method: str,
    n_frames: int,
    coord_aspect_mode: str,
    output_dir: Path,
    inr_device: str,
    deepinv_device: str,
    batch_k: int | None,
    map_tv_max_iter: int,
    map_tv_workers: int,
) -> tuple[tuple[str, ...], str, int]:
    command = [
        "uv",
        "run",
        "python",
        "scripts/train_stage3.py",
        method,
        "--n-frames",
        str(int(n_frames)),
        "--coord-aspect-mode",
        coord_aspect_mode,
        "--output-dir",
        str(output_dir),
    ]
    if method in {"siren", "wire"}:
        command.extend(["--device", inr_device])
        if batch_k is not None:
            command.extend(["--batch-k", str(int(batch_k))])
        return tuple(command), device_resource(inr_device), {"siren": 10, "wire": 20}[method]
    if method == "deepinv_dip":
        command.extend(["--device", deepinv_device])
        if batch_k is not None:
            command.extend(["--batch-k", str(int(batch_k))])
        return tuple(command), device_resource(deepinv_device), 30
    command.extend(["--max-iter", str(int(map_tv_max_iter)), "--workers", str(int(map_tv_workers))])
    return tuple(command), "cpu", 0


def build_plan(args: argparse.Namespace) -> list[Task]:
    frames = parse_frame_list(args.frames)
    if args.max_frame is not None:
        frames = [frame for frame in frames if frame <= int(args.max_frame)]
    tasks: list[Task] = []
    for n_frames in frames:
        for command_method in METHOD_SEQUENCE:
            normalized = normalize_method(command_method)
            output_dir = default_output_dir(
                normalized,
                n_frames=n_frames,
                patch_shape=None,
                coord_aspect_mode=args.coord_aspect_mode,
            )
            command, resource, priority = task_command(
                method=command_method,
                n_frames=n_frames,
                coord_aspect_mode=args.coord_aspect_mode,
                output_dir=output_dir,
                inr_device=args.inr_device,
                deepinv_device=args.deepinv_device,
                batch_k=args.batch_k,
                map_tv_max_iter=args.map_tv_max_iter,
                map_tv_workers=args.map_tv_workers,
            )
            tasks.append(
                Task(
                    method=normalized,
                    command_method=command_method,
                    n_frames=n_frames,
                    coord_aspect_mode=args.coord_aspect_mode,
                    output_dir=output_dir,
                    command=command,
                    resource=resource,
                    priority=priority,
                )
            )
    return sorted(tasks, key=lambda task: (task.n_frames, task.priority, task.method))


def launch_record_path(task: Task) -> Path:
    return RUN_RECORD_DIR / f"{task.run_name}.launch.json"


def log_path(task: Task) -> Path:
    return LOG_DIR / f"{task.run_name}.log"


def pid_matches_record(pid: int, record: dict[str, Any]) -> bool:
    cmdline_path = Path("/proc") / str(pid) / "cmdline"
    if not cmdline_path.exists():
        return False
    try:
        cmdline = cmdline_path.read_text(encoding="utf-8", errors="ignore").replace("\x00", " ")
    except OSError:
        return True
    output_dir = str(record.get("output_dir", ""))
    return "train_stage3.py" in cmdline and (not output_dir or output_dir in cmdline)


def is_pid_alive(record: dict[str, Any]) -> bool:
    try:
        pid = int(record.get("pid", -1))
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return pid_matches_record(pid, record)


def assess_metrics(
    task: Task,
    metrics: dict[str, Any],
    *,
    policy: HealthPolicy | None = None,
    check_artifacts: bool = True,
) -> dict[str, Any]:
    policy = policy or HealthPolicy()
    failures: list[str] = []
    warnings: list[str] = []

    gate = metrics.get("stage_gate") or metrics.get("stage1_gate")
    if gate != "complete":
        failures.append(f"stage gate is {gate!r}, expected 'complete'")

    actual_frames = metrics.get("n_frames")
    if actual_frames is not None and int(actual_frames) != int(task.n_frames):
        failures.append(f"n_frames is {actual_frames}, expected {task.n_frames}")

    holdout = finite_float(metrics.get("holdout_residual"))
    if holdout is None:
        failures.append("holdout_residual is missing or non-finite")
    elif holdout < policy.holdout_min or holdout > policy.holdout_max:
        failures.append(
            f"holdout_residual={holdout:.6g} outside [{policy.holdout_min}, {policy.holdout_max}]"
        )

    split = finite_float(metrics.get("split_half_nrmse"))
    if split is None:
        failures.append("split_half_nrmse is missing or non-finite")
    elif split >= policy.split_half_max:
        failures.append(f"split_half_nrmse={split:.6g} >= {policy.split_half_max}")

    artifact = finite_float(metrics.get("artifact_score"))
    if artifact is None:
        failures.append("artifact_score is missing or non-finite")
    else:
        if artifact > policy.artifact_max:
            failures.append(f"artifact_score={artifact:.6g} > {policy.artifact_max}")
        elif artifact > policy.artifact_warn:
            warnings.append(f"artifact_score={artifact:.6g} > warning threshold {policy.artifact_warn}")

    raw_control = finite_float(metrics.get("raw_control_agreement"))
    if raw_control is None:
        failures.append("raw_control_agreement is missing or non-finite")

    p95 = finite_float(metrics.get("p95_gradient"))
    if p95 is None:
        warnings.append("p95_gradient is missing or non-finite")

    if task.method in NEURAL_METHODS:
        best_step = metrics.get("best_step")
        try:
            best_step_int = int(best_step)
        except (TypeError, ValueError):
            best_step_int = -1
        if best_step_int < policy.neural_min_best_step:
            failures.append(f"best_step={best_step!r} < {policy.neural_min_best_step}")
        if bool(metrics.get("early_stopped_before_500", False)):
            failures.append("early_stopped_before_500 is true")

    if check_artifacts:
        required_arrays = ["hr_image.npy", "hr_raw_control.npy"]
        if split is not None:
            required_arrays.extend(["split_half_a.npy", "split_half_b.npy"])
        missing_arrays = [name for name in required_arrays if not (task.output_dir / name).exists()]
        if missing_arrays:
            failures.append("missing required arrays: " + ", ".join(missing_arrays))

        missing_figures = [
            name
            for name in ("training_curve.png", "hr_highpass.png", "hr_raw_control.png")
            if not (task.output_dir / name).exists()
        ]
        if missing_figures:
            warnings.append("missing expected figures: " + ", ".join(missing_figures))

    return {
        "healthy": not failures,
        "failures": failures,
        "warnings": warnings,
        "gate": gate,
        "holdout_residual": holdout,
        "split_half_nrmse": split,
        "artifact_score": artifact,
        "raw_control_agreement": raw_control,
        "p95_gradient": p95,
    }


def assess_task(task: Task) -> dict[str, Any]:
    metrics_path = task.output_dir / "metrics.json"
    record_path = launch_record_path(task)
    record = load_json(record_path)
    metrics = load_json(metrics_path)
    log = log_path(task)

    if metrics:
        health = assess_metrics(task, metrics)
        state = "complete_healthy" if health["healthy"] else "complete_unhealthy"
    elif record and is_pid_alive(record):
        health = {"healthy": False, "failures": [], "warnings": [], "gate": None}
        state = "running"
    elif record:
        health = {
            "healthy": False,
            "failures": ["process exited before metrics.json was written"],
            "warnings": [],
            "gate": None,
        }
        state = "failed_no_metrics"
    elif task.output_dir.exists():
        health = {
            "healthy": False,
            "failures": ["output directory exists but metrics.json is missing"],
            "warnings": [],
            "gate": None,
        }
        state = "incomplete_dir"
    else:
        health = {"healthy": False, "failures": [], "warnings": [], "gate": None}
        state = "pending"

    return {
        "run": task.run_name,
        "method": task.method,
        "method_label": METHOD_LABELS.get(task.method, task.method),
        "n_frames": task.n_frames,
        "resource": task.resource,
        "state": state,
        "healthy": bool(health.get("healthy", False)),
        "health": health,
        "pid": record.get("pid"),
        "started_at": record.get("started_at"),
        "command": list(task.command),
        "output_dir": str(task.output_dir),
        "metrics_path": str(metrics_path),
        "log_path": str(log),
    }


def summarize(tasks: list[Task]) -> dict[str, Any]:
    runs = [assess_task(task) for task in tasks]
    frames = sorted({run["n_frames"] for run in runs})
    phases: list[dict[str, Any]] = []
    current_phase: int | None = None
    blocked = False
    for frame in frames:
        phase_runs = [run for run in runs if run["n_frames"] == frame]
        states = {run["state"] for run in phase_runs}
        if all(run["state"] == "complete_healthy" for run in phase_runs):
            phase_state = "complete_healthy"
        elif any(run["state"] in {"complete_unhealthy", "failed_no_metrics", "incomplete_dir"} for run in phase_runs):
            phase_state = "blocked"
            blocked = True
        elif any(run["state"] == "running" for run in phase_runs):
            phase_state = "running"
        else:
            phase_state = "pending"
        phases.append({"n_frames": frame, "state": phase_state})
        if current_phase is None and phase_state != "complete_healthy":
            current_phase = frame
    overall = "complete" if current_phase is None else "blocked" if blocked else "active"
    return {
        "generated_at": now_iso(),
        "overall_state": overall,
        "current_phase": current_phase,
        "phases": phases,
        "runs": runs,
    }


def running_resources(summary: dict[str, Any]) -> set[str]:
    return {run["resource"] for run in summary["runs"] if run["state"] == "running"}


def launch_task(task: Task) -> dict[str, Any]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_RECORD_DIR.mkdir(parents=True, exist_ok=True)
    task.output_dir.mkdir(parents=True, exist_ok=True)
    log = log_path(task)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with log.open("ab") as log_file:
        log_file.write(f"\n\n===== launch {now_iso()} =====\n".encode("utf-8"))
        log_file.write((" ".join(task.command) + "\n").encode("utf-8"))
        process = subprocess.Popen(
            list(task.command),
            cwd=str(EP08_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
    record = {
        "pid": process.pid,
        "started_at": now_iso(),
        "command": list(task.command),
        "cwd": str(EP08_ROOT),
        "output_dir": str(task.output_dir),
        "log_path": str(log),
        "resource": task.resource,
        "n_frames": task.n_frames,
        "method": task.method,
    }
    save_json(launch_record_path(task), record)
    return record


class TickLock:
    def __init__(self, path: Path, stale_seconds: int = 1800) -> None:
        self.path = path
        self.stale_seconds = stale_seconds
        self.fd: int | None = None

    def __enter__(self) -> "TickLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            age = datetime.now().timestamp() - self.path.stat().st_mtime
            if age > self.stale_seconds:
                self.path.unlink(missing_ok=True)
        try:
            self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError(f"controller lock exists: {self.path}") from exc
        os.write(self.fd, f"pid={os.getpid()} started_at={now_iso()}\n".encode("utf-8"))
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.fd is not None:
            os.close(self.fd)
        self.path.unlink(missing_ok=True)


def select_launches(tasks: list[Task], summary: dict[str, Any]) -> list[Task]:
    current_phase = summary["current_phase"]
    if current_phase is None or summary["overall_state"] == "blocked":
        return []
    run_by_name = {run["run"]: run for run in summary["runs"]}
    used = running_resources(summary)
    launches: list[Task] = []
    for task in sorted((task for task in tasks if task.n_frames == current_phase), key=lambda item: item.priority):
        run = run_by_name[task.run_name]
        if run["state"] != "pending":
            continue
        if task.resource in used:
            continue
        launches.append(task)
        used.add(task.resource)
    return launches


def write_status(summary: dict[str, Any]) -> None:
    save_json(STATUS_JSON, summary)


def print_status(summary: dict[str, Any]) -> None:
    print(f"EP08 Stage 4 status @ {summary['generated_at']}")
    print(f"overall={summary['overall_state']} current_phase={summary['current_phase']}")
    print("")
    print(f"{'frames':>6}  {'method':<13}  {'resource':<7}  {'state':<18}  {'pid':>8}  {'holdout':>10}  {'split':>10}  {'best/gate':>12}")
    for run in sorted(summary["runs"], key=lambda item: (item["n_frames"], item["method"])):
        health = run.get("health", {})
        holdout = health.get("holdout_residual")
        split = health.get("split_half_nrmse")
        holdout_text = "-" if holdout is None else f"{float(holdout):.4g}"
        split_text = "-" if split is None else f"{float(split):.4g}"
        gate = health.get("gate") or "-"
        pid = run.get("pid") or "-"
        print(
            f"{run['n_frames']:>6}  {run['method']:<13}  {run['resource']:<7}  "
            f"{run['state']:<18}  {str(pid):>8}  {holdout_text:>10}  {split_text:>10}  {str(gate):>12}"
        )
        failures = health.get("failures") or []
        warnings = health.get("warnings") or []
        if failures:
            print(" " * 32 + "fail: " + "; ".join(failures))
        if warnings:
            print(" " * 32 + "warn: " + "; ".join(warnings))
    print("")
    print(f"status_json={STATUS_JSON}")
    print(f"logs_dir={LOG_DIR}")


def command_plan(args: argparse.Namespace) -> int:
    tasks = build_plan(args)
    for task in tasks:
        print(f"# {task.run_name} [{task.resource}]")
        print(" ".join(task.command))
    return 0


def command_status(args: argparse.Namespace) -> int:
    tasks = build_plan(args)
    summary = summarize(tasks)
    write_status(summary)
    if args.json:
        print(json.dumps(jsonable(summary), indent=2, sort_keys=True))
    else:
        print_status(summary)
    return 0 if summary["overall_state"] != "blocked" else 2


def command_tick(args: argparse.Namespace) -> int:
    tasks = build_plan(args)
    with TickLock(LOCK_PATH):
        summary = summarize(tasks)
        launches = select_launches(tasks, summary)
        actions: list[dict[str, Any]] = []
        for task in launches:
            if args.dry_run:
                actions.append({"action": "would_launch", "run": task.run_name, "command": list(task.command)})
            else:
                record = launch_task(task)
                actions.append({"action": "launched", "run": task.run_name, "pid": record["pid"], "log_path": record["log_path"]})
        summary = summarize(tasks)
        summary["tick"] = {"dry_run": bool(args.dry_run), "actions": actions, "ran_at": now_iso()}
        write_status(summary)
    print_status(summary)
    if actions:
        print("Actions:")
        for action in actions:
            print(json.dumps(jsonable(action), sort_keys=True))
    elif summary["overall_state"] == "blocked":
        print("No launch: current phase is blocked. Inspect failures above and the relevant log.")
    elif summary["overall_state"] == "complete":
        print("No launch: selected Stage 4 plan is complete.")
    else:
        print("No launch: resources are busy or current phase is waiting for prerequisite runs.")
    return 0 if summary["overall_state"] != "blocked" else 2


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--frames", default="64,128,255", help="Comma-separated progressive frame counts.")
    parser.add_argument("--max-frame", type=int, default=None, help="Only include phases up to this frame count.")
    parser.add_argument("--coord-aspect-mode", choices=["preserve", "stretch"], default="preserve")
    parser.add_argument("--inr-device", default="cuda:1", help="Device for SIREN and WIRE.")
    parser.add_argument("--deepinv-device", default="cuda:0", help="Device for DeepInv-DIP.")
    parser.add_argument("--batch-k", type=int, default=None, help="Optional neural-method batch_k override.")
    parser.add_argument("--map-tv-max-iter", type=int, default=50)
    parser.add_argument("--map-tv-workers", type=int, default=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remote-friendly EP08 Stage 4 progressive training controller."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="Print the exact training commands without launching.")
    add_common_args(plan_parser)
    plan_parser.set_defaults(func=command_plan)

    status_parser = subparsers.add_parser("status", help="Inspect run state and numeric health gates.")
    add_common_args(status_parser)
    status_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    status_parser.set_defaults(func=command_status)

    tick_parser = subparsers.add_parser(
        "tick",
        help="Advance the current phase by launching eligible pending jobs, then exit.",
    )
    add_common_args(tick_parser)
    tick_parser.add_argument("--dry-run", action="store_true", help="Show launches that would happen.")
    tick_parser.set_defaults(func=command_tick)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "HealthPolicy",
    "Task",
    "assess_metrics",
    "build_parser",
    "build_plan",
    "parse_args",
    "summarize",
]
