#!/usr/bin/env python3
"""CPU-only EP16 child process for one MAP-TGV reconstruction."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLBACKEND", "Agg")
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
ALGO_ROOT = SCRIPT_DIR.parents[0]
PROJECT_ROOT = SCRIPT_DIR.parents[2]
for _path in (
    PROJECT_ROOT / "algos" / "ep10_tgv_sr" / "src",
    PROJECT_ROOT / "algos" / "ep06_sr_poc" / "src",
    PROJECT_ROOT / "core" / "src",
):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from ep10_tgv_sr import get_tgv_backend_provenance, reconstruct_map_tgv  # noqa: E402


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    started = time.perf_counter()
    result_path = Path(spec["result_json"])
    try:
        indices = np.asarray(spec["indices"], dtype=np.int64)
        hp_frames = np.load(spec["hp_frames_npy"], mmap_mode="r")
        shifts = np.load(spec["shifts_npy"]).astype(np.float32, copy=False)
        frames = np.asarray(hp_frames[indices], dtype=np.float32)
        if shifts.shape != (len(indices), 2):
            raise ValueError(f"shifts shape {shifts.shape} does not match {len(indices)} indices")

        hr, records = reconstruct_map_tgv(
            frames,
            shifts,
            lambda_tv=float(spec["lambda_tv"]),
            psf_sigma=float(spec["psf_sigma"]),
            alpha_ratio=float(spec["alpha_ratio"]),
            max_iter=int(spec["max_iter"]),
            tgv_inner_iter=int(spec["tgv_inner_iter"]),
            aniso_ratio_y=float(spec["aniso_ratio_y"]),
            coverage_weighted=bool(spec["coverage_weighted"]),
            tgv_device="cpu",
            workers=int(spec["workers"]),
            scale=2,
        )
        hr = np.asarray(hr, dtype=np.float32)
        hr_path = Path(spec["hr_npy"])
        conv_path = Path(spec["convergence_csv"])
        hr_path.parent.mkdir(parents=True, exist_ok=True)
        conv_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(hr_path, hr)
        pd.DataFrame.from_records(records).to_csv(conv_path, index=False)
        backend = get_tgv_backend_provenance()
        _write_json(
            result_path,
            {
                "run_id": spec["run_id"],
                "status": "success",
                "runtime_sec": float(time.perf_counter() - started),
                "hr_npy": str(hr_path),
                "convergence_csv": str(conv_path),
                "iterations": int(len(records)),
                "tgv_backend": backend,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            },
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - child must record and continue.
        _write_json(
            result_path,
            {
                "run_id": spec.get("run_id", args.spec.stem),
                "status": "failed",
                "runtime_sec": float(time.perf_counter() - started),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

