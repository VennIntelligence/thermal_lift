from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
SRC = ROOT / "src"
EP06_SRC = PROJECT_ROOT / "algos" / "ep06_sr_poc" / "src"
for path in (SRC, EP06_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import data_loader as ep06_data  # noqa: E402
from common import forward_model as ep06_forward  # noqa: E402
from ep08.forward import ForwardOperator, adjoint, forward  # noqa: E402
from ep08.highpass import highpass_preprocess, offset_correction  # noqa: E402
from ep08.splits import build_train_val_split  # noqa: E402
from ep08.utils import save_json  # noqa: E402


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def validate_forward() -> dict[str, object]:
    rng = np.random.default_rng(801)
    rows: list[dict[str, float | str]] = []
    for psf_sigma in (0.0, 0.35, 1.0):
        x_hr = rng.normal(size=(24, 26))
        shift = np.array([0.23, 0.37], dtype=np.float64)
        expected = ep06_forward.forward(x_hr, shift, psf_sigma=psf_sigma)
        actual = forward(
            torch.as_tensor(x_hr, dtype=torch.float64),
            torch.as_tensor(shift, dtype=torch.float64),
            psf_sigma=psf_sigma,
        ).detach().cpu().numpy()
        rows.append({"case": f"forward_psf_{psf_sigma:g}", "max_abs_error": _max_abs(actual, expected)})

    y_lr = rng.normal(size=(11, 13))
    shift = np.array([0.41, 0.19], dtype=np.float64)
    expected_adj = ep06_forward.adjoint(y_lr, shift, psf_sigma=0.45, hr_shape=(22, 26))
    actual_adj = adjoint(
        torch.as_tensor(y_lr, dtype=torch.float64),
        torch.as_tensor(shift, dtype=torch.float64),
        psf_sigma=0.45,
        hr_shape=(22, 26),
    ).detach().cpu().numpy()
    rows.append({"case": "adjoint_psf_0.45", "max_abs_error": _max_abs(actual_adj, expected_adj)})

    shifts = np.array([[0.0, 0.0], [0.15, 0.25], [0.45, 0.1]], dtype=np.float64)
    x_hr = rng.normal(size=(20, 24))
    ep06_op = ep06_forward.build_observation_operator(x_hr.shape, shifts=shifts, psf_sigma=0.25)
    ep08_op = ForwardOperator(x_hr.shape, (10, 12), torch.as_tensor(shifts, dtype=torch.float64), psf_sigma=0.25)
    rows.append(
        {
            "case": "operator_forward_all",
            "max_abs_error": _max_abs(
                ep08_op.forward_all(torch.as_tensor(x_hr, dtype=torch.float64)).detach().cpu().numpy(),
                ep06_op.forward_all(x_hr),
            ),
        }
    )

    max_error = max(float(row["max_abs_error"]) for row in rows)
    return {
        "status": "passed" if max_error < 1e-5 else "failed",
        "max_abs_error": max_error,
        "threshold": 1e-5,
        "source": "EP08 validate_p0.py vs EP06 common.forward_model",
        "cases": rows,
    }


def validate_highpass() -> dict[str, object]:
    rng = np.random.default_rng(802)
    rows: list[dict[str, float | str]] = []

    frames = rng.normal(size=(4, 17, 19)).astype(np.float32)
    expected_hp = ep06_data.highpass_preprocess(frames, sigma_bg=2.0, mode="nearest")
    actual_hp = highpass_preprocess(torch.as_tensor(frames), sigma_bg=2.0, mode="nearest").detach().cpu().numpy()
    rows.append({"case": "highpass_stack", "max_abs_error": _max_abs(actual_hp, expected_hp)})

    expected_offset, expected_offsets = ep06_data.offset_correction(frames, method="median", return_offsets=True)
    actual_offset, actual_offsets = offset_correction(frames, method="median", return_offsets=True)
    rows.append({"case": "offset_corrected_frames", "max_abs_error": _max_abs(actual_offset, expected_offset)})
    rows.append({"case": "offset_values", "max_abs_error": _max_abs(actual_offsets, expected_offsets)})

    max_error = max(float(row["max_abs_error"]) for row in rows)
    return {
        "status": "passed" if max_error < 1e-5 else "failed",
        "max_abs_error": max_error,
        "threshold": 1e-5,
        "source": "EP08 validate_p0.py vs EP06 common.data_loader",
        "cases": rows,
    }


def validate_split() -> dict[str, object]:
    shifts = np.array(
        [[0.01 * idx, 0.17 * idx] for idx in range(48)],
        dtype=np.float64,
    )
    frames = np.zeros((len(shifts), 4, 4), dtype=np.float32)
    first = build_train_val_split(frames, shifts, val_ratio=0.2, seed=42)
    second = build_train_val_split(frames, shifts, val_ratio=0.2, seed=42)
    bit_exact = all(np.array_equal(a, b) for a, b in zip(first, second))
    return {
        "status": "passed" if bit_exact else "failed",
        "bit_exact": bool(bit_exact),
        "n_train": int(len(first[0])),
        "n_val": int(len(first[1])),
        "source": "EP08 build_train_val_split seed=42",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write EP08 P0 validation artifacts.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output" / "ep08_inr_sr")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    forward_payload = validate_forward()
    highpass_payload = validate_highpass()
    split_payload = validate_split()

    save_json(args.output_dir / "forward_validation.json", forward_payload)
    save_json(args.output_dir / "highpass_validation.json", highpass_payload)
    save_json(args.output_dir / "split_validation.json", split_payload)

    pd.DataFrame(forward_payload["cases"]).to_csv(args.output_dir / "forward_validation.csv", index=False)
    pd.DataFrame(highpass_payload["cases"]).to_csv(args.output_dir / "highpass_validation.csv", index=False)
    pd.DataFrame([split_payload]).to_csv(args.output_dir / "split_validation.csv", index=False)

    statuses = [forward_payload["status"], highpass_payload["status"], split_payload["status"]]
    print(f"P0 validation: {statuses}")
    if any(status != "passed" for status in statuses):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
