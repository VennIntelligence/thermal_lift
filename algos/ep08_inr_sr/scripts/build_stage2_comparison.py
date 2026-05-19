#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from thermal_core.plotting import savefig_academic, setup_academic_style

EP08_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EP08_ROOT.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "ep08_inr_sr"
BASELINE_CONFIG = EP08_ROOT / "configs" / "ep06_baseline_metrics.json"
EP06_PATCH_METRICS = OUTPUT_DIR / "ep06_patch_baseline" / "metrics.json"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _metric(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    return value_f if np.isfinite(value_f) else None


def _method_row(method: str, family: str, source_dir: str, stage_gate_key: str = "stage1_gate") -> dict[str, Any]:
    source = OUTPUT_DIR / source_dir / "metrics.json"
    payload = _read_json(source)
    return {
        "method": method,
        "family": family,
        "holdout_residual": _metric(payload, "holdout_residual"),
        "split_half_nrmse": _metric(payload, "split_half_nrmse"),
        "artifact_score": _metric(payload, "artifact_score"),
        "raw_control_agreement": _metric(payload, "raw_control_agreement"),
        "p95_gradient": _metric(payload, "p95_gradient"),
        "best_step": payload.get("best_step"),
        "stage_gate": payload.get("stage_gate", payload.get(stage_gate_key, "missing" if not source.exists() else "unknown")),
        "source_type": "ep08_patch_protocol",
        "unavailable_reason": "" if source.exists() else f"missing {source.relative_to(PROJECT_ROOT)}",
        "source": str(source.relative_to(PROJECT_ROOT)),
    }


def _ep06_baseline_row() -> dict[str, Any]:
    if EP06_PATCH_METRICS.exists():
        payload = _read_json(EP06_PATCH_METRICS)
        return {
            "method": "EP06 MAP-TV",
            "family": "classic_opt",
            "holdout_residual": _metric(payload, "holdout_residual"),
            "split_half_nrmse": _metric(payload, "split_half_nrmse"),
            "artifact_score": _metric(payload, "artifact_score"),
            "raw_control_agreement": _metric(payload, "raw_control_agreement"),
            "p95_gradient": _metric(payload, "p95_gradient"),
            "best_step": payload.get("best_step"),
            "stage_gate": payload.get("stage_gate", "unknown"),
            "source_type": "ep08_patch_protocol",
            "unavailable_reason": "",
            "source": str(EP06_PATCH_METRICS.relative_to(PROJECT_ROOT)),
        }

    baseline = _read_json(BASELINE_CONFIG)
    map_tv = baseline.get("map_tv", {})
    source = baseline.get("source", {})
    reason = (
        "EP08 patch MAP-TV metrics are absent; using legacy EP06 full-frame proxy. "
        "holdout_residual and split_half_nrmse remain unavailable because the proxy did not use the EP08 seed=42 split."
    )
    return {
        "method": "EP06 MAP-TV",
        "family": "classic_opt",
        "holdout_residual": _metric(map_tv, "holdout_residual"),
        "split_half_nrmse": _metric(map_tv, "split_half_nrmse"),
        "artifact_score": _metric(map_tv, "artifact_score"),
        "raw_control_agreement": _metric(map_tv, "raw_control_agreement"),
        "p95_gradient": _metric(map_tv, "p95_gradient"),
        "best_step": None,
        "stage_gate": source.get("status", "proxy_only_patch_metrics_missing"),
        "source_type": "ep06_fullframe_proxy",
        "unavailable_reason": reason,
        "source": str(BASELINE_CONFIG.relative_to(PROJECT_ROOT)),
    }


def build_rows() -> list[dict[str, Any]]:
    return [
        _method_row("SIREN", "inr_sine", "siren_stage1"),
        _method_row("WIRE", "inr_gabor", "wire_stage1"),
        _method_row("Deep Decoder", "cnn_decoder", "deep_decoder_stage2"),
        _method_row("DeepInverse-DIP", "cnn_dip_reference", "deepinv_dip_stage2", stage_gate_key="stage_gate"),
        _ep06_baseline_row(),
    ]


def save_table(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(rows: list[dict[str, Any]], path: Path) -> None:
    setup_academic_style()
    metrics = [
        ("holdout_residual", "Hold-out residual"),
        ("split_half_nrmse", "Split-half NRMSE"),
        ("artifact_score", "Artifact score"),
        ("raw_control_agreement", "Raw-control agreement"),
        ("p95_gradient", "P95 gradient"),
    ]
    colors = ["#4C72B0", "#C44E52", "#55A868", "#8172B2", "#8C8C8C"]
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 4.8))
    methods = [row["method"] for row in rows]
    for ax, (key, label) in zip(axes.ravel(), metrics):
        values = [row.get(key) for row in rows]
        x = np.arange(len(rows))
        numeric = [float(v) if v is not None and np.isfinite(float(v)) else np.nan for v in values]
        ax.bar(x, np.nan_to_num(numeric, nan=0.0), color=colors[: len(rows)])
        for xpos, value in zip(x, numeric, strict=True):
            if np.isnan(value):
                ax.text(xpos, 0.02, "n/a", ha="center", va="bottom", rotation=90, transform=ax.get_xaxis_transform(), fontsize=7)
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=28, ha="right")
        finite = [v for v in numeric if np.isfinite(v) and v > 0]
        if key in {"holdout_residual", "artifact_score", "p95_gradient"} and finite:
            ax.set_yscale("log")
    axes.ravel()[-1].axis("off")
    savefig_academic(fig, path)


def main() -> None:
    rows = build_rows()
    save_table(rows, OUTPUT_DIR / "stage2_comparison.csv")
    save_json = OUTPUT_DIR / "stage2_comparison.json"
    save_json.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True), encoding="utf-8")
    save_figure(rows, OUTPUT_DIR / "stage2_comparison.png")
    print(f"saved {OUTPUT_DIR / 'stage2_comparison.csv'}")
    print(f"saved {save_json}")
    print(f"saved {OUTPUT_DIR / 'stage2_comparison.png'}")


if __name__ == "__main__":
    main()
