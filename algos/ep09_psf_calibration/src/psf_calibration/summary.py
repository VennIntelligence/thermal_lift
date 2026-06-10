"""EP09 result aggregation, global config update, and report rendering."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import CONFIG_PATH, OUTPUT_DIR, PROJECT_ROOT, REPORT_DIR, ensure_dir, read_json, relative, write_json


def _safe_read(path: Path) -> dict[str, Any] | None:
    return read_json(path) if path.exists() else None


def _ci_width(ci: list[float] | None) -> float:
    if not ci or len(ci) != 2 or not np.all(np.isfinite(ci)):
        return float("nan")
    return float(ci[1] - ci[0])


def _route_records(forward: dict | None, esf: dict | None, joint: dict | None) -> list[dict[str, Any]]:
    records = []
    if forward:
        records.append(
            {
                "route": "A_forward",
                "sigma_lr_px": float(forward["sigma_forward_lr_px"]),
                "ci95_lr_px": forward.get("ci95_lr_px"),
                "n": int(forward.get("n_frames", 0)),
                "status": "primary",
            }
        )
    if esf:
        records.append(
            {
                "route": "B_esf",
                "sigma_lr_px": float(esf["sigma_esf_median_lr_px"]),
                "ci95_lr_px": esf.get("bootstrap_ci95_lr_px"),
                "n": int(esf.get("n_valid", 0)),
                "status": "cross_check",
            }
        )
    if joint:
        records.append(
            {
                "route": "C_joint",
                "sigma_lr_px": float(joint["sigma_joint_lr_px"]),
                "ci95_lr_px": None,
                "n": int(joint.get("n_val", 0)),
                "status": "cross_check",
            }
        )
    return records


def _verdict_from_sigma(sigma: float, consistent: bool) -> tuple[str, str]:
    if not consistent:
        return (
            "not_cleared_inconsistent_routes",
            "4x is not cleared because the independent calibration routes disagree beyond the EP09 gate.",
        )
    if sigma <= 0.25:
        return (
            "window_open_for_4x_pnp",
            "sigma <= 0.25 LR px, so a constrained 4x PnP experiment is worth attempting.",
        )
    if sigma >= 0.35:
        return (
            "focus_2x_4x_not_physical",
            "sigma >= 0.35 LR px, so 4x contour recovery is not physically supported by this calibration.",
        )
    return (
        "gray_zone_require_risk_limited_4x",
        "sigma lies between 0.25 and 0.35 LR px; 4x should only be treated as a risk-limited diagnostic.",
    )


def summarize_calibration(
    *,
    output_dir: str | Path = OUTPUT_DIR,
    report_dir: str | Path = REPORT_DIR,
    config_path: str | Path = CONFIG_PATH,
    consistency_tolerance: float = 0.05,
    ci_width_gate: float = 0.10,
    residual_depth_gate: float = 1e-3,
) -> dict[str, Any]:
    """Aggregate route outputs and write summary/config/report artifacts."""

    out_dir = ensure_dir(output_dir)
    rep_dir = ensure_dir(report_dir)
    forward = _safe_read(out_dir / "sigma_forward.json")
    esf = _safe_read(out_dir / "sigma_esf.json")
    joint = _safe_read(out_dir / "sigma_joint.json")
    routes = _route_records(forward, esf, joint)
    if not routes:
        raise FileNotFoundError(f"No route JSON files found in {out_dir}")
    route_values = np.asarray([r["sigma_lr_px"] for r in routes if np.isfinite(r["sigma_lr_px"])], dtype=float)
    if route_values.size == 0:
        raise ValueError("No finite route sigma estimates were available")

    # Route A is the primary physical forward-model estimate. Use it as the
    # final value when present, with route spread reported explicitly.
    final_sigma = float(forward["sigma_forward_lr_px"]) if forward else float(np.median(route_values))
    route_spread = float(np.max(route_values) - np.min(route_values)) if len(route_values) > 1 else 0.0
    consistent = bool(route_spread <= float(consistency_tolerance))
    forward_ci = forward.get("ci95_lr_px") if forward else None
    final_ci = forward_ci if forward_ci else [float("nan"), float("nan")]
    ci_width = _ci_width(final_ci)
    ci_gate_pass = bool(np.isfinite(ci_width) and ci_width <= float(ci_width_gate))
    esf_gate_pass = bool(esf and int(esf.get("n_valid", 0)) >= 10)
    forward_diag = forward.get("fine_diagnostic", {}) if forward else {}
    forward_single_min = bool(
        forward
        and forward_diag.get("monotone_to_minimum", False)
        and not forward_diag.get("minimum_at_grid_edge", True)
        and float(forward_diag.get("relative_depth_vs_best_edge", 0.0)) >= float(residual_depth_gate)
    )
    verdict_key, verdict_text = _verdict_from_sigma(final_sigma, consistent)
    completed_at = dt.datetime.now(dt.timezone.utc).isoformat()

    summary = {
        "episode": "EP09_psf_calibration",
        "completed_at_utc": completed_at,
        "final_sigma_lr_px": final_sigma,
        "final_sigma_hr_px_at_2x": final_sigma * 2.0,
        "final_ci95_lr_px": final_ci,
        "final_ci95_width_lr_px": ci_width,
        "route_spread_lr_px": route_spread,
        "routes_consistent_within_0p05_px": consistent,
        "route_consistency_tolerance_lr_px": float(consistency_tolerance),
        "ci_width_gate_pass": ci_gate_pass,
        "forward_curve_single_minimum_gate_pass": forward_single_min,
        "forward_residual_depth_gate": float(residual_depth_gate),
        "esf_valid_segment_gate_pass": esf_gate_pass,
        "overall_gate_pass": bool(consistent and ci_gate_pass and forward_single_min),
        "four_x_verdict": verdict_key,
        "four_x_verdict_text": verdict_text,
        "routes": routes,
        "source_files": {
            "forward": relative(out_dir / "sigma_forward.json") if forward else None,
            "esf": relative(out_dir / "sigma_esf.json") if esf else None,
            "joint": relative(out_dir / "sigma_joint.json") if joint else None,
        },
    }
    write_json(out_dir / "calibration_summary.json", summary)

    config = {
        "psf_sigma_lr_px": final_sigma,
        "psf_sigma_hr_px_at_2x": final_sigma * 2.0,
        "confidence_interval_95_lr_px": final_ci,
        "calibration_episode": "EP09_psf_calibration",
        "status": "validated" if summary["overall_gate_pass"] else "provisional_needs_review",
        "four_x_verdict": verdict_key,
        "four_x_verdict_text": verdict_text,
        "updated_at_utc": completed_at,
        "summary_file": relative(out_dir / "calibration_summary.json"),
        "notes": [
            "Sigma is reported in LR detector pixels unless a field name says HR.",
            "Route A is the primary estimate; routes B/C are cross-checks and gate diagnostics.",
        ],
    }
    write_json(config_path, config)
    _write_verdict_markdown(out_dir / "4x_feasibility_verdict.md", summary)
    _write_report_markdown(rep_dir / "psf_calibration_report.md", summary, forward, esf, joint)
    _write_route_table(out_dir / "route_sigma_summary.csv", routes)
    return summary


def _write_route_table(path: Path, routes: list[dict[str, Any]]) -> None:
    table = pd.DataFrame(
        [
            {
                "route": r["route"],
                "sigma_lr_px": r["sigma_lr_px"],
                "ci95_low_lr_px": r["ci95_lr_px"][0] if r.get("ci95_lr_px") else np.nan,
                "ci95_high_lr_px": r["ci95_lr_px"][1] if r.get("ci95_lr_px") else np.nan,
                "n": r["n"],
                "status": r["status"],
            }
            for r in routes
        ]
    )
    table.to_csv(path, index=False)


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(number):
        return "NaN"
    return f"{number:.{digits}f}"


def _write_verdict_markdown(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(
        "\n".join(
            [
                "# EP09 4x Feasibility Verdict",
                "",
                f"- Final sigma: **{_fmt(summary['final_sigma_lr_px'], 3)} LR px** "
                f"({_fmt(summary['final_sigma_hr_px_at_2x'], 3)} HR px on the 2x grid)",
                f"- 95% CI: **[{_fmt(summary['final_ci95_lr_px'][0], 3)}, {_fmt(summary['final_ci95_lr_px'][1], 3)}] LR px**",
                f"- Route spread: **{_fmt(summary['route_spread_lr_px'], 3)} LR px**",
                f"- Gate status: **{'PASS' if summary['overall_gate_pass'] else 'REVIEW'}**",
                f"- Verdict: **{summary['four_x_verdict']}**",
                "",
                summary["four_x_verdict_text"],
                "",
                "Interpretation: this is a physics/forward-model gate, not a claim that any 4x reconstruction is already valid.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _route_md(routes: list[dict[str, Any]]) -> str:
    lines = ["| Route | Sigma LR px | 95% CI LR px | N | Role |", "|---|---:|---:|---:|---|"]
    for route in routes:
        ci = route.get("ci95_lr_px")
        ci_text = f"[{_fmt(ci[0], 3)}, {_fmt(ci[1], 3)}]" if ci else "n/a"
        lines.append(
            f"| {route['route']} | {_fmt(route['sigma_lr_px'], 3)} | {ci_text} | {int(route['n'])} | {route['status']} |"
        )
    return "\n".join(lines)


def _write_report_markdown(
    path: Path,
    summary: dict[str, Any],
    forward: dict[str, Any] | None,
    esf: dict[str, Any] | None,
    joint: dict[str, Any] | None,
) -> None:
    forward_diag = forward.get("fine_diagnostic", {}) if forward else {}
    forward_n = int(forward.get("n_frames", 0)) if forward else 0
    forward_hr_input = forward.get("hr_input", "missing") if forward else "missing"
    lines = [
        "# EP09 — PSF Sigma Calibration Report",
        "",
        "## Executive Summary",
        "",
        f"EP09 estimates the Gaussian PSF sigma as **{_fmt(summary['final_sigma_lr_px'], 3)} LR px** "
        f"with route spread **{_fmt(summary['route_spread_lr_px'], 3)} LR px**. "
        f"The 4x gate verdict is **{summary['four_x_verdict']}**.",
        "",
        _route_md(summary["routes"]),
        "",
        "## Gate Results",
        "",
        f"- Route consistency within 0.05 px: **{summary['routes_consistent_within_0p05_px']}**",
        f"- Forward residual single-minimum gate: **{summary['forward_curve_single_minimum_gate_pass']}**",
        f"- 95% CI width <= 0.10 px: **{summary['ci_width_gate_pass']}** "
        f"(width={_fmt(summary['final_ci95_width_lr_px'], 3)} px)",
        f"- ESF valid segment count >= 10: **{summary['esf_valid_segment_gate_pass']}**",
        f"- Overall gate: **{summary['overall_gate_pass']}**",
        "",
        "## Route A: Forward-Model Residual",
        "",
        "Route A uses the EP06 MAP-TV 2x highpass reconstruction as a pseudo-HR scene and sweeps sigma in the EP06 forward model. "
        f"The score is the cropped LR highpass residual against the {forward_n} clean main-session observations loaded from the EP06 SR metadata filter, "
        "split by acquisition order into train and validation frames.",
        "",
        f"- Sigma: **{_fmt(forward['sigma_forward_lr_px'], 4) if forward else 'missing'} LR px**",
        f"- EP06 pseudo-HR input: `{forward_hr_input}`",
        f"- Train/validation sigma delta: **{_fmt(forward.get('train_val_abs_delta_lr_px', np.nan) if forward else np.nan, 4)} px**",
        f"- Relative residual depth vs best edge: **{_fmt(forward_diag.get('relative_depth_vs_best_edge', np.nan), 4)}**",
        f"- Minimum at grid edge: **{forward_diag.get('minimum_at_grid_edge', 'missing')}**",
        "",
        "## Route B: 1D ESF Fitting",
        "",
        "Route B fits an error-function edge model on EP04 outer contour anchors after contrast/projection/R2 gates. "
        "It is an independent check, but it can include true thermal edge width in addition to the optical PSF.",
        "",
        f"- Median sigma: **{_fmt(esf['sigma_esf_median_lr_px'], 4) if esf else 'missing'} LR px**",
        f"- Valid segments: **{int(esf.get('n_valid', 0)) if esf else 0}**",
        "",
        "## Route C: Joint MAP-TV Hold-Out Sweep",
        "",
        "Route C reconstructs a short-budget MAP-TV HR image for each candidate sigma on a deterministic frame subset, then scores held-out frames. "
        "It is intentionally lower budget than EP06 and is used as a cross-check rather than the primary estimate.",
        "",
        f"- Joint sigma: **{_fmt(joint['sigma_joint_lr_px'], 4) if joint else 'missing'} LR px**",
        f"- Grid minimum: **{_fmt(joint.get('sigma_joint_grid_min_lr_px', np.nan) if joint else np.nan, 4)} LR px**",
        "",
        "## 4x Decision",
        "",
        summary["four_x_verdict_text"],
        "",
        "The decision is conservative: 4x should not be promoted unless the calibrated sigma, route consistency, and confidence interval all clear the gates.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
