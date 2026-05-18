"""EP05 notebook helpers for SR capacity and alignment baseline outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


METHOD_ORDER = [
    "no_alignment",
    "old_stage_model",
    "filename_affine_fit",
    "data_driven_ncc_init",
    "data_driven_contour_refined",
]

METHOD_LABELS = {
    "no_alignment": "No alignment",
    "old_stage_model": "Stage prior",
    "filename_affine_fit": "Filename affine",
    "data_driven_ncc_init": "NCC init",
    "data_driven_contour_refined": "Contour refined",
}


def load_capacity_outputs(output_dir: Path) -> dict:
    """Load all EP05 capacity-check outputs used by the notebook."""
    output_dir = Path(output_dir)
    with open(output_dir / "alignment_sr_capacity_summary.json", encoding="utf-8") as f:
        summary_json = json.load(f)
    return {
        "summary_json": summary_json,
        "method_summary": pd.read_csv(output_dir / "alignment_method_summary.csv"),
        "holdout_scores": pd.read_csv(output_dir / "alignment_method_holdout_scores.csv"),
        "phase_summary_2x": pd.read_csv(output_dir / "phase_bin_summary_2x.csv"),
        "phase_counts_2x": pd.read_csv(output_dir / "phase_bin_counts_2x.csv"),
        "overlay_density": pd.read_csv(output_dir / "alignment_overlay_density_metrics.csv"),
    }


def ordered_method_table(method_summary: pd.DataFrame) -> pd.DataFrame:
    """Return the alignment comparison table in decision-order, not winner-order."""
    df = method_summary.copy()
    df["method_label"] = df["method"].map(METHOD_LABELS).fillna(df["method"])
    df["method_order"] = df["method"].map({name: i for i, name in enumerate(METHOD_ORDER)})
    cols = [
        "method_label",
        "n_frames",
        "holdout_chamfer_median_px",
        "holdout_chamfer_p90_px",
        "gradient_corr_median",
        "gradient_corr_p10",
        "shift_norm_median_px",
        "shift_norm_p90_px",
    ]
    return df.sort_values("method_order")[cols].reset_index(drop=True)


def phase_capacity_table(phase_summary: pd.DataFrame) -> pd.DataFrame:
    """Return the 2x phase-capacity table with readable method labels."""
    df = phase_summary.copy()
    df["method_label"] = df["method"].map(METHOD_LABELS).fillna(df["method"])
    df["method_order"] = df["method"].map({name: i for i, name in enumerate(METHOD_ORDER)})
    cols = [
        "method_label",
        "n_frames",
        "occupied_bins",
        "bad_bins",
        "total_bins",
        "entropy_fraction",
        "min_count",
        "max_count",
    ]
    return df.sort_values("method_order")[cols].reset_index(drop=True)


def overlay_density_table(overlay_density: pd.DataFrame) -> pd.DataFrame:
    """Return a compact contour-stack density table for the notebook."""
    df = overlay_density.copy()
    df["method_label"] = df["method"].map(METHOD_LABELS).fillna(df["method"])
    cols = [
        "method_label",
        "sampled_frames",
        "density_peak",
        "density_p99",
        "near_reference_edge_mean",
        "off_reference_edge_mean",
        "near_off_ratio",
    ]
    return df[cols].reset_index(drop=True)


def ep06_decision_table() -> pd.DataFrame:
    """Return the EP06 handoff decision table."""
    return pd.DataFrame(
        [
            {
                "item": "EP06 recommended alignment",
                "decision": "Use data-driven NCC init plus contour refinement gate",
                "reason": "Lowest held-out contour Chamfer while preserving a data-constrained local anchor.",
            },
            {
                "item": "Phase prior",
                "decision": "Keep filename affine and NCC init",
                "reason": "Both provide continuous 2x phase coverage and remain useful before local refinement.",
            },
            {
                "item": "Control groups",
                "decision": "No alignment, stage prior, filename affine, NCC init",
                "reason": "These separate display gain from true data-driven alignment gain.",
            },
            {
                "item": "Primary failure risks",
                "decision": "Thermal drift, local contour ambiguity, PSF/SNR ceiling",
                "reason": "These can create visually plausible stacks without stable held-out contour improvement.",
            },
            {
                "item": "Acceptance metrics",
                "decision": "Split-half consistency, held-out contour Chamfer, phase-bin coverage, visual contour gain",
                "reason": "Back-projection residual or Tenengrad alone is not sufficient evidence for SR success.",
            },
        ]
    )
