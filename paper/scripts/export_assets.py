#!/usr/bin/env python3
"""Export selected experiment artifacts into the paper workspace.

The script copies compact, stable figures and CSV tables from ``output/`` into
``paper/figures`` and ``paper/tables``. It deliberately does not rewrite the
source outputs and can be rerun after notebooks or sweeps are regenerated.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = PROJECT_ROOT / "paper"
FIGURE_DIR = PAPER_ROOT / "figures"
TABLE_DIR = PAPER_ROOT / "tables"


FIGURES = {
    "ep01_session_detection.png": "output/ep01_data_processing/session_detection_a.png",
    "ep01_raster_trajectory.png": "output/ep01_data_processing/acquisition_raster_trajectory.png",
    "ep02_data_driven_alignment.png": "output/ep02_displacement_calibration/ep02_data_driven_alignment_comparison.png",
    "ep03_mtf_snr_recoverability.png": "output/ep03_theoretical_limits/mtf_snr_recoverability_heatmap.png",
    "ep04_gate_recommendations.png": "output/ep04_global_validation/ep06_gate_recommendations.png",
    "ep05_alignment_methods.png": "output/ep05_alignment_sr_capacity/alignment_method_comparison.png",
    "ep05_alignment_tuning_heatmap.png": "output/ep05_alignment_tuning_study/tuning_heatmap_heldout_chamfer.png",
    "ep05_tuned_alignment_comparison.png": "output/ep05_alignment_tuning_study/candidate_alignment_comparison.png",
    "ep06_main_comparison.png": "output/ep06_sr_poc/comparison_fullview.png",
    "ep06_raw_control.png": "output/ep06_sr_poc/comparison_control_track.png",
    "ep06_center_raw_temperature.png": "output/ep06_sr_poc/comparison_center_raw_temperature.png",
    "ep06_alignment_ablation.png": "output/ep06_alignment_ablation/strategy_split_half_nrmse.png",
    "ep06_alignment_artifact.png": "output/ep06_alignment_ablation/strategy_gradient_artifact.png",
    "ep06_sweep_metrics.png": "output/ep06_sr_poc_data_driven_align_sweep/summary/sweep_metric_bars.png",
    "ep06_sweep_lambda.png": "output/ep06_sr_poc_data_driven_align_sweep/summary/sweep_map_tv_lambda_selection.png",
    "ep06_sweep_delta.png": "output/ep06_sr_poc_data_driven_align_sweep/summary/sweep_delta_vs_baseline.png",
}


TABLES = {
    "ep01_sr_data_basis_summary.csv": "output/ep01_data_processing/sr_data_basis_summary.csv",
    "ep02_alignment_evidence_decision_table.csv": "output/ep02_displacement_calibration/ep02_alignment_evidence_decision_table.csv",
    "ep03_mtf_snr_recoverability_gate_summary.csv": "output/ep03_theoretical_limits/mtf_snr_recoverability_gate_summary.csv",
    "ep04_ep06_gate_recommendation_summary.csv": "output/ep04_global_validation/ep06_gate_recommendation_summary.csv",
    "ep05_alignment_method_summary.csv": "output/ep05_alignment_sr_capacity/alignment_method_summary.csv",
    "ep05_tuning_summary.csv": "output/ep05_alignment_tuning_study/tuning_summary.csv",
    "ep05_full_candidate_eval93_summary.csv": "output/ep05_alignment_tuning/full_candidate_eval93_summary.csv",
    "ep06_evaluation_summary.csv": "output/ep06_sr_poc/evaluation_summary.csv",
    "ep06_alignment_strategy_metrics.csv": "output/ep06_alignment_ablation/strategy_metrics.csv",
    "ep06_sweep_method_metrics.csv": "output/ep06_sr_poc_data_driven_align_sweep/summary/sweep_method_metrics.csv",
    "ep06_sweep_map_tv_lambda.csv": "output/ep06_sr_poc_data_driven_align_sweep/summary/sweep_map_tv_lambda.csv",
    "ep06_sweep_delta_vs_baseline.csv": "output/ep06_sr_poc_data_driven_align_sweep/summary/sweep_delta_vs_baseline.csv",
}


def copy_manifest(manifest: dict[str, str], target_dir: Path) -> list[dict[str, str]]:
    target_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for dest_name, src_rel in manifest.items():
        src = PROJECT_ROOT / src_rel
        dest = target_dir / dest_name
        row = {"target": str(dest.relative_to(PAPER_ROOT)), "source": src_rel}
        if src.exists():
            shutil.copy2(src, dest)
            row["status"] = "copied"
        else:
            row["status"] = "missing"
        rows.append(row)
    return rows


def main() -> None:
    figure_rows = copy_manifest(FIGURES, FIGURE_DIR)
    table_rows = copy_manifest(TABLES, TABLE_DIR)
    summary = {
        "figures": figure_rows,
        "tables": table_rows,
        "n_figures_copied": sum(row["status"] == "copied" for row in figure_rows),
        "n_tables_copied": sum(row["status"] == "copied" for row in table_rows),
    }
    manifest_path = PAPER_ROOT / "asset_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Figures copied: {summary['n_figures_copied']} / {len(figure_rows)}")
    print(f"Tables copied: {summary['n_tables_copied']} / {len(table_rows)}")
    missing = [row for row in figure_rows + table_rows if row["status"] != "copied"]
    if missing:
        print("Missing assets:")
        for row in missing:
            print(f"  - {row['source']}")
    print(f"Manifest: {manifest_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

