"""Collect promoted episode figures into output/paper_figures with stable S-ids.

EP01-EP15 episode pipelines already produce several CVPR-styled figures that
directly support the supplementary narrative (audit chain, alignment gates,
theta verification, kinematics, MAP-TV anchor evidence, MTF/SNR bounds).
This script copies the promoted set into the canonical paper-figure directory
under stable ``figSxx_*`` names and writes a provenance manifest, so LaTeX can
include everything from one directory while each asset remains rebuildable by
its episode pipeline (see docs/paper/09_figures_tables_assets.md).

Run from the repository root (after the episode pipelines / notebooks):
    uv run python scripts/paper_figures/collect_promoted_supp.py

历史定位: scripts/paper_figures/ 是 2026-06 时代的旧论文图脚本；现行权威图集见
docs/publication_figures/（每图一个脚本、自带规范）。
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "output" / "paper_figures"

# figS name -> episode-canonical source (relative to repo root).
PROMOTED: dict[str, str] = {
    # S-F7: alignment chain + gate (supp B.2)
    "figS07a_alignment_chain.png": "output/ep05_alignment_sr_capacity/alignment_method_comparison.png",
    "figS07b_phase_bins.png": "output/ep05_alignment_sr_capacity/phase_bin_coverage_2x.png",
    "figS07c_gate_map.png": "output/ep04_global_validation/pass_fail_contour_map.png",
    # S-F11: data audit chain (supp B.1)
    "figS11a_order_comparison.png": "output/ep01_data_processing/order_comparison.png",
    "figS11b_raster_acquisition.png": "output/ep01_data_processing/session_detection_a.png",
    # S-F12: AVI theta forest plot (supp A.5.2)
    "figS12_theta_forest.png": "output/ep02_displacement_calibration/avi_theta_forest_plot.png",
    # S-F13: measured cumulative trajectory (supp B.3)
    "figS13_cumulative_trajectory.png": "output/ep05_sr_reassessment/main_session_cumulative_trajectory.png",
    # S-F14: MAP-TV anchor structural evidence (supp D.2.3)
    "figS14a_zigzag_profiles.png": "output/ep15_info_limit/m4_deconv_anchor/zigzag_profiles.png",
    "figS14b_four_arm_highpass.png": "output/ep15_info_limit/m4_deconv_anchor/four_arm_highpass.png",
    # S-F15: MTF / effective-SNR bounds (supp A.1)
    "figS15a_mtf_snr_heatmap.png": "output/ep03_theoretical_limits/mtf_snr_recoverability_heatmap.png",
    "figS15b_mtf_frequency_response.png": "output/ep03_theoretical_limits/mtf_psf_frequency_response.png",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, str]] = {}
    missing: list[str] = []
    for dst_name, src_rel in PROMOTED.items():
        src = PROJECT_ROOT / src_rel
        if not src.exists():
            missing.append(src_rel)
            continue
        dst = OUT_DIR / dst_name
        shutil.copy2(src, dst)
        manifest[dst_name] = {
            "source": src_rel,
            "source_mtime": datetime.fromtimestamp(src.stat().st_mtime).isoformat(timespec="seconds"),
        }
        print(f"collected {dst_name}  <-  {src_rel}")

    manifest_path = OUT_DIR / "promoted_supp_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"manifest: {manifest_path} ({len(manifest)} assets)")

    if missing:
        raise SystemExit(
            "missing sources (run the episode pipelines first):\n  " + "\n  ".join(missing)
        )


if __name__ == "__main__":
    main()
