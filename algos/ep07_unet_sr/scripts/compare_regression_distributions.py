#!/usr/bin/env python3
"""Compare solver_regression_suite metric distributions across good/bad cases.

Stage 0d follow-up (ACL-046): the suite's hand-picked thresholds failed known-good
checkpoints on 3/4 probes, so pass/fail cannot gate training yet.  This tool takes
several ``--output-json`` files from ``solver_regression_suite.py``, extracts one
scalar per probe metric, and reports per-metric good/bad separability plus a
suggested threshold (geometric midpoint) where the classes separate.  It only
REPORTS; threshold changes remain an owner decision.

Usage:
    python scripts/compare_regression_distributions.py \
        --case good:v11_40k:output/.../goodcase_v11_40k.json \
        --case bad:v8_k4:output/.../badcase_v8k4.json \
        --output-csv output/.../regression_metric_distributions.csv
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

# probe -> list of (metric_label, extractor_path, threshold_key)
# extractor_path items starting with "candidates." are resolved inside the
# selected candidate's dict; all metrics are "higher is worse" in suite v1.
METRIC_SPECS: dict[str, list[tuple[str, str, str]]] = {
    "flat_roi_artifact": [
        ("lowpass_p95_abs_delta_c", "candidates.lowpass_p95_abs_delta_c", "flat_lowpass_p95_delta_c"),
        ("lowpass_std_delta_c", "candidates.lowpass_std_delta_c", "flat_lowpass_std_delta_c"),
    ],
    "tiled_full_halo_extent_consistency": [
        ("nrmse_vs_tiled_std", "nrmse_vs_tiled_std", "extent_nrmse"),
        ("p95_abs_diff_c", "p95_abs_diff_c", "extent_p95_abs_diff_c"),
    ],
    "seam_spectrum": [
        ("max_abs_autocorr", "candidates.max_abs_autocorr", "seam_autocorr"),
    ],
    "beading_probe": [
        ("edge_signal_ratio_vs_reference", "candidates.edge_signal_ratio_vs_reference", "bead_edge_ratio"),
        ("edge_excess_p95_c", "candidates.edge_excess_p95_c", "bead_excess_p95_c"),
    ],
}


def parse_case(spec: str) -> tuple[str, str, Path]:
    parts = str(spec).split(":", 2)
    if len(parts) != 3 or parts[0] not in ("good", "bad"):
        raise argparse.ArgumentTypeError(f"--case must be good|bad:name:path.json, got {spec!r}")
    return parts[0], parts[1], Path(parts[2])


def extract_metric(case: dict, probe: str, path: str, candidate: str) -> float:
    for entry in case.get("cases", []):
        if entry.get("name") != probe:
            continue
        metrics = entry.get("metrics", {})
        if path.startswith("candidates."):
            leaf = path.split(".", 1)[1]
            cand = metrics.get("candidates", {}).get(candidate, {})
            value = cand.get(leaf)
        else:
            value = metrics.get(path)
        return float(value) if value is not None else math.nan
    return math.nan


def extract_threshold(case: dict, probe: str, threshold_key: str) -> float:
    for entry in case.get("cases", []):
        if entry.get("name") == probe:
            value = entry.get("thresholds", {}).get(threshold_key)
            return float(value) if value is not None else math.nan
    return math.nan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", required=True, type=parse_case,
                        help="good|bad:name:path.json (repeatable)")
    parser.add_argument("--candidate", default="full_halo96",
                        help="which per-candidate block to read (default: full_halo96, the E3 mainline inference mode)")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-separability-csv", type=Path, default=None)
    args = parser.parse_args()

    classes = {cls for cls, _, _ in args.case}
    rows: list[dict] = []
    for cls, name, path in args.case:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for probe, specs in METRIC_SPECS.items():
            for metric_label, metric_path, threshold_key in specs:
                value = extract_metric(payload, probe, metric_path, args.candidate)
                threshold = extract_threshold(payload, probe, threshold_key)
                rows.append(
                    {
                        "case": name,
                        "class": cls,
                        "probe": probe,
                        "metric": metric_label,
                        "value": value,
                        "current_threshold": threshold,
                        "exceeds_current_threshold": bool(value > threshold)
                        if math.isfinite(value) and math.isfinite(threshold)
                        else None,
                    }
                )
    long_df = pd.DataFrame(rows)

    sep_rows: list[dict] = []
    for (probe, metric), part in long_df.groupby(["probe", "metric"], sort=False):
        good = part[part["class"].eq("good")]["value"].dropna()
        bad = part[part["class"].eq("bad")]["value"].dropna()
        worst_good = float(good.max()) if len(good) else math.nan
        best_bad = float(bad.min()) if len(bad) else math.nan
        separable = (
            math.isfinite(worst_good) and math.isfinite(best_bad) and worst_good < best_bad
        )
        if separable and worst_good > 0 and best_bad > 0:
            suggested = math.sqrt(worst_good * best_bad)
        elif separable:
            suggested = 0.5 * (worst_good + best_bad)
        else:
            suggested = math.nan
        sep_rows.append(
            {
                "probe": probe,
                "metric": metric,
                "n_good": int(len(good)),
                "n_bad": int(len(bad)),
                "good_min": float(good.min()) if len(good) else math.nan,
                "good_max": worst_good,
                "bad_min": best_bad,
                "bad_max": float(bad.max()) if len(bad) else math.nan,
                "current_threshold": float(part["current_threshold"].dropna().iloc[0])
                if part["current_threshold"].notna().any()
                else math.nan,
                "separable_good_below_bad": separable,
                "separation_ratio_badmin_over_goodmax": (best_bad / worst_good)
                if math.isfinite(worst_good) and math.isfinite(best_bad) and worst_good > 0
                else math.nan,
                "suggested_threshold_geomean": suggested,
            }
        )
    sep_df = pd.DataFrame(sep_rows)

    print("== per-case metric values ==")
    print(long_df.to_string(index=False))
    print()
    print("== good/bad separability (owner decides thresholds; this is data only) ==")
    print(sep_df.to_string(index=False))
    if "good" not in classes or "bad" not in classes:
        print("\nWARNING: need at least one good AND one bad case for separability to mean anything.")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        long_df.to_csv(args.output_csv, index=False)
        print(f"\nwrote {args.output_csv}")
    if args.output_separability_csv is not None:
        args.output_separability_csv.parent.mkdir(parents=True, exist_ok=True)
        sep_df.to_csv(args.output_separability_csv, index=False)
        print(f"wrote {args.output_separability_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
