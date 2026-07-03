"""Feed Stage 0a per-frame shift refinements back as an alignment artifact.

Stage 0f measured the per-frame shift error against ``contour_refined`` as a
zero-mean annulus of |delta| ~ 0.25-0.40 LR px (ACL-046/047).  This module
merges ``stage0a_best_shift_refinements.csv`` into the contour alignment CSV
schema (``refined_align_dx_px/refined_align_dy_px``) so every consumer that
accepts ``--alignment-csv`` can use the refined shifts unchanged.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_refined_alignment(
    *,
    refinements_csv: str | Path,
    base_alignment_csv: str | Path,
    max_initial_mismatch_px: float = 1e-3,
    expected_frames: int | None = 248,
) -> pd.DataFrame:
    """Merge Stage 0a refinements into the contour alignment schema.

    Every refinement row's ``initial_dx/dy_px`` must match the base CSV's
    refined columns (the refinement was computed FROM that alignment); a
    mismatch means the refinements came from a different alignment source and
    the merge would silently corrupt shifts.
    """

    refinements = pd.read_csv(refinements_csv)
    required = {"file", "initial_dx_px", "initial_dy_px", "refined_dx_px", "refined_dy_px"}
    missing = required - set(refinements.columns)
    if missing:
        raise ValueError(f"refinements CSV missing columns: {sorted(missing)}")
    if refinements["file"].duplicated().any():
        dupes = refinements.loc[refinements["file"].duplicated(), "file"].head(5).tolist()
        raise ValueError(f"refinements CSV has duplicate files (multiple candidates?): {dupes}")
    if expected_frames is not None and len(refinements) != expected_frames:
        raise ValueError(
            f"expected {expected_frames} refinement rows (use-all-score run), got {len(refinements)}; "
            "pass --expected-frames 0 only for explicit partial audits"
        )

    base = pd.read_csv(base_alignment_csv)
    for col in ("file", "refined_align_dx_px", "refined_align_dy_px"):
        if col not in base.columns:
            raise ValueError(f"base alignment CSV missing column {col!r}: {base_alignment_csv}")

    merged = base.merge(
        refinements[["file", "initial_dx_px", "initial_dy_px", "refined_dx_px", "refined_dy_px", "delta_dx_px", "delta_dy_px"]],
        on="file",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(refinements):
        raise ValueError(
            f"file mismatch between base alignment ({len(base)} rows) and refinements "
            f"({len(refinements)} rows): only {len(merged)} matched"
        )

    mismatch_dx = (merged["refined_align_dx_px"] - merged["initial_dx_px"]).abs().max()
    mismatch_dy = (merged["refined_align_dy_px"] - merged["initial_dy_px"]).abs().max()
    if max(float(mismatch_dx), float(mismatch_dy)) > float(max_initial_mismatch_px):
        raise ValueError(
            "refinements were not computed from this base alignment: max |initial - base| = "
            f"({mismatch_dx:.6f}, {mismatch_dy:.6f}) px exceeds {max_initial_mismatch_px} px"
        )

    out = base.copy()
    out = out.merge(
        refinements[["file", "refined_dx_px", "refined_dy_px", "delta_dx_px", "delta_dy_px"]],
        on="file",
        how="inner",
        validate="one_to_one",
    )
    out["stage0a_delta_dx_px"] = out["delta_dx_px"]
    out["stage0a_delta_dy_px"] = out["delta_dy_px"]
    out["refined_align_dx_px"] = out["refined_dx_px"]
    out["refined_align_dy_px"] = out["refined_dy_px"]
    out = out.drop(columns=["refined_dx_px", "refined_dy_px", "delta_dx_px", "delta_dy_px"])
    return out
