"""EP01 helpers for SR data-basis and main-session modeling."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thermal_core.io import detect_sessions, load_frame
from thermal_core.plotting import METHOD_COLOR_LIST, make_figure, setup_academic_style

NOISE_FLOOR_C = 0.0724


@dataclass(frozen=True)
class SessionModel:
    """Acquisition-order session model used by EP01 and downstream SR work."""

    df_acquisition: pd.DataFrame
    df_filename: pd.DataFrame
    session_summary: pd.DataFrame
    main_session: int
    session_ids: np.ndarray
    break_indices: np.ndarray
    threshold: float
    filename_session_ids: np.ndarray
    filename_break_indices: np.ndarray
    filename_threshold: float
    boundary_jumps: pd.DataFrame


def add_robust_temperature_stats(
    df: pd.DataFrame,
    data_dir: Path,
) -> pd.DataFrame:
    """Add frame median, robust trimmed mean, and central quantiles.

    ``audit_all_frames`` already reads the frame once for mean/std/min/max.  EP01
    needs median and robust curves for SR session modeling, so this helper keeps
    the extra statistics local to EP01 without changing the shared IO contract.
    """
    required = {"T_median", "T_q05", "T_q95", "T_robust"}
    if required.issubset(df.columns):
        return df.copy()

    rows: list[dict[str, float | str]] = []
    for fname in df["file"].astype(str):
        frame = load_frame(data_dir / fname).astype(float, copy=False)
        finite = frame[np.isfinite(frame)]
        q05, q50, q95 = np.quantile(finite, [0.05, 0.50, 0.95])
        trimmed = finite[(finite >= q05) & (finite <= q95)]
        rows.append(
            {
                "file": fname,
                "T_median": float(q50),
                "T_q05": float(q05),
                "T_q95": float(q95),
                "T_robust": float(np.mean(trimmed)),
            }
        )

    robust = pd.DataFrame(rows)
    return df.drop(columns=list(required & set(df.columns)), errors="ignore").merge(
        robust, on="file", how="left"
    )


def build_session_model(df: pd.DataFrame) -> SessionModel:
    """Build filename-order and acquisition-order session views."""
    df_filename = df.sort_values("file").reset_index(drop=True)
    df_acquisition = df.sort_values(["acquisition_order", "file"]).reset_index(drop=True)

    filename_session_ids, filename_break_indices, filename_threshold = detect_sessions(
        df_filename
    )
    session_ids, break_indices, threshold = detect_sessions(df_acquisition)

    df_acquisition = df_acquisition.copy()
    df_acquisition["session"] = session_ids
    df_acquisition["session_source"] = "acquisition_order_mtime"

    session_lookup = df_acquisition[["file", "session"]]
    df_filename = (
        df_filename.drop(columns=["session"], errors="ignore")
        .merge(session_lookup, on="file", how="left")
        .reset_index(drop=True)
    )

    summary_kwargs = {
        "n_frames": ("file", "count"),
        "first_order": ("acquisition_order", "min"),
        "last_order": ("acquisition_order", "max"),
        "first_file": ("file", "first"),
        "last_file": ("file", "last"),
        "mean_temp": ("T_mean", "mean"),
        "median_temp": ("T_mean", "median"),
        "min_mean_temp": ("T_mean", "min"),
        "max_mean_temp": ("T_mean", "max"),
    }
    if "T_median" in df_acquisition.columns:
        summary_kwargs["frame_median_temp"] = ("T_median", "median")
    if "T_robust" in df_acquisition.columns:
        summary_kwargs["robust_temp"] = ("T_robust", "median")

    session_summary = (
        df_acquisition.groupby("session")
        .agg(**summary_kwargs)
        .reset_index()
    )
    main_session = int(
        session_summary.loc[session_summary["n_frames"].idxmax(), "session"]
    )
    df_acquisition["is_main_session"] = df_acquisition["session"].eq(main_session)
    df_filename["is_main_session"] = df_filename["session"].eq(main_session)

    boundary_rows = []
    t_mean = df_acquisition["T_mean"].to_numpy(dtype=float)
    for brk in break_indices:
        before = df_acquisition.iloc[int(brk)]
        after = df_acquisition.iloc[int(brk) + 1]
        boundary_rows.append(
            {
                "boundary_after_order": int(before["acquisition_order"]),
                "from_session": int(before["session"]),
                "to_session": int(after["session"]),
                "from_file": str(before["file"]),
                "to_file": str(after["file"]),
                "delta_mean_C": float(t_mean[int(brk) + 1] - t_mean[int(brk)]),
                "abs_delta_mean_C": float(abs(t_mean[int(brk) + 1] - t_mean[int(brk)])),
            }
        )
    boundary_jumps = pd.DataFrame(boundary_rows)

    return SessionModel(
        df_acquisition=df_acquisition,
        df_filename=df_filename,
        session_summary=session_summary,
        main_session=main_session,
        session_ids=session_ids,
        break_indices=break_indices,
        threshold=float(threshold),
        filename_session_ids=filename_session_ids,
        filename_break_indices=filename_break_indices,
        filename_threshold=float(filename_threshold),
        boundary_jumps=boundary_jumps,
    )


def coordinate_count_grid(
    df: pd.DataFrame,
    valid_coords: list[int],
) -> np.ndarray:
    """Return count-per-coordinate grid indexed as Y rows by X columns."""
    vals = sorted(int(v) for v in valid_coords)
    x_index = {x: j for j, x in enumerate(vals)}
    y_index = {y: i for i, y in enumerate(vals)}
    grid = np.zeros((len(vals), len(vals)), dtype=int)

    for row in df.itertuples(index=False):
        x = int(getattr(row, "X"))
        y = int(getattr(row, "Y"))
        if x in x_index and y in y_index:
            grid[y_index[y], x_index[x]] += 1
    return grid


def make_bmp_txt_pairing_table(
    pairing: dict,
    *,
    rename_mapping: pd.DataFrame | None = None,
    rename_mapping_path: str | Path | None = None,
) -> pd.DataFrame:
    """Summarize TXT/BMP pairing and rename provenance without inventing data."""
    only_txt = sorted(str(v) for v in pairing.get("only_txt", []))
    only_bmp = sorted(str(v) for v in pairing.get("only_bmp", []))
    mapping_available = rename_mapping is not None and not rename_mapping.empty

    if mapping_available:
        txt_rows = int(rename_mapping["ext"].astype(str).str.lower().eq(".txt").sum())
        bmp_rows = int(rename_mapping["ext"].astype(str).str.lower().eq(".bmp").sum())
        mapping_value = f"{len(rename_mapping)} rows ({txt_rows} TXT, {bmp_rows} BMP)"
        mapping_use = "Use for raw-name provenance and special-case rename audit."
    else:
        path_text = str(rename_mapping_path) if rename_mapping_path else "rename_mapping.csv"
        mapping_value = f"missing: {path_text}"
        mapping_use = "Record provenance gap only; do not reconstruct raw names by guesswork."

    rows = [
        (
            "TXT matrices",
            int(pairing.get("n_txt", 0)),
            "Numerical LWIR temperature inputs for downstream audit.",
        ),
        (
            "BMP previews",
            int(pairing.get("n_bmp", 0)),
            "Same-name visual references; not SR numerical input.",
        ),
        (
            "Paired stems",
            int(pairing.get("n_paired", 0)),
            "Frames with both TXT and BMP companions.",
        ),
        (
            "TXT without BMP",
            ", ".join(only_txt) if only_txt else "none",
            "Still numerically readable, but harder to inspect visually.",
        ),
        (
            "BMP without TXT",
            ", ".join(only_bmp) if only_bmp else "none",
            "Cannot be used as raw temperature input.",
        ),
        (
            "Rename provenance",
            mapping_value,
            mapping_use,
        ),
    ]
    return pd.DataFrame(rows, columns=["Check", "Value", "Downstream meaning"])


def make_missing_coordinate_table(
    df: pd.DataFrame,
    valid_coords: list[int],
    known_missing: list[list[int]] | list[tuple[int, int]] | None = None,
    *,
    main_session: int | None = None,
) -> pd.DataFrame:
    """Return explicit missing-coordinate rows for the expected command grid."""
    vals = sorted(int(v) for v in valid_coords)
    expected = {(x, y) for y in vals for x in vals}
    actual = {
        (int(row.X), int(row.Y))
        for row in df[["X", "Y"]].drop_duplicates().itertuples(index=False)
    }
    known = {tuple(map(int, coord)) for coord in (known_missing or [])}

    all_counts = df.groupby(["X", "Y"]).size()
    if main_session is not None and "session" in df.columns:
        main_counts = df[df["session"].eq(main_session)].groupby(["X", "Y"]).size()
    else:
        main_counts = pd.Series(dtype=int)

    rows = []
    for x, y in sorted(expected - actual, key=lambda c: (c[1], c[0])):
        rows.append(
            {
                "X_um": int(x),
                "Y_um": int(y),
                "all_frames": int(all_counts.get((x, y), 0)),
                "main_session_frames": int(main_counts.get((x, y), 0)),
                "configured_known_missing": "yes" if (x, y) in known else "no",
                "status": "known coordinate-level absence"
                if (x, y) in known
                else "unexpected coordinate-level absence",
                "downstream_handling": "Exclude from usable coordinate grid; do not synthesize a frame.",
            }
        )

    columns = [
        "X_um",
        "Y_um",
        "all_frames",
        "main_session_frames",
        "configured_known_missing",
        "status",
        "downstream_handling",
    ]
    return pd.DataFrame(rows, columns=columns)


def make_raster_row_table(
    df_acquisition: pd.DataFrame,
    valid_coords: list[int],
    known_missing: list[list[int]] | list[tuple[int, int]] | None = None,
    *,
    r_value: int = 0,
) -> pd.DataFrame:
    """Summarize row-wise raster order for one repeat ID.

    This is an acquisition-order diagnostic only. It checks whether command
    coordinates follow the expected step-and-shoot raster; it is not an image
    alignment measurement.
    """
    vals = sorted(int(v) for v in valid_coords)
    known = {tuple(map(int, coord)) for coord in (known_missing or [])}
    df_r = (
        df_acquisition[df_acquisition["R"].astype(int).eq(int(r_value))]
        .sort_values(["acquisition_order", "file"])
        .copy()
    )

    rows = []
    previous_last_order: int | None = None
    for y in vals:
        group = df_r[df_r["Y"].astype(int).eq(y)].sort_values("acquisition_order")
        observed = group["X"].astype(int).tolist()
        expected = [x for x in vals if (x, y) not in known]
        missing_x = [x for x in vals if x not in set(observed)]
        monotonic = observed == sorted(observed)
        matches_expected = observed == expected
        first_order = int(group["acquisition_order"].min()) if not group.empty else None
        last_order = int(group["acquisition_order"].max()) if not group.empty else None
        row_gap = (
            int(first_order - previous_last_order)
            if first_order is not None and previous_last_order is not None
            else "start"
        )
        if last_order is not None:
            previous_last_order = last_order
        rows.append(
            {
                "Y_um": y,
                "first_order": first_order,
                "last_order": last_order,
                "gap_from_previous_row": row_gap,
                "n_r0_frames": int(len(group)),
                "first_X_um": int(observed[0]) if observed else None,
                "last_X_um": int(observed[-1]) if observed else None,
                "missing_X_um": ", ".join(str(x) for x in missing_x) if missing_x else "none",
                "X_monotonic_in_acquisition": bool(monotonic),
                "matches_expected_after_known_missing": bool(matches_expected),
            }
        )

    return pd.DataFrame(rows)


def make_acquisition_order_audit_table(df_acquisition: pd.DataFrame) -> pd.DataFrame:
    """Return the per-frame acquisition audit columns used by later episodes."""
    cols = ["file", "X", "Y", "R", "T_mean", "mtime", "acquisition_order", "session"]
    if "source_file" in df_acquisition.columns:
        cols.insert(1, "source_file")
    return df_acquisition.sort_values(["acquisition_order", "file"])[cols].copy()


def make_boundary_jump_table(
    boundary_jumps: pd.DataFrame,
    noise_floor_c: float,
) -> pd.DataFrame:
    """Compare each detected session boundary jump with the noise floor."""
    columns = [
        "boundary_after_order",
        "to_order",
        "transition",
        "from_file",
        "to_file",
        "delta_mean_C",
        "abs_delta_mean_C",
        "noise_floor_C",
        "abs_delta_over_noise_floor",
        "diagnostic",
    ]
    if boundary_jumps.empty:
        return pd.DataFrame(columns=columns)

    out = boundary_jumps.copy()
    out["to_order"] = out["boundary_after_order"].astype(int) + 1
    out["transition"] = (
        "S"
        + out["from_session"].astype(int).astype(str)
        + " -> S"
        + out["to_session"].astype(int).astype(str)
    )
    out["noise_floor_C"] = float(noise_floor_c)
    if noise_floor_c > 0:
        out["abs_delta_over_noise_floor"] = (
            out["abs_delta_mean_C"].astype(float) / float(noise_floor_c)
        )
    else:
        out["abs_delta_over_noise_floor"] = np.nan
    out["diagnostic"] = np.where(
        out["abs_delta_over_noise_floor"].ge(10),
        "thermal-state jump; keep sessions isolated",
        "small jump relative to noise floor; review manually",
    )
    return out[columns].reset_index(drop=True)


def make_frame_audit_contract_table(*, include_source_file: bool = False) -> pd.DataFrame:
    """Document the downstream contract of frame_audit.csv columns."""
    rows = [
        (
            "file",
            "Standard X_Y_R TXT filename.",
            "Join key for raw TXT/BMP files and later per-frame metrics.",
            "required",
        ),
        (
            "X, Y, R",
            "Stage command coordinate and repeat ID parsed from filename.",
            "Command prior and grid bookkeeping only; not alignment truth.",
            "required",
        ),
        (
            "filename_order",
            "Alphabetical order after renaming.",
            "Audit/debug only; do not use for session detection or timelines.",
            "required",
        ),
        (
            "mtime",
            "Filesystem modification time used to recover capture order.",
            "Provenance for acquisition_order.",
            "required",
        ),
        (
            "acquisition_order",
            "Zero-based frame order sorted by mtime then filename.",
            "Canonical time axis for sessions, raster diagnostics, and frame selection.",
            "required",
        ),
        (
            "rows, cols",
            "Detector matrix shape.",
            "Validate that all TXT frames share the 480 x 640 grid.",
            "required",
        ),
        (
            "T_min, T_max, T_mean, T_std",
            "Basic per-frame temperature statistics.",
            "Detect bad frames and session-level thermal jumps.",
            "required",
        ),
        (
            "T_q05, T_median, T_q95, T_robust",
            "Robust per-frame temperature statistics.",
            "Check that session conclusions are not driven by extreme pixels.",
            "required",
        ),
        (
            "session",
            "Temperature segment ID detected in acquisition order.",
            "Physical thermal-state diagnostic; do not use alone for SR input after repeat exclusion.",
            "required",
        ),
        (
            "session_source",
            "Method used to assign the session field.",
            "Provenance guard against filename-order session artifacts.",
            "required",
        ),
        (
            "is_raw_main_session",
            "Boolean flag for the largest acquisition-order temperature segment before repeat exclusion.",
            "Preserves the 255-frame physical session-2 definition for diagnostics.",
            "required",
        ),
        (
            "is_repeat_frame",
            "Boolean flag for nonzero repeat IDs (R != 0).",
            "Exclude from downstream SR inputs; keep only for repeat diagnostics.",
            "required",
        ),
        (
            "has_repeat_sibling",
            "Whether the same (X, Y) coordinate has more than one repeat ID in the full audit.",
            "Use for repeat-acquisition diagnostics and provenance checks.",
            "required",
        ),
        (
            "repeat_exclusion_reason",
            "Reason a repeat frame is excluded from clean SR input.",
            "Documents prewarm/main/post-main repeat handling without deleting audit rows.",
            "required",
        ),
        (
            "is_clean_main_session",
            "Boolean flag for raw main-session frames after repeat-frame exclusion.",
            "Clean thermal baseline for later alignment and SR work.",
            "required",
        ),
        (
            "is_sr_usable",
            "Boolean flag for the default downstream SR input set.",
            "Primary frame-selection gate for reconstruction code.",
            "required",
        ),
        (
            "sr_input_index",
            "Zero-based index within is_sr_usable frames sorted by acquisition_order.",
            "Stable per-frame index for clean SR inputs; blank for excluded frames.",
            "required",
        ),
        (
            "sr_exclusion_reason",
            "Reason a frame is outside the default SR input set.",
            "Audit trail for repeat, non-main-session, and invalid-frame exclusion.",
            "required",
        ),
        (
            "frame_role",
            "Human-readable role assigned by EP01.",
            "Quickly separates sr_default, repeat_diagnostic, and non-main diagnostic frames.",
            "required",
        ),
        (
            "is_main_session",
            "Compatibility alias for is_sr_usable after repeat exclusion.",
            "Legacy loaders that filter is_main_session now receive the clean SR input set.",
            "required",
        ),
    ]
    if include_source_file:
        rows.insert(
            1,
            (
                "source_file",
                "Original raw filename before standardization.",
                "Rename provenance and special-case filename audit.",
                "optional",
            ),
        )
    return pd.DataFrame(
        rows,
        columns=["Column", "Meaning", "Downstream use", "Contract status"],
    )


def add_sr_input_selection_columns(
    df_acquisition: pd.DataFrame,
    *,
    raw_main_session: int,
) -> pd.DataFrame:
    """Add repeat-exclusion and clean-SR selection columns to frame audit rows."""
    out = df_acquisition.copy()
    raw_main = out["session"].astype(int).eq(int(raw_main_session))
    repeat = out["R"].astype(int).ne(0)
    valid_matrix = (
        out["rows"].eq(480)
        & out["cols"].eq(640)
        & ~out["has_nan"].astype(bool)
        & ~out["has_inf"].astype(bool)
    )

    coord_repeat_counts = out.groupby(["X", "Y"])["R"].transform("nunique")
    out["is_raw_main_session"] = raw_main
    out["is_repeat_frame"] = repeat
    out["has_repeat_sibling"] = coord_repeat_counts.gt(1)

    repeat_reason = np.full(len(out), "", dtype=object)
    clean_candidate = raw_main & ~repeat & valid_matrix
    if clean_candidate.any():
        first_clean_order = float(out.loc[clean_candidate, "acquisition_order"].min())
        last_clean_order = float(out.loc[clean_candidate, "acquisition_order"].max())
    else:
        first_clean_order = float("inf")
        last_clean_order = float("-inf")
    acq_order = out["acquisition_order"].astype(float)
    repeat_reason[repeat & acq_order.lt(first_clean_order)] = "prewarm_or_main_start_repeat"
    repeat_reason[repeat & acq_order.gt(last_clean_order)] = "post_main_repeat"
    repeat_reason[repeat & (repeat_reason == "")] = "in_main_repeat"
    out["repeat_exclusion_reason"] = repeat_reason

    clean_main = clean_candidate
    out["is_clean_main_session"] = clean_main
    out["is_sr_usable"] = clean_main

    sr_reason = np.full(len(out), "", dtype=object)
    sr_reason[~raw_main] = "not_raw_main_session"
    sr_reason[raw_main & repeat] = "repeat_frame"
    sr_reason[raw_main & ~repeat & ~valid_matrix] = "invalid_matrix"
    out["sr_exclusion_reason"] = sr_reason

    roles = np.full(len(out), "other_session", dtype=object)
    roles[raw_main & ~repeat] = "sr_default"
    roles[repeat] = "repeat_diagnostic"
    roles[raw_main & repeat] = "repeat_diagnostic"
    out["frame_role"] = roles

    out["sr_input_index"] = pd.NA
    sr_order = out.loc[out["is_sr_usable"]].sort_values(["acquisition_order", "file"]).index
    out.loc[sr_order, "sr_input_index"] = np.arange(len(sr_order), dtype=int)

    # Compatibility: downstream loaders historically used this column as the
    # default reconstruction gate.  After repeat exclusion it aliases is_sr_usable.
    out["is_main_session"] = out["is_sr_usable"]
    return out


def _adjacent_jump_table(frame_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Collect adjacent-frame mean-temperature jumps above threshold."""
    t_means = frame_df["T_mean"].to_numpy(dtype=float)
    files = frame_df["file"].astype(str).to_numpy()
    rows: list[dict[str, float | str | int]] = []
    for idx in range(len(t_means) - 1):
        delta = float(t_means[idx + 1] - t_means[idx])
        if abs(delta) <= threshold:
            continue
        rows.append(
            {
                "break_index": idx,
                "x_pos": float(idx) + 0.5,
                "y_before": float(t_means[idx]),
                "y_after": float(t_means[idx + 1]),
                "from_file": files[idx],
                "to_file": files[idx + 1],
                "delta_C": delta,
                "abs_delta_C": abs(delta),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "break_index",
                "x_pos",
                "y_before",
                "y_after",
                "from_file",
                "to_file",
                "delta_C",
                "abs_delta_C",
            ]
        )
    return (
        pd.DataFrame(rows)
        .sort_values("abs_delta_C", ascending=False)
        .reset_index(drop=True)
    )


def _format_jump_label(from_file: str, to_file: str, delta_c: float) -> str:
    sign = "+" if delta_c >= 0 else ""
    return f"{from_file} → {to_file}\nΔT={sign}{delta_c:.2f}°C"


def _select_spread_jumps(
    jumps: pd.DataFrame,
    top_k: int,
    *,
    min_label_gap: float = 35.0,
) -> pd.DataFrame:
    """Pick the largest jump in each index band to avoid label clustering."""
    if jumps.empty or len(jumps) <= top_k:
        return jumps

    x_max = float(jumps["x_pos"].max())
    band_edges = np.linspace(0.0, x_max + 1.0, top_k + 1)
    selected_rows: list[pd.Series] = []
    used_indices: set[int] = set()
    used_positions: list[float] = []

    def _far_enough(x_pos: float) -> bool:
        return all(abs(x_pos - prev) >= min_label_gap for prev in used_positions)

    for band_idx in range(top_k):
        lo = float(band_edges[band_idx])
        hi = float(band_edges[band_idx + 1])
        band = jumps[
            jumps["x_pos"].between(lo, hi, inclusive="left")
            & ~jumps["break_index"].isin(used_indices)
        ]
        if band.empty:
            continue
        for _, row in band.iterrows():
            x_pos = float(row["x_pos"])
            if not _far_enough(x_pos):
                continue
            selected_rows.append(row)
            used_indices.add(int(row["break_index"]))
            used_positions.append(x_pos)
            break

    if len(selected_rows) < top_k:
        for _, row in jumps.iterrows():
            if int(row["break_index"]) in used_indices:
                continue
            x_pos = float(row["x_pos"])
            if not _far_enough(x_pos):
                continue
            selected_rows.append(row)
            used_indices.add(int(row["break_index"]))
            used_positions.append(x_pos)
            if len(selected_rows) >= top_k:
                break

    return pd.DataFrame(selected_rows).reset_index(drop=True)


def _annotate_top_jumps(
    ax: plt.Axes,
    frame_df: pd.DataFrame,
    threshold: float,
    top_k: int,
    *,
    spread_across_timeline: bool = False,
) -> int:
    """Plot mean-temperature curve and mark the largest adjacent jumps."""
    jumps = _adjacent_jump_table(frame_df, threshold)
    t_means = frame_df["T_mean"].to_numpy(dtype=float)
    idx = np.arange(len(t_means))
    ax.plot(idx, t_means, color="#666666", linewidth=1.0, zorder=3)

    if jumps.empty or top_k <= 0:
        return 0

    labeled = (
        _select_spread_jumps(jumps, top_k)
        if spread_across_timeline
        else jumps.head(top_k)
    )

    for row in labeled.itertuples(index=False):
        x_pos = float(row.x_pos)
        y_mid = 0.5 * (float(row.y_before) + float(row.y_after))
        ax.scatter(
            [x_pos],
            [y_mid],
            s=16,
            color=METHOD_COLOR_LIST[2],
            zorder=4,
            edgecolors="white",
            linewidths=0.4,
        )

    return int(len(jumps))


def _highlight_main_session(
    ax: plt.Axes,
    frame_df: pd.DataFrame,
    session_ids: np.ndarray,
    main_session: int,
) -> None:
    mask = session_ids == int(main_session)
    if not np.any(mask):
        return
    positions = np.flatnonzero(mask)
    ax.axvspan(
        int(positions[0]) - 0.5,
        int(positions[-1]) + 0.5,
        color=METHOD_COLOR_LIST[1],
        alpha=0.10,
        linewidth=0,
        zorder=0,
    )


def plot_order_comparison(
    model: SessionModel,
    *,
    top_k_filename: int = 4,
    top_k_acquisition: int = 3,
    save_path: str | Path | None = None,
    save_fn=None,
) -> plt.Figure:
    """Compare filename vs acquisition order; label only the largest adjacent jumps."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # Disable constrained layout temporarily so we can manually adjust wspace
    setup_academic_style()
    _cl_backup = plt.rcParams.get("figure.constrained_layout.use", False)
    plt.rcParams["figure.constrained_layout.use"] = False

    fig, axes = make_figure("double_col", nrows=1, ncols=2, height=3.4, sharey=True, constrained_layout=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    plt.rcParams["figure.constrained_layout.use"] = _cl_backup

    filename_n_sessions = int(model.filename_session_ids.max()) + 1
    acquisition_n_sessions = int(model.session_ids.max()) + 1
    filename_threshold = float(model.filename_threshold)
    acquisition_threshold = float(model.threshold)

    n_filename_jumps = _annotate_top_jumps(
        axes[0],
        model.df_filename,
        filename_threshold,
        top_k_filename,
        spread_across_timeline=True,
    )
    _highlight_main_session(
        axes[1],
        model.df_acquisition,
        model.session_ids,
        int(model.main_session),
    )
    n_acquisition_jumps = _annotate_top_jumps(
        axes[1],
        model.df_acquisition,
        acquisition_threshold,
        top_k_acquisition,
    )

    axes[0].set_title(
        "(a) Filename order\n"
        f"{filename_n_sessions} apparent sessions, "
        f"{n_filename_jumps} jumps > {filename_threshold:.2f}°C "
        f"(top {min(top_k_filename, n_filename_jumps)} marked)"
    )
    axes[1].set_title(
        "(b) Acquisition order (mtime)\n"
        f"{acquisition_n_sessions} physical segments, "
        f"{n_acquisition_jumps} jumps > {acquisition_threshold:.2f}°C "
        f"(all marked)"
    )
    for ax, xlabel in zip(
        axes,
        ["Frame index (filename sort)", "Frame index (acquisition order)"],
        strict=True,
    ):
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)
        ax.set_xlim(-0.5, len(model.df_filename) - 0.5)

    axes[0].set_ylabel("Frame mean temperature [°C]")
    axes[1].legend(
        handles=[
            Line2D([0], [0], color="#666666", linewidth=1.2, label="Frame mean temperature"),
            Patch(facecolor=METHOD_COLOR_LIST[1], alpha=0.12, label="Main session (255 frames)"),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=METHOD_COLOR_LIST[2],
                markersize=5,
                label="Marked adjacent jump",
            ),
        ],
        loc="lower right",
        fontsize=7,
        frameon=False,
    )

    # Manually adjust margins and spacing (with a larger wspace for more breathing room)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.82, wspace=0.18)

    if save_fn and save_path:
        save_fn(fig, save_path)
    return fig


def _plot_order_curve_panel(
    ax: plt.Axes,
    frame_df: pd.DataFrame,
    *,
    title: str,
    xlabel: str,
    color: str,
    jump_threshold: float | None = None,
) -> int:
    df = frame_df.reset_index(drop=True)
    x = np.arange(len(df))
    ax.plot(x, df["T_mean"].to_numpy(float), color=color, linewidth=1.0, zorder=3)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_xlim(-0.5, max(len(df) - 0.5, 0.5))

    if jump_threshold is None:
        return 0
    jumps = _adjacent_jump_table(df, float(jump_threshold))
    if jumps.empty:
        return 0
    ax.scatter(
        jumps["x_pos"],
        0.5 * (jumps["y_before"] + jumps["y_after"]),
        s=12,
        color=METHOD_COLOR_LIST[2],
        edgecolors="white",
        linewidths=0.3,
        zorder=4,
    )
    return int(len(jumps))


def plot_repeat_exclusion_order_comparison(
    model: SessionModel,
    clean_sr_df: pd.DataFrame,
    *,
    save_path: str | Path | None = None,
    save_fn=None,
) -> plt.Figure:
    """Show mean-temperature timelines before and after repeat-frame exclusion."""
    setup_academic_style()
    _cl_backup = plt.rcParams.get("figure.constrained_layout.use", False)
    plt.rcParams["figure.constrained_layout.use"] = False
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(7.1, 5.2),
        sharey=True,
        constrained_layout=False,
    )
    fig.set_layout_engine(None)
    plt.rcParams["figure.constrained_layout.use"] = _cl_backup

    before_filename = model.df_filename.sort_values("file").reset_index(drop=True)
    before_acq = model.df_acquisition.sort_values(["acquisition_order", "file"]).reset_index(drop=True)
    after_filename = clean_sr_df.sort_values("file").reset_index(drop=True)
    after_acq = clean_sr_df.sort_values(["acquisition_order", "file"]).reset_index(drop=True)

    filename_jumps_before = _plot_order_curve_panel(
        axes[0, 0],
        before_filename,
        title=f"(a) Before exclusion: filename order (n={len(before_filename)})",
        xlabel="Frame index (filename sort)",
        color="#666666",
        jump_threshold=float(model.filename_threshold),
    )
    acq_jumps_before = _plot_order_curve_panel(
        axes[0, 1],
        before_acq,
        title=f"(b) Before exclusion: acquisition order (n={len(before_acq)})",
        xlabel="Frame index (acquisition order)",
        color="#666666",
        jump_threshold=float(model.threshold),
    )
    filename_jumps_after = _plot_order_curve_panel(
        axes[1, 0],
        after_filename,
        title=f"(c) After exclusion: filename order (n={len(after_filename)})",
        xlabel="Frame index (filename sort)",
        color=METHOD_COLOR_LIST[0],
        jump_threshold=float(model.filename_threshold),
    )
    acq_jumps_after = _plot_order_curve_panel(
        axes[1, 1],
        after_acq,
        title=f"(d) After exclusion: acquisition order (n={len(after_acq)})",
        xlabel="Frame index (acquisition order)",
        color=METHOD_COLOR_LIST[1],
        jump_threshold=float(model.threshold),
    )

    axes[0, 0].set_ylabel("Frame mean temperature [°C]")
    axes[1, 0].set_ylabel("Frame mean temperature [°C]")
    for ax, n_jumps in zip(
        axes.flat,
        [filename_jumps_before, acq_jumps_before, filename_jumps_after, acq_jumps_after],
        strict=True,
    ):
        ax.text(
            0.06,
            0.06,
            f"Jumps > gate: {n_jumps}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7,
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "#cccccc",
                "linewidth": 0.4,
                "alpha": 0.86,
            },
        )

    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.09, top=0.95, hspace=0.40, wspace=0.14)

    if save_fn and save_path:
        save_fn(fig, save_path)
    return fig


def plot_robust_temperature_curve(
    model: SessionModel,
    *,
    save_path: str | Path | None = None,
    save_fn=None,
) -> plt.Figure:
    """Plot acquisition-order mean/median/robust temperature curves with a split X-axis."""
    df = model.df_acquisition.reset_index(drop=True)
    x = np.arange(len(df))

    # Apply standard academic styles first
    setup_academic_style()

    # Split the figure into two axes with a custom aspect ratio.
    # IMPORTANT: setup_academic_style() sets rcParams["figure.constrained_layout.use"] = True.
    # Constrained layout completely ignores subplots_adjust(), so we must temporarily
    # disable it at the rcParam level before creating the figure.
    _cl_backup = plt.rcParams.get("figure.constrained_layout.use", False)
    plt.rcParams["figure.constrained_layout.use"] = False
    fig, (ax1, ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(4.5, 3.5),
        sharey=True,
        gridspec_kw={"width_ratios": [1, 2.2]},
    )
    fig.set_layout_engine(None)  # belt-and-suspenders: also clear any layout engine on fig
    plt.rcParams["figure.constrained_layout.use"] = _cl_backup  # restore global default
    # Now subplots_adjust works correctly — leave whitespace above for the title
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.14, top=0.91, wspace=0.08)

    # Plot everything on both axes
    for ax in [ax1, ax2]:
        for sid, group in df.groupby("session", sort=True):
            start = int(group.index.min())
            end = int(group.index.max())
            is_main = int(sid) == model.main_session
            color = METHOD_COLOR_LIST[1] if is_main else METHOD_COLOR_LIST[2]
            ax.axvspan(
                start - 0.5,
                end + 0.5,
                color=color,
                alpha=0.08 if is_main else 0.12,
                linewidth=0,
            )

        if {"T_q05", "T_q95"}.issubset(df.columns):
            ax.fill_between(
                x,
                df["T_q05"],
                df["T_q95"],
                color="#FF9F1C",  # Vibrant warm orange/yellow band
                alpha=0.22,       # High visibility transparency
                linewidth=0,
                label="Frame central 90% pixel band" if ax == ax2 else "",
            )

        ax.plot(x, df["T_mean"], color="#777777", linewidth=0.9, label="Mean" if ax == ax2 else "")
        if "T_median" in df.columns:
            ax.plot(x, df["T_median"], color=METHOD_COLOR_LIST[0], linewidth=1.2, label="Median" if ax == ax2 else "")
        if "T_robust" in df.columns:
            ax.plot(
                x,
                df["T_robust"],
                color=METHOD_COLOR_LIST[1],
                linewidth=1.2,
                linestyle="--",
                label="5-95% trimmed mean" if ax == ax2 else "",
            )

        for brk in model.break_indices:
            ax.axvline(int(brk) + 0.5, color="#333333", linestyle="--", linewidth=0.7)

        ax.grid(axis="y", alpha=0.25, linewidth=0.5)

    # Set boundaries for non-uniform zoom
    split_x = 11.5
    ax1.set_xlim(-0.5, split_x)
    ax2.set_xlim(split_x, len(df) - 0.5)

    # Place text labels at the right side of boundaries in the top 1/6 area
    last_x_ax1 = -999.0
    for sid, group in df.groupby("session", sort=True):
        start = int(group.index.min())
        is_main = int(sid) == model.main_session
        if is_main:
            label_ax = ax2
            text_x = split_x + 10.0  # Safe offset to the right of the split boundary on ax2 to avoid break marks
        else:
            label_ax = ax1
            candidate_x = max(0.0, start - 0.2)
            # Ensure at least 5.5 units of separation on ax1 to prevent overlap between S0 and S1
            if candidate_x < last_x_ax1 + 5.5:
                text_x = last_x_ax1 + 5.5
            else:
                text_x = candidate_x
            last_x_ax1 = text_x

        label_ax.text(
            text_x,
            0.83,  # Top 1/6 height of axes (1 - 1/6 ≈ 0.83)
            f"S{int(sid)} (N={len(group)})",
            transform=label_ax.get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=8,
            color="#333333",
            fontweight="bold",
        )

    # Broken axis decorations: hide spines and tick lines on the break side
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(left=False, labelleft=False)

    # Draw diagonal break marks on the bottom spine at the break gap
    d = 0.5  # slant size
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=8,
                  linestyle="none", color="black", mec="black", mew=1, clip_on=False)
    ax1.plot([1], [0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0], [0], transform=ax2.transAxes, **kwargs)

    # Figure-level labels
    ax1.set_ylabel("Temperature [deg C]", fontsize=9.5)
    fig.supxlabel("Frame index (acquisition order)", fontsize=9.5, y=0.03)

    ax2.legend(loc="upper right", fontsize=8)

    if save_fn and save_path:
        save_fn(fig, save_path)
    return fig


def _dedupe_legend_entries(handles: list, labels: list[str]) -> tuple[list, list[str]]:
    combined_handles: list = []
    combined_labels: list[str] = []
    seen: set[str] = set()
    for handle, label in zip(handles, labels, strict=True):
        if label in seen:
            continue
        seen.add(label)
        combined_handles.append(handle)
        combined_labels.append(label)
    return combined_handles, combined_labels


def _place_panel_legend(
    fig: plt.Figure,
    ax: plt.Axes,
    handles: list,
    labels: list[str],
    *,
    ncol: int = 3,
    legend_gap: float = 0.05,
) -> None:
    """Place a borderless legend centered below *ax* without overlapping it."""
    fig.canvas.draw()
    pos = ax.get_position()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(pos.x0 + pos.width / 2, pos.y0 - legend_gap),
        bbox_transform=fig.transFigure,
        ncol=ncol,
        fontsize=7,
        frameon=False,
        borderaxespad=0.0,
    )


def plot_session_detection_raster(
    df_acquisition: pd.DataFrame,
    valid_coords: list[int],
    main_session: int,
    known_missing: list[list[int]] | list[tuple[int, int]] | None = None,
    *,
    save_path: str | Path | None = None,
    save_fn=None,
) -> plt.Figure:
    """Panel (a): raster grid in acquisition order."""
    vals = sorted(int(v) for v in valid_coords)
    known = {tuple(map(int, coord)) for coord in (known_missing or [])}
    df = df_acquisition.sort_values(["acquisition_order", "file"]).copy()
    main = df[df["session"].eq(main_session)]
    non_main = df[~df["session"].eq(main_session)]
    r0 = df[df["R"].astype(int).eq(0)]
    coord_pad = 2
    coord_lo = min(vals) - coord_pad
    coord_hi = max(vals) + coord_pad

    setup_academic_style()
    _cl_backup = plt.rcParams.get("figure.constrained_layout.use", False)
    plt.rcParams["figure.constrained_layout.use"] = False
    fig = plt.figure(figsize=(4.6, 5.0))
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.0, 0.06],
        left=0.14,
        right=0.86,
        bottom=0.34,
        top=0.90,
        wspace=0.14,
    )
    fig.set_layout_engine(None)
    plt.rcParams["figure.constrained_layout.use"] = _cl_backup

    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    xx, yy = np.meshgrid(vals, vals)
    ax.scatter(
        xx.ravel(),
        yy.ravel(),
        s=8,
        color="#dddddd",
        edgecolor="none",
        label="Expected command grid",
        zorder=1,
    )
    for _, group in r0.groupby("Y", sort=True):
        ordered = group.sort_values("acquisition_order")
        ax.plot(
            ordered["X"],
            ordered["Y"],
            color="#777777",
            linewidth=0.6,
            alpha=0.45,
            zorder=2,
        )

    if not main.empty:
        sc = ax.scatter(
            main["X"],
            main["Y"],
            c=main["acquisition_order"],
            cmap="viridis",
            s=22,
            edgecolor="white",
            linewidth=0.25,
            label="Main-session frames",
            zorder=3,
        )
        fig.colorbar(sc, cax=cax).set_label("Acquisition order")
        first = main.iloc[0]
        last = main.iloc[-1]
        ax.scatter(
            [first["X"]],
            [first["Y"]],
            marker="s",
            s=42,
            color=METHOD_COLOR_LIST[1],
            edgecolor="white",
            linewidth=0.35,
            label="Main start",
            zorder=5,
        )
        ax.scatter(
            [last["X"]],
            [last["Y"]],
            marker="D",
            s=38,
            color=METHOD_COLOR_LIST[3],
            edgecolor="white",
            linewidth=0.35,
            label="Main end",
            zorder=5,
        )

    if not non_main.empty:
        ax.scatter(
            non_main["X"],
            non_main["Y"],
            s=34,
            facecolors="none",
            edgecolors=METHOD_COLOR_LIST[2],
            linewidth=0.9,
            label="Other sessions",
            zorder=4,
        )

    if known:
        missing_x = [coord[0] for coord in sorted(known, key=lambda c: (c[1], c[0]))]
        missing_y = [coord[1] for coord in sorted(known, key=lambda c: (c[1], c[0]))]
        ax.scatter(
            missing_x,
            missing_y,
            marker="x",
            s=52,
            color=METHOD_COLOR_LIST[2],
            linewidth=1.2,
            label="Known missing",
            zorder=6,
        )

    ax.set_xticks(vals)
    ax.set_yticks(vals)
    ax.set_xlim(coord_lo, coord_hi)
    ax.set_ylim(coord_lo, coord_hi)
    ax.set_xlabel("Command X [um]")
    ax.set_ylabel("Command Y [um]")
    ax.set_title("(a) Raster grid in acquisition order")
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    ax.grid(alpha=0.2, linewidth=0.5)

    handles, labels = _dedupe_legend_entries(*ax.get_legend_handles_labels())
    _place_panel_legend(fig, ax, handles, labels, ncol=3, legend_gap=0.09)

    if save_fn and save_path:
        save_fn(fig, save_path)
    return fig


def plot_session_detection_timeline(
    df_acquisition: pd.DataFrame,
    valid_coords: list[int],
    main_session: int,
    *,
    save_path: str | Path | None = None,
    save_fn=None,
) -> plt.Figure:
    """Panel (b): command coordinates over acquisition order."""
    from matplotlib.patches import Patch

    vals = sorted(int(v) for v in valid_coords)
    df = df_acquisition.sort_values(["acquisition_order", "file"]).copy()
    non_main = df[~df["session"].eq(main_session)]
    coord_pad = 2
    coord_lo = min(vals) - coord_pad
    coord_hi = max(vals) + coord_pad

    setup_academic_style()
    _cl_backup = plt.rcParams.get("figure.constrained_layout.use", False)
    plt.rcParams["figure.constrained_layout.use"] = False
    fig = plt.figure(figsize=(5.0, 5.0))
    fig.set_layout_engine(None)
    plt.rcParams["figure.constrained_layout.use"] = _cl_backup
    fig.subplots_adjust(left=0.14, right=0.92, bottom=0.28, top=0.90)

    ax = fig.add_subplot(111)

    for sid, group in df.groupby("session", sort=True):
        start = int(group["acquisition_order"].min())
        end = int(group["acquisition_order"].max())
        color = METHOD_COLOR_LIST[1] if int(sid) == int(main_session) else METHOD_COLOR_LIST[2]
        ax.axvspan(
            start - 0.5,
            end + 0.5,
            color=color,
            alpha=0.08 if int(sid) == int(main_session) else 0.12,
            linewidth=0,
        )
    ax.plot(
        df["acquisition_order"],
        df["X"],
        color=METHOD_COLOR_LIST[0],
        linewidth=1.0,
        label="Command X",
    )
    ax.step(
        df["acquisition_order"],
        df["Y"],
        where="post",
        color=METHOD_COLOR_LIST[2],
        linewidth=1.0,
        label="Command Y",
    )
    if not non_main.empty:
        ax.scatter(
            non_main["acquisition_order"],
            non_main["X"],
            facecolors="none",
            edgecolors=METHOD_COLOR_LIST[2],
            s=18,
            linewidth=0.8,
            label="Other-session X",
        )

    ax.set_xlabel("Acquisition order (mtime)")
    ax.set_ylabel("Command coordinate [um]")
    ax.set_title("(b) Command coordinates over time")
    ax.set_ylim(coord_lo, coord_hi)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)

    span_handles = [
        Patch(
            facecolor=METHOD_COLOR_LIST[1],
            alpha=0.08,
            edgecolor="none",
            label="Main session span",
        ),
        Patch(
            facecolor=METHOD_COLOR_LIST[2],
            alpha=0.12,
            edgecolor="none",
            label="Other session span",
        ),
    ]
    handles_time, labels_time = ax.get_legend_handles_labels()
    handles, labels = _dedupe_legend_entries(
        span_handles + handles_time,
        [item.get_label() for item in span_handles] + labels_time,
    )
    _place_panel_legend(fig, ax, handles, labels, ncol=3, legend_gap=0.09)

    if save_fn and save_path:
        save_fn(fig, save_path)
    return fig


def plot_acquisition_raster_trajectory(
    df_acquisition: pd.DataFrame,
    valid_coords: list[int],
    main_session: int,
    known_missing: list[list[int]] | list[tuple[int, int]] | None = None,
    *,
    save_path: str | Path | None = None,
    save_fn=None,
) -> plt.Figure:
    """Backward-compatible alias: save panel (a) only when a single path is given."""
    return plot_session_detection_raster(
        df_acquisition,
        valid_coords,
        main_session,
        known_missing,
        save_path=save_path,
        save_fn=save_fn,
    )


def plot_session_coverage_heatmaps(
    df_acquisition: pd.DataFrame,
    valid_coords: list[int],
    main_session: int,
    *,
    save_path: str | Path | None = None,
    save_fn=None,
) -> plt.Figure:
    """Plot coordinate coverage for the main session and all other frames."""
    vals = sorted(int(v) for v in valid_coords)
    main_df = df_acquisition[df_acquisition["session"].eq(main_session)]
    other_df = df_acquisition[~df_acquisition["session"].eq(main_session)]
    grids = [
        ("Main session", coordinate_count_grid(main_df, vals)),
        ("Other sessions", coordinate_count_grid(other_df, vals)),
    ]

    vmax = max(int(grid.max()) for _, grid in grids)
    fig, axes = make_figure("double_col", nrows=1, ncols=2, height=3.8)
    images = []

    for ax, (title, grid) in zip(axes, grids):
        im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=vmax, aspect="equal")
        images.append(im)
        for i in range(len(vals)):
            for j in range(len(vals)):
                val = int(grid[i, j])
                text_color = "white" if val >= max(1, vmax // 2) else "black"
                ax.text(j, i, str(val), ha="center", va="center", fontsize=6, color=text_color)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(vals, rotation=90)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(vals)
        ax.set_xlabel("X [um]")
        ax.set_title(f"{title} (n={int(grid.sum())})")
    axes[0].set_ylabel("Y [um]")
    cbar = fig.colorbar(images[-1], ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
    cbar.set_label("Frame count per coordinate")

    if save_fn and save_path:
        save_fn(fig, save_path)
    return fig


def make_ep01_summary_table(
    *,
    n_txt: int,
    n_bmp: int,
    n_paired: int,
    frame_shape: tuple[int, int],
    coord_count: int,
    main_session: int,
    main_frames: int,
    main_coord_count: int,
    boundary_jumps: pd.DataFrame,
    main_mean_range: float,
    sr_input_frames: int | None = None,
    sr_coord_count: int | None = None,
    noise_floor_c: float | None = None,
) -> pd.DataFrame:
    """Create the compact summary table requested by downstream SR agents."""
    if boundary_jumps.empty:
        jump_text = "0 deg C"
    else:
        max_jump = float(boundary_jumps["abs_delta_mean_C"].max())
        med_jump = float(boundary_jumps["abs_delta_mean_C"].median())
        if noise_floor_c and noise_floor_c > 0:
            jump_text = (
                f"{med_jump:.2f} deg C median, {max_jump:.2f} deg C max "
                f"({med_jump / noise_floor_c:.0f}x / {max_jump / noise_floor_c:.0f}x noise floor)"
            )
        else:
            jump_text = f"{med_jump:.2f} deg C median, {max_jump:.2f} deg C max"

    rows = [
        ("Raw TXT/BMP files", f"{n_txt} TXT, {n_bmp} BMP, {n_paired} paired", "Use TXT as SR input; BMP is visual reference."),
        ("Matrix size", f"{frame_shape[0]} x {frame_shape[1]}", "All frames share one detector grid."),
        ("Total frames", str(n_txt), "Full audit population before session gating."),
        ("Raw main session frames", f"session {main_session}: {main_frames}", "Physical temperature segment before repeat exclusion."),
        (
            "Clean SR input frames",
            str(sr_input_frames if sr_input_frames is not None else main_frames),
            "Default input set for micro-scan SR after repeat exclusion.",
        ),
        ("Session temperature jump scale", jump_text, "Cross-session frames should not be mixed."),
        ("Main-session drift span", f"{main_mean_range:.2f} deg C", "Default input frames stay within this temperature band."),
        (
            "Coordinate coverage",
            (
                f"{coord_count}/256 total; {main_coord_count}/256 raw main; "
                f"{sr_coord_count if sr_coord_count is not None else main_coord_count}/256 clean SR"
            ),
            "Clean SR input covers the usable coordinate grid after repeat exclusion.",
        ),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value", "SR use"])


_SUMMARY_METRIC_ZH = {
    "Raw TXT/BMP files": "原始 TXT/BMP 文件",
    "Matrix size": "矩阵尺寸",
    "Total frames": "总帧数",
    "Main session frames": "主 session 帧",
    "Raw main session frames": "原始主 session 帧",
    "Clean SR input frames": "干净 SR 输入帧",
    "Session temperature jump scale": "Session 温度跳变尺度",
    "Main-session drift span": "主 session 温度漂移跨度",
    "Coordinate coverage": "坐标覆盖",
}

_SUMMARY_SR_USE_ZH = {
    "Use TXT as SR input; BMP is visual reference.": "SR 输入使用 TXT；BMP 仅作视觉参考。",
    "All frames share one detector grid.": "所有帧共享同一探测器网格。",
    "Full audit population before session gating.": "session 门控前的完整审计样本。",
    "Default input set for micro-scan SR.": "微扫描 SR 的默认输入集。",
    "Physical temperature segment before repeat exclusion.": "重复帧剔除前的物理温度段。",
    "Default input set for micro-scan SR after repeat exclusion.": "重复帧剔除后的微扫描 SR 默认输入集。",
    "Cross-session frames should not be mixed.": "不同 session 的帧不可混合使用。",
    "Default input frames stay within this temperature band.": "默认输入帧应落在此温度带内。",
    "Main session covers the full usable coordinate grid.": "主 session 覆盖完整可用坐标网格。",
    "Clean SR input covers the usable coordinate grid after repeat exclusion.": "干净 SR 输入在重复帧剔除后覆盖可用坐标网格。",
}

_FRAME_AUDIT_CONTRACT_ZH: dict[str, tuple[str, str]] = {
    "file": (
        "标准化 X_Y_R TXT 文件名。",
        "原始 TXT/BMP 及后续逐帧指标的关联键。",
    ),
    "source_file": (
        "标准化前的原始文件名。",
        "重命名溯源与特殊文件名审计。",
    ),
    "X, Y, R": (
        "从文件名解析的 stage 命令坐标与 repeat 编号。",
        "仅作命令先验与网格 bookkeeping；不是对齐真值。",
    ),
    "filename_order": (
        "重命名后的字母序。",
        "仅用于审计/调试；不可用于 session 检测或时间线。",
    ),
    "mtime": (
        "用于恢复采集顺序的文件系统修改时间。",
        "acquisition_order 的来源依据。",
    ),
    "acquisition_order": (
        "按 mtime 再按文件名排序的零基帧序。",
        "session、raster 诊断与帧选择的规范时间轴。",
    ),
    "rows, cols": (
        "探测器矩阵尺寸。",
        "校验所有 TXT 帧共享 480×640 网格。",
    ),
    "T_min, T_max, T_mean, T_std": (
        "逐帧基础温度统计量。",
        "识别坏帧与 session 级温度跳变。",
    ),
    "T_q05, T_median, T_q95, T_robust": (
        "逐帧稳健温度统计量。",
        "确认 session 结论不受极端像素主导。",
    ),
    "session": (
        "按采集顺序检测到的温度段 ID。",
        "下游对齐与重建输入的 session 门控。",
    ),
    "session_source": (
        "session 字段的赋值方法。",
        "防止文件名排序 session 伪影的来源守卫。",
    ),
    "is_raw_main_session": (
        "重复帧剔除前最大采集顺序温度段的布尔标记。",
        "保留 session 2 共 255 帧的物理诊断定义。",
    ),
    "is_repeat_frame": (
        "非零重复编号帧（R != 0）的布尔标记。",
        "下游 SR 输入必须剔除；仅保留作重复采集诊断。",
    ),
    "has_repeat_sibling": (
        "同一 (X, Y) 坐标是否存在多个 R 编号。",
        "用于重复采集诊断和来源核查。",
    ),
    "repeat_exclusion_reason": (
        "重复帧被剔除出干净 SR 输入的原因。",
        "在不删除审计行的前提下记录预热/主段/补采重复帧处理。",
    ),
    "is_clean_main_session": (
        "原始主 session 剔除重复帧后的布尔标记。",
        "后续 alignment 与 SR 的干净热背景基线。",
    ),
    "is_sr_usable": (
        "默认下游 SR 输入集的布尔标记。",
        "重建代码的主筛选字段。",
    ),
    "sr_input_index": (
        "在 is_sr_usable 帧内按 acquisition_order 排列的零基索引。",
        "干净 SR 输入的稳定逐帧索引；排除帧为空。",
    ),
    "sr_exclusion_reason": (
        "帧不属于默认 SR 输入集的原因。",
        "记录重复帧、非主 session 和无效矩阵等剔除依据。",
    ),
    "frame_role": (
        "EP01 赋予每帧的人类可读角色。",
        "快速区分 sr_default、repeat_diagnostic 和非主段诊断帧。",
    ),
    "is_main_session": (
        "重复帧剔除后 is_sr_usable 的兼容别名。",
        "旧 loader 若筛 is_main_session，也会拿到干净 SR 输入集。",
    ),
}

_CONTRACT_STATUS_ZH = {
    "required": "必需",
    "optional": "可选",
}


def _translate_summary_value(metric: str, value: str) -> str:
    if metric == "Raw TXT/BMP files":
        match = re.fullmatch(r"(\d+) TXT, (\d+) BMP, (\d+) paired", value)
        if match:
            n_txt, n_bmp, n_paired = match.groups()
            return f"{n_txt} 个 TXT，{n_bmp} 个 BMP，{n_paired} 对已配对"
    if metric in {"Main session frames", "Raw main session frames"}:
        match = re.fullmatch(r"session (\d+): (\d+)", value)
        if match:
            return f"session {match.group(1)}：{match.group(2)} 帧"
    if metric == "Clean SR input frames":
        return f"{value} 帧"
    if metric == "Session temperature jump scale":
        if value == "0 deg C":
            return "0 °C"
        match = re.fullmatch(
            r"([\d.]+) deg C median, ([\d.]+) deg C max \((\d+)x / (\d+)x noise floor\)",
            value,
        )
        if match:
            med, max_jump, med_x, max_x = match.groups()
            return (
                f"中位 {med} °C，最大 {max_jump} °C"
                f"（分别为噪声底 {med_x}× / {max_x}×）"
            )
        match = re.fullmatch(r"([\d.]+) deg C median, ([\d.]+) deg C max", value)
        if match:
            return f"中位 {match.group(1)} °C，最大 {match.group(2)} °C"
    if metric == "Main-session drift span":
        return value.replace("deg C", "°C")
    if metric == "Coordinate coverage":
        match = re.fullmatch(r"(\d+)/256 total; (\d+)/256 in main session", value)
        if match:
            return f"共 {match.group(1)}/256；主 session {match.group(2)}/256"
        match = re.fullmatch(r"(\d+)/256 total; (\d+)/256 raw main; (\d+)/256 clean SR", value)
        if match:
            return (
                f"共 {match.group(1)}/256；原始主 session {match.group(2)}/256；"
                f"干净 SR {match.group(3)}/256"
            )
    return value


def translate_ep01_summary_table_zh(table: pd.DataFrame) -> pd.DataFrame:
    """Return a Chinese display copy of the EP01 SR data-basis summary table."""
    translated = table.copy()
    translated["Value"] = [
        _translate_summary_value(str(metric), str(value))
        for metric, value in zip(table["Metric"], table["Value"], strict=True)
    ]
    translated["Metric"] = translated["Metric"].map(_SUMMARY_METRIC_ZH)
    translated["SR use"] = translated["SR use"].map(_SUMMARY_SR_USE_ZH)
    return translated.rename(columns={"Metric": "指标", "Value": "数值", "SR use": "SR 用途"})


def translate_ep01_frame_audit_contract_zh(table: pd.DataFrame) -> pd.DataFrame:
    """Return a Chinese display copy of the frame_audit.csv column contract."""
    rows: list[dict[str, str]] = []
    for _, row in table.iterrows():
        column = str(row["Column"])
        meaning_zh, downstream_zh = _FRAME_AUDIT_CONTRACT_ZH[column]
        status = str(row["Contract status"])
        rows.append(
            {
                "字段": column,
                "含义": meaning_zh,
                "下游用途": downstream_zh,
                "合同状态": _CONTRACT_STATUS_ZH.get(status, status),
            }
        )
    return pd.DataFrame(rows, columns=["字段", "含义", "下游用途", "合同状态"])
