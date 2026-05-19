"""EP01 helpers for SR data-basis and main-session modeling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thermal_core.io import detect_sessions, load_frame
from thermal_core.plotting import METHOD_COLOR_LIST, make_figure


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
            "Session gate for downstream alignment and reconstruction inputs.",
            "required",
        ),
        (
            "session_source",
            "Method used to assign the session field.",
            "Provenance guard against filename-order session artifacts.",
            "required",
        ),
        (
            "is_main_session",
            "Boolean flag for the largest usable temperature segment.",
            "Default filter for the 255-frame main-session input set.",
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


def plot_order_comparison(
    model: SessionModel,
    *,
    save_path: str | Path | None = None,
    save_fn=None,
) -> plt.Figure:
    """Compare filename sorting against true acquisition-order sorting."""
    fig, axes = make_figure("double_col", nrows=1, ncols=2, height=3.2, sharey=True)
    panels = [
        (
            axes[0],
            model.df_filename,
            model.filename_session_ids,
            model.filename_break_indices,
            "Filename order",
            f"{int(model.filename_session_ids.max()) + 1} apparent sessions",
        ),
        (
            axes[1],
            model.df_acquisition,
            model.session_ids,
            model.break_indices,
            "Acquisition order (mtime)",
            f"{int(model.session_ids.max()) + 1} physical temperature segments",
        ),
    ]

    for ax, frame_df, session_ids, breaks, xlabel, subtitle in panels:
        idx = np.arange(len(frame_df))
        for sid in range(int(session_ids.max()) + 1):
            mask = session_ids == sid
            color = METHOD_COLOR_LIST[sid % len(METHOD_COLOR_LIST)]
            ax.plot(
                idx[mask],
                frame_df.loc[mask, "T_mean"],
                "-",
                color=color,
                linewidth=1.3 if mask.sum() > 2 else 0,
                marker="o" if mask.sum() <= 8 else None,
                markersize=3,
                label=f"S{sid} (n={int(mask.sum())})",
            )
        for brk in breaks:
            ax.axvline(int(brk) + 0.5, color="#555555", linewidth=0.5, alpha=0.4)
        ax.set_xlabel(xlabel)
        ax.set_title(subtitle)
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)

    axes[0].set_ylabel("Frame mean temperature [deg C]")
    axes[1].legend(loc="lower right", fontsize=7)

    if save_fn and save_path:
        save_fn(fig, save_path)
    return fig


def plot_robust_temperature_curve(
    model: SessionModel,
    *,
    save_path: str | Path | None = None,
    save_fn=None,
) -> plt.Figure:
    """Plot acquisition-order mean/median/robust temperature curves."""
    df = model.df_acquisition.reset_index(drop=True)
    x = np.arange(len(df))
    fig, ax = make_figure("double_col", height=3.5)

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
        ax.text(
            (start + end) / 2,
            0.96,
            f"S{int(sid)} n={len(group)}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7,
            color="#333333",
        )

    if {"T_q05", "T_q95"}.issubset(df.columns):
        ax.fill_between(
            x,
            df["T_q05"],
            df["T_q95"],
            color="#999999",
            alpha=0.12,
            linewidth=0,
            label="Frame central 90% pixel band",
        )

    ax.plot(x, df["T_mean"], color="#777777", linewidth=0.9, label="Mean")
    if "T_median" in df.columns:
        ax.plot(x, df["T_median"], color=METHOD_COLOR_LIST[0], linewidth=1.2, label="Median")
    if "T_robust" in df.columns:
        ax.plot(
            x,
            df["T_robust"],
            color=METHOD_COLOR_LIST[1],
            linewidth=1.2,
            linestyle="--",
            label="5-95% trimmed mean",
        )

    for brk in model.break_indices:
        ax.axvline(int(brk) + 0.5, color="#333333", linestyle="--", linewidth=0.7)

    ax.set_xlabel("Frame index (acquisition order)")
    ax.set_ylabel("Temperature [deg C]")
    ax.set_title("Robust Temperature Timeline and Main Session")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)

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
    """Visualize acquisition-order raster facts without treating shifts as truth."""
    vals = sorted(int(v) for v in valid_coords)
    known = {tuple(map(int, coord)) for coord in (known_missing or [])}
    df = df_acquisition.sort_values(["acquisition_order", "file"]).copy()
    main = df[df["session"].eq(main_session)]
    non_main = df[~df["session"].eq(main_session)]
    r0 = df[df["R"].astype(int).eq(0)]

    fig, axes = make_figure("double_col", nrows=1, ncols=2, height=3.9)
    ax_map, ax_time = axes

    xx, yy = np.meshgrid(vals, vals)
    ax_map.scatter(
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
        ax_map.plot(
            ordered["X"],
            ordered["Y"],
            color="#777777",
            linewidth=0.6,
            alpha=0.45,
            zorder=2,
        )

    if not main.empty:
        sc = ax_map.scatter(
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
        fig.colorbar(sc, ax=ax_map, fraction=0.046, pad=0.03).set_label(
            "Acquisition order"
        )
        first = main.iloc[0]
        last = main.iloc[-1]
        ax_map.scatter(
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
        ax_map.scatter(
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
        ax_map.scatter(
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
        ax_map.scatter(
            missing_x,
            missing_y,
            marker="x",
            s=52,
            color=METHOD_COLOR_LIST[2],
            linewidth=1.2,
            label="Known missing",
            zorder=6,
        )

    ax_map.set_xticks(vals)
    ax_map.set_yticks(vals)
    ax_map.set_xlabel("Command X [um]")
    ax_map.set_ylabel("Command Y [um]")
    ax_map.set_title("(a) Raster grid in acquisition order")
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.grid(alpha=0.2, linewidth=0.5)
    ax_map.legend(loc="upper left", fontsize=6.5, frameon=True, framealpha=0.9)

    for sid, group in df.groupby("session", sort=True):
        start = int(group["acquisition_order"].min())
        end = int(group["acquisition_order"].max())
        color = METHOD_COLOR_LIST[1] if int(sid) == int(main_session) else METHOD_COLOR_LIST[2]
        ax_time.axvspan(
            start - 0.5,
            end + 0.5,
            color=color,
            alpha=0.08 if int(sid) == int(main_session) else 0.12,
            linewidth=0,
        )
    ax_time.plot(
        df["acquisition_order"],
        df["X"],
        color=METHOD_COLOR_LIST[0],
        linewidth=1.0,
        label="Command X",
    )
    ax_time.step(
        df["acquisition_order"],
        df["Y"],
        where="post",
        color=METHOD_COLOR_LIST[2],
        linewidth=1.0,
        label="Command Y",
    )
    ax_time.scatter(
        non_main["acquisition_order"],
        non_main["X"],
        facecolors="none",
        edgecolors=METHOD_COLOR_LIST[2],
        s=18,
        linewidth=0.8,
        label="Other-session X",
    )
    ax_time.set_xlabel("Acquisition order (mtime)")
    ax_time.set_ylabel("Command coordinate [um]")
    ax_time.set_title("(b) Command coordinates over time")
    ax_time.set_ylim(min(vals) - 2, max(vals) + 2)
    ax_time.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax_time.legend(loc="upper right", fontsize=7)

    if save_fn and save_path:
        save_fn(fig, save_path)
    return fig


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
        ("Main session frames", f"session {main_session}: {main_frames}", "Default input set for micro-scan SR."),
        ("Session temperature jump scale", jump_text, "Cross-session frames should not be mixed."),
        ("Main-session drift span", f"{main_mean_range:.2f} deg C", "Default input frames stay within this temperature band."),
        ("Coordinate coverage", f"{coord_count}/256 total; {main_coord_count}/256 in main session", "Main session covers the full usable coordinate grid."),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value", "SR use"])
