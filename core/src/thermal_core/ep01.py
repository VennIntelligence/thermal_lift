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
        ("Main-session drift span", f"{main_mean_range:.2f} deg C", "SR alignment should model frames within this temperature band."),
        ("Coordinate coverage", f"{coord_count}/256 total; {main_coord_count}/256 in main session", "Main session covers the full usable coordinate grid."),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value", "SR use"])
