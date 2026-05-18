"""EP02 raster-path, stage-prior, and alignment-evidence helpers."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from thermal_core.displacement import coordinate_to_shift
from thermal_core.plotting import METHOD_COLOR_LIST, make_figure, savefig_academic


EP02_DIRNAME = "ep02_displacement_calibration"


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_stage_config(project_root: Path) -> dict:
    return load_json(project_root / "configs" / "stage_calibration.json")


def load_frame_audit(project_root: Path) -> pd.DataFrame:
    path = project_root / "output" / "ep01_data_processing" / "frame_audit.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run EP01 before building EP02."
        )
    return pd.read_csv(path)


def ep02_output_dir(project_root: Path) -> Path:
    path = project_root / "output" / EP02_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def main_session(frame_audit: pd.DataFrame, session_id: int = 2) -> pd.DataFrame:
    return (
        frame_audit[frame_audit["session"].eq(session_id)]
        .sort_values("acquisition_order")
        .reset_index(drop=True)
    )


def main_raster_r0(frame_audit: pd.DataFrame, session_id: int = 2) -> pd.DataFrame:
    return (
        main_session(frame_audit, session_id=session_id)
        .query("R == 0")
        .sort_values("acquisition_order")
        .reset_index(drop=True)
    )


def add_stage_prior(
    df: pd.DataFrame,
    *,
    theta_deg: float,
    pixel_size_um: float,
) -> pd.DataFrame:
    out = df.copy()
    dx, dy = coordinate_to_shift(
        out["X"].to_numpy(float),
        out["Y"].to_numpy(float),
        theta_deg=theta_deg,
        pixel_size_um=pixel_size_um,
    )
    out["stage_prior_dx_px"] = dx
    out["stage_prior_dy_px"] = dy
    out["stage_prior_phase_x"] = np.mod(dx, 1.0)
    out["stage_prior_phase_y"] = np.mod(dy, 1.0)
    out["phase2_x_bin"] = np.floor(out["stage_prior_phase_x"] * 2.0).astype(int).clip(0, 1)
    out["phase2_y_bin"] = np.floor(out["stage_prior_phase_y"] * 2.0).astype(int).clip(0, 1)
    out["phase2_bin"] = (
        out["phase2_y_bin"].astype(str) + "," + out["phase2_x_bin"].astype(str)
    )
    return out


def phase2_table(prior_df: pd.DataFrame) -> pd.DataFrame:
    table = pd.crosstab(prior_df["phase2_y_bin"], prior_df["phase2_x_bin"])
    return table.reindex(index=[0, 1], columns=[0, 1], fill_value=0)


def plot_raster_acquisition_path(
    frame_audit: pd.DataFrame,
    output_path: Path,
) -> plt.Figure:
    """Plot the main-session raster path and coordinate timelines."""
    raster = main_raster_r0(frame_audit)
    if raster.empty:
        raise ValueError("No session=2, R=0 raster frames found.")

    fig, axes = make_figure("double_col", nrows=2, ncols=2, height=4.2)
    ax_path, ax_cov, ax_x, ax_y = np.asarray(axes).ravel()

    points = raster[["X", "Y"]].to_numpy(float)
    orders = raster["acquisition_order"].to_numpy(float)
    segments = np.stack([points[:-1], points[1:]], axis=1)
    same_row = raster["Y"].to_numpy()[:-1] == raster["Y"].to_numpy()[1:]
    seg_order = (orders[:-1] + orders[1:]) / 2.0

    for mask, linewidth, linestyle, label in [
        (same_row, 1.2, "-", "within-row X step"),
        (~same_row, 1.0, "--", "row transition"),
    ]:
        if np.any(mask):
            collection = LineCollection(
                segments[mask],
                cmap="viridis",
                linewidths=linewidth,
                linestyles=linestyle,
                alpha=0.92,
                label=label,
            )
            collection.set_array(seg_order[mask])
            collection.set_clim(orders.min(), orders.max())
            ax_path.add_collection(collection)

    sc = ax_path.scatter(
        raster["X"],
        raster["Y"],
        c=orders,
        cmap="viridis",
        s=15,
        zorder=3,
        edgecolor="none",
    )
    ax_path.scatter(raster.iloc[0]["X"], raster.iloc[0]["Y"], marker="*", s=95, color="#222222", zorder=4)
    ax_path.scatter(raster.iloc[-1]["X"], raster.iloc[-1]["Y"], marker="X", s=60, color=METHOD_COLOR_LIST[2], zorder=4)
    ax_path.plot([], [], color="#555555", ls="-", lw=1.2, label="X step")
    ax_path.plot([], [], color="#555555", ls="--", lw=1.0, label="row jump")
    ax_path.set_title("Raster Path by Acquisition Order")
    ax_path.set_xlabel("Stage X [um]")
    ax_path.set_ylabel("Stage Y [um]")
    ax_path.set_xlim(-2, 42)
    ax_path.set_ylim(-2, 42)
    ax_path.set_aspect("equal", adjustable="box")
    ax_path.grid(alpha=0.22)
    ax_path.legend(loc="lower right", fontsize=7)
    cbar = fig.colorbar(sc, ax=ax_path, fraction=0.045, pad=0.02)
    cbar.set_label("Acquisition order", fontsize=8)

    x_vals = sorted(raster["X"].unique())
    y_vals = sorted(raster["Y"].unique())
    coverage = (
        raster.pivot_table(index="Y", columns="X", values="file", aggfunc="count")
        .reindex(index=y_vals, columns=x_vals)
        .fillna(0)
    )
    ax_cov.imshow(coverage.to_numpy(), origin="lower", cmap="Greys", vmin=0, vmax=1.2, aspect="auto")
    missing = np.argwhere(coverage.to_numpy() == 0)
    for yi, xi in missing:
        ax_cov.scatter(xi, yi, marker="x", s=35, color=METHOD_COLOR_LIST[2], lw=1.1)
    ax_cov.set_title("R=0 Grid Coverage")
    ax_cov.set_xlabel("Stage X [um]")
    ax_cov.set_ylabel("Stage Y [um]")
    ax_cov.set_xticks(np.arange(len(x_vals))[::3])
    ax_cov.set_xticklabels([int(v) for v in x_vals[::3]])
    ax_cov.set_yticks(np.arange(len(y_vals))[::3])
    ax_cov.set_yticklabels([int(v) for v in y_vals[::3]])
    ax_cov.text(
        0.03,
        0.96,
        f"{int(coverage.to_numpy().sum())}/{coverage.size} R=0 points",
        transform=ax_cov.transAxes,
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.95},
    )

    ax_x.plot(orders, raster["X"], color=METHOD_COLOR_LIST[0], marker="o", ms=2.0, lw=1.0)
    ax_x.set_title("X Timeline")
    ax_x.set_xlabel("Acquisition order")
    ax_x.set_ylabel("X [um]")
    ax_x.grid(axis="y", alpha=0.3)

    ax_y.step(orders, raster["Y"], where="post", color=METHOD_COLOR_LIST[2], lw=1.2)
    ax_y.scatter(orders, raster["Y"], color=METHOD_COLOR_LIST[2], s=7)
    ax_y.set_title("Y Timeline")
    ax_y.set_xlabel("Acquisition order")
    ax_y.set_ylabel("Y [um]")
    ax_y.grid(axis="y", alpha=0.3)

    fig.suptitle("EP02 Raster Acquisition Path", fontsize=11, fontweight="bold")
    savefig_academic(fig, output_path)
    return fig


def raster_summary(frame_audit: pd.DataFrame) -> pd.DataFrame:
    main = main_session(frame_audit)
    raster = main_raster_r0(frame_audit)
    x_transitions = (
        (raster["Y"].to_numpy()[:-1] == raster["Y"].to_numpy()[1:])
        & (np.diff(raster["X"].to_numpy()) > 0)
    )
    row_transitions = raster["Y"].to_numpy()[:-1] != raster["Y"].to_numpy()[1:]
    return pd.DataFrame(
        [
            ("main session frames", len(main), "session=2 only"),
            ("R=0 raster frames", len(raster), "primary step-and-shoot grid"),
            ("within-row X transitions", int(np.sum(x_transitions)), "time-adjacent motion"),
            ("row transitions", int(np.sum(row_transitions)), "Y advance plus X reset"),
            ("unique coordinates", main[["X", "Y"]].drop_duplicates().shape[0], "filename coordinate coverage"),
        ],
        columns=["metric", "value", "interpretation"],
    )


def plot_stage_prior_coverage(
    frame_audit: pd.DataFrame,
    *,
    theta_deg: float,
    pixel_size_um: float,
    output_path: Path,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Plot stage-command prior coverage in detector coordinates and 2x bins."""
    main = main_session(frame_audit)
    unique = main[["X", "Y", "acquisition_order"]].drop_duplicates(["X", "Y"]).copy()
    prior = add_stage_prior(unique, theta_deg=theta_deg, pixel_size_um=pixel_size_um)
    all_prior = add_stage_prior(main, theta_deg=theta_deg, pixel_size_um=pixel_size_um)
    bins = phase2_table(all_prior)

    fig, axes = make_figure("double_col", nrows=1, ncols=2, height=3.5)
    ax_cloud, ax_bins = np.asarray(axes).ravel()

    sc = ax_cloud.scatter(
        prior["stage_prior_dx_px"],
        prior["stage_prior_dy_px"],
        c=prior["acquisition_order"],
        cmap="viridis",
        s=17,
        alpha=0.86,
        edgecolor="none",
    )
    ax_cloud.set_title("Stage Prior Coverage in Detector Coordinates")
    ax_cloud.set_xlabel("prior dx [detector px]")
    ax_cloud.set_ylabel("prior dy, y-up [detector px]")
    ax_cloud.set_aspect("equal", adjustable="box")
    ax_cloud.grid(alpha=0.25)
    ax_cloud.axhline(0, color="#777777", lw=0.6)
    ax_cloud.axvline(0, color="#777777", lw=0.6)
    cbar = fig.colorbar(sc, ax=ax_cloud, fraction=0.045, pad=0.02)
    cbar.set_label("Acquisition order", fontsize=8)

    image = ax_bins.imshow(bins.to_numpy(), origin="lower", cmap="viridis", vmin=0)
    for y in range(2):
        for x in range(2):
            ax_bins.text(x, y, int(bins.loc[y, x]), ha="center", va="center", color="white", fontweight="bold")
    ax_bins.set_title("2x Phase Bin Coverage")
    ax_bins.set_xlabel("phase-x bin")
    ax_bins.set_ylabel("phase-y bin")
    ax_bins.set_xticks([0, 1])
    ax_bins.set_yticks([0, 1])
    ax_bins.set_xticklabels(["[0, 0.5)", "[0.5, 1)"])
    ax_bins.set_yticklabels(["[0, 0.5)", "[0.5, 1)"])
    fig.colorbar(image, ax=ax_bins, fraction=0.045, pad=0.02, label="frame count")

    fig.suptitle(
        f"Stage-Command Prior: theta={theta_deg:.1f} deg, pitch={pixel_size_um:.1f} um/pixel",
        fontsize=11,
        fontweight="bold",
    )
    savefig_academic(fig, output_path)
    return fig, bins


def stage_prior_summary(
    frame_audit: pd.DataFrame,
    *,
    theta_deg: float,
    pixel_size_um: float,
) -> pd.DataFrame:
    main = main_session(frame_audit)
    prior = add_stage_prior(main, theta_deg=theta_deg, pixel_size_um=pixel_size_um)
    return pd.DataFrame(
        [
            ("theta_deg", theta_deg, "configured stage-to-detector rotation"),
            ("pixel_size_um", pixel_size_um, "detector sampling pitch"),
            ("dx_span_px", float(prior["stage_prior_dx_px"].max() - prior["stage_prior_dx_px"].min()), "stage-prior x coverage"),
            ("dy_span_px", float(prior["stage_prior_dy_px"].max() - prior["stage_prior_dy_px"].min()), "stage-prior y-up coverage"),
            ("phase2_bins_nonempty", int((phase2_table(prior).to_numpy() > 0).sum()), "2x phase bins represented"),
        ],
        columns=["metric", "value", "interpretation"],
    )


def load_ep02_tables(output_dir: Path) -> dict[str, pd.DataFrame]:
    required = {
        "gap": "coordinate_pair_time_gap_audit.csv",
        "time": "time_adjacent_method_measurements.csv",
        "time_summary": "time_adjacent_method_summary.csv",
        "y_summary": "y_coordinate_method_summary.csv",
    }
    tables: dict[str, pd.DataFrame] = {}
    missing = []
    for key, filename in required.items():
        path = output_dir / filename
        if path.exists():
            tables[key] = pd.read_csv(path)
        else:
            missing.append(filename)
    if missing:
        raise FileNotFoundError(f"Missing EP02 tables in {output_dir}: {missing}")
    return tables


def small_step_metrics(output_dir: Path) -> pd.DataFrame:
    tables = load_ep02_tables(output_dir)
    time = tables["time"]
    y_summary = tables["y_summary"]
    gap = tables["gap"]

    x_hp = time.query(
        "move_type == 'x_step' and method_label == 'highpass_ncc' and fit_ok == True and edge_peak == False"
    ).copy()
    y_hp = y_summary.query("method_label == 'highpass_ncc'").copy()
    y2 = float(y_hp.query("delta_um == 2.0")["median_parallel_px"].iloc[0])
    y4 = float(y_hp.query("delta_um == 4.0")["median_parallel_px"].iloc[0])
    x2 = float(x_hp.query("delta_um == 2.0")["parallel_px"].median())
    x4 = float(x_hp.query("delta_um == 4.0")["parallel_px"].median())
    x2_ref = float(x_hp.query("delta_um == 2.0")["ref_mag_px"].median())
    y2_ref = float(y_hp.query("delta_um == 2.0")["median_ref_mag_px"].iloc[0])
    y4_ref = float(y_hp.query("delta_um == 4.0")["median_ref_mag_px"].iloc[0])

    rows = [
        ("X time-adjacent gap median", float(gap.query("scan_axis == 'x'")["order_gap"].median()), "frames", "valid local smoke test"),
        ("Y coordinate-neighbor gap median", float(gap.query("scan_axis == 'y'")["order_gap"].median()), "frames", "not time-adjacent"),
        ("X 2um visible projection", x2, "px", "local NCC response"),
        ("X 2um nominal prior", x2_ref, "px", "stage prior magnitude"),
        ("X 4um/2um visible ratio", x4 / x2, "ratio", "short-time linearity check"),
        ("Y 2um visible projection", y2, "px", "contaminated coordinate-neighbor result"),
        ("Y 4um nominal prior", y4_ref, "px", "stage prior magnitude"),
        ("Y 4um/2um visible ratio", y4 / y2, "ratio", "fails calibration monotonicity"),
        ("Y 2um nominal prior", y2_ref, "px", "stage prior magnitude"),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "unit", "interpretation"])


def plot_small_step_diagnostics(output_dir: Path, output_path: Path) -> tuple[plt.Figure, pd.DataFrame]:
    tables = load_ep02_tables(output_dir)
    gap = tables["gap"].copy()
    time = tables["time"].copy()
    y_summary = tables["y_summary"].copy()
    metrics = small_step_metrics(output_dir)

    x_hp = time.query(
        "move_type == 'x_step' and method_label == 'highpass_ncc' and fit_ok == True and edge_peak == False"
    ).copy()
    y_hp = y_summary.query("method_label == 'highpass_ncc'").copy()

    fig, axes = make_figure("double_col", nrows=1, ncols=3, height=3.3)
    ax_gap, ax_x, ax_y = np.asarray(axes).ravel()

    rng = np.random.default_rng(12)
    for idx, (axis, label, color) in enumerate(
        [("x", "X coord pairs", METHOD_COLOR_LIST[0]), ("y", "Y coord pairs", METHOD_COLOR_LIST[2])]
    ):
        values = gap.query("scan_axis == @axis")["order_gap"].to_numpy(float)
        jitter = rng.normal(0, 0.035, values.size)
        ax_gap.scatter(np.full(values.size, idx) + jitter, values, s=10, alpha=0.42, color=color, edgecolor="none")
        ax_gap.plot([idx - 0.24, idx + 0.24], [np.median(values), np.median(values)], color="#222222", lw=1.2)
        ax_gap.text(idx, np.median(values) + 1.1, f"med={np.median(values):.0f}", ha="center", fontsize=8)
    ax_gap.set_xticks([0, 1])
    ax_gap.set_xticklabels(["X", "Y"])
    ax_gap.set_ylabel("Acquisition gap [frames]")
    ax_gap.set_title("Coordinate Neighbor Time Gap")
    ax_gap.set_ylim(0, max(32, gap["order_gap"].max() + 2))
    ax_gap.grid(axis="y", alpha=0.3)

    for delta, color, marker in [(2.0, METHOD_COLOR_LIST[0], "o"), (4.0, METHOD_COLOR_LIST[1], "s"), (6.0, METHOD_COLOR_LIST[3], "^")]:
        subset = x_hp[x_hp["delta_um"].eq(delta)]
        if subset.empty:
            continue
        x = subset["ref_mag_px"].to_numpy(float)
        y = subset["parallel_px"].to_numpy(float)
        ax_x.scatter(x, y, s=13, color=color, marker=marker, alpha=0.42, edgecolor="none", label=f"{delta:.0f} um")
        ax_x.scatter(np.median(x), np.median(y), s=55, color=color, marker=marker, edgecolor="#222222", linewidth=0.5)
    lim = max(0.7, float(x_hp["ref_mag_px"].max()) * 1.1)
    ax_x.plot([0, lim], [0, lim], color="#222222", ls="--", lw=0.9, label="nominal")
    ax_x.set_xlim(0, lim)
    ax_x.set_ylim(0, lim)
    ax_x.set_title("X Time-Adjacent NCC Projection")
    ax_x.set_xlabel("stage-prior magnitude [px]")
    ax_x.set_ylabel("visible projection [px]")
    ax_x.grid(alpha=0.25)
    ax_x.legend(loc="upper left", fontsize=7)

    methods = ["raw_ncc", "highpass_ncc", "gradient_ncc"]
    labels = ["raw", "high-pass", "gradient"]
    xpos = np.arange(len(methods))
    width = 0.34
    for offset, delta, color in [(-width / 2, 2.0, METHOD_COLOR_LIST[2]), (width / 2, 4.0, METHOD_COLOR_LIST[4])]:
        values = [
            float(y_summary.query("method_label == @method and delta_um == @delta")["median_projection_ratio"].iloc[0])
            for method in methods
        ]
        bars = ax_y.bar(xpos + offset, values, width=width, color=color, alpha=0.75, label=f"Y {delta:.0f} um")
        for bar, value in zip(bars, values):
            ax_y.text(bar.get_x() + bar.get_width() / 2, value + 0.04, f"{value:.2f}", ha="center", va="bottom", fontsize=7)
    ax_y.axhline(1.0, color="#333333", ls="--", lw=0.8)
    ax_y.axhline(2.0, color="#333333", ls=":", lw=0.8)
    ax_y.set_xticks(xpos)
    ax_y.set_xticklabels(labels)
    ax_y.set_ylabel("projection / 2um nominal")
    ax_y.set_title("Y Coordinate-Pair Smoke Test")
    ax_y.set_ylim(0, 2.35)
    ax_y.grid(axis="y", alpha=0.3)
    ax_y.legend(loc="upper right", fontsize=7)

    fig.suptitle("Small-Step Diagnostics Are Local Smoke Tests, Not SR Feasibility Proofs", fontsize=11, fontweight="bold")
    savefig_academic(fig, output_path)
    return fig, metrics


def load_alignment_comparison(project_root: Path) -> tuple[pd.DataFrame, str]:
    """Load EP05 alignment scores if present, otherwise return an EP02 proxy."""
    ep05_path = project_root / "output" / "ep05_alignment_sr_capacity" / "alignment_method_summary.csv"
    if ep05_path.exists():
        df = pd.read_csv(ep05_path)
        labels = {
            "no_alignment": "No alignment",
            "old_stage_model": "Stage prior only",
            "filename_affine_fit": "Filename affine prior",
            "data_driven_ncc_init": "Data-driven NCC init",
            "data_driven_contour_refined": "Data-driven contour refined",
        }
        order = {
            "no_alignment": 0,
            "old_stage_model": 1,
            "filename_affine_fit": 2,
            "data_driven_ncc_init": 3,
            "data_driven_contour_refined": 4,
        }
        out = df.copy()
        out["display_label"] = out["method"].map(labels).fillna(out["method"])
        out["plot_order"] = out["method"].map(order).fillna(99)
        out = out.sort_values("plot_order").reset_index(drop=True)
        return out, "ep05_alignment_sr_capacity"

    output_dir = ep02_output_dir(project_root)
    metrics = small_step_metrics(output_dir)
    prior_error = float(metrics.query("metric == 'X 2um nominal prior'")["value"].iloc[0] - metrics.query("metric == 'X 2um visible projection'")["value"].iloc[0])
    proxy = pd.DataFrame(
        [
            {
                "method": "stage_prior_local_proxy",
                "display_label": "Stage prior local proxy",
                "holdout_chamfer_median_px": abs(prior_error),
                "holdout_chamfer_p90_px": abs(prior_error),
                "gradient_corr_median": np.nan,
                "gradient_corr_p10": np.nan,
                "shift_norm_median_px": float(metrics.query("metric == 'X 2um nominal prior'")["value"].iloc[0]),
                "shift_norm_p90_px": np.nan,
                "plot_order": 0,
            },
            {
                "method": "data_driven_x_ncc_proxy",
                "display_label": "Data-driven X NCC proxy",
                "holdout_chamfer_median_px": 0.0,
                "holdout_chamfer_p90_px": 0.0,
                "gradient_corr_median": np.nan,
                "gradient_corr_p10": np.nan,
                "shift_norm_median_px": float(metrics.query("metric == 'X 2um visible projection'")["value"].iloc[0]),
                "shift_norm_p90_px": np.nan,
                "plot_order": 1,
            },
        ]
    )
    return proxy, "ep02_lightweight_proxy"


def plot_alignment_comparison(
    project_root: Path,
    output_path: Path,
) -> tuple[plt.Figure, pd.DataFrame, str]:
    summary, source = load_alignment_comparison(project_root)
    fig, axes = make_figure("double_col", nrows=1, ncols=2, height=3.4)
    ax_chamfer, ax_corr = np.asarray(axes).ravel()

    labels = summary["display_label"].to_numpy()
    y = np.arange(len(summary))
    colors = [
        METHOD_COLOR_LIST[2] if "Stage" in label else METHOD_COLOR_LIST[1] if "Data-driven" in label else METHOD_COLOR_LIST[0]
        for label in labels
    ]
    ax_chamfer.barh(y, summary["holdout_chamfer_median_px"], color=colors, alpha=0.78)
    for idx, value in enumerate(summary["holdout_chamfer_median_px"]):
        if np.isfinite(value):
            ax_chamfer.text(value + 0.006, idx, f"{value:.3f}", va="center", fontsize=8)
    ax_chamfer.set_yticks(y)
    ax_chamfer.set_yticklabels(labels)
    ax_chamfer.invert_yaxis()
    ax_chamfer.set_xlabel("median holdout Chamfer [px]")
    ax_chamfer.set_title("Contour Holdout Error")
    ax_chamfer.grid(axis="x", alpha=0.3)

    corr = summary["gradient_corr_median"].to_numpy(float)
    ax_corr.barh(y, corr, color=colors, alpha=0.78)
    for idx, value in enumerate(corr):
        if np.isfinite(value):
            ax_corr.text(value + 0.006, idx, f"{value:.3f}", va="center", fontsize=8)
    ax_corr.set_yticks(y)
    ax_corr.set_yticklabels([])
    ax_corr.invert_yaxis()
    ax_corr.set_xlim(0, 1.03 if np.isfinite(corr).any() else 1)
    ax_corr.set_xlabel("median gradient correlation")
    ax_corr.set_title("Contour/Gradient Agreement")
    ax_corr.grid(axis="x", alpha=0.3)

    fig.suptitle(f"Data-Driven Alignment Evidence ({source})", fontsize=11, fontweight="bold")
    savefig_academic(fig, output_path)

    csv_path = output_path.with_suffix(".csv")
    summary.to_csv(csv_path, index=False)
    return fig, summary, source


def alignment_improvement_summary(summary: pd.DataFrame) -> pd.DataFrame:
    by_method = summary.set_index("method")
    rows = []
    if {"old_stage_model", "data_driven_contour_refined"}.issubset(by_method.index):
        stage = by_method.loc["old_stage_model"]
        data = by_method.loc["data_driven_contour_refined"]
        rows.append(
            (
                "data-driven vs stage-prior chamfer reduction",
                100.0 * (stage["holdout_chamfer_median_px"] - data["holdout_chamfer_median_px"]) / stage["holdout_chamfer_median_px"],
                "percent",
                "lower holdout contour error is better",
            )
        )
        rows.append(
            (
                "data-driven vs stage-prior gradient-corr gain",
                data["gradient_corr_median"] - stage["gradient_corr_median"],
                "correlation",
                "higher gradient agreement is better",
            )
        )
    if {"no_alignment", "data_driven_contour_refined"}.issubset(by_method.index):
        no_align = by_method.loc["no_alignment"]
        data = by_method.loc["data_driven_contour_refined"]
        rows.append(
            (
                "data-driven vs no-alignment chamfer reduction",
                100.0 * (no_align["holdout_chamfer_median_px"] - data["holdout_chamfer_median_px"]) / no_align["holdout_chamfer_median_px"],
                "percent",
                "main-session contour registration gain",
            )
        )
    return pd.DataFrame(rows, columns=["metric", "value", "unit", "interpretation"])


def decision_table() -> pd.DataFrame:
    rows = [
        {
            "evidence": "Stage/filename coordinate prior",
            "use_for": "coverage planning, initialization, regularization",
            "do_not_use_for": "alignment truth or success metric",
            "reason": "commanded X/Y is metadata; real frame alignment must be checked in the image data",
        },
        {
            "evidence": "Data-driven contour/NCC alignment",
            "use_for": "alignment anchor and quality gate before 2x contour-level SR",
            "do_not_use_for": "replacing the physical coordinate system",
            "reason": "holdout contour and gradient scores directly test image agreement",
        },
        {
            "evidence": "X time-adjacent small steps",
            "use_for": "local direction and short-time linearity smoke test",
            "do_not_use_for": "global SR feasibility claim or absolute stage-amplitude truth",
            "reason": "these pairs are acquisition-adjacent but only probe a local response under one ROI/preprocess choice",
        },
        {
            "evidence": "Y coordinate-adjacent pairs",
            "use_for": "raster-path failure diagnosis and coordinate metadata",
            "do_not_use_for": "Y displacement calibration",
            "reason": "fixed-X Y neighbors are separated by about one raster row, so thermal evolution contaminates NCC",
        },
        {
            "evidence": "AVI continuous scans",
            "use_for": "auxiliary direction and naming sanity check",
            "do_not_use_for": "SR input or high-precision theta replacement",
            "reason": "AVI is rendered 8-bit video with many duplicate frames, not raw temperature matrices",
        },
    ]
    return pd.DataFrame(rows)


def write_decision_table(output_dir: Path) -> pd.DataFrame:
    table = decision_table()
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "ep02_alignment_evidence_decision_table.csv", index=False)
    return table
