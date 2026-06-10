"""EP02 raster-path, stage-prior, and alignment-evidence helpers."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from thermal_core.displacement import coordinate_to_shift
from thermal_core.plotting import METHOD_COLOR_LIST, make_figure, savefig_academic, setup_academic_style


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


def _bool_mask(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float).ne(0.0)
    return series.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def raw_main_session(frame_audit: pd.DataFrame, session_id: int = 2) -> pd.DataFrame:
    """Return the raw physical main temperature segment before repeat exclusion."""
    return (
        frame_audit[frame_audit["session"].eq(session_id)]
        .sort_values("acquisition_order")
        .reset_index(drop=True)
    )


def clean_sr_input(frame_audit: pd.DataFrame, session_id: int = 2) -> pd.DataFrame:
    """Return the repeat-excluded default SR input set.

    EP01 defines `is_sr_usable` as the primary downstream gate.  The older
    `is_main_session` flag is now a compatibility alias for the same clean set.
    """
    for column in ("is_sr_usable", "is_clean_main_session", "is_main_session"):
        if column in frame_audit.columns:
            return (
                frame_audit[_bool_mask(frame_audit[column])]
                .sort_values("acquisition_order")
                .reset_index(drop=True)
            )
    return (
        frame_audit[frame_audit["session"].eq(session_id) & frame_audit["R"].astype(int).eq(0)]
        .sort_values("acquisition_order")
        .reset_index(drop=True)
    )


def main_session(frame_audit: pd.DataFrame, session_id: int = 2) -> pd.DataFrame:
    """Compatibility wrapper: EP02's default main input is clean SR input."""
    return clean_sr_input(frame_audit, session_id=session_id)


def main_raster_r0(frame_audit: pd.DataFrame, session_id: int = 2) -> pd.DataFrame:
    return clean_sr_input(frame_audit, session_id=session_id)


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
    """Plot the main-session raster coordinate timelines."""
    raster = main_raster_r0(frame_audit)
    if raster.empty:
        raise ValueError("No clean SR raster frames found; expected is_sr_usable == True.")

    fig, axes = make_figure("double_col", nrows=1, ncols=2, height=2.8)
    ax_x, ax_y = np.asarray(axes).ravel()

    orders = raster["acquisition_order"].to_numpy(float)

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

    savefig_academic(fig, output_path)
    return fig


def raster_summary(frame_audit: pd.DataFrame) -> pd.DataFrame:
    raw_main = raw_main_session(frame_audit)
    main = clean_sr_input(frame_audit)
    raster = main_raster_r0(frame_audit)
    x_transitions = (
        (raster["Y"].to_numpy()[:-1] == raster["Y"].to_numpy()[1:])
        & (np.diff(raster["X"].to_numpy()) > 0)
    )
    row_transitions = raster["Y"].to_numpy()[:-1] != raster["Y"].to_numpy()[1:]
    return pd.DataFrame(
        [
            ("default main input frames", len(main), "is_sr_usable=True clean input"),
            ("raw main session frames", len(raw_main), "session=2 before repeat exclusion"),
            ("clean SR input frames", len(main), "is_sr_usable=True after repeat exclusion"),
            ("R=0 raster frames", len(raster), "clean primary step-and-shoot grid"),
            ("within-row X transitions", int(np.sum(x_transitions)), "time-adjacent motion"),
            ("row transitions", int(np.sum(row_transitions)), "Y advance plus X reset"),
            ("unique coordinates", main[["X", "Y"]].drop_duplicates().shape[0], "clean coordinate coverage"),
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
    max_val = bins.to_numpy().max()
    for y in range(2):
        for x in range(2):
            val = int(bins.loc[y, x])
            text_color = "black" if val > max_val * 0.5 else "white"
            ax_bins.text(x, y, val, ha="center", va="center", color=text_color, fontweight="bold")
    ax_bins.set_title("2x Phase Bin Coverage")
    ax_bins.set_xlabel("phase-x bin")
    ax_bins.set_ylabel("phase-y bin")
    ax_bins.set_xticks([0, 1])
    ax_bins.set_yticks([0, 1])
    ax_bins.set_xticklabels(["[0, 0.5)", "[0.5, 1)"])
    ax_bins.set_yticklabels(["[0, 0.5)", "[0.5, 1)"])
    fig.colorbar(image, ax=ax_bins, fraction=0.045, pad=0.02, label="frame count")

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


def stage_prior_contract_table(
    frame_audit: pd.DataFrame,
    *,
    theta_deg: float,
    pixel_size_um: float,
    max_rows: int = 10,
) -> pd.DataFrame:
    main = main_session(frame_audit)
    cols = ["acquisition_order", "file", "X", "Y", "R"]
    prior = add_stage_prior(main[cols], theta_deg=theta_deg, pixel_size_um=pixel_size_um)
    prior = prior.sort_values("acquisition_order").head(max_rows).copy()
    prior["stage_prior_dx_px"] = prior["stage_prior_dx_px"].round(4)
    prior["stage_prior_dy_px"] = prior["stage_prior_dy_px"].round(4)
    prior["stage_prior_phase_x"] = prior["stage_prior_phase_x"].round(4)
    prior["stage_prior_phase_y"] = prior["stage_prior_phase_y"].round(4)
    prior["contract"] = "prior/init/regularization only"
    return prior[
        [
            "acquisition_order",
            "file",
            "X",
            "Y",
            "R",
            "stage_prior_dx_px",
            "stage_prior_dy_px",
            "stage_prior_phase_x",
            "stage_prior_phase_y",
            "phase2_bin",
            "contract",
        ]
    ]


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

    # Disable constrained layout temporarily so we can manually adjust wspace
    setup_academic_style()
    _cl_backup = plt.rcParams.get("figure.constrained_layout.use", False)
    plt.rcParams["figure.constrained_layout.use"] = False

    fig, axes = make_figure("double_col", nrows=1, ncols=3, height=3.0, constrained_layout=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    plt.rcParams["figure.constrained_layout.use"] = _cl_backup

    ax_gap, ax_x, ax_y = np.asarray(axes).ravel()

    # Set box aspect to 1 for all subplots to make them square
    for ax in [ax_gap, ax_x, ax_y]:
        ax.set_box_aspect(1)

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
    ax_gap.set_title("(a) Coordinate Neighbor Time Gap")
    ax_gap.set_ylim(0, max(32, gap["order_gap"].max() + 2))
    ax_gap.grid(axis="y", alpha=0.3)

    # Filter to only planned design steps (2um and 4um)
    x_hp = x_hp[x_hp["delta_um"].isin([2.0, 4.0])].copy()

    for delta, color, marker in [(2.0, METHOD_COLOR_LIST[0], "o"), (4.0, METHOD_COLOR_LIST[1], "s")]:
        subset = x_hp[x_hp["delta_um"].eq(delta)]
        if subset.empty:
            continue
        x = subset["ref_mag_px"].to_numpy(float)
        y = subset["parallel_px"].to_numpy(float)
        ax_x.scatter(x, y, s=13, color=color, marker=marker, alpha=0.42, edgecolor="none", label=f"{delta:.0f} um")
        ax_x.scatter(np.median(x), np.median(y), s=55, color=color, marker=marker, edgecolor="#222222", linewidth=0.5)
    data_max = max(float(x_hp["ref_mag_px"].max()), float(x_hp["parallel_px"].max()))
    lim = max(data_max * 1.15, 0.05)
    ax_x.plot([0, lim], [0, lim], color="#222222", ls="--", lw=0.9, label="nominal")
    ax_x.set_xlim(0, lim)
    ax_x.set_ylim(0, lim)
    ax_x.set_title("(b) X Time-Adjacent NCC Projection")
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
    ax_y.set_title("(c) Y Coordinate-Pair Smoke Test")
    ax_y.set_ylim(0, 2.35)
    ax_y.grid(axis="y", alpha=0.3)
    ax_y.legend(loc="upper right", fontsize=7)

    # Adjust spacing and margins with larger horizontal spacing (wspace) for y-axis labels
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.85, wspace=0.38)

    savefig_academic(fig, output_path)
    return fig, metrics


def load_alignment_comparison(project_root: Path) -> tuple[pd.DataFrame, str]:
    """Load compatible EP05 alignment scores if present, otherwise return an EP02 proxy."""
    ep05_path = project_root / "output" / "ep05_alignment_sr_capacity" / "alignment_method_summary.csv"
    expected_frames = len(clean_sr_input(load_frame_audit(project_root)))
    if ep05_path.exists():
        df = pd.read_csv(ep05_path)
        frame_counts = set(df["n_frames"].dropna().astype(int)) if "n_frames" in df.columns else set()
        if frame_counts == {expected_frames}:
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
    source = "ep02_lightweight_proxy"
    if ep05_path.exists():
        stale_counts = ",".join(str(v) for v in sorted(frame_counts)) if frame_counts else "missing"
        source = f"ep02_lightweight_proxy_ep05_n_frames_{stale_counts}_expected_{expected_frames}"
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
    return proxy, source


def plot_alignment_comparison(
    project_root: Path,
    output_path: Path,
) -> tuple[plt.Figure, pd.DataFrame, str]:
    summary, source = load_alignment_comparison(project_root)

    # Disable constrained layout temporarily so we can manually adjust wspace
    setup_academic_style()
    _cl_backup = plt.rcParams.get("figure.constrained_layout.use", False)
    plt.rcParams["figure.constrained_layout.use"] = False

    fig, axes = make_figure("double_col", nrows=1, ncols=2, height=3.4, constrained_layout=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    plt.rcParams["figure.constrained_layout.use"] = _cl_backup

    ax_chamfer, ax_corr = np.asarray(axes).ravel()

    # Set box aspect to 1 for both subplots to make them square
    for ax in [ax_chamfer, ax_corr]:
        ax.set_box_aspect(1)

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
    ax_chamfer.set_title("(a) Contour Holdout Error")
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
    ax_corr.set_title("(b) Contour/Gradient Agreement")
    ax_corr.grid(axis="x", alpha=0.3)

    # Adjust spacing and margins with larger horizontal spacing (wspace) and enough left margin for labels
    fig.subplots_adjust(left=0.24, right=0.96, bottom=0.15, top=0.85, wspace=0.32)

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
            "reason": "commanded X/Y is metadata; alignment evidence must be supported by the image data",
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


def _read_required_csv(output_dir: Path, filename: str) -> pd.DataFrame:
    path = output_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing EP02 table: {path}")
    return pd.read_csv(path)


def _format_ci(lower: float, upper: float, digits: int = 2) -> str:
    return f"[{lower:.{digits}f}, {upper:.{digits}f}]"


def time_adjacent_method_comparison(output_dir: Path) -> pd.DataFrame:
    """Summarize true time-adjacent X steps and row transitions by method."""
    summary = _read_required_csv(output_dir, "time_adjacent_method_summary.csv")
    method_order = {"raw_ncc": 0, "highpass_ncc": 1, "gradient_ncc": 2, "phase_corr": 3}
    move_order = {"x_step": 0, "row_transition": 1}
    method_label = {
        "raw_ncc": "raw NCC",
        "highpass_ncc": "high-pass NCC",
        "gradient_ncc": "gradient NCC",
        "phase_corr": "phase correlation",
    }
    move_label = {
        "x_step": "X-step, acquisition gap=1",
        "row_transition": "row transition, Y advance + X reset",
    }
    out = summary.copy()
    out["method_sort"] = out["method_label"].map(method_order).fillna(99)
    out["move_sort"] = out["move_type"].map(move_order).fillna(99)
    out = out.sort_values(["move_sort", "method_sort"]).reset_index(drop=True)
    out["evidence window"] = out["move_type"].map(move_label).fillna(out["move_type"])
    out["method"] = out["method_label"].map(method_label).fillna(out["method_label"])
    out["pairs"] = out["n_pairs"].astype(int)
    out["prior mag [px]"] = out["median_ref_mag_px"].round(4)
    out["visible/prior projection"] = out["median_projection_ratio"].round(3)
    out["RMS vs prior [px]"] = out["rms_ref_residual_px"].round(4)
    out["perp median [px]"] = out["median_abs_perpendicular_px"].round(4)
    out["median score"] = out["median_peak_score"].round(4)
    out["use boundary"] = np.where(
        out["move_type"].eq("x_step"),
        "local direction/linearity smoke test only",
        "not a clean Y-only calibration pair",
    )
    return out[
        [
            "evidence window",
            "method",
            "pairs",
            "prior mag [px]",
            "visible/prior projection",
            "RMS vs prior [px]",
            "perp median [px]",
            "median score",
            "use boundary",
        ]
    ]


def y_coordinate_failure_table(output_dir: Path) -> pd.DataFrame:
    """Summarize coordinate-neighbor Y-only failure across preprocessing methods."""
    summary = _read_required_csv(output_dir, "y_coordinate_method_summary.csv")
    method_order = {"raw_ncc": 0, "highpass_ncc": 1, "gradient_ncc": 2}
    method_label = {
        "raw_ncc": "raw NCC",
        "highpass_ncc": "high-pass NCC",
        "gradient_ncc": "gradient NCC",
    }
    rows = []
    for method, group in summary.groupby("method_label"):
        by_delta = group.set_index("delta_um")
        if not {2.0, 4.0}.issubset(by_delta.index):
            continue
        two = by_delta.loc[2.0]
        four = by_delta.loc[4.0]
        rows.append(
            {
                "method_sort": method_order.get(method, 99),
                "method": method_label.get(method, method),
                "pairs 2um / 4um": f"{int(two['n_pairs'])} / {int(four['n_pairs'])}",
                "2um projection/prior": round(float(two["median_projection_ratio"]), 3),
                "4um projection/prior": round(float(four["median_projection_ratio"]), 3),
                "visible 4um/2um": round(float(four["median_parallel_px"] / two["median_parallel_px"]), 3),
                "expected 4um/2um": 2.0,
                "RMS 2um / 4um [px]": f"{float(two['rms_ref_residual_px']):.4f} / {float(four['rms_ref_residual_px']):.4f}",
                "interpretation": "stable failure; do not use as Y calibration",
            }
        )
    out = pd.DataFrame(rows).sort_values("method_sort").drop(columns="method_sort")
    return out.reset_index(drop=True)


def avi_theta_compact_table(output_dir: Path, *, reference_theta_deg: float = 47.6) -> pd.DataFrame:
    """Format AVI theta estimates as auxiliary validation evidence."""
    if not (output_dir / "avi_theta_summary.csv").exists():
        return pd.DataFrame(
            [{"note": "AVI theta artifacts not available; raw TXT EP02 cache is still valid."}]
        )
    summary = _read_required_csv(output_dir, "avi_theta_summary.csv")
    method_order = {"gradient": 0, "highpass": 1}
    source_order = {"x-only": 0, "y-only": 1, "combined": 2}
    out = summary.copy()
    out["method_sort"] = out["method"].map(method_order).fillna(99)
    out["source_sort"] = out["source"].map(source_order).fillna(99)
    out = out.sort_values(["method_sort", "source_sort"]).reset_index(drop=True)
    out["theta mean [deg]"] = out["mean_deg"].round(2)
    out["theta median [deg]"] = out["median_deg"].round(2)
    out["95% CI [deg]"] = [
        _format_ci(lo, hi, digits=2)
        for lo, hi in zip(out["ci_lower_deg"].to_numpy(float), out["ci_upper_deg"].to_numpy(float))
    ]
    out["range [deg]"] = [
        _format_ci(lo, hi, digits=2)
        for lo, hi in zip(out["min_deg"].to_numpy(float), out["max_deg"].to_numpy(float))
    ]
    out[f"covers {reference_theta_deg:.1f} deg"] = out["within_ci_47_6"].map({True: "yes", False: "no"})
    out["use boundary"] = np.where(
        out["source"].eq("combined") & out["method"].eq("gradient"),
        "auxiliary validation; keep config theta",
        "diagnostic subgroup, not config replacement",
    )
    return out[
        [
            "method",
            "source",
            "n",
            "theta mean [deg]",
            "theta median [deg]",
            "95% CI [deg]",
            "range [deg]",
            f"covers {reference_theta_deg:.1f} deg",
            "use boundary",
        ]
    ]


def avi_txt_line_match_table(output_dir: Path) -> pd.DataFrame:
    """Summarize AVI filename-to-TXT line mapping checks for X and Y scans."""
    if not (output_dir / "avi_txt_xline_match_summary.csv").exists() or not (
        output_dir / "avi_txt_yline_match_summary.csv"
    ).exists():
        return pd.DataFrame(
            [{"note": "AVI-TXT line-match artifacts not available; skipped optional naming check."}]
        )
    specs = [
        ("X-scan AVI", "avi_txt_xline_match_summary.csv", "fixed_y", "xN.avi -> TXT fixed Y=N", "expected"),
        ("X-scan AVI", "avi_txt_xline_match_summary.csv", "fixed_x", "xN.avi -> TXT fixed X=N", "rejected"),
        ("Y-scan AVI", "avi_txt_yline_match_summary.csv", "fixed_x", "yN.avi -> TXT fixed X=N", "expected"),
        ("Y-scan AVI", "avi_txt_yline_match_summary.csv", "fixed_y", "yN.avi -> TXT fixed Y=N", "rejected"),
    ]
    rows = []
    for avi_group, filename, mapping, hypothesis, decision in specs:
        df = _read_required_csv(output_dir, filename)
        subset = df[df["mapping"].eq(mapping)]
        if subset.empty:
            continue
        rows.append(
            {
                "AVI group": avi_group,
                "mapping hypothesis": hypothesis,
                "decision": decision,
                "lines": int(subset["avi"].nunique()),
                "contour axis diff [deg]": round(float(subset["contour_axis_diff_to_avi_highpass_deg"].median()), 2),
                "high-pass axis diff [deg]": round(float(subset["highpass_axis_diff_to_avi_median_deg"].median()), 2),
                "gradient axis diff [deg]": round(float(subset["gradient_axis_diff_to_avi_median_deg"].median()), 2),
                "acquisition gap median": round(float(subset["acquisition_gap_median"].median()), 1),
                "interpretation": (
                    "filename mapping is correct"
                    if decision == "expected"
                    else "orthogonal line hypothesis"
                ),
            }
        )
    return pd.DataFrame(rows)


def historical_ncc_failure_audit(output_dir: Path) -> pd.DataFrame:
    """Record why coordinate-adjacent NCC remains bounded diagnostic evidence.

    This intentionally uses the current rebuildable EP02 cache, not obsolete
    early coordinate-adjacent outputs such as theta_estimate.json or linearity.csv.
    """
    gap = _read_required_csv(output_dir, "coordinate_pair_time_gap_audit.csv")
    time_summary = _read_required_csv(output_dir, "time_adjacent_method_summary.csv")
    x_fit = _read_required_csv(output_dir, "time_adjacent_x_step_fit.csv")
    y_failure = y_coordinate_failure_table(output_dir)

    y_gap_median = float(gap.query("scan_axis == 'y'")["order_gap"].median())
    hp_ratio = float(
        y_failure.loc[y_failure["method"].eq("high-pass NCC"), "visible 4um/2um"].iloc[0]
    )
    phase = time_summary[
        time_summary["method_label"].eq("phase_corr") & time_summary["move_type"].eq("x_step")
    ]
    phase_ratio = float(phase["median_projection_ratio"].iloc[0]) if not phase.empty else np.nan
    hp_fit = x_fit[x_fit["method_label"].eq("highpass_ncc")]
    hp_pairs = int(hp_fit["n_pairs"].iloc[0]) if not hp_fit.empty else 0
    rows = [
        {
            "diagnostic boundary": "clean input contract",
            "value": "is_sr_usable=True; no R!=0 repeat frames",
            "current interpretation": "EP02 displacement diagnostics now run on the repeat-excluded 248-frame input",
        },
        {
            "diagnostic boundary": "Y coordinate-neighbor acquisition gap",
            "value": f"median {y_gap_median:.0f} frames",
            "current interpretation": "fixed-X Y neighbors are not time-adjacent and remain contaminated by thermal evolution",
        },
        {
            "diagnostic boundary": "Y high-pass visible 4um/2um",
            "value": f"{hp_ratio:.3f} (expected about 2)",
            "current interpretation": "Y-only coordinate neighbors fail monotonicity even after repeat exclusion",
        },
        {
            "diagnostic boundary": "phase correlation on X tiny steps",
            "value": f"visible/prior projection {phase_ratio:.3f}",
            "current interpretation": "phase correlation degenerates on this subpixel local diagnostic",
        },
        {
            "diagnostic boundary": "high-pass X-step fit pairs",
            "value": f"{hp_pairs} clean time-adjacent X pairs",
            "current interpretation": "usable only as local direction/linearity smoke test, not alignment truth",
        },
        {
            "diagnostic boundary": "obsolete coordinate-adjacent outputs",
            "value": "removed from current cache",
            "current interpretation": "old theta/linearity/repeatability files are not required to rebuild EP02",
        },
    ]
    return pd.DataFrame(rows)
