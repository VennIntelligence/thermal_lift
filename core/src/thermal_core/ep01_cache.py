"""EP01 cache builder and loader — notebook reads artifacts; script rebuilds them."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from thermal_core.ep01 import (
    add_sr_input_selection_columns,
    add_robust_temperature_stats,
    build_session_model,
    make_acquisition_order_audit_table,
    make_boundary_jump_table,
    make_bmp_txt_pairing_table,
    make_ep01_summary_table,
    make_frame_audit_contract_table,
    make_missing_coordinate_table,
    make_raster_row_table,
    plot_order_comparison,
    plot_repeat_exclusion_order_comparison,
    plot_robust_temperature_curve,
    plot_session_detection_raster,
    plot_session_detection_timeline,
    plot_session_coverage_heatmaps,
)
from thermal_core.io import audit_all_frames, check_bmp_txt_pairing, build_coord_repeat_map
from thermal_core.notebook_cache import clear_output_dir
from thermal_core.plotting import savefig_academic, setup_academic_style
from thermal_core.viz import plot_coverage_heatmap, plot_temperature_histograms

EP01_CACHE_VERSION = 2

EP01_TABLE_ARTIFACTS = (
    "frame_audit.csv",
    "acquisition_order_audit.csv",
    "sr_data_basis_summary.csv",
    "session_summary.csv",
    "boundary_jump_table.csv",
    "bmp_txt_pairing_detail.csv",
    "coord_repeat_summary.csv",
    "r_distribution.csv",
    "repeat_exclusion_summary.csv",
    "missing_coordinate_table.csv",
    "raster_row_summary.csv",
    "rename_special.csv",
    "frame_audit_contract.csv",
    "matrix_audit_summary.json",
)

EP01_FIGURE_ARTIFACTS = (
    "coordinate_coverage_map.png",
    "frame_temperature_statistics.png",
    "robust_temperature_timeline.png",
    "order_comparison.png",
    "repeat_exclusion_order_comparison.png",
    "session_detection_a.png",
    "session_detection_b.png",
    "session_coordinate_coverage.png",
)

EP01_ARTIFACTS = (*EP01_TABLE_ARTIFACTS, *EP01_FIGURE_ARTIFACTS, "cache_manifest.json")


@dataclass(frozen=True)
class Ep01Cache:
    """Loaded EP01 artifacts for notebook display."""

    output_dir: Path
    df: pd.DataFrame
    pairing: dict
    pairing_detail: pd.DataFrame
    rename_special: pd.DataFrame
    rename_mapping_path: Path
    coord_repeat_summary: pd.DataFrame
    r_distribution: pd.DataFrame
    repeat_exclusion_summary: pd.DataFrame
    missing_coordinate_table: pd.DataFrame
    session_summary: pd.DataFrame
    boundary_jump_table: pd.DataFrame
    summary_table: pd.DataFrame
    frame_audit_contract: pd.DataFrame
    raster_row_summary: pd.DataFrame
    raster_row_mismatches: pd.DataFrame
    acquisition_order_audit: pd.DataFrame
    matrix_audit_summary: dict
    noise_floor_c: float
    valid_coords: set[int]
    coord_config: dict
    manifest: dict

    @property
    def model(self):
        from thermal_core.ep01 import SessionModel

        return build_session_model(self.df)

    def figure_path(self, name: str) -> Path:
        return self.output_dir / name


def _project_root(start: Path | None = None) -> Path:
    root = (start or Path.cwd()).resolve()
    while not (root / "AGENTS.md").exists() and root != root.parent:
        root = root.parent
    if not (root / "AGENTS.md").exists():
        raise FileNotFoundError("Could not locate project root (AGENTS.md missing).")
    return root


def _save_figure(fig, path: Path) -> None:
    savefig_academic(fig, path)


def _markdown_table(table: pd.DataFrame) -> str:
    headers = list(table.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in table.iterrows():
        values = [str(row[col]).replace("\n", " ") for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_audit_report(
    *,
    report_dir: Path,
    pairing: dict,
    pairing_detail: pd.DataFrame,
    model,
    summary_table: pd.DataFrame,
    boundary_jump_table: pd.DataFrame,
    coord_repeat_summary: pd.DataFrame,
    r_distribution: pd.DataFrame,
    repeat_exclusion_summary: pd.DataFrame,
    missing_coordinate_table: pd.DataFrame,
    raster_row_summary: pd.DataFrame,
    frame_audit_contract: pd.DataFrame,
    raw_main_df: pd.DataFrame,
    sr_df: pd.DataFrame,
    all_coords: set[tuple[int, int]],
    valid_coords: set[int],
    noise_floor_c: float,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    main_session = int(model.main_session)
    main_row = model.session_summary.loc[model.session_summary["session"].eq(main_session)].iloc[0]
    raw_main_coords = set(zip(raw_main_df["X"].astype(int), raw_main_df["Y"].astype(int)))
    sr_coords = set(zip(sr_df["X"].astype(int), sr_df["Y"].astype(int)))
    n_sessions = int(model.session_ids.max()) + 1
    frame_shape = (
        int(raw_main_df["rows"].mode().iat[0]) if "rows" in raw_main_df.columns else 480,
        int(raw_main_df["cols"].mode().iat[0]) if "cols" in raw_main_df.columns else 640,
    )
    missing_coords = sorted(
        set((x, y) for x in sorted(valid_coords) for y in sorted(valid_coords)) - all_coords
    )
    boundary_text = (
        _markdown_table(model.boundary_jumps.round(3))
        if not model.boundary_jumps.empty
        else "No detected boundaries."
    )
    boundary_jump_md = (
        _markdown_table(boundary_jump_table.round(3))
        if not boundary_jump_table.empty
        else "No detected boundaries."
    )

    report = f"""# EP01 — SR Data Basis and Main-Session Model

## Scope

EP01 audits the raw LWIR TXT/BMP dataset and turns it into a reproducible input model for micro-scan super-resolution. The goal is not to decide SR success or failure; it is to define which frames can be used together, what time order they have, and how session-level temperature drift constrains reconstruction.

## Executable Summary

{_markdown_table(summary_table)}

## Frame Inventory

All `{pairing['n_txt']}` TXT thermal matrices are readable `{frame_shape[0]} x {frame_shape[1]}` arrays with no NaN/Inf frames, and all have matching BMP companions. TXT remains the numerical input for SR; BMP is retained as same-name visual reference only.

TXT/BMP pairing and rename provenance:

{_markdown_table(pairing_detail)}

Coordinate/repeat coverage:

{_markdown_table(coord_repeat_summary)}

Repeat-ID distribution:

{_markdown_table(r_distribution)}

Repeat-frame exclusion summary:

{_markdown_table(repeat_exclusion_summary)}

The dataset contains `{len(all_coords)}/256` actual coordinates. Missing coordinates are `{missing_coords}`. These gaps are coordinate-level absences, not merely missing `R=0` repeats.

Explicit missing-coordinate table:

{_markdown_table(missing_coordinate_table)}

## Acquisition Order and Sessions

Filename order is not acquisition order. Sorting by renamed filename produces `{int(model.filename_session_ids.max()) + 1}` apparent temperature sessions because repeat and early frames are interleaved with the raster grid. Sorting by file modification time recovers `{n_sessions}` physical temperature segments:

{_markdown_table(model.session_summary.round(3))}

Boundary jumps in acquisition order:

{boundary_text}

Boundary jumps compared with the `{noise_floor_c:.4f}` deg C noise floor:

{boundary_jump_md}

The raw physical main session is session `{main_session}` with `{int(main_row['n_frames'])}` frames. It spans acquisition orders `{int(main_row['first_order'])}` to `{int(main_row['last_order'])}` and covers `{len(raw_main_coords)}/256` coordinates before repeat exclusion. The clean SR input set excludes all `R != 0` repeat frames and contains `{len(sr_df)}` frames across `{len(sr_coords)}/256` coordinates. Its acquisition-order span is `{int(sr_df['acquisition_order'].min())}` to `{int(sr_df['acquisition_order'].max())}` and its mean-temperature span is `{float(sr_df['T_mean'].max() - sr_df['T_mean'].min()):.3f}` deg C.

R=0 raster row-order diagnostic:

{_markdown_table(raster_row_summary)}

## SR Input Rule

Downstream SR should inherit `frame_audit.csv` and use `acquisition_order` plus `is_sr_usable == True` as the frame-selection contract. `session == {main_session}` remains the raw physical temperature segment (`{int(main_row['n_frames'])}` frames), while `is_sr_usable` / the compatibility alias `is_main_session` define the repeat-excluded clean SR input (`{len(sr_df)}` frames). Stage/filename coordinates are useful as command priors for initialization or regularization, but actual alignment must be constrained by image data and later EP04 localization quality gates.

Cross-session frames should not be mixed into one reconstruction pass. The detected session-boundary jumps are `{model.boundary_jumps['abs_delta_mean_C'].median():.2f}` deg C median and `{model.boundary_jumps['abs_delta_mean_C'].max():.2f}` deg C max, which are about `{model.boundary_jumps['abs_delta_mean_C'].median() / noise_floor_c:.0f}x` and `{model.boundary_jumps['abs_delta_mean_C'].max() / noise_floor_c:.0f}x` the `{noise_floor_c:.4f}` deg C noise floor.

`frame_audit.csv` downstream contract:

{_markdown_table(frame_audit_contract)}

## Output Files

Rebuild cache with `uv run python scripts/build_ep01_cache.py`.

- `frame_audit.csv`
- `acquisition_order_audit.csv`
- `sr_data_basis_summary.csv`
- `coordinate_coverage_map.png`
- `frame_temperature_statistics.png`
- `robust_temperature_timeline.png`
- `order_comparison.png`
- `repeat_exclusion_order_comparison.png`
- `session_detection_a.png`
- `session_detection_b.png`
- `session_coordinate_coverage.png`
"""
    (report_dir / "audit_report.md").write_text(report, encoding="utf-8")


def build_ep01_cache(
    *,
    project_root: Path | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    report_dir: Path | None = None,
    force: bool = False,
) -> Ep01Cache:
    """Read raw TXT/BMP once, write EP01 CSV/PNG artifacts and audit report."""
    root = _project_root(project_root)
    data_dir = (data_dir or root / "data" / "data_raw" / "infrared_avi").resolve()
    output_dir = (output_dir or root / "output" / "ep01_data_processing").resolve()
    report_dir = (report_dir or root / "paper" / "reports" / "ep01_data_processing").resolve()
    if force:
        clear_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "cache_manifest.json"
    if not force and manifest_path.exists():
        missing = [name for name in EP01_ARTIFACTS if not (output_dir / name).exists()]
        if not missing:
            return load_ep01_cache(output_dir=output_dir, project_root=root)

    with open(root / "configs" / "coordinate_set.json", encoding="utf-8") as f:
        coord_config = json.load(f)
    with open(root / "configs" / "noise_floor.json", encoding="utf-8") as f:
        noise_floor_c = float(json.load(f)["noise_floor_celsius"])

    valid_coords = set(int(v) for v in coord_config["x_coords_um"])
    valid_coord_list = sorted(valid_coords)
    known_missing = coord_config.get("known_missing_r0", [])
    rename_mapping_path = output_dir / "rename_mapping.csv"

    setup_academic_style()

    df = audit_all_frames(data_dir)
    df = add_robust_temperature_stats(df, data_dir)
    rename_mapping = (
        pd.read_csv(rename_mapping_path) if rename_mapping_path.exists() else pd.DataFrame()
    )
    if not rename_mapping.empty:
        txt_mapping = (
            rename_mapping[rename_mapping["ext"].str.lower().eq(".txt")]
            [["old_name", "new_name"]]
            .rename(columns={"old_name": "source_file", "new_name": "file"})
        )
        df = df.merge(txt_mapping, on="file", how="left")

    model = build_session_model(df)
    df_sorted = add_sr_input_selection_columns(
        model.df_acquisition.copy(),
        raw_main_session=int(model.main_session),
    )

    pairing = check_bmp_txt_pairing(data_dir)
    pairing_detail = make_bmp_txt_pairing_table(
        pairing,
        rename_mapping=rename_mapping,
        rename_mapping_path=rename_mapping_path,
    )
    coord_map = build_coord_repeat_map(df_sorted)
    missing_coordinate_table = make_missing_coordinate_table(
        df_sorted, valid_coord_list, known_missing
    )
    repeat_counts = {k: len(v) for k, v in coord_map.items()}
    coord_repeat_summary = (
        pd.Series(repeat_counts, name="frame_count")
        .value_counts()
        .sort_index(ascending=False)
        .rename_axis("frames_per_coordinate")
        .reset_index(name="n_coordinates")
    )
    r_distribution = (
        df_sorted.groupby("R")
        .size()
        .rename("n_frames")
        .to_frame()
        .join(
            df_sorted.drop_duplicates(["R", "X", "Y"])
            .groupby("R")
            .size()
            .rename("n_unique_coordinates")
        )
        .reset_index()
    )
    repeat_exclusion_summary = (
        df_sorted.groupby(["frame_role", "repeat_exclusion_reason", "sr_exclusion_reason"], dropna=False)
        .size()
        .rename("n_frames")
        .reset_index()
    )
    raster_row_summary = make_raster_row_table(
        df_sorted, valid_coord_list, known_missing, r_value=0
    )
    raster_row_mismatches = raster_row_summary[
        ~raster_row_summary["matches_expected_after_known_missing"]
    ].copy()
    acquisition_order_audit = make_acquisition_order_audit_table(df_sorted)
    boundary_jump_table = make_boundary_jump_table(model.boundary_jumps, noise_floor_c)
    frame_audit_contract = make_frame_audit_contract_table(
        include_source_file="source_file" in df_sorted.columns
    )

    raw_main_df = df_sorted[df_sorted["is_raw_main_session"]].copy()
    sr_df = df_sorted[df_sorted["is_sr_usable"]].copy()
    all_coords = set(zip(df_sorted["X"].astype(int), df_sorted["Y"].astype(int)))
    raw_main_coords = set(zip(raw_main_df["X"].astype(int), raw_main_df["Y"].astype(int)))
    sr_coords = set(zip(sr_df["X"].astype(int), sr_df["Y"].astype(int)))
    frame_shape = (int(df_sorted["rows"].mode().iat[0]), int(df_sorted["cols"].mode().iat[0]))
    summary_table = make_ep01_summary_table(
        n_txt=pairing["n_txt"],
        n_bmp=pairing["n_bmp"],
        n_paired=pairing["n_paired"],
        frame_shape=frame_shape,
        coord_count=len(all_coords),
        main_session=int(model.main_session),
        main_frames=len(raw_main_df),
        main_coord_count=len(raw_main_coords),
        sr_input_frames=len(sr_df),
        sr_coord_count=len(sr_coords),
        boundary_jumps=model.boundary_jumps,
        main_mean_range=float(sr_df["T_mean"].max() - sr_df["T_mean"].min()),
        noise_floor_c=noise_floor_c,
    )

    if rename_mapping.empty:
        rename_special = pd.DataFrame(
            [
                {
                    "status": "rename_mapping.csv missing or empty",
                    "interpretation": (
                        "current files are already in X_Y_R form; "
                        "no raw-name mapping table is available in output"
                    ),
                }
            ]
        )
    else:
        special_old_names = {
            "0200.txt", "0240.txt", "0280.txt", "0400.txt",
            "2000.txt", "2020.txt", "2040.txt", "2060.txt", "2080.txt",
            "2400.txt", "2，400.txt",
            "4000.txt", "4020.txt", "4040.txt", "4060.txt", "4080.txt",
        }
        rename_special = (
            rename_mapping[
                rename_mapping["old_name"].isin(special_old_names)
                | rename_mapping["old_name"].astype(str).str.contains("，", regex=False)
            ][["old_name", "new_name", "X", "Y", "R", "ext"]]
            .sort_values(["ext", "old_name"])
            .reset_index(drop=True)
        )

    matrix_audit_summary = {
        "n_frames": int(len(df_sorted)),
        "rows": int(frame_shape[0]),
        "cols": int(frame_shape[1]),
        "n_nan_frames": int(df_sorted["has_nan"].sum()),
        "n_inf_frames": int(df_sorted["has_inf"].sum()),
        "t_min_global": float(df_sorted["T_min"].min()),
        "t_max_global": float(df_sorted["T_max"].max()),
        "t_mean_min": float(df_sorted["T_mean"].min()),
        "t_mean_max": float(df_sorted["T_mean"].max()),
        "t_median_min": float(df_sorted["T_median"].min()),
        "t_median_max": float(df_sorted["T_median"].max()),
        "acquisition_order_min": int(df_sorted["acquisition_order"].min()),
        "acquisition_order_max": int(df_sorted["acquisition_order"].max()),
    }

    # ── CSV artifacts ──
    df_sorted.to_csv(output_dir / "frame_audit.csv", index=False)
    acquisition_order_audit.to_csv(output_dir / "acquisition_order_audit.csv", index=False)
    summary_table.to_csv(output_dir / "sr_data_basis_summary.csv", index=False)
    model.session_summary.to_csv(output_dir / "session_summary.csv", index=False)
    boundary_jump_table.to_csv(output_dir / "boundary_jump_table.csv", index=False)
    pairing_detail.to_csv(output_dir / "bmp_txt_pairing_detail.csv", index=False)
    coord_repeat_summary.to_csv(output_dir / "coord_repeat_summary.csv", index=False)
    r_distribution.to_csv(output_dir / "r_distribution.csv", index=False)
    repeat_exclusion_summary.to_csv(output_dir / "repeat_exclusion_summary.csv", index=False)
    missing_coordinate_table.to_csv(output_dir / "missing_coordinate_table.csv", index=False)
    raster_row_summary.to_csv(output_dir / "raster_row_summary.csv", index=False)
    rename_special.to_csv(output_dir / "rename_special.csv", index=False)
    frame_audit_contract.to_csv(output_dir / "frame_audit_contract.csv", index=False)
    (output_dir / "matrix_audit_summary.json").write_text(
        json.dumps(matrix_audit_summary, indent=2), encoding="utf-8"
    )

    # ── Figure artifacts ──
    from thermal_core.io import compute_coverage_grid

    def _save(fig, name: str) -> None:
        _save_figure(fig, output_dir / Path(name).name)

    grid = compute_coverage_grid(coord_map, valid_coord_list)
    plot_coverage_heatmap(
        grid, valid_coord_list, save_path="coordinate_coverage_map.png", save_fn=_save
    )
    plot_temperature_histograms(
        df_sorted,
        save_path="frame_temperature_statistics.png",
        save_fn=_save,
    )
    plot_robust_temperature_curve(
        model, save_path="robust_temperature_timeline.png", save_fn=_save
    )
    plot_order_comparison(model, save_path="order_comparison.png", save_fn=_save)
    plot_repeat_exclusion_order_comparison(
        model,
        sr_df,
        save_path="repeat_exclusion_order_comparison.png",
        save_fn=_save,
    )
    plot_session_detection_raster(
        df_sorted,
        valid_coord_list,
        int(model.main_session),
        known_missing,
        save_path="session_detection_a.png",
        save_fn=_save,
    )
    plot_session_detection_timeline(
        df_sorted,
        valid_coord_list,
        int(model.main_session),
        save_path="session_detection_b.png",
        save_fn=_save,
    )
    plot_session_coverage_heatmaps(
        df_sorted,
        valid_coord_list,
        int(model.main_session),
        save_path="session_coordinate_coverage.png",
        save_fn=_save,
    )

    manifest = {
        "version": EP01_CACHE_VERSION,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "n_frames": int(len(df_sorted)),
        "n_raw_main_session_frames": int(len(raw_main_df)),
        "n_sr_usable_frames": int(len(sr_df)),
        "artifacts": list(EP01_ARTIFACTS),
        "rebuild_command": "uv run python scripts/build_ep01_cache.py",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    _write_audit_report(
        report_dir=report_dir,
        pairing=pairing,
        pairing_detail=pairing_detail,
        model=model,
        summary_table=summary_table,
        boundary_jump_table=boundary_jump_table,
        coord_repeat_summary=coord_repeat_summary,
        r_distribution=r_distribution,
        repeat_exclusion_summary=repeat_exclusion_summary,
        missing_coordinate_table=missing_coordinate_table,
        raster_row_summary=raster_row_summary,
        frame_audit_contract=frame_audit_contract,
        raw_main_df=raw_main_df,
        sr_df=sr_df,
        all_coords=all_coords,
        valid_coords=valid_coords,
        noise_floor_c=noise_floor_c,
    )

    return Ep01Cache(
        output_dir=output_dir,
        df=df_sorted,
        pairing=pairing,
        pairing_detail=pairing_detail,
        rename_special=rename_special,
        rename_mapping_path=rename_mapping_path,
        coord_repeat_summary=coord_repeat_summary,
        r_distribution=r_distribution,
        repeat_exclusion_summary=repeat_exclusion_summary,
        missing_coordinate_table=missing_coordinate_table,
        session_summary=model.session_summary.copy(),
        boundary_jump_table=boundary_jump_table,
        summary_table=summary_table,
        frame_audit_contract=frame_audit_contract,
        raster_row_summary=raster_row_summary,
        raster_row_mismatches=raster_row_mismatches,
        acquisition_order_audit=acquisition_order_audit,
        matrix_audit_summary=matrix_audit_summary,
        noise_floor_c=noise_floor_c,
        valid_coords=valid_coords,
        coord_config=coord_config,
        manifest=manifest,
    )


def load_ep01_cache(
    *,
    output_dir: Path | None = None,
    project_root: Path | None = None,
) -> Ep01Cache:
    """Load EP01 CSV artifacts without re-reading raw TXT matrices."""
    root = _project_root(project_root)
    output_dir = (output_dir or root / "output" / "ep01_data_processing").resolve()
    missing = [name for name in EP01_ARTIFACTS if not (output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "EP01 cache incomplete. Missing: "
            + ", ".join(missing)
            + "\nRun: uv run python scripts/build_ep01_cache.py"
        )

    with open(output_dir / "cache_manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(root / "configs" / "coordinate_set.json", encoding="utf-8") as f:
        coord_config = json.load(f)
    with open(root / "configs" / "noise_floor.json", encoding="utf-8") as f:
        noise_floor_c = float(json.load(f)["noise_floor_celsius"])
    with open(output_dir / "matrix_audit_summary.json", encoding="utf-8") as f:
        matrix_audit_summary = json.load(f)

    df = pd.read_csv(output_dir / "frame_audit.csv")
    pairing_detail = pd.read_csv(output_dir / "bmp_txt_pairing_detail.csv")
    rename_special = pd.read_csv(output_dir / "rename_special.csv")
    coord_repeat_summary = pd.read_csv(output_dir / "coord_repeat_summary.csv")
    r_distribution = pd.read_csv(output_dir / "r_distribution.csv")
    repeat_exclusion_summary = pd.read_csv(output_dir / "repeat_exclusion_summary.csv")
    repeat_exclusion_summary = repeat_exclusion_summary.fillna("")
    missing_coordinate_table = pd.read_csv(output_dir / "missing_coordinate_table.csv")
    session_summary = pd.read_csv(output_dir / "session_summary.csv")
    boundary_jump_table = pd.read_csv(output_dir / "boundary_jump_table.csv")
    summary_table = pd.read_csv(output_dir / "sr_data_basis_summary.csv")
    frame_audit_contract = pd.read_csv(output_dir / "frame_audit_contract.csv")
    raster_row_summary = pd.read_csv(output_dir / "raster_row_summary.csv")
    raster_row_mismatches = raster_row_summary[
        ~raster_row_summary["matches_expected_after_known_missing"].astype(bool)
    ].copy()
    acquisition_order_audit = pd.read_csv(output_dir / "acquisition_order_audit.csv")

    def _pair_value(label: str) -> int:
        row = pairing_detail.loc[pairing_detail["Check"].eq(label), "Value"]
        return int(row.iat[0]) if not row.empty else 0

    n_txt = _pair_value("TXT matrices")
    n_bmp = _pair_value("BMP previews")
    n_paired = _pair_value("Paired stems")
    pairing = {
        "n_txt": n_txt,
        "n_bmp": n_bmp,
        "n_paired": n_paired,
        "only_txt": [],
        "only_bmp": [],
    }

    return Ep01Cache(
        output_dir=output_dir,
        df=df,
        pairing=pairing,
        pairing_detail=pairing_detail,
        rename_special=rename_special,
        rename_mapping_path=output_dir / "rename_mapping.csv",
        coord_repeat_summary=coord_repeat_summary,
        r_distribution=r_distribution,
        repeat_exclusion_summary=repeat_exclusion_summary,
        missing_coordinate_table=missing_coordinate_table,
        session_summary=session_summary,
        boundary_jump_table=boundary_jump_table,
        summary_table=summary_table,
        frame_audit_contract=frame_audit_contract,
        raster_row_summary=raster_row_summary,
        raster_row_mismatches=raster_row_mismatches,
        acquisition_order_audit=acquisition_order_audit,
        matrix_audit_summary=matrix_audit_summary,
        noise_floor_c=noise_floor_c,
        valid_coords=set(int(v) for v in coord_config["x_coords_um"]),
        coord_config=coord_config,
        manifest=manifest,
    )


def require_ep01_cache(**kwargs) -> Ep01Cache:
    """Load cache or raise with rebuild instructions."""
    return load_ep01_cache(**kwargs)
