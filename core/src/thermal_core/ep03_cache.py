"""EP03 cache builder and loader — notebook reads artifacts; script rebuilds them."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from thermal_core.ep03 import (
    build_crb_gate_summary_table,
    build_crb_localization_table,
    build_crb_sensitivity_table,
    build_mtf_attenuation_table,
    build_mtf_snr_recoverability_table,
    build_output_grid_nyquist_table,
    build_sampling_resolution_table,
    build_snr_reference_table,
    load_frame_by_row,
    measure_contour_observability,
    plot_crb_esf_localization,
    plot_crb_sensitivity_surface,
    plot_local_anchor_confidence,
    plot_local_contour_candidate_map,
    plot_mtf_psf_curves,
    plot_mtf_snr_recoverability_heatmap,
    plot_noise_floor_snr,
    plot_sampling_resolution_diagram,
    select_main_scan,
    select_reference_frame_row,
)
from thermal_core.io import load_frame
from thermal_core.notebook_cache import cache_is_complete, project_root, write_manifest
from thermal_core.plotting import savefig_academic, setup_academic_style

EP03_CACHE_VERSION = 1
EP03_DIRNAME = "ep03_theoretical_limits"
REBUILD_COMMAND = "uv run python scripts/build_ep03_cache.py"

PSF_SIGMAS = (0.2, 0.35, 0.5)
CRB_SIGMAS = (0.2, 0.35, 0.5, 1.0)
CRB_CONTRASTS = (0.3, 0.7, 1.0, 2.0)
CRB_N_FRAMES = (1, 4, 16, 64, 255)
CRB_PHASE_COVERAGE = (0.0, 0.5, 1.0)
RNG_SEED = 7
TARGET_GRID_UM = 5.0

EP03_TABLE_ARTIFACTS = (
    "sampling_resolution_distinction.csv",
    "output_grid_nyquist_periods.csv",
    "pixel_pitch_measurement_summary.csv",
    "pixel_size_measurement.json",
    "mtf_psf_attenuation.csv",
    "snr_noise_reference.csv",
    "local_contour_observability_segments.csv",
    "local_contour_observability_summary.csv",
    "crb_esf_localization_bounds.csv",
    "crb_sensitivity_scan.csv",
    "crb_sensitivity_gate_summary.csv",
    "mtf_snr_recoverability.csv",
    "mtf_snr_recoverability_gate_summary.csv",
)

EP03_FIGURE_ARTIFACTS = (
    "sampling_resolution_distinction.png",
    "pixel_size_measurement.png",
    "mtf_psf_frequency_response.png",
    "noise_floor_snr_contrast.png",
    "local_contour_candidate_map.png",
    "local_anchor_confidence_scatter.png",
    "crb_esf_localization_anchor.png",
    "crb_sensitivity_surface.png",
    "mtf_snr_recoverability_heatmap.png",
)

EP03_ARTIFACTS = (*EP03_TABLE_ARTIFACTS, *EP03_FIGURE_ARTIFACTS, "cache_manifest.json")


@dataclass(frozen=True)
class Ep03Cache:
    """Loaded EP03 artifacts for notebook display."""

    output_dir: Path
    manifest: dict
    stage_config: dict
    noise_config: dict
    theta_deg: float
    detector_pitch_um: float
    spatial_resolution_um: float
    noise_sigma_c: float
    target_grid_um: float
    main_session_frames: int
    reference_file: str
    reference_order: int
    reference_xy: tuple[int, int]
    sampling_resolution: pd.DataFrame
    grid_nyquist: pd.DataFrame
    pixel_pitch_summary: pd.DataFrame
    mtf_table: pd.DataFrame
    snr_reference: pd.DataFrame
    observability_summary: pd.DataFrame
    segments: pd.DataFrame
    crb_table: pd.DataFrame
    crb_sensitivity: pd.DataFrame
    crb_gate_summary: pd.DataFrame
    recoverability_table: pd.DataFrame
    recoverability_gate_summary: pd.DataFrame

    def figure_path(self, name: str) -> Path:
        return self.output_dir / Path(name).name


def _save_figure(fig, path: Path) -> None:
    savefig_academic(fig, path)


def _run_pixel_measurement(project_root: Path, output_dir: Path) -> None:
    script = project_root / "scripts" / "measure_pixel_size.py"
    if not script.exists():
        raise FileNotFoundError(f"Missing measurement script: {script}")
    data_dir = project_root / "data" / "data_raw" / "infrared_avi"
    bmp = data_dir / "10_16_0.bmp"
    txt = data_dir / "10_16_0.txt"
    if not bmp.exists() or not txt.exists():
        raise FileNotFoundError(
            f"Missing BMP/TXT pair for pixel measurement: {bmp}, {txt}"
        )
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--bmp",
            str(bmp),
            "--txt",
            str(txt),
            "--output-dir",
            str(output_dir),
        ],
        cwd=project_root,
        check=True,
    )


def _build_pixel_pitch_summary(output_dir: Path) -> pd.DataFrame:
    with open(output_dir / "pixel_size_measurement.json", encoding="utf-8") as f:
        pixel_measurement = json.load(f)
    axis_result = pixel_measurement["axis_method"]
    contour_result = pixel_measurement["contour_cross_check"]
    resolution_result = pixel_measurement["resolution_distinction"]
    summary = pd.DataFrame(
        [
            {
                "measurement": "BMP mm-axis ticks",
                "value_um_per_pixel": float(axis_result["pixel_size_mean_um"]),
                "evidence": "Rendered data crop is 640 x 480; 1 mm tick spacing is 100 rendered px",
            },
            {
                "measurement": "TXT/BMP contour cross-check",
                "value_um_per_pixel": float(contour_result["pixel_size_mean_um"]),
                "evidence": f"Outer-mask IoU={contour_result['mask_iou']:.4f}",
            },
            {
                "measurement": "Current spatial resolution",
                "value_um_per_pixel": float(resolution_result["current_spatial_resolution_um"]),
                "evidence": "Calibrated resolving scale, not detector pitch",
            },
        ]
    )
    summary.to_csv(output_dir / "pixel_pitch_measurement_summary.csv", index=False)
    return summary


def build_ep03_cache(
    *,
    project_root_arg: Path | None = None,
    output_dir: Path | None = None,
    force: bool = False,
) -> Ep03Cache:
    """Build EP03 theoretical-limit tables and figures from configs and reference frame."""
    root = project_root(project_root_arg)
    output_dir = (output_dir or root / "output" / EP03_DIRNAME).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not force and cache_is_complete(output_dir, EP03_ARTIFACTS):
        return load_ep03_cache(output_dir=output_dir, project_root_arg=root)

    ep01_audit = root / "output" / "ep01_data_processing" / "frame_audit.csv"
    if not ep01_audit.exists():
        raise FileNotFoundError(
            f"Missing {ep01_audit}. Run: uv run python scripts/build_ep01_cache.py"
        )

    with open(root / "configs" / "stage_calibration.json", encoding="utf-8") as f:
        stage_config = json.load(f)
    with open(root / "configs" / "noise_floor.json", encoding="utf-8") as f:
        noise_config = json.load(f)

    theta_deg = float(stage_config["theta_deg"])
    detector_pitch_um = float(stage_config["pixel_size_um"])
    spatial_resolution_um = float(stage_config["current_spatial_resolution_um"])
    noise_sigma_c = float(noise_config["noise_floor_celsius"])

    setup_academic_style()
    data_dir = root / "data" / "data_raw" / "infrared_avi"

    audit_df = pd.read_csv(ep01_audit)
    main_df = select_main_scan(audit_df)
    ref_row = select_reference_frame_row(main_df)
    reference_frame = load_frame_by_row(data_dir, ref_row, load_frame)

    sampling_resolution = build_sampling_resolution_table(
        detector_pitch_um=detector_pitch_um,
        spatial_resolution_um=spatial_resolution_um,
        target_grid_um=TARGET_GRID_UM,
    )
    sampling_resolution.to_csv(output_dir / "sampling_resolution_distinction.csv", index=False)

    grid_nyquist = build_output_grid_nyquist_table(
        detector_pitch_um=detector_pitch_um,
        spatial_resolution_um=spatial_resolution_um,
        grid_factors=(1, 2, 4),
    )
    grid_nyquist.to_csv(output_dir / "output_grid_nyquist_periods.csv", index=False)

    _run_pixel_measurement(root, output_dir)
    pixel_pitch_summary = _build_pixel_pitch_summary(output_dir)

    mtf_table = build_mtf_attenuation_table(
        detector_pitch_um=detector_pitch_um,
        sigmas_px=PSF_SIGMAS,
        grid_factors=(1, 2, 4),
    )
    mtf_table.to_csv(output_dir / "mtf_psf_attenuation.csv", index=False)

    segments, observability_summary, _outer_mask, outer_contour, inner_contours = (
        measure_contour_observability(
            reference_frame,
            theta_deg=theta_deg,
            noise_sigma_c=noise_sigma_c,
        )
    )
    segments.to_csv(output_dir / "local_contour_observability_segments.csv", index=False)
    observability_summary.to_csv(
        output_dir / "local_contour_observability_summary.csv", index=False
    )

    snr_reference = build_snr_reference_table(noise_sigma_c, observability_summary)
    snr_reference.to_csv(output_dir / "snr_noise_reference.csv", index=False)

    crb_table = build_crb_localization_table(
        noise_sigma_c,
        contrasts_c=CRB_CONTRASTS,
        sigma_values_px=(0.5, 1.0),
        n_frames=16,
        phase_coverage_px=1.0,
    )
    crb_table.to_csv(output_dir / "crb_esf_localization_bounds.csv", index=False)

    crb_sensitivity = build_crb_sensitivity_table(
        noise_sigma_c,
        contrasts_c=CRB_CONTRASTS,
        sigma_values_px=CRB_SIGMAS,
        n_frames_values=CRB_N_FRAMES,
        phase_coverage_values_px=CRB_PHASE_COVERAGE,
    )
    crb_sensitivity.to_csv(output_dir / "crb_sensitivity_scan.csv", index=False)

    crb_gate_summary = build_crb_gate_summary_table(crb_sensitivity)
    crb_gate_summary.to_csv(output_dir / "crb_sensitivity_gate_summary.csv", index=False)

    recoverability_contrasts = snr_reference[snr_reference["label"].ne("Noise floor")].copy()
    recoverability_table = build_mtf_snr_recoverability_table(
        recoverability_contrasts,
        noise_sigma_c=noise_sigma_c,
        sigmas_px=PSF_SIGMAS,
        grid_factors=(1, 2, 4),
        detector_pitch_um=detector_pitch_um,
    )
    recoverability_table.to_csv(output_dir / "mtf_snr_recoverability.csv", index=False)

    recoverability_gate_summary = (
        recoverability_table.groupby(["grid_label", "sigma_psf_px"], as_index=False)
        .agg(
            min_effective_snr=("effective_snr", "min"),
            median_effective_snr=("effective_snr", "median"),
            max_effective_snr=("effective_snr", "max"),
            pass_3x_fraction=("passes_3x_noise", "mean"),
            pass_5x_fraction=("passes_5x_noise", "mean"),
        )
    )
    recoverability_gate_summary.to_csv(
        output_dir / "mtf_snr_recoverability_gate_summary.csv", index=False
    )

    _save_figure(
        plot_sampling_resolution_diagram(
            sampling_resolution,
            detector_pitch_um=detector_pitch_um,
            spatial_resolution_um=spatial_resolution_um,
            target_grid_um=TARGET_GRID_UM,
        ),
        output_dir / "sampling_resolution_distinction.png",
    )
    _save_figure(plot_mtf_psf_curves(mtf_table, sigmas_px=PSF_SIGMAS), output_dir / "mtf_psf_frequency_response.png")
    _save_figure(
        plot_noise_floor_snr(snr_reference, segments, noise_sigma_c=noise_sigma_c),
        output_dir / "noise_floor_snr_contrast.png",
    )
    _save_figure(
        plot_local_contour_candidate_map(
            reference_frame,
            outer_contour,
            inner_contours,
            segments,
        ),
        output_dir / "local_contour_candidate_map.png",
    )
    _save_figure(plot_local_anchor_confidence(segments), output_dir / "local_anchor_confidence_scatter.png")
    _save_figure(
        plot_crb_esf_localization(crb_table, noise_sigma_c=noise_sigma_c, seed=RNG_SEED),
        output_dir / "crb_esf_localization_anchor.png",
    )
    _save_figure(
        plot_crb_sensitivity_surface(
            crb_sensitivity,
            sigma_values_px=(0.35, 0.5),
            phase_coverage_values_px=(0.5, 1.0),
        ),
        output_dir / "crb_sensitivity_surface.png",
    )
    _save_figure(
        plot_mtf_snr_recoverability_heatmap(recoverability_table),
        output_dir / "mtf_snr_recoverability_heatmap.png",
    )

    manifest = write_manifest(
        output_dir,
        version=EP03_CACHE_VERSION,
        artifacts=EP03_ARTIFACTS,
        rebuild_command=REBUILD_COMMAND,
        extra={
            "theta_deg": theta_deg,
            "detector_pitch_um": detector_pitch_um,
            "spatial_resolution_um": spatial_resolution_um,
            "noise_sigma_c": noise_sigma_c,
            "main_session_frames": int(len(main_df)),
            "reference_file": str(ref_row["file"]),
            "reference_order": int(ref_row["acquisition_order"]),
            "reference_x_um": int(ref_row["X"]),
            "reference_y_um": int(ref_row["Y"]),
        },
    )

    return Ep03Cache(
        output_dir=output_dir,
        manifest=manifest,
        stage_config=stage_config,
        noise_config=noise_config,
        theta_deg=theta_deg,
        detector_pitch_um=detector_pitch_um,
        spatial_resolution_um=spatial_resolution_um,
        noise_sigma_c=noise_sigma_c,
        target_grid_um=TARGET_GRID_UM,
        main_session_frames=int(len(main_df)),
        reference_file=str(ref_row["file"]),
        reference_order=int(ref_row["acquisition_order"]),
        reference_xy=(int(ref_row["X"]), int(ref_row["Y"])),
        sampling_resolution=sampling_resolution,
        grid_nyquist=grid_nyquist,
        pixel_pitch_summary=pixel_pitch_summary,
        mtf_table=mtf_table,
        snr_reference=snr_reference,
        observability_summary=observability_summary,
        segments=segments,
        crb_table=crb_table,
        crb_sensitivity=crb_sensitivity,
        crb_gate_summary=crb_gate_summary,
        recoverability_table=recoverability_table,
        recoverability_gate_summary=recoverability_gate_summary,
    )


def load_ep03_cache(
    *,
    output_dir: Path | None = None,
    project_root_arg: Path | None = None,
) -> Ep03Cache:
    """Load EP03 CSV/JSON artifacts without re-reading raw TXT matrices."""
    root = project_root(project_root_arg)
    output_dir = (output_dir or root / "output" / EP03_DIRNAME).resolve()

    missing = [name for name in EP03_ARTIFACTS if not (output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "EP03 cache incomplete. Missing: "
            + ", ".join(missing)
            + f"\nRun: {REBUILD_COMMAND}"
        )

    manifest_path = output_dir / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    with open(root / "configs" / "stage_calibration.json", encoding="utf-8") as f:
        stage_config = json.load(f)
    with open(root / "configs" / "noise_floor.json", encoding="utf-8") as f:
        noise_config = json.load(f)

    return Ep03Cache(
        output_dir=output_dir,
        manifest=manifest,
        stage_config=stage_config,
        noise_config=noise_config,
        theta_deg=float(stage_config["theta_deg"]),
        detector_pitch_um=float(stage_config["pixel_size_um"]),
        spatial_resolution_um=float(stage_config["current_spatial_resolution_um"]),
        noise_sigma_c=float(noise_config["noise_floor_celsius"]),
        target_grid_um=TARGET_GRID_UM,
        main_session_frames=int(manifest.get("main_session_frames", 255)),
        reference_file=str(manifest.get("reference_file", "")),
        reference_order=int(manifest.get("reference_order", 0)),
        reference_xy=(
            int(manifest.get("reference_x_um", 0)),
            int(manifest.get("reference_y_um", 0)),
        ),
        sampling_resolution=pd.read_csv(output_dir / "sampling_resolution_distinction.csv"),
        grid_nyquist=pd.read_csv(output_dir / "output_grid_nyquist_periods.csv"),
        pixel_pitch_summary=pd.read_csv(output_dir / "pixel_pitch_measurement_summary.csv"),
        mtf_table=pd.read_csv(output_dir / "mtf_psf_attenuation.csv"),
        snr_reference=pd.read_csv(output_dir / "snr_noise_reference.csv"),
        observability_summary=pd.read_csv(output_dir / "local_contour_observability_summary.csv"),
        segments=pd.read_csv(output_dir / "local_contour_observability_segments.csv"),
        crb_table=pd.read_csv(output_dir / "crb_esf_localization_bounds.csv"),
        crb_sensitivity=pd.read_csv(output_dir / "crb_sensitivity_scan.csv"),
        crb_gate_summary=pd.read_csv(output_dir / "crb_sensitivity_gate_summary.csv"),
        recoverability_table=pd.read_csv(output_dir / "mtf_snr_recoverability.csv"),
        recoverability_gate_summary=pd.read_csv(
            output_dir / "mtf_snr_recoverability_gate_summary.csv"
        ),
    )


def require_ep03_cache(**kwargs) -> Ep03Cache:
    """Load cache or raise with rebuild instructions."""
    return load_ep03_cache(**kwargs)
