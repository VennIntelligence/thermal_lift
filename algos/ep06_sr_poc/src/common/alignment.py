"""EP05 alignment and quality-weight loading for EP06 SR."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import DEFAULT_FRAME_AUDIT_PATH, load_main_session_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ALIGNMENT_SCORES_PATH = (
    PROJECT_ROOT / "output" / "ep05_alignment_sr_capacity" / "alignment_method_holdout_scores.csv"
)
DEFAULT_CONTOUR_ALIGNMENT_PATH = PROJECT_ROOT / "output" / "ep05_contour_alignment" / "contour_alignment_results.csv"

_METHOD_ALIASES = {
    "ncc_init": "data_driven_ncc_init",
    "data_driven_ncc_init": "data_driven_ncc_init",
    "contour_refined": "data_driven_contour_refined",
    "data_driven_contour_refined": "data_driven_contour_refined",
    "filename_affine": "filename_affine_fit",
    "filename_affine_fit": "filename_affine_fit",
}


def _metadata(metadata: pd.DataFrame | None = None, frame_metadata: pd.DataFrame | None = None) -> pd.DataFrame:
    if metadata is not None:
        return metadata.reset_index(drop=True).copy()
    if frame_metadata is not None:
        return frame_metadata.reset_index(drop=True).copy()
    return load_main_session_metadata(DEFAULT_FRAME_AUDIT_PATH)


def _merge(meta: pd.DataFrame, values: pd.DataFrame) -> pd.DataFrame:
    if "file" in meta.columns and "file" in values.columns:
        vals = values.drop_duplicates("file", keep="first").copy()
        vals["file"] = vals["file"].astype(str)
        out = meta.assign(file=meta["file"].astype(str)).merge(vals, on="file", how="left")
        return out
    if "acquisition_order" in meta.columns and "acquisition_order" in values.columns:
        vals = values.drop_duplicates("acquisition_order", keep="first").copy()
        return meta.merge(vals, on="acquisition_order", how="left")
    raise ValueError("metadata and alignment table share neither file nor acquisition_order")


def _scores_rows(path: Path, method: str) -> pd.DataFrame:
    scores = pd.read_csv(path)
    key = _METHOD_ALIASES.get(method, method)
    rows = scores[scores["method"].eq(key)].copy()
    rows = rows.rename(
        columns={
            "align_dx_px": "dx_px",
            "align_dy_px": "dy_px",
            "holdout_chamfer_px": "chamfer_px",
        }
    )
    keep = [c for c in ["file", "acquisition_order", "dx_px", "dy_px", "chamfer_px", "gradient_corr"] if c in rows]
    return rows[keep]


def _contour_rows(path: Path, method: str) -> pd.DataFrame | None:
    if _METHOD_ALIASES.get(method, method) == "filename_affine_fit":
        return None
    contour = pd.read_csv(path)
    if _METHOD_ALIASES.get(method, method) == "data_driven_ncc_init":
        rename = {
            "init_align_dx_px": "dx_px",
            "init_align_dy_px": "dy_px",
            "init_holdout_chamfer_px": "chamfer_px",
        }
    else:
        rename = {
            "refined_align_dx_px": "dx_px",
            "refined_align_dy_px": "dy_px",
            "refined_holdout_chamfer_px": "chamfer_px",
            "gradient_corr_refined": "gradient_corr",
        }
    rows = contour.rename(columns=rename)
    if "success" in rows:
        rows = rows[rows["success"].astype(str).str.lower().isin({"true", "1", "yes"})].copy()
    keep = [c for c in ["file", "acquisition_order", "dx_px", "dy_px", "chamfer_px", "ncc_peak", "gradient_corr"] if c in rows]
    return rows[keep]


def load_alignment_table(
    method: str = "ncc_init",
    *,
    metadata: pd.DataFrame | None = None,
    frame_metadata: pd.DataFrame | None = None,
    frame_audit_path: str | Path | None = None,
    alignment_csv: str | Path | None = None,
    alignment_scores_path: str | Path | None = None,
    contour_alignment_path: str | Path | None = None,
    strict: bool = True,
) -> pd.DataFrame:
    """Return one alignment row per metadata row.

    Shifts are ``dx_px, dy_px`` in LR pixels and follow the EP05 convention:
    applying the shift moves the observed frame into the reference coordinate
    system. Forward-model prediction uses the inverse displacement internally.
    """

    if method not in _METHOD_ALIASES:
        raise ValueError(f"Unknown alignment method: {method}")
    meta = _metadata(metadata, frame_metadata)
    if frame_audit_path is not None and metadata is None and frame_metadata is None:
        meta = load_main_session_metadata(frame_audit_path)

    contour_path = Path(contour_alignment_path or alignment_csv or DEFAULT_CONTOUR_ALIGNMENT_PATH)
    scores_path = Path(alignment_scores_path or DEFAULT_ALIGNMENT_SCORES_PATH)

    contour = _contour_rows(contour_path, method) if contour_path.exists() else None
    primary = contour if contour is not None else _scores_rows(scores_path, method)
    if contour is not None and scores_path.exists():
        scores = _scores_rows(scores_path, method)
        key = "file" if "file" in primary.columns and "file" in scores.columns else "acquisition_order"
        fb = scores.drop_duplicates(key, keep="first")
        primary = primary.merge(fb, on=key, how="left", suffixes=("", "_score"))
        for col in ("dx_px", "dy_px", "chamfer_px", "gradient_corr"):
            alt = f"{col}_score"
            if alt in primary:
                primary[col] = primary[col].where(primary[col].notna(), primary[alt])
                primary = primary.drop(columns=[alt])

    out = _merge(meta, primary)
    missing = out[["dx_px", "dy_px"]].isna().any(axis=1)
    if strict and bool(missing.any()):
        examples = out.loc[missing, "file"].astype(str).head(8).tolist() if "file" in out else []
        raise ValueError(f"Missing {int(missing.sum())} alignment shifts for method={method}; examples={examples}")
    return out


def load_alignment_shifts(
    method: str = "ncc_init",
    *,
    metadata: pd.DataFrame | None = None,
    frame_metadata: pd.DataFrame | None = None,
    frame_audit_path: str | Path | None = None,
    alignment_csv: str | Path | None = None,
    alignment_scores_path: str | Path | None = None,
    contour_alignment_path: str | Path | None = None,
    strict: bool = True,
) -> np.ndarray:
    """Return ``(N, 2)`` shifts as ``[dx, dy]`` in LR pixels."""

    table = load_alignment_table(
        method,
        metadata=metadata,
        frame_metadata=frame_metadata,
        frame_audit_path=frame_audit_path,
        alignment_csv=alignment_csv,
        alignment_scores_path=alignment_scores_path,
        contour_alignment_path=contour_alignment_path,
        strict=strict,
    )
    shifts = table[["dx_px", "dy_px"]].to_numpy(dtype=np.float32)
    return np.nan_to_num(shifts, nan=0.0) if not strict else shifts


def load_quality_weights(
    method: str = "contour_refined",
    *,
    metadata: pd.DataFrame | None = None,
    frame_metadata: pd.DataFrame | None = None,
    frame_audit_path: str | Path | None = None,
    alignment_csv: str | Path | None = None,
    alignment_scores_path: str | Path | None = None,
    contour_alignment_path: str | Path | None = None,
    return_table: bool = False,
) -> np.ndarray | tuple[np.ndarray, pd.DataFrame]:
    """Load per-frame quality weights normalized to mean 1."""

    table = load_alignment_table(
        method,
        metadata=metadata,
        frame_metadata=frame_metadata,
        frame_audit_path=frame_audit_path,
        alignment_csv=alignment_csv,
        alignment_scores_path=alignment_scores_path,
        contour_alignment_path=contour_alignment_path,
        strict=False,
    )
    n = len(table)
    if n == 0:
        weights = np.empty(0, dtype=np.float32)
        return (weights, table) if return_table else weights

    signal_terms: list[np.ndarray] = []
    for col in ("ncc_peak", "gradient_corr"):
        if col not in table:
            continue
        values = pd.to_numeric(table[col], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        if finite.any():
            filled = values.copy()
            filled[~finite] = float(np.nanmedian(values[finite]))
            if np.nanmin(filled) < 0:
                filled = 0.5 * (filled + 1.0)
            signal_terms.append(np.clip(filled, 0.0, 1.0))
    signal = np.mean(np.vstack(signal_terms), axis=0) if signal_terms else np.ones(n, dtype=float)

    if "chamfer_px" in table:
        chamfer = pd.to_numeric(table["chamfer_px"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(chamfer) & (chamfer > 0)
        if finite.any():
            med = float(np.nanmedian(chamfer[finite]))
            chamfer = np.where(finite, chamfer, med)
            inverse_chamfer = med / np.maximum(chamfer, 1e-6)
        else:
            inverse_chamfer = np.ones(n, dtype=float)
    else:
        inverse_chamfer = np.ones(n, dtype=float)

    weights = np.clip(signal * inverse_chamfer, 0.05, 5.0)
    weights = weights / max(float(np.mean(weights)), 1e-12)
    weights = weights.astype(np.float32, copy=False)
    return (weights, table) if return_table else weights
