"""Manifest helpers for TCForge datasets."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SceneManifest:
    """Serializable metadata for one generated ThermalChipPhantom scene."""

    scene_id: str
    scene_dir: str
    difficulty: str
    seed: int
    split: str = "test"
    scale: int = 2
    lr_shape: tuple[int, int] = (480, 640)
    hr_shape: tuple[int, int] = (960, 1280)
    forward_mode: str = "exact_ep06_point"
    shift_profile: str = "real_default_contour_refined"
    shift_convention: str = "LR-to-reference alignment shift"
    drift_model: str = "none"
    files: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["lr_shape"] = list(self.lr_shape)
        data["hr_shape"] = list(self.hr_shape)
        return data

    def write_json(self, path: str | Path) -> None:
        write_json(self.to_dict(), path)


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"JSON file not found: {src}")
    return json.loads(src.read_text(encoding="utf-8"))


def write_manifest_csv(records: list[SceneManifest] | list[dict[str, Any]], path: str | Path) -> None:
    """Write a flat CSV scene manifest."""

    rows = [record.to_dict() if isinstance(record, SceneManifest) else dict(record) for record in records]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = {
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else value
                for key, value in row.items()
            }
            writer.writerow(flat)


def validate_file_list(
    scene_dir: str | Path,
    required_files: list[str] | tuple[str, ...],
    *,
    allow_empty: bool = False,
) -> list[Path]:
    """Return required paths if present; raise a clear error on missing files."""

    root = Path(scene_dir)
    if not root.exists():
        raise FileNotFoundError(f"scene_dir not found: {root}")
    missing: list[Path] = []
    found: list[Path] = []
    for rel in required_files:
        path = root / rel
        if not path.exists():
            missing.append(path)
            continue
        if not allow_empty and path.is_file() and path.stat().st_size == 0:
            raise ValueError(f"required file is empty: {path}")
        found.append(path)
    if missing:
        preview = ", ".join(str(path) for path in missing[:8])
        suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} total)"
        raise FileNotFoundError(f"missing required files: {preview}{suffix}")
    return found


def validate_scene_manifest(manifest: SceneManifest | dict[str, Any]) -> None:
    """Check core manifest fields before writing or consuming a scene."""

    data = manifest.to_dict() if isinstance(manifest, SceneManifest) else dict(manifest)
    required = {"scene_id", "scene_dir", "difficulty", "seed", "lr_shape", "hr_shape", "scale", "forward_mode"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"manifest missing required fields: {sorted(missing)}")
    if int(data["scale"]) <= 0:
        raise ValueError("manifest scale must be > 0")
    for key in ("lr_shape", "hr_shape"):
        shape = data[key]
        if len(shape) != 2 or int(shape[0]) <= 0 or int(shape[1]) <= 0:
            raise ValueError(f"manifest {key} must be a positive (rows, cols) pair")
