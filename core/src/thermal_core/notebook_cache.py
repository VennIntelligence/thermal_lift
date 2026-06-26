"""Shared helpers for EP notebook cache builders and display."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from IPython.display import Image as NotebookImage
from IPython.display import display


def project_root(start: Path | None = None) -> Path:
    root = (start or Path.cwd()).resolve()
    while not (root / "AGENTS.md").exists() and root != root.parent:
        root = root.parent
    if not (root / "AGENTS.md").exists():
        raise FileNotFoundError("Could not locate project root (AGENTS.md missing).")
    return root


def figure_path(output_dir: Path, name: str) -> Path:
    return output_dir / Path(name).name


def missing_artifacts(output_dir: Path, names: Iterable[str]) -> list[str]:
    return [name for name in names if not (output_dir / name).exists()]


def clear_output_dir(output_dir: Path) -> None:
    """Remove a cache output directory before a forced rebuild."""

    if output_dir.exists():
        shutil.rmtree(output_dir)


def require_artifacts(
    output_dir: Path,
    names: Iterable[str],
    *,
    rebuild_command: str,
) -> None:
    missing = missing_artifacts(output_dir, names)
    if missing:
        raise FileNotFoundError(
            f"Cache incomplete in {output_dir}. Missing: {', '.join(missing)}\n"
            f"Run: {rebuild_command}"
        )


def write_manifest(
    output_dir: Path,
    *,
    version: int,
    artifacts: Iterable[str],
    rebuild_command: str,
    extra: dict | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": version,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": list(artifacts),
        "rebuild_command": rebuild_command,
    }
    if extra:
        manifest.update(extra)
    path = output_dir / "cache_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_manifest(output_dir: Path) -> dict:
    path = output_dir / "cache_manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def cache_is_complete(output_dir: Path, artifacts: Iterable[str]) -> bool:
    return not missing_artifacts(output_dir, artifacts)


def show_fig(
    output_dir: Path,
    name: str,
    *,
    rebuild_command: str | None = None,
) -> None:
    """Display a cached PNG via NotebookImage (retina). For use in notebook fragments."""
    path = figure_path(output_dir, name)
    if not path.exists():
        hint = rebuild_command or "uv run python scripts/build_epXX_cache.py"
        raise FileNotFoundError(f"Missing cached figure: {path}\nRun: {hint}")
    display(NotebookImage(filename=str(path), retina=True))
