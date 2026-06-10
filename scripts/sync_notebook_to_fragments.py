#!/usr/bin/env python3
"""
Sync notebook cell sources back to fragment .py files.

This is the reverse operation of scripts/build_notebook.py for the common case
where a generated .ipynb was edited directly and the fragment cell layout is
still unchanged.

Usage:
    uv run python scripts/sync_notebook_to_fragments.py notebooks/ep01_data_processing
    uv run python scripts/sync_notebook_to_fragments.py notebooks/ep01_data_processing --write

Notes:
    - The .ipynb does not store fragment filenames, so this tool uses the
      existing fragment files and their current cell counts as boundaries.
    - Empty code cells are ignored by default because they are common notebook
      editing artifacts and are not useful fragment source.
    - If the notebook has other added/removed/reordered cells, the tool refuses
      to write. Fix the fragment layout first, then run it again.
    - Only cell source is synced; outputs, execution counts, and notebook
      metadata are intentionally ignored.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from build_notebook import parse_py_to_cells


def source_to_text(source: str | list[str]) -> str:
    """Normalize notebook source to the text shape used by build_notebook."""
    if isinstance(source, list):
        text = "".join(source)
    else:
        text = source
    return text.rstrip("\n")


def load_notebook_cells(
    notebook_path: Path,
    *,
    keep_empty_code_cells: bool = False,
) -> tuple[list[dict[str, str]], int]:
    """Load notebook cells as minimal source-only records."""
    try:
        nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"Notebook not found: {notebook_path}") from None
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid notebook JSON: {notebook_path}: {exc}") from None

    cells = []
    ignored_empty_code_cells = 0
    for idx, cell in enumerate(nb.get("cells", [])):
        cell_type = cell.get("cell_type")
        if cell_type not in {"code", "markdown"}:
            raise SystemExit(f"Unsupported cell type at index {idx}: {cell_type}")
        source = source_to_text(cell.get("source", ""))
        if cell_type == "code" and source.strip() == "" and not keep_empty_code_cells:
            ignored_empty_code_cells += 1
            continue
        cells.append({
            "cell_type": cell_type,
            "source": source,
        })
    return cells, ignored_empty_code_cells


def fragment_files(target_dir: Path) -> tuple[Path, list[Path]]:
    """Return the fragment directory and ordered fragment files."""
    fragments_dir = target_dir / "fragments"
    if not fragments_dir.is_dir():
        fragments_dir = target_dir

    manifest = fragments_dir / "manifest.txt"
    if manifest.exists():
        files = [
            fragments_dir / line.strip()
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    else:
        files = sorted(
            f for f in fragments_dir.glob("*.py")
            if re.match(r"^\d+_", f.name)
        )

    if not files:
        raise SystemExit(f"No fragment files found in: {fragments_dir}")

    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise SystemExit("Manifest references missing fragment files:\n" + "\n".join(missing))

    return fragments_dir, files


def render_cell(cell: dict[str, str]) -> str:
    """Render one notebook cell as jupytext percent-format Python."""
    source = cell["source"]
    if cell["cell_type"] == "markdown":
        lines = ["# %% [markdown]"]
        for line in source.split("\n") if source else []:
            lines.append("#" if line == "" else f"# {line}")
        return "\n".join(lines).rstrip() + "\n"

    return "# %%\n" + source.rstrip() + "\n"


def render_fragment(cells: list[dict[str, str]]) -> str:
    """Render a sequence of cells as one fragment file."""
    return "\n".join(render_cell(cell).rstrip("\n") for cell in cells).rstrip() + "\n"


def normalize_cells(cells: list[dict[str, str]]) -> list[dict[str, str]]:
    """Normalize cell records before deciding whether a fragment changed."""
    return [
        {
            "cell_type": cell["cell_type"],
            "source": source_to_text(cell["source"]),
        }
        for cell in cells
    ]


def split_cells_by_fragment(
    notebook_cells: list[dict[str, str]],
    files: list[Path],
) -> list[tuple[Path, list[dict[str, str]]]]:
    """Split notebook cells using existing fragment cell counts."""
    counts = [len(parse_py_to_cells(path)) for path in files]
    expected = sum(counts)
    actual = len(notebook_cells)
    if expected != actual:
        detail = "\n".join(f"  {path.name}: {count}" for path, count in zip(files, counts))
        raise SystemExit(
            "Cell count mismatch; refusing to guess fragment boundaries.\n"
            f"Notebook cells: {actual}\n"
            f"Fragment cells: {expected}\n"
            f"Current fragment layout:\n{detail}\n\n"
            "If cells were added or removed in the notebook, first add/remove the "
            "matching # %% cell blocks in fragments/, then run this tool again."
        )

    chunks = []
    cursor = 0
    for path, count in zip(files, counts):
        chunks.append((path, notebook_cells[cursor:cursor + count]))
        cursor += count
    return chunks


def default_notebook_path(target_dir: Path) -> Path:
    """Match build_notebook.py's output naming convention."""
    return target_dir / f"{target_dir.name}.ipynb"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync a generated .ipynb back to its fragments/*.py sources.",
    )
    parser.add_argument("target_dir", type=Path, help="Notebook episode directory")
    parser.add_argument(
        "--notebook",
        type=Path,
        help="Notebook path; defaults to <target_dir>/<target_dir.name>.ipynb",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write changed fragment files. Without this, only report changes.",
    )
    parser.add_argument(
        "--keep-empty-code-cells",
        action="store_true",
        help="Treat empty code cells as real cells instead of ignoring them.",
    )
    args = parser.parse_args()

    target_dir = args.target_dir
    if not target_dir.is_dir():
        raise SystemExit(f"Target directory not found: {target_dir}")

    notebook_path = args.notebook or default_notebook_path(target_dir)
    fragments_dir, files = fragment_files(target_dir)
    notebook_cells, ignored_empty_code_cells = load_notebook_cells(
        notebook_path,
        keep_empty_code_cells=args.keep_empty_code_cells,
    )
    chunks = split_cells_by_fragment(notebook_cells, files)

    changed: list[Path] = []
    for path, cells in chunks:
        old_cells = normalize_cells(parse_py_to_cells(path))
        new_cells = normalize_cells(cells)
        if old_cells != new_cells:
            changed.append(path)
            if args.write:
                new_text = render_fragment(cells)
                path.write_text(new_text, encoding="utf-8")

    mode = "wrote" if args.write else "would update"
    print(f"Notebook: {notebook_path}")
    print(f"Fragments: {fragments_dir}")
    print(f"Cells: {len(notebook_cells)} across {len(files)} files")
    if ignored_empty_code_cells:
        print(f"Ignored empty code cells: {ignored_empty_code_cells}")

    if changed:
        print(f"{mode} {len(changed)} fragment file(s):")
        for path in changed:
            print(f"  {path.relative_to(target_dir.parent.parent)}")
        if not args.write:
            print("Dry run only. Re-run with --write to apply these changes.")
    else:
        print("No fragment source changes needed.")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(1)
