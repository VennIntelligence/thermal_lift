# Thermal Lift Paper Workspace

This directory contains two lightweight writing tracks:

- `aaai/`: AAAI-style paper skeleton for a short, submission-shaped argument.
- `slides/`: Beamer deck skeleton for the full experimental story and project report.

The current AAAI files use the official AAAI-26 author kit downloaded from:

`https://aaai.org/authorkit26-1/`

Only the LaTeX style files needed for a local smoke test are kept under
`aaai/template/`. Before any real submission, replace these files with the
author kit for the target conference year.

## Build

Export stable figures and tables from `output/`:

```bash
uv run python paper/scripts/export_assets.py
```

Build the AAAI paper skeleton:

```bash
cd paper/aaai
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Build the Beamer report:

```bash
cd paper/slides
pdflatex main.tex
pdflatex main.tex
```

## Writing Boundary

The current claim is a 2x contour-level thermal SR POC on the 255-frame main
TXT session. Do not write this as a 4x SR result, a 5 um spatial-resolution
claim, or absolute-temperature metrology. Stage commands and filename-derived
motion are priors/controls, not alignment ground truth.

