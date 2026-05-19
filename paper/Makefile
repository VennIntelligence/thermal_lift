.PHONY: assets aaai slides all clean

assets:
	cd .. && uv run python paper/scripts/export_assets.py

aaai:
	cd aaai && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd aaai && bibtex main || true
	cd aaai && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd aaai && pdflatex -interaction=nonstopmode -halt-on-error main.tex

slides:
	cd slides && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd slides && pdflatex -interaction=nonstopmode -halt-on-error main.tex

all: assets aaai slides

clean:
	find aaai slides -maxdepth 1 -type f \( \
		-name '*.aux' -o -name '*.bbl' -o -name '*.blg' -o \
		-name '*.fdb_latexmk' -o -name '*.fls' -o -name '*.log' -o \
		-name '*.nav' -o -name '*.out' -o -name '*.snm' -o \
		-name '*.synctex.gz' -o -name '*.toc' \
	\) -delete

