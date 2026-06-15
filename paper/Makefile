.PHONY: assets aaai zh_conf zh_conf_supp slides all clean

assets:
	cd .. && uv run python paper/scripts/export_assets.py

aaai:
	cd aaai && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd aaai && bibtex main || true
	cd aaai && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd aaai && pdflatex -interaction=nonstopmode -halt-on-error main.tex

zh_conf:
	bash zh_conf/scripts/setup_fonts.sh
	cd zh_conf && xelatex -interaction=nonstopmode -halt-on-error main.tex
	cd zh_conf && bibtex main || true
	cd zh_conf && xelatex -interaction=nonstopmode -halt-on-error main.tex
	cd zh_conf && xelatex -interaction=nonstopmode -halt-on-error main.tex

zh_conf_supp:
	bash zh_conf/scripts/setup_fonts.sh
	cd zh_conf && xelatex -interaction=nonstopmode -halt-on-error supp.tex
	cd zh_conf && bibtex supp || true
	cd zh_conf && xelatex -interaction=nonstopmode -halt-on-error supp.tex
	cd zh_conf && xelatex -interaction=nonstopmode -halt-on-error supp.tex

slides:
	cd slides && pdflatex -interaction=nonstopmode -halt-on-error main.tex
	cd slides && pdflatex -interaction=nonstopmode -halt-on-error main.tex

all: assets aaai zh_conf zh_conf_supp slides

clean:
	find aaai zh_conf slides -maxdepth 1 -type f \( \
		-name '*.aux' -o -name '*.bbl' -o -name '*.blg' -o \
		-name '*.fdb_latexmk' -o -name '*.fls' -o -name '*.log' -o \
		-name '*.nav' -o -name '*.out' -o -name '*.snm' -o \
		-name '*.synctex.gz' -o -name '*.toc' \
	\) -delete

