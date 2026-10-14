PYTHON ?= python
LATEXMK ?= /home/jovyan/.TinyTeX/bin/x86_64-linux/latexmk

.PHONY: paper paper-check paper-clean tables

paper: paper-check
	cd paper && $(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error article.tex

paper-check:
	$(PYTHON) paper/lint_paper.py

paper-clean:
	cd paper && $(LATEXMK) -C article.tex

tables:
	$(PYTHON) scripts/paper/token_lengths.py
