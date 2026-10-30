.PHONY: pdf figs clean
pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
	cp main.pdf eurekazero_whitepaper.pdf
figs:
	python3 figures/scripts/make_flywheel_figs.py
clean:
	latexmk -C
