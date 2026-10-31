.PHONY: pdf figs clean
pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
	cp main.pdf eurekazero_whitepaper.pdf
figs:
	python3 figures/scripts/make_flywheel_figs.py
	python3 figures/scripts/make_schematics.py
	python3 figures/scripts/make_promotions_figure.py
	python3 figures/scripts/make_promotions_figure.py --target-progress
	python3 figures/scripts/make_promotions_figure.py --target-history
	python3 figures/scripts/make_recent_development_figure.py
	python3 figures/scripts/make_recent_development_figure.py --follow-up
clean:
	latexmk -C
