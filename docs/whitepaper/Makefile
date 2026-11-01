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
	python3 figures/scripts/make_recent_development_figure.py --later-window
	python3 figures/scripts/make_completed_time_figure.py
	python3 figures/scripts/make_event_distribution_figure.py
	python3 figures/scripts/make_subsequent_window_figure.py
	python3 figures/scripts/make_subsequent_utility_figure.py
	python3 figures/scripts/make_subsequent_rates_figure.py
	python3 figures/scripts/make_live_admissions_figures.py
clean:
	latexmk -C
