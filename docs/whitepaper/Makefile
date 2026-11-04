.PHONY: pdf figs clean supplement
pdf: supplement
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
	cp main.pdf eurekazero_whitepaper.pdf
supplement:
	SOURCE_DATE_EPOCH=1793753726 latexmk -pdf -interaction=nonstopmode -halt-on-error supplement_parameter_displacement.tex
	cp supplement_parameter_displacement.pdf eurekazero_supplement.pdf
figs:
	python3 figures/scripts/make_flywheel_figs.py
	python3 figures/scripts/make_schematics.py
	python3 figures/scripts/make_promotions_figure.py
	python3 figures/scripts/make_promotions_figure.py --target-progress --source-dir "$(HISTORICAL_TARGET_INPUTS)"
	python3 figures/scripts/make_promotions_figure.py --committed-target-history
	python3 figures/scripts/make_recent_development_figure.py
	python3 figures/scripts/make_recent_development_figure.py --follow-up
	python3 figures/scripts/make_recent_development_figure.py --later-window
	python3 figures/scripts/make_completed_time_figure.py
	python3 figures/scripts/make_event_distribution_figure.py
	python3 figures/scripts/make_subsequent_window_figure.py
	python3 figures/scripts/make_subsequent_utility_figure.py
	python3 figures/scripts/make_subsequent_rates_figure.py
	python3 figures/scripts/make_live_admissions_figures.py
	python3 figures/scripts/make_live_development_figure.py
	python3 figures/scripts/make_round_throughput_figure.py
	python3 figures/scripts/make_round_throughput_figure.py --live-window
	python3 figures/scripts/make_acceptance_over_rounds_figure.py
	python3 figures/scripts/make_promotions_per_hour_figure.py
	python3 figures/scripts/make_latest_admissions_figure.py
	python3 figures/scripts/make_latest_event_rates_figure.py
	python3 figures/scripts/make_latest_event_rates_figure.py --round-clock-only
	python3 figures/scripts/make_latest_event_rates_figure.py --current-program-clock
	python3 figures/scripts/make_candidate_retained_trace.py
	python3 figures/scripts/make_candidate_retained_trace.py --recent
	python3 figures/scripts/make_admission_block_rates.py
	python3 figures/scripts/make_program_gain_distribution.py
	python3 figures/scripts/make_program_gain_distribution.py --rolling-admissions
	python3 figures/scripts/make_program_gain_distribution.py --selection-changes
	python3 figures/scripts/make_program_gain_distribution.py --retained-utility-lift
	python3 figures/scripts/make_program_gain_distribution.py --admission-outcomes
	python3 figures/scripts/make_adapter_validation_trace.py
	python3 figures/scripts/make_pool_insertion_cadence.py
	python3 figures/scripts/make_event_counters.py
	python3 figures/scripts/make_pool_insertion_windows.py
	python3 figures/scripts/make_program_promotion_margins.py
	python3 figures/scripts/make_program_journal_rates.py
	python3 figures/scripts/make_program_round_counts.py
	python3 figures/scripts/make_recent_program_margins.py
	python3 figures/scripts/make_round_phase_budget.py
	python3 figures/scripts/make_program_24h.py
	python3 figures/scripts/make_adapter_admission_rates.py
	python3 figures/scripts/make_full_cohort_program_rates.py
	python3 figures/scripts/make_candidate_margin_window.py
	python3 figures/scripts/make_decision_cadence.py
	python3 figures/scripts/make_parameter_displacement.py
	python3 figures/scripts/make_gradient_norms.py
	python3 figures/scripts/make_full_retained_trajectory.py
	python3 figures/scripts/make_full_round_wall_time.py
	python3 figures/scripts/make_training_minibatch_losses.py
	python3 figures/scripts/make_recent_discovery_retention.py
clean:
	latexmk -C
