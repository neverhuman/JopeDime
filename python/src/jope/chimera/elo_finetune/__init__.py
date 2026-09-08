"""Elo-objective fine-tuning of the deployed Starlight regressor.

The model competes as `tabpfn_v3_regressor` in the regression benchmark
archive; this package trains its deployed weights directly against the
frozen competitor utilities with a differentiable Elo (soft win mass) loss,
measures the real per-batch Elo via an exact spliced replay, and reports
debiased EWMA micro-Elo with confidence bands.

See docs/elo_training.md for the full design.
"""
