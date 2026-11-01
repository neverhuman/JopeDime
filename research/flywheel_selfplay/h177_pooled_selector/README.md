# H177: Pooled Cross-Fit Confirmatory Selector (`205`-Family Pooled `B` Selection + Single Prespecified Candidate per Arm + Separate `A`-Family Confirmation) on `r882` Continuation

Hypothesis and fixed protocol: [PREREG.md](PREREG.md). Preregistered four-arm continuation (`CTRL` live `1B=8`, `ACC4` mean-grad `4B=32`, `ACC16` mean-grad `16B=128` covering all `123` fit families, and `OC15` one-cycle LR/inverse-momentum `1e-5 -> 1.5e-4 -> 1e-6`) evaluated at a **single prespecified candidate step (`CANDIDATE_STEP = 40`)** per arm with **pooled cross-fit `B` selection over all `205` `DL-Train` families** (each family in $B_k$ exactly once across the `5` folds, `3` seeds `177101–177103` averaged within family before `10,000`-draw paired family-bootstrap resampling, `fold_seed = 177140`, `boot_seed = 177141`, unchanged `gain > +0.001 + 1e-12` & `LCB95 > 0` & Holm `< 0.05` across arms, ties reject) and a separate confirmation readout on `A` families (`fsync`-gated after `pooled_b_selection_lock.json`) plus descriptive `91`-context / `19`-family `DL-Val`.

Status: **preregistration + data-free harness only (`PENDING_H175_CLOSEOUT`)**; no `H177` scientific scores or optimizer updates yet. Execution is gated by [`h175_power_anchor.json`](h175_power_anchor.json) until `H175` (`#1490`) closes with `SELECTOR_UNDERPOWERED` or `MIXED` (or closes as `NOT_RUN` if `H175` returns `CEILING_LIKELY`).

Run the data-free test suite with:
```bash
PYTHONPATH=research/flywheel_selfplay/h177_pooled_selector pytest -q research/flywheel_selfplay/h177_pooled_selector/test_h177.py
```
