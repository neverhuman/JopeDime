# H159 — TRAIN-calibrated `fail_logit` veto at selection time (zero-update) on `r882`

**Status: `PREREGISTERED_NOT_EXECUTED` (zero-update constructor smoke `PASS` on `xbabe2`, `0` scored contexts).**

Prospective zero-update CPU experiment testing whether a TRAIN-calibrated candidate failure-probability veto ($\arg\min$ `total_cost` among candidates with $\sigma(\text{fail\_logit}) < t$, else fallback to plain $\arg\min$, $t \in \{0.50, 0.60, 0.70, 0.80, 0.90, 0.95\}$) cuts `-2` gate-cliff picks and raises equal-family utility over frozen `r882` (`1.1377942`). See [PREREG.md](PREREG.md), [manifest.json](manifest.json), and [fold_manifest.json](fold_manifest.json).
