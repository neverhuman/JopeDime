# H126 — Robust Gradient Aggregation (`TRIM` & `TRIM_CLIP` vs `CONTROL`)

CPU-only offline study authorized by `/home/ubuntu/nudges/rd_argon_t1025.md` (`rd_argon`). [PREREG.md](PREREG.md) freezes the hypothesis, fresh `DL-Train` family split (`fold_seed = 126070`, `41` assessment families in `fold 0`, `164` fit families in `folds 1..4`), paired seeds `70–72`, unchanged-`r882` regression gate, frozen `CONTROL` calibration rule, coordinate-wise `20%`-each-tail per-family trimmed-mean gradient aggregation (`TRIM`) and calibration-median global norm clip (`TRIM_CLIP`), strict admission gates (`median_gain > 0`, `LCB95 > +0.001`, `harm <= 10%` in every seed), and `xbabe2` CPU/memory/load/disk guardrails before any execution.

[manifest.json](manifest.json) pins source, model, and input SHA-256 identities; [metrics.json](metrics.json) records `NOT_RUN` prior to calibration and execution. No outcome, admission, or runtime qualification is claimed in this preregistration PR.
