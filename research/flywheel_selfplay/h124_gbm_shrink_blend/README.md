# H124 — fixed-lambda GBM shrink blend

**Status: preregistered; scoring stopped at the strict RSS gate.** H119 found a positive primary U4 gain for `GBM_PROG412_CTX` but harmed 40/205 families, so the fixed hypothesis is that shrinking its utility scores halfway toward r882 reduces the harm tail without choosing lambda from the results. H119 statistics are motivation only. No H119 split, seed, prediction, adapter, or outcome enters H124.

H124 uses a new family-grouped split (`default_rng(12400)`), model seeds 64–66, and the pinned real-S3 H50 DL-Train packet plus H58 feature cache. The H119/H85 HistGBM recipe, sampling size, and fit-only positive-slope calibration are reproduced in a self-contained runner. For each pool, both candidate and r882 cost vectors are negated to utility scores, then lambda 0.5 is applied exactly as preregistered. The primary is H119 Track-2 U4, paired by family against the matching fresh r882 reference. Lambda 1.0 is descriptive only. r882's earlier exposure to all H50 DL-Train families makes this conditional development evidence, not an unseen-family or serving result.

Admission requires a positive median of seed mean family gains, a one-sided family-bootstrap LCB95 above +0.001, and at most 10% of families with three-seed mean gain below -0.01. The report includes Elo vs r882=1500 with a 95% family-bootstrap interval; AUC is not used.

The runner provides a synthetic `--cpu-dry-run` and a separate explicit `--run --root ...` path. The run path validates the committed preregistration and pinned inputs, checks load/memory/RSS/scratch limits, stages a fresh r882 copy, and executes one fit at a time using lazy program loads and an in-place design matrix. The default RSS abort threshold is 6 GiB. Scratch is capped at 2 GB; preserve the external receipt before any owner-authorized scratch reclamation.

See [PREREG.md](PREREG.md) for the frozen split, seeds, formula, acceptance gates, and source/input identities. The 2026-10-31 CPU attempt completed the fresh r882 DL-Train reference pass but reached 7,052,092 KiB peak RSS, exceeding the 6 GiB worker limit. The process stopped before any GBM fit; no blend metrics, Elo, or admission verdict were produced. The partial receipt and copied artifacts are at `/home/ubuntu/jope_plan/evidence/H124/partial-run-blocker.json` and `/home/ubuntu/jope_plan/evidence/H124/partial-run/`. See [metrics.json](metrics.json). No GPU, qualification lock, FIFO ticket, S3 write, or live xbabe1 path is used.

## Audit corrections for future execution

The [dated audit](AUDIT_t0855.md) covers the original source. OPUS Amendment 1
adds train-row cache reads, lazy/streamed inputs, median seed Elo and an RSS
watchdog. Owner t0955 preserves those commits and closes the remaining prefetch
and guard-failure paths in Amendment 2; existing running jobs are untouched.

Feature reads now use unbuffered archive/header/row access, validate row layout
before payload reads, and refuse DL-Val spans. Context IDs and projection offsets
are metadata. Whole-file hashing still reads opaque archive bytes, as explicitly
disclosed in the amendment; validation features are not decoded into model inputs.
`registered_estimator=median_seed_elo` identifies the headline and its paired
family-bootstrap interval. The pooled-score estimate is labelled additional.

The watchdog samples current and peak RSS at entry, exit and nominal 0.25-second
intervals. RSS breaches and monitor/receipt failures terminate the worker with
code 86. It is a sampled abort guard, not a kernel allocation limit. Partial
outputs from a breached/nonzero-exit run cannot establish completion. A new source
commit cannot retroactively qualify an old run: retain each run's actual source,
preregistration and terminal resource evidence. No scoring or retry is started by
these corrections. The operator must avoid duplicating the existing attempt.
