# H124 — fixed-lambda GBM shrink blend

**Status: preregistered; CPU-only scoring in progress.** H119 found a positive primary U4 gain for `GBM_PROG412_CTX` but harmed 40/205 families, so the fixed hypothesis is that shrinking its utility scores halfway toward r882 reduces the harm tail without choosing lambda from the results. H119 statistics are motivation only. No H119 split, seed, prediction, adapter, or outcome enters H124.

H124 uses a new family-grouped split (`default_rng(12400)`), model seeds 64–66, and the pinned real-S3 H50 DL-Train packet plus H58 feature cache. The H119/H85 HistGBM recipe, sampling size, and fit-only positive-slope calibration are reproduced in a self-contained runner. For each pool, both candidate and r882 cost vectors are negated to utility scores, then lambda 0.5 is applied exactly as preregistered. The primary is H119 Track-2 U4, paired by family against the matching fresh r882 reference. Lambda 1.0 is descriptive only. r882's earlier exposure to all H50 DL-Train families makes this conditional development evidence, not an unseen-family or serving result.

Admission requires a positive median of seed mean family gains, a one-sided family-bootstrap LCB95 above +0.001, and at most 10% of families with three-seed mean gain below -0.01. The report includes Elo vs r882=1500 with a 95% family-bootstrap interval; AUC is not used.

The runner provides a synthetic `--cpu-dry-run` and a separate explicit `--run --root ...` path. The run path validates the committed preregistration and pinned inputs, checks load/memory/RSS/scratch limits, stages a fresh r882 copy, and executes one fit at a time to remain below 6 GiB RSS. Scratch is capped at 2 GB and is removed only after the compact external receipt is verified.

See [PREREG.md](PREREG.md) for the frozen split, seeds, formula, acceptance gates, and source/input identities. Current metrics placeholder: [metrics.json](metrics.json). No GPU, qualification lock, FIFO ticket, S3 write, or live xbabe1 path is used.
