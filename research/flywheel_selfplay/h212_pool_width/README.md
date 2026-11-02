# H212 — frozen-r882 candidate-pool width sweep

**Attempt 001: INCOMPLETE due to a runner defect; no efficacy verdict. Attempt 002: owner-authorized D1, not yet launched.** The frozen preregistration is [PREREG.md](PREREG.md); the current fix validates H58's documented envelope representation.

Attempt 001 passed frozen r882 validation exactly (91/91, mean `1.1377941898164525`, maximum error `0.0`). DOPE stayed at the pinned SHA, with zero optimizer updates and CUDA uninitialized. The runner then stopped before proposals because it compared the decompressed H58 payload's SHA to the raw selected TRAIN table SHA. That check was a false positive: the selected H50 raw table matches its record, while H58 intentionally stores `training_cv_envelope(raw)`. The H58 gzip object SHA matches its manifest, and its decompressed bytes equal the pinned envelope function applied to the selected raw table. No data corruption or leakage occurred.

There were zero proposal calls, candidate evaluations, or treatment utility scores in attempt 001. Its raw receipts remain unchanged under `/home/ubuntu/jope-data/h212/attempt-001/`. Owner nudge `hw_t0125_luna.md` authorizes attempt 002 as deviation D1, using the same preregistered cohort, seeds, widths, endpoints, and guards, from a fresh `attempt-002` root after this fix is merged. This authorization does not change attempt 001's `INCOMPLETE` status or create an efficacy result.

See [metrics.json](metrics.json) for exact attempt-001 hashes and the D1 authorization. The finish README/metrics/index will be updated with attempt-002's outcome, three-seed median of equal-family mean gains, paired family-bootstrap CI, Holm m=2 across N32/N64, and Elo versus r882. No protected, sealed, holdout, or meta-test payload was read.
