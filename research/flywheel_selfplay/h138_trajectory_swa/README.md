# H138 — retained-trajectory SWA/EMA

**PREREGISTERED; not executed.** [PREREG.md](PREREG.md) fixes R882, SWA4_UNIFORM,
SWA2_LATE and EMA_0.5 before candidate inference. Only the 47 named adapter
parameter tensors / 29,211 parameters are averaged; all buffers and frozen
state come from r882. No optimizer updates.

Hypothesis: averaging the flywheel's retained 782/785/859/882 trajectory may
improve r882 without further gradient steps. Four copied checkpoints are SHA
pinned. Unscored constructor proof verifies all 67 DOPE parameter/normalizer
tensors byte-for-byte equal, the canonical c51a56d0 hash, 47 adapter parameter tensors,
one observed OS thread and 944,128,000-byte peak RSS. Initial input hashing
reads the explicit train+val closure; constructor-only dataset reads are zero.

First require exact 91/91 r882 regression parity. Primary evaluates fixed
predictions over 205 DL-Train families with three five-fold family partitions;
these partitions are not independent fits or unseen-family evidence. Family
bootstrap intervals, Holm, strict margin, all-positive partitions and <=10%
strictly harmed families must all pass before selecting one arm. Only that
arm receives one candidate readout on 91-context validation. No primary pass
means no candidate DL-Val inference. Report Elo with r882=1500, never AUC.

Results are pending. [metrics.json](metrics.json) contains no invented scores.
[manifest.json](manifest.json) pins source, checkpoints, evaluation identity,
[fold_manifest.json](fold_manifest.json) and resource limits. CPU only on xbabe1,
nice 19, <=4 actual threads, RSS < 6 GB, available filesystem space >= 40 GB, total
scratch < 2 GB, two-hour wall limit including pressure pauses. No GPU or lock.

Next: central preregistration merge, actual-source verification, baseline gate,
one bounded run and numerical S3 closeout. Selector-exposed validation cannot
establish sealed-test performance or authorize serving promotion.

Prelaunch review caught and repaired an imported cache-reader contract mismatch:
pass the complementary context identities as forbidden rows, so the cache
identity closure is complete while only the admitted split payload is decoded.
A synthetic stored-cache test reproduces the original refusal and verifies both
train-only and validation-only range access. No scored attempt preceded this fix.
