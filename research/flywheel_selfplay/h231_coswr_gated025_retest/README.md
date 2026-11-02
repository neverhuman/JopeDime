# H231 — COSWR × GATED_025 loader retest

PREREGISTERED, NOT RUN. Separately authorized retest after H230 failed during setup with 0 fits/updates. The raw NPZ loader now normalizes `program_sha256` into internal `pids` as H136 did. H230 arms, seeds, folds and strict +0.002 native/DL-Val margins are unchanged; this is not an independent replication.

[PREREG.md](PREREG.md) freezes the protocol and loader regression test, native91 check, TRAIN smoke reused in the 45-HistGBM/600-update inventory, resource limits and fixed deadline. [manifest.json](manifest.json), [fold_manifest.json](fold_manifest.json) and [provenance.json](provenance.json) bind identities.

Next: pass the raw-container regression, owner-authorized exact-head local checks/review/squash and TRAIN smoke, then execute the one CPU attempt. No treatment outcome or qualification claim.

A1 (`hw_t2100_astra_h231_merge.md`): inherited remote CI failures are non-gates; exact-head local checks control the owner-authorized admin squash. Merge/launch cutoff22:30Z Nov2; compute03:30Z and closeout04Z Nov3 unchanged. No outcomes read.
