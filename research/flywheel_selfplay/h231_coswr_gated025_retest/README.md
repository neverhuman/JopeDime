# H231 — COSWR × GATED_025 loader retest

PREREGISTERED, NOT RUN. Separately authorized retest after H230 failed during setup with 0 fits/updates. The raw NPZ loader now normalizes `program_sha256` into internal `pids` as H136 did. H230 arms, seeds, folds and strict +0.002 native/DL-Val margins are unchanged; this is not an independent replication.

[PREREG.md](PREREG.md) freezes the protocol and loader regression test, native91 check, TRAIN smoke reused in the 45-HistGBM/600-update inventory, resource limits and fixed deadline. [manifest.json](manifest.json), [fold_manifest.json](fold_manifest.json) and [provenance.json](provenance.json) bind identities.

Next: pass the raw-container regression, normal exact-head CI/review/merge and TRAIN smoke, then execute the one CPU attempt. No treatment outcome or qualification claim.
