# H132 — Fisher-anchored (`EWC`) and `BITFIT`-restricted adapter fine-tuning

**Preregistered; not run.** H132 tests whether directional constraints on adapter updates—either a fit-fold empirical diagonal Fisher anchor (`EWC`) or restricting optimization to the `691` adapter bias + output-layer parameters (`BITFIT`)—eliminate the family harm seen under unconstrained 40-step fine-tuning (`CONTROL`) and yield a strict improvement over unchanged `r882`. See [PREREG.md](PREREG.md) and [manifest.json](manifest.json) for the frozen protocol, and [fold_manifest_132100.json](fold_manifest_132100.json) (`sha256 = a2a9dd147601ecca76e247576a4575970450d1facc77c332232a5899ac54f440`) for the 5-fold `DL-Train` family split (`fold_seed = 132100`).

No real-data training or `fold 0` assessment scoring has occurred (`--synthetic-smoke` passed with `0` optimizer updates). Execution waits for this preregistration PR to merge into `origin/main`.
