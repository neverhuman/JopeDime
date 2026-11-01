# H164 — SAM/ASAM continuation retry

**PREREGISTERED; not executed.** H153 stopped before baseline because its launch receipt was outside the audit boundary. H164 fixes that path and tests CTRL, SAM rho.01/.05 and ASAM rho.5 across three seeds and five family folds:60 cells,2400 planned updates.

[PREREG.md](PREREG.md) fixes the selector, four-arm Holm rule, descriptive unconstrained/best-nonzero panels and CPU limits. [manifest.json](manifest.json) pins inputs/source; [fold_manifest.json](fold_manifest.json) fixes family membership. Exact91-context baseline must pass before updates. No serving promotion or sealed qualification.
