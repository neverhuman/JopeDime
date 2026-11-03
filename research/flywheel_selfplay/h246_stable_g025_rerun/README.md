# H246 — Fresh-Seed `STABLE_G025` Pick-Stability Unanimity Vote Gate Rerun (`h246-stable-g025-rerun-20261103`)

- **Status**: `PREREGISTERED` (awaiting TRAIN smoke + 1 primary attempt after preregistration PR merge)
- **Owner**: `rd_argon` (`[ARGON]`, screen `4187849.rd_argon`)
- **Authority**: Owner-delegated H-wave routine (`/home/ubuntu/nudges/hw_t0505_argon_h246.md`)
- **Preregistration**: [PREREG.md](PREREG.md)
- **Fold Manifest**: [fold_manifest.json](fold_manifest.json)
- **Manifest**: [manifest.json](manifest.json)
- **Metrics**: [metrics.json](metrics.json) ([metrics.schema.json](metrics.schema.json))

## Summary

Wave **H246** reruns the unchanged **H233** `STABLE_G025` pick-stability unanimity vote gate inside the frozen **H217** `GATED_025` recipe (`0` neural network optimizer updates; `DOPE c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` and `r882` read-only) on three fresh seeds (`246101–246103`, fold seeds `246141–246143`, maximum best-match Jaccard `0.1928 <= 0.20` vs `H203–H245`).

- **Recipe (`STABLE_G025`)**: Apply the frozen H217 `GATED_025` blend `r882 + 0.25 * (prior_0 - r882)` to a context **only when** (a) the outcome-free `zgap <= tau_cell` (where `tau_cell` is chosen on the `3` `FIT` folds under the `<= 5%` FIT family harm target) **and** (b) the blended first-argmin pick is unanimous across `3` outer HistGBM members (`member 0` = unweighted H217 fit, `members 1–2` = FIT-family cluster-bootstrap refits); otherwise fall back to `r882`.
- **Arms**: `R882` (incumbent), `G025` (replication anchor), `STABLE_G025` (sole primary, Holm `m = 1`), `STABLE_SHUF` (same vote rate, member agreement mask permuted across contexts within each FIT-calibrated cell), and `BAG_VOTE_UNGATED` (report-only).
- **Only Change vs H233**: Sequential member fitting (holding at most `1` HistGBM model in memory at a time and freeing heap pages after each member) and the cooperative H239 A1 unit/member-boundary resource guard (`xbabe2` `taskset -c 52-55`, `nice -n 19`, `<= 4` threads, `< 4 GiB` RSS, `>= 300 GB` free disk; load admit `<= 100`, pause `> 108`, resume `<= 100`; memory admit `>= 30 GiB`, pause `< 20 GiB`, resume `>= 25 GiB` sustained `30 s`, hard stop only `< 12 GiB` sustained `30 s` or `< 6 GiB` instant; compute stop `2026-11-03T10:00:00Z`).
