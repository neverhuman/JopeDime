# H266 (`h266_rerank4_confirm`) — Confirmatory Fresh-Split Replication + Feature-Block Ablation of H265 `RERANK4`

- **Wave ID**: `h266-rerank4-confirm-20261103`
- **Owner**: Argon (`xbabe2` CPU lane `taskset -c 57-60`, `nice -n 19`, `threads <= 4`, `0` GPU)
- **Status**: `PREREGISTERED` (`0` outcomes read for `H266`)
- **Primary Endpoint**: `DL-Train` (`1,219` ctx / `205` fam) repeated nested family-grouped CV (`3` fresh fold seeds `266140, 266141, 266142` $\times$ `5` folds $\times$ `3` fresh model seeds `266101, 266102, 266103` = `45` cells), OOF single-pick (`0` test-time reveals) equal-family utility `RERANK4 - CONTROL(r882)` (`10,000`-draw paired family bootstrap `boot_seed = 266150`, per-seed medians across fold seeds).
- **Secondary Endpoint**: `DL-Val91` (`91` ctx / `19` fam) fresh-seed replication (`model_seeds = 266101–266103`, `fold_seed = 266140`) with `H265` gates (`Native91 = 1.1377941898164525` `91/91` bit-exact first).
- **Placebo & Multiplicity**: `RERANK4_SHUF` (within-family label shuffle); `Holm m = 2` over `{PRIMARY, SECONDARY}` (and within-split `Holm m = 2`).
- **Descriptive Feature-Block Ablations (`fold_seed = 266140`)**: `SCORE_ONLY` (`cols 0..31`), `SCORE_ONLY_20` (`cols 0..19`), `NO_SCOREGEOM`, `NO_SEM24`, `NO_CTX76`.

See [`PREREG.md`](PREREG.md) for the frozen protocol and confirmatory decision gates.
