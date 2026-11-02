# H218 — nested blend-weight (`lambda in {0.125, 0.25, 0.5}`) + `zgap` gate on FIT, xbabe2 CPU, 0 updates

**Status: preregistered; not yet run.** It runs only after this prereg merges. See [PREREG.md](PREREG.md).

- **Why:** [H210](../h210_blend025_gated_retry/README.md) found `CANDIDATE: GATED_025` (`BLEND` with fixed `lambda = 0.25`
  of calibrated HistGBM vs `r882`, gated per context by outcome-free `zgap`, `FIT`-only `tau`), giving pooled-`A` native
  gain `+0.0140 [+0.0022, +0.0355]` (`3/3` seeds, harm `3.4%`, Elo `1505.6`), while `DL-Val` `flywheel91` moved only
  `+0.0010` with `lambda` fixed at `0.25`.
- **Question:** Does choosing `lambda in {0.125, 0.25, 0.5}` jointly with `tau`, nested on `FIT` only (inner 3-fold
  family CV out-of-fold calibrated predictions, maximizing `FIT` equal-family native gain subject to `FIT` harm
  `<= 5%`, else `r882`), lift the `DL-Val` `flywheel91` plateau beyond `+0.002` while maintaining held-out `A` native
  gain?
- **Recipe:** The functions `zgap`, `disagreement`, `tau_grid`, `gated`, `choose_tau`, and `blend` are byte-identical to
  `H210`/`H217` (verified by a unit test); only `LAMBDAS` expands to `{0.125, 0.25, 0.5}` and `choose_lambda_tau`
  selects `(lambda, tau)` on `FIT` only.
- **Arms:**
  - `R882` (reference, `Elo 1500`; `DL-Val` `91/91` bit-exact `1.1377941898164525` first).
  - `GATED_NEST` (primary & co-primary): `(lambda, tau)` chosen nested on `FIT` only across `lambda in {0.125, 0.25, 0.5}`.
  - `GATED_NEST_SHUF` (placebo): same nested `(lambda, tau)` rule with `zgap` permuted across contexts (`218160`).
  - `GATED_025` (fixed `lambda = 0.25`, report-only reference).
  - `lambda_selection_counts` reported per cell across all `15` `(seed, outer)` cells.
- **Fresh partitions:** Seeds `218101–218103` with fold seeds `218141–218143`, one distinct 5-fold family partition per
  seed (`boot_seed = 218150`). Pairwise best-match Jaccard within `H218` is `0.145–0.152`, and at most `0.176` vs
  `H203`, `0.185` vs `H210`, and `0.178` vs `H217` (reuse refused).
- **Decision:**
  - **Primary (`GATED_NEST` vs `R882`, pooled `A`, `205` families, Holm over primary only `m = 1`):**
    `CANDIDATE: GATED_NEST` iff `LCB95 > 0`, `3/3` seeds positive, and harm `<= 10%`; else `NO_ADVANCE`.
  - **Co-primary (`PLATEAU_WIN` on `DL-Val` `91` contexts / `19` families):** `PLATEAU_WIN` iff context-mean gain vs
    `r882` (`1.1377942`) `> +0.002` with family-bootstrap `LCB95 > 0` (ties are not wins); else `NO_PLATEAU_WIN`.
- **Host & guards:** `xbabe2` CPU, `nice 19`, `<= 4` worker cores (`OMP_NUM_THREADS=4`), stop if `MemAvailable < 20 GiB`
  or `xbabe2 free disk < 200 GB`, `load <= 85`, no GPU, no `qualification.lock`, deadline `14:30Z`.
- **Code:** [runner.py](runner.py), [analysis.py](analysis.py), [monitor.py](monitor.py), and
  [test_h218.py](test_h218.py).
