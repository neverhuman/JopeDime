# H226 — frozen GATED_025 on the retained-best r3175 base (fresh-fold confirmation), xbabe2 CPU, 0 updates

**Status: preregistered; not yet run.** It runs only after this prereg merges (`merge prereg PR -> smoke -> full run`).
See [PREREG.md](PREREG.md).

- **Why.** [H210](../h210_blend025_gated_retry/README.md) (+0.0140, harm 3.4%, Elo 1505.6) and
  [H217](../h217_gated025_confirm/README.md) (+0.0216, harm 4.4%, Elo 1504.8) confirmed `GATED_025` on `r882`, lifting
  DL-Val flywheel91 by `+0.00145` (`1.139246`, `0.00055` shy of the `+0.002` plateau margin). Meanwhile, the live
  flywheel has advanced to retained-best `r3175` (`1.1389572`, `+0.001163` over `r882`).
- **Hypothesis.** Applying the frozen H217 `GATED_025` recipe to the retained-best `r3175` base (`G3175`, zero adapter
  updates) compounds with `r3175` on 205-family pooled cross-fit and lifts DL-Val91 past both `+0.002` vs `R882` and
  `R3175` itself (`G3175 - R3175` family-bootstrap `LCB95 > 0`).
- **Arms.**
  - `G3175` (primary and co-primary): H217 `GATED_025` recipe on `r3175` (FIT-only $\tau$ over `R3175` s.t. FIT harm
    $\le 5\%$).
  - `R3175`: retained-best `round-3175.pt` (`1d6ea900…`) alone (exact DL-Val91 gate `1.1389572403748425`).
  - `R882`: reference `round-882.pt` (`0084479a…`, Elo 1500; bit-exact DL-Val91 gate `1.1377941898164525`).
  - `G882`: H217 `GATED_025` recipe on `r882` (replication anchor on fresh folds).
  - `G3175_DIS`: H129 disagreement placebo on `r3175`.
  - `B3175`, `B882`: ungated $\lambda = 0.25$ blend components (report-only).
- **Fresh partitions.** Seeds `226101–226103` with fold seeds `226141–226143`. Max best-match Jaccard is `0.189` ($\le
  0.20$) across `H203`, `H210`, `H217`, `H218`, `H222`, `H223`, `H224` and `H225`.
- **Decision.**
  - **Primary:** `CONFIRMED: G3175` iff pooled-A native `G3175` vs `R882` has `LCB95 > 0`, `3/3` seeds positive, and
    `harm <= 10%`.
  - **Co-primary:** `PLATEAU_WIN` iff DL-Val91 context-mean gain of `G3175` vs `R882` $> +0.002$ with family-bootstrap
    `LCB95 > 0`, AND `G3175` vs `R3175` DL-Val91 family-bootstrap `LCB95 > 0`.
  - **Diagnostics:** Elo (`R882 = 1500`) and DL-Val91 top-1 pick flip counts (`G3175 vs R3175`, `G3175 vs R882`,
    `R3175 vs R882`, `G882 vs R882`, `G3175 vs G882`). No AUC. No speed.
- **Host and guards (hw_t1435).** `xbabe2` CPU only, `nice -n 19`, whole-tree affinity $\le 4$ cores, RSS $< 4\text{ GiB}$,
  `MemAvailable >= 20 GiB`, free disk $\ge 300\text{ GB}$. Load guard pauses (`load5 > 60` $\to$ sleep/SIGSTOP, recheck
  every 60 s, resume at $\le 55$). Compute stop `18:30Z`, hard closeout `19:00Z`.
- **Code.** [runner.py](runner.py), [analysis.py](analysis.py), [monitor.py](monitor.py), [fold_manifest.json](fold_manifest.json)
  and [test_h226.py](test_h226.py).
