# H233 — `STABLE_G025` (pick-stability unanimity vote gate inside the frozen H217 `GATED_025` recipe)

- **Status:** `PREREGISTERED` (awaiting `TRAIN` smoke + one primary attempt on `xbabe2` CPU; 0 optimizer updates)
- **Owner:** `rd_argon` `[ARGON]` (`~/nudges/hw_t2205_argon_h233.md`, claim `~/jope_plan/ns/receipts/H233_claim.txt`)
- **Preregistration:** [PREREG.md](PREREG.md) (`SHA256 6f2bec113c1fc3161db0ff822c8472881ffbd83262772eb597f0d4e9114fddad`)
- **Fold Manifest:** [fold_manifest.json](fold_manifest.json) (fresh seeds `233101–233103`, fold seeds `233141–233143`, max best-match Jaccard vs `H203–H231` is `0.1993 <= 0.20`)
- **Arms:** `R882` (reference `1.1377942`), `G025` (frozen H217 replication anchor), `STABLE_G025` (sole primary, `Holm m=1`), `STABLE_SHUF` (per-cell permuted agreement-mask placebo), `BAG_VOTE_UNGATED` (vote without `zgap`, report-only)
- **Budget:** `5 outer x 3 members + 10 inner x 1 member = 25 HistGBM fits/seed` (`75` total) on `xbabe2` CPU only (`nice 19`, `taskset <= 4` cores or up to `8` when `load5 < 50`, `RSS < 4 GiB`, `MemAvailable >= 20 GiB`, `free >= 300 GB`, pause `load5 > 60` / resume `<= 55`).
