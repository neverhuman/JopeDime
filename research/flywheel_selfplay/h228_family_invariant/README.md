# H228 — Family-Invariant Adapter Continuation: AND-Mask ($\tau=0.8$) and GroupDRO ($\eta=0.01$)

**Status**: `PREREGISTERED` (no GPU outcomes yet; merge preregistration before Native91 baseline, timing smoke, or full 60-fit execution).

## Hypothesis

Restricting `r882` adapter continuation updates to coordinate directions whose gradient signs agree in $\ge \tau = 0.8$ of families (`ANDMASK`) or up-weighting the highest-loss families via GroupDRO (`GDRO`, $\eta = 0.01$) suppresses family-idiosyncratic overfitting and yields positive held-out-family lift versus `r882` where plain AdamW continuation (`CTRL`) and random-mask density-matched placebo (`ANDMASK_SHUF`) do not.

## Preregistered Design Summary

| Parameter | Specification |
|---|---|
| Wave / Owner | `H228` / `rd_argon` (`[ARGON]`, `~/nudges/hw_t1640_argon_h228.md`) |
| Base Incumbent | Read-only copy of `round-882.pt` (`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`); DOPE frozen `c51a56d0` verified before & after every fit |
| Trainable Parameters | `47` adapter tensors (`29,211` parameters; `41` active gradients, `6` inactive) |
| Arms (`40` updates/fit) | `CTRL` (H219 `CONTROL`), `ANDMASK` ($\tau=0.8$ over H219 `FBAL` $\ge 4$ families/step), `GDRO` ($\eta=0.01$ over families), `ANDMASK_SHUF` (random coordinate mask matching `ANDMASK` per-step kept fraction) |
| Primary & Multiplicity | Primary `ANDMASK` vs `R882`; Holm-Bonferroni (`m=2`) over `("ANDMASK", "GDRO")` (`CTRL` and `ANDMASK_SHUF` are controls) |
| Cohort & Partitions | `1,219` real-S3 `DL-Train` contexts (`205` families), 5-fold family-grouped OOF (`seeds = 228101–228103`, `fold_seeds = 228141–228143`) + post-lock `DL-Val91` (`91` contexts / `19` families) |
| Pre-Fit Gate | Bit-exact Native91 `r882` utility vector and mean `1.1377941898164525` before any fit; static TRAIN parity across all `1,219` contexts |
| Total Fits & Updates | `60` OOF fits (`2,400` full updates) + `1` timing smoke fit (`4` updates) = `2,404` total updates |
| Admission Rule | `median_gain > +0.002` AND `CI95_lower > 0` across 3 seeds (`harm <= 10%`, `3/3` positive seeds; exact ties reject); `BREAKTHROUGH` requires passing both TRAIN OOF and `DL-Val91` |
| Compute & Guards | `xbabe1` `GPU0` (`RTX 4090`, `GPU-784030c4-b1f9-d5a3-f0eb-9337f133f503`) under `xbabe1` `~/qualification.lock` FIFO (`hw_t1905_argon_h228_gpulost.md` pre-outcome retarget after `xbabe2` `Xid 79/154` with `admitted=false, outcomes_read=false`), `nice 19`, $\le 4$ CPU cores (`taskset`), `RSS < 6 GiB`, `GPU free VRAM >= 6 GiB` (`6,144 MiB`, never preempt flywheel), `MemAvailable >= 8 GiB`, `load5 <= 85`, `free disk >= 150 GB`; wait on guard trip without swap-thrashing |
| Deadlines | Prereg merge by `18:30Z` (`#1917` @ `f78f0d6e3`); compute stop `23:30Z`; closeout PR by `00:00Z` |

See [PREREG.md](PREREG.md) for the complete mathematical and operational protocol, [manifest.json](manifest.json) for source/input pins, [fold_manifest.json](fold_manifest.json) and [fold_jaccard.json](fold_jaccard.json) for fold partitions and overlap checks, and [test_h228.py](test_h228.py) for the data-free verification suite.
