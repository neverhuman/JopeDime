# H269 — Preregistered Fresh-Fold Confirmation of `NO_SCOREGEOM` (`NOSG`) + Outcome-Free Invalid-Gate Veto (`NOSG_VETO`)

- **Wave ID**: `h269-nosg-veto-rerank4-20261103` (`H269`)
- **Study Path**: `research/flywheel_selfplay/h269_nosg_veto_rerank4`
- **Owner**: `Argon`
- **Status**: `PREREGISTERED` (`outcomes_read = 0`)
- **Verdict**: `PENDING`

## Overview

`H269` performs a preregistered confirmatory replication of `H266`'s `NO_SCOREGEOM` (`528`-col feature representation, dropping all `32` `r882` scalar score/gap/cosine columns `0..31`) on **3 fresh fold seeds (`269140, 269141, 269142`)** and **3 fresh model seeds (`269101, 269102, 269103`)**, plus `NOSG_VETO` (`NOSG` paired with a frozen inner-`FIT` `LGBMClassifier` `-2.0` invalid-gate veto with `tau in {0.05, 0.10, 0.20}`). See `PREREG.md` for full preregistered specification.
