# H190 — `ENS10` Confirmation with Independent Family-Fold Partitions per Outer Seed (`190141`, `190142`, `190143`)

- **Wave**: `H190`
- **Owner**: `gpu_fused_agy` (`[FUSED-AGY]`)
- **Status**: `PREREGISTERED` (pending review, merge, and single guarded CPU execution on `xbabe1`)
- **Specification**: [PREREG.md](PREREG.md)
- **Manifest & Pins**: [manifest.json](manifest.json), [fold_manifest.json](fold_manifest.json), [provenance.json](provenance.json)
- **Implementation**: [analysis.py](analysis.py), [treatment.py](treatment.py), [runner.py](runner.py), [monitor.py](monitor.py), [test_h190.py](test_h190.py)

## Overview

`H190` tests whether a 10-member fresh-adapter score ensemble (`ENS10`, `40` updates/member on `FIT_{s, k}` only, candidate costs averaged before `argmin`) and its untrained 10-member control (`RENS10`, exact same `10` fresh initializations at `0` updates) beat `round-882.pt` (`R882`, Holm $m = 2$) under the imported `H177`/`H178` two-stage `205`-family pooled cross-fit selector when **each outer seed (`190101, 190102, 190103`) uses its own independent `3/1/1` family fold partition (`fold_seeds 190141, 190142, 190143`)** rather than sharing a single fold partition across seeds as in `H182` and `H185`.
