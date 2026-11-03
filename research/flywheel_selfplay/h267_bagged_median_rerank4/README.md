# H267: bagged median-rank top-4 re-ranker (zero-update, single-pick)

**Status: preregistered, not executed.** [PREREG.md](PREREG.md) is the frozen protocol. It is published on `v1/next`
before anything runs.

## Hypothesis

**What is known.** H265 RERANK4 advanced on DL-Val91 (+0.0187), but it is seed-unstable, harms 3–5 families per seed,
and its 205-family DL-Train out-of-fold result is not significant.

**What is new.** Family-bootstrap bagging (B = 25) of the frozen H265 LGBMRanker with median-rank aggregation. The
question is whether that makes the lift stable and positive on DL-Train out of fold and keeps it on DL-Val91.

## Setup

- **Frozen from H265:** features, labels, the LGBMRanker recipe and grid, the strict selector and the margin rule,
  imported unchanged from a byte-identical copy ([h265_core.py](h265_core.py)).
- **Arms:**
  - **CONTROL:** r882.
  - **RERANK4:** the H265 reference.
  - **BAG_MED (primary):** 25 family-bootstrap bags at the cell's selected (lr, n_est); picks the slot with the
    smallest median rank, ties to r882.
  - **BAG_SHUF (placebo):** labels shuffled within family.

  The bagged arms are admitted by an out-of-bag strict rule (gain > .001 and LCB95 > 0, else r882).
- **Design.** Repeated nested family CV with one fresh fold seed per model seed (267140/141/142 ↔ 267101/102/103):
  15 cells. All selections are locked before any DL-Val91 read.
- **Decision.**
  - **PRIMARY:** DL-Train out of fold, BAG_MED − r882: median > +0.002, Holm-level LCB > 0, 3/3 seeds, and beats
    BAG_SHUF.
  - **SECONDARY:** DL-Val91 > 0 on 3/3 seeds.
  - Holm m = 2 over the two.
- **Host.** xbabe2 CPU cores 44–47, nice 19, under the H239 A1 memory guard. The compute stop is 20:30Z.

## Files

| File | Contents |
|---|---|
| [runner.py](runner.py) | the study runner (imports the H265 core) |
| [h265_core.py](h265_core.py) | byte-identical copy of the H265 runner |
| [analysis.py](analysis.py) | folds, bags, median-rank aggregation, OOB admission and inference |
| [monitor.py](monitor.py) | the A1 guard |
| [test_h267.py](test_h267.py) | data-free controls |
| [manifest.json](manifest.json) | input, source and package pins |
| [fold_manifest.json](fold_manifest.json) | the three partitions |
| [provenance.json](provenance.json) | the real-S3 closure |
| [metrics.json](metrics.json) | the metrics schema placeholder |
