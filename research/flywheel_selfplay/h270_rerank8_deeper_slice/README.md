# H270: re-ranking a deeper slice (top-8) of r882's own pool (zero-update, single-pick)

**Status: preregistered, not executed.** [PREREG.md](PREREG.md) is the frozen protocol. It is published on `v1/next`
before anything runs.

## Hypothesis

**What is known.** The H265 top-4 re-ranker captures only a small part of the top-4 oracle headroom, and the whole-pool
oracle is far above the top-4 oracle.

**What is new.** The same zero-update ranker re-ranks r882's top-**8** (K = 8) instead of its top-4. Only K changes.
The question is whether that lifts DL-Train out-of-fold single-pick utility over r882.

## Setup

- **Frozen from H265:** features, labels, the LGBMRanker grid, the strict selector and the margin rule.
  [topk.py](topk.py) is the K-generic form and reproduces [h265_core.py](h265_core.py) exactly at K = 4 (tested).
- **Arms:**
  - **CONTROL:** r882.
  - **RERANK4:** the H265 reference.
  - **RERANK8 (primary):** the same pipeline over slots 0–7.
  - **RERANK8_SHUF (placebo):** labels shuffled within family.
- **Design.** A full 45-cell cross: fold seeds 270140/141/142 × 5 folds × model seeds 270101–270103. All selections
  are locked before any DL-Val91 read.
- **Decision.**
  - **PRIMARY:** DL-Train OOF RERANK8 − r882: median > +0.002, Holm-level LCB > 0, 3/3 seeds, beats the placebo, and
    no more −2.0 picks than r882.
  - **SECONDARY:** DL-Val91 > 0 on 3/3 seeds. DL-Val91 is heavily reused; DL-Train OOF is the endpoint.
  - Holm m = 2 over the two.
- **Host.** xbabe2 CPU cores 44–47, nice 19, under the H239 A1 memory guard. The compute stop is 20:50Z.

## Files

| File | Contents |
|---|---|
| [runner.py](runner.py) | the study runner |
| [topk.py](topk.py) | the K-generic H265 pipeline |
| [h265_core.py](h265_core.py) | byte-identical copy of the H265 runner |
| [analysis.py](analysis.py) | the 45-cell protocol and inference |
| [monitor.py](monitor.py) | the A1 guard |
| [test_h270.py](test_h270.py) | data-free controls, including K = 4 exactness against H265 |
| [manifest.json](manifest.json) | input, source and package pins |
| [fold_manifest.json](fold_manifest.json) | the three partitions |
| [provenance.json](provenance.json) | the real-S3 closure |
| [metrics.json](metrics.json) | the metrics schema placeholder |
