# H272: fresh-split confirmation of the H270 RERANK8 re-ranker, with depth arms

**Status: preregistered, not executed.** [PREREG.md](PREREG.md) is the frozen protocol. It is published on `v1/next`
before anything runs.

## Hypothesis

**What is known.** H270 ADVANCED: re-ranking r882's own top-8 lifted DL-Train out of fold by +0.052. That was one run,
with a reused DL-Val91 and visible family harm.

**What H272 asks.** Does RERANK8 replicate on **fresh** family splits and seeds, using the byte-identical H270 code?
How do depths 12 and 16 compare (descriptive)?

## Setup

- **Frozen sources.** [topk.py](topk.py) and [h265_core.py](h265_core.py) are byte-identical to the H270 execution
  source. The smoke reproduces one real H270 cell exactly before compute.
- **Arms.**
  - **CONTROL:** r882.
  - **RERANK8:** confirmatory.
  - **RERANK8_SHUF:** placebo.
  - **RERANK4:** reference.
  - **RERANK12 and RERANK16:** descriptive.
- **Design.**
  - **PRIMARY:** the full 45-cell cross on fresh fold seeds 272140/141/142 × model seeds 272101–272103.
  - **SECONDARY:** DL-Val91 (fold seed 272140).
  - **TERTIARY:** descriptive. The 41 least-similar families are held out as a cluster panel.
- **Decision.** CONFIRMED iff PRIMARY median > +0.002, Holm LCB > 0, 3/3 seeds, beats the placebo, SECONDARY 3/3,
  and no more DL-Train −2.0 picks than r882.
- **Host.** xbabe2 CPU cores 44–47, nice 19, under the H239 A1 guard. The compute stop is 21:40Z.

## Files

| File | Contents |
|---|---|
| [runner.py](runner.py) | phased runner (confirm, depth, tertiary, then DL-Val91) |
| [topk.py](topk.py), [h265_core.py](h265_core.py) | the frozen H270 pipeline (byte-identical) |
| [analysis.py](analysis.py) | protocol, cluster holdout and inference |
| [monitor.py](monitor.py) | the A1 guard |
| [test_h272.py](test_h272.py) | data-free controls |
| [manifest.json](manifest.json), [fold_manifest.json](fold_manifest.json), [provenance.json](provenance.json) | pins, folds, the held-out cluster and the S3 closure |
| [metrics.json](metrics.json) | the metrics schema placeholder |
