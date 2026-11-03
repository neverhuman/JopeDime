# H264: zero-update kNN residual memory on frozen r882 embeddings

**Status: preregistered, not executed.** [PREREG.md](PREREG.md) is the frozen protocol. It is published on `v1/next`
before anything runs.

## Hypothesis

**What is known.** Every gradient continuation of r882 was flat or negative, and the parametric adapter looks
data-saturated (H176).

**What is new.** Non-parametric memory. A kNN residual over DL-Train FIT (context, candidate) pairs, keyed by frozen
r882 embeddings, adds information without moving any weight. The question is whether it lifts DL-Val91 single-pick
utility over r882 with no reveals.

## Setup

- **Policy.** score = r882 cost + λ × the softmax(−d/τ)-weighted mean of the k nearest FIT residuals (true cost −
  r882 cost), excluding the query's own family. Keys are normalised concat(context, candidate) embeddings. Memory and
  queries use each context's 64 lowest-r882-cost candidates (compute bound; outcome-free).
- **Arms.**
  - **KNN (primary):** (k, τ, λ) from {8, 32, 128} × {0.05, 0.2} × {0.25, 0.5, 1.0}, chosen by the strict inner
    selector on SELECT; λ = 0 (r882) wins ties.
  - **KNN_SHUF (placebo):** the same config with residuals permuted within family.
  - **CONTROL:** r882.
- **Design.** Zero optimizer updates. One fresh fold seed 264140, seeds 264101–264103, 15 cells, selections locked
  globally before any ASSESS or DL-Val91 read.
- **Decision.** On DL-Val91 single-pick: median family gain > +0.002, LCB95 > 0, Holm p < .05 (m = 2), 3/3 seeds
  positive, and KNN beats KNN_SHUF.
- **Host.** xbabe2 CPU cores 44–47, nice 19, under the H239 A1 memory guard. The compute stop is 19:45Z.

## Files

| File | Contents |
|---|---|
| [runner.py](runner.py) | the study runner |
| [treatment.py](treatment.py) | the frozen forward and embedding extraction |
| [analysis.py](analysis.py) | the shortlist kNN search and residual, the selector and inference |
| [monitor.py](monitor.py) | the A1 guard |
| [test_h264.py](test_h264.py) | data-free controls |
| [manifest.json](manifest.json) | input, asset and source pins |
| [fold_manifest.json](fold_manifest.json) | the folds |
| [provenance.json](provenance.json) | the real-S3 closure |
| [metrics.json](metrics.json) | the metrics schema placeholder |
