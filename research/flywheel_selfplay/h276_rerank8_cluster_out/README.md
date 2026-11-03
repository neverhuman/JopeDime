# H276: leave-family-cluster-out test of the confirmed RERANK8 re-ranker

**Status: preregistered, not executed.** [PREREG.md](PREREG.md) is the frozen protocol. It is published on `v1/next`
before anything runs.

## Hypothesis

**What is known.** RERANK8 replicated twice (H272 +0.0633; H273 clean-room +0.0571), but on a held-out cluster of
dissimilar families its lift was small and not significant.

**What H276 asks.** Does RERANK8 still lift single-pick utility over r882 when whole outcome-blind family clusters are
held out? Does the H273 invalid-gate veto keep that lift without extra invalid picks?

## Setup

- **Frozen science.** The H272 science ([topk.py](topk.py), [h265_core.py](h265_core.py)) is byte-identical. One H272
  cell is reproduced exactly before compute.
- **Arms.**
  - **CONTROL:** r882.
  - **RERANK4:** the H265 reference.
  - **RERANK8:** H272 frozen.
  - **RERANK8_VETO:** the H273 invalid gate, τ 0.10 ([veto.py](veto.py)).
  - **RERANK8_SHUF_UNC:** the placebo with fallback disabled.
- **Folds.** 4 pre-declared outcome-blind partitions (k-means k = 5 with 3 seeds, plus Ward k = 5) × 5 held-out
  clusters × 3 model seeds = 60 cells. Inner selection uses family-grouped CV inside the training clusters.
- **Decision.** PASS iff median > +0.010, Holm-level LCB > 0 (two-level cluster bootstrap; Holm m = 2 over RERANK8
  and RERANK8_VETO), 3/3 seeds, ≥ 4/5 clusters non-negative in every partition, −2.0 picks ≤ r882, and placebo
  ≤ +0.005.
- **Host.** xbabe2 CPU cores 44–47, nice 19, with a swap-in-rate guard. The compute stop is 23:00Z.

## Files

| File | Contents |
|---|---|
| [runner.py](runner.py) | the cluster-out runner |
| [veto.py](veto.py) | the RERANK8_VETO (H273 invalid gate) |
| [topk.py](topk.py), [h265_core.py](h265_core.py) | the frozen H272/H270 science (byte-identical) |
| [analysis.py](analysis.py) | the clusterings, cluster bootstrap and PASS gates |
| [analysis_h272.py](analysis_h272.py) | the H272 analysis, used only for the regression cell |
| [monitor.py](monitor.py) | the A1 guard with the swap-in-rate pause |
| [test_h276.py](test_h276.py) | data-free controls |
| [manifest.json](manifest.json), [fold_manifest.json](fold_manifest.json), [provenance.json](provenance.json) | pins, the clusterings and the S3 closure |
| [metrics.json](metrics.json) | the metrics schema placeholder |
