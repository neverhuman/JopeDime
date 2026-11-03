# H283: frozen RERANK8_VETO on a fresh regression cohort (GPU-exact)

**Status: preregistered, not executed.** [PREREG.md](PREREG.md) is the frozen protocol. It is published on `v1/next`
before any fresh context is built or scored.

**What it tests.** The frozen H278 RERANK8_VETO models are compared against r3175's own slot-0 pick on 60 fresh
regression families that no flywheel round, RERANK fit or selection has touched. This is the EVEN half of the shared
`fresh_cohort_v1.json` (sha256 `12d2b1f3…`).

**How.**
1. The contexts are built with the corpus's own construction, regression-checked bit-exact on DL-Val91.
2. They are scored on xbabe1's GPU with H281's writer-path scorer, after an in-process exact r3175 regression.

## Files

| File | Contents |
|---|---|
| [fresh_cohort.py](fresh_cohort.py) | the shared outcome-blind cohort (metadata only) |
| [build_fresh.py](build_fresh.py) | fresh contexts: train-only table, pool and H58 features (with a corpus regression mode) |
| [gpu_rescore.py](gpu_rescore.py), [run_gpu.sh](run_gpu.sh) | xbabe1 GPU: the r3175 DL-Val91 regression, then fresh scoring |
| [evaluate.py](evaluate.py), [analysis.py](analysis.py) | xbabe2 CPU: frozen models and the random null; the primary rule |
| [test_h283.py](test_h283.py), [manifest.json](manifest.json) | data-free controls; pins |
| [metrics.json](metrics.json) | the metrics schema placeholder |
