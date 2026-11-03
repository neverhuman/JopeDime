# H278: frozen RERANK8_VETO export and transfer to the r3175 pool

**Status: preregistered, not executed.** [PREREG.md](PREREG.md) is the frozen protocol. It is published on `v1/next`
before any evaluation outcome is read.

## Hypothesis

**What is known.** RERANK8_VETO passed H270/H272/H273/H276, but no frozen model weights were ever exported, so the
flywheel shadow observer is blocked (BLOCKED_MISSING_FROZEN_MODEL).

**What H278 does.** It fits one frozen model per seed on all DL-Train, exports it, checks that it reproduces the lift,
and tests whether the lift transfers to the live incumbent r3175's top-8 pool.

## Setup

- **Part A.** 3 frozen RERANK8_VETO models (seeds 278101–278103) are fit with the byte-identical H276 science. The
  hyperparameters come from family-grouped inner CV on DL-Train. The exported text models are reload-verified
  bit-exact, a fit-lock is written before any DL-Val91 read, and everything is exported to S3 with an export manifest.
- **Part B.**
  - **P1:** DL-Val91, r882 top-8 (supporting).
  - **P2:** r3175 top-8 (the adoption gate). Its pre-condition, an exact re-score of r3175's stored utility, fails on
    CPU (1.1366829 vs 1.1389572; 90/91 contexts; a near-tie). P2 is therefore declared BLOCKED, and the CPU-r3175
    comparison is reported as EXPLORATORY only.
- **Host.** xbabe2 CPU cores 44–47, nice 19, with the swap-in-rate guard. The compute stop is 00:30Z.

## Files

| File | Contents |
|---|---|
| [runner.py](runner.py) | Part A (freeze, export, fit-lock) and Part B (transfer evaluation) |
| [freeze.py](freeze.py) | the frozen fit (veto.run_cell_veto without an ASSESS fold) and exported-model inference |
| [topk.py](topk.py), [h265_core.py](h265_core.py), [veto.py](veto.py) | the H276 science (byte-identical) |
| [analysis.py](analysis.py), [analysis_h272.py](analysis_h272.py) | Part B statistics; the H272 analysis is used only for the regression cell |
| [monitor.py](monitor.py), [test_h278.py](test_h278.py) | the A1 guard; data-free controls |
| [manifest.json](manifest.json), [fold_manifest.json](fold_manifest.json), [provenance.json](provenance.json) | pins, the inner family folds and the S3 closure |
| [metrics.json](metrics.json) | the metrics schema placeholder |
