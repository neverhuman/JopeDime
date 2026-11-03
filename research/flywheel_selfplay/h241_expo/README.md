# H241: zero-update adapter extrapolation along the accepted trajectory

**Status: preregistered, not executed.** [PREREG.md](PREREG.md) is the frozen protocol. It merges before any study
outcome is read.

## Hypothesis

**The candidate.** θ(α) = r882 + α·(r882 − r_prev) moves past the last accepted adapter along the accepted flywheel
trajectory. This is ExPO-style extrapolation (Zheng et al. 2024), and it costs 0 training updates.

**Why it is new.** Earlier waves covered interpolation soups (H152), EMA/SWA, ensembles and distillation. None
extrapolated.

## Setup

- **Directions.** r_prev is r782 (primary) or r859 (secondary). Both are the exact saved adapters H152 used.
- **Grid.** α ∈ {0.25, 0.5, 1.0}, plus CONTROL α = 0, which is r882 exactly.
- **What moves.** Adapter tensors only: 47 tensors, 29,211 parameters. DOPE `c51a56d0` is frozen and checked.
- **Selection.** α is selected per (seed, outer) cell on DL-Train FIT folds only, using:
  - seeds 241101–241103;
  - fold seeds 241141–241143;
  - the strict selector, where ties keep r882.

  Every decision is locked before DL-Val91 is scored once.
- **Decision.** It is made on DL-Val91: median gain > +0.002, family LCB95 > 0, Holm p < .05 over the 2 directions,
  and ≥ 2/3 seeds positive. Elo is reported against r882 = 1500.
- **Host.** xbabe2 CPU cores 44–47, nice 19, under the H239 A1 memory guard. The compute stop is 07:00Z.

## Files

| File | Contents |
|---|---|
| [runner.py](runner.py) | the study runner |
| [expo.py](expo.py) | the extrapolation math |
| [analysis.py](analysis.py) | selection and inference |
| [monitor.py](monitor.py) | the A1 guard |
| [test_h241.py](test_h241.py) | data-free controls |
| [manifest.json](manifest.json) | input, asset and source pins |
| [fold_manifest.json](fold_manifest.json) | the per-seed folds |
| [provenance.json](provenance.json) | the real-S3 closure inherited from H237 |
| [metrics.json](metrics.json) | the metrics schema placeholder |
