# H139 — winner's-curse / selection-rule audit of the r882 plateau

**Status: preregistered (CPU, zero optimizer updates); not yet run.** See [PREREG.md](PREREG.md).

- **Question.** Is r882's 1.1377942 on the flywheel's 91-context val a selection artefact of single-argmax admission?
- **Arms.** R782, R859 and the rejected round-2482 candidate, compared with R882 on DL-Train.
  - DL-Train is independent of the selection step but not of training.
  - Comparisons are family-grouped, with a bootstrap CI and Holm.
  - R1703 and R2481 are disclosed as bitwise r882 copies.
- **Regression first.** R882 must reproduce 1.1377942 on all 91 val contexts.
- **Secondary.** Val-vs-DL-Train rank agreement.
- **Decision.** SELECTION_ARTEFACT vs PLATEAU_REAL, followed by a proposed bootstrap-median admission rule.
- **Code.** [runner.py](runner.py) reuses the pinned H124 forward pass. [test_h139.py](test_h139.py) has 9 synthetic
  controls.
