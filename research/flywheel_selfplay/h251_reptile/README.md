# H251: first-order Reptile family meta-continuation of the r882 adapter

**Status: preregistered, not executed.** [PREREG.md](PREREG.md) is the frozen protocol. It merges before anything
runs.

## Hypothesis

**What has been tried.** Every gradient continuation of r882 (AdamW, one-cycle, SAM, Lion, gradient noise, clipping,
GroupDRO/ANDMASK, AdaBelief/RAdam, Muon, Grokfast, the low-rank side branch) tied with or lost to r882. They all move
toward pooled-loss minima that do not transfer across families.

**What is new.** Meta-learning across families. First-order Reptile (Nichol et al. 2018) adapts on ONE family per
outer step and moves the weights part of the way toward the adapted weights, which implicitly rewards gradients that
agree across minibatches. The question is whether that lifts DL-Val91 over r882.

## Setup

- **Arms.** All use the same CTRL loss, 8 × 24 batch shape, clip 1.0 and 40 gradient evaluations per cell from a
  read-only r882 copy:
  - **CONTROL:** exact H170 CTRL AdamW.
  - **REPTILE (primary):** 8 outer steps × 5 inner plain-SGD steps (lr 1e-3) on one FIT family, then
    θ ← θ + 0.5 (φ − θ).
  - **REPTILE_K1 (mechanism control):** 40 outer steps × 1 inner step, same lr, ε and family stream.
- **Hypothesis family.** REPTILE and REPTILE_K1 against r882, with Holm m = 2. REPTILE − REPTILE_K1 tests the mechanism.
- **Design.** One fresh fold seed 251140, seeds 251101–251103, 15 cells per arm, the strict inner selector, and
  selections locked before assessment.
- **Decision.** It is made on DL-Val91: median gain > +0.002, LCB95 > 0, Holm p < .05, ≥ 2/3 seeds positive.
- **Also reported:** the step-40 unconstrained diagnostic, REPTILE − CONTROL, REPTILE − REPTILE_K1, drift, harm
  families and the cliff context #27.
- **Host.** xbabe2 CPU cores 44–47, nice 19, under the H239 A1 memory guard. The compute stop is 15:00Z.

## Files

| File | Contents |
|---|---|
| [runner.py](runner.py) | the study runner |
| [treatment.py](treatment.py) | Reptile and CTRL |
| [analysis.py](analysis.py) | the family plan, selection and inference |
| [monitor.py](monitor.py) | the A1 guard |
| [test_h251.py](test_h251.py) | data-free controls |
| [schedules.json](schedules.json) | the frozen settings |
| [manifest.json](manifest.json) | input, asset and source pins |
| [fold_manifest.json](fold_manifest.json) | the folds |
| [provenance.json](provenance.json) | the real-S3 closure |
| [metrics.json](metrics.json) | the metrics schema placeholder |
