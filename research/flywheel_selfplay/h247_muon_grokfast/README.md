# H247: Muon and Grokfast-EMA for the r882 adapter continuation

**Status: preregistered, not executed.** [PREREG.md](PREREG.md) is the frozen protocol. It merges before anything
runs.

## Hypothesis

**What has been tried.** Every earlier optimizer variant of the 40-update r882 continuation tied with or lost to r882.

**What is new.** Two update-direction geometries: Muon (orthogonalized Nesterov momentum on 2-D adapter matrices,
Jordan et al. 2024) and Grokfast-EMA (slow-gradient amplification, Lee et al. 2024). The question is whether either
lifts DL-Val91 over r882.

## Setup

- **Arms.** All use the same CTRL loss, batches, data order, clip 1.0 and 40 updates from a read-only r882 copy:
  - **CONTROL:** exact H170 CTRL AdamW.
  - **MUON:** lr 2e-4, momentum .95 Nesterov, Newton–Schulz 5. Matrices only; other tensors use CTRL AdamW.
  - **GROKFAST:** α .98, λ 2.0 on CTRL AdamW.
- **Hypothesis family.** MUON and GROKFAST are co-primary, with Holm m = 2.
- **Design.** One fresh fold seed 247140, seeds 247101–247103, 15 cells per arm, the strict inner selector, and
  selections locked before assessment.
- **Decision.** It is made on DL-Val91: median gain > +0.002, LCB95 > 0, Holm p < .05, ≥ 2/3 seeds positive.
- **Also reported:** the step-40 unconstrained diagnostic and the cliff context #27.
- **Host.** xbabe2 CPU cores 44–47, nice 19, under the H239 A1 memory guard. The compute stop is 09:30Z.

## Files

| File | Contents |
|---|---|
| [runner.py](runner.py) | the study runner |
| [treatment.py](treatment.py) | Muon, Grokfast and CTRL |
| [analysis.py](analysis.py) | selection and inference |
| [monitor.py](monitor.py) | the A1 guard |
| [test_h247.py](test_h247.py) | data-free controls |
| [schedules.json](schedules.json) | the frozen settings |
| [manifest.json](manifest.json) | input, asset and source pins |
| [fold_manifest.json](fold_manifest.json) | the folds |
| [provenance.json](provenance.json) | the real-S3 closure |
| [metrics.json](metrics.json) | the metrics schema placeholder |
