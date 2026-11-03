# H242: zero-update LASER-style spectral rank reduction of the r882 adapter

**Status: preregistered, not executed.** [PREREG.md](PREREG.md) is the frozen protocol. It merges before any study
outcome is read.

## Hypothesis

**The candidate.** Remove the small-singular-value components of r882's own adapter matrices. This follows LASER
(Sharma et al. 2023) and costs 0 training updates. Each 2-D adapter weight matrix keeps its top-k SVD components
holding at least the energy fraction ρ of Σσ².

**Why it is new.** No earlier wave did spectral denoising or rank truncation.

## Setup

- **Arms:**
  - **SVD_SELECT (primary):** ρ ∈ {.99, .95, .90}, applied to all 16 matrices and chosen per FIT cell.
  - **LASER_SINGLE (exploratory):** one matrix truncated at ρ = .90, chosen on FIT.
  - **RANDPROJ:** a report-only placebo with matched per-matrix Frobenius change.
  - **CONTROL:** ρ = 1, which is r882 exactly.
- **What changes.** Adapter tensors only. DOPE `c51a56d0` is frozen and checked.
- **The census.** [spectral_inventory.json](spectral_inventory.json) freezes the matrix list, shapes and kept ranks.
- **Selection.** Strict selection on DL-Train FIT folds: seeds 242101–242103, fold seeds 242141–242143, ties keep
  r882. Every decision is locked before DL-Val91 is scored once.
- **Decision.** It is made on DL-Val91: median gain > +0.002, family LCB95 > 0, Holm p < .05 over the 2 decision arms,
  and ≥ 2/3 seeds positive. Elo is reported against r882 = 1500.
- **Host.** xbabe2 CPU cores 44–47, nice 19, under the H239 A1 memory guard. The compute stop is 07:30Z.

## Files

| File | Contents |
|---|---|
| [runner.py](runner.py) | the study runner |
| [spectral.py](spectral.py) | the truncation, placebo and census |
| [analysis.py](analysis.py) | selection and inference |
| [monitor.py](monitor.py) | the A1 guard |
| [test_h242.py](test_h242.py) | data-free controls |
| [manifest.json](manifest.json) | input, asset and source pins |
| [fold_manifest.json](fold_manifest.json) | the per-seed folds |
| [spectral_inventory.json](spectral_inventory.json) | the frozen matrix census |
| [provenance.json](provenance.json) | the real-S3 closure |
| [metrics.json](metrics.json) | the metrics schema placeholder |
