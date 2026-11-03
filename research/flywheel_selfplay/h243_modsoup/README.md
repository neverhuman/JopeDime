# H243: zero-update module-wise soup of r882 and r782

**Status: preregistered, not executed.** [PREREG.md](PREREG.md) is the frozen protocol. It merges before any study
outcome is read.

## Hypothesis

**What has been tried.** Every earlier merge (H152, H241, H105, H242) used one global coefficient.

**The idea.** H145 found that r782 beats r882 on TRAIN, while global moves toward r782 hurt DL-Val91 (H241). That
suggests some modules of r782 are better than r882's and others worse.

**The candidate.** Replace one module group of r882 by r782's (β ∈ {0.5, 1.0}), or build a greedy soup of such groups
(Wortsman et al. 2022). This costs 0 training updates.

## Setup

- **Groups.** 11 module groups that partition the 47 adapter tensors, frozen in [groups.json](groups.json).
- **Arms:**
  - **MODSWAP_SELECT (primary):** the best single (g, β) per FIT cell, strict.
  - **GREEDY_SOUP (exploratory):** greedy forward addition on FIT, strict.
  - **PLACEBO:** report-only, a matched-L2 random move of the selected group.
  - **CONTROL:** r882 exactly.
- **Selection.** On DL-Train FIT folds only: seeds 243101–243103, fold seeds 243141–243143, 15 cells. Every decision
  is locked before DL-Val91 is scored once.
- **Decision.** It is made on DL-Val91: median gain > +0.002, family LCB95 > 0, Holm p < .05 over the 2 decision
  arms, and ≥ 2/3 seeds positive. Elo is reported against r882 = 1500. The H152 cliff context #27 is reported
  separately and never excluded.
- **Host.** xbabe2 CPU cores 44–47, nice 19, under the H239 A1 memory guard. The compute stop is 07:30Z.

## Files

| File | Contents |
|---|---|
| [runner.py](runner.py) | the study runner |
| [soup.py](soup.py) | module soups and the placebo |
| [analysis.py](analysis.py) | the selector, greedy soup and inference |
| [monitor.py](monitor.py) | the A1 guard |
| [test_h243.py](test_h243.py) | data-free controls |
| [groups.json](groups.json) | the frozen module groups |
| [manifest.json](manifest.json) | input, asset and source pins |
| [fold_manifest.json](fold_manifest.json) | the per-seed folds |
| [provenance.json](provenance.json) | the real-S3 closure |
| [metrics.json](metrics.json) | the metrics schema placeholder |
