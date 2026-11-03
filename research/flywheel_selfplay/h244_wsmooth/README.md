# H244 — weight-perturbation prediction smoothing

**PREREGISTERED; NOT RUN.** Primary WSMOOTH_SELECT averages r882 plus eight fixed RMS-scaled Gaussian adapter perturbations; sigma is selected on TRAIN FIT families only. WSINGLE uses the first perturbation at the same selected sigma as a report-only placebo. Zero optimizer updates.

[Preregistered protocol](PREREG.md), [tensor RMS and seeds](perturbations.json), [family folds](fold_manifest.json), [input/source custody](manifest.json), [metrics schema](metrics.schema.json).

Native91 must pass exactly before the single CPU run. All 15 selections are fsync-locked before post-baseline DL-Val scoring. Primary ADVANCE requires median family gain >.002, LCB95>0, Holm m=1 p<.05 and2/3 positive seeds. Context #27 remains included and receives a separate diagnostic. No outcome or speed claim yet; no serving promotion.
