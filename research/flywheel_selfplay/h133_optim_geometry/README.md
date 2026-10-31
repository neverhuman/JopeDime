# H133 — optimizer geometry / step size vs harm

**Status: preregistration READY; main study not run.** H133 tests whether reducing AdamW's step size or replacing adaptive normalization with momentum SGD retains family utility while reducing harm, using fresh `r882` copies and frozen DOPE. The only assessment split is the new `fold 0` family partition; all calibration before preregistration uses fit folds 1–4 only and records parameter drift, not an outcome metric.

Calibration implementation is pinned in [calibrate_fit_only.py](calibrate_fit_only.py). See [PREREG.md](PREREG.md), [manifest.json](manifest.json), [fold_manifest_133100.json](fold_manifest_133100.json), and the pinned [fit-only calibration receipt](calibration_fit_only.json). A prior local candidate generated from combined selection metadata was rejected and is not used. The committed manifest is regenerated from `train_only_export.json`, which contains only the streamed top-level DL-Train rows (`sha256=ffe506f68d7aade13a7eef6f0cf55d6e79b200dd5e742f8f035dccd0ee27e3ff`).

Fit-only calibration selected SGDM LR `0.003` (CONTROL-160 drift `2.06922246`; SGDM drift `1.75007866`). Attempt 1 was voided on the load guard and is disclosed in `PREREG.md`; attempt 2 completed with zero fold-0 payload or outcome reads. Main experiment execution and all assessment scoring remain gated on the preregistration merge.
