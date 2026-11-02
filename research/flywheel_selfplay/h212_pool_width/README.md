# H212 — frozen-r882 candidate-pool width sweep

**CLOSED INCOMPLETE after three authorized attempts; no efficacy verdict.** H212 is not advanced or rejected by scientific outcomes. The frozen preregistration is [PREREG.md](PREREG.md).

Attempt 001 passed frozen r882 validation exactly (91/91, mean `1.1377941898164525`, maximum error `0.0`) but stopped before proposals on a false-positive H58 identity check. The H50 raw TRAIN table matched its selected-record SHA; the H58 gzip payload correctly contained `training_cv_envelope(raw)`.

Attempt 002 passed the same baseline parity and stopped at 390.4 seconds with `KeyError: 'context_sha256'` after proposal generation and Rust evaluation began, but before any registered treatment outcome. H124 returns feature-only dictionaries keyed externally by context ID; `predict_dynamic()` incorrectly expected the ID inside the feature dictionary. The traceback and D2 closeout are retained in [metrics.json](metrics.json).

Attempt 003 was the last owner-authorized attempt. It passed exact baseline parity and the TRAIN-only all-width smoke (N16/N32/N64 proposal generation, Rust evaluation, dynamic prediction, three pool records and an analysis row; scratch removed). At 368.4 seconds it stopped before the treatment loop with `FileExistsError` because the runner created `rust_eval` twice after the smoke. It reported zero registered treatment outcomes. Its raw root remains unchanged. The duplicate directory creation is removed for H214 runner reuse; H212 is closed and will not be retried.

All attempt roots and evidence hashes are listed in [metrics.json](metrics.json). DOPE remained unchanged, CUDA was not initialized, optimizer updates were zero, and no holdout, sealed, or meta-test data was accessed. Three-seed medians, family bootstrap intervals, Holm correction, and Elo are unavailable because no attempt completed treatment scoring.
