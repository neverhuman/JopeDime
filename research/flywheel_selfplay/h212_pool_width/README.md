# H212 — frozen-r882 candidate-pool width sweep

**Attempts 001 and 002: INCOMPLETE due to runner defects; no efficacy verdict. Attempt 003: owner-authorized D3 and last authorized attempt.** The frozen preregistration is [PREREG.md](PREREG.md).

Attempt 001 passed frozen r882 validation exactly (91/91, mean `1.1377941898164525`, maximum error `0.0`) but stopped before proposals on a false-positive H58 identity check. The H50 raw TRAIN table matched its selected-record SHA; the H58 gzip payload correctly contained `training_cv_envelope(raw)`. Its root remains unchanged.

Attempt 002 used the merged H58 envelope fix and again passed exact baseline parity. It stopped at 390.4 seconds with `KeyError: 'context_sha256'` after proposal generation and Rust evaluation began, but before any candidate utility outcomes were reported. The traceback identifies `predict_dynamic()` reading `context_row["context_sha256"]`; H124 supplies feature-only dictionaries keyed externally by context ID. This is corrected by passing the context ID explicitly. Attempt 002's root remains unchanged and is retained as raw evidence. This is deviation D2.

The runner now requires an end-to-end smoke on one TRAIN context across widths 16, 32, and 64 before treatment processing. The smoke performs proposal generation, Rust evaluation, dynamic prediction, pool selection, and analysis-row writing in a temporary scratch root, validates its receipt, then removes the scratch. It is bound to the exact merged source through the launch manifest. A regression test covers the smoke contract and explicit context identity path.

Owner nudge `hw_t0200_luna.md` authorizes attempt 003 as deviation D3 and the last attempt, with the same preregistered design and guards. If attempt 003 fails on infrastructure, close H212 as INCOMPLETE and stop. At finish, report the three-seed median of equal-family mean gains, family bootstrap CI, Holm m=2, and Elo versus r882 (anchor 1500; ties REJECT). No AUC or speed verdicts.

See [metrics.json](metrics.json) for closeout records. No protected, sealed, holdout, or meta-test payload was read.
