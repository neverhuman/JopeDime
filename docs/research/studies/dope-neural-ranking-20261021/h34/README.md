# H34: fixed regret weighting versus ordinal and cached MSE

Status: **VERIFIED_DEVELOPMENT_ONLY_NO_EXTERNAL_CONFIRMATION**. This is an adaptive development study on historically exposed 850 contexts/162 components, not independent confirmation or an established Dope/landmarking lift.

Four new fixed 120-requested-tree GAP/ORDINAL models use the identical REAL636 representation and exact gradient input/scaler/support/native policy of two cached H33 REAL MSE controls. No MSE refit. Primaries are GAP−MSE and GAP−ORDINAL: minimized raw constrained fitness, equal component/context weights,100000 paired known-union draws, central98.75% each/FW97.5%. Both adjusted upper bounds<0 and matched support are required; external lift is never inferred here.

Read [protocol](protocol.json), [results](results.json), [all curves](curves.json), [failure/cost ledger](failure-cost-ledger.json), and [source identities](source-receipts.json). Pending endpoints, curves, fitted counts, costs and unknowns remain null. Producer-only values remain marked unverified until the full different-author replay and exact root acceptance close.

Curves retain every supplied observation. Fit TRAIN 24 points and optional reconstructed TRAIN 24 + MONITOR 24 are separate records of the same four models, not new independent models. Expose actual fitted tree count and effective min(requested,fitted) for 20/40/60/80/100/120. Repeated effective prefixes are not distinct trained checkpoints. All diagnostics are postfit; no live validation, intermediate OOF, early stopping or favorable checkpoint selection.

Own GAP/ORDINAL losses have different weighting/units. Common descriptive weighted centered L2 retains prediction offsets, is not decision regret, and is not pooled across fold scales. Gate failures retain full denominators, without inference filtering. Same native prediction engine and FP64-callback/native-FP32 optimization limits remain. Earlier source/control failures and costs remain separate from actual model attempts.

This capsule draft is neither publication, Git commit nor PR. Root owns final source-bound approval and publication; full private arrays/models/rows/locators/grants remain outside public Git.
