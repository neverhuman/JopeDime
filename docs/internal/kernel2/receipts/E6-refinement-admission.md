# E6 pipeline refinement admission

This partial E6 prerequisite connects the guarded constant optimizer to training-only
pipeline refinement and generation child measurement. The caller supplies a deadline
and cancellation predicate. A false predicate prevents the next prefix, trial or child
measurement; admitted work may finish. Actual trials, including rejected trials, share
one evaluation budget with child measurements. A stopped refinement retains its best
already measured training result and does not read held-out data for a new measurement.

The old refinement API, numerical helpers, legacy observation JSON and all evaluator
identity inputs remain byte-identical. Always-admitted tests compare the old and new
training reports and child records byte-for-byte, including fault failures and budgets.
Malformed training and wrong-cut parents fail before control checks or trial charges.

Acceptance: `cargo test -p jope-pipeline --lib --locked kernel2::refinement_guarded_tests`.
Full baseline/after suites, named exclusions, Clippy ratchets, golden 16, strict emitters,
fitted sanitizer/C++20 matrices and lane audit are retained under
`/home/ubuntu/jope-evidence/kernel2/bt/TRACK_E_06_REFINEMENT_ADMISSION/`.

The shipping search coordinator, CLI profile/fault flags, receipt v4, fitted-C++ default
and EEC clause(b) still need separate qualification before B05. This library slice does
not establish a shipped kernel2 search loop or live qualification.
