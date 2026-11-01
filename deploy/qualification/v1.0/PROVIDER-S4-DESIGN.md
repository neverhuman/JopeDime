# REL-07 S4 objective authority

The S4 provider verifies the supplied full, non-development candidate and its
embedded source SHA, then requires the same clean source checkout before,
between and after four exact `jope-service` tests. It writes a new external
`jope.rel07.gate-execution/v1` receipt bound to the full artifact SHA-256.

The selected product assertions cover:

- A newly defined custom maximization objective uses the runtime parser/scorer
  and drives both seed and proposal promotions in the actual discovery loop,
  despite an opposing RMSE surrogate. Every rung-1 decision is observed.
- Switching that custom objective between minimization and maximization changes
  elite order while preserving every prepared ranker input bit, revealed and
  proposal identity, and dimensions for both normal and exploration slates.
- Elite capacity and all combinations of objective/surrogate direction preserve
  user-objective membership and promotion rank, independently of the surrogate.

The custom expression is `rel07_untrained_pred_mean`, outside the trained metric
catalog; its scores use the real runtime objective implementation. Controlled
predictions and executor responses isolate the scheduling boundary. They do not
claim learned-weight quality, native model-fit parity, Q1 or serving promotion.
The unchanged-input comparison holds the same measured programs, surrogate
measurements, features and conditioning fixed. It does not require identical
contexts after the objective legitimately changes elite membership.

Each selector must execute exactly one named passing test. Missing, extra,
ignored, measured, duplicate and failed tests refuse PASS, as do source/artifact
drift, unverified candidates, symlinks, non-files and existing result paths.
Refusals exit 2; a failing test retains its exit status. CPU jobs are bounded to
eight with incremental compilation disabled and line-table debug information.
No dataset, GPU, signing or network service is required by the provider itself.

Invoke through `release-gates` using
`python3 -B deploy/qualification/v1.0/providers/s4.py` and the five standard
`JOPE_REL07_*` bindings in `GATE-RUNNER.md`. Run the 19 data-free orchestration
controls with `python3 -B deploy/qualification/v1.0/test_provider_s4/test_s4.py -v`.
Those synthetic candidate/child controls are not actual release qualification;
the selected Rust assertions are also run on the source under review. A real
candidate-bound S4 result requires the full release artifact supplied by its owner.

Reusing the existing strict S3/T1 provider contract avoids another receipt schema
or a generic successful-command wrapper. No production behavior, public API,
frozen model, dependency, worker protocol or release-controller wiring changes.
