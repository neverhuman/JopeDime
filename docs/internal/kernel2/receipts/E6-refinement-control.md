# Track E6 — per-trial refinement admission

This focused slice is a hard prerequisite of workplan PR14/E6 shipping search:
its deadline/cancellation predicate must be consulted before every objective
trial, including the initial trial. The existing E2 optimizer has no stop hook.
The E6 shipping loop, CLI search-profile/fault-policy flags, v4 receipts and
fitted-C++ default integration remain pending, as do EEC clause(b) and B05.

`MathExpr::optimize_constants_v2_guarded` accepts the existing profile, scope,
training-only rows/targets and objective plus `GuardedRefinementBudget`.
A false predicate starts no expression/objective evaluation and spends no trial.
A stop between trials returns the best completed result and actual counts.
A failed initial score returns no score and reports one rejected evaluation;
invalid configuration is refused before admission. Rejected subsequent trials
remain charged. The caller supplies its clock/cancellation/evaluation admission
policy; this library API does not implement a shipping deadline on its own.

The new cursor uses the existing private training scorer and AST helpers.
Original `optimize_constants_v2`, numerical helpers and public result types
are byte-unchanged. An always-true predicate produces the same expression,
score/scale bits, callback order and evaluation/refusal counts as the old API.
No scalar/operator, Model/codec, emitter, RNG/default, dependency, frozen fixture
or evaluator-identity change or regeneration is introduced.

## Acceptance

```sh
cargo test -p feat-math --locked --test kernel2_refinement_control
```

Seven nonignored tests cover always-allowed row/negative-constant parity,
callback/elementwise trial order, stop-before-initial, mid-search retention,
faulted trials, initial failure and invalid preflight. The tests-first missing
API failure is retained. Fixture corrections are also retained: the callback
must handle fallible row evaluation; the first negative-constant expression
exceeded the settled profile node limit and was reduced to a legal expression.
No profile limit or product behavior was weakened.

Fresh untouched-main baseline `add33f04f2960b85dc58145456ffa0fe85c9a7b0` was
qualified before source edits: PR1960/29, main4040/66, Python231/2,
golden16/16, fitted ASan/UBSan18 and C++2018. Identical full13 after commands
must add exactly seven tests (PR1967/29 and main4047/66), preserving every old
selected/skipped/ignored/excluded name. Clippy is the Amendment1 zero-new ratchet;
touched feat-math passes. Strict original O3 plus additional g++17 O2 outputs
must match; all38 identity inputs and71 frozen files must remain byte-identical.
Lane-only audit, exact-head automatic/manual CI, independent native approval and
guarded rebase merge are required before landing.

Durable receipts, failures, command/time/exit inventories and lifecycle readbacks:
`/home/ubuntu/jope-evidence/kernel2/bt/TRACK_E_06_PIPELINE/`.
