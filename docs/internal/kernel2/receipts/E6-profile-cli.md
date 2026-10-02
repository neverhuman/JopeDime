# E6 opt-in profile CLI and v4 evidence

This partial E6 prerequisite exposes the merged bounded profile coordinator through
`jope pipeline run`. It is required before the remaining Track E shipping/EEC
gates and B05. Legacy is still the default search profile; legacy run/evolve
dispatch, default emissions and receipt v1–3 remain unchanged.

Kernel2 runs require an explicit profile and refinement allowance, for example:

```sh
jope pipeline run --dataset DATASET --output OUTPUT \
  --search-profile kernel2-baseline --fault-policy reject-hard \
  --refine-top-k 1 --refine-trials 4 \
  --budget evaluations=64,seconds=300 --policy skip
jope pipeline verify-receipt OUTPUT/receipt.json --evidence
```

`kernel2-experimental` is also opt-in. Kernel2 defaults to `reject-hard`;
`penalize` requires explicit nonnegative `--lambda-domain`,
`--lambda-saturation` and `--fault-tie-break true|false`. Refinement limits are
explicit because their CLI defaults have not been settled. Zero limits disable
refinement. Every candidate measurement and objective trial spends the same
bounded allowance, including refused measurements and faulting trials.

Without `--emit`, kernel2 adds `cpp17-fitted` to the existing five languages.
An explicit emit list is honored exactly. Emission uses the existing recipe and
fitted backends and their exclusion reporting. Seeds are validated under the
selected profile before output is created. This slice supports one cut, one
thread and one island; unsupported concurrency, multiple cuts, incumbent receipts
and `--policy require` fail closed. The old legacy paths retain their capabilities.

Only kernel2 runs publish `jope.pipeline-receipt/v4`. The new schema is generated
by `cargo run --locked -p jope-pipeline --example receipt_v4_schema` and checked
against its Rust type. The receipt binds the shipped profile ID/digest, policy,
arm counts, full fault telemetry, candidate/objective/refinement accounting,
selected measurement and hashes of prepared/search/emitted evidence.
Observations and refinements are retained as sidecars; a refinement references
its child measurement rather than silently spending an additional allowance.

Evidence verification checks safe paths and file hashes, re-derives fault rates,
penalized scores, selection and refinement accounting, and binds the selected
program to the emission manifest and artifact bytes. It checks retained evidence;
it does not replay fits, reconstruct individual arm draws, authenticate a run or
establish live learning-loop qualification.

Acceptance:

```sh
cargo test --locked -p jope-pipeline --lib kernel2::cli_tests
cargo test --locked -p jope-cli --test pipeline_cli kernel2_
```

Fourteen nonignored library cases cover CLI defaults/options, real profile runs,
explicit penalties, seed/refinement accounting, receipt/evidence tampering,
safe artifact paths, unsupported shapes, cancellation, deterministic replay and
v4 schema freshness. Two umbrella CLI cases prove real child-process dispatch,
default fitted-C++ emission, evidence verification and refusal exit codes.
Implicit and explicit legacy CLI options produce identical receipt bytes with a
frozen clock. Existing schema v1–3, evaluator identity inputs, frozen fixtures
and emitter output remain byte-unchanged; no fixture regeneration is used.

Remaining gates: multiple-cut comparison, concurrent/island coordination,
incumbent/evolve integration, E7 policy interplay, EEC clause(b) and E8 equal-budget
qualification. This slice does not qualify all of E6, B05, a default flip or a
release. Exact source/gate/CI/review lifecycle evidence is retained outside the
checkout under `jope-evidence/kernel2/bt/TRACK_E_06_PROFILE_CLI`.
