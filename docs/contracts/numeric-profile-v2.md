# Numeric profile v2

This is an additive, opt-in CPU fitted-prediction profile. It does not change
legacy DSL/operator behavior, accepted evaluator registrations, model readers,
policy vocabularies, search pools or default emitted bytes. The semantic names
`numeric-profile-v2`, `numeric-operators-v2` and
`ridge-raw-train-range+missing-nonfinite+strict-width+quantile-partition-v2`
are draft I07/I08/I09 integration bindings, not registered evaluator digests.

## Supported behavior

`numeric_v2::NumericModelV2::fit` accepts an explicit schema and only training
rows/targets. Feature NaN/infinity are missing; genuine positive/negative zero
remain observed values. Per-column finite means are fitted on training rows;
all-missing columns use zero. Original missingness is inspected before any
sanitization. Labels must be finite and aligned. Selection uses unique schema
names; prediction requires the same names in the same order. Columns 17/63
are ordinary schema columns, not new native policy tokens. Standalone prediction
functions accept positional rows in this bound schema order; named-schema
validation is performed by the model API and manifest-aware caller.

The supported terminal head is ridge with finite nonnegative `lambda_frac`.
Raw targets are used directly. Ridge retains its existing standardized-feature
safety bound of ±20, but predictions use the training target range widened by
5% instead of the absolute ±60 window. Nonfinite fit, transform or prediction
results return an error. Constant targets use a 1e-9 minimum range to calculate
the margin. Existing legacy `target kind=zscore` was already available; this
profile does not insert it. Explicit target transforms and all other heads
are unsupported rather than silently run with legacy semantics.

Supported transforms are parameterless `impute`, `poly_interact` with positive
integer `topk` and Boolean `append`, and `quantile_bins`/`bin_onehot` with integer
`bins` in 2..64 and Boolean `append`. Unknown parameters/operators fail.
Interactions select exactly `topk` pairs in deterministic diagonal-then-cross
schema order, independent of labels. A request for unavailable pairs fails.
Widths greater than 512 fail instead of silently dropping columns. Quantile
edges use the existing linear interpolation, then a cumulative maximum removes
near-tie interpolation inversions. Partition-point lookup returns the first
edge greater than or equal to the value. This fitted-edge repair has a distinct
semantic identity from the existing opt-in legacy partition-point emitter.

`predict_oof_iid` requires declared IID row semantics and uses seeded shuffled,
balanced assignments. Each fold refits selection/imputation/transforms/ridge on
its training rows only. Ordered/grouped/temporal data are refused by this OOF
caller; their existing split paths remain available through legacy callers.
The legacy `StackModel` and its contiguous folds remain unchanged. V2 does not
claim to repair stack's complete multi-head training/emission implementation.

## Reproducer and support matrix

| Finding | Observed legacy behavior | Successor scope |
| --- | --- | --- |
| Raw targets outside ±60 | Ridge clamps to ±60 | Explicit raw ridge with train-relative bounds |
| Zero versus NaN | Sanitization collapses provenance; impute replaces zero | Pre-sanitize finite mask, train-only means |
| `poly_interact topk` | Values 1 and 2 produce identical old pairs | Exact deterministic pair-count selection |
| Appended width | Legacy cap can discard appended features | Explicit rejection beyond 512 |
| GBM first-N features | Existing `max_features` selects prefix | Unsupported head; no repair claim |
| Columns 17/63 | Parser can read them; old generator pool caps at 8 | Named schema adapter, old pool unchanged |
| Inner folds | Stack uses contiguous folds | Separate IID-only OOF ridge caller |
| Near-tie quantiles | Old helper algorithms may disagree on inverted edges | Monotone fitted edges and lower-bound lookup |

## Actual caller routes and held routes

| Caller | Selection | Support |
| --- | --- | --- |
| `model::fit`, old ops/heads | Existing defaults | Historical behavior retained |
| `transpile::emit`, C++17/JS recipes | Existing `EmitOptions` | Historical defaults and existing explicit partition-point option retained |
| `jope_pipeline::emit::emit_with_options` | Explicit `EmitOptions` | Python/Rust/C++17/C++20/JS recipe option forwarding |
| `jope_cli::kernel::emit_genome_with_options` | Explicit `EmitOptions` | Same recipe routes; fitted C++ refuses partition-point option |
| `numeric_v2::NumericModelV2` | Explicit profile | Supported fitted ridge/transforms above |
| `numeric_v2_emit::emit` | Explicit `OptionsV2` target | Python and Rust prediction-only fitted artifacts |
| `jope_pipeline::emit::emit_numeric_v2` | Profile, train rows, target | Exact source/codec/manifest write and verified reload |
| `jope_cli::kernel::emit_numeric_v2` | Profile, bounded training CSV, target | Kernel API; CSV still requires finite inputs |
| Existing bound/fitted/full emitters | Their existing options | No numeric-v2 support is claimed |
| Root CLI flags, I08/I09 canonical hooks | Integrator-owned | Pending; no automatic default rollout |
| C++/JS numeric-v2 fitted models | No target enum variant | Explicitly unavailable |

The source caller census is retained in lane test evidence. Existing
`advanced_v3::emit_bound` and its option semantics are unchanged. Root CLI
argument allocation, canonical graph/schema joins, I09 artifact registry and
policy/native dispatch are held for the integrator. Recipe option forwarding
is separate from a v2 fitted artifact; requesting partition-point recipes does
not opt a legacy model into the fitted-edge repair.

## Identity, artifact verification and rollback

Fitted artifacts use `numeric-fitted-artifact-v2` / `fitted-prediction-v2`.
The codec binds profile/operator/semantic names, complete schema and selection,
program, training input bit digest, fitted means/edges/pairs/coefficients and
safety range. The manifest binds exact target/options, codec/source SHA256 and
the task-local emitter template source SHA256 and framed numerical source closure
SHA256 (numeric fit, heads, operators, fold helper and RNG). Reload requires caller-supplied
expected options and expected fitted-state digest. It reads the actual shipped
source and codec, checks their bytes, and regenerates source from validated
state to reject a self-consistent source-hash substitution. These are integrity
bindings; they do not establish authorization from a manifest supplied by an
untrusted party.

The integrator must bind this draft adapter to accepted I08 typed schema and
I09 requested-target manifests, then generate the evaluator implementation
identity transition after I07/I09. This lane does not edit historical known
identity rows or shared transition fixtures. Missing dependency acceptance,
identity registration, shared discovery hooks, review and exact-head integrated
CI remain acceptance holds. Numeric scores cannot be compared under the old
evaluator identity. No optimizer update, policy export, serving promotion or
scientific improvement is asserted.

Rollback selects the retained legacy fit/emitter APIs with default options,
retains all old/new artifacts and failed receipts, and records the change of
selection. Disable the v2 caller at integration; never relabel v2 scores as old
evaluator outcomes. Revert through reviewed history if needed, without resets,
force pushes, fixture rewriting or deletion of evidence.
