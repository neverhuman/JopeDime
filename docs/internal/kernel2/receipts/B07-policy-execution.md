# Standalone numerical policy execution

Hard prerequisite of the numerical-policy card, following reader PR171. This
slice executes an explicitly selected bounded scalar program in the interpreter
and derived postorder compiler. It does not admit legacy models, fit state,
recipes, shipping search or generation. Full-card qualification remains held.

Base: `f36fc2be83138d2d132335267ee528fb7dff91c0`; sole reused lane
`bt-implementation`, private branch `kernel2/bt07-scalar-execution`.
Exact source, commands, environments, times, counts and immutable evidence:
`/home/ubuntu/jope-evidence/kernel2/bt/B07_V2_SCALAR_EXECUTION`.

## Execution contract

- Explicit `evaluate_row`/`evaluate_element` return `ScalarEvaluation`: a typed
  value/refusal and the faults observed before that result. Hard errors never
  become successful zeros. Row scope refuses `x`; missing columns refuse with
  `Bounds`. Interpreter and compiler preserve left-to-right operand order and
  stop at the same first error, retaining preceding faults.
- `LegacyV1`, including a missing selector, reuses existing raw protected
  variable/function/binary operations. Legacy intermediate values are not
  clamped at each node: the existing final sanitizer runs after the whole tree.
  The old interpreter and compiler bodies remain unchanged.
- Only explicit `StableV2` selects the already-reviewed asinh and four row
  reduction helpers. Their certified local bounds and fault sets are retained,
  including magnitude faults before rounded boundary tails disappear. Other
  functions, arithmetic and row standard deviation remain legacy operations.
  No new variance contract is claimed.
- Already-qualified inverse-link helpers execute with their owner-approved
  Acklam/central-logit contracts. Func21/22 remain parseable but lack qualified
  execution kernels: either anywhere in the AST refuses the entire execution
  and compilation before input reads. No textbook or legacy substitute runs.
- Construction/import/decoding retain the existing 64-node/16-depth bounds and
  arity checks. Derived code has at most 64 postorder instructions, retains its
  policy, and has no independent codec or mutable fitted state.
- Evaluating a decoded program leaves its bytes and domain-separated identity
  unchanged. Successful standalone execution does not make its legacy genome,
  model or any existing export route executable, nor make a reserved operator
  generatable. Decode itself still performs no evaluation.

The bridge lives in a non-hashed child of the existing compiled module so it
can reuse raw protected operations without copying or editing the frozen root
evaluator. No evaluator-identity input, historical fixture, v1 ID/codec/model,
default emission byte, helper core/twin or dependency changes. No regeneration
or identity rebaseline occurs.

## Tests first and no-regression gates

The initial focused test failed to compile on untouched main because the new
standalone API was absent. This is explicitly API-absence RED, not a reproduced
runtime regression; the original source/log/command is preserved in
`focused-red/`. Pre-freeze focused execution, policy reader and compatibility
tests passed 39/39, including thirteen new nonignored execution regressions.
Exact-head focused and complete after-gates remain required after freezing.

Acceptance:

```sh
cargo nextest run --locked -P pr -p feat-math --test numerical_policy_execution --test numerical_policy_ingress --test historical_v1 --test compat_freeze
cargo clippy -p feat-math --all-targets --locked -- -D warnings
just required
```

Fresh untouched-main baseline qualified: PR2070/29, main4150/66, Python231/2,
golden16/16, fitted ASan/UBSan18/18 and C++2018/18, original O3 plus additional
C++17 O2 emitter passes. After must add exactly thirteen tests, with every old
selection/skip/ignore/exclusion unchanged. All93 protected/38 identity inputs
are pinned. The initial qualifier's too-narrow Clippy reference and corrected
byte proof of six exposed held diagnostics are preserved separately.

Actual `just required` remains101 on the owner-accepted81-location/24-path
design floor: **NOT PASS**, and downstream required stages did not execute.
Full standing suites run separately; zero new diagnostics is still mandatory.
Read-only lane audit, exact-head automatic/manual CI and fresh independent
native approval precede guarded writer rebase merge. Prior merged-main guard
must succeed before publication; pending is not success.

## Remaining per-operation definition of done

Shortest-tree comparison, six-dataset utility, full runtime/interpreted/compiled
and emitted twins, guarded faults/arity/width/saturation, bounded Experimental
prior, equal-budget promotion, token/byte delta, lifecycle, train-only fit,
generation removal with parsing retained, append-only identities, EEC creation/
change/removal plus Rust/Python/C++17 train-predict recipes and fitted C++, IP
scrub and no new dependency remain the full card's gates. This slice establishes
standalone interpreted/compiled execution only. Model/fit/artifact bindings,
full recipes, Func21/22 numerical execution, profiles/generation/EEC and utility
remain held. No whole B07, B08, release or promotion completion is claimed.
