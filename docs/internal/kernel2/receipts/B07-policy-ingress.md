# Numerical policy program ingress

This is a parse-only prerequisite of the numerical-policy card. It carries an
explicit contract in the existing bounded scalar envelope; it does not admit
an evaluator, model, fitted predictor, recipe or shipping profile.

Base: `1251d4ed69d1f5f0d5180ee789de018bbca29d08`, after numerical helper PR170;
main-guard 37161722858 attempt 1 SUCCESS. Sole reused lane `bt-implementation`,
branch `kernel2/bt07-policy-ingress`. Exact source, commands and immutable logs:
`/home/ubuntu/jope-evidence/kernel2/bt/B07_V2_POLICY_INGRESS`.

## Contract and compatibility

- `NumericalPolicy::LegacyV1` is the default. Existing constructors, typed v1
  import and selector-free scalar payloads produce the exact existing bytes.
- Explicit `StableV2` adds required-feature bit 8 and byte selector 2 before the
  unchanged scalar AST. Existing inverse-link bit 2 and exponential bit 4 stay
  derived from that AST: accepted masks are 2/6 without policy and 10/14 with it.
  Readers predating this feature reject the unknown required bit before the AST.
- The selector records `v2` and the existing helper identity
  `feat-math.fdlibm-neumaier-lassq.v2`. Func13 and every existing Func/Var/node
  tag retain their meanings in the old codec; no new numeric operator ID is used.
- Unknown policy/version/feature declarations return `UnsupportedVersion`.
  Missing bytes, inconsistent body features, trailing bytes and exceeded
  budgets return `Malformed`. A bit-only policy downgrade cannot silently parse
  the remaining selector as a valid legacy scalar body.
- Explicitly selecting v1 removes the selector and restores the original bytes.
  The existing domain-separated program identity changes when the policy does.
- Node/depth/byte bounds remain 64/16/16 MiB. Decoding preserves raw IEEE literal
  payloads and performs no numerical evaluation. The decoded v2 scalar still
  refuses legacy execution, model loading and every export route.

No evaluator-identity input, v1 format, Model JSON, registry, profile, old fixture,
default emitter option or old emission byte changes. The previously reviewed
numerical core and emitted helper bodies are byte-identical. No regeneration,
allowlist expansion, new dependency or default switch occurs.

## Tests and qualification

The first test compiled against untouched main and failed with a typed refusal
of required feature 10. Its source, failure and command are preserved in
`focused-red/`. The corrected suite adds ten nonignored tests: exact old payload,
policy declaration/identity/removal, all scalar IDs, deterministic raw-bit
round-trips, every unknown selector, feature mismatch/stripping, all truncations,
trailing bytes/arity, existing resource bounds and closed legacy routes.

Fresh baseline: PR 2060 passed/29 skipped; main 4140 passed/66 skipped;
Python 231 passed/2 deselected; historical golden 16/16; fitted C++ ASan/UBSan
18/18 and C++20 18/18; strict original O3 and additional C++17 O2 matrix passed.
After qualification must add exactly ten tests and preserve every old selected,
skipped, ignored and excluded name. Full command outcomes, audit and hosted
exact-head CI/review are recorded externally, not inferred from these targets.

Actual `just required` exits 101 on the owner-accepted 81-location/24-path
design floor. It is **not PASS**, and its downstream tests are not executed.
Separately run full standing suites and zero-new Clippy ratchets remain required.

## Remaining per-operation definition of done

Shortest-tree comparison, six-dataset utility, interpreter/compiler/full-recipe
twins, guarded runtime/fault semantics, arity and deterministic width, bounded
Experimental prior, equal-budget promotion gates, token/byte delta, lifecycle,
train-only fit, saturation, removal from generation without removing parsing,
and append-only identities remain governed by the complete card. This reader
adds bounded policy serialization and refusal proof only. Full model/fit-state
policy and artifact manifests, three-language train-predict recipes plus fitted
C++, profile generation and EEC creation/change/removal/execution, utility and
promotion remain held. No whole-card, next-card or release completion is claimed.
IP hygiene and no new feat-math dependency remain mandatory.
