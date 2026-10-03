# B06 versioned expression ingress

Owner-selected next slice after B05 ingress PR168: read Func21 (expm1) and
Func22 (log1mexp) in the existing bounded scalar program. Both require one
argument. This is a necessary B06 prerequisite; it does not admit execution,
models, emitted recipes or shipping generation.

The v2 scalar envelope retains variant1 and minimum reader2. Existing Func0–20
bodies keep required feature mask2 and exact bytes. A body containing Func21/22
adds feature4, producing mask6. The writer derives features from all branches;
the reader requires the canonical matching set. Unknown sets/IDs/versions
return UnsupportedVersion; a known set inconsistent with the body is Malformed.
The preceding B05 reader refuses mask6 before decoding its body. Reserved v2
variant0/feature1 and all v1 routes remain unchanged. Limits remain 64 nodes,
16 levels and 16MiB, with arity checked before allocation and budgets before
recursion. No evaluator or probability/argument normalization runs at ingress;
IEEE literals retain their original bits, including signed zero and NaN payloads.

Acceptance:

```sh
cargo nextest run --locked -P pr -p feat-math --test stable_exponential_ingress_v2 --test inverse_links_ingress_v2
cargo clippy --locked -p feat-math --all-targets -- -D warnings
just required
```

Eight nonignored regressions cover append-only IDs/arity, nested binary/text
round-trip, exact old envelopes, derived features and mismatch refusal,
truncation/count/trailing-byte refusal, exact resource boundaries, unassigned
IDs/versions/features, and model/export/search refusal. The two existing unknown
Func ranges now start at23; explicit new21/22 tests replace those reservations
without removing any existing test. Tests-first E0599 missing-API proof and all
command/environment/time/exit/hash receipts are outside the checkout:
`/home/ubuntu/jope-evidence/kernel2/bt/B06_V2_EXPRESSION_INGRESS/receipt.md`.

Fresh base aa9e575476d339ed6a9b4112e36002687e56eb6d: PR2037/29 skipped,
main4117/66 skipped, Python231/2 deselected, golden16/16, fitted ASan/UBSan18/18
and C++2018/18. After counts must add exactly eight nonignored tests with no
change to existing selected/skipped/ignored/excluded names. Strict emitters
retain both original O3 and additional g++17 O2 passes. All93 protected inputs,
38 evaluator identity inputs and24 held design paths remain byte-identical;
no identity regeneration or dependency change.

Actual unchanged-main `just required` exits101 on the owner-accepted historical
81-location floor (2026-10-03T03:31Z). This run exposes71 locations; early
compiler termination can hide lib-test diagnostics, so this is no improvement
or PASS claim. All24 held paths are unchanged. Exact candidate exit, normalized
zero-new Clippy and separate full standing suites remain mandatory. No gate
weakening, lint suppression or held-path redesign.

## Per-op definition of done: remaining requirements

This ingress slice qualifies unary arity, bounded exact serialization,
append-only IDs, old-reader refusal and removal from generation while retaining
parsing. Both new functions remain nongeneratable. It introduces no numerical
exception: the B06 portable expm1, explicit -ln2 log1mexp branch, nonfinite,
domain and saturation contracts from the brief remain the required kernels.

Still held: shortest existing-tree comparison; at least six-dataset utility;
runtime/interpreted/compiled and emitted twins; guarded kernel and defined
NaN/Inf/zero/fault behavior; bounded Experimental prior; equal-budget five-seed
G-A…G-E receipt before Baseline; token/byte accounting; capability lifecycle;
train-only fit; deterministic width; saturation faults; append-only numerical
helpers; Rust/Python/C++17 train-and-predict EEC plus fitted C++; IP scrub of those
helpers; negative input/empty-row execution; independent numerical oracle and
three compiled-recipe output-bit/fault parity. No whole-B06 or release completion
is claimed. No new feat-math dependency is introduced.

Merge requires local after-gates and lane audit, exact-head automatic and full
manual CI, fresh independent neverhumanbot native approval and final readback.
No objection window under the owner directive. PR137/147/148, B02, cleanup_claude,
MASTER_CLEAN_UP, other checkouts, xbabe0 and paused/foreign pools remain untouched.
