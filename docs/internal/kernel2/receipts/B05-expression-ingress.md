# B05 versioned expression ingress

This is the next serial B05 prerequisite after #167, from origin/main
772a797955a6f54394dad6d4b23e3bebaed9a8a7, in the sole bt-implementation lane.
It adds a separate bounded scalar AST and version-2 binary/lossless-text reader
for Func19 (probit) and Func20 (logit), without changing the legacy AST or loader.
The envelope uses variant1, minimum reader2 and required feature mask2. Existing
reservation variant0/feature1 remains unchanged. Legacy node0–5, Func0–18 and
Var0–7 encodings retain their exact prefix, including nonfinite IEEE payloads,
signed zero and column indices. Admission validates arity and limits of 64 nodes,
16 levels and 16MiB before recursive allocation. Unknown versions, variants,
features and scalar tags refuse explicitly; malformed/truncated input never
falls back to the legacy loader.

Acceptance: `cargo nextest run --locked -P pr -p feat-math --test inverse_links_ingress_v2`.
Eight nonignored regressions cover successful nested round-trip and refusal at
all three binary/text ingress paths. Tests-first missing-API failure is retained
outside the checkout. Existing kernel_codec and versioned_conversion tests also
qualify the unchanged v1 and reservation-only routes.

Owner numerical decisions remain frozen: raw Acklam probit relative error
<=1.2e-9 without erfc refinement; certified stable central logit within1ULP on the
independent high-precision corpus. These are the documented B05 exceptions to
the plan2ULP target, not new choices. The old Blom/clamped/literal helpers remain
byte-identical. This slice does not change kernels or their oracle fixture.

## Remaining B05 gates

Parsing is not execution admission. No scalar-v2 program can become a legacy
genome/model or enter existing full Rust/Python/C++17/fitted/JavaScript export.
Shipping profiles still exclude the new functions. Interpreter/compiler and
model integration, full train/predict recipes, fitted export, search-profile
admission, EEC create/change/remove witnesses, six-dataset/five-seed utility and
promotion receipts remain held. No whole-card or release completion is claimed.
The owner's 2026-10-03T10:01Z order selects B06 as the next serial card after this
ingress slice merges; it does not waive these remaining qualification gates.

## Evidence and no regression

Durable live receipt, command/exit/time logs, inventories, frozen-file pins,
required-gate debt, audit and exact-head CI/review readbacks:
`/home/ubuntu/jope-evidence/kernel2/bt/B05_V2_EXPRESSION_INGRESS/receipt.md`.
Fresh untouched-main baseline: PR2029passed/29skipped; main4109passed/66skipped;
Python231passed/2deselected; golden16/16; fitted ASan/UBSan18/18 and C++2018/18.
Expected after counts are baseline plus eight tests, with identical old named
selection/skip/ignored/excluded sets. The original strict emitter O3 matrix and
additional g++17O2 pass are retained. All93 protected inputs,38 evaluator identity
inputs and24 owner-held design paths remain byte-identical; no identity rebaseline.

`just required` has the owner-accepted held Clippy floor (2026-10-03T03:31Z),
not a PASS. The historical ledger contains81 unique locations in24 unchanged
paths; current normalized workspace diagnostics are a separately measured set.
Exact-head required exit and zero-new classification must be recorded, alongside
separately passing full standing suites. No suppression or weakened gate.

Merge requires local after-gates/audit, exact-head automatic plus full manual CI,
a fresh independent neverhumanbot native approval and final readback. No READY
objection window per owner directive. Hands-off137/147/148/B02/cleanup_claude/
MASTER_CLEAN_UP/xbabe0/foreign containers and paused pools remain untouched.

## Audit naming correction

Initial d108fc8f2 passed all standing suites but added one hard audit filename
classification and one soft use-of-legacy classification for live v1 code.
Owner10:36Z/10:42Z authorizes purpose-based scalar_program / V1 naming. The
reader now lives in kernel/scalar_program.rs, with V1(Func) and from_v1 names
for its active v1 adapters. This changes no envelope, ID, arity, budget or byte
contract. No suppression, audit rule or allowlist change. Original results
remain immutable; full corrected-head qualification in correction-01 is required.
