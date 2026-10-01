# GP conversion proof inventory

This inventory maps the current GP tree conversion checks to their actual output routes. It does
not claim universal parity or admission for every genome form.

| Route | Executed proof | Boundary |
| --- | --- | --- |
| Paired Rust and Python recipes | [`parity_bits.rs`](../../crates/feat-math/tests/parity_bits.rs) compiles or runs each emitted source and compares prediction bits with the Rust evaluator. The emitted sidecar program is parsed back and checked against the executable tree. | The paired recipe planner must admit the program. |
| Fitted C++17 inference | The same test fits and emits C++17, compiles with `g++ -std=c++17 -O2 -ffp-contract=off -fno-fast-math`, runs a bit-input driver, and compares with the evaluator. | This is a fitted inference artifact, not the standalone C++17 training recipe. |
| Standalone C++17 training recipe | [`conversion_roundtrip.rs`](../../crates/feat-math/tests/conversion_roundtrip.rs) checks public admission and refusal after canonical text round-trip. | Only the plain pipeline in its seven-form matrix is admitted; this test does not execute that recipe. |

The 19-program scalar corpus in `parity_bits.rs` reaches every current named `MathExpr` function
(codec-local IDs 0–18), row variable (including `x` and indexed columns), binary operator (IDs
0–4), and negation across symbolic, `map`, and `derive` programs. Before any fit or emission, each
program passes parse → canonical render → parse and typed-envelope round-trips with identical
executable codec bytes. The three runtime lanes then run the round-tripped tree on deterministic
ordinary and boundary rows. The shared boundary corpus is byte-pinned; the existing
`parity_allowlist.v1.json` records 22 known X6 divergences and fails if any listed divergence stops
reproducing or a new one appears. Passing the test therefore means no divergence beyond that
explicit inventory, not bit equality on all rows.

The seven-form matrix covers plain, symbolic raw/affine, `map`/`derive`, target-stage, compact
portfolio, and explicit nested-portfolio text. It proves a canonical fixed point and identical
typed codec bytes for each. Its route assertions retain the public standalone C++17 refusals for
all but the plain pipeline and the paired Rust/Python recipe refusal for explicit nested
portfolio. The typed-literal test separately preserves negative-zero, subnormal, infinity and NaN
payload bits through the lossless program envelope. Nonfinite literals are refused by the readable
expression renderer; a direct negative-zero constant's typed node identity belongs to the typed
codec rather than the infix text parser.

Remaining work includes the 22 existing parity differences, executable proof for every
generatable wire operator, and broader form-specific execution and refusal cases. The standalone
C++17 training recipe still excludes symbolic expressions, `map`, `derive`, target stages and
portfolio forms; its admission check is not evidence that fitted inference covers those training
routes. The next wire/genome-form matrix adds coverage without changing frozen evaluator-identity
inputs or historical fixtures.
