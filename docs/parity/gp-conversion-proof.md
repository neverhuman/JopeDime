# GP conversion proof inventory

This inventory maps the current GP tree conversion checks to their actual output routes. It does
not claim universal parity or admission for every genome form.

| Route | Executed proof | Boundary |
| --- | --- | --- |
| Paired Rust and Python recipes | [`parity_bits.rs`](../../crates/feat-math/tests/parity_bits.rs) compiles or runs each emitted source and compares prediction bits with the Rust evaluator. The emitted sidecar program is parsed back and checked against the executable tree. | The paired recipe planner must admit the program. |
| Fitted C++17 inference | The same test fits and emits C++17, compiles with `g++ -std=c++17 -O2 -ffp-contract=off -fno-fast-math`, runs a bit-input driver, and compares with the evaluator. | This is a fitted inference artifact, not the standalone C++17 training recipe. |
| Standalone C++17 training recipe | [`conversion_wire_matrix.rs`](../../crates/feat-math/tests/conversion_wire_matrix.rs) compiles and runs every admitted direct-wire witness with `g++ -std=c++17 -O2 -ffp-contract=off -fno-fast-math` and compares prediction bits with the frozen evaluator. | Six of the 19 direct-wire programs are admitted; the other 13 keep their frozen refusal summaries. This is a training recipe, separate from fitted C++17 inference. |
| Standalone C++17 numeric ABI boundary | [`conversion_boundary.rs`](../../crates/feat-math/tests/conversion_boundary.rs) compiles a generated standardize/ridge recipe with the same strict flags. Three bounded seeds exercise NaN payload, positive/negative infinity, negative zero and subnormal feature values through the direct numeric ABI; all 24 prediction bits match the Rust evaluator. The driver also requires structured return-code refusal for nonfinite targets and malformed prediction width/length. | This proves one admitted training recipe through its direct C ABI. The B03 CSV command rejects nonfinite input and has a separate contract. |
| Registry and genome-form matrix | The same test ties current registry IDs 0–52 and all 35 search-generatable descriptors to immutable fitted, alias or quarantine records. It executes paired recipes for all 19 distinct direct-wire programs and for symbolic raw, symbolic affine and compact portfolio examples. | The seven quarantined IDs and four aliases are classified by the historical golden; explicit nested portfolio emission retains a pinned refusal. Classification alone is not execution of every parameter variant. |

The 19-program scalar corpus in `parity_bits.rs` reaches every current named `MathExpr` function
(codec-local IDs 0–18), row variable (including `x` and indexed columns), binary operator (IDs
0–4), and negation across symbolic, `map`, and `derive` programs. Before any fit or emission, each
program passes parse → canonical render → parse and typed-envelope round-trips with identical
executable codec bytes. The three runtime lanes then run the round-tripped tree on deterministic
ordinary and boundary rows. The shared boundary corpus is byte-pinned; the existing
`parity_allowlist.v1.json` records 22 known X6 divergences and fails if any listed divergence stops
reproducing or a new one appears. Passing the test therefore means no divergence beyond that
explicit inventory, not bit equality on all rows.

The seven-form matrix in `conversion_roundtrip.rs` covers plain, symbolic raw/affine, `map`/`derive`, target-stage, compact
portfolio, and explicit nested-portfolio text. It proves a canonical fixed point and identical
typed codec bytes for each. Its route assertions retain the public standalone C++17 refusals for
all but the plain pipeline and the paired Rust/Python recipe refusal for explicit nested
portfolio. The typed-literal test separately preserves negative-zero, subnormal, infinity and NaN
payload bits through the lossless program envelope. Nonfinite literals are refused by the readable
expression renderer; a direct negative-zero constant's typed node identity belongs to the typed
codec rather than the infix text parser.

The wire matrix uses the frozen 80-row, five-column toy input (first 64 rows train, last 16
predict) and requires bit equality for the 19 direct-wire programs in the evaluator and emitted
Rust/Python routes. Every generatable wire ID has a direct fitted witness in that set. The
standalone C++17 route executes its six admitted programs on the same input. It pins C++17
admission and refusal summaries for all 19 programs. Symbolic raw/affine and compact portfolio
examples also execute emitted Rust/Python against the evaluator; explicit nested portfolio
retains its exact paired and C++17 refusal. `map`/`derive` and target forms occur among the 19
executed direct-wire programs, with C++17 refusal where declared.

## Opt-in C++17 expression training recipes

`feat_math::kernel::recipe_v2::emit_cpp17` and `jope emit --lang cpp17-recipe-v2`
emit profile `feat-math.cpp17-expression-recipe.v2`. The generated numeric C ABI fits its own
training state for symbolic raw/affine programs and `map`/`derive` stages with Ridge/ElasticNet.
It reuses the existing planner, scalar lowering, helper dependency closure and C++ assembler.
Its profile identifies added backend admission; its scalar arithmetic remains the historical
policy. The default `emit_cpp17`, CLI `cpp17` and `all` retain their old bytes and refusals.
Symbolic IR must have matching readable source and tree; nonfinite literals and mismatched
source/tree pairs are refused before the historical planner can reparse them.

[`mathexpr_boundary_parity.rs`](../../crates/feat-math/tests/mathexpr_boundary_parity.rs) executes
nine scalar expression witnesses in four contexts (raw, affine, map, derive), covering Func
0–18, all variable forms, binary 0–4 and negation. Each executable tree passes canonical and
typed codec round-trips before fitting. Runtime input arrives after compilation, and predictions
must match evaluator bits with g++ C++17 O2, contraction disabled and fast math disabled.
Additional witnesses exercise whole-expression invalid-clip fallback through `min`/`sign`, missing
columns, protected division, append and multiple expression stages, and ElasticNet. Three recipes
run the O3, clang, C++20 and ASan/UBSan matrix. The direct ABI and generated Model methods refuse
zero-width/ragged tables; the ABI also refuses wrong prediction width/length and nonfinite targets.
The CLI test binds written `model.cpp` bytes to this public API and proves explicit opt-in.

The numeric ABI requires rectangular rows with positive training width, finite targets, and the
same prediction width. Nonfinite features are sanitized to zero, as in the default training recipe;
raw symbolic evaluator calls instead evaluate their raw features. The scalar bit witnesses use
finite features. Expression context/column/domain failures retain the historical zero fallback;
this profile does not introduce a fault stream or a new numerical policy. It inherits the scalar
helpers' existing X6 limitations; the unchanged 22-entry allowlist belongs to the existing parity
corpus, and is neither enlarged nor a waiver for these new execution witnesses.

Remaining work includes the 22 existing X6 parity differences, broader nonfinite and property-based
execution across emitted routes, parameter variants beyond the witnesses, and training recipe
support for target and portfolio forms. Ordered input/state metadata and `jope pipeline run`
integration remain separate B03 work. This numeric recipe has no ordered-state hook and is not a
full B03/1.0 qualification. Frozen evaluator-identity inputs and historical fixtures are unchanged.
