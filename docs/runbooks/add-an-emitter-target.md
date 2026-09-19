# Runbook: add an emitter target (or an op to an existing target)

Read [docs/emitters.md](../emitters.md) first: the language table, the per-target exclusion table
and the parity proofs are the contract you are extending. Decision record:
[ADR-0002](../adr/ADR-0002-feat-math-reconciliation.md). Everything below lives in
`crates/feat-math`.

## Adding an op to the C++17 or JavaScript emitter

The Python/Rust pair covers the whole DSL; C++ and JS cover 22 transforms and the `ridge` /
`elasticnet` heads. To add, say, `symlog` to both:

1. Implement it in `emit_cpp.rs` (`op_cpp`, `op_struct_cpp`, `op_calls_cpp`, a `HELPER_*`/`OP_*`
   source constant) and in `emit_js.rs` (`op_supported`, the op source table), **in the same
   evaluation order and with the same arithmetic as `emit_py.rs`/`emit_rs.rs`** — no FMA, no
   reassociation, sequential reductions, the portable math kernel for any libm call that is not
   correctly rounded everywhere (`portable_math.rs`).
2. Update the coverage gates: `supports_cpp17` / `supports_javascript` derive from the same
   tables, so the gate changes with the table; update the exclusion table in
   [docs/emitters.md](../emitters.md) (and the machine-checked table once `jope-contracts` lands).
3. Prove parity three ways:
   - a genome using the op in `tests/golden.rs` style with a `transpile.rs` C++ gate test (compiled
     through `cxx::command`, compared bit for bit with the Rust reference);
   - extend `tests/portable/full_ref.{py,rs,cpp,js}` if the op belongs in the portable reference
     (all five backends of `tests/portable_parity.rs` must agree; run with `JOPE_PARITY_STRICT=1`);
   - a matrix line: a `.dsl` fixture under `tests/fixtures/emitted/`, a `matrix` call in
     `deploy/ci/emitters.sh`, `UPDATE=1 bash deploy/ci/emitters.sh` once, commit the `.sha256` pins.
4. Run the environment matrix locally before the PR:
   ```bash
   JOPE_PARITY_STRICT=1 cargo test -p feat-math --locked
   bash deploy/ci/emitters.sh
   CXX=clang++ bash deploy/ci/emitters.sh
   JOPE_CXX_MARCH=znver2 bash deploy/ci/emitters.sh     # or native
   JOPE_CPP_STD=20 cargo test -p feat-math --locked --test portable_parity
   ```
   The predictions' sha256 must not change across any of them (the pins are the same file).
5. `lint_vocabulary` must stay clean: no search vocabulary, paths or URLs in an emitted artefact.

## Adding a head to C++ / JS

Same steps; heads are `head_supported` / `head_src_cpp` / `head_calls_cpp` in `emit_cpp.rs` and
the head table in `emit_js.rs`. A gradient-boosted head (`gbm`/`histgbm`, `histgbm2`) needs the
exact split/leaf arithmetic of `boost2.rs` reproduced, including its tie-breaking and its
early-stopping accounting — write the Python/Rust emitted source side by side with the C++ and
diff the arithmetic by hand before running anything.

## Adding a new target language

1. `emit_<lang>.rs` with the same structure as `emit_js.rs` (plan → ops table → head table →
   `Result<String, Vec<String>>` naming unsupported elements); `transpile::emit_<lang>` +
   `supports_<lang>`; a `Lang` entry in `tokens.rs`; a `<lang>` target in
   `examples/emit.rs` and `targets.txt`.
2. Decide the runtime contract: the Python/Rust artefacts are programs
   (`prog train.csv test_features.csv predictions`); C++/JS are libraries with a driver under
   `deploy/ci/emitters/driver.<ext>`. Pick one and document it in the language table.
3. Parity: a `tests/portable/full_ref.<ext>` reference and a backend in `portable_parity.rs`
   (strict under `JOPE_PARITY_STRICT=1`; a missing toolchain must fail, not skip); the matrix in
   `emitters.sh` (add the compiler/interpreter to its required-tool list and to the hosted
   `emitters` lane's setup); pins for both fixture genomes.
4. Determinism checklist for the new runtime: IEEE-754 doubles only; no FMA (compiler flag or
   language guarantee); sequential reductions; the portable math kernel for `exp/log/pow/tanh` if
   the platform's libm is not correctly rounded; integer RNG reproduced exactly; predictions
   printed with 17 significant digits.
5. Docs: language table + exclusion table in [docs/emitters.md](../emitters.md); the
   `jope emit --lang` enum; a line in the README quickstart if the toolchain is commonly installed.
6. Review: a second person compiles and runs the matrix on a different machine and pastes the
   `predictions.bits` sha256 into the PR.
