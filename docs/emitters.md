# Emitters: Python, Rust, C++17, C++20, JavaScript, TypeScript

> **Status on `main` (2026-09-30):** `jope emit` (all five targets, `--lang cpp20` included), `jope pipeline run --emit …`, `jope models` and `jope datasets` are on `main`. There is no `jope learn`, `jope hwm` or `jope scores`; the learning store is the separate `jope-learning` binary. The generated capability matrix is machine-checked by the manifest and closure tests; the hand table below summarizes it. See the README's "What this SHA actually does".

`crates/feat-math` lowers a fitted program to standalone source in each target. The interpreter
(`model::fit/predict`, f64) is the reference; every emitted program must reproduce its predictions
bit for bit. Plan section: "Code emission"; decisions: [ADR-0002](adr/ADR-0002-feat-math-reconciliation.md).

## Language table

| Target | Entry point | Artefact | Standard / runtime | Coverage today |
|---|---|---|---|---|
| Python | `transpile::emit` → `.python` | `model.py`, dependency-free script (`python3 model.py train.csv test_features.csv predictions`) | CPython 3.11+ | full DSL (every planned op and head) |
| Rust | `transpile::emit` → `.rust` | `model.rs`, single file, no crates (`rustc -O model.rs`) | Rust 2021 | full DSL; **bit-identical pair with Python by construction** |
| C++17 | `transpile::emit_cpp17`, gate `supports_cpp17` | `model.cpp`, a library with a C ABI; `deploy/ci/emitters/driver.cpp` makes it a program | `-std=c++17`; also a valid C++20 program | 22 transforms, heads `ridge`, `elasticnet` |
| C++20 | same source as C++17 | same `model.cpp` compiled with `-std=c++20` (`cxx::CxxStd::Cxx20`) | `-std=c++20` | identical to C++17: C++20 is a **compile target**, not a second emitter |
| TypeScript | `transpile::emit_typescript`, gate `supports_typescript` | `model.ts` + `model.kernel.rs`, compiled to JavaScript + `model.wasm` | strict TypeScript ES2020, Rust wasm32 | full Rust recipe coverage; fitted coverage follows bound Rust; [build and arithmetic contract](typescript.md) |
| JavaScript | `transpile::emit_javascript`, gate `supports_javascript` | `model.js`, ES2020/CommonJS functions; `deploy/ci/emitters/driver.js` makes it a program (`node program.js …`) | Node ≥ 20 (CI uses 26) | same 22 transforms and two heads as C++ |

The `emit` example (`cargo run -p feat-math --example emit -- [--all] <program.dsl> <out-dir>`)
writes `model.py`/`model.rs` always, `model.cpp`/`model.js` when their gates pass, a sidecar
`manifest.json` (sha256 and token counts of the sources, math-kernel policy) and `targets.txt`.
Without `--all` an uncovered target is reported and recorded, never silently dropped; with `--all`
it is an error naming the unsupported elements. `jope emit --lang {python,rust,cpp17,cpp20,js,all}`
is a smaller wrapper over the same `transpile` entry points: it sets the title `jope emit` (so its
Python and Rust bytes differ from the example's), writes no `manifest.json` or `targets.txt`, and
its `all` is the four sources. Its `cpp20` output is the `model.cpp` that `jope pipeline run`
writes, taken from the same function (`jope_pipeline::emit::cpp_source`).

## Per-target exclusion table

See [the generated capability matrix](internal/kernel2/capabilities.generated.md) for the checked rows and route support.

The C++17 and JavaScript emitters are independently certified and narrower than the Python/Rust
pair. What they do not cover today (`emit_cpp::op_cpp`/`head_supported`,
`emit_js::op_supported`/`head_supported`):

| Element | C++17 / C++20 | JavaScript | Python / Rust |
|---|---|---|---|
| Transforms `standardize robust_scale rank blom quantile_bins bin_onehot hinge winsor_clip cheb poly_interact rff_cos rff_sin rand_proj landmark_dist oblique_hinge row_stats pair_prod target_bin_encode sigmoid step norm_row row_diff_stats` (22) | yes | yes | yes |
| Transforms `polar lag roll fft_mag row_spectral svd_project nystrom map_expr derive_expr symlog select_corr impute pair_ratio spline` (14) | **no** | **no** | yes |
| Target transforms (`Plan::Pipeline { target: Some(_) }`) and symbolic formula programs | **no** | **no** | yes |
| Heads `ridge`, `elasticnet` | yes | yes | yes |
| Heads `ridge_gcv`, **`gbm` (the DSL's `histgbm`)**, `histgbm2`, `blend`, `krr`, `gpr`, `gpr_kernel`, `ksmooth`, `stack`, `portfolio` | **no** | **no** | yes |

Consequence you will see in CI: the champion `8944cbdf` (`… (histgbm depth=5 lr=0.15 rounds=150)`)
emits to Python and Rust only — `deploy/ci/emitters.sh` records `targets.txt = py rs` and pins that
too — while `portable-4lang` (the same transform stack over `ridge`) emits to all four sources and
runs as seven programs. The vocabulary test `crates/feat-math/tests/registry_vocabulary.rs` asserts
every op in `featmath-jain-v1.json` is executable in the registry. The generated matrix and its
manifest and closure tests provide the checked per-route rows; the hand table above summarizes them.

Adding a target or an op to a target: [runbooks/add-an-emitter-target.md](runbooks/add-an-emitter-target.md).

## Compiler environment contract (`crates/feat-math/src/cxx.rs`)

Every C++ compiler invocation in the crate goes through one helper (plan blocker B28), so one
environment contract applies to the fitted model images (`advanced_v3::fitted_cpp_runtime`), the
emitted-C++ gates in `transpile.rs`, the integration tests and, mirrored by hand, `build.rs`:

| Variable | Meaning | Default |
|---|---|---|
| `CXX` | the compiler binary | `g++` |
| `JOPE_CPP_STD` | default language standard, `17` or `20` (`c++17`/`c++20` accepted); sites that pin a standard (the parity matrix) ignore it; any other value is an error, never a silent fallback | `17` |
| `JOPE_CXX_MARCH` | an explicit `-march=` value. Emitted programs get **no** `-march` unless it is set; the fitted runtime and model images default to the portable baseline | unset (emitted) / `x86-64-v2` (fitted, `cxx::FITTED_DEFAULT_MARCH`) |

Fixed flags on every fitted build: `-O3 -ffp-contract=off -fno-fast-math` (`cxx::fitted_flags()`);
the matrix script adds the same to emitted programs. Those two flags are what make the output
deterministic — the ISA only changes speed. The fitted codec identity hashes the joined flag string
(`runtime_flags_sha256`), so a codec encoded under one environment decodes only under the same one;
historical node codecs were built with `-std=c++17 -O3 -march=znver2 -ffp-contract=off -fno-fast-math`
and `JOPE_CXX_MARCH=znver2` reproduces that identity ([ADR-0002](adr/ADR-0002-feat-math-reconciliation.md) §5).

## Parity proofs

| Proof | Where | What it asserts |
|---|---|---|
| Python ≡ Rust | `tests/golden.rs` (67 programs), `transpile.rs` unit tests | the emitted pair and the interpreter agree bit for bit; sources are byte-deterministic |
| Four-language portable parity + C++20 | `tests/portable_parity.rs` over `tests/portable/full_ref.{py,rs,cpp,js}` | Rust, C++17, **C++20**, Python and JavaScript backends produce equal bits; under `JOPE_PARITY_STRICT=1` (CI) a missing tool is a failure, never a skip |
| Emitted C++ gates | `transpile.rs` tests (g++, incl. ubsan/asan variants), `tests/champion_parity.rs`, `tests/kernel_emission_safety.rs` | compiled emitted C++ matches the Rust reference; the shipped-text lint (`lint_shipped_text`, asserted over every backend and the manifest by `tests/ip_scrub.rs`) rejects any artefact that reveals how the program was found, or mentions paths or URLs |
| Fitted runtime | `tests/advanced_v3_fitted_cpp.rs`, `tests/kernel_codec.rs` | the embedded `.so` and model images reproduce the interpreter; codec identity round-trips |
| The matrix | `deploy/ci/emitters.sh` | for `8944cbdf` and `portable-4lang` on `datasets/test/pmlb-229-pwlinear`: python, rustc, g++-17, g++-20, clang++-17, clang++-20, node — every `pred-*.txt` has the same IEEE-754 bit pattern per row and equals the committed `<fixture>.predictions.sha256`; every source equals `<fixture>.<target>.sha256`; `targets.txt` is pinned. `UPDATE=1` rewrites the pins (review the diff); `JOPE_CXX_MARCH` adds `-march` to every C++ build — verified identical under `znver2` |
| Cross-`-march` / cross-compiler | nightly `parity-matrix` (WP8): `JOPE_CXX_MARCH ∈ {x86-64-v3, znver2, native}` × g++/clang++ × c++17/20 | all equal the committed sha |

Measured on 2026-09-18 at `4235db6` (128-core host): `cargo test -p feat-math --locked` 13 m 01 s
including compilation, `bash deploy/ci/emitters.sh` 20 s, the `emit` example 0.2 s. Not
re-measured since.

## Why the numbers stay equal

- No fused multiply-add anywhere: `-ffp-contract=off` on every C++ build; the Rust and Python
  sources use only `+ - * /`, `sqrt`, `exp`, `log`, `tanh` … in the same order the interpreter
  evaluates them; the portable math kernel (`portable_math.rs`, `MathKernel::Portable`) replaces
  libm functions whose results differ between platforms with the same polynomial in every target.
- Reductions are sequential and in the same order in every target (no SIMD reassociation:
  `-fno-fast-math`).
- The RNG-bearing transforms (`rff_*`, `rand_proj`, `landmark_dist`) carry their seeds in the
  program and use the same integer generator in every target.
- Predictions are written with 17 significant digits, which round-trips a double exactly, and
  the matrix compares the resulting bit patterns, not decimal strings.
