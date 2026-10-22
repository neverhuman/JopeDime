# Native C17 model API

`jope emit --lang c17` (alias `c`) exports a complete training recipe behind a C17 API.
Its arithmetic is the same generated safe Rust implementation used by the Rust target.
**Rust is required to build the library. This is a C17 binding, not an independent C-only
numerical backend.** The compiled library needs neither Python nor JopeDime at deployment.

The target accepts the same numeric training recipes, transforms, symbolic expressions,
target transforms and heads as the Python/Rust recipe pair. It passes all explicit emission
options through to that pair. The default math policy remains platform libm; portable math
is refused wherever the existing pair refuses it. This does not claim parity across unrelated
architectures or libm implementations. Schema-bound fitted exports, the numeric-v2 fitted
format and packaged appliance models are separate APIs and are not C17 exports here.

## Export and build

```sh
jope emit --genome '(pipe X (standardize) (histgbm rounds=20))' --lang c17 --output exported --json
cd exported/c17
rustc --edition=2021 -O -C panic=unwind --crate-type=staticlib model_ffi.rs -o libmodel.a
gcc -std=c17 -O2 -Wall -Wextra -Werror -pedantic-errors -fno-fast-math -ffp-contract=off \
  caller.c libmodel.a -ldl -lpthread -lm -o caller
```

Linux GCC and Clang callers are exercised by the native contracts. `clang -std=c17` may
replace GCC. The header also supports C++ callers with `extern "C"` linkage. Other platforms
and cross compilation require their native Rust target and platform linker libraries and
are not qualified by these Linux tests. Use one generated model library per executable;
the exported symbol names are intentionally fixed. Separate shared libraries can be loaded
through separate library handles.

The bundle has four members:

| File | Purpose |
| --- | --- |
| `c17/model.h` | C17 declarations, status codes and buffer ownership contract |
| `c17/model_core.rs` | Full emitted Rust implementation plus safe typed facade |
| `c17/model_ffi.rs` | Checked native boundary and opaque handle management |
| `c17/manifest.json` | Recipe, options, exact source hashes and build recipe |

The target-complete route `--lang v3:c17` additionally binds all four members into the
outer `artifact-manifest-v3`. Pin that outer manifest independently when verifying artifacts.
`jope pipeline run --emit c17` and the public `transpile::emit_c17` API use the same emitter.
The emission example accepts `--c17`. Existing `all` sets remain their historical recipe sets;
request C17 explicitly. Manifests describe build inputs; emission alone does not execute them.

## Caller contract

See [the complete example](../examples/c17/README.md). `jope_model_fit` consumes a row-major
`double` matrix and labels and returns a nonzero `uint64_t` handle. `jope_model_predict`
fills the caller's output buffer; `jope_model_score` returns the recipe's R-squared score.
Release each successful handle with `jope_model_free`. Models retain their fitted state across
prediction calls. There is no fitted-state serialization API in this version.

Lengths count elements, not bytes. Training and scoring require positive rows and width;
prediction accepts zero rows with null buffers and zero lengths. Prediction width must match
training width. Supplied lengths, alignment and multiplication overflow are checked before
copying. C callers must still provide valid live allocations of the declared length: arbitrary
non-null addresses cannot be validated by a C ABI. Inputs are copied before output changes,
so input/output aliasing is supported. Floating-point bits are passed without CSV cleaning;
individual recipe operations retain their usual numerical semantics.

| Status | Meaning |
| --- | --- |
| `JOPE_MODEL_OK` (0) | Success |
| `JOPE_MODEL_INVALID_ARGUMENT` (1) | Shape, width, length, null or alignment error |
| `JOPE_MODEL_INVALID_HANDLE` (2) | Unknown, freed or zero handle |
| `JOPE_MODEL_PANIC` (3) | Recoverable unwinding numerical panic |
| `JOPE_MODEL_INTERNAL_ERROR` (4) | Internal result or handle-table error |

On a returned error, output buffers and handles are unchanged. Compile with `panic=unwind`;
abort builds, allocation exhaustion and invalid caller memory cannot be recovered. Panics may
write diagnostics to stderr. Handles are owned by one loaded library and never reused during
its lifetime. Concurrent predictions retain a shared owner while releasing the handle-table
lock before arithmetic. Free prevents future lookups; an already acquired prediction can finish.
Do not unload a library while callers or model handles remain active.

## Verification

```sh
JOPE_PARITY_STRICT=1 cargo test --locked -p feat-math --test golden c17_export::
cargo test --locked -p feat-math --test golden
cargo test --locked -p jope-pipeline --test artifact_target_binding
bash examples/c17/run.sh
CC=clang bash examples/c17/run.sh
```

The five native contract tests cover deterministic source identity, all compilation-input
bindings, strict C17 under GCC/Clang, C++ linkage, shared-library loading, shape/overflow/
alignment refusal, handle lifetime, concurrent prediction, aliasing, explicit runtime options
and injected panic containment. The existing 83-case golden suite now executes a real C caller
and compares prediction bits against the interpreter alongside Python/Rust, including the
fast-runtime fixtures. A passing test listing alone is not execution evidence.
