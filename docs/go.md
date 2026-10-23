# Go output

The explicit `go` target (alias `golang`) emits a Go library over the complete existing
Rust numerical implementation compiled to WebAssembly. Rust is required when building the
artifact. Deployed inference uses the pinned pure Go wazero runtime, with no CGO, Rust or
Python installation, WASI, native addon or host imports.

```sh
rustup target add wasm32-unknown-unknown
jope emit --genome '(pipe X (standardize) (ridge))' --lang go --output model
cd model
rustc --edition=2021 --target wasm32-unknown-unknown --crate-type cdylib \
  --crate-name model_kernel -C opt-level=3 -C panic=abort model.go.kernel.rs -o model.wasm
CGO_ENABLED=0 go build -mod=readonly ./...
```

Go 1.25+ is required. The generated module is `jope.generated/model`, package `model`.
`go.mod` and `go.sum` pin wazero v1.12.0 and its indirect x/sys v0.44.0 dependency.
Downloads happen during build, not prediction. Consumers can retain the module with a local
`replace` directive, or move the Go source into an application while retaining its dependencies.
Pass the compiled Wasm bytes to New; Go's embed package can include them in a single executable.
See the [runnable CLI-to-Go example](../examples/go/run.sh).

```go
m, err := model.New(ctx, wasmBytes)
if err != nil { return err }
defer m.Close(context.Background())
if err := m.Fit(ctx, trainingRows, targets); err != nil { return err }
predictions, err := m.Predict(ctx, testRows)
```

`NewWithEngine(ctx, bytes, model.Interpreter)` selects the portable interpreter; `Auto` uses
the host's default wazero engine. Each model owns an independent runtime. Calls serialize access,
and returned predictions are copies. Close is idempotent and safe alongside other methods.
Cancellation or a trap invalidates the instance. Caller validation errors preserve a usable model.
Do not copy a Model or change input slices concurrently with a call using them.

Fit requires nonempty rectangular rows and matching finite targets. Predict requires successful
training and the same width; empty batches are valid. Non-finite features retain the numerical
Rust implementation's handling. Score returns R-squared, zero for empty/constant targets, and
errors for non-finite targets or overflowed scores. An overflowed target mean is refused even
when predictions are perfect. Transfers are capped at 16,777,216 elements
or UTF-8 bytes. CSV encoding itself is bounded before copying oversized records. Each Wasm
instance has a 512 MiB linear-memory ceiling; allocation errors and cancellation are surfaced.

Fitted exports use `feat_math::go::emit_bound`, retaining the existing bound Rust emitter's
admitted models and refusals. `PredictStrings(ctx, rows)` consumes raw UTF-8 cells in saved
schema order and applies saved missing/category codecs, transforms, calibration and ordered
training tails. Fitted artifacts refuse Fit and numeric Predict. Recipe artifacts refuse string
input. The explicit `emit_scalar` and `emit_row_vector` APIs retain the existing scalar-v2
function/policy/carrier and row-vector recipes. Raw Legacy fitted wrappers and fitted portfolio
remain inherited Rust refusals. Numeric-v2 fitted images and model-object storage are separate
contracts; this target does not introduce a new model-state serialization format.

The numerical identity is `wasm-rust-std-v1`. Native-reference comparisons require finite
predictions within `1e-8 * (1 + abs(reference))`. This is not a universal bit-equality claim
across host libm and Wasm. Portable-math emission retains the existing explicit refusal.
Execution is sequential within one Wasm instance, including when recipe options request the
native fast-runtime profile. Historical default `all` target membership and numerical Rust/Python
source templates are unchanged.

CLI `--lang v3:go`, `transpile::emit_targets_v3` and bound v3 output bind the complete four-file
source closure: `model.go`, `model.go.kernel.rs`, `go.mod`, `go.sum`. Independent manifest pins
are required to detect substitutions. Compiler outputs are built separately. Pipeline `--emit go`
also records every source member. Use a separate package/directory for each generated model and language target.

Acceptance is `bash deploy/ci/go/qualify.sh`. It requires every registry case to execute under
both engines or match an explicit Rust refusal, every existing golden, fitted/publication
regressions and 100% raw Go facade statement coverage. Go has no built-in branch coverage metric;
the percentage excludes the inherited Rust kernel and runtime dependency. The registry census
and native oracle comparisons qualify their exercised behavior separately.

The required reusable CI matrix adds Go 1.25/1.26, race instrumentation, bounded lifecycle
fuzzing, six pure Go deployment cross-builds and native macOS/Windows census runs. A direct PR
trigger also runs on hosted workers independently of the repository's self-hosted queue.
Configured, pending or skipped checks are not passing execution evidence.

The dedicated golden target is `JOPE_GO_PARITY=1 cargo test --locked -p feat-math
--features go-parity-tests --test go_golden`. Build-time checked insertions add Go calls to the
unchanged native golden source; every original native assertion remains. The feature keeps this
expensive qualification separate from ordinary test discovery, and its runtime-presence test
refuses a missing opt-in flag. Native golden source and takeover snapshot hashes stay unchanged.
