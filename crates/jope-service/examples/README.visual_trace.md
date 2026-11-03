# Frozen fused inference render diagnostics

`visual-trace` is an opt-in observation feature. The default service graph has no
observer. It does not enable learned guidance, change bundle admission, train a
model, or supply a new checkpoint-loading route.

Build and run with the repository's supported Rust toolchain:

```sh
env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only \
  cargo run --locked -p jope-service --features visual-trace --example visual_trace -- \
  /external/signed-appliance /external/capture /external/engine-csvs
```

The first argument also accepts an existing authenticated r882 development
extraction. A signed appliance is opened and fully verified through `jope-assets`
and `load_r882_embedded`; a directory uses `load_r882_release`. Neither path accepts
unsigned replacement weights or test signing keys. A foreign executable's package
can be inspected without executing it.

The optional CSV directory contains `case-a.csv` and `case-b.csv`, each with exactly
96 finite numeric rows and headers `x0,x1,x2,x3,target` (or `y`). Its bytes are kept
unchanged. The production `random_split(96,76,20261103,"public-run/fit-v1")`
selects the 76 FIT rows. HOLDOUT rows never enter descriptor extraction. Without
this argument, the example generates two deterministic 96-row all-FIT diagnostic
tables. Neither mode is a benchmark or quality qualification.

The example calls the existing native descriptor recipe, `FusedForward::stages`
and `CpuAdapter::predict_costs`. Its 12 manually declared ridge programs form a
diagnostic slate; the first four fitness inputs are their measured FIT RMSE.
This is not the product's proposal/admission loop and does not prove that these
ranker calls caused a product run's exported winner.

`trace.json` is private evidence, not a public render package. It includes
checkpoint-derived values, native affine operands/outputs and ranker internals.
The output manifest is renamed into place only after both cases pass. Existing
captures are refused. `source` and `tokens_per_second` remain null because this
example performs neither source emission nor complete product execution.

Each affine record is row-major, with `weights[out,in]`, `bias[out]`,
`input[...,in]`, `output[...,out]` and a runtime sequence number. Every output is
checked by a separate f64 weighted sum against the observed native preactivation:

```text
abs(sum(x*w)+bias-output) / (sum(abs(x*w))+abs(bias)+1e-9) <= 1e-5
```

Native traced/untraced stage outputs and costs must match bit for bit. A corrupted
preactivation must fail closure. The feature records all existing affine helper
calls, plus three named readouts; it does not claim complete functional-operation,
attention-projection or full-forward coverage. Weights stay frozen across cases;
input-dependent activations and contributions change. Animated travel describes
dependency order with expanded presentation time, not physical signal transit.

The bounded thread-local capture accepts at most 128 records and 4,000,000 scalar
values. It applies only to synchronous operations on the calling thread. Nested
captures are refused; errors and unwinding clear incomplete capture state.

## Source emission measurement

After a separate product run's exact Rust source has executed and passed parity:

```sh
env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only \
  cargo run --locked -p jope-service --features visual-trace --example visual_emit_metric -- \
  /external/model-object-directory /external/verified-predictor.rs /external/emit-metric.json
```

The benchmark authenticates the saved `ModelObject`, regenerates Rust source and
requires byte equality with the separately verified source. It measures one outer
monotonic interval around repeated fresh source-emitter calls on that same model.
Allocation and loop overhead stay inside the interval. Model loading, warmup,
calibration, hashing, lexical counting and verification stay outside. Every timed
output is retained and compared afterward. Clock samples establish a minimum
reporting interval of at least 100 ms; a below-threshold batch is discarded and
repeated with more invocations within explicit repetition/memory bounds.

`viz.emit_batch.rust_tokens_per_s` uses `jope.source-lex/v1`. It is **warm repeated
emission of the same artifact**, not neural decoding, newly published unique code,
or an LLM tokenizer rate. The receipt states the build profile; the example's dev
build is a diagnostic local measurement, not release-performance qualification.

Focused checks:

```sh
env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only \
  cargo test --locked -p jope-service --features visual-trace --lib visual_trace::tests
```

Keep raw captures, model objects, downloaded packages and rendered media outside
the working tree. A public render requires a separate bounded projection of the
selected observations; never distribute this raw JSON as a scene asset.
