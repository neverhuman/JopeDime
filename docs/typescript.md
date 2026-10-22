# TypeScript model export

The `typescript` (alias `ts`) target emits a strict TypeScript API plus a WebAssembly
build source. The numerical implementation is the existing complete Rust emitter:
all accepted recipe transforms, target mappings, symbolic programs and regression
heads follow that emitter. The narrower JavaScript emitter does not determine this
target's coverage. Fitted export follows the existing bound Rust emitter's admitted
model forms, including schema encoding, missing cells, calibration and ordered rows.

This is a TypeScript/WebAssembly bundle. It is not a pure TypeScript numerical port.
Compile once with Rust and TypeScript; deploy JavaScript and `model.wasm` to Node or
a WebAssembly-capable browser. The deployed model needs no Rust installation, native
addon, Python process, npm runtime dependency, filesystem access or host imports.

## Emit and build

```sh
export CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only
cargo run --locked -p jope-cli --no-default-features --bin jope -- emit \
  --genome '(pipe X (winsor_clip q=0.1) (histgbm rounds=60 depth=3 lr=0.1 bins=32))' \
  --lang typescript --output target/ts-model --json

rustup target add wasm32-unknown-unknown
cd target/ts-model
rustc --edition=2021 --target wasm32-unknown-unknown --crate-type cdylib \
  --crate-name model_kernel -C opt-level=3 -C panic=abort model.kernel.rs -o model.wasm

# Use your installed TypeScript compiler. Qualification pins the repository's
# existing compiler version, 7.0.2.
tsc --strict --target ES2020 --module commonjs --lib ES2020,DOM --declaration model.ts
# For an ES module application, select --module ES2020 instead.
```

For a manifest binding the complete source bundle, select
`--lang v3:typescript --output <fresh-directory>`. It binds both `model.ts` and
`model.kernel.rs`, the selected program, options and declared build identity. The
source manifest does not certify the compiled binary: retain the compiler versions,
build command, binary digest and execution qualification alongside deployment records.

The pipeline accepts `--emit typescript` or `--emit rust,python,ts`.
The emitter example accepts `--typescript` / `--ts`. Historical `all` aliases
and the frozen default emitter sets retain their previous membership; request
TypeScript explicitly.

## Node

```ts
import { readFileSync } from "node:fs";
import { createModel } from "./model.js";

const model = await createModel(readFileSync("./model.wasm"));
model.fit([[0, 1], [1, 2], [2, 3], [3, 4]], [1, 3, 5, 7]);
const predictions: Float64Array = model.predict([[1.5, 2.5], [2.5, 3.5]]);
console.log(Array.from(predictions));
console.log(model.score([[0, 1], [1, 2]], [1, 3]));
model.dispose();
```

`createModel` accepts a `Uint8Array`, `ArrayBuffer`, or precompiled
`WebAssembly.Module`. Each call creates an independent model and linear memory.
The asynchronous work is module compilation/instantiation; fitting and prediction
are synchronous. Use a Web Worker or worker thread for long fits.

## Browser

```ts
import { createModel } from "./model.js";

const response = await fetch("./model.wasm");
if (!response.ok) throw new Error("model download failed");
const model = await createModel(await response.arrayBuffer());
model.fit([[0], [1], [2], [3]], [0, 2, 4, 6]);
console.log(model.predict([[1.5]])[0]);
model.dispose();
```

Deploy the compiled ES module and the WebAssembly binary together. Apply the
application's normal WebAssembly content-security policy. This target does not use
`eval`, `Function`, WASI, host imports or dynamic native loading.

## Rust API and fitted prediction

```rust,ignore
let program = feat_math::GenomeIr::parse("(pipe X (histgbm))")?;
let recipe = feat_math::transpile::emit_typescript(
    &program, &feat_math::transpile::EmitOptions::default(),
)?;
// recipe.source -> model.ts; recipe.kernel -> model.kernel.rs

let fitted = feat_math::typescript::emit_bound(
    &bound_model, &feat_math::transpile::EmitOptions::default(),
)?;
// Or request artifact_manifest::Target::TypeScript from emit_bound_targets_v3.
```

A fitted bundle already carries its trained state. Compile it with the same
commands, then use `model.predictStrings(rows)`, where each row contains raw
string cells in schema order, or `model.predictCsv(csv)` with the exact schema
header. CSV quoting, embedded commas/newlines, Unicode, categorical strings,
missing cells, ordered transforms and fitted tails follow the bound Rust source.
A fitted bundle rejects refitting. Unsupported fitted forms fail before publication,
as in the existing bound Rust/Python API.

## Runtime contract

| Operation | Behavior |
| --- | --- |
| `fit(rows, targets)` | Recipe only; rectangular numeric rows, nonempty width, exactly one finite target per row; returns the same model for chaining |
| `predict(rows)` | Recipe only; requires a fit, exact fitted width, permits an empty batch |
| `predictStrings(rows)` | Fitted only; raw string cells in schema order |
| `predictCsv(csv)` | Fitted only; exact header and existing Rust CSV/encoder semantics |
| `score(rows, targets)` | Recipe R², sequential reductions and the Rust constant-target convention |
| `dispose()` | Idempotent; subsequent calls refuse; returned predictions remain valid |
| Result ownership | A copied `Float64Array`; refitting, prediction, memory growth and disposal cannot detach or overwrite a returned result |
| Bounds | At most 16,777,216 transferred numeric elements or UTF-8 input bytes per call; model fitting can require additional memory |
| Failure | Shape errors refuse before fitting; a WebAssembly trap invalidates the instance |
| Parallelism | One synchronous instance; `fast_runtime` selects the sequential Rust numerical body for this target |

The bridge accepts lengths and dimensions, never caller-supplied memory addresses.
Its buffers and fitted model are private to each WebAssembly instance. The host
copies into allocator-produced addresses; no Rust unsafe block is introduced.
A module declaring host imports is refused.

The arithmetic identity is `wasm-rust-std-v1`. Full language coverage does not imply
cross-platform bit equality: Rust's WebAssembly transcendental implementations can
differ from native libm. The qualification tolerance is
`abs(actual-reference) <= 1e-8 * (1 + abs(reference))`; every prediction must also
be finite. Portable-math requests retain the existing fail-closed behavior.
No GPU execution, shared-memory threading, performance improvement or production
deployment is implied.

## Qualification

```sh
rustup target add wasm32-unknown-unknown
export CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only
npm ci --prefix deploy/ci/typescript --ignore-scripts --no-audit --no-fund
export PATH="$PWD/deploy/ci/typescript/node_modules/.bin:$PATH"
export JOPE_TYPESCRIPT_PARITY=1
cargo test --locked -p feat-math --no-default-features --test golden -- --test-threads=2
cargo test --locked -p feat-math --no-default-features --lib advanced_v3::emit:: -- --test-threads=2
cargo test --locked -p feat-math --no-default-features --test artifact_target_binding
cargo test --locked -p jope-pipeline --test artifact_target_binding
```

`tsc`, `node`, and the WebAssembly Rust target are required when this qualification
is selected. Missing tools fail. `JOPE_TSC` can name the TypeScript executable.

The existing golden tests now compile and execute TypeScript in both strict CommonJS
and strict ES2020 module mode when the flag is enabled. They compare against the
native emitted Rust predictions, which the existing suite independently compares
with the interpreter and Python. They exercise transforms, nonlinear heads, target
mappings, RNG, symbolic programs, refits, independent instances, empty predictions,
malformed dimensions and disposal. Existing fitted tests add ordered lag/roll tails
and control/Unicode categorical values. Existing manifest tests cover determinism,
complete source closure, deletion and tampering.

The dedicated `typescript` workflow retains exact source/run/attempt identities and
logs. A workflow definition or emitted file is not passing evidence; use the
actual exact-head run result before claiming qualification. Browser compatibility
follows the import-free module contract; the current executable matrix uses Node
and does not claim a real-browser run.
