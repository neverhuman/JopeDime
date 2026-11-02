# Browser prediction runtime

A prediction-only WASM ABI over the existing `jope-runtime` model-object decoder and
`feat-math` fitted predictors. It validates original manifests and part hashes and retains
native preprocessing, missing values, fitted parameters and target inversion. No model is
refitted and no model-supplied JavaScript, WASM, host import or executable source is loaded.

Build: `rustup target add wasm32-unknown-unknown`, then `bash web/browser-predict/build.sh`.
Verify: `bash web/browser-predict/check.sh` runs native contracts and Node/WASM parity over
native-generated ridge and composite-GPR fixtures, missing/out-of-range cells, corruption,
stale-model refusal and message limits. The cloud app separately verifies actual browsers,
owner-scoped transfer, its CSP/worker integration and the public development journey.

This is a standalone build workspace: it reuses the product crates by path, with the C++
runtime disabled, but adds no browser dependencies to the native appliance. Build output
uses the lane's existing root `target/` (or the already selected CI target). The checked-in
lock pins browser dependencies. `target/browser-predict/runtime-build.json` identifies the
source commit, toolchain, lock, ABI, memory ceiling and WASM hash for consumers.

`browser_input(n)` allocates the bounded message buffer. Copy UTF-8 JSON into that buffer,
then call `browser_load()` with `{manifest: original_manifest_string, parts: {name: base64}}`
or `browser_predict()` with `{values: [string_or_null, ...]}` in the model's column order.
Read `{ok, result}` or `{ok:false, error}` JSON at `browser_output()` / `browser_output_len()`.
The loaded metadata supplies input names/kinds, target and saved input-policy explanation.
`browser_dispose()` releases model/buffer state. All operations are serialized in a dedicated
first-party Web Worker; callers terminate work that exceeds the interactive time budget.

Admission is bounded to 32 MiB of decoded model parts and 256 columns, one prediction row,
8 KiB per cell and 512 MiB maximum linear memory. Current native decoders are reused;
unrecognized formats/capabilities and bound classification objects without a saved class
mapping are explicitly refused. Native appliance class-labelled predictions retain their
native probability output. Parallel native execution is disabled in this single worker.
