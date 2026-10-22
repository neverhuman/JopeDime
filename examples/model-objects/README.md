# Fit once, load in another process

Build the native executable, then run the dependency-free Node example in a new output directory:

```sh
cargo build --locked -p jope-appliance
node examples/model-objects/run.mjs target/debug/jopedime target/model-objects-example
```

This fits and saves the standard native object, verifies it, loads it for prediction, exports
complete bound JSON, then imports that state from an independent JavaScript producer. Both
producers use `models/sha256-…/manifest.json`. Outputs are create-only; use a new example output
directory for each run. TypeScript consumers get declarations from `client.d.mts`.

The adapter calls the native CLI using argument arrays and never invokes a shell. It provides
one practical path for new language clients to save/load the same fitted object immediately.
It does not claim a second numerical engine or native JavaScript training implementation.
`saveState` also supports other namespaced state codecs: integrity can be verified before a
compatible runtime exists, and unsupported execution remains explicit.

Python needs only its standard library to use the identical fit response and inference path:

```python
import json, subprocess

def jope(*args):
    return json.loads(subprocess.check_output(["jopedime", *args], text=True))

receipt = jope("fit", "--train", "train.csv", "--target", "y", "--task", "regression", "--json")
model_path = receipt["outputs"][0]["path"]
jope("model", "verify", "--model", model_path, "--require-executable")
jope("predict", "--model", model_path, "--input", "rows.csv", "--output", "predictions.csv", "--json")
```

C++/Java/Go and other clients use the same argument/JSON contract or MCP tools; Rust can call
`jope_runtime::model_object` directly. Independently trained language-local models follow the
[producer and decoder contract](../../docs/model-objects.md#adding-a-language-or-gp-variant).

## Acceptance and session rollback

Run `node examples/model-objects/check.mjs /absolute/path/to/jopedime` with an authenticated
appliance. It fits a model, reloads from disk after changing working directory, retains a second
opaque checkpoint, records both in one session, rolls back and compares prediction bytes. The
adapter pins executable paths containing a slash at construction time; a bare command such as
`jopedime` uses the caller's PATH on each invocation. Use an absolute executable in services.

The [format specification](../../docs/model-object-format.md) covers portable names, byte
identity and decoder requirements. The [session guide](../../docs/model-sessions.md) documents
record/show/rollback commands, expected-head conflicts and training checkpoint limitations.
`saveState` preserves supplied byte buffers; do not parse and reserialize foreign fitted state.
