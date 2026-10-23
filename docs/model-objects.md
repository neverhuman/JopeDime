# Fitted model objects

`jopedime fit` saves a complete, reusable model in `models/sha256-<manifest-digest>/`.
The JSON response's `outputs[0].path` is the directory to pass to `predict`. Set
`--model-root PATH` to choose another store. An explicit `--output model.jope` keeps
the existing single-file workflow. `fit-predict` remains the explicit transient operation.

```sh
cargo build --locked -p jope-appliance
# Use target/debug/jopedime below, or an installed, authenticated appliance.
jopedime fit --train train.csv --target y --task regression --method ridge --json > fit.json
# Read outputs[0].path from fit.json into MODEL using your language's JSON parser.
jopedime predict --model "$MODEL" --input rows.csv --output predictions.csv --json
jopedime model inspect --model "$MODEL"
jopedime model verify --model "$MODEL" --require-executable
jopedime model list --root models
jopedime model export --model "$MODEL" --format fitted --output model.jope
jopedime model export --model "$MODEL" --format bound-json --output state.json
jopedime model export --model "$MODEL" --format object --output copied-models
jopedime model import --input model.jope --root models
```

The object wrapper retains the exact native fitted file: preprocessing statistics,
category vocabularies, missing-value policy, fitted parameters, class order, training
procedure and provenance. Loading never refits. Existing `.jope` files remain readable.
The existing `models/manifest.json` and `models/bundles/` catalog describe pretrained
assets used by `jope models`; fitted estimators live in the separate `sha256-*`
subdirectories and are managed by `jopedime model`.
The default fit receipt's model `path` identifies the object directory; its `bytes` and
`sha256` describe `manifest.json`. The manifest binds the payload lengths and hashes.
Training-time timing/provenance is part of the retained model, so separate fits can
produce different object IDs even when predictions agree. IDs describe retained objects,
not a mathematical equivalence class of predictors.

## An open envelope for every producer

The owned, versioned format is **`jope.model-object/v1`**, specified here and in
[`schemas/jope.model-object.v1.schema.json`](../schemas/jope.model-object.v1.schema.json).
See the normative [format specification](model-object-format.md),
[session and rollback guide](model-sessions.md), and
[red-team review](reviews/model-object-red-team-20261022.md).
Its envelope uses ordinary UTF-8 JSON, raw payload files and SHA-256. No ONNX library,
Python pickle, language object dump, compiler or source execution is needed to retain it.
Rust implementation: `jope_runtime::model_object`.

| File or field | Contract |
| --- | --- |
| `models/sha256-<64 lowercase hex>/manifest.json` | Common commit marker and location for every language |
| `schema`, `minimum_reader_version` | Envelope and reader compatibility, independent of the state decoder |
| `format` | Namespaced, versioned state decoder; never a language-specific class name |
| `producer` | Writer name, version and language; informational provenance, not authority |
| `primary` | Main state file, explicitly listed in `files` |
| `files` | Every required part's flat filename, media type, byte length and SHA-256 |
| `required_capabilities` | Semantics a reader must understand before execution |
| `metadata` | Extensible descriptive metadata; cannot enable execution or override requirements |
| `model.jope` | Native appliance fitted state under `org.jopedime/appliance-fitted/v1` |
| `state.json` | Complete bound state under `org.jopedime/battle-artifact/v2` |
| `pipeline.json` + `input.json` | Fitted feat-math pipeline and its numeric input contract under `org.jopedime/math-pipeline/v1` |
| Additional `.bin` or `.json` parts | Optional typed buffers, operator parameters, support rows, calibration or other fitted state, as defined by that decoder |

The directory ID is `sha256-` plus SHA-256 of the **exact manifest bytes**. A reader
hashes retained bytes; it does not reserialize JSON. This avoids dependence on differences
in language float formatting, property ordering or Unicode escaping. Two different
manifest serializations intentionally have different IDs. Each part's digest covers its
exact file bytes. Never parse and rewrite a foreign manifest to copy an object.

Payload numeric types, endianness, dimensions, row semantics and operator semantics belong
to the versioned decoder contract. JSON numbers in descriptive metadata are not a tensor
wire format. A future decoder can specify little-endian IEEE-754 buffers, shapes and null
masks inside its state file and list every buffer in `files`. Preserve all training-derived
state needed at inference, including support rows and ordered-history state where applicable.
Dependencies must be included as parts; v1 does not fetch external URLs or install plugins.

The container can retain new program forms and arbitrary binary state without changing the
storage envelope. **Retention is broader than execution.** This build executes appliance
fitted state and validated `battle-artifact-v2` bound state through the Rust API. CLI/MCP
CSV prediction uses appliance objects because those carry the original user table and
class-label contract. A standalone bound object may be inspected, verified and exported;
use its typed `BoundModel` API for inference. Neither arbitrary future operators nor every
possible variant is already executable in every target language.

## Adding a language or program variant

A language calling the managed `jopedime fit --json` path automatically receives the
same persisted object convention. Python, Rust, TypeScript, JavaScript, C++, Java and other
languages can call the CLI without inventing a serializer. See the runnable
[Node/TypeScript-facing adapter](../examples/model-objects/client.mjs) and
[example walkthrough](../examples/model-objects/README.md).

An independently implemented estimator uses this adapter contract:

1. After a successful fit, serialize **all learned state**, preprocessing, schemas and output
   ordering with a declared versioned codec. A recipe or program alone is insufficient.
2. Write payload parts in a staging directory. Compute hashes and lengths over their exact
   bytes. Write the manifest, using the same schema and fields, with the producer language.
3. Call `jopedime model import --manifest STAGING/manifest.json --root models`, or MCP
   `model_import`. Both use the same validator/publisher. Import returns the object path and ID.
4. To load, verify the ID, envelope, part lengths and hashes; check the decoder and **every**
   required capability; then reconstruct state without fitting. Unknown capabilities prevent
   execution and remain explicit in inspection. A native framework's own opaque state can be
   retained under its own decoder ID, but is not thereby made portable to other runtimes.
5. Register a new decoder/version with a documented state schema, independent fixtures,
   numerical tolerances, malformed-state refusals and fit-save-process-exit-load-predict tests.
   Qualify each language executor separately before claiming parity. Never reuse an existing
   format ID for changed semantics, or hide a semantic extension in `metadata`.

Current bound program forms use `ModelObject::from_bound(&model)`. It serializes the complete
existing bound representation and calls its binding validator on reload; it does not lower
it into a smaller set of standard operators. For other forms, construct `Manifest`, `Part`
and `ModelObject::new` with the exact serialized bytes. Future schema versions and missing
readers fail explicitly; no implicit migration or silent fallback is performed.

```rust,ignore
use jope_runtime::model_object::{load_fitted, ModelObject};
use jope_runtime::limits::Limits;
let object = ModelObject::from_fitted(&fit_outcome.fitted)?;
let path = object.save(std::path::Path::new("models"), Limits::default().max_model_bytes)?;
let fitted = load_fitted(&path, Limits::default().max_model_bytes)?;
// fitted.predict(raw_rows) uses retained state; no training input or fit call is needed.
```

## Export and ONNX

`model export --format object` copies the exact envelope and payloads to another store.
`fitted` preserves the native `.jope` bytes; `bound-json` exports validated bound state.
`python`, `rust` and `javascript` use the existing portable fitted emitter and its explicit
coverage gate. Unsupported heads, transformations or state fail; nothing is omitted to make
an export appear successful. Source functions accept the existing encoded raw-table cells;
retain the original schema, category encoding and class order alongside them. This PR does
not expand the emitter's numerical coverage or retrofit persistence into every standalone
recipe's language-local `Model.fit`. Those recipes can adopt the adapter contract above.

An appliance fitted with an objective may retain an output calibration after its native model.
Object copies and `fitted` exports preserve that complete state and its predictions. When the
calibration changes predictions, `bound-json` and portable source exports return an unsupported
backend error before creating output; those formats cannot yet encode the extra map. Inspection,
verification and inference still use the complete fitted object. The model MCP prediction tool
accepts and ignores retained auxiliary columns just as CLI prediction does.

ONNX is a useful optional interoperability target. Its [IR](https://onnx.ai/onnx/repo-docs/IR.html)
and [custom operator mechanism](https://onnx.ai/onnx/intro/concepts.html) allow extension
namespaces, but a runtime still needs implementations for those operators. A private ONNX
fork would add maintenance and would not supply those implementations. The owned envelope
therefore preserves complete state independently of ONNX. A future exporter can produce
standard ONNX for an admitted subset or a versioned JopeDime custom-domain graph paired
with a compatible runtime and conformance tests. `--format onnx` currently returns backend
unavailable (exit 4), creates no output, and makes this missing capability explicit.

## MCP

Configure a stdio server with command `jopedime`, arguments
`["model", "mcp", "--root", "/absolute/path/to/models"]`. It authenticates packaged
executables before touching the store and speaks MCP `2025-11-25`, one JSON-RPC message per
line, with stdout reserved for protocol traffic.

| Tool | Behavior |
| --- | --- |
| `model_list` | List objects; report invalid or uncommitted entries explicitly |
| `model_inspect {id}` | Integrity verification plus execution capability report |
| `model_verify {id}` | Additionally require a supported, valid decoder |
| `model_export {id}` | Exact manifest text and `parts_hex` for language-neutral transport |
| `model_import {manifest, parts_hex}` | Validate and commit to the configured root; returns ID/path |
| `model_session_show {name}` | Read the session history and active model path |
| `model_session_record {name,id,expected_head,label?}` | Append a saved candidate after comparing the observed head |
| `model_session_rollback {name,version,expected_head,label?}` | Select a previous version while retaining history |
| `model_predict {id, csv}` | Load appliance state and infer on inline CSV; returns predictions CSV |

IDs must match `sha256-[0-9a-f]{64}` and cannot select arbitrary filesystem paths. The server
never runs uploaded code. Notifications do not execute tools. Frames/responses are limited
to 8 MiB, inline prediction CSV to 1 MiB/10,000 rows, and numerical work uses one thread.
Use the CLI for larger models. The protocol is intended for a trusted local MCP client;
it does not add authentication for network clients or sandbox numerical computation.

## Integrity, publication and limits

Readers check compatibility, primary membership, case-insensitively unique portable filenames, unique JSON keys, finite bounds,
checksums and supported decoder bindings. The manifest limit is 1 MiB; at most 256 parts
and 2 GiB total payload are accepted by default. Reads remain bounded if a file grows.
Direct object/part symlinks and path traversal are refused. The store is trusted local
filesystem space; concurrent hostile filesystem changes is outside this contract.

Publication validates supported state before store creation, creates owner-only directories
and files on Unix, writes and syncs create-only
parts, then atomically links `manifest.json` as the commit marker. It requires a local
filesystem supporting hard links. A reader never accepts an object lacking a committed
manifest. A crash can leave an uncommitted directory; it is listed as invalid. A retry resumes
when all existing declared parts agree, adds missing parts and commits the manifest. Conflicting
state is never overwritten or deleted. Concurrent identical publishers are safe. A retry of a
fully committed identical object is idempotent. Exports are create-only and cannot replace existing destinations. Model input
directories are protected against prediction output/receipt aliases. Preserve unique state
before manually reclaiming incomplete objects.

Objects are immutable by API convention, not OS write-protected or signed. A hash checks
integrity against retained metadata, not publisher authenticity or scientific quality.
Store copies and access controls remain the caller's responsibility. Kernel/smoother/support
models can retain training rows; the native header's retained-data notice is preserved.
Generated local `models/sha256-*/` is gitignored and is not silently uploaded or committed.

## Versioned sessions

Use `model session record` after a successful save to retain successive candidates in one
named session. Every update compares the previously observed head; `model session rollback`
appends a selection of a previous version without modifying or deleting any model. The returned
`model_path` is suitable for prediction. The [session guide](model-sessions.md) documents
commands, concurrent updates, backup/recovery and the additional state needed for exact
training continuation. Session history lives under `models/sessions/`, separately from each
immutable object. No automatic pruning or training-resume implementation is introduced.

## Capability matrix

| Producer or consumer | Managed fit persists objects | Native disk inference | Independent native executor |
| --- | --- | --- | --- |
| Rust runtime | `ModelObject::from_fitted` / `from_bound`, then `save` | Rust fitted/bound API | Shared native implementation |
| Native CLI or MCP client in any language | CLI `fit --json`; MCP imports already fitted bytes | CLI/MCP appliance objects | Delegates to packaged native executable |
| Node / TypeScript-facing example | `JopeModel.fit` / `saveState` | Adapter calls packaged executable | No JavaScript numerical executor added |
| Independent Python, C++, Java or other estimator | Implement declared codec and call CLI/MCP import | Requires a registered compatible decoder | Must qualify separately |
| Exported Python / Rust / JavaScript source | No automatic persistence added to standalone recipes | Existing emitter's admitted subset only | Export coverage gate may refuse |
| `math-pipeline/v1` object from an external producer | Producer writes `pipeline.json` + `input.json`, then CLI `model import --manifest` | CLI `predict` | `model export --format javascript` carries the fitted state; JavaScript slice only |
| ONNX export | Unimplemented, exit 4 | Unimplemented | A custom domain also needs a compatible runtime |

Inspection uses response schema `jope.model-object-inspection/v1`, distinct from the stored
manifest schema. `executable` reports registered decoder/binding acceptance, not complete
numerical-state validation, platform parity or resumability. Unknown state stays opaque and
retained. Byte bounds do not make hostile numerical state safe; see the format's trust limits.
