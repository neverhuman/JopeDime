# Model object format v1

This is the storage contract for `jope.model-object/v1`. See the
[workflow guide](model-objects.md), [session history contract](model-sessions.md),
[JSON Schema](../schemas/jope.model-object.v1.schema.json), and reference reader
[`jope_runtime::model_object`](../crates/jope-runtime/src/model_object.rs).
The repository's open-source licence also covers this specification. A new writer does
not need ONNX, a particular host language, or an archive implementation.

## Identity and directory layout

An object consists of `manifest.json` and one or more flat payload files in a directory
named `sha256-<digest>`. The digest is lowercase SHA-256 of the exact UTF-8 manifest bytes,
including whitespace and the optional final newline. Payload digests cover exact bytes.
JSON field ordering is not canonicalized. Reformatting a manifest creates a different ID.
Rewriting a payload requires updating its descriptor and therefore creates a different ID.
Repeated serialization is not a valid way to verify or copy an object.

The default root is `models`, relative to the process working directory at save time.
Writers should accept an explicit absolute root in applications and return the saved path.
The native default fit receipt returns the directory as `outputs[0].path`; that entry's
`bytes` and `sha256` describe the manifest. The manifest describes the payloads. Each fit
can carry different timing and provenance, so identical predictions need not have identical
IDs. Objects are content identities, independent of session names and version numbers.

The following is a complete manifest for a one-byte opaque state file containing byte
`0x00`. The producer must calculate the directory ID from the bytes it actually writes.

```json
{
  "schema": "jope.model-object/v1",
  "minimum_reader_version": 1,
  "format": "org.example/custom-state/v1",
  "producer": {"name": "example", "version": "1", "language": "python"},
  "primary": "state.bin",
  "required_capabilities": [],
  "files": [{
    "path": "state.bin",
    "media_type": "application/octet-stream",
    "bytes": 1,
    "sha256": "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d"
  }],
  "metadata": {"description": "retained, requires a custom decoder"}
}
```

## Envelope validation

| Field | v1 requirement |
| --- | --- |
| `schema` | Exactly `jope.model-object/v1` |
| `minimum_reader_version` | Integer `1`; a different value is refused |
| `format` | Namespaced decoder ID, at most 256 ASCII characters; nonempty slash-separated segments, characters `A-Z a-z 0-9 . : _ -` |
| `producer` | Exactly `name`, `version`, `language`, all nonempty strings |
| `primary` | Exact, case-sensitive name of one declared part |
| `required_capabilities` | Unique nonempty ASCII tokens, at most 256 characters each, characters `A-Z a-z 0-9 . / : _ -` |
| `files` | 1–256 descriptors, each exactly `path`, `media_type`, `bytes`, `sha256` |
| `metadata` | Any JSON value; informational and never an execution override |
| Part `path` | 1–128 ASCII characters from `A-Z a-z 0-9 . _ -`, first character alphanumeric or underscore or hyphen, no trailing period |
| Part `media_type` | Nonempty string; interpretation belongs to the decoder |
| Part `bytes` | Unsigned integer byte count; sum within the reader's limit |
| Part `sha256` | Exactly 64 lowercase hexadecimal digits |

All object keys must be unique, including keys inside arbitrary metadata. Escaped and
unescaped spellings of the same decoded key count as duplicates. Readers reject trailing
JSON values. The native JSON reader limits nesting to its standard recursion bound; v1
writers should keep structural depth below 128. Unsupported top-level/descriptor fields
are refused. Descriptive extensions belong in `metadata`; execution semantics belong in
an explicitly supported decoder version or required capability.

Part names must be unique under ASCII case folding. `manifest.json`, in any letter case,
is reserved. Windows device stems `CON`, `PRN`, `AUX`, `NUL`, `COM1`–`COM9`, `LPT1`–`LPT9`
are reserved case-insensitively, including names with extensions. No paths, directories,
spaces, colon, initial period, trailing period or non-ASCII filename characters are admitted.
These restrictions avoid platform-dependent aliases. JSON Schema checks the individual
fields; readers additionally enforce cross-field uniqueness, primary membership, total
bytes and actual checksums. Schema validation by itself does not establish integrity.

The native defaults are a 1 MiB manifest and 2 GiB total payload. Caller-specific smaller
limits still apply. Every file must be a regular file; direct object and part symlinks are
refused. The final resolved directory basename must equal the manifest ID, including when
the caller supplies `.`. Unlisted disk files are ignored and never deleted. Required state
must be listed; a decoder must not depend on an unlisted file or external URL.

## Decoder compatibility and learned state

Envelope version, decoder version, required capabilities, session format and session
version number have separate meanings. Changing one does not implicitly migrate another.

| Decoder | Retention | Execution in this build | Numeric wire contract |
| --- | --- | --- | --- |
| `org.jopedime/appliance-fitted/v1` | Complete `model.jope` | Native CLI/MCP table inference and Rust API | Existing native fitted file, with its header, section checksums and bound state |
| `org.jopedime/battle-artifact/v2` | Complete `state.json` | Typed Rust `BoundModel` API; no CLI CSV adapter | Existing serde representation of the full bound state, parsed with float roundtrip |
| `org.jopedime/math-pipeline/v1` | Complete `pipeline.json` and `input.json` | CLI `predict`, Rust API, `model export --format javascript` | Fitted `feat_math::model::Model` behind a versioned numeric input contract (historical f32 by default; explicit version 2 uses f64); see [below](#math-pipeline-v1) |
| Unknown decoder or any unknown required capability | Exact opaque parts | Refused, reported in inspection | Defined by the owning decoder, never guessed |

For known JSON state, the reader refuses unknown nested fields instead of silently dropping
future learned state. A new enum case, incompatible operator or changed default requires a
new decoder version or an explicitly implemented capability. This native decoder currently
implements no optional capability tokens: any nonempty requirement list prevents execution.
Unknown codecs can still be saved, copied and transported without being deserialized.
An unsupported envelope version is refused rather than partially interpreted.

A complete state codec must specify all of the following where applicable:

- Input feature names, order, types, null and unknown-category behavior, vocabularies,
  fitted imputation/scaling/encoding statistics, and output labels with their order.
- Every operator and its semantics, fitted parameters, calibration, target transforms,
  inverse transforms and final clamps; ensemble members, aggregation and weights.
- Support rows, kernels, tree structure, selected features and retained history. Stateful
  sample-time operators also need grouping, sort order, admission/cutoff and reset rules.
- Numeric scalar types, dimensions, byte order, indexing, buffer layout, null masks,
  finite-value policy, rounding/tolerances and deterministic defaults.
- Whether state supports inference only, warm starting, or exact training continuation.
  A training checkpoint additionally needs the state described in the session guide.

JSON metadata numbers are not a tensor format. Large integers can lose precision in
JavaScript; floating-point formatting, negative zero, nonfinite values and map order differ
across languages. Keep foreign payloads as bytes. A future codec can define little-endian
IEEE-754 buffers and shape descriptors without changing this envelope. Require that decoder
before interpretation. No arbitrary code, dynamic library or remote dependency is loaded by
the container reader. Decoder registration is a source change with its own conformance proof.

## math-pipeline v1

`org.jopedime/math-pipeline/v1` holds a regression pipeline fitted outside the native
appliance, so its producer's numbers can be reproduced without that producer. It has exactly
two parts and `primary` is `pipeline.json`; any other part is refused.

`pipeline.json` is `{"dsl", "model"}`: the program text and the fitted
`feat_math::model::Model` in its serde form. `input.json` is the input contract:

The input section has its own prediction version. This does not change the enclosing
`jope.model-object/v1` schema or the `org.jopedime/math-pipeline/v1` decoder ID. The
reference decoder accepts exactly these section versions:

| `input.json` marker | Numeric contract | Reader behavior |
| --- | --- | --- |
| `prediction_version` absent or `1` | Historical f32 behavior for parsing/encoding, model output, and target inversion; non-finite model output is replaced with `0.0` before target inversion. | The current decoder accepts either form. Readers predating the field accept the absent form; because they reject unknown fields, they may reject an explicit `1`. Version 1 serialization omits the marker, preserving legacy bytes and object IDs. |
| `prediction_version: 2` | f64 column scalers and encoding, model input/output, and target inversion. Missing and non-finite input cells still encode as `0.0`; a non-finite model output or target-unit prediction fails as `FITNESS_UNDEFINED`. | Readers predating version 2 reject the input section (typically as an unknown field). The current decoder loads it with the version-2 path. |

Version 1 is the compatibility default only when decoding. New producers must select the
version explicitly: construct the section with
`jope_runtime::math_pipeline::InputContract::new(columns, selected, target)`, then serialize
that contract into `input.json`. The constructor validates the contract and sets
`prediction_version` to `2`; do not omit the field or hand-build a legacy-default section
for a new object. A producer or finalizer that intentionally retains version 1 must choose
that legacy section explicitly. Changing the prediction section does not rewrite existing
objects: their manifest bytes, envelope identifiers, parts and content-derived object IDs
remain as saved. Readers do not migrate objects between prediction versions.

These two minimal contracts show the section-version difference; the remaining fields have
the same meanings and shapes. Both forms are exercised through the matching model-pipeline
decoder in `crates/jope-runtime/tests/math_pipeline.rs`.

Legacy contract (the marker is absent; explicit `1` is equivalent):

```json
{
  "task": "regression",
  "columns": [{"name": "x", "min": 0.0, "max": 1.0, "constant": false}],
  "selected": [0],
  "target": {"name": "y", "min": 0.0, "scale": 1.0}
}
```

New contract, also produced by `InputContract::new`:

```json
{
  "task": "regression",
  "columns": [{"name": "x", "min": 0.0, "max": 1.0, "constant": false}],
  "selected": [0],
  "target": {"name": "y", "min": 0.0, "scale": 1.0},
  "prediction_version": 2
}
```

| Field | Meaning |
| --- | --- |
| `task` | `regression`; v1 refuses every other task |
| `columns` | Encoded columns in order: `name`, `min` and `max` (f32 for absent/1, f64 for 2), and `constant` |
| `selected` | Distinct indices into `columns`, in the order the pipeline reads them |
| `target` | `name`, `min` and `scale` (f32 for absent/1, f64 for 2) that map pipeline output back to target units |
| `prediction_version` | Optional section version. Absent means historical version 1; version 2 must be explicit. Other values are refused. |

Both parts are decoded strictly: an unknown field anywhere, including a retained training
sample, is refused. The fitted pipeline must accept exactly `selected.len()` columns
(`feat_math::kernel::width::check_input_width`). For absent/1, prediction follows the
historical f32 path, per cell:

1. Trim (Rust `char::is_whitespace`). `""`, `na`, `n/a`, `null`, `none`, `nan`, `?` and
   `#n/a` (ASCII case-insensitive) are missing. `true`/`yes` read as 1 and `false`/`no` as 0.
   Otherwise parse as f64; a non-finite value, one outside the f32 range or unparseable text
   is missing. Parsed values are rounded to f32.
2. A missing cell or a `constant` column encodes as 0. Otherwise
   `((v - min) / (max - min)).clamp(0, 1)`.
3. The `selected` columns, widened to f64, go to `Model::predict`.
4. Each model output is rounded to f32 (non-finite becomes 0) and mapped back in f32 as
   `min + v * scale`, or `min` when `scale <= 1e-12`.

For version 2, parsing and arithmetic use f64 throughout. A missing cell, non-finite or
unparseable cell encodes as 0; otherwise the finite value is normalized as
`((v - min) / (max - min)).clamp(0, 1)` (constant columns encode as 0). Selected values
go to `Model::predict` as f64. Non-finite model outputs and non-finite target-unit results
fail that prediction with `FITNESS_UNDEFINED`; neither is replaced by zero. A constant
target still returns `min` when `scale <= 1e-12`.

Values outside a column's range therefore predict as the range's edge, and a missing cell
encodes as zero. `jopedime predict` matches columns by name in any order and reports the
others as ignored. `model export --format javascript` writes the program's ES2020 recipe,
the fitted state, and `predictRows(rows)` over raw cells in `MODEL_COLUMNS` order. Version 1
uses the same f32 steps (`Math.fround`); version 2 preserves f64 arithmetic and refuses
non-finite results, within the JavaScript emitter's transform and head slice.
Python and Rust source export, `bound-json` and `fitted` are refused for this decoder.

`crates/jope-runtime/tests/math_pipeline.rs` holds the acceptance case: a fitted pipeline from
an external producer and that producer's predictions for 78 generated rows covering in-range,
out-of-range, missing, boolean, infinite and unparseable cells. The decoder and the exported
JavaScript must each reproduce every prediction as the same f32.
`crates/jope-runtime/tests/fitted_js_qualification.rs` checks the fitted JavaScript against
`Model::predict` for every transform and head in the JavaScript slice.

## Publication, recovery and trust

The native publisher validates the envelope and any supported state decoder before creating
store paths. It creates an owner-only object directory and owner-only files on Unix, writes
and syncs each part, and atomically hard-links the complete manifest last. A successful return
also follows directory syncs on Unix. Local hard-link support is required; unsupported
filesystems fail instead of falling back to partial overwrite. Windows directory fsync and
ACL parity are not established by the Linux fixture tests; set the store ACL explicitly.

Two publishers of the same bytes can safely complete the same object. A complete matching
object is idempotent. For an incomplete directory, a retry verifies existing declared parts,
adds missing parts and commits the manifest. A differing/corrupt retained part blocks retry;
it is never replaced. A process can leave `.jope-write-*` temporary files. They are not
committed state and do not block a retry. Inventory and preserve unique content before manual
cleanup. A committed corrupt object is an integrity error; recover a verified backup or save
a new object. Readers never accept a directory without the final manifest.

Hashes detect disagreement with retained metadata, not author identity, trusted provenance,
statistical validity or safe resource use. The store and its ancestor directories must be
trusted local filesystem space. Hostile path replacement races, hostile hard links,
untrusted numerical payloads, object-store mounts, NFS consistency and power-loss behavior
on every platform are outside the qualified contract. Byte limits do not bound numerical
allocation or computation: a small malformed dimension can request large work. A successful
inspection means byte verification and acceptance by a registered decoder/binding validator;
it is not an exhaustive numerical-state audit or a training-resume certificate. Treat foreign
executable state as trusted input, or run inference in a separate resource-limited process.

Native tests cover all current 26 appliance recipes, additional bound program forms, tamper
refusal and local concurrency. Every additional language executor still needs independent
fit/save/process-exit/load/inference fixtures and malformed-state/resource tests. A shared
container does not establish cross-platform numerical parity.

Fitted JavaScript export validates input width and all prediction-time recipe literals
against fitted state: every append flag, bin counts, Chebyshev degree, landmark gamma
and step threshold. A contradictory declaration refuses export. Learned arrays remain
authoritative; fit-only seeds and regularization are not reconstructed or refitted.
