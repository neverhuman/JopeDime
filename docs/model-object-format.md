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
