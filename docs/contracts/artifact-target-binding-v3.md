# Requested artifact binding v3

This opt-in contract binds retained output bytes. It does not attest to scientific
execution, authorize training, or establish executable parity. Legacy recipe
`battle-artifact-v1`, bound `battle-artifact-v2`, existing reports and default
writers retain their original bytes and limitations.

The draft payload discriminator is `jope.battle-artifact/v3`, inside I02's
`jope.canonical-json/v1` envelope. The integrator owns its schema catalogue and
`schemas/jope.battle-artifact.v3.schema.json` adoption. The Rust adapter has a
closed set of ASCII field names and only strings, booleans, nulls, arrays and
safe integers. Its sorted JSON projection therefore follows I02's JCS ordering
without adding a dependency on jope-records. The payload ID is
`sha256(schema_utf8 || 0x00 || u64_be(payload_bytes.len()) || payload_bytes)`.
Own ID fields stay outside the payload. Duplicate fields, unknown fields,
noncanonical encodings and legacy envelopes refuse. Native, inference and storage
domain tags are transport annotations; they confer no execution authority.

The caller retains `ExpectedArtifactV3` independently of the output directory:
exact envelope SHA-256, selected canonical program SHA-256, and requested targets.
`verify_directory`, pipeline `read_identity_v3` and publication `verify_run_v3`
require that pin. Reconstructing it from a mutable receipt or manifest removes the
external trust anchor. A selected manifest hashes its canonical program and binds
the dialect, I07 registry source digest, all emission options, actual emitter
version, target product, source lengths/digests, declared build and fitted state.
Output sources are rendered before publication, then disk readback must match the
bound bytes. Missing files fail even when a loose receipt's digests were updated.

Each requested target has exactly one terminal record: `supported` with its
complete file closure or `unsupported` with a nonempty reason and registry source
digest. Empty requests, unknown targets and duplicates (including aliases) refuse.
`all_supported()` is false for any unsupported or invalid manifest. Supported
files are required; a missing artifact cannot become an exclusion. Writers require
new files and refuse existing destinations, symlinks and ambiguous paths. They
retain partial output after a failure for diagnosis; retry into a fresh directory.

| Request | Product | Bound files | Declared build |
| --- | --- | --- | --- |
| python / py | recipe v1, or bound v2 | model.py | Python 3 |
| rust / rs | recipe v1, or bound v2 | model.rs | Rust 2021, optimization |
| cpp17 / c++17 | recipe v1 | model.cpp | C++17, selected compiler and flags |
| cpp20 / c++20 | recipe v1 | model.cpp | C++20, selected compiler and flags |
| javascript / js | recipe v1 | model.js | ES2020 CommonJS, Node |
| go / golang | recipe v1 or bound v2 | model.go, model.go.kernel.rs, go.mod, go.sum | Go 1.25+, pinned wazero, Rust Wasm |
| cpp17-recipe-v2 | expression recipe v2 | cpp17-recipe-v2/model.cpp | C++17 |
| cpp17-fitted | fitted predictor | cpp17-fitted/model.cpp, model.codec.json, standalone.cpp | C++17, standalone define |

C++17 and C++20 explicitly share the same source only when the file binding is
identical. Their standards and build identities differ. Expression-recipe-v2 and
fitted sources use separate paths and products. C++ flags carry optimization,
optional ISA selection and the existing numeric policy flags. `execution: unknown`
means a declared build only; no compiler version, binary or runtime measurement is
invented. A measured compiler execution requires a separate receipt.

Fitted C++ binds actual model state, exact training-row bit patterns, task, schema,
vocabulary and all three output files. Missing or substituted codec, standalone
source or model source refuses. The bound-model adapter binds the existing model
and fitted input state and checks the supplied program against its stored program
identity; it preserves v2 Python/Rust bytes and explicitly excludes other targets.
Fitted-state digests identify the exact existing serialized state representation;
they are not a new canonical fitted-model codec.

`transpile::emit_targets_v3` uses actual emitter admission and I07's per-stage
recipe support declarations. Existing expression-recipe and fitted adapters retain
their independent versioned gates; I07 has no new fitted target allocation for
legacy registry rows. Runtime or admission errors cannot expand emitter coverage.
The pipeline writer returns manifest plus external pin. `publish_v3` also accepts
the bound-model adapter's rendered sources. Publication dispatches v3 envelopes
through complete target and exclusion checks. `verify_run` retains its existing
mutable evidence-chain trust model; `verify_run_v3` adds an independent pin.
Neither performs a fit replay or authenticates a run.

The new `kernel::emit_genome_v3` emit-only adapter preserves language aliases and
numeric training-CSV validation. Select it through the existing string argument,
for example `jope emit --lang v3:rust,js,cpp20 --output <new-directory> --json`.
It requires an output directory and returns `jope.emit.v3`. The `manifest-v3`
artifact in that report carries the root digest; the caller retains its requested
targets from the invocation and derives the program digest from the report's
canonical genome. Unsupported CLI requests return a nonzero refusal with their reasons and retained
manifest digest/path. The programmatic adapter returns the explicit terminal
states with `all_supported()` false. Default CLI invocation still uses the old route.
Additional parser options and receipt selections belong to the integrator. No old
receipt is upgraded by reading or reserializing it.

Integration must serialize the feat-math module export, v3 schema/catalogue,
CLI parser and run-receipt selection hooks, coverage/discovery and inventory rows,
and the I09 evaluator identity transition after I07/I08. Frozen files and historical
emitted pins stay unchanged. Lane tests distinguish byte binding, executable
fixture checks, legacy diagnostic coverage and pending integration gates.
