# Jankurai R2 P5 exact-file exceptions

**Owner authorization:** Jepson, delivered by Grok Bot in `jankurai_R2_P5_jk_p5_t1820.md` (2026-11-02). **Auditor:** Jankurai 1.7.6. **Review date:** 2026-11-02.

This register explains 46 P5 findings that were inspected and classified as genuine scanner false positives or as compatibility boundaries. The matching exact file paths are listed in `agent/audit-policy.toml`; the auditor excludes only those paths from Jankurai scoring. Rust/Python build and test tools, Gitleaks, and other security checks continue to scan them. No directory or broad glob is excluded here. Remove an entry when its justification no longer applies, and re-audit any source/format migration.

The code finding in `crates/jope-bench/src/automl.rs` was addressed in code by removing `clippy::all` and narrowly allowing the two lint classes reported in its byte-pinned entrant; that file is deliberately not excluded.

| Packet item | Exact path | Reason for the Jankurai exception |
|---:|---|---|
| 1 | `crates/battle-gpu/src/driver.rs` | `Library`'s `Send` implementation is preceded by a local safety invariant: the loader reference is owned, can move without thread affinity, and remains loaded until cleanup. Removing `Send` would prevent the mutex-owned backend from moving to its calling thread. |
| 2 | `crates/battle-gpu/src/lib.rs` | `Gpu` is moved only behind the backend `Mutex`; every call makes the context current after locking and cleanup is exclusive. The nearby `SAFETY` argument documents this invariant. |
| 3 | `crates/jope-api/src/bin/jope-api-openapi.rs` | This is one deterministic default output path used only when the optional CLI path is absent. It is not a fallback chain and does not hide an I/O failure. |
| 4 | `crates/jope-api/src/error.rs` | Public API `ErrorCode` represents HTTP/problem responses; the Smartcluster wire protocol has a different error vocabulary. A shared superset would broaden accepted/emitted protocol variants. |
| 5 | `crates/jope-api/src/events.rs` | API event bodies and learning-inventory event bodies describe different schemas and lifecycle boundaries; their variants are not interchangeable. |
| 6–8 | `crates/jope-api/src/runs.rs` | `Task`, `LandmarkMode`, and `Direction` are transport/API schema types. The flagged benchmark, dataset, arena, learning, and runtime enums have distinct variants and serialized contracts. |
| 9 | `crates/jope-api/src/warnings.rs` | API warning severity and telemetry alarm severity are independently consumed schemas; centralizing them would couple public API and telemetry vocabularies. |
| 10 | `crates/jope-appliance/src/mcp/dispatch.rs` | MCP `Outcome` is a response-dispatch result. Engine, training, lake-reader, outbox, and retention outcomes encode different state transitions and payloads. |
| 11 | `crates/jope-appliance/src/mcp/mod.rs` | `serve_stdio` is synchronous and creates its own Tokio runtime before calling `block_on`; it is not blocking from an async executor task. |
| 12 | `crates/jope-arena/src/family.rs` | “Stale” denotes an analysis blacklist that does not match the required dataset set; the error directs the operator to reconcile that real cache/freshness condition. |
| 13 | `crates/jope-assets/src/decode.rs` | Asset decoding `Codec` and parquet-retention `Codec` describe different byte formats and boundaries; combining the enums would expand decoder choices. |
| 14 | `crates/jope-bench/adapters/_panel.py` | The private panel CLI replays only bytes it just created in the same fit invocation. The artifact is created exclusively with mode `0600`, its complete digest is checked against the trusted in-process pin, and there is no artifact-import endpoint. The module and README explicitly state that pickle is executable and arbitrary/untrusted artifacts must never be loaded. This is a documented trust boundary, not a claim that pickle is safe. Revisit if external artifact loading is added. |
| 15–18 | `crates/jope-bench/adapters/test_panel.py` | These `pickle.loads` calls inspect artifacts freshly produced by the test's own fit helper or spy on that replay call. They do not deserialize user-supplied fixtures or remote inputs. |
| 20 | `crates/jope-bench/src/emit.rs` | The emitter `Stage` records distinct emission lifecycle states; the other listed stages are evaluation, reservation, capsule, wave-git, or CUDA execution vocabularies. |
| 21 | `crates/jope-cli/src/research_view.rs` | CLI research-view freshness is a display classification; learning rebuild freshness is a different persisted state machine. |
| 22 | `crates/jope-cli/src/schema.rs` | CLI `Kind` is a command/schema selector; progress inventory and retention policy `Kind` have different variants and meanings. |
| 23 | `crates/jope-client/src/inproc.rs` | The flagged `block_on` sites are test-only runtime setup on a dedicated server thread and a regression test that deliberately calls the synchronous adapter from async code. Product dispatch runs on the bridge runtime. |
| 24 | `crates/jope-datasets/src/blobs.rs` | “Stale” names a cached blob whose digest/size disagrees with its content-addressed key; the code removes the bad cache entry and refuses an offline substitute. |
| 25 | `crates/jope-datasets/src/cli.rs` | Dataset catalog commands are distinct from models, motifs, and pipeline command enums. A shared command union would enlarge unrelated CLIs. |
| 26 | `crates/jope-learning/src/clover/round_qual.rs` | `TODO` is one of the explicitly recognized placeholder sentinels rejected in caller-supplied preregistration material; removing it changes validation behavior. |
| 27 | `crates/jope-learning/src/lake/archive.rs` | Lake archive fidelity and edit-policy fidelity express different provenance guarantees and are not substitutable. |
| 28 | `crates/jope-learning/src/lake/roles.rs` | Lake storage purpose and replay-admission purpose govern different access/retention boundaries. |
| 29 | `crates/jope-learning/src/main.rs` | The matched text is the diagnostic format string `unsafe {what}`, not an unsafe Rust block. |
| 30 | `crates/jope-learning/src/metric.rs` | Learning metrics and pipeline/runtime metrics have distinct vocabularies and consumer schemas. |
| 31 | `crates/jope-learning/src/pools/pool.rs` | Pool `Status` describes pool lifecycle; observatory `Status` describes GP source observation. |
| 32 | `crates/jope-learning/src/progress/inventory.rs` | Learning inventory failure classes and Smartcluster job failure classes feed separate persisted protocols. |
| 33 | `crates/jope-learning/src/progress/segment.rs` | Segment insertion results and proposal-outbox insertion results are from separate transactions and return different state. |
| 34 | `crates/jope-learning/src/state.rs` | The matched text is the diagnostic format string `unsafe {what}`, not an unsafe Rust block. |
| 35 | `crates/jope-narrator/src/render/v2.rs` | Renderer revision 2 remains a dispatched replay implementation for records stamped with its revision; replacing it would change historical narration bytes. |
| 36 | `crates/jope-narrator/src/render/v3.rs` | Renderer revision 3 is the current renderer and wraps revision 2 while preserving both revision identities for replay. |
| 37 | `crates/jope-pipeline/src/experiment_v2.rs` | This public module validates the v2 evaluation-result schema; v1 remains readable, so the versioned type is an active compatibility contract. |
| 38 | `crates/jope-policy/src/decode.rs` | Policy decoding `FailureKind` and runtime error `FailureKind` classify different failure boundaries and are converted at the call site. |
| 39 | `crates/jope-records/src/repo_layout.rs` | `TempDir` is a typed repository-layout allowlist kind with count validation; “temp” is domain terminology, not a placeholder marker. |
| 40 | `crates/jope-runtime/src/fitness.rs` | Fitness aggregation and classification aggregation are separate computation contracts, even though their enum names coincide. |
| 41 | `crates/jope-service/src/store/mod.rs` | Service-store errors and Smartcluster daemon-store errors carry different operational context and cross crate boundaries. |
| 42–45 | `crates/jope-service/src/telemetry.rs` | The four zero initializers provide valid zeroed C ABI output storage for `cpu_set_t`, `statvfs`, and `vm_statistics64`; each has a nearby validity argument before the OS fills it. The types contain integer/bitset fields, not invalid Rust references or enums. |
| 46 | `crates/jope-service/src/worker/frame.rs` | Service worker framing errors and Smartcluster framing errors describe different framing APIs and are translated at the transport boundary. |
| 47 | `crates/smartcluster/jain-smartcluster-daemon/src/control.rs` | Daemon control errors and jainnode control errors belong to separate request protocols and retain different caller-facing variants. |

## Scope and review

These exceptions suppress only Jankurai's finding generation for the exact files above. They do not assert that every line in a file is safe, do not change runtime behavior, and do not waive unit tests, strict Clippy, Gitleaks, or the P5 acceptance audit. The trust-boundary exception for pickle is conditional on the current private fit/replay call graph. The P5 implementer performed a same-agent security review of this policy delta; there is no independent review claim.
