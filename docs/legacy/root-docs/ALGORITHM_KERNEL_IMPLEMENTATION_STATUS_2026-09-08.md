**Status: verified compatibility and scalar-expression work; the seven-package
upgrade and synchronized deployment are not complete.**

This records the initial implementation boundary after the audit. Later source
mirrors and verification are recorded in the
[action plan](ALGORITHM_KERNEL_VERIFICATION_ACTIONS_2026-09-08.md) and
[follow-up report](../parallel/kernel-verification/followup-20260908-01/REPORT.md).
It is not a release acceptance receipt. The supplied plan's thresholds remain unchanged. Research ideas were
assessed separately in [the Krantz assessment](KRANTZ_KERNEL_ASSESSMENT_2026-09-08.md).

| Package | Implemented and checked | Remaining before package acceptance |
|---|---|---|
| 1 — Preservation/builds | Local and remote inventories; canonical kernel/mirror verification; full kernel/native/grammar source packaging; immutable source staging and isolated build output; missing Hyperion helper source imported and checked against remote hashes | Complete all-consumer reconciliation, executable builds and dependency/environment qualification, checkpoint/corpus inventory and remote drift reconciliation |
| 2 — Safety/admission | Exact typed legacy codec and identities; explicit invalid-clip errors; whole-expression Rust/Python/C++ fallback; missing-column guards; repaired C++ artifact revision and historical artifact regeneration; separate pipeline slots | Full operation contracts and registry shared by fitting, task/schema/backend binding, compiled admission, and search consumers |
| 3 — Scalar evolution | Shared AST adapters; versioned subtree/constant mutation and crossover; bounded constant fitting; fixed-arity register execution; row-summary reuse; width/column checks in a separate pipeline API | Exact neural literal/token/cache integration, production search integration and frozen-control quality/throughput qualification |
| 4 — Programmable core | Not implemented | AlgorithmV1, tensors/records/state, functions/control flow, resource-accounted executor and expandable algorithms |
| 5 — Differentiation/tasks | Not implemented | AD, mathematical modules, structured outputs, genuine multilabel and forecasting contracts |
| 6 — Backend pipeline | Legacy expression safety work only | Shared new-core training/prediction exports, complete JavaScript/CUDA host/device lowering, fitted state and numeric-profile qualification |
| 7 — Search/neural migration | Not implemented | Versioned admitted-program consumers, V2 corpus/vocabulary/scorer/checkpoint migration, metrics and training/recovery qualification |

The new `kernel::Program` currently contains **only `LegacyV1`**. Lossless decoding
is not executable admission. `kernel::pipeline` checks slots and conservative
legacy widths; it does not certify every parameter, task, effect, backend or
resource requirement. No `CompiledProgram`/`bind` gate is wired into fitting yet.
Existing mutation APIs and their RNG streams remain unchanged. New scalar and
pipeline operators are separate APIs and have not replaced worker proposal routes.

**Compatibility repairs.**

Typed parameters, strings, signed zero, subnormal values and IEEE NaN payloads
roundtrip without passing through the frozen lossy DSL printer. Binary encoding
uses explicit tags, lengths, ordering and float bits. Duplicate/unordered keys,
truncation and oversized writes are rejected. Historical identities remain
available alongside the new domain-separated identity. Scalar AST encoding and
new compilation/evolution entrypoints check structural limits. This is not a
complete parser/codec fuzz qualification; general legacy text parsing still needs
adversarial boundary work.

Invalid NaN clip bounds return an error instead of panicking. Legacy callers retain
the established whole-expression zero fallback. Repaired Python and Rust recipes
have a distinct emitter identity when clipping is present. Fitted C++ expressions
guard missing columns before any evaluation and propagate clip failures to the
whole expression. Fitted C++ emitter revision 2 is recorded in runtime metadata.
Revision-1 containers are decoded and regenerated from their embedded fitted model
before the shared library is loaded; the original bytes are preserved. A test
constructs a historical unsafe library and verifies this path actually executes.

The pre-existing `golden.rs` change was formatting only. The original bytes and
patch were saved; formatting both versions produced identical files. The tracked
fixture was restored without editing any expected hash. See
[`golden-reconciliation.json`](audits/algorithm-kernel-upgrade-2026-09-08/golden-reconciliation.json).

**Verification evidence.**

| Check | Observed result and scope |
|---|---|
| Full kernel library tests | 267 passed, 2 ignored at the earlier compatibility boundary; subsequently added historical-regeneration and codec-allocation tests passed separately |
| Full golden suite | 83 passed in 631.63 seconds; final no-summary dispatch subsequently checked through operation and export tests |
| Legacy freeze | 3 passed, 1 intentionally ignored; expected hashes unchanged |
| Exact codec | 4 integration tests passed plus the allocation-limit unit test |
| Scalar evolution | 4 tests passed, including 100,000 deterministic scoped children with interpreter/register result comparison |
| Pipeline evolution | 3 tests passed, including 100,000 structural/width checks and a small actual-fit sample for the five omitted modern learner families |
| Final expression operation integration | 10 tests passed |
| Fitted C++ export | 13 tests passed, including ASan/UBSan checks for unsafe expression cases |
| Python/Rust invalid-expression export | 1 integration test passed after compiling/running both emitted programs |
| Portable scalar parity | Existing Rust/Python/C++17/JavaScript parity fixture passed |
| Source release tooling | 4 tests passed: coverage, isolation, tamper/drift/symlink handling and optional helper dependencies |
| Immutable parallel source | Main library and Hyperion helper CPU library compile with their separate locked dependency sets |
| Root application integration | **Blocked:** `src/hyperion_router.rs:17` imports `crate::router_contracts_v2`, whose source/module is absent in the supplied tree. This unrelated unfinished integration was not invented or removed. |

The 100,000-child checks are **not** the plan's all-profile admission gate: they do
not execute every child against every task/backend or establish full effect/resource
safety. Sanitizer coverage is limited to the saved cases. No full search-quality,
noninferiority, genuine multilabel, forecast-isolation, eight-GPU failure recovery,
or 24-hour integrated qualification result is claimed.

Test logs and inventories are in
[`docs/audits/algorithm-kernel-upgrade-2026-09-08`](audits/algorithm-kernel-upgrade-2026-09-08).
One intermediate compile failed due to a missing delimiter; its failure log is
retained, and the corrected operation/export checks passed.

**Performance and research evidence.**

The same-source expression microbenchmark found summary hoisting useful for the
tested wide-row formula and register overhead for a simple narrow-row formula.
Map/derive retain direct evaluation without row summaries. This does not establish
the required historical-control throughput threshold or qualify a fast profile.

The exact-rational approximation counterexamples and ten-seed actual-learner
screening probe support falsification tests for approximation assumptions. The
screening probe's 40 MSE values replayed exactly, excluding timing. No production
screening policy, regime encoder, learned controller or neural vocabulary was
added because of the research proposals.

**Release and remote state.**

The unqualified local source release is
`parallel/releases/kernel-v1-6a464377bc6f6f6fb25dee91997f0675bd6db4bf90ca86744baf24860f9d8e42`.
It contains 362 files, including 85 canonical kernel files and the 121-file
Hyperion source dependency. Its [content manifest](audits/algorithm-kernel-upgrade-2026-09-08/source-release.json)
records hashes, sizes, executable bits, dependency coverage and unqualified status.
Build output is separate from immutable source. Datasets and model assets still
require separate verified references; source inclusion does not qualify binaries.

At the initial boundary, remote operations were read-only. The service definitions and executable hashes
for CPU-v44, neural-v44 and the active monitor match the baseline. The old campaign
is still active; completed-cycle receipts are not a terminal campaign receipt.
No queue, vocabulary, checkpoint, service, executable route or campaign input was
changed. The helper source copy matched hashes before and after transfer.

Subsequent work mirrored immutable source and verification evidence into separate
remote `kernel-work` and `kernel-work-evidence` directories, with hash verification
and unchanged live identities. This supersedes the initial absence of remote
staging. There has been no runtime cutover, worker expansion, checkpoint migration,
integrated GPU qualification or rollback demonstration. Those remain gated by the
original plan and by completion of the current campaign.
