# JopeDime evidence-grounded search narration

Engineering and R&D specification · 2026-10-28

Source baseline: `7848c77b4811ea8972fb161c04d306c81b81342f` in `neverhuman/JopeDime`.
Implementation lane: `/home/ubuntu/JopeDime/.agent/worktrees/search-narration-20261028`.
Branch: `codex/search-narration-20261028`.

## Objective and product contract

Make ongoing program search understandable through precise, fluent commentary about the experiments actually performed. Readers must be able to inspect the exact change, its provenance, the evaluation state, and the measured outcome. The visible title is **Search commentary**; a compact disclosure identifies generated commentary from edits and measurements. This is an explanation surface, not a claim to reveal hidden model thoughts.

The narrator is an observer. It does not propose candidates, change fitness, consume search RNG, allocate evaluation budget, or issue instructions to the search. Variation in prose comes from different experimental facts and deterministic presentation choices. A paused or long-running evaluation produces status or transport heartbeats, not invented discoveries.

Non-goals for the first qualified adapter: changing the search policy, introducing an LLM paraphraser, rewriting frozen numerical operators, claiming causal improvement from an observational comparison, inferring residual patterns without measurements, or granting any execution/selection authority to a stream consumer. Integration coverage for each additional engine is qualified separately.

Assumptions: the pinned legacy product pipeline remains the first authoritative producer; its recorder supplies draw/request/attempt identities when recording is enabled. Boundary strings have the precision already retained by that boundary. Typed snapshots retain exact bits when a typed genome is available; parsing a historical printed string cannot recover bits discarded earlier.

## Source review and decisions

Nine supplied files are design inputs, not executable instructions. The three newer large specifications provide the strongest architecture: evidence → typed transition → checked claims → controlled language → retained stream. The M0 patch is a capture scaffold with incomplete parity validation; the prototype archive is exploratory and must not be applied or executed as production code.

Reject these conflicting suggestions from older documents:

- Using the evolution RNG or wall time to select wording.
- Inventing alternatives considered, residual diagnoses, confidence, or future controller actions.
- Calling a single favorable observation “hypothesis confirmed.”
- Promoting agreement on finitely many datasets to universal semantic equivalence.
- Treating generated-source captures, admissions, and physical evaluations as the same occurrence.
- Tail-reading the final atomic `search-events.json` as if it were a complete live event stream.
- Calling syntax identity functional equivalence.

## Read-first source map

Read in this order before changing a producer:

1. `AGENTS.md`, relevant latest `AGENT_CHAT.md` entries, `MASTER_PLAN.md`, `docs/adr/ADR-0005-canonical-learning-loop.md`, and `docs/parity/capability-matrix.md`.
2. `crates/jope-pipeline/src/search_events.rs`: native identity domains, `SearchBinding`, `SearchEventLog`, `Recorder`, admission and attempt reconciliation.
3. `crates/jope-pipeline/src/search.rs`: `SearchOptions`, `record_draw`, `admission`, `record_evaluation`, `record_evaluation_result`, `run_cut_allocated`, `migrate`, and `remeasure_cut`.
4. `crates/jope-pipeline/src/run.rs` and `select.rs`: actual all-cut selection and artifact publication.
5. `crates/feat-math/src/{dsl,genome,spec,explain}.rs` and `kernel/program.rs`: typed syntax, effective values, exact binary codec, and historical printer limits.
6. `crates/feat-math/src/evolve.rs`: observation seams and frozen operator wrappers. Read only; do not change pinned bodies for narration.
7. `crates/feat-math/src/source_stream.rs`, `crates/jope-observatory/src/gp_source.rs`, and `src/bin/gp-source-stream.rs`: existing persistence and streaming patterns.
8. `deploy/ci/{fast-lane.sh,rust-test-scope.sh,coverage.json,test-inventory.sh}`, `.github/workflows/ci.yml`, and `crates/jope-records/src/coverage/`: required CI and inventory contracts.

## Architecture and ownership

`jope-narrator` depends on `feat-math`, serde, serde_json, sha2, and existing error utilities. It owns facts, exact program views, diffs, claim construction, rendering, bounded capture, and the retained journal. It has no Tokio/HTTP dependency. `feat-math` does not depend back on the narrator.

`jope-pipeline` publishes owned facts through an optional bounded queue. Disabled mode performs no narration work beyond an option check. Search does not await clients or filesystem operations. `jope-observatory` provides read-only delivery and consumes the same persisted envelopes for WebSocket, CLI, finite pages, and MCP.

Write packets are disjoint:

| Owner | Write scope | Contract |
|---|---|---|
| Core worker | `crates/jope-narrator/**` | Facts, snapshots, diffs, claims, journal; publish API before adapter edits |
| Producer worker | `crates/jope-pipeline/src/{search,search_events,run,lib}.rs`, narration adapter, narration tests | Native IDs and actual lifecycle; no recorder-body or RNG changes |
| Transport worker | Observatory narration module, new binary, tests, module export | Read-only incremental index and shared transcript |
| Integrator | Workspace/consumer manifests, lockfile, CI routing, inventories, engineering docs, validation receipts | Sole index/commit owner; no agent edits another packet |

Forbidden paths: other worktrees, frozen compatibility fixtures except official owning regeneration procedures, model weights, datasets, credentials, training manifests, GPU jobs, and existing PR-owner scopes. Generated inventories are updated only through their owning tools or a documented additive merge of actual listings. No generated PASS receipts are edited by hand.

The human explicitly authorized a temporary fifth registered worktree and a parallel lane while existing PRs remain with their owners. This does not authorize removing or modifying those lanes. Claims and handoffs append under `AGENT_CHAT.md.lock`.

## Evidence and identity

Keep these identities distinct: run, renderer version, journal sequence, candidate occurrence, draw, logical request, physical attempt, program text digest, typed program identity, observation digest, source occurrence, emitted artifact, and narration message.

Native draw/request/attempt IDs are copied from the recorder, never replaced with an unrelated hash called the same name. When native recording is disabled, those references remain explicitly unavailable. A canonical program digest joins historical observations; a typed codec identity establishes exact retained parameter bits. Neither establishes semantic equivalence.

A stored envelope binds schema, run identity, decimal sequence, renderer version, original fact, evidence links, typed changes/claims, and rendered text. Cursor scope includes run identity. Large wire counters and seeds must have lossless representations when exposed to JavaScript clients; cursor parsing rejects noncanonical, overflow, wrong-run, and future values.

Every nontrivial statement belongs to an explicit class:

| Claim | Required evidence |
|---|---|
| Structural fact | Verified before/after program view and replayable net patch |
| Implementation fact | Versioned code-backed semantic rule for this dialect |
| Prospective hypothesis | Changes and prior evidence available before this candidate result |
| Measured fact | Matching observation/attempt and finite typed metric |
| Controller decision | Actual admission, survivor, or selection record |
| Interpretation | Labeled retrospective interpretation; no causal proof implied |
| Limitation | Missing trace, unsupported semantics, incomplete cuts, gaps, or precision loss |

Search telemetry is observed evidence, not an authenticated external authority. Journal integrity checks detect malformed or internally inconsistent records; cryptographic hashes alone do not authenticate their author.

## Exact diff and explanation compiler

1. Import the retained program using the exact codec. A typed `GenomeIr` path preserves `f64::to_bits()` including signed zero and adjacent floats. Historical text import marks its precision boundary.
2. For pipelines, align ordered stages with deterministic bounded dynamic programming. Exact stage match precedes same-op retune, insertion/deletion, and replacement. Tie-breaking is explicit so repeated stages do not produce unstable messages.
3. Walk the union of typed parameter keys. Preserve added/removed values and type changes. Float comparison uses bits, not epsilon. Do not stringify a value then parse it back.
4. Represent patch coordinates with explicit before/after indices. Replay the patch and require exact child codec bytes. A failed reconstruction produces a bounded unsupported record, never a confident partial explanation.
5. Describe observed net change separately from historical edit attribution. A deleted duplicate does not prove the operator selected the later duplicate. An adjacent swap description establishes boundary structure; it does not establish which internal arm ran without trace evidence.
6. Preserve ordered crossover parents even when they are equal. Inheritance can be ambiguous; equal parent heads do not eliminate either possible donor. Describe net changes relative to each known parent without inventing cut sites.
7. Symbolic and portfolio adapters may retain a verified whole-program replacement until recursive structure is independently qualified. Explicit coverage status accompanies that fallback; no pipeline-specific mechanism is guessed.

Keep requested syntax separate from effective execution. Initial semantic facts should be narrow and directly checked against `spec.rs`: legacy `logistic` resolves to ridge; `lambda_frac` maps to a row-count-scaled ridge penalty with a lower bound; clipping and bin-count requests can clamp to the same effective value. Unknown operators get exact structural commentary without generic textbook promises.

An unchanged downstream stage definition can fit different state after an upstream change. Do not call fitted state unchanged from equal syntax. Compile, fit, predict, and numerical parity are separate milestones.

## Useful, varied language

Use a controlled grammar over validated claims. Vary focus between local delta, unchanged controls, prospective question, evaluation progress, outcome, selection reason, and limitation. Derive a separate deterministic presentation choice from run/event/content identity and renderer revision. Never consume search RNG.

Examples are illustrative, not measurements:

> Stage 2 changes `cheb.degree` from 4 to 6. This tests a larger polynomial basis; validation has not finished.

> The feature stages are unchanged. `ridge.lambda_frac` changes from 0.02 to 0.04. This tests stronger regularization under the current implementation.

> Evaluation finished on cut 1: RMSE 0.2215. Cross-cut selection is still pending.

Do not generate “I noticed heavy tails,” “I considered three alternatives,” “this caused the gain,” or a promised next experiment without corresponding retained evidence. Variety never weakens a mandatory qualification. Exact numeric values, units, source refs, and pending states survive every wording variant.

Compare outcomes only when task, cut, evaluator, fidelity, objective, and direction are compatible. Use the actual selection objective; RMSE improvement alone cannot explain an objective-loss winner. Absolute differences are safe; percentages require a finite strictly positive meaningful denominator. A partial-cut result is not an all-cut winner.

## Producer lifecycle

The initial pipeline adapter emits:

1. Search start with actual cut shape, budget, objective, islands, and seed.
2. Draw with coordinate, ordered parents, seed, child or explicit failure, outcome, and native draw ref when present.
3. Admission with the same draw reference.
4. Physical attempt start with request/attempt refs. Admission alone never implies a fit started.
5. Measurement with status, actual metric fields, observation digest, and physical attempt ref.
6. Generation closure only after survivors are selected.
7. Migration with from/to islands and actual admission result.
8. Comparison pool start plus comparison draw/admission/start/result records. Do not omit comparison draws.
9. Search stop and final selection after the actual complete-cut rule runs.
10. Terminal success, cancellation, or failure; an unfinished producer is not labeled successful.

Fixed prediction has a separate contract: model/input binding, prediction started, actual execution completion/failure, rows/outputs and measured runtime if available. It does not invent mutation, fit, or evolution. Additional neural, Kernel2, and advanced-engine adapters must publish explicit coverage and pass the same evidence audit before being described as complete.

## Queue, durability, and failure semantics

Producer API exposes a bounded `sync_channel::try_send` queue handle with atomic byte reservations, not an arbitrary callback that can block or panic search. Decoding occurs after removal from the channel and propagates errors. A separate `testing` feature exposes the validated collector used in fixtures; it is absent from normal production builds. Bound record size, queued records, and retained queued payload bytes. Admission accounts the owned serialized payload rather than an incomplete estimate of selected string lengths. Oversize and saturated submissions return an explicit outcome and increment loss counters.

Queue loss is represented by an aggregate gap record. Its count is exact under the stated counter contract; its location is approximate unless each rejected event has a retained sequence. Consumers must not infer complete causal coverage across a gap.

The writer assigns contiguous journal sequence, validates facts, constructs claims, constructs one retained wording, validates it by rederivation, and writes complete JSONL envelopes. Batches contain at most 64 records before data synchronization; a queued-empty boundary or 50 ms of batch processing also triggers synchronization. A synced, atomically replaced `.commit` watermark publishes only data-synced bytes and retains capture closure status. Flush is not described as fsync. Resource limits include total events/bytes; reaching a limit closes capture with an explicit status.

`finish` explicitly closes admission and drains accepted records even when clients retain cloned senders. It returns writer errors and a summary. A writer panic or I/O failure cannot disappear into stderr while the transcript is called complete. Search result qualification and narration completeness remain distinct.

After a crash, all bytes beyond the watermark are withheld, including complete but uncommitted lines. A final incomplete committed line is corruption; a suffix beyond the watermark is reported as a withheld tail. Complete-record corruption is an error; never silently truncate past it. Journal files are create-new and run scoped, preventing accidental overwrite. The versioned v1 reader returns stored wording, guarded by a retained golden transcript covering every v1 fact kind and grammar variant. New renderers add a versioned dispatch arm and preserve the v1 reader. Unsupported revisions are explicit errors; an updated rendering is a separate derived view.

## Real-time delivery

The Rust observatory binary indexes complete appended records once and stores bounded offset/length metadata. Per-client replay seeks to those offsets instead of rescanning the whole journal per poll. It validates source envelopes and run/sequence continuity. Bounds cover index entries, file bytes, frame size, finite page records/bytes, clients, and write time.

WebSocket uses exclusive `after`, run-scoped identity, deterministic replay, client caps, bounded writes, and responsive control frames. Loopback binding and exact Host/Origin validation follow the existing source service. Wrong-run, future, malformed, and unavailable cursors are explicit errors. Quiet heartbeats report transport state and latest retained sequence; they never masquerade as scientific observations.

CLI replay/follow, finite HTTP pages, and MCP read tools expose identical stored envelopes/text. MCP stdout contains protocol frames only. Requests cannot supply arbitrary filesystem paths. Requests/lines/pages are bounded; errors remain structured and do not include raw dataset contents or secrets. Terminal text sanitizes control characters while structured evidence retains exact bounded program data.

## Ordered implementation and acceptance

1. Add core crate and consumer dependencies; verify no dependency cycle or new network/runtime service.
2. Implement snapshots, bit-exact values, deterministic patches, reconstruction validation, and explicit unsupported coverage.
3. Implement typed fact validation and controlled renderer. Prefix invariance test: future measurements cannot change earlier prospective text.
4. Implement bounded capture/journal and adversarial lifecycle tests before connecting search.
5. Add producer hooks at existing wrappers; retain native recorder schemas/IDs and frozen numerical bodies.
6. Add comparison/cancellation/terminal paths and fixed-clock on/off parity fixtures.
7. Add incremental read-only transport, real WebSocket tests, finite MCP tests, and clean CLI behavior.
8. Route new targets through maintained CI, regenerate inventories through owning tools, and review workflow/dependency trust changes.
9. Freeze source, run focused gates, retain actual exit/count/timing evidence, review exact head, then publish a draft PR if the completed scope is coherent. Full hosted CI is a separate state from local success.

Core tests: signed zero/adjacent float/near-integer round trips; 10,000 bounded pipeline pairs with exact patch replay; repeated stages; insert/delete/retune/head changes; ordered equal crossover parents; invalid digest/nonfinite metric rejection; clamps/default/alias negative cases; zero-denominator comparison; future-result prefix invariance.

Capture tests: concurrent producers, record/byte/oversize caps, retained sender at finish, explicit final loss, writer failure, create-new refusal, complete corruption, partial final line, run mismatch, sequence continuity, renderer validation, stable retained replay.

Producer tests: frozen-clock narration off/on parity for candidate bytes, scores, decisions, winner and existing artifacts with one/two islands and multiple threads; actual comparison joins; duplicates do not become physical fits; cancellation before/during work; all-fit failures; recording-disabled missing refs; migration acceptance/rejection.

Transport tests: real WebSocket initial/reconnect/live replay, exclusive boundary, future/wrong-run cursor, partial tail then completion, corruption, client bounds, health heartbeat, ping/close responsiveness, finite MCP invalid/oversize requests and framing-clean stdout.

Commands run in the claimed lane with one reused `target/`, `CARGO_INCREMENTAL=0`, `CARGO_PROFILE_DEV_DEBUG=line-tables-only`, bounded build/test workers and math threads:

```sh
cargo fmt --all -- --check
cargo test --locked -p jope-narrator
cargo test --locked -p jope-pipeline --test search_narration
cargo test --locked -p jope-observatory --test narration
cargo test --locked -p jope-runtime --test prediction_narration
cargo clippy --locked -p jope-narrator -p jope-pipeline -p jope-observatory -p jope-runtime --all-targets -- -D warnings
cargo test --locked -p jope-pipeline --test search_events
cargo test --locked -p jope-pipeline --test gp_source_capture
cargo test --locked -p feat-math --test historical_v1
cargo run --locked -p jope-records -- coverage validate
bash deploy/ci/parity-matrix.sh --no-build
git diff --check
```

Actual target names and commands are reconciled against the final source before handoff. A zero-test command, missing tool, canceled job, unrun case, path skip, or stale-head green is never a pass. Existing identity fixtures are preserved; if source identity inputs change, use only official identity-mode procedures and compare all behavioral fixtures unchanged.

## Performance and R&D

Measure representative fixed-count search with identical inputs/threads/seed, narration disabled/enabled, and independent presentation. Report CPU, wall time, peak RSS, journal throughput, queue loss, and end-to-end visibility p50/p95/p99. Provisional targets: p95 diff/planning below 2 ms for admitted short pipelines; p95 visible event below 250 ms under declared load; fixed-count overhead below 2%. These are targets until measured, not product claims. Time-budget search cannot promise identical decisions when observer overhead changes time available.

Human evaluation compares factual accuracy, exact-difference usefulness, repetition, comprehensibility, and evidence inspection time. Fluency alone is insufficient. Include blinded examples with clamped no-ops, conflicting objectives, multiple edits, equal parents, partial cuts, and missing evidence. Do not optimize for concealing that commentary is derived.

Follow-on qualification packets, each with a separate adapter/test matrix: recursive symbolic/portfolio/graph diffs; effective-plan/fitted-state sidecars; neural proposal/decoder provenance; Kernel2 refinements; runtime prediction coverage; UI timeline with evidence drill-down; locale/accessibility; persistent retention/rotation; measured long-duration load; optional budgeted prospective experiment planning. Observer mode never silently becomes a controller.

## Receipts, completion, and stop conditions

Retain source SHA/tree, dirty inventory, exact commands/env, UTC interval, real exit status, collected/executed/pass/fail/skip counts, input identity, limits, and output digests under lane evidence storage. Preserve failed attempts and append corrections. Progress updates name findings and remaining uncertainty, not merely activity.

Completion requires supported adapters clearly listed, every claimed behavior implemented, adversarial tests and fixed-count parity passing, bounded resources demonstrated, maintained CI routing validated, exact-head substantive review retained, and current hosted CI status honestly reported. Serving deployment, release, broad engine coverage, human quality, and performance qualification require their own actual evidence.

Stop dependent work on ownership overlap, changed base/API contract, missing required evidence, unsafe dependency cycle, frozen-body drift, resource exhaustion, or failing exactness/parity. Preserve the failure and diagnose; do not weaken a check to produce green. Never stop unrelated owners' jobs.

Final handoff includes branch/full head, remote worktree path, dirty files, implemented coverage, actual test counts, CI URL/status, receipts, open defects and follow-on packets, and ownership disposition. A worktree remains until its source and evidence have a verified recovery route and its lifecycle owner releases it.

## Implemented adapter boundary and identity maintenance

The first adapter is the legacy pipeline search and its public run/invent entrypoints.
Fixed prediction covers the native runtime API and the shipped math-pipeline table/cell
methods. Automatic environment capture is capped at 64 session attempts per process;
retention across process restarts remains outside this delivery. Other engines remain
explicit qualification packets. The [user guide](../search-narration.md) records the
actual command, transport, prefix-durability and resource contracts.

The maintained rust-check lane invokes `ops/ci/narration.sh`, whose per-suite counts
reject zero-test successes. Product nextest scope also includes the new crate. Coverage
assignments and additive test inventories must derive from actual Cargo discovery and
nextest listings. Static discovery, local execution and hosted CI stay distinct.

Workspace manifest/lock changes mint a new ADR-0004 evaluator identity, and narration
changes to pipeline search sources also change SearchBinding source identity. The
identity fixture must be regenerated only via the existing `identity` mode at a real
source commit. Behavioral fixture payloads remain unchanged. The PR announces both
old and new evaluator identities; byte parity is asserted within the same build.
