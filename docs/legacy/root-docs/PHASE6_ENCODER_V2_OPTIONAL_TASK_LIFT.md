# Phase 6: Encoder V2, Optional-Task Evidence, and Defensible Lift

Status: proposed execution plan; shadow/offline work may begin after Phase 5 source integration, but no production activation is allowed before the Phase 5 recovery bundle completes its 86,400-second continuity gate.

Owners: Codex owns the V2 identity/cache contract, benchmark authority, recovery integration, and release gates. Claude owns an independent identity audit, capability-routing review, and bounded encoder/optional-task implementation slices agreed in `REALTIME_AGENT_CHAT.md`. Every material finding and ownership change is recorded there.

## 1. Mission

Build an encoder and evidence pipeline that is dramatically faster, reproducible enough to defend scientifically, and incapable of confusing bytes produced by different models, weights, devices, runtimes, datasets, or foundation inputs.

Phase 6 succeeds only if it produces all of the following:

1. A complete `JopeFoundationEncoderIdentity.v2` plus `JopeFoundationDatasetInput.v2` whose combined digest changes when any result-affecting raw-encoder or train-input byte changes, followed by a separately keyed `JopeContextTransformIdentity.v2` for Foundation projection, standardization, fit cohort, and task transforms.
2. A content-addressed `JopeExpandedEncodingCache.v2` namespace in which incompatible outputs can coexist and identity collisions fail closed.
3. Byte-exact same-capsule replay on a pre-registered sample, plus explicit cross-backend equivalence evidence when CPU and CUDA are intentionally different identities.
4. Measured encoding throughput and latency improvement under matched hardware and work, without borrowing capacity by cancelling evaluations or breaking the useful-work runway.
5. Restored, typed multiclass and time-series evidence with independent datasets, task-correct metrics, task-valid heads, and no test-derived training features.
6. A paired, matched-compute lift report with uncertainty intervals and an immutable experiment manifest. Raw best-of-N or unequal-attempt comparisons are not acceptable evidence.
7. A shadow catalog and shadow controller run that can be replayed from immutable artifacts and does not advance a production model, catalog, seal, evaluator, or promotion pointer.

## 2. Non-goals and frozen surfaces

The following remain frozen until their separate gates pass:

- Production model-pointer advancement.
- Promotion-panel publication or promotion-controller activation.
- Seal creation, seal replacement, or sealed-cohort regeneration.
- Population-learning or multi-generation production campaigns.
- Rewriting, deleting, or repairing V1 evidence manifests in place.
- Reusing a V1 cache path for V2 output.
- Treating corpus-wide producer Elo as a promotion gate. Eight-slot, per-context Bradley-Terry remains the controlled rating authority.
- Enabling `jope-selfplay-expansion.service` before the V2 identity, cache, replay, and shadow gates are green.

## 3. Proven starting point

The execution plan starts from evidence already established in Phase 5:

- Foundation V1 has 41,854 immutable objects totaling 348,225,280 bytes.
- Its content-addressed inventory is `235f2004…`, projection is `3df46159…`, and standardization is `daac224c…`.
- A complete migration replay returned identical document digests and reverified every legacy source and destination link.
- Existing V1 cache identity binds the dataset, protocol, and output hashes, but it does not fully bind the encoder executable, model weights, numerical runtime, backend actually executed, or foundation documents.
- Existing cache paths are keyed by task and dataset ID. That makes CPU/CUDA or weight variants collide even when their bytes legitimately differ.
- Historical slice/binary logs suggest roughly 570–588 datasets/hour on their old cohorts, while a real uncached CPU encode ran for roughly 30 minutes. These are incompatible observational contexts, not a fleet-wide baseline; the frozen V2 A/B plan is the only speed authority.
- Multiclass has substantial corpus evidence, but three-node replay must be capability-routed without hostname pins. Time-series evidence remains incomplete and must use rolling-origin, time-causal evaluation.
- Phase 5 keeps expansion inactive and production pointers frozen.

## 4. Target architecture

```text
encoder source/weights/runtime/backend identity
        +
train-only dataset/reduction input identity
        |
        v
encoder identity + train-input identity --SHA-256--> cache key
        |                                  |
        |                                  +--> immutable output objects
        |                                  +--> cache manifest v2
        |                                  +--> replay receipt
        v
raw 2,048-float vector
        +
Foundation documents + fit cohort + task transform
        |
        v
context-transform identity ------------> shadow expanded catalog
                                                |
                                                +--> optional-task evidence
                                                +--> task-aware shadow training
                                                +--> matched-compute lift report
```

V1 remains readable and immutable. V2 is additive and lives under a disjoint root. There is no in-place migration path.

## 5. Workstream A: close Phase 5 before runtime activation

This is the only runtime-critical predecessor.

### A1. Finish source and test closure

- Preserve global frozen multiclass spec revision v2.
- Introduce a replay-only spec revision for the new capability-gated replay form.
- Route the three replay roles through distinct, unprovisioned `slot-a`, `slot-b`, and `slot-c` capability markers.
- Keep missing markers fail-closed. Marker provisioning is a separate reviewed deployment step.
- Pass focused Rust/Python suites, full workspace tests, all-target check, boundary scan, spec-drift scan, and diff checks.

### A2. Integrate with bounded commits

- Stage explicit paths only.
- Keep campaigns, archives, caches, `__pycache__`, submodule changes, and unrelated user files out of commits.
- Create bounded source-policy/foundation, useful-work/recovery, bundle/audit, and task-aware reporting commits.
- Claude reviews Codex commits; Codex reviews Claude-owned paths.
- Record both bundle-creation acknowledgements in `REALTIME_AGENT_CHAT.md`.

### A3. Build and stage the recovery bundle

- Build release binaries from a clean detached worktree at the reviewed integration commit.
- Build one content-addressed bundle using Prime support from immutable bundle `9521ffce…`.
- Verify Prime source-bundle, package-tree, and per-file hashes.
- Verify evaluator retention closure and active service/unit closure.
- Synchronize and hash-verify both the new bundle and rollback bundle across xbabe0 through xbabe4 without advancing the active pointer.
- Do not activate until runway, freezer, causal, telemetry, rollback, unit, and continuity prerequisites pass.

## 6. Workstream B: define complete Encoder V2 identity

### B1. Deterministic identity split

Use three layered canonical documents so reusable raw-encoder provenance is not duplicated, downstream context transforms do not force raw recomputation, and no dataset input can be confused with encoder code.

`JopeFoundationEncoderIdentity.v2` must contain, at minimum:

- Schema and identity version.
- Encoder protocol name and a content-addressed protocol artifact.
- Phase 6 bundle digest, `foundation.py` hash, wrapper/runner hashes, native supervisor/attestor executable SHA-256, and build provenance: repository commit, Cargo/Python/native lock digests, compiler/runtime versions, target triple, enabled features, and profile. The numerical encoder remains the bundled Python/PyTorch implementation; the native component supervises it and verifies its receipt.
- Model architecture/config digest, expert implementation-tree digest, every expert source-file hash, and every model-weight/model-manifest SHA-256.
- Prime package/config digests and the immutable Prime support source-bundle digest.
- Preprocessing constants, the single pooled-vector shape, seed policy, and explicit feature layout (`mean`, `std`, `p10`, `p90` by hidden index).
- Numerical runtime contract: requested and required actual backend, device class, deterministic-mode flags, dtype, accumulation dtype, installed package/library names and versions plus each loaded package-tree digest, driver identity, thread counts, and relevant kernel policy.
- Batch-size and memory policy when they can affect numerical output.
- Output-format contract: exactly one finite float32 vector with 2,048 features.

Publishable raw identities require deterministic mode. A nondeterministic runtime may emit a failed/exploratory attempt receipt, but it cannot produce a healthy binding.

`JopeFoundationDatasetInput.v2` must contain:

- Task, immutable source-family identity, train SHA-256 and byte size, feature/target schema digest, and row/column counts.
- A train-only assertion and the exact content-addressed files the encoder is allowed to read.
- For time series, the direct-reduction code/version, metadata digest, horizon, frequency, lag policy, and known-future/past-only covariate declarations.
- Test hashes only as evaluation provenance in a disjoint experiment manifest. Test bytes and test-derived statistics are forbidden from the encoder input identity and execution sandbox.

`JopeContextTransformIdentity.v2` separately binds the raw encoder cache key and output SHA-256 to the Foundation inventory/projection/standardization documents, a content-addressed transform protocol, task transform/reduction identity, and frozen standardization fit-cohort digest. Raw 2,048-float outputs do not claim to consume these downstream documents, so changing a projection or fit cohort invalidates the context transform without recomputing valid raw encoder objects.

Canonical JSON independently produces `encoder_identity_sha256` and `input_identity_sha256`. The cache key is SHA-256 of a domain separator plus both digests. Dataset ID, path existence, and mutable path timestamps never participate. Missing, unknown, or unverified result-affecting fields fail closed; they do not serialize as healthy identity.

### B2. Backend attestation

The encoder, not its caller, reports the backend actually used.

- `cuda-requested` is not proof of CUDA execution.
- CPU-requested and CUDA-requested work use different encoder identities and namespaces.
- Any actual-device mismatch or CPU fallback after a CUDA request hard-fails the attempt. It is never silently relabeled and published as success.
- The receipt binds visible devices, selected device, backend, dtype, deterministic flags, and peak memory.
- Caller intent and encoder-observed execution are both retained. A mismatch is a typed failure, not a warning.
- Observational node, queue-wait, timing, utilization, and peak-memory facts live in an attempt receipt. They are bound to the deterministic identities and output digest but do not make the same deterministic object acquire a node-specific identity.

### B3. Identity validation tests

Add table-driven tests that change one field at a time and require a different capsule digest:

- One weight byte.
- Supervisor executable, protocol artifact, or loaded runtime package-tree digest.
- Foundation inventory/projection/standardization/fit-cohort digest at the context-transform layer.
- Dataset train or allowed-metadata digest. A test digest changes only the separate evaluator/experiment identity and must not change the raw cache key.
- CPU versus CUDA.
- Float32 versus mixed precision.
- Deterministic-mode false is rejected for publication.
- Task reduction at the input/context-transform layer.
- Raw output feature-layout schema.

Also test canonical field ordering, duplicate-field rejection, unknown critical-field rejection, and stable serialization across repeated runs.

The release sample contains at least 128 stratified inputs. Two reruns on the same declared backend must be byte-identical, and the V1 foundation tree's hashes, sizes, and read-only modes must remain unchanged.

## 7. Workstream C: content-addressed cache V2

### C1. Layout

Use a disjoint root such as:

```text
encodings/expanded-v2/
  objects/sha256/<prefix>/<output_sha256>
  bindings/<prefix>/<cache_key_sha256>/binding.v2.json
  attempts/<prefix>/<attempt_sha256>.json
  namespaces/<encoder_identity_sha256>/<input_identity_sha256>.pointer.json
  staging/<transaction_uuid>/
  object-metadata/<prefix>/<output_sha256>/
    cache-manifest.v2.json
    encoder-receipt.v2.json
    signature-manifest.v2.json
  context-transforms/<context_transform_sha256>/binding.v2.json
  datasets/<task>/<input_identity_sha256>/<encoder_identity_sha256>.pointer.json
  locks/<cache_key_sha256>.lock
  conflicts/<conflict_sha256>.json
  benchmarks/<benchmark_sha256>.json
```

Dataset projections point to immutable objects; they never own the bytes.

### C2. Publication transaction

1. Validate the dataset input and mirror identity.
2. Resolve and validate the complete encoder identity.
3. Acquire a lock on the combined cache key, not dataset ID.
4. Materialize in a same-filesystem staging directory.
5. Run the native supervisor around the bundled Python/PyTorch encoder with a receipt FD/path that the child must fill.
6. Verify observed backend and all capsule fields.
7. Hash the raw output and validate exactly one 2,048-element float32 vector, exact byte size, declared feature layout, and finite values.
8. Flush files and directory metadata, set immutable payload modes to `0444`, and publish by an exclusive same-filesystem hard link (`O_EXCL` semantics) into the output-SHA object namespace.
9. Publish the deterministic binding receipt only after the object exists and rehashes correctly; then flush its directory.
10. Publish a small content-addressed namespace/dataset pointer.
11. Publish the observational attempt receipt separately, including failed attempts.
12. Remove only the transaction's own staging directory after the durable binding exists.

Any existing binding for the same encoder/input key must name the same output SHA-256. A different output is a fatal nondeterminism collision: publish a conflict receipt, stop the lane, and block activation. Interrupted writes must leave no visible binding and no partial object addressable by a healthy pointer.

### C3. V1 compatibility

- V1 validation remains available for rollback and historical evidence.
- V1 caches are never repaired into V2 paths or rewritten with V2 fields.
- V1 and V2 may coexist for the same dataset.
- Shadow comparison explicitly names both cache identities.
- Garbage collection cannot remove V1 through the Phase 6 rollback window.

## 8. Workstream D: make encoding fast without corrupting identity

### D1. Establish the baseline first

Create `JopeEncoderBenchmarkPlan.v2` with a stratified, immutable dataset sample:

- All four tasks represented when evidence exists.
- Bins for rows, columns, class count/horizon, sparsity, and raw byte size.
- Fixed dataset identities and input hashes.
- Fixed hardware pool and explicit node capability receipts.
- Cold-cache and warm-cache lanes.
- CPU reference, current CUDA, and each proposed optimization lane.
- At least three timed repetitions after one untimed warm-up per lane.

Publish per-dataset wall time, kernel time, input/output time, peak RAM/VRAM, GPU utilization, output bytes, and cache-validation time. Report median, p90, p95, and throughput; do not report only the fastest example.

Every attempt also records queue wait, expert load, preprocess, encode, verify/publish, cache lookup, rows, features, bytes, actual device, peak RSS/VRAM, output hash, and typed failure class. Aggregate p50/p95/p99 and datasets/hour per reserved GPU slot and CPU-hour, partitioned by task, backend, shape bin, and cold/warm state. Missing or stale measurements fail closed.

Historical logs are context, not the benchmark authority: the slice-500 lane completed 500/500 at p50 6.31 seconds and p95 7.72 seconds (about 588/hour), while binary-42 completed 41/41 at p50 2.36 seconds and p95 25.35 seconds (about 570/hour). Only the frozen controlled V2 A/B cohort may support a release speed claim.

### D2. Optimization order

Optimize in this order so each gain is attributable:

1. Remove avoidable process startup and repeated model/weight loading.
2. Keep a bounded resident encoder worker per GPU capability lane.
3. Batch compatible datasets by shape/memory class while retaining one receipt per dataset.
4. Pipeline input hashing/copy, GPU execution, and output hashing with bounded queues.
5. Use reflinks/hard links for immutable inputs where safe.
6. Preallocate/reuse device and pinned-host buffers within the capsule policy.
7. Tune batch size using measured VRAM headroom, never retry-dependent silent fallback.
8. Add CPU parallelism only after proving it does not steal the four companion CPU credits required by a waiting GPU lane or violate useful-work runway.
9. Profile remaining kernel, serialization, and I/O hotspots before algorithmic changes.

### D3. Concurrency and recovery priority

- Use a dedicated, low-weight V2 project and an explicit CPU or GPU namespace; implicit device fallback is forbidden.
- Start with one long-lived worker per verified GPU capability lane and bounded chunks with a durable per-dataset commit.
- Ramp from one to two or more workers only after zero collisions/fallbacks, at least 1.5x measured RAM/VRAM headroom, and green availability thresholds: CPU PSI at most 0.25, memory PSI at most 0.05, I/O PSI at most 0.10, and free memory at least 15%.
- Pause before impairing recovery useful-work runway, the trainer, proposer, waiting four-CPU GPU companion credits, or evaluator service. Never cancel evaluations or release/cancel the three held multiclass jobs.

### D4. Performance release gates

Pre-register both absolute safety gates and relative speed gates:

- Zero identity collisions and zero unexplained output drift.
- 100% successful same-capsule replay on the release sample.
- No OOM, fallback, or backend mismatch hidden as success.
- Control-plane overhead below 5% of end-to-end time at steady state.
- Cache-hit validation p95 below the pre-registered budget and at least 20x faster than recomputation.
- Minimum safety gate: throughput per reserved slot at least the matched baseline and p95 latency no worse than 1.25x baseline.
- Screaming-fast release objective: candidate V2 median throughput at least 3x the matched current baseline and p95 latency at most 50% of baseline. Missing the objective requires publishing the bottleneck profile and a reviewed exception; it never waives the safety gate.
- During GPU encoding, useful work remains above its dynamic target and no evaluation is cancelled.

The 3x/50% targets are release objectives, not permission to discard slower datasets. Publish the full distribution and all failures.

## 9. Workstream E: optional-task evidence restoration

### E1. Shared admission contract

Introduce `JopeOptionalTaskReadiness.v2` with one row for exactly each optional task:

- Task.
- State: `absent`, `collecting`, `replay_pending`, `shadow_ready`, or `blocked`.
- Independent dataset count and minimum required count.
- Family-disjoint count.
- Valid cache V2 count.
- Robust-bank count and digest.
- Evaluator/protocol digest.
- Task-valid head count.
- Replay node/capability receipts.
- Freshness and explicit blocker list.

Absence remains typed; it is never coerced to zero performance.

Reuse completed coverage assets instead of rerunning them: multiclass has 272 datasets with 11,696/11,696 valid coverage rows; time series has 42 datasets with 126/126 valid coverage rows and 877 numeric comparator baselines plus five declared skips. These assets are test-adaptive and benchmark-invalid: they prove comparator/coverage availability only, never unbiased scientific lift.

### E2. Multiclass

- Preserve global campaign identities and historical receipts.
- Use replay-only v3 specs with three distinct role-capability markers.
- Require one successful, content-identical evaluator replay per role and validate distinct observed nodes.
- Maintain macro-F1 as the primary metric, accuracy as secondary, and complexity/hash ordering only as deterministic tie-breakers.
- Require task-valid multiclass terminal heads and reject target transforms or regression-only heads.
- Build a fixed-draw robust bank and eight-slot per-context rating evidence.
- Encode multiclass from train bytes only.

### E3. Time series

- Reach the independently verified dataset floor without synthesizing duplicate families.
- Bind frequency, horizon, seasonality, target position, known-future covariates, past-only covariates, and reduction version into dataset/context identity.
- Use rolling-origin evaluation only; random folds are forbidden.
- The authoritative primary utility is negative MASE and the native error is family-balanced MASE. Predeclare RMSE or sMAPE as a secondary when defined; a secondary never substitutes for missing MASE.
- Prevent future leakage in standardization, feature construction, lag creation, and foundation/context encoding.
- Produce a robust bank and typed telemetry row before optional-task admission is enabled.
- Encode only a content-pinned, train-only direct-reduction artifact; neither test bytes nor future-derived statistics may enter the input identity or encoder process.

Publish `JopeOptionalTaskEvidence.v1` for this restoration work. A task remains typed absence until encoder coverage, context identity, retained evaluator, comparator lock, and fresh unbiased shadow rows all verify.

### E4. Supply integration

- Optional tasks stay out of production supply until `shadow_ready`.
- Shadow supply may build task-specific waves under a separate namespace and hard cap.
- The protected two-wave reserve and 6,144+ useful-work contract continue to apply to production recovery work.
- Shadow encoding/proposal jobs yield to recovery and do not consume protected reserve manifests.

## 10. Workstream F: prove real lift

### F1. Immutable experiment manifest

Publish `JopeEncoderLiftExperiment.v2` before running the comparison. It binds:

- V1 and V2 capsule/cache identities.
- Dataset/family split and exclusion rules.
- Seeds, folds/rolling origins, draw count, candidate count, and compute budget.
- Task-specific primary/secondary metrics.
- Noninferiority margin and minimum practically important improvement.
- Confidence-interval method and multiplicity correction.
- Failure handling and missing-data policy.
- Stop rule and maximum compute.

### F2. Matched comparison

- Compare the same datasets, contexts, candidates, seeds, and evaluator versions.
- Equalize proposal attempts and training examples.
- Separate encoder-quality lift from search-volume lift and wall-clock lift.
- Use paired per-dataset deltas.
- Publish bootstrap confidence intervals and a sign/win-rate summary.
- Apply a pre-declared correction across task-level primary tests.
- Report all tasks, including negative or absent results.

### F3. Lift gates

V2 may advance from shadow only if:

- Same-capsule replay is exact.
- Scientific quality is noninferior on every activated task.
- At least one pre-registered primary task or aggregate shows a practically meaningful positive lift whose confidence interval clears the declared threshold, or the throughput gain is large enough to be the declared primary benefit while quality remains noninferior.
- No task crosses a catastrophic-regression boundary.
- Eight-slot task ratings and conventional task metrics agree directionally or the disagreement is investigated and published.
- Results reproduce from the immutable experiment manifest.

The primary matched panel uses equal eight-slot compute and family-disjoint datasets. Cluster bootstrap resampling operates at the declared family/dataset unit. The pre-registered advance gate requires all of:

- The 95% lower confidence bound of paired family-balanced utility is above zero.
- The task-native primary error improves rather than merely the aggregate rating.
- The 95% lower confidence bound of the strict-win rate is above 0.5.
- p95 inference/evaluation latency is no worse than 1.25x the comparator.
- Absolute failure-rate delta is at most 0.25 percentage points.

If multiplicity correction or the noninferiority margin is changed after the manifest is frozen, the run is exploratory and cannot satisfy the gate.

## 11. Workstream G: shadow integration and failure recovery

### G1. Shadow-only activation

- Add V2 readers behind an explicit shadow flag.
- Write a shadow expanded catalog and shadow proposal/training products to disjoint roots.
- Keep production expansion disabled.
- Make every shadow product name its capsule, cache, catalog, evaluator, and model identity.

### G2. Recovery drills

Test all of the following before any production enablement:

- Process crash during staging.
- Node loss during an encode.
- GPU OOM with no silent CPU identity substitution.
- Existing-object collision.
- Corrupt output file.
- Stale or missing Foundation document.
- Model-weight drift.
- Dataset mirror drift.
- Duplicate concurrent request for the same capsule.
- V1 rollback while V2 objects remain present.

Each drill must leave either one valid immutable object or no published object, plus a typed failure receipt.

## 12. Workstream H: production rollout gates

Production expansion remains inactive until all gates are green:

1. Phase 5 recovery continuity is green for 86,400 uninterrupted seconds.
2. One effective Phase 5 bundle/process hash exists across all five nodes.
3. Encoder V2 identity mutation tests pass.
4. Same-capsule replay passes on the full release sample.
5. Benchmark speed and resource gates pass.
6. Optional-task readiness is typed and fresh; no task is admitted by missing-data default.
7. Lift experiment is complete and reproducible.
8. V2 shadow catalog runs for a pre-registered observation window without drift.
9. V1 rollback is verified and V1 artifacts remain readable.
10. Codex and Claude explicitly acknowledge the V2 bundle and activation in `REALTIME_AGENT_CHAT.md`.

Rollout is canary-first: one bounded V2 tranche, one catalog merge, one shadow/training observation, then an explicit hold for review. No automatic model promotion is coupled to V2 activation.

## 13. Fast execution sequence

The schedule is expressed as dependency waves, not optimistic calendar promises.

### Wave 0: now, parallel with Phase 5 closure

- Finish full-suite and capability-marker fixes.
- Land the detailed plan and ownership ledger.
- Add pure V2 identity types, canonical hashing, and mutation tests only; no reader/writer activation.
- Build the benchmark inventory query and baseline manifest generator in audit-only mode.

### Wave 1: immediately after Phase 5 integration commit

- Build and stage the clean recovery bundle.
- Synchronize new and rollback bundles to all nodes after reciprocal acknowledgement.
- Implement the V2 object layout and atomic publisher behind offline CLI commands.
- Add the native supervisor/bundled-Python encoder receipt and attestation handshake.

### Wave 2: during the mandatory Phase 5 continuity window

- Run CPU/CUDA determinism and benchmark samples under spare capacity.
- Optimize startup, resident model loading, batching, and pipelining one change at a time.
- Restore multiclass replay evidence and time-series readiness in shadow namespaces.
- Publish benchmark and optional-readiness telemetry without enabling expansion.

### Wave 3: after identity and benchmark gates

- Freeze the immutable matched-compute lift experiment manifest.
- Run V1 versus V2 shadow comparisons.
- Reproduce the experiment from artifacts.
- Review negative cases and task disagreements.

### Wave 4: after Phase 5 continuity and V2 shadow gates

- Build a V2-capable bundle from a reviewed clean commit.
- Synchronize to all five nodes.
- Perform one bounded expansion canary with rollback armed.
- Observe, audit, and stop for explicit go/no-go. Promotion remains separate.

## 14. Parallel ownership and communication cadence

To move quickly without merging incompatible assumptions:

- Codex: identity schemas, cache namespace/publisher, benchmark authority, useful-work integration, lift design, bundle/audit, and rollout gates.
- Claude: independent identity threat model, native supervisor/Python encoder backend-attestation slice, optional-task replay/readiness slice, and performance review as assigned.
- Shared files require an ownership message before editing.
- Each owner stages explicit paths and reports the exact list.
- Each bounded commit receives reciprocal review before it enters an integration commit.
- Post a heartbeat to `REALTIME_AGENT_CHAT.md` at least every 15 minutes during long tests/builds/campaigns, and immediately on identity drift, scientific failure, scope change, runtime blocker, or material benchmark result.
- Never use chat acknowledgement as evidence by itself; link it to a commit, digest, test receipt, or immutable artifact.

## 15. Required public evidence

Phase 6 introduces or requires these interfaces:

- `JopeFoundationEncoderIdentity.v2`
- `JopeFoundationDatasetInput.v2`
- `JopeContextTransformIdentity.v2`
- `JopeEncoderExecutionReceipt.v2`
- `JopeExpandedEncodingCache.v2`
- `JopeExpandedCachePointer.v2`
- `JopeEncoderBenchmarkPlan.v2`
- `JopeEncoderBenchmarkReport.v2`
- `JopeOptionalTaskReadiness.v2`
- `JopeEncoderLiftExperiment.v2`
- `JopeEncoderLiftReport.v2`
- Existing `JopeFourTaskTelemetry.v1`, `JopeTaskRatingSnapshot.v1`, `JopeMicroWaveSupplyStatus.v2`, `JopeEvaluatorRetentionIndex.v1`, `JopeEffectiveUnitAudit.v1`, and `JopeRecoveryContinuity.v1`

Every status has a timestamp, freshness budget, explicit input digests, typed absence, and fail-closed validation.

## 16. Definition of screaming, defensible progress

Fast means parallel work on independent immutable artifacts, short feedback loops, resident/batched encoding, bounded queues, and no repeated expensive work when a verified cache exists.

Defensible means every claimed speedup has a matched baseline and full distribution; every claimed lift has equal attempts, paired datasets, uncertainty, and immutable provenance; every missing task remains explicitly absent; and every production transition has a verified rollback.

The phase is complete when the system is simultaneously faster, scientifically noninferior or better under the pre-registered lift test, replayable from content-addressed inputs, optional-task aware, and still able to roll back to the verified V1/Phase 5 state without changing an accepted model pointer.
