# v1.0.0 Jankurai release audit triage

This is the release-note rationale for the full read-only audit of `2e0b7723b56a4edfcde87a875d24555724fc6801`. The score is **47/85**, with **544 findings: 468 hard and 76 soft across 29 rule families**. The audit failed its conformance gate. A documented disposition does not mean a defect was fixed, a safety argument was proved, or a capability was qualified.

The [complete finding register](jankurai-v1-findings.json) retains every original finding ordinal, fingerprint, severity, path and line, with a hash of the source file and flagged line where they exist. Each entry has an explicit disposition, rationale and owner/closure route. Group totals below derive from that register. Raw secret-like snippets are redacted; their source identity is retained.

The installed Jankurai 1.6.11 binary has SHA-256 `9aa101c10cdc40c8a15e3f5c1341be3bce78e86f61754d849bbe8229e485a783`. Its digest differs from the earlier same-version binary used for the 1,995-finding report. The changed finding count is not evidence of a source improvement. The complete audit command was:

```sh
JANKURAI_NO_UPDATE_CHECK=1 jankurai audit . --full --no-score-history \
  --json /path/to/evidence/repo-score.json \
  --md /path/to/evidence/repo-score.md \
  --timings-json /path/to/evidence/timings.json
```

The original JSON report SHA-256 is `bbe87e72d5718f9e456c906c924d419d5178cde684bd3889059d3610dc824307`. This audit ran in a clean registered worktree. It took 308.35 seconds and exited 1 because the score/hard findings did not meet the gate. Output generation and a complete failed audit are separate from an audit execution error.

## Release limitations

The unresolved unsafe-site, handle-threading, unbounded-channel and executable-deserialization findings remain risks. This release must not advertise arbitrary artifacts as safely loadable, bounded live campaigns as qualified, complete GPU safety, complete W1–W12 remediation, or a clean Jankurai audit. Missing runtime/provenance proof remains missing. Existing replay fixtures, source pins, assertions, tolerances, failure records and test lanes remain intact.

The README badge is routed to Lane B's existing release change. ROOT separately adds a direct contributor link to AGENTS.md, independently confirmed absent on this source; that supplemental navigation repair was flagged by the earlier auditor but is not an additional finding in this 544-row report.

## Every finding class

| Rule family | Findings | Disposition and rationale groups |
| --- | ---: | --- |
| HLT-000-SCORE-DIMENSION | 3 | ci-audit (1): Record out-of-band audit without a passing gate claim; profile (2): Document the reference-profile exception |
| HLT-001-DEAD-MARKER | 75 | fallback (1): Keep bounded adapter failures visible; marker (72): Retain lexical warnings with their exact locations; shape (1): Defer module-size restructuring; unsupported (1): Preserve explicit unsupported-state refusal |
| HLT-003-OWNERLESS-PATH | 1 | context (1): Retain context/ownership dimension debt |
| HLT-005-PYTHON-PRODUCT-TRUTH | 2 | profile (2): Document the reference-profile exception |
| HLT-006-DIRECT-DB-WRONG-LAYER | 1 | boundary (1): Retain unqualified boundary/provenance scope |
| HLT-007-HANDWRITTEN-CONTRACT | 1 | boundary (1): Retain unqualified boundary/provenance scope |
| HLT-008-FALSE-GREEN-RISK | 1 | fixture-freeze (1): Preserve immutable historical compatibility bytes |
| HLT-009-GENERATED-SECURITY | 1 | security-lane (1): Separate actual security checks from reference names |
| HLT-010-SECRET-SPRAWL | 1 | secret-pattern (1): Redact a scanner-policy token without claiming secret-free history |
| HLT-015-CONTEXT-SETUP-GAP | 1 | context (1): Retain context/ownership dimension debt |
| HLT-016-SUPPLY-CHAIN-DRIFT | 2 | security-lane (2): Separate actual security checks from reference names |
| HLT-017-OPAQUE-OBSERVABILITY | 1 | observability (1): Retain partial observability coverage |
| HLT-019-STREAMING-RUNTIME-DRIFT | 1 | nats (1): Reject logarithmic-unit text as a streaming client |
| HLT-023-INPUT-BOUNDARY-GAP | 1 | sql-product (1): Preserve parameterized SQL in the ledger |
| HLT-025-RELEASE-READINESS-GAP | 1 | release-layout (1): Document the existing release runbook |
| HLT-026-COST-BUDGET-GAP | 1 | cost (1): Keep campaign budgets and CI cost proof separate |
| HLT-027-HUMAN-REVIEW-EVIDENCE-GAP | 1 | io-comment (1): Reject a file-reader comment as a review attestation |
| HLT-029-RUST-BAD-BEHAVIOR | 347 | channel (3): Defer bounded-queue redesign; raw (2): Review raw-view lifetime and bounds separately; send (2): Retain thread-safety review as a release limitation; unsafe (335): Defer complete unsafe-site qualification; unsafe-string (2): Reject an unsafe-code detector false positive; zero-mask (2): Document valid Linux affinity-mask initialization; zero-signal (1): Retain signal-structure proof debt |
| HLT-033-PYTHON-BAD-BEHAVIOR | 10 | checksum (1): Separate checksum verification from TLS; pickle (5): Restrict the claim to trusted executable artifacts; pickle-test (2): Preserve executable-artifact tests; pinned-exec (1): Retain the authenticated-code trust boundary; sql-fixture (1): Document parameterized test SQL |
| HLT-034-CI-BAD-BEHAVIOR | 9 | ci-concurrency (4): Defer workflow scheduling changes; ci-metadata (4): Reject a metadata-as-secret detector mismatch; ci-selfhost (1): Record runner isolation as a separate gate |
| HLT-035-GIT-BAD-BEHAVIOR | 5 | git-fixture (3): Preserve isolated Git fixture setup; git-help (2): Keep conditional recovery instructions |
| HLT-037-RELEASE-BAD-BEHAVIOR | 1 | git-fixture (1): Preserve isolated Git fixture setup |
| HLT-038-REFERENCE-PROFILE-STRUCTURE-GAP | 4 | profile (4): Document the reference-profile exception |
| HLT-040-REPO-ROT-BAD-BEHAVIOR | 26 | rot-disabled (1): Remove the hard-disabled duplicate CV loop; rot-doc (1): Reject a design-note-as-dead-code mismatch; rot-version (24): Preserve versioned contracts and qualified reference binaries |
| HLT-042-CI-LOCAL-PARITY | 10 | ci-layout (10): Document actual versioned CI entrypoints |
| HLT-043-COPY-PASTE-BAD-BEHAVIOR | 12 | copy (12): Defer semantic consolidation |
| HLT-046-UNNECESSARY-VARIETY | 22 | enum (22): Keep distinct namespace-local domain types |
| HLT-047-CANONICAL-README | 1 | badge (1): Route live CI badge to the release owner |
| HLT-048-CANONICAL-CI-GAP | 2 | ci-audit (1): Record out-of-band audit without a passing gate claim; ci-layout (1): Document actual versioned CI entrypoints |

Total: **544/544 dispositioned; 543 documented with rationale, 1 assigned to [the focused hard-disabled-CV fix PR #233](https://github.com/neverhuman/JopeDime/pull/233)**. Fix publication, exact-head hosted CI, independent approval and merge are recorded separately; a local correction is not a merged fix.

## Rationale details

### profile — 8 findings

**Document the reference-profile exception.** JopeDime is a Rust GP/ML and fitted-export repository. The auditor assumes a Rust/React/PostgreSQL application and canonical ops/ and python-ai/ folders. Keep existing supported languages and deployed paths; redesigning them during a release freeze requires a separate migration and qualification. This does not establish conformance to that reference stack.

Owner: Release owners. Closure: Verify actual boundaries and qualify a versioned repository-specific audit profile; do not lower the present score floor.

### shape — 1 findings

**Defer module-size restructuring.** The scorer reports large authored modules. Their size is maintenance debt; a split is not a safe name-only fix because visibility and qualification surfaces can change. Retain the finding and plan subsystem-specific refactors after the release freeze.

Owner: Core maintainers. Closure: Split one subsystem at a time with preserved tests and behavior.

### marker — 72 findings

**Retain lexical warnings with their exact locations.** A flagged word is not enough to prove unfinished behavior. Work-list variables named todo, stale-input refusals, temporary-file descriptions and historical identity names are valid uses. Each location is retained; unresolved behavioral TODOs need owner review rather than a bulk rename that improves the score without fixing behavior.

Owner: Respective source owner. Closure: Inspect the bound source location before changing behavior; preserve pinned source and historical records.

### unsupported — 1 findings

**Preserve explicit unsupported-state refusal.** genome.rs rejects the unsupported caruana blend rather than executing an incomplete implementation. The regression release does not claim this operator. Removing the refusal would weaken the input boundary.

Owner: GP owner. Closure: Implement and qualify the operator under a separate supported-version contract if it is needed.

### fallback — 1 findings

**Keep bounded adapter failures visible.** The benchmark adapter catches an exception to produce a failure result. An exception-handler marker alone does not establish successful fallback computation. Full failure-accounting and permitted-input qualification remain required; this release note is not proof of those operational gates.

Owner: Benchmark owner. Closure: Replay adapter failures and verify their terminal accounting before expanding its qualified scope.

### unsafe — 335 findings

**Defer complete unsafe-site qualification.** The audit lacks a nearby local safety argument at this source location. That is a real documentation/proof gap, not proof that the code is sound or unsound. A complete lifetime, bounds, alignment, ownership and synchronization review is larger than a cheap release fix. Preserve this unresolved risk; passing CPU tests or adding a comment alone will not close it.

Owner: Respective Rust subsystem owner. Closure: Review the exact operation and its callers; add a correct safety argument and adversarial receiving-path tests. No blanket unsafe allowance.

### unsafe-string — 2 findings

**Reject an unsafe-code detector false positive.** The match is the word unsafe followed by an interpolated field inside a refusal message, not a Rust unsafe expression. The receiving code rejects unsafe path components. Keep the refusal and document this lexical mismatch.

Owner: Learning owner. Closure: Use syntax-aware detection; preserve path-refusal tests.

### zero-mask — 2 findings

**Document valid Linux affinity-mask initialization.** core.rs initializes libc::cpu_set_t, an integer bit-mask structure, to zero and bounds CPU indices before access. The source has a local SAFETY argument. This specific type does not fabricate a Rust reference or invalid enum; do not generalize that reasoning to arbitrary zeroed types.

Owner: Campaign owner. Closure: Retain Linux type/bounds proof and affinity tests; re-evaluate on platform changes.

### zero-signal — 1 findings

**Retain signal-structure proof debt.** gp.rs zero-initializes libc::sigaction before assigning its handler/mask for a Linux syscall. This is not a general authorization for mem::zeroed. A platform-specific validity and signal-handler audit remains required.

Owner: Engine owner. Closure: Review all fields, signal safety, process-group cleanup and qualified platform assumptions.

### raw — 2 findings

**Review raw-view lifetime and bounds separately.** The cited call constructs a read-only slice. corpus.rs now checks mapping capacity and record bounds (#229); training.rs describes the live f32 byte view. Neither is an ownership transfer into an allocator. The detector wording is broader than the operation, but full lifetime/alignment proof remains the subsystem owner's responsibility.

Owner: Engine owner. Closure: Retain exact mapping bounds and byte-hash parity tests; review destruction and aliasing.

### send — 2 findings

**Retain thread-safety review as a release limitation.** An unsafe Send implementation needs proof about the library/device handle and every consumer. Nearby mutex comments are not a complete CUDA context/thread-affinity audit. Lane C owns the current GPU shape-bounds fix; ROOT does not overlap it or claim GPU safety qualification.

Owner: GPU owner / Lane C. Closure: Review handle ownership, locking and CUDA context assumptions with actual runtime qualification.

### channel — 3 findings

**Defer bounded-queue redesign.** The unbounded channel is a real resource/backpressure concern. Changing capacity during a release freeze can change scheduling, failure handling and deadlock behavior. Keep it visible; this release does not establish a bounded production campaign merely because a command exists.

Owner: Campaign owner. Closure: Freeze queue limits; test blocked consumers, cancellation, exhaustion and durable failure accounting.

### pickle — 5 findings

**Restrict the claim to trusted executable artifacts.** pickle/joblib deserialization can execute code. A caller-supplied SHA-256 is an integrity comparison, not producer authentication or a sandbox. These paths cannot be represented as safe loaders for arbitrary input. Preserve this limitation until authenticated production provenance and isolation, or a data-only format, are qualified.

Owner: Python/benchmark owner. Closure: Reject untrusted artifacts through the receiving boundary; qualify authenticated producer identity and execution isolation.

### pickle-test — 2 findings

**Preserve executable-artifact tests.** The match is in adapter tests using fitted artifact bytes. Test-local construction does not make the product loader safe for untrusted bytes. Keep the tests and retain the product limitation separately.

Owner: Benchmark owner. Closure: Keep positive and malicious-artifact refusal tests; do not treat digest parity as authentication.

### sql-fixture — 1 findings

**Document parameterized test SQL.** The flagged test_family_pin.py call has question-mark placeholders with a separately supplied parameter tuple. It builds a negative registry fixture in a temporary local database, not SQL by concatenating untrusted data.

Owner: DL lane. Closure: Preserve read-only-registry refusal tests and parameter binding.

### checksum — 1 findings

**Separate checksum verification from TLS.** TokenizedCorpus(..., verify=False) is a corpus checksum-loading option, not an HTTP certificate-verification switch. Parent corpus binding and process-spawn reload checks must be reviewed separately; this note does not qualify all cached loading paths.

Owner: Python trainer owner. Closure: Verify the admitted corpus and worker reload closure; keep TLS verification enabled in network clients.

### pinned-exec — 1 findings

**Retain the authenticated-code trust boundary.** v2_rowsel_build.py reads the GSK port with read_verified(GSK, GSK_SHA256) before compile/exec. It deliberately executes pinned code rather than arbitrary caller-provided source. The pin/provenance and filesystem custody remain trusted prerequisites, and this is not a general safe eval API.

Owner: DL lane. Closure: Verify producer authority, exact consumed bytes and source pin before execution; qualify a typed module replacement separately.

### ci-selfhost — 1 findings

**Record runner isolation as a separate gate.** A self-hosted label is not evidence of a Docker socket mount. fast-lane.sh explicitly says the CI image has no Docker socket; workflow permissions are read-only and checkout credentials are not persisted. Actual launcher isolation and hostile-PR qualification remain runner-owner gates. ROOT will not change ENG:JAILGUN runner controls.

Owner: Runner owner / ENG:JAILGUN. Closure: Inspect the current launch boundary and retain independent hostile-job isolation evidence.

### ci-concurrency — 4 findings

**Defer workflow scheduling changes.** These workflows have no workflow-level concurrency group. Cancelling main evidence or release work can destroy required coverage, so copying the PR cancel-in-progress policy is not a safe cheap fix. The dispatch-only release workflow is not the host release procedure.

Owner: CI/release owners. Closure: Design per-workflow serialization/cancellation with exact-head evidence retention; qualify under existing reservations.

### ci-metadata — 4 findings

**Reject a metadata-as-secret detector mismatch.** The cited expressions record builder_image/max_glibc or pass AWS_REGION. Those are runtime/build metadata, not credential file paths. This finding does not verify that every workflow artifact is credential-free; separate secret and runner guards still apply.

Owner: CI/release owners. Closure: Keep credential bytes out of artifacts and independently qualify publication boundaries.

### git-help — 2 findings

**Keep conditional recovery instructions.** release.sh prints tag-delete guidance for a locally created unpublished tag, or only after checking it is absent remotely. These cited lines are heredoc help text, not an executed destructive command. Remote state still must be checked by the operator.

Owner: Release owner. Closure: Retain fail-closed tag/publication tests and immutable published-tag policy.

### git-fixture — 4 findings

**Preserve isolated Git fixture setup.** test-release.sh stages a complete constructed fixture and mutates a tag in its local bare test remote. This does not authorize staging an agent checkout wholesale or changing a public release tag.

Owner: Release owner. Closure: Keep test-root confinement and real-publish refusal tests.

### rot-version — 24 findings

**Preserve versioned contracts and qualified reference binaries.** The path contains an explicit algorithm/schema/release version. Renaming it merely to silence a pattern would break source pins, replay commands or persisted identities. Keep it until a reviewed migration retires its consumers.

Owner: Respective versioned subsystem owner. Closure: Inventory consumers and preserve source/evidence before a version migration.

### rot-doc — 1 findings

**Reject a design-note-as-dead-code mismatch.** binary_cv.rs contains prose explaining the schema-bound public fit path and class-stratified folds. The flagged comment is not executable parked code. Classification remains outside the initial regression remediation scope.

Owner: Campaign owner. Closure: Retain the explanatory contract; qualify classification separately.

### rot-disabled — 1 findings

**Remove the hard-disabled duplicate CV loop.** final_cv contains an if false block and an unreachable-code allowance after the active pooled/nonpooled evaluators. Removing only that unreachable block preserves the active scores and fit allowance while eliminating misleading parked implementation.

Owner: ROOT. Closure: Focused HLT-040 PR; existing 14 strategy-screen tests, formatting and detector replay. Exact-head CI/review still required.

### copy — 12 findings

**Defer semantic consolidation.** The detector finds repeated grammar/GP units, pinned Lambda snapshots and empty Python package markers. Consolidation requires dependency/version and behavior review; empty package initializers are not duplicate algorithm implementations. Do not remove pinned snapshots or merge semantic variants solely to raise the score.

Owner: Engine/Lambda/DL owners. Closure: Classify each duplicated unit, prove semantic equivalence, then update source pins and differential tests together.

### enum — 22 findings

**Keep distinct namespace-local domain types.** Same short enum names across crate/module namespaces do not establish interchangeable meanings. A shared enum migration needs field/variant and persisted-schema proof. Width alone or spelling alone is not identity.

Owner: Respective type owners. Closure: Review each cited pair; consolidate only semantically identical contracts with compatibility tests.

### badge — 1 findings

**Route live CI badge to the release owner.** The README lacks a health badge. Lane B owns the release README badge/version/GP changes. Record this pending presentation fix; do not publish a green audit badge for a score of 47 or treat a badge as current execution evidence.

Owner: Lane B. Closure: Release PR adds a CI link/badge, retains source/head-scoped evidence and all failures.

### ci-layout — 11 findings

**Document actual versioned CI entrypoints.** The reference profile requires ops/ci and scripts/ci-* paths. This repository uses deploy/ci/fast-lane.sh, deploy/ci/check.sh, .githooks and just required; workflow jobs wrap those commands in receipts. Keep the path mismatch visible without claiming the local required gate covers every hosted lane.

Owner: CI owners. Closure: Verify actual local/hosted command parity and per-lane receipts; no directory-name-only rewrite.

### ci-audit — 2 findings

**Record out-of-band audit without a passing gate claim.** There is no Jankurai CI gate and the retained full audit fails 85. The owner explicitly permits a fully documented triage for the release criterion. Do not add an always-green wrapper, score exclusion or lower threshold to manufacture conformance.

Owner: Release/CI owners. Closure: Qualify a reproducible audit lane/profile separately; preserve every hard finding and the failing score.

### security-lane — 3 findings

**Separate actual security checks from reference names.** deploy/ci/fast-lane.sh invokes cargo-deny, and deploy/ci/check.sh invokes the secret-scan wrapper and the private-key-text guard. The detector expects different workflow names. These source hooks do not establish current execution, complete supply-chain coverage or hardened runner isolation.

Owner: Security/CI owners. Closure: Read exact-head security receipts and maintain pinned dependency/secret checks; retain uncovered scope.

### secret-pattern — 1 findings

**Redact a scanner-policy token without claiming secret-free history.** The matched line is the gitleaks allowlist paths field containing a synthetic fixture/private-key-armor pattern, not a verified live credential. Default gitleaks rules remain enabled. The broad exception policy still needs independent review. Raw secret-like evidence is not copied into this public register.

Owner: Security owner. Closure: Review each allowlist scope and run history/tree scans; rotate only independently confirmed live credentials.

### fixture-freeze — 1 findings

**Preserve immutable historical compatibility bytes.** The match is inside a historical content-addressed emitted-source fixture, not a changed current test assertion. Rewriting it would destroy the baseline. Current replay tests and approved transition records are required separately; the fixture alone is not proof.

Owner: GP/release owners. Closure: Retain exact historical bytes and require maintained current replay/parity tests.

### io-comment — 1 findings

**Reject a file-reader comment as a review attestation.** io.rs describes accepting plain/gzip/zstd inputs and fixture compression. It does not certify independent human review or a numerical result. Keep the actual PR review and verification records separate.

Owner: Arena owner. Closure: Retain decompression fixtures; bind any future review claim to a named review and full source SHA.

### sql-product — 1 findings

**Preserve parameterized SQL in the ledger.** The cited statement uses task=?, dataset_hash=?, semantic_sha256=? and protocol_sha256=? placeholders with bound parameters. Adjacent transactional recheck and insertion stay intact. The token match does not establish an unvalidated input sink.

Owner: Campaign ledger owner. Closure: Retain race/idempotency and malicious-value tests without string interpolation.

### nats — 1 findings

**Reject logarithmic-unit text as a streaming client.** decode_outcomes.rs:8 uses nats for logit spacing in a fixed-logit sampler fixture. It does not instantiate a NATS client or expose an unbounded streaming adapter.

Owner: Policy owner. Closure: Retain sampler refusal/exhaustion fixtures; improve syntax/context-aware detection.

### release-layout — 1 findings

**Document the existing release runbook.** The expected docs/release.md does not exist; docs/release/README.md contains the host release procedure and its limitations. Do not confuse documented source procedure with a completed dry run, signed package or tag.

Owner: Release owner. Closure: Retain the runbook and qualify actual release evidence before publishing.

### cost — 1 findings

**Keep campaign budgets and CI cost proof separate.** docs/testing.md documents test families; compute studies have separate immutable admissions and deadlines. A documentation score does not prove spend control. No training/GPU/storage grant or renewed budget is inferred by this audit.

Owner: Campaign/CI owners. Closure: Bind each executed campaign to its original resource ceiling, deadline and complete attempt accounting.

### observability — 1 findings

**Retain partial observability coverage.** The dimension score is below 85. CI receipt/result machinery and testing docs exist, but the score is not an operational recovery or complete logging proof. Missing runtime evidence remains open.

Owner: CI/core owners. Closure: Qualify failure injection and reconstruction using source-bound execution receipts.

### context — 2 findings

**Retain context/ownership dimension debt.** AGENTS.md and ownership records exist, but navigation/context dimensions are below the reference floor. No source file or ownership map by itself closes the operational contracts.

Owner: Release/core owners. Closure: Keep synchronized claims and tighten subsystem navigation with owner-reviewed documentation.

### boundary — 2 findings

**Retain unqualified boundary/provenance scope.** The reference data/contract dimension is below 85. Actual schema, fit-authority and provenance controls must be evaluated at their receiving paths. This triage does not establish all W1-W12 gates, live training, S3 restoration or serving qualification.

Owner: Core/data owners. Closure: Continue receiving-boundary, ancestry, forbidden-input and independent replay qualification; report each evidence scope separately.

## Verification and release integration

[PR #233](https://github.com/neverhuman/JopeDime/pull/233) has frozen head `ac631826d1f0db51d75409337a58c0331efb4ce6`. The 27-line disabled loop is the sole deletion; active code and all maintained test bytes are unchanged. All 14 strategy-screen tests passed again at that exact head after main integration. Hosted CI, independent review and merge remain separate pending gates.

`just required` on unchanged main failed with exit 101 at `crates/jope-campaigns/src/bin/jope_task_learning_controller.rs:3982`: `clippy::field_reassign_with_default` in a test initializer. Hygiene and formatting completed; PR-tier Rust tests and Python checks were not reached. A second unchanged-main Clippy replay also exposes six needless borrows in `jope-bench/src/panel/inference.rs`, a useless conversion in `jope-learning/src/eligibility.rs`, type complexity in `jope-learning/src/state.rs`, and two cloned-reference-to-slice warnings in `jope-learning/src/progress/tests/reconcile.rs`. This is a release blocker separate from Jankurai triage, and the precise repro/logs were posted to the owning lanes. Do not call the full required gate passed.

The release owner must carry a link to this document and its limitations into CHANGELOG `[1.0.0]`, which supplies the shipping release notes. Until the audit documentation and focused fix pass review and that release-note integration lands, criterion 4 remains pending publication; this file does not authorize tagging or promotion.
