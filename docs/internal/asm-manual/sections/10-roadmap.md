# Prioritized acceptance checklist and roadmap

## Purpose and scope

This checklist is the requested property-list of remaining work. `roadmap.json` is its machine-readable source, with stable IDs, priorities, evidence levels, dependencies, commands, required receipts, budgets and source references. It describes source68f56404 and the stopped training64/product28/warm/winner follow-ons. PR83 is merged: remote main `a75e543d2166017cb0139e0a251d51d9a0c98e08` has the reviewed source tree, and main-guard35889731910 attempt1 succeeded. Clean canonical main was verified before the documentation branch. LIFE-01 records the completed readback; later source work still needs its own lifecycle.

P0 blocks the specified downstream proof or integration; it does not mean every item is a newly discovered defect. P1 is required for the full requested demonstration or manual. P2 is explicit subsequent work. Commands are unexecuted acceptance templates unless their named evidence says otherwise. Variables must resolve to actual final-source artifacts and emitted identities. An empty command list records a missing truthful interface rather than inventing one.

## Critical order

Verified reviewed main → guarded source integration → actual tests, host qualification and final packaging → fresh qualified contexts and preregistered real teachers → all-eligible family-disjoint corpus → one GPU-hour training with continuation → explicit new-checkpoint consumption → real fitted warm demonstration → unchanged thirty-dataset campaign, selected-state backend qualification inside its remaining window, and report → eligible motif reuse.

Library parity fixtures precede the full panel. Actual selected-winner exports require complete confirmations and must still fit the original execution window. They cannot consume its reporting reserve or silently run after expiry. Documentation can proceed alongside source preparation; actual numerical examples need an allocated execution window.

Do not spend the GPU hour until the regression consumer and its actual fixture tests are ready. Do not freeze the final official panel configuration until its checkpoint is chosen and consumed. Separate proof allowances suggested in older plans are proposals, not new budget authorization.

## Compact task index

| ID | Priority | Stage | Workstream | Current state | Dependencies |
|---|---|---:|---|---|---|
| LIFE-01 | P0 | 0 | coordination | complete with retained readback | None |
| DOC-01 | P1 | 0 | manual | canonical private manual assembled | LIFE-01 |
| DOC-02 | P1 | 0 | manual examples | complete at named recipe/refit scope | DOC-01 |
| DOC-03 | P1 | 0 | manual figures | figures rendered; final PDF inspection pending | DOC-01 |
| DOC-04 | P1 | 0 | manual quality | rendered; final publication gates pending | DOC-01, DOC-02, DOC-03 |
| INT-01 | P0 | 0 | serial integration | stopped unmerged packets | LIFE-01 |
| INT-02 | P0 | 0 | integration qualification | pending | INT-01 |
| BENCH-01 | P1 | 1 | frozen benchmark foundation | implemented; retain during integration | INT-02 |
| BENCH-02 | P1 | 1 | external comparators | refresh at final environment | INT-02 |
| EXEC-01 | P1 | 2 | bounded workers | implemented; CI coverage gap retained | INT-02 |
| EXEC-02 | P1 | 2 | fault regression tests | test gap: drain-and-fixtures-F2 | INT-01 |
| EXEC-03 | P1 | 2 | receipt replay regression tests | test gap: resume-fix-F2 | INT-01 |
| EXEC-04 | P1 | 2 | cutoff semantics regression | test gap: regressions-F2 | INT-01 |
| EXEC-05 | P1 | 2 | accounting and unknown outcomes | pending execution | INT-02 |
| NEURAL-01 | P0 | 3 | context qualification | requires final-source binding | INT-02 |
| TEACH-01 | P0 | 6 | HWM comparison identity | not runtime qualified | INT-02 |
| TEACH-02 | P0 | 6 | label exposure readers | residual eligibility-B6 | TEACH-01 |
| TEACH-03 | P0 | 3 | real teacher collection | blocked on fresh admitted evidence | BENCH-01, BENCH-02, NEURAL-01, TEACH-01, TEACH-02 |
| TEACH-04 | P0 | 3 | real corpus materialization | blocked on two real families | TEACH-03 |
| TRAIN-01 | P0 | 3 | actual trainer tests | residual training-F3 | INT-02 |
| TRAIN-02 | P0 | 3 | native supervised training | host proof pending | TRAIN-01 |
| TRAIN-03 | P0 | 3 | bounded real GPU training | blocked on admitted data and consumption readiness | TEACH-04, TRAIN-02, NEURAL-02 |
| TRAIN-04 | P0 | 3 | checkpoint continuation and export | requires actual campaign | TRAIN-03 |
| NEURAL-02 | P0 | 3 | trained-checkpoint consumer readiness | compile/runtime fixture pending | INT-02, TRAIN-01 |
| NEURAL-03 | P0 | 3 | live newly trained consumption | newly trained consumption pending | TRAIN-04, NEURAL-01 |
| INFER-01 | P0 | 4 | fitted state and privacy | qualification pending | INT-02 |
| INFER-02 | P1 | 4 | exact selected-winner portability | actual export/parity unrun | INFER-01, PANEL-01 |
| INSTALL-01 | P1 | 5 | private installation | cold/live path pending | INT-02 |
| INSTALL-02 | P1 | 5 | final input/runtime preparation | pending final source | INSTALL-01, BENCH-01, BENCH-02 |
| DEMO-01 | P1 | 5 | measured warm fitted demonstration | realtrained warm proof pending | NEURAL-03, INSTALL-02 |
| PANEL-01 | P1 | 5 | unchanged thirty-dataset campaign | pending finaltrained configuration | BENCH-01, BENCH-02, NEURAL-03, DEMO-01, INFER-01, INSTALL-02 |
| REPORT-01 | P1 | 5 | complete measured report | fullcampaign report pending | PANEL-01, EXEC-05 |
| MOTIF-01 | P1 | 6 | eligible reusable fragments | minefront door incomplete | TEACH-01, TEACH-02, TEACH-03 |
| MOTIF-02 | P1 | 6 | reuse validation | live reuse pending | MOTIF-01 |
| MOTIF-03 | P2 | 6 | new operators | future conditional work | MOTIF-02 |
| IP-01 | P1 | 0 | public privacy boundaries | residual D-IP4 and nextPRtextwork | INT-01 |
| MAINT-01 | P2 | 0 | broad diagnostic qualification | not closedbyfastCI | LIFE-01 |
| FUTURE-01 | P2 | 4 | CUDA inference and serving promotion | deferred | INFER-02, REPORT-01 |


## Detailed acceptance


### LIFE-01 — coordination

**Priority:** P0. **Current evidence:** reviewed merge, exact-tree remote main and successful main-guard. **Status:** complete with retained readback.

**Action:** Retain the verified lifecycle receipt; future source changes repeat exact-head CI and eligible independent native review.

**Dependencies:** None.

**Acceptance commands:**

```sh
/home/ubuntu/.local/bin/gh-role writer pr view 83 --json mergedAt,mergeCommit,headRefOid,state
/home/ubuntu/.local/bin/gh-role writer run view 35889731910 --json headSha,attempt,status,conclusion,jobs
```

**Required evidence:** PR83 merged2026-09-23T16:34:47Z; remote main a75e543d2166017cb0139e0a251d51d9a0c98e08 has tree176b75bc36d76c7ce4eaeb7234677ec013468889, equal to reviewed68. Main-guard35889731910 attempt1 succeeded; job107278860508 finished16:40:11Z. Clean canonical main verified before docs/asm-maintainer-manual branch claim.

**Resources:** Read-only/static custody work; no fitting, models, GPU or campaign admission.

**Source/evidence references:** `AGENTS.md:Pull requests`, `status.md:Scope`, `maintenance.md:Coverage and reviewed lifecycle`, `neural-integration/lifecycle-68f56404c3e199e06413bfd645d3fe0803d7b96f/remote-main-verification.json`.

**Qualification limit:** This task is complete; subsequent implementation remains pending. Keep the measured68 source identity distinct from the rebase-merge commit identity.


### DOC-01 — manual

**Priority:** P1. **Current evidence:** assembled thirteen chapters and complete data companions. **Status:** canonical private manual assembled.

**Action:** Maintain the dated evidence inventory and current source/pending distinctions.

**Dependencies:** LIFE-01.

**Acceptance commands:**

```sh
python3 docs/internal/asm-manual/tools/build_manual.py
```

**Required evidence:** Input manifest pins every chapter, figure dataset, source receipt and render tool; duplicates resolved; current/pending/historical/unknown labels retained.

**Resources:** Read-only/static custody work; no fitting, models, GPU or campaign admission.

**Source/evidence references:** `architecture-training-evidence.json`, `product-language-source-pins.json`, `dataset-inventory.json`, `performance-evidence.json`.

**Qualification limit:** Root integrated the book and renderer. Claude historical plot inputs were copied under stable before/after hashes; no stopped external renderer handoff is required.


### DOC-02 — manual examples

**Priority:** P1. **Current evidence:** fourteen actual commands passed; five backend predictions bit-identical. **Status:** complete at named recipe/refit scope.

**Action:** Retain the completed fourteen-command receipt and rerun after meaningful recipe/runtime changes; qualify saved fitted winners separately.

**Dependencies:** DOC-01.

**Acceptance commands:**

```sh
python3 "$EXAMPLES/validate-outputs.py" --recipe-output "$EXAMPLE_RUN/recipe"
"$J" pipeline verify-receipt --receipt "$EXAMPLE_RUN/pipeline/receipt.json" --json
```

**Required evidence:** examples/commands.json instantiated with actual paths and pinned source/tool versions; five outputs with8finite values and complete-bit comparison; pipeline receipt verification. Stopped audit SHA256 363fbe62a706b44030205da78411243ba206ad2366d546a277babe966b802c91.

**Resources:** Completed separate 300-second allocation: one CPU, 4 GiB, zero swap/GPU, network denial declared; 30-second numerical and 60-second compiler caps. Actual receipt retained; independent live property readback limitation is disclosed. No Cargo or frozen-panel campaign; no deadline reset.

**Source/evidence references:** `examples/commands.json`, `examples/validate-outputs.py`, `languages-examples.md:Example A`, `languages-examples.md:Example B`.

**Qualification limit:** Actual acceptance: data/example-acceptance/audit.json. Five recipe refits each predict eight synthetic rows; separate legacy pmlb229 pipeline has four successful measurements. No fitted-export, ASM demo, training or full-panel qualification. Resource-readback race retained.


### DOC-03 — manual figures

**Priority:** P1. **Current evidence:** ten local figure artifacts generated and source-checked. **Status:** figures rendered; final PDF inspection pending.

**Action:** Retain historical/current labels, complete source CSVs and final PDF visual checks.

**Dependencies:** DOC-01.

**Acceptance commands:**

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 docs/internal/asm-manual/tools/make_inventory_figures.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 docs/internal/asm-manual/tools/make_figures.py
```

**Required evidence:** Plot source CSV/JSON, tool source, plot hash, precise captions, missing-stat flags and historical-vs-current labels; no invented sealed statistics or causal speedup.

**Resources:** Read-only/static custody work; no fitting, models, GPU or campaign admission.

**Source/evidence references:** `data-evidence.md`, `dataset-inventory.json`, `performance-evidence.json`.

**Qualification limit:** Historical Elo belongs to its original ladder/pool/protocol; the current one-dataset proof is not a new Elo ladder. Source and figure checks are retained; authored architecture SVG is a repository source artifact.


### DOC-04 — manual quality

**Priority:** P1. **Current evidence:** thirteen chapters rendered; full-text and sampled-page checks available. **Status:** rendered; final publication gates pending.

**Action:** Render docs/*.pdf, extract PDF text, inspect all dense tables/code/figures, verify local links and filenames, and produce a readback checklist.

**Dependencies:** DOC-01, DOC-02, DOC-03.

**Acceptance commands:**

```sh
pdftotext -layout "$MANUAL_PDF" "$MANUAL_TEXT"
```

**Required evidence:** PDF SHA/page count; extracted-text receipt; visual inspections of title/contents, widest tables, commands, plots and final checklist; no clipping/omitted rows; Markdown counterparts present.

**Resources:** Read-only/static custody work; no fitting, models, GPU or campaign admission.

**Source/evidence references:** `maintenance.md:Updating the book`.

**Qualification limit:** A successful renderer exit is insufficient: the PDF must remain readable and complete.


### INT-01 — serial integration

**Priority:** P0. **Current evidence:** two exact custody replays + guarded hashes. **Status:** stopped unmerged packets.

**Action:** Refresh guards only after verified main; integrate training64 then product28 then warm/winner add-ons without replacing newer source corrections.

**Dependencies:** LIFE-01.

**Acceptance commands:**

```sh
python3 "$TRAINING_PACKET/replay.py" --expect-head "$SOURCE_SHA" --manifest-sha256 "$TRAINING_MANIFEST_SHA" --output "$TRAINING_PACKET/replay-NEW"
python3 "$PRODUCT_PACKET/replay.py" --expect-head "$SOURCE_SHA" --manifest-sha256 "$PRODUCT_MANIFEST_SHA" --output "$PRODUCT_PACKET/replay-NEW"
```

**Required evidence:** Exact-source guard receipts; unchanged original packet hashes/skips/resolutions; new applied diff; root scorer digest/cutoff/terminal persistence and current ledger hooks preserved.

**Resources:** Read-only/static custody work; no fitting, models, GPU or campaign admission.

**Source/evidence references:** `agent-training-integration/refresh-68f56404/HANDOFF.json`, `agent-product-final-68f56404/HANDOFF.json`, `agent-delivery-refresh-68f56404/HANDOFF.json`.

**Qualification limit:** NEW is an unused custody directory name. Add-on replay invocation must follow its own pinned helper. Never force stale guards or apply constituent accounting/coverage twice.


### INT-02 — integration qualification

**Priority:** P0. **Current evidence:** static syntax/feature checks; no integrated compilation yet. **Status:** pending.

**Action:** Compile/type-check the actual combined source, format, run meaningful suites and strict Clippy, regenerate coverage/inventories through owners, then obtain exact-head CI/native approval.

**Dependencies:** INT-01.

**Acceptance commands:**

```sh
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-bench --features asm
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-cli --features asm
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-pipeline --features qualified-context
cargo fmt --all --check
python3 deploy/ci/coverage.py discover
python3 deploy/ci/coverage.py validate
python3 deploy/ci/test-training-proof.py
TMPDIR="$ACCEPTANCE/tmp" JOPE_BUILD_JOBS=4 bash deploy/ci/test-inventory.sh --update
```

**Required evidence:** Collection/exit logs, actual tests passed/failed/ignored/deselected, refreshed inventory diff, feature/dependency graph, build executable hashes, current-attemptCI and independent exact-head approval.

**Resources:** Existing exclusive CPU CI allocation; one reused lane target, jobs4 unless lane policy sets otherwise; no concurrent numerical campaign.

**Source/evidence references:** `deploy/ci/coverage.json`, `deploy/ci/training-proof.py (pending)`, `deploy/ci/fast-lane.sh`, `deploy/ci/full-lane.sh`.

**Qualification limit:** Custody syntax tests are not this gate. Required18Python/27Rust/10policy names are proof minima, not total coverage. New code needs actual test collection.


### BENCH-01 — frozen benchmark foundation

**Priority:** P1. **Current evidence:** current source + local/exact-head tests. **Status:** implemented; retain during integration.

**Action:** Reverify unchanged manifest and all prepared inputs, actual row/group assignments, original category/missing/label semantics, loss directions, full class probability ordering and task method inventory.

**Dependencies:** INT-02.

**Acceptance commands:**

```sh
"$J" benchmark verify-manifest --manifest "$M"
"$J" benchmark verify-inputs --manifest "$M" --datasets "$D"
"$J" benchmark methods
```

**Required evidence:** Original panel SHAba7dace4…,30datasets10/task, exact input-byte verification, native17/5/4recipes and4external families; alias/configuration inventory and unavailable reasons.

**Resources:** Read-only/static custody work; no fitting, models, GPU or campaign admission.

**Source/evidence references:** `crates/jope-bench/src/panel/data.rs`, `crates/jope-bench/src/panel/methods.rs`, `crates/jope-bench/src/panel/score.rs`, `benchmarks/panel.json`.

**Qualification limit:** Verification is metadata/data validation, no new fits. Do not replace a failed panel dataset.


### BENCH-02 — external comparators

**Priority:** P1. **Current evidence:** four adapters implemented; earlier runtime inventories. **Status:** refresh at final environment.

**Action:** Pin all runtime/package/adapter bytes and actual constructors; execute relevant adapter contract tests and preserve unavailable dependencies.

**Dependencies:** INT-02.

**Acceptance commands:**

```sh
python3 deploy/prepare-panel-runtime.py --binary "$J" --native-command benchmark --build-receipt "$CLI_BUILD_RECEIPT" --source-sha "$SOURCE_SHA" --python "$EXTERNAL_PYTHON" --site-packages "$EXTERNAL_SITE_PACKAGES" --adapters crates/jope-bench/adapters --output "$ACCEPTANCE/runtimes.json"
"$EXTERNAL_PYTHON" -m pytest crates/jope-bench/adapters/test_panel.py -q
```

**Required evidence:** Current4engine version/source/package hashes, constructors for3tasks, actual contract-test receipt and explicit unavailability inventory.

**Resources:** Preparation owns sharedCPUallocation; tests use allocatedCPU; no panel/training window begins.

**Source/evidence references:** `deploy/prepare-panel-runtime.py`, `crates/jope-bench/adapters/_panel.py`, `crates/jope-bench/adapters/test_panel.py`.

**Qualification limit:** Do not nest an outer allocation lock around a front door already taking it. Environment mutation invalidates old inventory.


### EXEC-01 — bounded workers

**Priority:** P1. **Current evidence:** ten real synthetic host cases passed at 68. **Status:** implemented; CI coverage gap retained.

**Action:** Requalify final worker/controller bytes through ten current host cases and make required-lane coverage of critical async cleanup paths explicit.

**Dependencies:** INT-02.

**Acceptance commands:**

```sh
python3 deploy/ci/qualify-execution.py --test-binary "$PIPELINE_TEST_BINARY" --output "$ACCEPTANCE/host-execution"
```

**Required evidence:** Ten nonzero actual cases,20concurrentworkers,limit readbacks, process-group drainage, setup failures/cancel gates, original-window equality and fixture code reachability.

**Resources:** Existing exclusive CPU CI allocation; one reused lane target, jobs4 unless lane policy sets otherwise; no concurrent numerical campaign.

**Source/evidence references:** `deploy/ci/qualify-execution.py`, `crates/jope-pipeline/src/execution.rs`.

**Qualification limit:** Current host qualification is not automatically covered by the ordinary required unit lane; missing privileges fail, not skip.


### EXEC-02 — fault regression tests

**Priority:** P1. **Current evidence:** production terminal-persistence fix qualified indirectly. **Status:** test gap: drain-and-fixtures-F2.

**Action:** Add a real async-reap fixture with pre-gate BudgetExhausted and an undrained group. Prove terminal record survives before cleanup and admission remains blocked until drain.

**Dependencies:** INT-01.

**Acceptance commands:**

```sh
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-pipeline --features qualified-context execution::tests
```

**Required evidence:** Reproduction fails when persist is moved back under drained branch or admission_blocked handling is removed; corrected actual test passes with cleanup ordering evidence.

**Resources:** Existing exclusive CPU CI allocation; one reused lane target, jobs4 unless lane policy sets otherwise; no concurrent numerical campaign.

**Source/evidence references:** `crates/jope-pipeline/src/execution.rs:812`, `AGENT_CHAT.md:2026-09-23T16:17 delta review`.

**Qualification limit:** Exact new test name is not present yet; existing module command is the truthful current gate. Ten old host cases do not refute this coverage gap.


### EXEC-03 — receipt replay regression tests

**Priority:** P1. **Current evidence:** shared per-fold helper fixed and real68 recovered report passed. **Status:** test gap: resume-fix-F2.

**Action:** Add PoolBackend-shaped candidate receipt roundtrip asserting live Feedback equals recovered FitAdapter Feedback, including digest and partial completed-fold evidence.

**Dependencies:** INT-01.

**Acceptance commands:**

```sh
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-bench --features asm
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-pipeline --features qualified-context neural_search::tests
```

**Required evidence:** Regression reproduces d0 mismatch; exercises Some(shared fold digest), changed completed outputs refuse, never-admitted normalization cannot gain evidence.

**Resources:** Existing exclusive CPU CI allocation; one reused lane target, jobs4 unless lane policy sets otherwise; no concurrent numerical campaign.

**Source/evidence references:** `crates/jope-pipeline/src/neural_search.rs:1735`, `crates/jope-bench/src/panel/optimize_fit.rs`, `AGENT_CHAT.md:resume-fix-F2`.

**Qualification limit:** The two original FixtureBackend reproductions use absent digests and do not cover the same_folds path.


### EXEC-04 — cutoff semantics regression

**Priority:** P1. **Current evidence:** production cutoff ordering corrected; existing test too weak. **Status:** test gap: regressions-F2.

**Action:** Strengthen cancellation-at-cutoff test to assert original_deadline stop reason and persisted receipt, not only is_ok.

**Dependencies:** INT-01.

**Acceptance commands:**

```sh
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-pipeline --features qualified-context cancellation_at_cutoff_preserves_observed_generation_on_resume
```

**Required evidence:** Test fails if cancellation check moves before deadline check; successful resume emits no new fit and leaves observed bytes unchanged.

**Resources:** Existing exclusive CPU CI allocation; one reused lane target, jobs4 unless lane policy sets otherwise; no concurrent numerical campaign.

**Source/evidence references:** `crates/jope-pipeline/src/neural_search.rs:1380`, `AGENT_CHAT.md:regressions-F2`.


### EXEC-05 — accounting and unknown outcomes

**Priority:** P1. **Current evidence:** additive accounting union source + four authored Rust cases. **Status:** pending execution.

**Action:** Execute started/finished-union accounting tests; retain started-only unknowns separately from terminal refusals and qualify report replay without child retries.

**Dependencies:** INT-02.

**Acceptance commands:**

```sh
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-bench --features asm asm_campaign
```

**Required evidence:** Actual four new accounting tests collected; started-only attempts counted unknown, no inferredzero, exports counted once inASMworker totals, recovered reporting no launch.

**Resources:** Existing exclusive CPU CI allocation; one reused lane target, jobs4 unless lane policy sets otherwise; no concurrent numerical campaign.

**Source/evidence references:** `agent-accounting-union/training-bridge/HANDOFF.json`, `crates/jope-bench/src/panel/asm_campaign.rs (pending)`.


### NEURAL-01 — context qualification

**Priority:** P0. **Current evidence:** eight fixed native cases passed at 68. **Status:** requires final-source binding.

**Action:** Build exact context/policy/proof workers; qualify unchanged native references for final source and preserve eight fixed-case plus actual dataset process evidence.

**Dependencies:** INT-02.

**Acceptance commands:**

```sh
"$CONTEXT_PROOF" prepare --sources "$REFERENCE_SOURCES" --sources-sha256 "$REFERENCE_SOURCES_SHA" --encoder-files "$ENCODER_FILES" --encoder-files-sha256 "$ENCODER_FILES_SHA" --worker "$CONTEXT_WORKER" --worker-sha256 "$CONTEXT_WORKER_SHA" --source-sha40 "$SOURCE_SHA" --resources "$CONTEXT_RESOURCES" --resources-sha256 "$CONTEXT_RESOURCES_SHA" --output "$ACCEPTANCE/context-proof"
"$CONTEXT_PROOF" run --plan "$CONTEXT_PLAN" --plan-sha256 "$CONTEXT_PLAN_SHA"
```

**Required evidence:** All eight fixed cases pass the unchanged component/context maximum absolute difference of0.0003; source, assets and program identities match. Preserve actual bounded execution receipts and the qualified construction token, original labels and row order.

**Resources:** Separate explicit resource manifest and approved allowance required before admission;20 CPUs/workers maximum,64 GiB maximum,zero swap,one thread,30 seconds per fit,120 seconds per arm maximum. No new allowance granted by this checklist.

**Source/evidence references:** `crates/jope-policy/src/context_producer/admission.rs:4`, `crates/jope-pipeline/src/cpu_context_bridge.rs`.

**Qualification limit:** Metadata/path equality or checksum alone is not numerical qualification; no changed tolerance or checkpoint substitution.


### TEACH-01 — HWM comparison identity

**Priority:** P0. **Current evidence:** pending full-identity source + test plan. **Status:** not runtime qualified.

**Action:** Apply and execute full comparison identity enforcement before feedback; ensure stale evaluator/vocab/preprocessing/protocol/class order cannot collide with an incumbent.

**Dependencies:** INT-02.

**Acceptance commands:**

```sh
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-learning
```

**Required evidence:** Changed identity negative cases, old record migration/refusal behavior, hash-chain integrity and ties/strict improvement semantics.

**Resources:** Existing exclusive CPU CI allocation; one reused lane target, jobs4 unless lane policy sets otherwise; no concurrent numerical campaign.

**Source/evidence references:** `crates/jope-learning/src/record.rs (pending)`, `crates/jope-learning/src/hwm.rs (pending)`, `crates/jope-learning/src/eligibility.rs (pending)`.


### TEACH-02 — label exposure readers

**Priority:** P0. **Current evidence:** burn writer and effective-role mapping prepared. **Status:** residual eligibility-B6.

**Action:** Wire durable label-burn checks into every path that may claim an untouched test score, including HWM ingest/freeze and report publication, or explicitly prohibit that claim.

**Dependencies:** TEACH-01.

**Acceptance commands:**

```sh
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-learning
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-bench --features asm
```

**Required evidence:** Production caller inventory for require_unburned_test_partition; an integration test proving a burned partition is refused by test-valid readers; attempted sealed teaching refused; adaptive development reporting remains truthful.

**Resources:** Existing exclusive CPU CI allocation; one reused lane target, jobs4 unless lane policy sets otherwise; no concurrent numerical campaign.

**Source/evidence references:** `crates/jope-learning/src/eligibility.rs:253 (pending)`, `AGENT_CHAT.md:eligibility-B6`, `crates/jope-bench/src/panel/selection_exposure.rs`.

**Qualification limit:** No current live leak was observed because no producer emits benchmark-valid panel records. A burn file writer alone is not downstream enforcement.


### TEACH-03 — real teacher collection

**Priority:** P0. **Current evidence:** zero eligible current teachers; two prospective evidenced families. **Status:** blocked on fresh admitted evidence.

**Action:** Preregister teaching before two separately bounded proof plans for machineCPU and geyser. Obtain actual finite OOF improvement and improvements on four of five folds, or retain the rejection.

**Dependencies:** BENCH-01, BENCH-02, NEURAL-01, TEACH-01, TEACH-02.

**Acceptance commands:**

```sh
"$J" optimize proof-plan --base-plan "$BASE_PLAN" --base-plan-sha256 "$BASE_ID" --configuration "$TEACHING_CONFIGURATION" --configuration-sha256 "$TEACHING_CONFIGURATION_SHA" --dataset "$FAMILY_DATASET" --resources "$TEACHING_RESOURCES" --resources-sha256 "$TEACHING_RESOURCES_SHA" --output "$FAMILY_PLAN"
"$J" optimize run --plan "$FAMILY_PLAN" --plan-sha256 "$FAMILY_PLAN_ID" --manifest "$M" --datasets "$D" --journal "$FAMILY_JOURNAL"
"$J" optimize report --plan "$FAMILY_PLAN" --plan-sha256 "$FAMILY_PLAN_ID" --manifest "$M" --datasets "$D" --journal "$FAMILY_JOURNAL"
```

**Required evidence:** Before-fit freeze, vocabulary and source pins; two independent families and original windows; nine context proofs per dataset; complete five-fold predictions and trivial baselines. Apply the1e-12 absolute plus1e-6 relative gain margin and four-of-five rule. Retain no-signal refusals.

**Resources:** Separate explicit resource manifest and approved allowance required before admission;20 CPUs/workers maximum,64 GiB maximum,zero swap,one thread,30 seconds per fit,120 seconds per arm maximum. No new allowance granted by this checklist.

**Source/evidence references:** `agent-training-readiness/readiness-v3.json`, `operations-training.md:Present teacher readiness`, `crates/jope-bench/src/panel/teaching.rs (pending)`.

**Qualification limit:** Execute separately for230_machine_cpu and712_chscase_geyser1; the prospective choice predates losses. Old55e9,3fd and68 operational receipts cannot be retrofitted. The older queue suggested300 secondseconds per family without admitting that allowance.


### TEACH-04 — real corpus materialization

**Priority:** P0. **Current evidence:** native exporter prepared. **Status:** blocked on two real families.

**Action:** Build the request from every eligible requested comparison and its original native-context proof; freeze a family-disjoint corpus and the exact vocabulary.

**Dependencies:** TEACH-03.

**Acceptance commands:**

```sh
"$J" train prepare-inputs --learning-state "$LEARNING_STATE" --request "$CORPUS_REQUEST" --request-sha256 "$CORPUS_REQUEST_SHA" --output "$CORPUS_OUTPUT"
```

**Required evidence:** At least two real families; every eligible requested comparison included; seed7 partition derived from source families; exact1536+512 context values. No invented context, parameter-bin snapping, label remapping or omission of difficult examples.

**Resources:** Read-only/static custody work; no fitting, models, GPU or campaign admission.

**Source/evidence references:** `crates/jope-cli/src/train/corpus.rs (pending)`, `operations-training.md:Build the real corpus`.

**Qualification limit:** Only if exactly two families qualify does the recorded machineCPU-development/geyser-training assignment apply. More eligible families must all remain admitted.


### TRAIN-01 — actual trainer tests

**Priority:** P0. **Current evidence:** seven pure input tests passed; actual model cases deselected. **Status:** residual training-F3.

**Action:** Execute actual optimizer, reload and continuation tests across regression, binary and multiclass, plus task, vocabulary and runtime compatibility negatives.

**Dependencies:** INT-02.

**Acceptance commands:**

```sh
python/.venv/bin/python -m pytest python/tests/test_policy_training_contract.py python/tests/test_training_campaign.py python/tests/test_training_corpus_contract.py python/tests/test_experimental_proposal.py python/tests/test_experimental_runtime.py -q --junitxml="$ACCEPTANCE/training-fixtures.xml"
python3 deploy/ci/test-training-proof.py
```

**Required evidence:** Collected test IDs include all three parameters of test_actual_trained_classifier_reload_sampling_uses_task_head. Prove positive updates, changed parameters, task reward routing, reload and continuation; reject zero updates and missing architecture. The proof gate must see actual execution.

**Resources:** Existing exclusive CPU CI allocation; one reused lane target, jobs4 unless lane policy sets otherwise; no concurrent numerical campaign.

**Source/evidence references:** `python/tests/test_experimental_proposal.py (pending)`, `python/src/jope/policy/train_v3.py (pending)`, `deploy/ci/training-proof.py (pending)`.

**Qualification limit:** Required-name inventories and deselected tests do not count as execution. A synthetic CPU checkpoint remains fixture evidence.


### TRAIN-02 — native supervised training

**Priority:** P0. **Current evidence:** supervisor/cgroup orchestration prepared. **Status:** host proof pending.

**Action:** Exercise the native training front door with bounded synthetic normal and cancellation cases: subgroup entry, release, gate refusal, cleanup and original-window resume.

**Dependencies:** TRAIN-01.

**Acceptance commands:**

```sh
"$J" train plan --state "$FIXTURE_TRAIN_STATE" --python "$TRAINING_PYTHON" --synthetic-fixture --task all-admitted --device cpu --source-sha "$SOURCE_SHA" --seed 7 --passes 2
"$J" train run --state "$FIXTURE_TRAIN_STATE"
"$J" train verify --state "$FIXTURE_TRAIN_STATE"
```

**Required evidence:** Actual normal and interrupted CPU workers; controller-owned subgroup joined before execution; original deadline, device and runtime unchanged; children drained and resumed state compatible. The native wrapper must enforce admission.

**Resources:** Explicitbounded CPU fixture allocation; same planned3600 seconds/1200scaps are maxima, not authorization to spend them twice; no GPU.

**Source/evidence references:** `crates/jope-cli/src/train.rs (pending)`, `python/src/jope/policy/training_campaign.py (pending)`.

**Qualification limit:** Author the cancellation harness against the actual front door and target only its owned processes. CPU mode requires explicit fixture provenance.


### TRAIN-03 — bounded real GPU training

**Priority:** P0. **Current evidence:** implemented follow-on; no GPU campaign started. **Status:** blocked on admitted data and consumption readiness.

**Action:** Run one all-admitted CUDA campaign with real regression and two passes. Retain synthetic classifier fallback separately and report every unavailable category.

**Dependencies:** TEACH-04, TRAIN-02, NEURAL-02.

**Acceptance commands:**

```sh
"$J" train plan --state "$TRAIN_STATE" --python "$TRAINING_PYTHON" --inputs "$CORPUS_OUTPUT/inputs.json" --task all-admitted --device cuda:0 --source-sha "$SOURCE_SHA" --learning-state "$LEARNING_STATE" --seed 7 --passes 2
"$J" train run --state "$TRAIN_STATE"
```

**Required evidence:** Original global/category windows; one RTX3090 UUID; positive updates, changed parameter hashes and task coverage with data_kind/fallback reasons. Save and reload model plus optimizer; record CPU/GPU cost where measured. Requested CUDA must not silently fall back to CPU.

**Resources:** One original3600 seconds total window;≤1200 seconds per category including both passes;one RTX 3090,20CPU,64 GiB,zero swap,1Torch thread;no exclusiveCPU CI overlap. Resume never resets.

**Source/evidence references:** `python/src/jope/policy/training_campaign.py:780 (pending)`, `operations-training.md:Plan, run, resume, verify and export`.

**Qualification limit:** NEURAL-02 gates source and fixture readiness before the GPU hour. It does not depend on a newly trained live export, avoiding a circular prerequisite.


### TRAIN-04 — checkpoint continuation and export

**Priority:** P0. **Current evidence:** verification/export source prepared. **Status:** requires actual campaign.

**Action:** Verify two committed passes, optimizer counters and the immutable receipt chain; export complete architecture, vocabulary, task, source, encoder and runtime bindings.

**Dependencies:** TRAIN-03.

**Acceptance commands:**

```sh
"$J" train resume --state "$TRAIN_STATE"
"$J" train verify --state "$TRAIN_STATE"
"$J" train export --state "$TRAIN_STATE" --output "$TRAIN_EXPORT"
```

**Required evidence:** Resume only when needed inside the original window. Prove increased optimizer steps, identical parameters after reload, changed parameters relative to the prior checkpoint, matching export/resume proof and completed task coverage. Keep experimental non-serving status.

**Resources:** One original3600 seconds total window;≤1200 seconds per category including both passes;one RTX 3090,20CPU,64 GiB,zero swap,1Torch thread;no exclusiveCPU CI overlap. Resume never resets.

**Source/evidence references:** `python/src/jope/policy/training_campaign.py:1203 (pending)`, `python/src/jope/policy/training_campaign.py:1303 (pending)`.

**Qualification limit:** Resume is conditional, not an instruction to start a third pass after completing the plan. A new state or plan cannot extend the approved hour.


### NEURAL-02 — trained-checkpoint consumer readiness

**Priority:** P0. **Current evidence:** independent source review + purechecks; regression packet appended. **Status:** compile/runtime fixture pending.

**Action:** Qualify the explicit experimental regression consumer before GPU use. Preserve accepted-baseline hard pins and strict architecture, task, vocabulary, source and encoder compatibility.

**Dependencies:** INT-02, TRAIN-01.

**Acceptance commands:**

```sh
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-pipeline --features qualified-context experimental_regression_requires_matching_task_and_executable_head
python/.venv/bin/python -m pytest python/tests/test_experimental_proposal.py -q
```

**Required evidence:** Execute all three actual task-model cases. Test accepted-versus-experimental decoding and refusal of incomplete architecture, missing state, changed vocabulary, source or encoder. The operational regression route requires a real admitted export.

**Resources:** Existing exclusive CPU CI allocation; one reused lane target, jobs4 unless lane policy sets otherwise; no concurrent numerical campaign.

**Source/evidence references:** `agent-experimental-regression/HANDOFF.json`, `crates/jope-pipeline/src/experimental_producer.rs (pending)`, `python/src/jope/policy/experimental_proposal.py (pending)`.

**Qualification limit:** The new regression consumer is not serving promotion. Synthetic regression may test mechanics but cannot enter the real operational route.


### NEURAL-03 — live newly trained consumption

**Priority:** P0. **Current evidence:** current accepted checkpoint origin/evolution proved onone dataset. **Status:** newly trained consumption pending.

**Action:** Pin the actual exported checkpoints and runtime/canonicalizer into the final configuration. Prove new-checkpoint proposals reach real generation-zero fits and completed descendants.

**Dependencies:** TRAIN-04, NEURAL-01.

**Acceptance commands:**

```sh
"$J" optimize proof-plan --base-plan "$BASE_PLAN" --base-plan-sha256 "$BASE_ID" --configuration "$TRAINED_CONFIGURATION" --configuration-sha256 "$TRAINED_CONFIGURATION_SHA" --dataset 230_machine_cpu --resources "$CONSUMPTION_RESOURCES" --resources-sha256 "$CONSUMPTION_RESOURCES_SHA" --output "$CONSUMPTION_PLAN"
"$J" optimize run --plan "$CONSUMPTION_PLAN" --plan-sha256 "$CONSUMPTION_PLAN_ID" --manifest "$M" --datasets "$D" --journal "$CONSUMPTION_JOURNAL"
"$J" optimize report --plan "$CONSUMPTION_PLAN" --plan-sha256 "$CONSUMPTION_PLAN_ID" --manifest "$M" --datasets "$D" --journal "$CONSUMPTION_JOURNAL"
```

**Required evidence:** The exported checkpoint hash appears in actual proposal and fit evidence;64+16 versus80 origins remain authenticated. Retain completed descendants where claimed, predictions and fitted states. Regression must be admitted and classifier fixture limits explicit; replay the original window successfully.

**Resources:** Separate explicit resource manifest and approved allowance required before admission;20 CPUs/workers maximum,64 GiB maximum,zero swap,one thread,30 seconds per fit,120 seconds per arm maximum. No new allowance granted by this checklist.

**Source/evidence references:** `crates/jope-bench/src/panel/experimental_proposal.rs (pending)`, `crates/jope-bench/src/panel/origin_proof.rs (pending)`.

**Qualification limit:** The current proof-plan scopes one regression dataset. Classification needs its implemented campaign/configuration route; this command alone cannot prove binary or multiclass live consumption.


### INFER-01 — fitted state and privacy

**Priority:** P0. **Current evidence:** native fitted state exists; product28source prepared. **Status:** qualification pending.

**Action:** Execute the lossless private codec, complete fitted preprocessing/state reload, and public allowlist/canary tests before export. Unsupported states remain explicit.

**Dependencies:** INT-02.

**Acceptance commands:**

```sh
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p feat-math fitted_portable
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-bench --features asm inference
bash deploy/ci/ip-scrub.sh
```

**Required evidence:** Complete prediction parity after reload; missing/unseen category, class order and full probability-matrix cases. Preserve the private codec identity; public bundles must omit recipe, discovery and provenance canaries.

**Resources:** Existing exclusive CPU CI allocation; one reused lane target, jobs4 unless lane policy sets otherwise; no concurrent numerical campaign.

**Source/evidence references:** `crates/feat-math/src/advanced_v3/fitted_portable.rs (pending)`, `crates/jope-bench/src/panel/inference.rs (pending)`, `languages-examples.md:State restrictions`.

**Qualification limit:** The private C++ head gate is broader than the public gate. Python/Rust14of26, C++2of26 and JavaScript6of26 are static head counts, not runtime qualification rates.


### INFER-02 — exact selected-winner portability

**Priority:** P1. **Current evidence:** winner/add-on source + statictests. **Status:** actual export/parity unrun.

**Action:** Qualify the selected existing fold state in order: Python, Rust, the same C++ emitter under C++17 and C++20, then JavaScript. Preserve unsupported external winners and the complete ranking.

**Dependencies:** INFER-01, PANEL-01.

**Acceptance commands:**

```sh
"$J" benchmark export-winner --plan "$PLAN" --plan-sha256 "$PLAN_ID" --manifest "$M" --datasets "$D" --journal "$JOURNAL" --dataset "$DATASET_ID" --fold "$FOLD"
"$J" benchmark qualify-export --prepared "$PREPARED_EXPORT" --prepared-sha256 "$PREPARED_EXPORT_ID" --tools "$TOOLS" --tools-sha256 "$TOOLS_SHA"
"$J" benchmark report --plan "$PLAN" --plan-sha256 "$PLAN_ID" --manifest "$M" --datasets "$D" --journal "$JOURNAL"
```

**Required evidence:** Exact state, artifact, input, class, toolchain and numerical-policy hashes; actual five-backend outcomes. Python/Rust/C++ use bitwise comparison; JavaScript uses1e-12 absolute and relative tolerance. Recover interrupted work without retrying its child; read-only reports launch nothing. Unsupported never replaces the winner.

**Resources:** Same original campaign execution window and worker caps; no new job after t0 + 6900 seconds for official panel. No post-panel allowance is implemented.

**Source/evidence references:** `crates/jope-bench/src/panel/fitted_export.rs (pending)`, `crates/jope-bench/src/panel/fitted_export_run.rs (pending)`, `languages-examples.md:Example C`.

**Qualification limit:** Library fixtures precede the panel. Actual selected winners require complete confirmations from PANEL-01 and qualification before the original execution cutoff. No separately bounded post-panel qualification front door exists. One fitted fold does not qualify the other four states.


### INSTALL-01 — private installation

**Priority:** P1. **Current evidence:** installer source and15metadata tests passed. **Status:** cold/live path pending.

**Action:** Build the CLI from final source, measure cold build separately, checksum-install the exact Linux binary, and verify actual command help plus dependency/ELF boundaries.

**Dependencies:** INT-02.

**Acceptance commands:**

```sh
python3 deploy/install-private.py source --checkout "$R" --source-sha "$SOURCE_SHA" --output "$ACCEPTANCE/cli-build" --jobs 4
python3 deploy/install-private.py binary --binary "$PRIVATE_BINARY" --sha256 "$BINARY_SHA" --source-sha "$SOURCE_SHA" --prefix "$ACCEPTANCE/install"
"$J" doctor
```

**Required evidence:** Cold build/install durations, cache and already_present flags; Cargo executable messages, features, toolchain, dependency graph and ELF checks. Retain checksum/help readbacks and failure receipts. CLI dependencies exclude the engine, Torch and CUDA.

**Resources:** Existing exclusive CPU CI allocation; one reused lane target, jobs4 unless lane policy sets otherwise; no concurrent numerical campaign.

**Source/evidence references:** `deploy/install-private.py (pending)`, `product.md:Build and install`.

**Qualification limit:** An installer receipt is not policy/context-worker build authority. Do not include training credentials or model bodies in the public inference bundle.


### INSTALL-02 — final input/runtime preparation

**Priority:** P1. **Current evidence:** prepared panel existing; runtime refresh required. **Status:** pending final source.

**Action:** Install the pinned final Python wheel without changing CUDA dependencies; build worker inventories and verify prepared datasets before campaign planning.

**Dependencies:** INSTALL-01, BENCH-01, BENCH-02.

**Acceptance commands:**

```sh
"$J" benchmark verify-inputs --manifest "$M" --datasets "$D"
"$J" benchmark plan --manifest "$M" --datasets "$D" --source-sha "$SOURCE_SHA" --runtimes "$RUNTIMES" --output "$BASE_PLAN"
```

**Required evidence:** Wheel, source, runtime and dependency inventory; context/policy workers identified from actual Cargo messages; semantic base-plan identity; zero fits and no execution clock started.

**Resources:** Read-only/static custody work; no fitting, models, GPU or campaign admission.

**Source/evidence references:** `product.md:Prepare the panel`, `agent-acceptance-queue/QUEUE.md`.

**Qualification limit:** Packaging commands depend on final reviewed artifacts and the actual virtual environment. Preserve versions and file identities; changing the environment invalidates a pinned campaign.


### DEMO-01 — measured warm fitted demonstration

**Priority:** P1. **Current evidence:** warmtargetadd-onprepared;historicalfirstfit≈50.4s. **Status:** realtrained warm proof pending.

**Action:** Produce actual trained-neural and generic fitted artifacts, reloaded predictions, baseline/comparison and truthful target_met. Do not clip arm budgets to pass sixty seconds.

**Dependencies:** NEURAL-03, INSTALL-02.

**Acceptance commands:**

```sh
"$J" demo --asm-plan "$WARM_PLAN" --asm-plan-sha256 "$WARM_PLAN_ID" --manifest "$M" --datasets "$D" --output "$WARM_OUTPUT" --export python --export rust
python3 deploy/ci/qualify-demo.py --binary "$J" --sha256 "$J_SHA" --source-sha "$SOURCE_SHA" --output "$ACCEPTANCE/generic-three-task" --require-under-60
```

**Required evidence:** Actual fitted state, predictions and baseline; both full paired allowances; start/end, warm elapsed, execution_complete and target_met. Keep generic synthetic qualification separate from trained-neural proof.

**Resources:** Separate explicit resource manifest and approved allowance required before admission;20 CPUs/workers maximum,64 GiB maximum,zero swap,one thread,30 seconds per fit,120 seconds per arm maximum. No new allowance granted by this checklist.

**Source/evidence references:** `crates/jope-bench/src/panel/demo_asm.rs (pending)`, `docs/private-warm-target.md (pending)`, `product.md:Warm fitted demonstration`.

**Qualification limit:** Under sixty seconds is a measured target. The existing twenty-second-arm example is unqualified. An explicitly admitted longer proof may satisfy execution while missing the target. This checklist grants no implicit300-second allowance.


### PANEL-01 — unchanged thirty-dataset campaign

**Priority:** P1. **Current evidence:** protocol/currentfoundation qualified;full panel unstarted. **Status:** pending finaltrained configuration.

**Action:** Freeze the final roster/configuration and run the unchanged thirty datasets once. Screen fold0, freeze the top five, confirm five development folds, and add asm.neural diagnostically when absent. Retain every failure.

**Dependencies:** BENCH-01, BENCH-02, NEURAL-03, DEMO-01, INFER-01, INSTALL-02.

**Acceptance commands:**

```sh
"$J" optimize plan --base-plan "$BASE_PLAN" --base-plan-sha256 "$BASE_ID" --configuration "$TRAINED_CONFIGURATION" --configuration-sha256 "$TRAINED_CONFIGURATION_SHA" --output "$PANEL_PLAN"
"$J" benchmark run --plan "$PANEL_PLAN" --plan-sha256 "$PANEL_PLAN_ID" --manifest "$M" --datasets "$D" --journal "$PANEL_JOURNAL"
"$J" benchmark report --plan "$PANEL_PLAN" --plan-sha256 "$PANEL_PLAN_ID" --manifest "$M" --datasets "$D" --journal "$PANEL_JOURNAL"
```

**Required evidence:** All thirty dataset identities, ten per task; every applicable method slot; screen/frozen/confirmation losses; selection-exposure row intersection hashes; no sealed reads; winner backend outcomes and actual win counts.

**Resources:** Exactly7200 seconds including300 seconds reporting reserve;20 CPUs/workers,64 GiB,zero swap,one thread per fit,30 seconds per fit,120 seconds per search arm; sealed labels unopened; existing window never reset.

**Source/evidence references:** `benchmarks/panel.json`, `crates/jope-bench/src/panel/protocol.rs`, `crates/jope-bench/src/panel/selection_exposure.rs`.

**Qualification limit:** Current code forces asm.neural into extra confirmation, not asm.generic. Selected five-fold summaries are adaptive development, and ten wins per category are not promised. Qualify exports before the original6900-second cutoff or preserve unattempted status.


### REPORT-01 — complete measured report

**Priority:** P1. **Current evidence:** one-dataset832predictor/164recipe operational audit. **Status:** fullcampaign report pending.

**Action:** Publish all thirty losses/gaps, outcomes, time/cost per unique artifact and recipe, coverage and failure inventories. Compare the supplied one-hour/ten-million-token scenario only using supported quantities.

**Dependencies:** PANEL-01, EXEC-05.

**Acceptance commands:**

```sh
"$J" benchmark report --plan "$PANEL_PLAN" --plan-sha256 "$PANEL_PLAN_ID" --manifest "$M" --datasets "$D" --journal "$PANEL_JOURNAL"
```

**Required evidence:** Report recovery without fits; union of started/finished records; worker CPU distinguished from overlapping service CPU; known and unknown costs; training, installation, preparation and export accounted separately. Tokens and money stay unknown unless measured; retain win/loss/gap tables.

**Resources:** Original300 secondsreportingreserve forpanel; laterreadonlyauditopensnofitbudget.

**Source/evidence references:** `status.md:Cost`, `data-evidence.md`, `crates/jope-bench/src/panel/asm_campaign.rs (pending)`.

**Qualification limit:** The one-hour comparison supplies wall-duration context, not a matched speedup. Unique artifact hashes do not establish behavioral uniqueness. Never add aggregate service CPU to worker CPU already included in it.


### MOTIF-01 — eligible reusable fragments

**Priority:** P1. **Current evidence:** separate typed library branch exists. **Status:** minefront door incomplete.

**Action:** Integrate the production eligible-State reader, typed fragment/macro miner and umbrella CLI. Preserve full comparison identity, provenance and private/public separation.

**Dependencies:** TEACH-01, TEACH-02, TEACH-03.

**Acceptance commands:**

```sh
"$J" motifs mine --help
"$J" motifs validate --help
```

**Required evidence:** Actual mine CLI contract/help; eligible-input identity inventory; deterministic typed fragment output. Exclude rejected, no-signal, unbound and sealed examples.

**Resources:** Read-only/static custody work; no fitting, models, GPU or campaign admission.

**Source/evidence references:** `status.md:Pending acceptance`, `agent-acceptance-queue/QUEUE.md:motifs`.

**Qualification limit:** These help commands are currently unavailable at 68 and are acceptance targets. No truthful exact mine invocation exists yet; record it after implementation.


### MOTIF-02 — reuse validation

**Priority:** P1. **Current evidence:** validationlibrary/preparation only. **Status:** live reuse pending.

**Action:** Validate fragments/macros on independent host contexts, lossless roundtrips, deterministic fit compatibility and privacy. Measure reuse benefit under the full comparison identity.

**Dependencies:** MOTIF-01.

**Command boundary:** no truthful executable acceptance command yet; implement or obtain missing interface first.

**Required evidence:** Exact implemented validate command and input table; at least three hosts per fragment where its existing contract requires that count; roundtrip and actual fit receipts, independent context identity and canary absence. Support frequency is separate from measured uplift.

**Resources:** Explicitnewboundedreusequalification allocation required; noresetofpanelortrainingwindows.

**Source/evidence references:** `agent-acceptance-queue/QUEUE.md:motifs`, `docs/adr/ADR-0005-canonical-learning-loop.md`.

**Qualification limit:** Do not invent mine flags. The historical validate argument shape alone does not establish umbrella wiring or production State admission.


### MOTIF-03 — new operators

**Priority:** P2. **Current evidence:** useracceptance requirement. **Status:** future conditional work.

**Action:** Admit new operators only after measured ablation and grammar, checkpoint and backend compatibility tests. Preserve the old vocabulary identity.

**Dependencies:** MOTIF-02.

**Command boundary:** no truthful executable acceptance command yet; implement or obtain missing interface first.

**Required evidence:** Preregistered ablation recipes, budget and contexts; measured marginal gain; task/type/codec/emitter/vocabulary compatibility; new identity and migration policy.

**Resources:** Separatelypreregistered ablationbudget; noneadmitted here.

**Source/evidence references:** `user approval plan:Eligible optimization and reuse`, `languages-examples.md:Future work`.

**Qualification limit:** Typed reusable macros may stay within existing operators. Fragment frequency alone does not justify adding an operator.


### IP-01 — public privacy boundaries

**Priority:** P1. **Current evidence:** allowlists/canaries prepared; historicallintblindspots. **Status:** residual D-IP4 and nextPRtextwork.

**Action:** Resolve the approved public vocabulary/naming scope and test actual CLI help, receipts and NOTICE. Preserve the explicitly requested internal manual and the strict public inference allowlist.

**Dependencies:** INT-01.

**Acceptance commands:**

```sh
bash deploy/ci/ip-scrub.sh
```

**Required evidence:** Actual shipped-file inventory and help/receipt/NOTICE canaries; private provenance absent; allowlist negative cases. Keep the internal requested manual distinct from public bundles.

**Resources:** Existing exclusive CPU CI allocation; one reused lane target, jobs4 unless lane policy sets otherwise; no concurrent numerical campaign.

**Source/evidence references:** `AGENT_CHAT.md:D-IP4`, `deploy/ci/ip-scrub.sh`, `crates/feat-math/src/advanced_v3/fitted_portable.rs (pending)`.

**Qualification limit:** Current lint success does not prove unscanned surfaces clean. The user explicitly requested the private GP explanation, so ordinary creation of this manual needs no additional approval.


### MAINT-01 — broad diagnostic qualification

**Priority:** P2. **Current evidence:** historicalfull-default failures inventoried. **Status:** not closedbyfastCI.

**Action:** Reproduce and repair recorded full-default, scratch, policy and neural families through their owning full lanes. Retain unavailable capabilities and resource limits.

**Dependencies:** LIFE-01.

**Acceptance commands:**

```sh
bash deploy/ci/full-lane.sh full-default
bash deploy/ci/full-lane.sh full-scratch
bash deploy/ci/full-lane.sh full-policy
bash deploy/ci/full-lane.sh full-neural
```

**Required evidence:** Source-bound full-suite collection and failures, reproductions and corrected receipts. Keep fast-CI and full-diagnostic totals separate; zero tests or skips cannot count as passes.

**Resources:** Existing exclusive CPU CI allocation; one reused lane target, jobs4 unless lane policy sets otherwise; no concurrent numerical campaign.

**Source/evidence references:** `deploy/ci/full-lane.sh`, `maintenance.md:Coverage and reviewed lifecycle`, `AGENT_CHAT.md:full-default failures`.

**Qualification limit:** Check lane environment requirements before invocation. The owning qualification supplies allocation and budget; this manual does not admit them.


### FUTURE-01 — CUDA inference and serving promotion

**Priority:** P2. **Current evidence:** not qualified in this delivery. **Status:** deferred.

**Action:** After portable CPU backends, design CUDA fitted-inference qualification and independent experimental-to-serving promotion. Preserve the accepted baseline until promotion passes.

**Dependencies:** INFER-02, REPORT-01.

**Command boundary:** no truthful executable acceptance command yet; implement or obtain missing interface first.

**Required evidence:** New backend contract, fixed numerical policy, exact device/toolchain/state parity and promotion evaluation; reviewed pointer compare-and-swap and rollback where used.

**Resources:** NoCUDAinference/promotionbudgetadmitted bythisroadmap.

**Source/evidence references:** `docs/adr/ADR-0005-canonical-learning-loop.md`, `architecture.md:Experimental trained models`.

**Qualification limit:** Training GPU availability does not prove CUDA inference parity. Export is not serving promotion.


## Stopping and completion rules

An unsuccessful prerequisite remains a recorded refusal or failure. Do not substitute synthetic real-data teachers, another winner, easier datasets or a new deadline. Source completion does not close a runtime task. A plan, help output or checkpoint file is not the corresponding live receipt.

Full delivery requires all thirty outcomes, losses and gaps, backend limitations, measured unique-predictor/recipe costs, honest training category scope and actual newly trained consumption. Ten wins per category and an under-sixty-second warm result remain measured targets. The one-hour/ten-million-token comparison must retain unknown hardware, workload, token and currency quantities.

The document gate requires one readable PDF plus matching section Markdown, reproducible figure inputs, source/lifecycle references and this maintained checklist. The maintainer-private examples are authorized documentation; their existence does not expand public inference bundle contents.
