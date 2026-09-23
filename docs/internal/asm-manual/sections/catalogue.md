# Full task catalogue, source/schema index, receipts and glossary

## Primary acceptance items by stage

### Stage 0: coordination, documentation and cross-cutting work

#### LIFE-01 — coordination

**Priority:** P0. **Current evidence:** reviewed merge, exact-tree remote main and successful main-guard. **Status:** complete with retained readback.

**Action:** Retain the verified lifecycle receipt; future source changes repeat exact-head CI and eligible independent native review.

**Dependencies:** None.

**Acceptance commands:**

```sh
/home/ubuntu/.local/bin/gh-role writer pr view 83 --json mergedAt,mergeCommit,headRefOid,state
/home/ubuntu/.local/bin/gh-role writer run view 35889731910 --json headSha,attempt,status,conclusion,jobs
```

**Required evidence:** PR83 merged 2026-09-23T16:34:47Z; remote main a75e543d2166017cb0139e0a251d51d9a0c98e08 has tree 176b75bc36d76c7ce4eaeb7234677ec013468889, the same tree as reviewed head 68f56404c3e1. Main-guard 35889731910 attempt 1 succeeded; job 107278860508 finished 16:40:11Z. Clean canonical main verified before docs/asm-maintainer-manual branch claim.

**Resources:** Read-only/static custody work; no fitting, models, GPU or campaign admission.

**Source/evidence references:** `AGENTS.md:Pull requests`, [§](status.md#implementation-snapshot-and-this-manuals-own-identity), [§](maintenance.md#coverage-and-reviewed-lifecycle), `neural-integration/lifecycle-68f56404c3e199e06413bfd645d3fe0803d7b96f/remote-main-verification.json`.

**Qualification limit:** This task is complete; subsequent implementation remains pending. Keep the measured source identity 68f56404c3e1 distinct from the rebase-merge commit identity a75e543d2166.

#### DOC-01 — manual

**Priority:** P1. **Current evidence:** assembled seventeen chapters with source-audited training and target histories. **Status:** canonical private manual assembled.

**Action:** Maintain the dated evidence inventory and current source/pending distinctions.

**Dependencies:** LIFE-01.

**Acceptance commands:**

```sh
python3 docs/internal/asm-manual/tools/build_manual.py
```

**Required evidence:** Input manifest pins every chapter, figure dataset, source receipt and render tool; duplicates resolved; current/pending/historical/unknown labels retained.

**Resources:** Read-only/static custody work; no fitting, models, GPU or campaign admission.

**Source/evidence references:** `architecture-training-evidence.json`, `product-language-source-pins.json`, `dataset-inventory.json`, `performance-evidence.json.gz`.

**Qualification limit:** Root integrated the book and renderer. Claude historical plot inputs were copied under stable before/after hashes; no stopped external renderer handoff is required.

#### DOC-02 — manual examples

**Priority:** P1. **Current evidence:** fourteen actual commands passed; five backend predictions bit-identical. **Status:** complete at named recipe/refit scope.

**Action:** Retain the completed fourteen-command receipt and rerun after meaningful recipe/runtime changes; qualify saved fitted winners separately.

**Dependencies:** DOC-01.

**Acceptance commands:**

```sh
python3 "$EXAMPLES/validate-outputs.py" --recipe-output "$EXAMPLE_RUN/recipe"
"$J" pipeline verify-receipt --receipt "$EXAMPLE_RUN/pipeline/receipt.json" --json
```

**Required evidence:** examples/commands.json instantiated with actual paths and pinned source/tool versions; five outputs with 8 finite values and complete-bit comparison; pipeline receipt verification. Stopped audit SHA256 363fbe62a706b44030205da78411243ba206ad2366d546a277babe966b802c91.

**Resources:** Completed separate 300-second allocation: one CPU, 4 GiB, zero swap/GPU, network denial declared; 30-second numerical and 60-second compiler caps. Actual receipt retained; independent live property readback limitation is disclosed. No Cargo or frozen-panel campaign; no deadline reset.

**Source/evidence references:** `examples/commands.json`, `examples/validate-outputs.py`, [§](inference.md#example-a-a-tiny-private-recipe-fit-and-predictions), [§](inference.md#example-b-actual-legacy-pipelinesearch-shape).

**Qualification limit:** Actual acceptance: data/example-acceptance/audit.json. Five recipe refits each predict eight synthetic rows; separate legacy pmlb229 pipeline has four successful measurements. No fitted-export, ASM demo, training or full-panel qualification. Resource-readback race retained.

#### DOC-03 — manual figures

**Priority:** P1. **Current evidence:** twenty-one local figure artifacts; original metrics, complete recovered per-step series and generation points retained. **Status:** figures rendered; final PDF inspection pending.

**Action:** Retain historical/current labels, complete source CSVs and final PDF visual checks.

**Dependencies:** DOC-01.

**Acceptance commands:**

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 docs/internal/asm-manual/tools/make_inventory_figures.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 docs/internal/asm-manual/tools/make_figures.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 docs/internal/asm-manual/tools/make_training_figures.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 docs/internal/asm-manual/tools/make_target_figures.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 docs/internal/asm-manual/tools/make_legacy_training_figure.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 docs/internal/asm-manual/tools/make_per_step_figure.py
```

**Required evidence:** Plot source CSV/JSON, tool source, plot hash, precise captions, missing-stat flags and historical-vs-current labels; no invented sealed statistics or causal speedup.

**Resources:** Read-only/static custody work; no fitting, models, GPU or campaign admission.

**Source/evidence references:** [§](datasets.md#frozen-dataset-coverage), [§](gp-targets.md#ranking-figures-interpretation-contract), `dataset-inventory.json`, `performance-evidence.json.gz`.

**Qualification limit:** Historical Elo belongs to its original ladder/pool/protocol; the current one-dataset proof is not a new Elo ladder. Source and figure checks are retained; authored architecture SVG is a repository source artifact.

#### DOC-04 — manual quality

**Priority:** P1. **Current evidence:** first edition rendered; expanded seventeen-chapter final checks required. **Status:** rendered; final publication gates pending.

**Action:** Render docs/*.pdf, extract PDF text, inspect all dense tables/code/figures, verify local links and filenames, and produce a readback checklist.

**Dependencies:** DOC-01, DOC-02, DOC-03.

**Acceptance commands:**

```sh
pdftotext -layout "$MANUAL_PDF" "$MANUAL_TEXT"
```

**Required evidence:** PDF SHA/page count; extracted-text receipt; visual inspections of title/contents, widest tables, commands, plots and final checklist; no clipping/omitted rows; Markdown counterparts present.

**Resources:** Read-only/static custody work; no fitting, models, GPU or campaign admission.

**Source/evidence references:** [§](maintenance.md#updating-the-book).

**Qualification limit:** A successful renderer exit is insufficient: the PDF must remain readable and complete.

#### DOC-05 — real training and target history

**Priority:** P1. **Status:** Original training and target values verified; final composed-document inspection required.

**Action:** Publish exact observed optimizer/corpus/encoding/target history, component-loss curves, checkpoint lineage, source hashes and explicit comparability/missing-history limits.

**Dependencies:** DOC-01.

**Required evidence:** Audited original JSON-pointer-to-CSV equality; every curve bound to model/corpus/monitor/objective; original failed/inconclusive observations retained.

**Commands:**

```sh
python3 docs/internal/asm-manual/tools/make_training_figures.py
python3 docs/internal/asm-manual/tools/make_target_figures.py
python3 docs/internal/asm-manual/tools/make_legacy_training_figure.py
python3 docs/internal/asm-manual/tools/make_per_step_figure.py
```

The recovered series preserves 63,000 rows and 378,000 exact metric tokens. Keep all 21 corpus blocks, 20 intervening boundaries and the legacy internal-ladder limitations explicit.

**Resources:** No new fits, training or model loads; plotting retained metrics only.

**Sources:** [§](training-history.md#dated-lineage-and-evidence-coverage), [§](learning.md#what-the-historical-training-and-target-evidence-establishes).

#### DOC-06 — versions, KPIs and continuation runbook

**Priority:** P1. **Status:** Current and pending interfaces source-audited; independent chapter review complete.

**Action:** Publish the product/model/encoding/vocabulary/corpus/protocol version tuple, north-star and KPI definitions, real continuation evidence, and exact current-versus-pending commands and code.

**Dependencies:** DOC-01.

**Required evidence:** Source-bound command/flag/API audit; held native grant path disclosed; warm-start distinct from optimizer resume; full input closure and 64 GiB budget compatibility stated.

**Commands:** No complete executable front door is claimed; use the linked chapter’s current/pending interface map.

**Resources:** Documentation and metadata validation only; commands do not admit a new training window.

**Sources:** [§](status.md#what-version-are-we-running), [§](plan.md#the-continuation-interfaces-that-exist-today), [§](learning.md#what-better-and-better-must-mean), [§](search.md#focused-single-dataset-search).

#### INT-01 — serial integration

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

#### INT-02 — integration qualification

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

**Qualification limit:** Custody syntax tests are not this gate. Required 18 Python / 27 Rust / 10 policy names are proof minima, not total coverage. New code needs actual test collection.

#### IP-01 — public privacy boundaries

**Priority:** P1. **Current evidence:** allowlists/canaries prepared; historicallintblindspots. **Status:** residual D-IP4 and future PR text work.

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

#### MAINT-01 — broad diagnostic qualification

**Priority:** P2. **Current evidence:** historicalfull-default failures inventoried. **Status:** not closed by fast CI.

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

**Source/evidence references:** `deploy/ci/full-lane.sh`, [§](maintenance.md#coverage-and-reviewed-lifecycle), `AGENT_CHAT.md:full-default failures`.

**Qualification limit:** Check lane environment requirements before invocation. The owning qualification supplies allocation and budget; this manual does not admit them.

### Stage 1: correct benchmark integration

#### BENCH-01 — frozen benchmark foundation

**Priority:** P1. **Current evidence:** current source + local/exact-head tests. **Status:** implemented; retain during integration.

**Action:** Reverify unchanged manifest and all prepared inputs, actual row/group assignments, original category/missing/label semantics, loss directions, full class probability ordering and task method inventory.

**Dependencies:** INT-02.

**Acceptance commands:**

```sh
"$J" benchmark verify-manifest --manifest "$M"
"$J" benchmark verify-inputs --manifest "$M" --datasets "$D"
"$J" benchmark methods
```

**Required evidence:** Original panel SHA ba7dace4…, 30 datasets, 10 per task, exact input-byte verification, native 17/5/4 recipes and 4 external families; alias/configuration inventory and unavailable reasons.

**Resources:** Read-only/static custody work; no fitting, models, GPU or campaign admission.

**Source/evidence references:** `crates/jope-bench/src/panel/data.rs`, `crates/jope-bench/src/panel/methods.rs`, `crates/jope-bench/src/panel/score.rs`, `benchmarks/panel.json`.

**Qualification limit:** Verification is metadata/data validation, no new fits. Do not replace a failed panel dataset.

#### BENCH-02 — external comparators

**Priority:** P1. **Current evidence:** four adapters implemented; earlier runtime inventories. **Status:** refresh at final environment.

**Action:** Pin all runtime/package/adapter bytes and actual constructors; execute relevant adapter contract tests and preserve unavailable dependencies.

**Dependencies:** INT-02.

**Acceptance commands:**

```sh
python3 deploy/prepare-panel-runtime.py --binary "$J" --native-command benchmark --build-receipt "$CLI_BUILD_RECEIPT" --source-sha "$SOURCE_SHA" --python "$EXTERNAL_PYTHON" --site-packages "$EXTERNAL_SITE_PACKAGES" --adapters crates/jope-bench/adapters --output "$ACCEPTANCE/runtimes.json"
"$EXTERNAL_PYTHON" -m pytest crates/jope-bench/adapters/test_panel.py -q
```

**Required evidence:** Current 4 engine version/source/package hashes, constructors for 3 tasks, actual contract-test receipt and explicit unavailability inventory.

**Resources:** Preparation owns sharedCPUallocation; tests use allocatedCPU; no panel/training window begins.

**Source/evidence references:** `deploy/prepare-panel-runtime.py`, `crates/jope-bench/adapters/_panel.py`, `crates/jope-bench/adapters/test_panel.py`.

**Qualification limit:** Do not nest an outer allocation lock around a front door already taking it. Environment mutation invalidates old inventory.

### Stage 2: shared execution and accounting

#### EXEC-01 — bounded workers

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

#### EXEC-02 — fault regression tests

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

#### EXEC-03 — receipt replay regression tests

**Priority:** P1. **Current evidence:** shared per-fold helper fixed and real 68 recovered report passed. **Status:** test gap: resume-fix-F2.

**Action:** Add PoolBackend-shaped candidate receipt roundtrip asserting live Feedback equals recovered FitAdapter Feedback, including digest and partial completed-fold evidence.

**Dependencies:** INT-01.

**Acceptance commands:**

```sh
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-bench --features asm
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only cargo test --locked -p jope-pipeline --features qualified-context neural_search::tests
```

**Required evidence:** Regression reproduces the d0f5748b7db1 fitness/replay mismatch; exercises Some(shared fold digest), changed completed outputs refuse, never-admitted normalization cannot gain evidence.

**Resources:** Existing exclusive CPU CI allocation; one reused lane target, jobs4 unless lane policy sets otherwise; no concurrent numerical campaign.

**Source/evidence references:** `crates/jope-pipeline/src/neural_search.rs:1735`, `crates/jope-bench/src/panel/optimize_fit.rs`, `AGENT_CHAT.md:resume-fix-F2`.

**Qualification limit:** The two original FixtureBackend reproductions use absent digests and do not cover the same_folds path.

#### EXEC-04 — cutoff semantics regression

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

#### EXEC-05 — accounting and unknown outcomes

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

### Stage 3: neural consumption and training

#### NEURAL-01 — context qualification

**Priority:** P0. **Current evidence:** eight fixed native cases passed at 68. **Status:** requires final-source binding.

**Action:** Build exact context/policy/proof workers; qualify unchanged native references for final source and preserve eight fixed-case plus actual dataset process evidence.

**Dependencies:** INT-02.

**Acceptance commands:**

```sh
"$CONTEXT_PROOF" prepare --sources "$REFERENCE_SOURCES" --sources-sha256 "$REFERENCE_SOURCES_SHA" --encoder-files "$ENCODER_FILES" --encoder-files-sha256 "$ENCODER_FILES_SHA" --worker "$CONTEXT_WORKER" --worker-sha256 "$CONTEXT_WORKER_SHA" --source-sha40 "$SOURCE_SHA" --resources "$CONTEXT_RESOURCES" --resources-sha256 "$CONTEXT_RESOURCES_SHA" --output "$ACCEPTANCE/context-proof"
"$CONTEXT_PROOF" run --plan "$CONTEXT_PLAN" --plan-sha256 "$CONTEXT_PLAN_SHA"
```

**Required evidence:** All eight fixed cases pass the unchanged component/context maximum absolute difference of 0.0003; source, assets and program identities match. Preserve actual bounded execution receipts and the qualified construction token, original labels and row order.

**Resources:** Separate explicit resource manifest and approved allowance required before admission;20 CPUs/workers maximum,64 GiB maximum,zero swap,one thread,30 seconds per fit,120 seconds per arm maximum. No new allowance granted by this checklist.

**Source/evidence references:** `crates/jope-policy/src/context_producer/admission.rs:4`, `crates/jope-pipeline/src/cpu_context_bridge.rs`.

**Qualification limit:** Metadata/path equality or checksum alone is not numerical qualification; no changed tolerance or checkpoint substitution.

#### TEACH-03 — real teacher collection

**Priority:** P0. **Current evidence:** zero eligible current teachers; two prospective evidenced families. **Status:** blocked on fresh admitted evidence.

**Action:** Preregister teaching before two separately bounded proof plans for machineCPU and geyser. Obtain actual finite OOF improvement and improvements on four of five folds, or retain the rejection.

**Dependencies:** BENCH-01, BENCH-02, NEURAL-01, TEACH-01, TEACH-02.

**Acceptance commands:**

```sh
"$J" optimize proof-plan --base-plan "$BASE_PLAN" --base-plan-sha256 "$BASE_ID" --configuration "$TEACHING_CONFIGURATION" --configuration-sha256 "$TEACHING_CONFIGURATION_SHA" --dataset "$FAMILY_DATASET" --resources "$TEACHING_RESOURCES" --resources-sha256 "$TEACHING_RESOURCES_SHA" --output "$FAMILY_PLAN"
"$J" optimize run --plan "$FAMILY_PLAN" --plan-sha256 "$FAMILY_PLAN_ID" --manifest "$M" --datasets "$D" --journal "$FAMILY_JOURNAL"
"$J" optimize report --plan "$FAMILY_PLAN" --plan-sha256 "$FAMILY_PLAN_ID" --manifest "$M" --datasets "$D" --journal "$FAMILY_JOURNAL"
```

**Required evidence:** Before-fit freeze, vocabulary and source pins; two independent families and original windows; nine context proofs per dataset; complete five-fold predictions and trivial baselines. Apply the 1e-12 absolute plus 1e-6 relative gain margin and four-of-five rule. Retain no-signal refusals.

**Resources:** Separate explicit resource manifest and approved allowance required before admission;20 CPUs/workers maximum,64 GiB maximum,zero swap,one thread,30 seconds per fit,120 seconds per arm maximum. No new allowance granted by this checklist.

**Source/evidence references:** `agent-training-readiness/readiness-v3.json`, [§](teachers.md#present-teacher-readiness-and-the-smallest-real-path), `crates/jope-bench/src/panel/teaching.rs (pending)`.

**Qualification limit:** Execute separately for 230_machine_cpu and 712_chscase_geyser1; the prospective choice predates losses. Both are frozen-panel regression datasets, so if either is used to manufacture teachers, later evaluations on it must be labelled exposed/adaptive ([§](learning.md#freezing-the-experiment-and-choosing-families); OD-8 in [§](plan.md#proposed-owner-decisions-and-external-dependencies)). Old operational receipts from sources 55e9ed688faf, 3fd7d5b2ebb6 and 68f56404c3e1 cannot be retrofitted. The older queue suggested 300 seconds per family without admitting that allowance.

#### TEACH-04 — real corpus materialization

**Priority:** P0. **Current evidence:** native exporter prepared. **Status:** blocked on two real families.

**Action:** Build the request from every eligible requested comparison and its original native-context proof; freeze a family-disjoint corpus and the exact vocabulary.

**Dependencies:** TEACH-03.

**Acceptance commands:**

```sh
"$J" train prepare-inputs --learning-state "$LEARNING_STATE" --request "$CORPUS_REQUEST" --request-sha256 "$CORPUS_REQUEST_SHA" --output "$CORPUS_OUTPUT"
```

**Required evidence:** At least two real families; every eligible requested comparison included; seed 7 partition derived from source families; exact 1536 + 512 context values. No invented context, parameter-bin snapping, label remapping or omission of difficult examples.

**Resources:** Read-only/static custody work; no fitting, models, GPU or campaign admission.

**Source/evidence references:** `crates/jope-cli/src/train/corpus.rs (pending)`, [§](training.md#build-the-real-corpus-from-admitted-evidence).

**Qualification limit:** Only if exactly two families qualify does the recorded machineCPU-development/geyser-training assignment apply. More eligible families must all remain admitted.

#### TRAIN-01 — actual trainer tests

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

#### TRAIN-02 — native supervised training

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

**Resources:** Explicitbounded CPU fixture allocation; same planned 3600 seconds/1200scaps are maxima, not authorization to spend them twice; no GPU.

**Source/evidence references:** `crates/jope-cli/src/train.rs (pending)`, `python/src/jope/policy/training_campaign.py (pending)`.

**Qualification limit:** Author the cancellation harness against the actual front door and target only its owned processes. CPU mode requires explicit fixture provenance.

#### TRAIN-03 — bounded real GPU training

**Priority:** P0. **Current evidence:** implemented follow-on; no GPU campaign started. **Status:** blocked on admitted data and consumption readiness.

**Action:** Run one all-admitted CUDA campaign with real regression and two passes. Retain synthetic classifier fallback separately and report every unavailable category.

**Dependencies:** TEACH-04, TRAIN-02, NEURAL-02.

**Acceptance commands:**

```sh
"$J" train plan --state "$TRAIN_STATE" --python "$TRAINING_PYTHON" --inputs "$CORPUS_OUTPUT/inputs.json" --task all-admitted --device cuda:0 --source-sha "$SOURCE_SHA" --learning-state "$LEARNING_STATE" --seed 7 --passes 2
"$J" train run --state "$TRAIN_STATE"
```

**Required evidence:** Original global/category windows; one RTX3090 UUID; positive updates, changed parameter hashes and task coverage with data_kind/fallback reasons. Save and reload model plus optimizer; record CPU/GPU cost where measured. Requested CUDA must not silently fall back to CPU.

**Resources:** One original 3600 seconds total window;≤1200 seconds per category including both passes;one RTX 3090,20CPU,64 GiB,zero swap,1Torch thread;no exclusiveCPU CI overlap. Resume never resets.

**Source/evidence references:** `python/src/jope/policy/training_campaign.py:780 (pending)`, [§](training.md#plan-run-resume-verify-and-export).

**Qualification limit:** NEURAL-02 gates source and fixture readiness before the GPU hour. It does not depend on a newly trained live export, avoiding a circular prerequisite.

#### TRAIN-04 — checkpoint continuation and export

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

**Resources:** One original 3600 seconds total window;≤1200 seconds per category including both passes;one RTX 3090,20CPU,64 GiB,zero swap,1Torch thread;no exclusiveCPU CI overlap. Resume never resets.

**Source/evidence references:** `python/src/jope/policy/training_campaign.py:1203 (pending)`, `python/src/jope/policy/training_campaign.py:1303 (pending)`.

**Qualification limit:** Resume is conditional, not an instruction to start a third pass after completing the plan. A new state or plan cannot extend the approved hour.

#### NEURAL-02 — trained-checkpoint consumer readiness

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

#### NEURAL-03 — live newly trained consumption

**Priority:** P0. **Current evidence:** current accepted checkpoint origin/evolution proved onone dataset. **Status:** newly trained consumption pending.

**Action:** Pin the actual exported checkpoints and runtime/canonicalizer into the final configuration. Prove new-checkpoint proposals reach real generation-zero fits and completed descendants.

**Dependencies:** TRAIN-04, NEURAL-01.

**Acceptance commands:**

```sh
"$J" optimize proof-plan --base-plan "$BASE_PLAN" --base-plan-sha256 "$BASE_ID" --configuration "$TRAINED_CONFIGURATION" --configuration-sha256 "$TRAINED_CONFIGURATION_SHA" --dataset 230_machine_cpu --resources "$CONSUMPTION_RESOURCES" --resources-sha256 "$CONSUMPTION_RESOURCES_SHA" --output "$CONSUMPTION_PLAN"
"$J" optimize run --plan "$CONSUMPTION_PLAN" --plan-sha256 "$CONSUMPTION_PLAN_ID" --manifest "$M" --datasets "$D" --journal "$CONSUMPTION_JOURNAL"
"$J" optimize report --plan "$CONSUMPTION_PLAN" --plan-sha256 "$CONSUMPTION_PLAN_ID" --manifest "$M" --datasets "$D" --journal "$CONSUMPTION_JOURNAL"
```

**Required evidence:** The exported checkpoint hash appears in actual proposal and fit evidence;64+16 versus 80 origins remain authenticated. Retain completed descendants where claimed, predictions and fitted states. Regression must be admitted and classifier fixture limits explicit; replay the original window successfully.

**Resources:** Separate explicit resource manifest and approved allowance required before admission;20 CPUs/workers maximum,64 GiB maximum,zero swap,one thread,30 seconds per fit,120 seconds per arm maximum. No new allowance granted by this checklist.

**Source/evidence references:** `crates/jope-bench/src/panel/experimental_proposal.rs (pending)`, `crates/jope-bench/src/panel/origin_proof.rs (pending)`.

**Qualification limit:** The current proof-plan scopes one regression dataset. Classification needs its implemented campaign/configuration route; this command alone cannot prove binary or multiclass live consumption.

#### PACK-01 — packed training consumer and incremental loading

**Priority:** P2. **Status:** Legacy binary store exists; pending Python category route still materializes JSONL.

**Action:** Design and qualify a lossless packed reader for the bounded trainer, sharing repeated contexts and immutable shards while excluding sealed-family context vectors; measure host/device loading and fresh-example exposure.

**Dependencies:** TEACH-04, TRAIN-01.

**Required evidence:** JSONL-versus-packed token/context/label/sampler equivalence; corruption/schema tests; finite complete labels; startup/RSS/I/O/padding measurements; incremental shard reuse and actual optimizer input counts.

**Commands:** No complete executable front door is claimed; use the linked chapter’s current/pending interface map.

**Resources:** Source/fixture planning first; any measured training remains inside a separately declared remaining authorized window.

**Sources:** [§](data-estate.md#record-layout), `python/src/jope/policy/train_v3.py`, `crates/jope-engine/src/teaching_store.rs`.

#### LOOP-01 — repeatable measured learning flywheel

**Priority:** P2. **Status:** New complete improved-target-to-improved-search cycle not yet qualified.

**Action:** Demonstrate linked cycles in which newly eligible target improvements are actually sampled, change parameters, and yield measured next-search benefit against both parent checkpoint and generic search.

**Dependencies:** NEURAL-03, TEACH-04, TRAIN-04.

**Required evidence:** Per-cycle target deltas, actual fresh exposures, component losses on fixed monitors, optimizer/parent/checkpoint hashes, full failures/cost and matched fitted-program outcomes; promotion separately decided.

**Commands:** No complete executable front door is claimed; use the linked chapter’s current/pending interface map.

**Resources:** Original one GPU-hour cap remains binding. Further campaigns require their own explicit allocation; this item grants none.

**Sources:** [§](training-history.md#dated-lineage-and-evidence-coverage), [§](plan.md#the-continuation-interfaces-that-exist-today), [§](learning.md#what-better-and-better-must-mean).

### Stage 4: portable fitted inference

#### INFER-01 — fitted state and privacy

**Priority:** P0. **Current evidence:** native fitted state exists; product28 source prepared. **Status:** qualification pending.

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

**Source/evidence references:** `crates/feat-math/src/advanced_v3/fitted_portable.rs (pending)`, `crates/jope-bench/src/panel/inference.rs (pending)`, [§](inference.md#state-restrictions-beyond-the-head-table).

**Qualification limit:** The private C++ head gate is broader than the public gate. Python/Rust 14 of 26, C++ 2 of 26 and JavaScript 6 of 26 are static head counts, not runtime qualification rates.

#### INFER-02 — exact selected-winner portability

**Priority:** P1. **Current evidence:** winner/add-on source + statictests. **Status:** actual export/parity unrun.

**Action:** Qualify the selected existing fold state in order: Python, Rust, the same C++ emitter under C++17 and C++20, then JavaScript. Preserve unsupported external winners and the complete ranking.

**Dependencies:** INFER-01, PANEL-01.

**Acceptance commands:**

```sh
"$J" benchmark export-winner --plan "$PLAN" --plan-sha256 "$PLAN_ID" --manifest "$M" --datasets "$D" --journal "$JOURNAL" --dataset "$DATASET_ID" --fold "$FOLD"
"$J" benchmark qualify-export --prepared "$PREPARED_EXPORT" --prepared-sha256 "$PREPARED_EXPORT_ID" --tools "$TOOLS" --tools-sha256 "$TOOLS_SHA"
"$J" benchmark report --plan "$PLAN" --plan-sha256 "$PLAN_ID" --manifest "$M" --datasets "$D" --journal "$JOURNAL"
```

**Required evidence:** Exact state, artifact, input, class, toolchain and numerical-policy hashes; actual five-backend outcomes. Python/Rust/C++ use bitwise comparison; JavaScript uses 1e-12 absolute and relative tolerance. Recover interrupted work without retrying its child; read-only reports launch nothing. Unsupported never replaces the winner.

**Resources:** Same original campaign execution window and worker caps; no new job after t0 + 6900 seconds for official panel. No post-panel allowance is implemented.

**Source/evidence references:** `crates/jope-bench/src/panel/fitted_export.rs (pending)`, `crates/jope-bench/src/panel/fitted_export_run.rs (pending)`, [§](inference.md#example-c-exact-fitted-winner-export-after-pending-integration).

**Qualification limit:** Library fixtures precede the panel. Actual selected winners require complete confirmations from PANEL-01 and qualification before the original execution cutoff. No separately bounded post-panel qualification front door exists. One fitted fold does not qualify the other four states.

#### FUTURE-01 — CUDA inference and serving promotion

**Priority:** P2. **Current evidence:** not qualified in this delivery. **Status:** deferred.

**Action:** After portable CPU backends, design CUDA fitted-inference qualification and independent experimental-to-serving promotion. Preserve the accepted baseline until promotion passes.

**Dependencies:** INFER-02, REPORT-01.

**Command boundary:** no truthful executable acceptance command yet; implement or obtain missing interface first.

**Required evidence:** New backend contract, fixed numerical policy, exact device/toolchain/state parity and promotion evaluation; reviewed pointer compare-and-swap and rollback where used.

**Resources:** NoCUDAinference/promotionbudgetadmitted bythisroadmap.

**Source/evidence references:** `docs/adr/ADR-0005-canonical-learning-loop.md`, [§](training.md#what-the-pending-campaign-trains), [§](training.md#compatibility-checks-and-fail-closed-rules).

**Qualification limit:** Training GPU availability does not prove CUDA inference parity. Export is not serving promotion.

### Stage 5: installation and measured demonstration

#### INSTALL-01 — private installation

**Priority:** P1. **Current evidence:** installer source and 15 metadata tests passed. **Status:** cold/live path pending.

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

**Source/evidence references:** `deploy/install-private.py (pending)`, [§](installation.md#build-and-install).

**Qualification limit:** An installer receipt is not policy/context-worker build authority. Do not include training credentials or model bodies in the public inference bundle.

#### INSTALL-02 — final input/runtime preparation

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

**Source/evidence references:** [§](installation.md#prepare-the-panel-and-external-runtimes), `agent-acceptance-queue/QUEUE.md`.

**Qualification limit:** Packaging commands depend on final reviewed artifacts and the actual virtual environment. Preserve versions and file identities; changing the environment invalidates a pinned campaign.

#### DEMO-01 — measured warm fitted demonstration

**Priority:** P1. **Current evidence:** warmtargetadd-onprepared;historicalfirstfit≈50.4s. **Status:** real trained warm proof pending.

**Action:** Produce actual trained-neural and generic fitted artifacts, reloaded predictions, baseline/comparison and truthful target_met. Do not clip arm budgets to pass sixty seconds.

**Dependencies:** NEURAL-03, INSTALL-02.

**Acceptance commands:**

```sh
"$J" demo --asm-plan "$WARM_PLAN" --asm-plan-sha256 "$WARM_PLAN_ID" --manifest "$M" --datasets "$D" --output "$WARM_OUTPUT" --export python --export rust
python3 deploy/ci/qualify-demo.py --binary "$J" --sha256 "$J_SHA" --source-sha "$SOURCE_SHA" --output "$ACCEPTANCE/generic-three-task" --require-under-60
```

**Required evidence:** Actual fitted state, predictions and baseline; both full paired allowances; start/end, warm elapsed, execution_complete and target_met. Keep generic synthetic qualification separate from trained-neural proof.

**Resources:** Separate explicit resource manifest and approved allowance required before admission;20 CPUs/workers maximum,64 GiB maximum,zero swap,one thread,30 seconds per fit,120 seconds per arm maximum. No new allowance granted by this checklist.

**Source/evidence references:** `crates/jope-bench/src/panel/demo_asm.rs (pending)`, `docs/private-warm-target.md (pending)`, [§](search.md#warm-fitted-demonstration-after-integration).

**Qualification limit:** Under sixty seconds is the declared goal, not a promise; no qualifying measurement exists yet. The existing twenty-second-arm example is unqualified. An explicitly admitted longer proof may satisfy execution while missing the target. This checklist grants no implicit 300-second allowance.

#### PANEL-01 — unchanged thirty-dataset campaign

**Priority:** P1. **Current evidence:** protocol/currentfoundation qualified;full panel unstarted. **Status:** pending final trained configuration.

**Action:** Freeze the final roster/configuration and run the unchanged thirty datasets once. Screen fold0, freeze the top five, confirm five development folds, and add asm.neural diagnostically when absent. Retain every failure.

**Dependencies:** BENCH-01, BENCH-02, NEURAL-03, DEMO-01, INFER-01, INSTALL-02.

**Acceptance commands:**

```sh
"$J" optimize plan --base-plan "$BASE_PLAN" --base-plan-sha256 "$BASE_ID" --configuration "$TRAINED_CONFIGURATION" --configuration-sha256 "$TRAINED_CONFIGURATION_SHA" --output "$PANEL_PLAN"
"$J" benchmark run --plan "$PANEL_PLAN" --plan-sha256 "$PANEL_PLAN_ID" --manifest "$M" --datasets "$D" --journal "$PANEL_JOURNAL"
"$J" benchmark report --plan "$PANEL_PLAN" --plan-sha256 "$PANEL_PLAN_ID" --manifest "$M" --datasets "$D" --journal "$PANEL_JOURNAL"
```

**Required evidence:** All thirty dataset identities, ten per task; every applicable method slot; screen/frozen/confirmation losses; selection-exposure row intersection hashes; no sealed reads; winner backend outcomes and actual win counts.

**Resources:** Exactly 7200 seconds including 300 seconds reporting reserve;20 CPUs/workers,64 GiB,zero swap,one thread per fit,30 seconds per fit,120 seconds per search arm; sealed labels unopened; existing window never reset.

**Source/evidence references:** `benchmarks/panel.json`, `crates/jope-bench/src/panel/protocol.rs`, `crates/jope-bench/src/panel/selection_exposure.rs`.

**Qualification limit:** Current code forces asm.neural into extra confirmation, not asm.generic. Selected five-fold summaries are adaptive development, and ten wins per category are not promised. Qualify exports before the original 6900-second cutoff or preserve unattempted status.

#### REPORT-01 — complete measured report

**Priority:** P1. **Current evidence:** one-dataset, 832 predictor/164recipe operational audit. **Status:** full campaign report pending.

**Action:** Publish all thirty losses/gaps, outcomes, time/cost per unique artifact and recipe, coverage and failure inventories. Compare the supplied one-hour/ten-million-token scenario only using supported quantities.

**Dependencies:** PANEL-01, EXEC-05.

**Acceptance commands:**

```sh
"$J" benchmark report --plan "$PANEL_PLAN" --plan-sha256 "$PANEL_PLAN_ID" --manifest "$M" --datasets "$D" --journal "$PANEL_JOURNAL"
```

**Required evidence:** Report recovery without fits; union of started/finished records; worker CPU distinguished from overlapping service CPU; known and unknown costs; training, installation, preparation and export accounted separately. Tokens and money stay unknown unless measured; retain win/loss/gap tables.

**Resources:** Original 300 secondsreportingreserve forpanel; laterreadonlyauditopensnofitbudget.

**Source/evidence references:** [§](scheduling.md#worked-example-cpu-wall-time-and-per-predictor-cost-of-the-one-dataset-proof), [§](status.md#canonical-proof-record-and-per-arm-facts), `crates/jope-bench/src/panel/asm_campaign.rs (pending)`.

**Qualification limit:** The one-hour comparison supplies wall-duration context, not a matched speedup. Unique artifact hashes do not establish behavioral uniqueness. Never add aggregate service CPU to worker CPU already included in it.

### Stage 6: eligible optimization and reuse

#### TEACH-01 — HWM comparison identity

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

#### TEACH-02 — label exposure readers

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

**Qualification limit:** The panel at source 68f56404c3e1 has no benchmark-valid producer. Separate historical closure questions remain: indexed meta-test contexts and externally reported exposed high-water marks do not by themselves establish training-label leakage. A burn writer alone is not downstream enforcement.

#### MOTIF-01 — eligible reusable fragments

**Priority:** P1. **Current evidence:** separate typed library branch exists. **Status:** mining front door incomplete.

**Action:** Integrate the production eligible-State reader, typed fragment/macro miner and umbrella CLI. Preserve full comparison identity, provenance and private/public separation.

**Dependencies:** TEACH-01, TEACH-02, TEACH-03.

**Acceptance commands:**

```sh
"$J" motifs mine --help
"$J" motifs validate --help
```

**Required evidence:** Actual mine CLI contract/help; eligible-input identity inventory; deterministic typed fragment output. Exclude rejected, no-signal, unbound and sealed examples.

**Resources:** Read-only/static custody work; no fitting, models, GPU or campaign admission.

**Source/evidence references:** [§](status.md#pending-acceptance), `agent-acceptance-queue/QUEUE.md:motifs`.

**Qualification limit:** These help commands are unavailable at source 68f56404c3e1 and at main a75e543d2166, which has the same tree; they are acceptance targets. No truthful exact mine invocation exists yet; record it after implementation.

#### MOTIF-02 — reuse validation

**Priority:** P1. **Current evidence:** validationlibrary/preparation only. **Status:** live reuse pending.

**Action:** Validate fragments/macros on independent host contexts, lossless roundtrips, deterministic fit compatibility and privacy. Measure reuse benefit under the full comparison identity.

**Dependencies:** MOTIF-01.

**Command boundary:** no truthful executable acceptance command yet; implement or obtain missing interface first.

**Required evidence:** Exact implemented validate command and input table; at least three hosts per fragment where its existing contract requires that count; roundtrip and actual fit receipts, independent context identity and canary absence. Support frequency is separate from measured uplift.

**Resources:** Explicitnewboundedreusequalification allocation required; noresetofpanelortrainingwindows.

**Source/evidence references:** `agent-acceptance-queue/QUEUE.md:motifs`, `docs/adr/ADR-0005-canonical-learning-loop.md`.

**Qualification limit:** Do not invent mine flags. The historical validate argument shape alone does not establish umbrella wiring or production State admission.

#### MOTIF-03 — new operators

**Priority:** P2. **Current evidence:** useracceptance requirement. **Status:** future conditional work.

**Action:** Admit new operators only after measured ablation and grammar, checkpoint and backend compatibility tests. Preserve the old vocabulary identity.

**Dependencies:** MOTIF-02.

**Command boundary:** no truthful executable acceptance command yet; implement or obtain missing interface first.

**Required evidence:** Preregistered ablation recipes, budget and contexts; measured marginal gain; task/type/codec/emitter/vocabulary compatibility; new identity and migration policy.

**Resources:** Separatelypreregistered ablationbudget; noneadmitted here.

**Source/evidence references:** `user approval plan:Eligible optimization and reuse`, [§](inference.md#future-work-and-acceptance-boundaries).

**Qualification limit:** Typed reusable macros may stay within existing operators. Fragment frequency alone does not justify adding an operator.

#### FOCUS-01 — sustained single-dataset improvement

**Priority:** P2. **Status:** Current bounded proof exists; no incumbent-seeded attack front door.

**Action:** Add an authenticated incumbent-aware multi-wave request using full comparison identity, retained original deadline, bounded workers and durable improvement events. Preserve exploration and all non-improvement outcomes.

**Dependencies:** TEACH-01, TEACH-02, TEACH-03.

**Required evidence:** Improving, tied, worse, stale and concurrent incumbents; full OOF/baseline admission; no deadline reset or sealed input; successful no-improvement completion; actual cost per eligible improvement.

**Commands:** No complete executable front door is claimed; use the linked chapter’s current/pending interface map.

**Resources:** No indefinite loop or additional allowance implied; whole focused experiment has one predeclared finite budget.

**Sources:** [§](search.md#focused-single-dataset-search), `crates/jope-bench/src/panel/origin_proof.rs`, `crates/jope-pipeline/src/neural_search.rs`.

## Coordinated future backlog: 24 proposals

These are Claude’s coordinated follow-on proposals, retained alongside their overlap with the primary checklist. They are not 24 additional completed features, new budgets, or modifications to the frozen panel. Proposed `OD-*` decisions have not been silently adopted. No sealed evaluation is authorized by this list. The JSON/property list retains each proposed owner and original acceptance description.

### Documentation, onboarding and repository hygiene

#### Proposal HANDOFF-01

**Proposed owner:** Claude. **Priority:** P1. **Primary overlap:** DOC-01, DOC-05.

**Proposed acceptance:** track-a-20260922/manual-20260923/release/SHA256SUMS verifies; FINDINGS.json posted; sections 06/07/09/11 and figures/tools/data handed off

**Disposition:** Track with the overlapping primary item; exact implementation and acceptance remain pending.

#### Proposal HYG-01

**Proposed owner:** Claude. **Priority:** P1. **Primary overlap:** INT-02.

**Proposed acceptance:** cargo metadata shows workspace default members; offline_cli cpp20 test passes; jope-bench run defaults to min(20, cores)

**Disposition:** Track with the overlapping primary item; exact implementation and acceptance remain pending.

#### Proposal HELLO-01

**Proposed owner:** Claude docs, Codex verbs. **Priority:** P1. **Primary overlap:** DEMO-01.

**Proposed acceptance:** a quickstart of at most 5 commands routing to jope demo runs top to bottom from a clean checkout; transcript retained

**Disposition:** Track with the overlapping primary item; exact implementation and acceptance remain pending.

#### Proposal B6-01

**Proposed owner:** Claude. **Priority:** P2. **Primary overlap:** MAINT-01.

**Proposed acceptance:** A2 seed diagnostic and retained-evidence asserts; spec_drift repointed (3 ids); sotarena repoint; scratch budget 900/1500 s; neural-env.sh resolves CUDA_HOME; 13 ids added to known-broken-tests.txt

**Disposition:** Track with the overlapping primary item; exact implementation and acceptance remain pending.

#### Proposal DOCFIX-01

**Proposed owner:** Claude patch, Codex integrates. **Priority:** P1. **Primary overlap:** DOC-01.

**Proposed acceptance:** learning.md, architecture.md, quickstart.md and datasets.md corrected; AGENTS bucket table with sizes; catalogs/s3.json generated from DATA-01

**Disposition:** Track with the overlapping primary item; exact implementation and acceptance remain pending.

#### Proposal LAYOUT-01

**Proposed owner:** Claude. **Priority:** P2. **Primary overlap:** INT-02.

**Proposed acceptance:** crate package names equal directories; jope-contracts shared types; fewer duplicate Task enums/fold planners

**Disposition:** Track with the overlapping primary item; exact implementation and acceptance remain pending.

### Privacy and naming

#### Proposal IP-Q1

**Proposed owner:** Claude. **Priority:** P1. **Primary overlap:** IP-01.

**Proposed acceptance:** CLI help/doctor/README/NOTICE/docs/Cargo descriptions clean; ip-scrub covers NOTICE and schemas; new lint tests fail on old main and pass after

**Disposition:** Track with the overlapping primary item; exact implementation and acceptance remain pending.

#### Proposal IP-Q2

**Proposed owner:** Claude. **Priority:** P1. **Primary overlap:** IP-01.

**Proposed acceptance:** emitter text 2.3.0 with re-pinned fixtures; manifest key reader switch; codec serde aliases

**Disposition:** Track with the overlapping primary item; exact implementation and acceptance remain pending.

#### Proposal IP-Q3

**Proposed owner:** Claude. **Priority:** P2. **Primary overlap:** IP-01.

**Proposed acceptance:** receipts v3 and PR 83 ids renamed with a dual-name journal loader

**Disposition:** Track with the overlapping primary item; exact implementation and acceptance remain pending.

#### Proposal IP-Q4

**Proposed owner:** Claude. **Priority:** P2. **Primary overlap:** IP-01, PANEL-01.

**Proposed acceptance:** same-cut v3 evidence; afm.* method id renamed

**Disposition:** Track with the overlapping primary item; exact implementation and acceptance remain pending.

#### Proposal IP-Q5

**Proposed owner:** Claude. **Priority:** P2. **Primary overlap:** IP-01.

**Proposed acceptance:** publication boundary: private-path rule for docs/internal/** and docs/asm.md plus a packaging exclusion check

**Disposition:** Track with the overlapping primary item; exact implementation and acceptance remain pending.

#### Proposal NAME-01

**Proposed owner:** Claude. **Priority:** P1. **Primary overlap:** IP-01.

**Proposed acceptance:** no public AFM/ASPFM/Prime/Lime text; lint enforces

**Disposition:** Track with the overlapping primary item; exact implementation and acceptance remain pending.

### Data, teachers and training

#### Proposal DATA-01

**Proposed owner:** Claude. **Priority:** P1. **Primary overlap:** DOC-01, DOC-05.

**Proposed acceptance:** metadata-only inventory of archive-use2, usw2 replicas, custody dev/prod (sealed: counts only, after Codex agrees), bucket configuration, learning per-prefix reconciliation, replica parity, local roots, single-copy risk list, disk headroom; JSON in the s3-estate schema; zero body reads

**Disposition:** Author stopped the inventory at 19:30 UTC; this manual includes only the independently checked aggregate subset and 19 qualified risk summaries. The raw packet retains a failed sealed-name gate and is excluded. DOCFIX-01 remains pending; no object-body or numerical access.

#### Proposal DATA-02

**Proposed owner:** Claude design, Codex prepares. **Priority:** P1. **Primary overlap:** TEACH-04.

**Proposed acceptance:** frozen teaching manifest: seeded, selected before outcomes, families disjoint from all 30 panel families

**Disposition:** Track with the overlapping primary item; exact implementation and acceptance remain pending.

#### Proposal DATA-03

**Proposed owner:** Owner, Claude. **Priority:** P2. **Primary overlap:** TEACH-03.

**Proposed acceptance:** licence-known multiclass corpus with family registry and frozen splits

**Disposition:** Track with the overlapping primary item; exact implementation and acceptance remain pending.

#### Proposal TEACH-05

**Proposed owner:** Codex runs, Claude audits. **Priority:** P1. **Primary overlap:** TEACH-03.

**Proposed acceptance:** per-wave admission and refusal receipts; every admitted family counted

**Disposition:** Track with the overlapping primary item; exact implementation and acceptance remain pending.

#### Proposal TRAIN-05

**Proposed owner:** Codex. **Priority:** P2. **Primary overlap:** TRAIN-03, LOOP-01.

**Proposed acceptance:** second GPU campaign; consumption reproven

**Disposition:** Second campaign is outside the existing one-GPU-hour allowance; no additional budget or deadline is granted.

#### Proposal HWM-01

**Proposed owner:** Codex implements, Claude designs and tests. **Priority:** P2. **Primary overlap:** FOCUS-01.

**Proposed acceptance:** incumbent-seeded single-dataset attack; replacement only when the candidate beats the incumbent on at least 4 of 5 folds by the margin; negative tests for stale identity, tie, test-high/OOF-flat and burned partition

**Disposition:** FOCUS-01 covers the user’s focused-search need. Extra incumbent four-of-five-fold rule is a proposed policy, separate from required baseline admission.

### Evaluation, export, reuse and cost

#### Proposal SEALED-01

**Proposed owner:** Codex runs, owner authorizes. **Priority:** P1. **Primary overlap:** REPORT-01.

**Proposed acceptance:** preregistered one-shot sealed evaluation; burn receipts; no reselection

**Disposition:** Outside current execution scope: frozen sealed evaluation remains untouched. This proposal does not authorize opening it.

#### Proposal EMIT-01

**Proposed owner:** Claude. **Priority:** P1. **Primary overlap:** INFER-02.

**Proposed acceptance:** fitted C++17/20 and JS coverage for winning families with library parity fixtures

**Disposition:** Track with the overlapping primary item; exact implementation and acceptance remain pending.

#### Proposal INFER-03

**Proposed owner:** Codex. **Priority:** P2. **Primary overlap:** INFER-02.

**Proposed acceptance:** post-panel export qualification in its own admitted window

**Disposition:** Track with the overlapping primary item; exact implementation and acceptance remain pending.

#### Proposal EMIT-03

**Proposed owner:** Claude. **Priority:** P2. **Primary overlap:** INFER-01.

**Proposed acceptance:** language-neutral DSL semantics spec, golden corpus, numeric-policy conformance harness

**Disposition:** Track with the overlapping primary item; exact implementation and acceptance remain pending.

#### Proposal MOTIF-00

**Proposed owner:** Claude. **Priority:** P1. **Primary overlap:** MOTIF-01.

**Proposed acceptance:** jope motifs mine|validate --help exit 0; deterministic fixture output; --locked build and tests

**Disposition:** Track with the overlapping primary item; exact implementation and acceptance remain pending.

#### Proposal COST-01

**Proposed owner:** Owner constants, Claude table. **Priority:** P1. **Primary overlap:** REPORT-01.

**Proposed acceptance:** labelled scenario table separate from the measured table; no quality-parity claim

**Disposition:** Track with the overlapping primary item; exact implementation and acceptance remain pending.

## Planned execution and review ownership

Codex owns canonical integration and numerical admission. Claude contributes source and evidence work in the serial queue. A contributor does not become the independent native reviewer of that same change; the final review is assigned to a separate eligible process. The table below is the current assignment; the original ownership proposal is retained unchanged as history in `data/coordinated-roadmap-proposal.json`.

| Item | Execution | Claude contribution / review scope |
| --- | --- | --- |
| LIFE-01 | Completed PR83 lifecycle | Independent native review retained in lifecycle evidence. |
| DOC-01 | Codex canonical integration | Contribution/source findings only; exact-head native review uses a separate non-contributor Claude process. |
| DOC-02 | Codex (done 16:46Z) | none |
| DOC-03 | Codex | Fresh non-contributing independent native reviewer |
| DOC-04 | Codex | Fresh non-contributing independent native reviewer |
| INT-01 | Codex | pre-review of each refreshed replay; native review at exact head |
| INT-02 | Codex | pre-review; native review |
| BENCH-01 | Codex | read-only receipt audit |
| BENCH-02 | Codex | read-only receipt audit |
| EXEC-01 | Codex | review |
| EXEC-02 | Claude writes the tests; Codex runs them under the CPU lock | custody patch (drain-and-fixtures-F2: async reap, pre-gate BudgetExhausted, undrained group) |
| EXEC-03 | Claude writes the tests; Codex runs them | custody patch (resume-fix-F2: PoolBackend receipt through verify_receipt, Some(shared fold digest)) |
| EXEC-04 | Claude writes the tests; Codex runs them | custody patch (regressions-F2: assert original_deadline stop reason and persisted search receipt) |
| EXEC-05 | Codex | review |
| NEURAL-01 | Codex | receipt audit |
| TEACH-01 | Codex | review |
| TEACH-02 | Codex | optional custody patch: production caller wiring for require_unburned_test_partition plus an integration test |
| TEACH-03 | Codex | audit that freeze/vocabulary pins precede fits; independently recompute per-fold gains and the 4-of-5 rule |
| TEACH-04 | Codex | audit that every eligible comparison is included |
| TRAIN-01 | Codex | review |
| TRAIN-02 | Codex | review |
| TRAIN-03 | Codex | review |
| TRAIN-04 | Codex | review |
| NEURAL-02 | Codex | receipt audit |
| NEURAL-03 | Codex | audit that the new checkpoint hash appears in proposal and fit evidence |
| INFER-01 | Codex | review of emitter/parity parts |
| INFER-02 | Codex | review of emitter/parity parts |
| INSTALL-01 | Codex | new-user walkthrough of quickstart and doctor output |
| INSTALL-02 | Codex | new-user walkthrough |
| DEMO-01 | Codex | wins/cost table spec; independent recomputation |
| PANEL-01 | Codex | independent read-only recomputation from the journal |
| REPORT-01 | Codex | independent wins/losses and cost recomputation; every loss published |
| MOTIF-01 | Claude code, Codex live run | via MOTIF-00 |
| MOTIF-02 | Claude design, Codex windows | preregistered ablation design |
| MOTIF-03 | Claude design, Codex windows | two-stage promotion design |
| IP-01 | Claude | IP-Q1..Q5 |
| MAINT-01 | Claude (B6-01); Codex runs full lanes | author |
| FUTURE-01 | deferred | CUDA design after EMIT-01 |
| DOC-05 | Codex integration | Independent source/data cross-check |
| DOC-06 | Codex integration | Independent source/data cross-check |
| PACK-01 | Codex integration | Independent source/data cross-check |
| FOCUS-01 | Codex integration | Independent source/data cross-check |
| LOOP-01 | Codex integration | Independent source/data cross-check |

## Source index

### Current implementation references

These links bind the reviewed implementation commit `68f56404c3e1`, not a moving branch. PR83 merged it on 2026-09-23 as main `a75e543d2166`, which has the same source tree (`176b75bc36d7`), so each pinned file is also the file at that main commit. Hashes below are SHA256 of file bytes; Git commit/tree identifiers use Git’s separate object identity. A link needs access to the private repository.

| Current source | SHA256 |
| --- | --- |
| [benchmarks/panel.json](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/benchmarks/panel.json#L1) | `ba7dace4cd535687002cc95538dbc7f7aaf77ec973177c7159b3c6d22471eee0` |
| [benchmarks/prepare.py](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/benchmarks/prepare.py#L1) | `7600de95f0769309d814d80c6723d65f0588124b3e0d1a9875d8155fcc4a91c9` |
| [crates/feat-math/src/advanced_v3/fitted_cpp.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/feat-math/src/advanced_v3/fitted_cpp.rs#L1) | `c20d7e9e7d530102435bdfd07a7c5275de041742a38d681842b87147e6c82fef` |
| [crates/feat-math/src/emit_cpp.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/feat-math/src/emit_cpp.rs#L1) | `531e4d006caac9c41f3934df8f0f84d1f62749a9380ab4b67d1ec21180e65157` |
| [crates/feat-math/src/emit_js.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/feat-math/src/emit_js.rs#L1) | `2e18c3a0fa85ff4fb0013d610be9404079956d9331c9988486b828bc7ac805cf` |
| [crates/feat-math/src/emit_py.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/feat-math/src/emit_py.rs#L1) | `138a8c2a279da1b9df7b3c155606ac7e18c145982400ca7ace40f120cf386ca6` |
| [crates/feat-math/src/emit_rs.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/feat-math/src/emit_rs.rs#L1) | `92d1277a90886073ba5b3aa19aa9fb461f97ad3290c2c4a91a1ccaeec7c1d405` |
| [crates/feat-math/src/transpile.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/feat-math/src/transpile.rs#L1) | `374d7c1e54c9199b932963f564306deec8ffcc1f0ac7add5554fdcee04e0892a` |
| [crates/jope-bench/src/panel/campaign.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-bench/src/panel/campaign.rs#L1) | `95e162538f7601f40bd2e7019639a6622259e516fa38cd1f92e83a0d4117b20c` |
| [crates/jope-bench/src/panel/campaign_cli.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-bench/src/panel/campaign_cli.rs#L18) | `419262eab7516ba0ede8432a5d1f162fd4d64047681764e55edab9e438bb2ad1` |
| [crates/jope-bench/src/panel/cli.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-bench/src/panel/cli.rs#L1) | `f1f9879d0b2de5473f8fcd9b3722e5c41147d5e33b26adac0948d0f4541a7b21` |
| [crates/jope-bench/src/panel/methods.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-bench/src/panel/methods.rs#L1) | `170b3fcd3fadc140c5d3006e34372a8042d6c8939395a3f633085b884e1cdaa6` |
| [crates/jope-bench/src/panel/optimize_cli.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-bench/src/panel/optimize_cli.rs#L1) | `17af561a039fd4832dd1bd23c226bbde5065b09b5273eb6e0b1353899042bd80` |
| [crates/jope-bench/src/panel/receipt_timing.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-bench/src/panel/receipt_timing.rs#L1) | `0a479d36f84c9c22381310e9dd83ca4cf4c48594a7541c666587c0226f4a7367` |
| [crates/jope-bench/src/panel/score.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-bench/src/panel/score.rs#L1) | `10093ed25a81940c7ab985843e742efc49990f3d361c92764d6d6dcae07e81e9` |
| [crates/jope-bench/src/panel/selection_exposure.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-bench/src/panel/selection_exposure.rs#L9) | `a591787e6eaa95bb2e11d4c46776707076016b06785c4fad1dbc71d1a4d9fc8e` |
| [crates/jope-cli/src/kernel.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-cli/src/kernel.rs#L1) | `d8f26b2eb49652d61600c699155e7b69664bd0e868e1fc76229da012e6e547c8` |
| [crates/jope-cli/src/main.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-cli/src/main.rs#L1) | `4b2990e5294d0668a8d5f7a6c1e809d91fbb93be1418fc3fb54f291c88251265` |
| [crates/jope-cli/src/schema.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-cli/src/schema.rs#L1) | `307ce6a4fa3c22bef7c15e7272bd834100ba78715eba7b1c0547a2e1b6b04d1d` |
| [crates/jope-paths/src/lib.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-paths/src/lib.rs#L1) | `abc95bfa0e258642e0101b2f82c07706492dfe1d6c8a6fd99043a475893aa120` |
| [crates/jope-pipeline/src/cli.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-pipeline/src/cli.rs#L1) | `1bd10526c2d58c233f5bb1e9bd575bb5c7afa82691b2238ad2f6993f88bd6ae8` |
| [crates/jope-pipeline/src/cpu_context_bridge.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-pipeline/src/cpu_context_bridge.rs#L18) | `48dbe30216bf42520df668ca5b3d4a4138d21ae2731ba07c349fe6ec232729ff` |
| [crates/jope-pipeline/src/dataset.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-pipeline/src/dataset.rs#L1) | `3f34436e6d0bbb747c846d63dfe3bcbc4e1ce4e34863caa37d7dce0c7183e7ba` |
| [crates/jope-pipeline/src/emit.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-pipeline/src/emit.rs#L1) | `bb351af0947653caf9bfb5d276efd8b2645599503a101f1100b9fc20bb07edb9` |
| [crates/jope-pipeline/src/execution.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-pipeline/src/execution.rs#L1) | `e38f68bbb765918f421e69d82fa697d822d60a53eb10c51e30cc9f1ec8066a76` |
| [crates/jope-pipeline/src/experiment.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-pipeline/src/experiment.rs#L1) | `cec0f36073f73de8c09f508512c18e0098bb3f4028af08081d1b6a8639ab2e1e` |
| [crates/jope-pipeline/src/front_door.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-pipeline/src/front_door.rs#L1) | `71dc0402ff413f020a25983a5ba98386324c16af03363f5deae4fafceafd856a` |
| [crates/jope-pipeline/src/neural_search.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-pipeline/src/neural_search.rs#L27) | `a8cd55c9c4f7b5a1060eab5979deb76db24484360389f2072f6b906c4bae0390` |
| [crates/jope-pipeline/src/search_context.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-pipeline/src/search_context.rs#L8) | `e423e9676a75426cdb6fdbbaf908e80c4bc1c89c5d93007aedcdfc0189cd382e` |
| [crates/jope-pipeline/src/supervisor.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-pipeline/src/supervisor.rs#L1) | `a61265692d61580fe174c44a5f1cc2971e6512b803d91bc480d38c90a73a16b6` |
| [crates/jope-policy/src/config.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-policy/src/config.rs#L17) | `a740ca64edde353c9624f5fdc3b63878c1200ee95c159a71bca1556abc28ed37` |
| [crates/jope-policy/src/context_producer.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-policy/src/context_producer.rs#L35) | `f218d0539e273df4f53263c8ca4f06639d68456ab2057a195bf3253e27aa32d2` |
| [crates/jope-policy/src/context_producer/admission.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-policy/src/context_producer/admission.rs#L4) | `2e98364dfe3994a21ccc2d6115adb7d7396c3de3a9babf15bcae473d4d59c441` |
| [crates/jope-policy/src/context_producer/input.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-policy/src/context_producer/input.rs#L5) | `0fc3df5ed519439dd47fd27022e4be11594924a46fff4c926901cb514f5e4609` |
| [crates/jope-policy/src/producer.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-policy/src/producer.rs#L12) | `bfa9ecfe92bc6224d0409e2a66048a9858958a4833102cd57c0201fc4b083895` |
| [crates/jope-policy/src/bin/jope-policy-producer.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-policy/src/bin/jope-policy-producer.rs#L13) | `73a4678ac5fa362396f943ba02e8bc68e1fd5d00d4610d4de5ba597463fb3e6b` |
| [crates/jope-policy/src/context_producer/worker.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-policy/src/context_producer/worker.rs#L9) | `c106275c81e706c3eef8ff4644aa74efaf0d8a3e26eb97eb36b9222d052d82c4` |
| [crates/jope-policy/src/seeds.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-policy/src/seeds.rs#L42) | `434d8a0cde2e34c57291ec3657bf92707de6fbaf99f53e0fbe6f1b4aee969c15` |
| [crates/jope-policy/src/lib.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-policy/src/lib.rs#L133) | `2b0da78dcd569de31fdd0922c90e56e19e404ba9f1e184a1eb61cf634ac733a8` |
| [crates/jope-policy/src/model.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-policy/src/model.rs#L28) | `7d0f464fe4bd51ec4b2afa2257678565f959760c74bc347a7d1cebff3efe1e25` |
| [crates/jope-policy/src/ranking.rs](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/crates/jope-policy/src/ranking.rs#L22) | `92c31cc55f3943aea2246bc0191325fbd625d2019fd49ce33c54f505b751843c` |
| [deploy/ci/e2e-smoke.sh](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/deploy/ci/e2e-smoke.sh#L1) | `6becfc97830d87907bfb594aae247a117fb23162382d77bc75fbd770b4668354` |
| [deploy/ci/emitters.sh](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/deploy/ci/emitters.sh#L1) | `beb99faf01c60b4f39b0179eab46696bcd5cc10658b8cdd690962bbd35254e6b` |
| [deploy/ci/emitters/compare-bits.py](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/deploy/ci/emitters/compare-bits.py#L1) | `eaee5cc96d3d3161c85fefc02d636874adb4d399c340863534d83db5e8fe6249` |
| [deploy/ci/emitters/driver.cpp](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/deploy/ci/emitters/driver.cpp#L1) | `899ad86994c93ac37312a66acde6900948948b22bfae1dd0be1516c7e99ca416` |
| [deploy/ci/emitters/driver.js](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/deploy/ci/emitters/driver.js#L1) | `754b98bc8d694ec482e23826412e7dcbde76717e7d2f72d18a9adeb76ad12f75` |
| [deploy/prepare-panel-runtime.py](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/deploy/prepare-panel-runtime.py#L1) | `f9a0d1d9ca97dfd7664a39a8df6b1e364e738fdfaaa13a0e327402c073ee46f5` |
| [python/src/jope/policy/model_v3.py](https://github.com/veox-ai/JopeDime/blob/68f56404c3e199e06413bfd645d3fe0803d7b96f/python/src/jope/policy/model_v3.py#L93) | `11ab74d12bda8a2fb817b073a685d128eff06479f769e4301488339424314ffe` |

Current interface/semantics: `crates/jope-cli/src/main.rs`, `crates/jope-cli/src/kernel.rs`, `crates/jope-pipeline/src/cli.rs`, `crates/jope-pipeline/src/front_door.rs`, `crates/jope-paths/src/lib.rs`, `crates/jope-bench/src/panel/{cli,campaign_cli,campaign,optimize_cli,methods,score,receipt_timing}.rs`, `crates/jope-pipeline/src/{experiment,execution,supervisor,neural_search}.rs`, `deploy/prepare-panel-runtime.py`, `benchmarks/{panel.json,prepare.py}` and `AGENTS.md`.

### Pending source references

These are stopped custody patch views. They are not installed capabilities and are not linked as if they existed at the reviewed Git commit. Their patch identities and acceptance gates are named in [§](training.md#the-pending-bounded-extension-path-sequence-and-limits), [§](installation.md#command-surfaces-and-the-stopped-integration-order) and the integration card [§](#int-01--serial-integration).

| Pending path | SHA256 of inspected view |
| --- | --- |
| `agent-delivery-refresh-68f56404/replay-001/source-text/crates/jope-bench/src/panel/demo_asm.rs` | `dd1a287c4ff3c1447a20e81d2c2990913923cd48c667f3be600f5d6b3c972f62` |
| `agent-delivery-refresh-68f56404/replay-001/source-text/crates/jope-bench/src/panel/fitted_export.rs` | `2f6e75d70e6d89bbfeac7db7ec67c585e6d7fc5c6287b134851db931ef5ffb30` |
| `agent-delivery-refresh-68f56404/replay-001/source-text/crates/jope-bench/src/panel/fitted_export_run.rs` | `3b007f44198aaa7b96fbca4936ed7ad551feea086f6018f1ff0458ddee6c7295` |
| `agent-delivery-refresh-68f56404/replay-001/source-text/docs/private-fitted-qualification.md` | `abf27b0dd7fabc0ee4a1f095d37d2ee4f77c5ffdfd2988699990521089f61ef8` |
| `agent-delivery-refresh-68f56404/replay-001/source-text/docs/private-warm-target.md` | `4234468c7f9b5348f7a264f815d10d6f62baf1f4b5397c6cf1fed644a12d97a8` |
| `agent-product-final-68f56404/replay-001/source-text/crates/feat-math/src/advanced_v3/fitted_portable.rs` | `f2f8998251f2e7e97900563a7032b7d8b8c80bf43e080b7b622b2e195f75737f` |
| `agent-product-final-68f56404/replay-001/source-text/crates/feat-math/src/advanced_v3/fitted_portable/scalar.rs` | `34823e72c9576c7025d6bbb31a44203f397e1d571642d516a63b5ef328413970` |
| `agent-product-final-68f56404/replay-001/source-text/crates/jope-bench/src/panel/demo.rs` | `00751a3094b42c6f742cffed5f7804803977f47d9f2837b161421e884f7efdfc` |
| `agent-product-final-68f56404/replay-001/source-text/crates/jope-bench/src/panel/demo_asm.rs` | `74c2c6164015bc65f53d021c77f78d9a9ae7238d6c68bb007da424328707064d` |
| `agent-product-final-68f56404/replay-001/source-text/crates/jope-bench/src/panel/inference.rs` | `c4929b553dd6893a48b836c2c3838b62b9d96fbd2c5f76634e235eaeed98e626` |
| `agent-product-final-68f56404/replay-001/source-text/deploy/install-private.py` | `3e4e16d5d7e224624c970aca7217952339c8e679b12752aa8317452c27120e9c` |
| `agent-product-final-68f56404/replay-001/source-text/docs/private-install-demo.md` | `05d728bba5fdc78c1568650ef8179d71ce19cf5c8860ad23317c8cea53dfebb6` |
| `crates/jope-bench/src/panel/experimental_proposal.rs` | `d03c765ac08d1ecc039a82f3dbea0c40bc5158baac042e8df348f0c91a90a018` |
| `crates/jope-bench/src/panel/optimize_cli.rs` | `487a299045d9a24604511cdecd8d4fa0b64ff055e4557339ce34120858549eca` |
| `crates/jope-bench/src/panel/origin_proof.rs` | `229d26685d60646887e5ec30ec4734aa78dfdb2f37788ffbf30f878ce593b675` |
| `crates/jope-cli/src/train.rs` | `486235575fc972306a6294233d94f8a37179945931e74bfb298e342752c634a2` |
| `crates/jope-cli/src/train/corpus.rs` | `a239fd863384817821427033963d8c0c5ebc54642130b087a9848aa51f2d93c5` |
| `crates/jope-learning/src/eligibility.rs` | `839e1de7b5f2c869817564ed39f3c30843208f7a00a1a277d27fe50857ce1e8d` |
| `crates/jope-learning/src/eligibility/inner.rs` | `b5fb468fcc554ec7da9227244d435d84593910c254678fb3f4174b5d7a4e9215` |
| `crates/jope-learning/src/record.rs` | `4360d96a4a6c0a27f5ba78fbcab4f5be10a3a800f56acee6379ef805b3815d61` |
| `crates/jope-pipeline/src/experimental_producer.rs` | `9a15e7077651ded2adf364a26a579720ae21aa92f0362c0122d132e34162411c` |
| `crates/jope-pipeline/src/search_plan.rs` | `fe8e402357b9b61359b03760401da9a568b8a1e7092b1222947fc30d2c8fb817` |
| `crates/jope-policy/src/context_producer/classification.rs` | `465af30abcdc7bcc02c046d92a71e0b7a22be910320e22158ef16b216adc4fa6` |
| `metadata-receipts/veox-jopedime-models-use2--bundles--hyperion-v7--manifest.json` | `fdaf938fc7f7957c06959f12c130b2bd81cdaef5e5bb2198ba75cf35da1c5233` |
| `metadata-receipts/veox-jopedime-models-use2--bundles--jope-prime-gen31-v2--manifest.json` | `c8d8d13a2f4cb2eeb21eea364c33e14429c84d344e6b50eb38a472173b97b6f2` |
| `metadata-receipts/veox-jopedime-models-use2--bundles--policy-step-60000--manifest.json` | `d8e259b26ef8d580ddb99ae5826c9501bcdc0c32045945eef9010926fd796e9e` |
| `python/src/jope/policy/experimental_proposal.py` | `f0338f0505a27c9f2f5eac7579e2c506da5c116b3a206f44c52a1cf87cf9c0aa` |
| `python/src/jope/policy/training_campaign.py` | `9d9cfb3257ecf2b6072cab8adc6d830f3df096605a06ff0677e6338c2aad5a2c` |

Pending interfaces: final product postimages of `deploy/install-private.py`, `docs/private-install-demo.md` and panel `{demo,demo_asm,inference}.rs`; stopped add-on postimages of panel `{fitted_export,fitted_export_run}.rs` and `docs/{private-warm-target,private-fitted-qualification}.md`. Exact source and receipt hashes accompany this appendix's custody handoff.

## Machine-readable companions

### Companion files

The manual directory, `docs/internal/asm-manual/`, contains the following locally reproducible evidence. Paths are relative to it, except the property list, which is `docs/asm-maintainer-status.plist` from the repository root. The original metadata/custody paths inside receipts describe where observations were obtained; use the copies under this directory when available. Historical preparation receipts keep their original pre-execution wording; the current state of the examples is the executed result in `data/example-acceptance/result.json` (14 commands, `ok: true`).

| Companion | Purpose |
| --- | --- |
| `data/dataset-inventory.json` | Thirty complete development dataset records and 522 feature records |
| `data/catalog-summary.json` | Corpus task, row, feature, origin and license metadata totals |
| `data/metadata-receipts/` | Catalog JSONL and collection/model metadata; no payload bodies |
| `data/s3-estate.json`, `data/local-estate.json` | Dated current-object and local logical-byte inventories |
| `data/data-estate-extension-summary.json`, `data/data-estate-extension-release.json` | Reconciled aggregate archive/replica/local-root inventories and 19 qualified recovery risks; exact source mappings and raw-packet exclusion boundary |
| `data/performance-evidence.json.gz` | Current proof, historical score cells, reports, ladder identities and scope |
| `data/plot-*.csv` | Compact plot inputs with snapshot and source pins |
| `data/roadmap.json` | Detailed 43-item dependency/acceptance checklist and 24 separately marked coordinated proposals |
| `docs/asm-maintainer-status.plist` | XML property-list representation of the same checklist |
| `data/example-acceptance/` | Actual fourteen-command result, audit and handoff |
| `examples/` | Private synthetic fixture, exact argument packet and complete-output validator |
| `figures/` | Reproducible charts and associated historical data inputs |
| `render-receipt.json` | PDF digest, pages, consumed input hashes and rendering environment |

The machine-readable companions are `dataset-inventory.json` (all thirty datasets, 522 feature records, complete available upstream metadata and detailed development statistics), `performance-evidence.json.gz` (measured current origin plus historical score cells, timing and ladders), `catalog-summary.json`, `local-estate.json`, `s3-estate.json`, and `metadata-receipts/`. The catalog JSONL retains every available per-dataset catalog field for all 3,859 corpus datasets. No claim is made to enumerate every mathematically possible statistic.

### Lossless evidence packaging

The full performance inventory and original catalog JSONL are stored as `data/performance-evidence.json.gz` and `data/metadata-receipts/veox-jopedime-datasets-use2--catalog--v1--catalog.jsonl.gz` to respect the repository’s 5 MiB file limit. Gzip preserves every original byte; no rows, fields, numeric precision or metadata were removed. `data/evidence-packaging.json` records both compressed and original SHA256 values, sizes and preservation paths. Historical handoffs and the initial assembly manifest intentionally retain their original uncompressed paths and hashes; the packaging receipt supplies the current location mapping.

Read either companion directly with Python’s standard library:

```python
import gzip
import json
from pathlib import Path

root = Path("docs/internal/asm-manual/data")
with gzip.open(root / "performance-evidence.json.gz", "rt") as stream:
    performance = json.load(stream)
with gzip.open(root / "metadata-receipts/veox-jopedime-datasets-use2--catalog--v1--catalog.jsonl.gz", "rt") as stream:
    catalog = [json.loads(line) for line in stream if line.strip()]
assert len(catalog) == 3859
```

The plotting tools consume the separately retained CSVs and summary JSON, so this storage change does not change their inputs or measurements. The renderer hashes the compressed companions as well as the packaging receipt.

### Training-history and continuation companions

#### Companion inventory

The learning expansion is documented in [actual training history](training-history.md#dated-lineage-and-evidence-coverage), [target history and packing](learning.md#what-the-historical-training-and-target-evidence-establishes), [versions, KPIs and continuation](plan.md#the-continuation-interfaces-that-exist-today), and [flywheel/focused search](learning.md#what-better-and-better-must-mean).

| Companion | Purpose |
| --- | --- |
| `data/training-history/*.csv` | Original loss/diagnostic observations and checkpoint lineage, with source hashes and line/JSON Pointer anchors |
| `data/training-history/metric-definitions.json` | Exact historical loss and KPI meanings; objective changes and missing source-build binding |
| `data/training-history/verification.json` | Original-value verification of 3,218 metrics and 78 checkpoint records |
| `data/training-history/figure-receipt.json` | Input/output hashes for seven historical training figures |
| `data/legacy-training/` | Four original round-1 and reported self-play summary files, source hashes, recomputed diagnostics and a vector-plot receipt; internal Elo is not benchmark-valid |
| `data/per-step-history/` | All 63,000 retained recovery steps and 378,000 exact metric tokens in compressed CSV, 21 raw-log hashes, corpus boundaries and explicit 100-step means |
| `data/per-step-history/independent-review/` | Second complete comparison with original logs, checkpoint identities and all derived block/bin statistics; numerical/source review is separate from final-head CI |
| `figures/per-step-figure-receipt.json` | Source hashes and recomputation checks for all 630 plotted loss bins and 78 adaptive validation points |
| `data/packing-routing-audit.json` | Selected external packing metadata, recomputed sizes and strictly scoped paired final-search observations; excludes unjoined campaign rows |
| `data/original-target-receipts-custody.json` | Durable local archive and verification hashes for all 26,357 original development search receipts; archive stays outside Git, with seeds and off-host backup explicitly excluded |
| `data/target-history/*.csv` | All 159 repeated historical comparison summaries and all 213 illustrative arm/generation points; exact source identities and limits |
| `data/target-history/manifest.json`, `data/target-history/verification.json` | Six corpus inventories, binary-format source pins and original target-observation verification |
| `data/target-history/figure-receipt.json` | Input/output hashes for two historical GP-target figures |
| `data/extension-guide/` | Current-source version/command/API pins, stopped handoff and static/metadata-only verification |
| `examples/continue-python-policy.py` | Current Python resume API illustration; metadata-only by default, numerical branch unexecuted and externally supervised |
| `data/single-dataset-help-receipt.json` | Six actual metadata/help commands, with immutable CLI identity and zero fits |

#### Reusing the evidence

The adjacent `data/training-history` CSVs are long-form original measurements. `block-endpoint-metrics.csv` has 1,296 rows; `block-adaptive-validation.csv` 432; `fixed-monitor-metrics.csv` 16; `gen31-epoch-metrics.csv` 462; `a100-checkpoint-lineage.csv` 78; `a100-checkpoint-validation.csv` 312; and `rtx60100-step-metrics.csv` 700. The two original small logs are retained alongside the normalized data. Source hashes and line/JSON Pointer anchors make a plotted point independently traceable. The initial extraction audit covers four CSVs. The shipped `verify_history.py` checks all 3,218 metric `value` cells across the measurement CSVs, plus seven fields (step, model/parameter/optimizer/corpus hashes, world and global batch) and the state-file hash for each of the 78 lineage rows. It does not check every auxiliary numeric column, such as elapsed time, parent/local step, requested fresh fraction or row counts. Running that helper writes `verification.json` beside the CSVs; use a custody copy when preserving a stopped evidence bundle. Model bodies and sealed target values were not read for this reconstruction.

The additional `data/per-step-history` package preserves the 3,809,970-byte compressed step CSV (SHA `aa431f5a2f5f4d6bfc2e8568ef57bb10b100356750d22eb615dd68cfa1647627`), 21 full source-log hashes, block statistics, explicit 100-step bins and boundary comparisons. `tools/extract_per_step.py` regenerates into a new output directory from the retained raw logs and previously audited checkpoint CSVs; it refuses to overwrite an existing extraction. `tools/make_per_step_figure.py` independently recomputes every plotted bin from the compressed row data before drawing it. Run `python3 docs/internal/asm-manual/tools/make_per_step_figure.py` from the repository root to regenerate this one figure without loading models. The original seven-figure builder remains separate. Both figure receipts bind their exact input hashes.

## Receipts

### Important local receipts

The custody root is the single host-local directory `/home/ubuntu/jope-rescue-20260921/asm-integration-20260923/`; this manual records no off-host copy of it. These are host-local receipt locations, not promises that every reader has the same files. The structured documentation evidence identifies the source observations copied into this edition. Weight bodies, source dataset bodies and sealed labels are not embedded in the manual.

| Relative custody path | Meaning |
| --- | --- |
| `neural-integration/lifecycle-68f56404c3e199e06413bfd645d3fe0803d7b96f/` | Full author/committer inventory, CI, native reviews, merge and main-guard readbacks |
| `neural-integration/pr83-final-review-68f56404c3e199e06413bfd645d3fe0803d7b96f/` | Independent review prompt, source/proof inspection and native approval receipt |
| `neural-integration/source-match-68f56404c3e199e06413bfd645d3fe0803d7b96f.json` | All 2,855 Git blobs match the immutable build input inventory |
| `neural-integration/native-context-68f56404c3e199e06413bfd645d3fe0803d7b96f/` | Eight native encoder qualification cases under unchanged numerical policy |
| `neural-integration/origin-run-68f56404.json` | Final operational run command, timing and exit status |
| `neural-integration/origin-report-68f56404.json` | Independent recovered-report command and timing |
| `agent-final-acceptance/measured-origin-68f56404/audit.json` | Source/report/worker inventory and independently bound supervisor accounting |
| `neural-integration/final-head-pipeline-bench-68f56404.json` | Local 53 benchmark and 106 pipeline test results |
| `neural-integration/final-head-host-ten-68f56404.json` | Ten actual privileged host cases and binary identity |
| `agent-final-acceptance/failed-origin-d0f5748b/audit.json` | Preserved failed prior run; not an accepted current score |
| `agent-maintainer-manual/` | Section drafts, structured data inventory, example preparation and documentation observations |

Treat a receipt's command, source SHA, input hashes, runtime and original deadline as one evidence unit. Copying a number out of that unit can lose its scope. A local path alone proves neither continued availability nor integrity: verify the retained hash and the contents required for the claim.

### The PR83 lifecycle record

| Gate | Exact observation |
| --- | --- |
| PR | `veox-ai/JopeDime#83`, merged |
| Manual CI dispatch | Run 35887284772, attempt 1, source `68f56404…`, success |
| PR CI | Run 35887255898, attempt 1, source `68f56404…`, success |
| Required CI jobs | Eight lanes and aggregate succeeded in both runs |
| CI test inventory | 937 Rust passed, 14 skipped; 221 Python passed per run |
| First independent native review | 5293748183, `neverhuman`, APPROVED, exact full reviewed head, 16:29:01 UTC |
| Second independent native review | 5293788772, same eligible account, APPROVED, same head, 16:32:43 UTC; includes residual disclosures |
| Author eligibility | All twelve commits associated with `jepsontaylor` as author and committer; reviewer differs |
| Merge | Guarded rebase with `--match-head-commit` at the reviewed head, completed at 16:34:47 UTC as main `a75e543d2166…` with the reviewed tree `176b75bc36d7…` |
| Remote-main verification | Fetched main and compared complete Git tree; canonical clean main observed before documentation branch |
| Main guard | Run 35889731910, attempt 1, success at resulting main; guard job 107278860508 |

Two review processes overlapped during coordination. Both were independent of the source writer and both approvals are retained; they are not two different human reviewers. One process's supplementary disclosures identify outstanding work rather than silently treating it as fixed. The writer did not select another credential to approve its own source.

The full-main diagnostic run 35889732032 that the PR83 merge queued was cancelled while still queued to preserve the admitted local allocation and let main-guard run. It did not pass or execute. Earlier full-main run 35852022466 failed and remains historical diagnostic evidence requiring repair. Passing fast CI does not replace that outstanding broad qualification.

## Reproduction boundaries

Rendering this manual uses its committed section Markdown, structured data, figure inputs and Python rendering tools. It should not fetch a corpus, load a model, run a fit or contact the sealed store. Plotting historical evidence is distinct from rerunning historical experiments. The included figure data keeps missingness, eligibility and snapshot identities.

The private toy example has its own synthetic input and output receipt. Even a successful five-language recipe/refit comparison cannot qualify a different selected fitted predictor. The pending fitted-inference acceptance must authenticate the retained winner and compare complete saved-state predictions with its original numerical policy.

Operational replay needs more than the document: the exact source/build, immutable workers, model/checkpoint/vocabulary, runtime dependencies, allowed data inputs, original plan and journals, and an available resource allocation. Reproduction that resumes an expired campaign cannot start fresh jobs by assigning a new deadline. A new experiment needs a separately recorded plan; it does not repair the old receipt.

## Names and glossary

ASM is the current product name, settled by the user plan. Historical method IDs such as `afm.lite-race.v1` and `afm.lite-race.v1.resampled` remain exact evidence identifiers in committed same-cut results and the current legacy registry. Use ASM in new prose, but do not rename old identifiers or rewrite frozen reports in place. A method-identity change requires an explicitly versioned new comparison/evidence record. Sources: `crates/jope-bench/src/methods.rs:165`, `:168`; the 2026-09-23 16:07 coordination record.

| Term | Meaning in this repository |
| --- | --- |
| ASM | Algorithm Synthesis Model: learns to generate and optimize executable algorithms from data |
| Algorithm / recipe / program | Executable typed procedure and configuration; reusable across fits. The legacy CLI flag for a recipe is `--genome`. A recipe is not its native fitted predictor, an emitted recipe in another language or a fitted inference bundle ([§](architecture.md#gp-representation-recipes-fitted-predictors-and-the-four-artifact-types)) |
| Method | A comparable benchmark entrant: a registered native recipe (`rust.*`), an external library adapter (`external.*`), an ASM entrant (`asm.neural`, `asm.generic`) or, in historical ladders, a whole system |
| Candidate | A recipe placed in one search slot |
| GP | Genetic programming: search over executable programs using variation and measured selection |
| Arm | One search population in a campaign, with a fixed proposal source and its own declared allowance: the neural arm (policy pool plus exploration slots) or the generic arm (no policy proposals). Historical searches also ran a shuffled-context arm. Unrelated to the Hyperion asset label `arm_b` (the v5 arm-B weights, `hyperion_v5_arm_b.safetensors`) |
| Fitted predictor | One recipe plus preprocessing and model state learned from exact training inputs |
| Checkpoint | Serialized neural proposal-model parameters and compatibility metadata; distinct from a fitted tabular predictor |
| Context | Authenticated numerical dataset/target representation supplied to the proposal policy |
| Dataset encoder | Learned or specified mapping from permitted dataset structure/rows to context features |
| Target encoder / target context | Representation of allowed target information; task-specific and restricted to permitted fitting rows. In the regression context it is the 512-float block computed from min–max-rescaled training targets. The 1,024 foundation values in the 1,536-float “dataset” block also use the training targets ([§](architecture.md#component-dimensions-and-assembly)) |
| Vocabulary | Token identities, grammar and executable restrictions associated with a checkpoint |
| Proposal policy | The neural model that proposes program tokens for a context; the accepted regression policy is the lineage-B step-60000 checkpoint |
| Generation zero | Initial candidate slots before evolutionary descendants; 80 per arm in the current ASM search |
| Seed pool / RNG seed | Seed pool: the programs placed into generation zero of an arm (historically including incumbents and retained teachers). RNG seed: a random-number seed, such as seed 7 |
| Exploration candidate | An admitted generic candidate in the neural arm; not falsely labelled neural-origin |
| Offspring | Mutation or crossover descendant with authenticated parent identities and actual fit evidence |
| OOF | Out-of-fold: predict rows using a fit that excluded those rows; adaptive recipe selection can still expose them at another stage |
| Same-cut cuts (historical) | Hashed cuts built by the historical same-cut benchmark command; not comparable with the frozen panel's cuts ([§](benchmark.md#metrics-and-class-order)) |
| Development-hold score | The score of a final refit on a cut's fitting rows, measured on that cut's held fold (report field `development_hold_score`, `crates/jope-bench/src/panel/origin_proof.rs:321`); the one-dataset proof's RMSE uses 34 held rows |
| Policy-validation families | In the pending teaching partition, the families held out to validate the policy; older text calls them “development” families. Not the panel's development rows |
| Exposure / adaptive | Exposure: use of a partition's labels for selection, optimization or reporting. Adaptive: measured on data that also helped choose what is measured, so not an independent generalization estimate ([§](benchmark.md#adaptive-development-exposure)) |
| Cut identity | Frozen identity of the training/scoring row assignment and evaluator context |
| Inner folds | Five whole-group folds inside a cut's fitting rows, used for out-of-fold selection and, in the pending eligibility path, for teaching admission ([§](teachers.md#the-preregistered-freeze-and-train-side-inner-split)) |
| Frozen panel | `asm-private-30-v1` under protocol `asm-panel-v1`: 30 PMLB datasets, ten per task, manifest `ba7dace4…` ([§](data.md#the-frozen-thirty-dataset-panel-identity-totals-groups-and-immutability)). Older text also says official30 or the unchanged thirty-dataset panel. Training-side validation sets that change between checkpoints are not this panel |
| Task / category | Task: regression, binary classification or multiclass classification. Training text, `--task all-admitted` and the owner's goal of ten wins per category use “category” in this sense; elsewhere “category” means a categorical feature value |
| Label | The column a dataset asks a model to predict; one of the senses of “target” |
| Family | A group of related datasets from one source, such as the UCI computer-hardware source of `230_machine_cpu`. The teaching partition assigns whole families to training or validation. The north star (proposed; owner ratification pending) is measured on previously unseen families ([§](status.md#the-proposed-north-star-and-matched-comparisons)) |
| Development data | Data visible to optimization or recipe/model selection, regardless of a file's name; in the panel, every non-sealed row |
| Development folds 0–4 | The five original folds of the panel's development rows; every group stays within one fold ([§](data.md#the-frozen-thirty-dataset-panel-identity-totals-groups-and-immutability)) |
| Cut k | Fit on the development folds other than k and hold out fold k (`dataset.cut(k)`, `crates/jope-bench/src/panel/data.rs:314`). Older text calls cut 0 “fold0” or “development fold zero”. For `230_machine_cpu`, cut 0 has 135 fitting rows and 34 held rows |
| Sealed evaluation | Separately protected evaluation outside the learning and optimization closure. The “seal-protected shared memory” of the training reader (memfd seals) is unrelated |
| Burned partition / burn receipt | In the pending training64 patch: a dataset cut originally in the `test` role whose labels were made visible to optimization, so that they now count as development data, and the immutable `jope.label-burn/v1` receipt that records this relabel. Sealed and `meta_test` labels are refused rather than burned. `require_unburned_test_partition` refuses to present a burned partition as untouched; it has no production caller yet (TEACH-02) |
| HWM | High-water mark: the best retained result for a comparison identity. At `a75e543d2` the current-HWM pointer is keyed only by task, dataset hash and cut (`crates/jope-learning/src/state.rs:13`), so a bound result under a different evaluator, vocabulary or protocol with the same metric can still displace the incumbent (`crates/jope-learning/src/hwm.rs:64`). The pending eligibility/HWM correction (TEACH-01) uses the complete comparison identity (task, dataset, cut, evaluator, split, vocabulary, preprocessing and numerical protocol), so that a result under a different identity cannot displace an unrelated incumbent ([§](teachers.md#the-admission-guard-label-exposure-and-hwm-identity)) |
| PANEL-WINS (proposed) | **Proposed definition, pending owner ratification; no code at `a75e543d2` computes it.** Per task, the number of the ten frozen datasets on which `asm.neural`'s confirmed five-fold mean development loss is strictly below the minimum over all eligible frozen non-ASM comparators (every `rust.*` and `external.*` recipe). Incomplete, failed or unconfirmed datasets count as not won. `asm.generic` is reported in its own column. Reported as (regression, binary, multiclass); the owner's goal of ten wins per task is (10, 10, 10), a 30/30 sweep ([§](learning.md#win-counting-and-a-proposed-acceptance-record)) |
| Teaching admission | Eligibility decision permitting a measured example into policy training |
| Motif / macro | Typed reusable program fragment with eligible provenance; not automatically a new primitive operator |
| Numerical policy | Declared arithmetic, probability, ordering, tolerance and runtime assumptions fixed before scoring |
| Alias | Alternate spelling of an existing recipe; not another independent method fit. The regression `ridge` head carries the historical alias `logistic` (`crates/feat-math/src/advanced_v3/grammar.rs:1263`), a known misnomer |
| Front door | The supported CLI entry point for an operation, which enforces its admission rules, identities and budgets |
| Head | Name the sense: PR head commit, reward or attention head, estimator head (such as the ridge head), or serving/training pointer |
| Unsupported | A backend or task deliberately refuses a capability outside its admitted contract |
| Unavailable | A dependency or required runtime is missing; different from a model failure |
| Failed | An attempted operation ended unsuccessfully; retain its reason and measured/unknown costs |
| Skipped | A known test or operation did not execute; not passing coverage |
| Journal | Incremental durable started/finished/observation records for audit and original-window recovery |
| Receipt | Evidence binding command, source, inputs, runtime, resources, times, status and artifacts |
| Provenance | The chain of source, data, model, recipe and execution identities supporting an artifact |
| Custody | Host-local preservation directories outside Git, such as `/home/ubuntu/jope-rescue-20260921/asm-integration-20260923/`, that hold stopped patches, receipts and evidence. “Custody buckets” (also “custody dev/prod”) are something else: the account's separate `jopedime-714044927654-us-east-2-dev`, `-prod` and `-sealed` S3 buckets ([§](data-estate.md#additional-and-replica-buckets)) |
| Elo | Relative Bradley–Terry rating on the arena's declared method universe and eligible pairwise scores |
| Warm demo | Demonstration after verified preparation; report separately from cold installation |
| Promotion | Explicit qualification of an exact artifact/model/backend for its intended serving use; a merge alone is insufficient |
