# JopeDime release closure workplan — 2026-10-19

Status: **active implementation plan; release NOT qualified**. Admiral adoption: SC184M assigns `ADM-JOPE-RC-C00` through `C19` to the existing owners. Coordination is manual while native execution is unqualified; see [delivery status](IMPLEMENTATION_STATUS.md). This is an additive execution overlay on `MASTER_PLAN.md`, the 47-task `major-release-20261018` ledger and the separate older 66-item milestone. It does not replace, renumber or silently waive any of them.

## Operational amendment — 2026-10-19 19:30 UTC (controlling)

SC184AC adopts the human's process-local CI direction, verified directly in Codex chat `01a1559b-47ca-7e70-973f-973586db8ab7`. It supersedes older compile-only JA06/test-hold and new remote-CI-dispatch instructions in this dated overlay. Keep the existing owners, one integration PR, exact source/input/test receipts, independent exact-head review and ordinary protected merge. Do not dispatch GitHub Actions or paid remote CI; do not label local receipts as hosted checks. If a protected merge enforces an unavailable remote requirement, route that precise conflict to the existing Admiral/merge owner without bypassing it.

Run local mechanical gates in each owner's existing checkout/target using `/home/ubuntu/xbabe2_kicker/fleetcap/run-capped`, nice10, jobs4, `CARGO_INCREMENTAL=0`, `CARGO_PROFILE_DEV_DEBUG=line-tables-only`, and no globally set `CARGO_TARGET_DIR`. Root owns W execution; captain must not duplicate it. For C03 use `cargo test --locked --offline -p jope-datasets --lib -j4 -- --test-threads=2` and `cargo test --locked --offline -p jope-models --lib release:: -j4 -- --test-threads=2`, then scoped all-target clippy/fmt/prose/diff checks. W2 owner executes admission/library/compile-fail doctest gates at its actual stopped head, records every failed/skipped/filtered case, and supplies physical worker receipts. Whole integration acceptance still requires all relevant feature/runtime checks; a small green subset does not qualify the release.

The binding ENG:watcher clarification (sha256 `4d6bfe0784d18f0ed225c7e49d1ed0a3e089da6fa2b5a32b49bf9ed3621561cf`) and SC184AD retain the training hold: code/fixtures/local cargo tests, storage/cache work and non-training GPU benchmarks are allowed; actual learner exposure, training, refit, self-play and training/eval runs are held until Jepson approves in the ENG:watcher chat. The existing captain freezes/reviews the bounded lineage/evaluator/resource/attempt/stop-rule pilot, marks `READY-AWAITING-APPROVAL`, and posts it to COMMS. Stop there rather than treating mechanical test authorization as scientific or custody admission. Record finite benchmark terminal once, archive it and do not loop identical benchmarks.

This is an additive policy correction; preserve the original master/47-task/66-task ledgers and execution receipts. The separate authorized flywheel/storage plan packet is being prepared outside active source trees for integration by the same owners. Current deliverables and remaining gates are in IMPLEMENTATION_STATUS.md and TASKS.json.

The adopted priority detail is [storage value and feedback proof](FLYWHEEL_VALUE_WORKPLAN.md), prepared from the three latest human-supplied audits. Use its existing C09/C12/C14 mappings and the [focused local CI guide](../../../deploy/ci/flywheel-local.md). Packet-v2 supersedes the earlier stopped packet; the binding training hold above still controls every numerical pilot. A manifest missing source/data/adapters remains incomplete even if its status says ready.

## Objective and finish line

Deliver one reviewed integration PR onto current GitHub `main`, with an accurate README and agent entry guide, passing exact-head and receiving-main CI, usable versioned encoder/policy downloads, implemented Direct/Thinking and independent AutoML/ensemble options, durable research records, and explicit remaining scientific/release gates. Complete the declared major-release engineering scope and actual flywheel qualification before publishing its major release or claiming DONE.

The fastest useful next scientific result is a measured, fixed-budget feedback loop using the existing model and GP design: propose → execute/measure lawful new programs → publish verified labels → train from those fresh labels → compare against matched frozen controls on untouched families. Improving training loss alone, re-ranking a frozen pool or repeating CUDA latency runs does not close that loop.

Non-goals: a new model architecture, a wholesale GP grammar change, a new orchestrator/dashboard, unrestricted corpus ingestion, blanket S3 cleanup, a new paid provider, or filling GPU utilization with repeated benchmarks. No change to protected evaluator, split, custody, safety or resource gates. Admiralty's code-first CI exception applies to its cutover, not JopeDime.

## Current baseline and controlling evidence

Observed 2026-10-19 16:54–17:05 UTC; refresh identities before execution:

- PR384, head `1187217099e2acf15291b535d26fece3fbed9f79`, merged at 16:54:17Z to GitHub main `9f8d1a74ab08fc200214a82beaf2ee96e316061a`. Both required runs `39280859112` and `39286144220` passed all 11 executions; independent `neverhumanbot` approval `5574850332` binds that head.
- Main `main-guard` `39289511526` passed. `full-main-cpu` `39289511196` was queued/running at the planning snapshot; do not call it passed or replace it for queue delay. An older green `833fe058...` run is not evidence for `9f8d1a74...`.
- Open PR queue was empty after PR384. Re-query before opening the single next integration PR. Source workspace version is 0.3.0; do not infer a published major release from `Cargo.toml`.
- Committed MR ledger records 4 pass, 1 partial, 1 in-progress, 41 todo. These are recorded states, not a scientifically qualified completion percentage. Off-main MR14/MR20 receipts must be reconciled with dependencies and actual final-main acceptance before increasing completion.
- Existing P01 development analysis selected the historical v1 parent on 263 reused development contexts (loss 0.897327 versus 1.343791 for the cold 429-step run). That comparison is not an unseen-test result or authority for continuation. Preserve both parents and the original failures.
- Native research wave has 6,857 train contexts, 263 dev contexts, 429 completed updates, 858 proposals, zero demonstrated newly measured/published feedback labels and zero demonstrated updates from fresh feedback. One epoch is not evidence of convergence.
- Corrected CV/OOF/ensemble source is on the existing flywheel lane. At 17:03 its head was `4b4d3d59fe9cbc003d0e5791e271e08051cbb4b5`, branch `feat/flywheel-cpu-fitter-catalog-testscore`, with active catalog edits. This supersedes older branch pointers; acquire a stopped clean handoff, not a moving head.
- `SC-DEC-184L` gives this Codex chat sole README/agent/closure-doc ownership and sole next integration PR/merge ownership in existing W. Flywheel retains pipeline/CLI source; captain retains operational receipts and research packaging. Read its full written lease before changing paths.
- SC184L separates finite kernel replicates from held neural training/replay/continuation. Existing finite benchmark jobs have owners; do not interrupt or duplicate them. A live benchmark process is not neural training. Obtain one verified scope-specific run admission before neural execution.
- Existing Admiralty integration chat's last tool-visible state at 17:05 was `systemError` (last turn blocked by safety systems), not an execution receipt. Route through the Admiral and existing platform owner; do not assume a healthy observer feed means native autonomous execution works. No bypass or duplicate platform writer.

## Read first, in this order

1. `AGENTS.md`, `AGENT_CHAT.md`, `MASTER_CLEAN_UP.md`, and the local user's `RTK.md`; then fresh applicable `gb_sc_fix_*`, `ROOT_*`, `JEPSON_RULING_*`, SC184L and claims/owner receipts. Commands on this user's workstation start with `rtk`.
2. `MASTER_PLAN.md`; `docs/plans/major-release-20261018/{RECONCILED_SPEC.md,GP_UPGRADE_WORKPLAN.md,GP_SPEC_COVERAGE.json,MAJOR_RELEASE_TRACKER.json,MAJOR_RELEASE_TRACKER.md,validate_gp_scope.py}`. Read source-spec packet text referenced by the coverage file for every claimed task. Preserve all 47 IDs, MR45/MR46/MRFV included, and the separate 66 ledger.
3. `README.md`, `docs/{architecture,pipeline,models,datasets,learning,emitters}.md`, `docs/release/README.md`, ADR-0004/0005/0006 and `docs/AGENT_START_HERE.md`.
4. `models/{manifest.json,PINNED,bundles/*.json}`, `schemas/jope.model-bundle.v2.schema.json`, `crates/jope-models/src/{cli,manifest,index,cache,fetch,publish,repo}.rs`.
5. Flywheel handoff's `crates/jope-pipeline/src/{cli,pathway,model_build}.rs`, `model_build/{config,selection,oof,candidates,ensemble,artifact,native,catalog}.rs` when present, and authored tests. Also `evaluation_score.rs`, current `test_score.rs`, `kernel2` and fitted emitter paths. Existence in a worker tree is not final-main availability.
6. `deploy/dl-flywheel/{README.md,SOURCE_MANIFEST.json,RESEARCH_SOURCE_CLASSIFICATIONS.json,check_source.py}`, actual native/continuous producer and `consumer/dltrain/model.py`; `jope_flywheel/{batch_stream,training_admission,training_snapshot_native,s3_library,score_join}.py` and relevant tests. Read stopped warm-resume v3 plus original producer before adapting it.
7. `docs/research/waves/INDEX.jsonl`, recent wave manifests/cards/metrics/context; `catalogs/training-artifacts.json`, `docs/runbooks/training-artifact-locations.md`, `schemas/jope.wave-evidence.v1.schema.json` and existing retention/archive contracts.
8. `.github/workflows/{ci,full-main,main-guard,release}.yml`, `deploy/ci/{check.sh,coverage.json,fast-lane.sh,aggregate.py,full-main.py}`, `crates/jope-records/src/paper_trace.rs`, `deploy/release/release.sh` and the release runbook.

## Hosts, checkouts, ownership and forbidden zones

On xbabe2 (`ubuntu@162.218.217.123:6969`):

```sh
R=/home/ubuntu/JopeDime
W="$R/.agent/worktrees/fleet-main-296d07e1"
W2="$R/.agent/worktrees/pr371-gate"
D=/home/ubuntu/xbabe2_kicker/captains-20261017
A=/home/ubuntu/jope-evidence/storage-audit-20261019T1455Z
E=/home/ubuntu/jope-evidence/release-closure-20261019
```

| Owner | Write scope / required deliverable |
| --- | --- |
| Admiral `4060655.super_captain:0` | Decisions, fleet admission, task/lease reconciliation, actual Admiralty dispatch/adoption and unblock priorities |
| This Codex chat `01a13a98...`, W | README, `docs/AGENT_START_HERE.md`, this closure plan; P02 `jope-models` and versioned artifacts; single next integration PR and ordinary reviewed merge |
| Flywheel `1061448.jopedime_flywheel:0`, W2 | Leased pipeline/CLI, concrete estimator/OOF/refit/predictor source and tests; stopped commits for integrator pickup; no README edits or competing PR |
| Captain `2724936.captain_jopedime:1` | P01/P04 operational evidence, existing host runs and scope-specific admission, compact finished-wave publication packet, tracker evidence preparation |
| Reviewer `487851.jope_review_claude:0` | Independent source/math review and native exact-head approval after checks; read-only workflow access, no cancellations |
| Existing Admiralty integration owner / Commodore | Platform integration, deployment and authoritative execution feed; no Jope source handoff implied |

Claim and recheck under `AGENT_CHAT.md.lock` immediately before any checkout/index mutation. The lock protects the record, not the entire checkout. Read every overlapping claim. An idle screen is not permission to take its tree. Every handoff supplies stopped full SHA, branch/base, dirty/untracked list, files, source/tests/logs and owner release. No forced history rewrite; shared branches update by history-preserving merge. Integrator may pick reviewed stopped commits into its own lane with authorship retained; do not cherry-pick a moving working directory.

Forbidden: xbabe0/Jeryu; other project repos; canonical R source writes without fresh handoff; `.git` internals except ordinary Git operations; other agents' index or target directories; `jopedime-paper-wt`, `review-i01-aa06bc31` and the `jope-dime` symlink; secrets; immutable S3 objects; historical receipt edits. The paper tree remains locked. No resets, force pushes, `git clean`, broad `rm`, new worktrees/screens/clones, provider changes or unapproved spend.

Generated/read-only zones include README `arena`/`samecut` blocks, `target/health`, compatibility fixture banks, source inventories and receipt projections. Use their owning generator and identity transition procedure. Changing `Cargo.lock` can change evaluator identity even without numerical edits; never silently regenerate a historical result.

Disjoint preparation can proceed in W (models/docs), W2 (pipeline), and A/E (captain receipts) under leases. Integrator alone serializes Cargo/workspace/coverage, tracker projection and CLI registration conflicts. Pipeline and model owners agree an adapter contract before writing their respective sides. Admiralty imports tasks only into its current authoritative store through its existing owner; no second dashboard or task ledger becomes authoritative.

## Ordered work packets and acceptance

### C00 — Finish PR384 receiving-main validation (P00)

Owner: Codex; ready now. Preserve the merge receipt and exact review/run history. Query native GitHub run39289511196 and record every matrix job, nonzero evidence and conclusion. Inspect a red job's actual log and source/runner identity before deciding a correction or one specifically authorized retry. Do not cancel a valid running workflow. Re-read effective required status/review/conversation rules before the next merge. Completion: main-guard and all required full-main families at actual receiving SHA, with logs/hash-bound receipt. A queued run is pending, never PASS. Code preparation continues while waiting.

### C01 — Freeze ownership and adopt the plan in Admiralty

Owner: Admiral; Codex provides packet. Import/adopt C00–C19 as child execution packets of P00–P07, preserving MR01–MR46/MRFV links. For each: owner, checkout, full base/head, dependency IDs, exact command/artifact, cap/deadline, next deliverable and evidence locator. Use current native task mechanisms only after owner demonstrates task write → dispatch → worker result → acceptance transition on a safe existing task. Record actual IDs/receipt or mark `manual-coordination-active / native-execution-unqualified`; an ACK, HTTP200 or fresh observer timestamp is insufficient. Recover the existing platform task through its owner if its systemError persists, without retrying blocked actions or bypassing platform controls. At least every 30min workers commit and push useful source, or publish the precise blocker and next actor. Avoid ACK loops and duplicated dispatches.

### C02 — Source/tracker reconciliation and integration manifest

Owner: Codex plus stopped owner handoffs; depends C00 for PR opening. Freeze current `origin/main`. Inventory all refs (`git log --all --not --remotes`), registered trees, dirty/ignored assets and current PR queue; do not scan large data bodies gratuitously. Capture:
- full MR20 preserved branch head `3445618cb2842a4087da3874f43fc559ef0f5c58`, including source commit and its receipt commit (SC184L's abbreviated a261 source alone omits the latter);
- current flywheel branch/head, corrected JA02–04 commits, concrete fitter/catalog/test-score additions and remaining dirty work;
- P02/docs commits and captain's wave card files/hashes;
- canonical MR06 unique source and review-i01 unique work, each with owner disposition rather than automatic adoption/deletion.

Write `E/INTEGRATION_MANIFEST.json` with include/defer/obsolete-with-proof disposition per source commit. Retain duplicates until tree-equivalence and coverage proof; keep original authorship. Reconcile MR20's MR18 dependency, MR19 partial state and MR14 witnesses. Fix the known `validate_gp_scope.py` MR09 receipt inconsistency in the owning proof lane; do not weaken the validator. Every MR pass needs its exact cases and current-main applicability. Any packet referring to only 44 tasks is corrected to the full 47; keep the old66 ledger separate.

### C03 — P02 immutable encoder/policy release set

Owner: Codex; depends on agreed pipeline boundary, not on new training. Edit `crates/jope-models/src/{lib,cli,manifest,index,cache,fetch,publish,repo}.rs` only where needed; add `release.rs` and `schemas/jope.model-release.v1.schema.json` for a release-set wrapper over existing bundle-v2. Add `models/releases/<immutable-version>/manifest.json`, additive bundle IDs/index pins, and `docs/models.md`. Never modify an already-published bundle's bytes.

Proposed contract (final field names agree with pipeline owner before implementation):

```rust
struct VerifiedRelease { // construct only after validating trusted pinned manifest + files
    release_id: String,
    source_commit: String,
    encoder_bundle_ids: Vec<String>,
    policy_bundle_id: String,
    tokenizer_sha256: String,
    preprocessing_sha256: String,
    runtime_sha256: String,
    checkpoint_sha256: String,
    qualification: Qualification, // research / serving; no implied promotion
}
```

The manifest also binds architecture, vocabulary/DSL/evaluator, weights/byte counts, license/notice, exact S3 VersionIds, input/config/seed identities and compatibility constraints. Separate legacy Hyperion/step60000 from Kumo/282-token native PolicyLM; no weight-shape/vocab coercion. Reuse `BundleManifest`/`Fetcher`/`ModelsCache`, content-addressed blobs and manifest-last create-only publication. Trusted source expectation must come from a pinned release registry or independently verified signed source, not the same operator claim being checked. A claim must not create `VerifiedRelease` by deserialization.

Expose one verified release fetch/verify/path flow and explicit version folder. Document only command spellings actually implemented; keep research downloads usable without claiming serving qualification. Preserve existing bundle commands. Tests in `crates/jope-models/tests/release_set.rs` (new) cover fresh-cache and offline reload, partial/missing/corrupt files, traversal/symlink escape, wrong VersionId/source/runtime/vocab/encoder, incompatible components, rights refusal, interrupted manifest-last publication, atomic no-partial-install and old bundle compatibility. File-backed fake-store CI proves plumbing; one separately admitted exact-version self-hosted cold readback proves deployment. No production writer credentials inside PR runners.

### C04 — Real Direct and Thinking execution

Owner: flywheel; source prep can proceed alongside C03. Edit existing `pathway.rs`, `model_build/config.rs`, `cli.rs`, current neural/policy adapter and emitter seams rather than adding an unrelated engine. Map P02 `VerifiedRelease` into the existing `ReleaseClosure` / `verify_checkpoint` boundary; preserve explicit refusal for missing or incompatible components. Implement the one-arm `neural_search` path: read encoding, run actual inference, parse/check the exact GP, fit it on allowed train rows, then emit/run supported Rust/Python/C++17 targets. A planning enum or a successful refusal is not positive execution coverage.

Thinking seeds generation0 from that actual neural GP; remeasure seed, evolve under 64/256/1024/4096 whole-request evaluation caps, obey stricter time/work/memory limits, retain every attempted/failed/no-op child and parent identities. Use existing request ledger; avoid multiplying the global cap by fold, island, candidate or retry count. Export the selected fitted artifact and validate in a fresh process on runtime query inputs. Unsupported target/program combinations refuse explicitly. Keep legacy CPU behavior through the named `run-legacy-gp` migration and document its acceptance/compatibility controls. Tests: existing `pathway_plumbing.rs`, `model_build_flags.rs`; add a real admitted tiny execution fixture and negative release/seed/target/budget cases. P02/real runtime missing remains a blocker, not a fallback to generic GP.

### C05 — Concrete AutoML roster and leak-free CV

Owner: flywheel. Retain flag > file > default precedence with explicit negative flags; AutoML and ensemble independent. Inspect `candidates.rs`, `catalog.rs`, `native.rs`, `selection.rs`, `oof.rs`, existing model-zoo contract. Bind each enabled entry to source revision, algorithm vs preset, runtime/FFI library hash, license, task types, metric/probability behavior and persistence capability. Catalog 655 entries and 415 normalized strings are not 415 independently reviewed algorithms. Source-present is not executable/admitted. Initial concrete ridge/logit adapters may be qualified as that limited subset; do not call the broad zoo complete.

Extend the real fitter, including neural-GP candidates and admissible model-zoo algorithms, one verified binding at a time. Bind the actual NativeData feature/target arrays, ordered row IDs and class encoding to the same trusted request/dataset identity that BakeoffPlan labels describe; a callback echoing a supplied request hash cannot prove that binding. Fit transforms only within each fold. Enforce exactly five common train-side folds, one fold per group, complete row/class coverage and unchanged outer-test seal. Charge attempted candidate evaluations before callbacks; measure physical starts from real worker receipts, including internal fits, and keep unknown counts unknown. Persist actual fold fitted artifacts and fit/scoring row identities; `AdmittedBundle` must validate against trusted plan and fold receipts, not caller-authored matching hashes. Fix failing real branches rather than trimming roster/cases silently. Tests cover crossed groups, shuffled rows/classes, empty/failed fold artifacts, candidate failure, zero/multiple physical fits, timeout, one-class folds and resource exhaustion. New tests `model_build_automl.rs` distinguish real fitting from mocks and declare the admitted task matrix.

### C06 — Ensemble, refit, persistence, fresh-process test score

Owner: flywheel; depends C05 and C03/C04 for neural members. Complete `ensemble.rs`, `artifact.rs`, `test_score.rs` and fitted predictor consumers. Train-side selection/report partition is fixed before outcomes; choose nonnegative weights from valid OOF predictions, retain best-single fallback and why; guard same-class/order/rows/metric support AND a single trusted request/input/fold-plan identity across all members. Ordinary OOF alone does not make the meta report half independent: base fits for selection rows may already include report-half labels. Use nested cross-fitting with report labels excluded from every upstream fit used for selection, or explicitly classify that report as reused development diagnostics without an independent estimate. Add a report-label poison test through the real fitter, not only a mocked OOF bundle. Never use outer test scores for candidate selection, ensemble weights, early stopping or configuration.

Refit winners on full allowed train partition after selection, persist member state plus normalization/class schema/weights/IDs/runtime requirements, reload in a separate process with no training data or in-memory fitter, and predict unseen inputs. `artifact::verify_reload` metadata alone cannot substitute for numerical prediction parity. Extend `jope.pipeline-test-score/v1` through typed persisted predictors, preserve same-registry earlier-look history, and score outer test once per declared identity. Validate exported Rust/Python/C++17 support or record unsupported members; no silently substituted estimator. Tests `model_build_ensemble.rs`, `model_build_persistence.rs` (new): real single/ensemble parity, failed candidate, malformed model/state, changed runtime, invalid probabilities, no data/label leakage and legacy predictor behavior. Execute only under JA06/current approved test envelope.

### C07 — Native warm-resume source and independent qualification

Owner: captain supplies stopped adapter; flywheel/Codex integrate only after disjoint source lease. Preserve A's v1/v2/v3 artifacts and their original hashes. Actual parent429 `state.safetensors` has 48 model tensors, 144 named optimizer tensors, torch CPU/CUDA RNG and no embedded metadata; step/config/parameter IDs live in its sidecar. Use named parameters and producer's tied-weight digest, not an invented replacement hash. Preserve producer AdamW decay/no_decay groups, betas(0.9,0.999), constant lr0.0003; changing schedule/betas creates a distinctly admitted fork.

Close sidecar/header/source binding, tensor and optimizer restore, RNG source audit, stream cursor/step/no-pending alignment, outbox-head semantics, SQLite `Connection.backup()` and no-partial-mutation refusal. Honor separate caps: at most20 retained new states /8GiB and at most2,574 total unresolved proposals including parent858. No literal from an erroneous ruling overrides the corrected cap. Independently qualify interrupted-versus-uninterrupted behavior on a tiny authorized fixture before any real-parent tensor load/forward/update; current design/header-only tests do not prove that. Integrate into owning `deploy/dl-flywheel/pipelines/native` producer and continuous adapter only with source-inventory updates and claimed paths. A real-parent read/eval/continuation follows its specific execution gate, never merely source tests passing.

### C08 — Admit the next finite neural run, then train to a reasoned stopping rule

Owner: Admiral/captain; depends C00/C03/C07/C15 and explicit current execution disposition. Resolve contradictory operational hold fields into one scope-qualified record; do not infer authority from a key nickname or a file claiming to quote a human. Record concrete actor, authority, source/config/parent/input/evaluator hashes, rights/split/admission, host/rank, optimizer/update/token/wall caps, retention, expected outputs and stop conditions. Preserve parent choice evidence and declare continuation versus warm-start versus cold run accurately.

Use fixed development checkpoints for learning-curve and overfitting diagnosis; do not stop simply because one epoch ended. Proposed bounded continuation evaluates improvement at predeclared checkpoints, gradient/update norms, held-out dev gap, optimizer health, invalid decode and program success rates, with patience and maximum budget defined before launch. Reused dev is not confirmatory test. Never launch a long run just to exhaust all S3 objects. Stop on nonfinite loss/state, source drift, rights/split failure, checkpoint corruption, resource pressure or no useful new labels; save terminal state and reason. No simultaneous duplicate runs under the same budget. Existing other services and GPU jobs retain their owners.

### C09 — Close the real fresh-feedback loop before scaling

Owner: captain scientific lead, flywheel source owner; depends C04/C07/C08 plus actual evaluator/data admission. Use P01 frozen-pool StageA as a ranking diagnostic, then StageB 12 TRAIN families ×3seeds ×2rounds ×4matched-budget arms, subject to frozen admission and adequate independent cluster power. Primary paired family log-loss-ratio endpoint and one-sided97.5% upper bound/practical margin are preregistered; source signs and loss direction are verified before results. The .02 practical benefit threshold is retained from the existing packet, not retuned after observation. Untouched outer families stay sealed; retain all failures and cost.

For at least two actual rounds bind proposal IDs → real fit/evaluation receipts → durable published labels → admitted incremental training snapshot → optimizer updates/checkpoint → next-round proposal and measurement IDs. Compare frozen neural/no-update, ordinary GP, active feedback and intended control under equal total attempts/resources, not equal accepted successes. Prove feedback labels are new, not recycled parent/pool rows; deduplicate semantically and prevent split ancestry leakage. Add/complete `crates/jope-learning/tests/flywheel_measured_feedback_e2e.rs` for mechanism with a tiny fixture; actual experiment receipts are additional. No qualified self-play claim until actual strict execution/preregistration/evaluator/outer-test gates close. A negative or inconclusive result is preserved and guides targeted weak-family GP/data acquisition, not post-hoc metric changes.

### C10 — Remaining 47-task GP scope, in dependency order

Owner: Admiral allocates existing lanes; no exhausted agy-provider retries. Use `MR_TASKS.json` snapshot and authoritative tracker, exact tests and GP workplan per task; reassign a blocked owner in writing. Proposed scheduling groups preserve the original DAG:

1. Authority/receipt/data/old-route foundations MR01–MR07 and MR11–MR13; close admitted fleet proof gaps rather than interpreting unit fixtures as live authority.
2. Numeric/strict ingress/IR/control-flow foundations MR14–MR19; complete MR18 before dependent MR20 acceptance. MR20's 100k edit result is useful source evidence, not permission to skip lazy-control-flow or full source coverage.
3. Fit/export/scope/cross-fit closure MR21–MR24; learner/solver/state primitives MR25–MR32, with independent mathematical and train/query leakage checks.
4. Grammar/action-policy/abstraction/resource lanes MR33–MR36; all original no-truncation, support, tied-gradient, default-off and actual runtime witnesses retained.
5. Experiment/value/recovery/final capability requirements MR37–MR40; analytical tensor reference MR41, actual fit/mutate/crossover/refit/export MR42, state/update edits MR43 and real numeric-v2 route/export MR44.
6. Retain all MR45 GAM/PCA/centroid, MR46 closure-repair experiment and MRFV final obligations; dependencies in the original tracker govern, not this grouping's prose.

Do not freeze or enlarge the candidate protocol silently. Do not introduce architecture/GP changes to explain a failed scientific pilot without preregistered, separately admitted evidence. `validate_gp_scope.py` proves coverage integrity only; every implementation acceptance command must collect the named tests and produce the declared cases. Full feature/runtime/Python/emitter/GPU evidence cannot be replaced by `cargo test --workspace` alone.

### C11 — Preserve completed research and publish compact Git waves now

Owner: captain packet; Codex integrates. Verify new 180-rep archive pack `9a8751a6...` VersionId `bRcGSbfZtk0rtG3q04WLWjLTfwvHuJan`, index `2d6f64e6...` VersionId `k3naEzmBnwN_wx7O5Hoz9xM9Q7qROqVS`, independent restoration receipt `2365a1e8...` covering 2,174 members. P01 child pack `dca2121c...` VersionId `NO7sLton___HoBuEu1.5tqAPwOYpcoyV` and index `eeaca6ba...` VersionId `bdt6WVch6VkNSyNp1Tf5QaTRIvGO3Z7v` now have independent exact-version readback proof: 7/7 regular members, 263 decoded JSONL rows, zero mismatches; receipt `cc19ae2430c0ece85bd488c7f74e9f71d6d7353c4c7c450d62b486a290c1b6fd` at E/independent-p01-restore/independent-restore-receipt.json. This closes restoration only. Correct the staged summary byte count to actual24,811 (not24,839); bind member roots to actual enclosing pack/index plus member path. Do not assign the pack VersionId to a later independent-restore receipt that is not in the pack. Preserve original drafts and publish an explicit correction. Resolve actual bucket/key from receipts; do not copy an order's incorrect bucket label.

Add distinct compact cards/metrics/context/manifests and INDEX.jsonl entries for finished benchmark and P01 comparison. Keep original source/build/dirty status, hardware, input/split/protocol/config/seed, actual counts, errors, negative outcomes and scope. Old source c956 is not relabeled9f8. Link old data rather than duplicate it. Store real publication commit/PR and main readback; a local packaging JSON is not Git publication. Ensure original429 checkpoint archive gaps remain explicitly held rather than declaring all430states protected by a pack containing only four. For every new wave use the same schema and compression/index contract.

### C12 — Storage audit, high-information retention, safe reclamation

Owner: captain custody/data with independent restoration reviewer. Existing ~35.16M objects /~2.95TiB daily inventory covers 8/9buckets and contains metadata/attempt/history/archive objects, not that many training examples. Version bloat was not the primary observed driver. Freeze inventory date/coverage, count current versus noncurrent versions/delete markers separately and classify canonical data, joins/pools, program results, checkpoints, redundant context/attempt shards and custody archives.

First compact/deduplicate high-volume small metadata into immutable compressed partitions with a sparse columnar/query index, semantic IDs and old→new locators; measure bytes, compression, access and restoration cost. Preserve learning-relevant examples/attempts/failures, lineage, all unique checkpoints required for reproducibility, licenses and protected sources. Trainable coverage funnel must show object→body→eligible identity→join→dedup→split→admitted row→consumed sample, with exclusion counts and reasons. More data/evaluations target demonstrated weak families/uncertainty and measured marginal value, not raw object count.

No version is deletion-certified yet. A later retirement manifest lists exact bucket/key/VersionId or local non-symlink scratch path, owners/consumers/locks/retention, archive identity, independent restored-file hashes, expected reclaimed bytes and explicit authorization. An ETag, local checksum, merged PR or compressed pack alone is insufficient. Do not delete symlinks, containing folders with links/unknown assets, ignored files, active targets or unique source. Preserve the irreversible-deletion decision separately from this plan.

### C13 — README and agent onboarding complete for actual state

Owner: Codex, exclusive SC184L scope. Update README now: Direct/Thinking diagram; independent AutoML/ensemble matrix and configuration precedence; actual current versus feature/planned command examples; versioned downloads and legacy/native distinction; current bounded research/429-update limitations; dashboard and all47 task links; compact Git/S3 result policy; exact CI definition-versus-execution counts. Preserve generated benchmark bytes. Add `docs/AGENT_START_HERE.md` and update `docs/models.md` within P02 scope.

`README_COVERAGE.json` maps every direct human request to section, implementation dependency, runnable example/test and current qualification. SC184M approves exactly the README/agent-guide technical-prose exemptions in the existing path/reason mechanism to honor the direct human GP documentation request; retain all other scanner/emitter/runtime contracts and the preserved original failure. Run existing prose/link/model-manifest/paper-trace/generated-block checks; do not add a test mirroring a prose edit. Before final release, replace planned examples only after their real positive acceptance commands pass on the final head. Run each documented example in a fresh admitted environment/cache; record unsupported/missing/credential-bound examples explicitly. README100% coverage means no omitted request or misleading capability claim; it does not mean unfinished engineering is declared available.

### C14 — Integrate once, finish JA06 and final CI on frozen head

Owner: Codex; depends C00 and stopped ready packets C02–C07/C11/C13. Prepare one branch `codex/major-release-closure-20261019` in existing W from current main after collision check. Integrate preserved source commits, models and docs; resolve shared registration/tracker/coverage conflicts with owners, then stop source churn for final qualification. If remaining MR scope is not ready, maintain one truthful draft integration PR; do not open stacks or label a partial PR the completed major release.

Run validation matrix in `VALIDATION.md`: formatting/doc/model/schema and actual new targets first, then required fast lanes, full final declared package/feature/Python/emitter/admission/recovery tests under admitted resources. Preserve original failures and repair source; missing target/zero collection/unknown runtime is a blocker. Register genuinely new test targets in `deploy/ci/coverage.json` and regenerate inventory/matrix via owning tools after handoff; no hand-inflated counts. Bind source and build checksums. Final main contains source and all compact research facts, not raw artifacts. CI deferral for source authoring ends here; red or pending required checks do not become green from a workplan.

### C15 — Independent exact-head review, ordinary merge and main readback

Owner: Codex merge; independent existing reviewer. Re-query PR author and all GitHub-associated commit authors; reviewer identity must differ from every one. Check native approved fullhead equals branch head; green checks bind that head; no unresolved required conversations or branch divergence. Read effective protections, and do not use --admin or rewrite attribution. Use `/home/ubuntu/.local/bin/gh-role writer pr merge <PR> --rebase --match-head-commit <FULL_SHA>`. Any changed head invalidates old review/check eligibility. No blind retries or manual workflow cancellations.

Read actual resulting GitHub main SHA, verify intended commits/tree/manifest and release files, then new main-guard/full-main runs, runtime smoke and cold artifact download/reload on receiving main. Old PR CI is distinct from receiving-main qualification. Preserve PR/review/check/merge timestamps and logs. Merge or fix the one PR promptly after green; close only work proven obsolete with preservation evidence and owner disposition. Do not simply close valuable unmerged work to make the queue look clean.

Post-merge experiments can produce evidence only after C15 under the current admission order. The single next integration PR carries engineering and existing results; later experimental result/release updates use sequential reviewed PRs, one open at a time. Do not fabricate future outcomes to fit one PR, hold ready source indefinitely for an unadmitted run, or call the first source merge a completed scientific major release. Admiral records this sequencing explicitly.

### C16 — Major release, delivery and rollback

Owner: Codex/release owner after all required MR/P/JA and scientific gates, independent review and current main validation. Select version using actual release policy (not an invented tag). Update CHANGELOG, version/model pins and README through the same reviewed lifecycle, run `deploy/release/release.sh` dry run per `docs/release/README.md`, verify binary/runtime portability, immutable manifest, SBOMs, hashes/signature and checkpoint cold readback. Publish only when authority/gates are present, then verify tag→main→binary→release-set identities and rollback to previous qualified set. Artifact availability, scientific efficacy and serving promotion each get their own state. No main-only code tag may falsely imply all encoders/checkpoints or self-play are qualified.

### C17 — Local Git/worktree and folder retirement

Owner: owning captain plus custody reviewer; after integration and fresh consumer closure. Inventory registered trees AND matching sibling temporary directories with `lstat` (do not follow/delete symlinks). Record commits/unpushed refs, tracked/dirty/untracked/ignored data, processes/open consumers and build/evidence content. For a candidate to remove: prove merge/tree inclusion or explicit obsolete disposition, preserve all unique assets, independently restore exact commits/trees and non-Git files outside original store without alternates, then recheck no active consumer/lock. Log approved exact paths and reclaimed bytes; use ordinary `git worktree remove` for eligible registered lanes, no force. Reuse active W/W2; paper/review/source/linked/unknown assets stay held. The maintenance reporter is advisory, not a deletion engine.

### C18 — Dashboard and operational handoff

Owner: existing observatory/Admiralty owners; no competing deploy. Tie displayed source/build/model/checkpoint to published manifest, counters to actual run output and stale/unknown states to observation age. KPIs: train/dev loss and gap, LR/gradient/update norms, examples/tokens/updates, throughput/ETA/resource/budget, invalid proposal/fit/fault/duplicate rates, new evaluations/labels, fresh-feedback updates, family coverage/weakness, matched-cost gains and confidence interval; GP library unique programs/families/operations/preset vs algorithm drilldown; dataset eligibility/consumption funnel; S3 bytes/objects/versions/compression/deletion certification; checkpoints/archive restoration; PR/mainCI and task acceptance age. Avoid adding every metric as zero; expose missing telemetry honestly. Deploy current qualified source through owner, preserve retired reporting, and verify actual worker dispatch/result ingestion separately from observer freshness.

### C19 — Final closeout receipt

Owner: Codex with Admiral/captain sign-off. Produce `E/FINAL_CLOSURE.json` and compact Git context: final source/tree/build/release/model/checkpoint IDs; PR/native review/CI/receiving-main links; every original47 and separate66 task disposition with case/log hashes; experimental protocol/counts/outcomes and remaining claim limits; archive VersionIds/restore proof; clean/owned dirty/unpushed/worktree disposition; deployment and rollback receipt. Human-readable closeout states exactly what shipped and what did not. No DONE while required engineering, scientific, custody or release gates remain open.

## Stop conditions and failure handling

Stop the dependent action, preserve state and escalate one concrete packet when ownership collides, source/runtime/manifest identity changes, a required capability is missing, rights/split/evaluator/custody fails, budgets or85% whole-host CPU pressure are exceeded, a job's outcome is unknown, or review/checks do not bind the exact head. Do not stop independent admitted source/documentation work. Never repair a failed audit by changing original timestamps/hashes, discarding failure rows, reducing expected cases, relaxing gates or rerunning until lucky.

A blocker packet names task, full head/path, actual failing command/log/hash, required actor, next executable deliverable, and whether source/tests/experiment/merge alone is held. A CI queue delay is a resource/scheduling state, not a test failure. A systemError in Admiralty is routed through its existing owner, not interpreted as task completion.

## Progress accounting and handoff

Use `TASKS.json` for closure packet states and `MR_TASKS.json` as a dated reference; only the authoritative existing MR JSON/Markdown pair can change MR state after evidence. Count acceptance cases on actual final main; no single aggregate percentage merges source implementation, local proof, CI, science and deployment. Append short UTC status with full SHA and immutable evidence to existing coordination; store large logs compressed outside Git. Push work at least every30min or publish the exact blocker; never force-push. One actionable owner dispatch per new issue; suppress duplicate reminders while unchanged.

At handoff include exact host/checkout/branch/base/stopped head, dirty/untracked/ignored ownership, patch/commit list, task/acceptance mapping, test census and environments, log/artifact hashes and locations, PR/review/CI/main receipts, resource/admission/retention end conditions, remaining blockers and next ready task. Have the Admiral acknowledge plan adoption with actual task IDs/owner leases or a concrete correction. No automatic new monitor is created by this finite plan; the human's existing30min monitor remains separately controlled.


## Combined source integration — 2026-10-20T03:55Z

Root01a13a98 is consolidating stopped W2 `e66442c8847fe3155a08f1fadb6a899d95111349`
(tree `3cea9f8cbf7d8c244c2b45c8e4a54549405cc3b9`) in the existing main-based
`codex/major-release-closure-20261019` branch, preserving both histories. No new lane or
competing PR is created. C14 receipts are under
`/home/ubuntu/jope-evidence/release-closure-20261019/monitor-0340-20261020/c14-integration/`.

The stopped component has 626 passed / 0 failed / 3 ignored, including 33 native-study and 8 compile-fail
doctests (both included in 626), plus noncached clippy exit0. Independent driver-delta review
`review_w2_e66442c88_w1_native_driver_20261020.json` SHA256
`acefed409a34af1924f96de4e284dd0a9c6c9df9f397cc3f018f456ecc2890dd` has no blocking findings.
The original producer receipt has a future timestamp and omitted executed doctests; addendum
`ccb6ec4e21f4152d5af2d5bc49435779f312dfb9c16ea39669011835b15780a7` preserves it and corrects
those metadata errors. Observed full-log SHA256:
`cd3dc2c2473efbaaf36ad521f5295a844a383882b2d2cf3f4eed203919548880`.

Component results do not substitute for combined-head qualification or final native approval.
Combined tests, generated inventory, README consistency and exact-head review remain pending
until their actual receipts exist. The library validates the whole plan before any fit, uses
one ledger/counter, freezes all arms before any O and refuses retry/reassessment of an existing
study. It preserves the legacy finite budget API, which has no production caller.

Real admission root/provider/loader, a production study entry point and physical E9 linkage
remain open. Separate Qwen9B recovery-r3 performs inference and GP evaluation, not native
optimizer continuation; native training remains at 429 updates. No MR47 or older I66 task is
accepted by this integration note.


Count audit: [validation count notes](VALIDATION_COUNT_NOTES.json) separates 625 component top-level passes from its successful fresh-process replay, and 903 combined top-level passes from its replay. The earlier 626/904 aggregate summaries remain preserved. Combined 5c software tests passed; scoped clippy and CI hygiene/e2e still required repairs. No full release or scientific acceptance follows.
