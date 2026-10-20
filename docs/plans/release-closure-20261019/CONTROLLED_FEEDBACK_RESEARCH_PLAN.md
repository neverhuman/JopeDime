# Research plan revision 2.2 — retain the current model; isolate replacement research

Human clarification received 2026-10-20: a decision to replace the current deep learning model has **not** been made. This instruction supersedes any plan language that implies a model swap is part of the major release, or that Qwen/another candidate becomes the default after a pilot. It does not authorize a new experiment, provider or worker.

## Ownership and branch boundary

The current production model and its inference pathway remain the incumbent. The 893,957-parameter/282-token native experimental learner, the 401-action production actor, the historical 17,773-update checkpoint and Qwen 4–9B research candidates are distinct identities. Do not call whichever is newest the production baseline. Before a comparison, resolve and freeze the actual deployed incumbent's code, model/encoder/checkpoint hashes, grammar/vocabulary, preprocessing and download manifest; if that identity cannot be resolved, record the gap and do not claim superiority.

Model-replacement code, architecture/configuration changes, weights and experiment results belong on a **claimed research branch**, outside the release integration branch. The existing research Root `01a155bf-9fcd-7501-a15a-495c9435725b` owns R4 and related candidates; coordinate its current branch/lease before a source change. Require a recorded branch, base commit and exact experiment source identity before the next candidate implementation. Do not create another checkout, worker or PR merely to satisfy this instruction. If the current frozen experiment sources exist only in an evidence directory, preserve them and have their owner record a recoverable research-branch commit without altering the frozen run or its hashes. This plan is an instruction to establish that source lineage, not evidence that it already exists.

The existing integration writer retains PR387 for generic pipeline plumbing, tests, documentation and supported existing product routes. Do not merge a replacement architecture/weight set/default switch into that PR. Candidate support or opt-in research adapters must not select themselves for the normal Direct path, alter existing model bundles, or rewrite production/latest aliases. Same-architecture feedback/continuation experiments may proceed through their existing admissions, but their child checkpoints also remain unpromoted until evaluated. Research results and manifests can be published without promoting a candidate.

## Ordered comparison and promotion packet

1. Read `AGENTS.md`, `MASTER_PLAN.md`, the release-closure workplan and this revision, then the actual serving/download configuration and research owner's source claim. Write `incumbent.lock.json` and `candidate.lock.json` with exact code/encoder/model/checkpoint/tokenizer/grammar/preprocessing identities and the claimed research branch/base/head. Hash metadata and keep bulk weights outside Git.
2. Run engineering parity before scientific evaluation: encoding and input shape, GP AST/grammar and native transport, code generation and execution in supported targets, output schema, deterministic replay where promised, and checkpoint download/hash/restore plumbing. A parser repair or better raw text validity is not a model-quality win.
3. Freeze TRAIN/development/untouched family membership, seeds, evaluator and objective, direction of benefit, practical minimum effect, confidence procedure, resource limits and failure treatment before accessing outcomes. Match logical GP opportunities and optimizer exposure where relevant; report cold/warm latency, inference tokens, GPU/CPU seconds, memory and actual numerical fits separately. No validation/O-derived training targets and no post-hoc family deletion. Retain current-model, random/GP-only and other applicable frozen controls; the five-arm feedback test keeps architecture fixed and remains a different hypothesis from replacing the architecture.
4. Compare **end-to-end held-out quality and family-level uncertainty** against the pinned incumbent, including failures, GP validity/executability, diversity, latency, memory and cost. Training loss alone and more successful fits alone cannot establish superiority, especially with different tokenizers or objectives. A better result on a small development pilot is promising research; it is not confirmed generalization or serving approval. Freeze decisions and use a separately admitted untouched cohort for confirmation.
5. Produce `comparison.json`, compact per-family metrics, a short failure-aware interpretation card, exact source/config/input identities and immutable compressed raw/checkpoint references with object VersionIds, compressed/decoded/member hashes and restore receipts. Present the packet to the same Admiral and the human. **No automatic promotion:** keep the incumbent and production aliases until a separate explicit promotion decision, independent review, required checks, ordinary protected merge and receiving-main/runtime validation are complete. Preserve a rollback identity.

## Planned implementation and validation boundaries

Use the existing model-build config/selection code, `crates/jope-models/src/{release.rs,cache.rs}`, download contracts, README and native pipeline tests as the read-first authority. Before editing, resolve their actual paths/functions at the claimed head and write a disjoint path lease. Proposed tests must prove: default configuration selects the incumbent; a research opt-in requires explicit candidate identity and separate output prefix; a candidate cannot update production/latest aliases through the research path; download manifests bind every encoder and the model; and comparison reports preserve failed arms and exact resource denominators. These are planned acceptance cases, not tests claimed to have run in this documentation update.

Stop or hand off on missing source/ownership/resource admission, unresolved incumbent identity, test leakage, parser mismatch, an identity-changing source update after freeze, or a required review/check failure. Do not change the current model to work around any blocker. Report promising, inconclusive, adverse and confirmed outcomes distinctly. Keep the original failures, costs and timestamps.

## Current disposition

The Admiral adopted bounded local F0 metadata work and F1/F2 design packets at06:36:54Z. F3 numerical execution and model promotion remain unadmitted. The F0 metadata census is a partial inventory, not a training grant or a complete GP semantic union. Revision2.1's R3 engineering-inconclusive record remains below. R4 belongs to the existing research owner; this clarification does not restart or modify it. PR387's 4c40 head completed all11 current CI jobs successfully at06:49:22Z; a subsequent documentation commit must receive its own exact-head checks and review.


---

# Research plan revision 2.1 — R3 parser-interface finding

Observed 2026-10-20T06:35:57.730903+00:00. This addendum supersedes the earlier instruction to let R3 finish. The earlier plan's 06:12 running observation remains dated evidence; its source in PR387 at 4c40dbd5 is preserved.

The existing research Root, chat `01a155bf-9fcd-7501-a15a-495c9435725b`, classified R3 **INCONCLUSIVE_PARSER_INTERFACE_CONFOUND** and stopped its own controller at 06:20:11 UTC. The current response finished at natural EOS; the worker exited 0 and released the GPU at 06:26:46 UTC. This is an owned engineering-integrity stop, not an unverified SSH-key incident, training convergence or a scientific negative. The controller's last counters are 414 successful numerical fits, 440 charged slots, 13/20 frozen ranks and zero O assessments. Worker terminal counters include 11 new completed calls and 103,638 generated tokens; the last completed response was not incorporated into the controller's terminal counters. Preserve both counters and their timestamps instead of silently reconciling them.

Independent audit `new-control-round-audit-0611/AUDIT.json`, SHA256 `0a25f4c687c49d44baead103547c98e7ee916fe666fb9aba4346485174b947e3`, found all eight family09 shuffled-round3 strings had internal spaces. Python's validator removed whitespace, while the driver sent the original text to a native parser that does not skip it. All eight were rejected before numeric fitting: eight logical opportunities, zero physical fits, 8,468 inference tokens plus parsing/driver cost. The family retained 14/24 usable model proposals, so this did not trigger a nonexistent per-batch usability minimum or the actual zero-usable-family threshold. The design-level interface confound, rather than a new threshold, motivated the owner's stop.

Read the immutable audit, `INTERFACE_DESIGN_AND_REUSE_ADDENDUM.json` SHA256 `1f9ddceff770e00cbffd8c062a51ed2ae23165a3a965082bf3b34b5a32fc00df`, and `engineering-parser-interface-20261020T0611Z/{INTENT.json,ACTION_RECEIPT.json}` under `/home/ubuntu/jope-evidence/reasoner-eos-9b-20261020/recovery-r3/` on xbabe1. The same research Root retains the next revision and experiment; do not restart R3, alter its sources or rerun its rejected slots under old receipts.

## New first packet: parser and transport parity

Before another model/GP evaluation, the existing owner prepares a separately frozen prospective interface revision and exact source/test path lease. Read the frozen `v3_support.py::canonical`, `driver_v3.py` native `inspect` call and native `GpProgram::parse`/`parse_gp_node` source copies identified in the audit. Preserve the current four arithmetic operations, node/depth/leaf rules and evaluator semantics. No architecture change or operator expansion follows from a serialization disagreement.

1. Define one accepted wire grammar and a versioned canonical serialization. Validate a parsed AST against the allowed grammar, then produce the native representation. Keep raw output plus canonical form and an explicit transformation receipt. Do not globally remove whitespace: other supported GP dialects can contain quoted names or literals whose internal spaces matter.
2. Add differential positive/negative fixtures through **both the actual validator and actual native inspection boundary**: nested spaces, tabs/newlines, canonical text, nested operators/leaves, arity, node/depth limits, unknown leaves/operators, forbidden constants, malformed/trailing input and quoted/literal behavior for the relevant full-library dialect. Check identical AST/semantic identity where accepted, matched refusals and zero numerical-fit claims from parser-only tests. Pin actual binaries and source identities. A Python-only fixture cannot qualify this boundary.
3. Freeze the new interface/protocol before new outcomes. Recompute requests and cache eligibility against exact prompt bytes, request ID, seed, tokenizer/model/revision/precision, decoding and source policy. Keep every old slot, failure and cost. A change before a later adaptive prompt requires recomputing that prompt; stop cache reuse from the first divergence unless exact equality is independently proven. No fuzzy/quality-selected reuse, retroactive slot repair or cost reset.
4. Independently review the new revision and obtain current resource/evaluator admission through the existing owner. Archive R3's stopped source, responses, rejected strings, terminal ledgers and interpretation in compact Git wave records with immutable compressed bulk references and restoration evidence. Run the new revision only after its qualified source/input/ownership boundary is explicit.

After that packet, the five-arm feedback plan, full GP/S3 score-reuse census, explicit CE-target changes, exact checkpoint recovery and family-level cost-controlled comparisons remain the next phases. Parser parity also becomes a prerequisite for F0's claimed executable/reachable surface and F1's real outcome path. The native learner still has 429 completed updates; this incident does not supply new loss evidence.

## Integration and owner handoff

The published plan/proof at PR387 head 4c40dbd5 remains clean and its current CI is running. This dated addendum is delivered outside the source tree and in the PR description so it does not cancel working CI. Fold it into the same repository workplan in the next ordinary documentation update after that run reaches a terminal state; preserve the original dated observations. No extra PR or training launch is authorized. The Admiral is asked to acknowledge the new prerequisite and the existing research owner, rather than dispatch a competing parser fix or restart.


---

# Controlled feedback research plan — revision 2, 2026-10-20

Status: proposed priorities for the existing Admiral and owners; no new experiment is admitted by this document. This revision incorporates the human-supplied external review and the request to use the full GP library and historical S3 measurements. It refines existing C09/C12/C14 work and the separate MR47/I66 ledgers; it creates no replacement task count.

## Objective, scope and completion

Demonstrate a small complete cycle: policy proposes → evaluator measures → verified outcomes are published → outcomes change explicit learning targets → a child checkpoint consumes those targets → subsequent search uses that child → a controlled comparison measures the benefit. Keep the encoder, model architecture, GP operators, decoding support and objective coefficients fixed in the first experiment. Prove feedback value separately from more optimization and better retrieval memory.

Non-goals: changing R3 in flight, expanding its sample or retry budget, new models/providers/screens/worktrees, ingesting every S3 object, changing the production admission policy, a support expansion or coefficient sweep, declaring self-play from loss alone, or deleting storage. Existing Direct/Thinking and optional AutoML/ensemble product requirements remain in the major-release plan; the scientific trial does not silently turn on AutoML.

Completion has distinct levels:

1. **Plumbing:** at least one genuinely evaluated TRAIN outcome reaches a verified child snapshot, changes a declared target and is proven consumed. This proves a connection, not useful learning.
2. **Development pilot:** all frozen comparison arms complete with valid controls, nonzero measured outcomes, exact cost accounting, actual child-guided search and a family-level analysis. A null or adverse effect is a valid completed experiment.
3. **Scientific confirmation:** after development choices are locked, a separately admitted untouched family cohort establishes the preregistered primary contrast with uncertainty and practical effect size. Repeated qualified cycles are required for a self-improvement claim. One favorable recovery or validation result is insufficient.
4. **Release:** exact-head review/checks, ordinary protected merge and receiving-main validation plus the remaining runtime/model/physical-worker gates. None of the preceding levels substitutes for this.

Assumptions to verify: lawful TRAIN membership and parent families can be resolved; real evaluator/cache identities and an incumbent baseline are available; current checkpoint semantics can be recovered; a new snapshot can retain immutable ancestry. Missing assumptions produce a specific blocker and next owner action, not invented rewards or a smaller unreported study.

## Evidence checked for this revision

The supplied review is advisory source material, not a new owner ruling or a fresh fleet observation. Its historical 294-fit snapshot is superseded by actual observations: recovery-r3 had **414 successful fits, 440 charged slots, 10 new terminal EOS responses and 13/20 frozen ranks at 06:12 UTC**. O assessments and optimizer updates remained zero. The frozen Qwen inference process is separate from the native learner.

The native learner completed one epoch, 429 updates, 6,857 TRAIN contexts and 263 development-validation contexts. The same 16-context validation panel fell from 4.39456 to 1.38138; final full-263 validation was 1.34379, a different denominator. There are 429 pending TRAIN pairs and 429 VALIDATION pairs, not 858 trainable rewards. Only 2,267/6,857 TRAIN contexts had comparison-related support; 4,590 did not. Even an optimistic 429 newly supported distinct contexts reaches only 2,696/6,857 = 39.3%, before overlap, failures and comparability exclusions. This is an upper bound, not a forecast.

Two review hypotheses have been checked against the actual source:

- `pipelines/native/neural_native_train.py::run` constructs AdamW with constant learning rate 0.0003 and calls no scheduler. Checkpoint 000429 records 0.0003. An exhausted cosine schedule is **not** the observed explanation for this run. Resume qualification must still bind schedule identity explicitly, including `none`; a future schedule reset is a separate named intervention.
- `neural_native_batch.py::objective` and `neural_native_train.py::run_microbatches` already normalize each term by its supported-context denominator across the complete macro batch. Missing support contributes connected zero with denominator zero. Sparse support alone does **not** prove CE gradient dominance or batch-wide dilution. Preserve those semantics and audit actual gradients before changing weights. Gradient norm before clipping at 1 is already recorded; learning rate, update norm and occasional per-term shared-parameter gradients need complete reporting.

Checkpoint state SHA-256 is `d412f7b67e2130dc95a63021c6894940b3f081f0021aa3e4c564aa90e400e5b3`; parameter SHA-256 is `b888b5d66fda926c3609fa068a42a022a079e2d4bbc8db3b5badec8f51acaf20`. The 893,957-parameter/282-token learner uses a fixed 6,640-program training-teacher trie. This is neither the full GP syntax nor the separate 401-action production actor. The historical 17,773-update policy is another candidate requiring compatible lineage and equal-identity replay; neither age nor “latest” establishes the strongest baseline.

## Read first, ownership and writable scope

Resolve all repository-relative paths under:

```text
R=/home/ubuntu/JopeDime
W=/home/ubuntu/JopeDime/.agent/worktrees/fleet-main-296d07e1
D=/home/ubuntu/xbabe2_kicker/captains-20261017
C=/home/ubuntu/jope-evidence/campaigns/flywheel_native_derived_20261019T1303Z
```

Read in order:

1. `W/AGENTS.md`, `R/AGENT_CHAT.md`, fresh relevant rulings/claims/receipts in D, `MASTER_PLAN.md`, `docs/adr/ADR-0005-canonical-learning-loop.md` and the current open PR queue. Verify the existing source writer and reviewer before any edit.
2. `docs/plans/release-closure-20261019/{WORKPLAN.md,TASKS.json,FLYWHEEL_VALUE_WORKPLAN.md,IMPLEMENTATION_STATUS.md}`, the unchanged `major-release-20261018/{GP_UPGRADE_WORKPLAN.md,GP_SPEC_COVERAGE.json,MAJOR_RELEASE_TRACKER.json}` and the original algorithmic-selfplay plan. This is an additive priority refinement.
3. C's frozen `config.json`, `derived_spec.json`, `neural_native_train.py`, `neural_storage.py`, `neural_outbox.py`, `training/{progress.json,batch-manifest.json,outbox.sqlite,checkpoints/000429/manifest.json}`. Open SQLite with `mode=ro`. Treat C as immutable evidence; implement in W, never patch a completed campaign.
4. `deploy/dl-flywheel/pipelines/native/{neural_native_train,neural_storage,neural_native_batch,neural_native_stream,neural_outbox,neural_decode}.py`; `deploy/dl-flywheel/jope_flywheel/{training_snapshot_native,s3_library}.py`; existing `tests_native/test_neural_{native_train,native_batch,outbox,decode}.py` and `test_training_snapshot_native.py`.
5. Full GP authorities: `crates/feat-math/{registry/operators.toml,src/registry.rs,src/dsl.rs,src/full_rust.rs,src/heads.rs,src/evolve.rs,src/math_expr/,src/advanced_v3/grammar.rs,src/kernel/{capabilities.rs,profile.rs,registry_v2/,ir/},src/numeric_v2.rs}`; `docs/contracts/operator-registry-v2.md`; `tests/{registry_agreement,capability_join_audit,registry_vocabulary}.rs`; actual engine grammar/vocabulary assets identified by those tests. `src/gpr.rs` is the Gaussian-process estimator; its name is not a complete genetic-programming grammar.
6. S3/data/score authorities: `catalogs/{s3.json,training-artifacts.json}`, `docs/runbooks/training-artifact-locations.md`, current `docs/research/waves/INDEX.jsonl`, `crates/jope-learning/src/{retention/,exe_cache.rs,replay/,pools/,retrieval.rs}` and `crates/jope-engine/src/training/{replay_bridge,consumption,checkpoint_commit,continuation,ancestry}.rs`. Reuse the October-19 storage audit and October-18 exact-version inventory manifests; retain their observation dates.

The **same Fleet Admiral** adopts priorities/resource admission. Existing Root W owns all integration/index/shared docs and the single PR387. The existing research Root `01a155bf…` owns R3 and its controller; this plan does not create another controller. The project captain owns the outside-tree evaluator/cohort operational packet; an exhausted provider is not retried. The same independent reviewer finishes current-main I03/I04 receipts, then reviews the stopped corrected PR head; do not poach it mid-command. Native approval identity must be verified separately from process independence.

New implementation needs a fresh written disjoint path lease from the existing owner. Likely source additions, **not yet implemented**, are `pipelines/native/neural_feedback.py`, `neural_acquisition.py`, `neural_resume.py` and `tests_native/test_neural_feedback.py`, `test_neural_acquisition.py`, `test_neural_resume.py`. Extend existing functions before adding a second trainer/store: `checkpoint`, `BatchStream.reserve/commit/adopt_manifest/verify`, `Outbox.record_batch/claim/finish/verify`, `NativeContextLoader.get`, `run_microbatches` and `run`. Any resume module under `pipelines/neural/` is a different historical lane; inspect its contract, do not confuse it with the proposed native-path file.

Forbidden writes: C, live R3/policy files, canonical dirty R source, the other owner's checkout/index, generated registry tables/fixtures/grammar assets, sealed partitions, keys/provider settings, protected evaluator hashes, other projects, symlinks and unknown containing directories. Shared manifests, inventories and targets serialize through W. No new clone/build lane. Inventory/grammar changes use their existing generators and independent proof, never hand-edited hashes.

## Packet R0 — finish R3 without intervention

Owner: existing research Root. Preserve its current protocol, models, controls, all interrupted/censored histories and finite scientific endpoints. Freeze all 20 ranks before O; independently audit cheap, retrieval, random and shuffled controls before a positive interpretation. Ranks/fits/tokens are not independent families. Keep adoption false until its existing gates qualify. Record terminal outcome even if inconclusive or negative; do not relaunch identical benchmarks for utilization. This packet is independent of the following source/planning work.

## Packet F0 — inventory GP reachability and reusable S3 evidence

Owner: captain outside-tree census; W integrates reviewed compact results. Reuse metadata before new body reads. Produce versioned `inputs.lock.json`, `gp-surface-map.json`, `historical-score-reuse.json`, `coverage.json` and `blockers.json` under a fresh evidence directory chosen by the owner.

1. Map dialect + spelling + semantic version/profile, aliases, parameter domains/effects, input schema, row axis, task type, evaluator, admission, generation mask and each emitter route. The checked additive v2 registry has **90 rows: 61 legacy-recipe and 29 advanced-v3**, including compatibility/quarantined/disabled entries. It is not the union of every historical grammar or every executable program. Join `Manifest::current`, `registry_v2_manifest`, `operation_profiles`, symbolic/row-vector forms and the actual checkpoint tokenizer/trie. Count declared, runtime-supported, generatable and checkpoint-reachable separately. For example, legacy `lag/roll` act within a feature row while advanced versions concern ordered sample rows; spellings cannot be merged by name.
2. For each measured or evolved program record canonical/source hash, dialect/profile/semantic identity, DSL validity, allowed evaluator, tokenizer representability, trie membership, training-interface representability and actual next-search availability. Preserve out-of-support children for retrieval/ordinary-GP study; exclude them from the first fixed-support learned-emission contrast. Expanding the grammar or trie requires a separate support-transition experiment, not an in-flight correction.
3. Build the storage funnel: physical objects/versions → decoded programs/contexts → comparable score records → role/admission-valid observations → snapshot members → actually consumed targets. The dated estate total 35,161,503 objects/2.95 TiB is not a corpus size. The all-version learning inventory covered 34,287,355 rows; lower-bound historical teacher groups held 30,848,574 rows. Small context claims, attempts, dashboards, checkpoints, archives and replicas must not count as independent labels. September catalog bucket counts are historical, not a fresh global census.
4. Join `experiments/gp-teacher-v6/{contexts,attempts,campaigns}` metadata, admitted score/cache indexes, retained HWM/derived snapshots, BeyondArena dataset/view inventories, historical teaching shards and compact research-wave results. Keep cross-project DOPE/Jarambit namespaces, sealed/custody data, restricted licenses and unknown consumers held. Historical `reference_v1`/qualification shards are useful study candidates, not current strict-label admission.
5. A reusable score key must bind dataset/context and content hash, parent/family, row/split/cut/preprocessing, program, evaluator/runtime/profile, label/baseline protocol, source attempt, status/censoring, role/exposure and exact immutable object reference. Different split/evaluator/metric or unknown ancestry means noncomparable; use as background evidence only. Match a cache record by full key and verify its referenced bytes before reuse. The same program on different contexts is useful compatibility data, not an automatic duplicate.
6. Report diversity by parent family, numeric/categorical/missingness/shape strata, task, program skeleton/operator family, error/feasibility regime and runtime band. Separate already exposed development data from genuinely unseen confirmation families. Do not compute “all data diversity” from object names alone. Rank recovery opportunities by new rankable TRAIN families/comparisons per projected cost, not raw byte count or historical winner-only score.

Accept F0 when all included prefix/dialect sources have a pinned observation time/hash, exclusions/unknowns reconcile and an actionable eligible TRAIN pool is produced. If metadata cannot establish payload or role closure, leave that group pending and name its owner/action. No blanket S3 listing/body download/deletion or lifecycle change. Refresh recent publisher/consumer deltas rather than rescanning 35 million bodies.

## Packet F1 — prove evaluated outcomes change learning targets

Owners: existing evaluator/source owners, serialized through W. First use a bounded proposed plumbing cohort of 16 eligible TRAIN contexts from at least 8 parent families; this proposal is not execution admission.

1. Read the 429 pending TRAIN pairs without mutating the historical outbox. Create an immutable child request set with original emission IDs, source checkpoint, full score key, role and attempted-state lineage. Never reset a historical unknown lease. Classify cache reuse, new evaluation, refusal and unresolved outcome separately; VALIDATION pairs remain evaluation-only.
2. Exercise the actual trusted evaluator path under a current written resource/data/evaluator grant, then `Outbox.finish(..., verify_outcome=...)`. A subprocess exit or a claimed score is insufficient. Verify measured result plus error/censoring evidence, append-only publication and snapshot membership. One accepted new outcome proves only plumbing.
3. Implement a versioned target policy, provisionally `feedback-targets/v1`, in the new `neural_feedback.py` overlay. Preserve existing q/baseline/grid/tie/runtime conventions from `consumer/dltrain/labels.py`; never infer q from fitness. Define quality, feasibility and censored-runtime eligibility independently. Form BT comparisons only within the identical context/evaluator/split with distinct quality grids; list support needs at least three eligible measured programs. Infrastructure unknowns have no fabricated numeric reward. Program failures receive only the failure supervision the existing contract actually supports; never a positive CE target.
4. **CE transformation is mandatory:** the current loader sets CE only on the frozen HWM winner, so publishing a better score alone leaves generation targets unchanged. For the first fixed-support trial, require an admitted comparable incumbent and a measured, feasible, in-trie child whose quality improvement clears a preregistered grid/noise margin. Keep the incumbent until that condition holds. Select at most three measured programs tied in the best accepted quality grid, ordered by frozen complexity/runtime tie rules then program hash; use the existing uniform per-context CE-eligible mean. Preserve other useful measured programs for ranking. Store every inclusion/exclusion and target-policy identity. Freeze the margin and tie rules before outcomes; if noise calibration is unavailable, classify near ties as unresolved instead of claiming improvement.
5. Materialize a **new** snapshot identity and parent-to-child target diff: new pairs, rankable contexts, parent families, quality/feasibility/runtime labels, CE additions/removals, in/out-of-support counts, original result locators. Do not edit the old winner map or old manifest. Prove the batch loader exposes those exact target hashes and that a nonzero optimizer batch consumes them; database publication is not consumption.

Acceptance tests to add: role leakage refusal; context/evaluator mismatch; invalid cache; tied/noisy/failed outcomes; at least two distinct comparable scores before rankability; three before list support; better measured in-support child changes CE membership; out-of-trie child cannot silently expand decoding; duplicate publication/emission idempotency; child target membership actually appears in gradients/consumption receipt.

## Packet F2 — qualify continuation and diagnose the loss

Owner: existing native source writer after a stopped-head handoff; read-only checkpoint audit can proceed independently.

1. Extend `neural_storage.checkpoint` with a versioned explicit optimizer param-group/defaults record, current LR and scheduler identity/state (`none` for the inspected run), named tensor mapping, CPU/CUDA RNG, any Python/NumPy/local RNG actually consumed, preprocessing/tokenizer/support/source pins, planner epoch/cursor/shuffle and accumulation position. Existing state has weights, Adam moments and Torch RNG; that does not establish a working resume API. Record AMP scaler as explicitly unused if unused. Do not deserialize arbitrary pickle or infer absent state.
2. Separate `resume_exact` for the same admitted semantics from `warm_start_weights`/optimizer reset. Express the continuation's intentionally changed maximum epochs/updates as an allowlisted parent→child config diff; source/model/objective/data-support changes are not allowed under exact recovery. Dataset/target changes in F1 are a declared new child experiment, with their own lineage and adoption boundary.
3. On synthetic CPU fixtures compare uninterrupted N+K updates with checkpoint/restart N then K: parameters, optimizer moments/step/LR, next batches and emissions. Inject interruption at reservation, durable-emission ACK, optimizer checkpoint and planner commit boundaries. No duplicate/omitted update or emission; uncertain state stops for reconciliation. The existing one-epoch loop creates output exclusively and insists on fresh initialization; implement recovery rather than invoking it again in the old directory.
4. Retain per-term eligible contexts/examples/pairs/tokens, unweighted numerator/denominator, coefficient/weighted contribution, before-clip gradient norm, actual LR and parameter-update norm. Every 32 updates on one fixed TRAIN diagnostic macro batch, measure per-term shared-parameter gradients with dropout off while restoring RNG and without modifying `.grad`, optimizer or planner state. Count diagnostic compute. Do not interpret 33.1% support as 33.1% gradient weight.
5. Compare teacher-forced CE with exact constrained-trie probability assigned to useful programs, EOS legality, unique programs/skeletons, entropy/concentration and actual executed portfolio quality. Keep probabilities normalized under the actual constrained decoder; raw token CE and trie distribution are different quantities. Record whether lower loss merely repeats historical teachers.

Accept F2 only with true interrupted/uninterrupted parity and all consumed state attested. Otherwise report a named warm-start experiment and match that reset across trained arms. No automatic recovery or historical budget extension. Do not sweep coefficients until the measured-target and denominator audit works.

## Packet F3 — controlled development pilot

Prerequisites: F0 eligible pool, F1 target/consumer proof, F2 qualified initialization, independent review, and fresh scientific/resource admission. Proposed pilot settings for Admiral review: 128 TRAIN contexts across at least 32 parents if eligibility permits; at most 128 additional updates per trained arm, context batch size 16; identical constant LR 0.0003; 25% new-feedback sampling for feedback arms with a frozen old-data replay rule; 32 logical search slots per development evaluation context/arm. Freeze acquisition cost B, seeds, pilot parents, caps and cache rules in the manifest **before** launch. These proposed numbers do not enlarge any existing grant. If a sufficient independent family pool cannot be formed, deliver plumbing evidence and an explicit design blocker.

| Arm | Weights/state at start | Learning data | Retrieval/HWM during assessment | What it identifies |
|---|---|---|---|---|
| F | Frozen checkpoint 429 | No updates | Frozen M0 | Reference |
| C | Same 429 plus identical optimizer/RNG initialization | Old admitted snapshot | M0 | Extra optimization |
| U | Same start and update budget | Old + uniformly acquired outcomes | M0 | Added feedback beyond continuation |
| T | Same start and update budget | Old + targeted acquired outcomes | M0 | Acquisition benefit beyond uniform |
| M | Frozen 429 | No updates | Reviewed updated memory M1 | Memory-only improvement over F |

M1 is the locked union of admitted new outcomes for the memory diagnostic; it is not silently available to C/U/T. Main policy contrasts keep M0 identical and immutable. Fix retrieval, GP operators, search settings, decode support, preprocessing and evaluation ordering. Include ordinary-GP/cheap/retrieval baselines where the existing protocol requires them; do not weaken controls to save slots. A weights×memory interaction study is later work, not a reason to duplicate this first trial.

Match optimizer updates, initial optimizer state, diagnostic cadence and old/new sampling rules across C/U/T. C substitutes old admitted examples into the new-data sampling slots. Record actual tokens, supported denominators and gradient contributions; changing support can still change supervision even with identical coefficients. Choose one default checkpoint rule, e.g. final fixed update, before running; any development-selected alternative is secondary. The full 263-context set may guide development selection only with its selection history recorded; it is not sealed confirmation.

Use a simple active rule. On the remaining eligible TRAIN candidate pool, define a deterministic priority set from unsupported-context/family coverage first, then premeasurement decision-margin uncertainty, then neighborhoods of **already measured** errors, with stable hash ties. No unseen candidate's actual error is available for selection. T draws without replacement from a 75% priority-set/25% uniform mixture; U draws uniformly. Log conditional propensity `0.75*I(i in priority)/|priority| + 0.25/|remaining|`, eligible-pool hash and costs at every draw; if priority is empty, use uniform probability 1/|remaining|. Do not use naive one-step propensity as a full trajectory inclusion probability. The randomized component prevents zero-support exploration; no learned acquisition model is introduced yet.

Freeze equal logical acquisition budgets for U/T, count the cost of selecting/evaluating incumbents, all failed/censored attempts and diagnostic work, and predeclare how both arms see the cache. An arm running second must not receive uncharged warm-cache advantage: use a common immutable cache snapshot or independently meter reuse. Charge logical slots once per declared request, but separately record fresh CPU/GPU seconds, inference tokens/time, cache retrieval cost, unique outcomes, retries and infrastructure interruptions. A cache hit is not fresh-fit compute. Report quality at matched logical slots and at matched actual resource budgets/curves; the preregistered primary endpoint controls the conclusion.

Suggested primary contrast: U−C equal-logical-budget parent-level search improvement, demonstrating new feedback beyond more optimization. Secondary planned contrasts C−F, T−U and M−F identify continuation, acquisition and memory effects. Freeze direction, metric, failure treatment, practical improvement threshold and multiplicity handling before sealed outcomes. Include end-to-end acquisition/training costs alongside post-training search cost; a fixed-checkpoint quality win is not automatically an amortized compute win. Use paired parent-family aggregates and family-clustered intervals; seeds/fits/folds/ranks are repeated measures. A positive development interval nominates a new untouched confirmation cohort; intervals including zero mean inconclusive, not “almost passed.” Preserve all failures and external interruptions under the analysis policy.

## Validation, receipts and next report

Commands below are worker acceptance commands **after implementation**, not claims they have been executed. Use the existing W target and a pinned existing CPU PyTorch interpreter; no installs/fallback venv or GPU exposure. Required variables deliberately fail closed until the owner supplies the verified interpreter and resource scope.

```sh
cd /home/ubuntu/JopeDime/.agent/worktrees/fleet-main-296d07e1
rtk proxy env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=deploy/dl-flywheel/pipelines/native:deploy/dl-flywheel:deploy/dl-flywheel/tests_native \
  "${JOPE_NATIVE_FIXTURE_PYTHON:?pin existing CPU PyTorch interpreter}" -m unittest discover \
  -s deploy/dl-flywheel/tests_native -p test_neural_resume.py -v
# Repeat the same bounded invocation for test_neural_feedback.py and test_neural_acquisition.py;
# then the existing train/batch/outbox/snapshot regressions affected by the implementation.
rtk proxy env CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only \
  CARGO_NET_OFFLINE=true cargo test --offline --locked -j2 -p feat-math \
  --test capability_join_audit -- --exact zero_unclassified_primitives_across_all_surfaces
rtk proxy git diff --check
```

Run these through the current owner-approved fleetcap/taskset/nice envelope, not outside it. Verify nonzero collected cases and exact exits; missing interpreter, zero cases, filtered-only output and skipped GPU paths are not success. The GP audit is acceptance for a generated surface-map change, not a request to repeat it during planning. Source/classification/layout/prose/current-head CI and independent exact-head review follow the existing C14 workflow. One PR, no new stack or blind reruns.

Proposed records must use strict versioned schemas, with unknown admission fields null and `execution_admitted:false` until real owner disposition:

```text
experiment manifest: parent_source/checkpoint/optimizer, child_source, config_diff,
  target_policy, old/new snapshot hashes, split/exposure/parent roster, GP surface/support,
  arms, sampling/cache rules, cost caps, seed plan, primary/secondary analysis, stop rules
outcome/target row: context+parent, program+profile, role, evaluator+baseline,
  original attempt/status/censoring, immutable result ref, target masks, CE decision/reason
consumption row: child_checkpoint, snapshot+target-policy, batch/step, target-member hashes,
  denominators, parameter-before/after, outbox/planner commit and measured-result ancestry
```

For each wave commit small `card.json`, `metrics.json`, `wave.manifest.json`, dense `context.md` and an INDEX row under `docs/research/waves/`; update `catalogs/training-artifacts.json` and the location runbook through the owner. Bulk checkpoints, scores, raw failures and inventories stay content-addressed and compressed outside Git. Record exact S3 URI/VersionId, compressed/decoded/member hashes, schema/counts and independent restore evidence; a two-host local copy or metadata HEAD alone is not an S3 all-member restore. Do not duplicate shared corpora in each wave. Store target deltas/reusable unique outcomes once, preserve old manifests and original failures, and reference them. Git publication commit/PR and receiving-main status are explicit. Storage deletion still requires separate current consumer-closure and restoration proof.

The next owner report leads with: newly measured TRAIN pairs; newly rankable contexts **and parents**; usable/informative comparisons per logical slot and physical cost; dataset/operator/skeleton diversity; evaluation→publication→consumption latency; number/hash of changed CE/ranking targets; child checkpoint consumption and actual next-search use; equal-cost U−C/T−U/M−F effects with uncertainty. Loss/LR/gradient/update curves accompany these figures. Utilization alone is insufficient.

Stop dependent work on stale ownership/admission, changed source/model/data/evaluator pins, invalid controls, leakage, unsupported target/decoder expansion, missing baseline, nonfinite loss/gradient/update, broken journal/checkpoint mapping, unknown interrupted attempts, resource/storage guard breach or exhausted provider. Preserve failures and continue other independent admitted work. xbabe1's live Qwen jobs take priority; xbabe2/3 storage holds and current fleet resource rulings apply. Never touch xbabe0/Jeryu or signal another owner's jobs.

Admiral handoff must return the same owner per packet, admitted versus pending prerequisites, exact next executable deliverable and checkpoint date. R0 can finish alongside read-only F0/F2 audit. F1/F2 source writes share W and must serialize; F3 waits for both. No new worker is assigned by this plan.

## Research basis and limits

[Expert Iteration](https://arxiv.org/abs/1705.08439) motivates improving search targets and then improving search with learned guidance. [DAgger](https://arxiv.org/abs/1011.0686) motivates supervision on learner-induced cases under its reduction assumptions. These motivate this design; neither establishes a guarantee or measured JopeDime gain. [Cawley and Talbot](https://www.jmlr.org/papers/v11/cawley10a.html) explain why repeated model selection can bias a finite evaluation set; development checkpoint selection and sealed confirmation therefore remain separate. The control choices, acquisition mixture and proposed pilot sizes above are this plan's engineering proposals, not claims established by those papers.


## F0 delivery and F1/F2 implementation handoff

The [compact local metadata census](F0_METADATA_CENSUS_20261020.json) records actual counts, source/input hashes and a compressed metadata capsule with local and xbabe2 full-member restoration. It is partial: no S3 refresh, numerical evaluator, checkpoint load or training occurred. The proposed plumbing cohort contains16TRAIN contexts from16parents, selected before new outcomes; it is not admitted. The original9row/8parent selector remains immutable in the archive.

The ordered F1/F2 source/state packet is `/home/ubuntu/jope-evidence/release-closure-20261019/monitor-0638-20261020/F1_F2_DESIGN_AND_GAPS_20261020.json`, SHA256 `45bf641ccae68c62330adc13f6c3232de9d5e118d8a5a66b147cd4f305a1ce62`. It names actual source functions, proposed files, edge cases, validation and remaining source/data/evaluator/runtime grants. This is design delivery, not implemented feedback or resume. Existing owners serialize source work after exact-head handoff; candidate replacement remains on the claimed research branch.
