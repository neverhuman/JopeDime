# JopeDime evergreen GP learning flywheel

Implementation record and continuation plan, 2026-10-09 UTC. This document supplements `DL_FLYWHEEL_PLAN.md` and `DL_WHITE_PAPER_TRACKING.md`. The goal is 100,000 unique, traceable regression variants with fewer than 2,000 rows, freshly encoded and actually scored programs, then admission into measured training runs.

## Current correction: Kumo Small only and real optimizer accounting

The latest request supersedes the earlier eight-block forward plan. New encoding is **Kumo Small only,2,051 values**; eight-block historical artifacts remain preserved. This is a fresh candidate. Neither Wave4 nor mixed-input bf-kS-lion proves its performance. PriorLabs/TabPFN3.5 and other unused encoders are excluded from new encoding.

The live /paper/ page distinguishes published data runs, required encodings, measured GP results and actual optimizer exposure. At this correction the new trainer has not launched: zero actual batches and zero inputs consumed. GP16 fixed-candidate scoring supplies labels/opponents, not neural outputs. Kumo-only encoding is running for remaining112 variants. The original100-variant pilot is separate from the128-lineage evergreen cohort.

The new batch planner enforces shuffled exhaustive epochs, one parent lineage per batch, durable checkpoint/intent reconciliation and order-independent membership uniqueness across restarts. All128 cohort rows remain eligible, pending or refused. Original7,120-view admission into the generator remains an explicit gap. See DL_FLYWHEEL_CORRECTION_PLAN.md for exact ownership, paths, tests, stop conditions and completion criteria. Later completion receipts supersede these dated counts.

## Current verified state

- The corrected full Wave4 campaign completed 45 fits over 7,120 views, 1,579 dataset IDs and 1,068 retained lineages. It does not generate programs. Selected validation losses improved, but the matched LightGBM comparator led on fitness R²: 0.7849 versus 0.6902. Synthetic-data growth is an experiment, not evidence that this deficit is fixed.
- All 1,068 representative parent lineages are qualified for prepared research transforms. There are 1,014 train and 54 validation lineages (5.06%). Parent rows range from 64 to 1,838; widths range from 1 to 500. These are one representative view per lineage, not a replacement for the full 7,120-view corpus.
- The first pilot has 100 unique variants from one parent, with all eight fresh encoder blocks complete and in S3. The two publication stages contain 1,200 verified objects, including 800 vectors, totaling 33,202,223 bytes. Every published object has an exact VersionId readback and SHA256. The original metadata remains immutable; new completion receipts join the additional encodings.
- Two programs were actually scored by the owned local Lambda evaluator on one modified pilot dataset. Four inner-fold fits produced 384 predictions; the separately emitted and compiled Rust code matched all 384 bit-for-bit. Both measured program results and four native canonical score rows are now in S3 with exact-version artifact verification.
- An evergreen intake service is running on xbabe2. It generated and published its first 128 variants from 128 different parent lineages (125 train, three validation), with 512 exact-version S3 objects totaling 12,241,675 bytes. It is waiting for downstream encoding, scoring and admission. Seventy-eight variants changed targets with 1% noise scaled from designated inner-fit labels; fifty retained the original targets. Target: 100,000; current generated, published, encoded, scored and admitted counts are separate.
- The new program gate and score publisher are distinct from integration into a program-generating trainer. Existing Wave4 and token-CE validation do not themselves produce new GP programs. Do not claim that every future generator is covered until its emission path calls the gate and the end-to-end test passes.

## Source and ownership

Canonical product repository: `/home/ubuntu/JopeDime` on xbabe2. The current Grok checkout is `.agent/worktrees/dlhwm-sweep`. Its source ownership has not transferred for the flywheel. New source is prepared under the ignored `.agent/proposals` namespace and in owned execution directories; the predecessor checkout, index and results are preserved. Existing PRs must be reconciled under the repository contract before starting another implementation lane.

Read first:

1. `/home/ubuntu/JopeDime/AGENTS.md`, `MASTER_PLAN.md`, `AGENT_CHAT.md`, and `docs/adr/ADR-0005-canonical-learning-loop.md`.
2. `/home/ubuntu/jopedime_probe_exp/ofm-20261008/S3_ENCODINGS_GPHWM_REFERENCE.md`.
3. `/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/DL_TRAINING_DASHBOARD_HANDOFF.md` and its latest claims.
4. The exact current parent registry, campaign config, native worker build and execution qualification receipts listed below.
5. `deploy/dl-flywheel/jope_flywheel/{protocol,augmentation,target_noise,queue,evidence,evergreen,program_gate,s3_library}.py` and their focused tests.

No new worktree is authorized by this document. Never mutate generated historical receipts, the original pilot, Grok's source, sealed test sources, other agents' services, or xbabe1's benchmark compute. New controls operate only on explicitly owned services. Log new claims and handoffs under the shared append locks.

## Data identities and lineage

The parent registry is `/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/flywheel_parents_20261008_v1/DL_FLYWHEEL_PARENT_REGISTRY_V2.jsonl`, SHA256 `8fe409848aadd6256644466faca6a0ceab73ab67131d30243701e0d93d9d0593`. It includes exact context and compressed/raw payload hashes, dataset and retained-lineage identities, original cut/fold, current role authority, feature/target hashes, source preparation caveats, and allowed transforms.

603 original context payloads were copied into new immutable train-only files. Only the original training block was decoded; outer-test tails were hashed opaquely for custody and discarded. Another 465 existing stripped parents were fully reverified. Each newly stripped payload has a separate `preparation_receipt`; existing stripped parents use a null field plus the original context's strip evidence. Do not treat null as a path.

Every descendant inherits the stable parent-lineage assignment. Aliases and folds from a parent must remain together. The split namespace is `af9d3b32c55aa019a3437eb957d709a60f676fe6e67ecec8774f5809cd716df2`. Current sealed/rights exclusions take precedence. This is a new development/research partition; historical development exposure and pretrained encoder overlap are not erased. It is not a pristine holdout.

## Exact augmentation protocol

The version is `prepared-drop-nuisance-targetnoise/v1`. The per-lineage ordinal and hash-derived uint64 seed determine the recipe. Parent lineages rotate before another descendant is produced for the same lineage; the initial protocol permits at most 256 descendants per parent. 1,068 × 256 provides room for the 100,000 target while fresh parent lineages can later enter a new pinned registry.

- **Feature removal:** deterministic random selection of 0%, 5% or 10% of columns, rounded down, always retaining at least one. Metadata records every removed and retained source column.
- **Synthetic columns:** independent standard-Gaussian nuisance columns with deterministic per-row randomness, up to 10% of original width rounded down, at most eight, and total width at most 512.
- **Feature noise:** disabled on the current cohort because original continuous/categorical semantics are not qualified. Prepared numeric category codes are never silently treated as continuous measurements. A future version may enable train-scale Gaussian perturbations only for verified continuous columns.
- **Target noise:** 0%, 1%, 3% or 5% Gaussian standard deviation relative to the designated inner-fit target population standard deviation. This is not a bound on each individual perturbation; Gaussian tails are unbounded and no clipping is applied. Scale uses inner-label-0 rows only. Constant calibration targets cannot yield nonzero target noise.
- **Rows and folds:** preserve all admitted parent rows in original order and both original inner-fold labels. No row subsampling, duplication or invented outer-test data. New fold generation is a separate future protocol with grouping/duplicate controls, not a current capability.

A changed target obtains new target/content/context identities. All changed datasets require fresh encodings and GP evaluations. Parent scores, parent encodings and parent HWM values are never copied as labels for a variant. A transformation that produces byte-identical data is rejected; distinct recipes that produce the same payload cannot inflate the unique dataset counter.

Metadata records the complete parent descriptor, source checksum, recipe, seed, ordinal, dropped-column indices, output-column mapping, fitted noise state, target hash before/after, row and fold support, lineage/partition, NumPy/RNG version, prepared units, rights, exposure caveats and admission state. Target-noise task generation is declared separately from GP model fitting. It must not be described as a fold-proof learned preprocessing step.

## Evergreen process and resource behavior

Runtime: `/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/flywheel_evergreen_20261009` on xbabe2. Service: `jopedime-flywheel-evergreen-20261009`.

Config SHA256: `cd9ee8f882735c42b49d694b99dbbc4187035804099bc49f0ca81f8ec77a9b0a`. The first service window ends at Unix `1791592146.380285`; it must not silently reset its deadline. A scheduled continuation inspects progress and makes the next bounded work packet concrete.

The service uses at most four CPU cores, 2 GiB memory, no swap and no GPU. It waits when one-minute host load exceeds 96, available memory falls below 32 GiB, or free disk falls below 100 GiB. It allows at most 128 generated/published/reserved variants that are still waiting for downstream admission. Publishing a dataset does not free that capacity. GPUs are reserved for qualified encoder/training work on xbabe2 and xbabe3.

`data/expansion.sqlite` is a WAL/FULL-synchronous journal. A sequence and parent/ordinal are reserved transactionally before writing. Unknown interrupted reservations remain capacity-occupying. Immutable variant directories contain `encoder.jtt1.gz`, `metadata.json` and `generation.json`. The latter binds raw/compressed content and metadata hashes. The service checks the pinned role/source authorities before each generation cycle and authenticates each parent again before use.

Failure handling: source drift, protected rows, invalid shape, output escape, nonfinite data, unsupported policy, no-op variants or unknown outcomes stop the bounded worker and preserve evidence. Do not silently substitute another dataset or retry an external operation. Reconcile the recorded exact outcome before releasing capacity or resuming. Resource waits are distinct from failures.

## Mandatory new-program path

A generator's training and validation emission points must call the same gate with:

```python
result = gate.ensure(
    context, generated_dsl,
    phase="training",  # or "validation"
    checkpoint_sha256=checkpoint_sha,
    step=optimizer_step,
)
measurement = result.require_measurement()  # raises while evaluation/publication is pending
```

The key includes canonical program, exact dataset bytes, target, cut/split, evaluator, vocabulary, protocol, metric, units and direction. Program-only lookup is insufficient. A different target/noise variant, fold or evaluator must be reevaluated. S3 authentication/read failures are not cache misses.

1. Record the program origin durably, including phase, checkpoint and step.
2. Do an authoritative exact-context S3 lookup. Reuse only an entry whose complete versioned artifact closure verifies.
3. On a confirmed absence, enqueue one idempotent evaluation and its dependent publication. Several observations of the same context/program share the work.
4. Execute the canonical DSL with the owned local Lambda worker. It uses two inner folds, fit-row-only min/max scaling, actual prediction/target arrays and native serialized-model replay checks. No fabricated outer-test rows.
5. Emit the Rust source through the pinned native emitter, compile it and execute it on both exact fit/score folds. Compare predictions under a predeclared tolerance and retain bit-exact counts. The current qualification matched exactly.
6. Write full native S1 evaluation, input, canonical DSL, Rust source, executable, code-parity/provenance and canonical `jope.score/v1` rows. Canonical row hashing uses the native Rust serializer with blank `row_sha256`, not Python's float formatting.
7. Publish artifacts create-only, verify exact S3 VersionId and SHA256, then publish the lookup record last. Cache hits undergo the same full verification. Count only committed measured results as library additions.
8. Apply dataset/encoder/role/label admission independently. A verified score is not automatically optimizer-eligible.

The current owned evaluator permits zero outer-test rows while preserving the native two-fold protocol. Its SHA256 is `626d93a5cd11830e1b7cd43bf46f3fbde42c06273e750faa6d64b69a0c037526`; source commit `6bf65b12e9f0e5ec2697720a6119f13ea27d4835` plus the recorded parser patch. Original evaluators remain unchanged. The owned executable runs on the Lambda machines; this is not a request to invoke the AWS Lambda service.

## Qualified generic observation and execution path

On 2026-10-09 at 01:14 UTC, the generic bridge completed a real cold-miss proof on a new 102-row, six-feature variant with 1% target noise. Training and validation observations durably recorded the same candidate, `(pipe X (ridge))`, and atomically queued one evaluation with its dependent publication. The local Lambda host evaluated both 51-row inner folds, emitted Rust, compiled it, executed both folds, and matched all 102 native predictions bit-for-bit. The fixed tolerance remains `1e-9 * max(1, abs(reference))`; it was not relaxed.

Fresh RMSE was `[0.006271888006852582, 0.010831365897680825]` in the prepared synthetic target units. These values belong to this exact context and cannot be compared directly to the earlier 192-row pilot's errors. Seven artifacts and the lookup record were published with exact-version checks. Both subsequent phase observations returned the measured S3 result with no new jobs. The proof took 17.275 seconds and peaked at 298,561,536 bytes. This is one small functional proof, not a throughput benchmark or a trained model improvement.

- Source/config: `DL_FLYWHEEL_PROGRAM_BRIDGE_CONFIG.json`, SHA256 `3528daeb99d34419b94104f88da3f0829f176c3fc9fe507ce88a6a80f8b98c56`.
- Verification: `DL_FLYWHEEL_PROGRAM_BRIDGE_VERIFICATION.json`, SHA256 `8ebf8112183712e7473703475c760ee291e634bf7b500db14bbd91138f1c97bd`.
- Lookup record: `program-results/flywheel/v1/7109d76e1391c2cf6dc215c4ec8ab9bd37dc35c6dc49950e4e9dd7338c297ba5/record.json`, VersionId `2DKoDgqvjWQ8c_53P6qkR4Z4S3zVU03X` under the prefix below.
- Runtime: `/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/flywheel_program_bridge_20261009_v1` on xbabe2.

The CLI provides `describe-context`, `observe-program`, and `run-once`. Requests bind the full variant identity, exact typed context, descriptor hash, DSL, training/validation origin, checkpoint hash, step, and named producer. `program_observer.py` journals bounded emissions before semantic validation and S3 lookup. `program_worker.py` refuses evaluation without a matching dependent publication job. `queue.enqueue_many` commits both jobs together; capacity or storage failure leaves neither job behind. The regression reproduces the old orphan-evaluation failure and verifies rollback.

This version accepts only the 128 registered new prepared variant contexts. Original/native HWM contexts are rejected, rather than being treated as absent from the whole S3 library. Invalid DSL, changed roles, authentication failures, timeouts and uncertain outcomes remain explicit errors. They do not become numeric rewards and do not retry automatically. A separate lossless attempt-custody exporter now archives the complete observation journal and terminal receipts to S3. The first actual snapshot contains eight successful bridge events; failure handling was tested with explicit synthetic fixtures, not invented production failures. The public framework suite passes 144 tests; the captured-artifact worker orchestration suite passes 11 tests.

The proof uses a clearly labeled synthetic checkpoint provenance value. It does **not** demonstrate a hook in a live trained model. Existing foreign GP-capable gateway services were observed and remain outside this owned bridge. Their source owners must integrate the mandatory observation boundary or hand over the relevant source before universal coverage can be claimed.

## S3 layout

Bucket: `veox-jopedime-learning-use2`; root: `experiments/gp-policy-dl/v2-trainonly/`.

- `datasets/flywheel/v1/<full-variant-identity>/`: immutable input, metadata and fresh encoding blocks.
- `receipts/flywheel-datasets/v1/…`: immutable joins with exact object versions and producer hashes.
- `program-results/flywheel/v1/…`: a new versioned wrapper around ordinary native program evaluations and canonical scores. Ordinary losing programs remain useful library entries. Do not overwrite winner-only GP-HWM records with ordinary results.
- `paper-evidence/20261008/…`: committed paper evidence snapshots. Three archives totaling 131,532,800 bytes and 233 evidence events have verified S3 custody.

No existing dataset, score, winner or historical receipt is overwritten. Completion inventories add references to later encodings without changing the original metadata.

## Training admission and model tests

Admission must require exact source/role checks, all eight encoder vectors with producer provenance, freshly measured program labels, versioned S3 closure, finite dimensions/values and parent-group partition. Freeze a complete training manifest at run start; later arrivals enter the next snapshot. Keep validation descendants out of optimizer batches. Use lineage-balanced sampling and cap descendants' combined weight so one prolific source cannot dominate.

Wave4's outcome-only critic cannot learn which program is better from dataset features alone. The next program-conditioned scorer must include a verified representation of the candidate program. Compare normalized regression against within-context pairwise ranking using only measured candidates; split and confidence intervals by parent lineage. Store raw RMSE and baseline RMSE as well as a fit-derived stabilized log-relative error; never relabel that normalized objective as RMSE. Win rates require a frozen exact-context opponent pool and actual scores. Elo is descriptive and cannot replace those paired comparisons.

Before committing a larger training run: finish program-emitter integration, build the frozen admitted manifest, verify exact batch/epoch membership including short final batches, run a bounded GPU update/checkpoint test, and compare against the matched CPU baseline at the same parent support. Preserve the distinction between lowering validation loss and beating the external baseline.

## Validation and continuation requirements

Focused tests cover target-noise fit isolation/determinism, parent/source drift, protected rows, variant identity, small-feature behavior, native width bounds, stable lineage assignment, output path restrictions, backlog/resume behavior and exact metadata/content custody. The first evergreen packet passed 20 tests and an independent exact-source review. Actual host preflight reverified all 1,068 parents before launch.

Next work packets, in order:

1. Completed: publish the two initial programs and a third new-context program through the generic observation → local execution → publication bridge, including real-S3 reuse from both phase origins. The lossless S3 attempt exporter has also passed on the actual eight-event journal. Next, integrate every actual program-emitting producer and archive each new bounded journal snapshot, including any real failed attempts.
2. Completed: publish all 128 evergreen datasets with provenance. Completed: encode a diverse 16-lineage subset on the two GPUs, including seven changed-target tasks, three validation lineages and a one-column dataset. All 128 component vectors and their phase/coordinator ledgers are verified. S3 completion publication and the remaining 112 variants follow as separate packets. Advance counts only from actual receipts.
3. The generic queue/worker is connected and its actual cold-miss proof passed. Integrate actual emitting trainer hooks, with no bypass and explicit backpressure; retain failure/timeout and interrupted-publication behavior.
4. Score candidates for each newly encoded context, publish canonical rows, build admission receipts and freeze the next training snapshot.
5. Release only admitted backlog entries, allowing balanced generation to continue. Scale throughput from measured stage latency, not by removing resource limits.
6. Integrate the reviewed source through the repo's ownership/PR lifecycle, preserving all failed attempts and predecessor files. An unacknowledged transfer is not approval.

Completion means observed new datasets, new measured program/context results and verified training exposure, with a live dashboard and reproducible receipts. A planned 100,000 count, queued job, model prediction or emitted source file alone is not completion.

## Eight-encoder qualification on 16 evergreen lineages

The first diverse subset completed all eight encoders on xbabe2 and xbabe3. Each host handled eight tasks with CPUQuota=800%, MemoryMax=12 GiB, no swap, one GPU and a shared original 1,200-second deadline. xbabe3 completed in 82.92 seconds. xbabe2's first attempt failed after the 16 base vectors because a disk-usage inventory raced a legitimate Triton cache-file rename. The failed output and traceback remain preserved. A narrowly reviewed recovery tolerates a vanished temporary cache entry only; scientific artifact loss, symlinks and output limits still fail. xbabe2's separate recovery completed in 99.78 seconds before the original deadline.

All 16 tasks retain their original lineage roles and fresh target identities; none inherits a parent GP score. Full-vector repeat/invariance checks and phase-specific qualification passed, including the one-feature task. Exact sealed closure and fresh current-role verification are recorded in `DL_FLYWHEEL_EVERGREEN_ENCODE16_FINAL.json`, SHA256 `e128d43855ed38ebf239261b91eaf3ac6b3654513bca65f3b95011e18bccccb4`. This receipt establishes 128 local vectors, not automatic S3 publication, program labels or training admission.

## New training snapshot adapter

The native PolicyLM source audit identifies a safe route using an explicitly versioned `flywheel-policy-tranche/v1` input. The new eight-block layout is 20,143 floats: A0 88; Hyperion v6 all 512; Kumo L 4,099; Kumo S 2,051; Mitra v2 4,099; TabDPT 3,075; TabICL 2,051; DOPE 4,168. This is not the historical S1 encoder roster. Full 64-character variant identities remain intact; the old 16-character native-dataset validator is not bypassed or impersonated.

A snapshot must verify the exact S3 input, metadata, encoders, qualification evidence and program-result versions, reconstruct the fit-row-mean baseline from the opposite inner fold, and bind current parent-lineage roles. A lineage assigned to validation maps only to the native dev partition. Source, grammar, tokenizer, labels and sampler bytes are pinned. Token cross-entropy and pairwise groups require actual qualifying teachers: the native group of eight requires at least four distinct gate-passing teacher skeletons. Sparse or unsuccessful contexts remain explicitly feasibility-only. Repeating a program four times or inventing outcomes cannot create teacher support. The adapter is being implemented and has not yet admitted a new variant or launched a new training run.

## Program attempt custody in S3

`attempt_custody.py` exports one bounded, externally anchored SQLite read transaction without source SQL mutation. It preserves all emission envelopes, outcomes, queue transitions, registered exact contexts and terminal receipts. Counters distinguish observations, exact-context cache reuse, unique executions, publication, errors, failed jobs and uncertain outcomes. Failed or invalid attempts receive no numeric score.

The first actual export completed in 1.403 seconds with a 50,237,440-byte peak under CPU1/512 MiB/no-GPU limits. It published two create-only, exact-version-verified objects under `program-attempts/flywheel/v1/`: a 376,856-byte snapshot and a separate content-addressed receipt. The complete actual source contains four observations, eight events, two cache hits, one unique execution, and zero production failures. Fourteen tests include invalid DSL, unknown contexts, timeout outcomes, worker failures, unresolved observations, corruption, concurrent writers, exact-version reads and partial publication failure. Those test fixtures remain labeled as synthetic.

Verification: `DL_FLYWHEEL_PROGRAM_ATTEMPT_ARCHIVE_VERIFICATION.json`, SHA256 `30c7d77a51010c924e2bcb1b266f2f74fb9f37af08195d1563dcd1ccff226f37`. This snapshot is an auditable record of attempted work, not a new measured GP result or training-admission receipt.

## S3 closure for the 16-context encoder packet

All 128 vectors and 16 self-contained completion receipts were created and exact-version verified in S3 (144 objects, 2,354,705 bytes). Final receipt: `DL_FLYWHEEL_EVERGREEN_ENCODER_PUBLICATION.json`, SHA256 `c9614c853e201fb14eb2c9bec7599d473a06ca31833f28b4bc72d4083ddb952e`. Each completion receipt embeds the pinned qualification JSON bodies, original encoder records, component object versions, source/target/lineage identities and sealed phase/coordinator anchors. The publisher refreshed a stale current-role proof into a new append-only config; original configs/preflights were preserved and the original 600-second deadline was unchanged. Publication completed at 01:37:35 UTC before the 01:40:17 deadline.

Across the pilot and evergreen packets this establishes 116 distinct fully encoded variants and 928 fresh vectors in S3. The 100 pilot variants share one parent; the 16 evergreen variants have 16 distinct parents. Do not sum those as 116 independent source datasets.

## Scaling beyond a single bounded shard

The 100,000 target is a cumulative accepted-variant goal. The current generator's backlog cap is 128, and the first loader bounds a snapshot shard to 128 contexts and 256 MiB of unique exact-version artifact reads. These limits are explicit operational units, not evidence that the full 100,000-task learner already exists. A larger learner must join immutable admitted shards, stream or memory-map encoder blocks, preserve full context/program membership and lineage weights, and log actual shuffled batch exposure. Completed code/score objects are content-addressed; shared binaries may deduplicate, but unique-byte accounting must use actual object versions rather than an assumption.

A coordinator should keep encoding and labeling workers supplied, create an admitted shard only after its measured closure verifies, then transactionally release only those variants from the generator backlog. It must retain separate counts for generated, S3-published, all-eight-encoded, measured, optimizer-eligible, validation-only and actually consumed. New cohorts enter the next frozen run manifest; no mid-epoch mutation. This coordinator/admission integration remains outstanding.
