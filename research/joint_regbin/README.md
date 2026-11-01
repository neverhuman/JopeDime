# JRB-01 CPU study driver

This component prepares and audits an equal-budget study. It contains no qualified
learner or learning results. The existing live flywheel is
regression-only. Its code and running process are outside this component's scope.

## Design and preregistration

Freeze a metadata manifest and the [preregistration](preregistration.json) before
learning. Three arms (joint, regression only, binary only) share three seeds and
one five-fold family partition. Five held-family jobs plus one full-training job
per arm/seed make 54 jobs. Every job has 50 rounds, two updates per round, 32
contexts per update and 10 conditioning slots per context. These slots include
masked positions; they are not generated-code tokens or throughput. The joint
arm uses a fixed 1:1 regression/binary batch. Shuffled family blocks are cyclic
and deterministic. Validation families never train; a panel family appearing in
training uses its held-family fold model. The prior natural-ratio sensitivity arm
is deferred by the T0218 three-arm owner scope.

Training and evaluation run as separate processes. The training request contains
only that fold/arm's training references and batch schedule. Evaluation receives
the private training receipt and evaluation references, without feeding results
back into learning. No automatic live promotion, task split or mitigation launch
exists. Three seeds remain paired through family bootstrap resampling.

The report uses separate regression and binary Elo tables, with matching
single-task arm anchored at 1500 and ridge as a common contender. These study-local
ratings do not reproduce the published eight-method panel leaderboard. CV gets
paired 95% family bootstrap intervals; DL-Val gets task-specific loss curves;
panel_v1 requires 240 distinct datasets per task. `joint` is the declared
decision arm. The exact 25-Elo and two-consecutive-evaluation rule is in the preregistration. Incomplete evidence
has no scientific verdict. An inconclusive result requires a separately frozen,
single mitigation rerun, rather than selecting the best post-hoc result.

## Input contract

The metadata JSON has schema `jope.joint-regbin/v1`, `eligibility_verified:true`,
SHA256 pins `catalog_sha256`, `learner_sha256`, `initial_checkpoint_sha256`, and a
`contexts` list. Each context has exactly:

```json
{"id":"context-sha","dataset":"catalog-id","family":"authoritative-family",
 "task":"regression","role":"train","source_split":"train","origin":"real",
 "has_meta_test":false,"payload":"summaries/context.npz","payload_sha256":"<64hex>"}
```

Roles are `train`, `val`, `panel`; source rows must be `train` or `inner`.
The current executable catalog verifier accepts only recorded family keys;
provisional keys require the audited admission mapping described below before
they may enter a runnable manifest. Context roles and train-row-only summary
construction need provenance review against the frozen DL partition.
`eligibility_verified` records that review; it does not perform it. Multiclass
metadata is counted and excluded.
Unknown tasks, protected splits/paths, family leakage and payload escapes fail
before a payload is opened. Payloads, model weights and output remain outside Git.

### Current catalog admission ruling (ENG:watcher t2035, 2026-11-01)

The approved source is catalog v246 in `S3_DATASET_CATALOG.md`. The only holdout is
the P3 `sealed_excluded` family group defined by the Oct 7 Jepson ruling
(293 production members and 470 catalog members); exclude it by family and do
not read its payloads. All remaining catalog regression and binary records are
eligible for TRAIN or validation, including P3 `heldout` and `dev` records. A
manifest must retain the family-grouped role assignment and the ruling
reference. The sealed roster is recorded by reference only; no sealed payload
or roster contents are part of this repository.

For unkeyed binary records, promote the catalog's provisional
`srcfile:<stem>` key, add the source URL parent when present, and merge shared
recorded dedupe fingerprints before assigning folds. Record the source fields
and the resulting family key for every promoted dataset. `panel_v1` is the
admitted panel target at 240 datasets per task; if the eligible binary roster
falls short after family exclusions, include all eligible records and report
the measured shortfall.

This ruling changes data eligibility; it does not qualify an executable study
packet. The current verifier still needs an audited provisional-key admission
mapping before it accepts these new roles. The runtime manifest also needs
task-correct train-row-only cached pools with payload hashes, the existing
task-split validation contexts, and a pinned two-task learner/checkpoint. Until
those inputs pass preflight and the learner is qualified, no study result or
split-or-stay verdict is available.

### Catalog pool candidate inventory

`admission.build_catalog_pool_manifest_from_files(catalog_path, owner_alias_path)`
constructs a metadata-only regression/binary candidate inventory. It applies
the pinned `dltrain.family_keys` policy, unions source-file and URL-parent
aliases across both tasks, and binds the current owner raw-catalog alias
receipt. The output is `jope.joint-regbin.catalog-pools/v1` with
`admission_status: CATALOG_CANDIDATES_NOT_ADMITTED` and
`eligibility_verified: false`. It contains catalog train-row hashes and family
alias evidence, but assigns no train/validation/panel roles and does not read
payloads. The owner receipt still leaves 14 source-file aliases unresolved and
its recorded-key intersection check is binary-only, so this inventory does not
establish full owner-alias or sealed-family closure. It cannot be used as a
runnable study manifest until role, alias, payload, and worker evidence is
reviewed and admitted separately.

### Task-split role candidates

`admission.build_task_role_candidate_manifest_from_files(...)` binds the
catalog pool to the pinned flywheel train/dev selection and `panel_v1` metadata.
It preserves the 1,219 regression train and 91 regression validation contexts,
uses all 843 catalog binary rows as a family-grouped candidate pool, and
records task-specific panel rows. The five-fold assignment groups contexts by
the cross-task P2/source-file/URL family review. Panel evaluation excludes
training families that intersect its panel family set. Per T0830 owner ruling,
the 14 owner aliases without source-file metadata are collapsed into one
conservative family and excluded from role assignment.

The output schema is `jope.joint-regbin.task-role-candidates/v1` and remains
`ROLE_CANDIDATES_NOT_ADMITTED`: it records metadata and source hashes only; no
payload bytes are opened or verified, 11 regression panel identities do not
appear in the usable catalog, and the task-specific worker qualification is
still missing. It is a role-planning receipt for the worker lane, not a runnable
study manifest or scientific result.

### T1210 family-key candidate inventory and role guard

`admission.build_family_key_pool_manifest(...)` records catalog IDs and exactly
those keys returned by the pinned `family_keys.attrs_from_catalog_row()` and
`family_keys.derive()` calls. Family components preserve shared and transitive
returned keys across both tasks. No source-file, URL-parent, owner-alias, or
protected-roster inputs are added.

The frozen T1145/T1210 admission inputs require a family's nonsealed status to
be established before assigning TRAIN or validation roles. The approved v246
catalog supplies identities and TRAIN-file hashes, but no reviewed nonsealed
eligibility evidence. Presence in the catalog, a task label, and self-reported
flags cannot establish that permission. Every unverified family is therefore
withheld: `fold_by_family` and `fold_splits` stay empty, while
`nonsealed_eligibility` records the withheld family/dataset counts and ruling.
The preflight re-derives this candidate inventory and reports
`METADATA_ONLY_NONSEALED_STATUS_UNVERIFIED`, zero folds, and
`eligibility_verified:false`. Forged role assignments are rejected.

This withholds unknown eligibility; it does not classify P3 `heldout` or `dev`
as forbidden. The t2035 holdout remains exactly `sealed_excluded` by family.
Restoring five-fold assignments with fixed seed 136 requires approved,
reviewed nonsealed-family evidence that respects the no-protected-roster input
boundary. No role is inferred and no payload is opened in this metadata path.
Catalog train hashes also do not provide the candidate tensors or qualified
worker/checkpoint required for an executable learner packet. No campaign starts.

### Train-only candidate tensor builder (ENG:watcher R22)

`candidate_builder.py` supplies the train-only cache construction seam. It
rederives the zero-role family-key pool and requires an ADMITTED role manifest
that passes `admission.rederive_role_manifest`. Metadata-only role candidates
and provisional nonempty family folds are refused;
the current approved catalog has no reviewed nonsealed-family evidence, so the
family pool has no folds and cannot authorize payload reads. When a reviewed
admission exists, the builder reads only
`datasets/<id>/manifest.json` and the exact
`blobs/sha256/<catalog-train-sha256>` object from the approved dataset bucket.
It never lists a bucket or reads another split. CSV rows are numeric and finite;
binary labels are explicitly mapped to zero/one from their two observed class
values. A deterministic seed builds two inner folds, and the source-pinned
Flywheel Landmark-V2 proposer, DSL featurizer and native S1 evaluator build 16
candidate rows.

Regression worker targets are the existing S1 constrained fitness. Binary
worker targets are the existing two-fold `cv_gate_passed` labels, because the
task head is BCE-with-logits; this does not infer task from target cardinality
or introduce a new threshold. The R882/DOPE token tensors are supplied by the
caller after its read-only pinned initialization. Worker inputs have no revealed
candidate labels, so target values cannot leak through conditioning slots.
`make_worker_packer(...)` implements the callable expected by `WorkerCore`.
The synthetic suite checks that unverified family roles and metadata-only
candidate receipts cannot authorize an S3 request. Building a tensor cache does
not qualify a worker or admit a study packet.

Every generated pool remains
`CANDIDATE_TENSORS_NOT_ADMITTED` / `eligibility_verified:false`. This builder
does not qualify a worker, admit role candidates, train, reserve the study lock,
write checkpoints or mutate the score store. Use only role rows whose pinned
metadata proves nonsealed TRAIN/VAL eligibility. Current owner evidence still
does not provide an admitted complete role manifest or qualified two-task
worker, so no real pool was built by this change.

## Learner integration contract (qualification still required)

Supply a reviewed Python learner, pinned by its full file hash, accepting
`--request FILE --response FILE --packet DIR --initial-checkpoint FILE`.
This PR does **not** provide or qualify that two-task learner. It must use the
frozen DOPE weights, train only the JopeDime adapter, enforce GPU memory≤90% and
CPU≤85%, verify the inherited `JOPE_QUALIFICATION_LOCK_FD` identifies the real
lock, and avoid shared/live bank or checkpoint writes. A regression-only worker
cannot be renamed to satisfy this contract. No Phase2 kernel is authorized.

For `phase:train`, read only the supplied context references and execute exactly
the supplied batches. Save private checkpoint artifacts at each five-round
evaluation point. Return round receipts with actual optimizer steps, batch digest,
parameter hashes before/after, frozen DOPE hash, and checkpoint hashes/locations.
For `phase:evaluate`, use those checkpoints under inference-only evaluation.
Return source/job identities, actual budget, initial/final parameter hashes,
`dope_before/after`, all round receipts and `scores`. Each score supplies
`round,id,status,metric,value,baseline,curve,model_sha256,evaluation_sha256,evaluation_protocol`.
The input hash must match its frozen context, the protocol must be
`cached-pool-jrb01/v1`, and the checkpoint hash must match its training round. Regression uses positive
RMSE and ridge RMSE with exact log ratio; binary uses AUC for pairwise Elo and
log-loss for curves. Labels/heads must be task-correct, not inferred from target
cardinality in this driver. Scores must refer to the same frozen evaluation input
and protocol. Failed or missing cases remain unavailable; never fill with zero.

This seam deliberately prevents importing the running flywheel as an experiment
backend. Until a binary-capable worker and eligible binary pools are qualified,
the GPU study remains blocked. CPU tests exercise the contract using labelled
synthetic fixtures and never produce learning evidence.

## Commands

From the repository root, freeze the plan without loading payloads or a GPU:

```sh
python3 -m research.joint_regbin.driver plan MANIFEST.json --catalog "$HOME/jope_plan/master/source/S3_DATASET_CATALOG.json" --out PLAN.json
```

After input/learner review and when the real lock becomes available, dispatch:

```sh
nice -n 10 python3 -m research.joint_regbin.driver run PLAN.json --packet PACKET --worker LEARNER.py --checkpoint INITIAL.pt --out "$HOME/jope-data/jrb-01/attempt1" --score-store-module "$HOME/jope_plan/scores/score_store.py" --score-store "$HOME/jope_plan/scores/history.jsonl"
```

Before reserving compute, check the assembled inputs on CPU:

```sh
nice -n 10 python3 -m research.joint_regbin.driver preflight PLAN.json --packet PACKET --worker LEARNER.py --checkpoint INITIAL.pt
```

`preflight` verifies the frozen plan (including its fixed DOPE pin) and the worker, checkpoint and approved
packet file hashes. It rejects protected paths before hashing any input and
prints `INPUTS_VERIFIED` only when all identities match. It takes no GPU lock,
starts no worker, reserves no campaign and writes no score history. Missing or
changed inputs exit nonzero. The same checks run again at dispatch, so a passing
preflight cannot hide later input drift.

This verifies input identities, not a two-task learner or a scientific result:
the response retains `qualification_status: NOT_QUALIFIED` and a null verdict.
The task's sustained-regression-gains prerequisite and separate learner/input
qualification still apply before learning. Plans pin implementation bytes;
prepare a new plan after upgrading the driver, without resetting an existing
campaign reservation.

The dispatcher takes the real exclusive lock nonblocking. If busy, keep the same
preregistration and queue normally; do not substitute a lock or reset budgets.
There is one worker process at a time and one shared six-hour/10GiB campaign cap.
A durable plan-keyed reservation under `~/jope-data/jrb-01/campaigns/` prevents
changing the output path to reset that budget. Interrupted reservations remain;
automatic resume is unsupported and never starts a second campaign implicitly.

Admit results, append via the JRB-02 owning API and generate JSON/Markdown/raw
curve JSONL plus a standalone SVG plot:

```sh
python3 -m research.joint_regbin.report PLAN.json "$HOME/jope-data/jrb-01/attempt1" --out "$HOME/jope-data/jrb-01/report1" --score-store-module "$HOME/jope_plan/scores/score_store.py" --score-store "$HOME/jope_plan/scores/history.jsonl"
```

The dispatcher appends after each admitted evaluation job; report import also
replays every score idempotently. Worker failure preserves its private files and
logs; incomplete evaluations remain unavailable rather than zero.
Score source identities contain plan/job/context/round, so reimport is idempotent
and different seeds/folds/context variants remain distinct. Original histories
are never rewritten. No historical store is mutated by CPU tests.

```sh
mkdir -p target
pytest research/joint_regbin -q --basetemp=target/jrb01-test-data
```
