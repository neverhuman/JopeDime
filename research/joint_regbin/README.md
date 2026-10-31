# JRB-01 CPU study driver

This component prepares and audits an equal-budget study. It contains no trained
model, binary candidate pools or learning results. The existing live flywheel is
regression-only. Its code and running process are outside this component's scope.

## Design and preregistration

Freeze a metadata manifest and the [preregistration](preregistration.json) before
learning. Four arms (joint balanced, joint natural, regression only, binary only)
share three seeds and one five-fold family partition. Five held-family jobs plus
one full-training job per arm/seed make72 jobs. Every job has50 rounds, two updates
per round,32 contexts per update and10 conditioning slots per context. These slots
include masked positions; they are not generated-code tokens or throughput.
Joint natural uses the frozen training cohort's ratio; joint balanced uses1:1.
Shuffled family blocks are cyclic and deterministic. Validation families never
train; a panel family appearing in training uses its held-family fold model.

Training and evaluation run as separate processes. The training request contains
only that fold/arm's training references and batch schedule. Evaluation receives
the private training receipt and evaluation references, without feeding results
back into learning. No automatic live promotion, task split or mitigation launch
exists. Three seeds remain paired through family bootstrap resampling.

The report uses separate regression and binary Elo tables, with matching
single-task arm anchored at1500 and ridge as a common contender. These study-local
ratings do not reproduce the published eight-method panel leaderboard. CV gets
paired95% family bootstrap intervals; DL-Val gets task-specific loss curves;
panel_v1 requires240 distinct datasets per task. `joint-balanced` is the declared
decision arm; natural sampling is sensitivity evidence. The exact25-Elo and
two-consecutive-evaluation rule is in the preregistration. Incomplete evidence
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
Only authoritative catalog family keys are accepted; unreviewed derived keys are
refused. Context roles and train-row-only summary construction need provenance
review against the frozen DL partition. `eligibility_verified` records that
review; it does not perform it. Multiclass metadata is counted and excluded.
Unknown tasks, protected splits/paths, family leakage and payload escapes fail
before a payload is opened. Payloads, model weights and output remain outside Git.

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
