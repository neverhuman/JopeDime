# Search, Rust scoring, and the S3 score lifecycle

Inherited product code is commit `edf36182b3fb3254afd244c9be1f1f6f274b9e0c`.
The cache added by this lane is commit
`dfa461fd9c4035f9078ff9f49f29d67b847b62c6`.
The lifecycle repair (ledger reload under a lock, and no S3 prefix rule
from a listing) is `BudgetLedger.admit` and `proposal_from_rows` in
`s3_lifecycle.py`, at the line numbers cited below.
That repair is commit `5c456d5294380be2b9edf21652c717132038b50c`.
Measurements below were taken on this host on 2026-10-10.
Receipts live under `.agent/proposals/claude-rlfw-20261009/rlfw-grok/` and are
outside git (`.gitignore` ignores `.agent/`). A figure that was not measured
is marked **unmeasured**.

## Learner objective and losses

At `edf36182` the trainer objective used by name `successor-v1` is
`successor_v1` in
`deploy/dl-trainonly/lane/src/dl_training_v1/dltrain/objective.py:341`.
Its docstring states the weighted sum, and the coefficients are
`COEF_SUCCESSOR` at line 30:

| term | coefficient | what the function computes |
| --- | --- | --- |
| `ce` | 1.0 | masked sequence cross-entropy over teacher rows of group contexts (lines 354–364) |
| `bt` | 0.25 | softplus pairwise loss on `q_hat` for `q_known` pairs with distinct `q_grid` (lines 366–378) |
| `list` | 0.10 | list term on group contexts with at least 3 `q_known` members (`LIST_MIN_MEMBERS` is 3, line 44; lines 380–389) |
| `q` | 0.25 | Huber of `q_hat - q_opt` with delta `HUBER_DELTA = 1.0` (line 34; lines 391–397) |
| `feas` | 0.10 | 3-class cross-entropy on `feas_valid` members (lines 399–409) |
| `rt` | 0.05 | Huber on completed runs, squared hinge when `rt_censored`, local backend via `rt_valid` (lines 411–421) |

`_finish` applies those coefficients (line 423). A term with no support is the
connected zero `q_hat.sum() * 0.0` (line 350). A non-finite head, term, or
total raises `ObjectiveNonFinite` (lines 8–9 and the class at line 50).

A second objective, `baseline-v1`, has coefficients
`{"seq_nll": 1.0, "bt": 0.25, "list": 0.10, "rt": 0.10, "cf_bt": 0.05}`
(line 31). This lane did not train either objective. No loss curve from
`successor_v1` was produced here. That training loss is **unmeasured** in
this session.

`L_PI` and `tilt_weights` do not occur under
`deploy/dl-trainonly/lane/src/dl_training_v1/dltrain/` at `edf36182`.
That term is another lane's work and is **absent** from this commit.

## Training loop

`Trainer._step` (`train.py:747`) draws a batch, runs the network, calls the
objective selected by `_objective_fn` (lines 363–368), then
`zero_grad(set_to_none=True)`, `backward`,
`clip_grad_norm_` with `clip_norm` default `1.0` (line 85 and line 773),
`opt.step`, and `sched.step` (lines 771–779). A non-finite loss or gradient
norm finishes the run as `FAILED` / `nonfinite_objective` (lines 768–776).
The step appends one canonical journal record and `fsync`s it (lines 782–786).

This lane did not execute that loop. Optimizer updates and parameter changes
for the neural trainer are **unmeasured** here.

## Search

`jope pipeline run` defaults to the `legacy` search profile
(`crates/jope-pipeline/src/kernel2/messages.rs:4`). That profile returns no
kernel2 options (`kernel2/cli.rs:79–93`), so `cli.rs:246–251` calls
`crate::run::run`, which searches with `crates/jope-pipeline/src/search.rs`.

Constants at `edf36182`:

- `POPULATION = 12` (line 95). Generation 0 is one bootstrap pipeline
  `"(pipe X (ridge))"` (line 92) plus random canonical pipelines (lines 804–816).
- `ELITES = 4` (line 97). After a generation the best 4 by held-out RMSE are
  kept and are not fitted again (lines 911–914; the module note at lines 34 and 96).
- Later slots are `offspring` (line 600): `crossover` when two parents exist
  and the slot is even, otherwise `mutate` (lines 622–631). A duplicate is
  not fitted (lines 636–641 and the admission check at lines 852–857).
- A candidate is scored by `feat_math::fit` (`crates/feat-math/src/model.rs:36`)
  and `r2` / `rmse` (`crates/feat-math/src/metrics.rs:4` and `:24`).
  Non-finite scores are `SCORE_FAILED` (`search.rs:580–588`).

The receipt's engine string on the scratch run was
`feat-math canonical-pipeline GP (in-process)`.

### Scratch search, 2026-10-10T14:02:51Z

Command, under `systemd-run --user --scope` with `CPUQuota=200%`,
`MemoryMax=4G`, `MemorySwapMax=0`, `TasksMax=2048`, nice 15
(unit `grok-s3gp-pipeline1`):

`target/debug/jope pipeline run` on `datasets/test/pmlb-229-pwlinear`,
budget `evaluations=24,seconds=180`, `--cuts 1`, `--emit rust`,
`--threads 2`, `--cpu-ceiling 2000`, `--policy skip`,
`--source-commit edf36182b3fb3254afd244c9be1f1f6f274b9e0c`.

The binary is the dev profile built in this worktree (223,594,552 bytes).
It is not a release binary. `/usr/bin/time` reported `elapsed_s 4.79`,
`max_rss_kb 38352`, exit 0. The receipt is
`jope.pipeline-receipt/v3`, `ok: true`, `git_sha` the commit above,
`created_at` `2026-10-10T14:02:56Z`.

Cut 0: train 133 rows, hold 33 rows, 3 generations, 24 completed evaluations,
24 successful measurements, 0 failures, `stopped_by` `evaluations`.
Search-stage seconds in the receipt: 4.762869397.
Active search seconds: 4.761518901.
Evaluations per active second: 24 / 4.761518901 = 5.040408428.

Held-out RMSE by generation, from `run/search/cut-0/observations.json`
(24 rows):

| generation | measured | best held-out RMSE | change in the running best |
| --- | --- | --- | --- |
| 0 | 12 (1 `bootstrap`, 11 `random`) | 0.06733810931528403 | running best, from the bootstrap RMSE 0.1017061840553019, improved by 0.034368074740017865 |
| 1 | 7 (3 `crossover`, 4 `mutate`) | 0.06733810931528403 | 0 |
| 2 | 5 (3 `crossover`, 2 `mutate`) | 0.06763176878629071 for that generation | running best unchanged; that generation's best is 0.0002936594710066798 worse |

The selected pipeline is the generation-0 `random` row with that RMSE,
hold R² 0.8887899826991266, train R² 0.9736513096387953, fit seconds
0.050628978. Selection rule in the receipt:
`all_cuts_then_min_mean_validation_rmse_then_genome`.
The propose stage is `skipped` (the encode/propose stage is not in this
work package). `nvidia-smi` at the start and end of this process reported
GPU 0 utilization 0% and 1 MiB used. This search path did not use the GPU.
A GPU training rate is **unmeasured**.

## Rust scoring path

Two compilers exist in this tree.

The historical native scorer
(`deploy/dl-flywheel/pipelines/hpc_verified_v5/native_evaluate.py:89–103`)
runs the emitter, then:

`rustc --edition=2021 -O <dest>/emitted/model.rs -o <dest>/emitted/model`

and executes `<binary> <fold-train.csv> <fold-validation.csv> <predictions>`
once per fold. The same argv is in `program_worker.py:120`. The executable
cap is 16 MiB (`program_gate.py:19`). The in-process verified cache
(`pipelines/hpc_verified_v5/verified_cache.py:72`) checks object bytes and
SHA-256. It is not a disk cache of executables.

The emitter writes one `model.rs` (`jope-pipeline/src/emit.rs:161–166`).
`feat_math::emit_rust` starts at `emit_rs.rs:15`. The generated header's
usage line is produced at `emit_rs.rs:169`:
`./model train.csv test_features.csv predictions.csv`.

This lane did not invoke the Lambda worker. A local Lambda fit is
**unmeasured** in this session.

### Reproducible cache

`deploy/dl-flywheel/pipelines/rlfw_s3gp/score_cache.py` at `dfa461fd9`.
Protocol id `jope.repro-rust-executable/v1` (line 22). Flags (lines 25–32):
`--edition=2021`, `-C opt-level=2`, `-C codegen-units=1`, `-C debuginfo=0`,
`-C strip=symbols`, `-C link-arg=-Wl,--build-id=none`, plus
`--remap-path-prefix=<workdir>=/src`, and `SOURCE_DATE_EPOCH=0` (lines 130–131).
The cache key is protocol, source bytes, `rustc -vV`, those flags, and the
remap prefix (lines 65–75). `ensure` re-hashes before return (lines 111–114
and 155–160). `execute` re-reads the file immediately before `exec`
(lines 171–172). `run_batch` (line 196) calls `execute` per job, so a repeated
source compiles once.

A same-directory comparison of `-O` against `-C opt-level=2` was not run.
Flag equivalence of those two spellings is **unmeasured**. The scratch
comparison below is historical `-O` in two directories against the protocol
above.

### Scratch compile and execute

Source: the emitted `model.rs`, 19,555 bytes, SHA-256
`4fe74ecfcdba7f7c96a8b12c277fcc5509c2e378a5858a19e958ab680d3dcdfa`.
Receipt `runs/rust-score-1/receipt.json` (`jope.s3gp-rust-score/v1`).

Historical `rustc --edition=2021 -O` in two directories produced two
binaries of 4,468,432 bytes with different SHA-256 values
(`06318d7d…00085cf` and `1b9f3fb9…8d58a48`). They are not reproducible.

The protocol cache produced one binary of 433,944 bytes, SHA-256
`f990a717e67b3ccaa34cdc3a20a44896953201f4836529ef209b95d18398b105`.
Cold `ensure` took 0.831516358 s (`hit` false, `compiles` 1).
Warm `ensure` took 0.000422920 s (`hit` true). The binary contains
`/src/model.rs` and does not contain the cache directory path.
Four executions of that binary, after the one compile, took 0.053306455 s
wall, all exit 0, each 34 prediction lines and 661 bytes, one prediction
SHA-256 `3e6ef26c0d00b40ebf635bcd7343c5b33fa6d8249552f54745dd0808a09dc638`.

One historical `-O` binary, given the same `train.csv` and the same
feature file (test.csv with the last column removed), wrote the same 661
bytes (`runs/rust-score-1/prediction-parity.json`, `match` true).

## S3 lifecycle and cost

Bucket `veox-jopedime-learning-use2`, region `us-east-2`.
Library prefix at `edf36182` is
`experiments/gp-policy-dl/v2-trainonly/program-results/flywheel/v1/`
(`deploy/dl-flywheel/jope_flywheel/s3_library.py:11`).
Puts use `IfNoneMatch='*'` (line 50). Artifacts are stored at
`objects/<sha256>/<role>` (line 59). Reads require bucket, key, byte length,
SHA-256, and `VersionId` (lines 22–36).

### What was listed

Read-only, 2026-10-10, profile `veox-jepson-prod`, no put, delete, lifecycle,
or storage-class call. Receipt `receipts/measurement-v1.json`
(`jope.s3gp-measurement/v1`), wall 113.129 s.

`program-results/flywheel/v1/objects/` listed to completion: 21 pages,
`object_listing_truncated` false, 20,833 objects, 18,178,859,605 bytes,
every listed object `STANDARD`.

| role | objects | bytes |
| --- | --- | --- |
| `evaluation` | 3,891 | 140,021,609 |
| `input` | 1,664 | 524,239,652 |
| `program` | 1,812 | 276,029 |
| `rust_executable` | 3,889 | 17,344,244,624 |
| `rust_parity` | 3,889 | 125,407,889 |
| `rust_source` | 1,799 | 35,754,819 |
| `score_records` | 3,889 | 8,914,983 |

There are more `rust_executable` objects than `rust_source` objects
(3,889 versus 1,799). The 20 `record.json` GETs had 20 distinct executable
keys and 16 distinct source keys. Whether each extra executable is a second
compile of one source is **unmeasured**. The historical argv embeds the
per-evaluation directory, and the scratch compile showed that this produces
different binaries from one source.

Those 20 records (`per_evaluation_bytes`): n=20, min 4,488,091, max 4,918,722,
mean 4,688,510.25 bytes. That mean adds the record bytes to every artifact's
bytes, so a shared object is counted once per record. It is not the census.
Evaluator SHA-256 counts in those 20:
`626d93a5cd11830e1b7cd43bf46f3fbde42c06273e750faa6d64b69a0c037526` on 11,
`e3be938644722d14a1c1a0e82168518838c4d174cfe031076a339745c48870a8` on 9.
No current evaluator was designated, so none of these rows were classified
as stale.

`record.json` objects outside `objects/` were not fully listed. Their count
and bytes are **unmeasured**. `gp-hwm/`, `checkpoints/`, and
`paper-evidence/` were probed with `max-keys` 1. Their totals are
**unmeasured**.

Ten `HeadObject` calls returned `ObjectLockLegalHoldStatus` null and
`ObjectLockMode` null. `StorageClass` was omitted on those heads.
A legal-hold scan of the 20,833 objects is **unmeasured**.

`GetObjectLockConfiguration` saved at `logs/object-lock.json` returned
`ObjectLockEnabled: Enabled` and `Rule: null`.
`GetBucketLifecycleConfiguration` saved at `logs/lifecycle-live.json` has
three enabled rules: `abort-mpu` (prefix empty, abort incomplete multipart
uploads after 7 days), `observations-deep-archive` (prefix
`audit/observations/`, transition day 0 to `DEEP_ARCHIVE`), and
`checkpoint-blobs-gir` (prefix `blobs/` and tag `kind=checkpoint`,
transition day 30 to `GLACIER_IR`).
`TransitionDefaultMinimumObjectSize` is `all_storage_classes_128K`.
This lane did not put a lifecycle configuration.

### Prices and the dollar figure this listing supports

Rates are the 2026-09-01 `us-east-2` rows in
`.agent/proposals/claude-rlfw-20261009/w6/pricing/s3-us-east-2-price-receipt-v1.json`:

- `USE2-TimedStorage-ByteHrs`, first 50 TB: $0.023 per GB-month (the next
  tier, $0.022 from 51,200 GB, is a different row).
- `USE2-TimedStorage-GIR-ByteHrs`: $0.004 per GB-month.
- `USE2-Requests-Tier1`: $0.005 per 1,000 PUT, COPY, POST, or LIST.
- `USE2-Requests-Tier2`: $0.004 per 10,000 GET and all other requests.
  HEAD is charged with that row because the description says all other
  requests. A separate HEAD price row was not present.

The code constants are `s3_lifecycle.py:25–28` at `dfa461fd9`.
Standard-IA storage dollars per GB were not used. That rate is **unused**
here.

Census bytes at the first-tier Standard rate:
18,178,859,605 / 10^9 × 0.023 = $0.418113770915 per month.
This is the price of this prefix at that tier. Bucket-wide bytes that might
already sit in a higher tier are **unmeasured**, so this is not a bill.

`rust_executable` bytes at the same rate would be $0.398917626352 per month
if those bytes were absent. That figure is a counterfactual in
`receipts/dry-run-v1.json`. It is not a proposed deletion. The 3,889
executables stay until each source SHA-256 has a reproducible replacement
(`classify_object`, lines 143–152).

This run's own requests: 26 LIST, 20 GET, 10 HEAD.
26 × 0.000005 + 30 × 0.0000004 = $0.000142.
Production PUT and GET counts per day, and therefore production $/day, are
**unmeasured**. CloudWatch was not queried.

### Proposal rules that were not applied

`proposal_from_rows` (`s3_lifecycle.py:171`) sets `applied` false.
`refuse_apply(True)` raises (lines 253–255). `proposed_rules` is always
empty. An S3 prefix matches every key that begins with it, including keys
that were not in the listing, so a full object key is still a prefix.
`_listing_is_census` (line 164) records whether the note and `census` flag
describe a complete listing, and `listing_versioned` records whether every
input row has a version id. Neither flag emits a rule, because this
function is not given a role or tag filter. Rows that would expire or move
are listed in `object_candidates` with that object's full key
(`s3_lifecycle.py:206–232`). The head sample produced zero proposed rules:
3 keys untouched (`/gp-hwm/`, `/checkpoints/`, `/paper-evidence/`) and
7 retained (`executable_not_yet_superseded`, `evaluator_unmeasured`, or
`no_rule`). Proposal SHA-256
`c38278cf6654661b1140736b33e4fe06eae795b06ebdb9652adf26258fa8da10`.

Protected markers (lines 37–45): `/gp-hwm/`, `/checkpoints/`,
`/paper-evidence/`, `/provenance/`, `/sealed/`, `/judge/`, and the substring
`legal-hold`. A legal-hold status other than absent, empty, or `OFF`, or any
object-lock mode, is untouched (lines 64–68). An object whose legal hold was
not read is retained with reason `legal_hold_unmeasured` (lines 125–128).
A score whose evaluator SHA-256 is missing is retained as
`evaluator_unmeasured` (lines 140–141). A stale score (both evaluator ids
present and different) below 128 KiB is not individually transitioned
(lines 131–134), because the live configuration bills objects under 128 KiB
as 128 KiB. Glacier Instant Retrieval is proposed only at 90 days
(`GIR_MIN_DAYS`, line 31), which matches the receipt note that GIR has a
90-day minimum. Executable expiry is 14 days (line 32), which is inside that
minimum, so an object candidate expires rather than transitions.

Caps: 200 × 10^9 bytes per UTC day and 2,000,000 PUTs per UTC day, alarm at
80% on an admit (`BYTE_CAP`, `PUT_CAP`, `ALARM_RATIO`, lines 23–25).
`BudgetLedger` (line 258) is a local hash chain and does not call S3.
`admit` (line 311) takes an exclusive lock, reloads the file, then decides.
A second client opened on the same path before the first admit still sees
that admit and refuses a projection over the cap. A refuse is not appended.
A crash before the append leaves no reservation. `export_ledger` (line 357)
writes Parquet and
checks the row count through `read_parquet`. A second export to the same
DuckDB file replaces the table. The census export is
`receipts/objects-census.parquet`, 7 rows, SHA-256
`0be916761ae2808d4742d3ae0dc7647e9809866fda5ea11998de94c181ffd91d`,
874 bytes, with `receipts/objects-census.duckdb`.
The ledger event for this read-only run admitted 0 bytes and 0 PUTs.

## Integrity rules

1. Library writes are create-only (`IfNoneMatch='*'`, `s3_library.py:50`)
   and a read checks the exact version, length, and SHA-256 (lines 25–35).
2. This package does not delete, rewrite, or change storage class. Applying
   a proposal raises (`s3_lifecycle.py:253–255`).
3. Sealed, judge, legal-hold, high-water, checkpoint, paper-evidence, and
   provenance keys are untouched or, when the hold was not read, retained.
4. An executable expires only after its `rust_source_sha256` is in the
   reproducible-cache set. The live 3,889 executables are not in that set.
5. A cached executable is hashed before it is executed. A mismatch raises
   and the file is not run (`score_cache.py:111–114` and `171–172`).
6. The scratch historical binary and the cache binary agreed on prediction
   bytes for this one source and this one dataset. Agreement on any other
   source is **unmeasured**.
7. Outer-test labels were not read back into selection. The scratch score
   used the pipeline's prepared hold rows. `jope pipeline score` on the
   sealed test split was not run. A sealed-test number is **unmeasured**.

## Tests

`python3 -m unittest discover -s deploy/dl-flywheel/pipelines/rlfw_s3gp/tests`
on the repaired tree: 10 tests, OK, 3.065 s, including real `rustc`, a
second ledger client that refuses 60+60 bytes under a 100-byte cap, a
partial sample and a versioned census that both emit no prefix rule, and a
DuckDB round trip that
exports twice to the same file. The earlier `dfa461fd9` run was 8 tests, OK,
2.841 s, before those two cases existed.
The two test modules are assigned in `deploy/ci/coverage.json` and
`docs/internal/core-ip/findings/test-inventory.json`. The non-test modules
are hashed in `RESEARCH_SOURCE_CLASSIFICATIONS.json` and
`SOURCE_MANIFEST.json`. The Python lane command is
`python3 -m unittest discover -s deploy/dl-flywheel/pipelines/rlfw_s3gp/tests`.
Hosted CI for that registration is **unmeasured** until the run for this
commit finishes. The earlier hosted run on `50969b90a` failed hygiene with
two unassigned additions, and its Python lane was `skipped-by-path`.
