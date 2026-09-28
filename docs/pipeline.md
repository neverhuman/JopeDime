# CPU pipeline budget and selection contract

`jope pipeline run --dataset datasets/test/pmlb-229-pwlinear --cuts 3 \
  --budget evaluations=4,seconds=60 --rng-seed 7 --policy skip \
  --emit rust,python --output <new-durable-output-directory>`

The budget limits candidate attempts per cut across exploration and cross-cut
comparison together. Parsing, fitting, scoring and numerical failures all consume
an attempt; a program is attempted at most once on a cut. Previously, four
requested attempts could produce ten per cut after unbounded remeasurement.
Receipts from that implementation remain readable as v1, without a v2 guarantee.

For multiple cuts, exploration gets `ceil(evaluations / 2)` attempts and half the
seconds. Comparison receives the unspent total allowance. One cut uses the entire
allowance for exploration. A one-attempt budget can measure only the bootstrap;
receipts show that limited coverage. No extra fits are hidden after the limit.

Seconds measure cumulative active wall time in each cut's search and comparison
admission loops, including candidate generation during search and fit/scoring.
Time spent on other cuts, constructing the shared comparison pool, publishing
receipts, preparing data and emitting sources is outside this numerical allowance.
This is an admission deadline: the in-process fitter cannot be interrupted safely.
An admitted fit can overrun it, and then no new fit is admitted on that cut. The
receipt reports actual active seconds and `deadline_overrun`; it does not claim a
hard process deadline. A controller requiring hard cancellation still needs an
isolated numerical worker.

The comparison pool contains up to four measured programs from each cut, ranked
by that cut's RMSE and canonical program text. Candidates are visited in rank order,
round-robin across cuts, with duplicates removed. Missing measurements are attempted
in cut order until each cut's total budget is spent. A failed attempt is retained
without retry. The pool can remain incomplete when time, evaluations or failures
prevent coverage; each program lists measured, failed and missing cuts.

Selection requires successful measurements on **all requested cuts**. Among those
programs, minimum mean held-out RMSE wins; the canonical program text breaks ties. Programs
with partial coverage are excluded, even if their score is better on their smaller
cut set. If no program has complete coverage, selection fails and retains the
per-cut evidence. No partial or unmeasured fallback is emitted. This is development
selection over inner folds, not sealed outer-test evaluation or promotion.

New runs write `jope.pipeline-search/v2` and `jope.pipeline-receipt/v3`, binding the
requested total, phase allowances/counts/failures/stop reasons, active time and cut
coverage. For multiple cuts, the overall stop reason is comparison's outcome:
`seconds` takes precedence when both total limits are reached, then `evaluations`,
otherwise `comparison_pool_exhausted`. The exploration phase retains its own
reason (`evaluations`, `seconds` or `search_space_exhausted`). Single-cut runs use
the exploration reason and record comparison as `not_requested`.

## Seeding (`--seed-genomes`, `--incumbent`)

`--seed-genomes <seeds.json>` reads a `jope.pipeline-seeds/v1` file (the champion bank's
export format, and what `jope pipeline evolve` writes per round). `--incumbent <receipt.json>`
takes a parent run's receipt: it is re-derived from that run's own evidence (prepared cuts,
every cut's `search.json` and `observations.json`, the selection bit for bit, and the emitted
files) and refused if anything disagrees; its selected candidate is offered first. Offered
candidates are canonicalised, deduplicated (the bootstrap is always in generation 0 already) and capped at
four, then enter generation 0 right after the bootstrap with origin `incumbent`. They spend
evaluations like any other candidate and are re-measured on this run's cuts: no score is carried
over. A budget whose exploration allowance cannot measure the bootstrap plus every seed on each
cut is refused before anything is written, and the seconds bound never skips them: generation
0's bootstrap and seeds always start (only a stop request prevents it; an overrun is recorded as
`deadline_overrun`), so every seed has a generation-0 measurement on every cut, which is what
`verify-receipt --evidence` requires. The random draws after them are what the seconds bound
stops. The receipt's `search.seeding` records the offered
count, each admitted seed's `genome`, sha256 and source, the parent receipt's sha256 and the seeds
file's sha256; the v3 validator requires every seed to have been attempted on every requested
cut. Because the parent's champion is re-measured on the same cuts, a seeded run's champion is
never worse than its seed there.

`jope pipeline verify-receipt` validates v1, v2 or v3 according to its own schema; v2 and v3
fields are required and their totals are checked. The v1 and v2 schemas are unchanged.
`--evidence` also re-derives a v2/v3 receipt from its run directory. Generate the v3 schema
from code with:

```sh
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only \
  cargo run --locked -q -p jope-pipeline --example receipt_schema
```

## Parallelism (`--threads`, `--islands`, `--cpu-ceiling`)

`--threads N` evaluates each generation's admitted candidates on N workers. Admission (which
candidates are fitted, up to the evaluation bound) and the fold of results back into the search
both happen in slot order, so the thread count never changes a decision: `--threads 1` and
`--threads 8` publish the same observations, selection and receipt apart from wall-clock
readings (with the clock frozen in tests, byte for byte). The comparison phase runs one worker
per cut. The thread count is not recorded in the receipt, because it cannot change it. Every fit
runs on an evaluation-pool thread with a 32 MiB stack, `--threads 1` included (and the test-split
refit too), so a deep fit cannot measure at one worker count and overflow a smaller default
stack, aborting the process, at another.

`--islands M` runs M independent random streams over the same cuts. Island 0 is `--rng-seed`
itself, so one island is the plain search. Islands share each cut's evaluation budget and the
rule that no candidate is fitted twice on a cut; a candidate another island already measured joins
the drawing island's candidates with its existing score. Every `--migration-interval K`
generations (default 4) each island's top-1 is carried into every other island's elites; it
enters only if it beats one of them, and a migrant is never re-fitted. Observations carry their
island, and the receipt's `search.islands` records the count, the interval and every migration
(cut, generation, from, to, program sha256, RMSE, admitted). The evaluation pool has
`islands x threads` workers; a run is refused before anything is written when that exceeds the
CPUs `sched_getaffinity` allows, `available_parallelism`, or `--cpu-ceiling <millicores>`.

## Child-process use (`--source-commit`, SIGTERM)

`--source-commit <sha>` records the given commit as the receipt's `git_sha` and never spawns
`git` (without it the run uses `JOPE_GIT_SHA`, else `git rev-parse HEAD`). SIGTERM or SIGINT
stops a run between evaluations: in-flight fits finish, nothing new starts, no receipt is
written, and the process exits 143. A second signal exits at once.

## Targeted search (`jope pipeline evolve`)

```sh
jope pipeline evolve --dataset datasets/test/uci-parkinsons-telemonitoring --rounds 3 --control \
  --cuts 3 --budget evaluations=200,seconds=600 --threads 8 --policy skip --emit rust \
  --output <new-directory>
jope pipeline lineage verify --lineage <new-directory>/lineage.json --require-bred-candidates
```

Each round is a full `jope pipeline run` in `round-NNN/` on the identical prepared cuts (fixed
fold seed and cut count). Round 1 starts from `--seeds` (a `jope.pipeline-seeds/v1` file, such
as a champion-bank export) or unseeded; every later round is seeded from the previous round's
champion and ranked runners-up (at most four, `islands::next_seeds`), with the previous receipt
as its verified `--incumbent`, and its own random stream. Because the previous champion is
re-measured on the same cuts, no round's champion is worse than the round before *on those
cuts*. Every round selects on the same held-out folds, so part of a round-over-round gain is
selection on reused folds; only the sealed test split speaks to generalisation. `--control` adds
`control/`: one run at the final round's configuration (one round's budget and the final round's
random stream) with no seeds. It shows what one round of search finds without inherited seeds;
it is not compute-matched against the lineage, whose final round inherits every earlier round's
search on the same cuts (`lineage verify` reports the rounds' total evaluations beside the
control's), and one control on one stream cannot separate seeding from that extra compute or from
noise.

Every refusal a round could make is made before the output directory is created: the counts,
budget and parallelism; round 1's seeds (a seed that is not a pipeline, or more seeds than the
budget can re-measure); a dataset that does not resolve, is not a regression task or does not
parse; and, with more than one round, a budget whose exploration allowance cannot re-measure the
four seeds a later round may carry. A refused command leaves nothing behind, so the corrected
command can be re-run into the same `--output`.

`lineage.json` (`jope.pipeline-lineage/v1`) is rewritten atomically and fsynced after each
completed round: per round the `selected_genome` and program sha256, the mean held-out RMSE, R^2
and train R^2 on the identical cuts, the parent receipt sha256, the seed origin and seeds, each
cut's per-generation best-so-far and stop reasons, the generation-1+ measurements and the
receipt's stage seconds. A round interrupted by a signal writes no receipt and never enters it.
`jope pipeline lineage verify` re-derives every round and the control from its directory and
checks the chain: same dataset, cuts, budget and island model; each parent is the previous
receipt; each round's seeds are exactly the previous round's champion and runners-up; the
recorded numbers are what the receipts show; no champion got worse. `--require-bred-candidates`
also demands a measured candidate bred in generation 1 or later in every round: the search ran
past the candidates it started from, which says nothing about whether anything it bred was better. The
report says that separately: `champion_found_in_round` per round (the champion is neither the
bootstrap nor one of the seeds the round started from, so the round's own search produced it)
and `improved_on_previous_round` (strictly lower mean held-out RMSE than the round before);
`jope pipeline evolve` prints each round's champion as found in the round, carried from its
seeds, or the bootstrap.

`jope pipeline evolve` never writes to the learning store. HWM ingest and the champion bank are separate
`jope-learning` steps over its outputs (see `docs/learning.md`).

## Test-split scoring (`jope pipeline score --split test`)

Everything above measures on the training rows only. `jope pipeline score --receipt <r> --split
test` re-derives the receipt, re-resolves the sha-verified dataset (it must be the dataset and
`train.csv` the search measured), refits the selected candidate on every `train.csv` row and scores
it on `test.csv`, writing `jope.pipeline-test-score/v1` to
`<registry>/<dataset_hash>/<program_sha256>.json` (default registry
`$JOPE_STATE/pipeline/test-scores`). The path is checked before any fit and the file is created
exclusively, so each (dataset, candidate) pair is scored once in that registry: a repeat from the
same receipt, or from any other run that selected the same candidate, is refused. Nothing in the
pipeline reads a score back and the run's receipt is not touched, so the tools never feed it to
selection. What they cannot enforce is the operator's side: the registry is a local record the
caller chooses (a new `--registry` or `$JOPE_STATE` starts an empty one), and scoring several
candidates of one dataset and keeping the test-best would make the test split a selection input.
Each score receipt therefore records `earlier_looks`, the candidates already scored on that
dataset's test split in the same registry, and the command warns when there are any.

Verification: `cargo test --locked -p jope-pipeline` exercises budget exhaustion,
failed attempts, in-flight overrun, full-cut selection and legacy reading.
`E2E_OUT=<new-durable-directory> bash deploy/ci/e2e-smoke.sh` runs the committed
real-data development fixture, checks per-phase receipts against observations,
requires a non-bootstrap program successfully measured on all three cuts, and
verifies emitted program parity. The script refuses to overwrite existing evidence and uses a unique default directory
under `target/e2e-smoke/` when `E2E_OUT` is unset.
Use the repository's required Cargo environment for both commands.

Neural proposals remain explicitly skipped here. The learning-loop, S3 recovery,
worker isolation and promotion qualifications are separate follow-up work.
