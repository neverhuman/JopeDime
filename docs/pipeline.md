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
by that cut's RMSE and canonical genome. Candidates are visited in rank order,
round-robin across cuts, with duplicates removed. Missing measurements are attempted
in cut order until each cut's total budget is spent. A failed attempt is retained
without retry. The pool can remain incomplete when time, evaluations or failures
prevent coverage; each program lists measured, failed and missing cuts.

Selection requires successful measurements on **all requested cuts**. Among those
programs, minimum mean held-out RMSE wins; canonical genome breaks ties. Programs
with partial coverage are excluded, even if their score is better on their smaller
cut set. If no program has complete coverage, selection fails and retains the
per-cut evidence. No partial or unmeasured fallback is emitted. This is development
selection over inner folds, not sealed outer-test evaluation or promotion.

New runs write `jope.pipeline-search/v2` and `jope.pipeline-receipt/v2`, binding the
requested total, phase allowances/counts/failures/stop reasons, active time and cut
coverage. `jope pipeline verify-receipt` validates either v1 or v2 according to its
own schema; v2 fields are required and their totals are checked. The v1 schema is
unchanged. Generate the v2 schema from code with:

```sh
CARGO_INCREMENTAL=0 CARGO_PROFILE_DEV_DEBUG=line-tables-only \
  cargo run --locked -q -p jope-pipeline --example receipt_schema
```

Verification: `cargo test --locked -p jope-pipeline` exercises budget exhaustion,
failed attempts, in-flight overrun, full-cut selection and legacy reading.
`E2E_OUT=<new-durable-directory> bash deploy/ci/e2e-smoke.sh` runs the committed
real-data development fixture, checks per-phase receipts against observations,
requires a non-bootstrap program successfully measured on all three cuts, and
verifies emitted program parity. The script refuses to overwrite existing evidence.
Use the repository's required Cargo environment for both commands.

Neural proposals remain explicitly skipped here. The learning-loop, S3 recovery,
worker isolation and promotion qualifications are separate follow-up work.
