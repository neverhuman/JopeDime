# Q1 harness

This implementation prepares Q1; it does **not** grant permission to run it. The
preregistration is unsigned. No learned or sealed study has been run by this task.
The production discovery adapter and frozen QLT-02 cohort remain prerequisites.

## Local acceptance

Use the repository Python formatter and the owned tests:

```sh
python3 -m ruff format deploy/qualification/q1/harness/*.py
python3 -m ruff check deploy/qualification/q1/harness
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s deploy/qualification/q1/harness -p 'test_*.py' -v
rustfmt --edition 2021 --check deploy/qualification/q1/harness/native_heuristic.rs
nice -n 10 python3 deploy/qualification/q1/harness/build_backend.py
```

The build script uses locked Cargo artifact messages and this lane's existing
`target/`; it creates no extra workspace or Cargo lockfile. It records compiler
commands, actual library artifacts, source/lock identities and binary SHA-256 in
`target/qualification/q1/build-receipt.json`. The Rust reference adapter invokes
real per-candidate rung-0 CV, charges failed attempts, and reports nonzero fold
scores. It refuses learned/qualification requests. It is a plumbing reference,
not the product discovery loop or an M3/evaluator-parity proof.

For smoke acceptance, a curator first retrieves the three **train blobs only**
listed in `smoke-catalog.json`, to an external non-sealed data cache, named
`<train_sha256>.csv`. Verify their catalog/inventory `origin=real`, task/family,
`has_meta_test=false`, and full blob hashes before execution. No data download,
S3 credential or payload is embedded in this harness or working tree. The source
catalog's license status is unknown; no raw data is redistributed in this PR.

```sh
nice -n 10 env PYTHONDONTWRITEBYTECODE=1 RAYON_NUM_THREADS=8 \
  python3 deploy/qualification/q1/harness/q1.py smoke \
  --backend target/qualification/q1/native-heuristic \
  --data-root "$Q1_SMOKE_DATA_ROOT" --out "$Q1_SMOKE_RECEIPTS"
```

This runs three datasets × two seeds, four charged candidates and two folds per
candidate, using at most 128 rows/eight features from each non-sealed FIT blob.
The command rejects symlinked blobs, bad hashes, forbidden cache roots and existing
receipt files. A failure exits nonzero and preserves started/finished records with
unknown counts rather than zero/placeholder scores. The adjacent `.plan.json`
records the exact protocol, code and backend identity. No smoke result can produce
a Q1 decision. Archive these real smoke scores in the task-separated score store
with `stage=heuristic_smoke`; do not mix them into a learned validation headline.

## Qualification interface — remains locked

QLT-02/QLT-04 supply the actual metadata-only plan. `q1.py`'s closed plan/dataset
field sets are authoritative. Every dataset has an authoritative family and a
canonical dataset/cut SHA-256, `task=regression`, positive rows/features, and
`holdout_group=sealed_meta_test`; there are no qualification payload paths. Require
40 unique families disjoint from the nonempty frozen training-family closure and
at least five members of each prespecified subgroup. The plan pins the backend,
source, bundle, DOPE, evaluator, vocabulary, protocol and training-closure hashes,
seeds 0/1/2, 24 charged admissions, 600-second per-run limit, and absolute persistent
`ledger_root`. Opaque metadata requests are sent to the separately qualified
backend; this harness never opens sealed/DL-holdout rows. Do not substitute the
reference smoke adapter or development data for a qualification backend/cohort.

OWNER supplies the allowed-signers file and places its SHA-256 in the preregistration,
then signs the exact output of this command using identity `jepson` and OpenSSH
namespace `jopedime-q1`:

```sh
python3 deploy/qualification/q1/harness/q1.py approval-payload --plan "$Q1_PLAN"
```

There is deliberately no command to create an OWNER approval or signature. The
signature binds document + full plan + backend. PENDING, missing/bad signature,
changed backend/plan/document/key or changed ledger root blocks execution before
any qualification subprocess. The temporary signing key in unit tests is a test
fixture, not an OWNER key or approval of this study.

After actual OWNER authorization, QLT-04 can invoke `run --plan ... --backend ...
--out ... --signature ... --allowed-signers ... --ledger ...`. The signed ledger
root gets an exclusive per-bundle started marker; repeating that bundle is refused.
Completed/incomplete execution receipts are retained separately; interrupted
started-only outcomes remain unknown. This tool does not implement automatic
qualification retries/resume or budget resets.

The pinned evaluator consumes one `jope.q1-request/v1` JSON document on stdin and
returns one closed `jope.q1-response/v1` document on stdout. Requests bind mode,
plan/source/pins, dataset/cut, policy, seed, budget and request digest. Responses
bind that digest/policy/metric and provide actual charged/completed counts, nonzero
fold-score counts, per-attempt failure inventory, finite RMSE (or null on failure),
program identity and adapter identifier. Budget echoes alone are not proof of a
real production run; independently qualify the backend and retain its evidence.
Only the smoke mode includes numeric rows in the request.

`analyze --plan ... --receipts ... --signature ... --allowed-signers ... --out ...`
re-verifies authorization and accepts only a complete, unique, identity-matched
paired panel with corresponding starts. It averages seeds within datasets, then
computes the median relative improvement and 10,000 paired bootstrap resamples,
with the prespecified subgroup checks. Unknown/failed/missing results never become
zero or an apparent Q1 pass. It writes a new report and refuses overwrites.

## QLT-04 FIFO launch preparation

Prepare a create-only intent descriptor without acquiring the shared lock,
starting a waiter or reserving a live FIFO position:

```sh
python3 -B deploy/qualification/q1/harness/fifo_launch.py prepare \
  --out "$Q1_INTENT" --owner sol_v1_obj02
```

The descriptor says `prepared_not_enqueued`, with `active_ticket=null`. Use a
descriptive JSON filename, not the numeric PID ticket format. Preparation can
record the current unsigned state; it grants no execution authority. W1 must
release `qualification.lock` before Q1 can run, and all admission gates still
apply after that release.

Only after the production backend, actual authoritative cohort, model and M2/M3
proofs are admitted and OWNER signs the final protocol/plan/backend, launch from
a clean merged checkout at the plan's exact `source_commit`:

```sh
nice -n 10 python3 -B deploy/qualification/q1/harness/fifo_launch.py launch \
  --plan "$Q1_PLAN" --backend "$Q1_BACKEND" \
  --signature "$Q1_SIGNATURE" --allowed-signers "$Q1_ALLOWED_SIGNERS" \
  --ledger "$Q1_LEDGER" --run-dir "$Q1_RUN_DIR" \
  --wait-seconds "$Q1_WAIT_SECONDS" --wall-seconds "$Q1_WALL_SECONDS" \
  --authorized-launch
```

The operator flag supplements the real OWNER signature. Before creating a live
ticket, the wrapper verifies admission, frozen DOPE, executable backend hash,
clean merged source and the canonical owner-signed persistent ledger. It refuses
any existing started or finished marker for this bundle. It repeats admission
and compares input hashes after taking the existing exclusive lock. The backend's
scientific qualification and private input closure remain separate prerequisites;
a signature check alone does not prove them.

The launcher joins the existing numeric PID FIFO with a 45-second courtesy delay,
then polls every 30 seconds. The finite wait cap is at most three hours. It reads
other tickets without pruning them, never replaces the lock, and removes only its
own ticket. The CPU-only child environment uses eight Rayon threads and one
OpenMP/BLAS thread, with niceness 10; dispatch refuses host load above 85 and
xbabe0. Admission is checked again before analysis. No GPU qualification occurs.

Expected wall time is **unknown** until qualified backend timing exists. The
current Q1 driver runs policies serially: `N` datasets produce `6*N` policy runs,
`144*N` planned charged admissions and `3600*N` timeout-seconds. At the minimum
40 families, that is 240 runs, 5,760 planned admissions and a **40-hour timeout-only
bound**, not an ETA. Eight backend threads do not divide that bound by eight.
The caller must choose a finite whole-job wall cap no larger than that arithmetic
bound. It covers post-lock admission, run and analysis; queue wait is separate,
and abort cleanup may add a five-second grace interval. A shorter wall cap may
leave the candidate consumed with an unknown outcome; no retry is automatic.

The driver inherits the lock descriptor. The wrapper retains the child leader
until it signals the owned process group, including descendants, on completion,
cancellation or timeout. It preserves original ledgers, logs and receipts and
writes a create-only `launch-receipt.json`; failed or incomplete execution never
becomes zero counts. A completed analysis can still say `FAIL`. This wrapper
never changes the guidance default or promotes a model. The launcher tests use
private queues, locks, fixture signing keys and benign child processes; they are
software control proofs, not a Q1 study.
