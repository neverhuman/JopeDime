# Native fit authority core and remaining release gates

This document describes the locally implemented native core, not a production
readiness declaration. The release owner must keep fitting fenced until historical
coverage, the mandatory executor boundary, target storage/runtime compatibility,
and the long-lived authority service have passed their release gates. No production
namespace was enabled by the registry lane.

## Source and computation contracts

The implementation is `src/eval_registry.rs`,
`src/eval_registry/authority.rs`, and `src/eval_registry/service.rs`; synthetic
core and service tests are in their respective `tests.rs` modules. The old optional Rust v1 API is intentionally
removed. The independent Python shadow prototype is not a fit authority.

`VerifiedFit` owns the exact ordered numerical feature and target arrays together
with preprocessing and feature-view bytes. It uses the evaluator parser's
canonical representation, preserving operator order and parameter attachment.
The executor verifies its actual executable and mapped runtime files, numerical
thread environment, and permitted fit randomness. The supervisor and executor
independently reconstruct the same descriptor from sealed inputs.

`ordered_training_sha256` hashes dimensions and row-ordered f64 bits. It is not
the prepared CSV-byte digest. Consumers must reconstruct the actual numerical
arrays to compare this identity. `numerical_contract_sha256` includes numerical
settings, and excludes the descriptive protocol name. Scope policy permits only
declared training digests, numerical-contract digests, and fit seeds.

Query inputs, scoring targets, metric definitions and aggregate weights use
separate derivation keys. A new query, metric, family label, generation, search
seed, path, admission policy, or wall deadline does not change a fit key.

The full fit identity records the actual executable and loaded runtime. A separate
permanent numerical-work guard excludes those release identifiers and conservatively
blocks another physical start for the same canonical program, ordered arrays,
preprocessing, feature view, fit seed and numerical settings. A UI-only recompile
cannot grant numerical novelty. A genuinely changed implementation needs a
separately reviewed protocol; this implementation has no automatic bypass or
cross-release model-alias reuse.

## API and durable transitions

1. `Authority::initialize(root, witness, scopes)` explicitly creates a disabled
   registry. Existing paths fail. The independently retained witness must be
   outside the registry backup root. Independent retention remains an operational
   requirement; directory placement alone does not prove backup independence.
2. `Authority::open(root, witness)` opens existing SQLite without creation,
   verifies the journal and witness, and replays safety-row postimages against the
   indexed entries, exclusions, coverage and scope bindings.
3. `claim(fit, provenance)` checks scope, coverage and permanent exclusions. An
   existing or blocked request performs no journal mutation.
4. `authorize_with_deadline(fit, claim, executor, duration)` binds a single token
   to the executor's PID, process-birth ticks, boot ID, task and absolute monotonic
   deadline. The convenience `authorize` uses 30 seconds.
5. The actual executor calls `consume_start`. It rechecks history and loaded
   runtime, consumes the token once, and retains non-cloneable exclusive file
   custody throughout numerical work. Foreign/inherited tokens are rejected.
6. The supervisor uses `ObservedCompletion::wait` on its original `Child` handle.
   A later call cannot reset the original deadline. Only `complete_observed`
   creates a native completion. A nonzero exit, signal or timeout cannot become
   success because a valid-looking result exists.
7. `verify_committed_completion` proves a result belongs to the approved original
   authority, current index, retained journal and immutable artifacts. A
   self-hashed JSON completion alone is insufficient. `model_for_completion`
   reports unavailable if the original model is absent; it never dispatches.

Every actual mutation synchronizes the retained journal, atomically publishes and
synchronizes its independent witness, then commits SQLite with FULL synchronization.
Interrupted publication leaves an explicit disagreement that blocks dispatch.
SQLite uses indexed lookups and touched-row journal postimages. Subsequent calls
verify appended tails, while an unjournaled database change triggers complete
state verification. Large immutable artifact validation occurs outside the writer
critical section.

`claimed`, `authorized` and `running` entries have no automatic retry/reset API.
Ambiguous work remains unavailable. Authorized work whose executor fails before
consumption also remains unavailable. Storage divergence currently requires a
separately reviewed offline reconciliation; the core does not silently repair it.

## History and completion limits

`import_history` requires nonempty content-bound inventories, all files under the
declared roots, occurrence/exclusion evidence for every file, and a preserved
legacy-fence receipt. It retains the source bytes and publishes exclusions plus
coverage together. The external reviewer must establish that the declared roots
cover the entire legacy namespace and that the fence receipt proves dispatch has
stopped. This API is not an automated proof of namespace completeness or source
eligibility. It cannot import historical quality as a reusable completion.

Partial historical identities match richer later requests. Missing computational
fields are wildcards, and changing provenance scope cannot escape a program/input
exclusion. An unresolved containing scope blocks its authorized training inputs
across scope aliases. Permanent exclusions outlive model eviction.

Successful native fit receipts require a retained original model. Predictions and
metrics are optional derivations because model export can complete without an
evaluation query. The receipt's elapsed interval runs from dispatch authorization
to observed child exit, including intervening verification and dispatch time; it
must not be described as isolated numerical CPU time. Successful quality, terminal
failures, timeouts, infrastructure failures and unknown outcomes remain distinct.

## Qualified locally

The source-bound receipt is in `runs/native-fit-authority-20260910-local/`.
The tests execute synthetic state transitions and subprocess outcomes, never a
numerical fit. They cover 104 concurrent identical requests with one consumed
start, sequential/foreign token rejection, missing storage, database and journal
rollback against an independent witness, missing/torn journal bytes, failures
after journal/witness publication, deleted indexed safety rows, denied scope,
partial exclusions and import races, failed exit/signal/timeout, original model
reuse and unavailable export. The internal ignored test is a synthetic executor
invoked explicitly by its parent tests.

Run the scoped suite with the declared numerical environment:

```sh
rtk proxy env OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 cargo test --manifest-path parallel/Cargo.toml --lib eval_registry --no-default-features
```

The isolated receipt copies the compiled unit-test executable before execution so
another lane's build cannot replace a running test's mapped executable. Local
failure injection is not target-filesystem ENOSPC or device-loss qualification.

## Long-lived native service

`Service::bind` opens and validates one authority. `Service::run` owns original
worker creation, authenticated start consumption, process-group cleanup, original
exit observation and receipt commit. Clients and workers do not open SQLite.
`submit_to` accepts sealed numerical inputs and provenance; `verify_original`
checks an original committed receipt against the requested scope without opening
the authority in an admission process. `ServiceResult` carries authority-bound
receipt/model content references. `ServiceConfig::regression` uses fit seed zero,
f64, one numerical thread, a 30-second fit ceiling, and at most 104 workers.

The service verifies its actual executable and mapped runtime at startup and pins
the same executable for workers. A worker independently reconstructs the fit
identity from its inherited sealed descriptor. Its new Unix connection provides
kernel `SO_PEERCRED` for the actual original child; the service verifies PID,
process-birth and boot identity against its owned `Child` before consuming one
token. The worker receives explicit start permission and an exclusive custody FD.
Both the custody descriptor and original custody pathname must match. A lost
permission reply remains consumed; a lost client reply cannot cancel supervision
or cause replay.

Service identity binds the approved authority manifest/instance, independent
witness path, process incarnation, executable and original socket inode. Client
and worker authenticate this identity. Descriptor serialization, sealing and
hashing occur before connection. Connection establishment, frames and descriptor
transfer use bounded nonblocking operations. Full-backlog connect retries are
limited to a pre-transfer condition within the original connection deadline.

Each submission requires an integer `details.deadline_monotonic_ns` in the host's
`CLOCK_MONOTONIC` domain. It is provenance, never fit identity. The service checks
it before claim, bounds worker readiness by its remaining time, rechecks after
actual worker verification, and caps original authorization by the lesser of that
remaining time and its fit ceiling. Worker custody verification rejects expired
start permission. Existing successful work can be reused without another fit;
expired requests do not authorize new work.

A lifetime lease and dispatch mutex protect the single owner. Every claim and
start transition checks the original lease/socket while holding this mutex.
Custody loss fences dispatch before accepted handlers drain. Graceful stop accepts
only known pending worker peers, so unrelated connection floods cannot prolong
drain. The original service retains execution custody through observed exit and
commits accepted work even when a submitter disconnects. `waitid(WNOWAIT)` holds
the original PID identity until process-group cleanup and original `Child::wait`.

Startup refuses unresolved claimed, authorized or running entries. There is no
automatic task reset, stale socket replacement, empty database replacement or
journal/witness repair. Explicit offline reconciliation is required. This is
conservative: even a claimed task known not to have consumed a start remains
blocked until reviewed. Existing pre-guard databases require explicit migration;
normal startup does not create the missing safety table.

## Service verification and remaining release gates

The final combined local receipt is
`runs/native-fit-service-20260910-local/complete.json` (SHA256
`d133ba728f1af0654696e7a3e1643b1960d9a5312c02a62fc898a8623f0e763d`).
All 24 selected tests passed; two internal synthetic worker tests are marked
ignored for ordinary discovery and are invoked explicitly by parent tests.
Formatting passed and bound source remained unchanged. The final verification
used 121.998 seconds of wall time and 366.97333 CPU core-seconds, with zero
numerical fits and GPU tasks. These figures describe this final verification;
prior development runs remain separately recorded. Its source manifest binds
the core, service, tests, Cargo metadata and evaluator source. Tests execute
synthetic state transitions and fake numerical workers, not numerical fitting.
The earlier core-only receipt remains valid evidence for its original source;
the development service logs retain failed runs and subsequent corrections.

The local service cases include 104 concurrent identical clients with exactly one
original worker and one authority open; immutable reuse and scoped verification;
missing authority; stale/replaced socket; forged executor ticket; sealed/unsealed
FD validation; partial-frame and full-backlog deadlines; invalid request deadlines;
failed exit, signal, timeout and replaced custody; loss of either start permission
or final reply; and socket loss during worker readiness with authorization and
restart fenced. Local failure injection does not establish target-filesystem
ENOSPC, device-loss or live migration qualification.

No representative-history throughput, terminal p99 under 2 ms, claim/start p99
under 10 ms, or under-5-percent evaluator overhead has been established. The
104-client test proves concurrency correctness only. Historical namespace
completeness, verified reusable historical import, offline reconciliation,
independent witness retention, target fault/performance tests, mandatory public
entry-point integration and exact integrated release CI remain separate gates.
The automatic training hold and serving reference are outside this core's
authority and were not changed by the registry lane.
