# DSC-03 row-access wiring handoff (source `8b0309711`)

This is a CPU source audit and integration handoff for DSC-03-F01. It does not
wire product callsites, establish a trusted split, or qualify a live discovery
run. The DSC-03 card assigns actual callsite selection to the relevant
integration owner; this document identifies candidate seams for that owner to
select and claim before any source change.

## Existing contracts

`crates/jope-service/src/discover/audit.rs` defines `AuditRowId { dataset:
DatasetId, row: u64 }`, phase names and append-only `RowTouch` entries. The
current constructors and hooks are:

```rust
RowAudit::new(fit: Vec<AuditRowId>, holdout: Vec<AuditRowId>,
              test: Vec<AuditRowId>) -> Result<RowAudit, RowAuditError>
RowAudit::record(&mut self, phase: RowAuditPhase,
                 rows: &[AuditRowId]) -> Result<(), RowAuditError>
RowAudit::with_rows<T>(&mut self, phase: RowAuditPhase,
                       rows: &[AuditRowId],
                       read: impl FnOnce(&[AuditRowId]) -> T)
    -> Result<T, RowAuditError>
RowAudit::begin_finalization(&mut self)
RowAudit::entries(&self) -> &[RowTouch]
```

Search phases (`Landmarks`, `Rung0`, `Rung1`, `Preprocessing`, `Calibration`)
and finalization phases are separated by the one-way `begin_finalization`
boundary. A rejected batch is not logged and its closure is not called. A
successful `with_rows` logs the attempted touch before invoking the closure;
the caller must separately retain the closure's actual success, error or
panic outcome. The serialized `RowTouch` alone is not a read-success receipt.

`crates/jope-service/src/discover/folds.rs` exposes:

```rust
FoldPlan::new(rows: Vec<AuditRowId>, constraints: FoldConstraints, seed: u64)
    -> Result<FoldPlan, FoldError>
FoldPlan::folds(&self, phase: CvPhase, folds: usize, cap: usize)
    -> Result<Vec<CvFold>, FoldError>
FoldPlan::calibration_folds(&self, outer: &CvFold, folds: usize, cap: usize)
    -> Result<Vec<CvFold>, FoldError>
CvFold::train(&self) -> &[AuditRowId]
CvFold::validation(&self) -> &[AuditRowId]
```

The plan validates a single-dataset unique FIT roster and aligned group/time
metadata. It refuses infeasible group/time/gap constraints. Calibration folds
are made from the outer fold's training roster and inherit its aligned
constraints. These are row-selection descriptions; constructing folds does
not prove a worker or landmark reader used those rows.

`crates/jope-service/src/discover/loop.rs` has private `Plan::fold_spec(&self,
rung: u8) -> Result<Value>`; it maps matrix ordinals to `AuditRowId`s and uses
`FoldPlan` to serialize worker request folds. `Discovery::advance(&mut self,
space: &impl SearchSpace, admission: &impl Admission, executor: &impl Executor,
live_workers: usize, stop_after_rounds: usize) -> Result<Vec<PublicEvent>>`
dispatches evaluator requests and processes outcomes/refit. Neither function
owns a `RowAudit` or records actual worker reads. The hidden `run_m1(path:
&Path, executable: &Path) -> Result<Value>` creates the local store/worker and
executes search, but its input is an internal `Plan` over a FIT-only matrix;
its result is not the public prepared-partition/finalization adapter.

`crates/jope-service/src/worker/runtime.rs::serve` maps the request's immutable
matrix and calls its evaluator callback. It has no dataset-row roster or
`RowAudit` hook. `discover/landmark_pools.rs::select_landmark_preferred_pools(
landmark_v2_48: &[f64; 48]) -> (Vec<&'static str>, Vec<&'static str>)` consumes
precomputed contrasts and selects strings; it does not read source rows. The
landmark feature producer/caller, if it reads rows, is a separate actual-access
boundary to locate and claim.

`crates/jope-service/src/discover/finalize.rs` is the existing partial
integration. `Finalizer::finalize(&self, request: &FinalizationRequest,
executor: &impl Executor) -> Result<FinalizationReceipt>` builds a `RowAudit`
from prepared `Partition.rows`, claims once, enters the finalization boundary,
and wraps FIT fit, HOLDOUT score, optional FIT+HOLDOUT refit, and optional TEST
score with phase-specific `with_rows`. It validates provenance rosters and
partition digests before these reads. Its fixture tests exercise real local
partition reads and refusal/replay behavior. The module docs say HTTP, CLI and
constrained-split preparation remain adapters; the tests instantiate
`Finalizer` directly. No non-test public completion caller or trusted
prepared-roster producer was found in `crates/jope-service/src/public/` at the
audited source.

`discover/splits.rs::random_split(row_count, fit_rows, seed, label)` and its
related index helpers generate deterministic index splits; they do not accept
group/time constraints or provide trusted FIT/HOLDOUT/TEST provenance. They
must not be treated as the missing production partition producer. Outer
FIT/HOLDOUT selection must be performed by an owner-selected component that
enforces configured group/time policy before handing the FIT roster and aligned
metadata to `FoldPlan`.

## Owner-selected implementation seams

These are candidate files only, not claimed patch scope:

1. The discovery integration owner selects the actual worker-evaluation call
   boundary (candidate seams: `discover/loop.rs`, `worker/runtime.rs`, and the
   evaluator implementation invoked by the callback) and claims exact paths.
   At each real data read, validate and record the precise fold training rows
   immediately before read/preprocessing/fit. Record validation rows only for
   the score operation authorized by the protocol. Make an infeasible fold a
   refusal; do not fall back to random CV. Calibration must use
   `calibration_folds(outer, ...)` and only inner training rows for fit.
2. The landmark owner locates and claims its row-producing/access callsite.
   Record the actual landmark source rows before reading/deriving contrasts;
   the pool-selection function itself is not that boundary.
3. The prepared-partition/public-run owner supplies validated dataset IDs,
   matrix ordinals, source/matrix/split identities, and aligned group/time
   metadata. That owner selects the public completion adapter and connects it
   to `Finalizer`; no caller or producer is inferred by this handoff.
4. Persist attempted row touches together with source commit, matrix/split
   identities, phase, and the actual operation outcome. Reader error, worker
   failure, panic, cancellation, or unknown completion must not become a
   successful read by omission.

Before editing any candidate source path, the corresponding owner must claim
the precise paths and confirm the API boundary. This document does not change
worker frames, frozen public contracts, or task ownership.

## Acceptance plan for the owning integration

Use a small catalog-derived fixture whose split and matrix identities are
recorded. Demonstrate all of the following at the exact source commit:

* The trusted producer rejects duplicate dataset/ordinal identities and
  produces disjoint nonempty FIT/HOLDOUT (and optional TEST) rosters while
  respecting its configured group/time policy.
* A real discovery train records every landmark, rung, preprocessing and
  calibration read. Include instrumented reader-call counts/row identities
  proving the audit wrapper encloses the actual operation, not just fold
  construction.
* Rung 0 uses two folds; later rung and calibration caps/fold counts follow
  the plan. For each outer fold, inner calibration uses only outer training
  rows. An impossible group/time/gap request is refused without random fallback.
* Malicious HOLDOUT/TEST or duplicate-ordinal requests are refused before the
  reader is entered. Verify the reader counter stays zero for refused access.
* Finalization records exactly FIT fit, HOLDOUT score, optional FIT+HOLDOUT
  refit, and optional TEST score; tests assert call rows and outcomes. An
  error/interruption leaves attempted access and unknown outcome explicit and
  is not automatically replayed against protected rows.
* Receipt binds source SHA, dataset/matrix/split identity, phase touches and
  execution outcomes. Report collected/executed/skipped tests and owning
  package gates from the recorded exact head.

Component tests for `FoldPlan`/`RowAudit`, generated fold JSON, or metadata
alone do not satisfy this acceptance. This handoff reports no end-to-end
catalog train, no successful production access audit, and no whole-loop S3
qualification. No sealed or holdout payload was accessed for this audit.
