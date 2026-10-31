# Prepared run finalization handoff (DSC-09-F01)

Status: **owner handoff proposal; not an implemented or qualified run path**. Source evidence is pinned to `origin/main` `96e13174faba7c78f6eac809884192d1594e1c64` (2026-10-31). This document names the current seams, the minimum internal handoff, and the exact ownership needed before source integration. It does not change the frozen API, worker protocol, model format, split policy, or search policy.

## Current code seams and signatures

| Responsibility | Current path and signature | What it does today |
| --- | --- | --- |
| Search component | `crates/jope-service/src/discover/loop.rs`: `pub fn run_m1(path: &Path, executable: &Path) -> anyhow::Result<serde_json::Value>` | Opens a Store, creates a bounded `ProcessExecutor`, and runs the hidden M1 FIT-only search probe. It returns `stage = "search_finished"`; its own docs explicitly say public run/start/attach belongs to BAT-02/CLT-02. It does not call finalization or accept prepared FIT/HOLDOUT/TEST rosters. |
| Finalization component | `crates/jope-service/src/discover/finalize.rs`: `Finalizer::new(store: Arc<Store>) -> Finalizer`; `journal(&self) -> &Arc<EventLog>`; `finalize(&self, request: &FinalizationRequest, executor: &impl Executor) -> anyhow::Result<FinalizationReceipt>` | Validates the stored completed-search plan/champion, claims once before protected reads, performs finalization reads/fits/scores with the supplied executor, writes the retained model and receipt, and commits `RunFinished`/artifact/snapshot through its journal. On current main, all `Finalizer::new` / `FinalizationRequest` call sites are inside this module's tests. |
| Partition argument | `crates/jope-service/src/discover/finalize.rs`: `Partition { path: PathBuf, sha256: String, rows: Vec<AuditRowId> }` and `FinalizationRequest { run_id, fit, holdout, test, refit, model_root, scratch_root, max_input_bytes, max_model_bytes }` | This is the finalizer input shape, not proof that the caller obtained the files and row rosters from a trusted prepared-partition source. `AuditRowId` is dataset-local (`dataset: DatasetId`, `row: u64`). |
| Journal listeners | `crates/jope-service/src/http/sse.rs`: `event_routes(log: Arc<EventLog>, auth: LocalAuth) -> Router` | Serves event and long-poll status routes from the supplied log. The finalizer exposes its log with `journal()`, which can be shared with these listeners. |
| Engine/router | `crates/jope-service/src/http/router.rs`: `router(auth: LocalAuth) -> Router`; `crates/jope-service/src/engine.rs`: `InProcessEngine::open(home: impl AsRef<Path>) -> Result<InProcessEngine, EngineError>` | Registered non-system routes currently return `CapabilityUnavailable`. `InProcessEngine` owns a Store and home lock; it does not compose run handlers, an EventLog, or a finalizer. |
| Public CLI run adapter | `crates/jope-appliance/src/cmd/run.rs`: `dispatch(args: RunArgs) -> anyhow::Result<()>` | Uses the client/transport path for ordinary runs. Its hidden `--m1-plan` path invokes the FIT-only `run_m1` probe; the in-process production path currently refuses with `LocalHandlers`. |
| Public run listing/artifact retrieval | `crates/jope-appliance/src/cmd/runs.rs`: `run(command: RunsCommand) -> anyhow::Result<()>`; private `execute<T: Transport>(transport: &T, command: RunsCommand) -> anyhow::Result<Value>` | Handles list, delete, and feedback export only. The API registry already declares `GET /v1/artifacts/{id}` (`artifact_download`) and `GET /v1/runs/{id}` status, but the service router does not implement those registered operations. `RunSnapshot.best.model_artifact` and the existing `RunFinishedBody` can carry the completed artifact and score projection without a schema edit. |

## Minimum integration contract

The following is a proposed internal boundary for the owning adapter teams to accept or revise. `CompletedSearch`, `TrustedPartitionPreparer`, `SearchAccessReceiptRef`, and `TrustedPreparedFinalization` are design notation only; they do not exist on the pinned main and this document does not grant their source scope. The trusted preparer is the only constructor of the opaque prepared value.

```rust
pub struct SearchAccessReceiptRef {
    pub run_id: RunId,
    pub completed_plan_sha256: String,
    pub matrix_sha256: String,
    pub audit_artifact: ArtifactId,
    pub audit_sha256: String,
}

/// Constructible only by the trusted prepared-partition adapter after validating
/// source identity, dataset-local row ordinals, split policy, hashes, disjointness,
/// and the immutable completed plan. Not deserializable from a public request.
pub struct TrustedPreparedFinalization {
    request: FinalizationRequest,
    search_access: SearchAccessReceiptRef,
}

pub trait TrustedPartitionPreparer {
    fn prepare_finalization(
        &self,
        completed: &CompletedSearch,
    ) -> anyhow::Result<TrustedPreparedFinalization>;
}

pub fn finalize_completed_search(
    finalizer: &Finalizer,
    prepared: &TrustedPreparedFinalization,
    executor: &impl Executor,
) -> anyhow::Result<FinalizationReceipt>;
```

The adapter must verify `prepared.request.run_id == prepared.search_access.run_id`, and bind both proofs to the stored completed plan and FIT matrix hash before calling the existing `Finalizer::finalize`. The *same bounded `ProcessExecutor` instance* used by the selected search is passed through as `executor`; it is not reconstructed with a larger or different worker limit. FIT/HOLDOUT/optional TEST must be trusted prepared files whose row rosters preserve original dataset IDs and source ordinals. The preparer, not the finalizer, owns group/time split selection and validation.

Persist the search-access proof separately from the finalizer's `RowAudit`. Link them with run ID, completed plan digest, FIT matrix digest, and audit artifact/digest. The finalizer's row log covers finalization phases only; it cannot stand in for FIT-only search access evidence. A failed or unknown finalization claim remains terminal for protected-row rereads and must surface as an explicit recovery refusal.

Construct one finalizer per Store, then give its journal to listener composition:

```rust
let finalizer = Finalizer::new(store);
let event_log = Arc::clone(finalizer.journal());
let listeners = event_routes(event_log, auth);
// After the selected search has durably completed and trusted preparation succeeds:
let receipt = finalize_completed_search(&finalizer, &prepared, &same_bounded_executor)?;
```

The actual owner must decide where this sequence belongs. `run_m1` is not suitable as-is: it is a hidden FIT-only probe with no trusted HOLDOUT/TEST preparation, and it returns a search summary rather than a public completed run. `cmd/run.rs` is a client adapter, not a service-owned finalization callsite. The production service router must publish the committed `RunSnapshot`/`RunFinished` projection, keep private paths/rosters/audit out of public responses, and connect the existing artifact-download operation to the retained model CAS object. `jopedime predict <completed-run-id>` must resolve the same `model_artifact` recorded in the committed snapshot and download/verify that exact retained model before prediction; no alternate search or refit is allowed during retrieval.

## Required source-owner handoff

No exact existing prepared-partition producer or service-owned completion callsite is present on the pinned main. Before implementation, the integration owners must claim and name these exact paths:

1. The trusted partition/roster producer that validates prepared FIT/HOLDOUT/optional TEST files and emits the immutable-plan/search-access link.
2. The service-owned completed-search callsite that retains and reuses the search `ProcessExecutor` and invokes finalization once.
3. The engine/router and public run/artifact retrieval handlers that expose only committed scores and artifact IDs.

The current closest files (`discover/loop.rs`, `engine.rs`, `http/router.rs`, `http/sse.rs`, `appliance/src/cmd/run.rs`, `appliance/src/cmd/runs.rs`) are identified for owner review; this document does not assign or take their source paths. Source changes require explicit path claims by their owners and must not introduce a new frozen public API or split policy.

## Acceptance evidence for the eventual adapter PR

Use a TRAIN-derived fixture only. Demonstrate FIT-only search; FIT fit and HOLDOUT score; both `Refit::Fit` and `Refit::FitHoldout`; optional TEST scored exactly once with the selected retained model; committed completion/artifact lookup; `jopedime predict` loading that exact object; completed replay without protected rereads; failed/unknown claim refusal; and terminal run handling. Bind the search-access and finalization receipts to the same run/plan/matrix identities. Keep fixture inputs synthetic or TRAIN-derived; no external protected dataset is needed. Passing finalizer unit tests alone is not this adapter acceptance.
