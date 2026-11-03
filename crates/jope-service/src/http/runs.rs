//! Engine-owned, bounded CPU runs. No transport owns a second discovery engine.
use super::auth::problem;
use crate::{
    config::Config,
    dataset::{jmat, register::RegisteredDataset},
    discover::{
        audit::AuditRowId,
        effort::{BudgetOverrides, Effort as SearchEffort},
        finalize::{FinalizationRequest, Finalizer, Partition},
        r#loop::{Discovery, NativeSpace, Plan as SearchPlan, ProcessExecutor, SearchSpace},
        splits::random_split,
    },
    engine::{EngineActivity, EngineLock},
    events::{EventLog, RunUpdate},
    governor::{Governor, Requirements},
    idempotency::{CommandOutcome, RequestIdentity},
    run::service::{BackendExecutor, BackendPlan, BackendSpace},
    store::{Record, RecordKey, Store},
    telemetry::{Sensors, SystemSensors},
};
use axum::{
    body::Body,
    extract::{Extension, FromRequest, Query, Request},
    response::{IntoResponse, Response},
    Json,
};
use jope_api::{
    decimal::DecimalU64,
    error::{ErrorCode, Problem},
    *,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    num::NonZeroU32,
    path::PathBuf,
    sync::{mpsc, Arc, Mutex},
};

const WORKER_RAM: u64 = 1024 * 1024 * 1024;
const MODEL_LIMIT: u64 = 16 * 1024 * 1024;
const Q1_WALL_BUDGET: std::time::Duration = std::time::Duration::from_secs(600);
const Q1_ATTEMPT_WALL_CAP_MS: u64 = 600_000;
type Result<T> = std::result::Result<T, Box<Problem>>;

struct Q1ReceiptCompletion {
    attempt_events: Vec<EventBody>,
    elapsed: std::time::Duration,
    terminal_failure: Option<String>,
}

fn q1_remaining_wall_ms(elapsed: std::time::Duration) -> Option<u64> {
    let remaining = Q1_WALL_BUDGET.saturating_sub(elapsed);
    let remaining_ms = remaining
        .as_millis()
        .min(u128::from(Q1_ATTEMPT_WALL_CAP_MS)) as u64;
    (remaining_ms > 0).then_some(remaining_ms)
}

#[cfg(all(feature = "failpoints", debug_assertions))]
#[derive(Clone, Debug)]
struct Pub06InternalValues {
    dsl: String,
    operator: String,
    lineage: String,
    round: String,
    fail_after_seed: bool,
}

#[cfg(all(feature = "failpoints", debug_assertions))]
type Pub06Injection = Option<Pub06InternalValues>;
#[cfg(not(all(feature = "failpoints", debug_assertions)))]
type Pub06Injection = ();

pub(crate) struct RunService {
    store: Arc<Store>,
    pub(crate) journal: Arc<EventLog>,
    config: Config,
    home: PathBuf,
    executable: PathBuf,
    binary: String,
    activity: Arc<EngineActivity>,
    admission: Mutex<()>,
    // Worker jobs retain store ownership and the OS lease after a client drops.
    _lease: Arc<EngineLock>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Continuation {
    search: SearchPlan,
    finalization: FinalizationRequest,
    // Store the request and resolved placement, never the live graph or weights.
    backend: ContinuationBackend,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ContinuationBackend {
    spec: RunSpec,
    placement: Placement,
    #[serde(default)]
    landmarks: Option<crate::run::landmarks::FullFitLandmarks>,
}

impl ContinuationBackend {
    fn restore(
        &self,
        store: &Store,
        plan: &SearchPlan,
        roots: &[crate::store::BlobRef],
        admitted: Option<&crate::run::learned::AdmittedRanker>,
    ) -> Result<BackendPlan> {
        let mut backend = BackendPlan::resolve(&self.spec, admitted, plan.planned_workers)
            .map_err(|error| {
                issue(
                    crate::run::service::error_code(&error),
                    "The original run backend is unavailable.",
                )
            })?;
        if backend.bundle_digest != plan.bundle_digest || backend.placement != self.placement {
            return Err(unavailable(
                "Continuation requires the original backend placement and bundle.",
            ));
        }
        let fit = roots
            .iter()
            .find(|blob| {
                blob.sha256
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect::<String>()
                    == plan.matrix_sha256
            })
            .ok_or_else(|| unavailable("Original FIT blob is unavailable."))?;
        if store.verified_blob_path(fit).map_err(io)? != plan.matrix_path {
            return Err(unavailable("Original FIT path changed."));
        }
        backend
            .bind_fit(&store.read_blob(fit).map_err(io)?, plan)
            .map_err(io)?;
        if backend.full_fit_landmarks != self.landmarks {
            return Err(unavailable("Original FIT landmark evidence changed."));
        }
        Ok(backend)
    }
}

fn run_language_filter(
    languages: &[Language],
) -> Result<feat_math::evolve::api::capability::LanguageFilter> {
    use feat_math::evolve::api::capability::{self, Language as ExportLanguage};
    if languages.is_empty() {
        return Err(unavailable(
            "At least one supported export language is required.",
        ));
    }
    // Evaluation is always native Rust; requested portable exports add constraints.
    let mut requested = vec![ExportLanguage::Rust];
    for language in languages {
        requested.push(match language {
            Language::Rust => ExportLanguage::Rust,
            Language::Python => ExportLanguage::Python,
            Language::Javascript => ExportLanguage::Javascript,
            _ => {
                return Err(unavailable(
                    "This run backend supports Rust, Python, and JavaScript export languages.",
                ))
            }
        });
    }
    capability::allowed_ops(&requested).map_err(|_| {
        unavailable("Requested export languages have no supported native pipeline intersection.")
    })
}

fn continuation_key(run: &jope_api::ids::RunId) -> RecordKey {
    RecordKey::Command {
        principal: "public-run-continuation.v1".into(),
        idempotency_key: run.to_string(),
    }
}

struct Prepared {
    public: Plan,
    search: SearchPlan,
    fit: Vec<u8>,
    holdout: Vec<u8>,
    fit_rows: Vec<AuditRowId>,
    holdout_rows: Vec<AuditRowId>,
    backend: BackendPlan,
}

// Both Q1 arms use the same pinned folds and bounded numerical workers. Only
// admitted rung-0 attempts consume the policy budget; promotion/refit results
// remain in Discovery's own journal and are not invented as extra admissions.
struct Q1PinnedExecutor<E> {
    inner: E,
    folds: Value,
    started_at: std::time::Instant,
    attempts: Mutex<Vec<(crate::worker::Attempt, EvaluationAttempt)>>,
}
impl<E: crate::discover::r#loop::Executor> crate::discover::r#loop::Executor
    for Q1PinnedExecutor<E>
{
    fn evaluate(&self, request: &crate::worker::Request) -> crate::worker::Response {
        use crate::worker::{Response, WorkerFailure};
        let charged = request.attempt.slot < 65_536;
        let mut pinned = request.clone();
        pinned.fold_spec = self.folds.clone();
        pinned.fold_spec["version"] = json!(if charged { 2 } else { 1 });
        pinned.limits.wall_ms = pinned
            .limits
            .wall_ms
            .min(q1_remaining_wall_ms(self.started_at.elapsed()).unwrap_or(0));
        let mut result = if pinned.limits.wall_ms == 0 {
            Response::failed(pinned.attempt.clone(), WorkerFailure::Timeout, 0)
        } else {
            self.inner.evaluate(&pinned)
        };
        if result.attempt != pinned.attempt
            || !result.valid()
            || (result.failure.is_none()
                && (result.scores.len() != 3 || result.fold_scores.len() != 3))
        {
            result = Response::failed(pinned.attempt.clone(), WorkerFailure::Protocol, 0);
        }
        if charged {
            let failure = result.failure.map(q1_failure);
            let completed = failure.is_none();
            let mut attempts = self.attempts.lock().expect("Q1 attempt inventory poisoned");
            assert!(
                !attempts.iter().any(|(key, _)| key == &request.attempt),
                "Q1 cannot charge the same attempt twice"
            );
            let number = attempts.len() as u64;
            attempts.push((
                request.attempt.clone(),
                EvaluationAttempt {
                    attempt: number,
                    outcome: if completed {
                        EvaluationAttemptOutcome::Completed
                    } else {
                        EvaluationAttemptOutcome::Failed
                    },
                    fold_scores: if completed {
                        result.fold_scores.clone()
                    } else {
                        vec![]
                    },
                    loss: completed.then(|| {
                        result
                            .fold_scores
                            .iter()
                            .map(|score| score / result.fold_scores.len() as f64)
                            .sum::<f64>()
                    }),
                    program_sha256: completed.then(|| hash(request.dsl.as_bytes())),
                    failure,
                },
            ));
        }
        result
    }
    fn fitted_model(&self, request: &crate::worker::Request) -> anyhow::Result<Vec<u8>> {
        let mut pinned = request.clone();
        pinned.fold_spec = self.folds.clone();
        pinned.fold_spec["version"] = json!(1);
        pinned.limits.wall_ms = pinned
            .limits
            .wall_ms
            .min(q1_remaining_wall_ms(self.started_at.elapsed()).unwrap_or(0));
        anyhow::ensure!(
            pinned.limits.wall_ms > 0,
            "Q1 refit reached the policy deadline"
        );
        self.inner.fitted_model(&pinned)
    }
    fn revoke(&self, attempt: &crate::worker::Attempt) {
        self.inner.revoke(attempt);
    }
}

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Q1Request {
    schema: String,
    mode: String,
    source_commit: String,
    plan_sha256: String,
    pins: std::collections::BTreeMap<String, String>,
    metric: String,
    dataset: Q1DatasetIdentity,
    guidance: String,
    seed: u64,
    budget: u64,
    request_sha256: String,
}

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Q1DatasetIdentity {
    id: String,
    task: String,
    dataset_sha256: String,
    family: String,
    family_authoritative: bool,
    rows: u64,
    features: u64,
    cut_sha256: String,
    holdout_group: String,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Q1ProtocolManifest {
    schema: String,
    protocol_sha256: String,
    source_commit: String,
    plan_sha256: String,
    pins: std::collections::BTreeMap<String, String>,
    datasets: std::collections::BTreeMap<String, Q1DatasetBinding>,
}

mod q1_fit;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Q1DatasetBinding {
    id: String,
    task: String,
    dataset_sha256: String,
    family: String,
    family_authoritative: bool,
    rows: u64,
    features: u64,
    cut_sha256: String,
    holdout_group: String,
    catalog_version: String,
    catalog_record_sha256: String,
    dataset_binding_sha256: String,
    target: String,
    feature_columns: Vec<String>,
    train_rows: u64,
    train_blob_sha256: String,
    train_blob_bytes: u64,
    folds: Vec<Q1Fold>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Q1Fold {
    repeat: u8,
    train: Vec<usize>,
    validation: Vec<usize>,
}

/// Establish a bounded native baseline before considering expensive mutations.
/// The existing research/native starter order remains unchanged.
struct CpuRunSpace {
    allowed: feat_math::evolve::api::capability::LanguageFilter,
    #[cfg(all(feature = "failpoints", debug_assertions))]
    pub06_values: Option<Pub06InternalValues>,
}
impl SearchSpace for CpuRunSpace {
    fn canonical(&self, dsl: &str) -> anyhow::Result<String> {
        let canonical = NativeSpace.canonical(dsl)?;
        anyhow::ensure!(
            self.allowed
                .accepts_seed(&feat_math::GenomeIr::parse(&canonical)?),
            "pipeline is excluded by requested export languages"
        );
        Ok(canonical)
    }
    fn seeds(&self) -> anyhow::Result<Vec<String>> {
        #[cfg(all(feature = "failpoints", debug_assertions))]
        if let Some(values) = &self.pub06_values {
            // The test-only values live on the real SearchSpace used by the worker.
            // Touch each field during seed generation without changing the valid program.
            std::hint::black_box((
                values.dsl.as_str(),
                values.operator.as_str(),
                values.lineage.as_str(),
                values.round.as_str(),
            ));
        }
        Ok(vec![self.canonical("(pipe X (standardize) (ridge))")?])
    }
    fn propose(
        &self,
        parents: &[String],
        count: usize,
        rng: &mut rand_chacha::ChaCha12Rng,
    ) -> anyhow::Result<Vec<String>> {
        #[cfg(all(feature = "failpoints", debug_assertions))]
        if self
            .pub06_values
            .as_ref()
            .is_some_and(|values| values.fail_after_seed)
        {
            anyhow::bail!("PUB-06 test worker failure after internal-value injection");
        }
        Ok(NativeSpace
            .propose(parents, count, rng)?
            .into_iter()
            .filter(|dsl| self.canonical(dsl).is_ok())
            .collect())
    }
}

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn parse_digest(value: &str) -> Option<[u8; 32]> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return None;
    }
    let mut bytes = [0; 32];
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16).ok()?;
    }
    Some(bytes)
}

fn canonical_sha256(value: &Value) -> Result<String> {
    serde_json::to_vec(value)
        .map(|bytes| hash(&bytes))
        .map_err(io)
}

fn q1_cut_sha256(binding: &Q1DatasetBinding) -> Result<String> {
    // serde_json::Map serializes keys in sorted order; all array orders are semantic.
    canonical_sha256(&json!({
        "dataset_sha256": binding.dataset_sha256,
        "feature_columns": binding.feature_columns,
        "folds": binding.folds,
        "schema": "jope.q1-cut/v1",
        "target": binding.target,
        "train_blob_sha256": binding.train_blob_sha256,
        "train_rows": binding.train_rows,
    }))
}

fn validate_q1_request(request: &Q1Request, raw: &Value) -> Result<()> {
    if request.schema != "jope.q1-request/v1"
        || request.mode != "qualification"
        || request.metric != "rmse"
        || request.guidance != "heuristic" && request.guidance != "learned"
        || request.budget != 24
        || request.seed > 2
        || !valid_hex(&request.source_commit, 40)
        || !valid_hex(&request.plan_sha256, 64)
        || !valid_hex(&request.request_sha256, 64)
    {
        return Err(invalid(
            "Q1 request schema, policy, metric, identity or budget is unsupported.",
        ));
    }
    let expected_pins = [
        "bundle_sha256",
        "dope_sha256",
        "evaluator_sha256",
        "vocabulary_sha256",
        "protocol_sha256",
        "training_family_manifest_sha256",
    ];
    if request.pins.len() != expected_pins.len()
        || expected_pins.iter().any(|key| {
            request
                .pins
                .get(*key)
                .is_none_or(|value| !valid_hex(value, 64))
        })
    {
        return Err(invalid("Q1 request pins are incomplete or malformed."));
    }
    if !valid_hex(&request.dataset.id, 16)
        || !valid_hex(&request.dataset.dataset_sha256, 64)
        || !request
            .dataset
            .dataset_sha256
            .starts_with(&request.dataset.id)
        || !valid_hex(&request.dataset.cut_sha256, 64)
        || !valid_hex(&request.dataset.family, request.dataset.family.len())
        || ![40, 64].contains(&request.dataset.family.len())
        || !request.dataset.family_authoritative
        || request.dataset.task != "regression"
        || !matches!(
            request.dataset.holdout_group.as_str(),
            "sealed_meta_test" | "development_fit"
        )
        || request.dataset.rows == 0
        || request.dataset.features == 0
    {
        return Err(invalid(
            "Q1 dataset identity or holdout metadata is unsupported.",
        ));
    }
    let mut canonical_request = raw.clone();
    let object = canonical_request
        .as_object_mut()
        .ok_or_else(|| invalid("Q1 request must be a JSON object."))?;
    object.remove("request_sha256");
    if canonical_sha256(&canonical_request)? != request.request_sha256 {
        return Err(invalid("Q1 canonical request digest mismatch."));
    }
    Ok(())
}

fn valid_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn load_q1_manifest(home: &std::path::Path, protocol_sha: &str) -> Result<Q1ProtocolManifest> {
    let q1_dir = home.join("q1");
    let protocols_dir = q1_dir.join("protocols");
    for directory in [&q1_dir, &protocols_dir] {
        let metadata = std::fs::symlink_metadata(directory)
            .map_err(|_| unavailable("Pinned Q1 protocol manifest is not installed."))?;
        if !metadata.file_type().is_dir() {
            return Err(unavailable(
                "Pinned Q1 protocol path is not a real directory.",
            ));
        }
    }
    let path = protocols_dir.join(format!("{protocol_sha}.json"));
    let metadata = std::fs::symlink_metadata(&path)
        .map_err(|_| unavailable("Pinned Q1 protocol manifest is not installed."))?;
    if !metadata.file_type().is_file() || metadata.len() > 8 * 1024 * 1024 {
        return Err(unavailable(
            "Pinned Q1 protocol manifest is not a bounded regular file.",
        ));
    }
    let bytes = std::fs::read(path).map_err(io)?;
    serde_json::from_slice(&bytes).map_err(|_| invalid("Pinned Q1 protocol manifest is malformed."))
}

fn validate_q1_binding(binding: &Q1DatasetBinding, dataset: &Q1DatasetIdentity) -> Result<()> {
    let mut value = serde_json::to_value(binding).map_err(io)?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| invalid("Q1 dataset binding is malformed."))?;
    object.remove("dataset_binding_sha256");
    if binding.id != dataset.id
        || binding.task != dataset.task
        || binding.dataset_sha256 != dataset.dataset_sha256
        || binding.family != dataset.family
        || binding.family_authoritative != dataset.family_authoritative
        || binding.rows != dataset.rows
        || binding.features != dataset.features
        || binding.cut_sha256 != dataset.cut_sha256
        || binding.holdout_group != dataset.holdout_group
        || !binding.catalog_version.starts_with('v')
        || binding.catalog_version[1..].is_empty()
        || !binding.catalog_version[1..]
            .bytes()
            .all(|byte| byte.is_ascii_digit())
        || !valid_hex(&binding.catalog_record_sha256, 64)
        || !valid_hex(&binding.train_blob_sha256, 64)
        || !valid_hex(&binding.dataset_binding_sha256, 64)
        || hash(&serde_json::to_vec(&value).map_err(io)?) != binding.dataset_binding_sha256
        || q1_cut_sha256(binding)? != binding.cut_sha256
        || binding.target.is_empty()
        || binding.feature_columns.len() as u64 != dataset.features
        || binding.feature_columns.iter().any(String::is_empty)
        || binding
            .feature_columns
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != binding.feature_columns.len()
        || binding.feature_columns.contains(&binding.target)
        || binding.train_rows < 2
        || usize::try_from(binding.train_rows).is_err()
        || binding.folds.len() < 2
        || binding.folds.iter().any(|fold| {
            fold.train.is_empty()
                || fold.validation.is_empty()
                || fold
                    .train
                    .iter()
                    .any(|row| *row >= binding.train_rows as usize)
                || fold
                    .validation
                    .iter()
                    .any(|row| *row >= binding.train_rows as usize)
                || fold
                    .train
                    .iter()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
                    != fold.train.len()
                || fold
                    .validation
                    .iter()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len()
                    != fold.validation.len()
                || fold.train.iter().any(|row| fold.validation.contains(row))
        })
    {
        return Err(invalid(
            "Q1 dataset binding does not match its catalog/request identities.",
        ));
    }
    Ok(())
}

fn validate_q1_folds(binding: &Q1DatasetBinding) -> Result<()> {
    use std::collections::BTreeSet;
    let mut validation_seen = BTreeSet::new();
    let valid = binding.folds.len() == 3
        && binding.train_rows >= 6
        && binding.folds.iter().all(|fold| {
            let train: BTreeSet<_> = fold.train.iter().copied().collect();
            fold.repeat == 0
                && train.len() >= 2
                && train.len() == fold.train.len()
                && train.iter().all(|row| *row < binding.train_rows as usize)
                && fold.validation.len() >= 2
                && train.len() + fold.validation.len() == binding.train_rows as usize
                && fold.validation.iter().all(|row| {
                    *row < binding.train_rows as usize
                        && !train.contains(row)
                        && validation_seen.insert(*row)
                })
        })
        && validation_seen.len() == binding.train_rows as usize;
    if !valid {
        return Err(invalid(
            "Q1 requires three disjoint full-FIT validation partitions.",
        ));
    }
    Ok(())
}

#[cfg(test)]
fn q1_run_spec(request: &Q1Request, binding: &Q1DatasetBinding) -> Result<RunSpec> {
    let guidance = match request.guidance.as_str() {
        "heuristic" => Guidance::Heuristic,
        "learned" => Guidance::Learned,
        _ => return Err(invalid("Unsupported Q1 guidance policy.")),
    };
    Ok(RunSpec {
        dataset: format!("ds_{}", &request.dataset.dataset_sha256[..24])
            .parse()
            .map_err(|_| invalid("Q1 dataset identity cannot bind to a run."))?,
        target: binding.target.clone(),
        task: Task::Regression,
        ignore_columns: vec![],
        test_dataset: None,
        split: SplitSpec {
            kind: SplitKind::Random,
            holdout_fraction: 0.2,
            group_column: None,
            time_column: None,
        },
        objective: ObjectiveSpec::Name("rmse".into()),
        effort: jope_api::Effort::Fast,
        budget: jope_api::Budget {
            evaluations: Some(DecimalU64(request.budget)),
            wall_seconds: Some(DecimalU64(Q1_WALL_BUDGET.as_secs())),
        },
        landmarks: if guidance == Guidance::Learned {
            LandmarkMode::Full
        } else {
            LandmarkMode::None
        },
        guidance,
        languages: vec![Language::Rust],
        require_all_languages: false,
        device: "cpu".into(),
        seed: DecimalU64(request.seed),
        dry_run: false,
        plan_digest: None,
    })
}

#[cfg(test)]
fn q1_fold_spec(binding: &Q1DatasetBinding, matrix_sha256: &str, learned: bool) -> Value {
    if learned {
        json!({
            "version": 2,
            "matrix_sha256": matrix_sha256,
            "target": binding.target,
            "features": binding.feature_columns,
            "folds": binding.folds
        })
    } else {
        json!({
            "version": 1,
            "matrix_sha256": matrix_sha256,
            "target": binding.target,
            "features": binding.feature_columns,
            "folds": binding.folds,
            "refit_output": null,
            "refit_max_bytes": null
        })
    }
}

fn q1_command_key(request_sha256: &str) -> RecordKey {
    RecordKey::Command {
        principal: "q1-evaluation".into(),
        idempotency_key: request_sha256.into(),
    }
}

fn q1_replay_or_unknown(data: Value) -> Result<Option<Vec<u8>>> {
    if data["status"] == "completed" {
        return data["receipt_bytes"]
            .as_str()
            .map(|bytes| Some(bytes.as_bytes().to_vec()))
            .ok_or_else(|| io("Completed Q1 receipt bytes are missing."));
    }
    Err(unavailable(
        "A prior Q1 request has an unknown or incomplete outcome; it was not replayed.",
    ))
}

fn q1_running_snapshot(run_id: jope_api::ids::RunId, budget: u64) -> RunSnapshot {
    RunSnapshot {
        run_id,
        revision: DecimalU64(1),
        state: RunState::Running,
        phase: RunPhase::Encoding,
        pending: Some("pinned_training_evaluation".into()),
        effort: Effort::High,
        budget_used: BudgetUsed {
            evaluations: DecimalU64(0),
            wall_seconds: "0".into(),
        },
        budget_total: BudgetTotal {
            evaluations: DecimalU64(budget),
            wall_seconds: DecimalU64(600),
        },
        best: None,
        event_watermark: DecimalU64(0),
        warnings: vec![],
        completion_reason: None,
        evaluation_receipt: None,
    }
}

fn q1_failure(failure: crate::worker::WorkerFailure) -> jope_api::EvaluationFailure {
    let (code, reason) = match failure {
        crate::worker::WorkerFailure::Timeout => ("timeout", "attempt reached its time limit"),
        crate::worker::WorkerFailure::MemoryLimit => {
            ("memory_limit", "attempt exceeded its memory limit")
        }
        crate::worker::WorkerFailure::NonFinite => {
            ("non_finite", "attempt produced undefined fitness")
        }
        crate::worker::WorkerFailure::Unavailable => {
            ("backend_unavailable", "native evaluator unavailable")
        }
        crate::worker::WorkerFailure::InvalidRequest => {
            ("invalid_request", "native evaluator rejected the request")
        }
        other => (
            "evaluation_failed",
            match other {
                crate::worker::WorkerFailure::Panicked => "native evaluator panicked",
                crate::worker::WorkerFailure::ProcessExited => "worker process exited",
                crate::worker::WorkerFailure::Protocol => "worker protocol failed",
                crate::worker::WorkerFailure::StaleAttempt => "attempt fence was stale",
                _ => "attempt failed",
            },
        ),
    };
    jope_api::EvaluationFailure {
        code: code.into(),
        reason: reason.into(),
    }
}

#[cfg(test)]
fn q1_learned_seed<'a>(seeds: &'a [String], attempt: u64, measured: &[(String, f64)]) -> &'a str {
    let start = attempt as usize % seeds.len();
    seeds[start..]
        .iter()
        .chain(&seeds[..start])
        .find(|seed| !measured.iter().any(|(dsl, _)| dsl == *seed))
        .or_else(|| Some(&seeds[start]))
        .expect("nonempty seed inventory")
}
fn issue(code: ErrorCode, detail: impl Into<String>) -> Box<Problem> {
    Box::new(Problem { r#type: "about:blank".into(), title: code.as_str().into(),
        status: code.http_status().unwrap_or(500), detail: detail.into(), instance: None,
        code, retriable: false, remedy: "Use a supported bounded CPU regression RunSpec; inspect the local engine diagnostics for IO failures.".into(), field: None })
}
fn invalid(detail: &str) -> Box<Problem> {
    issue(ErrorCode::InvalidInput, detail)
}
fn unavailable(detail: &str) -> Box<Problem> {
    issue(ErrorCode::CapabilityUnavailable, detail)
}
fn io(error: impl std::fmt::Display) -> Box<Problem> {
    issue(ErrorCode::IoFailure, error.to_string())
}
fn response(error: &Problem) -> Response {
    problem(
        error.code,
        &error.detail,
        &error.remedy,
        error.field.as_deref(),
    )
}

mod q1_service;
mod routes;
mod run_service;
pub(super) use routes::{control, list, q1_evaluate, start};

#[cfg(test)]
mod tests;
