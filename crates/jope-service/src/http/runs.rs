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
            _ => {
                return Err(unavailable(
                    "This run backend supports Rust and Python export languages.",
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
                && (result.scores.len() != 3 || result.fold_scores.len() < 2))
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

#[cfg(test)]
fn validate_q1_learned_folds(binding: &Q1DatasetBinding) -> Result<()> {
    if binding.folds.len() != 2
        || binding.folds.iter().any(|fold| fold.repeat != 0)
        || binding
            .folds
            .iter()
            .any(|fold| fold.train.len() + fold.validation.len() != binding.train_rows as usize)
        || binding.folds[0].train != binding.folds[1].validation
        || binding.folds[0].validation != binding.folds[1].train
    {
        return Err(invalid(
            "Learned Q1 requires the pinned complementary two-fold FIT recipe.",
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

impl RunService {
    pub(crate) fn config(&self) -> &Config {
        &self.config
    }
    pub(crate) fn new(
        store: Arc<Store>,
        config: Config,
        home: PathBuf,
        executable: PathBuf,
        activity: Arc<EngineActivity>,
        lease: Arc<EngineLock>,
    ) -> anyhow::Result<Self> {
        let binary = format!(
            "{}:{}",
            env!("CARGO_PKG_VERSION"),
            hash(&std::fs::read(&executable)?)
        );
        Ok(Self {
            journal: Arc::new(EventLog::new(Arc::clone(&store))),
            store,
            config,
            home,
            executable,
            binary,
            activity,
            admission: Mutex::new(()),
            _lease: lease,
        })
    }

    fn q1_evaluate(&self, raw: Value) -> Result<Vec<u8>> {
        self.q1_evaluate_using(raw, || {
            ProcessExecutor::new(&self.executable, 1, &self.home.join("worker-scratch"))
        })
    }

    fn q1_evaluate_using<E: crate::discover::r#loop::Executor>(
        &self,
        raw: Value,
        make_executor: impl FnOnce() -> anyhow::Result<E>,
    ) -> Result<Vec<u8>> {
        let started_at = std::time::Instant::now();
        let request: Q1Request = serde_json::from_value(raw.clone())
            .map_err(|_| invalid("Expected a closed jope.q1-request/v1 object."))?;
        validate_q1_request(&request, &raw)?;
        let admission = self
            .admission
            .lock()
            .map_err(|_| io("Run admission lock is poisoned."))?;
        if let Some(record) = self
            .store
            .get(&q1_command_key(&request.request_sha256))
            .map_err(io)?
        {
            return q1_replay_or_unknown(record.data)?
                .ok_or_else(|| io("Q1 replay bytes missing."));
        }
        if self.activity.active_runs() != 0 {
            return Err(unavailable("The bounded evaluation backend is busy."));
        }
        // This spec only selects the trusted backend. No public-run split is
        // performed: the Q1 matrix and every evaluation fold come from the
        // independently pinned protocol manifest below.
        let backend_spec = RunSpec {
            dataset: format!("ds_{}", &request.dataset.dataset_sha256[..24])
                .parse()
                .map_err(io)?,
            target: String::new(),
            task: Task::Regression,
            ignore_columns: vec![],
            test_dataset: None,
            split: SplitSpec {
                kind: SplitKind::Random,
                holdout_fraction: 0.5,
                group_column: None,
                time_column: None,
            },
            objective: ObjectiveSpec::Name("rmse".into()),
            effort: Effort::High,
            budget: Budget {
                evaluations: Some(DecimalU64(request.budget)),
                wall_seconds: Some(DecimalU64(600)),
            },
            landmarks: LandmarkMode::Full,
            guidance: if request.guidance == "learned" {
                Guidance::Learned
            } else {
                Guidance::Heuristic
            },
            languages: vec![Language::Rust],
            require_all_languages: true,
            device: "cpu".into(),
            seed: DecimalU64(request.seed),
            dry_run: false,
            plan_digest: None,
        };
        let mut backend = BackendPlan::resolve(&backend_spec, self.activity.admitted_ranker(), 1)
            .map_err(|error| {
            issue(
                crate::run::service::error_code(&error),
                "The requested Q1 backend could not be prepared.",
            )
        })?;
        if backend.learned()
            && (backend.bundle_digest != request.pins["bundle_sha256"]
                || request.pins["dope_sha256"]
                    != "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814")
        {
            return Err(invalid(
                "Q1 bundle/DOPE pins differ from the admitted evaluator.",
            ));
        }

        let protocol_sha = &request.pins["protocol_sha256"];
        let manifest = load_q1_manifest(&self.home, protocol_sha)?;
        if manifest.schema != "jope.q1-protocol-manifest/v1"
            || manifest.protocol_sha256 != *protocol_sha
            || manifest.source_commit != request.source_commit
            || manifest.plan_sha256 != request.plan_sha256
            || manifest.pins != request.pins
        {
            return Err(invalid("Q1 protocol manifest identity mismatch."));
        }
        let binding = manifest
            .datasets
            .get(&request.dataset.id)
            .ok_or_else(|| unavailable("Dataset is absent from the pinned protocol manifest."))?;
        validate_q1_binding(binding, &request.dataset)?;
        if binding.train_blob_bytes > self.config.limits.max_dataset_bytes.value
            || binding.train_rows > self.config.limits.rung1_max_rows.value
        {
            return Err(unavailable(
                "Pinned Q1 training partition exceeds the local engine limits.",
            ));
        }
        let available = SystemSensors::without_gpu(&self.home)
            .mem_available()
            .map_err(|_| unavailable("Memory admission sensor unavailable."))?;
        if binding
            .train_blob_bytes
            .saturating_mul(4)
            .saturating_add(WORKER_RAM)
            > available
        {
            return Err(unavailable(
                "Insufficient available memory for the pinned Q1 training partition.",
            ));
        }

        let train_sha = parse_digest(&binding.train_blob_sha256)
            .ok_or_else(|| invalid("Pinned training blob hash is malformed."))?;
        let train_blob = crate::store::BlobRef {
            sha256: train_sha,
            bytes: binding.train_blob_bytes,
        };
        let train_csv = self.store.read_blob(&train_blob).map_err(io)?;
        let table = jope_runtime::table::parse_csv(
            &train_csv,
            &jope_runtime::limits::Limits {
                max_input_bytes: binding.train_blob_bytes,
                ..Default::default()
            },
        )
        .map_err(|_| invalid("Pinned training partition is not a valid CSV."))?;
        if table.rows.len() as u64 != binding.train_rows
            || binding.feature_columns.len() as u64 != request.dataset.features
            || table.headers.len() != binding.feature_columns.len() + 1
            || !table.headers.contains(&binding.target)
            || binding
                .feature_columns
                .iter()
                .any(|column| !table.headers.contains(column))
        {
            return Err(invalid(
                "Pinned training partition shape or columns mismatch.",
            ));
        }
        let matrix_bytes = jmat::encode(&table)
            .map_err(|_| invalid("Pinned training partition cannot be encoded."))?;
        let matrix_sha = hash(&matrix_bytes);
        let matrix = jmat::Matrix::open(matrix_bytes)
            .map_err(|_| invalid("Pinned training matrix is invalid."))?;
        let run_id = jope_api::ids::RunId::new();
        let matrix_blob = self.store.publish_blob(matrix.bytes()).map_err(io)?;
        // High patience cannot stop a 24-admission policy early; evaluations
        // and wall time remain explicitly capped, and every worker uses the pinned folds.
        let search_plan = SearchPlan {
            run_id: jope_api::ids::RunId::new(),
            dataset_id: backend_spec.dataset,
            problem_digest: request.dataset.dataset_sha256.clone(),
            bundle_digest: request.pins["bundle_sha256"].clone(),
            binary_version: self.binary.clone(),
            matrix_path: self.store.verified_blob_path(&matrix_blob).map_err(io)?,
            matrix_sha256: matrix_sha.clone(),
            target: binding.target.clone(),
            features: binding.feature_columns.clone(),
            objective: "rmse".into(),
            seed: request.seed,
            planned_workers: 1,
            effort: Effort::High,
            evaluations: request.budget,
            wall_seconds: 600,
            evaluation_wall_ms: 600_000,
            evaluation_rss_bytes: WORKER_RAM,
            groups: None,
            times: None,
            gap_rows: 0,
        };
        backend
            .bind_fit(matrix.bytes(), &search_plan)
            .map_err(|error| {
                issue(
                    crate::run::service::error_code(&error),
                    "Pinned Q1 FIT binding failed.",
                )
            })?;
        let inner = make_executor().map_err(io)?;
        let search_root = self.home.join("q1/search");
        match std::fs::create_dir(&search_root) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                if !std::fs::symlink_metadata(&search_root)
                    .map_err(io)?
                    .file_type()
                    .is_dir()
                {
                    return Err(invalid("Q1 search root is not a real directory."));
                }
            }
            Err(error) => return Err(io(error)),
        }
        // Give each pre-reservation setup attempt its own private journal
        // directory. A crash before request reservation must not poison a
        // later retry with an already-created directory.
        let search_home =
            search_root.join(format!("{}-{}", request.request_sha256, search_plan.run_id));
        std::fs::create_dir(&search_home).map_err(io)?;
        let search_store = Arc::new(Store::open(search_home.join("store")).map_err(io)?);
        let mut discovery = Discovery::open(search_store, search_plan).map_err(io)?;
        let started = self.reserve_q1_request(&request, &run_id)?;
        if let Some(bytes) = started {
            return Ok(bytes);
        }

        let _activity = self.activity.begin_run();
        drop(admission);
        let executor = Q1PinnedExecutor {
            inner,
            folds: json!({"version":1,"matrix_sha256":matrix_sha,"target":binding.target,
                "features":binding.feature_columns,"folds":binding.folds}),
            started_at,
            attempts: Mutex::new(vec![]),
        };
        // Match the proposal construction and numerical recipe across arms.
        // The verified backend chooses only the admission policy.
        let space = BackendSpace {
            native: NativeSpace,
            learned: true,
        };
        let mut terminal_failure = None;
        while discovery.completed().is_none() {
            if let Err(error) =
                discovery.advance(&space, &backend.admission, &executor, 1, usize::MAX)
            {
                if discovery.outcome_unknown() {
                    return Err(io(error));
                }
                terminal_failure = Some(format!(
                    "Q1 search stopped before completing its admitted budget: {error:#}"
                ));
                break;
            }
        }
        let used = discovery.snapshot().budget_used.evaluations.0;
        let attempts: Vec<_> = executor
            .attempts
            .into_inner()
            .map_err(|_| io("Q1 inventory poisoned."))?
            .into_iter()
            .map(|(_, receipt)| receipt)
            .collect();
        if attempts.len() as u64 != used || used > request.budget {
            terminal_failure.get_or_insert_with(|| {
                format!(
                    "Q1 worker inventory mismatch: {} actual worker attempts for {used} committed admissions (budget {}).",
                    attempts.len(), request.budget
                )
            });
        }
        let mut attempt_events = Vec::with_capacity(attempts.len() * 2);
        for attempt in &attempts {
            attempt_events.push(EventBody::ExperimentStarted(ExperimentStartedBody {
                experiment_id: format!("q1_{}", attempt.attempt),
                hypothesis: "Pinned Q1 evaluation attempt".into(),
            }));
            attempt_events.push(EventBody::ExperimentFinished(ExperimentFinishedBody {
                experiment_id: format!("q1_{}", attempt.attempt),
                outcome: if attempt.outcome == EvaluationAttemptOutcome::Completed {
                    ExperimentOutcome::NotImproved
                } else {
                    ExperimentOutcome::Failed
                },
                metric: request.metric.clone(),
                value: attempt.loss,
                failure: attempt.failure.as_ref().map(|failure| PublicFailure {
                    code: ErrorCode::NumericalFailure,
                    reason: format!("{}: {}", failure.code, failure.reason),
                }),
                attempt_receipt: Some(Box::new(attempt.clone())),
            }));
        }

        let charged = attempts.len() as u64;
        let champion = attempts
            .iter()
            .filter(|attempt| attempt.outcome == jope_api::EvaluationAttemptOutcome::Completed)
            .min_by(|left, right| left.loss.unwrap().total_cmp(&right.loss.unwrap()));
        let champion_program = champion.and_then(|attempt| attempt.program_sha256.clone());
        let receipt = EvaluationReceipt {
            schema: "jope.q1-evaluation-receipt/v1".into(),
            request_sha256: request.request_sha256.clone(),
            identity: EvaluationIdentity {
                source_commit: request.source_commit.clone(),
                plan_sha256: request.plan_sha256.clone(),
                pins: request.pins.clone(),
                dataset_id: request.dataset.id.clone(),
                dataset_sha256: request.dataset.dataset_sha256.clone(),
                cut_sha256: request.dataset.cut_sha256.clone(),
                guidance: request.guidance.clone(),
                metric: request.metric.clone(),
                seed: request.seed,
                requested_budget: request.budget,
                dataset_binding_sha256: binding.dataset_binding_sha256.clone(),
            },
            charged,
            attempts,
            champion_program,
            adapter: format!("jope-service/{}", self.binary),
        };
        let receipt_bytes = serde_json::to_vec(&receipt).map_err(io)?;
        self.commit_q1_receipt(
            &request,
            &run_id,
            receipt,
            receipt_bytes.clone(),
            Q1ReceiptCompletion {
                attempt_events,
                elapsed: started_at.elapsed(),
                terminal_failure,
            },
        )?;
        Ok(receipt_bytes)
    }

    fn reserve_q1_request(
        &self,
        request: &Q1Request,
        run_id: &jope_api::ids::RunId,
    ) -> Result<Option<Vec<u8>>> {
        let key = q1_command_key(&request.request_sha256);
        if let Some(record) = self.store.get(&key).map_err(io)? {
            return q1_replay_or_unknown(record.data);
        }
        let snapshot = q1_running_snapshot(run_id.clone(), request.budget);
        let result = self.journal.commit(
            RunUpdate {
                expected_revision: None,
                snapshot,
                checkpoint: json!({"kind":"q1","request_sha256":request.request_sha256}),
                blobs: vec![],
            },
            vec![EventBody::RunState(RunStateBody {
                state: RunState::Running,
                phase: RunPhase::Encoding,
                pending: Some("pinned_training_evaluation".into()),
                reason: None,
            })],
            |tx| {
                if tx.get(&key)?.is_some() {
                    return Err(crate::store::StoreError::Invalid("Q1 request already reserved"));
                }
                tx.put(
                    &key,
                    &crate::store::Record::metadata(
                        json!({"status":"started_outcome_unknown","request_sha256":request.request_sha256,"run_id":run_id}),
                        vec![],
                    ),
                )
            },
        );
        match result {
            Ok(_) => Ok(None),
            Err(_) => match self.store.get(&key).map_err(io)? {
                Some(record) => q1_replay_or_unknown(record.data),
                None => Err(io("Could not durably reserve Q1 request.")),
            },
        }
    }

    fn commit_q1_receipt(
        &self,
        request: &Q1Request,
        run_id: &jope_api::ids::RunId,
        receipt: EvaluationReceipt,
        receipt_bytes: Vec<u8>,
        completion: Q1ReceiptCompletion,
    ) -> Result<()> {
        let Q1ReceiptCompletion {
            mut attempt_events,
            elapsed,
            terminal_failure,
        } = completion;
        let mut snapshot = self
            .journal
            .snapshot(run_id)
            .map_err(io)?
            .ok_or_else(|| io("Q1 reservation disappeared."))?;
        let expected_revision = snapshot.revision.0;
        let completed = receipt
            .attempts
            .iter()
            .filter(|attempt| attempt.outcome == jope_api::EvaluationAttemptOutcome::Completed)
            .count();
        let loss = receipt
            .attempts
            .iter()
            .filter_map(|attempt| attempt.loss)
            .min_by(f64::total_cmp);
        snapshot.revision.0 = expected_revision
            .checked_add(1)
            .ok_or_else(|| io("Q1 receipt revision exhausted."))?;
        snapshot.state = if terminal_failure.is_some() || completed == 0 {
            RunState::Failed
        } else {
            RunState::Completed
        };
        snapshot.phase = RunPhase::Finalizing;
        snapshot.pending = None;
        snapshot.budget_used = BudgetUsed {
            evaluations: DecimalU64(receipt.charged),
            wall_seconds: format!("{:.3}", elapsed.as_secs_f64()),
        };
        snapshot.completion_reason = Some(CompletionReason::BudgetEvaluations);
        snapshot.evaluation_receipt = Some(receipt.clone());
        if let Some(reason) = terminal_failure.as_deref() {
            attempt_events.push(EventBody::RunState(RunStateBody {
                state: RunState::Failed,
                phase: RunPhase::Finalizing,
                pending: None,
                reason: Some(reason.to_owned()),
            }));
        }
        attempt_events.push(EventBody::RunFinished(RunFinishedBody {
            completion_reason: CompletionReason::BudgetEvaluations,
            champion: None,
            scores: FinalScores {
                cv: loss,
                holdout: None,
                test: None,
            },
            budget_used: snapshot.budget_used.clone(),
            nondeterminism: Nondeterminism {
                count: DecimalU64(0),
                experiment_ids: vec![],
            },
            evaluation_receipt: Some(Box::new(receipt.clone())),
        }));
        let command_key = q1_command_key(&request.request_sha256);
        let receipt_text = String::from_utf8(receipt_bytes.clone())
            .map_err(|_| io("Serialized Q1 receipt was not UTF-8."))?;
        self.journal.commit(
            RunUpdate {
                expected_revision: Some(expected_revision),
                snapshot,
                checkpoint: json!({"kind":"q1","request_sha256":request.request_sha256}),
                blobs: vec![],
            },
            attempt_events,
            |tx| {
                let command = tx.get(&command_key)?
                    .ok_or(crate::store::StoreError::Invalid("Q1 reservation missing"))?;
                if command.data["status"] != "started_outcome_unknown" {
                    return Err(crate::store::StoreError::Invalid("Q1 reservation state changed"));
                }
                tx.put(
                    &RecordKey::Evaluation { run: run_id.clone(), number: 0 },
                    &crate::store::Record::metadata(json!({
                        "request_sha256":request.request_sha256,
                        "receipt_bytes":receipt_text,
                        "charged":receipt.charged,
                        "completed":completed,
                        "terminal_failure":terminal_failure,
                        "fold_scores_completed":receipt.attempts.iter()
                            .filter(|attempt| attempt.outcome == jope_api::EvaluationAttemptOutcome::Completed)
                            .map(|attempt| attempt.fold_scores.len()).sum::<usize>(),
                    }), vec![]),
                )?;
                for attempt in &receipt.attempts {
                    tx.put(
                        &RecordKey::Evaluation { run: run_id.clone(), number: attempt.attempt + 1 },
                        &crate::store::Record::metadata(serde_json::to_value(attempt)?, vec![]),
                    )?;
                }
                tx.put(
                    &command_key,
                    &crate::store::Record::metadata(json!({
                        "status":"completed",
                        "request_sha256":request.request_sha256,
                        "run_id":run_id,
                        "receipt_bytes":receipt_text,
                        "terminal_failure":terminal_failure,
                    }), vec![]),
                )
            },
        ).map_err(io)?;
        Ok(())
    }

    fn prepare(&self, spec: &RunSpec) -> Result<Prepared> {
        if spec.task != Task::Regression {
            return Err(unavailable("This run backend requires task=regression."));
        }
        if jope_runtime::backend::DeviceRequest::parse(&spec.device).is_none() {
            return Err(invalid("Unknown device."));
        }
        let mut backend =
            BackendPlan::resolve(spec, self.activity.admitted_ranker(), 1).map_err(|error| {
                let code = crate::run::service::error_code(&error);
                // Keep the public catalog code without exposing private backend diagnostics.
                issue(
                    code,
                    format!("{code}: The requested run backend could not be prepared."),
                )
            })?;
        if spec.test_dataset.is_some()
            || spec.split.kind != SplitKind::Random
            || spec.split.group_column.is_some()
            || spec.split.time_column.is_some()
        {
            return Err(unavailable(
                "This run backend supports random FIT/internal assessment splits without a separate test dataset.",
            ));
        }
        let language_filter = run_language_filter(&spec.languages)?;
        let (objective, resolved_objective) = crate::run::service::run_objective(&spec.objective)
            .map_err(|_| {
            invalid("The objective is invalid or requires unsupported ordered/auxiliary inputs.")
        })?;
        if !spec.split.holdout_fraction.is_finite()
            || !(0.0..1.0).contains(&spec.split.holdout_fraction)
            || spec.split.holdout_fraction == 0.0
        {
            return Err(invalid(
                "holdout_fraction must be finite and strictly between zero and one.",
            ));
        }
        let record = self
            .store
            .get(&RecordKey::Dataset(spec.dataset.clone()))
            .map_err(io)?
            .ok_or_else(|| issue(ErrorCode::NotFound, "Registered dataset does not exist."))?;
        let dataset: RegisteredDataset = serde_json::from_value(record.data).map_err(io)?;
        if dataset.id != spec.dataset {
            return Err(io("Dataset identity mismatch."));
        }
        if dataset.csv.bytes > self.config.limits.max_dataset_bytes.value {
            return Err(invalid("Dataset exceeds configured maximum bytes."));
        }
        let fit_estimate =
            ((dataset.rows as f64) * (1.0 - spec.split.holdout_fraction)).floor() as u64;
        if fit_estimate > self.config.limits.rung1_max_rows.value {
            return Err(invalid("FIT exceeds limits.rung1_max_rows."));
        }
        let available = SystemSensors::without_gpu(&self.home)
            .mem_available()
            .map_err(|_| unavailable("Memory admission sensor unavailable."))?;
        if dataset
            .csv
            .bytes
            .saturating_mul(4)
            .saturating_add(WORKER_RAM)
            > available
        {
            return Err(issue(
                ErrorCode::BackendUnavailable,
                "Insufficient available memory for this bounded run.",
            ));
        }
        let source = self.store.read_blob(&dataset.csv).map_err(io)?;
        let mut table = jope_runtime::table::parse_csv(&source, &Default::default()).map_err(io)?;
        let matrix =
            jmat::Matrix::open(self.store.read_blob(&dataset.matrix).map_err(io)?).map_err(io)?;
        let target = table
            .headers
            .iter()
            .position(|v| v == &spec.target)
            .ok_or_else(|| invalid("Unknown target column."))?;
        if spec.ignore_columns.contains(&spec.target) {
            return Err(invalid("The target cannot be ignored."));
        }
        let ignored: std::collections::BTreeSet<_> = spec.ignore_columns.iter().collect();
        if ignored.len() != spec.ignore_columns.len()
            || ignored.iter().any(|v| !table.headers.contains(v))
        {
            return Err(invalid(
                "Ignored columns must be distinct registered columns.",
            ));
        }
        let selected: Vec<_> = table
            .headers
            .iter()
            .enumerate()
            .filter(|(_, name)| !ignored.contains(name))
            .map(|(i, _)| i)
            .collect();
        if selected.len() < 2
            || selected
                .iter()
                .any(|&i| matrix.columns()[i].kind != jmat::ColumnKind::Numeric)
            || table.rows.iter().any(|row| {
                row[target]
                    .as_deref()
                    .and_then(|v| v.trim().parse::<f64>().ok())
                    .is_none_or(|v| !v.is_finite())
            })
        {
            return Err(invalid(
                "At least one numeric feature and a finite numeric target in every row are required.",
            ));
        }
        let features: Vec<_> = selected
            .iter()
            .filter(|&&i| i != target)
            .map(|&i| table.headers[i].clone())
            .collect();
        table.headers = selected.iter().map(|&i| table.headers[i].clone()).collect();
        table.rows = table
            .rows
            .into_iter()
            .map(|row| selected.iter().map(|&i| row[i].clone()).collect())
            .collect();
        let fit_count =
            ((table.rows.len() as f64) * (1.0 - spec.split.holdout_fraction)).floor() as usize;
        let effort = SearchEffort::from_name(match spec.effort {
            Effort::Fast => "fast",
            Effort::Medium => "medium",
            Effort::High => "high",
            Effort::Ultra => "ultra",
        })
        .unwrap()
        .limits_with_budget(BudgetOverrides {
            evaluations: spec.budget.evaluations.map(|v| v.0),
            wall_seconds: spec.budget.wall_seconds.map(|v| v.0),
            ..Default::default()
        });
        if effort.evaluations == 0
            || effort.wall_seconds == 0
            || fit_count < usize::from(effort.rung1_folds)
            || fit_count > self.config.limits.rung1_max_rows.value as usize
            || fit_count >= table.rows.len()
        {
            return Err(invalid(
                "Budget or FIT row count is outside the backend's configured limits.",
            ));
        }
        let (fit_ids, holdout_ids) = random_split(
            table.rows.len(),
            fit_count,
            spec.seed.0,
            "public-run/fit-v1",
        )
        .map_err(io)?;
        let encode = |ids: &[usize]| -> Result<Vec<u8>> {
            let mut part = table.clone();
            part.rows = ids.iter().map(|&i| table.rows[i].clone()).collect();
            jmat::encode(&part).map_err(io)
        };
        let fit = encode(&fit_ids)?;
        let holdout = encode(&holdout_ids)?;
        let governor = Governor::new(
            std::num::NonZeroU64::new(u64::MAX).unwrap(),
            NonZeroU32::new(1).unwrap(),
            self.config.limits.disk_reserve.value,
        );
        let estimates = governor.preflight(
            Requirements {
                matrix_ram_bytes: dataset.csv.bytes.saturating_mul(4),
                per_worker_ram_bytes: WORKER_RAM,
                matrix_disk_bytes: (fit.len() + holdout.len()) as u64,
                artifact_bytes: MODEL_LIMIT,
                ..Default::default()
            },
            &mut SystemSensors::without_gpu(&self.home),
        )?;
        let timeout = self
            .config
            .limits
            .eval_timeout
            .value
            .as_millis()
            .min(u128::from(effort.wall_seconds) * 1000)
            .max(1) as u64;
        let mut normalized = serde_json::to_value(spec).map_err(io)?;
        normalized["dry_run"] = json!(false);
        normalized["plan_digest"] = Value::Null;
        let plan_digest = hash(
            &serde_json::to_vec(
                &json!({"backend":"cpu-ridge-seeded-rmse/v1", "spec":normalized,
            "dataset":dataset.csv, "fit":hash(&fit), "binary":self.binary, "timeout_ms":timeout,
            "worker_ram":WORKER_RAM, "workers":1, "estimates":estimates,
            "ranker_bundle":backend.bundle_digest,"placement":backend.placement}),
            )
            .map_err(io)?,
        );
        if spec.plan_digest.as_ref().is_some_and(|v| v != &plan_digest) {
            return Err(issue(
                ErrorCode::PlanChanged,
                "Plan digest changed; request a fresh dry-run plan.",
            ));
        }
        let mut public: Plan = serde_json::from_value(json!({"plan_digest":plan_digest,
            "task":{"value":"regression","inferred":false,"reason":"Explicit regression task."},
            "objective":resolved_objective,
            "rows":{"total":table.rows.len().to_string(),"fit":fit_count.to_string(),"holdout":holdout_ids.len().to_string(),"test":null},
            "effort":{"name":spec.effort,"evaluations":effort.evaluations.to_string(),"wall_seconds":effort.wall_seconds.to_string(),"cv_folds":effort.rung1_folds,"rounds_in_flight":1},
            "guidance":backend.guidance,"landmarks":{"mode":spec.landmarks,"sample_rows":0},
            "languages":{"requested":spec.languages,"excluded_operations":language_filter.excluded_ops(),"intersection_empty":false},
            "placement":backend.placement,
            "limits":{"eval_timeout_s":timeout.div_ceil(1000),"worker_rss_bytes":WORKER_RAM.to_string()},"estimates":estimates,"warnings":backend.warnings})).map_err(io)?;
        let search = SearchPlan {
            run_id: format!("run_{}", ulid::Ulid::new()).parse().map_err(io)?,
            dataset_id: spec.dataset.clone(),
            problem_digest: hash(&serde_json::to_vec(&json!({"spec":normalized,"csv_sha256":dataset.csv.sha256,
                "fit_sha256":hash(&fit),"objective_digest":public.objective.digest,"split":"public-run/fit-v1"})).map_err(io)?),
            bundle_digest: backend.bundle_digest.clone(),
            binary_version: self.binary.clone(),
            matrix_path: PathBuf::new(),
            matrix_sha256: hash(&fit),
            target: spec.target.clone(),
            features,
            objective,
            seed: spec.seed.0,
            planned_workers: 1,
            effort: spec.effort,
            evaluations: effort.evaluations,
            wall_seconds: effort.wall_seconds,
            evaluation_wall_ms: timeout,
            evaluation_rss_bytes: WORKER_RAM,
            groups: None,
            times: None,
            gap_rows: 0,
        };
        backend
            .bind_fit(&fit, &search)
            .map_err(|_| invalid("Unable to prepare numeric FIT landmarks."))?;
        public.landmarks.sample_rows = backend.landmark_rows as u32;
        let rows = |ids: Vec<usize>| {
            ids.into_iter()
                .map(|row| AuditRowId {
                    dataset: spec.dataset.clone(),
                    row: row as u64,
                })
                .collect()
        };
        Ok(Prepared {
            public,
            search,
            fit,
            holdout,
            fit_rows: rows(fit_ids),
            holdout_rows: rows(holdout_ids),
            backend,
        })
    }

    fn start(
        self: &Arc<Self>,
        spec: RunSpec,
        identity: RequestIdentity,
        _pub06_values: Pub06Injection,
    ) -> Response {
        let _guard = match self.admission.lock() {
            Ok(guard) => guard,
            Err(_) => return response(&io("Run admission lock is poisoned.")),
        };
        match self.store.idempotent_receipt(&identity) {
            Ok(Some(receipt)) => return receipt.into_response(),
            Ok(None) => {}
            Err(error) => return error.into_response(),
        }
        if !spec.dry_run && self.activity.active_runs() != 0 {
            return response(&issue(
                ErrorCode::BackendUnavailable,
                "The bounded CPU run backend is busy; retry this key after the active run drains.",
            ));
        }
        let mut prepared = match self.prepare(&spec) {
            Ok(value) => value,
            Err(error) => return response(&error),
        };
        if spec.dry_run {
            return match self.store.execute_idempotent(&identity, |_| {
                CommandOutcome::json(200, serde_json::to_value(&prepared.public)?)
            }) {
                Ok(receipt) => receipt.into_response(),
                Err(error) => error.into_response(),
            };
        }
        let fit = match self.store.publish_blob(prepared.fit.as_slice()) {
            Ok(value) => value,
            Err(error) => return response(&io(error)),
        };
        let holdout = match self.store.publish_blob(prepared.holdout.as_slice()) {
            Ok(value) => value,
            Err(error) => return response(&io(error)),
        };
        prepared.search.matrix_path = match self.store.verified_blob_path(&fit) {
            Ok(value) => value,
            Err(error) => return response(&io(error)),
        };
        let holdout_path = match self.store.verified_blob_path(&holdout) {
            Ok(value) => value,
            Err(error) => return response(&io(error)),
        };
        let request = FinalizationRequest {
            run_id: prepared.search.run_id.clone(),
            fit: Partition {
                path: prepared.search.matrix_path.clone(),
                sha256: prepared.search.matrix_sha256.clone(),
                rows: prepared.fit_rows,
            },
            holdout: Partition {
                path: holdout_path,
                sha256: hash(&prepared.holdout),
                rows: prepared.holdout_rows,
            },
            test: None,
            // Publish both genuine models once; exports select their recorded fit mode.
            refit: Refit::FitHoldout,
            model_root: self.home.join("models"),
            scratch_root: self.home.join("run-scratch"),
            max_input_bytes: self.config.limits.max_dataset_bytes.value,
            max_model_bytes: MODEL_LIMIT,
        };
        // This context precedes the first run commit, so every visible run can
        // resume from its original immutable plan and partition provenance.
        let context = Continuation {
            search: prepared.search.clone(),
            finalization: request.clone(),
            backend: ContinuationBackend {
                spec,
                placement: prepared.backend.placement.clone(),
                landmarks: prepared.backend.full_fit_landmarks.clone(),
            },
        };
        let persisted = self.store.transaction(|tx| {
            tx.put(
                &continuation_key(&prepared.search.run_id),
                &Record::metadata(
                    serde_json::to_value(&context)?,
                    vec![fit.clone(), holdout.clone()],
                ),
            )
        });
        if let Err(error) = persisted {
            return response(&io(error));
        }
        let sender = match self.worker_sender(
            _pub06_values,
            false,
            prepared.backend,
            context.backend.spec.languages.clone(),
        ) {
            Ok(sender) => sender,
            Err(error) => return response(&error),
        };
        let discovery = match Discovery::start_command(
            Arc::clone(&self.store),
            prepared.search,
            Arc::clone(&self.journal),
            identity.clone(),
            vec![fit, holdout],
            prepared.public.warnings,
        ) {
            Ok(value) => value,
            Err(error) => return response(&io(error)),
        };
        if sender.send((discovery, request)).is_err() {
            return response(&io("Run worker stopped before dispatch."));
        }
        match self.store.idempotent_receipt(&identity) {
            Ok(Some(receipt)) => receipt.into_response(),
            Ok(None) => response(&io("Committed run receipt missing.")),
            Err(error) => error.into_response(),
        }
    }

    fn worker_sender(
        self: &Arc<Self>,
        _pub06_values: Pub06Injection,
        explicit_continue: bool,
        backend: BackendPlan,
        languages: Vec<Language>,
    ) -> Result<mpsc::SyncSender<(Discovery, FinalizationRequest)>> {
        let (sender, receiver) = mpsc::sync_channel::<(Discovery, FinalizationRequest)>(1);
        let owner = Arc::clone(self);
        let allowed = run_language_filter(&languages)?;
        #[cfg(all(feature = "failpoints", debug_assertions))]
        let run_space = CpuRunSpace {
            allowed,
            pub06_values: _pub06_values,
        };
        #[cfg(not(all(feature = "failpoints", debug_assertions)))]
        let run_space = CpuRunSpace { allowed };
        let learned = backend.learned();
        let admission = backend.admission;
        let run_space = BackendSpace {
            native: run_space,
            learned,
        };
        let activity = self.activity.begin_run();
        if let Err(error) = std::thread::Builder::new()
            .name("jope-run".into())
            .spawn(move || {
                let _activity = activity;
                if let Ok((mut discovery, request)) = receiver.recv() {
                    let result = (|| -> anyhow::Result<()> {
                        let process = ProcessExecutor::new(
                            &owner.executable,
                            1,
                            &owner.home.join("worker-scratch"),
                        )?;
                        let executor = BackendExecutor {
                            inner: &process,
                            learned,
                        };
                        while discovery.completed().is_none() {
                            discovery.advance(&run_space, &admission, &executor, 1, usize::MAX)?;
                        }
                        let finalizer = Finalizer::with_journal(
                            Arc::clone(&owner.store),
                            Arc::clone(&owner.journal),
                        );
                        if explicit_continue {
                            finalizer.finalize_after_explicit_continue(&request, &executor)?;
                        } else {
                            finalizer.finalize(&request, &executor)?;
                        }
                        Ok(())
                    })();
                    if let Err(error) = result {
                        owner.fail(&request.run_id, &error.to_string());
                    }
                }
            })
        {
            return Err(io(error));
        }
        Ok(sender)
    }

    fn continue_run(
        self: &Arc<Self>,
        run: jope_api::ids::RunId,
        expected_revision: Option<u64>,
        identity: RequestIdentity,
    ) -> Response {
        let _guard = match self.admission.lock() {
            Ok(guard) => guard,
            Err(_) => return response(&io("Run admission lock is poisoned.")),
        };
        match self.store.idempotent_receipt(&identity) {
            Ok(Some(receipt)) => return receipt.into_response(),
            Ok(None) => {}
            Err(error) => return error.into_response(),
        }
        let attempt = (|| -> Result<()> {
            let snapshot = self
                .journal
                .snapshot(&run)
                .map_err(io)?
                .ok_or_else(|| issue(ErrorCode::NotFound, "Run does not exist."))?;
            if expected_revision.is_some_and(|revision| revision != snapshot.revision.0) {
                return Err(issue(
                    ErrorCode::RevisionConflict,
                    "Re-read the snapshot and retry.",
                ));
            }
            if snapshot.state != RunState::Paused || self.activity.active_runs() != 0 {
                return Err(issue(
                    ErrorCode::RevisionConflict,
                    "Continue requires an idle backend and a paused run.",
                ));
            }
            let stored_context = self
                .store
                .get(&continuation_key(&run))
                .map_err(io)?
                .ok_or_else(|| {
                    unavailable("Original continuation context is unavailable for this run.")
                })?;
            let context: Continuation = serde_json::from_value(stored_context.data).map_err(io)?;
            if context.search.run_id != run
                || context.finalization.run_id != run
                || context.search.binary_version != self.binary
            {
                return Err(unavailable(
                    "Continuation requires the original binary and run identity.",
                ));
            }
            // Rebuild only the original backend, retaining the checkpoint plan.
            let backend = context.backend.restore(
                &self.store,
                &context.search,
                &stored_context.blobs,
                self.activity.admitted_ranker(),
            )?;
            #[cfg(all(feature = "failpoints", debug_assertions))]
            let injection = None;
            #[cfg(not(all(feature = "failpoints", debug_assertions)))]
            let injection = ();
            let sender = self.worker_sender(
                injection,
                true,
                backend,
                context.backend.spec.languages.clone(),
            )?;
            let discovery = Discovery::continue_command(
                Arc::clone(&self.store),
                context.search,
                Arc::clone(&self.journal),
                identity.clone(),
            )
            .map_err(io)?;
            sender.send((discovery, context.finalization)).map_err(io)?;
            Ok(())
        })();
        if let Err(error) = attempt {
            return response(&error);
        }
        match self.store.idempotent_receipt(&identity) {
            Ok(Some(receipt)) => receipt.into_response(),
            Ok(None) => response(&io("Committed continue receipt missing.")),
            Err(error) => error.into_response(),
        }
    }

    fn fail(&self, run: &jope_api::ids::RunId, detail: &str) {
        let result = (|| -> anyhow::Result<()> {
            let record = self
                .store
                .get(&RecordKey::Run(run.clone()))?
                .ok_or_else(|| anyhow::anyhow!("Run missing"))?;
            let mut snapshot = self
                .journal
                .snapshot(run)?
                .ok_or_else(|| anyhow::anyhow!("Snapshot missing"))?;
            if snapshot.state != RunState::Running {
                return Ok(());
            }
            let expected = snapshot.revision.0;
            snapshot.revision.0 += 1;
            snapshot.state = RunState::Failed;
            self.journal.commit(
                RunUpdate {
                    expected_revision: Some(expected),
                    snapshot: snapshot.clone(),
                    checkpoint: record.data["checkpoint"].clone(),
                    blobs: record.blobs,
                },
                vec![EventBody::RunState(RunStateBody {
                    state: RunState::Failed,
                    phase: snapshot.phase,
                    pending: None,
                    reason: Some("The bounded CPU run failed; inspect local diagnostics.".into()),
                })],
                |_| Ok(()),
            )?;
            Ok(())
        })();
        let diagnostic = format!("run {run} failed: {detail}; failure journal: {result:?}");
        #[cfg(all(feature = "failpoints", debug_assertions))]
        crate::pub06_test::record_log(diagnostic.clone());
        eprintln!("{diagnostic}");
    }
}

pub(super) async fn start(
    Extension(service): Extension<Arc<RunService>>,
    request: Request<Body>,
) -> Response {
    let headers = request.headers().clone();
    #[cfg(all(feature = "failpoints", debug_assertions))]
    let pub06_values = match (
        headers.get("x-pub06-test-dsl"),
        headers.get("x-pub06-test-operator"),
        headers.get("x-pub06-test-lineage"),
        headers.get("x-pub06-test-round"),
    ) {
        (None, None, None, None) => None,
        (Some(dsl), Some(operator), Some(lineage), Some(round)) => {
            let parse = |value: &axum::http::HeaderValue| value.to_str().ok().map(str::to_owned);
            match (parse(dsl), parse(operator), parse(lineage), parse(round)) {
                (Some(dsl), Some(operator), Some(lineage), Some(round)) => {
                    Some(Pub06InternalValues {
                        dsl,
                        operator,
                        lineage,
                        round,
                        fail_after_seed: headers
                            .get("x-pub06-test-fail-after-seed")
                            .is_some_and(|value| value == "1"),
                    })
                }
                _ => return response(&invalid("Invalid PUB-06 test injection values.")),
            }
        }
        _ => {
            return response(&invalid(
                "PUB-06 test injection values must be supplied together.",
            ))
        }
    };
    // The in-process bridge uses an absolute URI; a listener receives origin
    // form. Authority is authenticated separately and is not command identity.
    let target = request
        .uri()
        .path_and_query()
        .map(|v| v.as_str().to_owned())
        .unwrap_or_else(|| "/v1/runs".into());
    let mut keys = headers.get_all("idempotency-key").iter();
    let Some(key) = keys.next().and_then(|v| v.to_str().ok()) else {
        return response(&invalid("Idempotency-Key is required."));
    };
    if keys.next().is_some() {
        return response(&invalid("Idempotency-Key must occur once."));
    }
    let Json(spec) = match Json::<RunSpec>::from_request(request, &()).await {
        Ok(value) => value,
        Err(_) => return response(&invalid("Expected a valid JSON RunSpec.")),
    };
    let body = match serde_json::to_value(&spec) {
        Ok(value) => value,
        Err(error) => return response(&io(error)),
    };
    let identity = match RequestIdentity::new(
        "local-engine-owner",
        key,
        6,
        &target,
        headers.get("if-match").and_then(|v| v.to_str().ok()),
        &body,
    ) {
        Ok(value) => value,
        Err(error) => return error.into_response(),
    };
    #[cfg(all(feature = "failpoints", debug_assertions))]
    let leak_value = std::env::var_os("JOPE_PUB06_TEST_LEAK")
        .and_then(|_| pub06_values.as_ref().map(|values| values.dsl.clone()));
    #[cfg(not(all(feature = "failpoints", debug_assertions)))]
    let pub06_values = ();
    let result = match tokio::task::spawn_blocking(move || {
        service.start(spec, identity, pub06_values)
    })
    .await
    {
        Ok(value) => value,
        Err(error) => response(&io(error)),
    };
    #[cfg(all(feature = "failpoints", debug_assertions))]
    if let Some(value) = leak_value {
        if let Ok(value) = axum::http::HeaderValue::from_str(&value) {
            let mut result = result;
            result.headers_mut().insert("x-pub06-test-leak", value);
            return result;
        }
    }
    result
}

pub(super) async fn control(
    Extension(service): Extension<Arc<RunService>>,
    path: std::result::Result<axum::extract::Path<String>, axum::extract::rejection::PathRejection>,
    request: Request<Body>,
) -> Response {
    let Ok(axum::extract::Path(run)) = path else {
        return response(&invalid("Invalid run path."));
    };
    let Ok(run) = run.parse::<jope_api::ids::RunId>() else {
        return response(&invalid("Invalid run id."));
    };
    let headers = request.headers().clone();
    let target = request
        .uri()
        .path_and_query()
        .map(|v| v.as_str().to_owned())
        .unwrap_or_default();
    let mut keys = headers.get_all("idempotency-key").iter();
    let Some(key) = keys.next().and_then(|v| v.to_str().ok()) else {
        return response(&invalid("Idempotency-Key is required."));
    };
    if keys.next().is_some() {
        return response(&invalid("Idempotency-Key must occur once."));
    }
    let content_type = headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split(';').next())
        .map(|v| v.trim().to_ascii_lowercase());
    if !content_type.as_deref().is_some_and(|v| {
        v == "application/json" || (v.starts_with("application/") && v.ends_with("+json"))
    }) {
        return response(&invalid("Expected JSON content type."));
    }
    let bytes = match axum::body::to_bytes(request.into_body(), 1024 * 1024).await {
        Ok(value) => value,
        Err(_) => return response(&invalid("Invalid control body.")),
    };
    // Preserve explicit nulls in retry identity; reject duplicate JSON keys.
    let value: Value = match jope_runtime::strict_json::decode(&bytes) {
        Ok(value) => value,
        Err(_) => return response(&invalid("Invalid control JSON.")),
    };
    let body: jope_api::registry::RunControlRequest = match serde_json::from_value(value.clone()) {
        Ok(value) => value,
        Err(_) => return response(&invalid("Expected a valid run control request.")),
    };
    let jope_api::registry::RunControlRequest::Continue { expected_revision } = body else {
        return response(&unavailable(
            "This bounded backend currently supports explicit continue only.",
        ));
    };
    let mut preconditions = headers.get_all("if-match").iter();
    let raw_if_match = preconditions.next();
    if preconditions.next().is_some() {
        return response(&invalid("If-Match must occur once."));
    }
    let if_match = match raw_if_match.map(|v| {
        v.to_str()
            .ok()
            .and_then(|s| s.trim_matches('"').parse::<u64>().ok())
    }) {
        None => None,
        Some(Some(value)) => Some(value),
        Some(None) => return response(&invalid("Invalid If-Match revision.")),
    };
    if if_match
        .zip(expected_revision)
        .is_some_and(|(a, b)| a != b.0)
    {
        return response(&invalid("Revision preconditions disagree."));
    }
    let identity = match RequestIdentity::new(
        "local-engine-owner",
        key,
        8,
        &target,
        raw_if_match.and_then(|v| v.to_str().ok()),
        &value,
    ) {
        Ok(v) => v,
        Err(e) => return e.into_response(),
    };
    match tokio::task::spawn_blocking(move || {
        service.continue_run(run, if_match.or(expected_revision.map(|v| v.0)), identity)
    })
    .await
    {
        Ok(value) => value,
        Err(error) => response(&io(error)),
    }
}

/// Execute one Q1 request and return the same compact bytes durably retained by the engine.
pub(super) async fn q1_evaluate(
    Extension(service): Extension<Arc<RunService>>,
    request: Request<Body>,
) -> Response {
    let Json(body) = match Json::<Value>::from_request(request, &()).await {
        Ok(value) => value,
        Err(_) => return response(&invalid("Expected a valid JSON Q1 request.")),
    };
    match tokio::task::spawn_blocking(move || service.q1_evaluate(body)).await {
        Ok(Ok(bytes)) => Response::builder()
            .status(axum::http::StatusCode::OK)
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .body(Body::from(bytes))
            .unwrap_or_else(|_| response(&io("Could not construct Q1 response."))),
        Ok(Err(error)) => response(&error),
        Err(error) => response(&io(error)),
    }
}

#[derive(Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ListQuery {
    after: Option<String>,
    limit: Option<usize>,
}

pub(super) async fn list(
    Extension(service): Extension<Arc<RunService>>,
    query: std::result::Result<Query<ListQuery>, axum::extract::rejection::QueryRejection>,
) -> Response {
    let Ok(Query(query)) = query else {
        return response(&invalid("Invalid run list query."));
    };
    let limit = query.limit.unwrap_or(100);
    if !(1..=256).contains(&limit) {
        return response(&invalid("limit must be between1 and256."));
    }
    let after = match query
        .after
        .map(|v| v.parse::<jope_api::ids::RunId>())
        .transpose()
    {
        Ok(value) => value,
        Err(_) => return response(&invalid("Invalid run list cursor.")),
    };
    let result = tokio::task::spawn_blocking(move || -> Result<jope_api::registry::RunPage> {
        let mut ids = service
            .store
            .run_ids_after(after.as_ref(), limit + 1)
            .map_err(io)?;
        let more = ids.len() > limit;
        ids.truncate(limit);
        let next_after = if more {
            ids.last().map(ToString::to_string)
        } else {
            None
        };
        let runs = ids
            .iter()
            .map(|id| {
                service
                    .journal
                    .snapshot(id)
                    .map_err(io)?
                    .ok_or_else(|| io("Listed run disappeared."))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(jope_api::registry::RunPage { runs, next_after })
    })
    .await;
    match result {
        Ok(Ok(page)) => Json(page).into_response(),
        Ok(Err(error)) => response(&error),
        Err(error) => response(&io(error)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::rand_core::SeedableRng;

    #[test]
    fn q1_cut_hash_matches_canonical_cross_language_control() {
        let binding = Q1DatasetBinding {
            id: "0000000000000001".into(),
            task: "regression".into(),
            dataset_sha256: "a".repeat(64),
            family: "f".repeat(40),
            family_authoritative: true,
            rows: 4,
            features: 2,
            cut_sha256: String::new(),
            holdout_group: "development_fit".into(),
            catalog_version: "v1".into(),
            catalog_record_sha256: "c".repeat(64),
            dataset_binding_sha256: String::new(),
            target: "y".into(),
            feature_columns: vec!["x1".into(), "x2".into()],
            train_rows: 4,
            train_blob_sha256: "b".repeat(64),
            train_blob_bytes: 1,
            folds: vec![
                Q1Fold {
                    repeat: 0,
                    train: vec![0, 1],
                    validation: vec![2, 3],
                },
                Q1Fold {
                    repeat: 0,
                    train: vec![2, 3],
                    validation: vec![0, 1],
                },
            ],
        };
        assert_eq!(
            q1_cut_sha256(&binding).unwrap(),
            "0c87e715ccb1a01fe05d4bc4e8b81f62f978d8aed0126248d97b866686acf249"
        );
        let identity = Q1DatasetIdentity {
            id: binding.id.clone(),
            task: binding.task.clone(),
            dataset_sha256: binding.dataset_sha256.clone(),
            family: binding.family.clone(),
            family_authoritative: binding.family_authoritative,
            rows: binding.rows,
            features: binding.features,
            cut_sha256: binding.cut_sha256.clone(),
            holdout_group: binding.holdout_group.clone(),
        };
        let mut mismatched = binding;
        mismatched.cut_sha256 = "d".repeat(64);
        let mut binding_value = serde_json::to_value(&mismatched).unwrap();
        binding_value
            .as_object_mut()
            .unwrap()
            .remove("dataset_binding_sha256");
        mismatched.dataset_binding_sha256 = hash(&serde_json::to_vec(&binding_value).unwrap());
        assert!(validate_q1_binding(&mismatched, &identity).is_err());
    }

    const Q1_FAIL_ALL_TEST_SEED: u64 = 2;

    #[test]
    #[ignore = "Q1 worker subprocess target; invoked by the worker process supervisor"]
    fn q1_worker_helper() {
        use std::io::Write as _;

        assert_eq!(std::env::var("JOPE_WORKER_FIXTURE").unwrap(), "worker");
        let mut stdout = std::io::stdout().lock();
        writeln!(stdout, "WORKER_FIXTURE_READY").unwrap();
        stdout.flush().unwrap();
        drop(stdout);
        crate::worker::serve(1, |request, matrix| {
            let request_seed = request
                .seed
                .wrapping_sub(request.attempt.attempt.saturating_sub(1));
            if request_seed == Q1_FAIL_ALL_TEST_SEED {
                Err(crate::worker::WorkerFailure::InvalidRequest)
            } else {
                crate::discover::r#loop::evaluate_native(request, matrix)
            }
        })
        .unwrap();
    }

    #[test]
    fn learned_seed_schedule_cycles_until_successful_measurements_exist() {
        let seeds = vec![
            "seed-a".to_owned(),
            "seed-b".to_owned(),
            "seed-c".to_owned(),
        ];
        let scheduled = (0..24)
            .map(|attempt| q1_learned_seed(&seeds, attempt, &[]))
            .collect::<Vec<_>>();
        assert_eq!(scheduled.len(), 24);
        assert!(scheduled
            .iter()
            .enumerate()
            .all(|(attempt, seed)| *seed == seeds[attempt % seeds.len()]));
        let measured = vec![("seed-b".to_owned(), 1.0)];
        assert_eq!(q1_learned_seed(&seeds, 1, &measured), "seed-c");
    }

    #[test]
    fn q1_attempt_timeout_is_bounded_by_one_total_wall_budget() {
        assert_eq!(
            q1_remaining_wall_ms(std::time::Duration::ZERO),
            Some(600_000)
        );
        assert_eq!(
            q1_remaining_wall_ms(std::time::Duration::from_millis(1250)),
            Some(598_750)
        );
        assert_eq!(
            q1_remaining_wall_ms(Q1_WALL_BUDGET - std::time::Duration::from_millis(1)),
            Some(1)
        );
        assert_eq!(q1_remaining_wall_ms(Q1_WALL_BUDGET), None);
        assert_eq!(
            q1_remaining_wall_ms(Q1_WALL_BUDGET + std::time::Duration::from_secs(1)),
            None
        );
    }
    use crate::{
        dataset::register,
        engine::InProcessEngine,
        http::auth::{BearerToken, LocalAuth},
    };
    use axum::{
        body::{to_bytes, Body},
        http::{header, Request, StatusCode},
    };
    use tower::ServiceExt;

    fn fixture() -> (tempfile::TempDir, InProcessEngine, RunSpec) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/int01");
        std::fs::create_dir_all(&root).unwrap();
        let directory = tempfile::tempdir_in(root).unwrap();
        let engine = InProcessEngine::open(directory.path()).unwrap();
        let raw = include_bytes!("../../../../test-support/datasets/v1/d2-regression/train.csv");
        assert_eq!(
            hash(raw),
            "d9a0b97e91ff9d04bc87fefa4671a49c40cee21dfac05be58a7fc7ce08045931"
        );
        let mut names: Vec<_> = (0..21).map(|i| format!("x{i}")).collect();
        names.push("target".into());
        let mut csv = names.join(",");
        csv.push('\n');
        for line in std::str::from_utf8(raw).unwrap().lines().take(96) {
            csv.push_str(line);
            csv.push('\n');
        }
        let dataset =
            register::register(engine.store(), csv.as_bytes(), &Default::default()).unwrap();
        let spec=serde_json::from_value(json!({"dataset":dataset.id,"target":"target","task":"regression",
            "ignore_columns":[],"test_dataset":null,"split":{"kind":"random","holdout_fraction":0.2,"group_column":null,"time_column":null},
            "objective":"rmse","effort":"fast","budget":{"evaluations":"4","wall_seconds":"30"},"landmarks":"none","guidance":"heuristic",
            "languages":["rust"],"require_all_languages":true,"device":"cpu","seed":"73","dry_run":true,"plan_digest":null})).unwrap();
        (directory, engine, spec)
    }

    fn continuation_backend_fixture() -> (
        tempfile::TempDir,
        InProcessEngine,
        ContinuationBackend,
        SearchPlan,
        crate::store::BlobRef,
    ) {
        let (directory, engine, spec) = fixture();
        let dataset: RegisteredDataset = serde_json::from_value(
            engine
                .store()
                .get(&RecordKey::Dataset(spec.dataset.clone()))
                .unwrap()
                .unwrap()
                .data,
        )
        .unwrap();
        let fit = dataset.matrix;
        let backend = BackendPlan::resolve(&spec, None, 1).unwrap();
        let plan = SearchPlan {
            run_id: "run_01ARZ3NDEKTSV4RRFFQ69G5FAV".parse().unwrap(),
            dataset_id: spec.dataset.clone(),
            problem_digest: "a".repeat(64),
            bundle_digest: backend.bundle_digest,
            binary_version: "continuation-guard-test".into(),
            matrix_path: engine.store().verified_blob_path(&fit).unwrap(),
            matrix_sha256: hash(&engine.store().read_blob(&fit).unwrap()),
            target: spec.target.clone(),
            features: (0..21).map(|i| format!("x{i}")).collect(),
            objective: "rmse".into(),
            seed: spec.seed.0,
            planned_workers: 1,
            effort: spec.effort,
            evaluations: 4,
            wall_seconds: 30,
            evaluation_wall_ms: 1000,
            evaluation_rss_bytes: WORKER_RAM,
            groups: None,
            times: None,
            gap_rows: 0,
        };
        let context = ContinuationBackend {
            spec,
            placement: backend.placement,
            landmarks: None,
        };
        (directory, engine, context, plan, fit)
    }

    #[test]
    fn continuation_restores_native_backend_from_persisted_request_without_replanning() {
        let (_directory, engine, context, plan, fit) = continuation_backend_fixture();
        // Legacy native contexts predate the optional landmark evidence.
        let mut legacy = serde_json::to_value(context).unwrap();
        legacy.as_object_mut().unwrap().remove("landmarks");
        let context: ContinuationBackend = serde_json::from_value(legacy).unwrap();
        let original = plan.clone();
        let restored = context
            .restore(engine.store(), &plan, &[fit], None)
            .unwrap();
        assert!(!restored.learned());
        assert_eq!(restored.placement, context.placement);
        assert_eq!(restored.bundle_digest, original.bundle_digest);
        assert_eq!(plan, original);
    }

    #[test]
    fn learned_continuation_without_admitted_capability_never_falls_back_to_native() {
        let (_directory, engine, mut context, plan, fit) = continuation_backend_fixture();
        context.spec.guidance = Guidance::Learned;
        context.spec.landmarks = LandmarkMode::Full;
        let error = context
            .restore(engine.store(), &plan, &[fit], None)
            .err()
            .unwrap();
        assert_eq!(error.code, ErrorCode::CapabilityUnavailable);
    }

    #[test]
    fn continuation_refuses_changed_backend_identity_and_tampered_fit() {
        let (_directory, engine, mut context, mut plan, fit) = continuation_backend_fixture();
        context.placement.workers = 2;
        assert_eq!(
            context
                .restore(engine.store(), &plan, std::slice::from_ref(&fit), None)
                .err()
                .unwrap()
                .code,
            ErrorCode::CapabilityUnavailable
        );
        context.placement.workers = 1;
        let original = plan.clone();
        plan.bundle_digest = "b".repeat(64);
        assert_eq!(
            context
                .restore(engine.store(), &plan, std::slice::from_ref(&fit), None)
                .err()
                .unwrap()
                .code,
            ErrorCode::CapabilityUnavailable
        );
        plan = original;
        plan.matrix_path = plan.matrix_path.with_extension("changed");
        assert_eq!(
            context
                .restore(engine.store(), &plan, std::slice::from_ref(&fit), None)
                .err()
                .unwrap()
                .code,
            ErrorCode::CapabilityUnavailable
        );
        plan.matrix_path = engine.store().verified_blob_path(&fit).unwrap();
        std::fs::write(&plan.matrix_path, b"corrupted owned TRAIN fixture").unwrap();
        assert!(context
            .restore(engine.store(), &plan, &[fit], None)
            .is_err());
    }

    fn app(engine: &InProcessEngine) -> axum::Router {
        engine
            .router(
                LocalAuth::new(BearerToken::from_hex(&"ab".repeat(32)).unwrap(), 31234).unwrap(),
            )
            .unwrap()
    }
    fn request(spec: &RunSpec, key: &str) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri("/v1/runs")
            .header(header::HOST, "127.0.0.1:31234")
            .header(header::AUTHORIZATION, format!("Bearer {}", "ab".repeat(32)))
            .header(header::CONTENT_TYPE, "application/json")
            .header("idempotency-key", key)
            .body(Body::from(serde_json::to_vec(spec).unwrap()))
            .unwrap()
    }
    async fn body(response: Response) -> Value {
        serde_json::from_slice(&to_bytes(response.into_body(), 1 << 20).await.unwrap()).unwrap()
    }

    struct InlineQ1Native {
        calls: Arc<std::sync::atomic::AtomicUsize>,
    }
    impl crate::discover::r#loop::Executor for InlineQ1Native {
        fn evaluate(&self, request: &crate::worker::Request) -> crate::worker::Response {
            use crate::worker::Response;
            self.calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let matrix = jmat::Matrix::open(std::fs::read(&request.matrix_path).unwrap()).unwrap();
            match crate::worker::evaluate::native(request, &matrix) {
                Ok(scores) => Response {
                    attempt: request.attempt.clone(),
                    scores: scores.values,
                    fold_scores: scores.fold_scores,
                    predictions_digest: scores.predictions_digest,
                    wall_ms: 0,
                    failure: None,
                },
                Err(error) => Response::failed(request.attempt.clone(), error, 0),
            }
        }
        fn fitted_model(&self, request: &crate::worker::Request) -> anyhow::Result<Vec<u8>> {
            let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/q1-native");
            std::fs::create_dir_all(&root)?;
            let directory = tempfile::tempdir_in(root)?;
            let output = directory.path().join("model.json");
            let mut request = request.clone();
            request.fold_spec["refit_output"] = json!(output);
            request.fold_spec["refit_max_bytes"] = json!(16 * 1024 * 1024);
            let response = self.evaluate(&request);
            anyhow::ensure!(response.failure.is_none(), "native fixture refit failed");
            let value: Value = serde_json::from_slice(&std::fs::read(output)?)?;
            let model = value["model_json"].as_str().unwrap();
            anyhow::ensure!(
                value["attempt"] == serde_json::to_value(&request.attempt)?
                    && value["model_digest"] == hash(model.as_bytes())
                    && value["predictions_digest"] == response.predictions_digest,
                "refit binding differs"
            );
            Ok(model.as_bytes().to_vec())
        }
    }

    #[test]
    fn q1_expired_policy_never_invokes_the_evaluator() {
        use crate::discover::r#loop::Executor;
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let executor = Q1PinnedExecutor {
            inner: InlineQ1Native {
                calls: Arc::clone(&calls),
            },
            folds: json!({}),
            started_at: std::time::Instant::now() - Q1_WALL_BUDGET,
            attempts: Mutex::new(vec![]),
        };
        let request = crate::worker::Request {
            attempt: crate::worker::Attempt {
                run: jope_api::ids::RunId::new(),
                round: 0,
                slot: 0,
                attempt: 1,
            },
            matrix_path: PathBuf::new(),
            fold_spec: json!({}),
            dsl: "(pipe X (ridge))".into(),
            objective: json!("rmse"),
            seed: 0,
            limits: crate::worker::Limits {
                wall_ms: 600_000,
                rss_bytes: WORKER_RAM,
            },
        };
        let response = executor.evaluate(&request);
        assert_eq!(
            response.failure,
            Some(crate::worker::WorkerFailure::Timeout)
        );
        assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 0);
        let attempts = executor.attempts.into_inner().unwrap();
        assert_eq!(attempts.len(), 1);
        assert_eq!(attempts[0].1.outcome, EvaluationAttemptOutcome::Failed);
        assert!(attempts[0].1.loss.is_none() && attempts[0].1.fold_scores.is_empty());
    }

    struct Q1AllFailed;
    impl crate::discover::r#loop::Executor for Q1AllFailed {
        fn evaluate(&self, request: &crate::worker::Request) -> crate::worker::Response {
            crate::worker::Response::failed(
                request.attempt.clone(),
                crate::worker::WorkerFailure::NonFinite,
                0,
            )
        }
        fn fitted_model(&self, _: &crate::worker::Request) -> anyhow::Result<Vec<u8>> {
            panic!("failed candidates cannot be refitted")
        }
    }

    #[test]
    fn q1_all_failed_attempts_commit_terminal_replayable_actual_counts() {
        let (directory, service) = q1_service_fixture();
        let request = q1_request(directory.path(), &service.store);
        let bytes = service
            .q1_evaluate_using(request.clone(), || Ok(Q1AllFailed))
            .unwrap();
        let receipt: EvaluationReceipt = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(receipt.charged, 24);
        assert_eq!(receipt.attempts.len(), 24);
        assert!(receipt.champion_program.is_none());
        assert!(receipt
            .attempts
            .iter()
            .enumerate()
            .all(|(number, a)| a.attempt == number as u64
                && a.outcome == EvaluationAttemptOutcome::Failed
                && a.failure.is_some()
                && a.loss.is_none()
                && a.fold_scores.is_empty()));
        let command = service
            .store
            .get(&q1_command_key(&receipt.request_sha256))
            .unwrap()
            .unwrap();
        assert_eq!(command.data["status"], "completed");
        let run: jope_api::ids::RunId = command.data["run_id"].as_str().unwrap().parse().unwrap();
        let snapshot = service.journal.snapshot(&run).unwrap().unwrap();
        assert_eq!(snapshot.state, RunState::Failed);
        assert!(snapshot.pending.is_none());
        assert_eq!(snapshot.budget_used.evaluations.0, 24);
        let replay = service
            .q1_evaluate_using(request, || -> anyhow::Result<Q1AllFailed> {
                panic!("terminal replay must never invoke an executor factory")
            })
            .unwrap();
        assert_eq!(bytes, replay);
    }

    #[test]
    fn q1_deterministic_search_stop_commits_only_actual_attempts() {
        let (directory, service) = q1_service_fixture();
        let request_value = q1_request(directory.path(), &service.store);
        let request: Q1Request = serde_json::from_value(request_value).unwrap();
        let manifest =
            load_q1_manifest(directory.path(), &request.pins["protocol_sha256"]).unwrap();
        let dataset_binding_sha256 = manifest.datasets[&request.dataset.id]
            .dataset_binding_sha256
            .clone();
        let run_id = jope_api::ids::RunId::new();
        assert!(service
            .reserve_q1_request(&request, &run_id)
            .unwrap()
            .is_none());
        let attempt = jope_api::EvaluationAttempt {
            attempt: 0,
            outcome: EvaluationAttemptOutcome::Failed,
            fold_scores: vec![],
            loss: None,
            program_sha256: None,
            failure: Some(jope_api::EvaluationFailure {
                code: "non_finite".into(),
                reason: "actual evaluator attempt failed".into(),
            }),
        };
        let receipt = EvaluationReceipt {
            schema: "jope.q1-evaluation-receipt/v1".into(),
            request_sha256: request.request_sha256.clone(),
            identity: EvaluationIdentity {
                source_commit: request.source_commit.clone(),
                plan_sha256: request.plan_sha256.clone(),
                pins: request.pins.clone(),
                dataset_id: request.dataset.id.clone(),
                dataset_sha256: request.dataset.dataset_sha256.clone(),
                cut_sha256: request.dataset.cut_sha256.clone(),
                guidance: request.guidance.clone(),
                metric: request.metric.clone(),
                seed: request.seed,
                requested_budget: request.budget,
                dataset_binding_sha256,
            },
            charged: 1,
            attempts: vec![attempt],
            champion_program: None,
            adapter: "test".into(),
        };
        let bytes = serde_json::to_vec(&receipt).unwrap();
        service
            .commit_q1_receipt(
                &request,
                &run_id,
                receipt.clone(),
                bytes.clone(),
                Q1ReceiptCompletion {
                    attempt_events: vec![],
                    elapsed: std::time::Duration::from_millis(5),
                    terminal_failure: Some("synthetic deterministic discovery stop".into()),
                },
            )
            .unwrap();

        let snapshot = service.journal.snapshot(&run_id).unwrap().unwrap();
        assert_eq!(snapshot.state, RunState::Failed);
        assert_eq!(snapshot.pending, None);
        assert_eq!(snapshot.budget_used.evaluations.0, 1);
        assert_eq!(snapshot.evaluation_receipt, Some(receipt));
        let command = service
            .store
            .get(&q1_command_key(&request.request_sha256))
            .unwrap()
            .unwrap();
        assert_eq!(command.data["status"], "completed");
        assert_eq!(
            command.data["terminal_failure"],
            "synthetic deterministic discovery stop"
        );
        assert_eq!(
            command.data["receipt_bytes"],
            String::from_utf8(bytes.clone()).unwrap()
        );
        assert_eq!(q1_replay_or_unknown(command.data).unwrap(), Some(bytes));
    }

    fn q1_service_fixture() -> (tempfile::TempDir, RunService) {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/q1-native");
        std::fs::create_dir_all(&root).unwrap();
        let directory = tempfile::tempdir_in(root).unwrap();
        let lease = Arc::new(EngineLock::acquire(directory.path()).unwrap());
        let store = Arc::new(Store::open(directory.path().join("store")).unwrap());
        let config = Config::load(
            &directory.path().join("config.toml"),
            &Default::default(),
            &Default::default(),
        )
        .unwrap();
        let service = RunService::new(
            store,
            config,
            directory.path().to_path_buf(),
            std::env::current_exe().unwrap(),
            Arc::new(EngineActivity::default()),
            lease,
        )
        .unwrap();
        (directory, service)
    }

    fn q1_request(directory: &std::path::Path, store: &Store) -> Value {
        let train = b"x,target\n0,0\n1,1\n2,2\n3,3\n4,4\n5,5\n";
        let blob = store.publish_blob(train.as_slice()).unwrap();
        let protocol_sha = "a".repeat(64);
        let plan_sha = "b".repeat(64);
        let source_commit = "c".repeat(40);
        let pins = [
            "bundle_sha256",
            "dope_sha256",
            "evaluator_sha256",
            "vocabulary_sha256",
            "protocol_sha256",
            "training_family_manifest_sha256",
        ]
        .into_iter()
        .map(|key| {
            (
                key.to_owned(),
                if key == "protocol_sha256" {
                    protocol_sha.clone()
                } else {
                    "d".repeat(64)
                },
            )
        })
        .collect::<std::collections::BTreeMap<_, _>>();
        let mut binding = Q1DatasetBinding {
            id: "0123456789abcdef".into(),
            task: "regression".into(),
            dataset_sha256: "0123456789abcdef".to_owned() + &"e".repeat(48),
            family: "f".repeat(40),
            family_authoritative: true,
            rows: 10,
            features: 1,
            cut_sha256: String::new(),
            holdout_group: "sealed_meta_test".into(),
            catalog_version: "v246".into(),
            catalog_record_sha256: "2".repeat(64),
            dataset_binding_sha256: String::new(),
            target: "target".into(),
            feature_columns: vec!["x".into()],
            train_rows: 6,
            train_blob_sha256: hash(train),
            train_blob_bytes: blob.bytes,
            folds: vec![
                Q1Fold {
                    repeat: 0,
                    train: vec![0, 1, 2],
                    validation: vec![3, 4, 5],
                },
                Q1Fold {
                    repeat: 0,
                    train: vec![3, 4, 5],
                    validation: vec![0, 1, 2],
                },
            ],
        };
        binding.cut_sha256 = q1_cut_sha256(&binding).unwrap();
        let mut binding_value = serde_json::to_value(&binding).unwrap();
        binding_value
            .as_object_mut()
            .unwrap()
            .remove("dataset_binding_sha256");
        binding.dataset_binding_sha256 = hash(&serde_json::to_vec(&binding_value).unwrap());
        let manifest = json!({
            "schema":"jope.q1-protocol-manifest/v1",
            "protocol_sha256":protocol_sha,
            "source_commit":source_commit,
            "plan_sha256":plan_sha,
            "pins":pins,
            "datasets":{"0123456789abcdef":binding}
        });
        let protocols = directory.join("q1/protocols");
        std::fs::create_dir_all(&protocols).unwrap();
        std::fs::write(
            protocols.join(format!("{}.json", "a".repeat(64))),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let mut request = json!({
            "schema":"jope.q1-request/v1",
            "mode":"qualification",
            "source_commit":source_commit,
            "plan_sha256":plan_sha,
            "pins":pins,
            "metric":"rmse",
            "dataset":{
                "id":"0123456789abcdef",
                "task":"regression",
                "dataset_sha256":"0123456789abcdefe".to_owned() + &"e".repeat(47),
                "family":"f".repeat(40),
                "family_authoritative":true,
                "rows":10,
                "features":1,
                "cut_sha256":binding.cut_sha256,
                "holdout_group":"sealed_meta_test"
            },
            "guidance":"heuristic",
            "seed":0,
            "budget":24
        });
        // Match the catalog's full dataset identity; the test partition is never stored here.
        request["dataset"]["dataset_sha256"] =
            json!("0123456789abcdef".to_owned() + &"e".repeat(48));
        let sha = canonical_sha256(&request).unwrap();
        request["request_sha256"] = json!(sha);
        request
    }

    fn q1_http_request(body: &Value) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri("/v1/q1/evaluations")
            .header(header::HOST, "127.0.0.1:31234")
            .header(header::AUTHORIZATION, format!("Bearer {}", "ab".repeat(32)))
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_vec(body).unwrap()))
            .unwrap()
    }

    #[tokio::test]
    #[ignore = "requires the authenticated signed r782 fixture and a source-bound native worker"]
    async fn q1_admitted_r782_learned_request_executes_real_guidance() {
        use crate::{ranker::cuda::release::load_r782_release, run::learned::AdmittedRanker};
        use std::sync::atomic::Ordering;
        let assets = PathBuf::from(std::env::var_os("JOPE_AUTHENTICATED_BUNDLE_DIR").unwrap());
        let worker = PathBuf::from(
            std::env::var_os("JOPE_Q1_WORKER_BIN").expect("source-bound native worker required"),
        );
        let ranker = AdmittedRanker::r782(load_r782_release(&assets).unwrap()).unwrap();
        let forwards = ranker.execution_counter();
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/q1-native");
        std::fs::create_dir_all(&root).unwrap();
        let directory = tempfile::tempdir_in(root).unwrap();
        let engine = InProcessEngine::open_with_ranker(directory.path(), &worker, ranker).unwrap();
        let mut request = q1_request(directory.path(), engine.store());
        request["pins"]["bundle_sha256"] =
            json!("94fd9587675c7806bd0a0582b6e97540cd2fdc7ebbc51958411b905185cad85c");
        request["pins"]["dope_sha256"] =
            json!("c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814");
        let path = directory
            .path()
            .join("q1/protocols")
            .join(format!("{}.json", "a".repeat(64)));
        let mut manifest: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        manifest["pins"] = request["pins"].clone();
        std::fs::write(path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        for guidance in ["heuristic", "learned"] {
            request["guidance"] = json!(guidance);
            request.as_object_mut().unwrap().remove("request_sha256");
            request["request_sha256"] = json!(canonical_sha256(&request).unwrap());
            let before = forwards.load(Ordering::Relaxed);
            let response = app(&engine)
                .oneshot(q1_http_request(&request))
                .await
                .unwrap();
            let status = response.status();
            let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
            assert_eq!(
                status,
                StatusCode::OK,
                "{}",
                String::from_utf8_lossy(&bytes)
            );
            let receipt: EvaluationReceipt = serde_json::from_slice(&bytes).unwrap();
            if let Some(output) = std::env::var_os("JOPE_Q1_TEST_RECEIPTS") {
                let output = PathBuf::from(output);
                std::fs::create_dir_all(&output).unwrap();
                std::fs::write(
                    output.join(format!("request-{guidance}.json")),
                    serde_json::to_vec_pretty(&request).unwrap(),
                )
                .unwrap();
                std::fs::write(output.join(format!("receipt-{guidance}.json")), &bytes).unwrap();
            }
            assert_eq!(receipt.identity.guidance, guidance);
            assert_eq!(receipt.charged, 24);
            assert_eq!(receipt.attempts.len(), 24);
            let completed = receipt
                .attempts
                .iter()
                .filter(|a| a.outcome == EvaluationAttemptOutcome::Completed)
                .count();
            assert!(completed > 0);
            let after = forwards.load(Ordering::Relaxed);
            if guidance == "learned" {
                assert!(after > before, "no actual neural guidance");
            } else {
                assert_eq!(after, before, "heuristic arm invoked neural guidance");
            }
            let replay = app(&engine)
                .oneshot(q1_http_request(&request))
                .await
                .unwrap();
            assert_eq!(replay.status(), StatusCode::OK);
            assert_eq!(bytes, to_bytes(replay.into_body(), 1 << 20).await.unwrap());
            assert_eq!(
                forwards.load(Ordering::Relaxed),
                after,
                "replay evaluated again"
            );
            println!(
                "{}",
                json!({"guidance":guidance,"charged":receipt.charged,
                "completed":completed,"actual_neural_forwards":after-before,
                "fixture_only":true,"sealed_payloads_read":0})
            );
        }
        request["pins"]["bundle_sha256"] = json!("e".repeat(64));
        request.as_object_mut().unwrap().remove("request_sha256");
        request["request_sha256"] = json!(canonical_sha256(&request).unwrap());
        let response = app(&engine)
            .oneshot(q1_http_request(&request))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(body(response).await["code"], "INVALID_INPUT");
        assert!(engine
            .store()
            .get(&q1_command_key(request["request_sha256"].as_str().unwrap()))
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn q1_learned_without_admitted_model_refuses_before_reservation() {
        let (directory, engine, _) = fixture();
        let mut request = q1_request(directory.path(), engine.store());
        request["guidance"] = json!("learned");
        request.as_object_mut().unwrap().remove("request_sha256");
        request["request_sha256"] = json!(canonical_sha256(&request).unwrap());
        let response = app(&engine)
            .oneshot(q1_http_request(&request))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(body(response).await["code"], "CAPABILITY_UNAVAILABLE");
        assert!(engine
            .store()
            .get(&q1_command_key(request["request_sha256"].as_str().unwrap()))
            .unwrap()
            .is_none());
        assert!(!directory.path().join("q1/search").exists());
    }

    #[test]
    fn q1_development_fit_scope_keeps_request_identity_validation() {
        let (directory, service) = q1_service_fixture();
        let mut request = q1_request(directory.path(), &service.store);
        request["dataset"]["holdout_group"] = json!("development_fit");
        request.as_object_mut().unwrap().remove("request_sha256");
        request["request_sha256"] = json!(canonical_sha256(&request).unwrap());
        let parsed: Q1Request = serde_json::from_value(request.clone()).unwrap();
        validate_q1_request(&parsed, &request).unwrap();
        request["dataset"]["holdout_group"] = json!("unapproved_group");
        request.as_object_mut().unwrap().remove("request_sha256");
        request["request_sha256"] = json!(canonical_sha256(&request).unwrap());
        let parsed: Q1Request = serde_json::from_value(request.clone()).unwrap();
        assert!(validate_q1_request(&parsed, &request).is_err());
    }

    #[tokio::test]
    async fn q1_receipt_uses_real_fold_values_and_replays_identical_durable_bytes() {
        let (directory, service) = q1_service_fixture();
        let request_body = q1_request(directory.path(), &service.store);
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let first_bytes = service
            .q1_evaluate_using(request_body.clone(), || {
                Ok(InlineQ1Native {
                    calls: Arc::clone(&calls),
                })
            })
            .unwrap();
        let receipt: EvaluationReceipt = serde_json::from_slice(&first_bytes).unwrap();
        assert_eq!(receipt.charged, 24);
        assert_eq!(receipt.attempts.len(), 24);
        let completed: Vec<_> = receipt
            .attempts
            .iter()
            .filter(|attempt| attempt.outcome == EvaluationAttemptOutcome::Completed)
            .collect();
        assert!(!completed.is_empty());
        assert!(completed
            .iter()
            .all(|attempt| attempt.fold_scores.len() == 2
                && attempt.loss.is_some()
                && attempt.program_sha256.is_some()
                && attempt.failure.is_none()));
        let programs: std::collections::BTreeSet<_> = completed
            .iter()
            .map(|attempt| attempt.program_sha256.as_ref().unwrap())
            .collect();
        assert!(
            programs.len() > 1,
            "Q1 must search rather than repeat one ridge DSL"
        );
        assert!(receipt
            .attempts
            .iter()
            .enumerate()
            .all(|(number, attempt)| attempt.attempt == number as u64
                && (attempt.outcome == EvaluationAttemptOutcome::Completed
                    || (attempt.failure.is_some()
                        && attempt.fold_scores.is_empty()
                        && attempt.loss.is_none()))));

        let before = calls.load(std::sync::atomic::Ordering::Relaxed);
        let replay_bytes = service
            .q1_evaluate_using(request_body, || {
                Ok(InlineQ1Native {
                    calls: Arc::clone(&calls),
                })
            })
            .unwrap();
        assert_eq!(
            calls.load(std::sync::atomic::Ordering::Relaxed),
            before,
            "idempotent replay must not evaluate again"
        );
        assert_eq!(first_bytes, replay_bytes);

        let stored = service
            .store
            .get(&q1_command_key(&receipt.request_sha256))
            .unwrap()
            .unwrap();
        assert_eq!(
            stored.data["receipt_bytes"],
            String::from_utf8(first_bytes.to_vec()).unwrap()
        );
        let run: jope_api::ids::RunId = stored.data["run_id"].as_str().unwrap().parse().unwrap();
        let receipt_row = service
            .store
            .get(&RecordKey::Evaluation {
                run: run.clone(),
                number: 0,
            })
            .unwrap()
            .unwrap();
        assert_eq!(receipt_row.data["request_sha256"], receipt.request_sha256);
        for attempt in 0..24 {
            assert!(service
                .store
                .get(&RecordKey::Evaluation {
                    run: run.clone(),
                    number: attempt + 1
                })
                .unwrap()
                .is_some());
        }
    }

    #[test]
    fn learned_q1_plan_and_folds_are_derived_from_the_pinned_manifest() {
        let (directory, engine, _) = fixture();
        let raw = q1_request(directory.path(), engine.store());
        let request: Q1Request = serde_json::from_value(raw.clone()).unwrap();
        let manifest = load_q1_manifest(
            directory.path(),
            request.pins.get("protocol_sha256").unwrap(),
        )
        .unwrap();
        let binding = manifest.datasets.get(&request.dataset.id).unwrap();

        let run_spec =
            q1_run_spec(&request_with_guidance(request.clone(), "learned"), binding).unwrap();
        assert_eq!(run_spec.guidance, Guidance::Learned);
        assert_eq!(run_spec.landmarks, LandmarkMode::Full);
        assert_eq!(run_spec.device, "cpu");
        validate_q1_learned_folds(binding).unwrap();

        let fold_spec = q1_fold_spec(binding, &"a".repeat(64), true);
        assert_eq!(fold_spec["version"], 2);
        assert_eq!(fold_spec["target"], binding.target);
        assert_eq!(fold_spec["features"], json!(binding.feature_columns));
        assert_eq!(fold_spec["folds"], json!(binding.folds));
        assert!(fold_spec.get("refit_output").is_none());

        let mut caller_controlled = raw;
        caller_controlled["target"] = json!("attacker_target");
        let mut unsigned = caller_controlled.clone();
        unsigned.as_object_mut().unwrap().remove("request_sha256");
        caller_controlled["request_sha256"] = json!(canonical_sha256(&unsigned).unwrap());
        assert!(serde_json::from_value::<Q1Request>(caller_controlled).is_err());
    }

    fn request_with_guidance(mut request: Q1Request, guidance: &str) -> Q1Request {
        request.guidance = guidance.into();
        request
    }

    #[tokio::test]
    async fn learned_q1_requires_a_verified_admitted_ranker() {
        let (directory, engine, _) = fixture();
        let mut request_body = q1_request(directory.path(), engine.store());
        request_body["guidance"] = json!("learned");
        request_body
            .as_object_mut()
            .unwrap()
            .remove("request_sha256");
        let request_sha = canonical_sha256(&request_body).unwrap();
        request_body["request_sha256"] = json!(request_sha);

        let denied = app(&engine)
            .oneshot(q1_http_request(&request_body))
            .await
            .unwrap();
        assert_eq!(denied.status(), StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(body(denied).await["code"], "CAPABILITY_UNAVAILABLE");
        assert!(engine
            .store()
            .get(&q1_command_key(
                request_body["request_sha256"].as_str().unwrap()
            ))
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn actual_store_bound_plan_retries_and_conflicts_without_starting_a_run() {
        let (_directory, engine, spec) = fixture();
        let app = app(&engine);
        let first = app
            .clone()
            .oneshot(request(&spec, "plan-key"))
            .await
            .unwrap();
        assert_eq!(first.status(), StatusCode::OK);
        let first = body(first).await;
        let plan: Plan = serde_json::from_value(first.clone()).unwrap();
        assert_eq!(plan.rows.total.0, 96);
        assert_eq!(plan.rows.fit.0, 76);
        assert_eq!(plan.rows.holdout.0, 20);
        assert_eq!(plan.placement.evaluation, "cpu");
        assert_eq!(plan.placement.workers, 1);
        assert_eq!(
            body(
                app.clone()
                    .oneshot(request(&spec, "plan-key"))
                    .await
                    .unwrap()
            )
            .await,
            first
        );
        let mut changed = spec.clone();
        changed.seed.0 += 1;
        let conflict = app
            .clone()
            .oneshot(request(&changed, "plan-key"))
            .await
            .unwrap();
        assert_eq!(conflict.status(), StatusCode::CONFLICT);
        assert_eq!(body(conflict).await["code"], "IDEMPOTENCY_CONFLICT");
        let mut stale = spec.clone();
        stale.plan_digest = Some("0".repeat(64));
        assert_eq!(
            body(app.oneshot(request(&stale, "stale-plan")).await.unwrap()).await["code"],
            "PLAN_CHANGED"
        );
        let list = Request::builder()
            .uri("/v1/runs?limit=1")
            .header(header::HOST, "127.0.0.1:31234")
            .header(header::AUTHORIZATION, format!("Bearer {}", "ab".repeat(32)))
            .body(Body::empty())
            .unwrap();
        let listed = super::tests::app(&engine).oneshot(list).await.unwrap();
        assert_eq!(listed.status(), StatusCode::OK);
        assert_eq!(body(listed).await, json!({"runs":[],"next_after":null}));
        for raw in [
            r#"{"action":"continue","action":"continue"}"#,
            r#"{"action":"continue","unexpected":true}"#,
            r#"{"action":"continue","expected_revision":"-1"}"#,
        ] {
            let request = Request::builder()
                .method("POST")
                .uri("/v1/runs/run_01ARZ3NDEKTSV4RRFFQ69G5FAV/control")
                .header(header::HOST, "127.0.0.1:31234")
                .header(header::AUTHORIZATION, format!("Bearer {}", "ab".repeat(32)))
                .header(header::CONTENT_TYPE, "application/json")
                .header("idempotency-key", "bad-continue")
                .body(Body::from(raw))
                .unwrap();
            let reply = super::tests::app(&engine).oneshot(request).await.unwrap();
            assert_eq!(reply.status(), StatusCode::BAD_REQUEST);
            assert_eq!(body(reply).await["code"], "INVALID_INPUT");
        }
        assert!(!_directory.path().join("worker-scratch").exists());
    }

    #[tokio::test]
    async fn run_device_errors_are_typed_before_dataset_lookup_or_worker_start() {
        let (directory, engine, mut spec) = fixture();
        spec.dataset = "ds_00112233445566778899aabb".parse().unwrap();
        let app = app(&engine);
        for dry_run in [false, true] {
            spec.dry_run = dry_run;
            for (device, status, code) in [
                (
                    "cuda:0",
                    StatusCode::SERVICE_UNAVAILABLE,
                    "BACKEND_UNAVAILABLE",
                ),
                (
                    "cuda:7",
                    StatusCode::SERVICE_UNAVAILABLE,
                    "BACKEND_UNAVAILABLE",
                ),
                ("cuda:", StatusCode::BAD_REQUEST, "INVALID_INPUT"),
                ("cuda:-1", StatusCode::BAD_REQUEST, "INVALID_INPUT"),
                ("cuda:4294967296", StatusCode::BAD_REQUEST, "INVALID_INPUT"),
                ("gpu", StatusCode::BAD_REQUEST, "INVALID_INPUT"),
            ] {
                spec.device = device.into();
                let denied = app
                    .clone()
                    .oneshot(request(&spec, "device-refusal"))
                    .await
                    .unwrap();
                assert_eq!(denied.status(), status, "{device} dry_run={dry_run}");
                let problem = body(denied).await;
                assert_eq!(problem["code"], code);
                assert!(problem.get("run_id").is_none());
            }
        }
        assert!(!directory.path().join("worker-scratch").exists());
        assert!(!directory.path().join("models").exists());
    }

    #[tokio::test]
    async fn actual_run_route_preserves_auth_host_origin_and_explicit_capability_refusals() {
        let (_directory, engine, spec) = fixture();
        let app = app(&engine);
        for (header_name, value, status) in [
            ("authorization", "Bearer wrong", StatusCode::UNAUTHORIZED),
            ("host", "attacker.invalid:31234", StatusCode::BAD_REQUEST),
            ("origin", "https://example.invalid", StatusCode::BAD_REQUEST),
        ] {
            let mut req = request(&spec, "denied");
            req.headers_mut()
                .insert(header_name, axum::http::HeaderValue::from_static(value));
            let denied = app.clone().oneshot(req).await.unwrap();
            assert_eq!(denied.status(), status);
            assert_eq!(denied.headers()["jope-api-version"], "1.0");
        }
        let mut unsupported = spec.clone();
        unsupported.guidance = Guidance::Learned;
        let denied = app
            .clone()
            .oneshot(request(&unsupported, "learned"))
            .await
            .unwrap();
        assert_eq!(denied.status(), StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(body(denied).await["code"], "CAPABILITY_UNAVAILABLE");
        let missing = Request::builder()
            .uri("/v1/runs/run_01JBD8Q6N4V2T5Y8K3M7P9R1S0/events")
            .header(header::HOST, "127.0.0.1:31234")
            .header(header::AUTHORIZATION, format!("Bearer {}", "ab".repeat(32)))
            .body(Body::empty())
            .unwrap();
        let missing = app.oneshot(missing).await.unwrap();
        assert_eq!(missing.status(), StatusCode::NOT_FOUND);
        assert_eq!(body(missing).await["code"], "NOT_FOUND");
    }
    #[test]
    fn requested_export_languages_filter_the_actual_native_seed_and_mutations() {
        let allowed = run_language_filter(&[Language::Rust, Language::Python]).unwrap();
        let space = CpuRunSpace {
            allowed: allowed.clone(),
            #[cfg(all(feature = "failpoints", debug_assertions))]
            pub06_values: None,
        };
        let seeds = space.seeds().unwrap();
        assert_eq!(seeds.len(), 1);
        let mut rng = rand_chacha::ChaCha12Rng::seed_from_u64(73);
        let proposals = space.propose(&seeds, 16, &mut rng).unwrap();
        assert!(!proposals.is_empty());
        for dsl in seeds.iter().chain(&proposals) {
            assert!(allowed.accepts_seed(&feat_math::GenomeIr::parse(dsl).unwrap()));
        }
        assert!(space.canonical("(sym affine \"c0\")").is_err());
        for languages in [
            vec![],
            vec![Language::Javascript],
            vec![Language::Typescript],
        ] {
            assert!(run_language_filter(&languages).is_err());
        }
    }

    #[tokio::test]
    async fn python_and_rust_plan_uses_the_declared_capability_intersection() {
        let (directory, engine, mut spec) = fixture();
        spec.languages = vec![Language::Rust, Language::Python];
        let reply = app(&engine)
            .oneshot(request(&spec, "portable-plan"))
            .await
            .unwrap();
        assert_eq!(reply.status(), StatusCode::OK);
        let plan: Plan = serde_json::from_value(body(reply).await).unwrap();
        assert_eq!(plan.languages.requested, spec.languages);
        assert_eq!(
            plan.languages.excluded_operations,
            run_language_filter(&spec.languages).unwrap().excluded_ops()
        );
        assert!(!plan.languages.intersection_empty);
        assert!(!directory.path().join("worker-scratch").exists());
    }
    #[tokio::test]
    async fn full_landmarks_plan_measures_real_fit_and_progressive_stays_unavailable() {
        let (directory, engine, mut spec) = fixture();
        spec.landmarks = LandmarkMode::Full;
        spec.guidance = Guidance::Auto;
        spec.languages = vec![Language::Rust, Language::Python];
        let app = app(&engine);
        let reply = app
            .clone()
            .oneshot(request(&spec, "full-plan"))
            .await
            .unwrap();
        assert_eq!(reply.status(), StatusCode::OK);
        let value = body(reply).await;
        let plan: Plan = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(plan.landmarks.mode, LandmarkMode::Full);
        assert_eq!(plan.landmarks.sample_rows, 76);
        assert_eq!(plan.guidance.value, Guidance::Heuristic);
        assert_eq!(plan.placement.dope, "none");
        assert_eq!(plan.languages.requested, spec.languages);
        assert_eq!(
            body(
                app.clone()
                    .oneshot(request(&spec, "full-plan"))
                    .await
                    .unwrap()
            )
            .await,
            value
        );
        spec.landmarks = LandmarkMode::Progressive;
        let denied = app
            .oneshot(request(&spec, "progressive-plan"))
            .await
            .unwrap();
        assert_eq!(denied.status(), StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(body(denied).await["code"], "CAPABILITY_UNAVAILABLE");
        assert!(!directory.path().join("worker-scratch").exists());
    }

    #[test]
    fn full_landmark_continuation_rebuilds_only_original_fit_and_refuses_receipt_drift() {
        let (_directory, engine, mut context, plan, fit) = continuation_backend_fixture();
        context.spec.landmarks = LandmarkMode::Full;
        let bytes = engine.store().read_blob(&fit).unwrap();
        let mut backend = BackendPlan::resolve(&context.spec, None, 1).unwrap();
        backend.bind_fit(&bytes, &plan).unwrap();
        let measured = backend.full_fit_landmarks.clone().unwrap();
        assert_eq!(measured.fit_sha256, plan.matrix_sha256);
        assert_eq!(measured.fit_rows, 96);
        assert_eq!(measured.sampled_fit_rows.len(), 96);
        assert!(measured.sampled_fit_rows.iter().all(|&r| r < 96));
        assert!(!measured.features.contains(&measured.target));
        assert_eq!(measured.vector.len(), 256);
        assert_eq!(measured.columns.len(), 64 * 17);
        assert!(measured
            .vector
            .iter()
            .chain(&measured.columns)
            .all(|v| v.is_finite()));
        assert!(measured.vector.iter().any(|v| *v != 0.));
        context.landmarks = Some(measured);
        let mut context: ContinuationBackend =
            serde_json::from_value(serde_json::to_value(context).unwrap()).unwrap();
        let before = plan.clone();
        let restored = context
            .restore(engine.store(), &plan, std::slice::from_ref(&fit), None)
            .unwrap();
        assert_eq!(restored.full_fit_landmarks, context.landmarks);
        assert_eq!(plan, before);
        context.landmarks.as_mut().unwrap().vector[0] += 1.;
        assert_eq!(
            context
                .restore(engine.store(), &plan, &[fit], None)
                .err()
                .unwrap()
                .code,
            ErrorCode::CapabilityUnavailable
        );
        let mut wrong = plan.clone();
        wrong.matrix_sha256 = "0".repeat(64);
        assert!(crate::run::landmarks::FullFitLandmarks::extract(&bytes, &wrong).is_err());
        wrong = plan;
        wrong.features.push(wrong.target.clone());
        assert!(crate::run::landmarks::FullFitLandmarks::extract(&bytes, &wrong).is_err());
    }
}
