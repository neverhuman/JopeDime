//! Heuristic M1 discovery. The scheduler owns deterministic waves; live workers only
//! change throughput. Final refit/holdout/export qualification belongs to DSC-09.
//!
//! A wave is planned against one committed elite context. Every round checkpoint
//! retains its uncommitted siblings, so recovery halfway through a wave never
//! replans them against a newer context. Only committed events leave this module.

use super::{
    audit::AuditRowId,
    effort::{self, BudgetOverrides},
    elites::{Elite, ElitePool, ScoreOrder},
    folds::{CvPhase, FoldConstraints, FoldPlan},
};
use crate::{
    dataset::jmat::{JmatCell, Matrix},
    events::{EventLog, RunUpdate},
    objective::regression::RegressionObjective,
    public::{phrase, project},
    store::{BlobRef, Record, RecordKey, Store, StoreError},
    worker::{self, Attempt, Request, Response, WorkerFailure, WorkerPool},
};
use anyhow::{bail, ensure, Context, Result};
use jope_api::{
    decimal::DecimalU64,
    ids::{ArtifactId, DatasetId, RunId},
    AlgorithmAcceptedBody, BestAlgorithm, BudgetTotal, BudgetUsed, CompletionReason, Effort,
    EventBody, ExperimentFinishedBody, ExperimentOutcome, ExperimentStartedBody, Nondeterminism,
    PublicEvent, PublicFailure, RunPhase, RunSnapshot, RunState,
};
use rand_chacha::{
    rand_core::{RngCore, SeedableRng},
    ChaCha12Rng,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[cfg(feature = "research")]
use crate::research::capture::{
    self, AuxiliaryAttempt, AuxiliaryResult, CaptureState, CapturedAttempt, CapturedRound,
    Measured, Slate,
};
#[cfg(feature = "research")]
use crate::research::{
    BehaviorPolicy, CaptureContext, ObservedSlate, RecordedOutcome, RecordedPolicyKind,
    ResearchRound, Rung0Measurement,
};

const CHECKPOINT_VERSION: u32 = 2;
const NATIVE_FOLDS_VERSION: u32 = 1;
const NATIVE_ARTIFACT_VERSION: u32 = 1;
const REFIT_HANDOFF_BYTES: u64 = 16 * 1024 * 1024;

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Internal M1 plan, not a new public API or worker wire schema. `matrix_path`
/// names an immutable FIT-only matrix: no HOLDOUT/TEST rows enter the worker.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub run_id: RunId,
    pub dataset_id: DatasetId,
    pub problem_digest: String,
    pub bundle_digest: String,
    pub binary_version: String,
    pub matrix_path: PathBuf,
    pub matrix_sha256: String,
    pub target: String,
    pub features: Vec<String>,
    pub objective: String,
    pub seed: u64,
    pub planned_workers: u32,
    pub effort: Effort,
    pub evaluations: u64,
    pub wall_seconds: u64,
    pub evaluation_wall_ms: u64,
    pub evaluation_rss_bytes: u64,
    #[serde(default)]
    pub groups: Option<Vec<u64>>,
    #[serde(default)]
    pub times: Option<Vec<f64>>,
    #[serde(default)]
    pub gap_rows: usize,
}

impl Plan {
    pub fn rounds_in_flight(&self) -> usize {
        (self.planned_workers / 4).max(1) as usize
    }

    fn limits(&self) -> effort::EffortLimits {
        effort::Effort::from_name(match self.effort {
            Effort::Fast => "fast",
            Effort::Medium => "medium",
            Effort::High => "high",
            Effort::Ultra => "ultra",
        })
        .expect("all effort variants are mapped")
        .limits_with_budget(BudgetOverrides {
            evaluations: Some(self.evaluations),
            wall_seconds: Some(self.wall_seconds),
            ..BudgetOverrides::default()
        })
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.planned_workers > 0 && self.planned_workers <= 1024,
            "invalid planned workers"
        );
        ensure!(
            self.evaluations > 0 && self.wall_seconds > 0,
            "empty discovery budget"
        );
        ensure!(
            self.evaluation_wall_ms > 0 && self.evaluation_rss_bytes > 0,
            "empty worker limit"
        );
        for identity in [
            &self.problem_digest,
            &self.bundle_digest,
            &self.matrix_sha256,
        ] {
            ensure!(
                identity.len() == 64 && identity.bytes().all(|b| b.is_ascii_hexdigit()),
                "invalid plan digest"
            );
        }
        ensure!(!self.binary_version.is_empty(), "binary identity missing");
        let bytes = fs::read(&self.matrix_path)?;
        ensure!(
            digest(&bytes) == self.matrix_sha256,
            "FIT matrix digest mismatch"
        );
        let matrix = Matrix::open(bytes)?;
        ensure!(
            !self.features.is_empty()
                && self.features.iter().collect::<BTreeSet<_>>().len() == self.features.len(),
            "invalid feature selection"
        );
        ensure!(
            !self.features.contains(&self.target),
            "target cannot be a feature"
        );
        for name in self.features.iter().chain(std::iter::once(&self.target)) {
            let column = matrix
                .columns()
                .iter()
                .find(|c| &c.name == name)
                .context("unknown column")?;
            ensure!(
                column.kind == crate::dataset::jmat::ColumnKind::Numeric,
                "M1 native probe requires numeric columns"
            );
        }
        let fitness = jope_runtime::fitness::Fitness::parse(&self.objective)?;
        ensure!(
            fitness.columns().is_empty(),
            "M1 auxiliary-column objectives require their later adapter"
        );
        ensure!(
            !fitness.requires_order() || self.times.is_some(),
            "ordered objective needs chronological folds"
        );
        self.fold_spec(0)?;
        self.fold_spec(1)?;
        Ok(())
    }

    fn fold_spec(&self, rung: u8) -> Result<Value> {
        let matrix = Matrix::open(fs::read(&self.matrix_path)?)?;
        let rows: Vec<_> = (0..matrix.rows())
            .map(|row| AuditRowId {
                dataset: self.dataset_id.clone(),
                row: row as u64,
            })
            .collect();
        let mut folds = vec![];
        let repeats = if rung == 0 {
            1
        } else {
            self.limits().promotion_repeats
        };
        for repeat in 0..repeats {
            let plan = FoldPlan::new(
                rows.clone(),
                FoldConstraints {
                    groups: self.groups.clone(),
                    times: self.times.clone(),
                    gap_rows: self.gap_rows,
                },
                if repeat == 0 {
                    self.seed
                } else {
                    evaluation_seed(self.seed, 0, u32::from(repeat), 255)
                },
            )?;
            let partition = plan.folds(
                if rung == 0 {
                    CvPhase::Rung0
                } else {
                    CvPhase::Rung1
                },
                if rung == 0 {
                    2
                } else {
                    self.limits().rung1_folds as usize
                },
                if rung == 0 { 20_000 } else { 100_000 },
            )?;
            folds.extend(partition.iter().map(|f| {
                json!({
                    "repeat": repeat,
                    "train": f.train().iter().map(|r| r.row).collect::<Vec<_>>(),
                    "validation": f.validation().iter().map(|r| r.row).collect::<Vec<_>>()
                })
            }));
        }
        Ok(json!({
            "version": NATIVE_FOLDS_VERSION,
            "matrix_sha256": self.matrix_sha256,
            "target": self.target,
            "features": self.features,
            "folds": folds
        }))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    pub algorithm_id: String,
    pub dsl: String,
}

/// Context-free policies remain supported through the `RoundAdmission` bridge.
pub trait Admission {
    fn admit(&self, proposals: Vec<String>, explore: bool, rng: &mut ChaCha12Rng) -> Vec<String>;

    #[cfg(feature = "research")]
    fn admit_recorded(
        &self,
        _proposals: Vec<String>,
        _explore: bool,
        _rng: &mut ChaCha12Rng,
        _observed: &ObservedSlate,
    ) -> Result<BehaviorPolicy> {
        bail!("same-invocation admission observations are unavailable")
    }
}

/// Committed rung-0 context. User-objective values are deliberately absent;
/// objective ordering still determines elite membership and search parents.
pub struct RoundMeasurement<'a> {
    pub algorithm_id: &'a str,
    pub dsl: &'a str,
    pub surrogate_fitness: f64,
}

pub struct RoundContext<'a> {
    pub plan: &'a Plan,
    pub round: u64,
    pub measured: Vec<RoundMeasurement<'a>>,
}

/// The immutable capture observation and conditioning that will be persisted
/// with this same invocation. Learned policies must verify both against their
/// actual model inputs before recording costs or consuming selection randomness.
#[cfg(feature = "research")]
pub struct RecordedAdmissionContext<'a> {
    pub slate: &'a ObservedSlate,
    pub conditioning: crate::ranker::stage1b::TokenProjection,
}

/// Fallible admission against the same committed context for every sibling in a
/// wave. Failure aborts planning before any scheduler checkpoint is committed.
pub trait RoundAdmission {
    fn admit_round(
        &self,
        context: &RoundContext<'_>,
        proposals: Vec<String>,
        explore: bool,
        rng: &mut ChaCha12Rng,
    ) -> Result<Vec<String>>;

    #[cfg(feature = "research")]
    fn admit_recorded_round(
        &self,
        _context: &RoundContext<'_>,
        _proposals: Vec<String>,
        _explore: bool,
        _rng: &mut ChaCha12Rng,
        _recorded: &RecordedAdmissionContext<'_>,
    ) -> Result<BehaviorPolicy> {
        bail!("same-invocation admission observations are unavailable")
    }
}

impl<T: Admission> RoundAdmission for T {
    fn admit_round(
        &self,
        _context: &RoundContext<'_>,
        proposals: Vec<String>,
        explore: bool,
        rng: &mut ChaCha12Rng,
    ) -> Result<Vec<String>> {
        Ok(self.admit(proposals, explore, rng))
    }

    #[cfg(feature = "research")]
    fn admit_recorded_round(
        &self,
        _context: &RoundContext<'_>,
        proposals: Vec<String>,
        explore: bool,
        rng: &mut ChaCha12Rng,
        recorded: &RecordedAdmissionContext<'_>,
    ) -> Result<BehaviorPolicy> {
        self.admit_recorded(proposals, explore, rng, recorded.slate)
    }
}

pub struct HeuristicAdmission;
impl Admission for HeuristicAdmission {
    #[cfg(feature = "research")]
    fn admit_recorded(
        &self,
        proposals: Vec<String>,
        explore: bool,
        rng: &mut ChaCha12Rng,
        _observed: &ObservedSlate,
    ) -> Result<BehaviorPolicy> {
        let admitted = self.admit(proposals.clone(), explore, rng);
        let indices = admitted
            .iter()
            .map(|dsl| {
                proposals
                    .iter()
                    .position(|p| p == dsl)
                    .context("admission returned an unknown proposal")
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(BehaviorPolicy {
            kind: RecordedPolicyKind::Heuristic,
            explore,
            slate_costs: None,
            admitted: indices,
        })
    }

    fn admit(
        &self,
        mut proposals: Vec<String>,
        explore: bool,
        rng: &mut ChaCha12Rng,
    ) -> Vec<String> {
        if !explore || proposals.len() <= 4 {
            proposals.truncate(4);
            return proposals;
        }
        let mut rest = proposals.split_off(2);
        for _ in 0..2 {
            let index = sample_index(rng, rest.len());
            proposals.push(rest.remove(index));
        }
        proposals
    }
}

fn sample_index(rng: &mut ChaCha12Rng, len: usize) -> usize {
    let limit = u64::MAX - u64::MAX % len as u64;
    loop {
        let value = rng.next_u64();
        if value < limit {
            return (value % len as u64) as usize;
        }
    }
}

pub trait SearchSpace {
    fn canonical(&self, dsl: &str) -> Result<String>;
    fn seeds(&self) -> Result<Vec<String>>;
    fn propose(
        &self,
        parents: &[String],
        count: usize,
        rng: &mut ChaCha12Rng,
    ) -> Result<Vec<String>>;
}

/// Pure numerical/protocol boundary. Implementations must return the echoed
/// attempt exactly and must not publish store effects.
pub trait Executor: Sync {
    fn evaluate(&self, request: &Request) -> Response;
    fn fitted_model(&self, request: &Request) -> Result<Vec<u8>>;
    fn revoke(&self, _attempt: &Attempt) {}
}

#[derive(Debug)]
struct RefitWorkerError(Response);
impl std::fmt::Display for RefitWorkerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("bounded promoted fit failed")
    }
}
impl std::error::Error for RefitWorkerError {}

#[derive(Debug)]
struct RefitIoError(std::io::Error);
impl std::fmt::Display for RefitIoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "private fit handoff: {}", self.0)
    }
}
impl std::error::Error for RefitIoError {}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RefitEnvelope {
    attempt: Attempt,
    model_json: String,
    model_digest: String,
    predictions_digest: String,
}

/// Linux parent-death containment is tied to the *creating thread*. Keep pool
/// callers alive for the entire run, including between waves and promotions;
/// short-lived scoped callers would kill idle reusable workers on thread exit.
pub struct ProcessExecutor {
    sender: Option<std::sync::mpsc::Sender<ProcessJob>>,
    threads: Vec<std::thread::JoinHandle<()>>,
    pool: Arc<WorkerPool>,
    scratch_root: PathBuf,
}
type ProcessJob = (Request, std::sync::mpsc::SyncSender<Response>);

impl ProcessExecutor {
    pub fn new(executable: &Path, workers: usize, scratch_root: &Path) -> Result<Self> {
        ensure!(workers > 0 && workers <= 1024, "invalid live workers");
        let mut builder = fs::DirBuilder::new();
        builder.recursive(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder.create(scratch_root)?;
        ensure!(
            !fs::symlink_metadata(scratch_root)?.file_type().is_symlink(),
            "private fit scratch root must not be a symlink"
        );
        let scratch_root = fs::canonicalize(scratch_root)?;
        let pool = Arc::new(WorkerPool::new(worker::PoolConfig::new(
            executable.to_path_buf(),
            workers,
        ))?);
        let (sender, receiver) = std::sync::mpsc::channel::<ProcessJob>();
        let receiver = Arc::new(Mutex::new(receiver));
        let mut threads = Vec::new();
        for _ in 0..workers {
            let pool = Arc::clone(&pool);
            let receiver = Arc::clone(&receiver);
            threads.push(std::thread::spawn(move || loop {
                let job = receiver.lock().expect("dispatch mutex poisoned").recv();
                let Ok((request, result)) = job else {
                    break;
                };
                let response = pool.evaluate(&request);
                let _ = result.send(response);
            }));
        }
        Ok(Self {
            sender: Some(sender),
            threads,
            pool,
            scratch_root,
        })
    }
}

impl Drop for ProcessExecutor {
    fn drop(&mut self) {
        self.sender.take();
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}

impl Executor for ProcessExecutor {
    fn evaluate(&self, request: &Request) -> Response {
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        if self
            .sender
            .as_ref()
            .is_none_or(|dispatch| dispatch.send((request.clone(), sender)).is_err())
        {
            return Response::failed(request.attempt.clone(), WorkerFailure::Unavailable, 0);
        }
        receiver.recv().unwrap_or_else(|_| {
            Response::failed(request.attempt.clone(), WorkerFailure::Unavailable, 0)
        })
    }
    fn fitted_model(&self, request: &Request) -> Result<Vec<u8>> {
        let directory = tempfile::Builder::new()
            .prefix("promoted-fit-")
            .tempdir_in(&self.scratch_root)
            .map_err(RefitIoError)?;
        let output = directory.path().join("model.json");
        let mut request = request.clone();
        let fields = request
            .fold_spec
            .as_object_mut()
            .context("fit request needs an object fold specification")?;
        fields.insert("refit_output".into(), json!(output));
        let maximum = REFIT_HANDOFF_BYTES.min(request.limits.rss_bytes);
        fields.insert("refit_max_bytes".into(), json!(maximum));
        // The unchanged pool supplies the same hard wall/RSS limits and fence.
        // Returned training scores are deliberately never used for selection.
        let response = checked_response(&request, self.evaluate(&request));
        if response.failure.is_some() {
            return Err(RefitWorkerError(response).into());
        }
        let invalid = || {
            RefitWorkerError(Response::failed(
                request.attempt.clone(),
                WorkerFailure::Protocol,
                response.wall_ms,
            ))
        };
        let metadata = fs::symlink_metadata(&output).map_err(RefitIoError)?;
        if !metadata.is_file() || metadata.len() > maximum {
            return Err(invalid().into());
        }
        let mut bytes = vec![];
        fs::File::open(&output)
            .map_err(RefitIoError)?
            .take(maximum + 1)
            .read_to_end(&mut bytes)
            .map_err(RefitIoError)?;
        if bytes.len() as u64 > maximum {
            return Err(invalid().into());
        }
        let envelope: RefitEnvelope = serde_json::from_slice(&bytes).map_err(|_| invalid())?;
        if envelope.attempt != request.attempt
            || envelope.predictions_digest != response.predictions_digest
            || digest(envelope.model_json.as_bytes()) != envelope.model_digest
        {
            return Err(invalid().into());
        }
        Ok(envelope.model_json.into_bytes())
    }
    fn revoke(&self, attempt: &Attempt) {
        self.pool.revoke(attempt);
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Champion {
    pub candidate: Candidate,
    pub loss: f64,
    pub value: f64,
    pub predictions_digest: String,
    pub model: BlobRef,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct PendingRound {
    number: u64,
    candidates: Vec<Candidate>,
    #[cfg(feature = "research")]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    capture: Option<BlobRef>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DiscardedEvaluation {
    round: u64,
    slot: u32,
    candidate: Candidate,
    reason: CompletionReason,
    response: Option<Response>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Checkpoint {
    version: u32,
    /// Immutable prepared inputs remain reachable for the run's full lifetime.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    input_roots: Vec<BlobRef>,
    #[cfg(feature = "research")]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    capture: Option<CaptureState>,
    plan: Plan,
    elites: ElitePool,
    programs: BTreeMap<String, String>,
    memo: BTreeSet<String>,
    champion: Option<Champion>,
    pending: Vec<PendingRound>,
    discarded: Vec<DiscardedEvaluation>,
    next_round: u64,
    admitted: u64,
    stalled: u32,
    // A recovery/control transaction allocates a fresh attempt namespace before
    // dispatch. Aborted round attempts can never collide with the next session.
    epoch: u32,
    nondeterminism: Vec<String>,
    elapsed_ms: u64,
    // Durable active-span anchor. Interruption through recovery is conservatively
    // charged; an explicit paused checkpoint has no active span.
    active_since_ms: Option<u64>,
    completed: Option<CompletionReason>,
}

// A monotonic span in production. The research-only unit-test clock also
// controls durable wall anchors, so setup scheduling cannot consume fake budget.
struct ElapsedSpan {
    started: Instant,
    #[cfg(all(test, feature = "research"))]
    clock: Option<(Arc<std::sync::atomic::AtomicU64>, u64)>,
}

#[cfg(all(test, feature = "research"))]
thread_local! {
    static TEST_CLOCK: std::cell::RefCell<Option<Arc<std::sync::atomic::AtomicU64>>> = const { std::cell::RefCell::new(None) };
}

#[cfg(all(test, feature = "research"))]
fn test_clock() -> Option<Arc<std::sync::atomic::AtomicU64>> {
    TEST_CLOCK.with(|clock| clock.borrow().clone())
}

impl ElapsedSpan {
    fn start() -> Self {
        Self {
            started: Instant::now(),
            #[cfg(all(test, feature = "research"))]
            clock: test_clock().map(|clock| {
                let start = clock.load(Ordering::SeqCst);
                (clock, start)
            }),
        }
    }

    fn millis(&self) -> u128 {
        #[cfg(all(test, feature = "research"))]
        if let Some((clock, start)) = &self.clock {
            return u128::from(
                clock
                    .load(Ordering::SeqCst)
                    .checked_sub(*start)
                    .expect("test clock moved backwards"),
            );
        }
        self.started.elapsed().as_millis()
    }
}

fn wall_clock_ms() -> Result<u64> {
    #[cfg(all(test, feature = "research"))]
    if let Some(clock) = test_clock() {
        return Ok(clock.load(Ordering::SeqCst));
    }

    Ok(u64::try_from(
        SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis(),
    )?)
}

impl Checkpoint {
    fn elapsed_at(&self, now: u64) -> Result<u64> {
        let active = self
            .active_since_ms
            .map(|start| {
                now.checked_sub(start)
                    .context("active wall clock moved backwards; resume refused")
            })
            .transpose()?
            .unwrap_or(0);
        self.elapsed_ms
            .checked_add(active)
            .context("wall budget counter exhausted")
    }
}

pub struct Discovery {
    #[cfg(feature = "research")]
    capture_context: Option<Arc<CaptureContext>>,
    store: Arc<Store>,
    log: Arc<EventLog>,
    snapshot: RunSnapshot,
    checkpoint: Checkpoint,
    attempt: u32,
    poisoned: bool,
}

struct StartCommand {
    journal: Arc<EventLog>,
    identity: crate::idempotency::RequestIdentity,
    input_roots: Vec<BlobRef>,
    warnings: Vec<jope_api::Warning>,
    resume: bool,
}

impl Discovery {
    /// Commit initial run state and its public retry receipt in one transaction.
    /// Workers may be dispatched only after this returns successfully.
    pub(crate) fn start_command(
        store: Arc<Store>,
        plan: Plan,
        journal: Arc<EventLog>,
        identity: crate::idempotency::RequestIdentity,
        input_roots: Vec<BlobRef>,
        warnings: Vec<jope_api::Warning>,
    ) -> Result<Self> {
        let start = Some(StartCommand {
            journal,
            identity,
            input_roots,
            warnings,
            resume: false,
        });
        #[cfg(feature = "research")]
        {
            Self::open_inner(store, plan, None, start)
        }
        #[cfg(not(feature = "research"))]
        {
            Self::open_inner(store, plan, start)
        }
    }
    /// Resume an owned paused run and atomically retain its public retry receipt.
    pub(crate) fn continue_command(
        store: Arc<Store>,
        plan: Plan,
        journal: Arc<EventLog>,
        identity: crate::idempotency::RequestIdentity,
    ) -> Result<Self> {
        let command = Some(StartCommand {
            journal,
            identity,
            input_roots: vec![],
            warnings: vec![],
            resume: true,
        });
        #[cfg(feature = "research")]
        {
            Self::open_inner(store, plan, None, command)
        }
        #[cfg(not(feature = "research"))]
        {
            Self::open_inner(store, plan, command)
        }
    }
    /// The caller exclusively owns this run and supplies a fresh worker pool on
    /// reopen. Store::open verifies committed blobs and marks interrupted runs.
    pub fn open(store: Arc<Store>, plan: Plan) -> Result<Self> {
        #[cfg(feature = "research")]
        {
            Self::open_inner(store, plan, None, None)
        }
        #[cfg(not(feature = "research"))]
        {
            Self::open_inner(store, plan, None)
        }
    }

    /// Requires the opaque, reviewed capture context; no production resolver is registered.
    #[cfg(feature = "research")]
    pub fn open_research(
        store: Arc<Store>,
        plan: Plan,
        context: Arc<CaptureContext>,
    ) -> Result<Self> {
        context.validate(&plan)?;
        Self::open_inner(store, plan, Some(context), None)
    }

    fn open_inner(
        store: Arc<Store>,
        plan: Plan,
        #[cfg(feature = "research")] capture_context: Option<Arc<CaptureContext>>,
        start: Option<StartCommand>,
    ) -> Result<Self> {
        plan.validate()?;
        let expected_version = CHECKPOINT_VERSION;
        #[cfg(feature = "research")]
        let expected_version = if capture_context.is_some() {
            capture::CHECKPOINT_VERSION
        } else {
            expected_version
        };
        let log = start
            .as_ref()
            .map(|s| Arc::clone(&s.journal))
            .unwrap_or_else(|| Arc::new(EventLog::new(Arc::clone(&store))));
        let old = log.snapshot(&plan.run_id)?;
        if let Some(command) = &start {
            ensure!(
                if command.resume {
                    old.as_ref()
                        .is_some_and(|snapshot| snapshot.state == RunState::Paused)
                } else {
                    old.is_none()
                },
                "command requires a new run or an explicitly continued paused run"
            );
        }
        let (mut checkpoint, mut snapshot) = if let Some(snapshot) = &old {
            let record = store
                .get(&RecordKey::Run(plan.run_id.clone()))?
                .context("run missing")?;
            let saved: Checkpoint = serde_json::from_value(record.data["checkpoint"].clone())?;
            ensure!(
                saved.version == expected_version && saved.plan == plan,
                "resume plan identity mismatch"
            );
            #[cfg(feature = "research")]
            match (&saved.capture, &capture_context) {
                (Some(state), Some(context)) => state.check_binding(&store, context)?,
                (None, None) => {}
                _ => bail!("capture mode cannot change on reopen"),
            }
            if let Some(c) = &saved.champion {
                store.verify_blob(&c.model)?;
            }
            let recovery = record
                .run
                .as_ref()
                .context("run recovery metadata missing")?;
            ensure!(
                saved.completed.is_some()
                    || saved.active_since_ms.is_some()
                    || (snapshot.state == RunState::Paused && !recovery.interrupted),
                "interrupted wall budget lacks an active-span anchor; resume refused"
            );
            (saved, snapshot.clone())
        } else {
            let checkpoint = Checkpoint {
                version: expected_version,
                input_roots: start
                    .as_ref()
                    .map(|s| s.input_roots.clone())
                    .unwrap_or_default(),
                #[cfg(feature = "research")]
                capture: capture_context
                    .as_deref()
                    .map(|context| CaptureState::new(&store, context))
                    .transpose()?,
                plan: plan.clone(),
                elites: ElitePool::new(ScoreOrder::Minimize, ScoreOrder::Minimize),
                programs: BTreeMap::new(),
                memo: BTreeSet::new(),
                champion: None,
                pending: vec![],
                discarded: vec![],
                next_round: 0,
                admitted: 0,
                stalled: 0,
                epoch: 0,
                nondeterminism: vec![],
                elapsed_ms: 0,
                active_since_ms: None,
                completed: None,
            };
            let snapshot = RunSnapshot {
                run_id: plan.run_id.clone(),
                revision: DecimalU64(1),
                state: RunState::Running,
                phase: RunPhase::Seeding,
                pending: None,
                effort: plan.effort,
                budget_used: BudgetUsed {
                    evaluations: DecimalU64(0),
                    wall_seconds: "0".into(),
                },
                budget_total: BudgetTotal {
                    evaluations: DecimalU64(plan.evaluations),
                    wall_seconds: DecimalU64(plan.wall_seconds),
                },
                best: None,
                event_watermark: DecimalU64(0),
                warnings: start
                    .as_ref()
                    .map(|s| s.warnings.clone())
                    .unwrap_or_default(),
                completion_reason: None,
                evaluation_receipt: None,
            };
            (checkpoint, snapshot)
        };
        #[cfg(feature = "research")]
        if let Some(state) = checkpoint.capture.as_mut() {
            // A reserved request with no durable observation may have executed.
            if state.unknown() {
                state.tainted = true;
            }
        }
        #[cfg(feature = "research")]
        if let Some(capture) = &checkpoint.capture {
            snapshot.budget_total.evaluations = DecimalU64(capture.config.max_attempts);
        }
        let now = wall_clock_ms()?;
        checkpoint.elapsed_ms = checkpoint.elapsed_at(now)?;
        if checkpoint.completed.is_none()
            && checkpoint.elapsed_ms >= plan.wall_seconds.saturating_mul(1000)
        {
            checkpoint.completed = Some(CompletionReason::BudgetTime);
            snapshot.phase = RunPhase::Finalizing;
        }
        if checkpoint.completed.is_some() {
            discard_pending(&mut checkpoint, None, &[], &[], &mut vec![])?;
        }
        checkpoint.active_since_ms = checkpoint.completed.is_none().then_some(now);
        snapshot.budget_used.wall_seconds = format!("{:.3}", checkpoint.elapsed_ms as f64 / 1000.0);
        checkpoint.epoch = checkpoint
            .epoch
            .checked_add(1)
            .context("attempt epochs exhausted")?;
        if old.is_some() {
            snapshot.revision.0 = snapshot
                .revision
                .0
                .checked_add(1)
                .context("revision exhausted")?;
        }
        snapshot.state = RunState::Running;
        let blobs = checkpoint_blobs(&checkpoint)?;
        let initial_response = serde_json::to_value(&snapshot)?;
        // Initialization/recovery is a control transaction, never a synthetic search event.
        let snapshot = log
            .commit(
                RunUpdate {
                    expected_revision: old.map(|s| s.revision.0),
                    snapshot,
                    checkpoint: serde_json::to_value(&checkpoint)?,
                    blobs,
                },
                vec![],
                |tx| {
                    if let Some(start) = &start {
                        tx.record_new_command(
                            &start.identity,
                            crate::idempotency::CommandOutcome::json(
                                if start.resume { 200 } else { 201 },
                                initial_response,
                            )?,
                        )?;
                    }
                    Ok(())
                },
            )?
            .snapshot;
        Ok(Self {
            #[cfg(feature = "research")]
            capture_context,
            store,
            log,
            snapshot,
            checkpoint,
            attempt: 0,
            poisoned: false,
        })
    }

    pub fn champion(&self) -> Option<&Champion> {
        self.checkpoint.champion.as_ref()
    }
    pub fn completed(&self) -> Option<CompletionReason> {
        self.checkpoint.completed
    }
    /// True when an attempted durable commit may have succeeded despite an
    /// error return. Callers must retain an unknown outcome in this case.
    pub fn outcome_unknown(&self) -> bool {
        self.poisoned
    }
    pub fn admitted(&self) -> u64 {
        self.checkpoint.admitted
    }
    pub fn budget_bound_by(&self) -> Option<&'static str> {
        match self.completed() {
            Some(CompletionReason::BudgetTime) => Some("time"),
            Some(CompletionReason::BudgetEvaluations) => Some("evaluations"),
            _ => None,
        }
    }
    pub fn snapshot(&self) -> &RunSnapshot {
        &self.snapshot
    }

    pub fn nondeterminism(&self) -> Nondeterminism {
        Nondeterminism {
            count: DecimalU64(self.checkpoint.nondeterminism.len() as u64),
            experiment_ids: self.checkpoint.nondeterminism.clone(),
        }
    }

    /// Canonical event projection under owner ruling rulings_t1900 R2 (§5.6).
    /// Operational envelopes and receipts retain run identity and elapsed usage.
    pub fn projection(&self) -> Result<Value> {
        let mut after = 0;
        let mut values = vec![];
        while after < self.snapshot.event_watermark.0 {
            let page = self
                .log
                .page(&self.checkpoint.plan.run_id, after, 256, &[])?
                .context("run missing")?;
            after = page.next_after.0;
            for event in page.events {
                let mut value = serde_json::to_value(event)?;
                let envelope = value.as_object_mut().context("event is not an object")?;
                envelope.remove("run_id");
                envelope.remove("time");
                if let Some(warnings) = value
                    .get_mut("body")
                    .and_then(|body| body.get_mut("warnings"))
                    .and_then(Value::as_array_mut)
                {
                    warnings.retain(|warning| warning.get("host") != Some(&Value::Bool(true)));
                }
                if let Some(budget) = value
                    .get_mut("body")
                    .and_then(|body| body.get_mut("budget_used"))
                    .and_then(Value::as_object_mut)
                {
                    budget.remove("wall_seconds");
                }
                if value
                    .get("body")
                    .and_then(|b| b.get("host"))
                    .and_then(Value::as_bool)
                    != Some(true)
                {
                    values.push(value);
                }
            }
        }
        Ok(Value::Array(values))
    }

    /// Pause only at a committed boundary. Pending siblings remain in the
    /// checkpoint; no uncommitted result is installed or publicly visible.
    pub fn pause(&mut self) -> Result<()> {
        let mut next = self.checkpoint.clone();
        next.elapsed_ms = next.elapsed_at(wall_clock_ms()?)?;
        next.active_since_ms = None;
        self.commit_control(next, RunState::Paused)
    }

    fn commit_control(&mut self, next: Checkpoint, state: RunState) -> Result<()> {
        ensure!(!self.poisoned, "commit outcome unknown: reopen the run");
        let mut snapshot = self.snapshot.clone();
        snapshot.state = state;
        snapshot.revision.0 = snapshot
            .revision
            .0
            .checked_add(1)
            .context("revision exhausted")?;
        snapshot.budget_used.wall_seconds = format!("{:.3}", next.elapsed_ms as f64 / 1000.0);
        #[cfg(feature = "research")]
        if next.capture.is_some() {
            snapshot.budget_used.evaluations = DecimalU64(rung0_budget_used(&next));
        }
        if next.completed.is_some() {
            snapshot.phase = RunPhase::Finalizing;
        }
        self.poisoned = true;
        let result = self.log.commit(
            RunUpdate {
                expected_revision: Some(self.snapshot.revision.0),
                snapshot,
                checkpoint: serde_json::to_value(&next)?,
                blobs: checkpoint_blobs(&next)?,
            },
            vec![],
            |_| Ok(()),
        )?;
        self.snapshot = result.snapshot;
        self.checkpoint = next;
        self.poisoned = false;
        Ok(())
    }

    /// Execute a fixed planned wave, or its retained suffix after recovery.
    /// `stop_after_rounds` permits a supervisor to request a bounded pause without
    /// changing the plan or event stream. Already computed siblings are discarded.
    pub fn advance(
        &mut self,
        space: &impl SearchSpace,
        admission: &impl RoundAdmission,
        executor: &impl Executor,
        live_workers: usize,
        stop_after_rounds: usize,
    ) -> Result<Vec<PublicEvent>> {
        ensure!(!self.poisoned, "commit outcome unknown: reopen the run");
        ensure!(
            self.snapshot.state == RunState::Running,
            "continue a paused run by reopening it"
        );
        ensure!(
            live_workers > 0 && live_workers <= 1024 && stop_after_rounds > 0,
            "invalid execution capacity"
        );
        if self.completed().is_some() {
            return Ok(vec![]);
        }
        let started = ElapsedSpan::start();
        let elapsed_before = self.checkpoint.elapsed_at(wall_clock_ms()?)?;
        if elapsed_before >= self.checkpoint.plan.wall_seconds.saturating_mul(1000) {
            let mut next = self.checkpoint.clone();
            next.elapsed_ms = elapsed_before;
            next.active_since_ms = None;
            next.completed = Some(CompletionReason::BudgetTime);
            discard_pending(&mut next, None, &[], &[], &mut vec![])?;
            self.commit_control(next, RunState::Running)?;
            return Ok(vec![]);
        }
        let mut next = self.checkpoint.clone();
        #[cfg(feature = "research")]
        ensure!(
            next.capture.as_ref().is_none_or(|s| !s.tainted),
            "capture has unknown outcomes; retained for inspection only"
        );
        if next.pending.is_empty() {
            #[cfg(feature = "research")]
            if let Some(context) = self.capture_context.as_deref() {
                plan_wave_inner(&mut next, space, admission, Some((&self.store, context)))?;
            } else {
                plan_wave(&mut next, space, admission)?;
            }
            #[cfg(not(feature = "research"))]
            plan_wave(&mut next, space, admission)?;
        }
        #[cfg(feature = "research")]
        let capture_work = if self.capture_context.is_some() {
            let result = self.evaluate_research_wave(&mut next, executor, live_workers);
            if result.is_err() {
                self.poisoned = true;
            }
            Some(result?)
        } else {
            None
        };
        let mut work = vec![];
        #[cfg(feature = "research")]
        let captured = capture_work.is_some();
        #[cfg(not(feature = "research"))]
        let captured = false;
        if !captured {
            for round in &next.pending {
                for (slot, candidate) in round.candidates.iter().enumerate() {
                    work.push(self.request(round.number, slot as u32, 0, candidate)?);
                }
            }
        }
        let outcomes;
        #[cfg(feature = "research")]
        if let Some((requests, responses)) = capture_work {
            work = requests;
            outcomes = responses;
        } else {
            outcomes = parallel_evaluate(executor, &work, live_workers);
        }
        #[cfg(not(feature = "research"))]
        {
            outcomes = parallel_evaluate(executor, &work, live_workers);
        }
        let mut offset = 0;
        let mut published = vec![];
        for _ in 0..stop_after_rounds.min(next.pending.len()) {
            let round = next.pending.remove(0);
            let end = offset + round.candidates.len();
            let mut events = vec![];
            let mut records = vec![];
            for (slot, candidate) in round.candidates.iter().enumerate() {
                let response = &outcomes[offset + slot];
                let id = experiment(&next.plan, round.number, slot, 0)?;
                record_result(&mut next, candidate, response, &id, &mut events)?;
                records.push(json!({"round":round.number,"slot":slot,"rung":0,"candidate":candidate,"worker_attempt":response.attempt,
                    "scores":response.scores,"predictions_digest":response.predictions_digest,"failure":response.failure}));
            }
            let best = round
                .candidates
                .iter()
                .filter_map(|c| {
                    let response =
                        &outcomes[offset + round.candidates.iter().position(|v| v == c)?];
                    response.failure.is_none().then(|| (c, response.scores[1]))
                })
                .min_by(|(a, x), (b, y)| {
                    x.partial_cmp(y)
                        .expect("validated finite objective")
                        .then_with(|| a.algorithm_id.cmp(&b.algorithm_id))
                });
            let mut promoted = false;
            if let Some((candidate, _)) = best.filter(|(c, _)| {
                next.elites
                    .rank_by_objective(&c.algorithm_id)
                    .is_some_and(|rank| rank < 4)
            }) {
                let request = self.request(round.number, 0, 1, candidate)?;
                #[cfg(feature = "research")]
                let mut response =
                    self.evaluate_research_aux(&mut next, executor, request.clone())?;
                #[cfg(not(feature = "research"))]
                let mut response = checked_response(&request, executor.evaluate(&request));
                let cv_response = response.clone();
                let mut refit_failure = None;
                let improving = response.failure.is_none()
                    && next
                        .champion
                        .as_ref()
                        .is_none_or(|c| response.scores[1] < c.loss);
                let bytes = if improving {
                    let refit = self.request(round.number, 0, 2, candidate)?;
                    // Numerical refit failure belongs to this candidate. Store IO
                    // failure below still aborts the round and requires recovery.
                    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        #[cfg(feature = "research")]
                        {
                            self.fit_research_aux(&mut next, executor, refit.clone())
                        }
                        #[cfg(not(feature = "research"))]
                        {
                            executor.fitted_model(&refit)
                        }
                    })) {
                        Ok(Ok(bytes)) if !bytes.is_empty() => Some(bytes),
                        Ok(Err(error)) if error.is::<RefitIoError>() => return Err(error),
                        failure => {
                            let worker = match &failure {
                                Ok(Err(error)) => error
                                    .downcast_ref::<RefitWorkerError>()
                                    .map(|e| e.0.clone()),
                                _ => None,
                            };
                            let kind = worker
                                .as_ref()
                                .and_then(|r| r.failure)
                                .unwrap_or(WorkerFailure::NonFinite);
                            refit_failure = worker;
                            response =
                                Response::failed(request.attempt.clone(), kind, response.wall_ms);
                            None
                        }
                    }
                } else {
                    None
                };
                response.fold_scores = cv_response.fold_scores.clone();
                let id = experiment(&next.plan, round.number, 0, 1)?;
                add_events(&mut next, candidate, &response, &id, &mut events)?;
                records.push(json!({"round":round.number,"slot":0,"rung":1,"candidate":candidate,"worker_attempt":response.attempt,
                    "scores":cv_response.scores,"predictions_digest":cv_response.predictions_digest,
                    "failure":response.failure,"cv_failure":cv_response.failure,"refit_failure":refit_failure}));
                if let Some(bytes) = bytes {
                    // Publication precedes the metadata reference. An aborted transaction
                    // may leave an unreferenced CAS object, never a partial champion.
                    let model = self.store.publish_blob(bytes.as_slice())?;
                    let previous = next.champion.as_ref().map(|c| c.value);
                    let artifact: ArtifactId = format!("art_{}", &digest(&bytes)[..24]).parse()?;
                    next.champion = Some(Champion {
                        candidate: candidate.clone(),
                        loss: response.scores[1],
                        value: response.scores[2],
                        predictions_digest: response.predictions_digest.clone(),
                        model,
                    });
                    let summary =
                        accepted_summary(&next.plan, candidate, response.scores[2], previous)?;
                    events.push(EventBody::AlgorithmAccepted(AlgorithmAcceptedBody {
                        algorithm_id: candidate.algorithm_id.parse()?,
                        metric: public_metric(&next.plan)?,
                        cv_value: response.scores[2],
                        previous_value: previous,
                        summary,
                        model_artifact: Some(artifact),
                        excerpt: String::new(),
                    }));
                    promoted = true;
                }
            }
            next.stalled = if promoted {
                0
            } else {
                next.stalled.saturating_add(1)
            };
            next.next_round = round
                .number
                .checked_add(1)
                .context("round counter exhausted")?;
            next.elapsed_ms =
                elapsed_before.saturating_add(started.millis().min(u128::from(u64::MAX)) as u64);
            next.completed = if next.stalled >= next.plan.limits().patience {
                Some(CompletionReason::Converged)
            } else if next.elapsed_ms >= next.plan.wall_seconds.saturating_mul(1000) {
                Some(CompletionReason::BudgetTime)
            } else if next.pending.is_empty()
                && rung0_budget_used(&next) >= rung0_budget_limit(&next)
            {
                Some(CompletionReason::BudgetEvaluations)
            } else {
                None
            };
            if next.completed.is_some() {
                discard_pending(&mut next, Some(executor), &work, &outcomes, &mut events)?;
            }
            next.active_since_ms = if next.completed.is_none() {
                Some(wall_clock_ms()?)
            } else {
                None
            };
            #[cfg(feature = "research")]
            if let Some(capture) = next.capture.as_mut() {
                if let Err(error) = capture.finish_round(&self.store, round.number) {
                    self.poisoned = true;
                    return Err(error);
                }
            }
            #[cfg(all(test, feature = "research"))]
            if next.capture.is_some() {
                capture::boundary("before_round");
            }
            let committed = self.commit_round(next.clone(), events, records)?;
            #[cfg(all(test, feature = "research"))]
            if next.capture.is_some() {
                capture::boundary("after_round");
            }
            published.extend(committed);
            offset = end;
            if self.completed().is_some() {
                break;
            }
        }
        Ok(published)
    }

    fn request(
        &mut self,
        round: u64,
        slot: u32,
        rung: u8,
        candidate: &Candidate,
    ) -> Result<Request> {
        self.attempt = self
            .attempt
            .checked_add(1)
            .context("session attempt counter exhausted")?;
        let plan = &self.checkpoint.plan;
        Ok(Request {
            attempt: Attempt {
                run: plan.run_id.clone(),
                round,
                slot: slot + u32::from(rung) * 65_536,
                attempt: (u64::from(self.checkpoint.epoch) << 32) | u64::from(self.attempt),
            },
            matrix_path: plan.matrix_path.clone(),
            fold_spec: plan.fold_spec(rung)?,
            dsl: candidate.dsl.clone(),
            objective: json!(plan.objective),
            seed: evaluation_seed(plan.seed, round, slot, rung),
            limits: worker::Limits {
                wall_ms: plan.evaluation_wall_ms,
                rss_bytes: plan.evaluation_rss_bytes,
            },
        })
    }

    fn commit_round(
        &mut self,
        next: Checkpoint,
        events: Vec<EventBody>,
        records: Vec<Value>,
    ) -> Result<Vec<PublicEvent>> {
        let mut snapshot = self.snapshot.clone();
        snapshot.revision.0 += 1;
        snapshot.phase = if next.completed.is_some() {
            RunPhase::Finalizing
        } else {
            RunPhase::Searching
        };
        snapshot.budget_used = BudgetUsed {
            evaluations: DecimalU64(rung0_budget_used(&next)),
            wall_seconds: format!("{:.3}", next.elapsed_ms as f64 / 1000.0),
        };
        snapshot.best = next
            .champion
            .as_ref()
            .map(|c| -> Result<_> {
                Ok(BestAlgorithm {
                    algorithm_id: c.candidate.algorithm_id.parse()?,
                    metric: public_metric(&next.plan)?,
                    cv_value: c.value,
                    model_artifact: Some(format!("art_{}", &hex_blob(&c.model)[..24]).parse()?),
                })
            })
            .transpose()?;
        let run = next.plan.run_id.clone();
        let round = next.next_round - 1;
        let champion = next.champion.clone();
        self.poisoned = true;
        let committed = self.log.commit(RunUpdate {
            expected_revision: Some(self.snapshot.revision.0), snapshot,
            checkpoint: serde_json::to_value(&next)?, blobs: checkpoint_blobs(&next)?,
        }, events, |tx| {
            if let Some(champion) = champion {
                let key = RecordKey::Artifact(format!("art_{}", &hex_blob(&champion.model)[..24]).parse()
                    .map_err(|_| StoreError::Invalid("artifact identity invalid"))?);
                if let Some(old) = tx.get(&key)? {
                    if old.blobs != [champion.model] { return Err(StoreError::Integrity("artifact identity collision")); }
                } else {
                    tx.put(&key, &Record::metadata(json!({"kind":"m1-native-fit", "model_digest":hex_blob(&champion.model)}), vec![champion.model]))?;
                }
            }
            for (slot, data) in records.into_iter().enumerate() {
                let number = round.checked_mul(65_536).and_then(|n| n.checked_add(slot as u64)).ok_or(StoreError::Invalid("evaluation key exhausted"))?;
                let key = RecordKey::Evaluation { run: run.clone(), number };
                if tx.get(&key)?.is_some() { return Err(StoreError::Integrity("round evaluation already committed")); }
                tx.put(&key, &Record::metadata(data, vec![]))?;
            }
            Ok(())
        })?;
        self.snapshot = committed.snapshot;
        self.checkpoint = next;
        self.poisoned = false;
        Ok(committed.events)
    }
}

#[cfg(feature = "research")]
impl Discovery {
    // Publish only the capture journal over the last committed GP state. The
    // local candidate may already contain uncommitted elite/promotion changes.
    fn journal_research(&mut self, next: &mut Checkpoint) -> Result<()> {
        let now = wall_clock_ms()?;
        let mut durable = self.checkpoint.clone();
        durable.capture = next.capture.clone();
        durable.elapsed_ms = durable.elapsed_at(now)?;
        durable.active_since_ms = Some(now);
        next.elapsed_ms = durable.elapsed_ms;
        next.active_since_ms = durable.active_since_ms;
        self.commit_control(durable, RunState::Running)
    }

    fn evaluate_research_wave(
        &mut self,
        next: &mut Checkpoint,
        executor: &impl Executor,
        workers: usize,
    ) -> Result<(Vec<Request>, Vec<Response>)> {
        let context = Arc::clone(
            self.capture_context
                .as_ref()
                .context("capture context missing")?,
        );
        let pending = next.pending.clone();
        let mut fresh = vec![];
        for round in &pending {
            let captured = next
                .capture
                .as_ref()
                .unwrap()
                .rounds
                .get(&round.number)
                .context("pending capture round missing")?
                .clone();
            ensure!(
                round.capture.as_ref() == Some(&captured.slate),
                "pending capture slate mismatch"
            );
            let slate: Slate = CaptureState::read(&self.store, &captured.slate)?;
            if !captured.attempts.is_empty() {
                ensure!(
                    captured.attempts.len() == slate.dispatch.len()
                        && captured.attempts.iter().all(|a| a.result.is_some()),
                    "unknown capture attempt; retry refused"
                );
                continue;
            }
            let mut admitted = BTreeMap::new();
            for (slot, candidate) in round.candidates.iter().enumerate() {
                let index = slate
                    .dsls
                    .iter()
                    .position(|dsl| dsl == &candidate.dsl)
                    .context("admitted candidate absent from slate")?;
                admitted.insert(
                    index,
                    (slot, self.request(round.number, slot as u32, 0, candidate)?),
                );
            }
            for index in &slate.dispatch {
                let (admitted_slot, request) = if let Some((slot, request)) = admitted.remove(index)
                {
                    (Some(slot), request)
                } else {
                    ensure!(
                        *index < 16 && round.number > 0,
                        "label-only slot outside approved v1 bounds"
                    );
                    let state = next.capture.as_mut().unwrap();
                    state.label_counter = state
                        .label_counter
                        .checked_add(1)
                        .context("label attempt counter exhausted")?;
                    let slot = 32 + *index as u32;
                    let request = Request {
                        attempt: Attempt {
                            run: next.plan.run_id.clone(),
                            round: round.number,
                            slot,
                            attempt: (u64::from(next.epoch) << 32) | u64::from(state.label_counter),
                        },
                        matrix_path: next.plan.matrix_path.clone(),
                        fold_spec: next.plan.fold_spec(0)?,
                        dsl: slate.dsls[*index].clone(),
                        objective: json!(next.plan.objective),
                        seed: context.provider.label_seed(round.number, *index)?,
                        limits: worker::Limits {
                            wall_ms: next.plan.evaluation_wall_ms,
                            rss_bytes: next.plan.evaluation_rss_bytes,
                        },
                    };
                    (None, request)
                };
                let capture = next.capture.as_mut().unwrap();
                let reference = capture.publish(&self.store, &request)?;
                let attempts = &mut capture.rounds.get_mut(&round.number).unwrap().attempts;
                let position = attempts.len();
                attempts.push(CapturedAttempt {
                    proposal_index: *index,
                    admitted_slot,
                    request: reference,
                    result: None,
                });
                fresh.push((round.number, position, request));
            }
            ensure!(
                admitted.is_empty(),
                "admitted request not reserved for capture"
            );
        }
        if !fresh.is_empty() {
            let now = wall_clock_ms()?;
            next.elapsed_ms = next.elapsed_at(now)?;
            next.active_since_ms = Some(now);
            if next.elapsed_ms >= next.plan.wall_seconds.saturating_mul(1000) {
                next.completed = Some(CompletionReason::BudgetTime);
                next.active_since_ms = None;
                discard_pending(next, None, &[], &[], &mut vec![])?;
                self.commit_control(next.clone(), RunState::Running)?;
                return Ok((vec![], vec![]));
            }
            // Slates, exact requests, roots and every reservation are durable
            // before the first executor call. No GP result is installed here.
            #[cfg(test)]
            capture::boundary("before_reservation");
            self.commit_control(next.clone(), RunState::Running)?;
            #[cfg(test)]
            capture::boundary("after_reservation");
            let requests: Vec<_> = fresh.iter().map(|(_, _, r)| r.clone()).collect();
            parallel_evaluate_research(
                executor,
                &requests,
                workers,
                next.plan
                    .wall_seconds
                    .saturating_mul(1000)
                    .saturating_sub(next.elapsed_at(wall_clock_ms()?)?),
                |index, response, wall_ms| {
                    let (number, position, request) = &fresh[index];
                    ensure!(
                        response.failure != Some(WorkerFailure::StaleAttempt),
                        "stale outcome is unknown, not a failed label"
                    );
                    let measurement = if let Some(code) = response.failure {
                        Rung0Measurement::Failed { code }
                    } else {
                        let measurement = context.provider.measurement(request, &response)?;
                        ensure!(
                            matches!(&measurement, Rung0Measurement::Success { surrogate, objective, .. }
                        if *surrogate == response.scores[0] && *objective == response.scores[1]),
                            "actual surrogate/objective or gate unavailable"
                        );
                        measurement
                    };
                    let capture = next.capture.as_mut().unwrap();
                    let attempt = &capture.rounds[number].attempts[*position];
                    ensure!(attempt.result.is_none(), "duplicate capture outcome");
                    let measured = Measured {
                        response,
                        outcome: RecordedOutcome {
                            proposal_index: attempt.proposal_index,
                            wall_ms,
                            measurement,
                        },
                    };
                    let reference = capture.publish(&self.store, &measured)?;
                    capture.rounds.get_mut(number).unwrap().attempts[*position].result =
                        Some(reference);
                    self.journal_research(next)?;
                    #[cfg(test)]
                    capture::boundary("after_outcome");
                    Ok(())
                },
            )?;
        }
        let capture = next.capture.as_ref().unwrap();
        let mut requests = vec![];
        let mut responses = vec![];
        for pending in &next.pending {
            let round = &capture.rounds[&pending.number];
            let mut ordered = BTreeMap::new();
            for attempt in &round.attempts {
                let request: Request = CaptureState::read(&self.store, &attempt.request)?;
                let measured: Measured = CaptureState::read(
                    &self.store,
                    attempt.result.as_ref().context("unknown capture outcome")?,
                )?;
                ensure!(
                    request.attempt == measured.response.attempt
                        && measured.response.valid()
                        && (measured.response.failure.is_some()
                            || measured.response.scores.len() == 3),
                    "invalid saved capture response"
                );
                if let Some(slot) = attempt.admitted_slot {
                    ordered.insert(slot, (request, measured.response));
                }
            }
            ensure!(
                ordered.len() == pending.candidates.len(),
                "admitted capture inventory mismatch"
            );
            for (expected, (slot, (request, response))) in ordered.into_iter().enumerate() {
                ensure!(
                    slot == expected && request.dsl == pending.candidates[slot].dsl,
                    "admitted capture order mismatch"
                );
                requests.push(request);
                responses.push(response);
            }
        }
        Ok((requests, responses))
    }

    fn prepare_research_aux(
        &mut self,
        next: &mut Checkpoint,
        request: Request,
    ) -> Result<(Request, Option<usize>)> {
        let wall_available = next.capture.is_none()
            || next.elapsed_at(wall_clock_ms()?)? < next.plan.wall_seconds.saturating_mul(1000);
        let Some(capture) = next.capture.as_mut() else {
            return Ok((request, None));
        };
        for (index, attempt) in capture.auxiliary.iter().enumerate() {
            let saved: Request = CaptureState::read(&self.store, &attempt.request)?;
            if saved.attempt.round == request.attempt.round
                && saved.attempt.slot == request.attempt.slot
            {
                let mut old = serde_json::to_value(&saved)?;
                let mut new = serde_json::to_value(&request)?;
                old.as_object_mut().unwrap().remove("attempt");
                new.as_object_mut().unwrap().remove("attempt");
                ensure!(
                    old == new && attempt.result.is_some(),
                    "unknown or changed auxiliary request"
                );
                return Ok((saved, Some(index)));
            }
        }
        ensure!(
            capture.auxiliary.len() < capture.config.max_rounds.saturating_mul(2),
            "auxiliary attempt bound exhausted"
        );
        ensure!(
            wall_available,
            "capture wall budget exhausted before auxiliary dispatch"
        );
        let reference = capture.publish(&self.store, &request)?;
        let index = capture.auxiliary.len();
        capture.auxiliary.push(AuxiliaryAttempt {
            request: reference,
            result: None,
            model: None,
        });
        self.journal_research(next)?;
        Ok((request, Some(index)))
    }

    fn finish_research_aux(
        &mut self,
        next: &mut Checkpoint,
        index: usize,
        result: AuxiliaryResult,
        model: Option<Vec<u8>>,
    ) -> Result<()> {
        let capture = next.capture.as_mut().unwrap();
        if let Some(bytes) = model {
            let total = capture
                .bytes
                .checked_add(bytes.len() as u64)
                .context("capture byte counter exhausted")?;
            ensure!(
                total <= capture.config.max_bytes,
                "capture model-root budget exhausted"
            );
            capture.auxiliary[index].model =
                Some(self.store.publish_blob(std::io::Cursor::new(bytes))?);
            capture.bytes = total;
        }
        let reference = capture.publish(&self.store, &result)?;
        capture.auxiliary[index].result = Some(reference);
        self.journal_research(next)
    }

    fn evaluate_research_aux(
        &mut self,
        next: &mut Checkpoint,
        executor: &impl Executor,
        request: Request,
    ) -> Result<Response> {
        if next.capture.is_none() {
            return Ok(checked_response(&request, executor.evaluate(&request)));
        }
        let result = self.evaluate_research_aux_inner(next, executor, request);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    fn evaluate_research_aux_inner(
        &mut self,
        next: &mut Checkpoint,
        executor: &impl Executor,
        request: Request,
    ) -> Result<Response> {
        let (request, index) = self.prepare_research_aux(next, request)?;
        let Some(index) = index else {
            return Ok(checked_response(&request, executor.evaluate(&request)));
        };
        if let Some(reference) = &next.capture.as_ref().unwrap().auxiliary[index].result {
            let result: AuxiliaryResult = CaptureState::read(&self.store, reference)?;
            if let AuxiliaryResult::Evaluation { response } = result {
                ensure!(
                    response.attempt == request.attempt && response.valid(),
                    "auxiliary response identity mismatch"
                );
                return Ok(response);
            }
            bail!("wrong auxiliary result kind");
        }
        let response = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            research_response(&request, executor.evaluate(&request))
        }))
        .map_err(|_| anyhow::anyhow!("auxiliary evaluator aborted; outcome remains unknown"))??;
        ensure!(
            response.failure != Some(WorkerFailure::StaleAttempt),
            "auxiliary stale response remains unknown"
        );
        self.finish_research_aux(
            next,
            index,
            AuxiliaryResult::Evaluation {
                response: response.clone(),
            },
            None,
        )?;
        Ok(response)
    }

    fn capture_io(&mut self, error: anyhow::Error) -> anyhow::Error {
        self.poisoned = true;
        RefitIoError(std::io::Error::other(error.to_string())).into()
    }

    fn fit_research_aux(
        &mut self,
        next: &mut Checkpoint,
        executor: &impl Executor,
        request: Request,
    ) -> Result<Vec<u8>> {
        let (request, index) = self
            .prepare_research_aux(next, request)
            .map_err(|error| self.capture_io(error))?;
        let Some(index) = index else {
            return executor.fitted_model(&request);
        };
        let attempt = next.capture.as_ref().unwrap().auxiliary[index].clone();
        if let Some(reference) = attempt.result {
            let result = CaptureState::read(&self.store, &reference)
                .map_err(|error| self.capture_io(error))?;
            return match result {
                AuxiliaryResult::Fit => {
                    let model = attempt
                        .model
                        .context("captured model root missing")
                        .map_err(|error| self.capture_io(error))?;
                    self.store
                        .read_blob(&model)
                        .map_err(|error| self.capture_io(error.into()))
                }
                AuxiliaryResult::Failure {
                    worker: Some(response),
                } => Err(RefitWorkerError(response).into()),
                AuxiliaryResult::Failure { worker: None } => {
                    bail!("retained numerical refit failure")
                }
                _ => Err(self.capture_io(anyhow::anyhow!("wrong auxiliary result kind"))),
            };
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            executor.fitted_model(&request)
        }));
        match result {
            Ok(Ok(bytes)) if !bytes.is_empty() => {
                self.finish_research_aux(next, index, AuxiliaryResult::Fit, Some(bytes.clone()))
                    .map_err(|error| self.capture_io(error))?;
                Ok(bytes)
            }
            Ok(Err(error)) if error.is::<RefitIoError>() => {
                self.poisoned = true;
                Err(error)
            }
            Ok(Err(error)) => {
                let worker = error
                    .downcast_ref::<RefitWorkerError>()
                    .map(|e| e.0.clone());
                self.finish_research_aux(next, index, AuxiliaryResult::Failure { worker }, None)
                    .map_err(|error| self.capture_io(error))?;
                Err(error)
            }
            Ok(Ok(_)) => {
                self.finish_research_aux(
                    next,
                    index,
                    AuxiliaryResult::Failure { worker: None },
                    None,
                )
                .map_err(|error| self.capture_io(error))?;
                bail!("empty fitted model")
            }
            Err(_) => {
                Err(self.capture_io(anyhow::anyhow!("refit aborted; outcome remains unknown")))
            }
        }
    }
}

#[cfg(feature = "research")]
fn parallel_evaluate_research(
    executor: &impl Executor,
    requests: &[Request],
    workers: usize,
    wall_ms: u64,
    mut persist: impl FnMut(usize, Response, u64) -> Result<()>,
) -> Result<()> {
    use std::sync::{atomic::AtomicBool, mpsc};
    let cursor = AtomicUsize::new(0);
    let halted = AtomicBool::new(false);
    let started = ElapsedSpan::start();
    let (sender, receiver) = mpsc::sync_channel(workers.max(1));
    std::thread::scope(|scope| {
        for _ in 0..workers.min(requests.len()) {
            let sender = sender.clone();
            let cursor = &cursor;
            let halted = &halted;
            let started = &started;
            scope.spawn(move || {
                while !halted.load(Ordering::Acquire) {
                    if started.millis() >= u128::from(wall_ms) {
                        break;
                    }
                    let index = cursor.fetch_add(1, Ordering::Relaxed);
                    let Some(request) = requests.get(index) else {
                        break;
                    };
                    let started = Instant::now();
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        research_response(request, executor.evaluate(request))
                    }))
                    .map_err(|_| anyhow::anyhow!("evaluator aborted; outcome remains unknown"))
                    .and_then(|result| result);
                    let wall = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
                    if sender.send((index, result, wall)).is_err() {
                        break;
                    }
                }
            });
        }
        drop(sender);
        let mut error = None;
        let mut received = 0;
        for (index, response, wall) in receiver {
            received += 1;
            if error.is_none() {
                let result = response.and_then(|response| persist(index, response, wall));
                if let Err(failure) = result {
                    halted.store(true, Ordering::Release);
                    error = Some(failure);
                }
            }
        }
        if error.is_none() && received != requests.len() {
            error = Some(anyhow::anyhow!(
                "capture wall budget exhausted; undispatched reservations remain unknown"
            ));
        }
        error.map_or(Ok(()), Err)
    })
}

#[cfg(feature = "research")]
fn research_response(request: &Request, response: Response) -> Result<Response> {
    ensure!(
        response.attempt == request.attempt
            && response.failure != Some(WorkerFailure::StaleAttempt)
            && response.valid()
            && (response.failure.is_some() || response.scores.len() == 3),
        "invalid or stale capture response; outcome remains unknown"
    );
    Ok(response)
}

fn checkpoint_blobs(state: &Checkpoint) -> Result<Vec<BlobRef>> {
    let roots: Vec<_> = state
        .input_roots
        .iter()
        .cloned()
        .chain(state.champion.iter().map(|c| c.model.clone()))
        .collect();
    #[cfg(feature = "research")]
    if let Some(capture) = &state.capture {
        let mut unique = BTreeMap::new();
        for reference in roots.into_iter().chain(capture.roots()) {
            if let Some(old) = unique.insert(reference.sha256, reference.clone()) {
                ensure!(old == reference, "conflicting CAS root metadata");
            }
        }
        return Ok(unique.into_values().collect());
    }
    Ok(roots)
}

fn rung0_budget_used(state: &Checkpoint) -> u64 {
    #[cfg(feature = "research")]
    if let Some(capture) = &state.capture {
        return capture.charged;
    }
    state.admitted
}

fn rung0_budget_limit(state: &Checkpoint) -> u64 {
    #[cfg(feature = "research")]
    if let Some(capture) = &state.capture {
        return capture.config.max_attempts;
    }
    state.plan.evaluations
}

fn hex_blob(blob: &BlobRef) -> String {
    blob.sha256.iter().map(|b| format!("{b:02x}")).collect()
}
fn projector(plan: &Plan) -> Result<project::Projector> {
    let mut bytes = [0_u8; 32];
    for (i, value) in bytes.iter_mut().enumerate() {
        *value = u8::from_str_radix(&plan.problem_digest[i * 2..i * 2 + 2], 16)?;
    }
    Ok(project::Projector::new(
        plan.run_id.clone(),
        project::ProblemDigest(bytes),
    ))
}

fn public_metric(plan: &Plan) -> Result<String> {
    let name = jope_runtime::fitness::Fitness::parse(&plan.objective)?.name();
    Ok(
        if jope_runtime::fitness::Fitness::catalog()
            .iter()
            .any(|entry| entry.name == name)
        {
            name
        } else {
            "custom objective".into()
        },
    )
}

fn experiment(plan: &Plan, round: u64, slot: usize, rung: u8) -> Result<String> {
    ensure!(
        slot < 65536 && rung < 4,
        "evaluation ordinal is out of range"
    );
    let ordinal = evaluation_ordinal(round, u32::from(rung) * 65536 + slot as u32)?;
    Ok(projector(plan)?.experiment_id(project::EvaluationOrdinal(ordinal)))
}

fn evaluation_ordinal(round: u64, encoded_slot: u32) -> Result<u64> {
    ensure!(encoded_slot < 262144, "evaluation ordinal is out of range");
    round
        .checked_mul(262144)
        .and_then(|n| n.checked_add(u64::from(encoded_slot)))
        .context("evaluation ordinal exhausted")
}

fn projected_summary(plan: &Plan, claim: project::Claim<'_>) -> Result<String> {
    let EventBody::Reasoning(body) = projector(plan)?.project(claim)? else {
        bail!("unexpected public projection body");
    };
    Ok(body.summary)
}

fn accepted_summary(
    plan: &Plan,
    candidate: &Candidate,
    value: f64,
    previous: Option<f64>,
) -> Result<String> {
    let key = feat_math::GenomeIr::parse(&candidate.dsl)
        .ok()
        .and_then(|genome| match genome {
            feat_math::GenomeIr::Pipeline(dsl) => dsl.stages.last().map(|stage| stage.op.clone()),
            feat_math::GenomeIr::Portfolio(_) => Some("portfolio".into()),
            _ => None,
        });
    let Some(head) = key.as_deref().and_then(|key| {
        feat_math::public_names::heads::HEADS
            .iter()
            .find(|h| h.key == key)
    }) else {
        return projected_summary(plan, project::Claim::Unknown);
    };
    let context = project::ScoreContext {
        problem: project::ProblemDigest({
            let mut b = [0_u8; 32];
            for (i, v) in b.iter_mut().enumerate() {
                *v = u8::from_str_radix(&plan.problem_digest[i * 2..i * 2 + 2], 16)?;
            }
            b
        }),
        folds: project::FoldsDigest(
            Sha256::digest(serde_json::to_vec(&plan.fold_spec(1)?)?).into(),
        ),
        metric: public_metric(plan)?,
        scope: format!("cv_fit_{}fold", plan.limits().rung1_folds),
    };
    // This describes the atomic post-commit view. Neither body nor model
    // reference becomes observable unless the containing transaction commits.
    projected_summary(
        plan,
        project::Claim::Accepted {
            champion: project::Champion {
                algorithm_id: candidate.algorithm_id.parse()?,
                head: phrase::CatalogEntry {
                    public_name: head.public_name,
                    purpose: head.purpose,
                },
                committed: true,
                score: project::Score {
                    context: context.clone(),
                    value,
                },
            },
            previous: previous.map(|value| project::Score { context, value }),
        },
    )
}
fn round_rng(seed: u64, round: u64) -> ChaCha12Rng {
    let mut hash = Sha256::new();
    hash.update(seed.to_le_bytes());
    hash.update(round.to_le_bytes());
    ChaCha12Rng::from_seed(hash.finalize().into())
}
pub(crate) fn evaluation_seed(seed: u64, round: u64, slot: u32, rung: u8) -> u64 {
    let mut hash = Sha256::new();
    hash.update(seed.to_le_bytes());
    hash.update(round.to_le_bytes());
    hash.update(slot.to_le_bytes());
    hash.update([rung]);
    u64::from_le_bytes(
        hash.finalize()[..8]
            .try_into()
            .expect("SHA256 contains eight bytes"),
    )
}

fn plan_wave(
    state: &mut Checkpoint,
    space: &impl SearchSpace,
    admission: &impl RoundAdmission,
) -> Result<()> {
    #[cfg(feature = "research")]
    {
        plan_wave_inner(state, space, admission, None)
    }
    #[cfg(not(feature = "research"))]
    {
        plan_wave_inner(state, space, admission)
    }
}

fn plan_wave_inner(
    state: &mut Checkpoint,
    space: &impl SearchSpace,
    admission: &impl RoundAdmission,
    #[cfg(feature = "research")] capture_context: Option<(&Store, &CaptureContext)>,
) -> Result<()> {
    let mut parents: Vec<_> = state
        .elites
        .by_objective()
        .iter()
        .filter_map(|e| state.programs.get(&e.algorithm_id).cloned())
        .collect();
    if parents.is_empty() {
        parents = space.seeds()?;
    }
    ensure!(!parents.is_empty(), "search space has no seeds");
    let rounds = if state.next_round == 0 {
        1
    } else {
        state.plan.rounds_in_flight()
    };
    for offset in 0..rounds {
        let number = state
            .next_round
            .checked_add(offset as u64)
            .context("round counter exhausted")?;
        let mut rng = round_rng(state.plan.seed, number);
        let explore = state.stalled >= state.plan.limits().stall_rounds;
        let proposals = if number == 0 {
            space.seeds()?
        } else {
            space.propose(&parents, if explore { 16 } else { 8 }, &mut rng)?
        };
        let mut seen = state.memo.clone();
        let mut unique = vec![];
        #[cfg(feature = "research")]
        let mut source_indices = vec![];
        #[cfg(feature = "research")]
        let sources = proposals.into_iter().enumerate();
        #[cfg(not(feature = "research"))]
        let sources = proposals.into_iter().map(|source| ((), source));
        for (_source_index, source) in sources {
            let dsl = space.canonical(&source)?;
            if seen.insert(digest(dsl.as_bytes())) {
                #[cfg(feature = "research")]
                source_indices.push(_source_index as u64);
                unique.push(dsl);
            }
        }
        let round_context = RoundContext {
            plan: &state.plan,
            round: number,
            measured: state
                .elites
                .by_objective()
                .iter()
                .map(|elite| {
                    Ok(RoundMeasurement {
                        algorithm_id: &elite.algorithm_id,
                        dsl: state
                            .programs
                            .get(&elite.algorithm_id)
                            .context("elite program missing")?,
                        surrogate_fitness: elite.surrogate,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
        };
        #[cfg(feature = "research")]
        let mut capture_ref = None;
        #[cfg(feature = "research")]
        let observed_admission = if let Some((store, context)) = capture_context {
            let captured = state.capture.as_ref().context("capture state missing")?;
            let remaining = captured.config.max_attempts - captured.charged;
            if remaining == 0 {
                break;
            }
            ensure!(
                captured.rounds.len() < captured.config.max_rounds,
                "capture round bound exhausted"
            );
            let keys = state
                .elites
                .by_objective()
                .iter()
                .map(|elite| {
                    state
                        .programs
                        .get(&elite.algorithm_id)
                        .map(|dsl| digest(dsl.as_bytes()))
                        .context("elite program missing")
                })
                .collect::<Result<Vec<_>>>()?;
            let measured = captured.measured_pool(store, &keys)?;
            let observed = context.observe(number, &unique, &measured)?;
            let policy = if number == 0 {
                BehaviorPolicy {
                    kind: RecordedPolicyKind::Heuristic,
                    explore: false,
                    slate_costs: None,
                    admitted: (0..unique.len()).collect(),
                }
            } else {
                admission.admit_recorded_round(
                    &round_context,
                    unique.clone(),
                    explore,
                    &mut rng,
                    &RecordedAdmissionContext {
                        slate: &observed,
                        conditioning: crate::ranker::stage1b::TokenProjection {
                            tokens: std::array::from_fn(|i| {
                                context.binding.tokens[i]
                                    .as_slice()
                                    .try_into()
                                    .expect("capture binding validated ten 32-wide rows")
                            }),
                            mask: std::array::from_fn(|i| {
                                if context.binding.mask[i] {
                                    1.0
                                } else {
                                    0.0
                                }
                            }),
                        },
                    },
                )?
            };
            let record = ResearchRound {
                schema: crate::research::VERSION.into(),
                identity: context.binding.identity.clone(),
                round: number,
                seed: state.plan.seed,
                tokens: context.binding.tokens.clone(),
                mask: context.binding.mask.clone(),
                revealed: observed.revealed,
                proposals: observed.programs,
                source_indices,
                policy,
                evaluate_all_proposals: context.binding.config.evaluate_all_proposals,
                dispatched: vec![],
                capture_complete: false,
                outcomes: vec![],
            };
            record.validate()?;
            let mut dispatch = record.policy.admitted.clone();
            if record.evaluate_all_proposals {
                dispatch.extend((0..unique.len()).filter(|i| !record.policy.admitted.contains(i)));
            }
            dispatch.truncate(remaining.min(usize::MAX as u64) as usize);
            let admitted = record
                .policy
                .admitted
                .iter()
                .filter(|i| dispatch.contains(i))
                .map(|i| unique[*i].clone())
                .collect::<Vec<_>>();
            let slate = Slate {
                record,
                dsls: unique.clone(),
                dispatch,
            };
            let captured = state.capture.as_mut().unwrap();
            let reference = captured.publish(store, &slate)?;
            captured.charged += slate.dispatch.len() as u64;
            ensure!(
                captured
                    .rounds
                    .insert(
                        number,
                        CapturedRound {
                            slate: reference.clone(),
                            attempts: vec![],
                            committed: None,
                            aborted: false
                        }
                    )
                    .is_none(),
                "capture round already planned"
            );
            capture_ref = Some(reference);
            Some(admitted)
        } else {
            None
        };
        #[cfg(feature = "research")]
        let admitted = if let Some(admitted) = observed_admission {
            admitted
        } else if number == 0 {
            unique
        } else {
            admission.admit_round(&round_context, unique, explore, &mut rng)?
        };
        #[cfg(not(feature = "research"))]
        let admitted = if number == 0 {
            unique
        } else {
            admission.admit_round(&round_context, unique, explore, &mut rng)?
        };
        let mut candidates = vec![];
        for dsl in admitted
            .into_iter()
            .take((state.plan.evaluations - state.admitted).min(usize::MAX as u64) as usize)
        {
            let algorithm_id = format!("alg_{:08}", state.admitted);
            state.admitted += 1;
            state.memo.insert(digest(dsl.as_bytes()));
            state.programs.insert(algorithm_id.clone(), dsl.clone());
            candidates.push(Candidate { algorithm_id, dsl });
        }
        state.pending.push(PendingRound {
            number,
            candidates,
            #[cfg(feature = "research")]
            capture: capture_ref,
        });
        if rung0_budget_used(state) >= rung0_budget_limit(state) {
            break;
        }
    }
    Ok(())
}

fn checked_response(request: &Request, response: Response) -> Response {
    if response.attempt != request.attempt
        || !response.valid()
        || (response.failure.is_none() && response.scores.len() != 3)
    {
        Response::failed(request.attempt.clone(), WorkerFailure::Protocol, 0)
    } else {
        response
    }
}

fn parallel_evaluate(
    executor: &impl Executor,
    requests: &[Request],
    workers: usize,
) -> Vec<Response> {
    let cursor = AtomicUsize::new(0);
    let results = Mutex::new(vec![None; requests.len()]);
    std::thread::scope(|scope| {
        for _ in 0..workers.min(requests.len()) {
            scope.spawn(|| loop {
                let index = cursor.fetch_add(1, Ordering::Relaxed);
                let Some(request) = requests.get(index) else {
                    break;
                };
                let response = checked_response(request, executor.evaluate(request));
                results.lock().expect("result mutex poisoned")[index] = Some(response);
            });
        }
    });
    results
        .into_inner()
        .expect("result mutex poisoned")
        .into_iter()
        .map(|r| r.expect("worker result missing"))
        .collect()
}

fn record_result(
    state: &mut Checkpoint,
    candidate: &Candidate,
    response: &Response,
    id: &str,
    events: &mut Vec<EventBody>,
) -> Result<()> {
    add_events(state, candidate, response, id, events)?;
    if response.failure.is_none() {
        state.elites.insert(Elite {
            algorithm_id: candidate.algorithm_id.clone(),
            surrogate: response.scores[0],
            objective: response.scores[1],
        })?;
    }
    Ok(())
}

fn add_events(
    state: &mut Checkpoint,
    candidate: &Candidate,
    response: &Response,
    id: &str,
    events: &mut Vec<EventBody>,
) -> Result<()> {
    if matches!(
        response.failure,
        Some(WorkerFailure::Timeout | WorkerFailure::MemoryLimit)
    ) {
        state.nondeterminism.push(id.into());
    }
    events.push(EventBody::ExperimentStarted(ExperimentStartedBody {
        experiment_id: id.into(),
        hypothesis: projected_summary(&state.plan, project::Claim::Unknown)?,
    }));
    events.push(EventBody::ExperimentFinished(ExperimentFinishedBody {
        experiment_id: id.into(),
        outcome: if response.failure.is_none() {
            ExperimentOutcome::NotImproved
        } else {
            ExperimentOutcome::Failed
        },
        metric: public_metric(&state.plan)?,
        value: (response.attempt.slot / 65536 == 1)
            .then(|| response.scores.get(2).copied())
            .flatten(),
        failure: response.failure.map(|failure| PublicFailure {
            code: match failure {
                WorkerFailure::NonFinite => jope_api::error::ErrorCode::FitnessUndefined,
                WorkerFailure::Unavailable => jope_api::error::ErrorCode::BackendUnavailable,
                WorkerFailure::InvalidRequest => jope_api::error::ErrorCode::InvalidInput,
                _ => jope_api::error::ErrorCode::NumericalFailure,
            },
            reason: public_failure(failure),
        }),
        attempt_receipt: Some(Box::new(jope_api::EvaluationAttempt {
            // The public receipt identifies the logical evaluation, just like its
            // experiment ID. Worker restart epochs remain in private run records.
            attempt: evaluation_ordinal(response.attempt.round, response.attempt.slot)?,
            outcome: if response.failure.is_some() {
                jope_api::EvaluationAttemptOutcome::Failed
            } else {
                jope_api::EvaluationAttemptOutcome::Completed
            },
            fold_scores: response.fold_scores.clone(),
            loss: response
                .scores
                .get(1)
                .copied()
                .filter(|_| response.failure.is_none()),
            program_sha256: response
                .failure
                .is_none()
                .then(|| digest(candidate.dsl.as_bytes())),
            failure: response.failure.map(|failure| jope_api::EvaluationFailure {
                code: format!("{failure:?}").to_ascii_lowercase(),
                reason: public_failure(failure),
            }),
        })),
    }));
    Ok(())
}

fn public_failure(failure: WorkerFailure) -> String {
    let reason = match failure {
        WorkerFailure::Timeout => "the evaluation reached its time limit",
        WorkerFailure::MemoryLimit => "the evaluation exceeded its memory limit",
        WorkerFailure::NonFinite => "the objective was undefined for its predictions",
        WorkerFailure::Unavailable => "the evaluation backend was unavailable",
        WorkerFailure::InvalidRequest => "the approach could not be evaluated",
        _ => "the evaluation did not complete",
    };
    let name = &feat_math::public_names::heads::HEADS[0];
    phrase::render_phrase(
        phrase::Template::Failure,
        &Sha256::digest(reason.as_bytes()).into(),
        phrase::CatalogEntry {
            public_name: name.public_name,
            purpose: name.purpose,
        },
        phrase::EventDetails {
            reason: Some(reason),
            ..Default::default()
        },
    )
    .expect("reviewed failure phrase")
}

fn discard_pending(
    state: &mut Checkpoint,
    executor: Option<&dyn Executor>,
    requests: &[Request],
    responses: &[Response],
    events: &mut Vec<EventBody>,
) -> Result<()> {
    let reason = state
        .completed
        .context("discard requires a completion reason")?;
    for pending in std::mem::take(&mut state.pending) {
        #[cfg(feature = "research")]
        if let Some(capture) = state.capture.as_mut() {
            if let Some(round) = capture.rounds.get_mut(&pending.number) {
                round.aborted = true;
            }
            capture.tainted = true;
        }
        for (slot, candidate) in pending.candidates.into_iter().enumerate() {
            let index = requests
                .iter()
                .position(|r| r.attempt.round == pending.number && r.attempt.slot == slot as u32);
            if let (Some(executor), Some(index)) = (executor, index) {
                executor.revoke(&requests[index].attempt);
            }
            let response = index.and_then(|i| responses.get(i)).cloned();
            if let Some(response) = &response {
                let id = experiment(&state.plan, pending.number, slot, 0)?;
                add_events(state, &candidate, response, &id, events)?;
            }
            state.discarded.push(DiscardedEvaluation {
                round: pending.number,
                slot: slot as u32,
                candidate,
                reason,
                response,
            });
        }
    }
    Ok(())
}

/// Native heuristic grammar. No learned checkpoint or bank is loaded by this
/// M1 adapter; learned proposer/ranker parity remains the RNK-10/M3 gate.
pub struct NativeSpace;
impl SearchSpace for NativeSpace {
    fn canonical(&self, dsl: &str) -> Result<String> {
        let genome = feat_math::GenomeIr::parse(dsl)?;
        feat_math::transpile::supports(&genome)
            .map_err(|e| anyhow::anyhow!("unsupported Rust pipeline: {e:?}"))?;
        Ok(genome.canonical())
    }

    fn seeds(&self) -> Result<Vec<String>> {
        let seeds: Vec<_> = feat_math::evolve::STARTERS
            .iter()
            .copied()
            .chain(std::iter::once(feat_math::CHAMPION_GENOME))
            .filter_map(|s| self.canonical(s).ok())
            .collect();
        ensure!(!seeds.is_empty(), "no supported native starters");
        Ok(seeds)
    }

    fn propose(
        &self,
        parents: &[String],
        count: usize,
        rng: &mut ChaCha12Rng,
    ) -> Result<Vec<String>> {
        use feat_math::evolve::{self, CanonicalMutationArm::*};
        ensure!(!parents.is_empty(), "no parents");
        let arms = [
            ParameterJitter,
            Insert,
            Delete,
            AdjacentSwap,
            HeadReplacement,
            Simplify,
        ];
        let mut proposals = vec![];
        for _ in 0..count.saturating_mul(16) {
            let parent = &parents[sample_index(rng, parents.len())];
            let arm = sample_index(rng, 8);
            let seed = rng.next_u64();
            let proposed = if arm < 6 {
                evolve::mutate_canonical_pipeline_with_arm(parent, seed, 6, false, arms[arm])
            } else if arm == 6 {
                let other = &parents[sample_index(rng, parents.len())];
                evolve::crossover_canonical_pipelines(parent, other, seed, 6)
            } else {
                Ok(evolve::random_canonical_pipeline(seed, 6))
            };
            if let Ok(dsl) = proposed.and_then(|s| self.canonical(&s).map_err(|e| e.to_string())) {
                proposals.push(dsl);
                if proposals.len() == count {
                    break;
                }
            }
        }
        Ok(proposals)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NativeFolds {
    version: u32,
    matrix_sha256: String,
    target: String,
    features: Vec<String>,
    folds: Vec<Indices>,
    refit_output: Option<PathBuf>,
    refit_max_bytes: Option<u64>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Indices {
    repeat: u8,
    train: Vec<usize>,
    validation: Vec<usize>,
}

/// Existing WRK-01 callback. It reads only its supplied FIT mapping and returns
/// three finite values: M1 RMSE surrogate, minimization objective, user value.
/// This heuristic surrogate is explicit; it is not qualified as bundle R0/M3.
pub fn evaluate_native<B: AsRef<[u8]> + Sync>(
    request: &Request,
    matrix: &Matrix<B>,
) -> std::result::Result<worker::Scores, WorkerFailure> {
    // A dedicated numerical pool overrides ambient Rayon settings. The binary's
    // fixed one-thread fits do not change when live worker capacity changes.
    let result = jope_runtime::budget::Budget::new(None, 1)
        .install(|| native_scores(request, matrix))
        .map_err(|_| WorkerFailure::Unavailable)?;
    result.map_err(|error| {
        if error.is::<UndefinedFitness>() {
            WorkerFailure::NonFinite
        } else {
            WorkerFailure::InvalidRequest
        }
    })
}

#[derive(Debug)]
struct UndefinedFitness;
impl std::fmt::Display for UndefinedFitness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("candidate has undefined fitness or numerical predictions")
    }
}
impl std::error::Error for UndefinedFitness {}

fn native_scores<B: AsRef<[u8]>>(request: &Request, matrix: &Matrix<B>) -> Result<worker::Scores> {
    let spec: NativeFolds = serde_json::from_value(request.fold_spec.clone())?;
    ensure!(
        spec.version == NATIVE_FOLDS_VERSION && spec.folds.len() >= 2,
        "invalid native folds"
    );
    ensure!(
        digest(matrix.bytes()) == spec.matrix_sha256,
        "mapped FIT matrix identity mismatch"
    );
    let objective = RegressionObjective::parse(
        request
            .objective
            .as_str()
            .context("objective must be a specification string")?,
    )?;
    feat_math::GenomeIr::parse(&request.dsl)?;
    match (&spec.refit_output, spec.refit_max_bytes) {
        (Some(path), Some(maximum)) => {
            return native_refit(request, matrix, &spec, &objective, path, maximum)
        }
        (None, None) => {}
        _ => bail!("incomplete private refit handoff"),
    }
    let mut targets = vec![];
    let mut predictions = vec![];
    let mut validation_rows = BTreeSet::new();
    let mut losses = vec![];
    let mut values = vec![];
    let mut surrogate = vec![];
    let mut fold_scores = vec![];
    for fold in spec.folds {
        let train: BTreeSet<_> = fold.train.iter().copied().collect();
        ensure!(
            !train.is_empty() && train.len() == fold.train.len() && !fold.validation.is_empty(),
            "empty or duplicated fold"
        );
        ensure!(
            fold.validation
                .iter()
                .all(|r| !train.contains(r) && validation_rows.insert((fold.repeat, *r))),
            "overlapping fold rows"
        );
        let (x, y) = numeric_rows(matrix, &spec.target, &spec.features, &fold.train)?;
        let fitted =
            NativeModel::fit(&spec.features, &request.dsl, &x, &y).map_err(|_| UndefinedFitness)?;
        let (vx, vy) = numeric_rows(matrix, &spec.target, &spec.features, &fold.validation)?;
        let prediction = fitted.predict(&vx).map_err(|_| UndefinedFitness)?;
        let score = objective
            .evaluate(&vy, &prediction)
            .map_err(|_| UndefinedFitness)?;
        let rmse = RegressionObjective::parse("rmse")?
            .evaluate(&vy, &prediction)
            .map_err(|_| UndefinedFitness)?;
        losses.push(score.loss);
        values.push(score.value);
        surrogate.push(rmse.loss);
        fold_scores.push(rmse.loss);
        targets.extend(vy);
        predictions.extend(prediction);
    }
    let mean = |v: &[f64]| v.iter().map(|x| x / v.len() as f64).sum::<f64>();
    let scores = vec![mean(&surrogate), mean(&losses), mean(&values)];
    ensure!(scores.iter().all(|x| x.is_finite()), "non-finite CV score");
    let mut bytes = Vec::with_capacity(predictions.len() * 16);
    for (y, p) in targets.into_iter().zip(predictions) {
        ensure!(p.is_finite(), "non-finite prediction");
        bytes.extend(y.to_le_bytes());
        bytes.extend(p.to_le_bytes());
    }
    Ok(worker::Scores {
        values: scores,
        fold_scores,
        predictions_digest: digest(&bytes),
    })
}

pub(crate) fn numeric_rows<B: AsRef<[u8]>>(
    matrix: &Matrix<B>,
    target: &str,
    features: &[String],
    rows: &[usize],
) -> Result<(feat_math::Rows, Vec<f64>)> {
    ensure!(
        !features.is_empty()
            && features.iter().collect::<BTreeSet<_>>().len() == features.len()
            && !features.iter().any(|f| f == target),
        "invalid native features"
    );
    let find = |name: &str| {
        matrix
            .columns()
            .iter()
            .position(|c| c.name == name)
            .context("unknown native column")
    };
    let target = find(target)?;
    let columns: Vec<_> = features.iter().map(|s| find(s)).collect::<Result<_>>()?;
    let mut x = vec![];
    let mut y = vec![];
    for &row in rows {
        let JmatCell::Numeric(label) =
            matrix.cell(row, target).context("row outside FIT matrix")?
        else {
            bail!("invalid numeric target");
        };
        ensure!(label.is_finite(), "non-finite target");
        y.push(label);
        x.push(
            columns
                .iter()
                .map(|&col| -> Result<f64> {
                    match matrix.cell(row, col) {
                        Some(JmatCell::Numeric(v)) if v.is_finite() => Ok(v),
                        Some(JmatCell::Missing) => Ok(f64::NAN),
                        _ => bail!("invalid numeric feature"),
                    }
                })
                .collect::<Result<_>>()?,
        );
    }
    Ok((x, y))
}

/// Internal CAS fit, deliberately separate from the frozen exported bundle
/// format. Every preprocessing parameter is fitted inside the training fold.
#[derive(Serialize, Deserialize)]
pub(crate) struct NativeModel {
    version: u32,
    features: Vec<String>,
    dsl: String,
    min: Vec<f64>,
    max: Vec<f64>,
    fill: Vec<f64>,
    target_min: f64,
    target_scale: f64,
    model: feat_math::Model,
}
impl NativeModel {
    pub(crate) fn fit(
        features: &[String],
        dsl: &str,
        x: &feat_math::Rows,
        y: &[f64],
    ) -> Result<Self> {
        ensure!(
            !x.is_empty() && x.len() == y.len() && y.iter().all(|y| y.is_finite()),
            "invalid training rows"
        );
        ensure!(
            x.iter().all(|row| row.len() == features.len()),
            "ragged features"
        );
        let mut min = vec![];
        let mut max = vec![];
        let mut fill = vec![];
        for col in 0..features.len() {
            let present: Vec<_> = x.iter().map(|r| r[col]).filter(|v| v.is_finite()).collect();
            min.push(present.iter().copied().reduce(f64::min).unwrap_or(0.0));
            max.push(present.iter().copied().reduce(f64::max).unwrap_or(0.0));
            fill.push(
                present
                    .iter()
                    .map(|v| v / present.len() as f64)
                    .sum::<f64>(),
            );
        }
        let target_min = y.iter().copied().reduce(f64::min).context("no target")?;
        let target_max = y.iter().copied().reduce(f64::max).context("no target")?;
        let target_scale = (target_max - target_min).max(1e-12);
        ensure!(
            target_scale.is_finite()
                && min.iter().chain(&max).chain(&fill).all(|v| v.is_finite())
                && min.iter().zip(&max).all(|(lo, hi)| (hi - lo).is_finite()),
            "preprocessing overflow"
        );
        let normalize = |rows: &feat_math::Rows| -> feat_math::Rows {
            rows.iter()
                .map(|r| {
                    r.iter()
                        .enumerate()
                        .map(|(j, &v)| {
                            let value = if v.is_finite() { v } else { fill[j] };
                            if max[j] == min[j] {
                                0.0
                            } else {
                                ((value - min[j]) / (max[j] - min[j])).clamp(0.0, 1.0)
                            }
                        })
                        .collect()
                })
                .collect()
        };
        let tx = normalize(x);
        let ty = y
            .iter()
            .map(|v| (v - target_min) / target_scale)
            .collect::<Vec<_>>();
        let model = feat_math::fit(&feat_math::GenomeIr::parse(dsl)?, &tx, &ty)?;
        Ok(Self {
            version: NATIVE_ARTIFACT_VERSION,
            features: features.to_vec(),
            dsl: dsl.into(),
            min,
            max,
            fill,
            target_min,
            target_scale,
            model,
        })
    }
    pub(crate) fn predict(&self, x: &feat_math::Rows) -> Result<Vec<f64>> {
        ensure!(
            x.iter().all(|r| r.len() == self.features.len()),
            "prediction width mismatch"
        );
        let tx = x
            .iter()
            .map(|r| {
                r.iter()
                    .enumerate()
                    .map(|(j, &v)| {
                        let value = if v.is_finite() { v } else { self.fill[j] };
                        if self.max[j] == self.min[j] {
                            0.0
                        } else {
                            ((value - self.min[j]) / (self.max[j] - self.min[j])).clamp(0.0, 1.0)
                        }
                    })
                    .collect()
            })
            .collect();
        let values: Vec<_> = self
            .model
            .predict(&tx)?
            .into_iter()
            .map(|p| p * self.target_scale + self.target_min)
            .collect();
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "non-finite native prediction"
        );
        Ok(values)
    }
}

// This function is reached only inside the supervised numerical worker.
// Private scratch output is never a CAS or redb write.
fn native_refit<B: AsRef<[u8]>>(
    request: &Request,
    matrix: &Matrix<B>,
    spec: &NativeFolds,
    objective: &RegressionObjective,
    output: &Path,
    maximum: u64,
) -> Result<worker::Scores> {
    ensure!(
        maximum > 0 && maximum <= REFIT_HANDOFF_BYTES,
        "invalid private fit bound"
    );
    ensure!(
        output.is_absolute() && output.file_name().is_some_and(|n| n == "model.json"),
        "invalid private fit path"
    );
    let parent = output.parent().context("private fit directory missing")?;
    ensure!(
        fs::symlink_metadata(parent)?.is_dir() && fs::canonicalize(parent)? == parent,
        "private fit directory must be canonical"
    );
    let (x, y) = numeric_rows(
        matrix,
        &spec.target,
        &spec.features,
        &(0..matrix.rows()).collect::<Vec<_>>(),
    )?;
    let fitted =
        NativeModel::fit(&spec.features, &request.dsl, &x, &y).map_err(|_| UndefinedFitness)?;
    let predictions = fitted.predict(&x).map_err(|_| UndefinedFitness)?;
    let score = objective
        .evaluate(&y, &predictions)
        .map_err(|_| UndefinedFitness)?;
    let surrogate = RegressionObjective::parse("rmse")?
        .evaluate(&y, &predictions)
        .map_err(|_| UndefinedFitness)?;
    let mut measured = Vec::with_capacity(y.len() * 16);
    for (label, prediction) in y.iter().zip(&predictions) {
        measured.extend(label.to_le_bytes());
        measured.extend(prediction.to_le_bytes());
    }
    let predictions_digest = digest(&measured);
    let model_json = serde_json::to_string(&fitted)?;
    let _: NativeModel = serde_json::from_str(&model_json).map_err(|_| UndefinedFitness)?;
    let envelope = RefitEnvelope {
        attempt: request.attempt.clone(),
        model_digest: digest(model_json.as_bytes()),
        model_json,
        predictions_digest: predictions_digest.clone(),
    };
    let bytes = serde_json::to_vec(&envelope)?;
    ensure!(
        bytes.len() as u64 <= maximum,
        "promoted model exceeds the M1 private handoff bound"
    );
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(output)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    Ok(worker::Scores {
        values: vec![surrogate.loss, score.loss, score.value],
        fold_scores: vec![],
        predictions_digest,
    })
}

/// Hidden M1 adapter. The plan is immutable and contains a FIT-only mapping;
/// `home` is a private store directory, `live_workers` affects throughput only.
/// Public run/start/attach is implemented by BAT-02/CLT-02, not this probe.
pub fn run_m1(path: &Path, executable: &Path) -> Result<Value> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Input {
        home: PathBuf,
        plan: Plan,
        live_workers: usize,
    }
    let input: Input = serde_json::from_slice(&fs::read(path)?)?;
    ensure!(
        input.live_workers > 0 && input.live_workers <= 1024,
        "invalid live workers"
    );
    let store = Arc::new(Store::open(&input.home)?);
    let pool = ProcessExecutor::new(
        executable,
        input.live_workers,
        &input.home.join("worker-scratch"),
    )?;
    let mut discovery = Discovery::open(store, input.plan)?;
    while discovery.completed().is_none() {
        discovery.advance(
            &NativeSpace,
            &HeuristicAdmission,
            &pool,
            input.live_workers,
            usize::MAX,
        )?;
    }
    ensure!(
        discovery.champion().is_some(),
        "search produced no finite rung-1 champion"
    );
    Ok(
        json!({"stage":"search_finished", "completion_reason":discovery.completed(),
        "budget_bound_by":discovery.budget_bound_by(),
        "fitting":"selection_only",
        "threads_per_worker":1,
        "wall_recovery_policy":"interruption is charged through recovery; explicit pause excludes idle time; backwards clock refuses resume",
        "champion":discovery.snapshot().best, "snapshot":discovery.snapshot(),
        "nondeterminism":discovery.nondeterminism(), "projection":discovery.projection()?,
        "qualification":"M1 heuristic numeric regression; finalization/export and bundle-surrogate parity remain separate gates"}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        io::{BufRead, BufReader, Write},
        process::{Command, Stdio},
        time::Duration,
    };

    const TRAIN: &[u8] =
        include_bytes!("../../../../test-support/datasets/v1/d2-regression/train.csv");
    const TRAIN_SHA: &str = "d9a0b97e91ff9d04bc87fefa4671a49c40cee21dfac05be58a7fc7ce08045931";

    fn plan(directory: &Path, rows: usize, effort: Effort, evaluations: u64) -> Plan {
        assert_eq!(digest(TRAIN), TRAIN_SHA);
        let mut headers: Vec<_> = (0..21).map(|j| format!("x{j}")).collect();
        headers.push("target".into());
        let mut csv = headers.join(",").into_bytes();
        csv.push(b'\n');
        csv.extend(TRAIN);
        let mut table =
            jope_runtime::table::parse_csv(&csv, &jope_runtime::limits::Limits::default()).unwrap();
        // Only the catalog TRAIN object is opened; the external TEST object is
        // never read. Workers receive only this fixed FIT subset of TRAIN.
        let (fit, _) = super::super::splits::random_split(
            table.rows.len(),
            table.rows.len() * 4 / 5,
            73,
            "dsc08/fit",
        )
        .unwrap();
        let source_rows: Vec<_> = fit.into_iter().take(rows).collect();
        table.rows = source_rows.iter().map(|&r| table.rows[r].clone()).collect();
        let bytes = crate::dataset::jmat::encode(&table).unwrap();
        let path = directory.join("fit.jmat");
        fs::write(&path, &bytes).unwrap();
        fs::write(
            directory.join("fit-source-rows.json"),
            serde_json::to_vec(&source_rows).unwrap(),
        )
        .unwrap();
        Plan {
            run_id: "run_01JBD8Q6N4V2T5Y8K3M7P9R1S0".parse().unwrap(),
            dataset_id: "ds_21fbfe8b53dbcd605d79486b".parse().unwrap(),
            problem_digest: digest(&bytes),
            bundle_digest: digest(b"DSC08 heuristic M1, no learned bundle"),
            binary_version: "dsc08-test-candidate-v1".into(),
            matrix_path: path,
            matrix_sha256: digest(&bytes),
            target: "target".into(),
            features: headers[..21].to_vec(),
            objective: "rmse".into(),
            seed: 73,
            planned_workers: 8,
            effort,
            evaluations,
            wall_seconds: 60,
            evaluation_wall_ms: 15_000,
            evaluation_rss_bytes: 2 * 1024 * 1024 * 1024,
            groups: None,
            times: None,
            gap_rows: 0,
        }
    }

    // Scheduler controls use an executor fixture, not ranker/quality evidence.
    struct RoundSpace;
    impl SearchSpace for RoundSpace {
        fn canonical(&self, dsl: &str) -> Result<String> {
            Ok(dsl.into())
        }
        fn seeds(&self) -> Result<Vec<String>> {
            Ok((0..4).map(|i| format!("seed-{i}")).collect())
        }
        fn propose(
            &self,
            parents: &[String],
            count: usize,
            rng: &mut ChaCha12Rng,
        ) -> Result<Vec<String>> {
            assert_eq!(parents[0], "seed-3", "parents retain user-objective order");
            let wave = rng.next_u64();
            Ok((0..count)
                .map(|i| format!("candidate-{wave}-{i}"))
                .collect())
        }
    }
    struct OpposingScores;
    impl Executor for OpposingScores {
        fn evaluate(&self, request: &Request) -> Response {
            let seed = request
                .dsl
                .strip_prefix("seed-")
                .and_then(|s| s.parse::<u32>().ok());
            let (surrogate, objective) =
                seed.map_or((50.0, 5.0), |i| (f64::from(i), 20.0 - f64::from(i)));
            Response {
                attempt: request.attempt.clone(),
                scores: vec![surrogate, objective, objective],
                fold_scores: vec![],
                predictions_digest: digest(request.dsl.as_bytes()),
                wall_ms: 0,
                failure: None,
            }
        }
        fn fitted_model(&self, request: &Request) -> Result<Vec<u8>> {
            Ok(request.dsl.as_bytes().to_vec())
        }
    }
    struct ContextAdmission {
        rounds: Mutex<Vec<u64>>,
        refuse: bool,
    }
    impl RoundAdmission for ContextAdmission {
        fn admit_round(
            &self,
            context: &RoundContext<'_>,
            proposals: Vec<String>,
            _: bool,
            _: &mut ChaCha12Rng,
        ) -> Result<Vec<String>> {
            self.rounds.lock().unwrap().push(context.round);
            ensure!(!self.refuse, "controlled admission refusal");
            assert_eq!(context.measured.len(), 4);
            assert_eq!(context.measured[0].dsl, "seed-3");
            assert_eq!(context.measured[0].algorithm_id, "alg_00000003");
            assert_eq!(context.measured[0].surrogate_fitness, 3.0);
            assert_eq!(context.plan.seed, 73);
            Ok(proposals.into_iter().skip(4).take(4).collect())
        }
    }

    #[test]
    fn round_context_admission_preserves_objective_promotions_and_pending_recovery() {
        let directory = tempfile::tempdir().unwrap();
        let p = plan(directory.path(), 30, Effort::Fast, 24);
        let store = Arc::new(Store::open(directory.path().join("store")).unwrap());
        let mut d = Discovery::open(store.clone(), p.clone()).unwrap();
        let admission = ContextAdmission {
            rounds: Mutex::new(vec![]),
            refuse: false,
        };
        d.advance(&RoundSpace, &admission, &OpposingScores, 1, 1)
            .unwrap();
        assert!(
            admission.rounds.lock().unwrap().is_empty(),
            "seed round bypasses admission"
        );
        assert_eq!(d.champion().unwrap().candidate.dsl, "seed-3");
        assert_eq!(d.champion().unwrap().loss, 17.0);
        d.advance(&RoundSpace, &admission, &OpposingScores, 1, 1)
            .unwrap();
        assert_eq!(*admission.rounds.lock().unwrap(), vec![1, 2]);
        assert_eq!(d.checkpoint.pending.len(), 1);
        assert!(d.champion().unwrap().candidate.dsl.ends_with("-4"));
        assert_eq!(
            d.champion().unwrap().loss,
            5.0,
            "objective improves despite worse surrogate"
        );
        let pending = d.checkpoint.pending[0].candidates.clone();
        assert!(pending
            .iter()
            .all(|c| c.dsl.rsplit('-').next().unwrap().parse::<u32>().unwrap() >= 4));
        d.pause().unwrap();
        drop(d);
        let mut d = Discovery::open(store, p).unwrap();
        let refusal = ContextAdmission {
            rounds: Mutex::new(vec![]),
            refuse: true,
        };
        d.advance(&RoundSpace, &refusal, &OpposingScores, 7, 1)
            .unwrap();
        assert!(
            refusal.rounds.lock().unwrap().is_empty(),
            "retained wave is not reranked on recovery/capacity change"
        );
    }

    #[test]
    #[ignore = "requires the hash-pinned RES-07 CPU packet and frozen RES-06 bundle"]
    fn learned_cpu_admission_recovery_matches_uninterrupted_execution() {
        use crate::ranker::{
            admission::{
                CpuRoundAdmission, RevealedRule, RoundInputProvider, RoundInputs, RoundProgram,
            },
            stage1b::TokenProjection,
        };
        use serde_json::Value;
        use sha2::{Digest, Sha256};
        use std::{
            collections::HashMap,
            env, fs,
            sync::atomic::{AtomicUsize, Ordering},
        };

        const WEIGHTS_SHA256: &str =
            "ce39c75f37624e64dfbf89e6493adbf74811a5248f9355997161491e93a132ef";
        const INDEX_SHA256_ENV: &str = "JOPE_GOLDEN_INDEX_SHA256";

        fn read_pinned(path: &Path, expected: &str) -> Vec<u8> {
            assert!(!fs::symlink_metadata(path).unwrap().file_type().is_symlink());
            let bytes = fs::read(path).unwrap();
            assert_eq!(
                format!("{:x}", Sha256::digest(&bytes)),
                expected,
                "{}",
                path.display()
            );
            bytes
        }

        fn tensor_f32(bytes: &[u8], name: &str) -> Vec<f32> {
            let tensors = safetensors::SafeTensors::deserialize(bytes).unwrap();
            let tensor = tensors.tensor(name).unwrap();
            assert_eq!(tensor.dtype(), safetensors::Dtype::F32);
            tensor
                .data()
                .chunks_exact(4)
                .map(|raw| f32::from_le_bytes(raw.try_into().unwrap()))
                .collect()
        }

        struct Slate {
            prog: Vec<f32>,
            proj: Vec<f32>,
            tokens: Vec<f32>,
            mask: Vec<f32>,
            gates: Vec<f32>,
            revealed_fitness: Vec<f32>,
        }

        fn load_slate(root: &Path, index: &Value, cid: &str, width: usize) -> Slate {
            let name = format!("slates/{cid}-{width}.safetensors");
            let entry = &index["files"][name.as_str()];
            let path = root.join(&name);
            assert!(path.canonicalize().unwrap().starts_with(root));
            let bytes = read_pinned(
                &path,
                entry["sha256"].as_str().expect("slate must be indexed"),
            );
            assert_eq!(bytes.len() as u64, entry["bytes"].as_u64().unwrap());
            Slate {
                prog: tensor_f32(&bytes, "input.prog_412"),
                proj: tensor_f32(&bytes, "input.proj_16"),
                tokens: tensor_f32(&bytes, "input.mm_tokens_10x32"),
                mask: tensor_f32(&bytes, "input.mm_mask_10"),
                gates: tensor_f32(&bytes, "input.revealed_gate"),
                revealed_fitness: tensor_f32(&bytes, "input.revealed_raw_f"),
            }
        }

        struct ReplayProvider {
            by_dsl: HashMap<String, usize>,
            slates: HashMap<u64, Slate>,
            calls: Arc<AtomicUsize>,
        }
        impl RoundInputProvider for ReplayProvider {
            fn observe(
                &self,
                context: &RoundContext<'_>,
                proposals: &[String],
            ) -> Result<RoundInputs> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                let slate = self
                    .slates
                    .get(&context.round)
                    .context("unexpected learned round")?;
                let program = |dsl: &str| -> Result<RoundProgram> {
                    let index = *self
                        .by_dsl
                        .get(dsl)
                        .context("program missing from pinned packet")?;
                    ensure!(index < 20, "replay program outside pinned packet subset");
                    Ok(RoundProgram {
                        dsl: dsl.to_owned(),
                        features412: slate.prog[index * 412..(index + 1) * 412]
                            .try_into()
                            .unwrap(),
                        projection16: slate.proj[index * 16..(index + 1) * 16].try_into().unwrap(),
                    })
                };
                let pool = context
                    .measured
                    .iter()
                    .map(|m| program(m.dsl))
                    .collect::<Result<Vec<_>>>()?;
                let pool_gates = context
                    .measured
                    .iter()
                    .map(|m| {
                        let index = *self.by_dsl.get(m.dsl).context("measured program missing")?;
                        ensure!(
                            index < 4,
                            "learned replay changed the measured four-program pool"
                        );
                        ensure!(
                            slate.gates[index] == 0.0 || slate.gates[index] == 1.0,
                            "nonbinary revealed gate"
                        );
                        Ok(slate.gates[index] == 1.0)
                    })
                    .collect::<Result<Vec<_>>>()?;
                let proposals = proposals
                    .iter()
                    .map(|dsl| program(dsl))
                    .collect::<Result<Vec<_>>>()?;
                let conditioning = TokenProjection {
                    tokens: std::array::from_fn(|row| {
                        std::array::from_fn(|col| slate.tokens[row * 32 + col])
                    }),
                    mask: slate.mask.as_slice().try_into().unwrap(),
                };
                Ok(RoundInputs {
                    pool,
                    pool_gates,
                    proposals,
                    conditioning,
                })
            }
        }

        struct ReplaySpace {
            dsls: Vec<String>,
            proposals: AtomicUsize,
        }
        impl SearchSpace for ReplaySpace {
            fn canonical(&self, dsl: &str) -> Result<String> {
                Ok(dsl.to_owned())
            }
            fn seeds(&self) -> Result<Vec<String>> {
                Ok(self.dsls[..4].to_vec())
            }
            fn propose(
                &self,
                _: &[String],
                count: usize,
                _: &mut ChaCha12Rng,
            ) -> Result<Vec<String>> {
                ensure!(count == 8, "expected non-exploring eight-proposal wave");
                let wave = self.proposals.fetch_add(1, Ordering::SeqCst);
                let start = 4 + wave * 8;
                ensure!(
                    start + 8 <= self.dsls.len(),
                    "test fixture proposal waves exhausted"
                );
                Ok(self.dsls[start..start + 8].to_vec())
            }
        }

        struct ReplayExecutor {
            indices: HashMap<String, usize>,
            seed_fitness: Vec<f32>,
        }
        impl Executor for ReplayExecutor {
            fn evaluate(&self, request: &Request) -> Response {
                let index = self.indices[&request.dsl];
                let (surrogate, objective) = if index < 4 {
                    (f64::from(self.seed_fitness[index]), index as f64)
                } else {
                    (index as f64, 100.0 + index as f64)
                };
                Response {
                    attempt: request.attempt.clone(),
                    scores: vec![surrogate, objective, objective],
                    fold_scores: vec![],
                    predictions_digest: digest(request.dsl.as_bytes()),
                    wall_ms: 0,
                    failure: None,
                }
            }
            fn fitted_model(&self, request: &Request) -> Result<Vec<u8>> {
                Ok(request.dsl.as_bytes().to_vec())
            }
        }

        fn run_case(
            root_dir: &Path,
            index: &Value,
            root: &Path,
            cid: &str,
            dsls: &[String],
            weights: &[u8],
            interrupted: bool,
        ) -> (Value, usize) {
            let case_dir = root_dir.join(if interrupted {
                "recovered"
            } else {
                "uninterrupted"
            });
            fs::create_dir_all(&case_dir).unwrap();
            let mut p = plan(&case_dir, 30, Effort::Fast, 20);
            p.planned_workers = 8;
            let store = Arc::new(Store::open(case_dir.join("store")).unwrap());
            let mut discovery = Discovery::open(store.clone(), p.clone()).unwrap();
            let calls = Arc::new(AtomicUsize::new(0));
            let slate12 = load_slate(root, index, cid, 12);
            let seed_fitness = slate12.revealed_fitness.clone();
            let slate20 = load_slate(root, index, cid, 20);
            let provider = ReplayProvider {
                by_dsl: dsls
                    .iter()
                    .enumerate()
                    .map(|(i, dsl)| (dsl.clone(), i))
                    .collect(),
                slates: [(1, slate12), (2, slate20)].into_iter().collect(),
                calls: calls.clone(),
            };
            let admission = CpuRoundAdmission::for_cpu_replay_weights(
                weights,
                p.clone(),
                provider,
                RevealedRule::StorageOrder,
            )
            .unwrap();
            let executor = ReplayExecutor {
                indices: dsls
                    .iter()
                    .enumerate()
                    .map(|(i, dsl)| (dsl.clone(), i))
                    .collect(),
                seed_fitness,
            };
            let space = ReplaySpace {
                dsls: dsls.to_vec(),
                proposals: AtomicUsize::new(0),
            };
            discovery
                .advance(&space, &admission, &executor, 8, 1)
                .unwrap();
            assert_eq!(
                calls.load(Ordering::SeqCst),
                0,
                "seed round bypasses learned admission"
            );
            discovery
                .advance(
                    &space,
                    &admission,
                    &executor,
                    8,
                    if interrupted { 1 } else { 2 },
                )
                .unwrap();
            assert_eq!(
                calls.load(Ordering::SeqCst),
                2,
                "both learned proposals were evaluated before recovery"
            );
            if interrupted {
                assert_eq!(
                    discovery.checkpoint.pending.len(),
                    1,
                    "one admitted sibling wave is durably pending"
                );
                let retained = serde_json::to_value(&discovery.checkpoint.pending).unwrap();
                discovery.pause().unwrap();
                drop(discovery);
                let mut recovered = Discovery::open(store, p).unwrap();
                assert_eq!(
                    serde_json::to_value(&recovered.checkpoint.pending).unwrap(),
                    retained
                );
                recovered
                    .advance(&space, &admission, &executor, 1, 1)
                    .unwrap();
                assert_eq!(
                    calls.load(Ordering::SeqCst),
                    2,
                    "recovery must consume persisted admission without reranking"
                );
                (
                    serde_json::json!({
                        "programs": recovered.checkpoint.programs,
                        "memo": recovered.checkpoint.memo,
                        "elites": recovered.checkpoint.elites,
                        "champion": recovered.checkpoint.champion,
                        "pending": recovered.checkpoint.pending,
                    }),
                    calls.load(Ordering::SeqCst),
                )
            } else {
                (
                    serde_json::json!({
                        "programs": discovery.checkpoint.programs,
                        "memo": discovery.checkpoint.memo,
                        "elites": discovery.checkpoint.elites,
                        "champion": discovery.checkpoint.champion,
                        "pending": discovery.checkpoint.pending,
                    }),
                    calls.load(Ordering::SeqCst),
                )
            }
        }

        let golden =
            PathBuf::from(env::var_os("JOPE_GOLDEN_DIR").expect("hash-pinned CPU packet required"));
        let golden = golden.canonicalize().unwrap();
        let index_sha = env::var(INDEX_SHA256_ENV).expect("trusted index SHA required");
        let index_bytes = read_pinned(&golden.join("index.json"), &index_sha);
        let index: Value = serde_json::from_slice(&index_bytes).unwrap();
        assert_eq!(index["schema"], "jope.golden-fixtures/v1");
        assert_eq!(index["contexts"], 50);
        let cid = index["files"]
            .as_object()
            .unwrap()
            .keys()
            .find_map(|name| name.strip_prefix("contexts/")?.strip_suffix(".safetensors"))
            .expect("pinned context inventory empty")
            .to_owned();
        let programs_path = golden.join("programs.jsonl");
        let programs_bytes = read_pinned(
            &programs_path,
            index["files"]["programs.jsonl"]["sha256"].as_str().unwrap(),
        );
        let dsls: Vec<String> = std::str::from_utf8(&programs_bytes)
            .unwrap()
            .lines()
            .take(20)
            .map(|line| {
                serde_json::from_str::<Value>(line).unwrap()["dsl"]
                    .as_str()
                    .unwrap()
                    .to_owned()
            })
            .collect();
        assert_eq!(dsls.len(), 20);
        let bundle = PathBuf::from(
            env::var_os("JOPE_PARITY_BUNDLE_DIR").expect("frozen RES-06 bundle required"),
        );
        let weights = read_pinned(&bundle.join("jope.safetensors"), WEIGHTS_SHA256);
        let dir = tempfile::tempdir().unwrap();
        let uninterrupted = run_case(dir.path(), &index, &golden, &cid, &dsls, &weights, false);
        let recovered = run_case(dir.path(), &index, &golden, &cid, &dsls, &weights, true);
        assert_eq!(
            uninterrupted.0, recovered.0,
            "recovered Discovery differs from uninterrupted execution"
        );
        assert_eq!(uninterrupted.0["programs"].as_object().unwrap().len(), 12);
        assert_eq!(uninterrupted.1, 2);
        assert_eq!(recovered.1, 2);
        println!("RNK10_S2_REAL_CPU_REPLAY contexts=1 proposal_rounds=2 interruptions=1 capacity=8->1 learned_invocations=2->2 state_equal=true");
    }

    #[test]
    fn round_admission_refusal_preserves_committed_state_and_retry_identity() {
        let directory = tempfile::tempdir().unwrap();
        let p = plan(directory.path(), 30, Effort::Fast, 24);
        let store = Arc::new(Store::open(directory.path().join("store")).unwrap());
        let mut d = Discovery::open(store, p).unwrap();
        let admission = ContextAdmission {
            rounds: Mutex::new(vec![]),
            refuse: false,
        };
        d.advance(&RoundSpace, &admission, &OpposingScores, 1, 1)
            .unwrap();
        let before = serde_json::to_value(&d.checkpoint).unwrap();
        let snapshot = d.snapshot().clone();
        let refusal = ContextAdmission {
            rounds: Mutex::new(vec![]),
            refuse: true,
        };
        for _ in 0..2 {
            assert!(d
                .advance(&RoundSpace, &refusal, &OpposingScores, 1, 1)
                .is_err());
            assert_eq!(serde_json::to_value(&d.checkpoint).unwrap(), before);
            assert_eq!(d.snapshot(), &snapshot);
        }
        assert_eq!(*refusal.rounds.lock().unwrap(), vec![1, 1]);
        d.advance(&RoundSpace, &admission, &OpposingScores, 1, 1)
            .unwrap();
        assert_eq!(*admission.rounds.lock().unwrap(), vec![1, 2]);
    }

    struct TinySpace;
    impl SearchSpace for TinySpace {
        fn canonical(&self, dsl: &str) -> Result<String> {
            Ok(dsl.into())
        }
        fn seeds(&self) -> Result<Vec<String>> {
            Ok(vec!["lucky-seed".into()])
        }
        fn propose(&self, _: &[String], _: usize, _: &mut ChaCha12Rng) -> Result<Vec<String>> {
            Ok(vec!["later-better".into(), "tie".into()])
        }
    }
    struct Controlled {
        failure: Option<WorkerFailure>,
    }
    impl Executor for Controlled {
        fn evaluate(&self, request: &Request) -> Response {
            if let Some(failure) = self.failure {
                return Response::failed(request.attempt.clone(), failure, 0);
            }
            let promotion = request.attempt.slot >= 65_536;
            let loss = if request.dsl == "lucky-seed" {
                if promotion {
                    10.0
                } else {
                    0.0
                }
            } else {
                if promotion {
                    5.0
                } else {
                    1.0
                }
            };
            Response {
                attempt: request.attempt.clone(),
                scores: vec![loss, loss, loss],
                fold_scores: vec![],
                predictions_digest: digest(request.dsl.as_bytes()),
                wall_ms: 0,
                failure: None,
            }
        }
        fn fitted_model(&self, request: &Request) -> Result<Vec<u8>> {
            Ok(request.dsl.as_bytes().to_vec())
        }
    }

    fn finish(
        discovery: &mut Discovery,
        space: &impl SearchSpace,
        executor: &impl Executor,
        workers: usize,
    ) {
        for _ in 0..300 {
            if discovery.completed().is_some() {
                return;
            }
            discovery
                .advance(space, &HeuristicAdmission, executor, workers, usize::MAX)
                .unwrap();
        }
        panic!("search did not stop within bounded rounds");
    }

    #[test]
    fn strict_rung1_promotion_is_not_blocked_by_a_lucky_seed_rung0() {
        let directory = tempfile::tempdir().unwrap();
        let p = plan(directory.path(), 30, Effort::Fast, 24);
        let store = Arc::new(Store::open(directory.path().join("store")).unwrap());
        let mut d = Discovery::open(Arc::clone(&store), p).unwrap();
        let executor = Controlled { failure: None };
        d.advance(&TinySpace, &HeuristicAdmission, &executor, 8, usize::MAX)
            .unwrap();
        assert_eq!(d.champion().unwrap().loss, 10.0);
        d.advance(&TinySpace, &HeuristicAdmission, &executor, 8, usize::MAX)
            .unwrap();
        assert_eq!(d.champion().unwrap().candidate.dsl, "later-better");
        assert_eq!(d.champion().unwrap().loss, 5.0);
        finish(&mut d, &TinySpace, &executor, 8);
        assert_eq!(d.completed(), Some(CompletionReason::Converged));
        let accepted = d
            .projection()
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .filter(|e| e["kind"] == "algorithm.accepted")
            .count();
        assert_eq!(accepted, 2);
        assert_eq!(d.admitted(), 3);
        let model_id: ArtifactId = format!("art_{}", &hex_blob(&d.champion().unwrap().model)[..24])
            .parse()
            .unwrap();
        assert!(store.get(&RecordKey::Artifact(model_id)).unwrap().is_some());
    }

    #[test]
    fn limit_hits_count_admissions_and_list_exact_experiments() {
        for failure in [
            WorkerFailure::Timeout,
            WorkerFailure::MemoryLimit,
            WorkerFailure::InvalidRequest,
            WorkerFailure::NonFinite,
        ] {
            let directory = tempfile::tempdir().unwrap();
            let p = plan(directory.path(), 30, Effort::Fast, 1);
            let store = Arc::new(Store::open(directory.path().join("store")).unwrap());
            let mut d = Discovery::open(store, p).unwrap();
            finish(
                &mut d,
                &TinySpace,
                &Controlled {
                    failure: Some(failure),
                },
                1,
            );
            assert_eq!(d.admitted(), 1);
            assert!(d.champion().is_none());
            assert_eq!(d.completed(), Some(CompletionReason::BudgetEvaluations));
            let expected = usize::from(matches!(
                failure,
                WorkerFailure::Timeout | WorkerFailure::MemoryLimit
            ));
            assert_eq!(d.nondeterminism().count.0, expected as u64);
            assert_eq!(
                d.nondeterminism().experiment_ids,
                if expected == 1 {
                    vec![experiment(&d.checkpoint.plan, 0, 0, 0).unwrap()]
                } else {
                    vec![]
                }
            );
            if failure == WorkerFailure::NonFinite {
                let projection = d.projection().unwrap();
                assert!(projection
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|e| e["body"]["failure"]["code"] == "FITNESS_UNDEFINED"));
            }
        }
    }

    #[test]
    fn heuristic_exploration_is_seeded_distinct_and_retains_first_two() {
        let proposals: Vec<_> = (0..16).map(|n| n.to_string()).collect();
        let a = HeuristicAdmission.admit(proposals.clone(), true, &mut round_rng(73, 4));
        let b = HeuristicAdmission.admit(proposals.clone(), true, &mut round_rng(73, 4));
        assert_eq!(a, b);
        assert_eq!(&a[..2], &["0", "1"]);
        assert_eq!(a.iter().collect::<BTreeSet<_>>().len(), 4);
        assert_eq!(
            HeuristicAdmission.admit(proposals, false, &mut round_rng(73, 4)),
            vec!["0", "1", "2", "3"]
        );
    }

    #[test]
    fn invalid_and_changed_resume_plans_are_refused() {
        let directory = tempfile::tempdir().unwrap();
        let mut p = plan(directory.path(), 30, Effort::Fast, 5);
        let store = Arc::new(Store::open(directory.path().join("store")).unwrap());
        let d = Discovery::open(Arc::clone(&store), p.clone()).unwrap();
        drop(d);
        p.seed += 1;
        assert!(Discovery::open(Arc::clone(&store), p.clone()).is_err());
        p.seed -= 1;
        p.features.push(p.target.clone());
        assert!(Discovery::open(store, p).is_err());
    }

    #[test]
    fn interrupted_active_span_consumes_remaining_wall_budget_before_replay() {
        let directory = tempfile::tempdir().unwrap();
        let mut p = plan(directory.path(), 30, Effort::Fast, 24);
        p.wall_seconds = 1;
        let home = directory.path().join("store");
        {
            let store = Arc::new(Store::open(&home).unwrap());
            let mut d = Discovery::open(store, p.clone()).unwrap();
            let mut checkpoint = d.checkpoint.clone();
            checkpoint.elapsed_ms = 250;
            checkpoint.active_since_ms = Some(wall_clock_ms().unwrap() - 1500);
            d.commit_control(checkpoint, RunState::Running).unwrap();
        }
        let store = Arc::new(Store::open(&home).unwrap());
        let mut recovered = Discovery::open(store, p).unwrap();
        assert!(recovered.checkpoint.elapsed_ms >= 1750);
        assert_eq!(recovered.completed(), Some(CompletionReason::BudgetTime));
        assert_eq!(recovered.budget_bound_by(), Some("time"));
        assert!(recovered
            .advance(
                &TinySpace,
                &HeuristicAdmission,
                &Controlled { failure: None },
                1,
                1
            )
            .unwrap()
            .is_empty());
        assert_eq!(recovered.admitted(), 0);
    }

    #[test]
    fn explicit_pause_excludes_idle_time_and_backwards_active_clock_refuses() {
        let directory = tempfile::tempdir().unwrap();
        let mut p = plan(directory.path(), 30, Effort::Fast, 24);
        p.wall_seconds = 1;
        let home = directory.path().join("store");
        {
            let store = Arc::new(Store::open(&home).unwrap());
            let mut d = Discovery::open(store, p.clone()).unwrap();
            let mut checkpoint = d.checkpoint.clone();
            checkpoint.elapsed_ms = 900;
            checkpoint.active_since_ms = None;
            assert_eq!(
                checkpoint.elapsed_at(u64::MAX).unwrap(),
                900,
                "known paused idle interval is not charged"
            );
            checkpoint.active_since_ms = Some(1000);
            assert!(checkpoint.elapsed_at(999).is_err());
            checkpoint.active_since_ms = None;
            d.commit_control(checkpoint, RunState::Paused).unwrap();
        }
        let store = Arc::new(Store::open(&home).unwrap());
        let recovered = Discovery::open(store, p).unwrap();
        assert_eq!(recovered.checkpoint.elapsed_ms, 900);
        assert_eq!(recovered.completed(), None);
        assert!(recovered.checkpoint.active_since_ms.is_some());
    }

    #[test]
    fn aborted_round_exposes_no_checkpoint_events_or_artifact_reference() {
        let directory = tempfile::tempdir().unwrap();
        let p = plan(directory.path(), 30, Effort::Fast, 2);
        let store = Arc::new(Store::open(directory.path().join("store")).unwrap());
        let mut d = Discovery::open(Arc::clone(&store), p.clone()).unwrap();
        let before = d.snapshot().clone();
        // Force the round's transaction hook to fail *after* EventLog has
        // staged checkpoint/event rows and the candidate CAS object exists.
        store
            .transaction(|tx| {
                tx.put(
                    &RecordKey::Evaluation {
                        run: p.run_id.clone(),
                        number: 0,
                    },
                    &Record::metadata(json!({"fixture":"collision"}), vec![]),
                )
            })
            .unwrap();
        assert!(d
            .advance(
                &TinySpace,
                &HeuristicAdmission,
                &Controlled { failure: None },
                1,
                1
            )
            .is_err());
        assert_eq!(d.snapshot(), &before);
        assert_eq!(d.log.snapshot(&p.run_id).unwrap().unwrap(), before);
        assert!(d.projection().unwrap().as_array().unwrap().is_empty());
        assert!(d.champion().is_none());
        assert_eq!(d.admitted(), 0);
        let artifact: ArtifactId = format!("art_{}", &digest(b"lucky-seed")[..24])
            .parse()
            .unwrap();
        assert!(store.get(&RecordKey::Artifact(artifact)).unwrap().is_none());
        assert!(
            d.advance(
                &TinySpace,
                &HeuristicAdmission,
                &Controlled { failure: None },
                1,
                1
            )
            .is_err(),
            "unknown commit outcome requires reopen"
        );
    }

    #[test]
    fn equal_rung1_scores_never_replace_the_champion() {
        struct Unique;
        impl SearchSpace for Unique {
            fn canonical(&self, s: &str) -> Result<String> {
                Ok(s.into())
            }
            fn seeds(&self) -> Result<Vec<String>> {
                TinySpace.seeds()
            }
            fn propose(
                &self,
                _: &[String],
                _: usize,
                rng: &mut ChaCha12Rng,
            ) -> Result<Vec<String>> {
                Ok(vec![format!("tie-{}", rng.next_u64())])
            }
        }
        let directory = tempfile::tempdir().unwrap();
        let p = plan(directory.path(), 30, Effort::Medium, 4);
        let store = Arc::new(Store::open(directory.path().join("store")).unwrap());
        let mut d = Discovery::open(store, p).unwrap();
        finish(&mut d, &Unique, &Controlled { failure: None }, 8);
        let events = d.projection().unwrap();
        assert_eq!(
            events
                .as_array()
                .unwrap()
                .iter()
                .filter(|e| e["kind"] == "algorithm.accepted")
                .count(),
            2
        );
        assert_eq!(
            events
                .as_array()
                .unwrap()
                .iter()
                .filter(|e| e["kind"] == "experiment.finished"
                    && e["body"]["experiment_id"]
                        .as_str()
                        .unwrap()
                        .starts_with("exp_"))
                .count(),
            8
        );
        assert_eq!(d.champion().unwrap().candidate.algorithm_id, "alg_00000001");
    }

    struct WaveSpace;
    impl SearchSpace for WaveSpace {
        fn canonical(&self, s: &str) -> Result<String> {
            Ok(s.into())
        }
        fn seeds(&self) -> Result<Vec<String>> {
            Ok(vec!["private-seed-program".into()])
        }
        fn propose(&self, _: &[String], _: usize, rng: &mut ChaCha12Rng) -> Result<Vec<String>> {
            Ok(vec![format!("private-proposal-program-{}", rng.next_u64())])
        }
    }
    struct LateImprovement {
        wait: bool,
        late_promotion: std::sync::atomic::AtomicBool,
        revoked: std::sync::atomic::AtomicUsize,
    }
    impl LateImprovement {
        fn new(wait: bool) -> Self {
            Self {
                wait,
                late_promotion: std::sync::atomic::AtomicBool::new(false),
                revoked: std::sync::atomic::AtomicUsize::new(0),
            }
        }
    }
    impl Executor for LateImprovement {
        fn evaluate(&self, request: &Request) -> Response {
            if request.attempt.round == 1 && request.attempt.slot >= 65536 && self.wait {
                std::thread::sleep(Duration::from_millis(1100));
            }
            if request.attempt.round >= 3 && request.attempt.slot >= 65536 {
                self.late_promotion.store(true, Ordering::SeqCst);
            }
            let loss = match request.attempt.round {
                0 => 1.0,
                1 | 2 => 2.0,
                _ => 0.0,
            };
            Response {
                attempt: request.attempt.clone(),
                scores: vec![loss, loss, loss],
                fold_scores: vec![],
                predictions_digest: digest(request.dsl.as_bytes()),
                wall_ms: 0,
                failure: None,
            }
        }
        fn fitted_model(&self, request: &Request) -> Result<Vec<u8>> {
            Ok(request.dsl.as_bytes().to_vec())
        }
        fn revoke(&self, _: &Attempt) {
            self.revoked.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn convergence_stops_and_fences_a_wave_before_a_late_better_result_can_promote() {
        let directory = tempfile::tempdir().unwrap();
        let mut p = plan(directory.path(), 30, Effort::Fast, 24);
        p.planned_workers = 12;
        let store = Arc::new(Store::open(directory.path().join("store")).unwrap());
        let mut d = Discovery::open(store, p).unwrap();
        let executor = LateImprovement::new(false);
        d.advance(&WaveSpace, &HeuristicAdmission, &executor, 8, usize::MAX)
            .unwrap();
        d.advance(&WaveSpace, &HeuristicAdmission, &executor, 8, usize::MAX)
            .unwrap();
        assert_eq!(d.completed(), Some(CompletionReason::Converged));
        assert_eq!(d.checkpoint.next_round, 3);
        assert_eq!(d.champion().unwrap().loss, 1.0);
        assert_eq!(d.checkpoint.discarded.len(), 1);
        assert_eq!(d.checkpoint.discarded[0].round, 3);
        assert!(d.checkpoint.pending.is_empty());
        assert!(!executor.late_promotion.load(Ordering::SeqCst));
        assert_eq!(executor.revoked.load(Ordering::SeqCst), 1);
        assert_eq!(
            d.admitted(),
            4,
            "already admitted discarded work remains charged"
        );
    }

    #[test]
    fn wall_completion_is_checked_at_each_round_and_discards_the_remaining_wave() {
        let directory = tempfile::tempdir().unwrap();
        let mut p = plan(directory.path(), 30, Effort::Medium, 24);
        p.planned_workers = 12;
        let store = Arc::new(Store::open(directory.path().join("store")).unwrap());
        let mut d = Discovery::open(store, p).unwrap();
        let executor = LateImprovement::new(true);
        d.advance(&WaveSpace, &HeuristicAdmission, &executor, 8, usize::MAX)
            .unwrap();
        // Model time already spent before this call without charging time
        // between the checkpoint setup and `advance`. The executor's deliberate
        // 1.1s delay below is what deterministically carries this round past the
        // 60s budget; scheduler load must not decide whether the round starts.
        d.checkpoint.elapsed_ms = 59_000;
        d.checkpoint.active_since_ms = None;
        d.advance(&WaveSpace, &HeuristicAdmission, &executor, 8, usize::MAX)
            .unwrap();
        assert_eq!(d.completed(), Some(CompletionReason::BudgetTime));
        assert_eq!(d.checkpoint.next_round, 2);
        assert_eq!(d.champion().unwrap().loss, 1.0);
        assert_eq!(d.checkpoint.discarded.len(), 2);
        assert!(!executor.late_promotion.load(Ordering::SeqCst));
        assert_eq!(executor.revoked.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn public_events_use_opaque_ids_and_projected_phrases_without_programs_or_search_counters() {
        let directory = tempfile::tempdir().unwrap();
        let mut p = plan(directory.path(), 30, Effort::Fast, 24);
        p.planned_workers = 12;
        let store = Arc::new(Store::open(directory.path().join("store")).unwrap());
        let mut d = Discovery::open(store, p).unwrap();
        finish(&mut d, &WaveSpace, &LateImprovement::new(false), 8);
        let projection = d.projection().unwrap();
        let text = serde_json::to_string(&projection).unwrap();
        for word in [
            "private-seed-program",
            "private-proposal-program",
            "round",
            "slot",
            "rung",
            "admitted",
            "stalled",
            "(pipe",
        ] {
            assert!(!text.contains(word), "public leak: {word}");
        }
        for event in projection.as_array().unwrap() {
            if let Some(id) = event["body"]["experiment_id"].as_str() {
                assert!(id.starts_with("exp_"));
                assert_eq!(id.len(), 20);
            }
            if event["kind"] == "algorithm.accepted" {
                assert_eq!(event["body"]["excerpt"], "");
            }
            for key in ["summary", "hypothesis"] {
                if let Some(value) = event["body"][key].as_str() {
                    phrase::lint_public_phrase(value).unwrap();
                }
            }
        }
    }

    #[test]
    fn numerical_artifact_refit_failure_fails_only_the_candidate() {
        struct Unfittable;
        impl Executor for Unfittable {
            fn evaluate(&self, request: &Request) -> Response {
                Controlled { failure: None }.evaluate(request)
            }
            fn fitted_model(&self, _: &Request) -> Result<Vec<u8>> {
                bail!("unit fixture numerical refit failure")
            }
        }
        let directory = tempfile::tempdir().unwrap();
        let p = plan(directory.path(), 30, Effort::Fast, 1);
        let store = Arc::new(Store::open(directory.path().join("store")).unwrap());
        let mut d = Discovery::open(store, p).unwrap();
        finish(&mut d, &TinySpace, &Unfittable, 1);
        assert!(d.champion().is_none());
        assert_eq!(d.completed(), Some(CompletionReason::BudgetEvaluations));
        assert_eq!(d.nondeterminism().count.0, 0);
        let projection = d.projection().unwrap();
        assert!(projection
            .as_array()
            .unwrap()
            .iter()
            .any(|e| e["body"]["failure"]["code"] == "FITNESS_UNDEFINED"));
    }

    #[test]
    fn ultra_evaluates_both_cv_repeats_without_treating_repeated_validation_as_leakage() {
        let directory = tempfile::tempdir().unwrap();
        let p = plan(directory.path(), 30, Effort::Ultra, 9);
        let store = Arc::new(Store::open(directory.path().join("store")).unwrap());
        let mut d = Discovery::open(store, p.clone()).unwrap();
        let candidate = Candidate {
            algorithm_id: "alg_0000".into(),
            dsl: NativeSpace.canonical("(pipe X (ridge))").unwrap(),
        };
        let request = d.request(0, 0, 1, &candidate).unwrap();
        let spec: NativeFolds = serde_json::from_value(request.fold_spec.clone()).unwrap();
        assert_eq!(spec.folds.len(), 10);
        assert_eq!(spec.folds.iter().filter(|f| f.repeat == 0).count(), 5);
        assert_eq!(spec.folds.iter().filter(|f| f.repeat == 1).count(), 5);
        let matrix = Matrix::open(fs::read(&p.matrix_path).unwrap()).unwrap();
        let score = evaluate_native(&request, &matrix).unwrap();
        assert_eq!(score.values.len(), 3);
        assert!(score.values.iter().all(|s| s.is_finite()));
    }

    // Build a tiny real worker launcher against the just-built library. No mock
    // scores, libtest stdout shim, protocol changes or extra Cargo target exist.
    fn native_worker(directory: &Path) -> PathBuf {
        native_worker_fault(directory, "")
    }
    fn native_worker_fault(directory: &Path, fault: &str) -> PathBuf {
        let deps = std::env::var_os("JOPE_DSC_DEPENDENCY_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                std::env::current_exe()
                    .unwrap()
                    .parent()
                    .unwrap()
                    .to_path_buf()
            });
        let library = std::env::var_os("JOPE_DSC_SERVICE_RLIB")
            .map(PathBuf::from)
            .unwrap_or_else(|| deps.parent().unwrap().join("libjope_service.rlib"));
        assert!(
            library.is_file(),
            "build the current service with cargo build -p jope-service --all-targets --locked"
        );
        let source = directory.join("native-worker.rs");
        let binary = directory.join("native-worker");
        let pids = directory.join("worker-pids");
        fs::create_dir_all(&pids).unwrap();
        let fault=match fault {
            "timeout"=>"if request.fold_spec.get(\"refit_output\").is_some() { loop { std::hint::spin_loop(); } }",
            "memory"=>"if request.fold_spec.get(\"refit_output\").is_some() { let mut bytes=vec![0_u8;128*1024*1024]; for i in (0..bytes.len()).step_by(4096) { bytes[i]=1; } loop { std::hint::black_box(&bytes); std::thread::sleep(std::time::Duration::from_millis(10)); } }",
            "r2-timeout"=>"if request.attempt.slot == 0 { loop { std::hint::spin_loop(); } }",
            "r2-memory"=>"if request.attempt.slot == 0 { let mut bytes=vec![0_u8;128*1024*1024]; for i in (0..bytes.len()).step_by(4096) { bytes[i]=1; } loop { std::hint::black_box(&bytes); std::thread::sleep(std::time::Duration::from_millis(10)); } }",
            ""=>"",_=>panic!("unknown owned fixture fault"),
        };
        fs::write(&source,format!("fn main() {{ std::fs::write(std::path::Path::new({:?}).join(std::process::id().to_string()), b\"native\").unwrap(); jope_service::worker::serve(1, |request,matrix| {{ {} jope_service::discover::r#loop::evaluate_native(request,matrix) }}).unwrap(); }}",pids,fault)).unwrap();
        let compiler = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
        let output = Command::new(&compiler)
            .arg("--edition=2021")
            .arg("--crate-name=dsc08_native_worker")
            .arg(&source)
            .arg("-L")
            .arg(format!("dependency={}", deps.display()))
            .arg("--extern")
            .arg(format!("jope_service={}", library.display()))
            .arg("-o")
            .arg(&binary)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "exact current service artifact could not link native worker: {}\n{}",
            library.display(),
            String::from_utf8_lossy(&output.stderr)
        );
        if let Some(receipt) = std::env::var_os("JOPE_DSC_RECEIPT_DIR") {
            let compiler_version = Command::new(&compiler).arg("-vV").output().unwrap();
            ensure_worker_receipt(&PathBuf::from(receipt), &library, &compiler_version.stdout);
        }
        binary
    }

    fn ensure_worker_receipt(receipt: &Path, library: &Path, compiler: &[u8]) {
        fs::create_dir_all(receipt).unwrap();
        fs::write(
            receipt.join("native-worker-build.json"),
            serde_json::to_vec_pretty(&json!({
                "library": library, "sha256": digest(&fs::read(library).unwrap()),
                "compiler": String::from_utf8_lossy(compiler),
                "source": std::env::var("JOPE_BUILD_SOURCE_SHA").unwrap_or_default(),
            }))
            .unwrap(),
        )
        .unwrap();
    }

    #[cfg(unix)]
    fn live_native_workers(directory: &Path) -> usize {
        fs::read_dir(directory)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| entry.file_name().to_string_lossy().parse::<u32>().ok())
            .filter(|pid| {
                let status = Command::new("/bin/kill")
                    .args(["-0", &pid.to_string()])
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status();
                status.is_ok_and(|status| status.success())
            })
            .count()
    }

    fn pool(binary: &Path, workers: usize) -> ProcessExecutor {
        ProcessExecutor::new(
            binary,
            workers,
            &binary.parent().unwrap().join("worker-scratch"),
        )
        .unwrap()
    }
    fn run_store(home: &Path, p: &Plan, binary: &Path, workers: usize) -> (Champion, Value) {
        let store = Arc::new(Store::open(home).unwrap());
        let mut d = Discovery::open(store, p.clone()).unwrap();
        let pool = pool(binary, workers);
        finish(&mut d, &NativeSpace, &pool, workers);
        #[cfg(unix)]
        assert_eq!(
            live_native_workers(&binary.parent().unwrap().join("worker-pids")),
            workers,
            "native worker process count while the pool is still alive"
        );
        assert_eq!(d.nondeterminism().count.0, 0);
        let projection = d.projection().unwrap();
        assert!(
            projection
                .as_array()
                .unwrap()
                .iter()
                .all(|e| e["kind"] != "experiment.finished" || e["body"]["failure"].is_null()),
            "S2 proof requires zero worker failures"
        );
        (d.champion().unwrap().clone(), projection)
    }

    fn same(label: &str, expected: &(Champion, Value), actual: &(Champion, Value)) {
        if let Some(receipt) = std::env::var_os("JOPE_DSC_RECEIPT_DIR") {
            let receipt = PathBuf::from(receipt);
            fs::create_dir_all(&receipt).unwrap();
            fs::write(
                receipt.join("s2-baseline.json"),
                serde_json::to_vec_pretty(expected).unwrap(),
            )
            .unwrap();
            fs::write(
                receipt.join(format!("s2-{label}.json")),
                serde_json::to_vec_pretty(actual).unwrap(),
            )
            .unwrap();
        }
        assert_eq!(expected.0, actual.0, "{label}: champion/model identity");
        let expected = expected.1.as_array().unwrap();
        let actual = actual.1.as_array().unwrap();
        assert_eq!(expected.len(), actual.len(), "{label}: event count");
        for (index, (a, b)) in expected.iter().zip(actual).enumerate() {
            assert_eq!(a, b, "{label}: first event difference at {index}");
        }
    }

    #[test]
    #[ignore = "dedicated bounded-process gate; build jope-service library first and run explicitly"]
    fn promoted_refits_enforce_worker_wall_and_rss_limits_without_engine_fitting() {
        struct RidgeOnly;
        impl SearchSpace for RidgeOnly {
            fn canonical(&self, s: &str) -> Result<String> {
                NativeSpace.canonical(s)
            }
            fn seeds(&self) -> Result<Vec<String>> {
                Ok(vec![self.canonical("(pipe X (ridge))")?])
            }
            fn propose(&self, _: &[String], _: usize, _: &mut ChaCha12Rng) -> Result<Vec<String>> {
                Ok(vec![])
            }
        }
        let engine = std::process::id();
        for (fault, expected) in [
            ("timeout", WorkerFailure::Timeout),
            ("memory", WorkerFailure::MemoryLimit),
        ] {
            let directory = tempfile::tempdir().unwrap();
            let mut p = plan(directory.path(), 30, Effort::Fast, 1);
            p.evaluation_wall_ms = 1000;
            p.evaluation_rss_bytes = 64 * 1024 * 1024;
            let binary = native_worker_fault(directory.path(), fault);
            let workers = pool(&binary, 1);
            let store = Arc::new(Store::open(directory.path().join("store")).unwrap());
            let mut d = Discovery::open(Arc::clone(&store), p.clone()).unwrap();
            finish(&mut d, &RidgeOnly, &workers, 1);
            assert_eq!(std::process::id(), engine);
            assert!(d.champion().is_none());
            let row = store
                .get(&RecordKey::Evaluation {
                    run: p.run_id.clone(),
                    number: 1,
                })
                .unwrap()
                .unwrap();
            assert!(
                row.data["cv_failure"].is_null(),
                "CV must succeed before the injected full-fit fault"
            );
            assert_eq!(
                row.data["scores"].as_array().unwrap().len(),
                3,
                "real CV measurements remain recorded"
            );
            assert_eq!(
                row.data["refit_failure"]["failure"],
                serde_json::to_value(expected).unwrap()
            );
            assert_eq!(row.data["refit_failure"]["attempt"]["slot"], 131072);
            assert_eq!(d.nondeterminism().count.0, 1);
            assert_eq!(
                d.nondeterminism().experiment_ids,
                vec![experiment(&p, 0, 0, 1).unwrap()]
            );
            assert!(!d
                .projection()
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .any(|e| e["kind"] == "algorithm.accepted"));
            println!("bounded promoted refit {fault}: CV succeeded; isolated worker failed as {expected:?}; engine alive; limit receipt retained");
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "dedicated REL-07 R2 native worker/RSS gate; build the current service library first"]
    fn rel07_r2_worker_faults_preserve_engine_memory_and_complete_search() {
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

        struct TwoSeeds;
        impl SearchSpace for TwoSeeds {
            fn canonical(&self, dsl: &str) -> Result<String> {
                NativeSpace.canonical(dsl)
            }
            fn seeds(&self) -> Result<Vec<String>> {
                ["(pipe X (ridge))", "(pipe X (standardize) (ridge))"]
                    .into_iter()
                    .map(|dsl| self.canonical(dsl))
                    .collect()
            }
            fn propose(&self, _: &[String], _: usize, _: &mut ChaCha12Rng) -> Result<Vec<String>> {
                Ok(vec![])
            }
        }
        struct Observed {
            inner: ProcessExecutor,
            failures: Mutex<Vec<(WorkerFailure, u64)>>,
        }
        impl Executor for Observed {
            fn evaluate(&self, request: &Request) -> Response {
                let start = Instant::now();
                let response = self.inner.evaluate(request);
                if let Some(failure) = response.failure {
                    self.failures
                        .lock()
                        .unwrap()
                        .push((failure, start.elapsed().as_millis() as u64));
                }
                response
            }
            fn fitted_model(&self, request: &Request) -> Result<Vec<u8>> {
                self.inner.fitted_model(request)
            }
            fn revoke(&self, attempt: &Attempt) {
                self.inner.revoke(attempt);
            }
        }

        // Warm an ordinary successful native run before establishing the RSS baseline.
        // This charges startup/store/allocator setup neither to the malicious candidate
        // nor to a fabricated extra memory allowance. The 10% limit stays exact.
        let warm = tempfile::tempdir().unwrap();
        let warm_plan = plan(warm.path(), 30, Effort::Fast, 1);
        let warm_worker = native_worker(warm.path());
        run_store(&warm.path().join("store"), &warm_plan, &warm_worker, 1);

        for (fault, expected) in [
            ("timeout", WorkerFailure::Timeout),
            ("memory", WorkerFailure::MemoryLimit),
        ] {
            let directory = tempfile::tempdir().unwrap();
            let mut p = plan(directory.path(), 30, Effort::Fast, 2);
            p.planned_workers = 1;
            p.evaluation_wall_ms = 1000;
            p.evaluation_rss_bytes = 96 * 1024 * 1024;
            let binary = native_worker_fault(directory.path(), &format!("r2-{fault}"));
            let store = Arc::new(Store::open(directory.path().join("store")).unwrap());
            let mut d = Discovery::open(Arc::clone(&store), p.clone()).unwrap();
            let observed = Observed {
                inner: pool(&binary, 1),
                failures: Mutex::new(vec![]),
            };
            let prime = d
                .request(
                    99,
                    0,
                    1,
                    &Candidate {
                        algorithm_id: "alg_00000099".into(),
                        dsl: NativeSpace.canonical("(pipe X (ridge))").unwrap(),
                    },
                )
                .unwrap();
            assert_eq!(observed.inner.evaluate(&prime).failure, None);
            let engine = std::process::id();
            let baseline = crate::sampler::sample(engine).unwrap().1;
            assert!(baseline > 0);
            let peak = AtomicU64::new(baseline);
            let samples = AtomicU64::new(0);
            let stop = AtomicBool::new(false);
            let sampling_failed = AtomicBool::new(false);
            std::thread::scope(|scope| {
                let observer = scope.spawn(|| {
                    while !stop.load(Ordering::Acquire) {
                        match crate::sampler::sample(engine) {
                            Ok((_, rss)) if rss > 0 => {
                                peak.fetch_max(rss, Ordering::Relaxed);
                                samples.fetch_add(1, Ordering::Relaxed);
                            }
                            _ => {
                                sampling_failed.store(true, Ordering::Release);
                                break;
                            }
                        }
                        std::thread::sleep(Duration::from_millis(5));
                    }
                });
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    finish(&mut d, &TwoSeeds, &observed, 1);
                }));
                stop.store(true, Ordering::Release);
                observer.join().unwrap();
                if let Err(panic) = outcome {
                    std::panic::resume_unwind(panic);
                }
            });
            assert!(
                !sampling_failed.load(Ordering::Acquire),
                "RSS sensor unavailable"
            );
            let sample_count = samples.load(Ordering::Relaxed);
            let peak_rss = peak.load(Ordering::Relaxed);
            assert!(sample_count > 0);
            assert!(
                peak_rss * 10 <= baseline * 11,
                "engine RSS exceeded 110% of warmed baseline: {peak_rss}/{baseline}"
            );
            assert!(
                d.completed().is_some(),
                "search must complete after the isolated fault"
            );
            let champion = d
                .champion()
                .expect("healthy candidate must still produce a champion");
            assert!(champion.value.is_finite() && champion.loss.is_finite());
            store.verify_blob(&champion.model).unwrap();
            assert_eq!(d.nondeterminism().count.0, 1);
            let failures = observed.failures.lock().unwrap();
            assert_eq!(failures.len(), 1);
            assert_eq!(failures[0].0, expected);
            let elapsed_ms = failures[0].1;
            assert!(
                elapsed_ms <= p.evaluation_wall_ms + 2000,
                "worker fault exceeded its limit plus two seconds"
            );
            drop(failures);
            let pids = directory.path().join("worker-pids");
            let workers_started = fs::read_dir(&pids).unwrap().count();
            assert!(workers_started >= 2, "faulted worker must be replaced");
            drop(observed);
            let workers_remaining = live_native_workers(&pids);
            assert_eq!(workers_remaining, 0, "owned worker cleanup must finish");
            println!(
                "REL07_R2 {}",
                json!({
                    "schema":"jope.rel07.worker-isolation/v1", "fault":fault,
                    "failure":format!("{expected:?}"), "wall_limit_ms":p.evaluation_wall_ms,
                    "elapsed_ms":elapsed_ms, "baseline_rss_bytes":baseline,
                    "peak_rss_bytes":peak_rss, "rss_samples":sample_count,
                    "completed":true, "champion_present":true,
                    "workers_started":workers_started, "workers_remaining":workers_remaining,
                    "train_sha256":TRAIN_SHA,
                    "source_sha":std::env::var("JOPE_BUILD_SOURCE_SHA").unwrap_or_default()
                })
            );
        }
    }

    #[test]
    #[ignore = "dedicated service M1 entry acceptance; build jope-service library first and run explicitly"]
    fn service_m1_entry_executes_native_discovery_and_returns_only_public_projection() {
        let directory = tempfile::tempdir().unwrap();
        let p = plan(directory.path(), 30, Effort::Fast, 1);
        let binary = native_worker(directory.path());
        let input = directory.path().join("m1.json");
        fs::write(
            &input,
            serde_json::to_vec(
                &json!({"home":directory.path().join("m1-store"),"plan":p,"live_workers":2}),
            )
            .unwrap(),
        )
        .unwrap();
        let result = run_m1(&input, &binary).unwrap();
        assert_eq!(result["stage"], "search_finished");
        assert!(result["champion"]["algorithm_id"]
            .as_str()
            .unwrap()
            .starts_with("alg_"));
        assert!(result["champion"].get("candidate").is_none());
        let text = serde_json::to_string(&result).unwrap();
        for word in [
            "(pipe", "\"dsl\"", "round-", "slot-", "rung-", "stalled", "admitted",
        ] {
            assert!(!text.contains(word), "M1 public leak: {word}");
        }
        assert_eq!(result["nondeterminism"]["count"], "0");
        if let Some(receipt) = std::env::var_os("JOPE_DSC_RECEIPT_DIR") {
            let receipt = PathBuf::from(receipt);
            fs::create_dir_all(&receipt).unwrap();
            fs::write(
                receipt.join("service-m1.json"),
                serde_json::to_vec_pretty(&result).unwrap(),
            )
            .unwrap();
        }
    }

    #[test]
    #[ignore = "dedicated native process gate: first cargo build -p jope-service --all-targets --locked, then run this exact test with --ignored"]
    fn real_native_workers_preserve_projection_across_capacity_pause_and_sigkill() {
        let directory = tempfile::tempdir().unwrap();
        let mut p = plan(directory.path(), 80, Effort::Medium, 25);
        p.wall_seconds = 600;
        p.evaluation_wall_ms = 60_000;
        assert_eq!(p.rounds_in_flight(), 2);
        let binary = native_worker(directory.path());
        let baseline = run_store(&directory.path().join("one"), &p, &binary, 1);
        same(
            "eight",
            &baseline,
            &run_store(&directory.path().join("eight"), &p, &binary, 8),
        );
        let pause_home = directory.path().join("pause");
        {
            let store = Arc::new(Store::open(&pause_home).unwrap());
            let mut d = Discovery::open(store, p.clone()).unwrap();
            let workers = pool(&binary, 8);
            d.advance(&NativeSpace, &HeuristicAdmission, &workers, 8, usize::MAX)
                .unwrap();
            d.advance(&NativeSpace, &HeuristicAdmission, &workers, 8, 1)
                .unwrap();
            assert_eq!(d.checkpoint.pending.len(), 1);
            d.pause().unwrap();
        }
        same("pause", &baseline, &run_store(&pause_home, &p, &binary, 1));
        #[cfg(unix)]
        {
            let kill_home = directory.path().join("kill");
            let plan_path = directory.path().join("plan.json");
            fs::write(&plan_path, serde_json::to_vec(&p).unwrap()).unwrap();
            let mut child = Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "discover::r#loop::tests::crash_parent",
                    "--ignored",
                    "--nocapture",
                    "--quiet",
                ])
                .env("DSC08_KILL_HOME", &kill_home)
                .env("DSC08_PLAN", &plan_path)
                .env("DSC08_WORKER", &binary)
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap();
            let mut reader = BufReader::new(child.stdout.take().unwrap());
            let (signal, observed) = std::sync::mpsc::sync_channel(1);
            let reader = std::thread::spawn(move || {
                let mut line = String::new();
                loop {
                    line.clear();
                    if reader.read_line(&mut line).unwrap() == 0 {
                        break;
                    }
                    if line.trim() == "DSC08_NATIVE_RESULT_UNCOMMITTED" {
                        let _ = signal.send(());
                        break;
                    }
                }
            });
            let ready = observed.recv_timeout(Duration::from_secs(45));
            // This Child is exclusively owned by this test; kill() sends SIGKILL
            // on Unix. WRK-01 also kills its native children on parent death.
            child.kill().unwrap();
            let status = child.wait().unwrap();
            reader.join().unwrap();
            assert!(
                ready.is_ok(),
                "native crash helper did not reach its uncommitted result within 45s"
            );
            use std::os::unix::process::ExitStatusExt;
            assert_eq!(status.signal(), Some(9));
            {
                let store = Arc::new(Store::open(&kill_home).unwrap());
                let record = store
                    .get(&RecordKey::Run(p.run_id.clone()))
                    .unwrap()
                    .unwrap();
                assert_eq!(record.run.as_ref().unwrap().state, RunState::Paused);
                assert!(record.run.unwrap().interrupted);
                let checkpoint: Checkpoint =
                    serde_json::from_value(record.data["checkpoint"].clone()).unwrap();
                assert_eq!(checkpoint.next_round, 2);
                assert_eq!(checkpoint.pending.len(), 1);
            }
            same("sigkill", &baseline, &run_store(&kill_home, &p, &binary, 1));
        }
        if let Some(receipt) = std::env::var_os("JOPE_DSC_RECEIPT_DIR") {
            let receipt = PathBuf::from(receipt);
            fs::write(
                receipt.join("s2-plan.json"),
                serde_json::to_vec_pretty(&p).unwrap(),
            )
            .unwrap();
            fs::copy(&p.matrix_path, receipt.join("s2-fit.jmat")).unwrap();
            fs::copy(
                directory.path().join("fit-source-rows.json"),
                receipt.join("s2-fit-source-rows.json"),
            )
            .unwrap();
        }
        println!("DSC08 S2: real native CV; planned R=2; workers1/8, mid-wave pause, SIGKILL/recovery: identical champion/model/projection; zero limits");
    }

    #[test]
    #[ignore = "supervised SIGKILL child; invoked by real_native_workers_preserve_projection_across_capacity_pause_and_sigkill"]
    fn crash_parent() {
        let home = PathBuf::from(std::env::var_os("DSC08_KILL_HOME").unwrap());
        let p: Plan =
            serde_json::from_slice(&fs::read(std::env::var_os("DSC08_PLAN").unwrap()).unwrap())
                .unwrap();
        let binary = PathBuf::from(std::env::var_os("DSC08_WORKER").unwrap());
        let store = Arc::new(Store::open(home).unwrap());
        let mut d = Discovery::open(store, p).unwrap();
        let workers = pool(&binary, 8);
        d.advance(&NativeSpace, &HeuristicAdmission, &workers, 8, usize::MAX)
            .unwrap();
        d.advance(&NativeSpace, &HeuristicAdmission, &workers, 8, 1)
            .unwrap();
        struct Signal<'a>(&'a ProcessExecutor);
        impl Executor for Signal<'_> {
            fn evaluate(&self, request: &Request) -> Response {
                let _response = self.0.evaluate(request);
                println!("DSC08_NATIVE_RESULT_UNCOMMITTED");
                std::io::stdout().flush().unwrap();
                loop {
                    std::thread::park_timeout(Duration::from_secs(1));
                }
            }
            fn fitted_model(&self, request: &Request) -> Result<Vec<u8>> {
                self.0.fitted_model(request)
            }
        }
        d.advance(
            &NativeSpace,
            &HeuristicAdmission,
            &Signal(&workers),
            8,
            usize::MAX,
        )
        .unwrap();
    }

    #[test]
    #[ignore = "dedicated real-data quality gate: first cargo build -p jope-service --locked, then run this exact test with --ignored"]
    fn medium_real_qlt05_regression_beats_ridge_cv() {
        let directory = tempfile::tempdir().unwrap();
        let p = plan(directory.path(), 800, Effort::Medium, 128);
        let binary = native_worker(directory.path());
        let workers = pool(&binary, 8);
        let store = Arc::new(Store::open(directory.path().join("quality")).unwrap());
        let mut d = Discovery::open(store, p.clone()).unwrap();
        let ridge = NativeSpace.canonical("(pipe X (ridge))").unwrap();
        let request = d
            .request(
                999,
                0,
                1,
                &Candidate {
                    algorithm_id: "alg_9999".into(),
                    dsl: ridge,
                },
            )
            .unwrap();
        let baseline = workers.evaluate(&request);
        assert_eq!(baseline.failure, None);
        finish(&mut d, &NativeSpace, &workers, 8);
        let champion = d
            .champion()
            .expect("medium search must produce a finite rung-1 model");
        println!("DSC08 QLT05 source={TRAIN_SHA}, FIT=800, seed=73, folds=5, medium128/R2; ridge_cv_rmse={}, champion_cv_rmse={}, admissions={}, reason={:?}, nondeterminism={:?}",baseline.scores[2],champion.value,d.admitted(),d.completed(),d.nondeterminism());
        assert!(
            champion.loss < baseline.scores[1],
            "medium must strictly improve ridge on the same pinned CV folds"
        );
        // Quality may be time-bound under the explicit spec §5.6 exception.
        // Every actual limit must be disclosed; only the separate zero-limit
        // S2 runs establish deterministic replay equality.
        let projection = d.projection().unwrap();
        let limits: Vec<String> = projection
            .as_array()
            .unwrap()
            .iter()
            .filter(|e| {
                matches!(
                    e["body"]["failure"]["reason"].as_str(),
                    Some(reason) if reason == public_failure(WorkerFailure::Timeout) || reason == public_failure(WorkerFailure::MemoryLimit)
                )
            })
            .map(|e| e["body"]["experiment_id"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(
            limits,
            d.nondeterminism().experiment_ids,
            "all real worker limits must be listed, in commit order"
        );
        assert_eq!(limits.len() as u64, d.nondeterminism().count.0);
        d.store.verify_blob(&champion.model).unwrap();
        if let Some(receipt) = std::env::var_os("JOPE_DSC_RECEIPT_DIR") {
            let receipt = PathBuf::from(receipt);
            fs::create_dir_all(&receipt).unwrap();
            fs::copy(&p.matrix_path, receipt.join("quality-fit.jmat")).unwrap();
            fs::copy(
                directory.path().join("fit-source-rows.json"),
                receipt.join("quality-fit-source-rows.json"),
            )
            .unwrap();
            fs::write(receipt.join("quality.json"),serde_json::to_vec_pretty(&json!({"plan":p,"source_train_sha256":TRAIN_SHA,"ridge_scores":baseline.scores,"champion":champion,"admissions":d.admitted(),"completion_reason":d.completed(),"budget_bound_by":d.budget_bound_by(),"nondeterminism":d.nondeterminism(),"projection":d.projection().unwrap()})).unwrap()).unwrap();
        }
    }
    #[cfg(feature = "research")]
    mod durable_capture {
        use super::*;
        use crate::research::{
            capture::{Binding, CaptureConfig, CaptureProvider, SlateObservation},
            outbox::materialize_controls,
            *,
        };

        struct ClockScope(Option<Arc<std::sync::atomic::AtomicU64>>);
        impl ClockScope {
            fn frozen() -> (Self, Arc<std::sync::atomic::AtomicU64>) {
                let clock = Arc::new(std::sync::atomic::AtomicU64::new(1_000_000));
                let previous = TEST_CLOCK.with(|current| current.replace(Some(Arc::clone(&clock))));
                (Self(previous), clock)
            }
        }
        impl Drop for ClockScope {
            fn drop(&mut self) {
                TEST_CLOCK.with(|clock| {
                    clock.replace(self.0.take());
                });
            }
        }

        struct Space {
            seeds: usize,
            duplicate: bool,
        }
        impl SearchSpace for Space {
            fn canonical(&self, dsl: &str) -> Result<String> {
                Ok(dsl.trim().into())
            }
            fn seeds(&self) -> Result<Vec<String>> {
                Ok((0..self.seeds).map(|i| format!("seed-{i}")).collect())
            }
            fn propose(
                &self,
                _: &[String],
                count: usize,
                rng: &mut ChaCha12Rng,
            ) -> Result<Vec<String>> {
                let nonce = rng.next_u64();
                let mut values: Vec<_> = (0..count).map(|i| format!("p-{nonce}-{i}")).collect();
                if self.duplicate {
                    values.insert(1, "seed-0".into());
                    values.insert(3, format!(" {} ", values[0]));
                }
                Ok(values)
            }
        }
        const SPACE: Space = Space {
            seeds: 4,
            duplicate: false,
        };

        struct Provider {
            observations: AtomicUsize,
            bad_gate: bool,
        }
        impl CaptureProvider for Provider {
            fn validate_binding(&self, _: &Binding, _: &Plan) -> Result<()> {
                ensure!(!self.bad_gate, "control gate unavailable before dispatch");
                Ok(())
            }

            fn observe(
                &self,
                round: u64,
                dsls: &[String],
                _: &[RecordedReveal],
            ) -> Result<SlateObservation> {
                self.observations.fetch_add(1, Ordering::SeqCst);
                Ok(SlateObservation {
                    programs: dsls
                        .iter()
                        .enumerate()
                        .map(|(i, dsl)| RecordedProgram {
                            program_sha256: digest(dsl.as_bytes()),
                            operator_family: "synthetic_control".into(),
                            feat412: vec![i as f32; 412],
                            proj16: vec![1.000_000_000_000_01 + i as f64; 16],
                        })
                        .collect(),
                    revealed_indices: if round == 0 { vec![] } else { (0..4).collect() },
                })
            }
            fn measurement(
                &self,
                request: &Request,
                response: &Response,
            ) -> Result<Rung0Measurement> {
                ensure!(!self.bad_gate, "control gate unavailable");
                Ok(Rung0Measurement::Success {
                    surrogate: response.scores[0],
                    objective: response.scores[1],
                    gate: request.dsl.ends_with('0'),
                })
            }
            fn label_seed(&self, round: u64, index: usize) -> Result<u64> {
                Ok(9_000_000 + round * 16 + index as u64)
            }
        }
        fn context(
            p: &Plan,
            attempts: u64,
            all: bool,
            bad_gate: bool,
        ) -> (Arc<CaptureContext>, Arc<Provider>) {
            let provider = Arc::new(Provider {
                observations: AtomicUsize::new(0),
                bad_gate,
            });
            let binding = Binding {
                config: CaptureConfig {
                    max_attempts: attempts,
                    max_rounds: 16.min(attempts as usize),
                    max_bytes: 8 * 1024 * 1024,
                    evaluate_all_proposals: all,
                    seed_protocol_sha256: digest(b"synthetic-control-extra-seeds-v1"),
                },
                identity: RecordIdentity {
                    run_id: p.run_id.to_string(),
                    task: RecordTask::Regression,
                    evidence: RecordEvidence::SyntheticControl,
                    partition: "train".into(),
                    source_commit: "a".repeat(40),
                    corpus_manifest_sha256: digest(b"control-corpus"),
                    dataset_sha256: p.matrix_sha256.clone(),
                    family_sha256: digest(b"control-family"),
                    problem_sha256: p.problem_digest.clone(),
                    bundle_sha256: p.bundle_digest.clone(),
                    conditioning_sha256: digest(b"control-conditioning"),
                    evaluator_sha256: digest(b"control-evaluator"),
                    dope_weights_sha256: DOPE_SHA256.into(),
                },
                seed: p.seed,
                tokens: vec![vec![0.25; 32]; 10],
                mask: vec![true; 10],
                matrix_sha256: p.matrix_sha256.clone(),
            };
            (
                Arc::new(CaptureContext::control(binding, provider.clone()).unwrap()),
                provider,
            )
        }
        fn control_plan(home: &Path, budget: u64) -> Plan {
            let csv = format!(
                "x,target\n{}",
                (0..40)
                    .map(|i| format!("{i},{}\n", i * 2))
                    .collect::<String>()
            );
            let table = jope_runtime::table::parse_csv(
                csv.as_bytes(),
                &jope_runtime::limits::Limits::default(),
            )
            .unwrap();
            let bytes = crate::dataset::jmat::encode(&table).unwrap();
            let matrix_path = home.join("control.jmat");
            fs::write(&matrix_path, &bytes).unwrap();
            Plan {
                run_id: "run_01JBD8Q6N4V2T5Y8K3M7P9R1S0".parse().unwrap(),
                dataset_id: "ds_21fbfe8b53dbcd605d79486b".parse().unwrap(),
                problem_digest: digest(&bytes),
                bundle_digest: digest(b"synthetic-control-bundle"),
                binary_version: "synthetic-control-v1".into(),
                matrix_path,
                matrix_sha256: digest(&bytes),
                target: "target".into(),
                features: vec!["x".into()],
                objective: "rmse".into(),
                seed: 73,
                planned_workers: 8,
                effort: Effort::Fast,
                evaluations: budget,
                wall_seconds: 60,
                evaluation_wall_ms: 15000,
                evaluation_rss_bytes: 1024 * 1024,
                groups: None,
                times: None,
                gap_rows: 0,
            }
        }
        struct Evaluator {
            calls: Mutex<Vec<Request>>,
            store: Arc<Store>,
            fail_extra: bool,
            panic_at: Option<usize>,
        }
        impl Evaluator {
            fn new(store: Arc<Store>) -> Self {
                Self {
                    calls: Mutex::new(vec![]),
                    store,
                    fail_extra: false,
                    panic_at: None,
                }
            }
        }
        impl Executor for Evaluator {
            fn evaluate(&self, request: &Request) -> Response {
                let count = {
                    let mut calls = self.calls.lock().unwrap();
                    calls.push(request.clone());
                    calls.len()
                };
                if let Some(row) = self
                    .store
                    .get(&RecordKey::Run(request.attempt.run.clone()))
                    .unwrap()
                {
                    if row.data["checkpoint"]["version"] == 3 {
                        let state: CaptureState =
                            serde_json::from_value(row.data["checkpoint"]["capture"].clone())
                                .unwrap();
                        state.validate_storage(&self.store).unwrap();
                        assert!(
                            state.roots().iter().all(|r| row.blobs.contains(r)),
                            "request ran without durable roots"
                        );
                        let reservations = state
                            .rounds
                            .values()
                            .map(|r| r.attempts.len())
                            .sum::<usize>();
                        assert_eq!(
                            state.charged, reservations as u64,
                            "all reservations must precede execution"
                        );
                    }
                }
                assert_ne!(self.panic_at, Some(count), "control evaluator abort");
                let extra = (32..65536).contains(&request.attempt.slot);
                if extra && self.fail_extra {
                    return Response::failed(request.attempt.clone(), WorkerFailure::Timeout, 7);
                }
                let loss = if extra {
                    -10000.0
                } else if request.dsl.starts_with("seed") {
                    100.0
                } else {
                    10.0
                };
                Response {
                    attempt: request.attempt.clone(),
                    scores: vec![loss, loss, 777.0],
                    fold_scores: vec![],
                    predictions_digest: digest(request.dsl.as_bytes()),
                    wall_ms: 7,
                    failure: None,
                }
            }
            fn fitted_model(&self, request: &Request) -> Result<Vec<u8>> {
                Ok(request.dsl.as_bytes().to_vec())
            }
        }
        fn records(d: &Discovery) -> Vec<ResearchRound> {
            d.checkpoint
                .capture
                .as_ref()
                .unwrap()
                .rounds
                .values()
                .filter_map(|r| r.committed.as_ref())
                .map(|r| CaptureState::read(&d.store, r).unwrap())
                .collect()
        }
        fn stored(d: &Discovery) -> Value {
            d.store
                .get(&RecordKey::Run(d.checkpoint.plan.run_id.clone()))
                .unwrap()
                .unwrap()
                .data["checkpoint"]
                .clone()
        }
        fn run_to_end(d: &mut Discovery, e: &Evaluator, workers: usize) {
            for _ in 0..16 {
                if d.completed().is_some() {
                    return;
                }
                d.advance(&SPACE, &HeuristicAdmission, e, workers, usize::MAX)
                    .unwrap();
            }
            panic!("control exceeded round bound");
        }

        #[test]
        fn labels_are_private_and_one_eight_workers_preserve_normal_gp() {
            let home = tempfile::tempdir().unwrap();
            let p = control_plan(home.path(), 48);
            let mut baseline = None;
            for workers in [1, 8] {
                let store =
                    Arc::new(Store::open(home.path().join(format!("capture-{workers}"))).unwrap());
                let (c, provider) = context(&p, 40, true, false);
                let mut d = Discovery::open_research(store.clone(), p.clone(), c).unwrap();
                let e = Evaluator::new(store);
                for _ in 0..2 {
                    d.advance(&SPACE, &HeuristicAdmission, &e, workers, usize::MAX)
                        .unwrap();
                }
                assert_eq!(provider.observations.load(Ordering::SeqCst), 3);
                assert_eq!(d.admitted(), 12);
                assert_eq!(d.checkpoint.capture.as_ref().unwrap().charged, 20);
                assert_eq!(
                    d.attempt, 16,
                    "extra slots must not consume admitted attempt counter"
                );
                let captured = records(&d);
                assert_eq!(captured.len(), 3);
                assert!(captured[1].outcomes.iter().any(|o| matches!(
                    o.measurement,
                    Rung0Measurement::Success {
                        objective: -10000.0,
                        ..
                    }
                )));
                assert!(captured.iter().all(|r| r.policy.slate_costs.is_none()));
                assert!(captured[1].revealed.iter().all(|r| r.surrogate == 100.0));
                let label_keys: Vec<_> = captured
                    .iter()
                    .flat_map(|r| {
                        r.proposals
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| !r.policy.admitted.contains(i))
                            .map(|(_, p)| p.program_sha256.clone())
                    })
                    .collect();
                assert!(label_keys.iter().all(|k| !d.checkpoint.memo.contains(k)));
                let public = d.projection().unwrap();
                let public_text = serde_json::to_string(&public).unwrap();
                for private in [
                    "feat412",
                    "proj16",
                    "conditioning_sha256",
                    "slate_costs",
                    "synthetic_control",
                    "-10000",
                ] {
                    assert!(!public_text.contains(private));
                }
                let state = serde_json::to_value((
                    &d.checkpoint.programs,
                    &d.checkpoint.elites,
                    &d.checkpoint.champion,
                ))
                .unwrap();
                if let Some((old_state, old_public)) = &baseline {
                    assert_eq!(&state, old_state);
                    assert_eq!(&public, old_public);
                } else {
                    baseline = Some((state, public));
                }
            }
            let store = Arc::new(Store::open(home.path().join("normal")).unwrap());
            let mut normal = Discovery::open(store.clone(), p).unwrap();
            let e = Evaluator::new(store);
            for _ in 0..2 {
                normal
                    .advance(&SPACE, &HeuristicAdmission, &e, 1, usize::MAX)
                    .unwrap();
            }
            assert_eq!(
                serde_json::to_value((
                    &normal.checkpoint.programs,
                    &normal.checkpoint.elites,
                    &normal.checkpoint.champion
                ))
                .unwrap(),
                baseline.as_ref().unwrap().0
            );
            assert_eq!(normal.projection().unwrap(), baseline.unwrap().1);
            assert_eq!(stored(&normal)["version"], 2);
            assert!(stored(&normal).get("capture").is_none());
            assert!(stored(&normal)["pending"]
                .as_array()
                .unwrap()
                .iter()
                .all(|r| r.get("capture").is_none()));
        }

        #[test]
        fn full_slates_keep_source_indices_and_original_precision() {
            let home = tempfile::tempdir().unwrap();
            let p = control_plan(home.path(), 40);
            let store = Arc::new(Store::open(home.path().join("store")).unwrap());
            let (c, _) = context(&p, 40, true, false);
            let mut d = Discovery::open_research(store.clone(), p, c).unwrap();
            let e = Evaluator::new(store);
            let space = Space {
                seeds: 4,
                duplicate: true,
            };
            d.advance(&space, &HeuristicAdmission, &e, 1, usize::MAX)
                .unwrap();
            d.checkpoint.stalled = 100; // Explicit software control for the exploring policy branch.
            d.commit_control(d.checkpoint.clone(), RunState::Running)
                .unwrap();
            d.advance(&space, &HeuristicAdmission, &e, 8, usize::MAX)
                .unwrap();
            let r = records(&d);
            assert_eq!(r[1].proposals.len(), 16);
            assert!(r[1].policy.explore);
            assert_eq!(&r[1].source_indices[..3], &[0, 2, 4]);
            assert_eq!(r[1].proposals[0].proj16[0], 1.000_000_000_000_01);
            assert_eq!(r[1].revealed.len(), 4);
            assert_eq!(r[1].outcomes.len(), 16);
            let mut rng = round_rng(d.checkpoint.plan.seed, 1);
            let raw = space.propose(&[], 16, &mut rng).unwrap();
            let mut seen = (0..4)
                .map(|i| digest(format!("seed-{i}").as_bytes()))
                .collect::<BTreeSet<_>>();
            let eligible = raw
                .iter()
                .map(|dsl| space.canonical(dsl).unwrap())
                .filter(|dsl| seen.insert(digest(dsl.as_bytes())))
                .collect();
            let expected = HeuristicAdmission
                .admit(eligible, true, &mut rng)
                .into_iter()
                .map(|dsl| digest(dsl.as_bytes()))
                .collect::<Vec<_>>();
            assert_eq!(
                r[1].policy
                    .admitted
                    .iter()
                    .map(|i| r[1].proposals[*i].program_sha256.clone())
                    .collect::<Vec<_>>(),
                expected
            );
        }

        #[test]
        fn materialization_reads_committed_state_and_python_counts_both_modes() {
            for all in [false, true] {
                let home = tempfile::tempdir().unwrap();
                let p = control_plan(home.path(), 40);
                let run = p.run_id.clone();
                let store = Arc::new(Store::open(home.path().join("store")).unwrap());
                let (c, _) = context(&p, if all { 20 } else { 12 }, all, false);
                let mut d = Discovery::open_research(store.clone(), p, c).unwrap();
                let mut e = Evaluator::new(store.clone());
                e.fail_extra = true;
                run_to_end(&mut d, &e, 8);
                let output = home.path().join("records.jsonl");
                assert!(
                    materialize_committed(&store, &run, &output).is_err(),
                    "controls cannot claim Engine evidence"
                );
                assert!(!output.exists());
                assert_eq!(materialize_controls(&store, &run, &output).unwrap(), 3);
                let bytes = fs::read(&output).unwrap();
                assert!(materialize_controls(&store, &run, &output).is_err());
                assert_eq!(fs::read(&output).unwrap(), bytes);
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    assert_eq!(
                        fs::metadata(&output).unwrap().permissions().mode() & 0o777,
                        0o600
                    );
                }
                let script = Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../python/src/jope/north_star/record_reader.py");
                let mut cmd = Command::new("python3");
                cmd.arg(script).arg(&output).arg("--allow-synthetic");
                if all {
                    cmd.arg("--require-evaluate-all");
                }
                let result = cmd.output().unwrap();
                assert!(
                    result.status.success(),
                    "{}",
                    String::from_utf8_lossy(&result.stderr)
                );
                let counts: Value = serde_json::from_slice(&result.stdout).unwrap();
                assert_eq!(
                    counts,
                    json!({"rounds":3,"proposals":20,"outcomes":if all{20}else{12},"failures":if all{8}else{0},"incomplete_rounds":0})
                );
                assert!(
                    fs::read_dir(home.path())
                        .unwrap()
                        .filter_map(Result::ok)
                        .filter(|e| e
                            .file_name()
                            .to_string_lossy()
                            .starts_with(".jope-research-"))
                        .count()
                        >= 2,
                    "staging evidence must remain"
                );
            }
        }

        #[test]
        fn budget_truncation_is_retained_but_not_exported() {
            let home = tempfile::tempdir().unwrap();
            let p = control_plan(home.path(), 40);
            let run = p.run_id.clone();
            let store = Arc::new(Store::open(home.path().join("store")).unwrap());
            let (c, _) = context(&p, 7, true, false);
            let mut d = Discovery::open_research(store.clone(), p, c).unwrap();
            let e = Evaluator::new(store.clone());
            run_to_end(&mut d, &e, 1);
            assert_eq!(d.checkpoint.capture.as_ref().unwrap().charged, 7);
            let r = records(&d);
            assert_eq!(r.len(), 2);
            assert_eq!(r[1].outcomes.len(), 3);
            assert!(!r[1].capture_complete);
            assert!(materialize_controls(&store, &run, &home.path().join("records")).is_err());
        }

        #[test]
        fn pause_reopen_reuses_known_wave_without_proposal_or_admission_replay() {
            struct NoSpace;
            impl SearchSpace for NoSpace {
                fn canonical(&self, _: &str) -> Result<String> {
                    panic!("recanonicalized")
                }
                fn seeds(&self) -> Result<Vec<String>> {
                    panic!("reseeded")
                }
                fn propose(
                    &self,
                    _: &[String],
                    _: usize,
                    _: &mut ChaCha12Rng,
                ) -> Result<Vec<String>> {
                    panic!("reproposed")
                }
            }
            let home = tempfile::tempdir().unwrap();
            let p = control_plan(home.path(), 40);
            let (c, provider) = context(&p, 20, true, false);
            let store = Arc::new(Store::open(home.path().join("store")).unwrap());
            let mut d = Discovery::open_research(store.clone(), p.clone(), c.clone()).unwrap();
            let e = Evaluator::new(store.clone());
            d.advance(&SPACE, &HeuristicAdmission, &e, 8, usize::MAX)
                .unwrap();
            d.advance(&SPACE, &HeuristicAdmission, &e, 8, 1).unwrap();
            assert_eq!(d.checkpoint.pending.len(), 1);
            assert_eq!(d.checkpoint.capture.as_ref().unwrap().charged, 20);
            d.pause().unwrap();
            drop(d);
            let before = e.calls.lock().unwrap().len();
            let mut d = Discovery::open_research(store, p, c).unwrap();
            d.advance(&NoSpace, &HeuristicAdmission, &e, 1, usize::MAX)
                .unwrap();
            assert_eq!(provider.observations.load(Ordering::SeqCst), 3);
            assert_eq!(
                e.calls.lock().unwrap().len() - before,
                0,
                "known pending outcomes must not execute again"
            );
            assert_eq!(records(&d).len(), 3);
            assert!(d.completed().is_some());
        }

        #[test]
        fn missing_gate_or_aborted_executor_retains_unknown_and_never_retries() {
            for bad_gate in [false, true] {
                let home = tempfile::tempdir().unwrap();
                let p = control_plan(home.path(), 40);
                let run = p.run_id.clone();
                let (c, _) = context(&p, 20, true, bad_gate);
                let store = Arc::new(Store::open(home.path().join("store")).unwrap());
                if bad_gate {
                    assert!(Discovery::open_research(store.clone(), p.clone(), c.clone()).is_err());
                    assert!(store.get(&RecordKey::Run(run)).unwrap().is_none());
                    continue;
                }
                let mut d = Discovery::open_research(store.clone(), p.clone(), c.clone()).unwrap();
                let mut e = Evaluator::new(store.clone());
                if !bad_gate {
                    e.panic_at = Some(2);
                }
                assert!(d
                    .advance(&SPACE, &HeuristicAdmission, &e, 1, usize::MAX)
                    .is_err());
                assert!(d.poisoned);
                assert_eq!(stored(&d)["capture"]["charged"], 4);
                assert!(d.projection().unwrap().as_array().unwrap().is_empty());
                drop(d);
                let mut d = Discovery::open_research(store.clone(), p, c).unwrap();
                let calls = e.calls.lock().unwrap().len();
                assert!(d
                    .advance(&SPACE, &HeuristicAdmission, &e, 1, usize::MAX)
                    .is_err());
                assert_eq!(calls, e.calls.lock().unwrap().len());
                assert!(d.checkpoint.capture.as_ref().unwrap().tainted);
                assert!(materialize_controls(&store, &run, &home.path().join("records")).is_err());
            }
        }

        #[test]
        fn empty_oversized_seed_and_unobserved_admission_refuse_before_dispatch() {
            for seeds in [0, 17] {
                let home = tempfile::tempdir().unwrap();
                let p = control_plan(home.path(), 40);
                let store = Arc::new(Store::open(home.path().join("store")).unwrap());
                let (c, _) = context(&p, 20, true, false);
                let mut d = Discovery::open_research(store.clone(), p, c).unwrap();
                let e = Evaluator::new(store);
                assert!(d
                    .advance(
                        &Space {
                            seeds,
                            duplicate: false
                        },
                        &HeuristicAdmission,
                        &e,
                        1,
                        usize::MAX
                    )
                    .is_err());
                assert!(e.calls.lock().unwrap().is_empty());
                assert_eq!(stored(&d)["capture"]["charged"], 0);
            }
            struct Unobserved;
            impl Admission for Unobserved {
                fn admit(&self, _: Vec<String>, _: bool, _: &mut ChaCha12Rng) -> Vec<String> {
                    panic!("must refuse, not invoke an unobserved policy")
                }
            }
            let home = tempfile::tempdir().unwrap();
            let p = control_plan(home.path(), 40);
            let store = Arc::new(Store::open(home.path().join("store")).unwrap());
            let (c, _) = context(&p, 20, true, false);
            let mut d = Discovery::open_research(store.clone(), p, c).unwrap();
            let e = Evaluator::new(store);
            d.advance(&SPACE, &HeuristicAdmission, &e, 1, usize::MAX)
                .unwrap();
            let calls = e.calls.lock().unwrap().len();
            assert!(d.advance(&SPACE, &Unobserved, &e, 1, usize::MAX).is_err());
            assert_eq!(e.calls.lock().unwrap().len(), calls);
        }

        #[test]
        fn capture_roots_survive_gc_and_normal_open_or_binding_drift_refuse() {
            let home = tempfile::tempdir().unwrap();
            let p = control_plan(home.path(), 40);
            let store = Arc::new(Store::open(home.path().join("store")).unwrap());
            let (c, _) = context(&p, 20, true, false);
            let mut d = Discovery::open_research(store.clone(), p.clone(), c).unwrap();
            let e = Evaluator::new(store.clone());
            run_to_end(&mut d, &e, 8);
            let roots = d.checkpoint.capture.as_ref().unwrap().roots();
            let orphan = store.publish_blob(&b"unreferenced-control"[..]).unwrap();
            let gc = store.gc(true).unwrap();
            assert_eq!(gc.blobs, vec![orphan]);
            assert!(gc.blobs.iter().all(|r| !roots.contains(r)));
            d.pause().unwrap();
            drop(d);
            assert!(Discovery::open(store.clone(), p.clone()).is_err());
            let (mut changed, _) = context(&p, 20, true, false);
            Arc::get_mut(&mut changed).unwrap().binding.tokens[0][0] = 0.5;
            assert!(Discovery::open_research(store, p, changed).is_err());
        }

        #[test]
        fn round_transaction_failure_never_installs_outbox_or_public_results() {
            let home = tempfile::tempdir().unwrap();
            let p = control_plan(home.path(), 40);
            let store = Arc::new(Store::open(home.path().join("store")).unwrap());
            let (c, _) = context(&p, 20, true, false);
            let mut d = Discovery::open_research(store.clone(), p.clone(), c.clone()).unwrap();
            let e = Evaluator::new(store.clone());
            store
                .transaction(|tx| {
                    tx.put(
                        &RecordKey::Evaluation {
                            run: p.run_id.clone(),
                            number: 0,
                        },
                        &Record::metadata(json!({"conflict-control":true}), vec![]),
                    )
                })
                .unwrap();
            assert!(d
                .advance(&SPACE, &HeuristicAdmission, &e, 1, usize::MAX)
                .is_err());
            assert!(d.poisoned);
            let cp = stored(&d);
            assert!(cp["elites"]["objective"].is_null() || cp["champion"].is_null());
            assert!(cp["capture"]["rounds"]["0"]["committed"].is_null());
            assert_eq!(cp["next_round"], 0);
            assert!(d.projection().unwrap().as_array().unwrap().is_empty());
            drop(d);
            let resumed = Discovery::open_research(store, p, c).unwrap();
            assert!(!resumed.checkpoint.capture.as_ref().unwrap().unknown());
            assert!(records(&resumed).is_empty());
        }

        #[test]
        fn corrupt_or_missing_capture_cas_never_materializes() {
            let home = tempfile::tempdir().unwrap();
            let p = control_plan(home.path(), 40);
            let run = p.run_id.clone();
            let (c, _) = context(&p, 20, true, false);
            let store_home = home.path().join("store");
            let store = Arc::new(Store::open(&store_home).unwrap());
            let mut d = Discovery::open_research(store.clone(), p.clone(), c.clone()).unwrap();
            let e = Evaluator::new(store.clone());
            run_to_end(&mut d, &e, 1);
            let reference = d.checkpoint.capture.as_ref().unwrap().rounds[&1]
                .slate
                .clone();
            let name = hex_blob(&reference);
            let path = store_home
                .join("cas/sha256")
                .join(&name[..2])
                .join(&name[2..4])
                .join(&name[4..]);
            let saved = fs::read(&path).unwrap();
            fs::write(&path, b"deliberately corrupted own control fixture").unwrap();
            let output = home.path().join("records.jsonl");
            assert!(materialize_controls(&store, &run, &output).is_err());
            assert!(!output.exists());
            assert!(Discovery::open_research(store.clone(), p.clone(), c.clone()).is_err());
            fs::write(&path, &saved).unwrap();
            let backup = home.path().join("retained-cas-control");
            fs::rename(&path, &backup).unwrap();
            assert!(materialize_controls(&store, &run, &output).is_err());
            assert!(!output.exists());
            fs::rename(backup, path).unwrap();
            assert_eq!(materialize_controls(&store, &run, &output).unwrap(), 3);
        }

        #[test]
        fn seed_endpoints_and_capture_input_bounds_are_explicit() {
            for seeds in [1, 16] {
                let home = tempfile::tempdir().unwrap();
                let p = control_plan(home.path(), 48);
                let (c, _) = context(&p, 16, true, false);
                let store = Arc::new(Store::open(home.path().join("store")).unwrap());
                let mut d = Discovery::open_research(store.clone(), p, c).unwrap();
                let e = Evaluator::new(store);
                d.advance(
                    &Space {
                        seeds,
                        duplicate: false,
                    },
                    &HeuristicAdmission,
                    &e,
                    1,
                    usize::MAX,
                )
                .unwrap();
                assert_eq!(records(&d)[0].proposals.len(), seeds);
                assert_eq!(records(&d)[0].outcomes.len(), seeds);
            }
            for case in 0..7 {
                let home = tempfile::tempdir().unwrap();
                let p = control_plan(home.path(), 40);
                let (mut c, _) = context(&p, 20, true, false);
                let b = &mut Arc::get_mut(&mut c).unwrap().binding;
                match case {
                    0 => b.config.max_attempts = 41,
                    1 => b.config.max_rounds = 0,
                    2 => b.config.max_bytes = 0,
                    3 => b.config.seed_protocol_sha256.clear(),
                    4 => b.identity.task = RecordTask::Binary,
                    5 => b.matrix_sha256 = digest(b"changed"),
                    _ => b.seed += 1,
                }
                let store = Arc::new(Store::open(home.path().join("store")).unwrap());
                assert!(Discovery::open_research(store.clone(), p.clone(), c).is_err());
                assert!(store.get(&RecordKey::Run(p.run_id)).unwrap().is_none());
            }
        }

        #[test]
        fn auxiliary_abort_or_stale_reply_stays_unknown() {
            for mode in 0..3 {
                struct AbortAux {
                    inner: Evaluator,
                    mode: usize,
                }
                impl Executor for AbortAux {
                    fn evaluate(&self, r: &Request) -> Response {
                        if r.attempt.slot == 65536 && self.mode < 2 {
                            assert_ne!(self.mode, 0, "owned auxiliary panic control");
                            let mut response = self.inner.evaluate(r);
                            response.attempt.attempt += 1;
                            return response;
                        }
                        self.inner.evaluate(r)
                    }
                    fn fitted_model(&self, _: &Request) -> Result<Vec<u8>> {
                        panic!("owned refit panic control")
                    }
                }
                let home = tempfile::tempdir().unwrap();
                let p = control_plan(home.path(), 40);
                let run = p.run_id.clone();
                let (c, _) = context(&p, 20, true, false);
                let store = Arc::new(Store::open(home.path().join("store")).unwrap());
                let mut d = Discovery::open_research(store.clone(), p.clone(), c.clone()).unwrap();
                let e = AbortAux {
                    inner: Evaluator::new(store.clone()),
                    mode,
                };
                assert!(d
                    .advance(&SPACE, &HeuristicAdmission, &e, 1, usize::MAX)
                    .is_err());
                assert!(d.poisoned);
                assert!(stored(&d)["capture"]["rounds"]["0"]["committed"].is_null());
                assert_eq!(stored(&d)["capture"]["charged"], 4);
                assert!(d.projection().unwrap().as_array().unwrap().is_empty());
                drop(d);
                let mut d = Discovery::open_research(store.clone(), p, c).unwrap();
                assert!(d.checkpoint.capture.as_ref().unwrap().tainted);
                assert!(d
                    .advance(&SPACE, &HeuristicAdmission, &e, 1, usize::MAX)
                    .is_err());
                assert!(materialize_controls(&store, &run, &home.path().join("refused")).is_err());
            }
        }

        #[test]
        fn exhausted_wall_stops_new_dispatch_without_refunding_reservations() {
            let (_scope, clock) = ClockScope::frozen();
            let home = tempfile::tempdir().unwrap();
            let mut p = control_plan(home.path(), 40);
            p.wall_seconds = 1;
            let (c, _) = context(&p, 20, true, false);
            let store = Arc::new(Store::open(home.path().join("store")).unwrap());
            let mut d = Discovery::open_research(store.clone(), p.clone(), c.clone()).unwrap();
            struct Deadline(Evaluator, Arc<std::sync::atomic::AtomicU64>);
            impl Executor for Deadline {
                fn evaluate(&self, r: &Request) -> Response {
                    self.1.fetch_add(1000, Ordering::SeqCst);
                    self.0.evaluate(r)
                }
                fn fitted_model(&self, _: &Request) -> Result<Vec<u8>> {
                    unreachable!()
                }
            }
            let e = Deadline(Evaluator::new(store.clone()), clock);
            // Work performed before advance leaves the frozen admission budget intact.
            let before = d.checkpoint.elapsed_at(wall_clock_ms().unwrap()).unwrap();
            assert_eq!(before, 0);
            assert!(d
                .advance(&SPACE, &HeuristicAdmission, &e, 1, usize::MAX)
                .is_err());
            assert!(d.poisoned);
            assert_eq!(e.0.calls.lock().unwrap().len(), 1);
            assert_eq!(stored(&d)["capture"]["charged"], 4);
            drop(d);
            let d = Discovery::open_research(store.clone(), p.clone(), c).unwrap();
            assert_eq!(d.completed(), Some(CompletionReason::BudgetTime));
            assert!(d.checkpoint.capture.as_ref().unwrap().tainted);
            assert!(materialize_controls(&store, &p.run_id, &home.path().join("refused")).is_err());
        }

        #[test]
        fn terminated_pending_suffix_retains_known_prefix_but_refuses_complete_export() {
            for timed_out in [false, true] {
                for workers in [1, 8] {
                    let (_scope, clock) = ClockScope::frozen();
                    let home = tempfile::tempdir().unwrap();
                    let mut p = control_plan(home.path(), 80);
                    if timed_out {
                        p.wall_seconds = 1;
                    }
                    let (context, _) = context(&p, 64, true, false);
                    let store_home = home.path().join("store");
                    let store = Arc::new(Store::open(&store_home).unwrap());
                    let mut d = Discovery::open_research(store.clone(), p.clone(), context.clone())
                        .unwrap();
                    struct FinishAtDeadline {
                        inner: Evaluator,
                        clock: Arc<std::sync::atomic::AtomicU64>,
                        timed_out: bool,
                    }
                    impl Executor for FinishAtDeadline {
                        fn evaluate(&self, request: &Request) -> Response {
                            self.inner.evaluate(request)
                        }
                        fn fitted_model(&self, request: &Request) -> Result<Vec<u8>> {
                            let bytes = self.inner.fitted_model(request)?;
                            if self.timed_out && request.attempt.round == 1 {
                                self.clock.fetch_add(1000, Ordering::SeqCst);
                            }
                            Ok(bytes)
                        }
                    }
                    let executor = FinishAtDeadline {
                        inner: Evaluator::new(store.clone()),
                        clock,
                        timed_out,
                    };
                    for _ in 0..4 {
                        if d.completed().is_some() {
                            break;
                        }
                        d.advance(&SPACE, &HeuristicAdmission, &executor, workers, usize::MAX)
                            .unwrap();
                    }
                    let reason = if timed_out {
                        CompletionReason::BudgetTime
                    } else {
                        CompletionReason::Converged
                    };
                    assert_eq!(d.completed(), Some(reason));
                    assert!(d.checkpoint.pending.is_empty());
                    let captured = d.checkpoint.capture.as_ref().unwrap().clone();
                    let prefix = records(&d);
                    assert_eq!(prefix.len(), if timed_out { 2 } else { 4 });
                    assert!(prefix.iter().all(|round| round.capture_complete));
                    let (number, suffix) = captured.rounds.last_key_value().unwrap();
                    assert_eq!(*number, prefix.len() as u64);
                    assert!(suffix.aborted && suffix.committed.is_none());
                    assert_eq!(suffix.attempts.len(), if timed_out { 8 } else { 16 });
                    assert!(suffix
                        .attempts
                        .iter()
                        .all(|attempt| attempt.result.is_some()));
                    assert!(captured.tainted && captured.unknown());
                    assert_eq!(captured.charged, if timed_out { 20 } else { 52 });
                    let rung0_calls = executor
                        .inner
                        .calls
                        .lock()
                        .unwrap()
                        .iter()
                        .filter(|r| r.attempt.slot < 65536)
                        .count();
                    assert_eq!(rung0_calls as u64, captured.charged);
                    assert_eq!(d.checkpoint.discarded.len(), 4);
                    assert!(d
                        .checkpoint
                        .discarded
                        .iter()
                        .all(|r| r.reason == reason && r.response.is_some()));
                    let before = store
                        .get(&RecordKey::Run(p.run_id.clone()))
                        .unwrap()
                        .unwrap();
                    assert!(captured
                        .roots()
                        .iter()
                        .all(|root| before.blobs.contains(root)));
                    let output = home.path().join("refused.jsonl");
                    let error = materialize_controls(&store, &p.run_id, &output)
                        .unwrap_err()
                        .to_string();
                    assert!(error.contains("unknown/aborted"), "{error}");
                    assert!(
                        error.contains("committed prefix is inspection-only"),
                        "{error}"
                    );
                    assert!(!output.exists());
                    assert!(!fs::read_dir(home.path()).unwrap().any(|entry| entry
                        .unwrap()
                        .file_name()
                        .to_string_lossy()
                        .starts_with(".jope-research-")));
                    assert!(store
                        .gc(true)
                        .unwrap()
                        .blobs
                        .iter()
                        .all(|root| !captured.roots().contains(root)));
                    drop(d);
                    drop(executor);
                    drop(store);
                    let store = Arc::new(Store::open(&store_home).unwrap());
                    let mut resumed =
                        Discovery::open_research(store.clone(), p.clone(), context).unwrap();
                    assert_eq!(resumed.completed(), Some(reason));
                    assert_eq!(
                        serde_json::to_value(records(&resumed)).unwrap(),
                        serde_json::to_value(prefix).unwrap()
                    );
                    assert_eq!(
                        resumed.checkpoint.capture.as_ref().unwrap().charged,
                        captured.charged
                    );
                    let no_work = Evaluator::new(store.clone());
                    assert!(resumed
                        .advance(&SPACE, &HeuristicAdmission, &no_work, workers, usize::MAX)
                        .unwrap()
                        .is_empty());
                    assert!(no_work.calls.lock().unwrap().is_empty());
                    assert!(materialize_controls(&store, &p.run_id, &output).is_err());
                    let after = store
                        .get(&RecordKey::Run(p.run_id.clone()))
                        .unwrap()
                        .unwrap();
                    assert_eq!(before.blobs, after.blobs);
                }
            }
        }

        #[test]
        #[ignore = "own child controlled by sigkill_at_capture_boundaries_preserves_committed_prefix"]
        fn capture_kill_child() {
            let home = PathBuf::from(std::env::var_os("QLT03_KILL_HOME").unwrap());
            let p = control_plan(&home, 40);
            let store = Arc::new(Store::open(home.join("store")).unwrap());
            let (c, _) = context(&p, 20, true, false);
            let mut d = Discovery::open_research(store.clone(), p.clone(), c).unwrap();
            let e = Evaluator::new(store.clone());
            run_to_end(&mut d, &e, 1);
            materialize_controls(&store, &p.run_id, &home.join("published.jsonl")).unwrap();
            panic!("requested process boundary was never reached");
        }

        #[cfg(unix)]
        #[test]
        fn sigkill_at_capture_boundaries_preserves_committed_prefix() {
            for stage in [
                "before_reservation",
                "after_reservation",
                "after_outcome",
                "before_round",
                "after_round",
                "after_file_sync",
                "after_file_publish",
            ] {
                let home = tempfile::tempdir().unwrap();
                let mut child = Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--exact",
                        "discover::r#loop::tests::durable_capture::capture_kill_child",
                        "--ignored",
                        "--nocapture",
                    ])
                    .env("QLT03_KILL_HOME", home.path())
                    .env("QLT03_CAPTURE_STOP", stage)
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .spawn()
                    .unwrap();
                let mut reader = BufReader::new(child.stdout.take().unwrap());
                let (tx, rx) = std::sync::mpsc::sync_channel(1);
                let thread = std::thread::spawn(move || {
                    let mut line = String::new();
                    while reader.read_line(&mut line).unwrap() > 0 {
                        if line.contains("QLT03_CAPTURE_BOUNDARY:") {
                            let _ = tx.send(());
                            break;
                        }
                        line.clear();
                    }
                });
                let ready = rx.recv_timeout(Duration::from_secs(20));
                let _ = child.kill();
                let status = child.wait().unwrap();
                thread.join().unwrap();
                let mut diagnostic = String::new();
                child
                    .stderr
                    .take()
                    .unwrap()
                    .read_to_string(&mut diagnostic)
                    .unwrap();
                assert!(ready.is_ok(), "{stage}: {diagnostic}");
                assert!(!status.success());
                let p = control_plan(home.path(), 40);
                let store = Arc::new(Store::open(home.path().join("store")).unwrap());
                let (c, _) = context(&p, 20, true, false);
                let mut d = Discovery::open_research(store.clone(), p.clone(), c).unwrap();
                let e = Evaluator::new(store.clone());
                let expected_records = match stage {
                    "after_round" => 1,
                    "after_file_sync" | "after_file_publish" => 3,
                    _ => 0,
                };
                assert_eq!(records(&d).len(), expected_records, "{stage}");
                if matches!(stage, "after_reservation" | "after_outcome") {
                    let state = d.checkpoint.capture.as_ref().unwrap();
                    assert!(state.tainted);
                    assert_eq!(state.charged, 4);
                    assert_eq!(
                        state.rounds[&0]
                            .attempts
                            .iter()
                            .filter(|a| a.result.is_some())
                            .count(),
                        usize::from(stage == "after_outcome")
                    );
                    assert!(d
                        .advance(&SPACE, &HeuristicAdmission, &e, 8, usize::MAX)
                        .is_err());
                    assert!(e.calls.lock().unwrap().is_empty());
                    assert!(materialize_controls(
                        &store,
                        &p.run_id,
                        &home.path().join("refused.jsonl")
                    )
                    .is_err());
                } else {
                    assert!(!d.checkpoint.capture.as_ref().unwrap().tainted);
                    run_to_end(&mut d, &e, 8);
                    assert_eq!(records(&d).len(), 3);
                    let output = home.path().join("recovered.jsonl");
                    assert_eq!(materialize_controls(&store, &p.run_id, &output).unwrap(), 3);
                    let published = home.path().join("published.jsonl");
                    assert_eq!(published.exists(), stage == "after_file_publish");
                    if published.exists() {
                        assert_eq!(fs::read(published).unwrap(), fs::read(output).unwrap());
                    }
                }
            }
        }
    }

    #[test]
    fn public_start_commits_snapshot_and_receipt_together_and_aborts_competing_creation() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/int01");
        fs::create_dir_all(&root).unwrap();
        let directory = tempfile::tempdir_in(root).unwrap();
        let plan = plan(directory.path(), 24, Effort::Fast, 4);
        let store = Arc::new(Store::open(directory.path().join("store")).unwrap());
        let log = Arc::new(EventLog::new(Arc::clone(&store)));
        let input = store
            .publish_blob(fs::read(&plan.matrix_path).unwrap().as_slice())
            .unwrap();
        let identity = crate::idempotency::RequestIdentity::new(
            "local-engine-owner",
            "same-key",
            6,
            "/v1/runs",
            None,
            &json!({"test":"actual-native-plan-creation"}),
        )
        .unwrap();
        let run = Discovery::start_command(
            Arc::clone(&store),
            plan.clone(),
            Arc::clone(&log),
            identity.clone(),
            vec![input.clone()],
            vec![],
        )
        .unwrap();
        let original = store.idempotent_receipt(&identity).unwrap().unwrap();
        assert_eq!(
            original.outcome.body(),
            &serde_json::to_value(run.snapshot()).unwrap()
        );
        assert_eq!(
            log.snapshot(&plan.run_id).unwrap().as_ref(),
            Some(run.snapshot())
        );
        assert_eq!(
            run.admitted(),
            0,
            "no worker admission before the committed creation boundary"
        );
        assert!(store
            .get(&RecordKey::Run(plan.run_id.clone()))
            .unwrap()
            .unwrap()
            .blobs
            .contains(&input));
        let mut competing = plan.clone();
        competing.run_id = "run_01JBD8Q6N4V2T5Y8K3M7P9R1S1".parse().unwrap();
        assert!(Discovery::start_command(
            Arc::clone(&store),
            competing.clone(),
            Arc::clone(&log),
            identity.clone(),
            vec![input],
            vec![]
        )
        .is_err());
        assert!(
            log.snapshot(&competing.run_id).unwrap().is_none(),
            "competing run state escaped its aborted receipt transaction"
        );
        assert_eq!(
            store
                .idempotent_receipt(&identity)
                .unwrap()
                .unwrap()
                .outcome,
            original.outcome
        );
        assert_eq!(store.run_ids_after(None, 257).unwrap(), vec![plan.run_id]);
    }
}
