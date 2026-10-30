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
    store::{BlobRef, Record, RecordKey, Store, StoreError},
    worker::{self, Attempt, Request, Response, WorkerFailure, WorkerPool},
};
use anyhow::{bail, ensure, Context, Result};
use jope_api::{
    decimal::DecimalU64,
    ids::{ArtifactId, DatasetId, RunId},
    AlgorithmAcceptedBody, BestAlgorithm, BudgetTotal, BudgetUsed, CompletionReason, Effort,
    EventBody, ExperimentFinishedBody, ExperimentOutcome, ExperimentStartedBody, Nondeterminism,
    PublicEvent, PublicFailure, ReasoningBody, RunPhase, RunSnapshot, RunState,
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
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::Instant,
};

const CHECKPOINT_VERSION: u32 = 1;

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
            "version": CHECKPOINT_VERSION,
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

/// Admission can later be replaced by learned guidance without changing the
/// scheduler, worker protocol or transaction boundary.
pub trait Admission {
    fn admit(&self, proposals: Vec<String>, explore: bool, rng: &mut ChaCha12Rng) -> Vec<String>;
}

pub struct HeuristicAdmission;
impl Admission for HeuristicAdmission {
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
    fn fitted_model(&self, plan: &Plan, dsl: &str) -> Result<Vec<u8>>;
}

/// Linux parent-death containment is tied to the *creating thread*. Keep pool
/// callers alive for the entire run, including between waves and promotions;
/// short-lived scoped callers would kill idle reusable workers on thread exit.
pub struct ProcessExecutor {
    sender: Option<std::sync::mpsc::Sender<ProcessJob>>,
    threads: Vec<std::thread::JoinHandle<()>>,
}
type ProcessJob = (Request, std::sync::mpsc::SyncSender<Response>);

impl ProcessExecutor {
    pub fn new(executable: &Path, workers: usize) -> Result<Self> {
        ensure!(workers > 0 && workers <= 1024, "invalid live workers");
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
    fn fitted_model(&self, plan: &Plan, dsl: &str) -> Result<Vec<u8>> {
        let matrix = Matrix::open(fs::read(&plan.matrix_path)?)?;
        fit_artifact(&matrix, &plan.target, &plan.features, dsl)
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
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Checkpoint {
    version: u32,
    plan: Plan,
    elites: ElitePool,
    programs: BTreeMap<String, String>,
    memo: BTreeSet<String>,
    champion: Option<Champion>,
    pending: Vec<PendingRound>,
    next_round: u64,
    admitted: u64,
    stalled: u32,
    // A recovery/control transaction allocates a fresh attempt namespace before
    // dispatch. Aborted round attempts can never collide with the next session.
    epoch: u32,
    nondeterminism: Vec<String>,
    elapsed_ms: u64,
    completed: Option<CompletionReason>,
}

pub struct Discovery {
    store: Arc<Store>,
    log: Arc<EventLog>,
    snapshot: RunSnapshot,
    checkpoint: Checkpoint,
    attempt: u32,
    poisoned: bool,
}

impl Discovery {
    /// The caller exclusively owns this run and supplies a fresh worker pool on
    /// reopen. Store::open verifies committed blobs and marks interrupted runs.
    pub fn open(store: Arc<Store>, plan: Plan) -> Result<Self> {
        plan.validate()?;
        let log = Arc::new(EventLog::new(Arc::clone(&store)));
        let old = log.snapshot(&plan.run_id)?;
        let (mut checkpoint, mut snapshot) = if let Some(snapshot) = &old {
            let record = store
                .get(&RecordKey::Run(plan.run_id.clone()))?
                .context("run missing")?;
            let saved: Checkpoint = serde_json::from_value(record.data["checkpoint"].clone())?;
            ensure!(
                saved.version == CHECKPOINT_VERSION && saved.plan == plan,
                "resume plan identity mismatch"
            );
            if let Some(c) = &saved.champion {
                store.verify_blob(&c.model)?;
            }
            (saved, snapshot.clone())
        } else {
            let checkpoint = Checkpoint {
                version: CHECKPOINT_VERSION,
                plan: plan.clone(),
                elites: ElitePool::new(ScoreOrder::Minimize, ScoreOrder::Minimize),
                programs: BTreeMap::new(),
                memo: BTreeSet::new(),
                champion: None,
                pending: vec![],
                next_round: 0,
                admitted: 0,
                stalled: 0,
                epoch: 0,
                nondeterminism: vec![],
                elapsed_ms: 0,
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
                warnings: vec![],
                completion_reason: None,
            };
            (checkpoint, snapshot)
        };
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
        let blobs = checkpoint
            .champion
            .iter()
            .map(|c| c.model.clone())
            .collect();
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
                |_| Ok(()),
            )?
            .snapshot;
        Ok(Self {
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

    /// Search events only contain deterministic values. Elapsed wall usage is
    /// available in the snapshot/receipt; DSC-09 owns run.finished/finalization.
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
                value
                    .as_object_mut()
                    .context("event is not an object")?
                    .remove("time");
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
        ensure!(!self.poisoned, "commit outcome unknown: reopen the run");
        let mut snapshot = self.snapshot.clone();
        snapshot.state = RunState::Paused;
        snapshot.revision.0 += 1;
        self.poisoned = true;
        let result = self.log.commit(
            RunUpdate {
                expected_revision: Some(self.snapshot.revision.0),
                snapshot,
                checkpoint: serde_json::to_value(&self.checkpoint)?,
                blobs: self
                    .checkpoint
                    .champion
                    .iter()
                    .map(|c| c.model.clone())
                    .collect(),
            },
            vec![],
            |_| Ok(()),
        )?;
        self.snapshot = result.snapshot;
        self.poisoned = false;
        Ok(())
    }

    /// Execute a fixed planned wave, or its retained suffix after recovery.
    /// `stop_after_rounds` permits a supervisor to request a bounded pause without
    /// changing the plan or event stream. Already computed siblings are discarded.
    pub fn advance(
        &mut self,
        space: &impl SearchSpace,
        admission: &impl Admission,
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
        let started = Instant::now();
        let elapsed_before = self.checkpoint.elapsed_ms;
        let mut next = self.checkpoint.clone();
        if next.pending.is_empty() {
            plan_wave(&mut next, space, admission)?;
        }
        let mut work = vec![];
        for round in &next.pending {
            for (slot, candidate) in round.candidates.iter().enumerate() {
                work.push(self.request(round.number, slot as u32, 0, candidate)?);
            }
        }
        let outcomes = parallel_evaluate(executor, &work, live_workers);
        let mut offset = 0;
        let mut published = vec![];
        for _ in 0..stop_after_rounds.min(next.pending.len()) {
            let round = next.pending.remove(0);
            let end = offset + round.candidates.len();
            let mut events = vec![];
            let mut records = vec![];
            for (slot, candidate) in round.candidates.iter().enumerate() {
                let response = &outcomes[offset + slot];
                let id = experiment(round.number, slot, 0);
                record_result(&mut next, candidate, response, &id, &mut events)?;
                records.push(json!({"round":round.number,"slot":slot,"rung":0,"candidate":candidate,
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
                let response = checked_response(&request, executor.evaluate(&request));
                let id = experiment(round.number, 0, 1);
                add_events(&mut next, &response, &id, &mut events);
                records.push(json!({"round":round.number,"slot":0,"rung":1,"candidate":candidate,
                    "scores":response.scores,"predictions_digest":response.predictions_digest,"failure":response.failure}));
                if response.failure.is_none()
                    && next
                        .champion
                        .as_ref()
                        .is_none_or(|c| response.scores[1] < c.loss)
                {
                    // Publication precedes the metadata reference. An aborted transaction
                    // may leave an unreferenced CAS object, never a partial champion.
                    let bytes = executor.fitted_model(&next.plan, &candidate.dsl)?;
                    ensure!(!bytes.is_empty(), "empty fitted artifact");
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
                    events.push(EventBody::AlgorithmAccepted(AlgorithmAcceptedBody {
                        algorithm_id: candidate.algorithm_id.parse()?,
                        metric: next.plan.objective.clone(),
                        cv_value: response.scores[2],
                        previous_value: previous,
                        summary: if previous.is_none() {
                            "first model"
                        } else {
                            "strict rung-1 improvement"
                        }
                        .into(),
                        model_artifact: Some(artifact),
                        excerpt: candidate
                            .dsl
                            .lines()
                            .take(12)
                            .collect::<Vec<_>>()
                            .join("\n"),
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
            next.elapsed_ms = elapsed_before
                .saturating_add(started.elapsed().as_millis().min(u128::from(u64::MAX)) as u64);
            if next.pending.is_empty() {
                next.completed = if next.admitted >= next.plan.evaluations {
                    Some(CompletionReason::BudgetEvaluations)
                } else if next.stalled >= next.plan.limits().patience {
                    Some(CompletionReason::Converged)
                } else if next.elapsed_ms >= next.plan.wall_seconds.saturating_mul(1000) {
                    Some(CompletionReason::BudgetTime)
                } else {
                    None
                };
            }
            events.push(EventBody::Reasoning(ReasoningBody {
                stage: "searching".into(),
                algorithm_id: None,
                summary: format!(
                    "round {} committed; admitted {}; stalled {}",
                    round.number, next.admitted, next.stalled
                ),
                evidence: None,
            }));
            let committed = self.commit_round(next.clone(), events, records)?;
            published.extend(committed);
            offset = end;
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
            evaluations: DecimalU64(next.admitted),
            wall_seconds: format!("{:.3}", next.elapsed_ms as f64 / 1000.0),
        };
        snapshot.best = next
            .champion
            .as_ref()
            .map(|c| -> Result<_> {
                Ok(BestAlgorithm {
                    algorithm_id: c.candidate.algorithm_id.parse()?,
                    metric: next.plan.objective.clone(),
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
            checkpoint: serde_json::to_value(&next)?, blobs: next.champion.iter().map(|c| c.model.clone()).collect(),
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

fn hex_blob(blob: &BlobRef) -> String {
    blob.sha256.iter().map(|b| format!("{b:02x}")).collect()
}
fn experiment(round: u64, slot: usize, rung: u8) -> String {
    format!("round-{round}/slot-{slot}/rung-{rung}")
}
fn round_rng(seed: u64, round: u64) -> ChaCha12Rng {
    let mut hash = Sha256::new();
    hash.update(seed.to_le_bytes());
    hash.update(round.to_le_bytes());
    ChaCha12Rng::from_seed(hash.finalize().into())
}
fn evaluation_seed(seed: u64, round: u64, slot: u32, rung: u8) -> u64 {
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
    admission: &impl Admission,
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
        for source in proposals {
            let dsl = space.canonical(&source)?;
            if seen.insert(digest(dsl.as_bytes())) {
                unique.push(dsl);
            }
        }
        let admitted = if number == 0 {
            unique
        } else {
            admission.admit(unique, explore, &mut rng)
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
        state.pending.push(PendingRound { number, candidates });
        if state.admitted >= state.plan.evaluations {
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
    add_events(state, response, id, events);
    if response.failure.is_none() {
        state.elites.insert(Elite {
            algorithm_id: candidate.algorithm_id.clone(),
            surrogate: response.scores[0],
            objective: response.scores[1],
        })?;
    }
    Ok(())
}

fn add_events(state: &mut Checkpoint, response: &Response, id: &str, events: &mut Vec<EventBody>) {
    if matches!(
        response.failure,
        Some(WorkerFailure::Timeout | WorkerFailure::MemoryLimit)
    ) {
        state.nondeterminism.push(id.into());
    }
    events.push(EventBody::ExperimentStarted(ExperimentStartedBody {
        experiment_id: id.into(),
        hypothesis: "native heuristic candidate".into(),
    }));
    events.push(EventBody::ExperimentFinished(ExperimentFinishedBody {
        experiment_id: id.into(),
        outcome: if response.failure.is_none() {
            ExperimentOutcome::NotImproved
        } else {
            ExperimentOutcome::Failed
        },
        metric: state.plan.objective.clone(),
        value: response.scores.get(2).copied(),
        failure: response.failure.map(|failure| PublicFailure {
            code: jope_api::error::ErrorCode::InvalidInput,
            reason: format!("worker {failure:?}"),
        }),
    }));
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
pub fn evaluate_native<B: AsRef<[u8]>>(
    request: &Request,
    matrix: &Matrix<B>,
) -> std::result::Result<worker::Scores, WorkerFailure> {
    native_scores(request, matrix).map_err(|_| WorkerFailure::InvalidRequest)
}

fn native_scores<B: AsRef<[u8]>>(request: &Request, matrix: &Matrix<B>) -> Result<worker::Scores> {
    let spec: NativeFolds = serde_json::from_value(request.fold_spec.clone())?;
    ensure!(
        spec.version == CHECKPOINT_VERSION && spec.folds.len() >= 2,
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
    let mut targets = vec![];
    let mut predictions = vec![];
    let mut validation_rows = BTreeSet::new();
    let mut losses = vec![];
    let mut values = vec![];
    let mut surrogate = vec![];
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
        let fitted = NativeModel::fit(&spec.features, &request.dsl, &x, &y)?;
        let (vx, vy) = numeric_rows(matrix, &spec.target, &spec.features, &fold.validation)?;
        let prediction = fitted.predict(&vx)?;
        let score = objective.evaluate(&vy, &prediction)?;
        let rmse = RegressionObjective::parse("rmse")?.evaluate(&vy, &prediction)?;
        losses.push(score.loss);
        values.push(score.value);
        surrogate.push(rmse.loss);
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
        predictions_digest: digest(&bytes),
    })
}

fn numeric_rows<B: AsRef<[u8]>>(
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
struct NativeModel {
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
    fn fit(features: &[String], dsl: &str, x: &feat_math::Rows, y: &[f64]) -> Result<Self> {
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
            version: CHECKPOINT_VERSION,
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
    fn predict(&self, x: &feat_math::Rows) -> Result<Vec<f64>> {
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

fn fit_artifact<B: AsRef<[u8]>>(
    matrix: &Matrix<B>,
    target: &str,
    features: &[String],
    dsl: &str,
) -> Result<Vec<u8>> {
    let (x, y) = numeric_rows(
        matrix,
        target,
        features,
        &(0..matrix.rows()).collect::<Vec<_>>(),
    )?;
    Ok(serde_json::to_vec(&NativeModel::fit(
        features, dsl, &x, &y,
    )?)?)
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
    let pool = ProcessExecutor::new(executable, input.live_workers)?;
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
        "champion":discovery.champion(), "snapshot":discovery.snapshot(),
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
                predictions_digest: digest(request.dsl.as_bytes()),
                wall_ms: 0,
                failure: None,
            }
        }
        fn fitted_model(&self, _: &Plan, dsl: &str) -> Result<Vec<u8>> {
            Ok(dsl.as_bytes().to_vec())
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
            let expected = usize::from(failure != WorkerFailure::InvalidRequest);
            assert_eq!(d.nondeterminism().count.0, expected as u64);
            assert_eq!(
                d.nondeterminism().experiment_ids,
                if expected == 1 {
                    vec!["round-0/slot-0/rung-0".to_string()]
                } else {
                    vec![]
                }
            );
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
                        .ends_with("rung-1"))
                .count(),
            4
        );
        assert_eq!(d.champion().unwrap().candidate.algorithm_id, "alg_00000001");
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
        let deps = std::env::current_exe()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let mut libraries: Vec<_> = fs::read_dir(&deps)
            .unwrap()
            .filter_map(|p| p.ok())
            .map(|p| p.path())
            .filter(|p| {
                p.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with("libjope_service-")
                    && p.extension().is_some_and(|e| e == "rlib")
            })
            .collect();
        libraries.sort_by_key(|p| std::cmp::Reverse(fs::metadata(p).unwrap().modified().unwrap()));
        let source = directory.join("native-worker.rs");
        let binary = directory.join("native-worker");
        let pids = directory.join("worker-pids");
        fs::create_dir_all(&pids).unwrap();
        fs::write(&source,format!("fn main() {{ std::fs::write(std::path::Path::new({:?}).join(std::process::id().to_string()), b\"native\").unwrap(); jope_service::worker::serve(1, jope_service::discover::r#loop::evaluate_native).unwrap(); }}",pids)).unwrap();
        for library in libraries {
            let output = Command::new("rustc")
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
            if output.status.success() {
                return binary;
            }
        }
        panic!("current jope-service library could not link native worker; build the library first with cargo build -p jope-service --locked");
    }

    fn pool(binary: &Path, workers: usize) -> ProcessExecutor {
        ProcessExecutor::new(binary, workers).unwrap()
    }
    fn run_store(home: &Path, p: &Plan, binary: &Path, workers: usize) -> (Champion, Value) {
        let store = Arc::new(Store::open(home).unwrap());
        let mut d = Discovery::open(store, p.clone()).unwrap();
        let pool = pool(binary, workers);
        finish(&mut d, &NativeSpace, &pool, workers);
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
    #[ignore = "dedicated native process gate: first cargo build -p jope-service --locked, then run this exact test with --ignored"]
    fn real_native_workers_preserve_projection_across_capacity_pause_and_sigkill() {
        let directory = tempfile::tempdir().unwrap();
        let mut p = plan(directory.path(), 80, Effort::Medium, 25);
        p.wall_seconds = 600;
        p.evaluation_wall_ms = 60_000;
        assert_eq!(p.rounds_in_flight(), 2);
        let binary = native_worker(directory.path());
        let baseline = run_store(&directory.path().join("one"), &p, &binary, 1);
        assert_eq!(
            fs::read_dir(directory.path().join("worker-pids"))
                .unwrap()
                .count(),
            1,
            "one live native worker observed"
        );
        same(
            "eight",
            &baseline,
            &run_store(&directory.path().join("eight"), &p, &binary, 8),
        );
        assert_eq!(
            fs::read_dir(directory.path().join("worker-pids"))
                .unwrap()
                .count(),
            9,
            "eight additional live native workers observed"
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
            fn fitted_model(&self, p: &Plan, dsl: &str) -> Result<Vec<u8>> {
                self.0.fitted_model(p, dsl)
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
                    Some("worker Timeout" | "worker MemoryLimit")
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
        let native_artifact = workers.fitted_model(&p, &champion.candidate.dsl).unwrap();
        assert_eq!(digest(&native_artifact), hex_blob(&champion.model));
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
}
