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

mod admission;
mod commit;
mod evaluation;
mod executor;
mod helpers;
mod native;
mod planning;
#[cfg(feature = "research")]
mod research;
#[cfg(feature = "research")]
mod research_helpers;
mod start;
#[cfg(test)]
mod tests;

#[cfg(feature = "research")]
pub use admission::RecordedAdmissionContext;
pub use admission::{
    Admission, Candidate, HeuristicAdmission, RoundAdmission, RoundContext, RoundMeasurement,
};
use evaluation::{add_events, checked_response, discard_pending, parallel_evaluate, record_result};
pub use executor::{Executor, ProcessExecutor, SearchSpace};
pub(crate) use helpers::evaluation_seed;
use helpers::{
    accepted_summary, checkpoint_blobs, evaluation_ordinal, experiment, hex_blob,
    projected_summary, public_metric, round_rng, rung0_budget_limit, rung0_budget_used,
};
pub use native::{evaluate_native, run_m1, NativeSpace};
use planning::plan_wave;
#[cfg(feature = "research")]
use planning::plan_wave_inner;
#[cfg(feature = "research")]
pub(super) use research_helpers::{parallel_evaluate_research, research_response};

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
