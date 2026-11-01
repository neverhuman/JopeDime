//! Engine-owned, bounded CPU runs. No transport owns a second discovery engine.
use super::auth::problem;
use crate::{
    config::Config,
    dataset::{jmat, register::RegisteredDataset},
    discover::{
        audit::AuditRowId,
        effort::{BudgetOverrides, Effort as SearchEffort},
        finalize::{FinalizationRequest, Finalizer, Partition},
        r#loop::{
            Discovery, HeuristicAdmission, NativeSpace, Plan as SearchPlan, ProcessExecutor,
            SearchSpace,
        },
        splits::random_split,
    },
    engine::{EngineActivity, EngineLock},
    events::{EventLog, RunUpdate},
    governor::{Governor, Requirements},
    idempotency::{CommandOutcome, RequestIdentity},
    ranker::admission::{resolve_unqualified, GuidanceUnavailableReason},
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
type Result<T> = std::result::Result<T, Box<Problem>>;

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
    #[serde(default = "legacy_run_languages")]
    languages: Vec<Language>,
}

fn legacy_run_languages() -> Vec<Language> {
    vec![Language::Rust]
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

    fn prepare(&self, spec: &RunSpec) -> Result<Prepared> {
        if spec.task != Task::Regression {
            return Err(unavailable("This run backend requires task=regression."));
        }
        let device = jope_runtime::backend::DeviceRequest::parse(&spec.device)
            .ok_or_else(|| invalid(&format!("Unknown device {:?}.", spec.device)))?;
        // This bounded run instantiates the native CPU evaluator. Select that
        // actual backend before reading data or publishing any run effects.
        jope_runtime::backend::select(device).map_err(|error| {
            issue(
                ErrorCode::BackendUnavailable,
                format!("{}: {error:#}", ErrorCode::BackendUnavailable),
            )
        })?;
        if spec.landmarks != LandmarkMode::None {
            return Err(unavailable("This run backend requires landmarks=none."));
        }
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
        if spec.objective != ObjectiveSpec::Name("rmse".into()) {
            return Err(unavailable(
                "This run backend currently qualifies the rmse regression objective.",
            ));
        }
        let (guidance, warnings) = resolve_unqualified(
            spec.guidance,
            spec.landmarks,
            GuidanceUnavailableReason::Capability,
        )
        .map_err(|code| issue(code, "Requested guidance is unavailable."))?;
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
            "worker_ram":WORKER_RAM, "workers":1, "estimates":estimates}),
            )
            .map_err(io)?,
        );
        if spec.plan_digest.as_ref().is_some_and(|v| v != &plan_digest) {
            return Err(issue(
                ErrorCode::PlanChanged,
                "Plan digest changed; request a fresh dry-run plan.",
            ));
        }
        let public: Plan = serde_json::from_value(json!({"plan_digest":plan_digest,
            "task":{"value":"regression","inferred":false,"reason":"Explicit regression task."},
            "objective":{"name":"rmse","digest":jope_runtime::fitness::Fitness::parse("rmse").map_err(io)?.identity().map_err(io)?,"direction":"minimize","fitting":"squared_error"},
            "rows":{"total":table.rows.len().to_string(),"fit":fit_count.to_string(),"holdout":holdout_ids.len().to_string(),"test":null},
            "effort":{"name":spec.effort,"evaluations":effort.evaluations.to_string(),"wall_seconds":effort.wall_seconds.to_string(),"cv_folds":effort.rung1_folds,"rounds_in_flight":1},
            "guidance":guidance,"landmarks":{"mode":"none","sample_rows":0},
            "languages":{"requested":spec.languages,"excluded_operations":language_filter.excluded_ops(),"intersection_empty":false},
            "placement":{"dope":"none","ranker":"heuristic","evaluation":"cpu","workers":1},
            "limits":{"eval_timeout_s":timeout.div_ceil(1000),"worker_rss_bytes":WORKER_RAM.to_string()},"estimates":estimates,"warnings":warnings})).map_err(io)?;
        let search = SearchPlan {
            run_id: format!("run_{}", ulid::Ulid::new()).parse().map_err(io)?,
            dataset_id: spec.dataset.clone(),
            problem_digest: hash(&serde_json::to_vec(&json!({"spec":normalized,"csv_sha256":dataset.csv.sha256,
                "fit_sha256":hash(&fit),"objective_digest":public.objective.digest,"split":"public-run/fit-v1"})).map_err(io)?),
            bundle_digest: hash(b"cpu-native-no-learned-bundle/v1"),
            binary_version: self.binary.clone(),
            matrix_path: PathBuf::new(),
            matrix_sha256: hash(&fit),
            target: spec.target.clone(),
            features,
            objective: "rmse".into(),
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
            // Finalize both models once: default exports use FIT+HOLDOUT, while
            // explicit FIT exports retain the model used to score HOLDOUT.
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
            languages: spec.languages.clone(),
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
        let sender = match self.worker_sender(_pub06_values, false, spec.languages.clone()) {
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
        let activity = self.activity.begin_run();
        if let Err(error) = std::thread::Builder::new()
            .name("jope-run".into())
            .spawn(move || {
                let _activity = activity;
                if let Ok((mut discovery, request)) = receiver.recv() {
                    let result = (|| -> anyhow::Result<()> {
                        let executor = ProcessExecutor::new(
                            &owner.executable,
                            1,
                            &owner.home.join("worker-scratch"),
                        )?;
                        while discovery.completed().is_none() {
                            discovery.advance(
                                &run_space,
                                &HeuristicAdmission,
                                &executor,
                                1,
                                usize::MAX,
                            )?;
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
            let context = self
                .store
                .get(&continuation_key(&run))
                .map_err(io)?
                .ok_or_else(|| {
                    unavailable("Original continuation context is unavailable for this run.")
                })?;
            let context: Continuation = serde_json::from_value(context.data).map_err(io)?;
            if context.search.run_id != run
                || context.finalization.run_id != run
                || context.search.binary_version != self.binary
            {
                return Err(unavailable(
                    "Continuation requires the original binary and run identity.",
                ));
            }
            #[cfg(all(feature = "failpoints", debug_assertions))]
            let injection = None;
            #[cfg(not(all(feature = "failpoints", debug_assertions)))]
            let injection = ();
            let sender = self.worker_sender(injection, true, context.languages.clone())?;
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
        use rand_chacha::rand_core::SeedableRng;
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
}
