//! Bounded frozen-panel comparator controller. Scores remain parent-owned.
//! Neural search supplies its frozen recipes through the same worker boundary.
use super::{
    data::{DevelopmentCut, DevelopmentData},
    methods::RustMethod,
    protocol::{self, Observation},
    score::{self, Predictions},
    worker::RustWorkerOutput,
};
use anyhow::{ensure, Context, Result};
use jope_datasets::catalog::Task;
use jope_pipeline::{
    execution::{Artifact, ExecutionPool, WorkerCommand, WorkerJob, WorkerOutcome},
    experiment::{
        EvaluationRequest, EvaluationStage, EvaluationStatus, Experiment, Panel, REQUEST_SCHEMA,
    },
    experiment_v2::{Consumption, EvaluationResult, RESULT_SCHEMA},
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    sync::atomic::AtomicBool,
};

pub const PLAN_SCHEMA: &str = "jope.panel-campaign/v2";

/// Every byte affecting a fit is either an executable/input artifact or a pinned
/// runtime inventory. Runtime inventories are verified before admission and after
/// execution; per-job inputs pin their immutable receipt as well as the adapter.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Runtime {
    pub program: Artifact,
    pub argument_prefix: Vec<String>,
    pub inputs: Vec<Artifact>,
    /// Complete installed package inventory, authenticated once before/after a run.
    pub package_files: Vec<Artifact>,
    pub environment: BTreeMap<String, String>,
    pub version: String,
    pub unavailable: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Method {
    Synthesized {
        id: String,
        recipe: jope_pipeline::synthesized_recipe::SynthesizedRecipe,
        checkpoint_sha256: Option<String>,
    },
    Rust {
        recipe: RustMethod,
    },
    External {
        id: String,
        engine: String,
        runtime: String,
    },
}
impl Method {
    pub fn id(&self) -> &str {
        match self {
            Self::Rust { recipe } => &recipe.id,
            Self::External { id, .. } | Self::Synthesized { id, .. } => id,
        }
    }
    pub(crate) fn runtime(&self) -> &str {
        match self {
            Self::Rust { .. } | Self::Synthesized { .. } => "rust",
            Self::External { runtime, .. } => runtime,
        }
    }
    pub(crate) fn identity(&self) -> Result<String> {
        match self {
            Self::Rust { recipe } => Ok(jope_pipeline::sha256_hex(recipe.dsl.as_bytes())),
            Self::External { .. } => digest(self),
            Self::Synthesized { recipe, .. } => {
                recipe.validate()?;
                Ok(recipe.executable_sha256.clone())
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CampaignPlan {
    pub schema: String,
    pub experiment: Experiment,
    pub panel: Panel,
    pub methods: BTreeMap<String, Vec<Method>>,
    pub runtimes: BTreeMap<String, Runtime>,
    /// A complete panel requires the neural comparison to be attached before
    /// execution. This controller's comparator qualification is named explicitly.
    pub scope: String,
}
impl CampaignPlan {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.schema == PLAN_SCHEMA, "unknown campaign plan");
        self.experiment.validate()?;
        self.panel.validate()?;
        ensure!(
            self.experiment.panel_sha256 == super::FROZEN_PANEL_SHA256,
            "panel identity differs from approved manifest"
        );
        // The original approved bytes are checked by read_panel at construction;
        // retain the semantic panel identity as part of the immutable plan.
        ensure!(
            self.experiment.methods_sha256 == digest(&(&self.methods, &self.runtimes))?,
            "method/runtime roster changed"
        );
        ensure!(
            (self.experiment.protocol == super::PROTOCOL
                || (self.scope == "asm-origin-proof"
                    && self.experiment.protocol == "jope.asm-origin-proof/v1"))
                && self.experiment.numerical_policy == score::NUMERICAL_POLICY,
            "campaign protocol or numerical policy changed"
        );
        ensure!(
            self.scope == "comparator-qualification"
                || ((self.scope == "asm-panel" || self.scope == "asm-origin-proof")
                    && cfg!(feature = "asm")
                    && self.runtimes.contains_key("asm-configuration")),
            "unsupported campaign scope"
        );
        let native_runtime = self.runtimes.get("rust").context("native runtime absent")?;
        ensure!(
            native_runtime.argument_prefix == ["benchmark"]
                || native_runtime.argument_prefix == ["panel"],
            "native runtime must pin its benchmark or panel command"
        );
        for task in [Task::Regression, Task::Binary, Task::Multiclass] {
            let methods = self
                .methods
                .get(task.as_str())
                .context("missing task roster")?;
            let mut ids = BTreeSet::new();
            let native = super::methods::rust_methods(task)?;
            for method in methods {
                ensure!(
                    ids.insert(method.id()) && !method.id().is_empty(),
                    "duplicate method"
                );
                let runtime = self
                    .runtimes
                    .get(method.runtime())
                    .context("runtime absent")?;
                ensure!(!runtime.version.is_empty(), "runtime version/reason absent");
                match method {
                    Method::Synthesized { .. } => anyhow::bail!(
                        "generated roster entries must come from retained train-side selection"
                    ),
                    Method::Rust { recipe } => {
                        ensure!(native.contains(recipe), "changed native roster")
                    }
                    Method::External { engine, .. } => ensure!(
                        ["xgboost", "lightgbm", "catboost", "sklearn"].contains(&engine.as_str()),
                        "unknown external engine"
                    ),
                }
            }
            ensure!(
                native.iter().all(|r| methods.iter().any(|m| matches!(m,
                Method::Rust {recipe} if recipe == r))),
                "incomplete native roster"
            );
            for engine in ["xgboost", "lightgbm", "catboost", "sklearn"] {
                ensure!(
                    methods
                        .iter()
                        .filter(|m| matches!(m,
                    Method::External {engine: e, ..} if e == engine))
                        .count()
                        == 1,
                    "each external engine requires one explicit outcome slot"
                );
            }
        }
        Ok(())
    }
    pub fn identity(&self) -> Result<String> {
        self.validate()?;
        digest(self)
    }
}

pub struct PendingFit {
    pub method: Method,
    pub request: EvaluationRequest,
    pub job: WorkerJob,
    pub input_sha256: String,
    pub expected_engine_version: String,
}

pub(crate) fn digest(value: &impl Serialize) -> Result<String> {
    Ok(jope_pipeline::sha256_hex(&serde_json::to_vec(value)?))
}

/// Create once or compare exact JSON value on resume. Never replace evidence.
pub(crate) fn immutable(path: &Path, value: &impl Serialize) -> Result<()> {
    if path.try_exists()? {
        let previous: Value = serde_json::from_slice(&fs::read(path)?)?;
        ensure!(
            previous == serde_json::to_value(value)?,
            "stale evidence: {}",
            path.display()
        );
        Ok(())
    } else {
        super::worker::write_private(path, value)
    }
}

pub(crate) fn request(
    dataset: &DevelopmentData,
    cut: &DevelopmentCut,
    method: &Method,
    experiment: &Experiment,
    stage: EvaluationStage,
) -> Result<EvaluationRequest> {
    let request = EvaluationRequest {
        schema: REQUEST_SCHEMA.into(),
        experiment_sha256: experiment.identity()?,
        dataset_sha256: dataset.dataset.source.sha256.clone(),
        split_sha256: dataset.dataset.split.sha256.clone(),
        fitting_rows_sha256: cut.train.rows_sha256.clone(),
        scoring_rows_sha256: cut.hold.rows_sha256.clone(),
        fitting_rows: cut.train.rows.len(),
        scoring_rows: cut.hold.rows.len(),
        recipe_sha256: method.identity()?,
        evaluator_sha256: jope_pipeline::sha256_hex(super::EVALUATOR.as_bytes()),
        checkpoint_sha256: match method {
            Method::Synthesized {
                checkpoint_sha256, ..
            } => checkpoint_sha256.clone(),
            _ => None,
        },
        task: dataset.dataset.task,
        metric: dataset.dataset.metric,
        classes: dataset.dataset.classes.clone(),
        fold: cut.fold,
        stage,
        seed: experiment.seed,
        numerical_policy: experiment.numerical_policy.clone(),
    };
    request.validate()?;
    Ok(request)
}

/// Holdout labels never enter the subprocess input or argv. Original nulls,
/// category strings and training labels are retained at the Python boundary.
pub fn prepare_fit(
    plan: &CampaignPlan,
    dataset: &DevelopmentData,
    cut: &DevelopmentCut,
    method: &Method,
    stage: EvaluationStage,
    journal: &Path,
) -> Result<PendingFit> {
    prepare_fit_impl(plan, dataset, cut, method, stage, journal, true)
}

pub(crate) fn prepare_fit_impl(
    plan: &CampaignPlan,
    dataset: &DevelopmentData,
    cut: &DevelopmentCut,
    method: &Method,
    stage: EvaluationStage,
    journal: &Path,
    persist: bool,
) -> Result<PendingFit> {
    let request = request(dataset, cut, method, &plan.experiment, stage)?;
    let id = request.identity()?;
    let input = journal.join("inputs").join(format!("{id}.json"));
    let input_value = match method {
        Method::Synthesized { recipe, .. } => {
            serde_json::to_value(super::synthesized::SynthesizedFitRequest {
                task: dataset.dataset.task,
                schema: dataset.schema.clone(),
                classes: dataset.dataset.classes.clone(),
                train: cut.train.table.clone(),
                targets: cut.train.targets.clone(),
                predict: cut.hold.table.clone(),
                recipe: recipe.clone(),
                scheduling_seed: request.seed,
            })?
        }
        Method::Rust { recipe } => {
            let (_, input) =
                super::evaluation_input(dataset, cut, recipe, &plan.experiment, stage)?;
            serde_json::to_value(input)?
        }
        Method::External { .. } => json!({
            "contract":super::methods::ADAPTER_CONTRACT, "task":dataset.dataset.task,
            "columns":dataset.dataset.features, "classes":dataset.dataset.classes,
            "train":{"features":cut.train.rows.iter().map(|r| &r.values).collect::<Vec<_>>(),
                "targets":cut.train.labels},
            "predict":{"features":cut.hold.rows.iter().map(|r| &r.values).collect::<Vec<_>>()}
        }),
    };
    if persist {
        immutable(&input, &input_value)?;
    } else {
        let retained: Value = serde_json::from_slice(&fs::read(&input)?)?;
        ensure!(
            retained == input_value,
            "retained fit input changed original panel rows"
        );
    }
    let input_sha256 = jope_pipeline::sha256_hex(&fs::read(&input)?);
    let runtime = &plan.runtimes[method.runtime()];
    let directory = journal.join("jobs").join(id);
    let output = directory.join("output.json");
    let mut args = runtime.argument_prefix.clone();
    let mut outputs = vec![PathBuf::from("output.json")];
    match method {
        Method::Rust { .. } | Method::Synthesized { .. } => args.extend(
            [
                if matches!(method, Method::Synthesized { .. }) {
                    "synthesized-fit-worker"
                } else {
                    "fit-worker"
                },
                "--input",
            ]
            .map(str::to_owned)
            .into_iter()
            .chain([
                input.display().to_string(),
                "--output".into(),
                output.display().to_string(),
            ]),
        ),
        Method::External { .. } => {
            args.extend([
                "--panel-v2".into(),
                input.display().to_string(),
                output.display().to_string(),
            ]);
            outputs.push("output.json.fitted.pkl".into());
        }
    }
    let mut inputs = runtime.inputs.clone();
    inputs.push(Artifact {
        path: input,
        sha256: input_sha256.clone(),
    });
    let job = WorkerJob {
        request: request.clone(),
        command: WorkerCommand {
            // A declared missing dependency has an impossible-to-satisfy private
            // sentinel; it can never accidentally run a now-present executable.
            program: if runtime.unavailable.is_some() {
                Artifact {
                    path: journal.join("unavailable").join(method.runtime()),
                    sha256: "0".repeat(64),
                }
            } else {
                runtime.program.clone()
            },
            args,
            inputs,
            environment: runtime.environment.clone(),
        },
        directory,
        outputs,
        arm: None,
    };
    job.identity()?;
    Ok(PendingFit {
        method: method.clone(),
        request,
        job,
        input_sha256,
        expected_engine_version: runtime.version.clone(),
    })
}

pub(crate) fn checked_predictions(
    fit: &PendingFit,
    outcome: &WorkerOutcome,
) -> Result<(Predictions, String)> {
    #[cfg(feature = "asm")]
    if matches!(fit.method, Method::Synthesized { .. }) {
        let journal = fit
            .job
            .directory
            .parent()
            .and_then(|p| p.parent())
            .context("fit outside journal")?;
        super::receipt_timing::completed_fit(journal, &fit.job, outcome)?;
    }
    outcome.verify_outputs()?;
    ensure!(
        outcome.job_sha256 == fit.job.identity()?
            && outcome.request_sha256 == fit.request.identity()?,
        "stale worker receipt"
    );
    let expected_paths: BTreeSet<_> = fit
        .job
        .outputs
        .iter()
        .map(|p| fit.job.directory.join(p))
        .collect();
    ensure!(
        outcome.outputs.len() == expected_paths.len()
            && outcome
                .outputs
                .iter()
                .map(|a| a.path.clone())
                .collect::<BTreeSet<_>>()
                == expected_paths,
        "worker output inventory changed"
    );
    let bytes = fs::read(fit.job.directory.join("output.json"))?;
    match &fit.method {
        Method::Synthesized { recipe, .. } => {
            let result: super::synthesized::SynthesizedWorkerOutput =
                serde_json::from_slice(&bytes)?;
            ensure!(
                result.schema == "jope.panel-synthesized-worker-output/v1"
                    && result.input_sha256 == fit.input_sha256
                    && result.executable_recipe_sha256 == recipe.executable_sha256
                    && result.physical_fit_starts == 1
                    && result.artifact.scheduling_seed == fit.request.seed
                    && result.artifact.task == fit.request.task
                    && result.artifact.classes == fit.request.classes
                    && digest(&result.artifact.recipe)? == digest(recipe)?,
                "generated fitted envelope changed"
            );
            result.artifact.validate()?;
            ensure!(
                result.artifact_sha256 == jope_pipeline::sha256_hex(&result.artifact.bytes()?),
                "generated fitted artifact checksum mismatch"
            );
            Ok((result.predictions, result.artifact_sha256))
        }
        Method::Rust { recipe } => {
            let result: RustWorkerOutput = serde_json::from_slice(&bytes)?;
            ensure!(
                result.schema == "jope.panel-rust-worker-output/v1"
                    && result.input_sha256 == fit.input_sha256,
                "worker input or schema mismatch"
            );
            result.artifact.validate()?;
            ensure!(
                &result.artifact.method == recipe
                    && result.artifact.task == fit.request.task
                    && result.artifact.classes == fit.request.classes,
                "fitted envelope changed"
            );
            ensure!(
                result.artifact_sha256 == jope_pipeline::sha256_hex(&result.artifact.bytes()?),
                "complete fitted artifact checksum mismatch"
            );
            Ok((result.predictions, result.artifact_sha256))
        }
        Method::External { .. } => {
            let result: Value = serde_json::from_slice(&bytes)?;
            ensure!(
                result["contract"] == super::methods::ADAPTER_CONTRACT
                    && result["input_sha256"] == fit.input_sha256
                    && result["engine_version"] == fit.expected_engine_version,
                "adapter contract, input or version changed"
            );
            let fitted = &result["fitted"];
            let artifact_path = fit.job.directory.join("output.json.fitted.pkl");
            let artifact = outcome
                .outputs
                .iter()
                .find(|a| a.path == artifact_path)
                .context("missing external fitted artifact")?;
            ensure!(
                fitted["path"] == "output.json.fitted.pkl"
                    && fitted["sha256"] == artifact.sha256
                    && fitted["reload_verified"] == true
                    && fitted["format"] == "jope.panel-fitted-python/v1"
                    && fitted["preprocessing_policy"] == super::methods::INPUT_POLICY
                    && fitted["bytes"].as_u64() == Some(fs::metadata(&artifact_path)?.len()),
                "external fitted artifact metadata mismatch"
            );
            Ok((
                serde_json::from_value(result["predictions"].clone())?,
                artifact.sha256.clone(),
            ))
        }
    }
}

pub fn score_fit(
    fit: &PendingFit,
    outcome: &WorkerOutcome,
    labels: &[String],
) -> Result<Observation> {
    ensure!(
        outcome.schema == "jope.worker-execution/v2"
            && outcome.job_sha256 == fit.job.identity()?
            && outcome.request_sha256 == fit.request.identity()?,
        "stale worker outcome"
    );
    let mut result = EvaluationResult {
        schema: RESULT_SCHEMA.into(),
        request_sha256: fit.request.identity()?,
        status: outcome.command_status,
        reason: outcome.reason.clone(),
        primary: None,
        secondary: BTreeMap::new(),
        scored_rows: 0,
        predictions_sha256: None,
        fitted_state_sha256: None,
        worker_receipt_sha256: digest(outcome)?,
        reused_from_request_sha256: None,
        consumption: Consumption {
            wall_seconds: outcome.consumption.wall_seconds,
            cpu_seconds: outcome.consumption.cpu_seconds,
            gpu_seconds: None,
            peak_rss_bytes: outcome.consumption.peak_rss_bytes,
            cgroup_peak_memory_bytes: outcome.consumption.cgroup_peak_memory_bytes,
            sampled_peak_rss_bytes: outcome.consumption.sampled_peak_rss_bytes,
            accounting_reason: outcome.consumption.accounting_reason.clone(),
        },
    };
    if outcome.command_status == EvaluationStatus::Completed {
        let scored = (|| -> Result<_> {
            let (predictions, artifact) = checked_predictions(fit, outcome)?;
            let score = score::score_predictions(
                fit.request.task,
                &fit.request.classes,
                labels,
                &predictions,
            )?;
            Ok((score.value, digest(&predictions)?, artifact))
        })();
        match scored {
            Ok((score, predictions, artifact)) => {
                result.primary = Some(score);
                result.scored_rows = labels.len();
                result.predictions_sha256 = Some(predictions);
                result.fitted_state_sha256 = Some(artifact);
            }
            Err(error) => {
                result.status = EvaluationStatus::Failed;
                result.reason = Some(format!("controller rejected output: {error:#}"));
            }
        }
    }
    result.validate(&fit.request)?;
    Ok(Observation {
        method: fit.method.id().into(),
        request: fit.request.clone(),
        result,
    })
}

/// Fold zero is the same physical fit as screening. Reuse its authenticated
/// predictor and observations, while retaining the original resource receipt.
pub(crate) fn reuse_fold_zero(
    screen: &Observation,
    confirm: EvaluationRequest,
) -> Result<Observation> {
    let mut original = confirm.clone();
    original.stage = EvaluationStage::Screen;
    ensure!(
        original.identity()? == screen.request.identity()?,
        "reuse changes the original cut"
    );
    let mut result = screen.result.clone();
    result.reused_from_request_sha256 = Some(screen.request.identity()?);
    result.request_sha256 = confirm.identity()?;
    result.consumption = Consumption {
        wall_seconds: Some(0.),
        cpu_seconds: Some(0.),
        accounting_reason: "reused screen predictor; physical cost retained on original receipt"
            .into(),
        ..Default::default()
    };
    result.validate(&confirm)?;
    Ok(Observation {
        method: screen.method.clone(),
        request: confirm,
        result,
    })
}

pub(crate) fn score_batch(
    fits: &[PendingFit],
    outcomes: Vec<WorkerOutcome>,
    data: &[DevelopmentData],
    journal: &Path,
) -> Result<Vec<Observation>> {
    let mut by_request = BTreeMap::new();
    for outcome in outcomes {
        ensure!(
            by_request
                .insert(outcome.request_sha256.clone(), outcome)
                .is_none(),
            "duplicate worker result"
        );
    }
    ensure!(
        by_request.len() == fits.len(),
        "incomplete worker inventory"
    );
    let mut observations = Vec::new();
    for fit in fits {
        let dataset = data
            .iter()
            .find(|d| d.dataset.source.sha256 == fit.request.dataset_sha256)
            .context("unknown worker dataset")?;
        let cut = dataset.cut(fit.request.fold)?;
        let outcome = by_request
            .get(&fit.request.identity()?)
            .context("worker outcome absent")?;
        let observation = score_fit(fit, outcome, &cut.hold.labels)?;
        immutable(
            &journal
                .join("observations")
                .join(format!("{}.json", fit.request.identity()?)),
            &observation,
        )?;
        observations.push(observation);
    }
    Ok(observations)
}

/// One pool owns both stages. Re-entering this function authenticates original
/// inputs and receipts and uses the pool's existing absolute deadline.
pub fn run(
    plan: &CampaignPlan,
    manifest: &Path,
    datasets: &Path,
    journal: &Path,
    pool: &mut ExecutionPool,
    cancel: &AtomicBool,
) -> Result<Value> {
    #[cfg(feature = "asm")]
    if plan.scope == super::origin_proof::SCOPE {
        return super::origin_proof::run(plan, manifest, datasets, journal, pool, cancel);
    }
    let controller_start = std::time::Instant::now();
    let controller_cpu_start = cpu_seconds()?;
    plan.validate()?;
    pool.reserve()?;
    // A run includes controller validation and preparation in its original budget.
    // The separate plan/preparation commands remain clock-free.
    pool.admit()?;
    for runtime in plan.runtimes.values().filter(|r| r.unavailable.is_none()) {
        runtime.program.verify()?;
        for input in runtime.inputs.iter().chain(&runtime.package_files) {
            input.verify()?;
        }
    }
    let (approved, hash) = super::cli::read_panel(manifest)?;
    ensure!(
        hash == plan.experiment.panel_sha256 && digest(&approved)? == digest(&plan.panel)?,
        "plan changed approved panel contents"
    );
    immutable(&journal.join("campaign.json"), plan)?;
    let data: Vec<_> = plan
        .panel
        .datasets
        .iter()
        .map(|d| super::data::load_development(datasets, d))
        .collect::<Result<_>>()?;
    let mut fits = Vec::new();
    for dataset in &data {
        let cut = dataset.cut(0)?;
        for method in &plan.methods[dataset.dataset.task.as_str()] {
            fits.push(prepare_fit(
                plan,
                dataset,
                &cut,
                method,
                EvaluationStage::Screen,
                journal,
            )?);
        }
    }
    immutable(
        &journal.join("screen-jobs.json"),
        &fits.iter().map(|f| &f.job).collect::<Vec<_>>(),
    )?;
    let outcomes = pool.run(fits.iter().map(|f| f.job.clone()).collect(), cancel)?;
    // Planned unavailable engines are immutable roster entries, not fit attempts.
    // Other never-started outcomes remain a resumable inventory.
    for outcome in &outcomes {
        let fit = fits
            .iter()
            .find(|fit| {
                fit.request
                    .identity()
                    .is_ok_and(|id| id == outcome.request_sha256)
            })
            .context("unknown dependency outcome")?;
        if plan.runtimes[fit.method.runtime()].unavailable.is_some() {
            immutable(
                &journal
                    .join("planned-dependencies")
                    .join(format!("{}.json", outcome.job_sha256)),
                outcome,
            )?;
        }
    }
    let has_pending = outcomes.iter().any(|outcome| {
        !journal
            .join("finished")
            .join(format!("{}.json", outcome.job_sha256))
            .exists()
            && !journal
                .join("planned-dependencies")
                .join(format!("{}.json", outcome.job_sha256))
                .exists()
    });
    // A missing supervisor is a dependency inventory, not a terminal scientific
    // attempt. Do not freeze empty winners or poison a future admitted resume.
    if has_pending
        || pool.window().is_none()
        || outcomes.iter().any(|outcome| {
            outcome
                .reason
                .as_ref()
                .is_some_and(|reason| reason.contains("execution admission unavailable"))
        })
    {
        return Ok(json!({"schema":"jope.panel-admission/v2", "ok":false,
            "experiment_sha256":plan.experiment.identity()?,"window":pool.window(),
            "outcomes":outcomes,"fits_started_by_this_call":null,
            "datasets":plan.panel.datasets.iter().map(|d|json!({"dataset":d.id,"task":d.task,"asm":"unknown_no_admitted_arm_outcome"})).collect::<Vec<_>>(),
            "accounting_note":"Partial inventory can include completed and interrupted attempts; no new-fit count is inferred from gate release.",
            "reason":"bounded execution unavailable; no selection or scientific result was frozen"}));
    }
    #[allow(unused_mut)]
    let mut screen = score_batch(&fits, outcomes, &data, journal)?;
    #[allow(unused_mut)]
    let mut roster: BTreeMap<String, Vec<Method>> = data
        .iter()
        .map(|d| {
            (
                d.dataset.id.clone(),
                plan.methods[d.dataset.task.as_str()].clone(),
            )
        })
        .collect();
    #[allow(unused_mut)]
    let mut asm_inventory = Value::Null;
    #[cfg(feature = "asm")]
    if plan.scope == "asm-panel" {
        let attached = super::asm_campaign::attach(plan, &data, journal, pool, cancel, &screen)?;
        asm_inventory = attached.inventory;
        if !attached.ready {
            return Ok(
                json!({"schema":"jope.asm-partial/v1","datasets":asm_inventory,"reason":"paired arms contain resumable inventory","window":pool.window(),"screen":screen}),
            );
        }
        for (dataset, methods) in attached.methods {
            roster
                .get_mut(&dataset)
                .context("unknown generated dataset")?
                .extend(methods);
        }
        screen.extend(attached.screen);
    }
    let mut selections = BTreeMap::new();
    let mut confirms = Vec::new();
    let mut reused = Vec::new();
    for dataset in &data {
        let dataset_screen: Vec<_> = screen
            .iter()
            .filter(|o| o.request.dataset_sha256 == dataset.dataset.source.sha256)
            .cloned()
            .collect();
        let expected: Vec<_> = dataset_screen
            .iter()
            .map(|o| (o.method.clone(), o.request.clone()))
            .collect();
        let asm_method = roster[&dataset.dataset.id]
            .iter()
            .find(|m| m.id() == "asm.neural")
            .map(Method::id);
        let selection = protocol::freeze_screen(&expected, &dataset_screen, asm_method)?;
        immutable(
            &journal
                .join("frozen")
                .join(format!("{}.json", dataset.dataset.id)),
            &selection,
        )?;
        for recipe in &selection.recipes {
            let method = roster[&dataset.dataset.id]
                .iter()
                .find(|m| m.id() == recipe.method)
                .context("frozen recipe absent")?;
            for fold in 0..5 {
                let cut = dataset.cut(fold)?;
                let fit = prepare_fit(
                    plan,
                    dataset,
                    &cut,
                    method,
                    EvaluationStage::Confirm,
                    journal,
                )?;
                if fold == 0 {
                    let screen_fit = dataset_screen
                        .iter()
                        .find(|o| o.method == recipe.method)
                        .context("screen absent")?;
                    let reuse = reuse_fold_zero(screen_fit, fit.request)?;
                    immutable(
                        &journal
                            .join("observations")
                            .join(format!("{}.json", reuse.request.identity()?)),
                        &reuse,
                    )?;
                    reused.push(reuse);
                } else {
                    confirms.push(fit);
                }
            }
        }
        selections.insert(
            dataset.dataset.id.clone(),
            (selection, expected, dataset_screen),
        );
    }
    immutable(
        &journal.join("confirm-jobs.json"),
        &confirms.iter().map(|f| &f.job).collect::<Vec<_>>(),
    )?;
    let outcomes = pool.run(confirms.iter().map(|f| f.job.clone()).collect(), cancel)?;
    if outcomes.iter().any(|outcome| {
        !journal
            .join("finished")
            .join(format!("{}.json", outcome.job_sha256))
            .exists()
    }) {
        return Ok(json!({"schema":"jope.panel-partial/v2","ok":false,
            "reason":"confirmation has resumable non-attempts","outcomes":outcomes,
            "window":pool.window(),"experiment_sha256":plan.experiment.identity()?}));
    }
    let mut confirmation = score_batch(&confirms, outcomes, &data, journal)?;
    confirmation.extend(reused);
    let mut datasets_report = Vec::new();
    for dataset in &data {
        let (frozen, expected_screen, screen_data) = &selections[&dataset.dataset.id];
        let observations: Vec<_> = confirmation
            .iter()
            .filter(|o| o.request.dataset_sha256 == dataset.dataset.source.sha256)
            .cloned()
            .collect();
        let expected: Vec<_> = observations
            .iter()
            .map(|o| (o.method.clone(), o.request.clone()))
            .collect();
        let confirmed = protocol::confirm(
            frozen,
            expected_screen,
            screen_data,
            roster[&dataset.dataset.id]
                .iter()
                .find(|m| m.id() == "asm.neural")
                .map(Method::id),
            &expected,
            &observations,
        )?;
        #[cfg(feature = "asm")]
        let fitted_inference = super::fitted_export::report(journal, &confirmed, &observations)?;
        #[cfg(not(feature = "asm"))]
        let fitted_inference = json!({"status":"unavailable_build_requires_asm_feature"});
        datasets_report.push(
            json!({"dataset":dataset.dataset.id,"task":dataset.dataset.task,
            "screen":screen_data,"frozen":frozen,"confirmation":observations,"recipes":confirmed,
            "fitted_inference":fitted_inference,
            "selection_exposure":super::selection_exposure::report(dataset,frozen,&roster[&dataset.dataset.id])?,
            "asm":asm_inventory.get(&dataset.dataset.id).cloned().unwrap_or_else(||json!("not_run_comparator_qualification_only"))}),
        );
    }
    let all: Vec<_> = screen.iter().chain(&confirmation).collect();
    for runtime in plan.runtimes.values().filter(|r| r.unavailable.is_none()) {
        for input in runtime.inputs.iter().chain(&runtime.package_files) {
            input.verify()?;
        }
    }
    let mut accounting = cost(&all);
    #[cfg(feature = "asm")]
    if plan.scope == "asm-panel" {
        accounting = super::asm_campaign::accounting(plan, journal)?;
    }
    accounting["controller_cpu_seconds_this_invocation"] =
        json!(cpu_seconds()? - controller_cpu_start);
    accounting["controller_wall_seconds_this_invocation"] =
        json!(controller_start.elapsed().as_secs_f64());
    let report = json!({"schema":"jope.panel-report/v2", "scope":plan.scope,
        "plan_sha256":plan.identity()?, "experiment_sha256":plan.experiment.identity()?,
        "window":pool.window(), "datasets":datasets_report, "cost":accounting,
        "sealed_inputs_opened":false,"optimization_visible_labels":"development",
        "confirmation_role":"adaptive_development","five_fold_summary_is_unbiased_generalization_estimate":false,
        "complete_asm_demonstration":false});
    let report_bytes = serde_json::to_vec(&report)?;
    let report_hash = jope_pipeline::sha256_hex(&report_bytes);
    immutable(
        &journal.join("reports").join(format!("{report_hash}.json")),
        &report,
    )?;
    Ok(report)
}

/// Unique physical receipts count once even if fold-zero evidence is reused.
pub fn cost(observations: &[&Observation]) -> Value {
    let mut receipts = BTreeSet::new();
    let mut predictors = BTreeSet::new();
    let mut recipes = BTreeSet::new();
    let mut cpu = 0.;
    let mut wall = 0.;
    let mut unknown_cpu = 0usize;
    let mut unknown_wall = 0usize;
    let mut completed = 0usize;
    for observation in observations {
        let result = &observation.result;
        if result.reused_from_request_sha256.is_some()
            || !receipts.insert(&result.worker_receipt_sha256)
        {
            continue;
        }
        if let Some(value) = result.consumption.cpu_seconds {
            cpu += value;
        } else {
            unknown_cpu += 1;
        }
        if let Some(value) = result.consumption.wall_seconds {
            wall += value;
        } else {
            unknown_wall += 1;
        }
        if result.status == EvaluationStatus::Completed {
            completed += 1;
            predictors.insert(&result.fitted_state_sha256);
            recipes.insert(&observation.request.recipe_sha256);
        }
    }
    json!({"worker_command_receipts":receipts.len(),"completed_fits":completed,
        "unique_fitted_predictors":predictors.len(),"unique_reusable_recipes":recipes.len(),
        "known_worker_cpu_seconds":cpu,"unknown_cpu_attempts":unknown_cpu,
        "known_sum_worker_wall_seconds":wall,"unknown_wall_attempts":unknown_wall,
        "cpu_seconds_per_unique_predictor_known":(!predictors.is_empty()).then(||cpu/predictors.len() as f64),
        "cpu_seconds_per_unique_recipe_known":(!recipes.is_empty()).then(||cpu/recipes.len() as f64),
        "controller_cpu_seconds":null,"gpu_seconds":null,
        "predictor_identity_policy":"complete retained fitted-artifact SHA256; behavioral equivalence is not inferred",
        "recipe_identity_policy":"canonical native DSL SHA256 or pinned external method configuration SHA256",
        "accounting_note":"Worker totals exclude controller work; unknown costs remain unknown. No cloud price is assumed."})
}

pub(crate) fn cpu_seconds() -> Result<f64> {
    let time = nix::time::clock_gettime(nix::time::ClockId::CLOCK_PROCESS_CPUTIME_ID)?;
    Ok(time.tv_sec() as f64 + time.tv_nsec() as f64 / 1e9)
}

/// Read-only recovery/reporting. Completed worker outputs are authenticated and
/// rescored with original parent-side labels. Missing receipts stay missing.
/// No process is launched, input generated, selection changed or clock started.
pub fn report(
    plan: &CampaignPlan,
    manifest: &Path,
    datasets: &Path,
    journal: &Path,
) -> Result<Value> {
    #[cfg(feature = "asm")]
    if plan.scope == super::origin_proof::SCOPE {
        return super::origin_proof::report(plan, manifest, datasets, journal);
    }
    plan.validate()?;
    let (approved, hash) = super::cli::read_panel(manifest)?;
    ensure!(
        hash == plan.experiment.panel_sha256 && digest(&approved)? == digest(&plan.panel)?,
        "report panel differs from approved input"
    );
    let retained: CampaignPlan = serde_json::from_slice(&fs::read(journal.join("campaign.json"))?)?;
    ensure!(
        retained.identity()? == plan.identity()?,
        "journal belongs to another plan"
    );
    let data = plan
        .panel
        .datasets
        .iter()
        .map(|d| super::data::load_development(datasets, d))
        .collect::<Result<Vec<_>>>()?;
    #[allow(unused_mut)]
    let mut roster: BTreeMap<String, Vec<Method>> = data
        .iter()
        .map(|d| {
            (
                d.dataset.id.clone(),
                plan.methods[d.dataset.task.as_str()].clone(),
            )
        })
        .collect();
    #[allow(unused_mut)]
    let mut asm_inventory = Value::Null;
    #[allow(unused_mut)]
    let mut aliases: BTreeMap<String, String> = BTreeMap::new();
    #[cfg(feature = "asm")]
    if plan.scope == "asm-panel" {
        let recovered = super::asm_campaign::recover(plan, &data, journal)?;
        asm_inventory = recovered.inventory;
        aliases = recovered.aliases;
        for (dataset, methods) in recovered.methods {
            roster
                .get_mut(&dataset)
                .context("unknown generated dataset")?
                .extend(methods);
        }
    }
    let mut jobs = BTreeMap::new();
    for stage in [
        "screen-jobs.json",
        "confirm-jobs.json",
        "asm-screen-jobs.json",
    ] {
        let path = journal.join(stage);
        if !path.try_exists()? {
            continue;
        }
        for job in serde_json::from_slice::<Vec<WorkerJob>>(&fs::read(path)?)? {
            ensure!(
                jobs.insert(job.request.identity()?, job).is_none(),
                "duplicate retained job"
            );
        }
    }
    let mut observations = Vec::new();
    let mut report = Vec::new();
    for dataset in &data {
        let mut screen = Vec::new();
        let mut confirmations = Vec::new();
        let mut inventory = Vec::new();
        for method in &roster[&dataset.dataset.id] {
            for (stage, folds) in [
                (EvaluationStage::Screen, 0..1),
                (EvaluationStage::Confirm, 0..5),
            ] {
                for fold in folds {
                    let cut = dataset.cut(fold)?;
                    let request = request(dataset, &cut, method, &plan.experiment, stage)?;
                    let request_id = request.identity()?;
                    let Some(job) = jobs.get(&request_id) else {
                        if stage == EvaluationStage::Screen {
                            inventory.push(json!({"method":method.id(),"stage":stage,"fold":fold,
                                "status":"not_started","reason":"no scheduled job"}));
                        }
                        continue;
                    };
                    ensure!(
                        job.request.identity()? == request_id
                            && job.directory == journal.join("jobs").join(&request_id),
                        "stale retained job"
                    );
                    let input_path = journal.join("inputs").join(format!("{request_id}.json"));
                    let input = job
                        .command
                        .inputs
                        .iter()
                        .find(|a| a.path == input_path)
                        .context("retained job lacks raw fit input")?;
                    input.verify()?;
                    let outcome_path = if plan.runtimes[method.runtime()].unavailable.is_some() {
                        journal
                            .join("planned-dependencies")
                            .join(format!("{}.json", job.identity()?))
                    } else {
                        journal
                            .join("finished")
                            .join(format!("{}.json", job.identity()?))
                    };
                    if !outcome_path.try_exists()? {
                        inventory.push(json!({"method":method.id(),"stage":stage,"fold":fold,
                            "status":"unknown","reason":"scheduled job has no terminal receipt"}));
                        continue;
                    }
                    let outcome: WorkerOutcome = serde_json::from_slice(&fs::read(outcome_path)?)?;
                    let execution_method = if stage == EvaluationStage::Screen {
                        aliases
                            .get(&format!("{}/{}", dataset.dataset.id, method.id()))
                            .and_then(|id| {
                                roster[&dataset.dataset.id].iter().find(|m| m.id() == id)
                            })
                            .unwrap_or(method)
                    } else {
                        method
                    };
                    let fit = prepare_fit_impl(
                        plan,
                        dataset,
                        &cut,
                        execution_method,
                        stage,
                        journal,
                        false,
                    )?;
                    ensure!(
                        fit.job.identity()? == job.identity()?,
                        "retained command changed the pinned runtime or input closure"
                    );
                    let mut observation = score_fit(&fit, &outcome, &cut.hold.labels)?;
                    observation.method = method.id().to_owned();
                    inventory.push(json!({"method":method.id(),"stage":stage,"fold":fold,
                        "status":observation.result.status,"reason":observation.result.reason}));
                    if stage == EvaluationStage::Screen {
                        screen.push(observation.clone());
                    } else {
                        confirmations.push(observation.clone());
                    }
                    observations.push(observation);
                }
            }
        }
        let expected = roster[&dataset.dataset.id]
            .iter()
            .map(|method| {
                Ok((
                    method.id().to_owned(),
                    request(
                        dataset,
                        &dataset.cut(0)?,
                        method,
                        &plan.experiment,
                        EvaluationStage::Screen,
                    )?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let frozen_path = journal
            .join("frozen")
            .join(format!("{}.json", dataset.dataset.id));
        let mut qualified = Value::Null;
        let mut selection_exposure = Value::Null;
        if frozen_path.try_exists()? {
            let frozen: protocol::FrozenSelection =
                serde_json::from_slice(&fs::read(frozen_path)?)?;
            ensure!(
                frozen
                    == protocol::freeze_screen(
                        &expected,
                        &screen,
                        roster[&dataset.dataset.id]
                            .iter()
                            .find(|m| m.id() == "asm.neural")
                            .map(Method::id)
                    )?,
                "stale frozen selection"
            );
            selection_exposure =
                super::selection_exposure::report(dataset, &frozen, &roster[&dataset.dataset.id])?;
            let mut expected_confirm = Vec::new();
            for recipe in &frozen.recipes {
                let method = roster[&dataset.dataset.id]
                    .iter()
                    .find(|m| m.id() == recipe.method)
                    .context("unknown frozen method")?;
                for fold in 0..5 {
                    let req = request(
                        dataset,
                        &dataset.cut(fold)?,
                        method,
                        &plan.experiment,
                        EvaluationStage::Confirm,
                    )?;
                    if fold == 0 {
                        let reuse = reuse_fold_zero(
                            screen
                                .iter()
                                .find(|o| o.method == recipe.method)
                                .context("screen result absent")?,
                            req.clone(),
                        )?;
                        confirmations.push(reuse.clone());
                        observations.push(reuse);
                    }
                    expected_confirm.push((method.id().to_owned(), req));
                }
            }
            if confirmations.len() == expected_confirm.len() {
                qualified = serde_json::to_value(protocol::confirm(
                    &frozen,
                    &expected,
                    &screen,
                    roster[&dataset.dataset.id]
                        .iter()
                        .find(|m| m.id() == "asm.neural")
                        .map(Method::id),
                    &expected_confirm,
                    &confirmations,
                )?)?;
            }
        }
        #[cfg(feature = "asm")]
        let fitted_inference = if qualified.is_null() {
            json!({"winner":null,"status":"unattempted_confirmation_incomplete","ranking_changed":false})
        } else {
            super::fitted_export::report(
                journal,
                &serde_json::from_value::<Vec<protocol::ConfirmedRecipe>>(qualified.clone())?,
                &confirmations,
            )?
        };
        #[cfg(not(feature = "asm"))]
        let fitted_inference = json!({"status":"unavailable_build_requires_asm_feature"});
        report.push(
            json!({"dataset":dataset.dataset.id,"task":dataset.dataset.task,
            "inventory":inventory,"screen":screen,"confirmation":confirmations,"recipes":qualified,
            "fitted_inference":fitted_inference,
            "selection_exposure":selection_exposure,
            "asm":asm_inventory.get(&dataset.dataset.id)}),
        );
    }
    let window: Option<jope_pipeline::experiment::ExecutionWindow> =
        if journal.join("window.json").try_exists()? {
            let window: jope_pipeline::experiment::ExecutionWindow =
                serde_json::from_slice(&fs::read(journal.join("window.json"))?)?;
            window.validate(&plan.experiment)?;
            Some(window)
        } else {
            None
        };
    #[allow(unused_mut)]
    let mut accounting = cost(&observations.iter().collect::<Vec<_>>());
    #[cfg(feature = "asm")]
    if plan.scope == "asm-panel" {
        accounting = super::asm_campaign::accounting(plan, journal)?;
    }
    Ok(
        json!({"schema":"jope.panel-recovery-report/v2","scope":plan.scope,
        "plan_sha256":plan.identity()?,"experiment_sha256":plan.experiment.identity()?,
        "window":window,"datasets":report,"cost":accounting,
        "sealed_inputs_opened":false,"fits_started":0,"clock_started":false,
        "optimization_visible_labels":"development","confirmation_role":"adaptive_development",
        "five_fold_summary_is_unbiased_generalization_estimate":false,"test_headline_eligible":false,
        "complete_asm_demonstration":false}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use jope_pipeline::{execution::ObservedConsumption, experiment::Metric};

    // These bytes are an explicit controller-validation fixture, not an estimator
    // or evidence of a successful external fit. No pickle is loaded in the parent.
    fn external_fixture(root: &Path, predictions: Predictions) -> (PendingFit, WorkerOutcome) {
        let method = Method::External {
            id: "fixture-external".into(),
            engine: "sklearn".into(),
            runtime: "sklearn".into(),
        };
        let request = EvaluationRequest {
            schema: REQUEST_SCHEMA.into(),
            experiment_sha256: "a".repeat(64),
            dataset_sha256: "b".repeat(64),
            split_sha256: "c".repeat(64),
            fitting_rows_sha256: "d".repeat(64),
            scoring_rows_sha256: "e".repeat(64),
            fitting_rows: 10,
            scoring_rows: 2,
            recipe_sha256: method.identity().unwrap(),
            evaluator_sha256: "f".repeat(64),
            checkpoint_sha256: None,
            task: Task::Regression,
            metric: Metric::Rmse,
            classes: vec![],
            fold: 0,
            stage: EvaluationStage::Screen,
            seed: 42,
            numerical_policy: score::NUMERICAL_POLICY.into(),
        };
        let directory = fs::canonicalize(root).unwrap();
        let artifact_path = directory.join("output.json.fitted.pkl");
        let artifact_bytes = b"explicit synthetic fitted-state fixture";
        fs::write(&artifact_path, artifact_bytes).unwrap();
        let artifact_hash = jope_pipeline::sha256_hex(artifact_bytes);
        let input_hash = "1".repeat(64);
        let output = json!({"contract":super::super::methods::ADAPTER_CONTRACT,
            "input_sha256":input_hash,"engine_version":"fixture-1","predictions":predictions,
            "fitted":{"path":"output.json.fitted.pkl","sha256":artifact_hash,
                "reload_verified":true,"format":"jope.panel-fitted-python/v1",
                "preprocessing_policy":super::super::methods::INPUT_POLICY,
                "bytes":artifact_bytes.len()}});
        let output_path = directory.join("output.json");
        let output_bytes = serde_json::to_vec(&output).unwrap();
        fs::write(&output_path, &output_bytes).unwrap();
        let job = WorkerJob {
            request: request.clone(),
            command: WorkerCommand {
                program: Artifact {
                    path: directory.join("unused-fixture-command"),
                    sha256: "2".repeat(64),
                },
                args: vec![],
                inputs: vec![],
                environment: BTreeMap::new(),
            },
            directory,
            outputs: vec!["output.json".into(), "output.json.fitted.pkl".into()],
            arm: None,
        };
        let outcome = WorkerOutcome {
            schema: "jope.worker-execution/v2".into(),
            job_sha256: job.identity().unwrap(),
            request_sha256: request.identity().unwrap(),
            command_status: EvaluationStatus::Completed,
            reason: None,
            exit_code: Some(0),
            signal: None,
            outputs: vec![
                Artifact {
                    path: output_path,
                    sha256: jope_pipeline::sha256_hex(&output_bytes),
                },
                Artifact {
                    path: artifact_path,
                    sha256: artifact_hash,
                },
            ],
            stdout: None,
            stderr: None,
            consumption: ObservedConsumption {
                wall_seconds: Some(2.),
                cpu_seconds: Some(1.5),
                accounting_reason: "synthetic fixture measurement".into(),
                ..Default::default()
            },
        };
        (
            PendingFit {
                method,
                request,
                job,
                input_sha256: input_hash,
                expected_engine_version: "fixture-1".into(),
            },
            outcome,
        )
    }

    #[test]
    fn parent_scores_original_labels_and_refuses_changed_fitted_bytes() {
        let root = tempfile::tempdir().unwrap();
        let (fit, outcome) = external_fixture(root.path(), Predictions::Values(vec![2., 4.]));
        let labels = vec!["1".into(), "3".into()];
        let accepted = score_fit(&fit, &outcome, &labels).unwrap();
        assert_eq!(accepted.result.status, EvaluationStatus::Completed);
        assert_eq!(accepted.result.primary, Some(1.));
        fs::write(fit.job.directory.join("output.json.fitted.pkl"), b"changed").unwrap();
        let rejected = score_fit(&fit, &outcome, &labels).unwrap();
        assert_eq!(rejected.result.status, EvaluationStatus::Failed);
        assert!(rejected.result.primary.is_none());
        assert!(rejected.result.reason.unwrap().contains("artifact changed"));
    }

    #[test]
    fn successful_command_with_bad_prediction_shape_has_no_score() {
        let root = tempfile::tempdir().unwrap();
        let (fit, outcome) = external_fixture(root.path(), Predictions::Values(vec![2.]));
        let result = score_fit(&fit, &outcome, &["1".into(), "3".into()])
            .unwrap()
            .result;
        assert_eq!(result.status, EvaluationStatus::Failed);
        assert!(result.primary.is_none());
        assert_eq!(result.scored_rows, 0);
        assert_eq!(result.consumption.cpu_seconds, Some(1.5));
    }

    #[test]
    fn stale_worker_request_and_runtime_version_cannot_publish_scores() {
        let root = tempfile::tempdir().unwrap();
        let (mut fit, mut outcome) =
            external_fixture(root.path(), Predictions::Values(vec![2., 4.]));
        let labels = vec!["1".into(), "3".into()];
        outcome.request_sha256 = "9".repeat(64);
        assert!(score_fit(&fit, &outcome, &labels).is_err());
        outcome.request_sha256 = fit.request.identity().unwrap();
        fit.expected_engine_version = "different".into();
        let result = score_fit(&fit, &outcome, &labels).unwrap().result;
        assert_eq!(result.status, EvaluationStatus::Failed);
        assert!(result.primary.is_none());
    }

    #[test]
    fn reuse_keeps_same_cut_and_counts_physical_cost_once() {
        let root = tempfile::tempdir().unwrap();
        let (fit, outcome) = external_fixture(root.path(), Predictions::Values(vec![2., 4.]));
        let screen = score_fit(&fit, &outcome, &["1".into(), "3".into()]).unwrap();
        let mut confirm = screen.request.clone();
        confirm.stage = EvaluationStage::Confirm;
        let reused = reuse_fold_zero(&screen, confirm.clone()).unwrap();
        let accounting = cost(&[&screen, &screen, &reused]);
        assert_eq!(accounting["completed_fits"], 1);
        assert_eq!(accounting["unique_fitted_predictors"], 1);
        assert_eq!(accounting["known_worker_cpu_seconds"], 1.5);
        confirm.scoring_rows_sha256 = "3".repeat(64);
        assert!(reuse_fold_zero(&screen, confirm).is_err());
    }

    #[test]
    fn interrupted_measurements_stay_unknown_and_evidence_is_immutable() {
        let root = tempfile::tempdir().unwrap();
        let (fit, mut outcome) = external_fixture(root.path(), Predictions::Values(vec![2., 4.]));
        outcome.command_status = EvaluationStatus::Cancelled;
        outcome.reason = Some("controller_interrupted".into());
        outcome.consumption = ObservedConsumption {
            accounting_reason: "interrupted: unknown".into(),
            ..Default::default()
        };
        let observation = score_fit(&fit, &outcome, &["1".into(), "3".into()]).unwrap();
        assert!(observation.result.primary.is_none());
        let accounting = cost(&[&observation]);
        assert_eq!(accounting["unknown_cpu_attempts"], 1);
        assert_eq!(accounting["unique_fitted_predictors"], 0);
        assert!(accounting["cpu_seconds_per_unique_predictor_known"].is_null());
        let path = root.path().join("evidence.json");
        immutable(&path, &json!({"original":true})).unwrap();
        immutable(&path, &json!({"original":true})).unwrap();
        assert!(immutable(&path, &json!({"original":false})).is_err());
    }
}
