//! One measured fitted-predictor demonstration through the shared worker boundary.
//! Synthetic inputs are explicit and this generic recipe demonstration is not neural evidence.
use super::{
    data::raw_table,
    methods::{digest, FittedRustArtifact, RustFitRequest, RustMethod},
    score::{self, Predictions},
    worker::RustWorkerOutput,
};
use anyhow::{ensure, Context, Result};
use clap::{Parser, ValueEnum};
use feat_math::advanced_v3::{ColumnSpec, DataSchema, RawTable, RowSemantics};
use jope_datasets::catalog::Task;
use jope_pipeline::experiment::{
    Column, ColumnKind, EvaluationRequest, EvaluationStage, EvaluationStatus, Experiment, Metric,
    Resources, EXPERIMENT_SCHEMA, REQUEST_SCHEMA,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    process::ExitCode,
    sync::atomic::AtomicBool,
    time::Instant,
};

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum DemoTask {
    Regression,
    Binary,
    Multiclass,
}
impl DemoTask {
    fn task(self) -> Task {
        match self {
            Self::Regression => Task::Regression,
            Self::Binary => Task::Binary,
            Self::Multiclass => Task::Multiclass,
        }
    }
}
#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum DemoExport {
    Python,
    Rust,
    Cpp17,
    Cpp20,
    Javascript,
}
impl DemoExport {
    pub(super) fn backend(self) -> super::inference::Backend {
        use super::inference::Backend;
        match self {
            Self::Python => Backend::Python,
            Self::Rust => Backend::Rust,
            Self::Cpp17 => Backend::Cpp17,
            Self::Cpp20 => Backend::Cpp20,
            Self::Javascript => Backend::JavaScript,
        }
    }
}

#[derive(Debug, Parser)]
#[command(group(clap::ArgGroup::new("input").required(true).args(["synthetic","dataset","plan","asm_plan"])))]
pub struct DemoCli {
    /// Explicit deterministic fixture; never substitutes for a failed real input.
    #[arg(long, value_enum)]
    pub synthetic: Option<DemoTask>,
    /// One dataset ID in the unchanged frozen panel; only development fold zero is used.
    #[arg(long,requires_all=["manifest","datasets"])]
    pub dataset: Option<String>,
    #[arg(long)]
    pub manifest: Option<PathBuf>,
    #[arg(long)]
    pub datasets: Option<PathBuf>,
    /// Existing authenticated one-dataset ASM origin plan; total budget must be <=60s.
    #[arg(long, requires_all=["asm_plan_sha256","manifest","datasets"], conflicts_with_all=["synthetic","dataset","method","plan"])]
    pub asm_plan: Option<PathBuf>,
    #[arg(long, requires = "asm_plan")]
    pub asm_plan_sha256: Option<String>,
    /// Private run directory; fresh demos require a new directory.
    #[arg(long)]
    pub output: PathBuf,
    /// Full reviewed build source SHA retained separately from the executable digest.
    #[arg(long, required_unless_present_any = ["plan", "asm_plan"])]
    pub source_sha: Option<String>,
    /// An exact registered method ID; default is ridge, logit_ridge or multiclass_ridge.
    #[arg(long)]
    pub method: Option<String>,
    /// Emit allowlisted fitted source; qualification is reported separately.
    #[arg(long, value_enum)]
    pub export: Vec<DemoExport>,
    #[arg(long, hide = true)]
    pub plan: Option<PathBuf>,
    #[arg(long, hide = true)]
    pub supervisor_unit: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum DemoInput {
    Synthetic {
        task: Task,
        version: String,
    },
    Frozen {
        manifest: PathBuf,
        datasets: PathBuf,
        dataset: String,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DemoPlan {
    pub schema: String,
    pub input: DemoInput,
    pub method: RustMethod,
    pub task: Task,
    pub experiment: Experiment,
    pub exports: Vec<super::inference::Backend>,
}
impl DemoPlan {
    pub fn identity(&self) -> Result<String> {
        self.validate()?;
        Ok(digest(&serde_json::to_vec(self)?))
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == "jope.warm-demo-plan/v1",
            "unsupported demo plan"
        );
        self.experiment.validate()?;
        ensure!(
            self.experiment.panel_sha256 == digest(&serde_json::to_vec(&self.input)?)
                && self.experiment.methods_sha256 == digest(&serde_json::to_vec(&self.method)?),
            "demo input or recipe identity changed"
        );
        ensure!(
            super::methods::rust_methods(self.task)?.contains(&self.method),
            "unregistered demo recipe"
        );
        ensure!(self.experiment.protocol=="jope.warm-demo/v1:development-fold=0:generic-single-recipe:train-constant-comparator" && self.experiment.numerical_policy==score::NUMERICAL_POLICY,"demo protocol changed");
        for (index, backend) in self.exports.iter().enumerate() {
            ensure!(
                !self.exports[..index].contains(backend),
                "duplicate export backend"
            );
        }
        let budget = &self.experiment.resources;
        ensure!(
            budget.workers == 1
                && budget.cpus == 1
                && budget.memory_bytes == (2u64 << 30)
                && budget.swap_bytes == 0
                && budget.gpu_tasks == 0
                && budget.threads_per_worker == 1
                && budget.total_seconds == 60
                && budget.reporting_seconds == 5
                && budget.per_fit_seconds == 30,
            "demo resource limits changed"
        );
        match &self.input {
            DemoInput::Synthetic { task, version } => ensure!(
                *task == self.task && version == "jope.synthetic-demo/v1",
                "synthetic fixture changed"
            ),
            DemoInput::Frozen {
                manifest, datasets, ..
            } => ensure!(
                manifest.is_absolute() && datasets.is_absolute(),
                "demo input paths must be absolute"
            ),
        };
        Ok(())
    }
}

struct Prepared {
    task: Task,
    classes: Vec<String>,
    features: Vec<Column>,
    schema: DataSchema,
    train: RawTable,
    targets: Vec<f64>,
    labels: Vec<String>,
    predict: RawTable,
    raw_predict: Vec<Vec<Option<String>>>,
    hold_labels: Vec<String>,
    dataset_sha256: String,
    split_sha256: String,
    train_sha256: String,
    hold_sha256: String,
}
fn prepare(plan: &DemoPlan) -> Result<Prepared> {
    match &plan.input {
        DemoInput::Synthetic { task, .. } => synthetic(*task),
        DemoInput::Frozen {
            manifest,
            datasets,
            dataset,
        } => {
            let (panel, _) = super::cli::read_panel(manifest)?;
            let spec = panel
                .datasets
                .iter()
                .find(|entry| &entry.id == dataset)
                .context("dataset absent from frozen panel")?;
            ensure!(spec.task == plan.task, "demo task changed");
            let data = super::data::load_development(datasets, spec)?;
            let cut = data.cut(0)?;
            Ok(Prepared {
                task: spec.task,
                classes: spec.classes.clone(),
                features: spec.features.clone(),
                schema: data.schema,
                train: cut.train.table,
                targets: cut.train.targets,
                labels: cut.train.labels,
                predict: cut.hold.table,
                raw_predict: cut.hold.rows.iter().map(|row| row.values.clone()).collect(),
                hold_labels: cut.hold.labels,
                dataset_sha256: spec.source.sha256.clone(),
                split_sha256: spec.split.sha256.clone(),
                train_sha256: cut.train.rows_sha256,
                hold_sha256: cut.hold.rows_sha256,
            })
        }
    }
}
fn synthetic(task: Task) -> Result<Prepared> {
    let features = vec![
        Column {
            name: "amount".into(),
            kind: ColumnKind::Numeric,
        },
        Column {
            name: "category".into(),
            kind: ColumnKind::Categorical,
        },
    ];
    let schema = DataSchema {
        columns: vec![
            ColumnSpec::numeric("amount"),
            ColumnSpec::categorical("category"),
        ],
        row_semantics: RowSemantics::Iid,
    };
    let classes: Vec<String> = match task {
        Task::Regression => vec![],
        Task::Binary => vec!["negative".into(), "positive".into()],
        Task::Multiclass => vec!["zebra".into(), "alpha".into(), "middle".into()],
        _ => anyhow::bail!("unsupported demo task"),
    };
    let mut rows = Vec::new();
    let mut all_labels = Vec::new();
    let mut all_targets = Vec::new();
    for i in 0..75 {
        let class = if classes.is_empty() {
            0
        } else {
            i % classes.len()
        };
        let amount = if classes.is_empty() {
            i as f64 / 8.0 - 4.0
        } else {
            class as f64 * 3.0 + (i / 3) as f64 / 100.0
        };
        let target = if classes.is_empty() {
            2.0 * amount + 1.0
        } else {
            class as f64
        };
        let category = if i % 11 == 0 {
            None
        } else {
            Some(["NA", "null", ""][i % 3].to_owned())
        };
        rows.push(vec![
            if i % 13 == 0 {
                None
            } else {
                Some(amount.to_string())
            },
            category,
        ]);
        all_labels.push(if classes.is_empty() {
            target.to_string()
        } else {
            classes[class].clone()
        });
        all_targets.push(target);
    }
    let raw_predict = rows[60..].to_vec();
    let dataset_sha256 = digest(&serde_json::to_vec(&(
        "explicitly synthetic",
        task,
        &rows,
        &all_labels,
    ))?);
    Ok(Prepared {
        task,
        classes,
        features: features.clone(),
        schema,
        train: raw_table(&features, &rows[..60])?,
        targets: all_targets[..60].to_vec(),
        labels: all_labels[..60].to_vec(),
        predict: raw_table(&features, &raw_predict)?,
        raw_predict,
        hold_labels: all_labels[60..].to_vec(),
        dataset_sha256,
        split_sha256: digest(b"jope.synthetic-demo/v1:train=0..60:development=60..75"),
        train_sha256: digest(&serde_json::to_vec(&(0..60).collect::<Vec<_>>())?),
        hold_sha256: digest(&serde_json::to_vec(&(60..75).collect::<Vec<_>>())?),
    })
}
fn baseline(data: &Prepared) -> Result<(Value, Predictions)> {
    let rows = data.raw_predict.len();
    if data.task == Task::Regression {
        let mean = data.targets.iter().sum::<f64>() / data.targets.len() as f64;
        ensure!(mean.is_finite(), "trivial baseline is nonfinite");
        Ok((
            json!({"kind":"train_mean","value":mean}),
            Predictions::Values(vec![mean; rows]),
        ))
    } else {
        let mut counts = vec![0usize; data.classes.len()];
        for &target in &data.targets {
            counts[target as usize] += 1;
        }
        let probabilities = counts
            .iter()
            .map(|count| *count as f64 / data.targets.len() as f64)
            .collect::<Vec<_>>();
        Ok((
            json!({"kind":"train_class_priors","classes":data.classes,"probabilities":probabilities}),
            Predictions::Probabilities {
                classes: data.classes.clone(),
                values: vec![probabilities; rows],
            },
        ))
    }
}

pub(super) fn exact_private(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}
pub(super) fn read_bounded(path: &Path) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take(128 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= 128 * 1024 * 1024,
        "demo artifact exceeds 128 MiB"
    );
    Ok(bytes)
}

/// Fresh private output only; evidence never lands in a checkout or /tmp.
pub(super) fn new_output(path: &Path) -> Result<PathBuf> {
    let absolute = std::path::absolute(path)?;
    ensure!(
        fs::symlink_metadata(&absolute).is_err(),
        "demo output already exists; retain its original evidence"
    );
    let ancestor = absolute
        .ancestors()
        .find(|p| p.exists())
        .context("output has no existing ancestor")?;
    let resolved = fs::canonicalize(ancestor)?.join(absolute.strip_prefix(ancestor)?);
    ensure!(
        !resolved.starts_with("/tmp") && !resolved.ancestors().any(|p| p.join(".git").exists()),
        "demo evidence requires a durable path outside Git checkouts and /tmp"
    );
    let parent = absolute.parent().context("output parent missing")?;
    fs::create_dir_all(parent)?;
    let parent = fs::canonicalize(parent)?;
    ensure!(
        !parent.starts_with("/tmp") && !parent.ancestors().any(|p| p.join(".git").exists()),
        "unsafe demo output parent"
    );
    let mut builder = fs::DirBuilder::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(&absolute)?;
    Ok(fs::canonicalize(absolute)?)
}

#[cfg(target_os = "linux")]
pub(super) fn program() -> Result<jope_pipeline::execution::Artifact> {
    let path = fs::canonicalize(std::env::current_exe()?)?;
    let mut hash = sha2::Sha256::new();
    use sha2::Digest;
    let mut file = fs::File::open(&path)?;
    let mut buffer = [0u8; 65536];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(jope_pipeline::execution::Artifact {
        path,
        sha256: format!("{:x}", hash.finalize()),
    })
}

#[cfg(target_os = "linux")]
pub fn run_in_supervisor(
    plan: &DemoPlan,
    output: &Path,
    supervisor: jope_pipeline::execution::Supervisor,
    cancel: &AtomicBool,
) -> Result<Value> {
    use jope_pipeline::execution::{Artifact, ExecutionPool, WorkerCommand, WorkerJob};
    plan.validate()?;
    let started = Instant::now();
    let journal = output.join("journal");
    let mut pool = ExecutionPool::open(&plan.experiment, &journal, supervisor)?;
    pool.admit()?;
    let data = prepare(plan)?;
    let preparation_seconds = started.elapsed().as_secs_f64();
    let input = RustFitRequest {
        task: data.task,
        schema: data.schema.clone(),
        classes: data.classes.clone(),
        train: data.train.clone(),
        targets: data.targets.clone(),
        predict: data.predict.clone(),
        method: plan.method.clone(),
    };
    input.validate()?;
    let request = EvaluationRequest {
        schema: REQUEST_SCHEMA.into(),
        experiment_sha256: plan.experiment.identity()?,
        dataset_sha256: data.dataset_sha256.clone(),
        split_sha256: data.split_sha256.clone(),
        fitting_rows_sha256: data.train_sha256.clone(),
        scoring_rows_sha256: data.hold_sha256.clone(),
        fitting_rows: data.labels.len(),
        scoring_rows: data.hold_labels.len(),
        recipe_sha256: digest(plan.method.dsl.as_bytes()),
        evaluator_sha256: digest(super::EVALUATOR.as_bytes()),
        checkpoint_sha256: None,
        task: data.task,
        metric: Metric::for_task(data.task)?,
        classes: data.classes.clone(),
        fold: 0,
        stage: EvaluationStage::Screen,
        seed: plan.experiment.seed,
        numerical_policy: score::NUMERICAL_POLICY.into(),
    };
    let directory = journal.join("jobs").join(request.identity()?);
    fs::create_dir_all(&directory)?;
    let input_path = directory.join("input.json");
    super::worker::write_private(&input_path, &input)?;
    let input_sha256 = digest(&read_bounded(&input_path)?);
    let result_path = directory.join("result.json");
    let job = WorkerJob {
        request: request.clone(),
        command: WorkerCommand {
            program: program()?,
            args: vec![
                "benchmark".into(),
                "fit-worker".into(),
                "--input".into(),
                input_path.to_string_lossy().into_owned(),
                "--output".into(),
                result_path.to_string_lossy().into_owned(),
            ],
            inputs: vec![Artifact {
                path: input_path,
                sha256: input_sha256.clone(),
            }],
            environment: BTreeMap::new(),
        },
        directory,
        outputs: vec![PathBuf::from("result.json")],
        arm: None,
    };
    let expected_job = job.identity()?;
    let expected_request = request.identity()?;
    let outcomes = pool.run(vec![job], cancel)?;
    ensure!(outcomes.len() == 1, "missing demo worker outcome");
    let outcome = &outcomes[0];
    outcome.verify_outputs()?;
    ensure!(
        outcome.job_sha256 == expected_job
            && outcome.request_sha256 == expected_request
            && outcome.outputs.len() == 1,
        "stale or unexpected demo worker outcome"
    );
    super::worker::write_private(&output.join("execution.json"), outcome)?;
    ensure!(
        outcome.command_status == EvaluationStatus::Completed,
        "demo fit did not complete: {:?}: {:?}",
        outcome.command_status,
        outcome.reason
    );
    let completed = outcome
        .outputs
        .iter()
        .find(|artifact| artifact.path == result_path)
        .context("worker result missing")?;
    completed.verify()?;
    let result: RustWorkerOutput = serde_json::from_slice(&read_bounded(&result_path)?)?;
    ensure!(
        result.schema == "jope.panel-rust-worker-output/v1" && result.input_sha256 == input_sha256,
        "worker result input identity differs"
    );
    ensure!(
        result.artifact.method == plan.method
            && result.artifact.task == data.task
            && result.artifact.classes == data.classes,
        "demo fitted envelope differs from request"
    );
    let reload_started = Instant::now();
    let fitted_bytes = result.artifact.bytes()?;
    ensure!(
        digest(&fitted_bytes) == result.artifact_sha256,
        "worker fitted state identity differs"
    );
    let fitted_path = output.join("fitted.json");
    exact_private(&fitted_path, &fitted_bytes)?;
    let restored =
        FittedRustArtifact::from_bytes(&read_bounded(&fitted_path)?, &result.artifact_sha256)?;
    let predictions = restored.predict_raw(&data.features, &data.raw_predict)?;
    ensure!(
        serde_json::to_vec(&predictions)? == serde_json::to_vec(&result.predictions)?,
        "saved fitted prediction parity failed"
    );
    let reload_prediction_seconds = reload_started.elapsed().as_secs_f64();
    super::worker::write_private(&output.join("predictions.json"), &predictions)?;
    let comparator_started = Instant::now();
    let (baseline_state, baseline_predictions) = baseline(&data)?;
    let model_score =
        score::score_predictions(data.task, &data.classes, &data.hold_labels, &predictions)?;
    let baseline_score = score::score_predictions(
        data.task,
        &data.classes,
        &data.hold_labels,
        &baseline_predictions,
    )?;
    super::worker::write_private(
        &output.join("baseline.json"),
        &json!({"state":baseline_state,"predictions":baseline_predictions,"score":baseline_score}),
    )?;
    let comparison_seconds = comparator_started.elapsed().as_secs_f64();
    let mut exports = Vec::new();
    for backend in &plan.exports {
        let export = super::inference::export_fitted(&restored, *backend)?;
        match export {
            super::inference::ExportOutcome::Unsupported { reason, .. } => {
                exports.push(json!({"backend":backend,"status":"unsupported","reason":reason}))
            }
            super::inference::ExportOutcome::Emitted { bundle } => {
                let folder = output
                    .join("exports")
                    .join(format!("{backend:?}").to_ascii_lowercase());
                fs::create_dir_all(&folder)?;
                for (name, text) in &bundle.files {
                    exact_private(&folder.join(name), text.as_bytes())?;
                }
                super::worker::write_private(&folder.join("manifest.json"), &bundle.manifest)?;
                exports.push(json!({"backend":backend,"status":"emitted-unqualified","bundle_sha256":bundle.identity()?,"directory":folder}));
            }
        }
    }
    let report = json!({"schema":"jope.warm-demo/v1","ok":true,"plan_sha256":plan.identity()?,"task":data.task,"classes":data.classes,"input":plan.input,"method":plan.method.id,"recipe_origin":"registered_generic","neural_consumed":false,"train_rows":data.labels.len(),"development_rows":data.hold_labels.len(),"sealed_inputs_opened":false,"fitted":{"path":fitted_path,"sha256":result.artifact_sha256,"reload_verified":true},"artifact_sha256":{"predictions.json":digest(&read_bounded(&output.join("predictions.json"))?),"baseline.json":digest(&read_bounded(&output.join("baseline.json"))?),"execution.json":digest(&read_bounded(&output.join("execution.json"))?)},"source_sha":plan.experiment.source_sha,"manifest_sha256":if matches!(plan.input, DemoInput::Frozen{..}){Some(super::FROZEN_PANEL_SHA256)}else{None},"model_score":model_score,"baseline_score":baseline_score,"loss_improvement":baseline_score.loss-model_score.loss,"stage_seconds":{"preparation":preparation_seconds,"fit":outcome.consumption.wall_seconds,"reload_prediction":reload_prediction_seconds,"comparison":comparison_seconds,"supervised_total":started.elapsed().as_secs_f64()},"limits":plan.experiment.resources,"execution_window":pool.window(),"worker_fit_attempts":1,"fitted_artifacts":1,"reusable_recipes":1,"constant_comparators":1,"consumption":outcome.consumption,"exports":exports,"qualification":"synthetic fixture only or one declared development fold; no panel or neural claim"});
    super::worker::write_private(&output.join("demo-result.json"), &report)?;
    Ok(report)
}

fn initial_plan(cli: &DemoCli) -> Result<DemoPlan> {
    let (input, task) = if let Some(task) = cli.synthetic {
        (
            DemoInput::Synthetic {
                task: task.task(),
                version: "jope.synthetic-demo/v1".into(),
            },
            task.task(),
        )
    } else {
        let manifest = fs::canonicalize(cli.manifest.as_ref().context("manifest required")?)?;
        let datasets = fs::canonicalize(cli.datasets.as_ref().context("datasets root required")?)?;
        let dataset = cli.dataset.as_ref().context("dataset required")?.clone();
        let (panel, _) = super::cli::read_panel(&manifest)?;
        let task = panel
            .datasets
            .iter()
            .find(|entry| entry.id == dataset)
            .context("dataset absent from fixed panel")?
            .task;
        (
            DemoInput::Frozen {
                manifest,
                datasets,
                dataset,
            },
            task,
        )
    };
    let default = match task {
        Task::Regression => "rust.ridge",
        Task::Binary => "rust.logit_ridge",
        _ => "rust.multiclass_ridge",
    };
    let id = cli.method.as_deref().unwrap_or(default);
    let method = super::methods::rust_methods(task)?
        .into_iter()
        .find(|method| method.id == id)
        .context("method is not a registered task-compatible recipe")?;
    let experiment = Experiment {
        schema: EXPERIMENT_SCHEMA.into(),
        panel_sha256: digest(&serde_json::to_vec(&input)?),
        methods_sha256: digest(&serde_json::to_vec(&method)?),
        source_sha: cli.source_sha.clone().context("source SHA required")?,
        checkpoints: BTreeMap::new(),
        seed: 20260923,
        resources: Resources {
            workers: 1,
            cpus: 1,
            memory_bytes: 2 << 30,
            swap_bytes: 0,
            gpu_tasks: 0,
            threads_per_worker: 1,
            total_seconds: 60,
            reporting_seconds: 5,
            per_fit_seconds: 30,
        },
        expected_artifacts: vec![
            "execution.json".into(),
            "fitted.json".into(),
            "predictions.json".into(),
            "baseline.json".into(),
            "demo-result.json".into(),
        ],
        protocol:
            "jope.warm-demo/v1:development-fold=0:generic-single-recipe:train-constant-comparator"
                .into(),
        numerical_policy: score::NUMERICAL_POLICY.into(),
    };
    let plan = DemoPlan {
        schema: "jope.warm-demo-plan/v1".into(),
        input,
        method,
        task,
        experiment,
        exports: cli.export.iter().map(|target| target.backend()).collect(),
    };
    plan.validate()?;
    Ok(plan)
}

#[cfg(target_os = "linux")]
fn run(cli: DemoCli) -> Result<Value> {
    let started = Instant::now();
    if cli.asm_plan.is_some() {
        #[cfg(feature = "asm")]
        return super::demo_asm::run(cli);
        #[cfg(not(feature = "asm"))]
        anyhow::bail!("warm ASM requires the compiled asm feature; no generic substitute was run");
    }
    if let Some(unit) = &cli.supervisor_unit {
        let path = cli
            .plan
            .as_ref()
            .context("supervised demo needs its private plan")?;
        let plan: DemoPlan = serde_json::from_slice(&read_bounded(path)?)?;
        plan.validate()?;
        let output = fs::canonicalize(&cli.output)?;
        ensure!(
            path == &output.join("demo-plan.json"),
            "unexpected supervised plan path"
        );
        let cancelled = jope_pipeline::supervisor::cancellation()?;
        let result = run_in_supervisor(
            &plan,
            &output,
            jope_pipeline::supervisor::for_unit(unit)?,
            cancelled.as_ref(),
        );
        if let Err(error) = &result {
            let _ = super::worker::write_private(
                &output.join("demo-failure.json"),
                &json!({"schema":"jope.warm-demo-failure/v1","ok":false,"plan_sha256":plan.identity()?,"error":format!("{error:#}"),"elapsed_seconds":started.elapsed().as_secs_f64()}),
            );
        }
        return result;
    }
    ensure!(
        cli.plan.is_none(),
        "private plans are started only by the shared supervisor"
    );
    let plan = initial_plan(&cli)?;
    let output = new_output(&cli.output)?;
    let plan_path = output.join("demo-plan.json");
    super::worker::write_private(&plan_path, &plan)?;
    let arguments = vec![
        "demo".into(),
        "--output".into(),
        output.to_string_lossy().into_owned(),
        "--plan".into(),
        plan_path.to_string_lossy().into_owned(),
    ];
    let program = program()?;
    let launched = jope_pipeline::supervisor::launch(
        &plan.experiment,
        &output.join("journal"),
        &program,
        &arguments,
    )?;
    let child = if output.join("demo-result.json").is_file() {
        Some(serde_json::from_slice::<Value>(&read_bounded(
            &output.join("demo-result.json"),
        )?)?)
    } else {
        None
    };
    let ok =
        launched.exit_code == Some(0) && child.as_ref().is_some_and(|value| value["ok"] == true);
    let elapsed = started.elapsed().as_secs_f64();
    let result_file_sha256 = if output.join("demo-result.json").is_file() {
        Some(digest(&read_bounded(&output.join("demo-result.json"))?))
    } else {
        None
    };
    let report = json!({"plan_file_sha256":digest(&read_bounded(&plan_path)?),"result_file_sha256":result_file_sha256,"program":program,"source_sha":plan.experiment.source_sha,"schema":"jope.warm-demo-invocation/v1","ok":ok,"plan_sha256":plan.identity()?,"warm_elapsed_seconds":elapsed,"under_60_seconds":ok&&elapsed<60.0,"cold_install_included":false,"launch":{"unit":launched.unit,"exit_code":launched.exit_code,"original_window":launched.original_window,"reason":launched.reason},"result":child});
    super::worker::write_private(&output.join("invocation-receipt.json"), &report)?;
    Ok(report)
}
#[cfg(not(target_os = "linux"))]
fn run(_cli: DemoCli) -> Result<Value> {
    anyhow::bail!("bounded warm demo requires Linux cgroup v2 supervision")
}
pub fn execute(cli: DemoCli) -> ExitCode {
    match run(cli) {
        Ok(report) => {
            println!("{report}");
            if report["ok"] == true {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(error) => {
            println!(
                "{}",
                json!({"schema":"jope.warm-demo-error/v1","ok":false,"error":format!("{error:#}")})
            );
            ExitCode::FAILURE
        }
    }
}

/// Read-only host observations; existence is never reported as execution qualification.
pub fn doctor() -> Value {
    let mut value = jope_pipeline::front_door::doctor();
    value["warm_demo"] = json!({"status":"implemented-unqualified","command":"jope demo --help","synthetic_inputs_require":"--synthetic TASK","real_input":"verified frozen development fold zero","limits":{"cpus":1,"memory_bytes":2u64<<30,"swap_bytes":0,"fit_seconds":30,"total_seconds":60,"reporting_seconds":5},"linux":cfg!(target_os="linux"),"cgroup_v2_present":Path::new("/sys/fs/cgroup/cgroup.controllers").is_file(),"qualification":"one saved fitted model, reload, predictions and a train-only constant comparator; each run records actual elapsed time"});
    value["warm_asm_demo"] = json!({"status":"implemented-unqualified","compiled":cfg!(all(feature="asm",target_os="linux")),"command":"jope demo --asm-plan PLAN --asm-plan-sha256 SHA --manifest MANIFEST --datasets ROOT --output NEW", "qualification":"shared authenticated origin proof; original <=60s plan, exact ASM and generic selected refits, native reload and predictions; no fallback-success", "workers":"separately authenticated policy/context binaries and eight-case qualification required", "neural_slots":64,"exploration_slots":16,"generic_slots":80,"under_60_seconds":"measured for each full invocation"});
    value["private_install"] = json!({"status":"script_available","command":"python3 deploy/install-private.py --help","source_build":"cargo build --locked --release -p jope-cli --features asm --bin jope","policy_context_workers":"separate authenticated worker-build inventory required","torch_engine_required":false,"cold_time_separate":true});
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn synthetic_demo_is_explicit_and_baseline_uses_training_only() {
        for task in [Task::Regression, Task::Binary, Task::Multiclass] {
            let mut data = synthetic(task).unwrap();
            assert_eq!(data.labels.len(), 60);
            assert_eq!(data.hold_labels.len(), 15);
            let (state, predictions) = baseline(&data).unwrap();
            score::score_predictions(task, &data.classes, &data.hold_labels, &predictions).unwrap();
            data.hold_labels.fill("changed held-out label".into());
            let (again, again_predictions) = baseline(&data).unwrap();
            assert_eq!(state, again);
            assert_eq!(predictions, again_predictions);
        }
    }
    #[test]
    fn demo_plan_pins_the_task_recipe_and_limits() {
        let cli = DemoCli {
            synthetic: Some(DemoTask::Binary),
            dataset: None,
            manifest: None,
            datasets: None,
            asm_plan: None,
            asm_plan_sha256: None,
            output: PathBuf::from("unused"),
            source_sha: Some("a".repeat(40)),
            method: None,
            export: vec![],
            plan: None,
            supervisor_unit: None,
        };
        let plan = initial_plan(&cli).unwrap();
        assert_eq!(plan.method.id, "rust.logit_ridge");
        let mut changed = plan.clone();
        changed.experiment.resources.total_seconds = 61;
        assert!(changed.validate().is_err());
        let value = serde_json::to_value(&plan).unwrap();
        assert!(!value.to_string().contains("heldout_targets"));
        assert_eq!(value["input"]["kind"], "synthetic");
        let mut duplicate = plan;
        duplicate.exports = vec![super::super::inference::Backend::Python; 2];
        assert!(duplicate.validate().is_err());
    }
    #[test]
    fn demo_evidence_refuses_temporary_and_checkout_paths_before_creation() {
        assert!(new_output(Path::new("/tmp/jope-warm-demo-must-not-exist")).is_err());
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("warm-demo-must-not-exist");
        assert!(new_output(&path).is_err());
        assert!(!path.exists());
    }
    #[test]
    fn asm_demo_requires_authenticated_plan_and_refuses_generic_overrides() {
        let args = [
            "demo",
            "--asm-plan",
            "proof.json",
            "--asm-plan-sha256",
            &"a".repeat(64),
            "--manifest",
            "panel.json",
            "--datasets",
            "data",
            "--output",
            "out",
        ];
        assert!(DemoCli::try_parse_from(args).is_ok());
        let mut bad = args.to_vec();
        bad.extend(["--method", "rust.ridge"]);
        assert!(DemoCli::try_parse_from(bad).is_err());
        assert!(
            DemoCli::try_parse_from(["demo", "--asm-plan", "proof.json", "--output", "out"])
                .is_err()
        );
    }
    #[test]
    fn demo_cli_refuses_implicit_or_mixed_input() {
        use clap::CommandFactory;
        DemoCli::command().debug_assert();
        assert!(DemoCli::try_parse_from([
            "demo",
            "--output",
            "unused",
            "--source-sha",
            &"a".repeat(40)
        ])
        .is_err());
        assert!(DemoCli::try_parse_from([
            "demo",
            "--synthetic",
            "binary",
            "--dataset",
            "penguins",
            "--output",
            "unused",
            "--source-sha",
            &"a".repeat(40)
        ])
        .is_err());
    }
}
