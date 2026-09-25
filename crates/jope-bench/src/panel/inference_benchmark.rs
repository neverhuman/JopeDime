//! Dedicated latency evidence, never a serving admission or predictive-quality claim.
use anyhow::{ensure, Context, Result};
use clap::Args;
use jope_pipeline::{
    execution::Artifact,
    experiment::{Experiment, Resources, EXPERIMENT_SCHEMA},
};
use jope_policy::context_producer::{
    self, Feature, FeatureKind, TrainingIdentity, TrainingInput, TrainingRow,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
};

pub const SCHEMA: &str = "jope.inference-benchmark/v2";
pub const TOLERANCE: f32 = 3e-4;
#[derive(Args)]
pub struct PlanArgs {
    #[arg(long)]
    pub manifest: PathBuf,
    #[arg(long)]
    pub datasets: PathBuf,
    #[arg(long)]
    pub configuration: PathBuf,
    #[arg(long)]
    pub campaign_plan: PathBuf,
    #[arg(long)]
    pub configuration_sha256: String,
    #[arg(long)]
    pub references: PathBuf,
    #[arg(long)]
    pub references_sha256: String,
    #[arg(long)]
    pub source_sha: String,
    #[arg(long)]
    pub output: PathBuf,
}
#[derive(Args)]
pub struct RunArgs {
    #[arg(long)]
    pub plan: PathBuf,
    #[arg(long)]
    pub plan_sha256: String,
    #[arg(long)]
    pub journal: PathBuf,
    #[arg(long, hide = true)]
    pub supervisor_unit: Option<String>,
}
#[derive(Args)]
pub struct ReportArgs {
    #[arg(long)]
    pub plan: PathBuf,
    #[arg(long)]
    pub plan_sha256: String,
    #[arg(long)]
    pub journal: PathBuf,
    #[arg(long)]
    pub output: PathBuf,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Dataset {
    pub id: String,
    pub raw: Artifact,
    pub gp: Option<GpInput>,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct GpInput {
    pub pair: jope_pipeline::neural_search::PairInputPlan,
    pub folds: Vec<(Artifact, Vec<String>)>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Plan {
    pub schema: String,
    pub experiment: Experiment,
    pub binary: Artifact,
    pub toolchain: BTreeMap<String, Artifact>,
    pub configuration: Artifact,
    pub references: Artifact,
    pub inputs: Vec<Dataset>,
    pub settings: Value,
}
pub fn digest(bytes: &[u8]) -> String {
    jope_policy::sha256_hex(bytes)
}
pub fn artifact(path: &Path) -> Result<Artifact> {
    let path = fs::canonicalize(path)?;
    Ok(Artifact {
        sha256: digest(&fs::read(&path)?),
        path,
    })
}
pub fn write(path: &Path, value: &impl Serialize) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)?;
    if path.exists() {
        ensure!(
            fs::read(path)? == bytes,
            "immutable output changed: {}",
            path.display()
        );
        return Ok(());
    }
    let mut f = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    f.write_all(&bytes)?;
    f.sync_all()?;
    Ok(())
}
pub fn custody(path: &Path) -> Result<PathBuf> {
    fs::create_dir_all(path)?;
    let path = fs::canonicalize(path)?;
    ensure!(
        !path.starts_with("/tmp")
            && !path.starts_with(env!("CARGO_MANIFEST_DIR").trim_end_matches("/crates/jope-bench")),
        "custody must be outside checkout and /tmp"
    );
    Ok(path)
}
fn settings() -> Value {
    json!({"warmups_per_dataset":2,"measured_seeds":[0,1,2,3,4,5,6,7,8,9],"cold_seed":0,"profile_seed":0,
        "rounds":8,"batch":128,"temperature":1,"max_tokens":128,"ranked":64,"exploration":16,
        "cut":0,"fitting_folds":[1,2,3,4],"tolerance":TOLERANCE,"gpu":"NVIDIA GeForce RTX 3090",
        "gpu_memory_mib":24576,"full_gp_datasets":["230_machine_cpu","auto_insurance_price","197_cpu_act"],"search_seconds_per_arm":120,"candidate_attempts":96,
        "timer":"synchronized wall clock before raw read through fitted export, compilation and first verified prediction",
        "cross_request_context_cache":false,"serving_promotion":false})
}
pub fn plan(args: PlanArgs) -> Result<Value> {
    ensure!(
        option_env!("JOPE_SOURCE_SHA") == Some(args.source_sha.as_str()),
        "source pin differs from compiled JOPE_SOURCE_SHA"
    );
    let directory = custody(args.output.parent().context("output parent")?)?;
    let mut toolchain = BTreeMap::new();
    for name in ["rustc", "python3", "g++"] {
        let output = if name == "rustc" {
            std::process::Command::new("rustup")
                .args(["which", "rustc"])
                .output()?
        } else {
            std::process::Command::new("which").arg(name).output()?
        };
        ensure!(output.status.success(), "missing tool {name}");
        toolchain.insert(
            name.to_string(),
            artifact(Path::new(String::from_utf8(output.stdout)?.trim()))?,
        );
    }
    let configuration = artifact(&args.configuration)?;
    ensure!(
        configuration.sha256 == args.configuration_sha256,
        "configuration pin mismatch"
    );
    let config: super::optimize::Configuration =
        serde_json::from_slice(&fs::read(&configuration.path)?)?;
    config.validate()?;
    for a in [
        &config.checkpoint_receipt,
        &config.policy_weights,
        &config.vocabulary,
        &config.frozen_transforms,
        &config.encoder_assets.configuration,
    ]
    .into_iter()
    .chain(config.encoder_assets.files.iter())
    {
        a.verify()?;
    }
    let references = artifact(&args.references)?;
    ensure!(
        references.sha256 == args.references_sha256,
        "reference pin mismatch"
    );
    ensure!(
        context_producer::case_inputs(&fs::read(&references.path)?, &references.sha256)?.len() == 8,
        "need eight native cases"
    );
    let campaign: super::campaign::CampaignPlan =
        serde_json::from_slice(&fs::read(&args.campaign_plan)?)?;
    campaign.validate()?;
    ensure!(
        campaign.experiment.seed == 0,
        "timing study requires campaign seed zero"
    );
    let (panel, panel_sha) = super::cli::read_panel(&args.manifest)?;
    let mut inputs = Vec::new();
    for dataset in panel
        .datasets
        .iter()
        .filter(|d| d.task == jope_datasets::catalog::Task::Regression)
    {
        let data = super::data::load_development(&args.datasets, dataset)?;
        let rows = data
            .rows
            .iter()
            .filter(|r| r.fold != 0)
            .map(|r| TrainingRow {
                id: r.id.clone(),
                values: r.values.clone(),
                label: r.target.clone(),
            })
            .collect::<Vec<_>>();
        let fitting_rows_sha256 = digest(&serde_json::to_vec(
            &rows.iter().map(|r| &r.id).collect::<Vec<_>>(),
        )?);
        let input = TrainingInput {
            schema: "jope.cpu-context-training/regression-v1".into(),
            identity: TrainingIdentity {
                dataset_sha256: dataset.source.sha256.clone(),
                prepared_sha256: dataset.split.development_sha256.clone(),
                training_sha256: context_producer::training_rows_sha256(&rows)?,
                fitting_rows_sha256,
                cut: 0,
            },
            features: dataset
                .features
                .iter()
                .map(|f| Feature {
                    name: f.name.clone(),
                    kind: match f.kind {
                        jope_pipeline::experiment::ColumnKind::Numeric => FeatureKind::Numeric,
                        _ => FeatureKind::Categorical,
                    },
                })
                .collect(),
            rows,
        };
        let path = directory.join(format!("raw-{}.json", dataset.id));
        write(&path, &input)?;
        let gp = if ["230_machine_cpu", "auto_insurance_price", "197_cpu_act"]
            .contains(&dataset.id.as_str())
        {
            let (_, mut pair) = super::optimize::build_pair_input(&campaign, &config, &data)?;
            pair.seconds = 120;
            pair.max_evaluations = 96;
            pair.incumbents = None;
            let cut = data.cut(0)?;
            let mut folds = Vec::new();
            for fold in 0..5 {
                let (fit_ids, predict_ids) = pair.selection.inner_rows(fold)?;
                let fit = cut
                    .train
                    .rows
                    .iter()
                    .filter(|r| fit_ids.contains(&r.id))
                    .collect::<Vec<_>>();
                let predict = cut
                    .train
                    .rows
                    .iter()
                    .filter(|r| predict_ids.contains(&r.id))
                    .collect::<Vec<_>>();
                let value = super::optimize_fit::fold_input(&data, &fit, &predict)?;
                let path = directory.join(format!("gp-{}-fold-{fold}.json", dataset.id));
                write(&path, &value)?;
                folds.push((
                    artifact(&path)?,
                    predict.iter().map(|r| r.target.clone()).collect(),
                ));
            }
            Some(GpInput { pair, folds })
        } else {
            None
        };
        inputs.push(Dataset {
            id: dataset.id.clone(),
            raw: artifact(&path)?,
            gp,
        });
    }
    inputs.sort_by(|a, b| a.id.cmp(&b.id));
    ensure!(inputs.len() == 10, "need ten regression datasets");
    let settings = settings();
    let experiment = Experiment {
        schema: EXPERIMENT_SCHEMA.into(),
        panel_sha256: panel_sha,
        methods_sha256: digest(&serde_json::to_vec(&(
            &configuration,
            &references,
            &inputs,
            &settings,
            &toolchain,
        ))?),
        source_sha: args.source_sha,
        checkpoints: BTreeMap::from([("policy".into(), jope_policy::ACCEPTED_STATE_SHA256.into())]),
        seed: 0,
        resources: Resources {
            workers: 1,
            cpus: 4,
            memory_bytes: 16 << 30,
            swap_bytes: 0,
            gpu_tasks: 1,
            threads_per_worker: 1,
            total_seconds: 3600,
            reporting_seconds: 300,
            per_fit_seconds: 3300,
        },
        expected_artifacts: vec!["qualification".into(), "requests".into()],
        protocol: SCHEMA.into(),
        numerical_policy: "finite; max absolute error <=3e-4; FP32".into(),
    };
    experiment.validate()?;
    let plan = Plan {
        schema: SCHEMA.into(),
        experiment,
        binary: artifact(&std::env::current_exe()?)?,
        toolchain,
        configuration,
        references,
        inputs,
        settings,
    };
    write(&args.output, &plan)?;
    Ok(
        json!({"plan":args.output,"sha256":artifact(&args.output)?.sha256,"gpu_window_started":false,"datasets":10}),
    )
}
pub fn read(path: &Path, sha: &str) -> Result<Plan> {
    let bytes = fs::read(path)?;
    ensure!(digest(&bytes) == sha, "plan pin mismatch");
    let p: Plan = serde_json::from_slice(&bytes)?;
    ensure!(
        p.schema == SCHEMA && p.settings == settings() && p.inputs.len() == 10,
        "benchmark protocol changed"
    );
    p.experiment.validate()?;
    let r = &p.experiment.resources;
    ensure!(
        r.workers == 1
            && r.cpus == 4
            && r.memory_bytes == 16 << 30
            && r.swap_bytes == 0
            && r.gpu_tasks == 1
            && r.threads_per_worker == 1
            && r.total_seconds == 3600
            && r.reporting_seconds == 300
            && r.per_fit_seconds == 3300,
        "benchmark resource limits changed"
    );
    ensure!(
        p.experiment.methods_sha256
            == digest(&serde_json::to_vec(&(
                &p.configuration,
                &p.references,
                &p.inputs,
                &p.settings,
                &p.toolchain
            ))?),
        "plan input identity changed"
    );
    ensure!(
        p.inputs.windows(2).all(|v| v[0].id < v[1].id),
        "duplicate or unordered dataset inputs"
    );
    Ok(p)
}
#[cfg(feature = "inference-cuda")]
#[path = "inference_cuda.rs"]
mod cuda;
#[cfg(feature = "inference-cuda")]
#[path = "inference_export.rs"]
mod export;
#[cfg(feature = "inference-cuda")]
#[path = "inference_gp.rs"]
mod gp;
pub fn run(args: RunArgs) -> Result<Value> {
    #[cfg(feature = "inference-cuda")]
    {
        cuda::run(args)
    }
    #[cfg(not(feature = "inference-cuda"))]
    {
        let _ = args;
        anyhow::bail!("build with inference-cuda; CPU fallback refused")
    }
}
#[cfg(feature = "inference-cuda")]
pub fn cold(plan: &Path, sha: &str, journal: &Path, index: usize) -> Result<Value> {
    cuda::cold(plan, sha, journal, index)
}
pub fn report(args: ReportArgs) -> Result<Value> {
    let _ = read(&args.plan, &args.plan_sha256)?;
    custody(&args.output)?;
    let status = std::process::Command::new("python3")
        .arg("-c")
        .arg(include_str!("inference_report.py"))
        .arg(&args.plan)
        .arg(&args.plan_sha256)
        .arg(&args.journal)
        .arg(&args.output)
        .status()?;
    ensure!(status.success(), "report generation failed");
    Ok(serde_json::from_slice(&fs::read(
        args.output.join("summary.json"),
    )?)?)
}
