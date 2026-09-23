//! Product front door for verified development inputs and immutable experiment planning.

use anyhow::{ensure, Context, Result};
use clap::{Parser, Subcommand};
use jope_datasets::catalog::Task;
use jope_pipeline::experiment::{Experiment, Panel, Resources, EXPERIMENT_SCHEMA};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    process::ExitCode,
};

#[derive(Parser)]
pub struct BenchmarkCli {
    #[command(subcommand)]
    pub command: BenchmarkCommand,
}

#[derive(Subcommand)]
pub enum BenchmarkCommand {
    /// Verify the approved frozen manifest; does not open dataset files or fit models.
    VerifyManifest {
        #[arg(long)]
        manifest: PathBuf,
    },
    /// Check every development file's bytes, actual folds, labels, groups and schema.
    VerifyInputs {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        datasets: PathBuf,
    },
    /// List executable Rust recipes, with aliases and configurations identified separately.
    Methods,
    /// Write an immutable Rust screening plan after all development inputs verify (zero fits).
    Plan {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        datasets: PathBuf,
        #[arg(long)]
        source_sha: String,
        /// Add pinned runtime inventory for a bounded comparator qualification.
        #[arg(long)]
        runtimes: Option<PathBuf>,
        #[arg(long)]
        output: PathBuf,
    },
    /// Execute or resume a pinned bounded campaign.
    #[cfg(target_os = "linux")]
    Run(super::campaign_cli::RunArgs),
    /// Authenticate retained evidence and recompute scores without fitting.
    #[cfg(target_os = "linux")]
    Report(super::campaign_cli::ReportArgs),
    /// Internal single-fit endpoint; the controller supplies resource isolation.
    #[command(hide = true)]
    FitWorker {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Private generated-recipe fit; called only through the admitted pool.
    #[command(hide = true)]
    SynthesizedFitWorker {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
}

pub fn read_panel(path: &Path) -> Result<(Panel, String)> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let hash = jope_pipeline::sha256_hex(&bytes);
    ensure!(
        hash == super::FROZEN_PANEL_SHA256,
        "manifest differs from the approved frozen panel"
    );
    let panel: Panel = serde_json::from_slice(&bytes)?;
    panel.validate()?;
    Ok((panel, hash))
}

pub fn registry() -> Result<BTreeMap<String, Vec<super::methods::RustMethod>>> {
    [Task::Regression, Task::Binary, Task::Multiclass]
        .into_iter()
        .map(|task| {
            Ok((
                task.as_str().to_owned(),
                super::methods::rust_methods(task)?,
            ))
        })
        .collect()
}

pub fn verify_inputs(panel: &Panel, root: &Path) -> Value {
    let outcomes: Vec<_> = panel
        .datasets
        .iter()
        .map(
            |dataset| match super::data::load_development(root, dataset) {
                Ok(_) => json!({"dataset":dataset.id,"task":dataset.task,"ok":true,
                "development_sha256":dataset.split.development_sha256,
                "rows":dataset.split.development_rows,"fold_rows":dataset.split.fold_rows}),
                Err(error) => json!({"dataset":dataset.id,"task":dataset.task,"ok":false,
                "error":format!("{error:#}")}),
            },
        )
        .collect();
    let ok = outcomes.iter().all(|o| o["ok"] == true);
    json!({"schema":"jope.panel-input-check/v1", "ok":ok, "panel":panel.panel_id,
        "datasets":outcomes, "fits":0, "execution_budget_started":false,
        "sealed_inputs_opened":false})
}

fn run(cli: BenchmarkCli) -> Result<Value> {
    match cli.command {
        BenchmarkCommand::VerifyManifest { manifest } => {
            let (panel, hash) = read_panel(&manifest)?;
            Ok(
                json!({"schema":"jope.panel-check/v1","ok":true,"panel":panel.panel_id,
                "manifest_sha256":hash,"datasets":panel.datasets.len(),
                "input_bytes_verified":false,"fits":0,"execution_budget_started":false}),
            )
        }
        BenchmarkCommand::VerifyInputs { manifest, datasets } => {
            let (panel, _) = read_panel(&manifest)?;
            Ok(verify_inputs(&panel, &datasets))
        }
        BenchmarkCommand::Methods => Ok(json!({"schema":"jope.panel-methods/v1", "ok":true,
            "rust":registry()?, "external_adapter_contract":super::methods::ADAPTER_CONTRACT,
            "note":"An executable recipe inventory is not a completed fit or a measured comparison."})),
        BenchmarkCommand::Plan {
            manifest,
            datasets,
            source_sha,
            runtimes,
            output,
        } => {
            if let Some(runtimes) = runtimes {
                #[cfg(target_os = "linux")]
                return super::campaign_cli::prepare(
                    &manifest, &datasets, source_sha, &runtimes, &output,
                );
                #[cfg(not(target_os = "linux"))]
                anyhow::bail!("bounded runtime plans require Linux");
            }
            let (panel, panel_sha256) = read_panel(&manifest)?;
            let verification = verify_inputs(&panel, &datasets);
            ensure!(
                verification["ok"] == true,
                "development verification failed: {verification}"
            );
            let registry = registry()?;
            let experiment = Experiment {
                schema: EXPERIMENT_SCHEMA.into(),
                panel_sha256,
                methods_sha256: jope_pipeline::sha256_hex(&serde_json::to_vec(&registry)?),
                source_sha,
                checkpoints: BTreeMap::new(),
                seed: panel.seed,
                resources: Resources {
                    workers: 20,
                    cpus: 20,
                    memory_bytes: 64 << 30,
                    swap_bytes: 0,
                    gpu_tasks: 1,
                    threads_per_worker: 1,
                    total_seconds: 7200,
                    reporting_seconds: 300,
                    per_fit_seconds: 30,
                },
                expected_artifacts: vec![
                    "journal.jsonl".into(),
                    "report.json".into(),
                    "fitted".into(),
                ],
                protocol: super::PROTOCOL.into(),
                numerical_policy: super::score::NUMERICAL_POLICY.into(),
            };
            let identity = experiment.identity()?;
            let mut requests = Vec::new();
            for spec in &panel.datasets {
                let dataset = super::data::load_development(&datasets, spec)?;
                let cut = dataset.cut(0)?;
                for method in &registry[spec.task.as_str()] {
                    let (request, _) = super::evaluation_input(
                        &dataset,
                        &cut,
                        method,
                        &experiment,
                        jope_pipeline::experiment::EvaluationStage::Screen,
                    )?;
                    requests.push(json!({"dataset":spec.id,"method":method.id,"request":request}));
                }
            }
            let plan = json!({"schema":"jope.panel-plan/v1", "experiment":experiment,
                "experiment_sha256":identity, "methods":registry,
                "screen_requests":requests,
                "verified_inputs":verification,"search_arm_seconds":120,
                "execution_window":null,"fits":0,
                "scope":"Rust screening inventory; external engine versions and ASM checkpoints must be pinned before campaign admission."});
            super::worker::write_private(&output, &plan)?;
            Ok(json!({"ok":true,"plan":output,"experiment_sha256":identity,
                "execution_budget_started":false,"fits":0}))
        }
        #[cfg(target_os = "linux")]
        BenchmarkCommand::Run(args) => super::campaign_cli::run(args),
        #[cfg(target_os = "linux")]
        BenchmarkCommand::Report(args) => super::campaign_cli::report(args),
        BenchmarkCommand::FitWorker { input, output } => {
            super::worker::fit(&input, &output)?;
            Ok(json!({"ok":true,"output":output}))
        }
        BenchmarkCommand::SynthesizedFitWorker { input, output } => {
            super::synthesized::worker(&input, &output)?;
            Ok(json!({"ok":true,"output":output}))
        }
    }
}

pub fn execute(cli: BenchmarkCli) -> ExitCode {
    match run(cli) {
        Ok(value) => {
            println!("{value}");
            if value["ok"] == false {
                ExitCode::FAILURE
            } else {
                ExitCode::SUCCESS
            }
        }
        Err(error) => {
            eprintln!("jope benchmark: {error:#}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn manifest_is_the_approved_bytes_not_merely_a_valid_shape() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let (panel, _) = read_panel(&root.join("benchmarks/panel.json")).unwrap();
        assert_eq!(panel.datasets.len(), 30);
        let tmp = tempfile::tempdir().unwrap();
        let mutated = tmp.path().join("panel.json");
        std::fs::write(&mutated, serde_json::to_vec(&panel).unwrap()).unwrap();
        assert!(read_panel(&mutated)
            .unwrap_err()
            .to_string()
            .contains("frozen panel"));
        let outcomes = verify_inputs(&panel, tmp.path());
        assert_eq!(outcomes["ok"], false);
        assert_eq!(outcomes["datasets"].as_array().unwrap().len(), 30);
        assert_eq!(outcomes["fits"], 0);
    }
}
