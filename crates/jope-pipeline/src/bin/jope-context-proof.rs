//! Internal qualification controller; installed public front doors may delegate
//! these same library APIs. Every numerical child uses the shared systemd pool.
#[cfg(target_os = "linux")]
fn main() -> anyhow::Result<()> {
    use anyhow::{ensure, Context};
    use clap::{Parser, Subcommand};
    use jope_pipeline::{
        context_qualification::{
            self as q, EncoderAssets, QualificationPlan, ReferenceSources, QUALIFICATION_PROTOCOL,
        },
        execution::{Artifact, ExecutionPool},
        experiment::{Experiment, Resources, EXPERIMENT_SCHEMA},
        sha256_hex, supervisor,
    };
    use jope_policy::context_producer::{self, AssetFile, EncoderFiles};
    use std::{collections::BTreeMap, fs, path::PathBuf};
    #[derive(Parser)]
    struct Cli {
        #[command(subcommand)]
        command: Command,
    }
    #[derive(Subcommand)]
    enum Command {
        /// Parent-only verified reference selection/materialization; no model load.
        Prepare {
            #[arg(long)]
            sources: PathBuf,
            #[arg(long)]
            sources_sha256: String,
            #[arg(long)]
            encoder_files: PathBuf,
            #[arg(long)]
            encoder_files_sha256: String,
            #[arg(long)]
            worker: PathBuf,
            #[arg(long)]
            worker_sha256: String,
            #[arg(long)]
            source_sha40: String,
            #[arg(long)]
            resources: PathBuf,
            #[arg(long)]
            resources_sha256: String,
            #[arg(long)]
            output: PathBuf,
        },
        /// Launch or resume the existing immutable plan under shared supervision.
        Run {
            #[arg(long)]
            plan: PathBuf,
            #[arg(long)]
            plan_sha256: String,
            #[arg(long, hide = true)]
            supervisor_unit: Option<String>,
        },
    }
    fn read<T: serde::de::DeserializeOwned>(
        path: PathBuf,
        sha256: String,
        max: u64,
    ) -> anyhow::Result<T> {
        Ok(serde_json::from_slice(
            &AssetFile { path, sha256 }.read(max)?,
        )?)
    }
    fn private_dir(path: &std::path::Path) -> anyhow::Result<()> {
        use std::os::unix::fs::{DirBuilderExt, PermissionsExt};
        if !path.exists() {
            fs::DirBuilder::new().mode(0o700).create(path)?;
        }
        ensure!(
            path.is_absolute()
                && path.canonicalize()? == path
                && fs::symlink_metadata(path)?.is_dir()
                && fs::metadata(path)?.permissions().mode() & 0o077 == 0,
            "proof directory must be private and physical"
        );
        Ok(())
    }
    match Cli::parse().command {
        Command::Prepare {
            sources,
            sources_sha256,
            encoder_files,
            encoder_files_sha256,
            worker,
            worker_sha256,
            source_sha40,
            resources,
            resources_sha256,
            output,
        } => {
            let sources: ReferenceSources = read(sources, sources_sha256, 1 << 20)?;
            let files: EncoderFiles = read(encoder_files, encoder_files_sha256, 1 << 20)?;
            let resources: Resources = read(resources, resources_sha256, 1 << 20)?;
            resources.validate()?;
            ensure!(
                resources.gpu_tasks == 0 && resources.per_fit_seconds <= 30,
                "CPU proof jobs require30s maximum and no GPU"
            );
            private_dir(&output)?;
            let reference_dir = output.join("references");
            private_dir(&reference_dir)?;
            let journal = output.join("journal");
            private_dir(&journal)?;
            let references = q::materialize_references(&sources, &reference_dir)?;
            let assets: EncoderAssets =
                q::prepare_assets(&files, &output.join("encoder-files.json"))?;
            let program = Artifact {
                path: worker,
                sha256: worker_sha256,
            };
            program.verify()?;
            let mut checkpoints = BTreeMap::new();
            checkpoints.insert(
                "context_transforms".into(),
                context_producer::TRANSFORMS_SHA256.into(),
            );
            let experiment = Experiment {
                schema: EXPERIMENT_SCHEMA.into(),
                panel_sha256: references.manifest.sha256.clone(),
                methods_sha256: sha256_hex(&serde_json::to_vec(&(
                    QUALIFICATION_PROTOCOL,
                    &program.sha256,
                    &assets.configuration.sha256,
                ))?),
                source_sha: source_sha40,
                checkpoints,
                seed: 20260923,
                resources,
                expected_artifacts: vec![
                    "context-qualification.json".into(),
                    "eight-native-case-outcomes".into(),
                ],
                protocol: QUALIFICATION_PROTOCOL.into(),
                numerical_policy: context_producer::verification_policy_sha256(),
            };
            let plan = q::plan_qualification(&experiment, references, program, assets, &journal)?;
            let bytes = serde_json::to_vec(&plan)?;
            println!(
                "{} {}",
                journal.join("context-qualification-plan.json").display(),
                sha256_hex(&bytes)
            );
        }
        Command::Run {
            plan,
            plan_sha256,
            supervisor_unit,
        } => {
            let config: QualificationPlan = read(plan.clone(), plan_sha256.clone(), 32 << 20)?;
            ensure!(
                config.schema == QUALIFICATION_PROTOCOL
                    && config.experiment.protocol == QUALIFICATION_PROTOCOL,
                "unknown context proof protocol"
            );
            if let Some(unit) = supervisor_unit {
                let mut pool = ExecutionPool::open(
                    &config.experiment,
                    &config.journal,
                    supervisor::for_unit(&unit)?,
                )?;
                let cancellation = supervisor::cancellation()?;
                let report = q::run_qualification(&mut pool, &config, &cancellation)?;
                println!("{}", serde_json::to_string(&report)?);
                ensure!(
                    report.completed.is_some(),
                    "context qualification unavailable: {}",
                    report
                        .validation_error
                        .as_deref()
                        .unwrap_or("unknown outcome")
                );
            } else {
                let program_path = std::env::current_exe()?.canonicalize()?;
                let program = Artifact {
                    sha256: jope_pipeline::sha256_file(&program_path)?,
                    path: program_path,
                };
                let arguments = vec![
                    "run".into(),
                    "--plan".into(),
                    plan.to_str().context("plan path must be UTF8")?.into(),
                    "--plan-sha256".into(),
                    plan_sha256,
                ];
                let launched =
                    supervisor::launch(&config.experiment, &config.journal, &program, &arguments)?;
                println!("{}", serde_json::to_string(&launched)?);
                ensure!(
                    launched.exit_code == Some(0),
                    "supervised context proof did not complete: {:?}",
                    launched.reason
                );
            }
        }
    }
    Ok(())
}
#[cfg(not(target_os = "linux"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("qualified CPU context proof requires the Linux supervisor")
}
