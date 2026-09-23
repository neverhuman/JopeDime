//! Public ASM front door. Optional numerical dependencies stay outside the
//! ordinary comparator build; unavailable support is reported explicitly.
use clap::{Parser, Subcommand};
use serde_json::{json, Value};
use std::{path::PathBuf, process::ExitCode};

#[derive(Parser)]
pub struct OptimizeCli {
    #[command(subcommand)]
    pub command: OptimizeCommand,
}
#[derive(Subcommand)]
pub enum OptimizeCommand {
    /// Read compiled support without loading any model or admitting a budget.
    Capabilities,
    /// Prepare a separately budgeted single-regression operational proof.
    ProofPlan {
        #[arg(long)]
        base_plan: PathBuf,
        #[arg(long)]
        base_plan_sha256: String,
        #[arg(long)]
        configuration: PathBuf,
        #[arg(long)]
        configuration_sha256: String,
        #[arg(long)]
        dataset: String,
        #[arg(long)]
        resources: PathBuf,
        #[arg(long)]
        resources_sha256: String,
        #[arg(long)]
        output: PathBuf,
    },
    /// Bind paired ASM arms and scheduling to an existing verified panel plan.
    Plan {
        #[arg(long)]
        base_plan: PathBuf,
        #[arg(long)]
        base_plan_sha256: String,
        #[arg(long)]
        configuration: PathBuf,
        #[arg(long)]
        configuration_sha256: String,
        #[arg(long)]
        output: PathBuf,
    },
    /// Execute or resume the original bounded panel window.
    #[cfg(target_os = "linux")]
    Run(super::campaign_cli::RunArgs),
    /// Authenticate retained fits and recompute development scores without fitting.
    #[cfg(target_os = "linux")]
    Report(super::campaign_cli::ReportArgs),
}
#[cfg(all(feature = "asm", target_os = "linux"))]
fn run_enabled(cli: OptimizeCli) -> anyhow::Result<Value> {
    use super::{
        campaign::{CampaignPlan, Runtime},
        optimize::Configuration,
    };
    use anyhow::{ensure, Context};
    match cli.command {
        OptimizeCommand::Capabilities => unreachable!("capabilities dispatch precedes resources"),
        OptimizeCommand::ProofPlan {
            base_plan,
            base_plan_sha256,
            configuration,
            configuration_sha256,
            dataset,
            resources,
            resources_sha256,
            output,
        } => super::origin_proof::prepare(
            &base_plan,
            &base_plan_sha256,
            &configuration,
            &configuration_sha256,
            &dataset,
            &resources,
            &resources_sha256,
            &output,
        ),
        OptimizeCommand::Plan {
            base_plan,
            base_plan_sha256,
            configuration,
            configuration_sha256,
            output,
        } => {
            let mut plan: CampaignPlan = serde_json::from_slice(&std::fs::read(base_plan)?)?;
            ensure!(
                plan.identity()? == base_plan_sha256 && plan.scope == "comparator-qualification",
                "base plan differs or was already admitted as ASM"
            );
            let configuration = jope_pipeline::execution::Artifact {
                path: std::fs::canonicalize(configuration)?,
                sha256: configuration_sha256,
            };
            configuration.verify()?;
            let config: Configuration =
                serde_json::from_slice(&std::fs::read(&configuration.path)?)?;
            config.validate()?;
            let native = plan.runtimes.get("rust").context("native runtime absent")?;
            plan.runtimes.insert(
                super::optimize::CONFIG_RUNTIME.into(),
                Runtime {
                    program: native.program.clone(),
                    argument_prefix: vec![],
                    inputs: vec![configuration],
                    package_files: vec![],
                    environment: Default::default(),
                    version: super::optimize::SCHEMA.into(),
                    unavailable: None,
                },
            );
            plan.experiment.methods_sha256 =
                super::optimize_fit::digest(&(&plan.methods, &plan.runtimes))?;
            plan.experiment.checkpoints.insert(
                "asm.accepted-policy-state".into(),
                jope_policy::ACCEPTED_STATE_SHA256.into(),
            );
            plan.scope = "asm-panel".into();
            let id = plan.identity()?;
            super::worker::write_private(&output, &plan)?;
            Ok(
                json!({"ok":true,"plan":output,"plan_sha256":id,"scope":plan.scope,"fits":0,"execution_budget_started":false,
                "neural_slots":64,"exploration_slots":16,"generic_slots":80,"maximum_evaluations_per_arm":config.max_evaluations,"seconds_per_arm":config.seconds,
                "dependencies":config.dependency_inventory(),"classification_neural":"unavailable_until_qualified_experimental_checkpoint_producer","checkpoint_training_exposure":"unknown",
                "claim":"Algorithm Synthesis Model (ASM): learns to generate and optimize executable algorithms from data."}),
            )
        }
        OptimizeCommand::Run(args) => super::campaign_cli::run(args),
        OptimizeCommand::Report(args) => super::campaign_cli::report(args),
    }
}
#[cfg(not(all(feature = "asm", target_os = "linux")))]
fn run_enabled(_cli: OptimizeCli) -> anyhow::Result<Value> {
    Ok(
        json!({"ok":false,"unavailable":"ASM execution requires the private Linux build with the asm feature; no fits or context production were run"}),
    )
}
fn run(cli: OptimizeCli) -> anyhow::Result<Value> {
    if matches!(cli.command, OptimizeCommand::Capabilities) {
        return Ok(
            json!({"schema":"jope.optimize-capabilities/v1","compiled":cfg!(all(feature="asm",target_os="linux")),"fits":0,"qualified":false}),
        );
    }
    run_enabled(cli)
}
pub fn execute(cli: OptimizeCli) -> ExitCode {
    match run(cli) {
        Ok(value) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&value).expect("JSON report")
            );
            if value.get("ok").and_then(Value::as_bool) == Some(false) {
                ExitCode::FAILURE
            } else {
                ExitCode::SUCCESS
            }
        }
        Err(error) => {
            eprintln!("ASM operation failed: {error:#}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn capabilities_are_metadata_only_and_never_qualification() {
        let value = run(OptimizeCli {
            command: OptimizeCommand::Capabilities,
        })
        .unwrap();
        assert_eq!(value["schema"], "jope.optimize-capabilities/v1");
        assert_eq!(
            value["compiled"],
            cfg!(all(feature = "asm", target_os = "linux"))
        );
        assert_eq!(value["fits"], 0);
        assert_eq!(value["qualified"], false);
    }
}
