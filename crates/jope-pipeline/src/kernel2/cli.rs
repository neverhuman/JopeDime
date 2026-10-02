//! Explicit CLI admission for the single-cut profile-search route.

use anyhow::{ensure, Context, Result};
use clap::Args;
use feat_math::evolve::FaultPolicy;
use std::{path::PathBuf, process::ExitCode};

use super::{receipt, run, search::Refinement};
use crate::{cli::SearchArgs, emit::Language};

#[derive(Args, Clone, Debug)]
pub struct ProfileArgs {
    /// Search profile: legacy (default), kernel2-baseline, or kernel2-experimental.
    #[arg(long, default_value = "legacy")]
    pub search_profile: String,
    /// Fault handling: legacy, reject-hard, or penalize. Kernel2 defaults to reject-hard.
    #[arg(long)]
    pub fault_policy: Option<String>,
    /// Training refinement parents per generation (0..=4); required for kernel2.
    #[arg(long)]
    pub refine_top_k: Option<usize>,
    /// Objective trials per parent, within the total evaluation budget; required for kernel2.
    #[arg(long)]
    pub refine_trials: Option<usize>,
    /// Explicit nonnegative Domain penalty; required with penalize.
    #[arg(long)]
    pub lambda_domain: Option<f64>,
    /// Explicit nonnegative Saturation penalty; required with penalize.
    #[arg(long)]
    pub lambda_saturation: Option<f64>,
    /// Explicit true/false fault tie-break; required with penalize.
    #[arg(long, action = clap::ArgAction::Set)]
    pub fault_tie_break: Option<bool>,
}

impl ProfileArgs {
    pub fn options(
        &self,
        search: &SearchArgs,
        output: PathBuf,
        signals: bool,
    ) -> Result<Option<run::Options>> {
        let penalty = self.lambda_domain.is_some()
            || self.lambda_saturation.is_some()
            || self.fault_tie_break.is_some();
        if self.search_profile == "legacy" {
            ensure!(
                self.fault_policy.as_deref().is_none_or(|p| p == "legacy")
                    && !penalty
                    && self.refine_top_k.is_none()
                    && self.refine_trials.is_none(),
                "legacy search refuses kernel2 fault/refinement options"
            );
            return Ok(None);
        }
        ensure!(
            ["kernel2-baseline", "kernel2-experimental"].contains(&self.search_profile.as_str()),
            "unknown --search-profile"
        );
        let policy = match self.fault_policy.as_deref().unwrap_or("reject-hard") {
            "reject-hard" => {
                ensure!(
                    !penalty,
                    "penalty parameters require --fault-policy penalize"
                );
                FaultPolicy::RejectHard
            }
            "penalize" => FaultPolicy::Penalize {
                lambda_domain: self
                    .lambda_domain
                    .context("penalize requires --lambda-domain")?,
                lambda_saturation: self
                    .lambda_saturation
                    .context("penalize requires --lambda-saturation")?,
                tie_break: self
                    .fault_tie_break
                    .context("penalize requires --fault-tie-break true|false")?,
            },
            _ => anyhow::bail!("kernel2 --fault-policy must be reject-hard or penalize"),
        };
        let refinement = Refinement {
            top_k: self
                .refine_top_k
                .context("kernel2 requires an explicit --refine-top-k (0..=4)")?,
            trials_per_program: self
                .refine_trials
                .context("kernel2 requires an explicit --refine-trials")?,
        };
        let mut run = search.options(output, signals)?;
        if search.emit.is_none() {
            run.languages = Language::parse_list("rust,python,cpp17,cpp20,js,cpp17-fitted")?;
        }
        Ok(Some(run::Options::new(
            run,
            &self.search_profile,
            policy,
            refinement,
        )?))
    }
}

pub(crate) fn print_run(options: &run::Options, json: bool) -> Result<ExitCode> {
    let receipt = run::run(options)?;
    if json {
        println!("{}", serde_json::to_string(&receipt)?);
    } else {
        println!("Pipeline ran {} with {}: {} measurements and {} training trials; selected {}; receipt in {}.",
            receipt.dataset.id, receipt.search.profile_id, receipt.search.candidate_measurements,
            receipt.search.objective_trials, receipt.search.best.observation.genome, options.run.output.display());
    }
    Ok(ExitCode::SUCCESS)
}

pub(crate) fn verify(path: &std::path::Path, evidence: bool, json: bool) -> Result<ExitCode> {
    let result = if evidence {
        super::publication::verify_run(path)
    } else {
        receipt::read(path)
    };
    match result {
        Ok(receipt) => {
            if json {
                println!("{}", serde_json::to_string(&receipt)?);
            } else {
                println!(
                    "ok {} ({}, profile {})",
                    path.display(),
                    receipt.schema,
                    receipt.search.profile_id
                );
            }
            Ok(ExitCode::SUCCESS)
        }
        Err(error) => {
            if json {
                println!(
                    "{}",
                    serde_json::json!({"schema":receipt::SCHEMA_ID,"ok":false,"error":format!("{error:#}")})
                );
            } else {
                eprintln!("invalid receipt: {error:#}");
            }
            Ok(ExitCode::FAILURE)
        }
    }
}
