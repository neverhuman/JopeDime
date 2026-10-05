//! Explicit CLI admission for the single-cut profile-search route.

use anyhow::{ensure, Context, Result};
use clap::Args;
use feat_math::evolve::FaultPolicy;
use std::{path::PathBuf, process::ExitCode};

use super::{
    messages, receipt, run, search::Refinement, ArchivePolicy, EngineSelection, RefineMethod,
};
use crate::{cli::SearchArgs, emit::Language};
use feat_math::kernel::expression_search::LmOptions;

#[derive(Args, Clone, Debug)]
pub struct ProfileArgs {
    /// Search profile: legacy (default), kernel2-baseline, or kernel2-experimental.
    #[arg(long, default_value = messages::DEFAULT_SEARCH_PROFILE)]
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
    /// Constant refinement for kernel2: coordinate (default) or lm, a damped least-squares fit of
    /// a derive stage's constants on its training rows, kept only when the whole pipeline's
    /// training loss strictly falls. lm accepts --refine-top-k 1 or more and --refine-trials 3 or
    /// more, but fits a derive stage only when the trials left for it reach its literal sites plus
    /// five; a map stage, or a derive stage with fewer trials left, uses the coordinate search.
    #[arg(long)]
    pub refine_method: Option<String>,
    /// Publish the loss and cost front of the run's measured programs as
    /// search/cut-0/archive.json (kernel2 only). It is evidence only: ranking, parents and the
    /// selected program never read it, so every measurement and refinement is unchanged.
    #[arg(long)]
    pub pareto_archive: bool,
    /// After each search round, simplify the expressions of the --refine-top-k best measured
    /// programs with exact rewrites (kernel2 only; needs --refine-top-k 1 or more). Each smaller
    /// program the profile admits is measured as a new program, one evaluation each; the rewrites
    /// keep every value bit for bit, so it scores exactly as its original. A smaller expression
    /// the profile refuses is discarded. Every proposal is written to search/cut-0/moves.json.
    #[arg(long)]
    pub rewrite_simplify: bool,
    /// After each search round, offer a bounded minimax linearization, degree 1 only, of the
    /// one-variable subtrees in the derive stages of the --refine-top-k best measured programs
    /// (kernel2 only; needs --refine-top-k 1 or more). Each subtree near the root of its
    /// expression that reads one variable is sampled on the stage's training input and fitted by
    /// its discrete minimax line. The line replaces it only when the fit's bracket closed, the
    /// line is smaller, its own error on those rows meets the fit's minimax bound, and the profile
    /// admits the result. Each such program is measured as a new program, one evaluation each,
    /// and ranks by its measured score, never by the line's error. Every proposal and refusal is
    /// written to search/cut-0/moves.json.
    #[arg(long)]
    pub linearize_subtrees: bool,
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
        if self.search_profile == messages::DEFAULT_SEARCH_PROFILE {
            ensure!(
                self.fault_policy
                    .as_deref()
                    .is_none_or(|p| p == messages::DEFAULT_SEARCH_PROFILE)
                    && !penalty
                    && self.refine_top_k.is_none()
                    && self.refine_trials.is_none()
                    && self.refine_method.is_none()
                    && !self.pareto_archive
                    && !self.rewrite_simplify
                    && !self.linearize_subtrees,
                messages::PROFILE_OPTIONS_REFUSAL
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
        let engine = self.engine()?;
        let mut run = search.options(output, signals)?;
        if search.emit.is_none() {
            run.languages = Language::parse_list("rust,python,cpp17,cpp20,js,cpp17-fitted")?;
        }
        let options = run::Options::new(run, &self.search_profile, policy, refinement)?;
        // The default selection keeps the existing single preflight.
        if engine == EngineSelection::default() {
            return Ok(Some(options));
        }
        Ok(Some(options.with_engine(engine)?))
    }

    // laneD:B17-lm-refinement begin
    /// The opt-in engine methods these flags select; none by default.
    fn engine(&self) -> Result<EngineSelection> {
        let refine = match self.refine_method.as_deref().unwrap_or("coordinate") {
            "coordinate" => RefineMethod::Coordinate,
            "lm" => RefineMethod::Lm(LmOptions::default()),
            _ => anyhow::bail!("kernel2 --refine-method must be coordinate or lm"),
        };
        Ok(EngineSelection {
            refine,
            // laneD:B18-pareto-archive begin
            archive: self.pareto_archive.then_some(ArchivePolicy::ArchiveOnly),
            // laneD:B18-pareto-archive end
            // laneD:B20-rewrite-simplify begin
            simplify: self.rewrite_simplify,
            // laneD:B20-rewrite-simplify end
            // laneD:B19-minimax-linearize begin
            linearize: self.linearize_subtrees,
            // laneD:B19-minimax-linearize end
        })
    }
    // laneD:B17-lm-refinement end
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
