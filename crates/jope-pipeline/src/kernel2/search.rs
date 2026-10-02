//! Bounded kernel2 generation search, ready for E6 CLI and v4 publication.
//! The legacy search entry points and their streams remain separate.

use anyhow::{ensure, Context, Result};
use feat_math::{evolve::FaultPolicy, kernel::profile::shipped_kernel2_profiles};
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

use super::{EvaluationBudget, Kernel2Evaluator, Kernel2Observation, RefinedChild};
use crate::{
    prepare::Cut,
    search::{SearchOptions, ELITES},
};

mod generation;
mod offspring;
#[cfg(test)]
mod tests;

/// Explicit training-refinement allowance; it shares the cut's total budget.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct Refinement {
    pub top_k: usize,
    pub trials_per_program: usize,
}

/// A resolved shipped profile and a caller-owned budget/cancellation contract.
#[derive(Clone, Debug)]
pub struct Options {
    pub search: SearchOptions,
    pub refinement: Refinement,
    evaluator: Kernel2Evaluator,
}

impl Options {
    pub fn new(
        search: SearchOptions,
        profile: &str,
        policy: FaultPolicy,
        refinement: Refinement,
    ) -> Result<Self> {
        let profile = shipped_kernel2_profiles()
            .into_iter()
            .find(|p| p.id == profile)
            .context("unknown shipped kernel2 profile")?;
        let evaluator = Kernel2Evaluator::new(profile)?.with_fault_policy(policy)?;
        let options = Self {
            search,
            refinement,
            evaluator,
        };
        options.check()?;
        Ok(options)
    }
    pub fn evaluator(&self) -> &Kernel2Evaluator {
        &self.evaluator
    }
    fn check(&self) -> Result<()> {
        self.search.budget.check()?;
        ensure!(
            self.search.threads == 1 && self.search.islands == 1,
            "kernel2 coordinator currently requires one thread and one island"
        );
        ensure!(
            self.refinement.top_k <= ELITES
                && self.refinement.trials_per_program <= self.search.budget.evaluations,
            "refinement allowance exceeds the cut's bounds"
        );
        Ok(())
    }
}

/// Search results retain every measurement and training trial for v4 accounting.
#[derive(Clone, Debug, Serialize)]
pub struct CutSearch {
    pub cut: usize,
    pub profile_id: &'static str,
    pub profile_sha256: String,
    pub fault_policy: PolicyRecord,
    pub rng_seed: u64,
    pub budget_evaluations: usize,
    pub budget_seconds: u64,
    pub refinement: Refinement,
    pub generations: usize,
    pub completed_evaluations: usize,
    pub candidate_measurements: usize,
    pub objective_trials: usize,
    pub rejected_trials: usize,
    pub successful_measurements: usize,
    pub stopped_by: &'static str,
    pub seconds: f64,
    /// All wrapper arm draws, including unsuccessful edits and duplicate retries.
    pub arm_counts: BTreeMap<&'static str, usize>,
    pub observations: Vec<Kernel2Observation>,
    pub refinements: Vec<RefinedChild>,
    pub best: Kernel2Observation,
}

/// Validated policy parameters needed to replay kernel2 ranking.
#[derive(Clone, Debug, PartialEq, Serialize, serde::Deserialize, schemars::JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PolicyRecord {
    RejectHard,
    Penalize {
        lambda_domain: f64,
        lambda_saturation: f64,
        tie_break: bool,
    },
}

impl PolicyRecord {
    fn from_validated(policy: FaultPolicy) -> Self {
        match policy {
            FaultPolicy::RejectHard => Self::RejectHard,
            FaultPolicy::Penalize {
                lambda_domain,
                lambda_saturation,
                tie_break,
            } => Self::Penalize {
                lambda_domain,
                lambda_saturation,
                tie_break,
            },
            FaultPolicy::Legacy => unreachable!("validated kernel2 policy"),
        }
    }
}

struct State {
    budget: EvaluationBudget,
    observations: Vec<Kernel2Observation>,
    refinements: Vec<RefinedChild>,
    seen: BTreeSet<String>,
    arms: BTreeMap<&'static str, usize>,
    generation: usize,
    barren: usize,
}

impl State {
    fn new(limit: usize) -> Self {
        Self {
            budget: EvaluationBudget::new(limit),
            observations: Vec::new(),
            refinements: Vec::new(),
            seen: BTreeSet::new(),
            arms: BTreeMap::new(),
            generation: 0,
            barren: 0,
        }
    }
    fn result(
        self,
        cut: &Cut,
        options: &Options,
        stopped_by: &'static str,
        seconds: f64,
    ) -> Result<CutSearch> {
        let ranked = options.evaluator.rank(&self.observations);
        let best = (*ranked
            .first()
            .context("no kernel2 candidate measured on this cut")?)
        .clone();
        let objective_trials = self
            .refinements
            .iter()
            .map(|r| r.refinement.evaluations)
            .sum::<usize>();
        ensure!(
            self.budget.used() == self.observations.len() + objective_trials,
            "measurement/trial budget accounting mismatch"
        );
        Ok(CutSearch {
            cut: cut.index,
            profile_id: options.evaluator.profile().id,
            profile_sha256: options.evaluator.profile().sha256(),
            rng_seed: options.search.rng_seed,
            fault_policy: PolicyRecord::from_validated(options.evaluator.fault_policy()),
            budget_evaluations: options.search.budget.evaluations,
            budget_seconds: options.search.budget.seconds,
            refinement: options.refinement,
            generations: self.generation + 1,
            completed_evaluations: self.budget.used(),
            candidate_measurements: self.observations.len(),
            objective_trials,
            rejected_trials: self
                .refinements
                .iter()
                .map(|r| r.refinement.rejected_trials)
                .sum(),
            successful_measurements: self
                .observations
                .iter()
                .filter(|o| o.observation.measured())
                .count(),
            stopped_by,
            seconds,
            arm_counts: self.arms,
            observations: self.observations,
            refinements: self.refinements,
            best,
        })
    }
}

/// Run a cut on the same evaluation stack as the shipping legacy search.
/// Each optimizer objective and each child measurement spend the same allowance.
/// Cancellation returns no result; an admitted operation is nonpreemptive.
pub fn run_cut(cut: &Cut, options: &Options) -> Result<CutSearch> {
    generation::check(cut, options)?;
    crate::parallel::on_evaluation_stack(|| generation::run(cut, options))?
}
