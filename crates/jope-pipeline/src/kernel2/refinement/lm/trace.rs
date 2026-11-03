//! Serializable records and validation for damped least-squares refinement.

use super::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The per-stage record of a damped least-squares refinement, kept with its training report.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RefinementTrace {
    /// The method id.
    pub method: String,
    /// One entry per stage refined, in stage order.
    pub stages: Vec<StageTrace>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum StageTrace {
    /// The coordinate search refined the stage, for the reason named.
    Coordinate {
        stage: usize,
        reason: String,
        evaluations: usize,
        rejected_trials: usize,
    },
    /// The start's whole-pipeline pass, or the training prefix before it, was refused: one refused
    /// evaluation, as the coordinate route charges, and the refinement failed.
    Failed { stage: usize },
    /// The guard refused the start's whole-pipeline pass: nothing was spent, and the refinement
    /// stopped.
    Stopped { stage: usize },
    /// The start's whole-pipeline pass, damped least squares on the training-prefix surrogate,
    /// then the whole-pipeline pass of a changed winner.
    Surrogate {
        stage: usize,
        surrogate: SurrogateTrace,
        /// The pipeline passes; a surrogate stage always records them.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pipeline: Option<PipelineCheck>,
    },
}

/// The surrogate fit's counts and iterations. Its charge is
/// `initial + trials + jacobian_cost * jacobians + rescore`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SurrogateTrace {
    pub status: String,
    pub changed: bool,
    pub sites: usize,
    pub jacobian_cost: usize,
    pub initial: usize,
    pub trials: usize,
    pub jacobians: usize,
    pub rescore: usize,
    pub rejected_trials: usize,
    pub solve_failures: usize,
    pub accepted: usize,
    /// The winner's surrogate re-score, as f64 bits.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rescored: Option<u64>,
    pub nonsmooth_rows: usize,
    pub saturated_rows: usize,
    pub iterations: Vec<IterationBits>,
}

/// One surrogate iteration with its values as f64 bits, so a replay compares them exactly.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct IterationBits {
    pub mu: u64,
    pub objective: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trial: Option<u64>,
    pub accepted: bool,
    pub solved: bool,
}

/// The whole-pipeline training passes of a surrogate stage: the start's, which runs before the
/// surrogate, and a changed winner's, which runs after it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PipelineCheck {
    /// 1, or 2 when the winner's pass ran.
    pub evaluations: usize,
    /// 1 when the winner's pass was refused.
    pub rejected_trials: usize,
    /// The start's training loss as f64 bits; a surrogate stage's start passed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start: Option<u64>,
    /// The winner's training loss as f64 bits; absent when its pass did not run or was refused.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub winner: Option<u64>,
    /// The winner replaced the start: its training loss is strictly lower.
    pub accepted: bool,
}

impl SurrogateTrace {
    pub(super) fn record(trace: &LmTrace) -> Self {
        Self {
            status: trace.status.as_str().into(),
            changed: trace.changed,
            sites: trace.sites,
            jacobian_cost: trace.jacobian_cost,
            initial: trace.initial,
            trials: trace.trials,
            jacobians: trace.jacobians,
            rescore: trace.rescore,
            rejected_trials: trace.rejected_trials,
            solve_failures: trace.solve_failures,
            accepted: trace.accepted,
            rescored: trace.rescored.map(f64::to_bits),
            nonsmooth_rows: trace.nonsmooth_rows,
            saturated_rows: trace.saturated_rows,
            iterations: trace
                .iterations
                .iter()
                .map(|i| IterationBits {
                    mu: i.mu.to_bits(),
                    objective: i.objective.to_bits(),
                    trial: i.trial.map(f64::to_bits),
                    accepted: i.accepted,
                    solved: i.solved,
                })
                .collect(),
        }
    }

    /// The evaluations and refusals charged, re-derived from the counts, after checking the counts
    /// against the iterations. `jacobian` is the selection's charge, `sites` or `one`.
    fn charged(&self, jacobian: &str) -> Result<(usize, usize)> {
        let status = LmStatus::ALL
            .into_iter()
            .find(|s| s.as_str() == self.status)
            .filter(|s| *s != LmStatus::Unsupported)
            .context("unknown surrogate status")?;
        let cost = match jacobian {
            "sites" => self.sites,
            "one" => 1,
            _ => anyhow::bail!("unknown refinement jacobian charge"),
        };
        ensure!(
            self.sites > 0 && self.jacobian_cost == cost,
            "surrogate jacobian charge disagrees"
        );
        ensure!(
            self.initial <= 1
                && self.rescore <= self.initial
                && (self.rescored.is_none() || self.rescore == 1)
                && (!self.changed || self.rescored.is_some()),
            "surrogate start or re-score disagrees"
        );
        let started = self.trials > 0 || self.jacobians > 0 || self.solve_failures > 0;
        ensure!(
            !started || self.initial == 1,
            "surrogate passes without an initial value"
        );
        if status == LmStatus::InitialRefused {
            ensure!(
                self.initial == 1 && !started && self.rescore == 0,
                "a refused start has passes after it"
            );
        }
        // The start's Jacobian comes before any iteration, and every accepted step is followed by
        // one, except a last step whose Jacobian the budget, the guard or an exact fit stopped; so
        // an accepted step with an iteration after it had its Jacobian. The re-score runs only
        // after an accepted step.
        let followed = self
            .iterations
            .iter()
            .rev()
            .skip(1)
            .filter(|i| i.accepted)
            .count();
        // The counts are untrusted here, so `jacobians - accepted` is taken checked: it is 0 or 1
        // exactly when `accepted <= jacobians <= accepted + 1`, with no overflow at any value.
        ensure!(
            self.jacobians
                .checked_sub(self.accepted)
                .is_some_and(|extra| extra <= 1)
                && (self.iterations.is_empty() || self.jacobians > followed)
                && (self.rescore == 0 || self.accepted > 0),
            "surrogate Jacobians or re-score disagree with its accepted steps"
        );
        ensure!(
            Some(self.iterations.len()) == self.trials.checked_add(self.solve_failures),
            "surrogate iterations disagree with trials and failed solves"
        );
        ensure!(
            self.iterations
                .iter()
                .all(|i| (i.solved || (i.trial.is_none() && !i.accepted))
                    && (!i.accepted || i.trial.is_some()))
                && self.iterations.iter().filter(|i| !i.solved).count() == self.solve_failures
                && self.iterations.iter().filter(|i| i.accepted).count() == self.accepted,
            "surrogate iterations disagree with their counts"
        );
        let refused = self
            .iterations
            .iter()
            .filter(|i| i.solved && i.trial.is_none())
            .count();
        ensure!(
            self.rejected_trials
                == refused
                    + usize::from(status == LmStatus::InitialRefused)
                    + usize::from(self.rescore == 1 && self.rescored.is_none()),
            "surrogate refusals disagree with its iterations"
        );
        let evaluations = cost
            .checked_mul(self.jacobians)
            .and_then(|j| j.checked_add(self.trials))
            .and_then(|e| e.checked_add(self.initial + self.rescore))
            .context("surrogate count overflow")?;
        Ok((evaluations, self.rejected_trials))
    }
}

impl PipelineCheck {
    fn charged(&self) -> Result<(usize, usize)> {
        let finite = |bits: Option<u64>| bits.is_none_or(|b| f64::from_bits(b).is_finite());
        ensure!(
            self.start.is_some()
                && finite(self.start)
                && finite(self.winner)
                && (1..=PIPELINE_PASSES).contains(&self.evaluations)
                && (self.winner.is_none() || self.evaluations == PIPELINE_PASSES),
            "pipeline passes disagree"
        );
        let refused = usize::from(self.evaluations == PIPELINE_PASSES && self.winner.is_none());
        let accepted = matches!((self.start, self.winner),
            (Some(start), Some(winner)) if f64::from_bits(winner) < f64::from_bits(start));
        ensure!(
            self.rejected_trials == refused && self.accepted == accepted,
            "pipeline decision does not re-derive"
        );
        Ok((self.evaluations, self.rejected_trials))
    }
}

impl StageTrace {
    fn charged(&self, jacobian: &str) -> Result<(usize, usize, usize)> {
        match self {
            Self::Coordinate {
                stage,
                reason,
                evaluations,
                rejected_trials,
            } => {
                ensure!(
                    FALLBACK_REASONS.contains(&reason.as_str()) && rejected_trials <= evaluations,
                    "invalid coordinate stage record"
                );
                Ok((*stage, *evaluations, *rejected_trials))
            }
            Self::Failed { stage } => Ok((*stage, 1, 1)),
            Self::Stopped { stage } => Ok((*stage, 0, 0)),
            Self::Surrogate {
                stage,
                surrogate,
                pipeline,
            } => {
                let pipeline = pipeline
                    .as_ref()
                    .context("surrogate stage without its pipeline passes")?;
                ensure!(
                    pipeline.evaluations < PIPELINE_PASSES || surrogate.changed,
                    "a winner's pipeline pass without a changed surrogate"
                );
                let (evaluations, rejected) = surrogate.charged(jacobian)?;
                let (more, refused) = pipeline.charged()?;
                Ok((
                    *stage,
                    evaluations
                        .checked_add(more)
                        .context("trace count overflow")?,
                    rejected
                        .checked_add(refused)
                        .context("trace count overflow")?,
                ))
            }
        }
    }
}

impl RefinementTrace {
    /// The evaluations and refusals the stages were charged, after checking every stage record.
    /// `jacobian` is the selection's charge, `sites` or `one`.
    pub(crate) fn charged(&self, jacobian: &str) -> Result<(usize, usize)> {
        ensure!(self.method == DAMPED_LS, "unknown refinement trace method");
        let (mut evaluations, mut rejected) = (0usize, 0usize);
        let mut last: Option<usize> = None;
        for stage in &self.stages {
            let (index, more, refused) = stage.charged(jacobian)?;
            ensure!(
                last.is_none_or(|previous| index > previous),
                "refinement trace stages out of order"
            );
            last = Some(index);
            evaluations = evaluations
                .checked_add(more)
                .context("trace count overflow")?;
            rejected = rejected
                .checked_add(refused)
                .context("trace count overflow")?;
        }
        Ok((evaluations, rejected))
    }

    /// [`Self::charged`], with every stage record bound to the `parent` program it refined under at
    /// most `trials_per_program` trials. The records are the parent's constant-bearing expression
    /// stages in order, as the route visits them. A `map` stage, and only a `map` stage, is an
    /// elementwise fallback. A `derive` stage is a fallback for an unsupported shape only outside
    /// the surrogate's site range, and one for a small allowance only within it and with fewer
    /// evaluations than its minimum. A surrogate stage, or a failed or stopped start, is a
    /// supported `derive` stage whose allowance could hold the minimum, and a surrogate stage
    /// reports the stage's literal sites and spent at most that allowance less the pipeline passes.
    /// A record that ends the refinement (a failed or stopped start, a stopped surrogate, or a
    /// changed winner whose pass did not run) is the last.
    pub(crate) fn check(
        &self,
        parent: &str,
        jacobian: &str,
        trials_per_program: usize,
    ) -> Result<(usize, usize)> {
        let charged = self.charged(jacobian)?;
        let charge = match jacobian {
            "sites" => LmJacobianCharge::Sites,
            "one" => LmJacobianCharge::One,
            _ => anyhow::bail!("unknown refinement jacobian charge"),
        };
        let pipe = parse_untrusted_pipeline(parent).context("refinement parent")?;
        let mut refined = Vec::new();
        for index in 0..pipe.stages.len() {
            if let Some(expression) = stage_expression(&pipe, index)? {
                if has_constant(expression.ast()) {
                    refined.push((index, expression.literal_sites()));
                }
            }
        }
        ensure!(
            self.stages.len() <= refined.len(),
            "refinement trace has more stages than the parent refines"
        );
        let mut spent = 0usize;
        for (position, (record, &(index, sites))) in self.stages.iter().zip(&refined).enumerate() {
            let (stage, evaluations, _) = record.charged(jacobian)?;
            ensure!(
                stage == index,
                "refinement trace stage is not the parent's next expression stage"
            );
            let op = pipe.stages[index].op.as_str();
            let last = position + 1 == self.stages.len();
            let supported = (1..=LM_MAX_SITES).contains(&sites);
            let minimum = stage_minimum(jacobian_cost(charge, sites));
            // The largest allowance the stage can have had.
            let room = trials_per_program
                .checked_sub(spent)
                .context("refinement trace overspends its allowance")?;
            let fitted = op == "derive" && supported && room >= minimum;
            let valid = match record {
                StageTrace::Coordinate { reason, .. } => match reason.as_str() {
                    ELEMENTWISE => op == "map",
                    UNSUPPORTED => op == "derive" && !supported,
                    SMALL_ALLOWANCE => op == "derive" && supported && evaluations < minimum,
                    _ => false,
                },
                StageTrace::Failed { .. } | StageTrace::Stopped { .. } => fitted && last,
                StageTrace::Surrogate {
                    surrogate,
                    pipeline,
                    ..
                } => {
                    let passes = pipeline.as_ref().map_or(0, |p| p.evaluations);
                    let ends = surrogate.status == LmStatus::Stopped.as_str()
                        || (surrogate.changed && passes < PIPELINE_PASSES);
                    fitted
                        && surrogate.sites == sites
                        && (evaluations - passes)
                            .checked_add(PIPELINE_PASSES)
                            .is_some_and(|needed| needed <= room)
                        && (last || !ends)
                }
            };
            ensure!(
                valid,
                "refinement trace stage {index} does not fit the parent program"
            );
            spent = spent
                .checked_add(evaluations)
                .context("trace count overflow")?;
        }
        Ok(charged)
    }
}
