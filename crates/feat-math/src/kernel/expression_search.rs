//! Public outcomes and training objectives for the separate version-two search.
pub use super::profile::ExpressionArm;
use crate::math_expr::MathExpr;

#[derive(Clone, Debug)]
pub struct ExpressionMutation {
    pub expression: MathExpr,
    pub arm: ExpressionArm,
    pub changed: bool,
}

/// The callback receives the candidate and training rows/targets only. The
/// optimizer checks the candidate's traced faults before invoking it.
pub type TrainingObjective<'a> =
    dyn FnMut(&MathExpr, &[Vec<f64>], &[f64]) -> anyhow::Result<f64> + 'a;

#[derive(Default)]
pub enum ConstantObjective<'a> {
    /// Fit a + b*f on training rows and minimize its residual MSE. This requires
    /// a row expression; elementwise expressions use a context-aware callback.
    #[default]
    LinearScaled,
    Callback(&'a mut TrainingObjective<'a>),
}

#[derive(Clone, Debug)]
pub struct ConstantRefinement {
    pub expression: MathExpr,
    /// Actual candidate evaluations, including the initial value and rejected trials.
    pub evaluations: usize,
    pub rejected_trials: usize,
    pub score: f64,
    /// [intercept, slope] for LinearScaled; absent for a caller objective.
    pub linear_scale: Option<[f64; 2]>,
}

/// E6 may stop a refinement before any initial or subsequent trial starts.
/// The caller owns the deadline/cancellation policy; a false predicate starts
/// no traced expression evaluation or objective and spends no trial.
pub struct GuardedRefinementBudget<'a> {
    pub max_evaluations: usize,
    pub may_start: &'a mut dyn FnMut() -> bool,
}

/// A completed or interrupted refinement, with actual attempted-trial counts.
/// A refused initial trial has no score and costs exactly one evaluation.
/// Invalid configuration instead returns an error before consulting the guard.
#[derive(Clone, Debug)]
pub struct GuardedConstantRefinement {
    pub refinement: Option<ConstantRefinement>,
    pub stopped: bool,
    pub failure: Option<String>,
}

impl GuardedConstantRefinement {
    pub fn evaluations(&self) -> usize {
        self.refinement
            .as_ref()
            .map_or(usize::from(self.failure.is_some()), |r| r.evaluations)
    }

    pub fn rejected_trials(&self) -> usize {
        self.refinement
            .as_ref()
            .map_or(usize::from(self.failure.is_some()), |r| r.rejected_trials)
    }
}
