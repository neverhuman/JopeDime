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
