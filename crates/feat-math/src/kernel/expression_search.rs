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

// laneD:B17-lm-refinement begin
/// What one Jacobian of the damped least-squares constant refinement is charged.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LmJacobianCharge {
    /// One evaluation per literal site, as a forward-difference Jacobian would cost, so an
    /// equal-budget comparison with the coordinate search stays fair.
    #[default]
    Sites,
    /// One evaluation per Jacobian.
    One,
}

/// The caller-set options of `MathExpr::optimize_constants_lm_guarded`. Every other setting of
/// the optimizer keeps its default.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LmOptions {
    /// The initial damping relative to the largest scaled curvature; finite and positive.
    pub tau: f64,
    pub jacobian: LmJacobianCharge,
}

impl Default for LmOptions {
    fn default() -> Self {
        Self {
            tau: 1e-3,
            jacobian: LmJacobianCharge::Sites,
        }
    }
}

/// Why a damped least-squares refinement ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LmStatus {
    /// The objective reached exactly zero.
    ExactFit,
    /// Every scaled gradient entry fell under the tolerance.
    Gradient,
    /// The step fell under the step tolerance.
    Step,
    IterationLimit,
    BudgetExhausted,
    /// A trial that was not accepted, or a failed solve, raised the damping past its bound.
    DampingLimit,
    /// A Jacobian or normal-equation entry was not finite.
    NonFiniteJacobian,
    /// The caller's guard refused a pass.
    Stopped,
    /// An elementwise scope, no literal site, or too many: nothing was spent.
    Unsupported,
    /// The start had a Domain or Hard fault, or a nonfinite objective, on a training row.
    InitialRefused,
}

impl LmStatus {
    pub const ALL: [Self; 10] = [
        Self::ExactFit,
        Self::Gradient,
        Self::Step,
        Self::IterationLimit,
        Self::BudgetExhausted,
        Self::DampingLimit,
        Self::NonFiniteJacobian,
        Self::Stopped,
        Self::Unsupported,
        Self::InitialRefused,
    ];

    /// A stable snake_case name for records.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ExactFit => "exact_fit",
            Self::Gradient => "gradient",
            Self::Step => "step",
            Self::IterationLimit => "iteration_limit",
            Self::BudgetExhausted => "budget_exhausted",
            Self::DampingLimit => "damping_limit",
            Self::NonFiniteJacobian => "non_finite_jacobian",
            Self::Stopped => "stopped",
            Self::Unsupported => "unsupported",
            Self::InitialRefused => "initial_refused",
        }
    }
}

/// One pass of the damped least-squares loop: a solved trial or a failed solve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LmIteration {
    pub mu: f64,
    /// `½ Σ r²` at the current point.
    pub objective: f64,
    /// The trial's objective; `None` for a refused trial or a failed solve.
    pub trial: Option<f64>,
    pub accepted: bool,
    pub solved: bool,
}

/// The deterministic record of one damped least-squares refinement. Its charged evaluations are
/// `initial + trials + jacobian_cost * jacobians + rescore` ([`LmTrace::evaluations`]).
#[derive(Clone, Debug, PartialEq)]
pub struct LmTrace {
    pub status: LmStatus,
    /// The winner's native re-score was strictly better than the start's, so it replaced it.
    pub changed: bool,
    /// Literal sites refined; the affine pair of the linear-scaled residuals is not counted.
    pub sites: usize,
    /// Evaluations one Jacobian is charged.
    pub jacobian_cost: usize,
    /// The start's native pass: 0 or 1.
    pub initial: usize,
    pub trials: usize,
    pub jacobians: usize,
    /// The winner's native re-score pass: 0 or 1.
    pub rescore: usize,
    /// Charged refusals: refused trials, a refused start and a refused re-score.
    pub rejected_trials: usize,
    pub solve_failures: usize,
    pub accepted: usize,
    /// The winner's native re-score, when it ran and was admitted.
    pub rescored: Option<f64>,
    /// Row evaluations over every Jacobian whose tangent was nonsmooth.
    pub nonsmooth_rows: usize,
    /// Row evaluations over every Jacobian that raised a Saturation fault.
    pub saturated_rows: usize,
    pub iterations: Vec<LmIteration>,
}

impl LmTrace {
    /// The evaluations this refinement was charged.
    pub fn evaluations(&self) -> usize {
        self.initial + self.trials + self.jacobian_cost * self.jacobians + self.rescore
    }
}

/// A guarded damped least-squares refinement in the coordinate search's terms, with its trace.
#[derive(Clone, Debug)]
pub struct GuardedLmRefinement {
    /// The winner, or the start, with the evaluations charged. A refused start is a failure that
    /// cost one evaluation; an unsupported shape spends nothing and reports no refinement.
    pub outcome: GuardedConstantRefinement,
    /// The start as admitted, for a caller that refuses the winner on its own check.
    pub start: MathExpr,
    pub trace: LmTrace,
}
// laneD:B17-lm-refinement end
