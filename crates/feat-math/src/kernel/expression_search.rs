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

/// The most literal sites `MathExpr::optimize_constants_lm_guarded` fits: its parameters are the
/// sites and the affine pair of the linear-scaled residuals. A row expression with more sites, or
/// with none, is unsupported and spends nothing.
pub const LM_MAX_SITES: usize = 14;

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

// laneD:B20-rewrite-simplify begin
/// Why the exact rewrite saturation of `MathExpr::simplify_v2` stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SaturationStop {
    /// A round changed nothing: it created no e-node and merged no classes.
    Saturated,
    /// The round limit was reached.
    IterationLimit,
    /// The next application might have passed the e-node cap.
    NodeLimit,
    /// The input failed the structure check, so saturation never started.
    Malformed,
}

impl SaturationStop {
    /// A stable snake_case name for records.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Saturated => "saturated",
            Self::IterationLimit => "iteration_limit",
            Self::NodeLimit => "node_limit",
            Self::Malformed => "malformed",
        }
    }
}

/// The outcome of `MathExpr::simplify_v2`, with how its saturation went. The counters are those
/// of the saturation run, whether or not its tree was kept.
#[derive(Clone, Debug)]
pub struct ExpressionSimplification {
    /// The smaller equivalent tree, admitted by the profile; `None` when saturation found no
    /// smaller tree, or found one the profile refused.
    pub simplified: Option<MathExpr>,
    /// Why the profile refused the smaller tree, which is then discarded.
    pub refusal: Option<String>,
    pub stop: SaturationStop,
    /// Saturation rounds run.
    pub iterations: usize,
    /// E-nodes after saturation.
    pub enodes: usize,
    /// Rewrite applications.
    pub applications: usize,
    /// The most e-nodes held at any point, the input's own included.
    pub peak: usize,
}
// laneD:B20-rewrite-simplify end

// laneD:B19-minimax-linearize begin
/// The only degree `MathExpr::linearize_v2` fits: a subtree is replaced by a line in its one
/// variable, `a0 + a1 x`, or not at all.
pub const LINEARIZATION_DEGREE: usize = 1;

/// Why a discrete minimax fit stopped. `max_error` and `levelled`, the two ends of a fit's error
/// bracket, are residuals of floating-point evaluation, not certified bounds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MinimaxStop {
    /// The samples lie on the line to a few ulps: `max_error ≤ 4 ε max|f|`.
    ExactFit,
    /// The bracket closed to `1e-12 max_error`.
    Converged,
    /// The bracket closed to the rounding level `64 ε max|f|`, and the exchange could not move.
    NoiseLimited,
    /// The exchange returned its own reference with the bracket open: the best fit seen.
    Stalled,
    /// Too few alternating extrema, or a singular reference system: the best fit seen.
    Degenerate,
    /// The step cap was reached with the bracket open: the best fit seen.
    IterationLimit,
}

impl MinimaxStop {
    pub const ALL: [Self; 6] = [
        Self::ExactFit,
        Self::Converged,
        Self::NoiseLimited,
        Self::Stalled,
        Self::Degenerate,
        Self::IterationLimit,
    ];

    /// A stable snake_case name for records.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ExactFit => "exact_fit",
            Self::Converged => "converged",
            Self::NoiseLimited => "noise_limited",
            Self::Stalled => "stalled",
            Self::Degenerate => "degenerate",
            Self::IterationLimit => "iteration_limit",
        }
    }

    /// Whether the bracket closed, to the stop's own tolerance, so that `max_error` is the
    /// discrete minimax error up to rounding. A line is offered only from such a fit; the other
    /// stops return the best fit seen, whose error need not be the minimax one.
    pub fn closed(self) -> bool {
        matches!(self, Self::ExactFit | Self::Converged | Self::NoiseLimited)
    }
}

/// The degree-1 discrete minimax fit of one subtree on its sampled training rows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinearFit {
    pub stop: MinimaxStop,
    /// `[a0, a1]`: the line `a0 + a1 x` in the subtree's variable, as it is lowered.
    pub coefficients: [f64; 2],
    /// The smallest and largest sampled value of the variable: the training range. The error is
    /// controlled only at the samples.
    pub lo: f64,
    pub hi: f64,
    /// The lower end of the bracket, never above `max_error`.
    pub levelled: f64,
    /// The fit's largest error over the samples, evaluated in the fit's own basis.
    pub max_error: f64,
    /// The largest `|f|` over the samples, which sets the rounding level of the bound.
    pub max_abs: f64,
    /// The exchange step that produced the fit.
    pub iterations: usize,
    /// The exchange steps run.
    pub steps: usize,
}

/// A line is refused before it is built when a training row faults in the subtree or its
/// variable, or gives a value that is not finite.
pub const LINEARIZE_UNSAMPLED: &str = "a training row faults or is not finite in the subtree";
/// A caller's refusal of every subtree of an expression before any sample: the rows the
/// expression reads could not be built without a fault (a pipeline stage's training input). The
/// live `MathExpr::linearize_v2` never gives it, since its caller passes the rows; the recorded
/// path takes it for a subtree without a fit, for every subtree of the expression or none.
pub const LINEARIZE_INPUT_FAULTS: &str = "the stage's training input faults";
/// Prefix of the refusal when the fit refuses its samples; the fit's reason follows.
pub const LINEARIZE_FIT_REFUSED: &str = "the fit refused its samples";
/// Prefix of the refusal when the fit's bracket did not close; the stop's name follows.
pub const LINEARIZE_OPEN_BRACKET: &str = "the fit's bracket did not close";
/// A coefficient of the line is not finite.
pub const LINEARIZE_NON_FINITE: &str = "a coefficient of the line is not finite";
/// The line has as many nodes as the subtree, or more.
pub const LINEARIZE_NOT_SMALLER: &str = "the line is not smaller than the subtree";
/// The line would take the expression over its node limit.
pub const LINEARIZE_OVER_BUDGET: &str = "the line is over the expression's node budget";
/// The line alone exceeds the expression limits.
pub const LINEARIZE_OVER_LIMITS: &str = "the line exceeds the expression limits";
/// The line's value leaves the evaluator's range on a training sample.
pub const LINEARIZE_OUT_OF_RANGE: &str = "the line leaves the evaluator's range on a sample";
/// The line's own largest error over the training samples is above the minimax bound.
pub const LINEARIZE_MISSES_BOUND: &str = "the line misses the minimax bound on the samples";
/// Prefix of the refusal when the profile refuses the expression with the line spliced in; the
/// profile's reason follows.
pub const LINEARIZE_PROFILE_REFUSED: &str = "the profile refused the linearized expression";

/// One subtree `MathExpr::linearize_v2` considered, in the order it considers them.
#[derive(Clone, Debug, PartialEq)]
pub struct SubtreeLinearization {
    /// The subtree's position: child indices from the root.
    pub path: Vec<usize>,
    /// The subtree as it stands in the admitted expression.
    pub subtree: MathExpr,
    /// Its nodes, counted as the expression limits count them.
    pub nodes: usize,
    /// The degree-1 fit; `None` when the subtree was not sampled or the fit refused its samples.
    pub fit: Option<LinearFit>,
    /// The nodes of the lowered line, once it was built.
    pub line_nodes: Option<usize>,
    /// The lowered line's own largest error over the training samples, in its own evaluation
    /// order, once it was measured. It is what the minimax bound holds.
    pub lowered_error: Option<f64>,
    /// The whole expression with the line in place of the subtree, admitted by the profile;
    /// `None` when the line is refused.
    pub expression: Option<MathExpr>,
    /// Why no expression is offered.
    pub refusal: Option<String>,
}

/// What `MathExpr::linearize_v2` measured on the training rows for one subtree, as a record
/// keeps it. Everything else in a [`SubtreeLinearization`] re-derives from the expression and
/// this, through `MathExpr::linearize_recorded_v2`.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearizationEvidence {
    pub fit: Option<LinearFit>,
    pub lowered_error: Option<f64>,
    /// Without a fit, why the subtree was not fitted; ignored with one.
    pub refusal: Option<String>,
}
// laneD:B19-minimax-linearize end
