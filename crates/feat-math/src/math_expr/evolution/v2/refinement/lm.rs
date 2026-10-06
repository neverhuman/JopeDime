//! Bounded Levenberg–Marquardt constant fitting with a dual-number Jacobian.
//!
//! Opt-in: the one public entry is [`MathExpr::optimize_constants_lm_guarded`], which admits by
//! the version-two search contract and fits the linear-scaled residuals. The raw residuals, the
//! structural admission and Levenberg's identity scaling serve this module's tests only.
//!
//! The optimizer is Algorithm 3.16 of Madsen, Nielsen and Tingleff, *Methods for Non-Linear Least
//! Squares Problems* (2nd ed., IMM DTU, 2004): Levenberg–Marquardt with Nielsen's damping update.
//! It runs on the Jacobi-scaled problem, whose scale is MINPACK's running maximum of `diag(JᵀJ)`
//! with a relative floor. Its damped steps are invariant to rescaling a parameter while the floor
//! raises no parameter's scale, before or after the rescaling. Where it raises one, it bounds that
//! parameter's raw step instead, and through the coupling of the parameters a rescaling can change
//! every step, including the rescaled parameter's own. The step stopping test (MNT 3.15b) is taken
//! on the raw parameters and is not invariant either.
//!
//! The damping is bounded to `[mu_min, mu_max]`: Nielsen's update after an accepted step is
//! clamped to both bounds, and a trial that is not accepted, or a failed solve, that would raise
//! it past `mu_max` stops the run.
//! A raise starts from at least the smallest normal number, so a zero damping still rises. The
//! iterations are bounded by `max_iterations` and the work by the caller's evaluation budget. The
//! damped system is solved by Cholesky with explicit pivot and reciprocal-condition checks. A
//! failed solve raises the damping and charges no evaluation.
//!
//! In the expression adapter every residual comes from the native protected evaluator
//! (`CompiledMathExpr::eval_row_traced`), and a trial with any Domain or Hard fault is refused, as
//! in the coordinate search. The Jacobian comes from forward-mode dual numbers ([`dual`]), whose
//! values must equal the native ones bit for bit. The winner replaces the start only when a fresh
//! native re-score is strictly better than the start's. Every native pass is charged: the first
//! Jacobian reuses the start's admitted values, and the re-score's fresh admission is its one pass.

// B17 core.

use super::*;
use crate::kernel::expression_search::{
    GuardedConstantRefinement, GuardedLmRefinement, GuardedRefinementBudget, LmIteration,
    LmJacobianCharge, LmOptions, LmStatus, LmTrace, LM_MAX_SITES,
};

mod dual;
use dual::{Program, MAX_PARAMS};

type Matrix = [[f64; MAX_PARAMS]; MAX_PARAMS];
type Vector = [f64; MAX_PARAMS];

/// Native and dual passes per thread, so a test can hold the work done to the evaluations charged.
#[cfg(test)]
mod passes {
    use std::cell::Cell;

    thread_local! {
        static NATIVE: Cell<usize> = const { Cell::new(0) };
        static DUAL: Cell<usize> = const { Cell::new(0) };
    }

    pub(super) fn native() {
        NATIVE.with(|count| count.set(count.get() + 1));
    }

    pub(super) fn dual() {
        DUAL.with(|count| count.set(count.get() + 1));
    }

    /// The `(native, dual)` passes since the last call.
    pub(super) fn take() -> (usize, usize) {
        (
            NATIVE.with(|count| count.replace(0)),
            DUAL.with(|count| count.replace(0)),
        )
    }
}

/// How the damping is scaled per parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scaling {
    /// Levenberg's `D = I` (MNT's examples).
    #[cfg(test)]
    Identity,
    /// Marquardt scaling by the running maximum of `diag(JᵀJ)` (MINPACK `mode = 1`), floored at
    /// `scale_floor` times its largest entry.
    Marquardt,
}

/// What one Jacobian costs against the evaluation budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum JacobianCost {
    /// One evaluation per expression site, as a forward-difference Jacobian would pay. Keeps an
    /// equal-budget comparison with the coordinate search fair.
    Sites,
    /// One evaluation.
    One,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Options {
    /// The initial damping relative to the largest scaled curvature (MNT 3.14): 1e-3 from a cold
    /// start, 1e-6 from a refined one.
    tau: f64,
    /// Stop when every scaled gradient entry is at most this times the residual scale.
    gradient_tolerance: f64,
    /// Stop when `‖h‖ ≤ step_tolerance (‖θ‖ + step_tolerance)` (MNT 3.15b), on the raw parameters,
    /// so this test depends on how a parameter is scaled even where the steps do not.
    step_tolerance: f64,
    mu_min: f64,
    mu_max: f64,
    max_iterations: usize,
    /// The smallest admitted 1-norm reciprocal condition of the damped system.
    rcond_min: f64,
    scale_floor: f64,
    scaling: Scaling,
    jacobian_cost: JacobianCost,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            tau: 1e-3,
            gradient_tolerance: 1e-10,
            step_tolerance: 1e-10,
            mu_min: 1e-9,
            mu_max: 1e10,
            max_iterations: 100,
            rcond_min: 1e-12,
            scale_floor: 1e-10,
            scaling: Scaling::Marquardt,
            jacobian_cost: JacobianCost::Sites,
        }
    }
}

/// Why a run ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Status {
    /// The objective reached exactly zero.
    ExactFit,
    /// Every scaled gradient entry fell below the tolerance.
    Gradient,
    /// The step fell below the step tolerance.
    Step,
    IterationLimit,
    BudgetExhausted,
    /// A trial that was not accepted (refused, or with no decrease) or a failed solve raised the
    /// damping past `mu_max`.
    DampingLimit,
    /// A Jacobian or normal-equation entry was not finite, or a dual value disagreed with the
    /// native one.
    NonFiniteJacobian,
    /// The caller's guard refused a start.
    Stopped,
    /// The adapter does not apply: an elementwise scope, no literal site, or too many.
    Unsupported,
    /// The start itself was refused (a Domain or Hard fault on a training row).
    InitialRefused,
}

/// One pass of the main loop, for the deterministic trace.
#[derive(Clone, Copy, Debug)]
struct Iteration {
    mu: f64,
    objective: f64,
    /// The trial's objective, or `None` for a refused trial or a failed solve.
    trial: Option<f64>,
    accepted: bool,
    solved: bool,
}

impl Iteration {
    #[cfg(test)]
    fn bits(&self) -> (u64, u64, Option<u64>, bool, bool) {
        (
            self.mu.to_bits(),
            self.objective.to_bits(),
            self.trial.map(f64::to_bits),
            self.accepted,
            self.solved,
        )
    }
}

/// A run of [`minimize`]: the best point, why it stopped, and what it spent.
#[derive(Clone, Debug)]
struct Outcome {
    theta: Vec<f64>,
    residuals: Vec<f64>,
    /// `½ Σ r²` at `theta`.
    objective: f64,
    status: Status,
    /// The initial evaluation, every trial and every Jacobian at its cost.
    evaluations: usize,
    trials: usize,
    rejected_trials: usize,
    jacobians: usize,
    solve_failures: usize,
    accepted: usize,
    trace: Vec<Iteration>,
}

impl Outcome {
    fn stop(mut self, status: Status) -> Self {
        self.status = status;
        self
    }
}

/// `JᵀJ` and `Jᵀr`, accumulated in row order.
#[derive(Clone, Copy, Debug)]
struct Normal {
    a: Matrix,
    g: Vector,
}

impl Normal {
    fn zero() -> Self {
        Self {
            a: [[0.0; MAX_PARAMS]; MAX_PARAMS],
            g: [0.0; MAX_PARAMS],
        }
    }

    /// Adds one Jacobian row and its residual to the upper triangle and the gradient.
    fn accumulate(&mut self, row: &Vector, residual: f64, width: usize) {
        for j in 0..width {
            self.g[j] += row[j] * residual;
            for k in j..width {
                self.a[j][k] += row[j] * row[k];
            }
        }
    }

    /// Mirrors the upper triangle, and checks every entry is finite.
    fn finish(mut self, width: usize) -> Option<Self> {
        for j in 0..width {
            for k in 0..j {
                self.a[j][k] = self.a[k][j];
            }
        }
        let finite = (0..width)
            .all(|j| self.g[j].is_finite() && self.a[j][..width].iter().all(|v| v.is_finite()));
        finite.then_some(self)
    }
}

/// A least-squares problem over `θ`.
trait Problem {
    /// The scale of the residuals, for the gradient test.
    fn scale(&self) -> f64;
    /// The residuals at `theta`, or `None` when the trial is refused.
    fn residuals(&mut self, theta: &[f64]) -> Option<Vec<f64>>;
    /// The normal equations at an accepted `theta` with its `residuals`, or `None` when an entry is
    /// not finite.
    fn normal_equations(&mut self, theta: &[f64], residuals: &[f64]) -> Option<Normal>;
    /// Moves an accepted point to one whose objective is no larger. The default keeps it.
    fn project(&mut self, _theta: &mut [f64], _residuals: &mut Vec<f64>) {}
}

/// What one run may spend.
#[derive(Clone, Copy, Debug)]
struct Limits {
    max_evaluations: usize,
    jacobian_cost: usize,
}

fn half_sum_squares(values: &[f64]) -> f64 {
    0.5 * values.iter().map(|v| v * v).sum::<f64>()
}

/// The Euclidean norm, scaled by the largest magnitude so that no square overflows: entries above
/// `sqrt(f64::MAX) ≈ 1.34e154` give their true norm, not infinity.
fn norm(values: &[f64]) -> f64 {
    let largest = values.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if largest == 0.0 || !largest.is_finite() {
        return largest;
    }
    largest
        * values
            .iter()
            .map(|v| (v / largest) * (v / largest))
            .sum::<f64>()
            .sqrt()
}

/// Levenberg–Marquardt from `theta`, whose `residuals` the caller has evaluated (one evaluation).
fn minimize(
    problem: &mut dyn Problem,
    theta: Vec<f64>,
    residuals: Vec<f64>,
    options: &Options,
    limits: Limits,
    may_start: &mut dyn FnMut() -> bool,
) -> Outcome {
    let p = theta.len();
    let objective = half_sum_squares(&residuals);
    let mut run = Outcome {
        theta,
        residuals,
        objective,
        status: Status::IterationLimit,
        evaluations: 1,
        trials: 0,
        rejected_trials: 0,
        jacobians: 0,
        solve_failures: 0,
        accepted: 0,
        trace: Vec::new(),
    };
    if run.objective == 0.0 {
        return run.stop(Status::ExactFit);
    }
    let tolerance = options.gradient_tolerance * problem.scale();
    let mut normal = match normal_at(problem, &mut run, limits, may_start) {
        Ok(normal) => normal,
        Err(status) => return run.stop(status),
    };
    let mut diagonal = [1.0; MAX_PARAMS];
    if options.scaling == Scaling::Marquardt {
        for (j, d) in diagonal.iter_mut().enumerate().take(p) {
            let curvature = normal.a[j][j];
            *d = if curvature > 0.0 { curvature } else { 1.0 };
        }
        floor(&mut diagonal, p, options.scale_floor);
    }
    let mut system = Scaled::new(&normal, &diagonal, p);
    if system.gradient_max() <= tolerance {
        return run.stop(Status::Gradient);
    }
    let curvature = (0..p).map(|j| system.a[j][j]).fold(0.0, f64::max);
    let mut mu = (options.tau * curvature).clamp(options.mu_min, options.mu_max);
    let mut nu = 2.0f64;
    for _ in 0..options.max_iterations {
        let Some(h_hat) = solve_damped(&system.a, &system.g, mu, p, options.rcond_min) else {
            run.solve_failures += 1;
            run.trace.push(Iteration {
                mu,
                objective: run.objective,
                trial: None,
                accepted: false,
                solved: false,
            });
            // From at least the smallest normal number, so a zero damping (mu_min = 0) rises and
            // 0 * inf never makes a NaN damping once nu overflows.
            mu = (mu * nu).max(f64::MIN_POSITIVE);
            nu *= 2.0;
            if mu > options.mu_max {
                return run.stop(Status::DampingLimit);
            }
            continue;
        };
        let step: Vec<f64> = (0..p).map(|j| system.s[j] * h_hat[j]).collect();
        if norm(&step) <= options.step_tolerance * (norm(&run.theta) + options.step_tolerance) {
            return run.stop(Status::Step);
        }
        if run.evaluations + 1 > limits.max_evaluations {
            return run.stop(Status::BudgetExhausted);
        }
        if !may_start() {
            return run.stop(Status::Stopped);
        }
        let candidate: Vec<f64> = run.theta.iter().zip(&step).map(|(t, h)| t + h).collect();
        run.evaluations += 1;
        run.trials += 1;
        // L(0) - L(h) = ½ ĥᵀ(μĥ - ĝ) on the scaled problem (MNT p. 25), positive for ĥ ≠ 0.
        let predicted = 0.5
            * (0..p)
                .map(|j| h_hat[j] * (mu * h_hat[j] - system.g[j]))
                .sum::<f64>();
        let trial = problem.residuals(&candidate);
        let trial_objective = trial.as_deref().map(half_sum_squares);
        if trial.is_none() {
            run.rejected_trials += 1;
        }
        let rho = trial_objective.map_or(f64::NEG_INFINITY, |f| (run.objective - f) / predicted);
        let accepted = rho > 0.0;
        run.trace.push(Iteration {
            mu,
            objective: run.objective,
            trial: trial_objective,
            accepted,
            solved: true,
        });
        let Some(trial) = trial.filter(|_| accepted) else {
            // From at least the smallest normal number, so a zero damping (mu_min = 0) rises and
            // 0 * inf never makes a NaN damping once nu overflows.
            mu = (mu * nu).max(f64::MIN_POSITIVE);
            nu *= 2.0;
            if mu > options.mu_max {
                return run.stop(Status::DampingLimit);
            }
            continue;
        };
        run.theta = candidate;
        run.residuals = trial;
        problem.project(&mut run.theta, &mut run.residuals);
        run.objective = half_sum_squares(&run.residuals);
        run.accepted += 1;
        if run.objective == 0.0 {
            return run.stop(Status::ExactFit);
        }
        normal = match normal_at(problem, &mut run, limits, may_start) {
            Ok(normal) => normal,
            Err(status) => return run.stop(status),
        };
        if options.scaling == Scaling::Marquardt {
            for (j, d) in diagonal.iter_mut().enumerate().take(p) {
                *d = d.max(normal.a[j][j]);
            }
            floor(&mut diagonal, p, options.scale_floor);
        }
        system = Scaled::new(&normal, &diagonal, p);
        if system.gradient_max() <= tolerance {
            return run.stop(Status::Gradient);
        }
        // Nielsen's factor exceeds 1 for a gain ratio below 1/2, so the update is clamped to both
        // bounds, not just the floor.
        mu = (mu * (1.0f64 / 3.0).max(1.0 - (2.0 * rho - 1.0).powi(3)))
            .clamp(options.mu_min, options.mu_max);
        nu = 2.0;
    }
    run.stop(Status::IterationLimit)
}

/// Charges and computes the normal equations at the run's point, or names why it cannot.
fn normal_at(
    problem: &mut dyn Problem,
    run: &mut Outcome,
    limits: Limits,
    may_start: &mut dyn FnMut() -> bool,
) -> Result<Normal, Status> {
    if run.evaluations + limits.jacobian_cost > limits.max_evaluations {
        return Err(Status::BudgetExhausted);
    }
    if !may_start() {
        return Err(Status::Stopped);
    }
    run.evaluations += limits.jacobian_cost;
    run.jacobians += 1;
    problem
        .normal_equations(&run.theta, &run.residuals)
        .ok_or(Status::NonFiniteJacobian)
}

/// `d_j ≥ eta · max_i d_i`, which bounds the raw step of a parameter with tiny curvature.
fn floor(diagonal: &mut Vector, p: usize, eta: f64) {
    let largest = diagonal[..p].iter().copied().fold(0.0, f64::max);
    for d in &mut diagonal[..p] {
        *d = d.max(eta * largest);
    }
}

/// The Jacobi-scaled system: `Â = S A S`, `ĝ = S g` with `S = diag(1/√d)`.
struct Scaled {
    a: Matrix,
    g: Vector,
    s: Vector,
    p: usize,
}

impl Scaled {
    fn new(normal: &Normal, diagonal: &Vector, p: usize) -> Self {
        let mut system = Self {
            a: [[0.0; MAX_PARAMS]; MAX_PARAMS],
            g: [0.0; MAX_PARAMS],
            s: [0.0; MAX_PARAMS],
            p,
        };
        for (s, d) in system.s.iter_mut().zip(diagonal).take(p) {
            *s = 1.0 / d.sqrt();
        }
        for j in 0..p {
            system.g[j] = system.s[j] * normal.g[j];
            for k in 0..p {
                system.a[j][k] = system.s[j] * normal.a[j][k] * system.s[k];
            }
        }
        system
    }

    fn gradient_max(&self) -> f64 {
        self.g[..self.p].iter().map(|g| g.abs()).fold(0.0, f64::max)
    }
}

/// `(Â + μI) ĥ = -ĝ` by Cholesky (MNT Algorithm A.4): every pivot finite and positive, the exact
/// 1-norm reciprocal condition at least `rcond_min`, and a finite solution; otherwise `None`.
fn solve_damped(a: &Matrix, g: &Vector, mu: f64, p: usize, rcond_min: f64) -> Option<Vector> {
    let mut m = *a;
    for (j, row) in m.iter_mut().enumerate().take(p) {
        row[j] += mu;
    }
    let mut l = [[0.0; MAX_PARAMS]; MAX_PARAMS];
    for k in 0..p {
        let mut pivot = m[k][k];
        for v in &l[k][..k] {
            pivot -= v * v;
        }
        if !(pivot.is_finite() && pivot > 0.0) {
            return None;
        }
        let diagonal = pivot.sqrt();
        l[k][k] = diagonal;
        for j in (k + 1)..p {
            let mut s = m[j][k];
            for (a, b) in l[j][..k].iter().zip(&l[k][..k]) {
                s -= a * b;
            }
            l[j][k] = s / diagonal;
        }
    }
    if rcond(&m, &l, p) < rcond_min {
        return None;
    }
    let mut rhs = [0.0; MAX_PARAMS];
    for (r, v) in rhs.iter_mut().zip(g).take(p) {
        *r = -v;
    }
    let h = cholesky_solve_lower(&l, p, &rhs);
    h[..p].iter().all(|v| v.is_finite()).then_some(h)
}

/// `L Lᵀ x = b`.
fn cholesky_solve_lower(l: &Matrix, p: usize, b: &Vector) -> Vector {
    let mut y = [0.0; MAX_PARAMS];
    for i in 0..p {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i][k] * y[k];
        }
        y[i] = s / l[i][i];
    }
    let mut x = [0.0; MAX_PARAMS];
    for i in (0..p).rev() {
        let mut s = y[i];
        for k in (i + 1)..p {
            s -= l[k][i] * x[k];
        }
        x[i] = s / l[i][i];
    }
    x
}

/// `1 / (‖M‖₁ ‖M⁻¹‖₁)`, with `M⁻¹` formed column by column from the factor (p ≤ 16).
fn rcond(m: &Matrix, l: &Matrix, p: usize) -> f64 {
    let norm = (0..p)
        .map(|j| (0..p).map(|i| m[i][j].abs()).sum::<f64>())
        .fold(0.0, f64::max);
    let mut inverse = 0.0f64;
    for j in 0..p {
        let mut unit = [0.0; MAX_PARAMS];
        unit[j] = 1.0;
        let column = cholesky_solve_lower(l, p, &unit);
        let sum = column[..p].iter().map(|v| v.abs()).sum::<f64>();
        if !sum.is_finite() {
            return 0.0;
        }
        inverse = inverse.max(sum);
    }
    if norm > 0.0 && inverse > 0.0 {
        1.0 / (norm * inverse)
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------------------------
// The expression adapter.

/// What the residuals compare.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Objective {
    /// `f - y`.
    #[cfg(test)]
    Raw,
    /// `α + β f - y`. The affine pair are two more parameters, reset to their closed form after
    /// every accepted step; the native score is the coordinate search's `LinearScaled` MSE.
    LinearScaled,
}

/// How a trial expression is admitted.
enum Admission<'a> {
    /// The version-two search contract (`Search::finish`): pool, scope limits and printed form.
    Profile(Search<'a>),
    /// The structural caps (64 nodes, depth 16) and the printed form, as `optimize_constants_v1`
    /// admits. For fixtures outside every kernel2 pool.
    #[cfg(test)]
    Structure,
}

impl Admission<'_> {
    fn finish(&self, root: Expr) -> Result<MathExpr> {
        match self {
            Self::Profile(search) => search.finish(root),
            #[cfg(test)]
            Self::Structure => {
                let source = MathExpr::from_ast(normalize(root))?.to_source()?;
                MathExpr::parse(&source)
            }
        }
    }

    fn elementwise(&self) -> bool {
        matches!(self, Self::Profile(search) if search.scope.elementwise)
    }
}

/// The outcome of [`MathExpr::optimize_constants_lm`].
#[derive(Clone, Debug)]
struct LmRefinement {
    /// The LM winner when its native re-score is strictly better than the start's; otherwise the
    /// start, with the start's score. Evaluations count the initial value, every trial, every
    /// Jacobian at its cost, and the re-score.
    refinement: ConstantRefinement,
    changed: bool,
    status: Status,
    /// The LM run, when one started.
    run: Option<Outcome>,
    /// The winner's native re-score, when one ran.
    rescore: Option<f64>,
    /// Row evaluations, over every Jacobian, whose tangent was nonsmooth. A row counts once per
    /// Jacobian, so this counts evaluations, not distinct rows.
    nonsmooth_rows: usize,
    /// Row evaluations, over every Jacobian, that raised a Saturation fault; counted the same way.
    saturated_rows: usize,
}

/// The admitted expression at a parameter point, and its native values.
struct Trial {
    sites: Vec<u64>,
    expression: MathExpr,
    values: Vec<f64>,
}

/// An expression's residuals on training rows, with its dual Jacobian.
struct ExprProblem<'a, 'b> {
    admission: &'a Admission<'b>,
    start: &'a Expr,
    sites: &'a [Vec<usize>],
    rows: &'a [Vec<f64>],
    targets: &'a [f64],
    objective: Objective,
    rejected: FaultSet,
    scale: f64,
    last: Option<Trial>,
    nonsmooth_rows: usize,
    saturated_rows: usize,
}

impl ExprProblem<'_, '_> {
    /// The expression at `theta`'s site values as admitted, with its native values, or `None`
    /// when admission refuses it, moves a site, or any row raises a Domain or Hard fault.
    fn admit(&self, theta: &[f64]) -> Option<Trial> {
        let values = &theta[..self.sites.len()];
        let mut root = self.start.clone();
        for (site, value) in self.sites.iter().zip(values) {
            replace(&mut root, site, literal(*value));
        }
        let expression = self.admission.finish(root).ok()?;
        if literal_paths(&expression.root) != self.sites {
            return None;
        }
        let bits: Vec<u64> = values.iter().map(|v| v.to_bits()).collect();
        let admitted: Option<Vec<u64>> = self
            .sites
            .iter()
            .map(|site| literal_value(at(&expression.root, site)).map(f64::to_bits))
            .collect();
        if admitted.as_ref() != Some(&bits) {
            return None;
        }
        let values = native_values(&expression, self.rows, self.rejected)?;
        Some(Trial {
            sites: bits,
            expression,
            values,
        })
    }

    fn residuals_of(&self, theta: &[f64], values: &[f64]) -> Vec<f64> {
        let p = self.sites.len();
        match self.objective {
            #[cfg(test)]
            Objective::Raw => values
                .iter()
                .zip(self.targets)
                .map(|(f, y)| f - y)
                .collect(),
            Objective::LinearScaled => {
                let (alpha, beta) = (theta[p], theta[p + 1]);
                values
                    .iter()
                    .zip(self.targets)
                    .map(|(f, y)| alpha + beta * f - y)
                    .collect()
            }
        }
    }

    /// The trial at `theta`'s site values: the last one when the bits match.
    fn trial_at(&mut self, theta: &[f64]) -> Option<&Trial> {
        let bits: Vec<u64> = theta[..self.sites.len()]
            .iter()
            .map(|v| v.to_bits())
            .collect();
        if self.last.as_ref().is_none_or(|trial| trial.sites != bits) {
            self.last = Some(self.admit(theta)?);
        }
        self.last.as_ref()
    }
}

impl Problem for ExprProblem<'_, '_> {
    fn scale(&self) -> f64 {
        self.scale
    }

    fn residuals(&mut self, theta: &[f64]) -> Option<Vec<f64>> {
        let trial = self.admit(theta)?;
        let residuals = self.residuals_of(theta, &trial.values);
        if residuals.iter().any(|r| !r.is_finite()) {
            return None;
        }
        self.last = Some(trial);
        Some(residuals)
    }

    fn normal_equations(&mut self, theta: &[f64], residuals: &[f64]) -> Option<Normal> {
        let p = self.sites.len();
        let width = theta.len();
        let (objective, rejected, rows) = (self.objective, self.rejected, self.rows);
        let sites = self.sites;
        let trial = self.trial_at(theta)?;
        let program = Program::lower(&trial.expression.root, sites).ok()?;
        debug_assert_eq!(program.params(), p);
        #[cfg(test)]
        passes::dual();
        let mut normal = Normal::zero();
        let (mut nonsmooth, mut saturated) = (0usize, 0usize);
        for (i, row) in rows.iter().enumerate() {
            let dual = program.row(&theta[..p], row);
            // The dual values are the native ones by construction; anything else is a defect.
            if dual.value.to_bits() != trial.values[i].to_bits() || dual.faults.intersects(rejected)
            {
                return None;
            }
            nonsmooth += usize::from(dual.nonsmooth);
            saturated += usize::from(dual.faults.intersects(FaultSeverity::Saturation.classes()));
            let mut jacobian = [0.0; MAX_PARAMS];
            match objective {
                #[cfg(test)]
                Objective::Raw => jacobian[..p].copy_from_slice(&dual.tangent[..p]),
                Objective::LinearScaled => {
                    let beta = theta[p + 1];
                    for (out, tangent) in jacobian.iter_mut().zip(&dual.tangent).take(p) {
                        *out = beta * tangent;
                    }
                    jacobian[p] = 1.0;
                    jacobian[p + 1] = dual.value;
                }
            }
            normal.accumulate(&jacobian, residuals[i], width);
        }
        self.nonsmooth_rows += nonsmooth;
        self.saturated_rows += saturated;
        normal.finish(width)
    }

    /// LinearScaled: the affine pair's closed form at the accepted expression values, kept only
    /// when it lowers nothing it should not (no larger objective, every residual finite).
    fn project(&mut self, theta: &mut [f64], residuals: &mut Vec<f64>) {
        if self.objective != Objective::LinearScaled {
            return;
        }
        let p = self.sites.len();
        let Some(trial) = self.last.as_ref() else {
            return;
        };
        let Ok((_, Some([alpha, beta]))) = linear_scaled(&trial.values, self.targets) else {
            return;
        };
        let mut candidate = theta.to_vec();
        candidate[p] = alpha;
        candidate[p + 1] = beta;
        let projected = self.residuals_of(&candidate, &trial.values);
        if projected.iter().all(|r| r.is_finite())
            && half_sum_squares(&projected) <= half_sum_squares(residuals)
        {
            theta.copy_from_slice(&candidate);
            *residuals = projected;
        }
    }
}

/// The native values on every row, or `None` when one raises a refused fault.
fn native_values(expression: &MathExpr, rows: &[Vec<f64>], rejected: FaultSet) -> Option<Vec<f64>> {
    let compiled = expression.compile().ok()?;
    #[cfg(test)]
    passes::native();
    rows.iter()
        .map(|row| {
            let (value, faults) = compiled.eval_row_traced(row);
            (!faults.intersects(rejected)).then_some(value)
        })
        .collect()
}

/// The native score: the residual MSE (Raw), or the coordinate search's LinearScaled MSE with
/// its affine pair.
fn native_score(
    objective: Objective,
    values: &[f64],
    targets: &[f64],
) -> Result<(f64, Option<[f64; 2]>)> {
    match objective {
        #[cfg(test)]
        Objective::Raw => {
            let sum = values
                .iter()
                .zip(targets)
                .map(|(f, y)| (f - y) * (f - y))
                .sum::<f64>();
            let score = sum / values.len() as f64;
            if !score.is_finite() {
                bail!("nonfinite raw objective");
            }
            Ok((score, None))
        }
        Objective::LinearScaled => linear_scaled(values, targets),
    }
}

impl MathExpr {
    /// Bounded LM over the literal sites of a row expression, on TRAINING rows only. Invalid
    /// training data or budget, or a start the admission refuses, is an error, as in the
    /// coordinate search. The winner must win a fresh native re-score.
    fn optimize_constants_lm(
        &self,
        admission: &Admission<'_>,
        rows: &[Vec<f64>],
        targets: &[f64],
        objective: Objective,
        options: &Options,
        budget: GuardedRefinementBudget<'_>,
    ) -> Result<LmRefinement> {
        if rows.is_empty()
            || rows.len() != targets.len()
            || budget.max_evaluations == 0
            || targets.iter().any(|v| !v.is_finite())
        {
            bail!("constant fitting needs finite aligned training targets and a budget");
        }
        if !(options.mu_min >= 0.0
            && options.mu_min <= options.mu_max
            && options.tau > 0.0
            && options.tau.is_finite())
        {
            bail!("invalid damping bounds");
        }
        let start = admission.finish(self.root.clone())?;
        let sites = literal_paths(&start.root);
        let p = sites.len();
        let width = p + if objective == Objective::LinearScaled {
            2
        } else {
            0
        };
        let unchanged =
            |evaluations: usize, rejected_trials: usize, score: f64| ConstantRefinement {
                expression: start.clone(),
                evaluations,
                rejected_trials,
                score,
                linear_scale: None,
            };
        let outcome = |refinement, status| LmRefinement {
            refinement,
            changed: false,
            status,
            run: None,
            rescore: None,
            nonsmooth_rows: 0,
            saturated_rows: 0,
        };
        if admission.elementwise() || p == 0 || width > MAX_PARAMS {
            return Ok(outcome(unchanged(0, 0, f64::NAN), Status::Unsupported));
        }
        let rejected = FaultSeverity::Domain
            .classes()
            .union(FaultSeverity::Hard.classes());
        if !(budget.may_start)() {
            return Ok(outcome(unchanged(0, 0, f64::NAN), Status::Stopped));
        }
        let Some(values) = native_values(&start, rows, rejected) else {
            return Ok(outcome(unchanged(1, 1, f64::NAN), Status::InitialRefused));
        };
        let Ok((start_score, start_scale)) = native_score(objective, &values, targets) else {
            return Ok(outcome(unchanged(1, 1, f64::NAN), Status::InitialRefused));
        };
        let mut theta: Vec<f64> = sites
            .iter()
            .map(|site| literal_value(at(&start.root, site)).unwrap())
            .collect();
        if let Some([alpha, beta]) = start_scale {
            theta.extend([alpha, beta]);
        }
        let target_norm = norm(targets);
        let mut problem = ExprProblem {
            admission,
            start: &start.root,
            sites: &sites,
            rows,
            targets,
            objective,
            rejected,
            scale: 0.0,
            last: None,
            nonsmooth_rows: 0,
            saturated_rows: 0,
        };
        let residuals = problem.residuals_of(&theta, &values);
        problem.scale = if target_norm > 0.0 {
            target_norm
        } else {
            norm(&residuals)
        };
        // The first Jacobian reuses the start's admitted expression and native values, so the
        // initial evaluation is the only native pass at the start.
        problem.last = Some(Trial {
            sites: theta[..p].iter().map(|v| v.to_bits()).collect(),
            expression: start.clone(),
            values,
        });
        let limits = Limits {
            // One evaluation stays reserved for the winner's native re-score.
            max_evaluations: budget.max_evaluations - 1,
            jacobian_cost: match options.jacobian_cost {
                JacobianCost::Sites => p,
                JacobianCost::One => 1,
            },
        };
        let run = if residuals.iter().all(|r| r.is_finite()) {
            minimize(
                &mut problem,
                theta,
                residuals,
                options,
                limits,
                budget.may_start,
            )
        } else {
            Outcome {
                theta,
                residuals,
                objective: f64::INFINITY,
                status: Status::NonFiniteJacobian,
                evaluations: 1,
                trials: 0,
                rejected_trials: 0,
                jacobians: 0,
                solve_failures: 0,
                accepted: 0,
                trace: Vec::new(),
            }
        };
        let mut result = LmRefinement {
            refinement: ConstantRefinement {
                expression: start.clone(),
                evaluations: run.evaluations,
                rejected_trials: run.rejected_trials,
                score: start_score,
                linear_scale: start_scale,
            },
            changed: false,
            status: run.status,
            run: None,
            rescore: None,
            nonsmooth_rows: problem.nonsmooth_rows,
            saturated_rows: problem.saturated_rows,
        };
        // A guard that refused during the run is never asked again. One that refuses only the
        // re-score stops the refinement, and the start stays.
        let rescore = run.accepted > 0
            && run.evaluations < budget.max_evaluations
            && run.status != Status::Stopped;
        if rescore && !(budget.may_start)() {
            result.status = Status::Stopped;
        } else if rescore {
            // The mandatory native re-score: a fresh admission, compile and evaluation, whose
            // values give the score. That admission is the re-score's one native pass.
            result.refinement.evaluations += 1;
            let rescored = problem.admit(&run.theta).and_then(|trial| {
                let (score, scale) = native_score(objective, &trial.values, targets).ok()?;
                Some((trial.expression, score, scale))
            });
            match rescored {
                Some((expression, score, scale)) => {
                    result.rescore = Some(score);
                    if score < start_score {
                        result.refinement.expression = expression;
                        result.refinement.score = score;
                        result.refinement.linear_scale = scale;
                        result.changed = true;
                    }
                }
                None => result.refinement.rejected_trials += 1,
            }
        }
        result.run = Some(run);
        Ok(result)
    }
}

// laneD:B17-lm-refinement begin
impl MathExpr {
    /// Bounded damped least squares over the literal sites of a row expression, on TRAINING rows
    /// only, admitted by the version-two search contract (`Search::finish`) and fitted to the
    /// coordinate search's linear-scaled residuals `α + β f - y`. The guard is asked before every
    /// pass, and the winner replaces the start only on a strictly better fresh native re-score.
    /// Invalid training data, budget or options, or a start the admission refuses, is an error,
    /// as in the coordinate search. An elementwise scope is unsupported and spends nothing.
    pub fn optimize_constants_lm_guarded(
        &self,
        profile: &SearchProfile,
        scope: ExpressionScope,
        rows: &[Vec<f64>],
        targets: &[f64],
        options: LmOptions,
        budget: GuardedRefinementBudget<'_>,
    ) -> Result<GuardedLmRefinement> {
        let admission = Admission::Profile(Search::new(profile, scope)?);
        let options = Options {
            tau: options.tau,
            jacobian_cost: match options.jacobian {
                LmJacobianCharge::Sites => JacobianCost::Sites,
                LmJacobianCharge::One => JacobianCost::One,
            },
            ..Options::default()
        };
        let result = self.optimize_constants_lm(
            &admission,
            rows,
            targets,
            Objective::LinearScaled,
            &options,
            budget,
        )?;
        // The same admission the optimizer made of its start: deterministic and uncharged.
        let start = admission.finish(self.root.clone())?;
        let sites = literal_paths(&start.root).len();
        Ok(result.guarded(start, sites, options.jacobian_cost))
    }

    /// The literal sites `optimize_constants_lm_guarded` fits in this expression, as its trace's
    /// `sites` reports them: one per maximal literal. Its admission keeps that count, since
    /// normalizing a literal and printing it again leaves one literal, so a caller can size a budget
    /// before it asks for the fit. A row expression is supported with 1 to [`LM_MAX_SITES`] sites.
    pub fn literal_sites(&self) -> usize {
        literal_paths(&self.root).len()
    }
}

// The fit's parameters are the literal sites and the affine pair of the linear-scaled residuals.
const _: () = assert!(LM_MAX_SITES + 2 == MAX_PARAMS);

impl Status {
    fn public(self) -> LmStatus {
        match self {
            Self::ExactFit => LmStatus::ExactFit,
            Self::Gradient => LmStatus::Gradient,
            Self::Step => LmStatus::Step,
            Self::IterationLimit => LmStatus::IterationLimit,
            Self::BudgetExhausted => LmStatus::BudgetExhausted,
            Self::DampingLimit => LmStatus::DampingLimit,
            Self::NonFiniteJacobian => LmStatus::NonFiniteJacobian,
            Self::Stopped => LmStatus::Stopped,
            Self::Unsupported => LmStatus::Unsupported,
            Self::InitialRefused => LmStatus::InitialRefused,
        }
    }
}

/// The failure a refused start reports; it cost one evaluation, as in the coordinate search.
const INITIAL_REFUSED: &str =
    "initial value refused: a traced fault or a nonfinite objective on training rows";

impl LmRefinement {
    /// The outcome in the coordinate search's terms, and the trace. A start refused or stopped
    /// before its pass, or an unsupported shape, reports no refinement.
    fn guarded(self, start: MathExpr, sites: usize, cost: JacobianCost) -> GuardedLmRefinement {
        let run = self.run.as_ref();
        let initial = usize::from(run.is_some() || self.status == Status::InitialRefused);
        let rescore = run.map_or(0, |r| self.refinement.evaluations - r.evaluations);
        let trace = LmTrace {
            status: self.status.public(),
            changed: self.changed,
            sites,
            jacobian_cost: match cost {
                JacobianCost::Sites => sites,
                JacobianCost::One => 1,
            },
            initial,
            trials: run.map_or(0, |r| r.trials),
            jacobians: run.map_or(0, |r| r.jacobians),
            rescore,
            rejected_trials: self.refinement.rejected_trials,
            solve_failures: run.map_or(0, |r| r.solve_failures),
            accepted: run.map_or(0, |r| r.accepted),
            rescored: self.rescore,
            nonsmooth_rows: self.nonsmooth_rows,
            saturated_rows: self.saturated_rows,
            iterations: run.map_or_else(Vec::new, |r| {
                r.trace
                    .iter()
                    .map(|i| LmIteration {
                        mu: i.mu,
                        objective: i.objective,
                        trial: i.trial,
                        accepted: i.accepted,
                        solved: i.solved,
                    })
                    .collect()
            }),
        };
        let outcome = match self.status {
            Status::InitialRefused => GuardedConstantRefinement {
                refinement: None,
                stopped: false,
                failure: Some(INITIAL_REFUSED.into()),
            },
            Status::Unsupported => GuardedConstantRefinement {
                refinement: None,
                stopped: false,
                failure: None,
            },
            Status::Stopped if initial == 0 => GuardedConstantRefinement {
                refinement: None,
                stopped: true,
                failure: None,
            },
            status => GuardedConstantRefinement {
                refinement: Some(self.refinement),
                stopped: status == Status::Stopped,
                failure: None,
            },
        };
        GuardedLmRefinement {
            outcome,
            start,
            trace,
        }
    }
}
// laneD:B17-lm-refinement end

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::profile::{profile, SearchProfile};
    use std::cell::Cell;

    fn always() -> impl FnMut() -> bool {
        || true
    }

    fn lm(
        source: &str,
        admission: &Admission<'_>,
        rows: &[Vec<f64>],
        targets: &[f64],
        objective: Objective,
        options: &Options,
        max_evaluations: usize,
    ) -> LmRefinement {
        let mut guard = always();
        MathExpr::parse(source)
            .unwrap()
            .optimize_constants_lm(
                admission,
                rows,
                targets,
                objective,
                options,
                GuardedRefinementBudget {
                    max_evaluations,
                    may_start: &mut guard,
                },
            )
            .unwrap()
    }

    fn sites(expression: &MathExpr) -> Vec<f64> {
        literal_paths(&expression.root)
            .iter()
            .map(|site| literal_value(at(&expression.root, site)).unwrap())
            .collect()
    }

    fn baseline() -> SearchProfile {
        profile("kernel2-baseline", 3).unwrap()
    }

    /// The exponential fixture: x = -1.55 + i/10 for i in 0..32 (the refinement-control grid).
    fn exp_data() -> (Vec<Vec<f64>>, Vec<f64>) {
        let rows: Vec<Vec<f64>> = (0..32).map(|i| vec![-1.55 + f64::from(i) / 10.0]).collect();
        let targets = rows.iter().map(|r| 2.5 * (0.8 * r[0]).exp()).collect();
        (rows, targets)
    }

    fn sine_data() -> (Vec<Vec<f64>>, Vec<f64>) {
        let rows: Vec<Vec<f64>> = (0..32).map(|i| vec![-1.55 + f64::from(i) / 10.0]).collect();
        let targets = rows
            .iter()
            .map(|r| 3.0 * (1.7 * r[0]).sin() + 0.5)
            .collect();
        (rows, targets)
    }

    fn trace_bits(result: &LmRefinement) -> Vec<(u64, u64, Option<u64>, bool, bool)> {
        result
            .run
            .as_ref()
            .unwrap()
            .trace
            .iter()
            .map(Iteration::bits)
            .collect()
    }

    #[test]
    fn recovers_a_times_exp_b_c0_to_one_part_in_a_million() {
        let (rows, targets) = exp_data();
        let options = Options::default();
        let run = || {
            lm(
                "1.0*exp(0.5*c0)",
                &Admission::Structure,
                &rows,
                &targets,
                Objective::Raw,
                &options,
                400,
            )
        };
        let result = run();
        assert!(result.changed, "{:?}", result.status);
        assert!(
            matches!(
                result.status,
                Status::ExactFit | Status::Gradient | Status::Step
            ),
            "{:?}",
            result.status
        );
        let fitted = sites(&result.refinement.expression);
        assert!((fitted[0] - 2.5).abs() <= 2.5e-6, "{fitted:?}");
        assert!((fitted[1] - 0.8).abs() <= 1e-6, "{fitted:?}");
        assert!(
            result.refinement.score < 1e-16,
            "{}",
            result.refinement.score
        );
        assert_eq!(result.rescore, Some(result.refinement.score));
        assert!(result.refinement.evaluations <= 400);
        // A bit-identical replay.
        let again = run();
        assert_eq!(
            again.refinement.expression.to_source().unwrap(),
            result.refinement.expression.to_source().unwrap()
        );
        assert_eq!(
            again.refinement.score.to_bits(),
            result.refinement.score.to_bits()
        );
        assert_eq!(again.refinement.evaluations, result.refinement.evaluations);
        assert_eq!(trace_bits(&again), trace_bits(&result));
    }

    #[test]
    fn the_jacobian_matches_central_differences_of_the_native_residuals() {
        let (rows, targets) = sine_data();
        let profile = baseline();
        let search = Search::new(&profile, ExpressionScope::search_profile(1, false)).unwrap();
        let admission = Admission::Profile(search);
        for (source, objective) in [
            ("sin(1.5*c0)+(0.25*c0)", Objective::Raw),
            ("sin(1.5*c0)+(0.25*c0)", Objective::LinearScaled),
            ("tanh(0.7*c0)*sigmoid(1.2-c0)", Objective::LinearScaled),
        ] {
            let start = admission
                .finish(MathExpr::parse(source).unwrap().root)
                .unwrap();
            let site_paths = literal_paths(&start.root);
            let rejected = FaultSeverity::Domain
                .classes()
                .union(FaultSeverity::Hard.classes());
            let mut problem = ExprProblem {
                admission: &admission,
                start: &start.root,
                sites: &site_paths,
                rows: &rows,
                targets: &targets,
                objective,
                rejected,
                scale: 1.0,
                last: None,
                nonsmooth_rows: 0,
                saturated_rows: 0,
            };
            let mut theta = sites(&start);
            if objective == Objective::LinearScaled {
                theta.extend([0.3, 1.7]);
            }
            let width = theta.len();
            let residuals = problem.residuals(&theta).unwrap();
            let normal = problem.normal_equations(&theta, &residuals).unwrap();
            // Rebuild J by central differences of the native residuals and compare JᵀJ and Jᵀr.
            let mut columns = Vec::new();
            for k in 0..width {
                let h = 1e-6 * theta[k].abs().max(1.0);
                let mut up = theta.clone();
                up[k] += h;
                let mut down = theta.clone();
                down[k] -= h;
                let plus = problem.residuals(&up).unwrap();
                let minus = problem.residuals(&down).unwrap();
                columns.push(
                    plus.iter()
                        .zip(&minus)
                        .map(|(a, b)| (a - b) / (up[k] - down[k]))
                        .collect::<Vec<f64>>(),
                );
            }
            for j in 0..width {
                let g: f64 = columns[j].iter().zip(&residuals).map(|(a, r)| a * r).sum();
                assert!(
                    (g - normal.g[j]).abs() <= 1e-6 * g.abs().max(1.0),
                    "{source} {objective:?} g[{j}]: dual {} central {g}",
                    normal.g[j]
                );
                for k in 0..width {
                    let a: f64 = columns[j].iter().zip(&columns[k]).map(|(x, y)| x * y).sum();
                    assert!(
                        (a - normal.a[j][k]).abs() <= 1e-6 * a.abs().max(1.0),
                        "{source} {objective:?} a[{j}][{k}]: dual {} central {a}",
                        normal.a[j][k]
                    );
                }
            }
        }
    }

    #[test]
    fn protected_fault_trials_are_refused_and_the_fit_recovers() {
        // sqrt(θ)·c0 against 0.01·c0: the first full steps land at θ < 0 (SQRT_PROJECTED).
        let rows: Vec<Vec<f64>> = (1..=10).map(|i| vec![f64::from(i)]).collect();
        let targets: Vec<f64> = rows.iter().map(|r| 0.01 * r[0]).collect();
        let profile = baseline();
        let search = Search::new(&profile, ExpressionScope::search_profile(1, false)).unwrap();
        let result = lm(
            "sqrt(1.0)*c0",
            &Admission::Profile(search),
            &rows,
            &targets,
            Objective::Raw,
            &Options::default(),
            400,
        );
        let run = result.run.as_ref().unwrap();
        assert!(run.rejected_trials >= 1, "{run:?}");
        // MNT's damping rule on this problem refuses exactly the first four trials (μ = 1e-3,
        // 2e-3, 8e-3, 6.4e-2) and accepts at μ = 1.024.
        let first_accept = run.trace.iter().position(|i| i.accepted).unwrap();
        assert_eq!(first_accept, 4, "{:?}", run.trace);
        assert!(run.trace[..4].iter().all(|i| i.trial.is_none()));
        // The scaled curvature of the one site is 1 up to rounding, so these are μ0 = τ and
        // Nielsen's μ ← μν, ν ← 2ν after each refusal.
        for (iteration, expected) in run.trace.iter().zip([1e-3, 2e-3, 8e-3, 6.4e-2, 1.024]) {
            assert!(
                (iteration.mu - expected).abs() <= 1e-12 * expected,
                "{:?}",
                run.trace
            );
        }
        // Accepted objectives never increase.
        let accepted: Vec<f64> = run
            .trace
            .iter()
            .filter(|i| i.accepted)
            .map(|i| i.trial.unwrap())
            .collect();
        assert!(accepted.windows(2).all(|w| w[1] <= w[0]), "{accepted:?}");
        assert!(result.changed);
        let theta = sites(&result.refinement.expression)[0];
        assert!((theta - 1e-4).abs() <= 1e-10, "{theta}");
    }

    #[test]
    fn rank_deficiency_converges_and_a_zero_damping_floor_still_terminates() {
        // a·(b·c0) against 6·c0: only the product is identifiable, so JᵀJ is singular.
        let rows: Vec<Vec<f64>> = (1..=10).map(|i| vec![f64::from(i)]).collect();
        let targets: Vec<f64> = rows.iter().map(|r| 6.0 * r[0]).collect();
        let result = lm(
            "1.0*(1.0*c0)",
            &Admission::Structure,
            &rows,
            &targets,
            Objective::Raw,
            &Options::default(),
            400,
        );
        assert!(
            !matches!(
                result.status,
                Status::DampingLimit | Status::NonFiniteJacobian
            ),
            "{:?}",
            result.status
        );
        let fitted = sites(&result.refinement.expression);
        assert!((fitted[0] * fitted[1] - 6.0).abs() <= 6e-10, "{fitted:?}");
        // With no damping floor and a first damping 1e-20 of the curvature, the singular system
        // is tried all but undamped: the solve refuses, the damping rises, and the fit still lands.
        let zero_floor = Options {
            mu_min: 0.0,
            tau: 1e-20,
            ..Options::default()
        };
        let run = || {
            lm(
                "1.0*(1.0*c0)",
                &Admission::Structure,
                &rows,
                &targets,
                Objective::Raw,
                &zero_floor,
                400,
            )
        };
        let first = run();
        let outcome = first.run.as_ref().unwrap();
        assert!(outcome.solve_failures >= 1, "{outcome:?}");
        assert!(
            !matches!(
                first.status,
                Status::DampingLimit | Status::NonFiniteJacobian
            ),
            "{:?}",
            first.status
        );
        let fitted = sites(&first.refinement.expression);
        assert!((fitted[0] * fitted[1] - 6.0).abs() <= 6e-10, "{fitted:?}");
        let second = run();
        assert_eq!(trace_bits(&first), trace_bits(&second));
    }

    #[test]
    fn the_damped_solver_refuses_singular_and_ill_conditioned_systems() {
        let mut a = [[0.0; MAX_PARAMS]; MAX_PARAMS];
        a[0][0] = 1.0;
        a[0][1] = 1.0;
        a[1][0] = 1.0;
        a[1][1] = 1.0;
        let mut g = [0.0; MAX_PARAMS];
        g[0] = 1.0;
        g[1] = 1.0;
        // Exactly singular without damping: the second pivot is zero.
        assert!(solve_damped(&a, &g, 0.0, 2, 1e-12).is_none());
        // Positive definite but with rcond near 5e-16.
        assert!(solve_damped(&a, &g, 1e-15, 2, 1e-12).is_none());
        // Well posed once damped: (A + I) h = -g gives h = -(1/3, 1/3).
        let h = solve_damped(&a, &g, 1.0, 2, 1e-12).unwrap();
        assert!(
            (h[0] + 1.0 / 3.0).abs() < 1e-15 && (h[1] + 1.0 / 3.0).abs() < 1e-15,
            "{h:?}"
        );
        // A non-finite entry never yields a step.
        a[0][1] = f64::NAN;
        a[1][0] = f64::NAN;
        assert!(solve_damped(&a, &g, 1.0, 2, 1e-12).is_none());
    }

    #[test]
    fn a_zero_jacobian_stops_on_the_gradient_test_without_a_change() {
        // The constant reaches the output only through floor, whose tangent is zero.
        let rows: Vec<Vec<f64>> = (1..=10).map(|i| vec![f64::from(i)]).collect();
        let targets: Vec<f64> = rows.iter().map(|r| 0.5 * r[0]).collect();
        let result = lm(
            "floor(0.3)+c0",
            &Admission::Structure,
            &rows,
            &targets,
            Objective::Raw,
            &Options::default(),
            50,
        );
        assert_eq!(result.status, Status::Gradient);
        assert!(!result.changed);
        assert_eq!(result.refinement.evaluations, 2);
    }

    /// Independent review of #195, R5: the division's slope (about -1e-5) underflowed to a silent
    /// 0 lane, so the run stopped on the gradient test after its first Jacobian, with no trial.
    /// With the slope kept, the fit moves toward the zero target.
    #[test]
    fn a_division_slope_kept_from_underflow_moves_the_fit() {
        let result = lm(
            "(c0/(1e13*c1))*c2",
            &Admission::Structure,
            &[vec![1e-311, 1e-24, 1e308]],
            &[0.0],
            Objective::Raw,
            &Options::default(),
            40,
        );
        let run = result.run.as_ref().unwrap();
        assert_ne!(result.status, Status::Gradient, "{run:?}");
        assert!(run.accepted > 0, "{run:?}");
        assert!(result.changed, "{:?}", result.status);
        assert!(
            result.refinement.score < 1e15,
            "{}",
            result.refinement.score
        );
    }

    /// Independent review of #195, R9: the `^` ratio form's `a'/a` overflowed for a base of 2^-1074,
    /// so the lane was erased and the run stopped on the gradient test after its first Jacobian,
    /// with no trial. With the slope kept (about 5.9e6 here) the run tries steps.
    #[test]
    fn a_power_slope_past_the_ratio_range_still_moves_the_fit() {
        let result = lm(
            "((1e-312*c0)^0.01)*c1",
            &Admission::Structure,
            &[vec![5e-12, 1e-300]],
            &[1.0],
            Objective::Raw,
            &Options::default(),
            40,
        );
        let run = result.run.as_ref().unwrap();
        assert_ne!(result.status, Status::Gradient, "{run:?}");
        assert!(run.trials > 0, "{run:?}");
    }

    /// Independent review of #195, R6: at c0 = 20 `tanh` has rounded to 1, so `1 - tanh²` gave a
    /// silent 0 lane and the run stopped on the gradient test with no trial, keeping a score of 4.
    /// With the slope from the argument (about 3.4e-16) the fit reaches the target.
    #[test]
    fn a_saturated_tanh_slope_still_moves_the_fit() {
        let result = lm(
            "tanh(1.0*c0)",
            &Admission::Structure,
            &[vec![20.0]],
            &[-1.0],
            Objective::Raw,
            &Options::default(),
            40,
        );
        let run = result.run.as_ref().unwrap();
        assert_ne!(result.status, Status::Gradient, "{run:?}");
        assert!(run.accepted > 0, "{run:?}");
        assert!(result.changed, "{:?}", result.status);
        assert!(result.refinement.score < 4.0, "{}", result.refinement.score);
    }

    #[test]
    fn a_non_finite_jacobian_returns_the_start() {
        // The value stays below 1e10 while ∂/∂(first site) is about 1e310.
        let rows: Vec<Vec<f64>> = (1..=8).map(|i| vec![f64::from(i)]).collect();
        let targets: Vec<f64> = rows.iter().map(|r| r[0]).collect();
        let result = lm(
            "sin((1.0*c0)*1e300)*1e10",
            &Admission::Structure,
            &rows,
            &targets,
            Objective::Raw,
            &Options::default(),
            50,
        );
        assert_eq!(result.status, Status::NonFiniteJacobian);
        assert!(!result.changed);
        assert_eq!(
            result.refinement.expression.to_source().unwrap(),
            MathExpr::parse("sin((1.0*c0)*1e300)*1e10")
                .unwrap()
                .to_source()
                .unwrap()
        );
    }

    #[test]
    fn unsupported_shapes_spend_nothing() {
        let rows = vec![vec![1.0], vec![2.0]];
        let targets = vec![1.0, 2.0];
        let profile = baseline();
        let elementwise = Admission::Profile(
            Search::new(&profile, ExpressionScope::search_profile(1, true)).unwrap(),
        );
        for (source, admission, objective) in [
            ("sin(1.5*x)", &elementwise, Objective::Raw),
            ("c0*c0", &Admission::Structure, Objective::Raw),
            // 17 sites; and 15 sites plus the affine pair.
            (
                "c0*(1.0+2.0+3.0+4.0+5.0+6.0+7.0+8.0)+(9.0+10.0+11.0+12.0+13.0+14.0+15.0+16.0+17.0)",
                &Admission::Structure,
                Objective::Raw,
            ),
            (
                "c0*(1.0+2.0+3.0+4.0+5.0+6.0+7.0+8.0)+(9.0+10.0+11.0+12.0+13.0+14.0+15.0)",
                &Admission::Structure,
                Objective::LinearScaled,
            ),
        ] {
            let mut starts = 0;
            let mut guard = || {
                starts += 1;
                true
            };
            let result = MathExpr::parse(source)
                .unwrap()
                .optimize_constants_lm(
                    admission,
                    &rows,
                    &targets,
                    objective,
                    &Options::default(),
                    GuardedRefinementBudget {
                        max_evaluations: 10,
                        may_start: &mut guard,
                    },
                )
                .unwrap();
            assert_eq!(result.status, Status::Unsupported, "{source}");
            assert_eq!(result.refinement.evaluations, 0, "{source}");
            assert!(!result.changed);
            assert_eq!(starts, 0, "{source}");
        }
    }

    #[test]
    fn every_evaluation_is_charged_and_the_guard_is_asked_first() {
        let (rows, targets) = exp_data();
        for budget in [1usize, 2, 3, 4, 7, 12, 40] {
            let starts = Cell::new(0usize);
            let mut guard = || {
                starts.set(starts.get() + 1);
                true
            };
            let result = MathExpr::parse("1.0*exp(0.5*c0)")
                .unwrap()
                .optimize_constants_lm(
                    &Admission::Structure,
                    &rows,
                    &targets,
                    Objective::Raw,
                    &Options::default(),
                    GuardedRefinementBudget {
                        max_evaluations: budget,
                        may_start: &mut guard,
                    },
                )
                .unwrap();
            let run = result.run.as_ref().unwrap();
            assert!(
                result.refinement.evaluations <= budget,
                "{budget}: {result:?}"
            );
            assert_eq!(
                run.evaluations,
                1 + run.trials + 2 * run.jacobians,
                "{budget}: two sites per Jacobian"
            );
            let rescored = usize::from(
                result.rescore.is_some() || result.refinement.evaluations > run.evaluations,
            );
            assert_eq!(
                result.refinement.evaluations,
                run.evaluations + rescored,
                "{budget}"
            );
            // One guard check per initial value, Jacobian, trial and re-score.
            assert_eq!(
                starts.get(),
                1 + run.jacobians + run.trials + rescored,
                "{budget}"
            );
        }
        // A refusing guard starts nothing.
        let mut never = || false;
        let stopped = MathExpr::parse("1.0*exp(0.5*c0)")
            .unwrap()
            .optimize_constants_lm(
                &Admission::Structure,
                &rows,
                &targets,
                Objective::Raw,
                &Options::default(),
                GuardedRefinementBudget {
                    max_evaluations: 50,
                    may_start: &mut never,
                },
            )
            .unwrap();
        assert_eq!(stopped.status, Status::Stopped);
        assert_eq!(stopped.refinement.evaluations, 0);
    }

    #[test]
    fn a_guard_that_refuses_partway_is_never_asked_again() {
        let (rows, targets) = exp_data();
        let start = MathExpr::parse("1.0*exp(0.5*c0)").unwrap();
        let refine = |allowed: usize| {
            let starts = Cell::new(0usize);
            let mut guard = || {
                starts.set(starts.get() + 1);
                starts.get() <= allowed
            };
            let result = start
                .optimize_constants_lm(
                    &Admission::Structure,
                    &rows,
                    &targets,
                    Objective::Raw,
                    &Options::default(),
                    GuardedRefinementBudget {
                        max_evaluations: 400,
                        may_start: &mut guard,
                    },
                )
                .unwrap();
            (result, starts.get())
        };
        let (full, total) = refine(usize::MAX);
        assert!(full.changed && full.rescore.is_some(), "{full:?}");
        // Every refusal point, the last of which is the re-score itself.
        for allowed in 1..total {
            let (result, starts) = refine(allowed);
            assert_eq!(starts, allowed + 1, "refused at question {}", allowed + 1);
            assert_eq!(result.status, Status::Stopped, "{allowed}");
            assert!(!result.changed, "{allowed}");
            assert_eq!(result.rescore, None, "{allowed}");
            assert_eq!(
                result.refinement.expression.to_source().unwrap(),
                start.to_source().unwrap()
            );
        }
    }

    #[test]
    fn a_unit_jacobian_charge_and_counted_kinks_and_saturation() {
        let (rows, targets) = exp_data();
        let unit = Options {
            jacobian_cost: JacobianCost::One,
            ..Options::default()
        };
        let result = lm(
            "1.0*exp(0.5*c0)",
            &Admission::Structure,
            &rows,
            &targets,
            Objective::Raw,
            &unit,
            400,
        );
        let run = result.run.as_ref().unwrap();
        assert_eq!(run.evaluations, 1 + run.trials + run.jacobians);
        assert!(result.changed);
        // Rows on a clip bound (ties) and a clamped exponential argument (Saturation) are counted;
        // neither refuses a trial nor stops the fit.
        let rows: Vec<Vec<f64>> = [-3.0, -2.0, -1.0, 0.5, 1.0, 2.0, 3.0, 70.0]
            .iter()
            .map(|v| vec![*v])
            .collect();
        let targets: Vec<f64> = rows.iter().map(|r| r[0].clamp(-1.5, 1.5)).collect();
        let result = lm(
            "clip(1.0*c0,-2.0,2.0)+(0.001*exp(0.5*c0))",
            &Admission::Structure,
            &rows,
            &targets,
            Objective::Raw,
            &Options::default(),
            100,
        );
        assert!(result.nonsmooth_rows >= 2, "{result:?}");
        assert!(result.saturated_rows >= 1, "{result:?}");
        assert!(result.run.as_ref().unwrap().jacobians >= 1);
    }

    #[test]
    fn a_refused_start_costs_one_evaluation() {
        // log(-1·c0) on positive rows: LOG_DOMAIN on every row.
        let rows: Vec<Vec<f64>> = (1..=4).map(|i| vec![f64::from(i)]).collect();
        let targets = vec![0.0; 4];
        let result = lm(
            "log((-1.0)*c0)",
            &Admission::Structure,
            &rows,
            &targets,
            Objective::Raw,
            &Options::default(),
            10,
        );
        assert_eq!(result.status, Status::InitialRefused);
        assert_eq!(result.refinement.evaluations, 1);
        assert_eq!(result.refinement.rejected_trials, 1);
    }

    /// Independent review of #195, R1: the first Jacobian re-admitted and re-evaluated the start,
    /// and the re-score evaluated twice, so native passes went uncharged. The work done is now
    /// held to the charges: each native pass is the initial value, a trial or the re-score, and
    /// each dual pass a Jacobian at its cost.
    #[test]
    fn every_native_and_dual_pass_is_charged() {
        let check =
            |source: &str, rows: &[Vec<f64>], targets: &[f64], options: &Options, budget: usize| {
                passes::take();
                let result = lm(
                    source,
                    &Admission::Structure,
                    rows,
                    targets,
                    Objective::Raw,
                    options,
                    budget,
                );
                let (native, dual) = passes::take();
                let run = result.run.as_ref().unwrap();
                let cost = match options.jacobian_cost {
                    JacobianCost::Sites => sites(&MathExpr::parse(source).unwrap()).len(),
                    JacobianCost::One => 1,
                };
                let rescored = usize::from(result.rescore.is_some());
                assert_eq!(
                    result.refinement.evaluations,
                    run.evaluations + rescored,
                    "{source} {budget}"
                );
                assert_eq!(dual, run.jacobians, "{source} {budget}");
                // Every trial on these fixtures is admitted, so each one is a native pass.
                assert_eq!(native, 1 + run.trials + rescored, "{source} {budget}");
                assert_eq!(
                    result.refinement.evaluations,
                    native + cost * dual,
                    "{source} {budget}"
                );
                (native, dual, result)
            };
        // The review's case: one trial, one Jacobian and the re-score in a budget of 4.
        let (native, dual, result) = check("0.45^c0", &[vec![2.0]], &[1.0], &Options::default(), 4);
        assert_eq!((native, dual), (3, 1));
        assert_eq!(result.refinement.evaluations, 4);
        assert!(result.changed);
        let (rows, targets) = exp_data();
        let unit = Options {
            jacobian_cost: JacobianCost::One,
            ..Options::default()
        };
        for options in [&Options::default(), &unit] {
            for budget in [1usize, 2, 3, 4, 7, 12, 40, 400] {
                check("1.0*exp(0.5*c0)", &rows, &targets, options, budget);
            }
        }
    }

    /// Independent review of #195, R2: a gain ratio in (0, 1/2) makes Nielsen's factor exceed 1,
    /// and the accepted-step update applied only the floor, so the next solve ran above `mu_max`.
    #[test]
    fn an_accepted_low_gain_step_keeps_the_damping_within_its_bounds() {
        let capped = Options {
            mu_max: 1e-3,
            ..Options::default()
        };
        let result = lm(
            "0.45^c0",
            &Admission::Structure,
            &[vec![2.0]],
            &[1.0],
            Objective::Raw,
            &capped,
            40,
        );
        let run = result.run.as_ref().unwrap();
        assert!(
            run.trace
                .iter()
                .all(|i| i.mu >= capped.mu_min && i.mu <= capped.mu_max),
            "{:?}",
            run.trace
        );
        // The first step is accepted with a gain ratio below 1/2. Unclamped, the next damping
        // would be 1.797e-3.
        assert!(run.trace[0].accepted, "{:?}", run.trace);
        assert_eq!(
            run.trace[1].mu.to_bits(),
            capped.mu_max.to_bits(),
            "{:?}",
            run.trace
        );
        // Both bounds: with the floor at 5e-4 the next accepted update (about 3.3e-4 unclamped)
        // lands on the floor.
        let both = Options {
            mu_min: 5e-4,
            mu_max: 1e-3,
            ..Options::default()
        };
        let result = lm(
            "0.45^c0",
            &Admission::Structure,
            &[vec![2.0]],
            &[1.0],
            Objective::Raw,
            &both,
            40,
        );
        let trace = &result.run.as_ref().unwrap().trace;
        assert_eq!(trace[1].mu.to_bits(), both.mu_max.to_bits(), "{trace:?}");
        assert_eq!(trace[2].mu.to_bits(), both.mu_min.to_bits(), "{trace:?}");
    }

    /// No pass runs without its own yes: nothing before the first guard question, nothing after
    /// a refusal, and at most one native or dual pass after each yes. The guard reads the pass
    /// counters before it answers, at every refusal point of two fixtures.
    #[test]
    fn no_pass_runs_without_its_own_yes() {
        fn check(source: &str, admission: &Admission<'_>, rows: &[Vec<f64>], targets: &[f64]) {
            let start = MathExpr::parse(source).unwrap();
            let run = |refuse_at: usize| {
                passes::take();
                let mut segments: Vec<(usize, usize)> = Vec::new();
                let mut answers: Vec<bool> = Vec::new();
                let mut guard = || {
                    segments.push(passes::take());
                    let yes = answers.len() + 1 < refuse_at;
                    answers.push(yes);
                    yes
                };
                start
                    .optimize_constants_lm(
                        admission,
                        rows,
                        targets,
                        Objective::Raw,
                        &Options::default(),
                        GuardedRefinementBudget {
                            max_evaluations: 400,
                            may_start: &mut guard,
                        },
                    )
                    .unwrap();
                segments.push(passes::take());
                (segments, answers)
            };
            let (_, full) = run(usize::MAX);
            for refuse_at in 1..=full.len() + 1 {
                let (segments, answers) = run(refuse_at);
                assert_eq!(segments[0], (0, 0), "{source} {refuse_at}");
                for (work, yes) in segments[1..].iter().zip(&answers) {
                    assert!(
                        work.0 + work.1 <= usize::from(*yes),
                        "{source} {refuse_at}: {segments:?} {answers:?}"
                    );
                }
            }
        }
        let (rows, targets) = exp_data();
        check("1.0*exp(0.5*c0)", &Admission::Structure, &rows, &targets);
        // The protected-fault fixture: trials refused for a Domain fault still run their pass.
        let rows: Vec<Vec<f64>> = (1..=10).map(|i| vec![f64::from(i)]).collect();
        let targets: Vec<f64> = rows.iter().map(|r| 0.01 * r[0]).collect();
        let profile = baseline();
        let search = Search::new(&profile, ExpressionScope::search_profile(1, false)).unwrap();
        check("sqrt(1.0)*c0", &Admission::Profile(search), &rows, &targets);
    }

    #[test]
    fn invalid_damping_options_are_refused() {
        let (rows, targets) = exp_data();
        for options in [
            Options {
                tau: f64::INFINITY,
                ..Options::default()
            },
            Options {
                tau: 0.0,
                ..Options::default()
            },
            Options {
                tau: f64::NAN,
                ..Options::default()
            },
            Options {
                mu_min: -1e-9,
                ..Options::default()
            },
            Options {
                mu_min: 1.0,
                mu_max: 0.5,
                ..Options::default()
            },
            Options {
                mu_max: f64::NAN,
                ..Options::default()
            },
        ] {
            let mut guard = always();
            let refused = MathExpr::parse("1.0*exp(0.5*c0)")
                .unwrap()
                .optimize_constants_lm(
                    &Admission::Structure,
                    &rows,
                    &targets,
                    Objective::Raw,
                    &options,
                    GuardedRefinementBudget {
                        max_evaluations: 50,
                        may_start: &mut guard,
                    },
                );
            assert!(refused.is_err(), "{options:?}");
        }
    }

    /// With `mu_min = 0` the damping can start at an exact 0: here the scaled curvature underflows
    /// (1e-163 squares to 0) while the gradient does not. A raise from 0 used to stay 0, and once
    /// `nu` overflowed it gave a NaN damping. Now every damping is finite and the run ends on its
    /// own test, not the iteration limit.
    #[test]
    fn a_zero_damping_still_rises() {
        let zero_floor = Options {
            mu_min: 0.0,
            max_iterations: 1100,
            ..Options::default()
        };
        let result = lm(
            "c1+(1.0*c0)",
            &Admission::Structure,
            &[vec![1e-163, 1e11]],
            &[1e-150],
            Objective::Raw,
            &zero_floor,
            100,
        );
        let trace = &result.run.as_ref().unwrap().trace;
        assert_eq!(trace[0].mu, 0.0, "{trace:?}");
        assert!(trace.iter().all(|i| i.mu.is_finite()), "{trace:?}");
        assert!(trace[1].mu > 0.0, "{trace:?}");
        assert_ne!(result.status, Status::IterationLimit, "{trace:?}");
    }

    #[test]
    fn linear_scaled_recovers_a_frequency_inside_the_kernel2_pool() {
        let (rows, targets) = sine_data();
        let profile = baseline();
        let scope = ExpressionScope::search_profile(1, false);
        let admission = Admission::Profile(Search::new(&profile, scope).unwrap());
        let result = lm(
            "sin(1.5*c0)",
            &admission,
            &rows,
            &targets,
            Objective::LinearScaled,
            &Options::default(),
            200,
        );
        assert!(result.changed, "{:?}", result.status);
        let frequency = sites(&result.refinement.expression)[0];
        assert!((frequency - 1.7).abs() <= 1e-6, "{frequency}");
        let [intercept, slope] = result.refinement.linear_scale.unwrap();
        assert!((intercept - 0.5).abs() <= 1e-6 && (slope - 3.0).abs() <= 1e-6);
        assert!(
            result.refinement.score <= 1e-12,
            "{}",
            result.refinement.score
        );
        // The coordinate search at the same budget, for the record.
        let coordinate = MathExpr::parse("sin(1.5*c0)")
            .unwrap()
            .optimize_constants_v2(
                &profile,
                scope,
                &rows,
                &targets,
                crate::kernel::expression_search::ConstantObjective::LinearScaled,
                result.refinement.evaluations,
            )
            .unwrap();
        assert!(result.refinement.score <= coordinate.score);
    }

    /// The train/held-out split a caller owns. The optimizer receives the training half only.
    struct Cut {
        train_x: Vec<Vec<f64>>,
        train_y: Vec<f64>,
        hold_x: Vec<Vec<f64>>,
        hold_y: Vec<f64>,
    }

    /// Held-out isolation comes from the signature: `optimize_constants_lm` takes the training
    /// rows and targets and nothing else, so no test can make a caller's held-out half reach it.
    /// This pins what can be tested. Two cuts with the same training half fit identically whatever
    /// their held-out halves hold. Held-out rows handed over by mistake, with non-finite labels,
    /// are refused rather than fitted.
    #[test]
    fn the_fit_sees_only_its_training_rows_and_refuses_non_finite_targets() {
        let (rows, targets) = exp_data();
        let clean = Cut {
            train_x: rows[..24].to_vec(),
            train_y: targets[..24].to_vec(),
            hold_x: rows[24..].to_vec(),
            hold_y: targets[24..].to_vec(),
        };
        let poisoned = Cut {
            hold_x: vec![vec![-1e100], vec![1e100]],
            hold_y: vec![f64::NAN, f64::INFINITY],
            ..Cut {
                train_x: clean.train_x.clone(),
                train_y: clean.train_y.clone(),
                hold_x: Vec::new(),
                hold_y: Vec::new(),
            }
        };
        let fit = |cut: &Cut| {
            lm(
                "1.0*exp(0.5*c0)",
                &Admission::Structure,
                &cut.train_x,
                &cut.train_y,
                Objective::Raw,
                &Options::default(),
                200,
            )
        };
        let (a, b) = (fit(&clean), fit(&poisoned));
        assert_eq!(
            a.refinement.expression.to_source().unwrap(),
            b.refinement.expression.to_source().unwrap()
        );
        assert_eq!(a.refinement.evaluations, b.refinement.evaluations);
        assert_eq!(trace_bits(&a), trace_bits(&b));
        // Held-out rows handed over by mistake are refused, not fitted.
        let mut leaked_x = poisoned.train_x.clone();
        leaked_x.extend(poisoned.hold_x.iter().cloned());
        let mut leaked_y = poisoned.train_y.clone();
        leaked_y.extend(poisoned.hold_y.iter().copied());
        let mut guard = always();
        assert!(MathExpr::parse("1.0*exp(0.5*c0)")
            .unwrap()
            .optimize_constants_lm(
                &Admission::Structure,
                &leaked_x,
                &leaked_y,
                Objective::Raw,
                &Options::default(),
                GuardedRefinementBudget {
                    max_evaluations: 10,
                    may_start: &mut guard,
                },
            )
            .is_err());
        assert!(clean.hold_x.len() == 8 && clean.hold_y.len() == 8);
    }

    // -----------------------------------------------------------------------------------------
    // The generic core on problems with published answers.

    /// A residual function with a hand-written Jacobian.
    struct Closed<F, J> {
        residuals: F,
        jacobian: J,
        scale: f64,
    }

    impl<F, J> Problem for Closed<F, J>
    where
        F: FnMut(&[f64]) -> Vec<f64>,
        J: FnMut(&[f64]) -> Vec<Vec<f64>>,
    {
        fn scale(&self) -> f64 {
            self.scale
        }

        fn residuals(&mut self, theta: &[f64]) -> Option<Vec<f64>> {
            let r = (self.residuals)(theta);
            r.iter().all(|v| v.is_finite()).then_some(r)
        }

        fn normal_equations(&mut self, theta: &[f64], residuals: &[f64]) -> Option<Normal> {
            let mut normal = Normal::zero();
            for (row, r) in (self.jacobian)(theta).iter().zip(residuals) {
                let mut lanes = [0.0; MAX_PARAMS];
                lanes[..row.len()].copy_from_slice(row);
                normal.accumulate(&lanes, *r, theta.len());
            }
            normal.finish(theta.len())
        }
    }

    fn solve_closed<F, J>(problem: &mut Closed<F, J>, theta: Vec<f64>, options: &Options) -> Outcome
    where
        F: FnMut(&[f64]) -> Vec<f64>,
        J: FnMut(&[f64]) -> Vec<Vec<f64>>,
    {
        let residuals = (problem.residuals)(&theta);
        let mut guard = always();
        minimize(
            problem,
            theta,
            residuals,
            options,
            Limits {
                max_evaluations: 100_000,
                jacobian_cost: 1,
            },
            &mut guard,
        )
    }

    #[test]
    fn rosenbrock_converges_as_madsen_nielsen_and_tingleff_report() {
        // MNT Example 3.13 with λ = 0: 17 iterations to ‖x - x*‖ = 2.78e-12 with D = I.
        let mut problem = Closed {
            residuals: |t: &[f64]| vec![10.0 * (t[1] - t[0] * t[0]), 1.0 - t[0]],
            jacobian: |t: &[f64]| vec![vec![-20.0 * t[0], 10.0], vec![-1.0, 0.0]],
            scale: 1.0,
        };
        let levenberg = Options {
            scaling: Scaling::Identity,
            step_tolerance: 1e-14,
            mu_min: 0.0,
            max_iterations: 200,
            ..Options::default()
        };
        let run = solve_closed(&mut problem, vec![-1.2, 1.0], &levenberg);
        let error = ((run.theta[0] - 1.0).powi(2) + (run.theta[1] - 1.0).powi(2)).sqrt();
        assert!(error <= 1e-10, "{run:?}");
        assert!(run.trace.len() <= 25, "{} iterations", run.trace.len());
        let marquardt = Options {
            step_tolerance: 1e-14,
            max_iterations: 200,
            ..Options::default()
        };
        let run = solve_closed(&mut problem, vec![-1.2, 1.0], &marquardt);
        let error = ((run.theta[0] - 1.0).powi(2) + (run.theta[1] - 1.0).powi(2)).sqrt();
        assert!(error <= 1e-8, "{run:?}");
        let accepted: Vec<f64> = run
            .trace
            .iter()
            .filter(|i| i.accepted)
            .map(|i| i.trial.unwrap())
            .collect();
        assert!(accepted.windows(2).all(|w| w[1] <= w[0]));
    }

    /// Independent review of #195, N1: the first damped step of `r = (θ0 + 1, θ1 + 1)` from 0, with
    /// θ0 expressed as `u φ0` and the step read back in θ. At u = 1e-3 the floor leaves φ0's scale
    /// alone and the step is u = 1's, -1/1.001. At u = 1e-6 φ0's curvature 1e-12 is floored to 1e-10
    /// of the largest, which bounds the step to -1/1.1.
    #[test]
    fn rescaling_keeps_the_damped_step_until_the_floor_raises_the_scale() {
        let first_step = |u: f64| {
            let mut problem = Closed {
                residuals: move |t: &[f64]| vec![u * t[0] + 1.0, t[1] + 1.0],
                jacobian: move |_: &[f64]| vec![vec![u, 0.0], vec![0.0, 1.0]],
                scale: 1.0,
            };
            let one = Options {
                max_iterations: 1,
                ..Options::default()
            };
            let run = solve_closed(&mut problem, vec![0.0, 0.0], &one);
            assert!(run.trace[0].accepted, "{run:?}");
            [u * run.theta[0], run.theta[1]]
        };
        let unit = first_step(1.0);
        let free = -1.0 / (1.0 + Options::default().tau);
        for step in [unit, first_step(1e-3)] {
            for v in step {
                assert!((v - free).abs() <= 1e-14, "{step:?}");
            }
        }
        let floored = first_step(1e-6);
        assert!((floored[0] + 1.0 / 1.1).abs() <= 1e-12, "{floored:?}");
        assert!((floored[1] - free).abs() <= 1e-14, "{floored:?}");
    }

    /// Independent review of #195, N1 follow-up: with coupled parameters the floor matters for the
    /// whole problem, not only for the rescaled parameter. A = [[1, 0.5], [0.5, 1]], g = (1, 1):
    /// expressing θ0 in units 1e3 times larger leaves every scale above the floor and keeps the
    /// first physical step; 1e6 times larger lifts θ0's curvature to 1e12, the floor raises θ1's
    /// scale from 1 to 100, and θ0's own step changes although its floor is inactive.
    #[test]
    fn a_coupled_problem_keeps_its_step_only_while_no_scale_is_floored() {
        // J = [[u, 0.5], [0, √0.75]] gives JᵀJ = A at u = 1; c makes g = Jᵀc = (1, 1) at 0.
        let root = 0.75f64.sqrt();
        let c = [1.0, 0.5 / root];
        let first_step = |u: f64| {
            let mut problem = Closed {
                residuals: move |t: &[f64]| vec![u * t[0] + 0.5 * t[1] + c[0], root * t[1] + c[1]],
                jacobian: move |_: &[f64]| vec![vec![u, 0.5], vec![0.0, root]],
                scale: 1.0,
            };
            let one = Options {
                max_iterations: 1,
                ..Options::default()
            };
            let run = solve_closed(&mut problem, vec![0.0, 0.0], &one);
            assert!(run.trace[0].accepted, "{run:?}");
            [u * run.theta[0], run.theta[1]]
        };
        let unit = first_step(1.0);
        let inactive = first_step(1e3);
        for (a, b) in inactive.iter().zip(unit) {
            assert!((a - b).abs() <= 1e-12 * b.abs(), "{inactive:?} vs {unit:?}");
        }
        let floored = first_step(1e6);
        assert!(
            (floored[0] - unit[0]).abs() > 1e-3,
            "{floored:?} vs {unit:?}"
        );
    }

    /// The log relative error -log10(|q - c| / |c|) (NIST StRD).
    fn lre(q: f64, c: f64) -> f64 {
        if q == c {
            f64::INFINITY
        } else {
            -((q - c).abs() / c.abs()).log10()
        }
    }

    const DANWOOD: [(f64, f64); 6] = [
        (2.138, 1.309),
        (3.421, 1.471),
        (3.597, 1.490),
        (4.340, 1.565),
        (4.882, 1.611),
        (5.660, 1.680),
    ];
    const DANWOOD_CERTIFIED: [f64; 2] = [7.6886226176E-01, 3.8604055871E+00];
    const DANWOOD_RSS: f64 = 4.3173084083E-03;

    const MISRA1A: [(f64, f64); 14] = [
        (10.07, 77.6),
        (14.73, 114.9),
        (17.94, 141.1),
        (23.93, 190.8),
        (29.61, 239.9),
        (35.18, 289.0),
        (40.02, 332.8),
        (44.82, 378.4),
        (50.76, 434.8),
        (55.05, 477.3),
        (61.01, 536.8),
        (66.40, 593.1),
        (75.47, 689.1),
        (81.78, 760.0),
    ];
    const MISRA1A_CERTIFIED: [f64; 2] = [2.3894212918E+02, 5.5015643181E-04];
    const MISRA1A_RSS: f64 = 1.2455138894E-01;

    type ResidualFn = fn(&[f64]) -> Vec<f64>;
    type JacobianFn = fn(&[f64]) -> Vec<Vec<f64>>;

    fn danwood_problem() -> Closed<ResidualFn, JacobianFn> {
        Closed {
            residuals: |t: &[f64]| {
                DANWOOD
                    .iter()
                    .map(|(y, x)| t[0] * x.powf(t[1]) - y)
                    .collect()
            },
            jacobian: |t: &[f64]| {
                DANWOOD
                    .iter()
                    .map(|(_, x)| vec![x.powf(t[1]), t[0] * x.powf(t[1]) * x.ln()])
                    .collect()
            },
            scale: norm(&DANWOOD.map(|(y, _)| y)),
        }
    }

    fn misra1a_problem() -> Closed<ResidualFn, JacobianFn> {
        Closed {
            residuals: |t: &[f64]| {
                MISRA1A
                    .iter()
                    .map(|(y, x)| t[0] * (1.0 - (-t[1] * x).exp()) - y)
                    .collect()
            },
            jacobian: |t: &[f64]| {
                MISRA1A
                    .iter()
                    .map(|(_, x)| vec![1.0 - (-t[1] * x).exp(), t[0] * x * (-t[1] * x).exp()])
                    .collect()
            },
            scale: norm(&MISRA1A.map(|(y, _)| y)),
        }
    }

    #[test]
    fn nist_lower_difficulty_problems_match_their_certified_values() {
        // Data integrity first: the certified parameters reproduce the certified RSS.
        let rss = |r: Vec<f64>| r.iter().map(|v| v * v).sum::<f64>();
        let mut danwood = danwood_problem();
        let mut misra = misra1a_problem();
        assert!(lre(rss((danwood.residuals)(&DANWOOD_CERTIFIED)), DANWOOD_RSS) >= 9.0);
        assert!(lre(rss((misra.residuals)(&MISRA1A_CERTIFIED)), MISRA1A_RSS) >= 9.0);
        let options = Options {
            gradient_tolerance: 1e-13,
            step_tolerance: 1e-13,
            max_iterations: 1000,
            ..Options::default()
        };
        for start in [[1.0, 5.0], [0.7, 4.0]] {
            let run = solve_closed(&mut danwood, start.to_vec(), &options);
            for (q, c) in run.theta.iter().zip(DANWOOD_CERTIFIED) {
                assert!(lre(*q, c) >= 6.0, "DanWood from {start:?}: {run:?}");
            }
            assert!(lre(2.0 * run.objective, DANWOOD_RSS) >= 6.0, "{run:?}");
        }
        for start in [[500.0, 1e-4], [250.0, 5e-4]] {
            let run = solve_closed(&mut misra, start.to_vec(), &options);
            for (q, c) in run.theta.iter().zip(MISRA1A_CERTIFIED) {
                assert!(lre(*q, c) >= 6.0, "Misra1a from {start:?}: {run:?}");
            }
            assert!(lre(2.0 * run.objective, MISRA1A_RSS) >= 6.0, "{run:?}");
        }
    }

    #[test]
    fn danwood_through_the_expression_uses_both_power_partials() {
        let rows: Vec<Vec<f64>> = DANWOOD.iter().map(|(_, x)| vec![*x]).collect();
        let targets: Vec<f64> = DANWOOD.iter().map(|(y, _)| *y).collect();
        let options = Options {
            gradient_tolerance: 1e-13,
            step_tolerance: 1e-13,
            max_iterations: 1000,
            ..Options::default()
        };
        let result = lm(
            "1.0*(c0^5.0)",
            &Admission::Structure,
            &rows,
            &targets,
            Objective::Raw,
            &options,
            10_000,
        );
        assert!(result.changed);
        let fitted = sites(&result.refinement.expression);
        for (q, c) in fitted.iter().zip(DANWOOD_CERTIFIED) {
            assert!(lre(*q, c) >= 6.0, "{fitted:?} {:?}", result.status);
        }
    }

    /// `optimize_constants_v1` on the LM fixtures, frozen at its current behavior so wiring the
    /// LM beside it can be checked to leave it untouched.
    #[test]
    fn v1_constant_fitting_on_the_lm_fixtures_is_frozen() {
        let (exp_rows, exp_targets) = exp_data();
        let (sine_rows, sine_targets) = sine_data();
        let fitted = |source: &str, rows: &[Vec<f64>], targets: &[f64], budget: usize| {
            let (expression, count) = MathExpr::parse(source)
                .unwrap()
                .optimize_constants_v1(rows, targets, budget)
                .unwrap();
            format!("{} | evaluations={count}", expression.to_source().unwrap())
        };
        let actual = [
            fitted("1.0*exp(0.5*c0)", &exp_rows, &exp_targets, 50),
            fitted("1.0*exp(0.5*c0)", &exp_rows, &exp_targets, 400),
            fitted("sin(1.5*c0)", &sine_rows, &sine_targets, 100),
        ];
        let expected: [&str; 3] = V1_FROZEN;
        assert_eq!(actual, expected);
    }

    const V1_FROZEN: [&str; 3] = [
        "(2.625*exp((0.75*c0))) | evaluations=50",
        "(2.5000009536743164*exp((0.7999997138977051*c0))) | evaluations=400",
        "sin((1.5245388001203537*c0)) | evaluations=100",
    ];
}

// laneD:B17-lm-refinement begin
/// The public entry: the same optimizer behind the version-two admission, in the coordinate
/// search's terms, with a trace that accounts for every charged pass.
#[cfg(test)]
mod public_tests {
    use super::*;
    use crate::kernel::profile::{profile, SearchProfile};
    use std::cell::Cell;

    fn baseline() -> SearchProfile {
        profile("kernel2-baseline", 3).unwrap()
    }

    fn sine_data() -> (Vec<Vec<f64>>, Vec<f64>) {
        let rows: Vec<Vec<f64>> = (0..32).map(|i| vec![-1.55 + f64::from(i) / 10.0]).collect();
        let targets = rows
            .iter()
            .map(|r| 3.0 * (1.7 * r[0]).sin() + 0.5)
            .collect();
        (rows, targets)
    }

    fn public(
        source: &str,
        rows: &[Vec<f64>],
        targets: &[f64],
        options: LmOptions,
        max_evaluations: usize,
        guard: &mut dyn FnMut() -> bool,
    ) -> GuardedLmRefinement {
        MathExpr::parse(source)
            .unwrap()
            .optimize_constants_lm_guarded(
                &baseline(),
                ExpressionScope::search_profile(1, false),
                rows,
                targets,
                options,
                GuardedRefinementBudget {
                    max_evaluations,
                    may_start: guard,
                },
            )
            .unwrap()
    }

    #[test]
    fn default_options_are_the_core_defaults() {
        let options = LmOptions::default();
        assert_eq!(options.tau.to_bits(), Options::default().tau.to_bits());
        assert_eq!(options.jacobian, LmJacobianCharge::Sites);
        assert_eq!(Options::default().jacobian_cost, JacobianCost::Sites);
        let names: Vec<&str> = LmStatus::ALL.iter().map(|s| s.as_str()).collect();
        let mut distinct = names.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), names.len());
    }

    /// At every budget, and with either Jacobian charge, the public entry reports what the
    /// staged optimizer does under the same admission, and the trace's charge is the outcome's.
    #[test]
    fn the_public_entry_reports_the_core_optimizer_and_its_charges() {
        let (rows, targets) = sine_data();
        let profile = baseline();
        for jacobian in [LmJacobianCharge::Sites, LmJacobianCharge::One] {
            let options = LmOptions {
                jacobian,
                ..LmOptions::default()
            };
            for source in ["sin(1.5*c0)", "sin(1.5*c0)+(0.25*c0)", "sqrt(1.0+(c0*c0))"] {
                for budget in [1usize, 2, 3, 4, 7, 12, 40, 200] {
                    let asked = Cell::new(0usize);
                    let mut guard = || {
                        asked.set(asked.get() + 1);
                        true
                    };
                    let actual = public(source, &rows, &targets, options, budget, &mut guard);
                    let search =
                        Search::new(&profile, ExpressionScope::search_profile(1, false)).unwrap();
                    let core = MathExpr::parse(source)
                        .unwrap()
                        .optimize_constants_lm(
                            &Admission::Profile(search),
                            &rows,
                            &targets,
                            Objective::LinearScaled,
                            &Options {
                                jacobian_cost: match jacobian {
                                    LmJacobianCharge::Sites => JacobianCost::Sites,
                                    LmJacobianCharge::One => JacobianCost::One,
                                },
                                ..Options::default()
                            },
                            GuardedRefinementBudget {
                                max_evaluations: budget,
                                may_start: &mut || true,
                            },
                        )
                        .unwrap();
                    let trace = &actual.trace;
                    let outcome = &actual.outcome;
                    let label = format!("{source} {jacobian:?} {budget}");
                    assert_eq!(trace.status, core.status.public(), "{label}");
                    assert_eq!(trace.changed, core.changed, "{label}");
                    assert_eq!(trace.evaluations(), outcome.evaluations(), "{label}");
                    assert_eq!(trace.evaluations(), core.refinement.evaluations, "{label}");
                    assert!(trace.evaluations() <= budget, "{label}");
                    assert_eq!(trace.rejected_trials, outcome.rejected_trials(), "{label}");
                    assert_eq!(
                        trace.iterations.len(),
                        trace.trials + trace.solve_failures,
                        "{label}"
                    );
                    assert_eq!(
                        trace
                            .iterations
                            .iter()
                            .map(|i| i.mu.to_bits())
                            .collect::<Vec<_>>(),
                        core.run.as_ref().map_or_else(Vec::new, |r| r
                            .trace
                            .iter()
                            .map(|i| i.mu.to_bits())
                            .collect()),
                        "{label}"
                    );
                    // One guard question per charged native pass and per Jacobian.
                    assert_eq!(
                        asked.get(),
                        trace.initial + trace.trials + trace.jacobians + trace.rescore,
                        "{label}"
                    );
                    let refinement = outcome.refinement.as_ref().unwrap();
                    assert_eq!(
                        refinement.expression.to_source().unwrap(),
                        core.refinement.expression.to_source().unwrap(),
                        "{label}"
                    );
                    assert_eq!(
                        refinement.score.to_bits(),
                        core.refinement.score.to_bits(),
                        "{label}"
                    );
                    assert!(!outcome.stopped && outcome.failure.is_none(), "{label}");
                    // The admitted start is the refinement whenever the winner did not replace it.
                    if !trace.changed {
                        assert_eq!(
                            actual.start.to_source().unwrap(),
                            refinement.expression.to_source().unwrap(),
                            "{label}"
                        );
                    }
                }
            }
        }
        // The frequency fixture is recovered within a modest budget.
        let mut always = || true;
        let fit = public(
            "sin(1.5*c0)",
            &rows,
            &targets,
            LmOptions::default(),
            200,
            &mut always,
        );
        assert!(fit.trace.changed, "{:?}", fit.trace.status);
        let refined = fit.outcome.refinement.unwrap();
        let root = &refined.expression.root;
        let sites = literal_paths(root);
        assert_eq!(sites.len(), 1);
        let frequency = literal_value(at(root, &sites[0])).unwrap();
        assert!((frequency - 1.7).abs() <= 1e-6, "{frequency}");
        assert_eq!(
            fit.trace.rescored.map(f64::to_bits),
            Some(refined.score.to_bits())
        );
    }

    #[test]
    fn unsupported_refused_and_stopped_starts_report_no_refinement() {
        let (rows, targets) = sine_data();
        // An elementwise scope spends nothing and never asks the guard.
        let mut never_asked = || panic!("an unsupported shape asks nothing");
        let elementwise = MathExpr::parse("sin(1.5*x)")
            .unwrap()
            .optimize_constants_lm_guarded(
                &baseline(),
                ExpressionScope::search_profile(1, true),
                &rows,
                &targets,
                LmOptions::default(),
                GuardedRefinementBudget {
                    max_evaluations: 10,
                    may_start: &mut never_asked,
                },
            )
            .unwrap();
        assert_eq!(elementwise.trace.status, LmStatus::Unsupported);
        assert!(elementwise.outcome.refinement.is_none());
        assert_eq!(elementwise.outcome.evaluations(), 0);
        assert_eq!(elementwise.trace.evaluations(), 0);
        // A refused guard starts nothing.
        let mut refuse = || false;
        let stopped = public(
            "sin(1.5*c0)",
            &rows,
            &targets,
            LmOptions::default(),
            10,
            &mut refuse,
        );
        assert_eq!(stopped.trace.status, LmStatus::Stopped);
        assert!(stopped.outcome.stopped && stopped.outcome.refinement.is_none());
        assert_eq!(stopped.outcome.evaluations(), 0);
        assert_eq!(stopped.trace.evaluations(), 0);
        // A start with a Domain fault on a training row is a failure that cost one evaluation.
        let mut always = || true;
        let refused = public(
            "sqrt(c0-5.0)",
            &rows,
            &targets,
            LmOptions::default(),
            10,
            &mut always,
        );
        assert_eq!(refused.trace.status, LmStatus::InitialRefused);
        assert!(refused.outcome.refinement.is_none() && !refused.outcome.stopped);
        assert_eq!(refused.outcome.failure.as_deref(), Some(INITIAL_REFUSED));
        assert_eq!(refused.outcome.evaluations(), 1);
        assert_eq!(refused.outcome.rejected_trials(), 1);
        assert_eq!(
            (refused.trace.evaluations(), refused.trace.rejected_trials),
            (1, 1)
        );
        // Invalid options are an error, as in the core.
        let mut always = || true;
        assert!(MathExpr::parse("sin(1.5*c0)")
            .unwrap()
            .optimize_constants_lm_guarded(
                &baseline(),
                ExpressionScope::search_profile(1, false),
                &rows,
                &targets,
                LmOptions {
                    tau: 0.0,
                    ..LmOptions::default()
                },
                GuardedRefinementBudget {
                    max_evaluations: 10,
                    may_start: &mut always,
                },
            )
            .is_err());
    }

    /// A guard that refuses partway keeps the start, is never asked again, and the trace still
    /// accounts for what was charged.
    #[test]
    fn a_partway_refusal_keeps_the_start_and_its_charges() {
        let (rows, targets) = sine_data();
        let mut always = || true;
        let full = public(
            "sin(1.5*c0)",
            &rows,
            &targets,
            LmOptions::default(),
            200,
            &mut always,
        );
        let total =
            full.trace.initial + full.trace.trials + full.trace.jacobians + full.trace.rescore;
        for allowed in 1..total {
            let asked = Cell::new(0usize);
            let mut guard = || {
                asked.set(asked.get() + 1);
                asked.get() <= allowed
            };
            let result = public(
                "sin(1.5*c0)",
                &rows,
                &targets,
                LmOptions::default(),
                200,
                &mut guard,
            );
            assert_eq!(asked.get(), allowed + 1, "{allowed}");
            assert_eq!(result.trace.status, LmStatus::Stopped, "{allowed}");
            assert!(result.outcome.stopped && !result.trace.changed, "{allowed}");
            assert_eq!(result.trace.evaluations(), result.outcome.evaluations());
            let kept = result.outcome.refinement.unwrap();
            assert_eq!(
                kept.expression.to_source().unwrap(),
                result.start.to_source().unwrap()
            );
        }
    }

    #[test]
    fn the_trace_replays_bit_for_bit() {
        let (rows, targets) = sine_data();
        let run = || {
            let mut always = || true;
            public(
                "sin(1.5*c0)+(0.25*c0)",
                &rows,
                &targets,
                LmOptions::default(),
                120,
                &mut always,
            )
        };
        let (a, b) = (run(), run());
        let bits = |t: &LmTrace| {
            t.iterations
                .iter()
                .map(|i| {
                    (
                        i.mu.to_bits(),
                        i.objective.to_bits(),
                        i.trial.map(f64::to_bits),
                        i.accepted,
                        i.solved,
                    )
                })
                .collect::<Vec<_>>()
        };
        assert!(!a.trace.iterations.is_empty());
        assert_eq!(bits(&a.trace), bits(&b.trace));
        assert_eq!(a.trace, b.trace);
        let (x, y) = (a.outcome.refinement.unwrap(), b.outcome.refinement.unwrap());
        assert_eq!(
            x.expression.to_source().unwrap(),
            y.expression.to_source().unwrap()
        );
        assert_eq!(x.score.to_bits(), y.score.to_bits());
    }

    /// `literal_sites` counts, before any fit, the sites the fit reports after its admission
    /// (negative and nested literals included), and the shape is unsupported exactly outside
    /// `1..=LM_MAX_SITES`.
    #[test]
    fn literal_sites_are_the_sites_the_fit_reports() {
        let (rows, targets) = sine_data();
        for (source, expected) in [
            ("sin(1.5*c0)", 1usize),
            ("sin(1.5*c0)+(0.25*c0)", 2),
            ("(-1.5*c0)+2.0", 2),
            ("(c0*-0.5)-(0.5-1.5)", 3),
            ("-(-1.5)*c0", 1),
            ("sin(c0)", 0),
        ] {
            let expression = MathExpr::parse(source).unwrap();
            assert_eq!(expression.literal_sites(), expected, "{source}");
            let mut always = || true;
            let fit = public(
                source,
                &rows,
                &targets,
                LmOptions::default(),
                30,
                &mut always,
            );
            assert_eq!(
                fit.trace.status == LmStatus::Unsupported,
                !(1..=LM_MAX_SITES).contains(&expected),
                "{source}"
            );
            assert_eq!(fit.trace.sites, expected, "{source}");
            assert_eq!(fit.start.literal_sites(), expected, "{source}");
        }
    }
}
// laneD:B17-lm-refinement end
