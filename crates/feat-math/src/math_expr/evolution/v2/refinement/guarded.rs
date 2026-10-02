//! E6 trial admission. The existing optimizer and numerical scorer stay intact.
use super::*;
use crate::kernel::expression_search::{GuardedConstantRefinement, GuardedRefinementBudget};

type ScoredTrial = Result<(MathExpr, (f64, Option<[f64; 2]>))>;

impl MathExpr {
    /// The v2 coordinate search with an explicit check before every trial.
    /// With an always-true guard, results and counts match the existing API.
    /// A stopped search retains its best already measured expression. Initial
    /// scoring failures are reported with one rejected trial and no score.
    pub fn optimize_constants_v2_guarded(
        &self,
        profile: &SearchProfile,
        scope: ExpressionScope,
        rows: &[Vec<f64>],
        targets: &[f64],
        objective: ConstantObjective<'_>,
        mut budget: GuardedRefinementBudget<'_>,
    ) -> Result<GuardedConstantRefinement> {
        validate_training(rows, targets, budget.max_evaluations)?;
        let search = Search::new(profile, scope)?;
        let expression = search.finish(self.root.clone())?;
        let mut training = Training {
            rows,
            targets,
            objective,
            rejected: FaultSeverity::Domain
                .classes()
                .union(FaultSeverity::Hard.classes()),
            elementwise: scope.elementwise,
        };
        if !(budget.may_start)() {
            return Ok(outcome(None, true));
        }
        match initial(&mut training, expression) {
            Ok(best) => Ok(advance(&search, &mut training, best, &mut budget)),
            Err(error) => Ok(GuardedConstantRefinement {
                refinement: None,
                stopped: false,
                failure: Some(format!("{error:#}")),
            }),
        }
    }
}

fn validate_training(rows: &[Vec<f64>], targets: &[f64], budget: usize) -> Result<()> {
    if rows.is_empty()
        || rows.len() != targets.len()
        || budget == 0
        || targets.iter().any(|v| !v.is_finite())
    {
        bail!("constant fitting needs finite aligned training targets and a budget");
    }
    Ok(())
}

fn initial(training: &mut Training<'_, '_>, expression: MathExpr) -> Result<ConstantRefinement> {
    let (score, linear_scale) = training.score(&expression)?;
    Ok(ConstantRefinement {
        expression,
        evaluations: 1,
        rejected_trials: 0,
        score,
        linear_scale,
    })
}

fn outcome(refinement: Option<ConstantRefinement>, stopped: bool) -> GuardedConstantRefinement {
    GuardedConstantRefinement {
        refinement,
        stopped,
        failure: None,
    }
}

fn advance(
    search: &Search<'_>,
    training: &mut Training<'_, '_>,
    mut best: ConstantRefinement,
    budget: &mut GuardedRefinementBudget<'_>,
) -> GuardedConstantRefinement {
    let mut coordinates = Coordinates::new(&best.expression.root);
    while best.evaluations < budget.max_evaluations {
        let Some((path, value)) = coordinates.next(&best.expression.root) else {
            break;
        };
        if !(budget.may_start)() {
            return outcome(Some(best), true);
        }
        best.evaluations += 1;
        let mut root = best.expression.root.clone();
        replace(&mut root, &path, literal(value));
        let trial = search
            .finish(root)
            .and_then(|expression| training.score(&expression).map(|score| (expression, score)));
        coordinates.improved |= retain(&mut best, trial);
    }
    outcome(Some(best), false)
}

fn retain(best: &mut ConstantRefinement, trial: ScoredTrial) -> bool {
    let Ok((expression, (score, linear_scale))) = trial else {
        best.rejected_trials += 1;
        return false;
    };
    if score >= best.score {
        return false;
    }
    best.expression = expression;
    best.score = score;
    best.linear_scale = linear_scale;
    true
}

/// Keep the base literal fixed for its negative/positive pair. Re-read the
/// next site's value from the latest winner, just as the existing search does.
struct Coordinates {
    paths: Vec<Vec<usize>>,
    site: usize,
    positive: bool,
    base: f64,
    step: f64,
    improved: bool,
}

impl Coordinates {
    fn new(root: &Expr) -> Self {
        Self {
            paths: literal_paths(root),
            site: 0,
            positive: false,
            base: 0.0,
            step: 1.0,
            improved: false,
        }
    }

    fn next(&mut self, root: &Expr) -> Option<(Vec<usize>, f64)> {
        if self.site == self.paths.len() {
            if !self.improved {
                self.step *= 0.5;
            }
            self.paths = literal_paths(root);
            self.site = 0;
            self.improved = false;
        }
        if self.paths.is_empty() || self.step < 1e-12 {
            return None;
        }
        let path = self.paths[self.site].clone();
        let value = if self.positive {
            self.site += 1;
            self.base + self.step
        } else {
            self.base = literal_value(at(root, &path)).unwrap();
            self.base - self.step
        };
        self.positive = !self.positive;
        Some((path, value))
    }
}
