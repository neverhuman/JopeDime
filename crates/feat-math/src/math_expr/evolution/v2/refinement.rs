use super::*;
use crate::kernel::expression_search::{ConstantObjective, ConstantRefinement};

struct Training<'a, 'b> {
    rows: &'a [Vec<f64>],
    targets: &'a [f64],
    objective: ConstantObjective<'b>,
    rejected: FaultSet,
    elementwise: bool,
}

impl Training<'_, '_> {
    fn score(&mut self, expression: &MathExpr) -> Result<(f64, Option<[f64; 2]>)> {
        let compiled = expression.compile()?;
        let mut values = Vec::with_capacity(self.rows.len());
        for row in self.rows {
            if self.elementwise {
                if row.is_empty() {
                    bail!("elementwise training row is empty");
                }
                let (_, faults) = compiled.map_row_traced(row);
                if faults.iter().any(|f| f.intersects(self.rejected)) {
                    bail!("faulting constant trial");
                }
            } else {
                let (value, faults) = compiled.eval_row_traced(row);
                if faults.intersects(self.rejected) {
                    bail!("faulting constant trial");
                }
                values.push(value);
            }
        }
        match &mut self.objective {
            ConstantObjective::LinearScaled => {
                if self.elementwise {
                    bail!("LinearScaled needs a row expression; use an elementwise callback");
                }
                linear_scaled(&values, self.targets)
            }
            ConstantObjective::Callback(callback) => {
                let score = callback(expression, self.rows, self.targets)?;
                if !score.is_finite() {
                    bail!("nonfinite training objective");
                }
                Ok((score, None))
            }
        }
    }
}

fn linear_scaled(values: &[f64], targets: &[f64]) -> Result<(f64, Option<[f64; 2]>)> {
    let n = values.len() as f64;
    let mean_f = values.iter().sum::<f64>() / n;
    let mean_y = targets.iter().sum::<f64>() / n;
    let variance = values.iter().map(|f| (f - mean_f).powi(2)).sum::<f64>();
    let covariance = values
        .iter()
        .zip(targets)
        .map(|(f, y)| (f - mean_f) * (y - mean_y))
        .sum::<f64>();
    let slope = if variance == 0.0 {
        0.0
    } else {
        covariance / variance
    };
    let intercept = mean_y - slope * mean_f;
    let score = values
        .iter()
        .zip(targets)
        .map(|(f, y)| (intercept + slope * f - y).powi(2))
        .sum::<f64>()
        / n;
    if !score.is_finite() || !intercept.is_finite() || !slope.is_finite() {
        bail!("nonfinite linear-scaled objective");
    }
    Ok((score, Some([intercept, slope])))
}

impl MathExpr {
    /// Deterministic coordinate refinement over supplied TRAINING data only.
    /// Budget counts every attempted candidate, including traced-fault refusals.
    pub fn optimize_constants_v2(
        &self,
        profile: &SearchProfile,
        scope: ExpressionScope,
        rows: &[Vec<f64>],
        targets: &[f64],
        objective: ConstantObjective<'_>,
        budget: usize,
    ) -> Result<ConstantRefinement> {
        if rows.is_empty()
            || rows.len() != targets.len()
            || budget == 0
            || targets.iter().any(|v| !v.is_finite())
        {
            bail!("constant fitting needs finite aligned training targets and a budget");
        }
        let search = Search::new(profile, scope)?;
        let expression = search.finish(self.root.clone())?;
        let rejected = FaultSeverity::Domain
            .classes()
            .union(FaultSeverity::Hard.classes());
        let mut training = Training {
            rows,
            targets,
            objective,
            rejected,
            elementwise: scope.elementwise,
        };
        let (score, linear_scale) = training.score(&expression)?;
        let mut best = ConstantRefinement {
            expression,
            evaluations: 1,
            rejected_trials: 0,
            score,
            linear_scale,
        };
        let mut step = 1.0;
        while best.evaluations < budget && step >= 1e-12 {
            let mut improved = false;
            let sites = literal_paths(&best.expression.root);
            if sites.is_empty() {
                break;
            }
            for path in sites {
                let value = literal_value(at(&best.expression.root, &path)).unwrap();
                for delta in [-step, step] {
                    if best.evaluations == budget {
                        return Ok(best);
                    }
                    best.evaluations += 1;
                    let mut root = best.expression.root.clone();
                    replace(&mut root, &path, literal(value + delta));
                    let trial = search.finish(root).and_then(|expression| {
                        training.score(&expression).map(|score| (expression, score))
                    });
                    match trial {
                        Ok((expression, (score, linear_scale))) if score < best.score => {
                            best.expression = expression;
                            best.score = score;
                            best.linear_scale = linear_scale;
                            improved = true;
                        }
                        Err(_) => best.rejected_trials += 1,
                        _ => {}
                    }
                }
            }
            if !improved {
                step *= 0.5;
            }
        }
        Ok(best)
    }
}

mod guarded;
// B17 bounded LM constant fitting: staged with no caller until B11 and B16 merge (OWNER v1.3).
#[cfg(test)]
mod lm;
