//! Portable regression objectives. Configuration is data, never executable code.
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

pub const FITNESS_VERSION: &str = "jope.regression-fitness/v1";
pub const MAX_SPEC_BYTES: usize = 32 * 1024;
/// Complete schema shared by tooling and MCP discovery. Runtime validation also bounds recursion.
pub fn schema() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../schemas/jope.regression-fitness.v1.schema.json"
    ))
    .expect("committed fitness schema")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    Minimize,
    Maximize,
}

/// Every parameter is part of the serialized objective identity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "metric", rename_all = "snake_case", deny_unknown_fields)]
pub enum Fitness {
    Rmse {},
    Mse {},
    Mae {},
    MedianAbsoluteError {},
    MaxError {},
    Mape {
        epsilon: f64,
    },
    Smape {
        epsilon: f64,
    },
    Wape {},
    Msle {},
    Rmsle {},
    R2 {},
    ExplainedVariance {},
    Pearson {},
    Spearman {},
    Huber {
        delta: f64,
    },
    PseudoHuber {
        delta: f64,
    },
    LogCosh {},
    Pinball {
        quantile: f64,
    },
    Expectile {
        quantile: f64,
    },
    PoissonDeviance {},
    GammaDeviance {},
    TweedieDeviance {
        power: f64,
    },
    /// y is the realized simple return; clipped prediction is a signed position.
    MeanReturn {
        max_position: f64,
        cost: f64,
    },
    Sharpe {
        max_position: f64,
        cost: f64,
        target: f64,
        periods_per_year: f64,
    },
    Sortino {
        max_position: f64,
        cost: f64,
        target: f64,
        periods_per_year: f64,
    },
    Custom {
        name: String,
        direction: Direction,
        expression: Expression,
        aggregation: Aggregation,
    },
    /// Terms are normalized to minimization before the positive weighted sum.
    Composite {
        terms: Vec<Term>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Term {
    pub weight: f64,
    pub scale: f64,
    pub fitness: Fitness,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Expression {
    Y {},
    Pred {},
    Constant { value: f64 },
    Add { left: Box<Self>, right: Box<Self> },
    Sub { left: Box<Self>, right: Box<Self> },
    Mul { left: Box<Self>, right: Box<Self> },
    Div { left: Box<Self>, right: Box<Self> },
    Min { left: Box<Self>, right: Box<Self> },
    Max { left: Box<Self>, right: Box<Self> },
    Abs { arg: Box<Self> },
    Square { arg: Box<Self> },
    Sqrt { arg: Box<Self> },
    Ln { arg: Box<Self> },
    Exp { arg: Box<Self> },
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Aggregation {
    Mean {},
    Sum {},
    Max {},
    Quantile { quantile: f64 },
}

#[derive(Clone, Debug, Serialize)]
pub struct Score {
    pub value: f64,
    /// The selection machinery always minimizes this value.
    pub loss: f64,
    pub direction: Direction,
    pub fitness_sha256: String,
}

fn positive(x: f64) -> bool {
    x.is_finite() && x > 0.0
}
fn probability(x: f64) -> bool {
    x.is_finite() && x > 0.0 && x < 1.0
}
fn finite(x: f64) -> Result<f64> {
    ensure!(x.is_finite(), "non-finite fitness value");
    Ok(x)
}
fn mean(v: &[f64]) -> f64 {
    v.iter().map(|x| x / v.len() as f64).sum()
}
fn quantile(v: &[f64], q: f64) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(f64::total_cmp);
    let p = q * (v.len() - 1) as f64;
    let i = p.floor() as usize;
    v[i] + (v[p.ceil() as usize] - v[i]) * (p - i as f64)
}
fn variance(v: &[f64]) -> f64 {
    let m = mean(v);
    mean(&v.iter().map(|x| (x - m).powi(2)).collect::<Vec<_>>())
}
fn correlation(a: &[f64], b: &[f64]) -> Result<f64> {
    let am = mean(a);
    let bm = mean(b);
    let av = variance(a);
    let bv = variance(b);
    ensure!(
        av > 0.0 && bv > 0.0,
        "correlation undefined for a constant vector"
    );
    finite(
        mean(
            &a.iter()
                .zip(b)
                .map(|(x, y)| (x - am) * (y - bm))
                .collect::<Vec<_>>(),
        ) / av.sqrt()
            / bv.sqrt(),
    )
}
fn ranks(v: &[f64]) -> Vec<f64> {
    let mut order: Vec<_> = (0..v.len()).collect();
    order.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).expect("finite inputs"));
    let mut out = vec![0.0; v.len()];
    let mut i = 0;
    while i < order.len() {
        let mut j = i + 1;
        while j < order.len() && v[order[i]] == v[order[j]] {
            j += 1;
        }
        for k in i..j {
            out[order[k]] = (i + j - 1) as f64 / 2.0;
        }
        i = j;
    }
    out
}

impl Expression {
    fn validate(&self, depth: usize, nodes: &mut usize) -> Result<()> {
        *nodes += 1;
        ensure!(
            depth <= 16 && *nodes <= 256,
            "fitness expression exceeds depth/node limit"
        );
        match self {
            Self::Constant { value } => {
                finite(*value)?;
            }
            Self::Y {} | Self::Pred {} => (),
            Self::Abs { arg }
            | Self::Square { arg }
            | Self::Sqrt { arg }
            | Self::Ln { arg }
            | Self::Exp { arg } => arg.validate(depth + 1, nodes)?,
            Self::Add { left, right }
            | Self::Sub { left, right }
            | Self::Mul { left, right }
            | Self::Div { left, right }
            | Self::Min { left, right }
            | Self::Max { left, right } => {
                left.validate(depth + 1, nodes)?;
                right.validate(depth + 1, nodes)?;
            }
        }
        Ok(())
    }
    fn eval(&self, y: f64, p: f64) -> Result<f64> {
        finite(match self {
            Self::Y {} => y,
            Self::Pred {} => p,
            Self::Constant { value } => *value,
            Self::Abs { arg } => arg.eval(y, p)?.abs(),
            Self::Square { arg } => arg.eval(y, p)?.powi(2),
            Self::Sqrt { arg } => arg.eval(y, p)?.sqrt(),
            Self::Ln { arg } => arg.eval(y, p)?.ln(),
            Self::Exp { arg } => arg.eval(y, p)?.exp(),
            Self::Add { left, right } => left.eval(y, p)? + right.eval(y, p)?,
            Self::Sub { left, right } => left.eval(y, p)? - right.eval(y, p)?,
            Self::Mul { left, right } => left.eval(y, p)? * right.eval(y, p)?,
            Self::Div { left, right } => left.eval(y, p)? / right.eval(y, p)?,
            Self::Min { left, right } => left.eval(y, p)?.min(right.eval(y, p)?),
            Self::Max { left, right } => left.eval(y, p)?.max(right.eval(y, p)?),
        })
    }
}

impl Fitness {
    /// A short catalog name or a strict inline JSON specification. File access belongs to CLI.
    pub fn parse(text: &str) -> Result<Self> {
        ensure!(
            text.len() <= MAX_SPEC_BYTES,
            "fitness specification is too large"
        );
        let spec = if text.trim_start().starts_with('{') {
            serde_json::from_str(text)?
        } else {
            Self::catalog()
                .into_iter()
                .find(|(n, _)| *n == text)
                .map(|(_, s)| s)
                .ok_or_else(|| {
                    anyhow::anyhow!("unknown fitness {text:?}; use `jopedime fitness list`")
                })?
        };
        spec.validate()?;
        Ok(spec)
    }
    pub fn catalog() -> Vec<(&'static str, Self)> {
        vec![
            ("rmse", Self::Rmse {}),
            ("mse", Self::Mse {}),
            ("mae", Self::Mae {}),
            ("median_absolute_error", Self::MedianAbsoluteError {}),
            ("max_error", Self::MaxError {}),
            ("mape", Self::Mape { epsilon: 1e-8 }),
            ("smape", Self::Smape { epsilon: 1e-8 }),
            ("wape", Self::Wape {}),
            ("msle", Self::Msle {}),
            ("rmsle", Self::Rmsle {}),
            ("r2", Self::R2 {}),
            ("explained_variance", Self::ExplainedVariance {}),
            ("pearson", Self::Pearson {}),
            ("spearman", Self::Spearman {}),
            ("huber", Self::Huber { delta: 1.0 }),
            ("pseudo_huber", Self::PseudoHuber { delta: 1.0 }),
            ("log_cosh", Self::LogCosh {}),
            ("pinball", Self::Pinball { quantile: 0.5 }),
            ("expectile", Self::Expectile { quantile: 0.5 }),
            ("poisson_deviance", Self::PoissonDeviance {}),
            ("gamma_deviance", Self::GammaDeviance {}),
            ("tweedie_deviance", Self::TweedieDeviance { power: 1.5 }),
            (
                "mean_return",
                Self::MeanReturn {
                    max_position: 1.0,
                    cost: 0.0,
                },
            ),
            (
                "sharpe",
                Self::Sharpe {
                    max_position: 1.0,
                    cost: 0.0,
                    target: 0.0,
                    periods_per_year: 252.0,
                },
            ),
            (
                "sortino",
                Self::Sortino {
                    max_position: 1.0,
                    cost: 0.0,
                    target: 0.0,
                    periods_per_year: 252.0,
                },
            ),
        ]
    }
    pub fn name(&self) -> &str {
        match self {
            Self::Rmse {} => "rmse",
            Self::Mse {} => "mse",
            Self::Mae {} => "mae",
            Self::MedianAbsoluteError {} => "median_absolute_error",
            Self::MaxError {} => "max_error",
            Self::Mape { .. } => "mape",
            Self::Smape { .. } => "smape",
            Self::Wape {} => "wape",
            Self::Msle {} => "msle",
            Self::Rmsle {} => "rmsle",
            Self::R2 {} => "r2",
            Self::ExplainedVariance {} => "explained_variance",
            Self::Pearson {} => "pearson",
            Self::Spearman {} => "spearman",
            Self::Huber { .. } => "huber",
            Self::PseudoHuber { .. } => "pseudo_huber",
            Self::LogCosh {} => "log_cosh",
            Self::Pinball { .. } => "pinball",
            Self::Expectile { .. } => "expectile",
            Self::PoissonDeviance {} => "poisson_deviance",
            Self::GammaDeviance {} => "gamma_deviance",
            Self::TweedieDeviance { .. } => "tweedie_deviance",
            Self::MeanReturn { .. } => "mean_return",
            Self::Sharpe { .. } => "sharpe",
            Self::Sortino { .. } => "sortino",
            Self::Custom { name, .. } => name,
            Self::Composite { .. } => "composite",
        }
    }
    pub fn direction(&self) -> Direction {
        match self {
            Self::R2 {}
            | Self::ExplainedVariance {}
            | Self::Pearson {}
            | Self::Spearman {}
            | Self::MeanReturn { .. }
            | Self::Sharpe { .. }
            | Self::Sortino { .. } => Direction::Maximize,
            Self::Custom { direction, .. } => *direction,
            _ => Direction::Minimize,
        }
    }
    pub fn requires_order(&self) -> bool {
        match self {
            Self::MeanReturn { .. } | Self::Sharpe { .. } | Self::Sortino { .. } => true,
            Self::Composite { terms } => terms.iter().any(|t| t.fitness.requires_order()),
            _ => false,
        }
    }
    pub fn validate(&self) -> Result<()> {
        self.validate_at(0, &mut 0)?;
        ensure!(
            serde_json::to_vec(self)?.len() <= MAX_SPEC_BYTES,
            "fitness specification is too large"
        );
        Ok(())
    }
    fn validate_at(&self, depth: usize, nodes: &mut usize) -> Result<()> {
        *nodes += 1;
        ensure!(
            depth <= 8 && *nodes <= 64,
            "fitness composition exceeds depth/node limit"
        );
        match self {
            Self::Mape { epsilon } | Self::Smape { epsilon } => {
                ensure!(positive(*epsilon), "epsilon must be positive and finite")
            }
            Self::Huber { delta } | Self::PseudoHuber { delta } => {
                ensure!(positive(*delta), "delta must be positive and finite")
            }
            Self::Pinball { quantile } | Self::Expectile { quantile } => {
                ensure!(probability(*quantile), "quantile must be in (0,1)")
            }
            Self::TweedieDeviance { power } => ensure!(
                power.is_finite() && *power > 1.0 && *power < 2.0,
                "Tweedie power must be in (1,2); use Poisson/Gamma at endpoints"
            ),
            Self::MeanReturn { max_position, cost }
            | Self::Sharpe {
                max_position, cost, ..
            }
            | Self::Sortino {
                max_position, cost, ..
            } => ensure!(
                positive(*max_position) && cost.is_finite() && *cost >= 0.0,
                "position must be positive; cost must be nonnegative; both finite"
            ),
            Self::Custom {
                name,
                expression,
                aggregation,
                ..
            } => {
                ensure!(
                    !name.is_empty()
                        && name.len() <= 64
                        && name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_'),
                    "custom name needs 1..64 ASCII letters, digits or underscores"
                );
                expression.validate(0, &mut 0)?;
                if let Aggregation::Quantile { quantile } = aggregation {
                    ensure!(
                        quantile.is_finite() && (0.0..=1.0).contains(quantile),
                        "aggregation quantile must be in [0,1]"
                    );
                }
            }
            Self::Composite { terms } => {
                ensure!(
                    !terms.is_empty() && terms.len() <= 16,
                    "composite needs 1..16 terms"
                );
                for t in terms {
                    ensure!(
                        positive(t.weight) && positive(t.scale),
                        "term weight and scale must be positive and finite"
                    );
                    t.fitness.validate_at(depth + 1, nodes)?;
                }
            }
            _ => (),
        }
        if let Self::Sharpe {
            target,
            periods_per_year,
            ..
        }
        | Self::Sortino {
            target,
            periods_per_year,
            ..
        } = self
        {
            ensure!(
                target.is_finite() && positive(*periods_per_year),
                "target must be finite and periods_per_year positive"
            );
        }
        Ok(())
    }
    pub fn identity(&self) -> Result<String> {
        self.validate()?;
        Ok(crate::sha256_hex(&serde_json::to_vec(&(
            FITNESS_VERSION,
            self,
        ))?))
    }
    pub fn evaluate(&self, y: &[f64], pred: &[f64]) -> Result<Score> {
        self.validate()?;
        ensure!(
            !y.is_empty() && y.len() == pred.len(),
            "fitness needs equal, nonempty target and prediction vectors"
        );
        ensure!(
            y.iter().chain(pred).all(|x| x.is_finite()),
            "fitness inputs must be finite"
        );
        let value = finite(self.value(y, pred)?)?;
        let direction = self.direction();
        Ok(Score {
            value,
            loss: if direction == Direction::Maximize {
                -value
            } else {
                value
            },
            direction,
            fitness_sha256: self.identity()?,
        })
    }
    fn value(&self, y: &[f64], p: &[f64]) -> Result<f64> {
        let errors: Vec<_> = y.iter().zip(p).map(|(y, p)| y - p).collect();
        if !matches!(
            self,
            Self::Custom { .. }
                | Self::Composite { .. }
                | Self::MeanReturn { .. }
                | Self::Sharpe { .. }
                | Self::Sortino { .. }
                | Self::Pearson {}
                | Self::Spearman {}
        ) {
            ensure!(
                errors.iter().all(|e| e.is_finite()),
                "non-finite regression residual"
            );
        }
        let squared = || mean(&errors.iter().map(|e| e * e).collect::<Vec<_>>());
        let abs: Vec<_> = errors.iter().map(|e| e.abs()).collect();
        let row_loss = |f: &dyn Fn(f64, f64) -> f64| {
            mean(&y.iter().zip(p).map(|(&y, &p)| f(y, p)).collect::<Vec<_>>())
        };
        Ok(match self {
            Self::Rmse {} => squared().sqrt(),
            Self::Mse {} => squared(),
            Self::Mae {} => mean(&abs),
            Self::MedianAbsoluteError {} => quantile(&abs, 0.5),
            Self::MaxError {} => abs.iter().copied().fold(0.0, f64::max),
            Self::Mape { epsilon } => row_loss(&|y, p| (y - p).abs() / y.abs().max(*epsilon)),
            Self::Smape { epsilon } => {
                row_loss(&|y, p| 2.0 * (y - p).abs() / (y.abs() + p.abs()).max(*epsilon))
            }
            Self::Wape {} => {
                let den = mean(&y.iter().map(|y| y.abs()).collect::<Vec<_>>());
                ensure!(den > 0.0, "WAPE undefined for all-zero targets");
                mean(&abs) / den
            }
            Self::Msle {} | Self::Rmsle {} => {
                ensure!(
                    y.iter().chain(p).all(|x| *x >= 0.0),
                    "log error needs nonnegative targets and predictions"
                );
                let v = row_loss(&|y, p| (y.ln_1p() - p.ln_1p()).powi(2));
                if matches!(self, Self::Rmsle {}) {
                    v.sqrt()
                } else {
                    v
                }
            }
            Self::R2 {} | Self::ExplainedVariance {} => {
                let den = variance(y);
                ensure!(
                    y.len() >= 2 && den > 0.0,
                    "R2/explained variance undefined for constant or singleton targets"
                );
                1.0 - if matches!(self, Self::R2 {}) {
                    squared() / den
                } else {
                    variance(&errors) / den
                }
            }
            Self::Pearson {} => correlation(y, p)?,
            Self::Spearman {} => correlation(&ranks(y), &ranks(p))?,
            Self::Huber { delta } => mean(
                &abs.iter()
                    .map(|e| {
                        if *e <= *delta {
                            0.5 * e * e
                        } else {
                            delta * (e - 0.5 * delta)
                        }
                    })
                    .collect::<Vec<_>>(),
            ),
            Self::PseudoHuber { delta } => mean(
                &errors
                    .iter()
                    .map(|e| delta * delta * ((1.0 + (e / delta).powi(2)).sqrt() - 1.0))
                    .collect::<Vec<_>>(),
            ),
            Self::LogCosh {} => mean(
                &abs.iter()
                    .map(|e| e + (-2.0 * e).exp().ln_1p() - std::f64::consts::LN_2)
                    .collect::<Vec<_>>(),
            ),
            Self::Pinball { quantile } => mean(
                &errors
                    .iter()
                    .map(|e| {
                        if *e >= 0.0 {
                            quantile * e
                        } else {
                            (quantile - 1.0) * e
                        }
                    })
                    .collect::<Vec<_>>(),
            ),
            Self::Expectile { quantile } => mean(
                &errors
                    .iter()
                    .map(|e| {
                        if *e >= 0.0 {
                            quantile * e * e
                        } else {
                            (1.0 - quantile) * e * e
                        }
                    })
                    .collect::<Vec<_>>(),
            ),
            Self::PoissonDeviance {} => {
                ensure!(
                    y.iter().all(|x| *x >= 0.0) && p.iter().all(|x| *x > 0.0),
                    "Poisson requires y>=0 and pred>0"
                );
                row_loss(&|y, p| {
                    2.0 * (if y == 0.0 {
                        p
                    } else {
                        y * (y / p).ln() - y + p
                    })
                })
            }
            Self::GammaDeviance {} => {
                ensure!(
                    y.iter().chain(p).all(|x| *x > 0.0),
                    "Gamma requires y>0 and pred>0"
                );
                row_loss(&|y, p| 2.0 * (-(y / p).ln() + y / p - 1.0))
            }
            Self::TweedieDeviance { power } => {
                ensure!(
                    y.iter().all(|x| *x >= 0.0) && p.iter().all(|x| *x > 0.0),
                    "Tweedie requires y>=0 and pred>0"
                );
                let q = *power;
                row_loss(&|y, p| {
                    2.0 * (y.powf(2.0 - q) / ((1.0 - q) * (2.0 - q))
                        - y * p.powf(1.0 - q) / (1.0 - q)
                        + p.powf(2.0 - q) / (2.0 - q))
                })
            }
            Self::MeanReturn { max_position, cost }
            | Self::Sharpe {
                max_position, cost, ..
            }
            | Self::Sortino {
                max_position, cost, ..
            } => {
                let mut last = 0.0;
                let returns: Vec<_> = y
                    .iter()
                    .zip(p)
                    .map(|(y, p)| {
                        let position = p.clamp(-*max_position, *max_position);
                        let r = position * y - cost * (position - last).abs();
                        last = position;
                        r
                    })
                    .collect();
                ensure!(
                    returns.iter().all(|x| x.is_finite()),
                    "non-finite strategy return"
                );
                match self {
                    Self::MeanReturn { .. } => mean(&returns),
                    Self::Sharpe {
                        target,
                        periods_per_year,
                        ..
                    }
                    | Self::Sortino {
                        target,
                        periods_per_year,
                        ..
                    } => {
                        let den = if matches!(self, Self::Sortino { .. }) {
                            mean(
                                &returns
                                    .iter()
                                    .map(|r| (r - target).min(0.0).powi(2))
                                    .collect::<Vec<_>>(),
                            )
                            .sqrt()
                        } else {
                            variance(&returns).sqrt()
                        };
                        ensure!(den > 0.0, "risk ratio undefined with zero risk denominator");
                        (mean(&returns) - target) / den * periods_per_year.sqrt()
                    }
                    _ => unreachable!(),
                }
            }
            Self::Custom {
                expression,
                aggregation,
                ..
            } => {
                let v = y
                    .iter()
                    .zip(p)
                    .map(|(&y, &p)| expression.eval(y, p))
                    .collect::<Result<Vec<_>>>()?;
                match aggregation {
                    Aggregation::Mean {} => mean(&v),
                    Aggregation::Sum {} => v.iter().sum(),
                    Aggregation::Max {} => v.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                    Aggregation::Quantile { quantile: q } => quantile(&v, *q),
                }
            }
            Self::Composite { terms } => terms
                .iter()
                .map(|t| Ok(t.weight * t.fitness.evaluate(y, p)?.loss / t.scale))
                .collect::<Result<Vec<_>>>()?
                .iter()
                .sum(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn near(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-10, "{a} != {b}");
    }
    fn error_expr() -> Expression {
        Expression::Sub {
            left: Box::new(Expression::Y {}),
            right: Box::new(Expression::Pred {}),
        }
    }
    #[test]
    fn standard_metrics_match_independent_hand_calculations() {
        for (spec, expected) in [
            (Fitness::Rmse {}, (2.0f64 / 3.0).sqrt()),
            (Fitness::Mse {}, 2.0 / 3.0),
            (Fitness::Mae {}, 2.0 / 3.0),
            (Fitness::MedianAbsoluteError {}, 1.0),
            (Fitness::MaxError {}, 1.0),
            (Fitness::Mape { epsilon: 1e-8 }, 4.0 / 9.0),
            (Fitness::Smape { epsilon: 1e-8 }, 16.0 / 45.0),
            (Fitness::Wape {}, 1.0 / 3.0),
            (Fitness::R2 {}, 0.0),
            (Fitness::ExplainedVariance {}, 0.0),
            (Fitness::Huber { delta: 1.0 }, 1.0 / 3.0),
            (
                Fitness::PseudoHuber { delta: 1.0 },
                2.0 * (2.0f64.sqrt() - 1.0) / 3.0,
            ),
            (Fitness::LogCosh {}, 2.0 * 1.0f64.cosh().ln() / 3.0),
            (Fitness::Pinball { quantile: 0.9 }, 1.0 / 3.0),
            (Fitness::Expectile { quantile: 0.8 }, 1.0 / 3.0),
        ] {
            near(
                spec.evaluate(&[1.0, 2.0, 3.0], &[2.0, 2.0, 2.0])
                    .unwrap()
                    .value,
                expected,
            );
        }
    }
    #[test]
    fn log_and_deviance_metrics_match_independent_calculations() {
        near(
            Fitness::Msle {}
                .evaluate(&[0.0, 3.0], &[1.0, 1.0])
                .unwrap()
                .value,
            2.0f64.ln().powi(2),
        );
        near(
            Fitness::Rmsle {}
                .evaluate(&[0.0, 3.0], &[1.0, 1.0])
                .unwrap()
                .value,
            2.0f64.ln(),
        );
        near(
            Fitness::PoissonDeviance {}
                .evaluate(&[0.0, 1.0, 2.0], &[1.0, 1.0, 1.0])
                .unwrap()
                .value,
            4.0 * 2.0f64.ln() / 3.0,
        );
        near(
            Fitness::GammaDeviance {}
                .evaluate(&[1.0, 2.0], &[2.0, 1.0])
                .unwrap()
                .value,
            0.5,
        );
        near(
            Fitness::TweedieDeviance { power: 1.5 }
                .evaluate(&[0.0, 1.0, 4.0], &[1.0, 1.0, 1.0])
                .unwrap()
                .value,
            8.0 / 3.0,
        );
    }
    #[test]
    fn correlations_handle_ties_signed_zero_and_direction() {
        near(
            Fitness::Pearson {}
                .evaluate(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0])
                .unwrap()
                .loss,
            1.0,
        );
        near(
            Fitness::Spearman {}
                .evaluate(&[0.0, -0.0, 2.0], &[2.0, 2.0, 5.0])
                .unwrap()
                .loss,
            -1.0,
        );
        let r = Fitness::R2 {}
            .evaluate(&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0])
            .unwrap();
        assert_eq!(r.direction, Direction::Maximize);
        near(r.value, 1.0);
        near(r.loss, -1.0);
    }
    #[test]
    fn malformed_nonfinite_and_undefined_values_never_become_winners() {
        for (y, p) in [
            (vec![], vec![]),
            (vec![1.0], vec![]),
            (vec![f64::NAN], vec![0.0]),
            (vec![1.0], vec![f64::INFINITY]),
        ] {
            assert!(Fitness::Rmse {}.evaluate(&y, &p).is_err());
        }
        assert!(Fitness::Mse {}.evaluate(&[f64::MAX], &[-f64::MAX]).is_err());
        assert!(Fitness::MedianAbsoluteError {}
            .evaluate(&[f64::MAX, 1.0, 2.0], &[-f64::MAX, 1.0, 2.0])
            .is_err());
        for f in [
            Fitness::R2 {},
            Fitness::ExplainedVariance {},
            Fitness::Pearson {},
            Fitness::Spearman {},
            Fitness::Wape {},
        ] {
            assert!(f.evaluate(&[0.0, 0.0], &[1.0, 2.0]).is_err(), "{f:?}");
        }
        assert!(Fitness::Msle {}.evaluate(&[-1.0], &[1.0]).is_err());
        assert!(Fitness::PoissonDeviance {}
            .evaluate(&[1.0], &[0.0])
            .is_err());
        assert!(Fitness::GammaDeviance {}.evaluate(&[0.0], &[1.0]).is_err());
        assert!(Fitness::parse("{\"metric\":\"mae\",\"typo\":1}").is_err());
        assert!(Fitness::parse("import os").is_err());
    }
    #[test]
    fn configuration_parameters_and_expression_limits_are_enforced() {
        for f in [
            Fitness::Huber { delta: 0.0 },
            Fitness::Mape { epsilon: f64::NAN },
            Fitness::Pinball { quantile: 1.0 },
            Fitness::Expectile { quantile: -0.1 },
            Fitness::TweedieDeviance { power: 2.0 },
            Fitness::Composite { terms: vec![] },
        ] {
            assert!(f.validate().is_err());
        }
        let mut expr = Expression::Y {};
        for _ in 0..18 {
            expr = Expression::Abs {
                arg: Box::new(expr),
            };
        }
        assert!(Fitness::Custom {
            name: "deep".into(),
            direction: Direction::Minimize,
            expression: expr,
            aggregation: Aggregation::Mean {}
        }
        .validate()
        .is_err());
        assert!(Fitness::parse(&" ".repeat(MAX_SPEC_BYTES + 1)).is_err());
        let f = Fitness::Custom {
            name: "zero_division".into(),
            direction: Direction::Minimize,
            expression: Expression::Div {
                left: Box::new(Expression::Y {}),
                right: Box::new(Expression::Constant { value: 0.0 }),
            },
            aggregation: Aggregation::Mean {},
        };
        assert!(f.evaluate(&[1.0], &[0.0]).is_err());
    }
    #[test]
    fn custom_aggregations_and_scaled_composition_preserve_semantics() {
        let expr = Expression::Abs {
            arg: Box::new(error_expr()),
        };
        for (aggregation, expected) in [
            (Aggregation::Mean {}, 1.0),
            (Aggregation::Sum {}, 3.0),
            (Aggregation::Max {}, 2.0),
            (Aggregation::Quantile { quantile: 0.75 }, 1.5),
        ] {
            let f = Fitness::Custom {
                name: "business_loss".into(),
                direction: Direction::Minimize,
                expression: expr.clone(),
                aggregation,
            };
            near(
                f.evaluate(&[0.0, 1.0, 2.0], &[0.0, 0.0, 0.0])
                    .unwrap()
                    .value,
                expected,
            );
        }
        let f = Fitness::Composite {
            terms: vec![
                Term {
                    weight: 2.0,
                    scale: 4.0,
                    fitness: Fitness::Mae {},
                },
                Term {
                    weight: 1.0,
                    scale: 2.0,
                    fitness: Fitness::R2 {},
                },
            ],
        };
        near(
            f.evaluate(&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0]).unwrap().loss,
            -0.5,
        );
    }
    #[test]
    fn return_ratios_use_positions_turnover_full_sample_downside_and_annualization() {
        let y = [0.1, -0.2, 0.3];
        let p = [1.0, 1.0, 1.0];
        let sortino = Fitness::Sortino {
            max_position: 1.0,
            cost: 0.0,
            target: 0.0,
            periods_per_year: 1.0,
        };
        near(
            sortino.evaluate(&y, &p).unwrap().value,
            (0.2 / 3.0) / (0.04f64 / 3.0).sqrt(),
        );
        let sharpe = Fitness::Sharpe {
            max_position: 1.0,
            cost: 0.0,
            target: 0.0,
            periods_per_year: 4.0,
        };
        near(
            sharpe.evaluate(&y, &p).unwrap().value,
            (0.2 / 3.0) / (0.12666666666666668f64 / 3.0).sqrt() * 2.0,
        );
        near(
            Fitness::MeanReturn {
                max_position: 1.0,
                cost: 0.01,
            }
            .evaluate(&[0.1, 0.1], &[2.0, -2.0])
            .unwrap()
            .value,
            -0.015,
        );
        assert!(sortino.requires_order());
        assert!(sortino.evaluate(&[0.1, 0.2], &[1.0, 1.0]).is_err());
        assert!(sharpe.evaluate(&[0.1, 0.1], &[1.0, 1.0]).is_err());
    }
    #[test]
    fn catalog_and_custom_identities_round_trip_and_bind_every_parameter() {
        assert_eq!(Fitness::catalog().len(), 25);
        for (name, f) in Fitness::catalog() {
            let json = serde_json::to_string(&f).unwrap();
            assert_eq!(Fitness::parse(name).unwrap(), f);
            assert_eq!(
                Fitness::parse(&json).unwrap().identity().unwrap(),
                f.identity().unwrap()
            );
        }
        assert_ne!(
            Fitness::Pinball { quantile: 0.5 }.identity().unwrap(),
            Fitness::Pinball { quantile: 0.9 }.identity().unwrap()
        );
        assert_ne!(
            Fitness::Mae {}.identity().unwrap(),
            Fitness::Mse {}.identity().unwrap()
        );
    }
    #[test]
    fn all_checked_in_fitness_examples_parse_and_score() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/fitness");
        let mut count = 0;
        for entry in std::fs::read_dir(root).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|e| e == "json")
                && !path
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .contains("request")
            {
                let f = Fitness::parse(&std::fs::read_to_string(&path).unwrap()).unwrap();
                let targets = if matches!(
                    f,
                    Fitness::Msle {}
                        | Fitness::Rmsle {}
                        | Fitness::PoissonDeviance {}
                        | Fitness::GammaDeviance {}
                        | Fitness::TweedieDeviance { .. }
                ) {
                    [0.01, 0.02, 0.03, 0.04]
                } else {
                    [0.01, 0.02, -0.03, 0.04]
                };
                f.evaluate(&targets, &[0.02, 0.01, 0.01, 0.03])
                    .unwrap_or_else(|e| panic!("{}: {e:#}", path.display()));
                count += 1;
            }
        }
        assert!(count >= 12);
    }
}
