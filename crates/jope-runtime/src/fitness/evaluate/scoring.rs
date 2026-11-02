use super::*;

impl Fitness {
    /// Score predictions of a specification that reads no auxiliary column.
    pub fn evaluate(&self, y: &[f64], pred: &[f64]) -> Result<Score> {
        self.evaluate_with(y, pred, &Columns::new())
    }
    /// Score predictions; `columns` holds every auxiliary column the specification reads,
    /// aligned with `y`.
    pub fn evaluate_with(&self, y: &[f64], pred: &[f64], columns: &Columns) -> Result<Score> {
        self.validate()?;
        let value = self.checked_value(y, pred, columns)?;
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
    /// The minimization loss of an already validated specification, without hashing it. The
    /// calibration search calls this many times on one validated objective.
    pub fn loss_unhashed(&self, y: &[f64], pred: &[f64], columns: &Columns) -> Result<f64> {
        let value = self.checked_value(y, pred, columns)?;
        Ok(if self.direction() == Direction::Maximize {
            -value
        } else {
            value
        })
    }
    fn checked_value(&self, y: &[f64], pred: &[f64], columns: &Columns) -> Result<f64> {
        ensure!(
            !y.is_empty() && y.len() == pred.len(),
            "fitness needs equal, nonempty target and prediction vectors"
        );
        ensure!(
            y.iter().chain(pred).all(|x| x.is_finite()),
            "fitness inputs must be finite"
        );
        for name in self.columns() {
            let values = columns
                .get(&name)
                .ok_or_else(|| anyhow::anyhow!("fitness column {name:?} was not supplied"))?;
            ensure!(
                values.len() == y.len() && values.iter().all(|v| v.is_finite()),
                "fitness column {name:?} needs one finite value per scored row"
            );
        }
        finite(self.value(y, pred, columns, None)?)
    }
    fn value(&self, y: &[f64], p: &[f64], columns: &Columns, w: Option<&[f64]>) -> Result<f64> {
        if w.is_some() {
            ensure!(
                self.supports_weights(),
                "{} does not accept row weights",
                self.name()
            );
        }
        let errors: Vec<_> = y.iter().zip(p).map(|(y, p)| y - p).collect();
        if !matches!(
            self,
            Self::Custom { .. }
                | Self::Weighted { .. }
                | Self::Composite { .. }
                | Self::MeanReturn { .. }
                | Self::Sharpe { .. }
                | Self::Sortino { .. }
                | Self::MaxDrawdown { .. }
                | Self::Calmar { .. }
                | Self::Omega { .. }
                | Self::ExpectedShortfall { .. }
                | Self::Pearson {}
                | Self::Spearman {}
                | Self::HitRate {}
                | Self::NormalizedGini {}
                | Self::Kge {}
                | Self::CaptureAt { .. }
        ) {
            ensure!(
                errors.iter().all(|e| e.is_finite()),
                "non-finite regression residual"
            );
        }
        let squared = || wmean(&errors.iter().map(|e| e * e).collect::<Vec<_>>(), w);
        let abs: Vec<_> = errors.iter().map(|e| e.abs()).collect();
        let row_loss = |f: &dyn Fn(f64, f64) -> f64| {
            wmean(
                &y.iter().zip(p).map(|(&y, &p)| f(y, p)).collect::<Vec<_>>(),
                w,
            )
        };
        let error_loss =
            |f: &dyn Fn(f64) -> f64| wmean(&errors.iter().map(|&e| f(e)).collect::<Vec<_>>(), w);
        Ok(match self {
            Self::Rmse {} => squared().sqrt(),
            Self::Mse {} => squared(),
            Self::Mae {} => wmean(&abs, w),
            Self::MedianAbsoluteError {} => wquantile(&abs, w, 0.5)?,
            Self::MaxError {} => abs.iter().copied().fold(0.0, f64::max),
            Self::Mape { epsilon } => row_loss(&|y, p| (y - p).abs() / y.abs().max(*epsilon)),
            Self::Smape { epsilon } => row_loss(&|y, p| {
                // Normalize before adding or doubling: finite same-sign inputs
                // can overflow the unscaled denominator and hide real error.
                let scale = y.abs().max(p.abs()).max(*epsilon);
                let denominator = (y.abs() / scale + p.abs() / scale).max(epsilon / scale);
                ((y - p).abs() / scale) * (2.0 / denominator)
            }),
            Self::Rmspe { epsilon } => {
                row_loss(&|y, p| ((y - p) / y.abs().max(*epsilon)).powi(2)).sqrt()
            }
            Self::Wape {} => {
                let den = wmean(&y.iter().map(|y| y.abs()).collect::<Vec<_>>(), w);
                ensure!(den > 0.0, "WAPE undefined for all-zero targets");
                wmean(&abs, w) / den
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
                let den = wvariance(y, w);
                ensure!(
                    den.is_finite(),
                    "R2/explained variance overflows: the target variance is not finite"
                );
                ensure!(
                    y.len() >= 2 && den > 0.0,
                    "R2/explained variance undefined for constant or singleton targets"
                );
                1.0 - if matches!(self, Self::R2 {}) {
                    squared() / den
                } else {
                    wvariance(&errors, w) / den
                }
            }
            Self::Pearson {} => correlation(y, p, w)?,
            Self::Spearman {} => correlation(&ranks(y), &ranks(p), None)?,
            Self::Huber { delta } => error_loss(&|e| {
                let e = e.abs();
                if e <= *delta {
                    0.5 * e * e
                } else {
                    delta * (e - 0.5 * delta)
                }
            }),
            Self::PseudoHuber { delta } => {
                error_loss(&|e| delta * delta * ((1.0 + (e / delta).powi(2)).sqrt() - 1.0))
            }
            Self::LogCosh {} => error_loss(&|e| {
                let e = e.abs();
                e + (-2.0 * e).exp().ln_1p() - std::f64::consts::LN_2
            }),
            Self::Pinball { quantile } => error_loss(&|e| {
                if e >= 0.0 {
                    quantile * e
                } else {
                    (quantile - 1.0) * e
                }
            }),
            Self::Expectile { quantile } => error_loss(&|e| {
                if e >= 0.0 {
                    quantile * e * e
                } else {
                    (1.0 - quantile) * e * e
                }
            }),
            Self::Linex { a } => error_loss(&|e| (a * e).exp_m1() - a * e),
            Self::AsymmetricAbsolute {
                under_cost,
                over_cost,
            } => error_loss(&|e| {
                if e >= 0.0 {
                    under_cost * e
                } else {
                    -over_cost * e
                }
            }),
            Self::AsymmetricSquared {
                under_cost,
                over_cost,
            } => error_loss(&|e| {
                if e >= 0.0 {
                    under_cost * e * e
                } else {
                    over_cost * e * e
                }
            }),
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
            Self::Mase { season } => {
                ensure!(
                    y.len() > *season,
                    "MASE needs more scored rows than its season"
                );
                let naive = mean(
                    &y.windows(season + 1)
                        .map(|w| (w[*season] - w[0]).abs())
                        .collect::<Vec<_>>(),
                );
                ensure!(
                    naive > 0.0,
                    "MASE undefined when the seasonal naive forecast is exact"
                );
                mean(&abs) / naive
            }
            Self::MeanReturn { max_position, cost } => {
                mean(&strategy_returns(y, p, *max_position, *cost)?)
            }
            Self::Sharpe {
                max_position,
                cost,
                target,
                periods_per_year,
            }
            | Self::Sortino {
                max_position,
                cost,
                target,
                periods_per_year,
            } => {
                let returns = strategy_returns(y, p, *max_position, *cost)?;
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
                ensure!(
                    den > 0.0 && den.is_finite(),
                    "risk ratio undefined with a zero or non-finite risk denominator"
                );
                finite((mean(&returns) - target) / den * periods_per_year.sqrt())?
            }
            Self::MaxDrawdown { max_position, cost } => {
                max_drawdown(&strategy_returns(y, p, *max_position, *cost)?)?
            }
            Self::Calmar {
                max_position,
                cost,
                periods_per_year,
            } => {
                let returns = strategy_returns(y, p, *max_position, *cost)?;
                let drawdown = max_drawdown(&returns)?;
                ensure!(drawdown > 0.0, "Calmar undefined without a drawdown");
                let end = *equity_curve(&returns)?.last().expect("nonempty");
                let growth = end.powf(periods_per_year / returns.len() as f64) - 1.0;
                growth / drawdown
            }
            Self::Omega {
                max_position,
                cost,
                threshold,
            } => {
                let returns = strategy_returns(y, p, *max_position, *cost)?;
                let gains: f64 = returns.iter().map(|r| (r - threshold).max(0.0)).sum();
                let losses: f64 = returns.iter().map(|r| (threshold - r).max(0.0)).sum();
                ensure!(gains.is_finite() && losses.is_finite(), "Omega overflowed");
                ensure!(losses > 0.0, "Omega undefined without a shortfall");
                gains / losses
            }
            Self::ExpectedShortfall {
                max_position,
                cost,
                alpha,
            } => {
                let losses: Vec<f64> = strategy_returns(y, p, *max_position, *cost)?
                    .iter()
                    .map(|r| -r)
                    .collect();
                tail_mean(&losses, None, *alpha)?
            }
            Self::HitRate {} => {
                let weight = |i: usize| w.map_or(1.0, |w| w[i]);
                let (mut hits, mut total) = (0.0, 0.0);
                for i in (0..y.len()).filter(|&i| y[i] != 0.0) {
                    total += weight(i);
                    if p[i] != 0.0 && (p[i] > 0.0) == (y[i] > 0.0) {
                        hits += weight(i);
                    }
                }
                ensure!(total > 0.0, "hit rate undefined for all-zero targets");
                hits / total
            }
            Self::NormalizedGini {} => {
                let model = lorenz_gini(&ranked(y, p, w))?;
                let perfect = lorenz_gini(&ranked(y, y, w))?;
                ensure!(
                    perfect != 0.0,
                    "normalized Gini undefined for constant targets"
                );
                model / perfect
            }
            Self::Kge {} => {
                ensure!(y.len() >= 2, "KGE needs at least two rows");
                let r = correlation(y, p, None)?;
                let alpha = (variance(p) / variance(y)).sqrt();
                let my = mean(y);
                ensure!(my != 0.0, "KGE undefined for a zero-mean target");
                let beta = mean(p) / my;
                1.0 - ((r - 1.0).powi(2) + (alpha - 1.0).powi(2) + (beta - 1.0).powi(2)).sqrt()
            }
            Self::CaptureAt { fraction } => {
                let total: f64 = y.iter().sum();
                ensure!(
                    total > 0.0 && total.is_finite(),
                    "capture needs a positive target total"
                );
                let mass = fraction * y.len() as f64;
                let (mut taken, mut captured) = (0.0f64, 0.0);
                for (_, target) in ranked(y, p, None) {
                    let take = (mass - taken).min(1.0);
                    if take <= 0.0 {
                        break;
                    }
                    captured += take * target;
                    taken += take;
                }
                finite(captured / total)?
            }
            Self::Custom {
                expression,
                aggregation,
                ..
            } => {
                let v = expression.rows(y, p, columns)?;
                match aggregation {
                    RegressionAggregation::Mean {} => wmean(&v, w),
                    RegressionAggregation::Sum {} => match w {
                        None => v.iter().sum(),
                        Some(w) => v.iter().zip(w).map(|(v, w)| v * w).sum(),
                    },
                    RegressionAggregation::Max {} => {
                        v.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                    }
                    RegressionAggregation::Quantile { quantile: q } => wquantile(&v, w, *q)?,
                    RegressionAggregation::TailMean { fraction } => tail_mean(&v, w, *fraction)?,
                    RegressionAggregation::RootMean {} => {
                        let m = wmean(&v, w);
                        ensure!(m >= 0.0, "root_mean needs a nonnegative mean");
                        m.sqrt()
                    }
                }
            }
            Self::Weighted { weights, fitness } => {
                ensure!(w.is_none(), "weighting cannot be nested");
                let weights = weights.rows(y, p, columns)?;
                let total: f64 = weights.iter().sum();
                ensure!(
                    weights.iter().all(|w| *w >= 0.0) && total.is_finite() && total > 0.0,
                    "weights must be nonnegative with a positive finite total"
                );
                fitness.value(y, p, columns, Some(&weights))?
            }
            Self::Composite { terms } => {
                let mut sum = 0.0;
                for t in terms {
                    let natural = finite(t.fitness.value(y, p, columns, w)?)?;
                    let loss = if t.fitness.direction() == Direction::Maximize {
                        -natural
                    } else {
                        natural
                    };
                    sum += t.weight * loss / t.scale;
                }
                sum
            }
        })
    }
}
