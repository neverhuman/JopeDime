use super::*;

impl Fitness {
    /// A catalog name, `name:key=value,…` overriding a catalog default's numeric parameters
    /// (`pinball:quantile=0.9`), or a strict inline JSON specification. File access belongs to
    /// the CLI.
    pub fn parse(text: &str) -> Result<Self> {
        ensure!(
            text.len() <= MAX_SPEC_BYTES,
            "fitness specification is too large"
        );
        let spec = if text.trim_start().starts_with('{') {
            serde_json::from_str(text)?
        } else {
            let (name, overrides) = match text.trim().split_once(':') {
                Some((name, overrides)) => (name, Some(overrides)),
                None => (text.trim(), None),
            };
            let base = Self::catalog()
                .into_iter()
                .find(|entry| entry.name == name)
                .map(|entry| entry.fitness)
                .ok_or_else(|| {
                    anyhow::anyhow!("unknown fitness {name:?}; use `jopedime fitness list`")
                })?;
            match overrides {
                None => base,
                Some(overrides) => {
                    let mut value = serde_json::to_value(&base)?;
                    let object = value.as_object_mut().expect("tagged specification");
                    for pair in overrides.split(',').filter(|p| !p.trim().is_empty()) {
                        let (key, raw) = pair
                            .split_once('=')
                            .ok_or_else(|| anyhow::anyhow!("override {pair:?} is not key=value"))?;
                        let key = key.trim();
                        ensure!(
                            key != "metric" && object.contains_key(key),
                            "{name} has no parameter {key:?}"
                        );
                        let number: serde_json::Value = serde_json::from_str(raw.trim())
                            .ok()
                            .filter(serde_json::Value::is_number)
                            .ok_or_else(|| {
                                anyhow::anyhow!("parameter {key} needs a number, got {raw:?}")
                            })?;
                        object.insert(key.to_owned(), number);
                    }
                    serde_json::from_value(value)?
                }
            }
        };
        spec.validate()?;
        Ok(spec)
    }

    /// Short names with their exact default parameters.
    pub fn catalog() -> Vec<CatalogEntry> {
        let strategy = |metric: &str| -> Fitness {
            let (max_position, cost) = (1.0, 0.0);
            match metric {
                "mean_return" => Self::MeanReturn { max_position, cost },
                "sharpe" => Self::Sharpe {
                    max_position,
                    cost,
                    target: 0.0,
                    periods_per_year: 252.0,
                },
                "sortino" => Self::Sortino {
                    max_position,
                    cost,
                    target: 0.0,
                    periods_per_year: 252.0,
                },
                "max_drawdown" => Self::MaxDrawdown { max_position, cost },
                "calmar" => Self::Calmar {
                    max_position,
                    cost,
                    periods_per_year: 252.0,
                },
                "omega" => Self::Omega {
                    max_position,
                    cost,
                    threshold: 0.0,
                },
                _ => Self::ExpectedShortfall {
                    max_position,
                    cost,
                    alpha: 0.05,
                },
            }
        };
        let entry = |name, fitness, description| CatalogEntry {
            name,
            fitness,
            description,
        };
        vec![
            entry(
                "rmse",
                Self::Rmse {},
                "Root mean squared error; emphasizes large misses.",
            ),
            entry("mse", Self::Mse {}, "Mean squared error."),
            entry(
                "mae",
                Self::Mae {},
                "Mean absolute error; linear cost per unit missed.",
            ),
            entry(
                "median_absolute_error",
                Self::MedianAbsoluteError {},
                "Median absolute error; ignores the worst half of misses.",
            ),
            entry(
                "max_error",
                Self::MaxError {},
                "Worst absolute error on scored rows.",
            ),
            entry(
                "mape",
                Self::Mape { epsilon: 1e-8 },
                "Mean absolute percentage error (ratio) with a denominator floor.",
            ),
            entry(
                "smape",
                Self::Smape { epsilon: 1e-8 },
                "Symmetric mean absolute percentage error (ratio, 0..2).",
            ),
            entry(
                "wape",
                Self::Wape {},
                "Total absolute error over total absolute target volume.",
            ),
            entry(
                "msle",
                Self::Msle {},
                "Mean squared log1p error; nonnegative values.",
            ),
            entry(
                "rmsle",
                Self::Rmsle {},
                "Root mean squared log1p error; nonnegative values.",
            ),
            entry(
                "r2",
                Self::R2 {},
                "Coefficient of determination (maximize).",
            ),
            entry(
                "explained_variance",
                Self::ExplainedVariance {},
                "Explained variance; ignores a constant bias (maximize).",
            ),
            entry(
                "pearson",
                Self::Pearson {},
                "Pearson correlation (maximize).",
            ),
            entry(
                "spearman",
                Self::Spearman {},
                "Rank correlation, the information coefficient (maximize).",
            ),
            entry(
                "huber",
                Self::Huber { delta: 1.0 },
                "Huber loss: quadratic inside delta.",
            ),
            entry(
                "pseudo_huber",
                Self::PseudoHuber { delta: 1.0 },
                "Smooth Huber loss.",
            ),
            entry(
                "log_cosh",
                Self::LogCosh {},
                "log(cosh(error)); smooth and robust.",
            ),
            entry(
                "pinball",
                Self::Pinball { quantile: 0.5 },
                "Quantile (pinball) loss; quantile=0.9 makes shortages 9x as costly.",
            ),
            entry(
                "expectile",
                Self::Expectile { quantile: 0.5 },
                "Asymmetric squared error.",
            ),
            entry(
                "poisson_deviance",
                Self::PoissonDeviance {},
                "Poisson deviance for counts; y>=0, pred>0.",
            ),
            entry(
                "gamma_deviance",
                Self::GammaDeviance {},
                "Gamma deviance for positive severities.",
            ),
            entry(
                "tweedie_deviance",
                Self::TweedieDeviance { power: 1.5 },
                "Tweedie deviance for zero-inflated positive costs.",
            ),
            entry(
                "linex",
                Self::Linex { a: 1.0 },
                "Linear-exponential loss; a>0 punishes under-prediction exponentially.",
            ),
            entry(
                "asymmetric_absolute",
                Self::AsymmetricAbsolute {
                    under_cost: 1.0,
                    over_cost: 1.0,
                },
                "Newsvendor cost: under_cost per unit short, over_cost per unit over.",
            ),
            entry(
                "asymmetric_squared",
                Self::AsymmetricSquared {
                    under_cost: 1.0,
                    over_cost: 1.0,
                },
                "Squared newsvendor cost.",
            ),
            entry(
                "rmspe",
                Self::Rmspe { epsilon: 1e-8 },
                "Root mean squared percentage error (ratio).",
            ),
            entry(
                "mase",
                Self::Mase { season: 1 },
                "MAE relative to the seasonal naive forecast; chronological rows.",
            ),
            entry(
                "mean_return",
                strategy("mean_return"),
                "Mean net strategy return per interval (maximize).",
            ),
            entry(
                "sharpe",
                strategy("sharpe"),
                "Annualized Sharpe ratio of net strategy returns (maximize).",
            ),
            entry(
                "sortino",
                strategy("sortino"),
                "Annualized Sortino ratio; penalizes only downside (maximize).",
            ),
            entry(
                "max_drawdown",
                strategy("max_drawdown"),
                "Largest peak-to-trough loss of compounded equity.",
            ),
            entry(
                "calmar",
                strategy("calmar"),
                "Compound annual growth over maximum drawdown (maximize).",
            ),
            entry(
                "omega",
                strategy("omega"),
                "Gains above threshold over shortfalls below it (maximize).",
            ),
            entry(
                "expected_shortfall",
                strategy("expected_shortfall"),
                "Mean loss of the worst alpha fraction of net returns (CVaR).",
            ),
            entry(
                "hit_rate",
                Self::HitRate {},
                "Share of nonzero targets whose sign is predicted (maximize).",
            ),
            entry(
                "normalized_gini",
                Self::NormalizedGini {},
                "Ranking quality: Gini over perfect Gini (maximize).",
            ),
            entry(
                "kge",
                Self::Kge {},
                "Kling-Gupta efficiency for hydrology (maximize).",
            ),
            entry(
                "capture_at",
                Self::CaptureAt { fraction: 0.1 },
                "Share of total target in the top-ranked fraction (maximize).",
            ),
        ]
    }

    pub fn name(&self) -> String {
        match self {
            Self::Custom { name, .. } => name.clone(),
            Self::Weighted { fitness, .. } => format!("weighted_{}", fitness.name()),
            Self::Composite { .. } => "composite".to_owned(),
            other => serde_json::to_value(other)
                .ok()
                .and_then(|v| v["metric"].as_str().map(str::to_owned))
                .expect("tagged built-in"),
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
            | Self::Sortino { .. }
            | Self::Calmar { .. }
            | Self::Omega { .. }
            | Self::HitRate {}
            | Self::NormalizedGini {}
            | Self::Kge {}
            | Self::CaptureAt { .. } => Direction::Maximize,
            Self::Custom { direction, .. } => *direction,
            Self::Weighted { fitness, .. } => fitness.direction(),
            _ => Direction::Minimize,
        }
    }
    /// Whether scoring depends on row order, so it needs chronological forward validation.
    pub fn requires_order(&self) -> bool {
        match self {
            Self::MeanReturn { .. }
            | Self::Sharpe { .. }
            | Self::Sortino { .. }
            | Self::MaxDrawdown { .. }
            | Self::Calmar { .. }
            | Self::Omega { .. }
            | Self::ExpectedShortfall { .. }
            | Self::Mase { .. } => true,
            Self::Weighted { fitness, .. } => fitness.requires_order(),
            Self::Composite { terms } => terms.iter().any(|t| t.fitness.requires_order()),
            _ => false,
        }
    }
    /// Auxiliary columns the specification reads (custom expressions and weights).
    pub fn columns(&self) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        self.collect_columns(&mut out);
        out
    }
    fn collect_columns(&self, out: &mut BTreeSet<String>) {
        match self {
            Self::Custom { expression, .. } => expression.collect_columns(out),
            Self::Weighted { weights, fitness } => {
                weights.collect_columns(out);
                fitness.collect_columns(out);
            }
            Self::Composite { terms } => {
                for term in terms {
                    term.fitness.collect_columns(out);
                }
            }
            _ => (),
        }
    }
}
