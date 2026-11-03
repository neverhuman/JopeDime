use super::*;

impl Fitness {
    /// Whether rows may carry weights here.
    pub(crate) fn supports_weights(&self) -> bool {
        match self {
            Self::MaxError {}
            | Self::Spearman {}
            | Self::Mase { .. }
            | Self::MeanReturn { .. }
            | Self::Sharpe { .. }
            | Self::Sortino { .. }
            | Self::MaxDrawdown { .. }
            | Self::Calmar { .. }
            | Self::Omega { .. }
            | Self::ExpectedShortfall { .. }
            | Self::Kge {}
            | Self::CaptureAt { .. }
            | Self::Resource { .. }
            | Self::Weighted { .. } => false,
            Self::Custom { aggregation, .. } => !matches!(
                aggregation,
                RegressionAggregation::Max {} | RegressionAggregation::Min {}
            ),
            Self::Composite { terms } => terms.iter().all(|t| t.fitness.supports_weights()),
            Self::Constrained { fitness, .. } => fitness.supports_weights(),
            _ => true,
        }
    }

    pub(crate) fn collect_resources(&self, out: &mut BTreeSet<ResourceVar>) {
        match self {
            Self::Resource { name } => {
                out.insert(*name);
            }
            Self::TrainTimeSeconds {} => {
                out.insert(ResourceVar::FitSeconds);
            }
            Self::InferenceTimeSeconds {} => {
                out.insert(ResourceVar::PredictSeconds);
            }
            Self::TotalTimeSeconds {} => {
                out.insert(ResourceVar::FitSeconds);
                out.insert(ResourceVar::PredictSeconds);
            }
            Self::InferenceUsPerRow {} => {
                out.insert(ResourceVar::PredictUsPerRow);
            }
            Self::TrainRowsPerSecond {} => {
                out.insert(ResourceVar::FitSeconds);
                out.insert(ResourceVar::TrainRows);
            }
            Self::InferenceRowsPerSecond {} => {
                out.insert(ResourceVar::PredictRowsPerSecond);
            }
            Self::TrainMemoryBytes {} => {
                out.insert(ResourceVar::FitPeakHeapBytes);
            }
            Self::InferenceMemoryBytes {} => {
                out.insert(ResourceVar::PredictPeakHeapBytes);
            }
            Self::TotalMemoryBytes {} => {
                out.insert(ResourceVar::FitPeakHeapBytes);
                out.insert(ResourceVar::PredictPeakHeapBytes);
            }
            Self::ModelBytes {} => {
                out.insert(ResourceVar::ModelBytes);
            }
            Self::ComplexityStages {} => {
                out.insert(ResourceVar::StageCount);
            }
            Self::ComplexityParameters {} => {
                out.insert(ResourceVar::ProgramNodes);
            }
            Self::Custom { expression, .. } => expression.collect_resources(out),
            Self::Weighted { weights, fitness } => {
                weights.collect_resources(out);
                fitness.collect_resources(out);
            }
            Self::Composite { terms } => {
                for t in terms {
                    t.fitness.collect_resources(out);
                }
            }
            Self::Constrained { fitness, limits } => {
                out.extend(limits.iter().map(|l| l.resource));
                fitness.collect_resources(out);
            }
            _ => (),
        }
    }

    /// Whether evaluation depends on [`ExecutionTelemetry`] or [`Resources`].
    pub fn uses_execution_telemetry(&self) -> bool {
        !self.resources().is_empty() || matches!(self, Self::FeatureCount {} | Self::FoldStd {})
    }

    pub fn validate(&self) -> Result<()> {
        self.validate_at(0, &mut 0, &mut 0, false)?;
        let columns = self.columns();
        ensure!(
            columns.len() <= MAX_COLUMNS,
            "a fitness specification may read at most {MAX_COLUMNS} columns"
        );
        ensure!(
            serde_json::to_vec(self)?.len() <= MAX_SPEC_BYTES,
            "fitness specification is too large"
        );
        Ok(())
    }

    fn validate_at(
        &self,
        depth: usize,
        nodes: &mut usize,
        expression_nodes: &mut usize,
        weighted: bool,
    ) -> Result<()> {
        *nodes += 1;
        ensure!(
            depth <= 8 && *nodes <= 64,
            "fitness composition exceeds depth/node limit"
        );
        if weighted {
            ensure!(
                self.supports_weights(),
                "{} does not accept row weights",
                self.name()
            );
        }
        let strategy = |max_position: f64, cost: f64| {
            ensure!(
                positive(max_position) && cost.is_finite() && cost >= 0.0,
                "position must be positive; cost must be nonnegative; both finite"
            );
            Ok(())
        };
        match self {
            Self::Mape { epsilon } | Self::Smape { epsilon } | Self::Rmspe { epsilon } => {
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
            Self::Linex { a } => ensure!(
                a.is_finite() && *a != 0.0,
                "LinEx a must be finite and nonzero"
            ),
            Self::AsymmetricAbsolute {
                under_cost,
                over_cost,
            }
            | Self::AsymmetricSquared {
                under_cost,
                over_cost,
            } => ensure!(
                positive(*under_cost) && positive(*over_cost),
                "under_cost and over_cost must be positive and finite"
            ),
            Self::Mase { season } => {
                ensure!((1..=10_000).contains(season), "season must be in 1..=10000")
            }
            Self::MeanReturn { max_position, cost } | Self::MaxDrawdown { max_position, cost } => {
                strategy(*max_position, *cost)?
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
                strategy(*max_position, *cost)?;
                ensure!(
                    target.is_finite() && positive(*periods_per_year),
                    "target must be finite and periods_per_year positive"
                );
            }
            Self::Calmar {
                max_position,
                cost,
                periods_per_year,
            } => {
                strategy(*max_position, *cost)?;
                ensure!(
                    positive(*periods_per_year),
                    "periods_per_year must be positive"
                );
            }
            Self::Omega {
                max_position,
                cost,
                threshold,
            } => {
                strategy(*max_position, *cost)?;
                ensure!(threshold.is_finite(), "threshold must be finite");
            }
            Self::ExpectedShortfall {
                max_position,
                cost,
                alpha,
            } => {
                strategy(*max_position, *cost)?;
                ensure!(probability(*alpha), "alpha must be in (0,1)");
            }
            Self::CaptureAt { fraction } => ensure!(
                fraction.is_finite() && *fraction > 0.0 && *fraction <= 1.0,
                "fraction must be in (0,1]"
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
                expression.validate(0, expression_nodes)?;
                match aggregation {
                    RegressionAggregation::Quantile { quantile } => ensure!(
                        quantile.is_finite() && (0.0..=1.0).contains(quantile),
                        "aggregation quantile must be in [0,1]"
                    ),
                    RegressionAggregation::TailMean { fraction } => ensure!(
                        fraction.is_finite() && *fraction > 0.0 && *fraction <= 1.0,
                        "tail_mean fraction must be in (0,1]"
                    ),
                    _ => (),
                }
            }
            Self::Weighted { weights, fitness } => {
                ensure!(!weighted, "weighting cannot be nested");
                weights.validate(0, expression_nodes)?;
                fitness.validate_at(depth + 1, nodes, expression_nodes, true)?;
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
                    t.fitness
                        .validate_at(depth + 1, nodes, expression_nodes, weighted)?;
                }
            }
            Self::Constrained { fitness, limits } => {
                ensure!(
                    !limits.is_empty() && limits.len() <= 16,
                    "constrained needs 1..16 resource limits"
                );
                for l in limits {
                    ensure!(
                        l.min.is_some() || l.max.is_some(),
                        "limit on {} needs at least one of min or max",
                        l.resource
                    );
                    if let Some(min) = l.min {
                        ensure!(
                            min.is_finite() && min >= 0.0,
                            "limit min on {} must be finite and nonnegative",
                            l.resource
                        );
                    }
                    if let Some(max) = l.max {
                        ensure!(
                            max.is_finite() && max >= 0.0,
                            "limit max on {} must be finite and nonnegative",
                            l.resource
                        );
                    }
                    if let (Some(min), Some(max)) = (l.min, l.max) {
                        ensure!(
                            min <= max,
                            "limit on {} has min ({min}) > max ({max})",
                            l.resource
                        );
                    }
                }
                fitness.validate_at(depth + 1, nodes, expression_nodes, weighted)?;
            }
            _ => (),
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String> {
        self.validate()?;
        struct Canonical;
        impl serde_json::ser::Formatter for Canonical {
            fn write_f64<W: ?Sized + std::io::Write>(
                &mut self,
                writer: &mut W,
                value: f64,
            ) -> std::io::Result<()> {
                let value = if value == 0.0 { 0.0 } else { value };
                serde_json::ser::CompactFormatter.write_f64(writer, value)
            }
        }
        let mut bytes = Vec::new();
        serde::Serialize::serialize(
            &(FITNESS_VERSION, self),
            &mut serde_json::Serializer::with_formatter(&mut bytes, Canonical),
        )?;
        Ok(crate::sha256_hex(&bytes))
    }
}

impl Expression {
    pub(super) fn validate(&self, depth: usize, nodes: &mut usize) -> Result<()> {
        *nodes += 1;
        ensure!(
            depth <= 16 && *nodes <= MAX_EXPRESSION_NODES,
            "fitness expressions exceed depth 16 or {MAX_EXPRESSION_NODES} nodes in one \
             specification"
        );
        match self {
            Self::Constant { value } => {
                finite(*value)?;
            }
            Self::Column { name } => validate_column_name(name)?,
            Self::Y {}
            | Self::Pred {}
            | Self::Resource { .. }
            | Self::TrainTimeSeconds {}
            | Self::InferenceTimeSeconds {}
            | Self::TotalTimeSeconds {}
            | Self::InferenceUsPerRow {}
            | Self::TrainRowsPerSecond {}
            | Self::InferenceRowsPerSecond {}
            | Self::TrainMemoryBytes {}
            | Self::InferenceMemoryBytes {}
            | Self::TotalMemoryBytes {}
            | Self::ModelBytes {}
            | Self::ComplexityStages {}
            | Self::ComplexityParameters {}
            | Self::FeatureCount {}
            | Self::FoldStd {}
            | Self::PredictionTurnover {}
            | Self::RowIndex {}
            | Self::RowFraction {}
            | Self::RowCount {}
            | Self::Residual {}
            | Self::AbsError {}
            | Self::SquaredError {}
            | Self::YMean {}
            | Self::YStd {}
            | Self::PredMean {}
            | Self::PredStd {} => (),
            Self::Abs { arg }
            | Self::Neg { arg }
            | Self::Square { arg }
            | Self::Sqrt { arg }
            | Self::Ln { arg }
            | Self::Log1p { arg }
            | Self::Exp { arg }
            | Self::Sign { arg }
            | Self::Tanh { arg }
            | Self::Sigmoid { arg }
            | Self::Cbrt { arg }
            | Self::Step { arg } => arg.validate(depth + 1, nodes)?,
            Self::Clip { arg, min, max } => {
                ensure!(
                    min.is_finite() && max.is_finite() && min <= max,
                    "clip needs finite min <= max"
                );
                arg.validate(depth + 1, nodes)?;
            }
            Self::Add { left, right }
            | Self::Sub { left, right }
            | Self::Mul { left, right }
            | Self::Div { left, right }
            | Self::Min { left, right }
            | Self::Max { left, right }
            | Self::Pow { left, right } => {
                left.validate(depth + 1, nodes)?;
                right.validate(depth + 1, nodes)?;
            }
            Self::Where {
                condition,
                then,
                otherwise,
            } => {
                condition.validate(depth + 1, nodes)?;
                then.validate(depth + 1, nodes)?;
                otherwise.validate(depth + 1, nodes)?;
            }
        }
        Ok(())
    }

    pub(crate) fn collect_columns(&self, out: &mut BTreeSet<String>) {
        match self {
            Self::Column { name } => {
                out.insert(name.clone());
            }
            Self::Abs { arg }
            | Self::Neg { arg }
            | Self::Square { arg }
            | Self::Sqrt { arg }
            | Self::Ln { arg }
            | Self::Log1p { arg }
            | Self::Exp { arg }
            | Self::Sign { arg }
            | Self::Tanh { arg }
            | Self::Sigmoid { arg }
            | Self::Cbrt { arg }
            | Self::Step { arg }
            | Self::Clip { arg, .. } => arg.collect_columns(out),
            Self::Add { left, right }
            | Self::Sub { left, right }
            | Self::Mul { left, right }
            | Self::Div { left, right }
            | Self::Min { left, right }
            | Self::Max { left, right }
            | Self::Pow { left, right } => {
                left.collect_columns(out);
                right.collect_columns(out);
            }
            Self::Where {
                condition,
                then,
                otherwise,
            } => {
                condition.collect_columns(out);
                then.collect_columns(out);
                otherwise.collect_columns(out);
            }
            _ => (),
        }
    }

    pub(crate) fn collect_resources(&self, out: &mut BTreeSet<ResourceVar>) {
        match self {
            Self::Resource { name } => {
                out.insert(*name);
            }
            Self::TrainTimeSeconds {} => {
                out.insert(ResourceVar::FitSeconds);
            }
            Self::InferenceTimeSeconds {} => {
                out.insert(ResourceVar::PredictSeconds);
            }
            Self::TotalTimeSeconds {} => {
                out.insert(ResourceVar::FitSeconds);
                out.insert(ResourceVar::PredictSeconds);
            }
            Self::InferenceUsPerRow {} => {
                out.insert(ResourceVar::PredictUsPerRow);
            }
            Self::TrainRowsPerSecond {} => {
                out.insert(ResourceVar::FitSeconds);
                out.insert(ResourceVar::TrainRows);
            }
            Self::InferenceRowsPerSecond {} => {
                out.insert(ResourceVar::PredictRowsPerSecond);
            }
            Self::TrainMemoryBytes {} => {
                out.insert(ResourceVar::FitPeakHeapBytes);
            }
            Self::InferenceMemoryBytes {} => {
                out.insert(ResourceVar::PredictPeakHeapBytes);
            }
            Self::TotalMemoryBytes {} => {
                out.insert(ResourceVar::FitPeakHeapBytes);
                out.insert(ResourceVar::PredictPeakHeapBytes);
            }
            Self::ModelBytes {} => {
                out.insert(ResourceVar::ModelBytes);
            }
            Self::ComplexityStages {} => {
                out.insert(ResourceVar::StageCount);
            }
            Self::ComplexityParameters {} => {
                out.insert(ResourceVar::ProgramNodes);
            }
            Self::Abs { arg }
            | Self::Neg { arg }
            | Self::Square { arg }
            | Self::Sqrt { arg }
            | Self::Ln { arg }
            | Self::Log1p { arg }
            | Self::Exp { arg }
            | Self::Sign { arg }
            | Self::Tanh { arg }
            | Self::Sigmoid { arg }
            | Self::Cbrt { arg }
            | Self::Step { arg }
            | Self::Clip { arg, .. } => arg.collect_resources(out),
            Self::Add { left, right }
            | Self::Sub { left, right }
            | Self::Mul { left, right }
            | Self::Div { left, right }
            | Self::Min { left, right }
            | Self::Max { left, right }
            | Self::Pow { left, right } => {
                left.collect_resources(out);
                right.collect_resources(out);
            }
            Self::Where {
                condition,
                then,
                otherwise,
            } => {
                condition.collect_resources(out);
                then.collect_resources(out);
                otherwise.collect_resources(out);
            }
            _ => (),
        }
    }
}
