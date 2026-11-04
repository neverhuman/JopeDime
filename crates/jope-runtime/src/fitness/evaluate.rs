use super::*;

mod scoring;
mod validation;

/// Candidate execution speed, memory footprint, and structural complexity telemetry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ExecutionTelemetry {
    /// Wall-clock seconds spent fitting the candidate model (`>= 0`).
    pub train_time_seconds: f64,
    /// Wall-clock seconds spent running candidate inference (`>= 0`).
    pub inference_time_seconds: f64,
    /// Number of training rows used during candidate fit.
    pub train_rows: usize,
    /// Number of evaluation rows scored during candidate inference.
    pub inference_rows: usize,
    /// Peak/working heap memory bytes used during training.
    pub train_memory_bytes: u64,
    /// Peak/working heap memory bytes used during inference.
    pub inference_memory_bytes: u64,
    /// Fitted model artifact footprint in bytes.
    pub model_bytes: u64,
    /// Pipeline stage count of the candidate.
    pub complexity_stages: usize,
    /// Fitted scalar parameter / node count of the candidate.
    pub complexity_parameters: usize,
    /// Active input feature count consumed by the candidate.
    pub feature_count: usize,
    /// Standard deviation of loss across cross-validation folds / regime cuts.
    pub fold_std: f64,
}

impl ExecutionTelemetry {
    pub fn total_time_seconds(&self) -> f64 {
        self.train_time_seconds.max(0.0) + self.inference_time_seconds.max(0.0)
    }

    pub fn total_memory_bytes(&self) -> u64 {
        self.train_memory_bytes
            .saturating_add(self.inference_memory_bytes)
    }

    pub fn inference_us_per_row(&self, scored_rows: usize) -> f64 {
        let rows = if self.inference_rows > 0 {
            self.inference_rows
        } else {
            scored_rows.max(1)
        };
        (self.inference_time_seconds.max(0.0) * 1_000_000.0) / (rows as f64)
    }

    pub fn train_rows_per_second(&self, scored_rows: usize) -> f64 {
        let rows = if self.train_rows > 0 {
            self.train_rows
        } else {
            scored_rows.max(1)
        };
        if self.train_time_seconds > 0.0 {
            (rows as f64) / self.train_time_seconds.max(1e-9)
        } else {
            rows as f64
        }
    }

    pub fn inference_rows_per_second(&self, scored_rows: usize) -> f64 {
        let rows = if self.inference_rows > 0 {
            self.inference_rows
        } else {
            scored_rows.max(1)
        };
        if self.inference_time_seconds > 0.0 {
            (rows as f64) / self.inference_time_seconds.max(1e-9)
        } else {
            rows as f64
        }
    }

    /// Project telemetry into a canonical [`Resources`] map.
    pub fn to_resources(&self, scored_rows: usize) -> Resources {
        let mut r = Resources::new();
        r.insert(ResourceVar::FitSeconds, self.train_time_seconds.max(0.0));
        r.insert(ResourceVar::FitCpuSeconds, self.train_time_seconds.max(0.0));
        r.insert(
            ResourceVar::PredictSeconds,
            self.inference_time_seconds.max(0.0),
        );
        let us_per_row = self.inference_us_per_row(scored_rows);
        r.insert(ResourceVar::PredictUsPerRow, us_per_row);
        r.insert(
            ResourceVar::PredictRowsPerSecond,
            self.inference_rows_per_second(scored_rows),
        );
        r.insert(ResourceVar::RowLatencyP50Us, us_per_row);
        r.insert(ResourceVar::RowLatencyP99Us, us_per_row);
        r.insert(
            ResourceVar::FitPeakHeapBytes,
            self.train_memory_bytes as f64,
        );
        r.insert(
            ResourceVar::PredictPeakHeapBytes,
            self.inference_memory_bytes as f64,
        );
        r.insert(ResourceVar::ModelBytes, self.model_bytes as f64);
        r.insert(ResourceVar::ProgramNodes, self.complexity_parameters as f64);
        r.insert(ResourceVar::StageCount, self.complexity_stages as f64);
        r.insert(
            ResourceVar::TrainRows,
            self.train_rows.max(scored_rows) as f64,
        );
        r.insert(
            ResourceVar::HoldRows,
            self.inference_rows.max(scored_rows) as f64,
        );
        r
    }
}

pub(super) fn read_resource(resources: &Resources, name: ResourceVar) -> Result<f64> {
    let &value = resources
        .get(&name)
        .ok_or_else(|| anyhow::anyhow!("fitness resource {name} is not available"))?;
    ensure!(
        value.is_finite() && value >= 0.0,
        "fitness resource {name} must be finite and nonnegative (got {value})"
    );
    Ok(value)
}

pub(super) struct BatchStats {
    pub(super) row_count: usize,
    pub(super) y_mean: f64,
    pub(super) y_std: f64,
    pub(super) pred_mean: f64,
    pub(super) pred_std: f64,
    pub(super) prediction_turnover: f64,
}

impl BatchStats {
    pub(super) fn compute(y: &[f64], p: &[f64]) -> Self {
        let y_mean = mean(y);
        let y_std = variance(y).max(0.0).sqrt();
        let pred_mean = mean(p);
        let pred_std = variance(p).max(0.0).sqrt();
        Self {
            row_count: y.len(),
            y_mean,
            y_std,
            pred_mean,
            pred_std,
            prediction_turnover: prediction_turnover(p),
        }
    }
}

#[derive(Clone, Copy)]
struct EvalContext<'a> {
    y: f64,
    p: f64,
    row: usize,
    columns: &'a Columns,
    resources: &'a Resources,
    stats: &'a BatchStats,
    telemetry: Option<&'a ExecutionTelemetry>,
}

impl Expression {
    fn eval(&self, context: EvalContext<'_>) -> Result<f64> {
        let EvalContext {
            y,
            p,
            row,
            columns,
            resources,
            stats,
            telemetry,
        } = context;
        let e = |x: &Self| x.eval(context);
        finite(match self {
            Self::Y {} => y,
            Self::Pred {} => p,
            Self::Constant { value } => *value,
            Self::Column { name } => *columns
                .get(name)
                .and_then(|values| values.get(row))
                .ok_or_else(|| anyhow::anyhow!("fitness column {name:?} is not available"))?,
            Self::Resource { name } => read_resource(resources, *name)?,
            Self::TrainTimeSeconds {} => read_resource(resources, ResourceVar::FitSeconds)?,
            Self::InferenceTimeSeconds {} => read_resource(resources, ResourceVar::PredictSeconds)?,
            Self::TotalTimeSeconds {} => {
                read_resource(resources, ResourceVar::FitSeconds)?
                    + read_resource(resources, ResourceVar::PredictSeconds)?
            }
            Self::InferenceUsPerRow {} => read_resource(resources, ResourceVar::PredictUsPerRow)?,
            Self::TrainRowsPerSecond {} => match telemetry {
                Some(t) => t.train_rows_per_second(stats.row_count),
                None => {
                    let fit = read_resource(resources, ResourceVar::FitSeconds)?;
                    let rows = resources
                        .get(&ResourceVar::TrainRows)
                        .copied()
                        .unwrap_or((stats.row_count as f64).max(1.0));
                    if fit <= 1e-12 {
                        rows * 1e6
                    } else {
                        rows / fit
                    }
                }
            },
            Self::InferenceRowsPerSecond {} => {
                read_resource(resources, ResourceVar::PredictRowsPerSecond)?
            }
            Self::TrainMemoryBytes {} => read_resource(resources, ResourceVar::FitPeakHeapBytes)?,
            Self::InferenceMemoryBytes {} => {
                read_resource(resources, ResourceVar::PredictPeakHeapBytes)?
            }
            Self::TotalMemoryBytes {} => {
                read_resource(resources, ResourceVar::FitPeakHeapBytes)?
                    + read_resource(resources, ResourceVar::PredictPeakHeapBytes)?
            }
            Self::ModelBytes {} => read_resource(resources, ResourceVar::ModelBytes)?,
            Self::ComplexityStages {} => read_resource(resources, ResourceVar::StageCount)?,
            Self::ComplexityParameters {} => read_resource(resources, ResourceVar::ProgramNodes)?,
            Self::FeatureCount {} => {
                telemetry
                    .ok_or_else(|| {
                        anyhow::anyhow!("fitness telemetry feature_count is not available")
                    })?
                    .feature_count as f64
            }
            Self::FoldStd {} => telemetry
                .ok_or_else(|| anyhow::anyhow!("fitness telemetry fold_std is not available"))?
                .fold_std
                .max(0.0),
            Self::PredictionTurnover {} => stats.prediction_turnover,
            Self::RowIndex {} => row as f64,
            Self::RowFraction {} => {
                if stats.row_count <= 1 {
                    1.0
                } else {
                    (row as f64) / ((stats.row_count - 1) as f64)
                }
            }
            Self::RowCount {} => stats.row_count as f64,
            Self::Residual {} => y - p,
            Self::AbsError {} => (y - p).abs(),
            Self::SquaredError {} => (y - p).powi(2),
            Self::YMean {} => stats.y_mean,
            Self::YStd {} => stats.y_std,
            Self::PredMean {} => stats.pred_mean,
            Self::PredStd {} => stats.pred_std,
            Self::Abs { arg } => e(arg)?.abs(),
            Self::Neg { arg } => -e(arg)?,
            Self::Square { arg } => e(arg)?.powi(2),
            Self::Sqrt { arg } => e(arg)?.sqrt(),
            Self::Ln { arg } => e(arg)?.ln(),
            Self::Log1p { arg } => e(arg)?.ln_1p(),
            Self::Exp { arg } => e(arg)?.exp(),
            Self::Sign { arg } => {
                let v = e(arg)?;
                if v > 0.0 {
                    1.0
                } else if v < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            }
            Self::Tanh { arg } => e(arg)?.tanh(),
            Self::Sigmoid { arg } => 1.0 / (1.0 + (-e(arg)?).exp()),
            Self::Cbrt { arg } => e(arg)?.cbrt(),
            Self::Step { arg } => {
                if e(arg)? > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Clip { arg, min, max } => e(arg)?.clamp(*min, *max),
            Self::Add { left, right } => e(left)? + e(right)?,
            Self::Sub { left, right } => e(left)? - e(right)?,
            Self::Mul { left, right } => e(left)? * e(right)?,
            Self::Div { left, right } => e(left)? / e(right)?,
            Self::Min { left, right } => e(left)?.min(e(right)?),
            Self::Max { left, right } => e(left)?.max(e(right)?),
            Self::Pow { left, right } => e(left)?.powf(e(right)?),
            Self::Where {
                condition,
                then,
                otherwise,
            } => {
                if e(condition)? > 0.0 {
                    e(then)?
                } else {
                    e(otherwise)?
                }
            }
        })
    }

    pub(super) fn rows_with_context(
        &self,
        y: &[f64],
        p: &[f64],
        columns: &Columns,
        resources: &Resources,
        telemetry: Option<&ExecutionTelemetry>,
    ) -> Result<Vec<f64>> {
        let stats = BatchStats::compute(y, p);
        y.iter()
            .zip(p)
            .enumerate()
            .map(|(row, (&y, &p))| {
                self.eval(EvalContext {
                    y,
                    p,
                    row,
                    columns,
                    resources,
                    stats: &stats,
                    telemetry,
                })
            })
            .collect()
    }
}

/// Build [`Columns`] from `(name, values)` pairs.
#[cfg(test)]
pub(super) fn columns_from_pairs(pairs: &[(&str, Vec<f64>)]) -> Columns {
    pairs
        .iter()
        .map(|(name, values)| ((*name).to_owned(), values.clone()))
        .collect()
}

/// Bail with a uniform message when a specification reads columns the caller cannot supply.
pub fn require_no_columns(fitness: &Fitness, context: &str) -> Result<()> {
    let columns = fitness.columns();
    if !columns.is_empty() {
        bail!(
            "{context} cannot supply fitness columns {:?}",
            columns.into_iter().collect::<Vec<_>>()
        );
    }
    Ok(())
}

/// Bail with a uniform message when a specification reads resources the caller cannot supply.
pub fn require_no_resources(fitness: &Fitness, context: &str) -> Result<()> {
    let resources = fitness.resources();
    if !resources.is_empty() {
        bail!(
            "{context} cannot supply fitness resources {:?}",
            resources
                .into_iter()
                .map(ResourceVar::as_str)
                .collect::<Vec<_>>()
        );
    }
    Ok(())
}
