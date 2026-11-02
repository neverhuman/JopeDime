use anyhow::{ensure, Result};
use jope_datasets::catalog::Task;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub fn sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(rename = "Metric")]
pub enum ExperimentMetric {
    Rmse,
    Auroc,
    LogLoss,
}
/// Compatibility path; the concrete enum has a domain-specific name.
pub type Metric = ExperimentMetric;

impl ExperimentMetric {
    pub fn for_task(task: Task) -> Result<Self> {
        Ok(match task {
            Task::Regression => Self::Rmse,
            Task::Binary => Self::Auroc,
            Task::Multiclass => Self::LogLoss,
            other => anyhow::bail!("task {} is outside this panel", other.as_str()),
        })
    }
    pub fn validate_score(self, value: f64) -> Result<()> {
        ensure!(
            value.is_finite() && value >= 0.,
            "nonfinite or negative primary score"
        );
        ensure!(self != Self::Auroc || value <= 1., "AUROC exceeds one");
        Ok(())
    }
    /// All primary metrics become a smaller-is-better selection loss.
    pub fn loss(self, value: f64) -> Result<f64> {
        self.validate_score(value)?;
        Ok(if self == Self::Auroc {
            1. - value
        } else {
            value
        })
    }
}
