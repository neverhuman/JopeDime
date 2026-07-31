use crate::metrics::{accuracy_at, auc, average_precision, brier, f1_at, logloss, mae, r2, rmse};
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const ARTIFACT_SCHEMA: &str = "battle-artifact-v2";

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskKind {
    Regression,
    Binary,
    Timeseries,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Objective {
    SquaredError,
    Huber,
    R2,
    Rmse,
    Mae,
    LogLoss,
    RocAuc,
    PrAuc,
    BrierScore,
    Accuracy,
    BalancedAccuracy,
    F1,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TaskSpec {
    pub kind: TaskKind,
    pub objective: Objective,
}

impl TaskSpec {
    pub fn regression(objective: Objective) -> Result<Self> {
        let task = Self {
            kind: TaskKind::Regression,
            objective,
        };
        task.validate()?;
        Ok(task)
    }

    pub fn binary(objective: Objective) -> Result<Self> {
        let task = Self {
            kind: TaskKind::Binary,
            objective,
        };
        task.validate()?;
        Ok(task)
    }

    pub fn timeseries(objective: Objective) -> Result<Self> {
        let task = Self {
            kind: TaskKind::Timeseries,
            objective,
        };
        task.validate()?;
        Ok(task)
    }

    pub fn validate(&self) -> Result<()> {
        let valid = match self.kind {
            TaskKind::Regression | TaskKind::Timeseries => matches!(
                self.objective,
                Objective::SquaredError
                    | Objective::Huber
                    | Objective::R2
                    | Objective::Rmse
                    | Objective::Mae
            ),
            TaskKind::Binary => matches!(
                self.objective,
                Objective::LogLoss
                    | Objective::RocAuc
                    | Objective::PrAuc
                    | Objective::BrierScore
                    | Objective::Accuracy
                    | Objective::BalancedAccuracy
                    | Objective::F1
            ),
        };
        if !valid {
            bail!(
                "objective {:?} is incompatible with {:?}",
                self.objective,
                self.kind
            );
        }
        Ok(())
    }

    pub fn score(&self, y: &[f64], predictions: &[f64], threshold: f64) -> f64 {
        match self.objective {
            Objective::SquaredError => -mean_squared_error(y, predictions),
            Objective::Huber => -huber_loss(y, predictions, 1.0),
            Objective::Rmse => -rmse(y, predictions),
            Objective::Mae => -mae(y, predictions),
            Objective::R2 => r2(y, predictions),
            Objective::LogLoss => -logloss(y, predictions),
            Objective::RocAuc => auc(y, predictions),
            Objective::PrAuc => average_precision(y, predictions),
            Objective::BrierScore => -brier(y, predictions),
            Objective::Accuracy => accuracy_at(y, predictions, threshold),
            Objective::BalancedAccuracy => balanced_accuracy_at(y, predictions, threshold),
            Objective::F1 => f1_at(y, predictions, threshold),
        }
    }
}

fn mean_squared_error(y: &[f64], predictions: &[f64]) -> f64 {
    let n = y.len().min(predictions.len());
    if n == 0 {
        return 0.0;
    }
    (0..n)
        .map(|row| (y[row] - predictions[row]).powi(2))
        .sum::<f64>()
        / n as f64
}

fn huber_loss(y: &[f64], predictions: &[f64], delta: f64) -> f64 {
    let n = y.len().min(predictions.len());
    if n == 0 {
        return 0.0;
    }
    (0..n)
        .map(|row| {
            let error = (y[row] - predictions[row]).abs();
            if error <= delta {
                0.5 * error * error
            } else {
                delta * (error - 0.5 * delta)
            }
        })
        .sum::<f64>()
        / n as f64
}

/// Deterministic class-stratified folds for binary labels. Regression labels
/// use deterministic round-robin assignment; the legacy `fold_of` API is not
/// called or changed.
pub fn stratified_fold_assignments(y: &[f64], folds: usize) -> Vec<usize> {
    let folds = folds.max(2);
    let binary = y.iter().all(|v| *v == 0.0 || *v == 1.0);
    let mut out = vec![0usize; y.len()];
    if binary {
        for class in [0.0, 1.0] {
            let mut rank = 0usize;
            for (row, value) in y.iter().enumerate() {
                if *value == class {
                    out[row] = rank % folds;
                    rank += 1;
                }
            }
        }
    } else {
        for (row, fold) in out.iter_mut().enumerate() {
            *fold = row % folds;
        }
    }
    out
}

pub fn balanced_accuracy_at(y: &[f64], p: &[f64], threshold: f64) -> f64 {
    let n = y.len().min(p.len());
    let (mut tp, mut tn, mut pos, mut neg) = (0usize, 0usize, 0usize, 0usize);
    for i in 0..n {
        if y[i] == 1.0 {
            pos += 1;
            tp += usize::from(p[i] >= threshold);
        } else {
            neg += 1;
            tn += usize::from(p[i] < threshold);
        }
    }
    if pos == 0 || neg == 0 {
        return 0.5;
    }
    0.5 * (tp as f64 / pos as f64 + tn as f64 / neg as f64)
}

pub(super) fn stable_hash<T: Serialize>(value: &T) -> Result<String> {
    let bytes = serde_json::to_vec(value)?;
    let digest = Sha256::digest(bytes);
    Ok(digest.iter().map(|b| format!("{b:02x}")).collect())
}

pub(super) fn text_hash(text: &str) -> String {
    Sha256::digest(text.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
