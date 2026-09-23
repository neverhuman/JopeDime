//! The tasks the benchmark scores, and per task: the primary score (what the screen and deep
//! stages rank by, higher is better), the ladder metric `jope-arena` ranks by, and the tier-1
//! heads.

use feat_math::advanced_v3::{Objective, TaskSpec};
use jope_arena::elo::Direction;
use serde::{Deserialize, Serialize};

/// A benchmarked task. Spelled as the dataset catalog spells it.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Task {
    /// Numeric target; the frozen kernel's heads.
    Regression,
    /// 0/1 target; the schema-bound v3 heads that produce probabilities.
    Binary,
    /// Integer class codes; the schema-bound v3 multiclass heads.
    Multiclass,
}

impl Task {
    /// Every benchmarked task.
    pub const ALL: [Task; 3] = [Task::Regression, Task::Binary, Task::Multiclass];

    /// The catalog spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Regression => "regression",
            Self::Binary => "binary",
            Self::Multiclass => "multiclass",
        }
    }

    /// From the catalog spelling.
    pub fn parse(text: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|task| task.as_str() == text)
    }

    /// The primary score, higher is better: `r2`, `auc` (ROC) or `macro_f1`. Screen ranks by
    /// it raw; deep ranks by its variance-penalised form ([`crate::eval::robust_fitness_for`]).
    pub fn score_metric(self) -> &'static str {
        match self {
            Self::Regression => "r2",
            Self::Binary => "auc",
            Self::Multiclass => "macro_f1",
        }
    }

    /// The metric a `jope-arena` track built from the reports ranks by, and its direction. The
    /// published regression ladder ranks by RMSE, so the regression track does too.
    pub fn ladder_metric(self) -> (&'static str, Direction) {
        match self {
            Self::Regression => ("rmse", Direction::Minimize),
            Self::Binary => ("auc", Direction::Maximize),
            Self::Multiclass => ("macro_f1", Direction::Maximize),
        }
    }

    /// Every metric the evaluator records for the task, in report order.
    pub fn metrics(self) -> &'static [&'static str] {
        match self {
            Self::Regression => &["rmse", "r2"],
            Self::Binary => &["auc", "logloss", "brier"],
            Self::Multiclass => &["macro_f1", "accuracy"],
        }
    }

    /// The tier-1 heads: every head the task's grammar accepts as the final stage of a
    /// single-head program. Regression: every distinct head of the frozen registry (aliases
    /// `logistic`, `gbm` and `ensemble_ridge_gbm` are spellings of `ridge`, `histgbm` and
    /// `ensemble`; `ridge_gcv` is accepted by the interpreter though absent from the registry).
    /// Binary and multiclass: the schema-bound v3 heads (`portfolio` needs members and is not a
    /// single-head program).
    pub fn default_heads(self) -> &'static [&'static str] {
        match self {
            Self::Regression => &[
                "ridge",
                "ridge_gcv",
                "elasticnet",
                "histgbm",
                "histgbm2",
                "ksmooth",
                "ensemble",
                "krr",
                "gpr",
                "stack",
            ],
            Self::Binary => &[
                "logit_ridge",
                "logistic",
                "tab_boost",
                "extra_trees",
                "gam_boost",
            ],
            Self::Multiclass => &[
                "multiclass_ridge",
                "multiclass_ovr_logistic",
                "multiclass_diagonal_lda",
                "multiclass_nearest_centroid",
            ],
        }
    }

    /// The schema-bound v3 task spec (`None` for regression, which runs on the frozen kernel).
    pub fn v3_spec(self) -> Option<TaskSpec> {
        match self {
            Self::Regression => None,
            Self::Binary => Some(
                TaskSpec::binary(Objective::RocAuc).expect("roc_auc is a valid binary objective"),
            ),
            Self::Multiclass => Some(
                TaskSpec::multiclass(Objective::MacroF1)
                    .expect("macro_f1 is a valid multiclass objective"),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spellings_round_trip_and_specs_validate() {
        for task in Task::ALL {
            assert_eq!(Task::parse(task.as_str()), Some(task));
            assert_eq!(serde_json::to_value(task).unwrap(), task.as_str());
            assert!(task.metrics().contains(&task.score_metric()));
            assert!(task.metrics().contains(&task.ladder_metric().0));
            assert!(!task.default_heads().is_empty());
            if let Some(spec) = task.v3_spec() {
                spec.validate().unwrap();
            }
        }
        assert_eq!(Task::parse("timeseries"), None);
        assert_eq!(Task::Regression.ladder_metric().1, Direction::Minimize);
        assert_eq!(Task::Binary.ladder_metric().1, Direction::Maximize);
    }
}
