//! The JA-03 bake-off roster: native estimator candidates after alias deduplication (JA-01 v2).
//!
//! Excluded, with the reason recorded in `EXCLUSIONS`: the internal ensembles `ensemble`,
//! `stack` and `portfolio`, which fix their own members and folds; the disabled
//! `multiclass_ridge`; and every alias (`logistic`, `gbm`, `ensemble_ridge_gbm`).
use jope_datasets::catalog::Task;

/// Whether a candidate may enter a bake-off without an explicit receipt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admission {
    Enabled,
    /// Registry admission is `receipt`: it needs an explicit admission receipt first.
    ReceiptRequired,
    /// Source exists, but its registry or runtime admission is not yet confirmed (JA-01 v2).
    Unconfirmed,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateSpec {
    pub id: &'static str,
    pub task: Task,
    pub admission: Admission,
}

/// Candidates left out of the roster, each with its reason.
pub const EXCLUSIONS: &[(&str, &str)] = &[
    (
        "rust.ensemble",
        "internal grammar ensemble with fixed members and folds",
    ),
    (
        "rust.stack",
        "internal stack with fixed member library and folds",
    ),
    (
        "rust.portfolio.core_v1",
        "internal portfolio with fixed members and folds",
    ),
    ("rust.multiclass_ridge", "registry admission is disabled"),
    ("logistic", "alias of ridge; not a separate candidate"),
    ("gbm", "alias of histgbm; not a separate candidate"),
    (
        "ensemble_ridge_gbm",
        "alias of ensemble; not a separate candidate",
    ),
];

pub fn bakeoff_roster(task: Task) -> Vec<CandidateSpec> {
    use Admission::{Enabled, ReceiptRequired, Unconfirmed};
    let rows: &[(&'static str, Admission)] = match task {
        Task::Regression => &[
            ("rust.ridge", Enabled),
            ("rust.ridge_gcv", Enabled),
            ("rust.elasticnet", Enabled),
            ("rust.histgbm", Enabled),
            ("rust.histgbm2", Enabled),
            ("rust.ksmooth", Enabled),
            ("rust.krr", Enabled),
            ("rust.gpr", Enabled),
            ("rust.gpr.rbf_linear", Enabled),
            ("rust.hurdle2", Unconfirmed),
            ("rust.tab_boost", ReceiptRequired),
            ("rust.extra_trees", ReceiptRequired),
            ("rust.gam_boost", ReceiptRequired),
            ("rust.pls", ReceiptRequired),
        ],
        Task::Binary => &[
            ("rust.logit_ridge", Enabled),
            ("rust.tab_boost", ReceiptRequired),
            ("rust.extra_trees", ReceiptRequired),
            ("rust.gam_boost", ReceiptRequired),
        ],
        Task::Multiclass => &[
            ("rust.multiclass_ovr_logistic", Unconfirmed),
            ("rust.multiclass_diagonal_lda", Unconfirmed),
            ("rust.multiclass_nearest_centroid", Unconfirmed),
        ],
        _ => &[],
    };
    rows.iter()
        .map(|&(id, admission)| CandidateSpec {
            id,
            task,
            admission,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_regression_roster_has_no_aliases_or_internal_ensembles() {
        let ids: Vec<&str> = bakeoff_roster(Task::Regression)
            .iter()
            .map(|c| c.id)
            .collect();
        for (excluded, _) in EXCLUSIONS {
            assert!(
                !ids.contains(excluded),
                "{excluded} must not enter the roster"
            );
        }
        let unique: std::collections::BTreeSet<_> = ids.iter().collect();
        assert_eq!(unique.len(), ids.len());
    }

    #[test]
    fn receipt_gated_candidates_are_marked_and_not_enabled_by_default() {
        let binary = bakeoff_roster(Task::Binary);
        let gated: Vec<_> = binary
            .iter()
            .filter(|c| c.admission == Admission::ReceiptRequired)
            .map(|c| c.id)
            .collect();
        assert_eq!(
            gated,
            vec!["rust.tab_boost", "rust.extra_trees", "rust.gam_boost"]
        );
    }

    #[test]
    fn xgboost_and_histgbm_are_never_listed_as_the_same_candidate() {
        for task in [Task::Regression, Task::Binary, Task::Multiclass] {
            assert!(bakeoff_roster(task)
                .iter()
                .all(|c| !c.id.contains("xgboost")));
        }
    }
}
