//! The JA-03 bake-off roster: native estimator candidates after alias deduplication (JA-01 v2).
//!
//! Excluded, with the reason recorded in `EXCLUSIONS`: the internal ensembles `ensemble`,
//! `stack` and `portfolio`, which fix their own members and folds; the disabled
//! `multiclass_ridge`; and every alias (`logistic`, `gbm`, `ensemble_ridge_gbm`).
//!
//! This roster is a source scaffold for the product heads. It does not enumerate the model-zoo
//! catalog. The catalog is bound by hash only (`CATALOG_BINDING`), and its counts are not
//! algorithm deduplication.
use jope_datasets::catalog::Task;
use serde::{Deserialize, Serialize};

/// Whether a candidate may enter a bake-off without an explicit receipt.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Admission {
    Enabled,
    /// Registry admission is `receipt`: it needs an explicit admission receipt first.
    ReceiptRequired,
    /// Source exists, but its registry or runtime admission is not yet confirmed (JA-01 v2).
    Unconfirmed,
}

/// What is known about a candidate's runtime. Unknown stays unknown: nothing here implies a
/// fit, predict or reload result.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeCapability {
    /// Source exists in this repository. Fit, predict and reload are not verified.
    SourcePresent,
    /// Neither source nor runtime admission is confirmed.
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateSpec {
    pub id: &'static str,
    pub task: Task,
    pub admission: Admission,
    pub runtime: RuntimeCapability,
}

/// Binding to the model-zoo catalog that JA-01 v2 read. A hash binding only.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CatalogBinding {
    pub source_head: &'static str,
    pub catalog_sha256: &'static str,
    pub entries: usize,
    /// Count of distinct entry names after stripping the task suffix. This is a string-name count,
    /// not an independently reviewed algorithm deduplication, and it is not a count of runnable or
    /// admitted algorithms.
    pub string_name_count_after_suffix_strip: usize,
}

pub const CATALOG_BINDING: CatalogBinding = CatalogBinding {
    source_head: "3f34f483c832823e0d506013e7865b294faa21a2",
    catalog_sha256: "a8d63e942ac78bb2a94dec4eed38d9a1c599f54d52559a252cf92ecf1e32f410",
    entries: 655,
    string_name_count_after_suffix_strip: 415,
};

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
    use RuntimeCapability::{SourcePresent, Unknown};
    let rows: &[(&'static str, Admission, RuntimeCapability)] = match task {
        Task::Regression => &[
            ("rust.ridge", Enabled, SourcePresent),
            ("rust.ridge_gcv", Enabled, SourcePresent),
            ("rust.elasticnet", Enabled, SourcePresent),
            ("rust.histgbm", Enabled, SourcePresent),
            ("rust.histgbm2", Enabled, SourcePresent),
            ("rust.ksmooth", Enabled, SourcePresent),
            ("rust.krr", Enabled, SourcePresent),
            ("rust.gpr", Enabled, SourcePresent),
            ("rust.gpr.rbf_linear", Enabled, SourcePresent),
            ("rust.hurdle2", Unconfirmed, SourcePresent),
            ("rust.tab_boost", ReceiptRequired, SourcePresent),
            ("rust.extra_trees", ReceiptRequired, SourcePresent),
            ("rust.gam_boost", ReceiptRequired, SourcePresent),
            ("rust.pls", ReceiptRequired, SourcePresent),
        ],
        Task::Binary => &[
            ("rust.logit_ridge", Enabled, SourcePresent),
            ("rust.tab_boost", ReceiptRequired, SourcePresent),
            ("rust.extra_trees", ReceiptRequired, SourcePresent),
            ("rust.gam_boost", ReceiptRequired, SourcePresent),
        ],
        Task::Multiclass => &[
            ("rust.multiclass_ovr_logistic", Unconfirmed, Unknown),
            ("rust.multiclass_diagonal_lda", Unconfirmed, Unknown),
            ("rust.multiclass_nearest_centroid", Unconfirmed, Unknown),
        ],
        _ => &[],
    };
    rows.iter()
        .map(|&(id, admission, runtime)| CandidateSpec {
            id,
            task,
            admission,
            runtime,
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
    fn unconfirmed_multiclass_candidates_carry_unknown_runtime_not_source_presence() {
        for c in bakeoff_roster(Task::Multiclass) {
            assert_eq!(c.admission, Admission::Unconfirmed);
            assert_eq!(c.runtime, RuntimeCapability::Unknown);
        }
    }

    #[test]
    fn the_catalog_count_is_labelled_as_a_string_name_count() {
        assert_eq!(CATALOG_BINDING.string_name_count_after_suffix_strip, 415);
        assert_eq!(CATALOG_BINDING.entries, 655);
        assert_eq!(CATALOG_BINDING.catalog_sha256.len(), 64);
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
