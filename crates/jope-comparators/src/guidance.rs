//! Agent-friendly repair guidance for every [`ComparatorError`].

use crate::error::ComparatorError;

/// Where the comparator contract, its inputs and its refusals are documented.
pub const COMPARATORS_DOCS_URL: &str = "crates/jope-comparators/README.md";

/// `ComparatorGuidance` is a struct for the typed repair surface contract in this module: what
/// control refused, the concrete reason, the usual fixes and the local rerun that proves a fix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComparatorGuidance {
    /// The control that refused the operation.
    pub purpose: &'static str,
    /// The concrete error, preserved verbatim.
    pub reason: String,
    /// Usual fixes, most likely first.
    pub common_fixes: Vec<&'static str>,
    /// Local documentation for the contract.
    pub docs_url: &'static str,
    /// The bounded local rerun that proves a fix.
    pub repair_hint: &'static str,
}

impl ComparatorError {
    /// Repair guidance for this error.
    pub fn guidance(&self) -> ComparatorGuidance {
        let (purpose, common_fixes, repair_hint) = match self {
            Self::EmptyFit
            | Self::EmptyScore
            | Self::NoFeatures
            | Self::ShapeMismatch { .. }
            | Self::ShapeOverflow => (
                "accept only a rectangular, non-empty row-major matrix",
                vec![
                    "pass rows × cols values with cols > 0",
                    "predict with the same feature columns used to fit",
                    "pass one target per training row",
                ],
                "rerun `cargo nextest run -p jope-comparators --test comparators refusals`",
            ),
            Self::TooManyRows { .. } | Self::TooManyFeatures { .. } => (
                "keep every fit within its bounded memory and time cost",
                vec![
                    "subsample rows before fitting",
                    "select or reduce features for linear models",
                ],
                "rerun the refusal tests, then the run with the reduced table",
            ),
            Self::NonFiniteTarget { .. } | Self::InfiniteFeature { .. } => (
                "fit only finite targets; NaN features mean missing, infinities are refused",
                vec![
                    "drop or repair rows whose target is NaN or infinite",
                    "replace infinite feature values with NaN (missing) or a finite cap",
                ],
                "rerun `cargo nextest run -p jope-comparators --test comparators refusals`",
            ),
            Self::InvalidParameter { .. } | Self::SingularSystem => (
                "fit only with parameters inside their documented ranges",
                vec![
                    "start from a preset such as ForestParams::random_forest",
                    "use a positive ridge alpha when columns may be collinear",
                ],
                "rerun the refusal tests for the named parameter",
            ),
            Self::DeadlineExceeded => (
                "stop a comparator fit at the caller's wall-clock ceiling",
                vec![
                    "raise the comparator ceiling for this effort tier",
                    "reduce trees or rounds in the comparator preset",
                ],
                "rerun the fit with a larger Deadline and compare its duration",
            ),
            Self::NonFinitePrediction { .. } => (
                "report only finite predictions and scores",
                vec![
                    "check the training targets' scale for overflow",
                    "reduce the boosting learning rate",
                ],
                "rerun `cargo nextest run -p jope-comparators --test comparators properties`",
            ),
        };
        ComparatorGuidance {
            purpose,
            reason: self.to_string(),
            common_fixes,
            docs_url: COMPARATORS_DOCS_URL,
            repair_hint,
        }
    }
}
