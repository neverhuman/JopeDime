//! Typed failures for comparator fitting, prediction and scoring, with agent-friendly repair
//! guidance for every failure.

use std::fmt;

/// `ComparatorError` is an enum for the comparator failure contract in this crate: every refusal
/// or runtime failure of a fit, prediction or score, reported as a value instead of a panic.
#[derive(Clone, Debug, PartialEq)]
pub enum ComparatorError {
    /// The training matrix has no rows.
    EmptyFit,
    /// A score was requested over zero rows.
    EmptyScore,
    /// The matrix declares zero feature columns.
    NoFeatures,
    /// A buffer length does not match the declared shape.
    ShapeMismatch { expected: usize, actual: usize },
    /// `rows × cols` overflows `usize`.
    ShapeOverflow,
    /// The training matrix has more rows than the row-index width supports.
    TooManyRows { limit: usize },
    /// The model family cannot fit this many feature columns within its bounded cost.
    TooManyFeatures { limit: usize },
    /// A training target is NaN or infinite.
    NonFiniteTarget { row: usize },
    /// A feature value is infinite. NaN marks a missing value and is accepted.
    InfiniteFeature { row: usize, column: usize },
    /// A parameter is outside its documented range.
    InvalidParameter { name: &'static str },
    /// The linear system is singular for an unregularized fit.
    SingularSystem,
    /// The caller's deadline passed before the fit finished.
    DeadlineExceeded,
    /// A prediction came out NaN or infinite.
    NonFinitePrediction { row: usize },
}

impl fmt::Display for ComparatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyFit => write!(f, "the training matrix has no rows"),
            Self::EmptyScore => write!(f, "a score needs at least one row"),
            Self::NoFeatures => write!(f, "the matrix has no feature columns"),
            Self::ShapeMismatch { expected, actual } => {
                write!(f, "expected {expected} values, found {actual}")
            }
            Self::ShapeOverflow => write!(f, "the declared matrix shape overflows"),
            Self::TooManyRows { limit } => write!(f, "more than {limit} training rows"),
            Self::TooManyFeatures { limit } => write!(f, "more than {limit} feature columns"),
            Self::NonFiniteTarget { row } => write!(f, "target in row {row} is not finite"),
            Self::InfiniteFeature { row, column } => {
                write!(f, "feature {column} in row {row} is infinite")
            }
            Self::InvalidParameter { name } => write!(f, "parameter `{name}` is out of range"),
            Self::SingularSystem => write!(f, "the unregularized linear system is singular"),
            Self::DeadlineExceeded => write!(f, "the fit deadline passed"),
            Self::NonFinitePrediction { row } => {
                write!(f, "prediction for row {row} is not finite")
            }
        }
    }
}

impl std::error::Error for ComparatorError {}

/// `ComparatorResult` is the result alias used by every fallible operation in this crate.
pub type ComparatorResult<T> = Result<T, ComparatorError>;

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
