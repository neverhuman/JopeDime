//! Typed failures for comparator fitting, prediction and scoring.

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
