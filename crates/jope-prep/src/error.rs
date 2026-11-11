//! Typed failures and agent-friendly repair guidance for client-side split, conditioning, and
//! JMAT v1 matrix encoding.

use std::fmt;

use crate::landmarks::RecipeError;

/// Where the `jope-prep` preparation contract and refusal semantics are documented.
pub const PREP_DOCS_URL: &str = "crates/jope-prep/README.md";

/// `PrepError` is an enum for the client-preparation failure contract in this crate: every split,
/// conditioning, wire-envelope, or JMAT v1 refusal is returned as a typed value without panicking.
#[derive(Clone, Debug, PartialEq)]
pub enum PrepError {
    /// Partitioning into non-empty FIT and HOLDOUT sets requires at least two rows.
    InsufficientRows { rows: usize },
    /// The holdout fraction is outside the open interval `(0.0, 1.0)` or non-finite.
    InvalidHoldoutFraction { fraction: f64 },
    /// A table declares zero feature columns.
    ZeroFeatures,
    /// Matrix or target slice lengths do not match the declared `(rows, features)` shape.
    ShapeMismatch { expected: usize, actual: usize },
    /// Arithmetic on dimensions or offsets overflowed `usize`.
    ArithmeticOverflow,
    /// A row index in a split plan lies outside `0..total_rows` or violates strict ordering.
    InvalidSplitIndex { index: usize, total_rows: usize },
    /// A FIT target value is `NaN` or infinite.
    NonFiniteFitTarget { fit_row: usize },
    /// Underlying landmark recipe failed validation.
    LandmarkRecipe(RecipeError),
    /// A `6,368`-byte `FitLandmarkBundle` wire envelope failed magic, version, length, or SHA-256 verification.
    InvalidLandmarkWire { reason: &'static str },
    /// A JMAT v1 table or byte buffer failed header, column, alignment, or SHA-256 validation.
    InvalidJmat { reason: &'static str },
}

impl fmt::Display for PrepError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientRows { rows } => {
                write!(f, "split requires at least 2 rows, found {rows}")
            }
            Self::InvalidHoldoutFraction { fraction } => {
                write!(
                    f,
                    "holdout fraction {fraction} must be finite and strictly between 0.0 and 1.0"
                )
            }
            Self::ZeroFeatures => f.write_str("feature table must have at least one column"),
            Self::ShapeMismatch { expected, actual } => {
                write!(f, "expected {expected} elements, found {actual}")
            }
            Self::ArithmeticOverflow => f.write_str("table dimension or byte offset overflowed"),
            Self::InvalidSplitIndex { index, total_rows } => {
                write!(
                    f,
                    "split row index {index} is invalid for total_rows={total_rows}"
                )
            }
            Self::NonFiniteFitTarget { fit_row } => {
                write!(f, "FIT target at row {fit_row} is not finite")
            }
            Self::LandmarkRecipe(err) => write!(f, "landmark conditioning failed: {err}"),
            Self::InvalidLandmarkWire { reason } => {
                write!(f, "invalid landmark wire bundle: {reason}")
            }
            Self::InvalidJmat { reason } => write!(f, "invalid JMAT v1 matrix: {reason}"),
        }
    }
}

impl std::error::Error for PrepError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::LandmarkRecipe(err) => Some(err),
            _ => None,
        }
    }
}

impl From<RecipeError> for PrepError {
    fn from(value: RecipeError) -> Self {
        Self::LandmarkRecipe(value)
    }
}

/// `PrepResult` is the result alias used across fallible operations in `jope-prep`.
pub type PrepResult<T> = Result<T, PrepError>;

/// `PrepGuidance` is a struct for the typed repair surface contract in `jope-prep`: what invariant
/// refused the input, the concrete reason, likely fixes, documentation path, and local rerun command.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrepGuidance {
    /// The invariant or boundary control that refused the call.
    pub purpose: &'static str,
    /// Exact formatted error string.
    pub reason: String,
    /// Concrete caller remedies, most likely first.
    pub common_fixes: Vec<&'static str>,
    /// Repository documentation path for the contract.
    pub docs_url: &'static str,
    /// Deterministic local test command verifying the refusal path.
    pub repair_hint: &'static str,
}

impl PrepError {
    /// Typed repair guidance for every `PrepError` variant.
    pub fn guidance(&self) -> PrepGuidance {
        let (purpose, common_fixes, repair_hint) = match self {
            Self::InsufficientRows { .. } | Self::InvalidHoldoutFraction { .. } => (
                "guarantee non-empty, disjoint FIT and HOLDOUT partitions",
                vec![
                    "supply a dataset with at least 2 rows",
                    "pass a finite holdout_fraction such as 0.20 in (0.0, 1.0)",
                ],
                "rerun `cargo test -p jope-prep --test prep split_refusals`",
            ),
            Self::ZeroFeatures
            | Self::ShapeMismatch { .. }
            | Self::ArithmeticOverflow
            | Self::InvalidSplitIndex { .. } => (
                "enforce exact rectangular matrix dimensions and in-bounds split indices",
                vec![
                    "pass row-major x of length total_rows * features with features >= 1",
                    "pass target y of length total_rows",
                    "use the SplitPlan constructed for the same total_rows count",
                ],
                "rerun `cargo test -p jope-prep --test prep conditioning_refusals`",
            ),
            Self::NonFiniteFitTarget { .. } | Self::LandmarkRecipe(_) => (
                "condition landmarks strictly on finite FIT targets and valid probe bounds",
                vec![
                    "drop or repair training rows whose target y is NaN or infinite",
                    "ensure the FIT partition contains finite numeric feature values",
                ],
                "rerun `cargo test -p jope-prep --test prep holdout_poison_invariance`",
            ),
            Self::InvalidLandmarkWire { .. } => (
                "authenticate 6,368-byte JLMB v1 landmark bundles before conditioning",
                vec![
                    "transmit the exact 6,368-byte buffer returned by FitLandmarkBundle::to_wire_bytes",
                    "verify transport compression did not truncate or corrupt trailing SHA-256 bytes",
                ],
                "rerun `cargo test -p jope-prep --test prep wire_roundtrip_and_tamper`",
            ),
            Self::InvalidJmat { .. } => (
                "accept only well-formed, SHA-256-verified JMAT v1 column-major matrices",
                vec![
                    "ensure column names are unique and non-empty",
                    "ensure numeric cells are finite (use None for missing values)",
                    "verify trailing 32-byte SHA-256 checksum matches payload bytes",
                ],
                "rerun `cargo test -p jope-prep --test prep jmat_roundtrip_and_refusals`",
            ),
        };
        PrepGuidance {
            purpose,
            reason: self.to_string(),
            common_fixes,
            docs_url: PREP_DOCS_URL,
            repair_hint,
        }
    }
}
