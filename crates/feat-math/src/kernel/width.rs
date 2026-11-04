//! The fitted-width contract (kernel2 WP-1.7).
//!
//! `Model::predict` indexes fitted per-column state by the column index of its
//! input, so rows narrower or wider than a stage was fitted on can panic, or
//! silently read state fitted for another column. `predict` now refuses
//! zero-width rows; this module checks the full fitted-width contract ahead of it:
//!
//! - [`required_input_width`]: the input width a fitted stage accepts, read
//!   from its fitted state.
//! - [`check_fitted_state`]: whether that state agrees with itself, so that no
//!   accepted width can make the stage read out of bounds.
//! - [`output_width`]: the width the stage produces, including the 512-column
//!   cap and the `[0.0]` padding that `apply_op` ends with.
//! - [`required_head_width`]: the width a head accepts, where its fitted state
//!   reveals one.
//! - [`check_input_width`] walks a model with those. [`Model::try_predict`]
//!   returns a [`WidthMismatch`] where `predict` would panic or misread, and
//!   otherwise returns `predict`'s own result.
//!
//! Every match over `FittedOp` or `FittedHead` here names each variant, with no
//! wildcard arm, so a new variant does not compile until its width is decided.
//!
//! Outside the contract: the internal state of heads, which read columns
//! defensively and never panic on width (their tables are trusted); the members
//! of a portfolio; and a fitted size so large that allocating its output
//! exhausts memory without overflowing (a `degree`, `bins` or `k` in the
//! billions), which aborts as it would in `predict`.
use super::{Error, Result};
use crate::heads::FittedHead;
use crate::model::Model;
use crate::ops::FittedOp;
use crate::portfolio::PortfolioModel;
use crate::stack::StackModel;
use crate::Rows;
use std::fmt;

mod models;
mod ops;

pub use models::{check_input_width, required_head_width};
pub(crate) use ops::stage_name;
pub use ops::{check_fitted_state, output_width, required_input_width};
use ops::{finished_width, past_max};

/// The widest row a `Vec<f64>` can hold. A wider output row is refused rather
/// than left to the capacity-overflow panic of `Vec`.
const MAX_ROW: usize = isize::MAX as usize / std::mem::size_of::<f64>();

/// The input width that a fitted stage or head accepts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WidthReq {
    /// Any width, zero included.
    Any,
    /// Exactly this many columns: the fitted state holds one entry per input column.
    Exactly(usize),
    /// At least this many columns: the fitted state names input columns below this.
    AtLeast(usize),
}

impl WidthReq {
    /// Whether rows `width` columns wide satisfy the requirement.
    pub fn admits(self, width: usize) -> bool {
        match self {
            WidthReq::Any => true,
            WidthReq::Exactly(n) => width == n,
            WidthReq::AtLeast(n) => width >= n,
        }
    }
}

impl fmt::Display for WidthReq {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WidthReq::Any => f.write_str("any number of"),
            WidthReq::Exactly(n) => write!(f, "exactly {n}"),
            WidthReq::AtLeast(n) => write!(f, "at least {n}"),
        }
    }
}

/// What [`check_input_width`] found wrong.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MismatchKind {
    /// Input row `row` is `got` columns wide, and every row must match row 0.
    /// Reported at stage 0.
    RaggedInput { row: usize },
    /// The input to fitted stage `stage` does not satisfy `expected`.
    StageInput,
    /// The input to the head does not satisfy `expected`. `stage` is the number
    /// of fitted stages.
    HeadInput,
    /// The fitted state of stage `stage` disagrees with itself (`reason`), so no
    /// input width is safe for it.
    InconsistentState { reason: &'static str },
    /// The output of stage `stage` for rows `got` wide overflows, or is wider than any row can be.
    OutputOverflow,
}

/// Why [`Model::try_predict`] refused an input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WidthMismatch {
    /// The index of the fitted stage whose input failed; the number of fitted
    /// stages for the head.
    pub stage: usize,
    /// The input width that stage accepts.
    pub expected: WidthReq,
    /// The input width it would have received.
    pub got: usize,
    /// What failed.
    pub kind: MismatchKind,
}

impl fmt::Display for WidthMismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let WidthMismatch {
            stage,
            expected,
            got,
            kind,
        } = self;
        match kind {
            MismatchKind::RaggedInput { row } => write!(
                f,
                "ragged input: row {row} has {got} columns, every row needs {expected}"
            ),
            MismatchKind::StageInput => {
                write!(f, "stage {stage} needs {expected} input columns, got {got}")
            }
            MismatchKind::HeadInput => {
                write!(f, "the head needs {expected} input columns, got {got}")
            }
            MismatchKind::InconsistentState { reason } => {
                write!(f, "stage {stage} has inconsistent fitted state: {reason}")
            }
            MismatchKind::OutputOverflow => {
                write!(
                    f,
                    "stage {stage} output width overflows at input width {got}"
                )
            }
        }
    }
}

impl std::error::Error for WidthMismatch {}

/// Refuse a row with no feature columns before a fitted operation can use its width.
pub(crate) fn reject_zero_width_rows(x: &Rows) -> std::result::Result<(), WidthMismatch> {
    if x.iter().any(Vec::is_empty) {
        Err(WidthMismatch {
            stage: 0,
            expected: WidthReq::AtLeast(1),
            got: 0,
            kind: MismatchKind::StageInput,
        })
    } else {
        Ok(())
    }
}
