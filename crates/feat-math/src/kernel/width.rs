//! The fitted-width contract (kernel2 WP-1.7).
//!
//! `Model::predict` indexes fitted per-column state by the column index of its
//! input, so rows narrower or wider than a stage was fitted on can panic, or
//! silently read state fitted for another column. `predict` stays as it is: its
//! bits are frozen and its callers guard widths themselves. This module states
//! the contract and checks it ahead of `predict`:
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
    /// The output of stage `stage` for rows `got` wide is wider than any row.
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

/// The input width that `op` accepts, read from its fitted state.
///
/// - `Exactly(n)`: per-column state indexed by the input column (a wider input
///   panics in `apply_op`, a narrower one is silently misread). The dot-product
///   stages (`rff_cos`, `rff_sin`, `rand_proj`, `oblique_hinge`,
///   `landmark_dist`, `nystrom`, `svd_project`) are in this class by meaning,
///   although `apply_op` truncates their dot products instead of panicking.
/// - `AtLeast(max + 1)`: fitted column indices (`select_corr`, `poly_interact`,
///   `pair_prod`, `pair_ratio`); wider input is read at the same indices.
/// - `AtLeast(1)`: `roll`, which divides by the width.
/// - `Any`: stateless stages, and `map` and `derive`, whose expressions
///   evaluate to 0.0 where they read a column beyond the row.
///
/// The requirement is never capped at `MAX_WIDTH`: a first stage may be fitted
/// on raw rows of any width.
///
/// Every variant is matched explicitly. For WP-1.7 a scratch variant was added
/// to `FittedOp`, and the crate stopped compiling with E0004 at this match (and
/// at the other exhaustive matches); the scratch variant was then removed.
pub fn required_input_width(op: &FittedOp) -> WidthReq {
    match op {
        FittedOp::Standardize { means, .. } => WidthReq::Exactly(means.len()),
        FittedOp::RobustScale { medians, .. } => WidthReq::Exactly(medians.len()),
        FittedOp::Rank { sorted_cols, .. } | FittedOp::Blom { sorted_cols, .. } => {
            WidthReq::Exactly(sorted_cols.len())
        }
        FittedOp::QuantileBins { edges, .. }
        | FittedOp::BinOnehot { edges, .. }
        | FittedOp::TargetBinEncode { edges, .. } => WidthReq::Exactly(edges.len()),
        FittedOp::Hinge { knots, .. } | FittedOp::Spline { knots, .. } => {
            WidthReq::Exactly(knots.len())
        }
        FittedOp::WinsorClip { lo, .. } => WidthReq::Exactly(lo.len()),
        FittedOp::Impute { fill, .. } => WidthReq::Exactly(fill.len()),
        FittedOp::RffCos { w, .. }
        | FittedOp::RffSin { w, .. }
        | FittedOp::RandProj { w, .. }
        | FittedOp::ObliqueHinge { w, .. } => first_row_width(w),
        FittedOp::LandmarkDist { landmarks, .. } | FittedOp::Nystrom { landmarks, .. } => {
            first_row_width(landmarks)
        }
        FittedOp::SvdProject { means, .. } => WidthReq::Exactly(means.len()),
        FittedOp::SelectCorr { indices, .. } => past_max(indices.iter().copied()),
        FittedOp::PolyInteract { pairs, .. }
        | FittedOp::PairProd { pairs, .. }
        | FittedOp::PairRatio { pairs, .. } => past_max(pairs.iter().flat_map(|&(a, b)| [a, b])),
        FittedOp::Roll { .. } => WidthReq::AtLeast(1),
        FittedOp::Cheb { .. }
        | FittedOp::RowStats { .. }
        | FittedOp::Sigmoid { .. }
        | FittedOp::Step { .. }
        | FittedOp::NormRow { .. }
        | FittedOp::Polar { .. }
        | FittedOp::Lag { .. }
        | FittedOp::FftMag { .. }
        | FittedOp::RowSpectral { .. }
        | FittedOp::RowDiffStats { .. }
        | FittedOp::Symlog { .. }
        | FittedOp::MapExpr { .. }
        | FittedOp::DeriveExpr { .. } => WidthReq::Any,
    }
}

/// Checks that the fitted state of `op` agrees with itself: the per-column
/// vectors have one length, the fitted column indices lie inside the fitted
/// columns, and every table `apply_op` reads an element of is non-empty. With
/// that, no width [`required_input_width`] admits can make `apply_op` read out
/// of bounds or panic. State that `fit_op` produced always passes; the check is
/// for state that arrived another way, such as a stored or edited model.
/// Returns a short reason on failure.
pub fn check_fitted_state(op: &FittedOp) -> std::result::Result<(), &'static str> {
    match op {
        FittedOp::Standardize { means, scales, .. } => ensure(
            scales.len() == means.len(),
            "means and scales differ in length",
        ),
        FittedOp::RobustScale {
            medians, scales, ..
        } => ensure(
            scales.len() == medians.len(),
            "medians and scales differ in length",
        ),
        FittedOp::BinOnehot { bins, .. } => ensure(*bins > 0, "bins is zero"),
        FittedOp::Hinge { knots, .. } => ensure(!knots.is_empty(), "there are no knot columns"),
        FittedOp::WinsorClip { lo, hi, .. } => {
            ensure(hi.len() == lo.len(), "lo and hi differ in length")?;
            ensure(
                lo.iter().zip(hi).all(|(low, high)| low <= high),
                "a lower clip bound is above its upper bound or NaN",
            )
        }
        FittedOp::RffCos { w, b, .. } | FittedOp::RffSin { w, b, .. } => {
            ensure(
                b.len() == w.len(),
                "weight rows and offsets differ in count",
            )?;
            rectangular(w, "weight rows differ in length")
        }
        FittedOp::RandProj { w, .. } => rectangular(w, "weight rows differ in length"),
        FittedOp::ObliqueHinge { w, t, .. } => {
            ensure(
                t.len() == w.len(),
                "weight rows and thresholds differ in count",
            )?;
            rectangular(w, "weight rows differ in length")
        }
        FittedOp::LandmarkDist { landmarks, .. } | FittedOp::Nystrom { landmarks, .. } => {
            rectangular(landmarks, "landmarks differ in length")
        }
        FittedOp::SvdProject {
            means, components, ..
        } => ensure(
            components.iter().all(|c| c.len() == means.len()),
            "a component and the means differ in length",
        ),
        FittedOp::TargetBinEncode { edges, enc, .. } => {
            ensure(
                enc.len() == edges.len(),
                "edges and encodings differ in length",
            )?;
            ensure(
                enc.iter().all(|column| !column.is_empty()),
                "a column has an empty encoding",
            )
        }
        FittedOp::Impute {
            fill,
            indicator_cols,
            ..
        } => ensure(
            indicator_cols.iter().all(|&j| j < fill.len()),
            "an indicator column is not a fitted column",
        ),
        FittedOp::PairRatio { pairs, eps, .. } => {
            ensure(eps.len() == pairs.len(), "pairs and guards differ in count")
        }
        FittedOp::Spline { knots, .. } => ensure(
            knots.iter().all(|column| column.len() >= 2),
            "a column has fewer than two knots",
        ),
        FittedOp::Rank { .. }
        | FittedOp::Blom { .. }
        | FittedOp::QuantileBins { .. }
        | FittedOp::Cheb { .. }
        | FittedOp::PolyInteract { .. }
        | FittedOp::RowStats { .. }
        | FittedOp::PairProd { .. }
        | FittedOp::Sigmoid { .. }
        | FittedOp::Step { .. }
        | FittedOp::NormRow { .. }
        | FittedOp::Polar { .. }
        | FittedOp::Lag { .. }
        | FittedOp::Roll { .. }
        | FittedOp::FftMag { .. }
        | FittedOp::RowSpectral { .. }
        | FittedOp::RowDiffStats { .. }
        | FittedOp::MapExpr { .. }
        | FittedOp::DeriveExpr { .. }
        | FittedOp::Symlog { .. }
        | FittedOp::SelectCorr { .. } => Ok(()),
    }
}

/// The width of every row that `apply_op(op, x)` returns when the rows of `x`
/// are `input` columns wide: the generated columns, after the input's own when
/// the stage appends, capped at [`crate::MAX_WIDTH`], and 1 when that leaves no
/// column (the empty row is padded with a single 0.0).
///
/// Errors: `Shape` when the fitted state is inconsistent or `input` is a width
/// that [`required_input_width`] does not admit; `Resource` when the output
/// width overflows or is wider than any row can be.
pub fn output_width(op: &FittedOp, input: usize) -> Result<usize> {
    if let Err(reason) = check_fitted_state(op) {
        return Err(Error::Shape(format!(
            "{} has inconsistent fitted state: {reason}",
            stage_name(op)
        )));
    }
    let required = required_input_width(op);
    if !required.admits(input) {
        return Err(Error::Shape(format!(
            "{} needs {required} input columns, got {input}",
            stage_name(op)
        )));
    }
    finished_width(op, input)
}

/// The input width that `head` accepts, as far as its fitted state reveals it.
///
/// Heads read columns defensively (`take`, `min` and bounds-checked routing),
/// so none panics on width; this is the contract, not a guard. Heads that keep
/// statistics of every training column reveal the exact width they were fitted
/// on. The tree heads (`n_feat` is capped by `max_features`) and `ridge_gcv`
/// (which may screen columns) reveal only a lower bound, and a constant
/// fallback that keeps no column reveals nothing. A composite head requires
/// what its members require, and its members, fitted on the same rows, agree.
pub fn required_head_width(head: &FittedHead) -> WidthReq {
    match head {
        FittedHead::Ridge(m) => WidthReq::Exactly(m.means.len()),
        FittedHead::RidgeGcv(m) => past_max(m.cols.iter().copied()),
        FittedHead::ElasticNet(m) => WidthReq::Exactly(m.means.len()),
        FittedHead::Gbm(m) => at_least(m.n_feat),
        FittedHead::Ensemble { ridge, gbm, .. } => {
            meet(WidthReq::Exactly(ridge.means.len()), at_least(gbm.n_feat))
        }
        FittedHead::HistGbm2(m) => at_least(m.n_feat),
        FittedHead::Krr(m) => WidthReq::Exactly(m.means.len()),
        FittedHead::Gpr(m) => WidthReq::Exactly(m.means.len()),
        FittedHead::GprKernel(m) => WidthReq::Exactly(m.means.len()),
        FittedHead::Ksmooth(m) => WidthReq::Exactly(m.means.len()),
        FittedHead::Hurdle2(m) => m
            .components
            .first()
            .map_or(WidthReq::Any, |c| WidthReq::Exactly(c.gate.means.len())),
        FittedHead::Stack(m) => stack_width(m),
        FittedHead::Portfolio(m) => portfolio_width(m),
    }
}

/// Checks that `model` accepts input rows `width` columns wide: walks the
/// fitted stages in order with the width each receives, checking its state,
/// its required input width and its output width, then checks the head.
/// `Model::Symbolic` accepts any width: where its expression reads a column
/// beyond the row, it evaluates to 0.0. A `Model::Portfolio` pads or truncates
/// rows to its fitted width, which it requires as a head does; its members are
/// not checked.
pub fn check_input_width(model: &Model, width: usize) -> std::result::Result<(), WidthMismatch> {
    match model {
        Model::Pipeline { ops, head, .. } => {
            let mut width = width;
            for (stage, op) in ops.iter().enumerate() {
                let expected = required_input_width(op);
                let mismatch = move |kind| WidthMismatch {
                    stage,
                    expected,
                    got: width,
                    kind,
                };
                if let Err(reason) = check_fitted_state(op) {
                    return Err(mismatch(MismatchKind::InconsistentState { reason }));
                }
                if !expected.admits(width) {
                    return Err(mismatch(MismatchKind::StageInput));
                }
                width = finished_width(op, width)
                    .map_err(|_| mismatch(MismatchKind::OutputOverflow))?;
            }
            head_admits(required_head_width(head), ops.len(), width)
        }
        Model::Symbolic { .. } => Ok(()),
        Model::Portfolio(m) => head_admits(portfolio_width(m), 0, width),
    }
}

impl Model {
    /// [`Model::predict`] behind the fitted-width contract.
    ///
    /// Returns a [`WidthMismatch`], before any stage runs, when the rows of `x`
    /// differ in width or [`check_input_width`] refuses their width. That covers
    /// every width at which `predict` would panic, and every width at which a
    /// stage or head holding one fitted entry per input column would silently
    /// misread the rows. Otherwise it returns `predict(x)` itself, so its bits
    /// are `predict`'s. With no rows there is no width to check, and it returns
    /// `predict(x)`.
    ///
    /// `Exactly` requirements make this stricter than `predict`, which accepts
    /// some narrower inputs; `predict` itself is unchanged.
    pub fn try_predict(&self, x: &Rows) -> std::result::Result<Vec<f64>, WidthMismatch> {
        if let Some(first) = x.first() {
            let width = first.len();
            if let Some((row, ragged)) = x.iter().enumerate().find(|(_, r)| r.len() != width) {
                return Err(WidthMismatch {
                    stage: 0,
                    expected: WidthReq::Exactly(width),
                    got: ragged.len(),
                    kind: MismatchKind::RaggedInput { row },
                });
            }
            check_input_width(self, width)?;
        }
        Ok(self.predict(x))
    }
}

/// The stage's name in the pipeline language, for messages.
pub(crate) fn stage_name(op: &FittedOp) -> &'static str {
    match op {
        FittedOp::Standardize { .. } => "standardize",
        FittedOp::RobustScale { .. } => "robust_scale",
        FittedOp::Rank { .. } => "rank",
        FittedOp::Blom { .. } => "blom",
        FittedOp::QuantileBins { .. } => "quantile_bins",
        FittedOp::BinOnehot { .. } => "bin_onehot",
        FittedOp::Hinge { .. } => "hinge",
        FittedOp::WinsorClip { .. } => "winsor_clip",
        FittedOp::Cheb { .. } => "cheb",
        FittedOp::PolyInteract { .. } => "poly_interact",
        FittedOp::RffCos { .. } => "rff_cos",
        FittedOp::RffSin { .. } => "rff_sin",
        FittedOp::RandProj { .. } => "rand_proj",
        FittedOp::LandmarkDist { .. } => "landmark_dist",
        FittedOp::ObliqueHinge { .. } => "oblique_hinge",
        FittedOp::RowStats { .. } => "row_stats",
        FittedOp::PairProd { .. } => "pair_prod",
        FittedOp::TargetBinEncode { .. } => "target_bin_encode",
        FittedOp::Sigmoid { .. } => "sigmoid",
        FittedOp::Step { .. } => "step",
        FittedOp::NormRow { .. } => "norm_row",
        FittedOp::Polar { .. } => "polar",
        FittedOp::Lag { .. } => "lag",
        FittedOp::Roll { .. } => "roll",
        FittedOp::FftMag { .. } => "fft_mag",
        FittedOp::RowSpectral { .. } => "row_spectral",
        FittedOp::RowDiffStats { .. } => "row_diff_stats",
        FittedOp::SvdProject { .. } => "svd_project",
        FittedOp::Nystrom { .. } => "nystrom",
        FittedOp::MapExpr { .. } => "map",
        FittedOp::DeriveExpr { .. } => "derive",
        FittedOp::Symlog { .. } => "symlog",
        FittedOp::SelectCorr { .. } => "select_corr",
        FittedOp::Impute { .. } => "impute",
        FittedOp::PairRatio { .. } => "pair_ratio",
        FittedOp::Spline { .. } => "spline",
    }
}

/// Whether the stage keeps its input columns ahead of the generated ones.
fn appends(op: &FittedOp) -> bool {
    match op {
        FittedOp::Standardize { append, .. }
        | FittedOp::RobustScale { append, .. }
        | FittedOp::Rank { append, .. }
        | FittedOp::Blom { append, .. }
        | FittedOp::QuantileBins { append, .. }
        | FittedOp::BinOnehot { append, .. }
        | FittedOp::Hinge { append, .. }
        | FittedOp::WinsorClip { append, .. }
        | FittedOp::Cheb { append, .. }
        | FittedOp::PolyInteract { append, .. }
        | FittedOp::RffCos { append, .. }
        | FittedOp::RffSin { append, .. }
        | FittedOp::RandProj { append, .. }
        | FittedOp::LandmarkDist { append, .. }
        | FittedOp::ObliqueHinge { append, .. }
        | FittedOp::RowStats { append }
        | FittedOp::PairProd { append, .. }
        | FittedOp::TargetBinEncode { append, .. }
        | FittedOp::Sigmoid { append }
        | FittedOp::Step { append, .. }
        | FittedOp::NormRow { append }
        | FittedOp::Polar { append }
        | FittedOp::Lag { append, .. }
        | FittedOp::Roll { append, .. }
        | FittedOp::FftMag { append, .. }
        | FittedOp::RowSpectral { append }
        | FittedOp::RowDiffStats { append }
        | FittedOp::SvdProject { append, .. }
        | FittedOp::Nystrom { append, .. }
        | FittedOp::MapExpr { append, .. }
        | FittedOp::DeriveExpr { append, .. }
        | FittedOp::Symlog { append, .. }
        | FittedOp::SelectCorr { append, .. }
        | FittedOp::Impute { append, .. }
        | FittedOp::PairRatio { append, .. }
        | FittedOp::Spline { append, .. } => *append,
    }
}

/// The columns the stage generates per row for rows `d` wide, before the input
/// is appended and the row is capped and padded; `None` on overflow. `d` is an
/// admitted width and the state is consistent.
fn generated_width(op: &FittedOp, d: usize) -> Option<usize> {
    Some(match op {
        FittedOp::Standardize { .. }
        | FittedOp::RobustScale { .. }
        | FittedOp::Rank { .. }
        | FittedOp::Blom { .. }
        | FittedOp::QuantileBins { .. }
        | FittedOp::WinsorClip { .. }
        | FittedOp::TargetBinEncode { .. }
        | FittedOp::Sigmoid { .. }
        | FittedOp::Step { .. }
        | FittedOp::NormRow { .. }
        | FittedOp::Polar { .. }
        | FittedOp::Lag { .. }
        | FittedOp::Roll { .. }
        | FittedOp::MapExpr { .. }
        | FittedOp::Symlog { .. } => d,
        FittedOp::BinOnehot { bins, .. } => d.checked_mul(*bins)?,
        FittedOp::Cheb { degree, .. } => d.checked_mul(*degree)?,
        FittedOp::Hinge { knots, .. } => knots
            .iter()
            .take(d)
            .try_fold(0usize, |sum, column| sum.checked_add(column.len()))?,
        FittedOp::Spline { knots, .. } => {
            knots.iter().take(d).try_fold(0usize, |sum, column| {
                sum.checked_add(spline_columns(column))
            })?
        }
        FittedOp::PolyInteract { pairs, .. } | FittedOp::PairProd { pairs, .. } => pairs.len(),
        // `apply_op` zips the pairs with their guards.
        FittedOp::PairRatio { pairs, eps, .. } => pairs.len().min(eps.len()),
        FittedOp::RffCos { w, .. }
        | FittedOp::RffSin { w, .. }
        | FittedOp::RandProj { w, .. }
        | FittedOp::ObliqueHinge { w, .. } => w.len(),
        FittedOp::LandmarkDist { landmarks, .. } | FittedOp::Nystrom { landmarks, .. } => {
            landmarks.len()
        }
        FittedOp::SvdProject { components, .. } => components.len(),
        FittedOp::SelectCorr { indices, .. } => indices.len(),
        FittedOp::Impute { indicator_cols, .. } => d.checked_add(indicator_cols.len())?,
        FittedOp::FftMag { k, .. } => *k,
        FittedOp::RowStats { .. } => 10,
        FittedOp::RowSpectral { .. } => 5,
        FittedOp::RowDiffStats { .. } => 8,
        FittedOp::DeriveExpr { .. } => 1,
    })
}

/// The spline columns of one input column: its raw value alone when its knots
/// span less than 1e-9 (the same comparison `apply_op` makes, so a NaN span
/// counts as wide), else the raw value and `k - 2` basis terms.
fn spline_columns(knots: &[f64]) -> usize {
    match knots {
        [first, .., last] if last - first < 1e-9 => 1,
        [_, _, ..] => knots.len() - 1,
        _ => 1,
    }
}

/// [`output_width`] once the state and `input` are known to be admitted.
fn finished_width(op: &FittedOp, input: usize) -> Result<usize> {
    let overflow = || {
        Error::Resource(format!(
            "{} output width overflows at input width {input}",
            stage_name(op)
        ))
    };
    let generated = generated_width(op, input).ok_or_else(overflow)?;
    let total = if appends(op) {
        input.checked_add(generated).ok_or_else(overflow)?
    } else {
        generated
    };
    if total > MAX_ROW {
        return Err(overflow());
    }
    Ok(if total == 0 {
        1
    } else {
        total.min(crate::MAX_WIDTH)
    })
}

fn head_admits(
    expected: WidthReq,
    stage: usize,
    width: usize,
) -> std::result::Result<(), WidthMismatch> {
    if expected.admits(width) {
        Ok(())
    } else {
        Err(WidthMismatch {
            stage,
            expected,
            got: width,
            kind: MismatchKind::HeadInput,
        })
    }
}

fn stack_width(m: &StackModel) -> WidthReq {
    [
        m.ridge_a.as_ref().map(|r| WidthReq::Exactly(r.means.len())),
        m.ridge_b.as_ref().map(|r| WidthReq::Exactly(r.means.len())),
        m.net.as_ref().map(|n| WidthReq::Exactly(n.means.len())),
        m.trees.as_ref().map(|t| at_least(t.n_feat)),
        m.kernel.as_ref().map(|k| WidthReq::Exactly(k.means.len())),
    ]
    .into_iter()
    .flatten()
    .fold(WidthReq::Any, meet)
}

/// A mean-only portfolio ignores its input; any other pads or truncates rows to
/// the width it was fitted on.
fn portfolio_width(m: &PortfolioModel) -> WidthReq {
    if m.mean_only {
        WidthReq::Any
    } else {
        WidthReq::Exactly(m.width)
    }
}

/// What two parts fitted on the same rows require together. They agree, so an
/// exact width stands for both.
fn meet(a: WidthReq, b: WidthReq) -> WidthReq {
    match (a, b) {
        (WidthReq::Exactly(n), _) | (_, WidthReq::Exactly(n)) => WidthReq::Exactly(n),
        (WidthReq::AtLeast(x), WidthReq::AtLeast(y)) => WidthReq::AtLeast(x.max(y)),
        (WidthReq::AtLeast(n), WidthReq::Any) | (WidthReq::Any, WidthReq::AtLeast(n)) => {
            WidthReq::AtLeast(n)
        }
        (WidthReq::Any, WidthReq::Any) => WidthReq::Any,
    }
}

fn at_least(n: usize) -> WidthReq {
    if n == 0 {
        WidthReq::Any
    } else {
        WidthReq::AtLeast(n)
    }
}

/// `AtLeast(max + 1)` over fitted column indices, `Any` when there is none.
fn past_max(indices: impl Iterator<Item = usize>) -> WidthReq {
    indices.max().map_or(WidthReq::Any, |max| {
        WidthReq::AtLeast(max.saturating_add(1))
    })
}

/// `Exactly` the width of the first fitted row over the input columns, `Any`
/// when there is none (the stage then generates no column).
fn first_row_width(rows: &[Vec<f64>]) -> WidthReq {
    rows.first()
        .map_or(WidthReq::Any, |row| WidthReq::Exactly(row.len()))
}

fn rectangular(rows: &[Vec<f64>], reason: &'static str) -> std::result::Result<(), &'static str> {
    let width = rows.first().map_or(0, Vec::len);
    ensure(rows.iter().all(|row| row.len() == width), reason)
}

fn ensure(ok: bool, reason: &'static str) -> std::result::Result<(), &'static str> {
    if ok {
        Ok(())
    } else {
        Err(reason)
    }
}
