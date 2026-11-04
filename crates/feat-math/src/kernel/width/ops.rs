use super::*;

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
/// that, at any width [`required_input_width`] admits, `apply_op` reads nothing
/// out of bounds and does not panic, unless a fitted size makes an output row
/// wider than any `Vec` can hold (a tampered `degree`, `bins` or `k`):
/// [`output_width`] refuses that too, and applies this check and the width
/// requirement as well. State that `fit_op` produced always passes; the check
/// is for state that arrived another way, such as a stored or edited model.
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
pub(super) fn finished_width(op: &FittedOp, input: usize) -> Result<usize> {
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

/// `AtLeast(max + 1)` over fitted column indices, `Any` when there is none.
pub(crate) fn past_max(indices: impl Iterator<Item = usize>) -> WidthReq {
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
