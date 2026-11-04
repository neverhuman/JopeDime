use super::*;

/// `finish` in `ops.rs`, line for line, counting the non-finite values it replaces: the append
/// merge, then per row the width cap, the replacement of non-finite values and the pad of an
/// empty row.
pub(crate) fn finish_counted(
    op: &FittedOp,
    input: &Rows,
    mut out: Rows,
    sanitized: &mut u64,
) -> Rows {
    if appends(op) {
        for (i, row) in out.iter_mut().enumerate() {
            let mut merged = input[i].clone();
            merged.append(row);
            *row = merged;
        }
    }
    for row in out.iter_mut() {
        if row.len() > MAX_WIDTH {
            row.truncate(MAX_WIDTH);
        }
        for v in row.iter_mut() {
            if !v.is_finite() {
                *v = 0.0;
                *sanitized += 1;
            }
        }
        if row.is_empty() {
            row.push(0.0);
        }
    }
    out
}

/// `op_append` in `ops.rs`, verbatim: every variant is listed and there is no wildcard, so a new
/// op does not compile until its append flag is read here too.
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

/// The non-finite cells of `x`: what `sanitize` replaces.
pub(crate) fn count_nonfinite(x: &Rows) -> u64 {
    x.iter().flatten().filter(|v| !v.is_finite()).count() as u64
}
