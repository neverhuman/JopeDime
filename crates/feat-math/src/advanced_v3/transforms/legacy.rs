use super::*;

pub(super) fn legacy_append(op: &FittedOp) -> bool {
    match op {
        FittedOp::Standardize { append, .. }
        | FittedOp::RobustScale { append, .. }
        | FittedOp::Rank { append, .. }
        | FittedOp::Blom { append, .. }
        | FittedOp::QuantileBins { append, .. }
        | FittedOp::BinOnehot { append, .. }
        | FittedOp::Hinge { append, .. }
        | FittedOp::TargetBinEncode { append, .. }
        | FittedOp::Cheb { append, .. }
        | FittedOp::PolyInteract { append, .. }
        | FittedOp::RffCos { append, .. }
        | FittedOp::RffSin { append, .. }
        | FittedOp::RandProj { append, .. }
        | FittedOp::LandmarkDist { append, .. }
        | FittedOp::ObliqueHinge { append, .. }
        | FittedOp::RowStats { append }
        | FittedOp::PairProd { append, .. }
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
        | FittedOp::WinsorClip { append, .. }
        | FittedOp::Impute { append, .. }
        | FittedOp::PairRatio { append, .. }
        | FittedOp::Spline { append, .. } => *append,
    }
}

pub(super) fn legacy_generated_to_block(
    op: &FittedOp,
    block: &FeatureBlock,
    generated: Rows,
) -> Result<FeatureBlock> {
    match op {
        FittedOp::Standardize { append, .. }
        | FittedOp::RobustScale { append, .. }
        | FittedOp::TargetBinEncode { append, .. }
        | FittedOp::Sigmoid { append }
        | FittedOp::Step { append, .. }
        | FittedOp::NormRow { append }
        | FittedOp::WinsorClip { append, .. } => {
            finish_one_to_one(block, generated, *append, "legacy_v3_transform")
        }
        FittedOp::SelectCorr {
            indices, append, ..
        } => finish_selected(block, generated, indices, *append, "select_corr"),
        FittedOp::Impute {
            indicator_cols,
            append,
            ..
        } => finish_impute(block, generated, indicator_cols, *append),
        FittedOp::Lag { .. } | FittedOp::Roll { .. } => {
            bail!("legacy row operations are not valid inside advanced-v3")
        }
        _ => {
            let generated_width = generated.first().map_or(0, Vec::len);
            let sources = (0..block.roles.len()).collect::<Vec<_>>();
            finish_generated_from_sets(
                block,
                generated,
                vec![sources; generated_width],
                vec![FeatureRole::Generated; generated_width],
                vec![vec![false; generated_width]; block.values.len()],
                legacy_append(op),
                "legacy_v3_transform",
            )
        }
    }
}

pub(super) fn legacy_to_block(op: &FittedOp, block: &FeatureBlock) -> Result<FeatureBlock> {
    check_legacy_width(op, block)?;
    let generated = crate::ops::apply_op_generated(op, &block.values);
    legacy_generated_to_block(op, block, generated)
}

/// The strict width boundary of a legacy stage in a schema-bound pipeline
/// (kernel2 WP-1.7): its fitted state must agree with itself, accept the
/// block's width and give output rows that a `Vec` can hold, or the stage is
/// refused with an error where `apply_op_generated` would panic or read state
/// fitted for another column. State fitted on the block always passes. Every
/// row of a block is `roles.len()` wide (`FeatureBlock::validate`).
///
/// A block with no rows has no width to compare. Nothing is read, and the
/// generated-width arm of `legacy_generated_to_block` takes its width from the
/// first row, so after such a stage the block is narrower than at fit time (no
/// column, or only the appended input). Without rows, the fitted state is
/// checked, and beyond it only the fitted column indices at which finishing the
/// stage looks the block up: `select_corr`'s indices, and `impute`'s indicators
/// when it does not append. Finishing panicked there. No output row is
/// allocated either, so there is no overflow to refuse. For state that
/// `fit_op` produced, every other outcome without rows is what it was before
/// these checks.
pub(super) fn check_legacy_width(op: &FittedOp, block: &FeatureBlock) -> Result<()> {
    use crate::kernel::width;
    let name = width::stage_name(op);
    if let Err(reason) = width::check_fitted_state(op) {
        bail!("legacy stage {name} has inconsistent fitted state: {reason}");
    }
    let got = block.roles.len();
    if block.values.is_empty() {
        let looked_up: &[usize] = match op {
            FittedOp::SelectCorr { indices, .. } => indices,
            FittedOp::Impute {
                indicator_cols,
                append: false,
                ..
            } => indicator_cols,
            _ => &[],
        };
        if let Some(index) = looked_up.iter().find(|&&index| index >= got) {
            bail!(
                "legacy stage {name} reads column {index}, but the feature block has {got} columns"
            );
        }
        return Ok(());
    }
    let required = width::required_input_width(op);
    if !required.admits(got) {
        bail!(
            "legacy stage {name} needs {required} input columns, but the feature block has {got}"
        );
    }
    // The state and the width are admitted above, so only an output row too
    // wide for any `Vec` is left for `output_width` to refuse.
    if width::output_width(op, got).is_err() {
        bail!("legacy stage {name} output width overflows at {got} input columns");
    }
    Ok(())
}

/// `corr_screen` reads every row at its fitted column indices, and finishing
/// the selection looks the block up at them, with rows or without.
pub(super) fn check_corr_screen_width(indices: &[usize], block: &FeatureBlock) -> Result<()> {
    let width = block.roles.len();
    if let Some(index) = indices.iter().find(|&&index| index >= width) {
        bail!("corr_screen reads column {index}, but the feature block has {width} columns");
    }
    Ok(())
}

/// `pls_project` reads the row and the fitted means at every index of a weight.
/// Without rows it reads nothing, and the block may be narrower than at fit
/// time (see `check_legacy_width`), so there is nothing to check.
pub(super) fn check_pls_project_width(
    means: &[f64],
    weights: &[Vec<f64>],
    block: &FeatureBlock,
) -> Result<()> {
    if block.values.is_empty() {
        return Ok(());
    }
    let width = block.roles.len();
    if let Some(weight) = weights
        .iter()
        .find(|weight| weight.len() > width.min(means.len()))
    {
        bail!(
            "pls_project weight spans {} columns, but the feature block has {width} and the fitted means {}",
            weight.len(),
            means.len()
        );
    }
    Ok(())
}

/// `piecewise_linear` finishes by gathering the block's missing mask at the column of every fitted
/// edge list that has an interval (two edges or more), so each such column must be in the block;
/// `piecewise_row` itself zips the row with the edge lists and reads nothing past either (kernel2
/// X3, closing the gap WP-1.7 recorded). Without rows nothing is read or gathered, and the block
/// may be narrower than at fit time (see `check_legacy_width`), so there is nothing to check.
pub(super) fn check_piecewise_linear_width(edges: &[Vec<f64>], block: &FeatureBlock) -> Result<()> {
    if block.values.is_empty() {
        return Ok(());
    }
    let width = block.roles.len();
    if let Some((column, _)) = edges
        .iter()
        .enumerate()
        .skip(width)
        .find(|(_, column_edges)| column_edges.len() > 1)
    {
        bail!("piecewise_linear reads column {column}, but the feature block has {width} columns");
    }
    Ok(())
}
