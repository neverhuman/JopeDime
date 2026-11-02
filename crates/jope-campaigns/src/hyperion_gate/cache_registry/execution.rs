use super::*;

#[derive(Debug)]
pub(super) struct StageExecution {
    pub(super) evidence: Vec<PairedFoldEvidenceV1>,
    pub(super) receipts: Vec<JopeHyperionFoldReceiptV1>,
    pub(super) encoding_milliseconds: u64,
    pub(super) scoring_milliseconds: u64,
}

pub(super) fn precomputed_fold_encoding(
    blocks: &PrecomputedTrainFeatureBlocks,
    fold: &GroupedFoldV1,
    plan_sha256: &str,
) -> Result<FoldEncoding, String> {
    let cell_fit = select_rows(&blocks.cell, &fold.fit_rows);
    let cell_hold = select_rows(&blocks.cell, &fold.hold_rows);
    let mut sidecars = vec![
        MatrixArtifactV1 {
            role: "cell_fit".into(),
            rows: cell_fit.len() as u32,
            columns: blocks.receipt.raw_feature_count,
            sha256: hash_matrix(&cell_fit),
        },
        MatrixArtifactV1 {
            role: "cell_hold".into(),
            rows: cell_hold.len() as u32,
            columns: blocks.receipt.raw_feature_count,
            sha256: hash_matrix(&cell_hold),
        },
    ];
    let (row_fit, row_hold) = if let Some(row128) = &blocks.row128 {
        let fit = select_rows(row128, &fold.fit_rows);
        let hold = select_rows(row128, &fold.hold_rows);
        sidecars.extend([
            MatrixArtifactV1 {
                role: "row_fit".into(),
                rows: fit.len() as u32,
                columns: ROW_EMBEDDING_WIDTH as u32,
                sha256: hash_matrix(&fit),
            },
            MatrixArtifactV1 {
                role: "row_hold".into(),
                rows: hold.len() as u32,
                columns: ROW_EMBEDDING_WIDTH as u32,
                sha256: hash_matrix(&hold),
            },
        ]);
        (fit, hold)
    } else {
        (Vec::new(), Vec::new())
    };
    let receipt_sha256 = blocks.receipt.identity()?;
    let cache_key_sha256 = canonical_sha256(&(
        "precomputed-train-feature-fold.v1",
        &receipt_sha256,
        plan_sha256,
        fold.fold_index,
        hash_matrix(&cell_fit),
        hash_matrix(&cell_hold),
    ))?;
    Ok(FoldEncoding {
        cell_fit,
        cell_hold,
        row_fit,
        row_hold,
        sidecars,
        cache_key_sha256,
        cache_receipt_sha256: receipt_sha256,
        milliseconds: 0,
        cache_replay: true,
    })
}

pub(super) fn sidecars_for_view(
    sidecars: &[MatrixArtifactV1],
    view: FeatureViewKindV1,
) -> Vec<MatrixArtifactV1> {
    sidecars
        .iter()
        .filter(|sidecar| match view {
            FeatureViewKindV1::Raw => false,
            FeatureViewKindV1::HyperionCell => sidecar.role.starts_with("cell_"),
            FeatureViewKindV1::HyperionRow128 => sidecar.role.starts_with("row_"),
            FeatureViewKindV1::HyperionBoth => true,
        })
        .cloned()
        .collect()
}
