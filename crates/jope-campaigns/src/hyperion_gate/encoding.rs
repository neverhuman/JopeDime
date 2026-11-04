use super::*;

/// Immutable output and dataset identities for a fold encoding.
#[derive(Clone, Copy)]
pub(crate) struct FoldEncodingScope<'a> {
    pub plan_sha256: &'a str,
    pub encoding_root: &'a Path,
    pub registry_sha256: &'a str,
    pub dataset: &'a TrainOnlyDatasetV1,
}

pub(crate) fn encode_fold(
    encoder: &HyperionEncoderIdentityV1,
    source: &RegressionTable,
    fold: &GroupedFoldV1,
    plan: &GroupedFoldPlanV1,
    scope: FoldEncodingScope<'_>,
    gate_started: Instant,
) -> Result<FoldEncoding, String> {
    let FoldEncodingScope {
        plan_sha256,
        encoding_root,
        registry_sha256,
        dataset,
    } = scope;

    let stage = match plan.stage {
        GateStageV1::Cv5Screen => "cv5",
        GateStageV1::Cv7Confirmation => "cv7",
    };
    let fit_input = csv_for_rows(source, &fold.fit_rows, false);
    let hold_input = csv_for_rows(source, &fold.hold_rows, true);
    let fit_input_sha256 = raw_sha256(&fit_input);
    let hold_input_sha256 = raw_sha256(&hold_input);
    let fit_row_ids_sha256 = hash_selected_strings(&dataset.stable_row_ids, &fold.fit_rows);
    let hold_row_ids_sha256 = hash_selected_strings(&dataset.stable_row_ids, &fold.hold_rows);
    let key = cache_key(
        registry_sha256,
        dataset,
        encoder,
        stage,
        plan_sha256,
        Some(fold.fold_index),
        &fit_row_ids_sha256,
        &hold_row_ids_sha256,
        &fit_input_sha256,
        &hold_input_sha256,
    )?;
    let key_sha256 = key.identity()?;
    let directory = encoding_root
        .join("datasets")
        .join(&dataset.dataset_id)
        .join(stage)
        .join(&key_sha256[..2])
        .join(&key_sha256);
    fs::create_dir_all(&directory)
        .map_err(|error| format!("create fold-local encoding directory: {error}"))?;
    let fit_path = directory.join("fit.csv");
    let hold_path = directory.join("hold.csv");
    write_immutable(&fit_path, &fit_input)?;
    write_immutable(&hold_path, &hold_input)?;
    let outputs = [
        (
            "cell_fit",
            directory.join("cell-fit.vs"),
            fold.fit_rows.len(),
            source.features[0].len(),
        ),
        (
            "cell_hold",
            directory.join("cell-hold.vs"),
            fold.hold_rows.len(),
            source.features[0].len(),
        ),
        (
            "row_fit",
            directory.join("row-fit.vs"),
            fold.fit_rows.len(),
            ROW_EMBEDDING_WIDTH,
        ),
        (
            "row_hold",
            directory.join("row-hold.vs"),
            fold.hold_rows.len(),
            ROW_EMBEDDING_WIDTH,
        ),
    ];
    let cache_path = directory.join("fold-encoding-cache.v2.json");
    if cache_path.is_file() {
        let cache: FoldEncodingCacheV2 = serde_json::from_slice(
            &fs::read(&cache_path).map_err(|error| format!("read fold encoding cache: {error}"))?,
        )
        .map_err(|error| format!("decode fold encoding cache: {error}"))?;
        cache.validate(encoder.device_ordinal)?;
        if cache.key != key || cache.key_sha256 != key_sha256 {
            return Err("fold-local encoding cache identity drifted".into());
        }
        let matrices = read_output_matrices(&outputs)?;
        let actual = output_artifacts(&outputs)?;
        if actual != cache.sidecars {
            return Err("fold-local sidecar changed after cache publication".into());
        }
        return Ok(FoldEncoding {
            cell_fit: matrices[0].clone(),
            cell_hold: matrices[1].clone(),
            row_fit: matrices[2].clone(),
            row_hold: matrices[3].clone(),
            sidecars: actual,
            cache_key_sha256: key_sha256,
            cache_receipt_sha256: canonical_sha256(&cache)?,
            milliseconds: 0,
            cache_replay: true,
        });
    }
    if outputs.iter().any(|(_, path, _, _)| path.exists()) {
        return Err("partial or unreceipted fold-local sidecar output exists".into());
    }
    let remaining = ENCODING_BUDGET
        .checked_sub(gate_started.elapsed())
        .ok_or_else(|| "eight-minute encoding budget expired".to_string())?;
    let execution = run_encoder_with_timeout(encoder, &fit_path, &hold_path, &outputs, remaining)?;
    if !execution.status.success() {
        return Err(format!(
            "fold-local Hyperion encoder exited with {}",
            execution.status
        ));
    }
    execution.proof.validate_namespace()?;
    let matrices = read_output_matrices(&outputs)?;
    let sidecars = output_artifacts(&outputs)?;
    let cache = FoldEncodingCacheV2 {
        schema: FoldEncodingCacheV2::SCHEMA.into(),
        key,
        key_sha256,
        fit_row_order_sha256: fit_row_ids_sha256,
        hold_row_order_sha256: hold_row_ids_sha256,
        hold_target_policy: FoldEncodingCacheV2::DUMMY_HOLD_TARGET_POLICY.into(),
        sidecars: sidecars.clone(),
        proof: execution.proof,
    };
    cache.validate(encoder.device_ordinal)?;
    write_immutable(&cache_path, &canonical_json_bytes(&cache)?)?;
    Ok(FoldEncoding {
        cell_fit: matrices[0].clone(),
        cell_hold: matrices[1].clone(),
        row_fit: matrices[2].clone(),
        row_hold: matrices[3].clone(),
        sidecars,
        cache_key_sha256: cache.key_sha256.clone(),
        cache_receipt_sha256: canonical_sha256(&cache)?,
        milliseconds: cache.proof.elapsed_milliseconds,
        cache_replay: false,
    })
}
