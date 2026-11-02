use super::*;

pub(super) fn encode_full_train(
    encoder: &HyperionEncoderIdentityV1,
    source: &RegressionTable,
    dataset: &TrainOnlyDatasetV1,
    registry_sha256: &str,
    cache_root: &Path,
) -> Result<CacheCoverageEntryV2, String> {
    let rows = (0..source.targets.len()).collect::<Vec<_>>();
    let train_input = csv_for_rows(source, &rows, false);
    let dummy_input = csv_for_rows(source, &rows, true);
    let row_order_sha256 = hash_selected_strings(&dataset.stable_row_ids, &rows);
    let plan_sha256 =
        raw_sha256(format!("full-train-v2|{}|{row_order_sha256}", dataset.train_sha256).as_bytes());
    let key = cache_key(
        registry_sha256,
        dataset,
        encoder,
        "full_train",
        &plan_sha256,
        None,
        &row_order_sha256,
        &row_order_sha256,
        &raw_sha256(&train_input),
        &raw_sha256(&dummy_input),
    )?;
    let key_sha256 = key.identity()?;
    let directory = cache_root
        .join("datasets")
        .join(&dataset.dataset_id)
        .join("full_train")
        .join(&key_sha256[..2])
        .join(&key_sha256);
    fs::create_dir_all(&directory)
        .map_err(|error| format!("create full-train encoding directory: {error}"))?;
    let train_path = directory.join("train.csv");
    let dummy_path = directory.join("dummy-hold.csv");
    write_immutable(&train_path, &train_input)?;
    write_immutable(&dummy_path, &dummy_input)?;
    let outputs = [
        (
            "cell_train",
            directory.join("cell-train.vs"),
            rows.len(),
            dataset.feature_count as usize,
        ),
        (
            "cell_shadow_hold",
            directory.join("cell-shadow-hold.vs"),
            rows.len(),
            dataset.feature_count as usize,
        ),
        (
            "row_train",
            directory.join("row-train.vs"),
            rows.len(),
            ROW_EMBEDDING_WIDTH,
        ),
        (
            "row_shadow_hold",
            directory.join("row-shadow-hold.vs"),
            rows.len(),
            ROW_EMBEDDING_WIDTH,
        ),
    ];
    let receipt_path = directory.join("full-train-encoding-cache.v2.json");
    if receipt_path.is_file() {
        let receipt: FullTrainEncodingCacheV2 = serde_json::from_slice(
            &fs::read(&receipt_path)
                .map_err(|error| format!("read full-train encoding cache: {error}"))?,
        )
        .map_err(|error| format!("decode full-train encoding cache: {error}"))?;
        receipt.validate(encoder.device_ordinal)?;
        if receipt.key != key || receipt.key_sha256 != key_sha256 {
            return Err("full-train encoding cache identity drifted".into());
        }
        let primary = [outputs[0].clone(), outputs[2].clone()];
        read_output_matrices(&primary)?;
        let actual = output_artifacts(&primary)?;
        if actual != receipt.sidecars {
            return Err("full-train sidecar changed after cache publication".into());
        }
        return Ok(CacheCoverageEntryV2 {
            fold_index: None,
            cache_key_sha256: key_sha256,
            cache_receipt_sha256: canonical_sha256(&receipt)?,
        });
    }
    if outputs.iter().any(|(_, path, _, _)| path.exists()) {
        return Err("partial or unreceipted full-train sidecar output exists".into());
    }
    let execution =
        run_encoder_with_timeout(encoder, &train_path, &dummy_path, &outputs, ENCODING_BUDGET)?;
    if !execution.status.success() {
        return Err(format!(
            "full-train Hyperion encoder exited with {}",
            execution.status
        ));
    }
    execution.proof.validate_namespace()?;
    read_output_matrices(&outputs)?;
    let primary = [outputs[0].clone(), outputs[2].clone()];
    let sidecars = output_artifacts(&primary)?;
    for path in [&outputs[1].1, &outputs[3].1] {
        fs::remove_file(path)
            .map_err(|error| format!("remove unused full-train shadow output: {error}"))?;
    }
    let receipt = FullTrainEncodingCacheV2 {
        schema: FullTrainEncodingCacheV2::SCHEMA.into(),
        key,
        key_sha256,
        train_row_order_sha256: row_order_sha256,
        sidecars,
        proof: execution.proof,
    };
    receipt.validate(encoder.device_ordinal)?;
    write_immutable(&receipt_path, &canonical_json_bytes(&receipt)?)?;
    Ok(CacheCoverageEntryV2 {
        fold_index: None,
        cache_key_sha256: receipt.key_sha256.clone(),
        cache_receipt_sha256: canonical_sha256(&receipt)?,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_measured_decision(
    dataset_id: &str,
    dataset: &TrainOnlyDatasetV1,
    predictor_sha256: &str,
    encoder_sha256: &str,
    cv5_plan_sha256: &str,
    cv7_plan_sha256: &str,
    thresholds: GateThresholdsV1,
    cv5_candidates: Vec<StageGateResultV1>,
    advanced: Option<FeatureViewKindV1>,
    confirmation: Option<StageGateResultV1>,
    fallback: Option<&str>,
    receipts: &[JopeHyperionFoldReceiptV1],
    encoding_milliseconds: u64,
    scoring_milliseconds: u64,
    started: Instant,
) -> Result<JopeHyperionGateDecisionV1, String> {
    let fallback = fallback.map(str::to_owned);
    let selected_view = if fallback.is_none() {
        confirmation
            .as_ref()
            .filter(|value| value.passed)
            .map(|value| value.arm)
            .unwrap_or(FeatureViewKindV1::Raw)
    } else {
        FeatureViewKindV1::Raw
    };
    let decision = JopeHyperionGateDecisionV1 {
        schema: JopeHyperionGateDecisionV1::SCHEMA.into(),
        dataset_id_sha256: raw_sha256(dataset_id.as_bytes()),
        source_training_sha256: dataset.train_sha256.clone(),
        raw_predictor_sha256: predictor_sha256.into(),
        encoder_identity_sha256: encoder_sha256.into(),
        cv5_fold_plan_sha256: cv5_plan_sha256.into(),
        cv7_fold_plan_sha256: cv7_plan_sha256.into(),
        thresholds,
        cv5_candidates,
        cv5_advanced_arm: advanced,
        cv7_confirmation: confirmation,
        selected_view,
        raw_fallback_reason: fallback,
        fold_receipt_sha256s: receipts
            .iter()
            .map(canonical_sha256)
            .collect::<Result<Vec<_>, _>>()?,
        encoding_milliseconds,
        scoring_milliseconds,
        wall_clock_milliseconds: u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
    };
    decision.validate()?;
    Ok(decision)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn execute_stage(
    encoder: &HyperionEncoderIdentityV1,
    dataset: &TrainOnlyDatasetV1,
    source: &RegressionTable,
    precomputed: &PrecomputedTrainFeatureBlocks,
    predictor: &FrozenPredictorV1,
    adapter: &CorpusEvalAdapter,
    plan: &GroupedFoldPlanV1,
    plan_sha256: &str,
    arms: &[FeatureViewKindV1],
    receipt_root: &Path,
    started: Instant,
    prior_encoding_milliseconds: u64,
    prior_scoring_milliseconds: u64,
) -> Result<StageExecution, String> {
    let mut evidence = Vec::with_capacity(plan.folds.len());
    let mut receipts = Vec::new();
    let mut encoding_milliseconds = 0u64;
    let mut scoring_milliseconds = 0u64;
    validate_plan_feature_feasibility(plan, arms, dataset.feature_count as usize)?;
    for fold in &plan.folds {
        if started.elapsed() >= WALL_CLOCK_BUDGET {
            return Err("ten-minute Hyperion gate wall-clock budget expired".into());
        }
        let encoding = precomputed_fold_encoding(precomputed, fold, plan_sha256)?;
        encoding_milliseconds += encoding.milliseconds;
        if prior_encoding_milliseconds + encoding_milliseconds > ENCODING_BUDGET.as_millis() as u64
        {
            return Err("eight-minute fold-local encoding budget expired".into());
        }
        let fitted = FittedFeatureBlocksV1::fit_available(
            &encoding.cell_fit,
            (!encoding.row_fit.is_empty()).then_some(encoding.row_fit.as_slice()),
            dataset.feature_count as usize,
        )?;
        let cell_fit = fitted.transform_cell(&encoding.cell_fit)?;
        let cell_hold = fitted.transform_cell(&encoding.cell_hold)?;
        let (row_fit, row_hold) = if encoding.row_fit.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            (
                fitted.transform_row(&encoding.row_fit)?,
                fitted.transform_row(&encoding.row_hold)?,
            )
        };
        let raw_scoring_started = Instant::now();
        let raw_prediction = score_view(
            FeatureViewKindV1::Raw,
            dataset,
            source,
            fold,
            &[],
            &[],
            &[],
            &[],
            predictor,
            adapter,
        )?;
        scoring_milliseconds += elapsed_milliseconds(raw_scoring_started);
        let mut arm_predictions = BTreeMap::new();
        for arm in arms {
            let scoring_started = Instant::now();
            let prediction = score_view(
                *arm, dataset, source, fold, &cell_fit, &cell_hold, &row_fit, &row_hold, predictor,
                adapter,
            )?;
            let scoring_elapsed = elapsed_milliseconds(scoring_started);
            scoring_milliseconds += scoring_elapsed;
            if prior_scoring_milliseconds + scoring_milliseconds > SCORING_BUDGET.as_millis() as u64
            {
                return Err("two-minute model scoring and verification budget expired".into());
            }
            let actual = fold
                .hold_rows
                .iter()
                .map(|row| source.targets[*row])
                .collect::<Vec<_>>();
            let feature_view = JopeFeatureViewV1::new(*arm, dataset.feature_count as usize)?;
            let expanded_fit = assemble_feature_view(
                *arm,
                &select_rows(&source.features, &fold.fit_rows),
                &cell_fit,
                &row_fit,
            )?;
            let expanded_hold = assemble_feature_view(
                *arm,
                &select_rows(&source.features, &fold.hold_rows),
                &cell_hold,
                &row_hold,
            )?;
            let receipt = JopeHyperionFoldReceiptV1 {
                schema: JopeHyperionFoldReceiptV1::SCHEMA.into(),
                stage: plan.stage,
                fold_index: fold.fold_index,
                fold_plan_sha256: plan_sha256.into(),
                feature_view_sha256: feature_view.identity()?,
                feature_view,
                predictor_sha256: predictor.semantic_sha256()?,
                encoder_identity_sha256: encoder.identity()?,
                cache_key_sha256: encoding.cache_key_sha256.clone(),
                cache_receipt_sha256: encoding.cache_receipt_sha256.clone(),
                source_training_sha256: dataset.train_sha256.clone(),
                fit_row_ids_sha256: hash_selected_strings(&dataset.stable_row_ids, &fold.fit_rows),
                hold_row_ids_sha256: hash_selected_strings(
                    &dataset.stable_row_ids,
                    &fold.hold_rows,
                ),
                fit_atomic_groups_sha256: hash_selected_strings(
                    &dataset.atomic_groups,
                    &fold.fit_rows,
                ),
                hold_atomic_groups_sha256: hash_selected_strings(
                    &dataset.atomic_groups,
                    &fold.hold_rows,
                ),
                sidecars: sidecars_for_view(&encoding.sidecars, *arm),
                expanded_fit_sha256: hash_matrix(&expanded_fit),
                expanded_hold_sha256: hash_matrix(&expanded_hold),
                raw_predictions_sha256: hash_f64(&raw_prediction),
                view_predictions_sha256: hash_f64(&prediction),
                raw_rmse: rmse(&actual, &raw_prediction),
                view_rmse: rmse(&actual, &prediction),
                encoding_milliseconds: encoding.milliseconds,
                scoring_milliseconds: scoring_elapsed,
                cache_replay: encoding.cache_replay,
            };
            receipt.validate()?;
            let receipt_bytes = canonical_json_bytes(&receipt)?;
            let stage_name = match plan.stage {
                GateStageV1::Cv5Screen => "cv5",
                GateStageV1::Cv7Confirmation => "cv7",
            };
            let arm_name = format!("{:?}", arm).to_ascii_lowercase();
            write_immutable(
                &receipt_root.join(format!(
                    "{stage_name}-fold-{}-{arm_name}.v1.json",
                    fold.fold_index
                )),
                &receipt_bytes,
            )?;
            arm_predictions.insert(*arm, prediction);
            receipts.push(receipt);
        }
        evidence.push(PairedFoldEvidenceV1 {
            fold_index: fold.fold_index,
            row_ids: fold
                .hold_rows
                .iter()
                .map(|row| dataset.stable_row_ids[*row].clone())
                .collect(),
            atomic_groups: fold
                .hold_rows
                .iter()
                .map(|row| dataset.atomic_groups[*row].clone())
                .collect(),
            actual: fold
                .hold_rows
                .iter()
                .map(|row| source.targets[*row])
                .collect(),
            raw_predictions: raw_prediction,
            arm_predictions,
        });
    }
    Ok(StageExecution {
        evidence,
        receipts,
        encoding_milliseconds,
        scoring_milliseconds,
    })
}

#[allow(clippy::too_many_arguments)]
fn score_view(
    view: FeatureViewKindV1,
    dataset: &TrainOnlyDatasetV1,
    source: &RegressionTable,
    fold: &GroupedFoldV1,
    cell_fit: &[Vec<f64>],
    cell_hold: &[Vec<f64>],
    row_fit: &[Vec<f64>],
    row_hold: &[Vec<f64>],
    predictor: &FrozenPredictorV1,
    adapter: &CorpusEvalAdapter,
) -> Result<Vec<f64>, String> {
    let fit_raw = select_rows(&source.features, &fold.fit_rows);
    let hold_raw = select_rows(&source.features, &fold.hold_rows);
    let fit_features = if view == FeatureViewKindV1::Raw {
        fit_raw
    } else {
        assemble_feature_view(view, &fit_raw, cell_fit, row_fit)?
    };
    let hold_features = if view == FeatureViewKindV1::Raw {
        hold_raw
    } else {
        assemble_feature_view(view, &hold_raw, cell_hold, row_hold)?
    };
    let fit_position = fold
        .fit_rows
        .iter()
        .enumerate()
        .map(|(position, row)| (*row, position))
        .collect::<BTreeMap<_, _>>();
    let hold_position = fold
        .hold_rows
        .iter()
        .enumerate()
        .map(|(position, row)| (*row, position))
        .collect::<BTreeMap<_, _>>();
    let rows = (0..source.targets.len())
        .map(|row| {
            let (features, target) = if let Some(position) = fit_position.get(&row) {
                (&fit_features[*position], source.targets[row])
            } else {
                (&hold_features[hold_position[&row]], 0.0)
            };
            let mut fields = features
                .iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>();
            fields.push(target.to_string());
            DatasetRowV1 {
                split: SplitRowV1 {
                    stable_row_id: dataset.stable_row_ids[row].clone(),
                    atomic_group: dataset.atomic_groups[row].clone(),
                    regression_target: Some(source.targets[row]),
                    binary_target: None,
                },
                csv_line: fields.join(","),
            }
        })
        .collect::<Vec<_>>();
    let csv_bytes = rows
        .iter()
        .flat_map(|row| [row.csv_line.as_bytes(), b"\n"].concat())
        .collect::<Vec<_>>();
    let table = DatasetTableV1 {
        task: GpTaskV1::Regression,
        dataset_sha256: dataset.dataset_sha256.clone(),
        training_sha256: raw_sha256(&csv_bytes),
        rows,
    };
    table.validate()?;
    let mut fit_ids = fold
        .fit_rows
        .iter()
        .map(|row| dataset.stable_row_ids[*row].clone())
        .collect::<Vec<_>>();
    let mut hold_ids = fold
        .hold_rows
        .iter()
        .map(|row| dataset.stable_row_ids[*row].clone())
        .collect::<Vec<_>>();
    fit_ids.sort();
    hold_ids.sort();
    let split = RowSplitV1 {
        split_index: fold.fold_index,
        seed: 1,
        train_row_ids: fit_ids,
        validation_row_ids: hold_ids.clone(),
        validation_groups: fold.hold_groups.clone(),
        balance_counts: BTreeMap::new(),
    };
    let candidates = predictor
        .pipelines()
        .into_iter()
        .cloned()
        .collect::<Vec<_>>();
    let outcomes = adapter.evaluate_search_split(&table, &split, &candidates)?;
    if outcomes.len() != candidates.len()
        || outcomes.iter().any(|outcome| !outcome.prediction.valid)
    {
        return Err("AFPM evaluator returned incomplete or invalid fold predictions".into());
    }
    let components = outcomes
        .iter()
        .map(|outcome| {
            if outcome.prediction.validation_row_ids != hold_ids {
                return Err("AFPM predictions changed hold-row ordering".into());
            }
            Ok(outcome
                .prediction
                .prediction_f64_bits
                .iter()
                .map(|bits| f64::from_bits(*bits))
                .collect::<Vec<_>>())
        })
        .collect::<Result<Vec<_>, String>>()?;
    let sorted_prediction = predictor.reconstruct(&components)?;
    let by_id = hold_ids
        .into_iter()
        .zip(sorted_prediction)
        .collect::<BTreeMap<_, _>>();
    fold.hold_rows
        .iter()
        .map(|row| {
            by_id
                .get(&dataset.stable_row_ids[*row])
                .copied()
                .ok_or_else(|| "AFPM prediction omitted a hold row".to_string())
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn cache_key(
    registry_sha256: &str,
    dataset: &TrainOnlyDatasetV1,
    encoder: &HyperionEncoderIdentityV1,
    stage: &str,
    fold_plan_sha256: &str,
    fold_index: Option<u32>,
    fit_row_ids_sha256: &str,
    hold_row_ids_sha256: &str,
    fit_input_sha256: &str,
    hold_input_sha256: &str,
) -> Result<HyperionCacheKeyV2, String> {
    let key = HyperionCacheKeyV2 {
        schema: HyperionCacheKeyV2::SCHEMA.into(),
        registry_sha256: registry_sha256.into(),
        dataset_id_sha256: raw_sha256(dataset.dataset_id.as_bytes()),
        source_training_sha256: dataset.train_sha256.clone(),
        stable_row_ids_sha256: hash_strings(&dataset.stable_row_ids),
        atomic_groups_sha256: hash_strings(&dataset.atomic_groups),
        stage: stage.into(),
        fold_plan_sha256: fold_plan_sha256.into(),
        fold_index,
        fit_row_ids_sha256: fit_row_ids_sha256.into(),
        hold_row_ids_sha256: hold_row_ids_sha256.into(),
        fit_input_sha256: fit_input_sha256.into(),
        hold_input_sha256: hold_input_sha256.into(),
        encoder_identity_sha256: encoder.identity()?,
        encoder_binary_sha256: encoder.binary.sha256.clone(),
        encoder_weights_sha256: encoder.weights.sha256.clone(),
        encoder_manifest_sha256: encoder.manifest.sha256.clone(),
        encoder_source_hashes_sha256: canonical_sha256(&encoder.source_hashes)?,
        seed: encoder.seed,
        sweeps: encoder.sweeps,
        scales: encoder.scales.clone(),
        batch_size: encoder.batch_size,
        runtime_versions_sha256: canonical_sha256(&encoder.runtime_versions)?,
        observed_device: "cuda:0".into(),
    };
    key.validate()?;
    Ok(key)
}
