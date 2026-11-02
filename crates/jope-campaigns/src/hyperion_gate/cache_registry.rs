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

/// Execute the train-only gate from its four authorized inputs.  Operational
/// failures after registry/predictor authorization are materialized as a raw
/// decision; malformed authority inputs are rejected without guessing.
pub fn run_gate_command(
    dataset_id: &str,
    registry_path: &Path,
    predictor_path: &Path,
    artifact_root: &Path,
) -> Result<JopeHyperionGateDecisionV1, String> {
    let started = Instant::now();
    validate_opaque_id(dataset_id)?;
    let registry_bytes =
        fs::read(registry_path).map_err(|error| format!("read train-only registry: {error}"))?;
    let registry: JopeHyperionTrainOnlyRegistryV2 = serde_json::from_slice(&registry_bytes)
        .map_err(|error| format!("decode train-only registry: {error}"))?;
    let registry_root = registry_path.parent().unwrap_or_else(|| Path::new("."));
    registry.validate(registry_root)?;
    let registry_sha256 = registry.identity()?;
    let dataset = registry
        .datasets
        .iter()
        .find(|value| value.dataset_id == dataset_id)
        .ok_or_else(|| "opaque dataset ID is absent from the train-only registry".to_string())?;
    let predictor_bytes =
        fs::read(predictor_path).map_err(|error| format!("read frozen AFPM predictor: {error}"))?;
    let predictor: FrozenPredictorV1 = serde_json::from_slice(&predictor_bytes)
        .map_err(|error| format!("decode frozen AFPM predictor: {error}"))?;
    predictor.validate()?;
    if predictor.license_track() != crate::jope_dlgp_best5::LicenseTrackV1::ProductionCompatible
        || predictor.pipelines().len() != predictor.component_count()
    {
        return Err("Hyperion gate requires a production-compatible pure-AFPM predictor".into());
    }
    let predictor_sha256 = predictor.semantic_sha256()?;
    let encoder_sha256 = registry.encoder.identity()?;
    let unavailable_cv5_sha256 =
        raw_sha256(format!("unavailable-cv5-plan-v1|{}", dataset.train_sha256).as_bytes());
    let unavailable_cv7_sha256 =
        raw_sha256(format!("unavailable-cv7-plan-v1|{}", dataset.train_sha256).as_bytes());
    let preflight = (|| -> Result<(RegressionTable, PrecomputedTrainFeatureBlocks), String> {
        let table = read_authorized_table(registry_root, dataset)?;
        let precomputed =
            load_precomputed_train_features(&registry, registry_root, &registry_sha256, dataset)?;
        require_frozen_precomputed_dataset_binding(&registry, registry_root, &precomputed.receipt)?;
        let cache_receipt_root = artifact_root.join("cache-receipts");
        fs::create_dir_all(&cache_receipt_root)
            .map_err(|error| format!("create cache receipt root: {error}"))?;
        write_immutable(
            &cache_receipt_root.join("precomputed-train-features.v1.json"),
            &canonical_json_bytes(&precomputed.receipt)?,
        )?;
        Ok((table, precomputed))
    })();
    let (authorized_table, precomputed) = match preflight {
        Ok(value) => value,
        Err(reason) => {
            let decision = raw_fallback_decision(
                dataset_id,
                &dataset.train_sha256,
                &predictor_sha256,
                &encoder_sha256,
                &unavailable_cv5_sha256,
                &unavailable_cv7_sha256,
                reason,
                started,
            );
            write_gate_artifacts(
                artifact_root,
                &decision,
                &predictor,
                dataset.feature_count as usize,
            )?;
            return Ok(decision);
        }
    };
    let cv5_plan = match build_grouped_fold_plan(GateStageV1::Cv5Screen, &dataset.atomic_groups) {
        Ok(plan) => plan,
        Err(reason) => {
            let decision = raw_fallback_decision(
                dataset_id,
                &dataset.train_sha256,
                &predictor_sha256,
                &encoder_sha256,
                &unavailable_cv5_sha256,
                &unavailable_cv7_sha256,
                reason,
                started,
            );
            write_gate_artifacts(
                artifact_root,
                &decision,
                &predictor,
                dataset.feature_count as usize,
            )?;
            return Ok(decision);
        }
    };
    let cv7_plan =
        match build_grouped_fold_plan(GateStageV1::Cv7Confirmation, &dataset.atomic_groups) {
            Ok(plan) => plan,
            Err(reason) => {
                let decision = raw_fallback_decision(
                    dataset_id,
                    &dataset.train_sha256,
                    &predictor_sha256,
                    &encoder_sha256,
                    &canonical_sha256(&cv5_plan)?,
                    &unavailable_cv7_sha256,
                    reason,
                    started,
                );
                write_gate_artifacts(
                    artifact_root,
                    &decision,
                    &predictor,
                    dataset.feature_count as usize,
                )?;
                return Ok(decision);
            }
        };
    let cv5_plan_sha256 = canonical_sha256(&cv5_plan)?;
    let cv7_plan_sha256 = canonical_sha256(&cv7_plan)?;

    let operational = (|| -> Result<JopeHyperionGateDecisionV1, String> {
        let (eligible_arms, excluded_arms) = independently_feasible_feature_views(
            &cv5_plan,
            &cv7_plan,
            &precomputed.receipt.available_views(),
            dataset.feature_count as usize,
        );
        if eligible_arms.is_empty() {
            return Err(format!(
                "no precomputed Hyperion arm has sufficient CV evidence: {excluded_arms:?}"
            ));
        }
        registry.evaluator.verify_artifact()?;
        let work_root = artifact_root.join("work");
        let receipt_root = artifact_root.join("fold-receipts");
        fs::create_dir_all(&receipt_root)
            .map_err(|error| format!("create fold receipt root: {error}"))?;
        let adapter = CorpusEvalAdapter::new(
            registry.evaluator.clone(),
            work_root.join("scoring"),
            SCORING_BUDGET.as_secs(),
        )?;
        let cv5 = execute_stage(
            &registry.encoder,
            dataset,
            &authorized_table,
            &precomputed,
            &predictor,
            &adapter,
            &cv5_plan,
            &cv5_plan_sha256,
            &eligible_arms,
            &receipt_root,
            started,
            0,
            0,
        )?;
        let thresholds = GateThresholdsV1::default();
        let (cv5_candidates, advanced) =
            screen_cv5_arms(&cv5.evidence, &eligible_arms, &thresholds)?;
        let Some(advanced_arm) = advanced else {
            return build_measured_decision(
                dataset_id,
                dataset,
                &predictor_sha256,
                &encoder_sha256,
                &cv5_plan_sha256,
                &cv7_plan_sha256,
                thresholds,
                cv5_candidates,
                None,
                None,
                Some("no CV5 arm passed every frozen advancement criterion"),
                &cv5.receipts,
                cv5.encoding_milliseconds,
                cv5.scoring_milliseconds,
                started,
            );
        };
        if started.elapsed() >= WALL_CLOCK_BUDGET {
            return Err("wall-clock budget expired before CV7 confirmation".into());
        }
        let cv7 = execute_stage(
            &registry.encoder,
            dataset,
            &authorized_table,
            &precomputed,
            &predictor,
            &adapter,
            &cv7_plan,
            &cv7_plan_sha256,
            &[advanced_arm],
            &receipt_root,
            started,
            cv5.encoding_milliseconds,
            cv5.scoring_milliseconds,
        )?;
        let confirmation = confirm_cv7(&cv7.evidence, advanced_arm, &thresholds)?;
        let fallback = (!confirmation.passed)
            .then_some("frozen CV5 arm failed one or more CV7 confirmation criteria");
        let mut receipts = cv5.receipts;
        receipts.extend(cv7.receipts);
        build_measured_decision(
            dataset_id,
            dataset,
            &predictor_sha256,
            &encoder_sha256,
            &cv5_plan_sha256,
            &cv7_plan_sha256,
            thresholds,
            cv5_candidates,
            Some(advanced_arm),
            Some(confirmation),
            fallback,
            &receipts,
            cv5.encoding_milliseconds + cv7.encoding_milliseconds,
            cv5.scoring_milliseconds + cv7.scoring_milliseconds,
            started,
        )
    })();

    let decision = match operational {
        Ok(decision) => decision,
        Err(reason) => raw_fallback_decision(
            dataset_id,
            &dataset.train_sha256,
            &predictor_sha256,
            &encoder_sha256,
            &cv5_plan_sha256,
            &cv7_plan_sha256,
            reason,
            started,
        ),
    };
    decision.validate()?;
    write_gate_artifacts(
        artifact_root,
        &decision,
        &predictor,
        dataset.feature_count as usize,
    )?;
    Ok(decision)
}

/// Precompute the pinned full-train, CV5, and CV7 encodings for either one
/// registry member or the complete frozen registry.  A complete run publishes
/// a coverage freeze; a targeted run only fills content-addressed entries.
pub fn build_registry_cache(
    registry_path: &Path,
    dataset_id: Option<&str>,
) -> Result<RegistryCacheCoverageV2, String> {
    if let Some(dataset_id) = dataset_id {
        validate_opaque_id(dataset_id)?;
    }
    let registry_bytes =
        fs::read(registry_path).map_err(|error| format!("read train-only registry: {error}"))?;
    let registry: JopeHyperionTrainOnlyRegistryV2 = serde_json::from_slice(&registry_bytes)
        .map_err(|error| format!("decode train-only registry: {error}"))?;
    let registry_root = registry_path.parent().unwrap_or_else(|| Path::new("."));
    registry.validate(registry_root)?;
    registry.encoder.verify_artifacts()?;
    require_cuda_device(registry.encoder.device_ordinal)?;
    let registry_sha256 = registry.identity()?;
    let cache_root = registry.resolved_cache_root(registry_root)?;
    let selected = registry
        .datasets
        .iter()
        .filter(|dataset| dataset_id.is_none_or(|wanted| dataset.dataset_id == wanted))
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return Err("requested dataset is absent from the train-only registry".into());
    }
    let mut datasets = Vec::with_capacity(selected.len());
    for dataset in selected {
        let source = read_authorized_table(registry_root, dataset)?;
        let full_train = encode_full_train(
            &registry.encoder,
            &source,
            dataset,
            &registry_sha256,
            &cache_root,
        )?;
        let cv5_plan = build_grouped_fold_plan(GateStageV1::Cv5Screen, &dataset.atomic_groups)?;
        let cv7_plan =
            build_grouped_fold_plan(GateStageV1::Cv7Confirmation, &dataset.atomic_groups)?;
        let mut cv5 = Vec::with_capacity(5);
        let cv5_sha256 = canonical_sha256(&cv5_plan)?;
        for fold in &cv5_plan.folds {
            let encoded = encode_fold(
                &registry.encoder,
                &source,
                fold,
                &cv5_plan,
                crate::hyperion_gate::FoldEncodingScope {
                    plan_sha256: &cv5_sha256,
                    encoding_root: &cache_root,
                    registry_sha256: &registry_sha256,
                    dataset,
                },
                Instant::now(),
            )?;
            cv5.push(CacheCoverageEntryV2 {
                fold_index: Some(fold.fold_index),
                cache_key_sha256: encoded.cache_key_sha256,
                cache_receipt_sha256: encoded.cache_receipt_sha256,
            });
        }
        let mut cv7 = Vec::with_capacity(7);
        let cv7_sha256 = canonical_sha256(&cv7_plan)?;
        for fold in &cv7_plan.folds {
            let encoded = encode_fold(
                &registry.encoder,
                &source,
                fold,
                &cv7_plan,
                crate::hyperion_gate::FoldEncodingScope {
                    plan_sha256: &cv7_sha256,
                    encoding_root: &cache_root,
                    registry_sha256: &registry_sha256,
                    dataset,
                },
                Instant::now(),
            )?;
            cv7.push(CacheCoverageEntryV2 {
                fold_index: Some(fold.fold_index),
                cache_key_sha256: encoded.cache_key_sha256,
                cache_receipt_sha256: encoded.cache_receipt_sha256,
            });
        }
        datasets.push(DatasetCacheCoverageV2 {
            dataset_id: dataset.dataset_id.clone(),
            full_train,
            cv5,
            cv7,
        });
    }
    let coverage = RegistryCacheCoverageV2 {
        schema: RegistryCacheCoverageV2::SCHEMA.into(),
        registry_sha256,
        datasets,
    };
    coverage.validate()?;
    if dataset_id.is_none() {
        if coverage.datasets.len() != registry.datasets.len() {
            return Err("refusing to freeze incomplete registry cache coverage".into());
        }
        fs::create_dir_all(&cache_root)
            .map_err(|error| format!("create registry cache root: {error}"))?;
        write_immutable(
            &cache_root.join("registry-cache-coverage.v2.json"),
            &canonical_json_bytes(&coverage)?,
        )?;
        let historical = build_historical_train3_inventory(&registry, registry_root)?;
        write_immutable(
            &cache_root.join("historical-train3-inventory.v1.json"),
            &canonical_json_bytes(&historical)?,
        )?;
        verify_complete_registry_cache(registry_path)?;
    }
    Ok(coverage)
}

/// Read-only freeze check.  This never invokes Hyperion: every registry member
/// must already have one validated full-train receipt and all frozen CV5/CV7
/// fold receipts with exact shapes and content hashes.
pub fn verify_complete_registry_cache(
    registry_path: &Path,
) -> Result<RegistryCacheCoverageV2, String> {
    let bytes =
        fs::read(registry_path).map_err(|error| format!("read train-only registry: {error}"))?;
    let registry: JopeHyperionTrainOnlyRegistryV2 = serde_json::from_slice(&bytes)
        .map_err(|error| format!("decode train-only registry: {error}"))?;
    let registry_root = registry_path.parent().unwrap_or_else(|| Path::new("."));
    registry.validate(registry_root)?;
    let registry_sha256 = registry.identity()?;
    let cache_root = registry.resolved_cache_root(registry_root)?;
    let coverage_path = cache_root.join("registry-cache-coverage.v2.json");
    let coverage: RegistryCacheCoverageV2 = serde_json::from_slice(
        &fs::read(&coverage_path)
            .map_err(|error| format!("read registry cache coverage: {error}"))?,
    )
    .map_err(|error| format!("decode registry cache coverage: {error}"))?;
    coverage.validate()?;
    if coverage.registry_sha256 != registry_sha256
        || coverage.datasets.len() != registry.datasets.len()
        || coverage
            .datasets
            .iter()
            .map(|dataset| dataset.dataset_id.as_str())
            .collect::<Vec<_>>()
            != registry
                .datasets
                .iter()
                .map(|dataset| dataset.dataset_id.as_str())
                .collect::<Vec<_>>()
    {
        return Err("registry cache coverage does not exactly match the frozen registry".into());
    }
    for (dataset, covered) in registry.datasets.iter().zip(&coverage.datasets) {
        let source = read_authorized_table(registry_root, dataset)?;
        verify_full_train_entry(
            &registry.encoder,
            &source,
            dataset,
            &registry_sha256,
            &cache_root,
            &covered.full_train,
        )?;
        for (stage, entries) in [
            (GateStageV1::Cv5Screen, covered.cv5.as_slice()),
            (GateStageV1::Cv7Confirmation, covered.cv7.as_slice()),
        ] {
            let plan = build_grouped_fold_plan(stage, &dataset.atomic_groups)?;
            let plan_sha256 = canonical_sha256(&plan)?;
            for (fold, entry) in plan.folds.iter().zip(entries) {
                verify_fold_entry(
                    &registry.encoder,
                    &source,
                    dataset,
                    &registry_sha256,
                    &cache_root,
                    &plan,
                    &plan_sha256,
                    fold,
                    entry,
                )?;
            }
        }
    }
    Ok(coverage)
}

#[allow(clippy::too_many_arguments)]
fn verify_fold_entry(
    encoder: &HyperionEncoderIdentityV1,
    source: &RegressionTable,
    dataset: &TrainOnlyDatasetV1,
    registry_sha256: &str,
    cache_root: &Path,
    plan: &GroupedFoldPlanV1,
    plan_sha256: &str,
    fold: &GroupedFoldV1,
    entry: &CacheCoverageEntryV2,
) -> Result<(), String> {
    let stage = match plan.stage {
        GateStageV1::Cv5Screen => "cv5",
        GateStageV1::Cv7Confirmation => "cv7",
    };
    let fit_input = csv_for_rows(source, &fold.fit_rows, false);
    let hold_input = csv_for_rows(source, &fold.hold_rows, true);
    let expected_key = cache_key(
        registry_sha256,
        dataset,
        encoder,
        stage,
        plan_sha256,
        Some(fold.fold_index),
        &hash_selected_strings(&dataset.stable_row_ids, &fold.fit_rows),
        &hash_selected_strings(&dataset.stable_row_ids, &fold.hold_rows),
        &raw_sha256(&fit_input),
        &raw_sha256(&hold_input),
    )?;
    if entry.fold_index != Some(fold.fold_index)
        || entry.cache_key_sha256 != expected_key.identity()?
    {
        return Err("coverage entry does not match the frozen fold order".into());
    }
    let directory = cache_root
        .join("datasets")
        .join(&dataset.dataset_id)
        .join(stage)
        .join(&entry.cache_key_sha256[..2])
        .join(&entry.cache_key_sha256);
    let receipt: FoldEncodingCacheV2 = serde_json::from_slice(
        &fs::read(directory.join("fold-encoding-cache.v2.json"))
            .map_err(|error| format!("read fold encoding cache: {error}"))?,
    )
    .map_err(|error| format!("decode fold encoding cache: {error}"))?;
    receipt.validate(encoder.device_ordinal)?;
    if receipt.key != expected_key
        || receipt.key_sha256 != entry.cache_key_sha256
        || canonical_sha256(&receipt)? != entry.cache_receipt_sha256
    {
        return Err("fold encoding receipt differs from frozen coverage".into());
    }
    let expected = BTreeMap::from([
        (
            "cell_fit",
            (
                fold.fit_rows.len(),
                dataset.feature_count as usize,
                "cell-fit.vs",
            ),
        ),
        (
            "cell_hold",
            (
                fold.hold_rows.len(),
                dataset.feature_count as usize,
                "cell-hold.vs",
            ),
        ),
        (
            "row_fit",
            (fold.fit_rows.len(), ROW_EMBEDDING_WIDTH, "row-fit.vs"),
        ),
        (
            "row_hold",
            (fold.hold_rows.len(), ROW_EMBEDDING_WIDTH, "row-hold.vs"),
        ),
    ]);
    verify_receipted_sidecars(&directory, &receipt.sidecars, &expected)
}

fn verify_full_train_entry(
    encoder: &HyperionEncoderIdentityV1,
    source: &RegressionTable,
    dataset: &TrainOnlyDatasetV1,
    registry_sha256: &str,
    cache_root: &Path,
    entry: &CacheCoverageEntryV2,
) -> Result<(), String> {
    let rows = (0..source.targets.len()).collect::<Vec<_>>();
    let train_input = csv_for_rows(source, &rows, false);
    let dummy_input = csv_for_rows(source, &rows, true);
    let row_order_sha256 = hash_selected_strings(&dataset.stable_row_ids, &rows);
    let plan_sha256 =
        raw_sha256(format!("full-train-v2|{}|{row_order_sha256}", dataset.train_sha256).as_bytes());
    let expected_key = cache_key(
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
    if entry.fold_index.is_some() || entry.cache_key_sha256 != expected_key.identity()? {
        return Err("full-train coverage entry identity drifted".into());
    }
    let directory = cache_root
        .join("datasets")
        .join(&dataset.dataset_id)
        .join("full_train")
        .join(&entry.cache_key_sha256[..2])
        .join(&entry.cache_key_sha256);
    let receipt: FullTrainEncodingCacheV2 = serde_json::from_slice(
        &fs::read(directory.join("full-train-encoding-cache.v2.json"))
            .map_err(|error| format!("read full-train encoding cache: {error}"))?,
    )
    .map_err(|error| format!("decode full-train encoding cache: {error}"))?;
    receipt.validate(encoder.device_ordinal)?;
    if receipt.key != expected_key
        || receipt.key_sha256 != entry.cache_key_sha256
        || canonical_sha256(&receipt)? != entry.cache_receipt_sha256
    {
        return Err("full-train encoding receipt differs from frozen coverage".into());
    }
    let expected = BTreeMap::from([
        (
            "cell_train",
            (rows.len(), dataset.feature_count as usize, "cell-train.vs"),
        ),
        (
            "row_train",
            (rows.len(), ROW_EMBEDDING_WIDTH, "row-train.vs"),
        ),
    ]);
    verify_receipted_sidecars(&directory, &receipt.sidecars, &expected)
}

fn verify_receipted_sidecars(
    directory: &Path,
    sidecars: &[MatrixArtifactV1],
    expected: &BTreeMap<&str, (usize, usize, &str)>,
) -> Result<(), String> {
    if sidecars.len() != expected.len() {
        return Err("receipted sidecar coverage is incomplete".into());
    }
    for sidecar in sidecars {
        sidecar.validate()?;
        let (rows, columns, file) = expected
            .get(sidecar.role.as_str())
            .ok_or_else(|| "receipted sidecar role is unexpected".to_string())?;
        if sidecar.rows as usize != *rows || sidecar.columns as usize != *columns {
            return Err("receipted sidecar shape differs from the frozen fold".into());
        }
        let path = directory.join(file);
        let bytes = fs::read(&path)
            .map_err(|error| format!("read receipted sidecar {}: {error}", path.display()))?;
        if bytes.is_empty() || raw_sha256(&bytes) != sidecar.sha256 {
            return Err("receipted sidecar is empty or changed".into());
        }
        read_vs_matrix(&path, *rows, *columns, &sidecar.role)?;
    }
    Ok(())
}
