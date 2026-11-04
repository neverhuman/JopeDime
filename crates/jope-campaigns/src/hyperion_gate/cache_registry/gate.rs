use super::*;

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
