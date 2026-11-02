fn load_ranked_evaluation(row: &RankedCandidate) -> Result<CandidateEvaluation> {
    let (evaluation, digest): (CandidateEvaluation, _) =
        read_receipt(&relocated_input_path(&row.evaluation_path))?;
    if digest != row.evaluation_sha256
        || evaluation.candidate != row.candidate
        || evaluation.score.robust_loss != row.robust_loss
        || evaluation.score.worst_fold_nrmse != row.worst_fold_nrmse
        || !evaluation.development_only
        || evaluation.confirmatory_valid
    {
        bail!("ranked candidate evaluation link drift");
    }
    Ok(evaluation)
}

fn one_se_candidate(row: &RankedCandidate) -> Result<OneSeCandidate> {
    let evaluation = load_ranked_evaluation(row)?;
    Ok(OneSeCandidate {
        semantic_sha256: row.candidate.semantic_sha256.clone(),
        complexity: row.candidate.complexity,
        robust_loss: row.robust_loss,
        repeat_nrmse: evaluation.score.repeat_nrmse,
        worst_fold_nrmse: row.worst_fold_nrmse,
    })
}

fn select_one_se(
    ranked: &[RankedCandidate],
    keep: impl Fn(&Candidate) -> bool,
) -> Result<Candidate> {
    let eligible = ranked
        .iter()
        .filter(|row| keep(&row.candidate))
        .collect::<Vec<_>>();
    if eligible.is_empty() {
        bail!("one-SE counterfactual has no eligible candidates");
    }
    let summaries = eligible
        .iter()
        .map(|row| one_se_candidate(row))
        .collect::<Result<Vec<_>>>()?;
    Ok(eligible[one_standard_error_select(&summaries)?]
        .candidate
        .clone())
}

fn selector_evaluations(
    ranked: &[RankedCandidate],
) -> Result<HashMap<String, CandidateEvaluation>> {
    ranked
        .iter()
        .map(|row| {
            Ok((
                row.candidate.semantic_sha256.clone(),
                load_ranked_evaluation(row)?,
            ))
        })
        .collect()
}

fn residual_correlation(
    left: &CandidateEvaluation,
    right: &CandidateEvaluation,
    targets: &[f64],
) -> Result<f64> {
    if left.repeat_predictions.len() != right.repeat_predictions.len()
        || left.repeat_predictions.is_empty()
    {
        bail!("residual correlation repeat mismatch");
    }
    let mut x = Vec::new();
    let mut y = Vec::new();
    for (left_repeat, right_repeat) in left
        .repeat_predictions
        .iter()
        .zip(&right.repeat_predictions)
    {
        if left_repeat.len() != targets.len() || right_repeat.len() != targets.len() {
            bail!("residual correlation row mismatch");
        }
        x.extend(
            targets
                .iter()
                .zip(left_repeat)
                .map(|(truth, prediction)| truth - prediction),
        );
        y.extend(
            targets
                .iter()
                .zip(right_repeat)
                .map(|(truth, prediction)| truth - prediction),
        );
    }
    let mean_x = x.iter().sum::<f64>() / x.len() as f64;
    let mean_y = y.iter().sum::<f64>() / y.len() as f64;
    let covariance = x
        .iter()
        .zip(&y)
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum::<f64>();
    let scale_x = x.iter().map(|x| (x - mean_x).powi(2)).sum::<f64>();
    let scale_y = y.iter().map(|y| (y - mean_y).powi(2)).sum::<f64>();
    if scale_x <= 0.0 || scale_y <= 0.0 {
        return Ok(1.0);
    }
    Ok((covariance / (scale_x * scale_y).sqrt()).clamp(-1.0, 1.0))
}

fn weighted_prediction(
    evaluations: &HashMap<String, CandidateEvaluation>,
    candidates: &[Candidate],
    weights: &[f64],
    repeat: usize,
) -> Result<Vec<f64>> {
    if candidates.is_empty()
        || candidates.len() != weights.len()
        || weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
        || (weights.iter().sum::<f64>() - 1.0).abs() > 1e-10
    {
        bail!("ensemble weights violate the convex contract");
    }
    let rows = evaluations
        .get(&candidates[0].semantic_sha256)
        .and_then(|evaluation| evaluation.repeat_predictions.get(repeat))
        .context("ensemble repeat is missing")?
        .len();
    let mut predictions = vec![0.0; rows];
    for (candidate, weight) in candidates.iter().zip(weights) {
        let values = evaluations
            .get(&candidate.semantic_sha256)
            .and_then(|evaluation| evaluation.repeat_predictions.get(repeat))
            .context("ensemble base evaluation is missing")?;
        if values.len() != rows {
            bail!("ensemble prediction row mismatch");
        }
        for (prediction, value) in predictions.iter_mut().zip(values) {
            *prediction += weight * value;
        }
    }
    Ok(predictions)
}

fn prediction_nrmse(targets: &[f64], predictions: &[f64], target_sd: f64) -> Result<f64> {
    Ok((squared_error(targets, predictions)? / targets.len() as f64).sqrt() / target_sd)
}

fn optimal_convex_alpha(targets: &[f64], current: &[f64], candidate: &[f64]) -> Result<f64> {
    if targets.len() != current.len() || targets.len() != candidate.len() || targets.is_empty() {
        bail!("convex mixing inputs have different shapes");
    }
    let numerator = targets
        .iter()
        .zip(current)
        .zip(candidate)
        .map(|((truth, current), candidate)| (truth - current) * (candidate - current))
        .sum::<f64>();
    let denominator = current
        .iter()
        .zip(candidate)
        .map(|(current, candidate)| (candidate - current).powi(2))
        .sum::<f64>();
    if denominator <= 0.0 {
        return Ok(0.0);
    }
    Ok((numerator / denominator).clamp(0.0, 1.0))
}

fn add_prediction(current: &[f64], candidate: &[f64], alpha: f64) -> Vec<f64> {
    current
        .iter()
        .zip(candidate)
        .map(|(current, candidate)| (1.0 - alpha) * current + alpha * candidate)
        .collect()
}

fn ensemble_nrmse(
    evaluations: &HashMap<String, CandidateEvaluation>,
    candidates: &[Candidate],
    weights: &[f64],
    targets: &[f64],
) -> Result<f64> {
    let sd = target_sd(targets)?;
    let repeats = evaluations
        .get(&candidates[0].semantic_sha256)
        .context("ensemble evaluation is absent")?
        .repeat_predictions
        .len();
    Ok((0..repeats)
        .map(|repeat| {
            prediction_nrmse(
                targets,
                &weighted_prediction(evaluations, candidates, weights, repeat)?,
                sd,
            )
        })
        .collect::<Result<Vec<_>>>()?
        .iter()
        .sum::<f64>()
        / repeats as f64)
}

fn build_ensemble(
    manifest: &TrainOnlyRedteamManifest,
    ranked: &[RankedCandidate],
    selected_single: &Candidate,
    evaluations: &HashMap<String, CandidateEvaluation>,
    targets: &[f64],
) -> Result<(Vec<Candidate>, Vec<f64>)> {
    let selected_evaluation = evaluations
        .get(&selected_single.semantic_sha256)
        .context("selected single evaluation is absent")?;
    if selected_evaluation.repeat_predictions.len() != 2 {
        bail!("selector ensemble requires exactly two repeats");
    }
    let mut leaders = vec![selected_single.clone()];
    let mut head_counts = HashMap::from([(selected_single.head_family.clone(), 1usize)]);
    for row in ranked {
        if leaders.len() == manifest.config.selector.ensemble_leaders {
            break;
        }
        if row.candidate.semantic_sha256 == selected_single.semantic_sha256
            || *head_counts.get(&row.candidate.head_family).unwrap_or(&0)
                >= manifest.config.selector.ensemble_head_family_cap
        {
            continue;
        }
        let evaluation = &evaluations[&row.candidate.semantic_sha256];
        if leaders.iter().any(|leader| {
            residual_correlation(evaluation, &evaluations[&leader.semantic_sha256], targets)
                .is_ok_and(|correlation| {
                    correlation > manifest.config.selector.residual_correlation_cap
                })
        }) {
            continue;
        }
        *head_counts
            .entry(row.candidate.head_family.clone())
            .or_default() += 1;
        leaders.push(row.candidate.clone());
    }

    let sd = target_sd(targets)?;
    let mut chosen = vec![selected_single.clone()];
    let mut weights = vec![1.0];
    while chosen.len() < manifest.config.selector.ensemble_max_models {
        let current_a = weighted_prediction(evaluations, &chosen, &weights, 0)?;
        let current_b = weighted_prediction(evaluations, &chosen, &weights, 1)?;
        let current_crossfit = (prediction_nrmse(targets, &current_a, sd)?
            + prediction_nrmse(targets, &current_b, sd)?)
            / 2.0;
        let mut best: Option<(f64, f64, Candidate)> = None;
        for candidate in &leaders {
            if chosen
                .iter()
                .any(|present| present.semantic_sha256 == candidate.semantic_sha256)
            {
                continue;
            }
            let evaluation = &evaluations[&candidate.semantic_sha256];
            let candidate_a = &evaluation.repeat_predictions[0];
            let candidate_b = &evaluation.repeat_predictions[1];
            let alpha_a = optimal_convex_alpha(targets, &current_a, candidate_a)?;
            let alpha_b = optimal_convex_alpha(targets, &current_b, candidate_b)?;
            let crossfit_loss = (prediction_nrmse(
                targets,
                &add_prediction(&current_b, candidate_b, alpha_a),
                sd,
            )? + prediction_nrmse(
                targets,
                &add_prediction(&current_a, candidate_a, alpha_b),
                sd,
            )?) / 2.0;
            let alpha = (alpha_a + alpha_b) / 2.0;
            if alpha > 0.0
                && best.as_ref().is_none_or(|(loss, _, best_candidate)| {
                    crossfit_loss < *loss
                        || (crossfit_loss == *loss
                            && candidate.semantic_sha256 < best_candidate.semantic_sha256)
                })
            {
                best = Some((crossfit_loss, alpha, candidate.clone()));
            }
        }
        let Some((loss, alpha, candidate)) = best else {
            break;
        };
        let improvement = (current_crossfit - loss) / current_crossfit.max(1e-15);
        if improvement
            < manifest
                .config
                .selector
                .crossfit_minimum_relative_improvement
        {
            break;
        }
        for weight in &mut weights {
            *weight *= 1.0 - alpha;
        }
        weights.push(alpha);
        chosen.push(candidate);
        let sum = weights.iter().sum::<f64>();
        for weight in &mut weights {
            *weight /= sum;
        }
    }
    Ok((chosen, weights))
}

struct SelectorDecision {
    ranked: Vec<RankedCandidate>,
    selected_single: Candidate,
    ensemble_candidates: Vec<Candidate>,
    ensemble_weights: Vec<f64>,
    audit_single_nrmse: f64,
    audit_ensemble_nrmse: f64,
    ensemble_accepted: bool,
    final_variant: FrozenVariant,
}

fn select_final_variant(
    root: &Path,
    manifest: &TrainOnlyRedteamManifest,
    dataset: &DatasetInput,
    candidates: Vec<Candidate>,
    workers: usize,
) -> Result<SelectorDecision> {
    let data = read_regression_csv(&dataset.train_path)?;
    let (bank, bank_digest): (FoldBankReceipt, _) = read_receipt(&dataset.fold_bank_path)?;
    if bank_digest != dataset.fold_bank_sha256 {
        bail!("selector fold-bank link drift");
    }
    let ranked = evaluate_candidates(
        root,
        manifest,
        dataset,
        &data,
        "selector",
        &bank.selector,
        &candidates,
        workers,
    )?;
    let selected_single = select_one_se(&ranked, |_| true)?;
    let selector_evaluation_map = selector_evaluations(&ranked)?;
    let (ensemble_candidates, ensemble_weights) = build_ensemble(
        manifest,
        &ranked,
        &selected_single,
        &selector_evaluation_map,
        &data.y,
    )?;

    let audit_candidates = {
        let mut unique = HashMap::new();
        merge_candidate(&mut unique, selected_single.clone());
        for candidate in &ensemble_candidates {
            merge_candidate(&mut unique, candidate.clone());
        }
        unique.into_values().collect::<Vec<_>>()
    };
    let audit_ranked = evaluate_candidates(
        root,
        manifest,
        dataset,
        &data,
        "audit",
        &bank.audit,
        &audit_candidates,
        workers,
    )?;
    let audit_evaluations = selector_evaluations(&audit_ranked)?;
    let audit_single_nrmse = ensemble_nrmse(
        &audit_evaluations,
        std::slice::from_ref(&selected_single),
        &[1.0],
        &data.y,
    )?;
    let audit_ensemble_nrmse = ensemble_nrmse(
        &audit_evaluations,
        &ensemble_candidates,
        &ensemble_weights,
        &data.y,
    )?;
    let ensemble_accepted = ensemble_candidates.len() > 1
        && (audit_single_nrmse - audit_ensemble_nrmse) / audit_single_nrmse.max(1e-15)
            >= manifest.config.selector.audit_minimum_relative_improvement;
    let final_variant = if ensemble_accepted {
        FrozenVariant {
            name: "smart-v3-final-ensemble".to_string(),
            candidates: ensemble_candidates.clone(),
            weights: ensemble_weights.clone(),
        }
    } else {
        FrozenVariant {
            name: "smart-v3-final-single-audit-fallback".to_string(),
            candidates: vec![selected_single.clone()],
            weights: vec![1.0],
        }
    };
    Ok(SelectorDecision {
        ranked,
        selected_single,
        ensemble_candidates,
        ensemble_weights,
        audit_single_nrmse,
        audit_ensemble_nrmse,
        ensemble_accepted,
        final_variant,
    })
}

fn finalize_combined_selector(
    root: &Path,
    manifest: &TrainOnlyRedteamManifest,
    dataset: &DatasetInput,
    workers: usize,
) -> Result<Option<(PathBuf, String)>> {
    let dataset_root = root.join("datasets").join(&dataset.dataset_hash);
    let g20_path = dataset_root.join("g20-extension-complete.v1.json");
    let smart_path = dataset_root.join("smart-v3-complete.v1.json");
    if !g20_path.exists() || !smart_path.exists() {
        return Ok(None);
    }
    let output = dataset_root.join("selector-complete.v1.json");
    if output.exists() {
        let (_, digest): (SelectorComplete, _) = read_receipt(&output)?;
        return Ok(Some((output, digest)));
    }
    let (g20, g20_digest): (G20Complete, _) = read_receipt(&g20_path)?;
    let (smart, smart_digest): (SmartComplete, _) = read_receipt(&smart_path)?;
    if !g20.development_only
        || g20.confirmatory_valid
        || !smart.development_only
        || smart.confirmatory_valid
    {
        bail!("arm disclosure drift before selector");
    }
    let mut pool = HashMap::new();
    for row in &g20.top_archive_search_oof {
        merge_candidate(&mut pool, row.candidate.clone());
    }
    for rows in smart.archive_top20_by_island.values() {
        for row in rows {
            merge_candidate(&mut pool, row.candidate.clone());
        }
    }
    let mut candidates = pool.into_values().collect::<Vec<_>>();
    candidates.sort_by(|left, right| left.semantic_sha256.cmp(&right.semantic_sha256));
    let decision = select_final_variant(root, manifest, dataset, candidates, workers)?;
    let smart_semantics = smart
        .archive_top20_by_island
        .values()
        .flatten()
        .map(|row| row.candidate.semantic_sha256.clone())
        .collect::<HashSet<_>>();
    let smart_single = select_one_se(&decision.ranked, |candidate| {
        smart_semantics.contains(&candidate.semantic_sha256)
    })?;
    let no_jitter = select_one_se(&decision.ranked, |candidate| {
        smart_semantics.contains(&candidate.semantic_sha256)
            && !candidate.proposal_origins.contains("jittered_context")
    })?;
    let original_seed = manifest.config.proposal.exact_decoder_seeds[0];
    let original_decoder_seed_only = select_one_se(&decision.ranked, |candidate| {
        smart_semantics.contains(&candidate.semantic_sha256)
            && !candidate.proposal_origins.contains("jittered_context")
            && (candidate.decoder_seeds.is_empty()
                || candidate.decoder_seeds == BTreeSet::from([original_seed]))
    })?;
    let receipt = SelectorComplete {
        schema: "TrainOnlyRedteamSelectorComplete.v1".to_string(),
        run_id: manifest.run_id.clone(),
        dataset_hash: dataset.dataset_hash.clone(),
        g20_complete_sha256: g20_digest,
        smart_complete_sha256: smart_digest,
        candidates: decision.ranked,
        selected_single: decision.selected_single,
        smart_single,
        no_jitter,
        original_decoder_seed_only,
        ensemble_candidates: decision.ensemble_candidates,
        ensemble_weights: decision.ensemble_weights,
        audit_single_nrmse: decision.audit_single_nrmse,
        audit_ensemble_nrmse: decision.audit_ensemble_nrmse,
        ensemble_accepted: decision.ensemble_accepted,
        final_variant: decision.final_variant,
        selection_frozen_before_test_access: true,
        scq_execution: scq_execution_provenance(manifest, Some(dataset), workers),
        development_only: true,
        confirmatory_valid: false,
    };
    let digest = write_receipt(&output, &receipt)?;
    Ok(Some((output, digest)))
}

fn finalize_smart_only_selector(
    root: &Path,
    manifest: &TrainOnlyRedteamManifest,
    dataset: &DatasetInput,
    workers: usize,
) -> Result<Option<(PathBuf, String)>> {
    let dataset_root = root.join("datasets").join(&dataset.dataset_hash);
    let smart_path = dataset_root.join("smart-v3-complete.v1.json");
    if !smart_path.exists() {
        return Ok(None);
    }
    let output = dataset_root.join("smart-v3-selector-complete.v1.json");
    if output.exists() {
        let (_, digest): (SmartV3SelectorComplete, _) = read_receipt(&output)?;
        return Ok(Some((output, digest)));
    }
    let (smart, smart_digest): (SmartComplete, _) = read_receipt(&smart_path)?;
    if smart.schema != "TrainOnlyRedteamSmartComplete.v1"
        || smart.run_id != manifest.run_id
        || smart.dataset_hash != dataset.dataset_hash
        || !smart.development_only
        || smart.confirmatory_valid
    {
        bail!("Smart-v3 completion receipt drift before smart-only selector");
    }
    let expected_islands = manifest
        .config
        .smart_v3
        .island_roots
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    if smart
        .archive_top20_by_island
        .keys()
        .cloned()
        .collect::<BTreeSet<_>>()
        != expected_islands
    {
        bail!("Smart-v3 selector archive islands drift");
    }
    let mut pool = HashMap::new();
    for rows in smart.archive_top20_by_island.values() {
        for row in rows {
            merge_candidate(&mut pool, row.candidate.clone());
        }
    }
    let mut candidates = pool.into_values().collect::<Vec<_>>();
    candidates.sort_by(|left, right| left.semantic_sha256.cmp(&right.semantic_sha256));
    let decision = select_final_variant(root, manifest, dataset, candidates, workers)?;
    let receipt = SmartV3SelectorComplete {
        schema: "TrainOnlyRedteamSmartV3SelectorComplete.v1".to_string(),
        run_id: manifest.run_id.clone(),
        dataset_hash: dataset.dataset_hash.clone(),
        smart_complete_sha256: smart_digest,
        candidates: decision.ranked,
        selected_single: decision.selected_single,
        ensemble_candidates: decision.ensemble_candidates,
        ensemble_weights: decision.ensemble_weights,
        audit_single_nrmse: decision.audit_single_nrmse,
        audit_ensemble_nrmse: decision.audit_ensemble_nrmse,
        ensemble_accepted: decision.ensemble_accepted,
        final_variant: decision.final_variant,
        selection_frozen_before_test_access: true,
        scq_execution: scq_execution_provenance(manifest, Some(dataset), workers),
        development_only: true,
        confirmatory_valid: false,
    };
    let digest = write_receipt(&output, &receipt)?;
    Ok(Some((output, digest)))
}

fn finalize_selector(
    root: &Path,
    manifest: &TrainOnlyRedteamManifest,
    dataset: &DatasetInput,
    selection_mode: SelectionMode,
    workers: usize,
) -> Result<Option<(PathBuf, String)>> {
    match selection_mode {
        SelectionMode::Combined => finalize_combined_selector(root, manifest, dataset, workers),
        SelectionMode::SmartOnly => finalize_smart_only_selector(root, manifest, dataset, workers),
    }
}

fn arm_name(arm: Arm) -> &'static str {
    match arm {
        Arm::G20Extension => "g20-extension",
        Arm::SmartV3 => "smart-v3",
    }
}

pub(super) fn run_arm(
    manifest_path: &Path,
    arm: Arm,
    selection_mode: SelectionMode,
    resume: bool,
    workers: usize,
    selected_dataset: Option<&str>,
) -> Result<PathBuf> {
    if workers == 0 {
        bail!("--workers must be positive");
    }
    if selection_mode == SelectionMode::SmartOnly && arm != Arm::SmartV3 {
        bail!("smart-only selection mode runs only the Smart-v3 arm");
    }
    let (manifest, manifest_digest) = validate_manifest(manifest_path)?;
    let root = manifest_path
        .parent()
        .context("red-team manifest has no run root")?;
    let datasets = manifest
        .datasets
        .iter()
        .filter(|dataset| selected_dataset.is_none_or(|selected| selected == dataset.dataset_hash))
        .collect::<Vec<_>>();
    if datasets.is_empty() {
        bail!("selected dataset is absent from the red-team manifest");
    }
    let aggregate_only = selected_dataset.is_none()
        && std::env::var("REDTEAM_AGGREGATE_ONLY").as_deref() == Ok("true");
    let mut receipt_sha256 = BTreeMap::new();
    let mut selected_output = None;
    for dataset in datasets {
        let (path, digest) = if aggregate_only {
            match arm {
                Arm::G20Extension => {
                    let path = root
                        .join("datasets")
                        .join(&dataset.dataset_hash)
                        .join("g20-extension-complete.v1.json");
                    let (receipt, digest): (G20Complete, _) = read_receipt(&path)?;
                    if receipt.schema != "TrainOnlyRedteamG20Complete.v1"
                        || receipt.run_id != manifest.run_id
                        || receipt.dataset_hash != dataset.dataset_hash
                        || !receipt.development_only
                        || receipt.confirmatory_valid
                    {
                        bail!("aggregate-only G20 receipt drift");
                    }
                    (path, digest)
                }
                Arm::SmartV3 => {
                    let path = root
                        .join("datasets")
                        .join(&dataset.dataset_hash)
                        .join("smart-v3-complete.v1.json");
                    let (receipt, digest): (SmartComplete, _) = read_receipt(&path)?;
                    if receipt.schema != "TrainOnlyRedteamSmartComplete.v1"
                        || receipt.run_id != manifest.run_id
                        || receipt.dataset_hash != dataset.dataset_hash
                        || !receipt.development_only
                        || receipt.confirmatory_valid
                    {
                        bail!("aggregate-only smart receipt drift");
                    }
                    (path, digest)
                }
            }
        } else {
            match arm {
                Arm::G20Extension => run_g20_dataset(root, &manifest, dataset, resume, workers)?,
                Arm::SmartV3 => run_smart_dataset(root, &manifest, dataset, resume, workers)?,
            }
        };
        receipt_sha256.insert(dataset.dataset_hash.clone(), digest);
        selected_output = Some(path);
        if !aggregate_only {
            let _ = finalize_selector(root, &manifest, dataset, selection_mode, workers)?;
        }
    }
    if selected_dataset.is_some() {
        return selected_output.context("arm produced no output");
    }
    if receipt_sha256.len() != manifest.datasets.len() {
        bail!("global arm completion requires every configured dataset");
    }
    let receipt = ArmComplete {
        schema: "TrainOnlyRedteamArmComplete.v1".to_string(),
        run_id: manifest.run_id.clone(),
        arm: arm_name(arm).to_string(),
        manifest_sha256: manifest_digest.clone(),
        dataset_receipt_sha256: receipt_sha256,
        scq_execution: scq_execution_provenance(&manifest, None, workers),
        development_only: true,
        confirmatory_valid: false,
    };
    let path = root.join(format!("{}-complete.v1.json", arm_name(arm)));
    write_receipt(&path, &receipt)?;

    let selector_filename = match selection_mode {
        SelectionMode::Combined => "selector-complete.v1.json",
        SelectionMode::SmartOnly => "smart-v3-selector-complete.v1.json",
    };
    if manifest.datasets.iter().all(|dataset| {
        root.join("datasets")
            .join(&dataset.dataset_hash)
            .join(selector_filename)
            .exists()
    }) {
        let selector_sha256 = manifest
            .datasets
            .iter()
            .map(|dataset| {
                let path = root
                    .join("datasets")
                    .join(&dataset.dataset_hash)
                    .join(selector_filename);
                let digest = match selection_mode {
                    SelectionMode::Combined => read_receipt::<SelectorComplete>(&path)?.1,
                    SelectionMode::SmartOnly => read_receipt::<SmartV3SelectorComplete>(&path)?.1,
                };
                Ok((dataset.dataset_hash.clone(), digest))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        let (cohort_path, schema) = match selection_mode {
            SelectionMode::Combined => (
                root.join("selector-complete.v1.json"),
                "TrainOnlyRedteamSelectorCohortComplete.v1",
            ),
            SelectionMode::SmartOnly => (
                root.join("smart-v3-selector-complete.v1.json"),
                "TrainOnlyRedteamSmartV3SelectorCohortComplete.v1",
            ),
        };
        write_receipt(
            &cohort_path,
            &json!({
                "schema": schema,
                "run_id": manifest.run_id,
                "manifest_sha256": manifest_digest,
                "selector_sha256": selector_sha256,
                "scq_execution": scq_execution_provenance(&manifest, None, workers),
                "selection_frozen_before_test_access": true,
                "development_only": true,
                "confirmatory_valid": false,
            }),
        )?;
    }
    Ok(path)
}
