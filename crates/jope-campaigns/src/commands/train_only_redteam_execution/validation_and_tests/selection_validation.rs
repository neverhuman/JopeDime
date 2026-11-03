use selection_math::{build_ensemble, ensemble_nrmse, select_one_se, selector_evaluations};

pub(crate) fn validate_dataset_selection(
    root: &Path,
    manifest: &TrainOnlyRedteamManifest,
    dataset: &DatasetInput,
    workers: usize,
) -> Result<DatasetValidationReplay> {
    let data = read_regression_csv(&dataset.train_path)?;
    let (bank, bank_digest): (FoldBankReceipt, _) = read_receipt(&dataset.fold_bank_path)?;
    if bank_digest != dataset.fold_bank_sha256 {
        bail!("validation fold-bank link drift");
    }
    validate_fold_bank(&bank, &manifest.config)?;

    let g20_path = root
        .join("datasets")
        .join(&dataset.dataset_hash)
        .join("g20-extension-complete.v1.json");
    let (g20, g20_digest): (G20Complete, _) = read_receipt(&g20_path)?;
    if g20.schema != "TrainOnlyRedteamG20Complete.v1"
        || g20.run_id != manifest.run_id
        || g20.dataset_hash != dataset.dataset_hash
        || g20.parent_generation_sha256 != dataset.parent_generation_sha256
        || g20.extension_generation_sha256.len() != 10
        || !g20.development_only
        || g20.confirmatory_valid
    {
        bail!("G20 completion receipt drift");
    }
    validate_g20_generation_replay(root, dataset, manifest, &data, &g20, workers)?;

    let smart_path = root
        .join("datasets")
        .join(&dataset.dataset_hash)
        .join("smart-v3-complete.v1.json");
    let (smart, smart_digest): (SmartComplete, _) = read_receipt(&smart_path)?;
    if smart.schema != "TrainOnlyRedteamSmartComplete.v1"
        || smart.run_id != manifest.run_id
        || smart.dataset_hash != dataset.dataset_hash
        || smart.generations_completed < manifest.config.smart_v3.minimum_generations
        || smart.generations_completed > manifest.config.smart_v3.maximum_generations
        || smart.generation_sha256.len() != smart.generations_completed
        || smart.archive_top20_by_island.len() != ISLANDS
        || !smart.development_only
        || smart.confirmatory_valid
    {
        bail!("Smart-v3 completion receipt drift");
    }
    let screening_path = root
        .join("datasets")
        .join(&dataset.dataset_hash)
        .join("smart")
        .join("screening.v1.json");
    let (screening, screening_digest): (ScreeningReceipt, _) = read_receipt(&screening_path)?;
    if screening_digest != smart.screening_sha256
        || !screening.development_only
        || screening.confirmatory_valid
        || screening.exact_candidates < 80
        || screening.jittered_candidates < 60
        || screening.transfer_candidates != 256
        || screening.advanced_exact.len() != 80
        || screening.advanced_jittered.len() != 60
        || screening.advanced_transfer.len() != 100
    {
        bail!("Smart-v3 screening receipt drift");
    }
    validate_smart_generation_replay(root, dataset, manifest, &smart, &screening)?;

    let selector_path = root
        .join("datasets")
        .join(&dataset.dataset_hash)
        .join("selector-complete.v1.json");
    let (selector, selector_digest): (SelectorComplete, _) = read_receipt(&selector_path)?;
    if selector.schema != "TrainOnlyRedteamSelectorComplete.v1"
        || selector.run_id != manifest.run_id
        || selector.dataset_hash != dataset.dataset_hash
        || selector.g20_complete_sha256 != g20_digest
        || selector.smart_complete_sha256 != smart_digest
        || !selector.selection_frozen_before_test_access
        || !selector.development_only
        || selector.confirmatory_valid
    {
        bail!("selector completion receipt drift");
    }
    let mut hashes = BTreeMap::new();
    validate_ranked_rows(
        &selector.candidates,
        manifest,
        dataset,
        &data,
        "selector",
        &bank.selector,
        true,
        &mut hashes,
    )?;
    if select_one_se(&selector.candidates, |_| true)? != selector.selected_single {
        bail!("one-SE selected-single replay drift");
    }
    let smart_semantics = smart
        .archive_top20_by_island
        .values()
        .flatten()
        .map(|row| row.candidate.semantic_sha256.clone())
        .collect::<HashSet<_>>();
    let original_seed = manifest.config.proposal.exact_decoder_seeds[0];
    if select_one_se(&selector.candidates, |candidate| {
        smart_semantics.contains(&candidate.semantic_sha256)
    })? != selector.smart_single
        || select_one_se(&selector.candidates, |candidate| {
            smart_semantics.contains(&candidate.semantic_sha256)
                && !candidate.proposal_origins.contains("jittered_context")
        })? != selector.no_jitter
        || select_one_se(&selector.candidates, |candidate| {
            smart_semantics.contains(&candidate.semantic_sha256)
                && !candidate.proposal_origins.contains("jittered_context")
                && (candidate.decoder_seeds.is_empty()
                    || candidate.decoder_seeds == BTreeSet::from([original_seed]))
        })? != selector.original_decoder_seed_only
    {
        bail!("Smart-v3 counterfactual selector replay drift");
    }
    let selector_evaluation_map = selector_evaluations(&selector.candidates)?;
    let (ensemble_candidates, ensemble_weights) = build_ensemble(
        manifest,
        &selector.candidates,
        &selector.selected_single,
        &selector_evaluation_map,
        &data.y,
    )?;
    if ensemble_candidates != selector.ensemble_candidates
        || ensemble_weights != selector.ensemble_weights
    {
        bail!("ensemble cross-fit replay drift");
    }
    let mut audit_rows = selector
        .ensemble_candidates
        .iter()
        .map(|candidate| {
            let path =
                evaluation_path(root, &dataset.dataset_hash, "audit", candidate, &bank.audit)?;
            if !path.exists() {
                bail!("validated audit cache is missing");
            }
            let (evaluation, digest): (CandidateEvaluation, _) = read_receipt(&path)?;
            Ok(RankedCandidate {
                candidate: candidate.clone(),
                evaluation_path: absolute(&path)?,
                evaluation_sha256: digest,
                robust_loss: evaluation.score.robust_loss,
                worst_fold_nrmse: evaluation.score.worst_fold_nrmse,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    audit_rows.sort_by(rank_loss);
    validate_ranked_rows(
        &audit_rows,
        manifest,
        dataset,
        &data,
        "audit",
        &bank.audit,
        true,
        &mut hashes,
    )?;
    let audit_evaluations = selector_evaluations(&audit_rows)?;
    let audit_single = ensemble_nrmse(
        &audit_evaluations,
        std::slice::from_ref(&selector.selected_single),
        &[1.0],
        &data.y,
    )?;
    let audit_ensemble = ensemble_nrmse(
        &audit_evaluations,
        &selector.ensemble_candidates,
        &selector.ensemble_weights,
        &data.y,
    )?;
    let accepted = selector.ensemble_candidates.len() > 1
        && (audit_single - audit_ensemble) / audit_single.max(1e-15)
            >= manifest.config.selector.audit_minimum_relative_improvement;
    if audit_single != selector.audit_single_nrmse
        || audit_ensemble != selector.audit_ensemble_nrmse
        || accepted != selector.ensemble_accepted
        || selector.final_variant
            != if accepted {
                FrozenVariant {
                    name: "smart-v3-final-ensemble".to_string(),
                    candidates: selector.ensemble_candidates.clone(),
                    weights: selector.ensemble_weights.clone(),
                }
            } else {
                FrozenVariant {
                    name: "smart-v3-final-single-audit-fallback".to_string(),
                    candidates: vec![selector.selected_single.clone()],
                    weights: vec![1.0],
                }
            }
    {
        bail!("audit fallback replay drift");
    }
    Ok(DatasetValidationReplay {
        dataset_hash: dataset.dataset_hash.clone(),
        g20_complete_sha256: g20_digest,
        smart_complete_sha256: smart_digest,
        selector_sha256: selector_digest,
        evaluation_sha256: hashes,
        deterministic_generation_replay: true,
        uncached_metric_parity: true,
        scq_execution: scq_execution_provenance(manifest, Some(dataset), workers),
        development_only: true,
        confirmatory_valid: false,
    })
}
