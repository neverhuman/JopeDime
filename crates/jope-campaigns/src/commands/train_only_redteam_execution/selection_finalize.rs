use super::selection_math::{build_ensemble, ensemble_nrmse, select_one_se, selector_evaluations};
use super::*;

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

pub(super) fn finalize_smart_only_selector(
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

pub(crate) fn finalize_selector(
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
