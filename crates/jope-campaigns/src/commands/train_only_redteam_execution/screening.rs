use super::*;

pub(super) fn build_screening(
    root: &Path,
    manifest: &TrainOnlyRedteamManifest,
    dataset: &DatasetInput,
    data: &RegressionData,
    bank: &FoldBankReceipt,
    workers: usize,
) -> Result<(ScreeningReceipt, String, PathBuf)> {
    let path = root
        .join("datasets")
        .join(&dataset.dataset_hash)
        .join("smart")
        .join("screening.v1.json");
    if path.exists() {
        let (screening, digest): (ScreeningReceipt, _) = read_receipt(&path)?;
        return Ok((screening, digest, path));
    }
    let (exact, jittered, transfer) = load_smart_candidates(manifest, dataset)?;
    let first = &bank.search[..1];
    let mut exact_first = evaluate_candidates(
        root,
        manifest,
        dataset,
        data,
        "search-screen-repeat-a",
        first,
        &exact,
        workers,
    )?;
    let mut jittered_first = evaluate_candidates(
        root,
        manifest,
        dataset,
        data,
        "search-screen-repeat-a",
        first,
        &jittered,
        workers,
    )?;
    let mut transfer_first = evaluate_candidates(
        root,
        manifest,
        dataset,
        data,
        "search-screen-repeat-a",
        first,
        &transfer,
        workers,
    )?;
    exact_first.truncate(80);
    jittered_first.truncate(60);
    transfer_first.truncate(100);
    let advanced_exact = evaluate_candidates(
        root,
        manifest,
        dataset,
        data,
        "search",
        &bank.search,
        &exact_first
            .iter()
            .map(|row| row.candidate.clone())
            .collect::<Vec<_>>(),
        workers,
    )?;
    let advanced_jittered = evaluate_candidates(
        root,
        manifest,
        dataset,
        data,
        "search",
        &bank.search,
        &jittered_first
            .iter()
            .map(|row| row.candidate.clone())
            .collect::<Vec<_>>(),
        workers,
    )?;
    let advanced_transfer = evaluate_candidates(
        root,
        manifest,
        dataset,
        data,
        "search",
        &bank.search,
        &transfer_first
            .iter()
            .map(|row| row.candidate.clone())
            .collect::<Vec<_>>(),
        workers,
    )?;
    let mut dl_ranked = advanced_exact
        .iter()
        .chain(&advanced_jittered)
        .cloned()
        .collect::<Vec<_>>();
    dl_ranked.sort_by(rank_loss);
    let mut dl_head_counts = HashMap::new();
    let mut transfer_head_counts = HashMap::new();
    let mut mixed_head_counts = HashMap::new();
    let head_family_cap = manifest
        .config
        .smart_v3
        .progressive_screen
        .max_per_terminal_head_family;
    let dl_population = select_with_head_cap(
        &dl_ranked,
        ISLAND_SIZE,
        &mut dl_head_counts,
        head_family_cap,
    )?;
    let transfer_population = select_with_head_cap(
        &advanced_transfer,
        ISLAND_SIZE,
        &mut transfer_head_counts,
        head_family_cap,
    )?;

    let mut mixed_candidates = Vec::new();
    let mut mixed_semantics = HashSet::new();
    for slot in 0..240 {
        let left = &dl_ranked[slot % dl_ranked.len()].candidate;
        let right = &advanced_transfer[(slot * 17 + 3) % advanced_transfer.len()].candidate;
        for retry in 0..100 {
            let seed = mixed_seed(&manifest.run_id, &dataset.dataset_hash, slot, retry);
            let Ok(genome) = crossover_canonical_pipelines(
                &left.canonical_genome,
                &right.canonical_genome,
                seed,
                SMART_MAX_TRANSFORMS,
            ) else {
                continue;
            };
            let mut roots = left.roots.clone();
            roots.extend(right.roots.iter().cloned());
            let mut origins = left.proposal_origins.clone();
            origins.extend(right.proposal_origins.iter().cloned());
            let mut decoder_seeds = left.decoder_seeds.clone();
            decoder_seeds.extend(right.decoder_seeds.iter().copied());
            let Ok(candidate) = candidate_from_parts(
                genome,
                roots,
                origins,
                decoder_seeds,
                json!({
                    "operation": "deterministic_dl_transfer_crossover",
                    "parents": [left.semantic_sha256, right.semantic_sha256],
                    "seed": seed,
                    "slot": slot,
                    "retry": retry,
                }),
                SMART_MAX_TRANSFORMS,
            ) else {
                continue;
            };
            if mixed_semantics.insert(candidate.semantic_sha256.clone()) {
                mixed_candidates.push(candidate);
            }
            break;
        }
    }
    if mixed_candidates.len() < 30 {
        bail!("cannot generate 30 unique deterministic mixed candidates");
    }
    let mixed_ranked = evaluate_candidates(
        root,
        manifest,
        dataset,
        data,
        "search",
        &bank.search,
        &mixed_candidates,
        workers,
    )?;
    let mixed_population = select_with_head_cap(
        &mixed_ranked,
        ISLAND_SIZE,
        &mut mixed_head_counts,
        head_family_cap,
    )?;
    let initial_populations = BTreeMap::from([
        ("dl".to_string(), dl_population),
        ("transfer".to_string(), transfer_population),
        ("mixed".to_string(), mixed_population),
    ]);
    validate_initial_populations(
        &initial_populations,
        ISLAND_SIZE,
        &["dl", "transfer", "mixed"],
        head_family_cap,
    )?;
    let screening = ScreeningReceipt {
        schema: "TrainOnlyRedteamSmartScreening.v1".to_string(),
        run_id: manifest.run_id.clone(),
        dataset_hash: dataset.dataset_hash.clone(),
        exact_candidates: exact.len(),
        jittered_candidates: jittered.len(),
        transfer_candidates: transfer.len(),
        advanced_exact,
        advanced_jittered,
        advanced_transfer,
        initial_populations,
        development_only: true,
        confirmatory_valid: false,
    };
    let digest = write_receipt(&path, &screening)?;
    Ok((screening, digest, path))
}
