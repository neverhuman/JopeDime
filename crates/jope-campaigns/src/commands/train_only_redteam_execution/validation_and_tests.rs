#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct DatasetValidationReplay {
    pub dataset_hash: String,
    pub g20_complete_sha256: String,
    pub smart_complete_sha256: String,
    pub selector_sha256: String,
    pub evaluation_sha256: BTreeMap<String, String>,
    pub deterministic_generation_replay: bool,
    pub uncached_metric_parity: bool,
    #[serde(default)]
    pub scq_execution: Option<ScqExecutionProvenance>,
    pub development_only: bool,
    pub confirmatory_valid: bool,
}

fn uncached_evaluation(
    manifest: &TrainOnlyRedteamManifest,
    dataset: &DatasetInput,
    data: &RegressionData,
    bank_name: &str,
    repeats: &[FoldRepeat],
    candidate: &Candidate,
) -> Result<CandidateEvaluation> {
    let parsed = GenomeIr::parse(&candidate.canonical_genome)?;
    let sd = target_sd(&data.y)?;
    let mut repeat_squared_error = Vec::with_capacity(repeats.len());
    let mut repeat_rows = Vec::with_capacity(repeats.len());
    let mut repeat_predictions = Vec::with_capacity(repeats.len());
    let mut fold_nrmse = Vec::with_capacity(repeats.len());
    for repeat in repeats {
        let mut oof = vec![f64::NAN; data.y.len()];
        let mut fold_scores = Vec::with_capacity(FOLDS);
        for validation in &repeat.validation_indices {
            let training = complement(data.y.len(), validation)?;
            let (train_x, train_y) = slice_rows(data, &training);
            let (validation_x, validation_y) = slice_rows(data, validation);
            let model = fit(&parsed, &train_x, &train_y)?;
            let predictions = model.predict(&validation_x)?;
            fold_scores.push(
                (squared_error(&validation_y, &predictions)? / validation.len() as f64).sqrt() / sd,
            );
            for (row, prediction) in validation.iter().zip(predictions) {
                oof[*row] = prediction;
            }
        }
        repeat_squared_error.push(squared_error(&data.y, &oof)?);
        repeat_rows.push(data.y.len());
        repeat_predictions.push(oof);
        fold_nrmse.push(fold_scores);
    }
    let score = robust_cv_score(&repeat_squared_error, &repeat_rows, fold_nrmse, sd)?;
    Ok(CandidateEvaluation {
        schema: "TrainOnlyRedteamCandidateEvaluation.v1".to_string(),
        run_id: manifest.run_id.clone(),
        dataset_hash: dataset.dataset_hash.clone(),
        train_sha256: dataset.train_sha256.clone(),
        bank: bank_name.to_string(),
        repeat_sha256: repeats
            .iter()
            .map(|repeat| repeat.repeat_sha256.clone())
            .collect(),
        candidate: candidate.clone(),
        target_sd: sd,
        repeat_squared_error,
        repeat_rows,
        repeat_predictions,
        score,
        development_only: true,
        confirmatory_valid: false,
    })
}

fn evaluations_match(left: &CandidateEvaluation, right: &CandidateEvaluation) -> bool {
    left.schema == right.schema
        && left.run_id == right.run_id
        && left.dataset_hash == right.dataset_hash
        && left.train_sha256 == right.train_sha256
        && left.bank == right.bank
        && left.repeat_sha256 == right.repeat_sha256
        && left.candidate == right.candidate
        && left.target_sd == right.target_sd
        && left.repeat_squared_error == right.repeat_squared_error
        && left.repeat_rows == right.repeat_rows
        && left.repeat_predictions == right.repeat_predictions
        && left.score.repeat_nrmse == right.score.repeat_nrmse
        && left.score.fold_nrmse == right.score.fold_nrmse
        && left.score.robust_loss == right.score.robust_loss
        && left.score.worst_fold_nrmse == right.score.worst_fold_nrmse
        && left.development_only == right.development_only
        && left.confirmatory_valid == right.confirmatory_valid
}

// Keep ranked receipt identity and the independent validation inputs explicit.
#[allow(clippy::too_many_arguments)]
fn validate_ranked_rows(
    rows: &[RankedCandidate],
    manifest: &TrainOnlyRedteamManifest,
    dataset: &DatasetInput,
    data: &RegressionData,
    bank_name: &str,
    repeats: &[FoldRepeat],
    uncached: bool,
    hashes: &mut BTreeMap<String, String>,
) -> Result<()> {
    let mut semantics = HashSet::new();
    for row in rows {
        if !semantics.insert(row.candidate.semantic_sha256.clone()) {
            bail!("ranked candidate set contains a duplicate");
        }
        let cached = load_ranked_evaluation(row)?;
        if cached.bank != bank_name {
            bail!("ranked candidate uses the wrong fold bank");
        }
        if uncached {
            let replay =
                uncached_evaluation(manifest, dataset, data, bank_name, repeats, &row.candidate)?;
            if !evaluations_match(&cached, &replay) {
                bail!("cached/uncached evaluation parity failure");
            }
        }
        hashes.insert(
            format!("{bank_name}:{}", row.candidate.semantic_sha256),
            row.evaluation_sha256.clone(),
        );
    }
    Ok(())
}

pub(super) fn validate_dataset_selection(
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
    let (parent_archive, _) = load_parent_archive(dataset)?;
    let mut history = parent_archive
        .iter()
        .map(|candidate| candidate.semantic_sha256.clone())
        .collect::<HashSet<_>>();
    let parent_checkpoint: Value = read_receipt::<Value>(&dataset.parent_generation_paths[9])?.0;
    let parent_ranked = parent_checkpoint
        .get("population")
        .and_then(Value::as_array)
        .context("parent generation nine lacks population")?
        .iter()
        .map(|value| {
            Ok(LegacyScoredCandidate {
                candidate: candidate_from_parent(value)?,
                split_r2: value
                    .get("split_r2")
                    .and_then(Value::as_array)
                    .context("parent score lacks split_r2")?
                    .iter()
                    .map(|value| value.as_f64().context("invalid parent split score"))
                    .collect::<Result<Vec<_>>>()?,
                robust_fitness: value.get("robust_fitness").and_then(Value::as_f64),
                worst_split_r2: value.get("worst_split_r2").and_then(Value::as_f64),
                failure_reason: value
                    .get("failure_reason")
                    .and_then(Value::as_str)
                    .map(str::to_string),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut expected_population = legacy_refill(
        &manifest.parent.run_id,
        &dataset.dataset_hash,
        10,
        &parent_ranked,
        &mut history,
    )?;
    let mut previous = None;
    for (offset, generation) in G20_GENERATIONS.enumerate() {
        let path = root
            .join("datasets")
            .join(&dataset.dataset_hash)
            .join("g20")
            .join(format!("generation-{generation:03}.v1.json"));
        let (checkpoint, digest): (G20GenerationReceipt, _) = read_receipt(&path)?;
        if digest != g20.extension_generation_sha256[offset]
            || checkpoint.generation != generation
            || checkpoint.previous_generation_sha256 != previous
            || checkpoint.parent_generation_nine_sha256 != dataset.parent_generation_sha256[9]
            || !checkpoint.development_only
            || checkpoint.confirmatory_valid
            || checkpoint.population.len() != 60
            || checkpoint.ranked_semantic_sha256
                != checkpoint
                    .population
                    .iter()
                    .map(|row| row.candidate.semantic_sha256.clone())
                    .collect::<Vec<_>>()
            || checkpoint.champion_semantic_sha256
                != checkpoint.population[0].candidate.semantic_sha256
            || expected_population
                .iter()
                .map(|candidate| &candidate.semantic_sha256)
                .collect::<Vec<_>>()
                != checkpoint
                    .population
                    .iter()
                    .map(|row| &row.candidate.semantic_sha256)
                    .collect::<Vec<_>>()
        {
            bail!("G20 deterministic checkpoint replay drift");
        }
        let rescored = score_legacy_population(
            &data,
            &dataset.dataset_hash,
            generation,
            &expected_population,
            workers,
        )?;
        if receipt_digest(&rescored)? != receipt_digest(&checkpoint.population)? {
            bail!("G20 uncached fitness replay drift");
        }
        let expected_next = if generation < 19 {
            legacy_refill(
                &manifest.parent.run_id,
                &dataset.dataset_hash,
                generation + 1,
                &rescored,
                &mut history,
            )?
        } else {
            Vec::new()
        };
        if receipt_digest(&expected_next)? != receipt_digest(&checkpoint.next_population)? {
            bail!("G20 deterministic offspring replay drift");
        }
        expected_population = expected_next;
        previous = Some(digest);
    }

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
    validate_initial_populations(
        &screening.initial_populations,
        ISLAND_SIZE,
        &["dl", "transfer", "mixed"],
        manifest
            .config
            .smart_v3
            .progressive_screen
            .max_per_terminal_head_family,
    )?;
    let mut histories = screening
        .initial_populations
        .iter()
        .map(|(island, rows)| {
            (
                island.clone(),
                rows.iter()
                    .map(|candidate| candidate.semantic_sha256.clone())
                    .collect::<HashSet<_>>(),
            )
        })
        .collect::<HashMap<_, _>>();
    let mut populations = screening.initial_populations.clone();
    let mut previous = None;
    let mut aggregate_counts = BTreeMap::new();
    let mut structural_niches = HashSet::new();
    let mut previous_best: Option<f64> = None;
    let mut stagnant = 0usize;
    for generation in 0..smart.generations_completed {
        let path = root
            .join("datasets")
            .join(&dataset.dataset_hash)
            .join("smart")
            .join(format!("generation-{generation:03}.v1.json"));
        let (checkpoint, digest): (SmartGenerationReceipt, _) = read_receipt(&path)?;
        if digest != smart.generation_sha256[generation]
            || checkpoint.generation != generation
            || checkpoint.previous_generation_sha256 != previous
            || !checkpoint.development_only
            || checkpoint.confirmatory_valid
            || checkpoint.populations.len() != ISLANDS
            || checkpoint
                .populations
                .values()
                .any(|rows| rows.len() != ISLAND_SIZE)
            || populations.iter().any(|(island, expected)| {
                expected
                    .iter()
                    .map(|candidate| &candidate.semantic_sha256)
                    .collect::<Vec<_>>()
                    != checkpoint.populations[island]
                        .iter()
                        .map(|row| &row.candidate.semantic_sha256)
                        .collect::<Vec<_>>()
            })
        {
            bail!("Smart-v3 deterministic generation replay drift");
        }
        for island in ["dl", "transfer", "mixed"] {
            let order = nsga_order(&checkpoint.populations[island]);
            let archive_path = root
                .join("datasets")
                .join(&dataset.dataset_hash)
                .join("smart")
                .join("archive")
                .join(format!("{island}-generation-{generation:03}.v1.json"));
            let (archive, archive_digest): (IslandArchiveReceipt, _) = read_receipt(&archive_path)?;
            if checkpoint.archive_sha256.get(island) != Some(&archive_digest)
                || !archive.development_only
                || archive.confirmatory_valid
                || archive.candidates.len() != 3
                || archive
                    .candidates
                    .iter()
                    .map(|row| &row.candidate.semantic_sha256)
                    .collect::<Vec<_>>()
                    != order[..3]
                        .iter()
                        .map(|index| {
                            &checkpoint.populations[island][*index]
                                .candidate
                                .semantic_sha256
                        })
                        .collect::<Vec<_>>()
            {
                bail!("Smart-v3 immutable archive drift");
            }
        }
        let best_loss = checkpoint
            .populations
            .values()
            .flatten()
            .map(|candidate| candidate.robust_loss)
            .min_by(f64::total_cmp)
            .context("Smart-v3 checkpoint has no scored candidates")?;
        let mut new_niche = false;
        for candidates in checkpoint.populations.values() {
            for (index, candidate) in candidates.iter().enumerate() {
                if is_nondominated(candidates, index) {
                    new_niche |= structural_niches.insert(structural_niche(&candidate.candidate)?);
                }
            }
        }
        let improved = previous_best.is_none_or(|previous| {
            (previous - best_loss) / previous.abs().max(1e-15)
                >= manifest.config.smart_v3.minimum_relative_improvement
        });
        if improved || new_niche {
            stagnant = 0;
        } else {
            stagnant += 1;
        }
        let can_stop = generation + 1 >= manifest.config.smart_v3.minimum_generations
            && stagnant >= manifest.config.smart_v3.stagnation_generations;
        let last = generation + 1 == manifest.config.smart_v3.maximum_generations;
        if checkpoint.best_robust_loss != best_loss
            || checkpoint.new_nondominated_structural_niche != new_niche
            || checkpoint.consecutive_stagnant_generations != stagnant
            || checkpoint.stopped != (can_stop || last)
        {
            bail!("Smart-v3 stopping-rule replay drift");
        }
        let mut expected_mutations = BTreeMap::new();
        let expected_next = if checkpoint.stopped {
            BTreeMap::new()
        } else {
            ["dl", "transfer", "mixed"]
                .into_iter()
                .map(|island| {
                    let (next, counts) = smart_refill(
                        manifest,
                        &dataset.dataset_hash,
                        island,
                        generation + 1,
                        &checkpoint.populations[island],
                        (island == "mixed").then(|| {
                            (
                                checkpoint.populations["dl"].as_slice(),
                                checkpoint.populations["transfer"].as_slice(),
                            )
                        }),
                        histories.get_mut(island).unwrap(),
                    )?;
                    for (operator, count) in counts {
                        *expected_mutations.entry(operator).or_default() += count;
                    }
                    Ok((island.to_string(), next))
                })
                .collect::<Result<BTreeMap<_, _>>>()?
        };
        if receipt_digest(&expected_next)? != receipt_digest(&checkpoint.next_populations)?
            || expected_mutations != checkpoint.mutation_counts
        {
            bail!("Smart-v3 deterministic offspring replay drift");
        }
        for (operator, count) in checkpoint.mutation_counts {
            *aggregate_counts.entry(operator).or_default() += count;
        }
        populations = checkpoint.next_populations;
        previous = Some(digest);
        previous_best = Some(best_loss);
    }
    if aggregate_counts != smart.mutation_counts {
        bail!("Smart-v3 mutation ledger drift");
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(id: char, loss: f64, complexity: usize) -> RankedCandidate {
        RankedCandidate {
            candidate: Candidate {
                canonical_genome: "(pipe X (ridge))".to_string(),
                semantic_sha256: id.to_string().repeat(64),
                complexity,
                head_family: "ridge".to_string(),
                roots: BTreeSet::from(["dl".to_string()]),
                proposal_origins: BTreeSet::new(),
                decoder_seeds: BTreeSet::new(),
                ancestry: Value::Null,
            },
            evaluation_path: PathBuf::from("unused"),
            evaluation_sha256: "0".repeat(64),
            robust_loss: loss,
            worst_fold_nrmse: loss,
        }
    }

    fn initial_candidate(island: &str, slot: usize, head_family: &str) -> Candidate {
        Candidate {
            canonical_genome: format!("(pipe {island} {slot} ({head_family}))"),
            semantic_sha256: format!("{island}-{slot}"),
            complexity: 1,
            head_family: head_family.to_string(),
            roots: BTreeSet::from([island.to_string()]),
            proposal_origins: BTreeSet::new(),
            decoder_seeds: BTreeSet::new(),
            ancestry: Value::Null,
        }
    }

    #[test]
    fn mutation_arm_frequency_contract_is_exact_over_full_cycles() {
        let mut counts = BTreeMap::new();
        for draw in 0..10_000 {
            *counts.entry(mutation_arm(draw).0).or_insert(0usize) += 1;
        }
        assert_eq!(counts["parameter_jitter"], 3_000);
        assert_eq!(counts["insertion"], 1_000);
        assert_eq!(counts["deletion"], 2_000);
        assert_eq!(counts["adjacent_swap"], 1_000);
        assert_eq!(counts["head_replacement"], 1_500);
        assert_eq!(counts["simplification"], 1_500);
    }

    #[test]
    fn nsga_order_preserves_loss_complexity_nondomination() {
        let population = vec![
            candidate('a', 0.10, 8),
            candidate('b', 0.11, 2),
            candidate('c', 0.12, 8),
            candidate('d', 0.13, 9),
        ];
        let order = nsga_order(&population);
        let first_front = &order[..2];
        assert!(first_front.contains(&0));
        assert!(first_front.contains(&1));
        assert!(
            order.iter().position(|index| *index == 2).unwrap()
                < order.iter().position(|index| *index == 3).unwrap()
        );
    }

    #[test]
    fn convex_alpha_is_bounded_and_improves_the_fit() {
        let truth = [0.0, 1.0, 2.0, 3.0];
        let current = [0.0, 0.5, 1.0, 1.5];
        let candidate = [0.0, 1.0, 2.0, 3.0];
        let alpha = optimal_convex_alpha(&truth, &current, &candidate).unwrap();
        assert_eq!(alpha, 1.0);
        let blended = add_prediction(&current, &candidate, alpha);
        assert_eq!(blended, truth);
    }

    #[test]
    fn smart_only_selector_freezes_without_a_g20_receipt() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path();
        let dataset_hash = DATASETS[0];
        let dataset_root = root.join("datasets").join(dataset_hash);
        fs::create_dir_all(&dataset_root).unwrap();
        let train_path = root.join("train.csv");
        let csv = (0..30)
            .map(|row| format!("{},{}", row as f64, 2.0 * row as f64 + 1.0))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&train_path, format!("{csv}\n")).unwrap();
        let train_sha256 = sha256_file(&train_path).unwrap();
        let data = read_regression_csv(&train_path).unwrap();
        let config = TrainOnlyRedteamConfig::sealed(SMART_V3_CPU_RUN_ID, DEFAULT_PARENT_RUN, false);
        let bank = build_fold_bank(
            SMART_V3_CPU_RUN_ID,
            dataset_hash,
            &train_sha256,
            &data.y,
            &config,
        )
        .unwrap();
        let fold_bank_path = dataset_root.join("fold-banks.v1.json");
        let fold_bank_sha256 = write_receipt(&fold_bank_path, &bank).unwrap();
        let canonical_genome = "(pipe X (ridge))".to_string();
        let selected = Candidate {
            semantic_sha256: semantic_identity(&canonical_genome),
            canonical_genome,
            complexity: 1,
            head_family: "ridge".to_string(),
            roots: BTreeSet::from(["dl".to_string()]),
            proposal_origins: BTreeSet::new(),
            decoder_seeds: BTreeSet::new(),
            ancestry: Value::Null,
        };
        let ranked = RankedCandidate {
            candidate: selected,
            evaluation_path: PathBuf::from("unused"),
            evaluation_sha256: "0".repeat(64),
            robust_loss: 1.0,
            worst_fold_nrmse: 1.0,
        };
        let smart = SmartComplete {
            schema: "TrainOnlyRedteamSmartComplete.v1".to_string(),
            run_id: SMART_V3_CPU_RUN_ID.to_string(),
            dataset_hash: dataset_hash.to_string(),
            generations_completed: 10,
            stopped_early: false,
            screening_sha256: "1".repeat(64),
            generation_sha256: vec!["2".repeat(64); 10],
            archive_top20_by_island: ["dl", "transfer", "mixed"]
                .into_iter()
                .map(|island| (island.to_string(), vec![ranked.clone()]))
                .collect(),
            mutation_counts: BTreeMap::new(),
            development_only: true,
            confirmatory_valid: false,
        };
        write_receipt(&dataset_root.join("smart-v3-complete.v1.json"), &smart).unwrap();
        assert!(!dataset_root.join("g20-extension-complete.v1.json").exists());
        let dataset = DatasetInput {
            dataset_hash: dataset_hash.to_string(),
            train_path,
            train_sha256,
            rows: data.y.len(),
            features: data.features,
            context_sha256: "3".repeat(64),
            transfer_ranking_path: root.join("unused-transfer.json"),
            transfer_ranking_sha256: "4".repeat(64),
            parent_generation_paths: Vec::new(),
            parent_generation_sha256: Vec::new(),
            fold_bank_path,
            fold_bank_sha256,
        };
        let manifest = TrainOnlyRedteamManifest {
            schema: "TrainOnlyRedteamManifest.v1".to_string(),
            run_id: SMART_V3_CPU_RUN_ID.to_string(),
            config,
            config_sha256: "5".repeat(64),
            development_only: true,
            confirmatory_valid: false,
            test_reuse_disclosure: DISCLOSURE.to_string(),
            parent: ParentLink {
                run_id: DEFAULT_PARENT_RUN.to_string(),
                manifest_path: root.join("unused-parent.json"),
                manifest_sha256: "6".repeat(64),
                validation_path: root.join("unused-validation.json"),
                validation_sha256: "7".repeat(64),
            },
            proposal_bundle: ProposalBundleLink {
                path: root.join("unused-proposals.jsonl"),
                sha256: "8".repeat(64),
                exact_rows_per_dataset: 256,
                jittered_rows_per_dataset: 192,
                scaling_statistic_sha256: "9".repeat(64),
            },
            scaling_statistic_path: root.join("unused-scaling.json"),
            scaling_statistic_sha256: "9".repeat(64),
            datasets: vec![dataset.clone()],
            firewall: SelectionFirewall {
                selection_workers_receive_training_only: true,
                contains_test_paths: false,
                contains_baseline_paths: false,
                test_paths_constructed: false,
                baseline_paths_constructed: false,
            },
        };
        let (path, _) = finalize_smart_only_selector(root, &manifest, &dataset, 1)
            .unwrap()
            .unwrap();
        let (receipt, _): (SmartV3SelectorComplete, _) = read_receipt(&path).unwrap();
        assert_eq!(receipt.schema, "TrainOnlyRedteamSmartV3SelectorComplete.v1");
        assert_eq!(receipt.final_variant.candidates.len(), 1);
        assert!(receipt.selection_frozen_before_test_access);
    }

    #[test]
    fn initialization_head_cap_is_applied_per_island() {
        let populations = ["dl", "transfer", "mixed"]
            .into_iter()
            .map(|island| {
                let rows = (0..30)
                    .map(|slot| {
                        initial_candidate(island, slot, ["ridge", "elastic", "hist"][slot / 10])
                    })
                    .collect::<Vec<_>>();
                (island.to_string(), rows)
            })
            .collect::<BTreeMap<_, _>>();

        validate_initial_populations(&populations, ISLAND_SIZE, &["dl", "transfer", "mixed"], 10)
            .unwrap();
    }

    #[test]
    fn global_head_cap_is_impossible_for_the_screened_pilot_pool() {
        let populations = ["dl", "transfer", "mixed"]
            .into_iter()
            .map(|island| {
                let rows = (0..30)
                    .map(|slot| {
                        initial_candidate(island, slot, ["ridge", "elastic", "hist"][slot / 10])
                    })
                    .collect::<Vec<_>>();
                (island.to_string(), rows)
            })
            .collect::<BTreeMap<_, _>>();
        let global_head_counts = populations.values().flatten().fold(
            HashMap::<&str, usize>::new(),
            |mut counts, candidate| {
                *counts.entry(candidate.head_family.as_str()).or_default() += 1;
                counts
            },
        );

        assert_eq!(global_head_counts["ridge"], 30);
        assert!(global_head_counts.values().any(|count| *count > 10));
        assert!(validate_initial_populations(
            &populations,
            ISLAND_SIZE,
            &["dl", "transfer", "mixed"],
            10,
        )
        .is_ok());
    }
}
