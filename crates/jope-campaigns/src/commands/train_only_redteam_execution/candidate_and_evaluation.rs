#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct Candidate {
    pub canonical_genome: String,
    pub semantic_sha256: String,
    pub complexity: usize,
    pub head_family: String,
    pub roots: BTreeSet<String>,
    pub proposal_origins: BTreeSet<String>,
    pub decoder_seeds: BTreeSet<u64>,
    pub ancestry: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct CandidateEvaluation {
    pub schema: String,
    pub run_id: String,
    pub dataset_hash: String,
    pub train_sha256: String,
    pub bank: String,
    pub repeat_sha256: Vec<String>,
    pub candidate: Candidate,
    pub target_sd: f64,
    pub repeat_squared_error: Vec<f64>,
    pub repeat_rows: Vec<usize>,
    pub repeat_predictions: Vec<Vec<f64>>,
    pub score: CvScore,
    pub development_only: bool,
    pub confirmatory_valid: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct LegacyScoredCandidate {
    candidate: Candidate,
    split_r2: Vec<f64>,
    robust_fitness: Option<f64>,
    worst_split_r2: Option<f64>,
    failure_reason: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct G20GenerationReceipt {
    schema: String,
    run_id: String,
    dataset_hash: String,
    generation: usize,
    parent_generation_nine_sha256: String,
    previous_generation_sha256: Option<String>,
    population: Vec<LegacyScoredCandidate>,
    ranked_semantic_sha256: Vec<String>,
    champion_semantic_sha256: String,
    next_population: Vec<Candidate>,
    development_only: bool,
    confirmatory_valid: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct G20Complete {
    pub schema: String,
    pub run_id: String,
    pub dataset_hash: String,
    pub parent_generation_sha256: Vec<String>,
    pub extension_generation_sha256: Vec<String>,
    pub raw_generation_19: Candidate,
    pub top_archive_search_oof: Vec<RankedCandidate>,
    pub archive_unique_candidates: usize,
    pub development_only: bool,
    pub confirmatory_valid: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct RankedCandidate {
    pub candidate: Candidate,
    pub evaluation_path: PathBuf,
    pub evaluation_sha256: String,
    pub robust_loss: f64,
    pub worst_fold_nrmse: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct IslandArchiveReceipt {
    schema: String,
    run_id: String,
    dataset_hash: String,
    island: String,
    generation: usize,
    candidates: Vec<RankedCandidate>,
    development_only: bool,
    confirmatory_valid: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct SmartGenerationReceipt {
    schema: String,
    run_id: String,
    dataset_hash: String,
    generation: usize,
    previous_generation_sha256: Option<String>,
    populations: BTreeMap<String, Vec<RankedCandidate>>,
    next_populations: BTreeMap<String, Vec<Candidate>>,
    archive_sha256: BTreeMap<String, String>,
    best_robust_loss: f64,
    new_nondominated_structural_niche: bool,
    consecutive_stagnant_generations: usize,
    mutation_counts: BTreeMap<String, usize>,
    stopped: bool,
    development_only: bool,
    confirmatory_valid: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct SmartComplete {
    pub schema: String,
    pub run_id: String,
    pub dataset_hash: String,
    pub generations_completed: usize,
    pub stopped_early: bool,
    pub screening_sha256: String,
    pub generation_sha256: Vec<String>,
    pub archive_top20_by_island: BTreeMap<String, Vec<RankedCandidate>>,
    pub mutation_counts: BTreeMap<String, usize>,
    pub development_only: bool,
    pub confirmatory_valid: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct ScreeningReceipt {
    schema: String,
    run_id: String,
    dataset_hash: String,
    exact_candidates: usize,
    jittered_candidates: usize,
    transfer_candidates: usize,
    advanced_exact: Vec<RankedCandidate>,
    advanced_jittered: Vec<RankedCandidate>,
    advanced_transfer: Vec<RankedCandidate>,
    initial_populations: BTreeMap<String, Vec<Candidate>>,
    development_only: bool,
    confirmatory_valid: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub(super) struct FrozenVariant {
    pub name: String,
    pub candidates: Vec<Candidate>,
    pub weights: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct SelectorComplete {
    pub schema: String,
    pub run_id: String,
    pub dataset_hash: String,
    pub g20_complete_sha256: String,
    pub smart_complete_sha256: String,
    pub candidates: Vec<RankedCandidate>,
    pub selected_single: Candidate,
    pub smart_single: Candidate,
    pub no_jitter: Candidate,
    pub original_decoder_seed_only: Candidate,
    pub ensemble_candidates: Vec<Candidate>,
    pub ensemble_weights: Vec<f64>,
    pub audit_single_nrmse: f64,
    pub audit_ensemble_nrmse: f64,
    pub ensemble_accepted: bool,
    pub final_variant: FrozenVariant,
    pub selection_frozen_before_test_access: bool,
    #[serde(default)]
    pub scq_execution: Option<ScqExecutionProvenance>,
    pub development_only: bool,
    pub confirmatory_valid: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct SmartV3SelectorComplete {
    pub schema: String,
    pub run_id: String,
    pub dataset_hash: String,
    pub smart_complete_sha256: String,
    pub candidates: Vec<RankedCandidate>,
    pub selected_single: Candidate,
    pub ensemble_candidates: Vec<Candidate>,
    pub ensemble_weights: Vec<f64>,
    pub audit_single_nrmse: f64,
    pub audit_ensemble_nrmse: f64,
    pub ensemble_accepted: bool,
    pub final_variant: FrozenVariant,
    pub selection_frozen_before_test_access: bool,
    #[serde(default)]
    pub scq_execution: Option<ScqExecutionProvenance>,
    pub development_only: bool,
    pub confirmatory_valid: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct ArmComplete {
    schema: String,
    run_id: String,
    arm: String,
    manifest_sha256: String,
    dataset_receipt_sha256: BTreeMap<String, String>,
    #[serde(default)]
    scq_execution: Option<ScqExecutionProvenance>,
    development_only: bool,
    confirmatory_valid: bool,
}

fn semantic_identity(genome: &str) -> String {
    sha256_bytes(genome.as_bytes())
}

fn candidate_from_parts(
    canonical_genome: String,
    roots: BTreeSet<String>,
    proposal_origins: BTreeSet<String>,
    decoder_seeds: BTreeSet<u64>,
    ancestry: Value,
    max_transforms: usize,
) -> Result<Candidate> {
    let parsed = GenomeIr::parse(&canonical_genome)?;
    let GenomeIr::Pipeline(pipeline) = &parsed else {
        bail!("candidate is not a pipeline genome");
    };
    let canonical = parsed.canonical();
    if canonical != canonical_genome || pipeline.stages.is_empty() {
        bail!("candidate is not canonical");
    }
    if let Some((index, _)) = pipeline
        .stages
        .iter()
        .enumerate()
        .find(|(_, stage)| stage.op == "target")
    {
        if index != 0 {
            bail!("target transform must be the first stage");
        }
        if pipeline.stages[index + 1..]
            .iter()
            .any(|stage| stage.op == "target")
        {
            bail!("target transform must be the first stage");
        }
    }
    let complexity = pipeline.stages.len() - 1;
    if complexity > max_transforms {
        bail!("candidate exceeds the transform ceiling");
    }
    let head_family = pipeline.stages.last().unwrap().op.clone();
    if matches!(head_family.as_str(), "logit_ridge" | "logistic") {
        bail!("classifier-only candidate in regression study");
    }
    Ok(Candidate {
        semantic_sha256: semantic_identity(&canonical),
        canonical_genome: canonical,
        complexity,
        head_family,
        roots,
        proposal_origins,
        decoder_seeds,
        ancestry,
    })
}

pub(super) fn candidate_from_parent(value: &Value) -> Result<Candidate> {
    let member = value.get("member").unwrap_or(value);
    let genome = value_string(member, "/canonical_genome")?;
    let roots = member
        .pointer("/ancestry/roots")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(|value| format!("parent_v2:{value}"))
                .collect::<BTreeSet<_>>()
        })
        .unwrap_or_else(|| BTreeSet::from(["parent_v2".to_string()]));
    candidate_from_parts(
        genome,
        roots,
        BTreeSet::new(),
        BTreeSet::new(),
        json!({
            "operation": "parent_v2",
            "source_ancestry": member.get("ancestry").cloned().unwrap_or(Value::Null),
        }),
        12,
    )
}

pub(super) fn relocated_input_path(path: &Path) -> PathBuf {
    let Some(root) = std::env::var_os("REDTEAM_PATH_ROOT").map(PathBuf::from) else {
        return path.to_path_buf();
    };
    match path.strip_prefix("/") {
        Ok(relative) => root.join(relative),
        Err(_) => path.to_path_buf(),
    }
}

pub(super) fn validate_manifest(path: &Path) -> Result<(TrainOnlyRedteamManifest, String)> {
    let (mut manifest, digest): (TrainOnlyRedteamManifest, _) = read_receipt(path)?;
    if manifest.schema != "TrainOnlyRedteamManifest.v1"
        || manifest.run_id != manifest.config.run_id
        || !manifest.development_only
        || manifest.confirmatory_valid
        || manifest.test_reuse_disclosure != DISCLOSURE
        || !manifest.firewall.selection_workers_receive_training_only
        || manifest.firewall.contains_test_paths
        || manifest.firewall.contains_baseline_paths
        || manifest.firewall.test_paths_constructed
        || manifest.firewall.baseline_paths_constructed
    {
        bail!("red-team manifest contract drift");
    }
    manifest.config.validate()?;
    manifest.parent.manifest_path = relocated_input_path(&manifest.parent.manifest_path);
    manifest.parent.validation_path = relocated_input_path(&manifest.parent.validation_path);
    manifest.proposal_bundle.path = relocated_input_path(&manifest.proposal_bundle.path);
    manifest.scaling_statistic_path = relocated_input_path(&manifest.scaling_statistic_path);
    for dataset in &mut manifest.datasets {
        dataset.train_path = relocated_input_path(&dataset.train_path);
        dataset.fold_bank_path = relocated_input_path(&dataset.fold_bank_path);
        dataset.transfer_ranking_path = relocated_input_path(&dataset.transfer_ranking_path);
        dataset.parent_generation_paths = dataset
            .parent_generation_paths
            .iter()
            .map(|path| relocated_input_path(path))
            .collect();
    }
    let (_, config_digest): (TrainOnlyRedteamConfig, _) =
        read_receipt(&path.parent().unwrap().join("config.v1.json"))?;
    let (_, scaling_digest): (Value, _) = read_receipt(&manifest.scaling_statistic_path)?;
    if config_digest != manifest.config_sha256
        || sha256_file(&manifest.proposal_bundle.path)? != manifest.proposal_bundle.sha256
        || scaling_digest != manifest.scaling_statistic_sha256
    {
        bail!("red-team manifest input drift");
    }
    let (_, parent_digest): (Value, _) = read_receipt(&manifest.parent.manifest_path)?;
    let (_, parent_validation_digest): (Value, _) = read_receipt(&manifest.parent.validation_path)?;
    if parent_digest != manifest.parent.manifest_sha256
        || parent_validation_digest != manifest.parent.validation_sha256
    {
        bail!("red-team parent link drift");
    }
    for dataset in &manifest.datasets {
        if sha256_file(&dataset.train_path)? != dataset.train_sha256
            || dataset.parent_generation_paths.len() != 10
            || dataset.parent_generation_sha256.len() != 10
        {
            bail!("red-team dataset input drift: {}", dataset.dataset_hash);
        }
        let (bank, bank_digest): (FoldBankReceipt, _) = read_receipt(&dataset.fold_bank_path)?;
        if bank_digest != dataset.fold_bank_sha256 {
            bail!("fold-bank digest drift: {}", dataset.dataset_hash);
        }
        validate_fold_bank(&bank, &manifest.config)?;
        let (_, transfer_digest): (Value, _) = read_receipt(&dataset.transfer_ranking_path)?;
        if transfer_digest != dataset.transfer_ranking_sha256 {
            bail!("transfer-ranking digest drift: {}", dataset.dataset_hash);
        }
        for (generation, source_path) in dataset.parent_generation_paths.iter().enumerate() {
            let (_, source_digest): (Value, _) = read_receipt(source_path)?;
            if source_digest != dataset.parent_generation_sha256[generation] {
                bail!(
                    "parent generation digest drift: {} generation {}",
                    dataset.dataset_hash,
                    generation
                );
            }
        }
    }
    Ok((manifest, digest))
}

fn complement(rows: usize, validation: &[usize]) -> Result<Vec<usize>> {
    let mut present = vec![false; rows];
    for row in validation {
        if *row >= rows || present[*row] {
            bail!("fold validation indices repeat or exceed row count");
        }
        present[*row] = true;
    }
    Ok(present
        .into_iter()
        .enumerate()
        .filter_map(|(row, validation)| (!validation).then_some(row))
        .collect())
}

fn slice_rows(data: &RegressionData, indices: &[usize]) -> (Vec<Vec<f64>>, Vec<f64>) {
    (
        indices.iter().map(|row| data.x[*row].clone()).collect(),
        indices.iter().map(|row| data.y[*row]).collect(),
    )
}

fn squared_error(truth: &[f64], predictions: &[f64]) -> Result<f64> {
    if truth.len() != predictions.len()
        || truth.is_empty()
        || truth
            .iter()
            .chain(predictions)
            .any(|value| !value.is_finite())
    {
        bail!("squared error inputs are invalid");
    }
    Ok(truth
        .iter()
        .zip(predictions)
        .map(|(truth, prediction)| (truth - prediction).powi(2))
        .sum())
}

fn evaluation_path(
    root: &Path,
    dataset_hash: &str,
    bank: &str,
    candidate: &Candidate,
    repeats: &[FoldRepeat],
) -> Result<PathBuf> {
    let key = receipt_digest(&json!({
        "bank": bank,
        "candidate": candidate.semantic_sha256,
        "repeat_sha256": repeats.iter().map(|repeat| &repeat.repeat_sha256).collect::<Vec<_>>(),
        "metric": "pooled-oof-nrmse-v1",
    }))?;
    Ok(root
        .join("evaluations")
        .join(dataset_hash)
        .join(bank)
        .join(format!("{key}.v1.json")))
}

fn evaluate_candidate(
    root: &Path,
    manifest: &TrainOnlyRedteamManifest,
    dataset: &DatasetInput,
    data: &RegressionData,
    bank: &str,
    repeats: &[FoldRepeat],
    candidate: &Candidate,
) -> Result<(CandidateEvaluation, String, PathBuf)> {
    let path = evaluation_path(root, &dataset.dataset_hash, bank, candidate, repeats)?;
    if path.exists() {
        let (evaluation, digest): (CandidateEvaluation, _) = read_receipt(&path)?;
        if evaluation.schema != "TrainOnlyRedteamCandidateEvaluation.v1"
            || evaluation.run_id != manifest.run_id
            || evaluation.dataset_hash != dataset.dataset_hash
            || evaluation.train_sha256 != dataset.train_sha256
            || evaluation.bank != bank
            || evaluation.candidate.semantic_sha256 != candidate.semantic_sha256
            || !evaluation.development_only
            || evaluation.confirmatory_valid
            || evaluation.repeat_sha256
                != repeats
                    .iter()
                    .map(|repeat| repeat.repeat_sha256.clone())
                    .collect::<Vec<_>>()
        {
            bail!("candidate evaluation cache identity drift");
        }
        return Ok((evaluation, digest, path));
    }
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
            let fold_error = squared_error(&validation_y, &predictions)?;
            fold_scores.push((fold_error / validation.len() as f64).sqrt() / sd);
            for (row, prediction) in validation.iter().zip(predictions) {
                oof[*row] = prediction;
            }
        }
        let error = squared_error(&data.y, &oof)?;
        repeat_squared_error.push(error);
        repeat_rows.push(data.y.len());
        repeat_predictions.push(oof);
        fold_nrmse.push(fold_scores);
    }
    let score = robust_cv_score(&repeat_squared_error, &repeat_rows, fold_nrmse, sd)?;
    let evaluation = CandidateEvaluation {
        schema: "TrainOnlyRedteamCandidateEvaluation.v1".to_string(),
        run_id: manifest.run_id.clone(),
        dataset_hash: dataset.dataset_hash.clone(),
        train_sha256: dataset.train_sha256.clone(),
        bank: bank.to_string(),
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
    };
    let digest = write_receipt(&path, &evaluation)?;
    Ok((evaluation, digest, path))
}

// Keep study identity, candidate pool, input folds and execution budget explicit.
#[allow(clippy::too_many_arguments)]
fn evaluate_candidates(
    root: &Path,
    manifest: &TrainOnlyRedteamManifest,
    dataset: &DatasetInput,
    data: &RegressionData,
    bank: &str,
    repeats: &[FoldRepeat],
    candidates: &[Candidate],
    workers: usize,
) -> Result<Vec<RankedCandidate>> {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()?;
    let mut ranked = pool.install(|| {
        candidates
            .par_iter()
            .map(|candidate| {
                let (evaluation, digest, path) =
                    evaluate_candidate(root, manifest, dataset, data, bank, repeats, candidate)?;
                Ok(RankedCandidate {
                    candidate: candidate.clone(),
                    evaluation_path: absolute(&path)?,
                    evaluation_sha256: digest,
                    robust_loss: evaluation.score.robust_loss,
                    worst_fold_nrmse: evaluation.score.worst_fold_nrmse,
                })
            })
            .collect::<Result<Vec<_>>>()
    })?;
    ranked.sort_by(rank_loss);
    Ok(ranked)
}

fn rank_loss(left: &RankedCandidate, right: &RankedCandidate) -> Ordering {
    left.robust_loss
        .total_cmp(&right.robust_loss)
        .then_with(|| left.worst_fold_nrmse.total_cmp(&right.worst_fold_nrmse))
        .then_with(|| left.candidate.complexity.cmp(&right.candidate.complexity))
        .then_with(|| {
            left.candidate
                .semantic_sha256
                .cmp(&right.candidate.semantic_sha256)
        })
}

fn legacy_metrics(truth: &[f64], predictions: &[f64]) -> Result<(f64, f64)> {
    let error = squared_error(truth, predictions)?;
    let mean = truth.iter().sum::<f64>() / truth.len() as f64;
    let total = truth
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>();
    if total <= 0.0 {
        bail!("legacy split target variance is zero");
    }
    Ok(((error / truth.len() as f64).sqrt(), 1.0 - error / total))
}

fn legacy_robust_fitness(scores: &[f64]) -> Result<f64> {
    if scores.is_empty() || scores.iter().any(|value| !value.is_finite()) {
        bail!("legacy robust fitness needs finite scores");
    }
    let clipped = scores
        .iter()
        .map(|value| value.max(-1.0))
        .collect::<Vec<_>>();
    let mean = clipped.iter().sum::<f64>() / clipped.len() as f64;
    let sd = (clipped
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / clipped.len() as f64)
        .sqrt();
    Ok(mean - 0.25 * sd.min(1.0))
}

fn legacy_split(dataset_hash: &str, generation: usize, targets: &[f64]) -> Result<Vec<usize>> {
    let mut sorted = (0..targets.len()).collect::<Vec<_>>();
    sorted.sort_by(|left, right| {
        targets[*left]
            .total_cmp(&targets[*right])
            .then_with(|| left.cmp(right))
    });
    let mut deciles = vec![Vec::new(); 10];
    let rows = sorted.len();
    for (rank, row) in sorted.into_iter().enumerate() {
        deciles[(rank * 10 / rows).min(9)].push(row);
    }
    let mut seed_hash = Sha256::new();
    seed_hash.update(b"checkpoint-assisted-split-seed-v2\0");
    seed_hash.update(20_260_729_u64.to_le_bytes());
    seed_hash.update(dataset_hash.as_bytes());
    seed_hash.update((generation as u64).to_le_bytes());
    let seed = u64::from_le_bytes(seed_hash.finalize()[..8].try_into().unwrap());
    let validation_total = targets.len().div_ceil(5);
    let mut ordered = Vec::new();
    let mut base_total = 0;
    for rows in deciles {
        let mut rows = rows;
        rows.sort_by_key(|row| {
            let mut hash = Sha256::new();
            hash.update(b"checkpoint-assisted-split-order-v2\0");
            hash.update(seed.to_le_bytes());
            hash.update((*row as u64).to_le_bytes());
            hash.finalize().to_vec()
        });
        let base = rows.len() / 5;
        let remainder = rows.len() % 5;
        base_total += base;
        ordered.push((rows, base, remainder));
    }
    let mut quotas = ordered.iter().map(|entry| entry.1).collect::<Vec<_>>();
    let mut extras = validation_total - base_total;
    let mut remainder_order = (0..10).collect::<Vec<_>>();
    remainder_order.sort_by(|left, right| {
        ordered[*right]
            .2
            .cmp(&ordered[*left].2)
            .then_with(|| left.cmp(right))
    });
    for decile in remainder_order {
        if extras == 0 {
            break;
        }
        if quotas[decile] < ordered[decile].0.len() {
            quotas[decile] += 1;
            extras -= 1;
        }
    }
    if extras != 0 {
        bail!("legacy validation quota allocation failed");
    }
    let mut validation = Vec::with_capacity(validation_total);
    for (decile, (rows, _, _)) in ordered.iter().enumerate() {
        validation.extend_from_slice(&rows[..quotas[decile]]);
    }
    validation.sort_unstable();
    Ok(validation)
}

fn score_legacy_population(
    data: &RegressionData,
    dataset_hash: &str,
    generation: usize,
    population: &[Candidate],
    workers: usize,
) -> Result<Vec<LegacyScoredCandidate>> {
    let validations = (0..=generation)
        .map(|split| legacy_split(dataset_hash, split, &data.y))
        .collect::<Result<Vec<_>>>()?;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .build()?;
    let mut scored = pool.install(|| {
        population
            .par_iter()
            .map(|candidate| {
                let evaluated = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                    || -> Result<Vec<f64>> {
                        let parsed = GenomeIr::parse(&candidate.canonical_genome)?;
                        let mut scores = Vec::with_capacity(validations.len());
                        for validation in &validations {
                            let training = complement(data.y.len(), validation)?;
                            let (train_x, train_y) = slice_rows(data, &training);
                            let (validation_x, validation_y) = slice_rows(data, validation);
                            let model = fit(&parsed, &train_x, &train_y)?;
                            let predictions = model.predict(&validation_x)?;
                            scores.push(legacy_metrics(&validation_y, &predictions)?.1);
                        }
                        Ok(scores)
                    },
                ));
                Ok(match evaluated {
                    Ok(Ok(split_r2)) => LegacyScoredCandidate {
                        robust_fitness: Some(legacy_robust_fitness(&split_r2)?),
                        worst_split_r2: split_r2.iter().copied().min_by(f64::total_cmp),
                        candidate: candidate.clone(),
                        split_r2,
                        failure_reason: None,
                    },
                    Ok(Err(error)) => LegacyScoredCandidate {
                        candidate: candidate.clone(),
                        split_r2: Vec::new(),
                        robust_fitness: None,
                        worst_split_r2: None,
                        failure_reason: Some(format!("{error:#}")),
                    },
                    Err(_) => LegacyScoredCandidate {
                        candidate: candidate.clone(),
                        split_r2: Vec::new(),
                        robust_fitness: None,
                        worst_split_r2: None,
                        failure_reason: Some("panic during legacy evaluation".to_string()),
                    },
                })
            })
            .collect::<Result<Vec<_>>>()
    })?;
    scored.sort_by(
        |left, right| match (left.robust_fitness, right.robust_fitness) {
            (Some(left_fitness), Some(right_fitness)) => right_fitness
                .total_cmp(&left_fitness)
                .then_with(|| {
                    right
                        .worst_split_r2
                        .unwrap()
                        .total_cmp(&left.worst_split_r2.unwrap())
                })
                .then_with(|| left.candidate.complexity.cmp(&right.candidate.complexity))
                .then_with(|| {
                    left.candidate
                        .semantic_sha256
                        .cmp(&right.candidate.semantic_sha256)
                }),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => left
                .candidate
                .semantic_sha256
                .cmp(&right.candidate.semantic_sha256),
        },
    );
    Ok(scored)
}
