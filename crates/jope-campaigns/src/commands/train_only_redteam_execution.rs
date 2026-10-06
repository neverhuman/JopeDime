use super::*;

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

#[derive(Clone, Debug)]
struct DeterministicRng(u64);

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
}

fn legacy_offspring_seed(
    parent_run_id: &str,
    dataset_hash: &str,
    generation: usize,
    slot: usize,
    retry: usize,
) -> u64 {
    let material = format!(
        "train-only-offspring-v1\0{parent_run_id}\0{dataset_hash}\0{generation}\0{slot}\0{retry}"
    );
    let digest = Sha256::digest(material.as_bytes());
    u64::from_le_bytes(digest[..8].try_into().unwrap())
}

fn legacy_refill(
    parent_run_id: &str,
    dataset_hash: &str,
    generation: usize,
    ranked: &[LegacyScoredCandidate],
    history: &mut HashSet<String>,
) -> Result<Vec<Candidate>> {
    if ranked
        .iter()
        .filter(|candidate| candidate.robust_fitness.is_some())
        .count()
        < 30
    {
        bail!("generation-20 control has fewer than 30 rankable parents");
    }
    let mut next = Vec::with_capacity(60);
    let mut generation_semantics = HashSet::new();
    for elite in ranked.iter().take(3) {
        let mut candidate = elite.candidate.clone();
        generation_semantics.insert(candidate.semantic_sha256.clone());
        candidate.ancestry = json!({
            "operation": "elite",
            "parents": [candidate.semantic_sha256],
            "generation": generation,
        });
        next.push(candidate);
    }
    for slot in 3..60 {
        let mut accepted = None;
        for retry in 0..100_000 {
            let mut rng = DeterministicRng::new(legacy_offspring_seed(
                parent_run_id,
                dataset_hash,
                generation,
                slot,
                retry,
            ));
            let tournament = |rng: &mut DeterministicRng| {
                let mut winner = (rng.next_u64() as usize) % 30;
                for _ in 1..3 {
                    winner = winner.min((rng.next_u64() as usize) % 30);
                }
                winner
            };
            let left = tournament(&mut rng);
            let crossover = rng.next_u64().is_multiple_of(4);
            let (genome, parents, operation, roots) = if crossover {
                let right = tournament(&mut rng);
                let Ok(genome) = crossover_canonical_pipelines(
                    &ranked[left].candidate.canonical_genome,
                    &ranked[right].candidate.canonical_genome,
                    rng.next_u64(),
                    12,
                ) else {
                    continue;
                };
                let mut roots = ranked[left].candidate.roots.clone();
                roots.extend(ranked[right].candidate.roots.iter().cloned());
                (
                    genome,
                    vec![
                        ranked[left].candidate.semantic_sha256.clone(),
                        ranked[right].candidate.semantic_sha256.clone(),
                    ],
                    "crossover",
                    roots,
                )
            } else {
                let Ok(genome) = mutate_canonical_pipeline_v2(
                    &ranked[left].candidate.canonical_genome,
                    rng.next_u64(),
                    12,
                    true,
                ) else {
                    continue;
                };
                (
                    genome,
                    vec![ranked[left].candidate.semantic_sha256.clone()],
                    "mutation",
                    ranked[left].candidate.roots.clone(),
                )
            };
            let Ok(candidate) = candidate_from_parts(
                genome,
                roots,
                BTreeSet::new(),
                BTreeSet::new(),
                json!({
                    "operation": operation,
                    "parents": parents,
                    "generation": generation,
                    "slot": slot,
                    "retry": retry,
                    "operator_version": "canonical-v2",
                }),
                12,
            ) else {
                continue;
            };
            if !generation_semantics.insert(candidate.semantic_sha256.clone())
                || !history.insert(candidate.semantic_sha256.clone())
            {
                generation_semantics.remove(&candidate.semantic_sha256);
                continue;
            }
            accepted = Some(candidate);
            break;
        }
        next.push(accepted.context("could not generate a unique v2 continuation child")?);
    }
    Ok(next)
}

fn load_parent_archive(dataset: &DatasetInput) -> Result<(Vec<Candidate>, Vec<Candidate>)> {
    let mut archive = Vec::new();
    let mut generation_nine = Vec::new();
    for (generation, path) in dataset.parent_generation_paths.iter().enumerate() {
        let (checkpoint, digest): (Value, _) = read_receipt(path)?;
        if digest != dataset.parent_generation_sha256[generation]
            || checkpoint.get("generation").and_then(Value::as_u64) != Some(generation as u64)
        {
            bail!("parent generation identity drift");
        }
        let population = checkpoint
            .get("population")
            .and_then(Value::as_array)
            .context("parent checkpoint lacks population")?;
        let candidates = population
            .iter()
            .map(candidate_from_parent)
            .collect::<Result<Vec<_>>>()?;
        if candidates.len() != 60 {
            bail!("parent checkpoint population drift");
        }
        archive.extend(candidates.clone());
        if generation == 9 {
            generation_nine = candidates;
        }
    }
    Ok((archive, generation_nine))
}

fn run_g20_dataset(
    root: &Path,
    manifest: &TrainOnlyRedteamManifest,
    dataset: &DatasetInput,
    resume: bool,
    workers: usize,
) -> Result<(PathBuf, String)> {
    let output = root
        .join("datasets")
        .join(&dataset.dataset_hash)
        .join("g20-extension-complete.v1.json");
    if output.exists() {
        if !resume {
            bail!("G20 output exists; pass --resume");
        }
        let (_, digest): (G20Complete, _) = read_receipt(&output)?;
        return Ok((output, digest));
    }
    let data = read_regression_csv(&dataset.train_path)?;
    let (mut archive, _) = load_parent_archive(dataset)?;
    let parent_checkpoint: Value = read_receipt::<Value>(&dataset.parent_generation_paths[9])?.0;
    let parent_population = parent_checkpoint
        .get("population")
        .and_then(Value::as_array)
        .context("parent generation nine lacks population")?;
    let parent_ranked = parent_population
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
    if parent_ranked
        .first()
        .and_then(|candidate| candidate.robust_fitness)
        .is_none()
    {
        bail!("parent generation-nine champion is not rankable");
    }
    let mut history = archive
        .iter()
        .map(|candidate| candidate.semantic_sha256.clone())
        .collect::<HashSet<_>>();
    let mut population = legacy_refill(
        &manifest.parent.run_id,
        &dataset.dataset_hash,
        10,
        &parent_ranked,
        &mut history,
    )?;
    let mut previous = None;
    let mut generation_digests = Vec::new();
    let mut raw_generation_19 = None;
    for generation in G20_GENERATIONS {
        let path = root
            .join("datasets")
            .join(&dataset.dataset_hash)
            .join("g20")
            .join(format!("generation-{generation:03}.v1.json"));
        if path.exists() {
            if !resume {
                bail!("G20 generation exists; pass --resume");
            }
            let (checkpoint, digest): (G20GenerationReceipt, _) = read_receipt(&path)?;
            if checkpoint.schema != "TrainOnlyRedteamG20Generation.v1"
                || checkpoint.run_id != manifest.run_id
                || checkpoint.dataset_hash != dataset.dataset_hash
                || !checkpoint.development_only
                || checkpoint.confirmatory_valid
                || checkpoint.previous_generation_sha256 != previous
                || checkpoint.generation != generation
            {
                bail!("G20 resume chain drift");
            }
            previous = Some(digest.clone());
            generation_digests.push(digest);
            archive.extend(
                checkpoint
                    .population
                    .iter()
                    .map(|candidate| candidate.candidate.clone()),
            );
            history.extend(
                checkpoint
                    .population
                    .iter()
                    .map(|candidate| candidate.candidate.semantic_sha256.clone()),
            );
            history.extend(
                checkpoint
                    .next_population
                    .iter()
                    .map(|candidate| candidate.semantic_sha256.clone()),
            );
            if generation == 19 {
                raw_generation_19 = checkpoint
                    .population
                    .first()
                    .map(|candidate| candidate.candidate.clone());
            } else {
                population = checkpoint.next_population;
            }
            continue;
        }
        let scored = score_legacy_population(
            &data,
            &dataset.dataset_hash,
            generation,
            &population,
            workers,
        )?;
        if scored
            .iter()
            .filter(|candidate| candidate.robust_fitness.is_some())
            .count()
            < 30
        {
            bail!("G20 generation {generation} has fewer than 30 rankable candidates");
        }
        let next_population = if generation < 19 {
            legacy_refill(
                &manifest.parent.run_id,
                &dataset.dataset_hash,
                generation + 1,
                &scored,
                &mut history,
            )?
        } else {
            Vec::new()
        };
        let checkpoint = G20GenerationReceipt {
            schema: "TrainOnlyRedteamG20Generation.v1".to_string(),
            run_id: manifest.run_id.clone(),
            dataset_hash: dataset.dataset_hash.clone(),
            generation,
            parent_generation_nine_sha256: dataset.parent_generation_sha256[9].clone(),
            previous_generation_sha256: previous.clone(),
            ranked_semantic_sha256: scored
                .iter()
                .map(|candidate| candidate.candidate.semantic_sha256.clone())
                .collect(),
            champion_semantic_sha256: scored[0].candidate.semantic_sha256.clone(),
            population: scored.clone(),
            next_population: next_population.clone(),
            development_only: true,
            confirmatory_valid: false,
        };
        let digest = write_receipt(&path, &checkpoint)?;
        archive.extend(scored.into_iter().map(|candidate| candidate.candidate));
        if generation == 19 {
            // The last appended population is ranked, so its first element is
            // 59 positions before the tail.
            raw_generation_19 = archive.get(archive.len().saturating_sub(60)).cloned();
        }
        previous = Some(digest.clone());
        generation_digests.push(digest);
        population = next_population;
    }
    let mut seen = HashSet::new();
    archive.retain(|candidate| seen.insert(candidate.semantic_sha256.clone()));
    let (bank, _): (FoldBankReceipt, _) = read_receipt(&dataset.fold_bank_path)?;
    let mut ranked = evaluate_candidates(
        root,
        manifest,
        dataset,
        &data,
        "search",
        &bank.search,
        &archive,
        workers,
    )?;
    ranked.truncate(10);
    let complete = G20Complete {
        schema: "TrainOnlyRedteamG20Complete.v1".to_string(),
        run_id: manifest.run_id.clone(),
        dataset_hash: dataset.dataset_hash.clone(),
        parent_generation_sha256: dataset.parent_generation_sha256.clone(),
        extension_generation_sha256: generation_digests,
        raw_generation_19: raw_generation_19.context("G20 raw generation-19 champion missing")?,
        top_archive_search_oof: ranked,
        archive_unique_candidates: archive.len(),
        development_only: true,
        confirmatory_valid: false,
    };
    let digest = write_receipt(&output, &complete)?;
    Ok((output, digest))
}

fn merge_candidate(candidates: &mut HashMap<String, Candidate>, mut candidate: Candidate) {
    if let Some(existing) = candidates.get_mut(&candidate.semantic_sha256) {
        existing.roots.append(&mut candidate.roots);
        existing
            .proposal_origins
            .append(&mut candidate.proposal_origins);
        existing.decoder_seeds.append(&mut candidate.decoder_seeds);
    } else {
        candidates.insert(candidate.semantic_sha256.clone(), candidate);
    }
}

fn load_smart_candidates(
    manifest: &TrainOnlyRedteamManifest,
    dataset: &DatasetInput,
) -> Result<(Vec<Candidate>, Vec<Candidate>, Vec<Candidate>)> {
    let mut exact = HashMap::new();
    let mut jittered = HashMap::new();
    let mut skipped = 0usize;
    for line in BufReader::new(File::open(&manifest.proposal_bundle.path)?).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: Value = serde_json::from_str(&line)?;
        if row.get("dataset_hash").and_then(Value::as_str) != Some(&dataset.dataset_hash) {
            continue;
        }
        let origin = value_string(&row, "/proposal_origin")?;
        let seed = row
            .get("invocation_seed")
            .and_then(Value::as_u64)
            .context("proposal lacks invocation_seed")?;
        let genome = value_string(&row, "/canonical_genome")?;
        if row.get("semantic_sha256").and_then(Value::as_str)
            != Some(semantic_identity(&genome).as_str())
        {
            bail!("proposal semantic identity drift");
        }
        let candidate = candidate_from_parts(
            genome,
            BTreeSet::from(["dl".to_string()]),
            BTreeSet::from([origin.clone()]),
            BTreeSet::from([seed]),
            json!({
                "operation": "dl_proposal",
                "source": origin,
                "decoder_seed": seed,
                "perturbation_seed": row.get("perturbation_seed").cloned().unwrap_or(Value::Null),
                "proposal_context_sha256": row.get("proposal_context_sha256").cloned().unwrap_or(Value::Null),
                "exact_context_sha256": row.get("exact_context_sha256").cloned().unwrap_or(Value::Null),
                "scaling_statistic_sha256": row.get("embedding_scaling_statistic_sha256").cloned().unwrap_or(Value::Null),
            }),
            SMART_MAX_TRANSFORMS,
        );
        let Ok(candidate) = candidate else {
            skipped += 1;
            continue;
        };
        match origin.as_str() {
            "exact_context" => merge_candidate(&mut exact, candidate),
            "jittered_context" => merge_candidate(&mut jittered, candidate),
            _ => bail!("unsupported smart proposal origin"),
        }
    }
    let (ranking, digest): (Value, _) = read_receipt(&dataset.transfer_ranking_path)?;
    if digest != dataset.transfer_ranking_sha256
        || ranking.get("target_dataset_id").and_then(Value::as_str) != Some(&dataset.dataset_hash)
        || ranking
            .get("direct_dl_sources_excluded")
            .and_then(Value::as_bool)
            != Some(true)
        || !ranking
            .get("excluded_dataset_ids")
            .and_then(Value::as_array)
            .is_some_and(|values| {
                values
                    .iter()
                    .any(|value| value.as_str() == Some(&dataset.dataset_hash))
            })
    {
        bail!("target-clean transfer ranking provenance drift");
    }
    let mut transfer = HashMap::new();
    for row in ranking
        .get("genomes")
        .and_then(Value::as_array)
        .context("transfer ranking lacks genomes")?
        .iter()
    {
        let genome = value_string(row, "/canonical_genome")?;
        let parsed = GenomeIr::parse(&genome)?;
        let GenomeIr::Pipeline(pipeline) = &parsed else {
            bail!("target-clean transfer candidate is not a pipeline genome");
        };
        if pipeline.stages.len().saturating_sub(1) > SMART_MAX_TRANSFORMS {
            continue;
        }
        let candidate = candidate_from_parts(
            genome,
            BTreeSet::from(["transfer".to_string()]),
            BTreeSet::from(["target_clean_transfer".to_string()]),
            BTreeSet::new(),
            json!({
                "operation": "target_clean_transfer",
                "source_ancestry": row,
                "ranking_sha256": dataset.transfer_ranking_sha256,
            }),
            SMART_MAX_TRANSFORMS,
        );
        let Ok(candidate) = candidate else {
            skipped += 1;
            continue;
        };
        merge_candidate(&mut transfer, candidate);
        if transfer.len() == 256 {
            break;
        }
    }
    let values = |map: HashMap<String, Candidate>| {
        let mut values = map.into_values().collect::<Vec<_>>();
        values.sort_by(|left, right| left.semantic_sha256.cmp(&right.semantic_sha256));
        values
    };
    let exact = values(exact);
    let jittered = values(jittered);
    let transfer = values(transfer);
    if exact.len() < 80 || jittered.len() < 60 || transfer.len() != 256 {
        bail!(
            "progressive screen cannot fill quotas: exact={}, jittered={}, transfer={}",
            exact.len(),
            jittered.len(),
            transfer.len()
        );
    }
    if skipped > 0 {
        eprintln!(
            "smart-v3 load_smart_candidates skipped {} malformed candidates for {}",
            skipped, dataset.dataset_hash
        );
    }
    Ok((exact, jittered, transfer))
}

fn select_with_head_cap(
    ranked: &[RankedCandidate],
    count: usize,
    head_counts: &mut HashMap<String, usize>,
    cap: usize,
) -> Result<Vec<Candidate>> {
    let mut selected = Vec::with_capacity(count);
    let mut semantics = HashSet::new();
    for row in ranked {
        if selected.len() == count {
            break;
        }
        let present = *head_counts.get(&row.candidate.head_family).unwrap_or(&0);
        if present >= cap || !semantics.insert(row.candidate.semantic_sha256.clone()) {
            continue;
        }
        *head_counts
            .entry(row.candidate.head_family.clone())
            .or_default() += 1;
        selected.push(row.candidate.clone());
    }
    if selected.len() != count {
        bail!("head-family cap prevents filling an initialization quota");
    }
    Ok(selected)
}

fn validate_initial_populations(
    populations: &BTreeMap<String, Vec<Candidate>>,
    island_size: usize,
    expected_islands: &[&str],
    head_family_cap: usize,
) -> Result<()> {
    if populations.len() != expected_islands.len()
        || expected_islands
            .iter()
            .any(|island| populations.get(*island).is_none())
        || populations
            .values()
            .any(|population| population.len() != island_size)
    {
        bail!("Smart-v3 initialization island size drift");
    }

    let mut semantics = HashSet::new();
    for island in expected_islands {
        let mut head_counts = HashMap::new();
        for candidate in &populations[*island] {
            if !semantics.insert(candidate.semantic_sha256.clone()) {
                bail!("initial Smart-v3 population is not canonically unique");
            }
            let count = head_counts
                .entry(candidate.head_family.clone())
                .or_insert(0usize);
            *count += 1;
            if *count > head_family_cap {
                bail!("Smart-v3 {island} initialization head-family cap exceeded");
            }
        }
    }
    Ok(())
}

fn mixed_seed(run_id: &str, dataset_hash: &str, slot: usize, retry: usize) -> u64 {
    let digest = Sha256::digest(
        format!("smart-v3-mixed-init-v1\0{run_id}\0{dataset_hash}\0{slot}\0{retry}").as_bytes(),
    );
    u64::from_le_bytes(digest[..8].try_into().unwrap())
}

fn build_screening(
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

fn nsga_order(population: &[RankedCandidate]) -> Vec<usize> {
    let dominates = |left: &RankedCandidate, right: &RankedCandidate| {
        left.robust_loss <= right.robust_loss
            && left.candidate.complexity <= right.candidate.complexity
            && (left.robust_loss < right.robust_loss
                || left.candidate.complexity < right.candidate.complexity)
    };
    let mut dominated = vec![Vec::new(); population.len()];
    let mut domination_count = vec![0usize; population.len()];
    for left in 0..population.len() {
        for right in 0..population.len() {
            if left == right {
                continue;
            }
            if dominates(&population[left], &population[right]) {
                dominated[left].push(right);
            } else if dominates(&population[right], &population[left]) {
                domination_count[left] += 1;
            }
        }
    }
    let mut fronts = Vec::new();
    let mut current = domination_count
        .iter()
        .enumerate()
        .filter_map(|(index, count)| (*count == 0).then_some(index))
        .collect::<Vec<_>>();
    while !current.is_empty() {
        current.sort_by(|left, right| rank_loss(&population[*left], &population[*right]));
        fronts.push(current.clone());
        let mut next = Vec::new();
        for index in current {
            for dominated_index in &dominated[index] {
                domination_count[*dominated_index] -= 1;
                if domination_count[*dominated_index] == 0 {
                    next.push(*dominated_index);
                }
            }
        }
        current = next;
    }
    let mut ordered = Vec::with_capacity(population.len());
    for mut front in fronts {
        let mut crowding = HashMap::<usize, f64>::new();
        for index in &front {
            crowding.insert(*index, 0.0);
        }
        for objective in 0..2 {
            front.sort_by(|left, right| {
                let left_value = if objective == 0 {
                    population[*left].robust_loss
                } else {
                    population[*left].candidate.complexity as f64
                };
                let right_value = if objective == 0 {
                    population[*right].robust_loss
                } else {
                    population[*right].candidate.complexity as f64
                };
                left_value.total_cmp(&right_value)
            });
            if front.len() > 1 {
                crowding.insert(front[0], f64::INFINITY);
                crowding.insert(*front.last().unwrap(), f64::INFINITY);
            }
            if front.len() > 2 {
                let first = if objective == 0 {
                    population[front[0]].robust_loss
                } else {
                    population[front[0]].candidate.complexity as f64
                };
                let last = if objective == 0 {
                    population[*front.last().unwrap()].robust_loss
                } else {
                    population[*front.last().unwrap()].candidate.complexity as f64
                };
                let scale = (last - first).abs().max(1e-15);
                for position in 1..(front.len() - 1) {
                    let before = if objective == 0 {
                        population[front[position - 1]].robust_loss
                    } else {
                        population[front[position - 1]].candidate.complexity as f64
                    };
                    let after = if objective == 0 {
                        population[front[position + 1]].robust_loss
                    } else {
                        population[front[position + 1]].candidate.complexity as f64
                    };
                    *crowding.get_mut(&front[position]).unwrap() += (after - before) / scale;
                }
            }
        }
        front.sort_by(|left, right| {
            crowding[right]
                .total_cmp(&crowding[left])
                .then_with(|| rank_loss(&population[*left], &population[*right]))
        });
        ordered.extend(front);
    }
    ordered
}

fn is_nondominated(population: &[RankedCandidate], index: usize) -> bool {
    let candidate = &population[index];
    !population.iter().any(|other| {
        other.robust_loss <= candidate.robust_loss
            && other.candidate.complexity <= candidate.candidate.complexity
            && (other.robust_loss < candidate.robust_loss
                || other.candidate.complexity < candidate.candidate.complexity)
    })
}

fn structural_niche(candidate: &Candidate) -> Result<String> {
    let parsed = GenomeIr::parse(&candidate.canonical_genome)?;
    let GenomeIr::Pipeline(pipeline) = parsed else {
        bail!("structural niche expects a pipeline");
    };
    Ok(format!(
        "{}|{}|{}",
        candidate.head_family,
        candidate.complexity,
        pipeline
            .stages
            .iter()
            .take(candidate.complexity)
            .map(|stage| stage.op.as_str())
            .collect::<Vec<_>>()
            .join(",")
    ))
}

fn smart_seed(
    run_id: &str,
    dataset_hash: &str,
    island: &str,
    generation: usize,
    slot: usize,
    retry: usize,
) -> u64 {
    let digest = Sha256::digest(
        format!(
            "smart-v3-offspring-v1\0{run_id}\0{dataset_hash}\0{island}\0{generation}\0{slot}\0{retry}"
        )
        .as_bytes(),
    );
    u64::from_le_bytes(digest[..8].try_into().unwrap())
}

fn mutation_arm(draw: u64) -> (&'static str, CanonicalMutationArm) {
    match draw % 100 {
        0..=29 => ("parameter_jitter", CanonicalMutationArm::ParameterJitter),
        30..=39 => ("insertion", CanonicalMutationArm::Insert),
        40..=59 => ("deletion", CanonicalMutationArm::Delete),
        60..=69 => ("adjacent_swap", CanonicalMutationArm::AdjacentSwap),
        70..=84 => ("head_replacement", CanonicalMutationArm::HeadReplacement),
        _ => ("simplification", CanonicalMutationArm::Simplify),
    }
}

fn smart_refill(
    manifest: &TrainOnlyRedteamManifest,
    dataset_hash: &str,
    island: &str,
    generation: usize,
    ranked: &[RankedCandidate],
    mixed_crossover_sources: Option<(&[RankedCandidate], &[RankedCandidate])>,
    history: &mut HashSet<String>,
) -> Result<(Vec<Candidate>, BTreeMap<String, usize>)> {
    let order = nsga_order(ranked);
    if order.len() != ISLAND_SIZE {
        bail!("Smart-v3 island population drift");
    }
    let mut next = order[..2]
        .iter()
        .map(|index| {
            let mut candidate = ranked[*index].candidate.clone();
            candidate.ancestry = json!({
                "operation": "elite",
                "parent": candidate.semantic_sha256,
                "generation": generation,
                "island": island,
            });
            candidate
        })
        .collect::<Vec<_>>();
    let mut generation_semantics = next
        .iter()
        .map(|candidate| candidate.semantic_sha256.clone())
        .collect::<HashSet<_>>();
    let mut counts = BTreeMap::new();
    for slot in 2..ISLAND_SIZE {
        let mut accepted = None;
        for retry in 0..100_000 {
            let mut rng = DeterministicRng::new(smart_seed(
                &manifest.run_id,
                dataset_hash,
                island,
                generation,
                slot,
                retry,
            ));
            let tournament = |rng: &mut DeterministicRng| {
                let mut best = (rng.next_u64() as usize) % 15;
                for _ in 1..3 {
                    best = best.min((rng.next_u64() as usize) % 15);
                }
                order[best]
            };
            let left_index = tournament(&mut rng);
            let left = &ranked[left_index].candidate;
            let crossover = rng.next_u64().is_multiple_of(4);
            let (genome, roots, origins, seeds, ancestry, operator) = if crossover {
                let (crossover_left, right) = if island == "mixed" {
                    let (dl, transfer) = mixed_crossover_sources
                        .context("mixed island lacks DL×transfer crossover sources")?;
                    let dl_order = nsga_order(dl);
                    let transfer_order = nsga_order(transfer);
                    let dl_index = dl_order[(rng.next_u64() as usize) % 15];
                    let transfer_index = transfer_order[(rng.next_u64() as usize) % 15];
                    (&dl[dl_index].candidate, &transfer[transfer_index].candidate)
                } else {
                    let right_index = tournament(&mut rng);
                    (left, &ranked[right_index].candidate)
                };
                let Ok(genome) = crossover_canonical_pipelines(
                    &crossover_left.canonical_genome,
                    &right.canonical_genome,
                    rng.next_u64(),
                    SMART_MAX_TRANSFORMS,
                ) else {
                    continue;
                };
                let mut roots = crossover_left.roots.clone();
                roots.extend(right.roots.iter().cloned());
                let mut origins = crossover_left.proposal_origins.clone();
                origins.extend(right.proposal_origins.iter().cloned());
                let mut seeds = crossover_left.decoder_seeds.clone();
                seeds.extend(right.decoder_seeds.iter().copied());
                (
                    genome,
                    roots,
                    origins,
                    seeds,
                    json!({
                        "operation": if island == "mixed" { "dl_transfer_crossover" } else { "crossover" },
                        "parents": [crossover_left.semantic_sha256, right.semantic_sha256],
                        "generation": generation,
                        "island": island,
                        "slot": slot,
                        "retry": retry,
                    }),
                    "crossover",
                )
            } else {
                let (name, arm) = mutation_arm(rng.next_u64());
                let Ok(genome) = mutate_canonical_pipeline_with_arm(
                    &left.canonical_genome,
                    rng.next_u64(),
                    SMART_MAX_TRANSFORMS,
                    true,
                    arm,
                ) else {
                    continue;
                };
                (
                    genome,
                    left.roots.clone(),
                    left.proposal_origins.clone(),
                    left.decoder_seeds.clone(),
                    json!({
                        "operation": name,
                        "parent": left.semantic_sha256,
                        "generation": generation,
                        "island": island,
                        "slot": slot,
                        "retry": retry,
                    }),
                    name,
                )
            };
            let Ok(candidate) = candidate_from_parts(
                genome,
                roots,
                origins,
                seeds,
                ancestry,
                SMART_MAX_TRANSFORMS,
            ) else {
                continue;
            };
            if island == "dl" && candidate.roots != BTreeSet::from(["dl".to_string()])
                || island == "transfer"
                    && candidate.roots != BTreeSet::from(["transfer".to_string()])
                || island == "mixed"
                    && !(candidate.roots.contains("dl") && candidate.roots.contains("transfer"))
            {
                bail!("Smart-v3 island ancestry boundary drift");
            }
            if !generation_semantics.insert(candidate.semantic_sha256.clone())
                || !history.insert(candidate.semantic_sha256.clone())
            {
                generation_semantics.remove(&candidate.semantic_sha256);
                continue;
            }
            *counts.entry(operator.to_string()).or_default() += 1;
            accepted = Some(candidate);
            break;
        }
        next.push(accepted.context("cannot generate a unique Smart-v3 offspring")?);
    }
    Ok((next, counts))
}

fn run_smart_dataset(
    root: &Path,
    manifest: &TrainOnlyRedteamManifest,
    dataset: &DatasetInput,
    resume: bool,
    workers: usize,
) -> Result<(PathBuf, String)> {
    let output = root
        .join("datasets")
        .join(&dataset.dataset_hash)
        .join("smart-v3-complete.v1.json");
    if output.exists() {
        if !resume {
            bail!("Smart-v3 output exists; pass --resume");
        }
        let (_, digest): (SmartComplete, _) = read_receipt(&output)?;
        return Ok((output, digest));
    }
    let data = read_regression_csv(&dataset.train_path)?;
    let (bank, bank_digest): (FoldBankReceipt, _) = read_receipt(&dataset.fold_bank_path)?;
    if bank_digest != dataset.fold_bank_sha256 {
        bail!("Smart-v3 fold-bank drift");
    }
    let (screening, screening_digest, _) =
        build_screening(root, manifest, dataset, &data, &bank, workers)?;
    let mut populations = screening.initial_populations;
    let mut histories = populations
        .iter()
        .map(|(island, candidates)| {
            (
                island.clone(),
                candidates
                    .iter()
                    .map(|candidate| candidate.semantic_sha256.clone())
                    .collect::<HashSet<_>>(),
            )
        })
        .collect::<HashMap<_, _>>();
    let mut immutable_archive = BTreeMap::<String, HashMap<String, RankedCandidate>>::from([
        ("dl".to_string(), HashMap::new()),
        ("transfer".to_string(), HashMap::new()),
        ("mixed".to_string(), HashMap::new()),
    ]);
    let mut structural_niches = HashSet::new();
    let mut previous = None;
    let mut generation_digests = Vec::new();
    let mut mutation_counts = BTreeMap::new();
    let mut previous_best = None;
    let mut stagnant = 0;
    let mut stopped_early = false;
    let mut generations_completed = 0;
    for generation in 0..manifest.config.smart_v3.maximum_generations {
        let generation_path = root
            .join("datasets")
            .join(&dataset.dataset_hash)
            .join("smart")
            .join(format!("generation-{generation:03}.v1.json"));
        if generation_path.exists() {
            if !resume {
                bail!("Smart-v3 generation exists; pass --resume");
            }
            let (checkpoint, digest): (SmartGenerationReceipt, _) = read_receipt(&generation_path)?;
            if checkpoint.schema != "TrainOnlyRedteamSmartGeneration.v1"
                || checkpoint.run_id != manifest.run_id
                || checkpoint.dataset_hash != dataset.dataset_hash
                || !checkpoint.development_only
                || checkpoint.confirmatory_valid
                || checkpoint.generation != generation
                || checkpoint.previous_generation_sha256 != previous
            {
                bail!("Smart-v3 resume chain drift");
            }
            for (island, archive_digest) in &checkpoint.archive_sha256 {
                let archive_path = root
                    .join("datasets")
                    .join(&dataset.dataset_hash)
                    .join("smart")
                    .join("archive")
                    .join(format!("{island}-generation-{generation:03}.v1.json"));
                let (archive, digest): (IslandArchiveReceipt, _) = read_receipt(&archive_path)?;
                if &digest != archive_digest
                    || !archive.development_only
                    || archive.confirmatory_valid
                {
                    bail!("Smart-v3 archive link drift");
                }
                for candidate in archive.candidates {
                    immutable_archive
                        .get_mut(island)
                        .unwrap()
                        .insert(candidate.candidate.semantic_sha256.clone(), candidate);
                }
            }
            for (operator, count) in checkpoint.mutation_counts {
                *mutation_counts.entry(operator).or_default() += count;
            }
            previous_best = Some(checkpoint.best_robust_loss);
            stagnant = checkpoint.consecutive_stagnant_generations;
            populations = checkpoint.next_populations;
            for (island, candidates) in &checkpoint.populations {
                histories
                    .get_mut(island)
                    .context("Smart-v3 resume island history missing")?
                    .extend(
                        candidates
                            .iter()
                            .map(|candidate| candidate.candidate.semantic_sha256.clone()),
                    );
                for (index, candidate) in candidates.iter().enumerate() {
                    if is_nondominated(candidates, index) {
                        structural_niches.insert(structural_niche(&candidate.candidate)?);
                    }
                }
            }
            for (island, candidates) in &populations {
                histories
                    .get_mut(island)
                    .context("Smart-v3 resume next-island history missing")?
                    .extend(
                        candidates
                            .iter()
                            .map(|candidate| candidate.semantic_sha256.clone()),
                    );
            }
            previous = Some(digest.clone());
            generation_digests.push(digest);
            generations_completed = generation + 1;
            if checkpoint.stopped {
                stopped_early = true;
                break;
            }
            continue;
        }
        let mut scored_populations = BTreeMap::new();
        let mut archive_sha256 = BTreeMap::new();
        let mut best_loss = f64::INFINITY;
        let mut new_niche = false;
        for island in ["dl", "transfer", "mixed"] {
            let population = populations
                .get(island)
                .with_context(|| format!("Smart-v3 missing island {island}"))?;
            let scored = evaluate_candidates(
                root,
                manifest,
                dataset,
                &data,
                "search",
                &bank.search,
                population,
                workers,
            )?;
            if scored.len() != ISLAND_SIZE {
                bail!("Smart-v3 island size drift");
            }
            let order = nsga_order(&scored);
            best_loss = best_loss.min(
                scored
                    .iter()
                    .map(|candidate| candidate.robust_loss)
                    .min_by(f64::total_cmp)
                    .unwrap(),
            );
            for index in order
                .iter()
                .take_while(|index| is_nondominated(&scored, **index))
            {
                new_niche |= structural_niches.insert(structural_niche(&scored[*index].candidate)?);
            }
            let archived = order
                .iter()
                .take(3)
                .map(|index| scored[*index].clone())
                .collect::<Vec<_>>();
            for candidate in &archived {
                immutable_archive.get_mut(island).unwrap().insert(
                    candidate.candidate.semantic_sha256.clone(),
                    candidate.clone(),
                );
            }
            let archive_receipt = IslandArchiveReceipt {
                schema: "TrainOnlyRedteamSmartArchive.v1".to_string(),
                run_id: manifest.run_id.clone(),
                dataset_hash: dataset.dataset_hash.clone(),
                island: island.to_string(),
                generation,
                candidates: archived,
                development_only: true,
                confirmatory_valid: false,
            };
            let archive_path = root
                .join("datasets")
                .join(&dataset.dataset_hash)
                .join("smart")
                .join("archive")
                .join(format!("{island}-generation-{generation:03}.v1.json"));
            archive_sha256.insert(
                island.to_string(),
                write_receipt(&archive_path, &archive_receipt)?,
            );
            scored_populations.insert(island.to_string(), scored);
        }
        let improved = previous_best
            .is_none_or(|previous| (previous - best_loss) / previous.abs().max(1e-15) >= 0.001);
        if improved || new_niche {
            stagnant = 0;
        } else {
            stagnant += 1;
        }
        let can_stop = generation + 1 >= manifest.config.smart_v3.minimum_generations
            && stagnant >= manifest.config.smart_v3.stagnation_generations;
        let last = generation + 1 == manifest.config.smart_v3.maximum_generations;
        let mut next_populations = BTreeMap::new();
        let mut generation_mutations = BTreeMap::new();
        if !can_stop && !last {
            for island in ["dl", "transfer", "mixed"] {
                let (next, counts) = smart_refill(
                    manifest,
                    &dataset.dataset_hash,
                    island,
                    generation + 1,
                    &scored_populations[island],
                    (island == "mixed").then(|| {
                        (
                            scored_populations["dl"].as_slice(),
                            scored_populations["transfer"].as_slice(),
                        )
                    }),
                    histories.get_mut(island).unwrap(),
                )?;
                next_populations.insert(island.to_string(), next);
                for (operator, count) in counts {
                    *generation_mutations.entry(operator.clone()).or_default() += count;
                    *mutation_counts.entry(operator).or_default() += count;
                }
            }
        }
        let checkpoint = SmartGenerationReceipt {
            schema: "TrainOnlyRedteamSmartGeneration.v1".to_string(),
            run_id: manifest.run_id.clone(),
            dataset_hash: dataset.dataset_hash.clone(),
            generation,
            previous_generation_sha256: previous.clone(),
            populations: scored_populations,
            next_populations: next_populations.clone(),
            archive_sha256,
            best_robust_loss: best_loss,
            new_nondominated_structural_niche: new_niche,
            consecutive_stagnant_generations: stagnant,
            mutation_counts: generation_mutations,
            stopped: can_stop || last,
            development_only: true,
            confirmatory_valid: false,
        };
        let digest = write_receipt(&generation_path, &checkpoint)?;
        previous = Some(digest.clone());
        previous_best = Some(best_loss);
        generation_digests.push(digest);
        populations = next_populations;
        generations_completed = generation + 1;
        if can_stop {
            stopped_early = true;
            break;
        }
    }
    if generations_completed < manifest.config.smart_v3.minimum_generations {
        bail!("Smart-v3 stopped before its minimum generation count");
    }
    let mut archive_top20_by_island = BTreeMap::new();
    for island in ["dl", "transfer", "mixed"] {
        let mut candidates = immutable_archive
            .remove(island)
            .unwrap()
            .into_values()
            .collect::<Vec<_>>();
        candidates.sort_by(rank_loss);
        candidates.truncate(20);
        archive_top20_by_island.insert(island.to_string(), candidates);
    }
    let complete = SmartComplete {
        schema: "TrainOnlyRedteamSmartComplete.v1".to_string(),
        run_id: manifest.run_id.clone(),
        dataset_hash: dataset.dataset_hash.clone(),
        generations_completed,
        stopped_early,
        screening_sha256: screening_digest,
        generation_sha256: generation_digests,
        archive_top20_by_island,
        mutation_counts,
        development_only: true,
        confirmatory_valid: false,
    };
    let digest = write_receipt(&output, &complete)?;
    Ok((output, digest))
}

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
