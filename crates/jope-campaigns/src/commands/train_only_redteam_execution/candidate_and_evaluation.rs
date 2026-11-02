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

#[path = "candidate_and_evaluation/evaluation.rs"]
mod evaluation;
use evaluation::*;
#[path = "candidate_and_evaluation/legacy_scoring.rs"]
mod legacy_scoring;
use legacy_scoring::*;
