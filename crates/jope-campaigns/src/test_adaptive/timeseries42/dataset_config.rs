fn dataset_worker_count() -> Result<usize> {
    let Ok(value) = std::env::var(DATASET_WORKERS_ENV) else {
        return Ok(DEFAULT_DATASET_WORKERS);
    };
    let workers = value
        .trim()
        .parse::<usize>()
        .with_context(|| format!("parse {DATASET_WORKERS_ENV}"))?;
    if workers == 0 {
        bail!("{DATASET_WORKERS_ENV} must be at least 1");
    }
    Ok(workers.min(MAX_DATASET_WORKERS))
}

fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .ok()
        .is_some_and(|value| matches!(value.trim(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct OfficialLeader {
    model: String,
    mase: f64,
    rmse: f64,
    report_path: String,
    report_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct ExternalReport {
    model: String,
    report_path: String,
    report_sha256: String,
    mase: f64,
    rmse: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct DatasetSpec {
    schema: String,
    dataset_hash: String,
    display_name: String,
    category: String,
    train_path: String,
    train_sha256: String,
    test_path: String,
    test_sha256: String,
    meta_path: String,
    meta_sha256: String,
    train_rows: usize,
    test_rows: usize,
    csv_features: usize,
    reduction_features: usize,
    horizon: usize,
    seasonality: usize,
    frequency: String,
    target_is_final_column: bool,
    known_future_covariates: Vec<String>,
    past_only_covariates: Vec<String>,
    source_real_data: bool,
    synthetic_origin: bool,
    /// Empty means ts-reduction-v1 (every historical campaign); campaigns opt
    /// into v2 by stamping ts-reduction-v2 here AND in CampaignConfig — the
    /// two pins must agree, which new campaign constructors are responsible
    /// for. Skipped when empty so historical manifests stay byte-identical.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    reduction_version: String,
    /// The immutable external envelope used by the all-models campaign.  This
    /// is absent from historical manifests so their canonical bytes remain
    /// unchanged when they are resumed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    accepted_external_reports: Vec<ExternalReport>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    comparator_lock_sha256: String,
    official_leader: OfficialLeader,
}

impl DatasetSpec {
    fn reduction(&self) -> Result<ReductionVersion> {
        if self.reduction_version.is_empty() {
            Ok(ReductionVersion::V1)
        } else {
            ReductionVersion::parse(&self.reduction_version)
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct AncestryRecord {
    operation: String,
    parents: Vec<String>,
    retry_count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Member {
    canonical_genome: String,
    semantic_sha256: String,
    complexity: usize,
    rust_source_len: usize,
    evolved_descendant: bool,
    ancestry: AncestryRecord,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct SeedRecord {
    slot: usize,
    canonical_genome: String,
    semantic_sha256: String,
    complexity: usize,
    rust_source_len: usize,
    source_family: String,
    source_file: String,
    source_file_sha256: String,
    source_record: usize,
    source_rank: usize,
    source_score: Option<f64>,
}

impl SeedRecord {
    fn member(&self) -> Member {
        Member {
            canonical_genome: self.canonical_genome.clone(),
            semantic_sha256: self.semantic_sha256.clone(),
            complexity: self.complexity,
            rust_source_len: self.rust_source_len,
            evolved_descendant: false,
            ancestry: AncestryRecord {
                operation: format!("seed:{}", self.source_family),
                parents: Vec::new(),
                retry_count: 0,
            },
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct SeedBank {
    schema: String,
    test_adaptive: bool,
    benchmark_valid: bool,
    dataset_hash: String,
    population: usize,
    selection: String,
    seeds: Vec<SeedRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct CampaignConfig {
    population: usize,
    generations: usize,
    shards: usize,
    shard_size: usize,
    nodes: Vec<String>,
    candidate_threads: usize,
    candidate_timeout_seconds: u64,
    infrastructure_retries: usize,
    #[serde(default, skip_serializing_if = "is_zero")]
    minimum_generations: usize,
    ranking: String,
    win_rule: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    lineage_salt: String,
    /// Empty means ts-reduction-v1. Frozen constructors (fixed/gen30/winall)
    /// never set this, so their configs and every historical manifest stay
    /// byte-identical; config equality checks therefore also pin v1 for the
    /// frozen flows.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    reduction_version: String,
}

impl CampaignConfig {
    fn fixed() -> Self {
        Self::fixed_with_generations(
            GENERATIONS,
            "finite evolved descendant with raw test MASE strictly lower than both official leader MASE and generation-zero best-seed MASE",
        )
    }

    fn gen30() -> Self {
        Self::fixed_with_generations(
            GEN30_GENERATIONS,
            "base generations 0-1 are immutable; generations 2-29 continue with 60 candidates/generation and stop non-base-wins only after a generation >=2 evolved descendant strictly beats both official leader MASE and generation-zero best MASE",
        )
    }

    fn forecast3_gen20() -> Self {
        let mut config = Self::fixed_with_generations(
            FORECAST3_GENERATIONS,
            "stop after the first confirmed finite candidate with raw test MASE strictly lower than the frozen authorized external comparator, including generation 0; otherwise exhaust after generation 19",
        );
        config.minimum_generations = 0;
        config
    }

    fn winall(generations: usize, start_generation: usize, lineage_salt: &str) -> Self {
        let mut config = Self::fixed_with_generations(
            generations,
            &format!(
                "source generations 0-{} are immutable; generations {start_generation}-{} continue source non-winners with 60 candidates/generation and stop after an evolved descendant strictly beats both official leader MASE and generation-zero best MASE",
                start_generation - 1,
                generations - 1
            ),
        );
        config.lineage_salt = lineage_salt.to_string();
        config
    }

    fn fixed_with_generations(generations: usize, win_rule: &str) -> Self {
        Self {
            population: POPULATION,
            generations,
            shards: SHARDS,
            shard_size: SHARD_SIZE,
            nodes: NODES.iter().map(|node| (*node).to_string()).collect(),
            candidate_threads: SHARD_SIZE,
            candidate_timeout_seconds: CANDIDATE_TIMEOUT_SECONDS,
            infrastructure_retries: NODES.len() - 1,
            minimum_generations: 0,
            ranking: "test_mase ASC, rmse ASC, complexity ASC, semantic_sha256 ASC".to_string(),
            win_rule: win_rule.to_string(),
            lineage_salt: String::new(),
            reduction_version: String::new(),
        }
    }
}

fn is_zero(value: &usize) -> bool {
    *value == 0
}

fn is_forecast3_gen20(run_id: &str) -> bool {
    run_id == FORECAST3_RUN_ID
}

fn base_campaign_config(run_id: &str) -> CampaignConfig {
    if is_forecast3_gen20(run_id) {
        CampaignConfig::forecast3_gen20()
    } else {
        CampaignConfig::fixed()
    }
}
