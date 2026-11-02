use super::*;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PackedLabelShardV3 {
    pub first_receipt: u64,
    pub last_receipt: u64,
    pub rows: usize,
    pub path: PathBuf,
    pub sha256: Sha256Digest,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reused_from_corpus_sha256: Option<Sha256Digest>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PackedColumnV3 {
    pub path: PathBuf,
    pub sha256: Sha256Digest,
    pub dtype: String,
    pub rows: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PackedCorpusV3 {
    pub schema: String,
    pub watermark: u64,
    pub catalog_binding_sha256: Sha256Digest,
    pub selected_row_index: PackedColumnV3,
    pub task_column: PackedColumnV3,
    pub split_column: PackedColumnV3,
    pub exact_pair_sidecar: PackedColumnV3,
    pub label_shards: Vec<PackedLabelShardV3>,
    pub created_at_ms: u128,
}

impl PackedCorpusV3 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == PACKED_CORPUS_V3_SCHEMA,
            "packed corpus schema is not v3"
        );
        ensure!(self.watermark > 0, "packed corpus watermark is zero");
        ensure!(
            !self.label_shards.is_empty(),
            "packed corpus has no content-addressed label shards"
        );
        let rows = self.selected_row_index.rows;
        ensure!(
            self.task_column.rows == rows
                && self.split_column.rows == rows
                && self
                    .label_shards
                    .iter()
                    .map(|shard| shard.rows)
                    .sum::<usize>()
                    == rows,
            "packed corpus selected index, columns, and shard rows differ"
        );
        ensure!(
            self.task_column.dtype == "u1" && self.split_column.dtype == "u1",
            "packed task/split columns must be memory-mappable u1 arrays"
        );
        ensure!(
            [
                &self.selected_row_index.path,
                &self.task_column.path,
                &self.split_column.path,
                &self.exact_pair_sidecar.path,
            ]
            .into_iter()
            .all(|path| path.file_name().and_then(|name| name.to_str())
                != Some("observations.jsonl")),
            "PackedCorpus.v3 must not copy observations.jsonl"
        );
        let mut previous = 0;
        let mut hashes = BTreeSet::new();
        for shard in &self.label_shards {
            ensure!(
                shard.rows > 0
                    && shard.first_receipt <= shard.last_receipt
                    && shard.first_receipt > previous
                    && shard.last_receipt <= self.watermark
                    && hashes.insert(&shard.sha256),
                "packed corpus shards overlap, repeat, or exceed the frozen watermark"
            );
            previous = shard.last_receipt;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationEvidenceV1 {
    pub error_correlation_lower_95: f64,
    pub interval_coverage: f64,
    pub target_coverage: f64,
    pub native_head_accuracy: f64,
    pub frequency_baseline_accuracy: f64,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UncertaintyRankingSourceV1 {
    Learned,
    EmpiricalEnsembleDispersion,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RankingFallbackV1 {
    pub uncertainty: UncertaintyRankingSourceV1,
    pub native_head_enabled: bool,
}

pub fn ranking_fallback(evidence: &CalibrationEvidenceV1) -> Result<RankingFallbackV1> {
    ensure!(
        [
            evidence.error_correlation_lower_95,
            evidence.interval_coverage,
            evidence.target_coverage,
            evidence.native_head_accuracy,
            evidence.frequency_baseline_accuracy,
        ]
        .into_iter()
        .all(f64::is_finite),
        "calibration evidence contains a non-finite metric"
    );
    let uncertainty = if evidence.error_correlation_lower_95 >= 0.0
        && (evidence.interval_coverage - evidence.target_coverage).abs() <= 0.05
    {
        UncertaintyRankingSourceV1::Learned
    } else {
        UncertaintyRankingSourceV1::EmpiricalEnsembleDispersion
    };
    Ok(RankingFallbackV1 {
        uncertainty,
        native_head_enabled: evidence.native_head_accuracy
            >= evidence.frequency_baseline_accuracy + 0.05,
    })
}
