use super::*;

impl TournamentSpec {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.candidate_checkpoint_sha256 != self.parent_checkpoint_sha256,
            "candidate and parent checkpoints are identical"
        );
        ensure!(
            self.family_disjoint
                && self.absent_from_candidate_corpus
                && self.absent_from_sealed_audit,
            "promotion cohort is not fresh and family-disjoint"
        );
        ensure!(
            !self.seeds.is_empty() && self.budget_per_dataset > 0,
            "tournament seeds and budget are required"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PromotionEvidence {
    pub datasets: usize,
    pub validity: f64,
    pub all_native: bool,
    pub all_contextualized: bool,
    pub direct_strict_wins: usize,
    pub conditional_advantage: f64,
    pub context_ablation_lower_bound: f64,
    pub latency_ratio: f64,
    pub family_utility_lower_bound: f64,
    pub family_failure_delta_upper_bound: f64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PromotionDecision {
    Accept,
    Continue { next_datasets: usize },
    Reject { reasons: Vec<&'static str> },
}

pub fn promotion_decision(evidence: PromotionEvidence) -> Result<PromotionDecision> {
    ensure!(
        (256..=2_048).contains(&evidence.datasets) && evidence.datasets.is_multiple_of(128),
        "promotion evidence must use deterministic 128-dataset blocks"
    );
    ensure!(
        [
            evidence.validity,
            evidence.conditional_advantage,
            evidence.context_ablation_lower_bound,
            evidence.latency_ratio,
            evidence.family_utility_lower_bound,
            evidence.family_failure_delta_upper_bound,
        ]
        .into_iter()
        .all(f64::is_finite),
        "promotion metrics must be finite"
    );
    let mut reasons = Vec::new();
    if evidence.validity < 0.99 {
        reasons.push("validity_below_99pct");
    }
    if !evidence.all_native {
        reasons.push("non_native_rows");
    }
    if !evidence.all_contextualized {
        reasons.push("context_mismatch");
    }
    if evidence.direct_strict_wins == 0 {
        reasons.push("no_direct_strict_win");
    }
    if evidence.conditional_advantage <= 0.0 {
        reasons.push("nonpositive_conditional_advantage");
    }
    if evidence.context_ablation_lower_bound <= 0.0 {
        reasons.push("context_ablation_not_positive");
    }
    if evidence.latency_ratio > 1.25 {
        reasons.push("latency_guard");
    }
    if evidence.family_utility_lower_bound <= 0.0 {
        reasons.push("utility_bound_not_clear");
    }
    if evidence.family_failure_delta_upper_bound > 0.0 {
        reasons.push("failure_bound_not_clear");
    }
    if reasons.is_empty() {
        return Ok(PromotionDecision::Accept);
    }
    if evidence.datasets == 2_048 {
        Ok(PromotionDecision::Reject { reasons })
    } else {
        Ok(PromotionDecision::Continue {
            next_datasets: (evidence.datasets + 128).min(2_048),
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncodingDataset {
    pub dataset_id: String,
    pub content_sha256: Sha256Digest,
    pub encoded: bool,
}

pub fn plan_encoding_chunks(
    datasets: &[EncodingDataset],
    encoder_identity_sha256: &Sha256Digest,
) -> Vec<EncodingChunkV2> {
    let mut pending = datasets
        .iter()
        .filter(|dataset| !dataset.encoded)
        .collect::<Vec<_>>();
    pending.sort_by(|left, right| {
        left.content_sha256
            .cmp(&right.content_sha256)
            .then_with(|| left.dataset_id.cmp(&right.dataset_id))
    });
    pending
        .chunks(ENCODING_CHUNK_SIZE)
        .map(|rows| {
            let datasets = rows
                .iter()
                .map(|row| super::EncodingDatasetIdentity {
                    dataset_id: row.dataset_id.clone(),
                    content_sha256: row.content_sha256.clone(),
                })
                .collect::<Vec<_>>();
            let identity = rows
                .iter()
                .map(|row| format!("{}:{}", row.dataset_id, row.content_sha256))
                .collect::<Vec<_>>()
                .join("|");
            EncodingChunkV2 {
                schema: ENCODING_CHUNK_SCHEMA.to_owned(),
                chunk_sha256: Sha256Digest::of_bytes(
                    format!("breadth-encode-v2|{encoder_identity_sha256}|{identity}").as_bytes(),
                ),
                encoder_identity_sha256: encoder_identity_sha256.clone(),
                attempt_number: 1,
                datasets,
                preemptible: true,
                reserved_cpu_cores: 4,
                required_node: "xbabe3".to_owned(),
                project: "jope-selfplay-breadth-encode".to_owned(),
            }
        })
        .collect()
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EncodingTrancheState {
    pub activated_datasets: usize,
    pub first_touch_datasets: usize,
    pub last_admission_attempt_ms: u64,
}

pub fn may_release_next_tranche(state: EncodingTrancheState) -> bool {
    state.activated_datasets == 0
        || state.activated_datasets <= ENCODING_TRANCHE_SIZE
            && state.first_touch_datasets as f64 / state.activated_datasets as f64
                >= FIRST_TOUCH_RELEASE_FRACTION
}

pub fn encoding_retry_due(now_ms: u64, last_attempt_ms: u64, gpu_free: bool) -> bool {
    gpu_free && now_ms.saturating_sub(last_attempt_ms) >= ENCODING_RETRY_INTERVAL_MS
}
