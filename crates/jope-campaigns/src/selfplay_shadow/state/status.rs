use super::*;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceIdentity {
    pub service: String,
    pub bundle_sha256: Sha256Digest,
    pub configuration_sha256: Sha256Digest,
    pub evaluator_sha256: Sha256Digest,
    pub started_at_ms: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeBundleV3Identity {
    pub schema: String,
    pub bundle_sha256: Sha256Digest,
    pub configuration_sha256: Sha256Digest,
    pub evaluator_sha256: Sha256Digest,
    pub services: Vec<ServiceIdentity>,
}

impl RuntimeBundleV3Identity {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == RUNTIME_BUNDLE_SCHEMA,
            "runtime bundle schema is not v3"
        );
        ensure!(!self.services.is_empty(), "runtime bundle has no services");
        let mut names = BTreeSet::new();
        for service in &self.services {
            ensure!(
                !service.service.trim().is_empty() && names.insert(service.service.clone()),
                "runtime service names must be nonempty and unique"
            );
            ensure!(
                service.started_at_ms > 0,
                "service start identity is absent"
            );
            ensure!(
                service.bundle_sha256 == self.bundle_sha256
                    && service.configuration_sha256 == self.configuration_sha256
                    && service.evaluator_sha256 == self.evaluator_sha256,
                "runtime service identity drift"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenRunIdentity {
    pub checkpoint_sha256: Sha256Digest,
    pub corpus_sha256: Sha256Digest,
    pub catalog_sha256: Sha256Digest,
    pub evaluator_sha256: Sha256Digest,
    pub proposal_sha256: Sha256Digest,
    pub context_sha256: Sha256Digest,
}

impl FrozenRunIdentity {
    pub fn validate_observed(&self, observed: &Self) -> Result<()> {
        ensure!(
            self == observed,
            "checkpoint, corpus, catalog, evaluator, proposal, or context drift"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ActionAccounting {
    pub direct_actions: usize,
    pub direct_evaluations: usize,
    pub direct_strict_wins: usize,
    pub matched_gp_evaluations: usize,
    pub matched_gp_strict_wins: usize,
    pub descendant_evaluations: usize,
    pub requested_d: usize,
    pub actual_d: usize,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CacheAccounting {
    pub exact_hits: usize,
    pub avoided_evaluations: usize,
    pub active_conflicts: usize,
    pub cross_protocol_conflicts: usize,
    pub requalifications: usize,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TaskCoverage {
    pub registry: usize,
    pub encoded: usize,
    pub active: usize,
    pub observed: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ShadowStatusV2 {
    pub schema: String,
    pub binary_front: AcceptedFront,
    pub regression_front: AcceptedFront,
    pub binary_actions: ActionAccounting,
    pub regression_actions: ActionAccounting,
    pub cache: CacheAccounting,
    pub binary_coverage: TaskCoverage,
    pub regression_coverage: TaskCoverage,
    pub frontier_contexts: usize,
    pub hard_contexts: usize,
    pub anchor_contexts: usize,
    pub proposal_validity: f64,
    pub proposal_uniqueness: f64,
    pub selector_calibrated: bool,
    pub zero_backfill: bool,
    pub corpus_fresh_labels: usize,
    pub parent_evidence_threshold: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_trainer_task: Option<Task>,
    pub proposer_active: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queue_hold_reason: Option<String>,
}

impl ShadowStatusV2 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == "SelfPlayShadowStatus.v2",
            "status schema is not v2"
        );
        self.binary_front.validate(Task::Binary)?;
        self.regression_front.validate(Task::Regression)?;
        ensure!(
            self.frontier_contexts + self.hard_contexts + self.anchor_contexts
                == super::DATASETS_PER_COHORT,
            "status cohort accounting does not total 64"
        );
        ensure!(
            self.proposal_validity.is_finite()
                && self.proposal_uniqueness.is_finite()
                && (0.0..=1.0).contains(&self.proposal_validity)
                && (0.0..=1.0).contains(&self.proposal_uniqueness),
            "proposal metrics are invalid"
        );
        ensure!(self.zero_backfill, "zero-backfill assertion failed");
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RolloutEvidence {
    pub consecutive_accepted_generations: usize,
    pub zero_backfill: bool,
    pub validity: f64,
    pub complete_lineage: bool,
    pub aggregate_dl_gp_lower_95: f64,
}

pub fn canary_ready(evidence: RolloutEvidence) -> bool {
    evidence.consecutive_accepted_generations >= 3
        && evidence.zero_backfill
        && evidence.validity >= 0.99
        && evidence.complete_lineage
        && evidence.aggregate_dl_gp_lower_95 > 1.0
}

pub fn canary_rung_ready(waves: usize, distinct_datasets: usize) -> bool {
    waves >= 3 && distinct_datasets >= 256
}

pub fn stable_canary_bucket(source_family: &str) -> u8 {
    let digest = Sha256::digest(format!("selfplay-canary-v2|{source_family}").as_bytes());
    digest[0] % 100
}
