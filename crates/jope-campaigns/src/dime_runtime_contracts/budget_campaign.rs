use super::*;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DlgpCampaignGateStateV1 {
    ReproductionRequired,
    LaunchGatesBlocked,
    Runnable,
    Complete,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DlgpActiveJobV1 {
    pub job_identity: String,
    pub trajectory_identity: String,
    pub attempt_number: u64,
    pub state: String,
    pub progress_evaluations: u64,
    pub reason: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct JopeDlgpBudgetCampaignStatusV1 {
    pub schema: String,
    pub generated_at_ms: u128,
    pub last_advanced_at_ms: u128,
    pub protocol_sha256: String,
    pub stage: String,
    pub gate_state: DlgpCampaignGateStateV1,
    pub trajectory_count: u64,
    pub completed_trajectory_count: u64,
    pub rung_counts: BTreeMap<String, u64>,
    pub evaluations_used: u64,
    pub evaluations_cap: u64,
    pub active_jobs: Vec<DlgpActiveJobV1>,
    pub robust_wins: u64,
    pub robust_wins_target: u64,
    pub robust_wins_denominator: u64,
    pub seed_agreement: String,
    pub license_track: String,
    pub production_compatible_state: String,
    pub research_only_state: String,
    pub test_opening_state: String,
    pub promotion_decisions: Vec<String>,
    pub blockers: Vec<String>,
}

impl JopeDlgpBudgetCampaignStatusV1 {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == JOPE_DLGP_BUDGET_CAMPAIGN_STATUS_SCHEMA,
            "budget campaign status schema is not v1"
        );
        require_sha256(&self.protocol_sha256, "campaign protocol")?;
        ensure!(
            self.generated_at_ms > 0
                && self.last_advanced_at_ms <= self.generated_at_ms
                && self.completed_trajectory_count <= self.trajectory_count
                && self.evaluations_used <= self.evaluations_cap
                && self.evaluations_cap == 200_000
                && self.robust_wins <= self.robust_wins_denominator
                && self.robust_wins_target == 7
                && self.robust_wins_denominator == 22,
            "budget campaign status accounting drift"
        );
        ensure!(
            matches!(
                self.license_track.as_str(),
                "production_compatible" | "research_only" | "separate"
            ),
            "budget campaign license track is invalid"
        );
        let identities = self
            .active_jobs
            .iter()
            .map(|job| job.job_identity.as_str())
            .collect::<BTreeSet<_>>();
        ensure!(
            identities.len() == self.active_jobs.len()
                && self.active_jobs.iter().all(|job| {
                    !job.job_identity.trim().is_empty()
                        && !job.trajectory_identity.trim().is_empty()
                        && job.attempt_number > 0
                }),
            "budget campaign active job identity is invalid or duplicated"
        );
        Ok(())
    }

    pub fn to_compact_json(&self) -> Result<Value> {
        self.validate()?;
        Ok(serde_json::json!({
            "schema": self.schema,
            "generated_at_ms": self.generated_at_ms,
            "last_advanced_at_ms": self.last_advanced_at_ms,
            "protocol_sha256": self.protocol_sha256,
            "stage": self.stage,
            "gate_state": self.gate_state,
            "trajectories": [self.completed_trajectory_count, self.trajectory_count],
            "rungs": self.rung_counts,
            "evaluations": [self.evaluations_used, self.evaluations_cap],
            "active_jobs": self.active_jobs,
            "robust_wins": [self.robust_wins, self.robust_wins_target, self.robust_wins_denominator],
            "seed_agreement": self.seed_agreement,
            "license_track": self.license_track,
            "test_opening_state": self.test_opening_state,
            "blockers": self.blockers,
        }))
    }
}

pub fn retry_backoff_ms(attempt_number: u64) -> u128 {
    match attempt_number {
        0 | 1 => 30_000,
        2 => 60_000,
        3 => 120_000,
        4 => 240_000,
        _ => 15 * 60 * 1_000,
    }
}
