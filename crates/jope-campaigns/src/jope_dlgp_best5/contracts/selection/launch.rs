use super::*;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LaunchEvidenceV1 {
    pub schema: String,
    pub scq_doctor_success: bool,
    pub evaluator_evidence_age_seconds: u64,
    pub supply_evidence_age_seconds: u64,
    pub max_evidence_age_seconds: u64,
    pub reviewed_action_acknowledged: bool,
    pub observed_project_priority_weight: u32,
    pub eligible_worker_disk_free_fractions: Vec<f64>,
}

impl LaunchEvidenceV1 {
    pub const SCHEMA: &'static str = "JopeDlgpBest5LaunchEvidence.v1";

    pub fn launch_allowed(&self) -> Result<bool, String> {
        self.launch_allowed_for(&ScqPolicyV1::frozen())
    }

    pub fn launch_allowed_for(&self, policy: &ScqPolicyV1) -> Result<bool, String> {
        schema(&self.schema, Self::SCHEMA)?;
        policy.validate()?;
        if self.max_evidence_age_seconds == 0 || self.eligible_worker_disk_free_fractions.is_empty()
        {
            return Err("launch evidence lacks freshness or eligible workers".into());
        }
        if self
            .eligible_worker_disk_free_fractions
            .iter()
            .any(|fraction| !fraction.is_finite() || !(0.0..=1.0).contains(fraction))
        {
            return Err("worker disk-free evidence is invalid".into());
        }
        Ok(self.scq_doctor_success
            && self.evaluator_evidence_age_seconds <= self.max_evidence_age_seconds
            && self.supply_evidence_age_seconds <= self.max_evidence_age_seconds
            && self.reviewed_action_acknowledged
            && self.observed_project_priority_weight == policy.priority_weight
            && self
                .eligible_worker_disk_free_fractions
                .iter()
                .all(|fraction| {
                    *fraction >= policy.min_worker_disk_free_fraction_micros as f64 / 1_000_000.0
                }))
    }

    pub fn disk_warning(&self) -> Result<bool, String> {
        schema(&self.schema, Self::SCHEMA)?;
        if self
            .eligible_worker_disk_free_fractions
            .iter()
            .any(|fraction| !fraction.is_finite() || !(0.0..=1.0).contains(fraction))
        {
            return Err("worker disk-free evidence is invalid".into());
        }
        Ok(self
            .eligible_worker_disk_free_fractions
            .iter()
            .any(|fraction| *fraction < 0.05))
    }
}
