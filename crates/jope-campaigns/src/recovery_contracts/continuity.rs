use super::*;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ContinuityNodeBundleReceipt {
    pub node: String,
    pub receipt_sha256: String,
    pub runtime_bundle_sha256: String,
    pub active_process_count: u64,
    pub verified: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ContinuityServiceReceipt {
    pub service: String,
    pub status_sha256: String,
    pub fresh: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveryContinuityObservationV1 {
    pub schema: String,
    pub generated_at_unix_ms: u64,
    pub freshness_budget_seconds: u64,
    pub input_digests: Vec<String>,
    pub incident_containment_sha256: String,
    pub quarantine_index_sha256: String,
    pub runtime_bundle_sha256: String,
    pub nodes: Vec<ContinuityNodeBundleReceipt>,
    pub active_process_bundle_sha256: Vec<String>,
    pub causal_evidence_sha256: String,
    pub causal_success: bool,
    pub telemetry_sha256: String,
    pub telemetry_rows_exact_and_fresh: bool,
    pub useful_work: u64,
    pub useful_work_target: u64,
    pub protected_complete_waves: u64,
    pub services: Vec<ContinuityServiceReceipt>,
    pub accepted_model_pointer_sha256: String,
    pub model_pointer_sha256: String,
    pub held_job_inventory_sha256: String,
    pub held_job_count: u64,
    pub ownerless_hold_count: u64,
    pub evaluation_cancellation_count_since_epoch: u64,
    pub v1_output_inventory_sha256: String,
    pub new_v1_output_count_since_epoch: u64,
    pub v1_quarantine_consumption_count_since_epoch: u64,
    pub expansion_active: bool,
    pub blockers: Vec<String>,
}

impl RecoveryContinuityObservationV1 {
    pub fn validate_at(&self, now_unix_ms: u64) -> Result<()> {
        ensure!(
            self.schema == RECOVERY_CONTINUITY_OBSERVATION_SCHEMA,
            "invalid recovery-continuity observation schema"
        );
        ensure!(
            self.generated_at_unix_ms > 0
                && self.generated_at_unix_ms <= now_unix_ms
                && self.freshness_budget_seconds > 0
                && self.freshness_budget_seconds <= 120
                && now_unix_ms - self.generated_at_unix_ms
                    <= self.freshness_budget_seconds.saturating_mul(1_000),
            "recovery-continuity observation is absent, future-dated, or stale"
        );
        ensure!(
            self.nodes.len() == REQUIRED_NODES.len(),
            "continuity node receipts are incomplete"
        );
        let mut expected_inputs = BTreeSet::new();
        for digest in [
            &self.incident_containment_sha256,
            &self.quarantine_index_sha256,
            &self.causal_evidence_sha256,
            &self.telemetry_sha256,
            &self.accepted_model_pointer_sha256,
            &self.model_pointer_sha256,
            &self.held_job_inventory_sha256,
            &self.v1_output_inventory_sha256,
        ] {
            require_sha256(digest, "continuity evidence digest")?;
            expected_inputs.insert(digest.clone());
        }
        require_sha256(&self.runtime_bundle_sha256, "continuity runtime bundle")?;
        expected_inputs.insert(self.runtime_bundle_sha256.clone());
        let mut nodes = BTreeSet::new();
        for receipt in &self.nodes {
            ensure!(
                REQUIRED_NODES.contains(&receipt.node.as_str())
                    && nodes.insert(receipt.node.as_str()),
                "continuity node receipt set is invalid"
            );
            require_sha256(&receipt.receipt_sha256, "continuity node receipt")?;
            ensure!(
                receipt.runtime_bundle_sha256 == self.runtime_bundle_sha256
                    && receipt.active_process_count > 0
                    && receipt.verified,
                "continuity node does not prove the common runtime bundle"
            );
            expected_inputs.insert(receipt.receipt_sha256.clone());
        }
        ensure!(
            self.active_process_bundle_sha256.len() >= REQUIRED_NODES.len()
                && self
                    .active_process_bundle_sha256
                    .iter()
                    .all(|digest| digest == &self.runtime_bundle_sha256),
            "active processes do not resolve to one runtime bundle"
        );
        ensure!(
            self.services.len() == CONTINUITY_SERVICES.len(),
            "continuity service receipts are incomplete"
        );
        for (receipt, expected) in self.services.iter().zip(CONTINUITY_SERVICES) {
            ensure!(
                receipt.service == expected && receipt.fresh,
                "continuity service is absent, out of order, or stale: {expected}"
            );
            require_sha256(&receipt.status_sha256, "continuity service status")?;
            expected_inputs.insert(receipt.status_sha256.clone());
        }
        ensure!(
            self.causal_success && self.telemetry_rows_exact_and_fresh,
            "causal evidence or exact four-task telemetry is not continuously green"
        );
        ensure!(
            self.useful_work >= self.useful_work_target
                && self.useful_work_target >= 6_144
                && self.protected_complete_waves >= 2,
            "useful-work target or protected reserves are not continuously satisfied"
        );
        ensure!(
            self.held_job_count == 3
                && self.ownerless_hold_count == 0
                && self.evaluation_cancellation_count_since_epoch == 0,
            "held-job ownership or evaluation-cancellation invariant failed"
        );
        ensure!(
            self.new_v1_output_count_since_epoch == 0
                && self.v1_quarantine_consumption_count_since_epoch == 0
                && !self.expansion_active,
            "V1 output, quarantine consumption, or expansion activation observed"
        );
        ensure!(
            self.blockers.is_empty(),
            "continuity observation has blockers"
        );
        ensure!(
            self.input_digests.windows(2).all(|pair| pair[0] < pair[1])
                && self.input_digests.iter().all(|digest| is_sha256(digest))
                && self.input_digests.iter().cloned().collect::<BTreeSet<_>>() == expected_inputs,
            "continuity input digest set is incomplete or noncanonical"
        );
        Ok(())
    }

    pub fn epoch_identity_sha256(&self, now_unix_ms: u64) -> Result<String> {
        self.validate_at(now_unix_ms)?;
        canonical_json_sha256(&json!({
            "domain": "jope-recovery-continuity-epoch-v1",
            "incident_containment_sha256": self.incident_containment_sha256,
            "quarantine_index_sha256": self.quarantine_index_sha256,
            "runtime_bundle_sha256": self.runtime_bundle_sha256,
            "accepted_model_pointer_sha256": self.accepted_model_pointer_sha256,
            "model_pointer_sha256": self.model_pointer_sha256,
            "held_job_inventory_sha256": self.held_job_inventory_sha256,
            "v1_output_inventory_sha256": self.v1_output_inventory_sha256,
        }))
    }
}
