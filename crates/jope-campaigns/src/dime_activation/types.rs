use super::*;

pub const ACTIVATION_TRANSACTION_SCHEMA: &str = "JopeDimeActivationTransaction.v1";
pub const EXTERNAL_ACTIVATION_GATE_SCHEMA: &str = "JopeDimeExternalActivationGate.v1";
pub const SCQ_CAPABILITY_RECEIPT_SCHEMA: &str = "JopeScqCapabilityReceipt.v1";
pub const CANCELLATION_OWNERSHIP_SCHEMA: &str = "JopeCancellationOwnershipEvidence.v1";
pub const CANDIDATE_REVIEW_GATE_SCHEMA: &str = "JopeDimeCandidateReviewGate.v1";
pub const CANDIDATE_REVIEW_ACK_SCHEMA: &str = "JopeDimeCandidateReviewAcknowledgement.v1";
pub const CANDIDATE_TEST_RUN_SCHEMA: &str = "JopeDimeCandidateTestRun.v1";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivationOperationV1 {
    Activate,
    Rollback,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivationStageV1 {
    Prepared,
    CandidateValidated,
    UnitsStaged,
    UnitsInstalled,
    RuntimeSwitched,
    ActivePublished,
    DaemonReloaded,
    TimersEnabled,
    ServicesRestarted,
    Complete,
    Aborted,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScqCapabilityObservationV1 {
    pub schema: String,
    pub node: String,
    pub observed_hostname: String,
    pub scheduler_binary_sha256: String,
    pub eligible_for_dime_gpu: bool,
    pub gpu_companion_core_reserve: Option<u64>,
    pub observed_at_unix_ms: u64,
    pub raw_output: RawEvidenceRefV2,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CancellationOwnershipEvidenceV1 {
    pub schema: String,
    pub owner: String,
    pub cancellation_message: String,
    pub source: String,
    pub observed_at_unix_ms: u64,
    pub raw_events: RawEvidenceRefV2,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DimeExternalActivationGateV1 {
    pub schema: String,
    pub candidate_bundle_sha256: String,
    pub generated_at_unix_ms: u64,
    pub expires_at_unix_ms: u64,
    pub scq_capability_receipts: Vec<JsonEvidenceRefV2>,
    pub cancellation_ownership: JsonEvidenceRefV2,
}

impl DimeExternalActivationGateV1 {
    pub fn validate_at(&self, evidence_root: &Path, now_unix_ms: u64) -> Result<()> {
        ensure!(
            self.schema == EXTERNAL_ACTIVATION_GATE_SCHEMA,
            "invalid DIME external activation-gate schema"
        );
        require_sha256(&self.candidate_bundle_sha256, "external gate candidate")?;
        ensure!(
            self.generated_at_unix_ms > 0
                && self.generated_at_unix_ms <= now_unix_ms
                && now_unix_ms < self.expires_at_unix_ms
                && self.expires_at_unix_ms - self.generated_at_unix_ms <= 60 * 60 * 1_000,
            "external activation gate is future-dated, stale, or overlong"
        );
        ensure!(
            self.scq_capability_receipts.len() == REQUIRED_NODES.len(),
            "external activation gate lacks five-node SCQ evidence"
        );
        let mut nodes = BTreeSet::new();
        for reference in &self.scq_capability_receipts {
            let receipt: ScqCapabilityObservationV1 = reference.load(evidence_root)?;
            ensure!(
                receipt.schema == SCQ_CAPABILITY_RECEIPT_SCHEMA
                    && REQUIRED_NODES.contains(&receipt.node.as_str())
                    && receipt.observed_hostname.split('.').next() == Some(receipt.node.as_str())
                    && receipt.observed_at_unix_ms >= self.generated_at_unix_ms
                    && receipt.observed_at_unix_ms < self.expires_at_unix_ms,
                "SCQ capability receipt has invalid node, host, or time scope"
            );
            require_sha256(&receipt.scheduler_binary_sha256, "SCQ scheduler binary")?;
            if receipt.eligible_for_dime_gpu {
                ensure!(
                    receipt
                        .gpu_companion_core_reserve
                        .is_some_and(|value| value >= 4),
                    "eligible node does not expose gpu_companion_core_reserve >= 4"
                );
            }
            receipt.raw_output.load(evidence_root)?;
            ensure!(
                nodes.insert(receipt.node),
                "duplicate SCQ capability node receipt"
            );
        }
        ensure!(
            REQUIRED_NODES.iter().all(|node| nodes.contains(*node)),
            "SCQ capability node set is incomplete"
        );
        let ownership: CancellationOwnershipEvidenceV1 =
            self.cancellation_ownership.load(evidence_root)?;
        ensure!(
            ownership.schema == CANCELLATION_OWNERSHIP_SCHEMA
                && !ownership.owner.trim().is_empty()
                && !ownership.source.trim().is_empty()
                && ownership.cancellation_message == "cancelled by operator"
                && ownership.observed_at_unix_ms >= self.generated_at_unix_ms
                && ownership.observed_at_unix_ms < self.expires_at_unix_ms,
            "cancellation ownership is absent, stale, or not tied to the operator message"
        );
        ownership.raw_events.load(evidence_root)?;
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DimeCandidateTestRunV1 {
    pub schema: String,
    pub candidate_bundle_sha256: String,
    pub integration_commit: String,
    pub command: Vec<String>,
    pub exit_status: i32,
    pub passed: u64,
    pub failed: u64,
    pub ignored: u64,
    pub started_at_unix_ms: u64,
    pub finished_at_unix_ms: u64,
    pub stdout: RawEvidenceRefV2,
    pub stderr: RawEvidenceRefV2,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DimeCandidateReviewAcknowledgementV1 {
    pub schema: String,
    pub reviewer: String,
    pub candidate_bundle_sha256: String,
    pub integration_commit: String,
    pub bundle_manifest_file_sha256: String,
    pub bundle_manifest_canonical_json_sha256: String,
    pub test_runs: Vec<JsonEvidenceRefV2>,
    pub reviewed_at_unix_ms: u64,
    pub raw_review: RawEvidenceRefV2,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DimeCandidateReviewGateV1 {
    pub schema: String,
    pub issued_at_unix_ms: u64,
    pub expires_at_unix_ms: u64,
    pub candidate_bundle_sha256: String,
    pub integration_commit: String,
    pub bundle_manifest_file_sha256: String,
    pub bundle_manifest_canonical_json_sha256: String,
    pub acknowledgements: Vec<JsonEvidenceRefV2>,
}

impl DimeCandidateReviewGateV1 {
    pub fn validate_at(&self, evidence_root: &Path, now_unix_ms: u64) -> Result<()> {
        ensure!(
            self.schema == CANDIDATE_REVIEW_GATE_SCHEMA,
            "invalid candidate review gate schema"
        );
        require_sha256(&self.candidate_bundle_sha256, "reviewed candidate")?;
        require_sha256(&self.bundle_manifest_file_sha256, "reviewed manifest file")?;
        require_sha256(
            &self.bundle_manifest_canonical_json_sha256,
            "reviewed manifest canonical JSON",
        )?;
        require_commit(&self.integration_commit)?;
        ensure!(
            self.issued_at_unix_ms > 0
                && self.issued_at_unix_ms <= now_unix_ms
                && now_unix_ms < self.expires_at_unix_ms
                && self.expires_at_unix_ms - self.issued_at_unix_ms <= 24 * 60 * 60 * 1_000,
            "candidate review gate is future-dated, stale, or overlong"
        );
        ensure!(
            self.acknowledgements.len() == 2,
            "candidate review gate requires exactly two acknowledgements"
        );
        let mut reviewers = BTreeSet::new();
        let mut evidence_paths = BTreeSet::new();
        let mut test_run_paths_by_reviewer = Vec::new();
        for reference in &self.acknowledgements {
            ensure!(
                evidence_paths.insert(reference.path.clone()),
                "candidate review acknowledgement is duplicated"
            );
            let acknowledgement: DimeCandidateReviewAcknowledgementV1 =
                reference.load(evidence_root)?;
            ensure!(
                acknowledgement.schema == CANDIDATE_REVIEW_ACK_SCHEMA
                    && matches!(acknowledgement.reviewer.as_str(), "codex" | "claude")
                    && acknowledgement.candidate_bundle_sha256 == self.candidate_bundle_sha256
                    && acknowledgement.integration_commit == self.integration_commit
                    && acknowledgement.bundle_manifest_file_sha256
                        == self.bundle_manifest_file_sha256
                    && acknowledgement.bundle_manifest_canonical_json_sha256
                        == self.bundle_manifest_canonical_json_sha256
                    && acknowledgement.reviewed_at_unix_ms >= self.issued_at_unix_ms
                    && acknowledgement.reviewed_at_unix_ms < self.expires_at_unix_ms,
                "candidate review acknowledgement scope mismatch"
            );
            ensure!(
                reviewers.insert(acknowledgement.reviewer.clone()),
                "candidate reviewer is duplicated"
            );
            acknowledgement.raw_review.load(evidence_root)?;
            ensure!(
                !acknowledgement.test_runs.is_empty(),
                "candidate reviewer supplied no reproduced test runs"
            );
            let mut reviewer_test_paths = BTreeSet::new();
            for test_reference in &acknowledgement.test_runs {
                ensure!(
                    reviewer_test_paths.insert(test_reference.path.clone()),
                    "candidate reviewer duplicated a test-run receipt"
                );
                let run: DimeCandidateTestRunV1 = test_reference.load(evidence_root)?;
                ensure!(
                    run.schema == CANDIDATE_TEST_RUN_SCHEMA
                        && run.candidate_bundle_sha256 == self.candidate_bundle_sha256
                        && run.integration_commit == self.integration_commit
                        && !run.command.is_empty()
                        && run
                            .command
                            .iter()
                            .all(|argument| !argument.trim().is_empty())
                        && run.exit_status == 0
                        && run.passed > 0
                        && run.failed == 0
                        && run.started_at_unix_ms >= self.issued_at_unix_ms
                        && run.finished_at_unix_ms >= run.started_at_unix_ms
                        && run.finished_at_unix_ms <= acknowledgement.reviewed_at_unix_ms,
                    "candidate test run is unbound, unsuccessful, or lacks exact counts"
                );
                run.stdout.load(evidence_root)?;
                run.stderr.load(evidence_root)?;
            }
            test_run_paths_by_reviewer.push(reviewer_test_paths);
        }
        ensure!(
            reviewers == BTreeSet::from(["claude".to_owned(), "codex".to_owned()]),
            "candidate review gate does not contain both independent reviewers"
        );
        ensure!(
            test_run_paths_by_reviewer[0].is_disjoint(&test_run_paths_by_reviewer[1]),
            "candidate reviewers did not independently reproduce test evidence"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ActivationUnitCandidateV1 {
    pub unit: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ActivationDropInPreimageV1 {
    pub relative_path: String,
    pub file: RawEvidenceRefV2,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ActivationUnitPreimageV1 {
    pub unit: String,
    pub base_file: Option<RawEvidenceRefV2>,
    pub drop_ins: Vec<ActivationDropInPreimageV1>,
    pub enabled: bool,
    pub active: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ActivationPreimageV1 {
    pub active_json: Option<RawEvidenceRefV2>,
    pub current_symlink_target: Option<String>,
    pub units: Vec<ActivationUnitPreimageV1>,
    pub recovery_tool: RawEvidenceRefV2,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DimeActivationTransactionV1 {
    pub schema: String,
    pub transaction_id: String,
    pub operation: ActivationOperationV1,
    pub stage: ActivationStageV1,
    pub candidate_bundle_sha256: String,
    pub rollback_capsule_sha256: String,
    pub candidate_review_gate_sha256: Option<String>,
    pub external_gate_sha256: Option<String>,
    pub active_document: Value,
    pub preimage: ActivationPreimageV1,
    pub candidate_units: Vec<ActivationUnitCandidateV1>,
    pub candidate_timers: Vec<String>,
    pub affected_services: Vec<String>,
    pub protected_science_sha256: BTreeMap<String, String>,
}

pub trait ActivationHost {
    fn is_enabled(&mut self, unit: &str) -> Result<bool>;
    fn is_active(&mut self, unit: &str) -> Result<bool>;
    fn validate_candidate(&mut self, staged_root: &Path, units: &[String]) -> Result<()>;
    fn daemon_reload(&mut self) -> Result<()>;
    fn enable_timers(&mut self, timers: &[String]) -> Result<()>;
    fn try_restart_services(&mut self, services: &[String]) -> Result<()>;
    fn restore_states(&mut self, units: &[ActivationUnitPreimageV1]) -> Result<()>;
}

pub struct ActivationRequest<'a> {
    pub output_root: &'a Path,
    pub user_unit_root: &'a Path,
    pub transaction_root: &'a Path,
    pub candidate_bundle_sha256: &'a str,
    pub rollback_capsule_sha256: &'a str,
    pub candidate_review_gate_sha256: Option<&'a str>,
    pub external_gate_sha256: Option<&'a str>,
    pub active_document: Value,
    pub operation: ActivationOperationV1,
}
