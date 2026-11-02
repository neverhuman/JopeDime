use super::*;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DimeRecoveryProjectionV1 {
    pub schema: String,
    pub role: DimeRoleV1,
    pub scientific_payload: DimeScientificPayloadV1,
    pub scientific_payload_sha256: String,
    pub current_job_identity: String,
    pub current_attempt_number: u64,
    pub last_attempt_receipt_sha256: Option<String>,
    pub retry_not_before_ms: u128,
    pub retry_pending_submission: bool,
    pub submitted_at_ms: u128,
    pub scientific_failures: u32,
    pub foreign_cancellation_timestamps_ms: Vec<u128>,
    pub contention_alarm_open: bool,
    pub content_sha256: String,
}

impl DimeRecoveryProjectionV1 {
    pub fn initial(
        role: DimeRoleV1,
        scientific_payload: DimeScientificPayloadV1,
        current_job_identity: String,
    ) -> Result<Self> {
        let scientific_payload_sha256 = scientific_payload.identity_sha256()?;
        ensure!(
            !current_job_identity.trim().is_empty(),
            "recovery projection job identity is empty"
        );
        let mut value = Self {
            schema: DIME_RECOVERY_PROJECTION_SCHEMA.to_owned(),
            role,
            scientific_payload,
            scientific_payload_sha256,
            current_job_identity,
            current_attempt_number: 1,
            last_attempt_receipt_sha256: None,
            retry_not_before_ms: 0,
            retry_pending_submission: false,
            submitted_at_ms: 0,
            scientific_failures: 0,
            foreign_cancellation_timestamps_ms: Vec::new(),
            contention_alarm_open: false,
            content_sha256: String::new(),
        };
        value.refresh_content_sha256()?;
        Ok(value)
    }

    fn computed_content_sha256(&self) -> Result<String> {
        let mut copy = self.clone();
        copy.content_sha256.clear();
        canonical_sha256(&copy)
    }

    fn refresh_content_sha256(&mut self) -> Result<()> {
        self.content_sha256 = self.computed_content_sha256()?;
        self.validate()
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == DIME_RECOVERY_PROJECTION_SCHEMA,
            "recovery projection schema is not v1"
        );
        self.scientific_payload.validate()?;
        ensure!(
            self.scientific_payload.identity_sha256()? == self.scientific_payload_sha256,
            "recovery projection payload binding drift"
        );
        ensure!(
            !self.current_job_identity.trim().is_empty() && self.current_attempt_number > 0,
            "recovery projection job/attempt identity is invalid"
        );
        if let Some(receipt) = &self.last_attempt_receipt_sha256 {
            require_sha256(receipt, "recovery projection attempt receipt")?;
        }
        ensure!(
            self.foreign_cancellation_timestamps_ms
                .windows(2)
                .all(|pair| pair[0] <= pair[1]),
            "recovery projection cancellation timestamps are unsorted"
        );
        require_sha256(&self.content_sha256, "recovery projection content")?;
        ensure!(
            self.computed_content_sha256()? == self.content_sha256,
            "recovery projection content hash drift"
        );
        Ok(())
    }

    pub fn apply_receipt(&mut self, receipt: &DimeInfrastructureAttemptV1) -> Result<()> {
        receipt.validate()?;
        ensure!(
            receipt.role == self.role
                && receipt.scientific_payload_sha256 == self.scientific_payload_sha256
                && receipt.scientific_payload == self.scientific_payload,
            "attempt receipt does not belong to the recovery projection"
        );
        if self.last_attempt_receipt_sha256.as_deref() == Some(&receipt.content_sha256) {
            return self.validate();
        }
        ensure!(
            receipt.attempt_number == self.current_attempt_number
                && receipt.job_identity == self.current_job_identity
                && receipt.prior_attempt_receipt_sha256 == self.last_attempt_receipt_sha256,
            "attempt receipt cannot advance the recovery projection"
        );
        if receipt.terminal.class == DimeTerminalClassV1::ScientificFailure {
            self.scientific_failures = self.scientific_failures.saturating_add(1);
        }
        if receipt.terminal.foreign_cancellation {
            self.foreign_cancellation_timestamps_ms
                .push(receipt.terminal.observed_at_ms);
        }
        self.contention_alarm_open = self
            .foreign_cancellation_timestamps_ms
            .iter()
            .rev()
            .take_while(|timestamp| {
                receipt.terminal.observed_at_ms.saturating_sub(**timestamp) <= 10 * 60 * 1_000
            })
            .count()
            >= 3;
        self.last_attempt_receipt_sha256 = Some(receipt.content_sha256.clone());
        self.retry_not_before_ms = receipt.retry_not_before_ms;
        self.retry_pending_submission = receipt.next_job_identity.is_some();
        self.submitted_at_ms = 0;
        if let Some(next_job_identity) = &receipt.next_job_identity {
            self.current_job_identity = next_job_identity.clone();
            self.current_attempt_number = receipt.attempt_number.saturating_add(1);
        }
        self.refresh_content_sha256()
    }

    pub fn mark_submitted(&mut self, job_identity: &str, now_ms: u128) -> Result<()> {
        ensure!(
            self.retry_pending_submission
                && self.current_job_identity == job_identity
                && now_ms >= self.retry_not_before_ms,
            "recovery projection submission is not due or has the wrong identity"
        );
        self.retry_pending_submission = false;
        self.submitted_at_ms = now_ms;
        self.retry_not_before_ms = now_ms.saturating_add(self.next_backoff_ms(now_ms));
        self.refresh_content_sha256()
    }

    pub fn record_initial_submission(&mut self, job_identity: &str, now_ms: u128) -> Result<()> {
        ensure!(
            self.last_attempt_receipt_sha256.is_none()
                && self.current_attempt_number == 1
                && self.current_job_identity == job_identity
                && !self.retry_pending_submission
                && self.submitted_at_ms == 0,
            "initial recovery projection submission is duplicated or mismatched"
        );
        self.submitted_at_ms = now_ms;
        self.retry_not_before_ms = now_ms.saturating_add(retry_backoff_ms(1));
        self.refresh_content_sha256()
    }

    pub fn next_backoff_ms(&self, now_ms: u128) -> u128 {
        if self
            .foreign_cancellation_timestamps_ms
            .iter()
            .rev()
            .take_while(|timestamp| now_ms.saturating_sub(**timestamp) <= 10 * 60 * 1_000)
            .count()
            >= 3
        {
            15 * 60 * 1_000
        } else {
            retry_backoff_ms(self.current_attempt_number)
        }
    }

    pub fn refresh_contention_alarm(&mut self, now_ms: u128) -> Result<bool> {
        self.contention_alarm_open = self
            .foreign_cancellation_timestamps_ms
            .iter()
            .rev()
            .take_while(|timestamp| now_ms.saturating_sub(**timestamp) <= 10 * 60 * 1_000)
            .count()
            >= 3;
        self.refresh_content_sha256()?;
        Ok(self.contention_alarm_open)
    }
}

pub fn recovery_projection_path(
    root: &Path,
    role: DimeRoleV1,
    scientific_payload_sha256: &str,
) -> Result<PathBuf> {
    require_sha256(scientific_payload_sha256, "recovery projection payload")?;
    Ok(root
        .join("ops/dime-recovery-projections")
        .join(role.as_str())
        .join(format!("{scientific_payload_sha256}.v1.json")))
}

pub fn rebuild_recovery_projection(
    root: &Path,
    role: DimeRoleV1,
    scientific_payload: &DimeScientificPayloadV1,
    initial_job_identity: &str,
) -> Result<DimeRecoveryProjectionV1> {
    let mut projection = DimeRecoveryProjectionV1::initial(
        role,
        scientific_payload.clone(),
        initial_job_identity.to_owned(),
    )?;
    for receipt in infrastructure_attempts(root, role, &scientific_payload.identity_sha256()?)? {
        projection.apply_receipt(&receipt)?;
    }
    Ok(projection)
}

pub fn load_or_rebuild_recovery_projection(
    root: &Path,
    role: DimeRoleV1,
    scientific_payload: &DimeScientificPayloadV1,
    initial_job_identity: &str,
) -> Result<DimeRecoveryProjectionV1> {
    let rebuilt =
        rebuild_recovery_projection(root, role, scientific_payload, initial_job_identity)?;
    let path = recovery_projection_path(root, role, &rebuilt.scientific_payload_sha256)?;
    if path.is_file() {
        let stored: DimeRecoveryProjectionV1 = serde_json::from_slice(&fs::read(&path)?)?;
        stored.validate()?;
        // The ledger is authoritative. A stale projection is repaired from it;
        // it is never used to manufacture another terminal receipt.
        if stored.role == rebuilt.role
            && stored.scientific_payload == rebuilt.scientific_payload
            && stored.scientific_payload_sha256 == rebuilt.scientific_payload_sha256
            && stored.current_job_identity == rebuilt.current_job_identity
            && stored.current_attempt_number == rebuilt.current_attempt_number
            && stored.last_attempt_receipt_sha256 == rebuilt.last_attempt_receipt_sha256
            && stored.scientific_failures == rebuilt.scientific_failures
            && stored.foreign_cancellation_timestamps_ms
                == rebuilt.foreign_cancellation_timestamps_ms
        {
            return Ok(stored);
        }
    }
    save_recovery_projection(root, &rebuilt)?;
    Ok(rebuilt)
}

pub fn save_recovery_projection(
    root: &Path,
    projection: &DimeRecoveryProjectionV1,
) -> Result<PathBuf> {
    projection.validate()?;
    let path =
        recovery_projection_path(root, projection.role, &projection.scientific_payload_sha256)?;
    atomic_json(&path, &serde_json::to_value(projection)?)?;
    Ok(path)
}

/// Cancellation never consumes a scientific failure artifact. Only the
/// scheduler's exact `failed` state may be paired with deterministic evidence.
pub fn terminal_classification(
    scheduler_state: &str,
    deterministic_scientific_evidence_sha256: Option<&str>,
) -> Result<DimeTerminalClassV1> {
    if scheduler_state == "failed" {
        if let Some(evidence) = deterministic_scientific_evidence_sha256 {
            require_sha256(evidence, "deterministic scientific failure evidence")?;
            return Ok(DimeTerminalClassV1::ScientificFailure);
        }
    }
    Ok(DimeTerminalClassV1::InfrastructureFailure)
}
