use super::*;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum PersistedSupplyStage {
    FreezeCohort,
    SubmitCohortProposer,
    FetchValidateProposals,
    CacheFilterResample,
    BuildWave,
    KeeperAcknowledgment,
}

impl PersistedSupplyStage {
    pub(super) fn ordinal(self) -> u8 {
        match self {
            Self::FreezeCohort => 0,
            Self::SubmitCohortProposer => 1,
            Self::FetchValidateProposals => 2,
            Self::CacheFilterResample => 3,
            Self::BuildWave => 4,
            Self::KeeperAcknowledgment => 5,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PersistedSupplyState {
    pub(super) schema: String,
    pub(super) wave: String,
    pub(super) checkpoint_sha256: String,
    pub(super) cohort: PathBuf,
    pub(super) cohort_sha256: String,
    pub(super) catalog_sha256: String,
    pub(super) decoder_configuration_sha256: String,
    pub(super) seed: u64,
    #[serde(default = "default_proposal_reservoir")]
    pub(super) proposal_reservoir_per_context: usize,
    pub(super) stage: PersistedSupplyStage,
    pub(super) revision: u64,
    pub(super) recovery_accounting_only: bool,
    #[serde(default)]
    pub(super) proposal_attempt: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) proposal_job_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) cohort_proposal_pointer: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) cohort_proposal_pointer_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) sanitized_proposal_pointer: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) sanitized_proposal_pointer_sha256: Option<String>,
    #[serde(default)]
    pub(super) runtime_pinned_attempts: Vec<RuntimePinnedIntentV1>,
}

fn default_proposal_reservoir() -> usize {
    64
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PinnedProposerInput {
    pub(super) name: String,
    pub(super) path: PathBuf,
    pub(super) sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RuntimePinnedIntentV1 {
    pub(super) schema: String,
    pub(super) runtime_bundle: PathBuf,
    pub(super) runtime_bundle_sha256: String,
    pub(super) catalog_snapshot: PathBuf,
    pub(super) catalog_snapshot_sha256: String,
    pub(super) proposer_inputs: Vec<PinnedProposerInput>,
    pub(super) immutable_spec_path: PathBuf,
    pub(super) immutable_spec_sha256: String,
    pub(super) job_identity: String,
    pub(super) attempt: u32,
}

impl RuntimePinnedIntentV1 {
    pub(super) fn validate(&self) -> Result<()> {
        if self.schema != RUNTIME_PINNED_INTENT_SCHEMA {
            bail!("runtime-pinned proposer intent schema is not v1");
        }
        validate_sha256(&self.runtime_bundle_sha256)?;
        validate_sha256(&self.catalog_snapshot_sha256)?;
        validate_sha256(&self.immutable_spec_sha256)?;
        if self.job_identity.trim().is_empty() {
            bail!("runtime-pinned proposer job identity is empty");
        }
        if !self.runtime_bundle.is_dir()
            || self
                .runtime_bundle
                .file_name()
                .and_then(|value| value.to_str())
                != Some(self.runtime_bundle_sha256.as_str())
        {
            bail!("runtime-pinned proposer bundle identity drift");
        }
        let bundle: Value =
            serde_json::from_slice(&fs::read(self.runtime_bundle.join("bundle.json"))?)?;
        if bundle.get("bundle_sha256").and_then(Value::as_str)
            != Some(self.runtime_bundle_sha256.as_str())
        {
            bail!("runtime-pinned proposer bundle manifest drift");
        }
        if !self.immutable_spec_path.is_file()
            || sha256_file(&self.immutable_spec_path)? != self.immutable_spec_sha256
        {
            bail!("runtime-pinned proposer immutable spec drift");
        }
        if self.proposer_inputs.is_empty() {
            bail!("runtime-pinned proposer has no immutable inputs");
        }
        let mut names = BTreeSet::new();
        for input in &self.proposer_inputs {
            validate_sha256(&input.sha256)?;
            if input.name.trim().is_empty()
                || !names.insert(input.name.as_str())
                || !input.path.is_file()
                || sha256_file(&input.path)? != input.sha256
            {
                bail!("runtime-pinned proposer input identity drift");
            }
        }
        if !names.contains("generator")
            || !names.contains("common")
            || !names.contains("pointer")
            || !names.contains("catalog")
            || !names.contains("data")
            || !names.contains("checkpoint")
            || !names.contains("cohort")
        {
            bail!("runtime-pinned proposer input set is incomplete");
        }
        let pointer = self
            .proposer_inputs
            .iter()
            .find(|input| input.name == "pointer")
            .context("runtime-pinned proposer has no catalog snapshot input")?;
        if pointer.path != self.catalog_snapshot || pointer.sha256 != self.catalog_snapshot_sha256 {
            bail!("runtime-pinned proposer catalog snapshot drift");
        }
        Ok(())
    }
}

impl PersistedSupplyState {
    pub(super) fn validate(&self, expected_wave: &str, expected_checkpoint: &str) -> Result<()> {
        if !matches!(
            self.schema.as_str(),
            SUPPLY_STATE_V1_SCHEMA | SUPPLY_STATE_V2_SCHEMA
        ) || self.wave != expected_wave
            || self.checkpoint_sha256 != expected_checkpoint
        {
            bail!("persisted cohort supply identity drift");
        }
        validate_sha256(&self.cohort_sha256)?;
        validate_sha256(&self.catalog_sha256)?;
        validate_sha256(&self.decoder_configuration_sha256)?;
        if !matches!(self.proposal_reservoir_per_context, 64 | 128 | 256) {
            bail!("cohort proposal reservoir must be 64, 128, or 256");
        }
        if self.stage.ordinal() >= PersistedSupplyStage::SubmitCohortProposer.ordinal()
            && self.proposal_job_id.is_none()
        {
            bail!("submitted cohort supply state lacks a proposer job identity");
        }
        if self.stage.ordinal() >= PersistedSupplyStage::FetchValidateProposals.ordinal()
            && (self.cohort_proposal_pointer.is_none()
                || self.cohort_proposal_pointer_sha256.is_none())
        {
            bail!("fetched cohort supply state lacks an immutable proposal pointer");
        }
        if self.stage.ordinal() >= PersistedSupplyStage::CacheFilterResample.ordinal()
            && (self.sanitized_proposal_pointer.is_none()
                || self.sanitized_proposal_pointer_sha256.is_none())
        {
            bail!("cache-filtered supply state lacks its proposal pointer");
        }
        for (index, intent) in self.runtime_pinned_attempts.iter().enumerate() {
            intent.validate()?;
            if intent.attempt as usize != index {
                bail!("runtime-pinned proposer attempts are not contiguous");
            }
        }
        if self.schema == SUPPLY_STATE_V2_SCHEMA
            && self.stage.ordinal() >= PersistedSupplyStage::SubmitCohortProposer.ordinal()
        {
            let intent = self
                .runtime_pinned_attempts
                .last()
                .context("submitted v2 supply state lacks a runtime-pinned intent")?;
            if Some(intent.job_identity.as_str()) != self.proposal_job_id.as_deref()
                || intent.attempt != self.proposal_attempt
            {
                bail!("submitted v2 supply state proposer attempt identity drift");
            }
        }
        Ok(())
    }

    pub(super) fn advance(&mut self, next: PersistedSupplyStage) -> Result<()> {
        if next.ordinal() != self.stage.ordinal() + 1 {
            bail!("cohort supply stage must advance exactly once");
        }
        self.stage = next;
        self.revision += 1;
        self.validate(&self.wave, &self.checkpoint_sha256)
    }

    pub(super) fn retry_proposer(
        &mut self,
        root: &Path,
        terminal_state: &str,
        observed_at_ms: u128,
    ) -> Result<ProposerRetryDisposition> {
        if self.stage != PersistedSupplyStage::SubmitCohortProposer {
            bail!("cohort proposer retry is allowed only in its submitted stage");
        }
        let prior = self
            .runtime_pinned_attempts
            .last()
            .cloned()
            .context("cohort proposer retry lacks its pinned original spec")?;
        prior.validate()?;
        let scientific_payload = proposer_scientific_payload(self, &prior)?;
        let mut projection = proposer_recovery_projection(root, self, &scientific_payload)?;
        let replayed_receipt =
            self.proposal_job_id.as_deref() != Some(projection.current_job_identity.as_str());
        reconcile_proposer_projection(root, self, &projection)?;
        if replayed_receipt {
            self.revision = self.revision.saturating_add(1);
            self.validate(&self.wave, &self.checkpoint_sha256)?;
            return Ok(ProposerRetryDisposition::Retried);
        }
        if projection.scientific_failures >= 5 {
            return Ok(ProposerRetryDisposition::ScientificCircuitOpen);
        }
        if projection.retry_pending_submission {
            return Ok(ProposerRetryDisposition::Backoff);
        }
        if projection.submitted_at_ms > 0 && observed_at_ms < projection.retry_not_before_ms {
            return Ok(ProposerRetryDisposition::Backoff);
        }
        if let Some(drain) = matching_drain_intent(
            root,
            DimeRoleV1::CohortProposer,
            &scientific_payload,
            observed_at_ms,
        )? {
            append_supply_event(
                root,
                "intentional_drain_honored",
                &json!({
                    "job_id": self.proposal_job_id,
                    "attempt": self.proposal_attempt,
                    "drain_receipt_sha256": drain.content_sha256,
                    "scientific_payload_sha256": scientific_payload.identity_sha256()?,
                }),
            )?;
            return Ok(ProposerRetryDisposition::Drained);
        }
        let current = projection.current_job_identity.clone();
        ensure!(
            self.proposal_job_id.as_deref() == Some(current.as_str()),
            "primary proposer state is not reconciled with its recovery projection"
        );
        let scientific = if terminal_state == "failed" {
            scientific_failure_receipt(root, &current, &scientific_payload)?
        } else {
            None
        };
        let terminal_class = terminal_classification(terminal_state, scientific.as_deref())?;
        let current_attempt_number = projection.current_attempt_number;
        let base = current
            .rsplit_once("-infra")
            .filter(|(_, suffix)| suffix.bytes().all(|byte| byte.is_ascii_digit()))
            .map(|(base, _)| base)
            .or_else(|| {
                current
                    .rsplit_once("-retry")
                    .filter(|(_, suffix)| suffix.bytes().all(|byte| byte.is_ascii_digit()))
                    .map(|(base, _)| base)
            })
            .unwrap_or(&current)
            .to_owned();
        let next_attempt_number = current_attempt_number
            .checked_add(1)
            .context("cohort proposer infrastructure attempt overflow")?;
        let next_job_identity = format!("{base}-infra{next_attempt_number:06}");
        let scientific_failures = projection.scientific_failures.saturating_add(u32::from(
            terminal_class == DimeTerminalClassV1::ScientificFailure,
        ));
        let retry_allowed = scientific_failures < 5;
        let evidence_sha256 = scientific.unwrap_or_else(|| {
            sha256_bytes(
                format!(
                    "dime-proposer-terminal-v1\0{}\0{}\0{}",
                    current, terminal_state, prior.immutable_spec_sha256
                )
                .as_bytes(),
            )
        });
        let recent_cancellations = projection
            .foreign_cancellation_timestamps_ms
            .iter()
            .rev()
            .take_while(|timestamp| observed_at_ms.saturating_sub(**timestamp) <= 10 * 60 * 1_000)
            .count()
            + usize::from(terminal_state == "cancelled");
        let backoff_ms = if recent_cancellations >= 3 {
            15 * 60 * 1_000
        } else {
            jope_dime_regressor::dime_runtime_contracts::retry_backoff_ms(current_attempt_number)
        };
        let retry_not_before_ms = observed_at_ms.saturating_add(backoff_ms);
        let receipt = DimeInfrastructureAttemptV1::new_terminal(
            DimeRoleV1::CohortProposer,
            scientific_payload,
            current_attempt_number,
            current.clone(),
            retry_allowed.then_some(next_job_identity.clone()),
            projection.last_attempt_receipt_sha256.clone(),
            prior
                .immutable_spec_path
                .metadata()
                .and_then(|metadata| metadata.modified())
                .ok()
                .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(1, |duration| duration.as_millis()),
            retry_not_before_ms,
            DimeTerminalEvidenceV1 {
                terminal_state: terminal_state.to_owned(),
                class: terminal_class,
                evidence_sha256,
                observed_at_ms,
                foreign_cancellation: terminal_state == "cancelled",
            },
        )?;
        append_infrastructure_attempt(root, &receipt)?;
        projection.apply_receipt(&receipt)?;
        let contention_alarm_open = projection.refresh_contention_alarm(observed_at_ms)?;
        save_recovery_projection(root, &projection)?;
        if contention_alarm_open {
            append_supply_event(
                root,
                "foreign_cancellation_contention_alarm",
                &json!({
                    "severity": "prominent",
                    "job_id": current,
                    "scientific_payload_sha256": projection.scientific_payload_sha256,
                    "retry_mode": "capped_half_open",
                    "retry_not_before_ms": projection.retry_not_before_ms,
                }),
            )?;
        }
        if !retry_allowed {
            append_supply_event(
                root,
                "scientific_failure_circuit_open",
                &json!({
                    "job_id": current,
                    "scientific_failures": projection.scientific_failures,
                    "attempt_receipt_sha256": receipt.content_sha256,
                }),
            )?;
            return Ok(ProposerRetryDisposition::ScientificCircuitOpen);
        }
        reconcile_proposer_projection(root, self, &projection)?;
        self.revision += 1;
        self.validate(&self.wave, &self.checkpoint_sha256)?;
        append_supply_event(
            root,
            if terminal_state == "cancelled" {
                "foreign_cancellation_recovered"
            } else {
                "infrastructure_attempt_recovered"
            },
            &json!({
                "prior_job_id": current,
                "next_job_id": self.proposal_job_id,
                "infrastructure_attempt": projection.current_attempt_number,
                "scientific_failures": projection.scientific_failures,
                "foreign_cancellations": projection.foreign_cancellation_timestamps_ms.len(),
                "attempt_receipt_sha256": receipt.content_sha256,
                "retry_not_before_ms": retry_not_before_ms,
                "contention_alarm_open": contention_alarm_open,
            }),
        )?;
        Ok(ProposerRetryDisposition::Retried)
    }
}
